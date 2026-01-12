#!/usr/bin/env python3
"""
Section 4.1: Image Classification on ImageNet
==============================================
Following LION Paper Section 4.1 EXACTLY:

ResNet-50 (90 epochs, batch=1024):
- AdamW: lr=1e-3, wd=1e-4, β=(0.9, 0.999)
- Lion:  lr=1e-4, wd=1.0,  β=(0.9, 0.99)

ViT-S/16, ViT-B/16 (300 epochs, batch=1024):
- AdamW: lr=1e-3, wd=0.3
- Lion:  lr=3e-4, wd=0.5

MEMORY FIX: num_workers=2, persistent_workers=False, gradient checkpointing
"""

import sys
def _patch_dill():
    try:
        import dill
        if not hasattr(dill, 'extend'):
            dill.extend = lambda use_dill=True: None
    except ImportError:
        pass
_patch_dill()

import os, gc, time, math, random, json, logging, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Optimizer
from torch.utils.checkpoint import checkpoint
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
import warnings
warnings.filterwarnings('ignore')


def setup_logger(output_dir, rank, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(output_dir / f"{name}_rank{rank}.log", mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
    logger.addHandler(fh)
    if rank == 0:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(ch)
    return logger


# =============================================================================
# Optimizers
# =============================================================================
class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p)
                m = state['m']
                beta1, beta2 = group['betas']
                p.add_((beta1 * m + (1 - beta1) * p.grad).sign_(), alpha=-group['lr'])
                m.mul_(beta2).add_(p.grad, alpha=1 - beta2)


class RLO(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, belief_coef=0.1, eps=1e-8):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, belief_coef=belief_coef, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                m, g = state["m"], p.grad
                beta1, beta2 = group["betas"]
                c = beta1 * m + (1 - beta1) * g
                delta = g - m
                update = c.sign() + group["belief_coef"] * (delta / delta.norm().clamp(min=group["eps"]))
                p.add_(update, alpha=-group["lr"])
                m.mul_(beta2).add_(g, alpha=1 - beta2)


class SmoothLiftedRLO(Optimizer):
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.99, eta=0.3,
                 weight_decay=0.0, lambda_b=0.1, eps=1e-8, gamma=5.0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eta=eta,
                        weight_decay=weight_decay, lambda_b=lambda_b, eps=eps, gamma=gamma)
        super().__init__(params, defaults)
        self.sqrt_dim = math.sqrt(sum(p.numel() for g in self.param_groups for p in g["params"]))

    @torch.no_grad()
    def step(self, closure=None):
        all_s, all_b, all_p = [], [], []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                g, m = p.grad, state["m"]
                c = group["beta1"] * m + (1 - group["beta1"]) * g
                s = torch.tanh(group["gamma"] * c)
                delta = g - m
                b = group["lambda_b"] * (delta / delta.norm().clamp(min=group["eps"]))
                all_s.append(s); all_b.append(b); all_p.append((p, group))
        if not all_p:
            return
        scale = self.sqrt_dim / sum((x * x).sum() for x in all_s).sqrt().clamp(min=1e-8)
        for (p, group), s, b in zip(all_p, all_s, all_b):
            state = self.state[p]
            if group["weight_decay"] != 0:
                p.mul_(1 - group["lr"] * group["weight_decay"])
            d = scale * s + b
            state["v"].mul_(1 - group["eta"]).add_(d, alpha=group["eta"])
            p.add_(state["v"], alpha=-group["lr"])
            state["m"].mul_(group["beta2"]).add_(p.grad, alpha=1 - group["beta2"])


# =============================================================================
# Models
# =============================================================================
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    def __init__(self, embed_dim=384, depth=12, num_heads=6, num_classes=1000):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, 16, 16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = checkpoint(blk, x, use_reentrant=False) if self.training else blk(x)
        return self.head(self.norm(x[:, 0]))


def create_model(name):
    if name == 'resnet50':
        return resnet50(weights=None, num_classes=1000)
    elif name == 'vit_s16':
        return ViT(embed_dim=384, depth=12, num_heads=6)
    elif name == 'vit_b16':
        return ViT(embed_dim=768, depth=12, num_heads=12)
    raise ValueError(f"Unknown: {name}")


# =============================================================================
# Data - MEMORY OPTIMIZED
# =============================================================================
def create_loader(data_path, batch_size, is_train, world_size, rank, augment=False):
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if is_train:
        transforms = [T.RandomResizedCrop(224), T.RandomHorizontalFlip()]
        if augment:
            transforms.append(T.RandAugment(2, 9))
        transforms.extend([T.ToTensor(), T.Normalize(MEAN, STD)])
        transform, split = T.Compose(transforms), 'train'
    else:
        transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
        split = 'val'
    dataset = ImageFolder(data_path / split, transform)
    sampler = DistributedSampler(dataset, world_size, rank, shuffle=is_train) if world_size > 1 else None
    # MEMORY FIX: num_workers=2, persistent_workers=False
    return DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None and is_train),
                     sampler=sampler, num_workers=2, pin_memory=True, drop_last=is_train,
                     persistent_workers=False), sampler


class Mixup:
    def __init__(self, alpha=0.8, num_classes=1000):
        self.alpha, self.num_classes = alpha, num_classes

    def __call__(self, x, y):
        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(x.size(0), device=x.device)
        y_oh = F.one_hot(y, self.num_classes).float()
        return lam * x + (1 - lam) * x[idx], lam * y_oh + (1 - lam) * y_oh[idx]


# =============================================================================
# Config - LION Paper EXACT Settings
# =============================================================================
CONFIGS = {
    'resnet50': {
        'epochs': 90, 'global_batch': 1024, 'warmup_epochs': 5,
        'augment': False, 'mixup': False, 'label_smoothing': 0.0,
        'optimizers': {
            'adamw': {'lr': 1e-3, 'wd': 1e-4, 'betas': (0.9, 0.999)},
            'lion': {'lr': 1e-4, 'wd': 1.0, 'betas': (0.9, 0.99)},
            'rlo': {'lr': 1e-4, 'wd': 1.0, 'betas': (0.9, 0.99), 'belief_coef': 0.1},
            'smooth_lifted_rlo': {'lr': 1e-4, 'wd': 1.0, 'lambda_b': 0.1, 'eta': 0.3},
        },
    },
    'vit_s16': {
        'epochs': 300, 'global_batch': 1024, 'warmup_epochs': 10,
        'augment': True, 'mixup': True, 'label_smoothing': 0.1,
        'optimizers': {
            'adamw': {'lr': 1e-3, 'wd': 0.3, 'betas': (0.9, 0.999)},
            'lion': {'lr': 3e-4, 'wd': 0.5, 'betas': (0.9, 0.99)},
            'rlo': {'lr': 3e-4, 'wd': 0.5, 'betas': (0.9, 0.99), 'belief_coef': 0.1},
            'smooth_lifted_rlo': {'lr': 3e-4, 'wd': 0.5, 'lambda_b': 0.1, 'eta': 0.3},
        },
    },
    'vit_b16': {
        'epochs': 300, 'global_batch': 1024, 'warmup_epochs': 10,
        'augment': True, 'mixup': True, 'label_smoothing': 0.1,
        'optimizers': {
            'adamw': {'lr': 1e-3, 'wd': 0.3, 'betas': (0.9, 0.999)},
            'lion': {'lr': 3e-4, 'wd': 0.5, 'betas': (0.9, 0.99)},
            'rlo': {'lr': 3e-4, 'wd': 0.5, 'betas': (0.9, 0.99), 'belief_coef': 0.1},
            'smooth_lifted_rlo': {'lr': 3e-4, 'wd': 0.5, 'lambda_b': 0.1, 'eta': 0.3},
        },
    },
}


def create_optimizer(model, name, cfg):
    params = model.parameters()
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['wd'], betas=cfg.get('betas', (0.9, 0.999)))
    elif name == 'lion':
        return Lion(params, lr=cfg['lr'], weight_decay=cfg['wd'], betas=cfg.get('betas', (0.9, 0.99)))
    elif name == 'rlo':
        return RLO(params, lr=cfg['lr'], weight_decay=cfg['wd'], betas=cfg.get('betas', (0.9, 0.99)),
                   belief_coef=cfg.get('belief_coef', 0.1))
    elif name == 'smooth_lifted_rlo':
        return SmoothLiftedRLO(params, lr=cfg['lr'], weight_decay=cfg['wd'],
                              lambda_b=cfg.get('lambda_b', 0.1), eta=cfg.get('eta', 0.3))
    raise ValueError(f"Unknown: {name}")


# =============================================================================
# Training
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast('cuda', torch.bfloat16):
            correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def train_one_model(model_name, opt_name, data_path, output_dir, rank, world_size, local_rank, logger):
    device = torch.device(f'cuda:{local_rank}')
    cfg = CONFIGS[model_name]
    opt_cfg = cfg['optimizers'][opt_name]
    per_gpu = cfg['global_batch'] // world_size

    logger.info("=" * 70)
    logger.info(f"Model: {model_name} | Opt: {opt_name} | LR: {opt_cfg['lr']} | WD: {opt_cfg['wd']}")

    train_loader, train_sampler = create_loader(data_path, per_gpu, True, world_size, rank, cfg['augment'])
    val_loader, _ = create_loader(data_path, per_gpu, False, 1, 0)
    steps = len(train_loader)

    model = create_model(model_name).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    logger.info(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M | Steps/epoch: {steps}")

    optimizer = create_optimizer(model, opt_name, opt_cfg)
    total_steps = steps * cfg['epochs']
    warmup_steps = steps * cfg['warmup_epochs']
    
    def get_lr(step):
        if step < warmup_steps:
            return opt_cfg['lr'] * step / max(1, warmup_steps)
        return opt_cfg['lr'] * 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

    mixup = Mixup() if cfg['mixup'] else None
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    scaler = torch.amp.GradScaler('cuda')

    history = {'train_loss': [], 'val_acc': [], 'lr': []}
    best_acc, step = 0.0, 0

    for epoch in range(cfg['epochs']):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        loss_sum, t0 = 0.0, time.time()

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if mixup:
                x, y_soft = mixup(x, y)
                use_soft = True
            else:
                y_soft, use_soft = None, False

            lr = get_lr(step)
            for g in optimizer.param_groups:
                g['lr'] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', torch.bfloat16):
                logits = model(x)
                loss = -torch.sum(F.log_softmax(logits, 1) * y_soft, 1).mean() if use_soft else criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()
            step += 1

        base_model = model.module if hasattr(model, 'module') else model
        val_acc = evaluate(base_model, val_loader, device)
        history['train_loss'].append(loss_sum / steps)
        history['val_acc'].append(val_acc)
        history['lr'].append(lr)

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        logger.info(f"E{epoch+1:3d}: loss={loss_sum/steps:.4f}, val={val_acc:.2f}%{'*' if is_best else ''}, "
                   f"{len(train_loader.dataset)/(time.time()-t0):.0f} img/s")

        if is_best and rank == 0:
            torch.save({'model': base_model.state_dict(), 'acc': best_acc}, output_dir / f"{model_name}_{opt_name}_best.pt")

        if (epoch + 1) % 10 == 0:
            gc.collect(); torch.cuda.empty_cache()

    if rank == 0:
        with open(output_dir / f"{model_name}_{opt_name}_results.json", 'w') as f:
            json.dump({'model': model_name, 'optimizer': opt_name, 'best_acc': best_acc, 'history': history}, f, indent=2)
    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['resnet50', 'vit_s16', 'vit_b16'])
    parser.add_argument('--optimizer', default='all')
    parser.add_argument('--data', default='/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder')
    parser.add_argument('--output', default='/blue/wdixon/wang.yixuan/rlo_results/classification')
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    if 'RANK' in os.environ:
        rank, world_size, local_rank = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl')
    else:
        rank, world_size, local_rank = 0, 1, 0

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = setup_logger(output_dir, rank, f"cls_{args.model}")

    opts = list(CONFIGS[args.model]['optimizers'].keys()) if args.optimizer == 'all' else [args.optimizer]
    results = {}
    for opt in opts:
        try:
            gc.collect(); torch.cuda.empty_cache()
            results[opt] = train_one_model(args.model, opt, Path(args.data), output_dir, rank, world_size, local_rank, logger)
        except Exception as e:
            logger.error(f"Error {opt}: {e}")
            import traceback; logger.error(traceback.format_exc())

    if rank == 0 and results:
        logger.info("=" * 70)
        for opt, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {opt}: {acc:.2f}%")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
