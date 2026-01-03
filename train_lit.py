#!/usr/bin/env python3
"""
Section 4.2: Vision-Language Contrastive Learning (LiT)
=======================================================

Following LION Paper Section 4.2:
- LiT (Locked-image Text Tuning)
- ViT-B/32 image encoder (frozen)
- Text encoder (trained)

Usage:
    torchrun --nproc_per_node=8 train_lit.py --optimizer all --epochs 30
"""

import os
import time
import math
import random
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Optimizer
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Logging
# =============================================================================

def setup_logger(output_dir, rank, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    fh = logging.FileHandler(output_dir / f"{name}_rank{rank}.log", mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
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

class RLO(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1, belief_coef=0.1, eps=1e-8):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, belief_coef=belief_coef, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                c = group["betas"][0] * m + (1 - group["betas"][0]) * g
                delta = g - m
                update = c.sign() + group["belief_coef"] * (delta / delta.norm().clamp(min=group["eps"]))
                p.add_(update, alpha=-group["lr"])
                m.mul_(group["betas"][1]).add_(g, alpha=1 - group["betas"][1])


class RLO_LambdaA(Optimizer):
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.99, beta3=0.999,
                 weight_decay=0.1, lambda_b=0.1, eps=1e-8, gamma=5.0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, beta3=beta3,
                        weight_decay=weight_decay, lambda_b=lambda_b, eps=eps, gamma=gamma)
        super().__init__(params, defaults)
        self.sqrt_dim = math.sqrt(sum(p.numel() for g in self.param_groups for p in g["params"]))

    @torch.no_grad()
    def step(self, closure=None):
        all_sp, all_b, all_p = [], [], []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["m"], state["s"] = torch.zeros_like(p), torch.zeros_like(p)
                m, s = state["m"], state["s"]
                s.mul_(group["beta3"]).addcmul_(g, g, value=1 - group["beta3"])
                c = group["beta1"] * m + (1 - group["beta1"]) * g
                sp = torch.tanh(group["gamma"] * c) / (s.sqrt() + group["eps"])
                delta = g - m
                b = group["lambda_b"] * (delta / delta.norm().clamp(min=group["eps"]))
                all_sp.append(sp); all_b.append(b); all_p.append((p, group, m))
        if not all_p:
            return
        scale = self.sqrt_dim / sum((x * x).sum() for x in all_sp).sqrt().clamp(min=1e-8)
        for (p, group, m), sp, b in zip(all_p, all_sp, all_b):
            update = scale * sp + b
            if group["weight_decay"] != 0:
                p.mul_(1 - group["lr"] * group["weight_decay"])
            p.add_(update, alpha=-group["lr"])
            m.mul_(group["beta2"]).add_(p.grad, alpha=1 - group["beta2"])


class SmoothLiftedRLO(Optimizer):
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.99, eta=0.3,
                 weight_decay=0.1, lambda_b=0.1, eps=1e-8, gamma=5.0):
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
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["m"], state["v"] = torch.zeros_like(p), torch.zeros_like(p)
                m = state["m"]
                c = group["beta1"] * m + (1 - group["beta1"]) * g
                s = torch.tanh(group["gamma"] * c)
                delta = g - m
                b = group["lambda_b"] * (delta / delta.norm().clamp(min=group["eps"]))
                all_s.append(s); all_b.append(b); all_p.append((p, group))
        if not all_p:
            return
        scale = self.sqrt_dim / sum((x * x).sum() for x in all_s).sqrt().clamp(min=1e-8)
        for (p, group), s, b in zip(all_p, all_s, all_b):
            update = scale * s + b
            state = self.state[p]
            if group["weight_decay"] != 0:
                p.mul_(1 - group["lr"] * group["weight_decay"])
            state["v"].mul_(1 - group["eta"]).add_(update, alpha=group["eta"])
            p.add_(state["v"], alpha=-group["lr"])
            state["m"].mul_(group["beta2"]).add_(p.grad, alpha=1 - group["beta2"])


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
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p)
                m = state['m']
                beta1, beta2 = group['betas']
                p.add_((beta1 * m + (1 - beta1) * g).sign_(), alpha=-group['lr'])
                m.mul_(beta2).add_(g, alpha=1 - beta2)


# =============================================================================
# Models
# =============================================================================

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


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


class VisionEncoder(nn.Module):
    """ViT-B/32 image encoder"""
    def __init__(self, embed_dim=768):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, 32, 32)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 50, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, 12) for _ in range(12)])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return F.normalize(self.norm(x[:, 0]), dim=-1)


class TextEncoder(nn.Module):
    """Text Transformer"""
    def __init__(self, vocab_size=49408, context_len=77, embed_dim=512, output_dim=768):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, context_len, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, 8) for _ in range(12)])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, output_dim, bias=False)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x):
        x = self.token_emb(x) + self.pos_emb[:, :x.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        return F.normalize(self.proj(self.norm(x).mean(dim=1)), dim=-1)


class LiT(nn.Module):
    """Locked-image Text tuning"""
    def __init__(self, embed_dim=768, temperature=0.07):
        super().__init__()
        self.image_encoder = VisionEncoder(embed_dim)
        self.text_encoder = TextEncoder(output_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def forward(self, image, text):
        img_feat = self.image_encoder(image)
        txt_feat = self.text_encoder(text)
        scale = self.logit_scale.exp()
        return scale * img_feat @ txt_feat.t(), scale * txt_feat @ img_feat.t()


# =============================================================================
# Dataset
# =============================================================================

TEMPLATES = ["a photo of a {}.", "a picture of a {}.", "an image of a {}."]


class ImageTextDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.dataset = ImageFolder(root / split, transform=transform)
        self.context_len = 77

    def tokenize(self, text):
        tokens = [1]
        for word in text.lower().replace('.', '').split()[:20]:
            tokens.append(hash(word) % 49400 + 4)
        tokens.append(2)
        tokens += [0] * (self.context_len - len(tokens))
        return torch.tensor(tokens[:self.context_len], dtype=torch.long)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        text = random.choice(TEMPLATES).format(f"class {label}")
        return img, self.tokenize(text), label


# =============================================================================
# Training
# =============================================================================

def contrastive_loss(logits_i, logits_t):
    labels = torch.arange(logits_i.shape[0], device=logits_i.device)
    return (F.cross_entropy(logits_i, labels) + F.cross_entropy(logits_t, labels)) / 2


CONFIGS = {
    'adamw': {'lr': 1e-3, 'wd': 0.1},
    'lion': {'lr': 1e-4, 'wd': 1.0},
    'rlo': {'lr': 1e-4, 'wd': 1.0},
    'rlo_lambda_a': {'lr': 1e-4, 'wd': 1.0},
    'smooth_lifted_rlo': {'lr': 1e-4, 'wd': 1.0},
}


def create_optimizer(params, name, lr, wd):
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif name == 'lion':
        return Lion(params, lr=lr, weight_decay=wd)
    elif name == 'rlo':
        return RLO(params, lr=lr, weight_decay=wd)
    elif name == 'rlo_lambda_a':
        return RLO_LambdaA(params, lr=lr, weight_decay=wd)
    elif name == 'smooth_lifted_rlo':
        return SmoothLiftedRLO(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown: {name}")


def train_lit(opt_name, data_path, output_dir, epochs, rank, world_size, local_rank, logger):
    device = torch.device(f'cuda:{local_rank}')
    cfg = CONFIGS[opt_name]
    global_batch, per_gpu = 4096, 4096 // world_size

    logger.info("=" * 60)
    logger.info(f"LiT Training: {opt_name}")
    logger.info(f"LR: {cfg['lr']}, WD: {cfg['wd']}, Global batch: {global_batch}")
    logger.info("=" * 60)

    # Data
    transform = T.Compose([
        T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
        T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = ImageTextDataset(data_path, 'train', transform)
    sampler = DistributedSampler(dataset, world_size, rank) if world_size > 1 else None
    loader = DataLoader(dataset, per_gpu, shuffle=(sampler is None), sampler=sampler,
                       num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)

    logger.info(f"Train: {len(dataset)} samples, {len(loader)} steps/epoch")

    # Model
    model = LiT().to(device)
    for p in model.image_encoder.parameters():
        p.requires_grad = False
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    base_model = model.module if hasattr(model, 'module') else model
    optimizer = create_optimizer(base_model.text_encoder.parameters(), opt_name, cfg['lr'], cfg['wd'])

    # Scheduler
    total_steps = len(loader) * epochs
    warmup_steps = len(loader) * 3

    def get_lr(step):
        if step < warmup_steps:
            return cfg['lr'] * step / max(1, warmup_steps)
        return cfg['lr'] * 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

    scaler = torch.amp.GradScaler('cuda')
    history = {'loss': [], 'lr': [], 'epoch_time': []}
    step = 0

    for epoch in range(epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        base_model.image_encoder.eval()
        loss_sum = 0.0
        t0 = time.time()

        for batch_idx, (imgs, tokens, _) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)

            lr = get_lr(step)
            for g in optimizer.param_groups:
                g['lr'] = lr

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast('cuda', torch.bfloat16):
                logits_i, logits_t = model(imgs, tokens)
                loss = contrastive_loss(logits_i, logits_t)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(base_model.text_encoder.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()
            step += 1

            if rank == 0 and (batch_idx + 1) % 100 == 0:
                logger.debug(f"E{epoch+1} S{batch_idx+1}: loss={loss.item():.4f}")

        epoch_time = time.time() - t0
        avg_loss = loss_sum / len(loader)
        history['loss'].append(avg_loss)
        history['lr'].append(lr)
        history['epoch_time'].append(epoch_time)

        logger.info(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, time={epoch_time:.1f}s")

    # Save
    if rank == 0:
        results = {'optimizer': opt_name, 'config': cfg, 'final_loss': history['loss'][-1],
                  'total_time': sum(history['epoch_time']), 'history': history}
        with open(output_dir / f"lit_{opt_name}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        torch.save(base_model.state_dict(), output_dir / f"lit_{opt_name}_model.pt")
        logger.info(f"Final loss: {history['loss'][-1]:.4f}")

    return history['loss'][-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='all')
    parser.add_argument('--data', default='/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder')
    parser.add_argument('--output', default='./results_lit')
    parser.add_argument('--epochs', type=int, default=30)
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
    logger = setup_logger(output_dir, rank, "lit")

    opts = list(CONFIGS.keys()) if args.optimizer == 'all' else [args.optimizer]
    results = {}
    
    for opt in opts:
        try:
            loss = train_lit(opt, Path(args.data), output_dir, args.epochs, rank, world_size, local_rank, logger)
            results[opt] = loss
        except Exception as e:
            logger.error(f"Error {opt}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    if rank == 0 and results:
        logger.info("=" * 60)
        logger.info("LiT Results (Loss)")
        for opt, loss in sorted(results.items(), key=lambda x: x[1]):
            logger.info(f"  {opt}: {loss:.4f}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
