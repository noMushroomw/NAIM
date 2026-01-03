#!/usr/bin/env python3
"""
Ablation Studies for RLO
========================

Studies:
1. belief_coef: [0.0, 0.05, 0.1, 0.2, 0.5]
2. gamma: [1.0, 3.0, 5.0, 10.0, 20.0]
3. lr_wd_grid: LR x WD sensitivity
4. batch_size: [256, 512, 1024, 2048, 4096]
5. components: LION vs RLO(no belief) vs RLO

Usage:
    torchrun --nproc_per_node=8 train_ablation.py --study belief_coef
    torchrun --nproc_per_node=8 train_ablation.py --study gamma
    torchrun --nproc_per_node=8 train_ablation.py --study components
    torchrun --nproc_per_node=8 train_ablation.py --study all
"""

import os
import time
import math
import random
import json
import logging
import argparse
from pathlib import Path
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
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
# Model (ViT-S/16 for ablation)
# =============================================================================

class Attention(nn.Module):
    def __init__(self, dim, num_heads=6):
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


class ViT_Small(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        embed_dim = 384
        self.patch_embed = nn.Conv2d(3, embed_dim, 16, 16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, 6) for _ in range(12)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, 0]))


# =============================================================================
# Data
# =============================================================================

def create_loader(data_path, per_gpu, is_train, world_size, rank):
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if is_train:
        transform = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.RandAugment(2, 9),
                              T.ToTensor(), T.Normalize(MEAN, STD)])
        split = 'train'
    else:
        transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
        split = 'val'
    dataset = ImageFolder(data_path / split, transform)
    sampler = DistributedSampler(dataset, world_size, rank, shuffle=is_train) if world_size > 1 else None
    return DataLoader(dataset, per_gpu, shuffle=(sampler is None and is_train), sampler=sampler,
                     num_workers=8, pin_memory=True, drop_last=is_train, persistent_workers=True), sampler


class Mixup:
    def __init__(self, alpha=0.8, num_classes=1000):
        self.alpha, self.num_classes = alpha, num_classes

    def __call__(self, x, y):
        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(x.size(0), device=x.device)
        x = lam * x + (1 - lam) * x[idx]
        y_oh = F.one_hot(y, self.num_classes).float()
        return x, lam * y_oh + (1 - lam) * y_oh[idx]


# =============================================================================
# Training Function
# =============================================================================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast('cuda', torch.bfloat16):
            out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def train_config(optimizer_fn, data_path, global_batch, epochs, rank, world_size, local_rank, logger, name="config"):
    """Train one configuration"""
    device = torch.device(f'cuda:{local_rank}')
    per_gpu = global_batch // world_size
    
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)

    train_loader, train_sampler = create_loader(data_path, per_gpu, True, world_size, rank)
    val_loader, _ = create_loader(data_path, per_gpu * 2, False, 1, 0)

    model = ViT_Small().to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    optimizer, base_lr = optimizer_fn(model.parameters())
    
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * 5
    
    def get_lr(step):
        if step < warmup_steps:
            return base_lr * step / max(1, warmup_steps)
        return base_lr * 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

    mixup = Mixup()
    scaler = torch.amp.GradScaler('cuda')
    history = {'val_acc': []}
    step = 0

    for epoch in range(epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x, y_mixed = mixup(x, y)

            lr = get_lr(step)
            for g in optimizer.param_groups:
                g['lr'] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', torch.bfloat16):
                out = model(x)
                loss = -torch.sum(F.log_softmax(out, dim=1) * y_mixed, dim=1).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1

        base_model = model.module if hasattr(model, 'module') else model
        val_acc = evaluate(base_model, val_loader, device)
        history['val_acc'].append(val_acc)

        if rank == 0:
            logger.info(f"  {name} E{epoch+1}/{epochs}: val={val_acc:.2f}%, time={time.time()-t0:.1f}s")

    return {'best_acc': max(history['val_acc']), 'final_acc': history['val_acc'][-1], 'history': history}


# =============================================================================
# Ablation Studies
# =============================================================================

def ablation_belief_coef(data_path, output_dir, rank, world_size, local_rank, logger):
    """Study 1: Belief coefficient sensitivity"""
    logger.info("=" * 60)
    logger.info("ABLATION: Belief Coefficient (λ_b)")
    logger.info("=" * 60)
    
    belief_coefs = [0.0, 0.05, 0.1, 0.2, 0.5]
    results = {}
    
    for belief in belief_coefs:
        logger.info(f"\nTraining with belief_coef={belief}")
        def opt_fn(params):
            return RLO(params, lr=1e-4, weight_decay=1.0, belief_coef=belief), 1e-4
        
        result = train_config(opt_fn, data_path, 1024, 30, rank, world_size, local_rank, logger, f"belief={belief}")
        results[f"belief_{belief}"] = result
    
    if rank == 0:
        with open(output_dir / "ablation_belief_coef.json", 'w') as f:
            json.dump(results, f, indent=2)
        logger.info("\nResults:")
        for k, v in results.items():
            logger.info(f"  {k}: {v['best_acc']:.2f}%")
    
    return results


def ablation_gamma(data_path, output_dir, rank, world_size, local_rank, logger):
    """Study 2: Gamma sensitivity"""
    logger.info("=" * 60)
    logger.info("ABLATION: Gamma (γ)")
    logger.info("=" * 60)
    
    gammas = [1.0, 3.0, 5.0, 10.0, 20.0]
    results = {}
    
    for gamma in gammas:
        logger.info(f"\nTraining with gamma={gamma}")
        def opt_fn(params, g=gamma):
            return RLO_LambdaA(params, lr=1e-4, weight_decay=1.0, gamma=g), 1e-4
        
        result = train_config(opt_fn, data_path, 1024, 30, rank, world_size, local_rank, logger, f"gamma={gamma}")
        results[f"gamma_{gamma}"] = result
    
    if rank == 0:
        with open(output_dir / "ablation_gamma.json", 'w') as f:
            json.dump(results, f, indent=2)
        logger.info("\nResults:")
        for k, v in results.items():
            logger.info(f"  {k}: {v['best_acc']:.2f}%")
    
    return results


def ablation_lr_wd_grid(data_path, output_dir, rank, world_size, local_rank, logger):
    """Study 3: LR x WD grid search"""
    logger.info("=" * 60)
    logger.info("ABLATION: LR x WD Grid")
    logger.info("=" * 60)
    
    lrs = [3e-5, 1e-4, 3e-4]
    wds = [0.5, 1.0, 2.0]
    results = {}
    
    for lr, wd in product(lrs, wds):
        logger.info(f"\nTraining with lr={lr}, wd={wd}")
        def opt_fn(params, _lr=lr, _wd=wd):
            return RLO(params, lr=_lr, weight_decay=_wd), _lr
        
        result = train_config(opt_fn, data_path, 1024, 30, rank, world_size, local_rank, logger, f"lr={lr}_wd={wd}")
        results[f"lr{lr}_wd{wd}"] = result
    
    if rank == 0:
        with open(output_dir / "ablation_lr_wd.json", 'w') as f:
            json.dump(results, f, indent=2)
        logger.info("\nResults:")
        for k, v in sorted(results.items(), key=lambda x: x[1]['best_acc'], reverse=True):
            logger.info(f"  {k}: {v['best_acc']:.2f}%")
    
    return results


def ablation_components(data_path, output_dir, rank, world_size, local_rank, logger):
    """Study 5: Component analysis - what makes RLO better?"""
    logger.info("=" * 60)
    logger.info("ABLATION: Component Analysis")
    logger.info("=" * 60)
    
    results = {}
    
    # 1. LION baseline
    logger.info("\n1. Training LION (baseline)")
    def lion_fn(params):
        return Lion(params, lr=1e-4, weight_decay=1.0), 1e-4
    results['lion'] = train_config(lion_fn, data_path, 1024, 30, rank, world_size, local_rank, logger, "LION")
    
    # 2. RLO without belief (should match LION)
    logger.info("\n2. Training RLO (belief=0, LION equivalent)")
    def rlo_no_belief_fn(params):
        return RLO(params, lr=1e-4, weight_decay=1.0, belief_coef=0.0), 1e-4
    results['rlo_no_belief'] = train_config(rlo_no_belief_fn, data_path, 1024, 30, rank, world_size, local_rank, logger, "RLO(belief=0)")
    
    # 3. RLO with belief
    logger.info("\n3. Training RLO (with belief)")
    def rlo_fn(params):
        return RLO(params, lr=1e-4, weight_decay=1.0, belief_coef=0.1), 1e-4
    results['rlo'] = train_config(rlo_fn, data_path, 1024, 30, rank, world_size, local_rank, logger, "RLO")
    
    # 4. RLO_LambdaA
    logger.info("\n4. Training RLO_LambdaA")
    def rlo_lambda_fn(params):
        return RLO_LambdaA(params, lr=1e-4, weight_decay=1.0), 1e-4
    results['rlo_lambda_a'] = train_config(rlo_lambda_fn, data_path, 1024, 30, rank, world_size, local_rank, logger, "RLO_LambdaA")
    
    # 5. AdamW reference
    logger.info("\n5. Training AdamW (reference)")
    def adamw_fn(params):
        return torch.optim.AdamW(params, lr=1e-3, weight_decay=0.1), 1e-3
    results['adamw'] = train_config(adamw_fn, data_path, 1024, 30, rank, world_size, local_rank, logger, "AdamW")
    
    if rank == 0:
        with open(output_dir / "ablation_components.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("\n" + "=" * 60)
        logger.info("COMPONENT ANALYSIS RESULTS")
        logger.info("=" * 60)
        for opt, res in sorted(results.items(), key=lambda x: x[1]['best_acc'], reverse=True):
            logger.info(f"  {opt:<20}: {res['best_acc']:.2f}%")
        
        # Key finding
        belief_gain = results['rlo']['best_acc'] - results['rlo_no_belief']['best_acc']
        logger.info(f"\nKey Finding: Belief correction adds {belief_gain:+.2f}% over LION-equivalent")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', default='all', choices=['all', 'belief_coef', 'gamma', 'lr_wd', 'components'])
    parser.add_argument('--data', default='/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder')
    parser.add_argument('--output', default='./results_ablation')
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if 'RANK' in os.environ:
        rank, world_size, local_rank = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl')
    else:
        rank, world_size, local_rank = 0, 1, 0

    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = setup_logger(output_dir, rank, "ablation")

    studies = {
        'belief_coef': ablation_belief_coef,
        'gamma': ablation_gamma,
        'lr_wd': ablation_lr_wd_grid,
        'components': ablation_components,
    }

    if args.study == 'all':
        for name, fn in studies.items():
            try:
                fn(data_path, output_dir, rank, world_size, local_rank, logger)
            except Exception as e:
                logger.error(f"Error in {name}: {e}")
    else:
        studies[args.study](data_path, output_dir, rank, world_size, local_rank, logger)

    if rank == 0:
        logger.info("\n" + "=" * 60)
        logger.info("ALL ABLATION STUDIES COMPLETE")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 60)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
