#!/usr/bin/env python3
# =============================================================================
# Dill Compatibility Patch (MUST be before any torch imports)
# =============================================================================
import sys
def _patch_dill():
    try:
        import dill
        if not hasattr(dill, 'extend'):
            dill.extend = lambda use_dill=True: None
    except ImportError:
        pass
_patch_dill()
# =============================================================================

"""
Section 4.1: Image Classification on ImageNet
==============================================

Following LION Paper Section 4.1:
- ResNet-50: 90 epochs, global batch = 1024
- ViT-S/16, ViT-B/16: 300 epochs, global batch = 4096

Usage:
    pip install dill==0.3.6 --break-system-packages
    torchrun --nproc_per_node=8 train_classification.py --model resnet50 --optimizer all
    torchrun --nproc_per_node=8 train_classification.py --model vit_s16 --optimizer all
    torchrun --nproc_per_node=8 train_classification.py --model vit_b16 --optimizer all
"""

import os
import sys
import time
import math
import random
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

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
from torchvision.models import resnet50

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logger(output_dir: Path, rank: int, name: str) -> logging.Logger:
    """Setup logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # File handler - all ranks
    fh = logging.FileHandler(output_dir / f"{name}_rank{rank}.log", mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(fh)
    
    # Console handler - rank 0 only
    if rank == 0:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(ch)
    
    return logger


# =============================================================================
# NAIM Tracker for mechanism evidence
# =============================================================================

class NAIMTracker:
    """Track optimization dynamics metrics"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.grad_norms = []
        self.momentum_norms = []
        self.update_norms = []
        self.belief_norms = []
        self.alignments = []
    
    def update(self, grad_norm, momentum_norm, update_norm, belief_norm=0.0, alignment=0.0):
        self.grad_norms.append(grad_norm)
        self.momentum_norms.append(momentum_norm)
        self.update_norms.append(update_norm)
        self.belief_norms.append(belief_norm)
        self.alignments.append(alignment)
    
    def get_stats(self):
        if not self.grad_norms:
            return {}
        return {
            'grad_norm': float(np.mean(self.grad_norms)),
            'momentum_norm': float(np.mean(self.momentum_norms)),
            'update_norm': float(np.mean(self.update_norms)),
            'belief_norm': float(np.mean(self.belief_norms)),
            'alignment': float(np.mean(self.alignments)),
        }


# =============================================================================
# Optimizers
# =============================================================================

class RLO(Optimizer):
    """Riemannian Lyapunov Optimizer"""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1,
                 belief_coef=0.1, eps=1e-8, track_stats=False):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                        belief_coef=belief_coef, eps=eps)
        super().__init__(params, defaults)
        self.track_stats = track_stats
        self.tracker = NAIMTracker() if track_stats else None

    @torch.no_grad()
    def step(self, closure=None):
        total_g, total_m, total_u, total_b, total_a, cnt = 0., 0., 0., 0., 0., 0

        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            beta1, beta2 = group["betas"]
            belief, eps = group["belief_coef"], group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]

                if wd != 0:
                    p.mul_(1 - lr * wd)

                c = beta1 * m + (1 - beta1) * g
                delta = g - m
                delta_norm = delta.norm().clamp(min=eps)
                belief_term = belief * (delta / delta_norm)
                update = c.sign() + belief_term

                p.add_(update, alpha=-lr)
                m.mul_(beta2).add_(g, alpha=1 - beta2)

                if self.track_stats:
                    total_g += g.norm().item() ** 2
                    total_m += m.norm().item() ** 2
                    total_u += update.norm().item() ** 2
                    total_b += belief_term.norm().item() ** 2
                    total_a += F.cosine_similarity(g.flatten().unsqueeze(0), m.flatten().unsqueeze(0)).item()
                    cnt += 1

        if self.track_stats and cnt > 0:
            self.tracker.update(math.sqrt(total_g), math.sqrt(total_m), 
                               math.sqrt(total_u), math.sqrt(total_b), total_a / cnt)


class RLO_LambdaA(Optimizer):
    """RLO with Lambda-A preconditioning"""
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.99, beta3=0.999,
                 weight_decay=0.1, lambda_b=0.1, eps=1e-8, gamma=5.0, track_stats=False):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, beta3=beta3,
                        weight_decay=weight_decay, lambda_b=lambda_b, eps=eps, gamma=gamma)
        super().__init__(params, defaults)
        self.sqrt_dim = math.sqrt(sum(p.numel() for g in self.param_groups for p in g["params"]))
        self.track_stats = track_stats
        self.tracker = NAIMTracker() if track_stats else None

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
                    state["m"] = torch.zeros_like(p)
                    state["s"] = torch.zeros_like(p)
                m, s = state["m"], state["s"]
                
                s.mul_(group["beta3"]).addcmul_(g, g, value=1 - group["beta3"])
                c = group["beta1"] * m + (1 - group["beta1"]) * g
                sp = torch.tanh(group["gamma"] * c) / (s.sqrt() + group["eps"])
                delta = g - m
                b = group["lambda_b"] * (delta / delta.norm().clamp(min=group["eps"]))
                
                all_sp.append(sp)
                all_b.append(b)
                all_p.append((p, group, m))

        if not all_p:
            return

        scale = self.sqrt_dim / sum((x * x).sum() for x in all_sp).sqrt().clamp(min=1e-8)
        total_u, total_b_norm = 0., 0.

        for (p, group, m), sp, b in zip(all_p, all_sp, all_b):
            update = scale * sp + b
            if group["weight_decay"] != 0:
                p.mul_(1 - group["lr"] * group["weight_decay"])
            p.add_(update, alpha=-group["lr"])
            m.mul_(group["beta2"]).add_(p.grad, alpha=1 - group["beta2"])
            
            if self.track_stats:
                total_u += update.norm().item() ** 2
                total_b_norm += b.norm().item() ** 2

        if self.track_stats:
            self.tracker.update(0, 0, math.sqrt(total_u), math.sqrt(total_b_norm), 0)


class SmoothLiftedRLO(Optimizer):
    """Smooth Lifted RLO with velocity tracking"""
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.99, eta=0.3,
                 weight_decay=0.1, lambda_b=0.1, eps=1e-8, gamma=5.0, track_stats=False):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eta=eta,
                        weight_decay=weight_decay, lambda_b=lambda_b, eps=eps, gamma=gamma)
        super().__init__(params, defaults)
        self.sqrt_dim = math.sqrt(sum(p.numel() for g in self.param_groups for p in g["params"]))
        self.track_stats = track_stats
        self.tracker = NAIMTracker() if track_stats else None

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
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                m = state["m"]
                
                c = group["beta1"] * m + (1 - group["beta1"]) * g
                s = torch.tanh(group["gamma"] * c)
                delta = g - m
                b = group["lambda_b"] * (delta / delta.norm().clamp(min=group["eps"]))
                
                all_s.append(s)
                all_b.append(b)
                all_p.append((p, group))

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
    """Lion optimizer (Google, 2023)"""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, track_stats=False):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.track_stats = track_stats
        self.tracker = NAIMTracker() if track_stats else None

    @torch.no_grad()
    def step(self, closure=None):
        total_g, total_m, total_u, cnt = 0., 0., 0., 0

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
                update = (beta1 * m + (1 - beta1) * g).sign_()
                p.add_(update, alpha=-group['lr'])
                m.mul_(beta2).add_(g, alpha=1 - beta2)

                if self.track_stats:
                    total_g += g.norm().item() ** 2
                    total_m += m.norm().item() ** 2
                    total_u += update.norm().item() ** 2
                    cnt += 1

        if self.track_stats and cnt > 0:
            self.tracker.update(math.sqrt(total_g), math.sqrt(total_m), math.sqrt(total_u), 0, 0)


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
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, num_classes=1000):
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
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, 0]))


def create_model(name):
    if name == 'resnet50':
        return resnet50(weights=None, num_classes=1000)
    elif name == 'vit_s16':
        return VisionTransformer(embed_dim=384, depth=12, num_heads=6)
    elif name == 'vit_b16':
        return VisionTransformer(embed_dim=768, depth=12, num_heads=12)
    raise ValueError(f"Unknown model: {name}")


# =============================================================================
# Data
# =============================================================================

def create_loader(data_path, per_gpu_batch, is_train, world_size, rank, num_workers=8, augment=False):
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    
    if is_train:
        transforms = [T.RandomResizedCrop(224), T.RandomHorizontalFlip()]
        if augment:
            transforms.append(T.RandAugment(2, 9))
        transforms.extend([T.ToTensor(), T.Normalize(MEAN, STD)])
        transform = T.Compose(transforms)
        split = 'train'
    else:
        transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
        split = 'val'

    dataset = ImageFolder(data_path / split, transform)
    sampler = DistributedSampler(dataset, world_size, rank, shuffle=is_train) if world_size > 1 else None
    
    return DataLoader(
        dataset, batch_size=per_gpu_batch, shuffle=(sampler is None and is_train),
        sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=is_train,
        persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None,
    ), sampler


class Mixup:
    def __init__(self, alpha=0.8, num_classes=1000):
        self.alpha, self.num_classes = alpha, num_classes

    def __call__(self, x, y):
        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(x.size(0), device=x.device)
        x = lam * x + (1 - lam) * x[idx]
        y_onehot = F.one_hot(y, self.num_classes).float()
        return x, lam * y_onehot + (1 - lam) * y_onehot[idx]


# =============================================================================
# Config
# =============================================================================

CONFIGS = {
    'resnet50': {
        'epochs': 90, 'global_batch': 1024, 'warmup_epochs': 5,
        'augment': False, 'mixup': False, 'label_smoothing': 0.0,
        'optimizers': {
            'adamw': {'lr': 1e-3, 'wd': 0.05},
            'lion': {'lr': 1e-4, 'wd': 0.5},
            'rlo': {'lr': 1e-4, 'wd': 0.5, 'belief_coef': 0.1},
            'rlo_lambda_a': {'lr': 1e-4, 'wd': 0.5, 'lambda_b': 0.1, 'gamma': 5.0},
            'smooth_lifted_rlo': {'lr': 1e-4, 'wd': 0.5, 'lambda_b': 0.1, 'gamma': 5.0, 'eta': 0.3},
        },
    },
    'vit_s16': {
        'epochs': 300, 'global_batch': 4096, 'warmup_epochs': 30,
        'augment': True, 'mixup': True, 'label_smoothing': 0.1,
        'optimizers': {
            'adamw': {'lr': 1e-3, 'wd': 0.1},
            'lion': {'lr': 1e-4, 'wd': 1.0},
            'rlo': {'lr': 1e-4, 'wd': 1.0, 'belief_coef': 0.1},
            'rlo_lambda_a': {'lr': 1e-4, 'wd': 1.0, 'lambda_b': 0.1, 'gamma': 5.0},
            'smooth_lifted_rlo': {'lr': 1e-4, 'wd': 1.0, 'lambda_b': 0.1, 'gamma': 5.0, 'eta': 0.3},
        },
    },
    'vit_b16': {
        'epochs': 300, 'global_batch': 4096, 'warmup_epochs': 30,
        'augment': True, 'mixup': True, 'label_smoothing': 0.1,
        'optimizers': {
            'adamw': {'lr': 1e-3, 'wd': 0.1},
            'lion': {'lr': 1e-4, 'wd': 1.0},
            'rlo': {'lr': 1e-4, 'wd': 1.0, 'belief_coef': 0.1},
            'rlo_lambda_a': {'lr': 1e-4, 'wd': 1.0, 'lambda_b': 0.1, 'gamma': 5.0},
            'smooth_lifted_rlo': {'lr': 1e-4, 'wd': 1.0, 'lambda_b': 0.1, 'gamma': 5.0, 'eta': 0.3},
        },
    },
}


def create_optimizer(model, name, cfg, track=False):
    params = model.parameters()
    lr, wd = cfg['lr'], cfg['wd']
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif name == 'lion':
        return Lion(params, lr=lr, weight_decay=wd, track_stats=track)
    elif name == 'rlo':
        return RLO(params, lr=lr, weight_decay=wd, belief_coef=cfg.get('belief_coef', 0.1), track_stats=track)
    elif name == 'rlo_lambda_a':
        return RLO_LambdaA(params, lr=lr, weight_decay=wd, lambda_b=cfg.get('lambda_b', 0.1),
                         gamma=cfg.get('gamma', 5.0), track_stats=track)
    elif name == 'smooth_lifted_rlo':
        return SmoothLiftedRLO(params, lr=lr, weight_decay=wd, lambda_b=cfg.get('lambda_b', 0.1),
                              gamma=cfg.get('gamma', 5.0), eta=cfg.get('eta', 0.3), track_stats=track)
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
            out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def train_model(model_name, opt_name, data_path, output_dir, rank, world_size, local_rank, logger):
    """Train one model with one optimizer"""
    device = torch.device(f'cuda:{local_rank}')
    cfg = CONFIGS[model_name]
    opt_cfg = cfg['optimizers'][opt_name]
    per_gpu = cfg['global_batch'] // world_size

    logger.info("=" * 70)
    logger.info(f"Model: {model_name} | Optimizer: {opt_name}")
    logger.info(f"Global batch: {cfg['global_batch']} | Per-GPU: {per_gpu} | Epochs: {cfg['epochs']}")
    logger.info(f"LR: {opt_cfg['lr']} | WD: {opt_cfg['wd']}")
    logger.info("=" * 70)

    # Data
    train_loader, train_sampler = create_loader(data_path, per_gpu, True, world_size, rank, 8, cfg['augment'])
    val_loader, _ = create_loader(data_path, per_gpu * 2, False, 1, 0, 8)
    
    steps_per_epoch = len(train_loader)
    logger.info(f"Train: {len(train_loader.dataset)} samples, {steps_per_epoch} steps/epoch")
    logger.info(f"Val: {len(val_loader.dataset)} samples")

    # Model
    model = create_model(model_name).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Optimizer
    optimizer = create_optimizer(model, opt_name, opt_cfg, track=True)
    
    # Scheduler
    total_steps = steps_per_epoch * cfg['epochs']
    warmup_steps = steps_per_epoch * cfg['warmup_epochs']
    
    def get_lr(step):
        if step < warmup_steps:
            return opt_cfg['lr'] * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return opt_cfg['lr'] * 0.5 * (1 + math.cos(math.pi * progress))

    # Training setup
    mixup = Mixup() if cfg['mixup'] else None
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    scaler = torch.amp.GradScaler('cuda')

    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'lr': [], 
               'naim': [], 'epoch_time': [], 'throughput': []}
    best_acc = 0.0
    step = 0

    for epoch in range(cfg['epochs']):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        t0 = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            if mixup:
                x, y_mixed = mixup(x, y)
                use_soft = True
            else:
                y_mixed, use_soft = None, False

            lr = get_lr(step)
            for g in optimizer.param_groups:
                g['lr'] = lr

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast('cuda', torch.bfloat16):
                out = model(x)
                if use_soft:
                    loss = -torch.sum(F.log_softmax(out, dim=1) * y_mixed, dim=1).mean()
                else:
                    loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()
            if not use_soft:
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
            step += 1

            if rank == 0 and (batch_idx + 1) % 100 == 0:
                logger.debug(f"E{epoch+1} S{batch_idx+1}/{steps_per_epoch}: loss={loss.item():.4f}, lr={lr:.2e}")

        # Epoch stats
        epoch_time = time.time() - t0
        throughput = len(train_loader.dataset) / epoch_time
        train_loss = loss_sum / steps_per_epoch
        train_acc = 100.0 * correct / total if total > 0 else 0.0

        # Validation
        base_model = model.module if hasattr(model, 'module') else model
        val_acc = evaluate(base_model, val_loader, device)

        # NAIM stats
        naim = {}
        if hasattr(optimizer, 'tracker') and optimizer.tracker:
            naim = optimizer.tracker.get_stats()
            optimizer.tracker.reset()

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(lr)
        history['naim'].append(naim)
        history['epoch_time'].append(epoch_time)
        history['throughput'].append(throughput)

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        # Log
        naim_str = f", belief={naim.get('belief_norm', 0):.3f}, align={naim.get('alignment', 0):.3f}" if naim else ""
        logger.info(f"E{epoch+1:3d}/{cfg['epochs']}: loss={train_loss:.4f}, train={train_acc:.1f}%, "
                   f"val={val_acc:.2f}%{'*' if is_best else ''}, {throughput:.0f} img/s{naim_str}")

        # Save best model
        if is_best and rank == 0:
            ckpt = {'epoch': epoch, 'model': base_model.state_dict(), 
                    'optimizer': optimizer.state_dict(), 'best_acc': best_acc}
            torch.save(ckpt, output_dir / f"{model_name}_{opt_name}_best.pt")

    # Save results
    if rank == 0:
        results = {
            'model': model_name, 'optimizer': opt_name, 'config': opt_cfg,
            'best_acc': best_acc, 'final_acc': history['val_acc'][-1],
            'total_time': sum(history['epoch_time']), 'history': history,
        }
        with open(output_dir / f"{model_name}_{opt_name}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Training complete. Best: {best_acc:.2f}%")

    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['resnet50', 'vit_s16', 'vit_b16'])
    parser.add_argument('--optimizer', default='all')
    parser.add_argument('--data', default='/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder')
    parser.add_argument('--output', default='./results_classification')
    args = parser.parse_args()

    # Setup
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Distributed
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl')
    else:
        rank, world_size, local_rank = 0, 1, 0

    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger = setup_logger(output_dir, rank, f"classification_{args.model}")
    logger.info(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
    logger.info(f"World size: {world_size} | Data: {data_path}")

    opts = list(CONFIGS[args.model]['optimizers'].keys()) if args.optimizer == 'all' else [args.optimizer]

    results = {}
    for opt in opts:
        try:
            acc = train_model(args.model, opt, data_path, output_dir, rank, world_size, local_rank, logger)
            results[opt] = acc
        except Exception as e:
            logger.error(f"Error {opt}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    if rank == 0 and results:
        logger.info("=" * 70)
        logger.info("FINAL RESULTS")
        logger.info("=" * 70)
        for opt, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {opt:<20}: {acc:.2f}%")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
