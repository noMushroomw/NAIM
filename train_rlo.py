#!/usr/bin/env python3
"""
RLO ImageNet Training - Following LION Paper Exactly
=====================================================

使用方法 (8 GPU):
    torchrun --nproc_per_node=8 train_rlo.py --experiment resnet50
    torchrun --nproc_per_node=8 train_rlo.py --experiment vit_s16
    torchrun --nproc_per_node=8 train_rlo.py --experiment vit_b16

LION Paper Settings (Section 4.1):
    - ResNet-50: 90 epochs, batch 1024, no augmentation
    - ViT-S/16, ViT-B/16: 300 epochs, batch 4096, RandAug + Mixup
    - Lion lr = 0.1x AdamW, Lion wd = 10x AdamW
"""

import os
import sys
import time
import math
import random
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

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
# 1. 优化器定义 (你的原始实现，不修改)
# =============================================================================

class RLO(Optimizer):
    """Riemannian Lyapunov Optimizer - 原始实现"""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1,
                 belief_coef=0.1, eps=1e-8):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                        belief_coef=belief_coef, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta1, beta2 = group["betas"]
            belief = group["belief_coef"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                m = state["exp_avg"]

                # Decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Compute direction
                c = beta1 * m + (1.0 - beta1) * g
                delta = g - m
                delta_norm = delta.norm(p=2).clamp(min=eps)
                d = c.sign() + belief * (delta / delta_norm)

                # Update
                p.add_(d, alpha=-lr)
                m.mul_(beta2).add_(g, alpha=(1.0 - beta2))

        return loss


class RLO_LambdaA(Optimizer):
    """RLO with Lambda-A preconditioning - 原始实现"""
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.99, beta3=0.999,
                 weight_decay=0.1, lambda_b=0.1, eps=1e-8, gamma=5.0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, beta3=beta3,
                        weight_decay=weight_decay, lambda_b=lambda_b,
                        eps=eps, gamma=gamma)
        super().__init__(params, defaults)
        self._init_sqrt_dim()

    def _init_sqrt_dim(self):
        total = sum(p.numel() for g in self.param_groups for p in g["params"])
        self.sqrt_dim = math.sqrt(total)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        all_smooth_pre = []
        all_belief = []
        all_params = []

        for group in self.param_groups:
            eps = group["eps"]
            gamma = group["gamma"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            beta3 = group["beta3"]
            lambda_b = group["lambda_b"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["s"] = torch.zeros_like(p)

                m = state["m"]
                s = state["s"]

                # Update second moment
                s.mul_(beta3).addcmul_(g, g, value=(1.0 - beta3))

                # Compute smooth direction
                c = beta1 * m + (1.0 - beta1) * g
                smooth = torch.tanh(gamma * c)
                smooth_pre = smooth / (s.sqrt() + eps)

                # Belief correction
                delta = g - m
                delta_norm = delta.norm(p=2).clamp(min=eps)
                belief = lambda_b * (delta / delta_norm)

                all_smooth_pre.append(smooth_pre)
                all_belief.append(belief)
                all_params.append((p, group))

        if not all_params:
            return loss

        # Global normalization
        s_norm = sum((sp * sp).sum() for sp in all_smooth_pre).sqrt().clamp(min=1e-8)
        scale = self.sqrt_dim / s_norm

        for (p, group), sp, b in zip(all_params, all_smooth_pre, all_belief):
            lr = group["lr"]
            wd = group["weight_decay"]
            beta2 = group["beta2"]

            d = scale * sp + b
            state = self.state[p]

            if wd != 0.0:
                p.mul_(1.0 - lr * wd)

            p.add_(d, alpha=-lr)
            state["m"].mul_(beta2).add_(p.grad, alpha=(1.0 - beta2))

        return loss


class SmoothLiftedRLO(Optimizer):
    """Smooth Lifted RLO with velocity tracking"""
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.99, eta=0.3,
                 weight_decay=0.1, lambda_b=0.1, eps=1e-8, gamma=5.0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eta=eta,
                        weight_decay=weight_decay, lambda_b=lambda_b,
                        eps=eps, gamma=gamma)
        super().__init__(params, defaults)
        self._init_sqrt_dim()

    def _init_sqrt_dim(self):
        total = sum(p.numel() for g in self.param_groups for p in g["params"])
        self.sqrt_dim = math.sqrt(total)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        all_s, all_b, all_params = [], [], []

        for group in self.param_groups:
            eps = group["eps"]
            gamma = group["gamma"]
            beta1 = group["beta1"]
            lambda_b = group["lambda_b"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                m = state["m"]
                c = beta1 * m + (1.0 - beta1) * g
                s = torch.tanh(gamma * c)

                delta = g - m
                delta_norm = delta.norm(p=2).clamp(min=eps)
                b = lambda_b * (delta / delta_norm)

                all_s.append(s)
                all_b.append(b)
                all_params.append((p, group))

        if not all_params:
            return loss

        s_norm = sum((s * s).sum() for s in all_s).sqrt().clamp(min=1e-8)
        scale = self.sqrt_dim / s_norm

        for (p, group), s, b in zip(all_params, all_s, all_b):
            lr = group["lr"]
            wd = group["weight_decay"]
            eta = group["eta"]
            beta2 = group["beta2"]

            d = scale * s + b
            state = self.state[p]
            m, v = state["m"], state["v"]

            if wd != 0.0:
                p.mul_(1.0 - lr * wd)

            v.mul_(1.0 - eta).add_(d, alpha=eta)
            p.add_(v, alpha=-lr)
            m.mul_(beta2).add_(p.grad, alpha=(1.0 - beta2))

        return loss


class Lion(Optimizer):
    """Lion optimizer (Google, 2023)"""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Decoupled weight decay
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                m = state['exp_avg']
                beta1, beta2 = group['betas']

                # Update: sign of interpolation
                update = (beta1 * m + (1 - beta1) * g).sign_()
                p.add_(update, alpha=-group['lr'])

                # Momentum update
                m.mul_(beta2).add_(g, alpha=1 - beta2)


# =============================================================================
# 2. 模型定义
# =============================================================================

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # Use Flash Attention if available
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(x))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT)"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.head(x[:, 0])


def create_model(name: str, num_classes: int = 1000):
    """Create model by name"""
    if name == 'resnet50':
        return resnet50(weights=None, num_classes=num_classes)
    elif name == 'vit_s16':
        # ViT-S/16: embed_dim=384, depth=12, heads=6
        return VisionTransformer(embed_dim=384, depth=12, num_heads=6)
    elif name == 'vit_b16':
        # ViT-B/16: embed_dim=768, depth=12, heads=12
        return VisionTransformer(embed_dim=768, depth=12, num_heads=12)
    else:
        raise ValueError(f"Unknown model: {name}")


# =============================================================================
# 3. 数据加载
# =============================================================================

def create_dataloader(
    data_path: Path,
    batch_size: int,
    is_train: bool,
    distributed: bool,
    num_workers: int = 8,
    use_randaugment: bool = False,
    use_mixup: bool = False,
):
    """Create dataloader following LION paper settings"""
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    if is_train:
        transforms_list = [
            T.RandomResizedCrop(224, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
        ]
        if use_randaugment:
            transforms_list.append(T.RandAugment(num_ops=2, magnitude=9))
        transforms_list.extend([
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
        transform = T.Compose(transforms_list)
        split = 'train'
    else:
        transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
        split = 'val'

    dataset = ImageFolder(data_path / split, transform)

    if distributed:
        sampler = DistributedSampler(dataset, shuffle=is_train)
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and is_train),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
        persistent_workers=True if num_workers > 0 else False,
    )

    return loader, sampler


class Mixup:
    """Mixup/CutMix data augmentation"""
    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, num_classes=1000):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.num_classes = num_classes

    def __call__(self, x, target):
        # Choose mixup or cutmix
        use_cutmix = random.random() < 0.5
        alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
        lam = np.random.beta(alpha, alpha)

        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        if use_cutmix:
            # CutMix
            _, _, H, W = x.shape
            cut_rat = math.sqrt(1 - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)
            cx = random.randint(0, W)
            cy = random.randint(0, H)
            x1 = max(cx - cut_w // 2, 0)
            y1 = max(cy - cut_h // 2, 0)
            x2 = min(cx + cut_w // 2, W)
            y2 = min(cy + cut_h // 2, H)
            x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
            lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
        else:
            # Mixup
            x = lam * x + (1 - lam) * x[index]

        # Create soft labels
        y = F.one_hot(target, self.num_classes).float()
        y_shuffled = F.one_hot(target[index], self.num_classes).float()
        y_mixed = lam * y + (1 - lam) * y_shuffled

        return x, y_mixed


# =============================================================================
# 4. 训练配置 (严格遵循LION论文)
# =============================================================================

# LION Paper Table 1 hyperparameters
EXPERIMENT_CONFIGS = {
    'resnet50': {
        'model': 'resnet50',
        'epochs': 90,
        'batch_size': 1024,  # Total batch size
        'warmup_epochs': 5,
        'use_randaugment': False,
        'use_mixup': False,
        'label_smoothing': 0.0,
        'optimizers': {
            'adamw': {'lr': 1e-3, 'wd': 0.05, 'betas': (0.9, 0.999)},
            'lion': {'lr': 1e-4, 'wd': 0.5, 'betas': (0.9, 0.99)},  # 10x smaller lr, 10x larger wd
            'rlo': {'lr': 1e-4, 'wd': 0.5, 'betas': (0.9, 0.99)},
            'rlo_lambda_a': {'lr': 1e-4, 'wd': 0.5},
            'smooth_lifted_rlo': {'lr': 1e-4, 'wd': 0.5},
        },
    },
    'vit_s16': {
        'model': 'vit_s16',
        'epochs': 300,
        'batch_size': 4096,  # LION paper uses 4096 for ViT
        'warmup_epochs': 30,
        'use_randaugment': True,
        'use_mixup': True,
        'label_smoothing': 0.1,
        'optimizers': {
            'adamw': {'lr': 1e-3, 'wd': 0.1, 'betas': (0.9, 0.999)},
            'lion': {'lr': 1e-4, 'wd': 1.0, 'betas': (0.9, 0.99)},
            'rlo': {'lr': 1e-4, 'wd': 1.0, 'betas': (0.9, 0.99)},
            'rlo_lambda_a': {'lr': 1e-4, 'wd': 1.0},
            'smooth_lifted_rlo': {'lr': 1e-4, 'wd': 1.0},
        },
    },
    'vit_b16': {
        'model': 'vit_b16',
        'epochs': 300,
        'batch_size': 4096,
        'warmup_epochs': 30,
        'use_randaugment': True,
        'use_mixup': True,
        'label_smoothing': 0.1,
        'optimizers': {
            'adamw': {'lr': 1e-3, 'wd': 0.1, 'betas': (0.9, 0.999)},
            'lion': {'lr': 1e-4, 'wd': 1.0, 'betas': (0.9, 0.99)},
            'rlo': {'lr': 1e-4, 'wd': 1.0, 'betas': (0.9, 0.99)},
            'rlo_lambda_a': {'lr': 1e-4, 'wd': 1.0},
            'smooth_lifted_rlo': {'lr': 1e-4, 'wd': 1.0},
        },
    },
}


def create_optimizer(model, opt_name: str, config: dict):
    """Create optimizer with LION paper settings"""
    params = model.parameters()
    lr = config['lr']
    wd = config['wd']

    if opt_name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=config.get('betas', (0.9, 0.999)))
    elif opt_name == 'lion':
        return Lion(params, lr=lr, weight_decay=wd, betas=config.get('betas', (0.9, 0.99)))
    elif opt_name == 'rlo':
        return RLO(params, lr=lr, weight_decay=wd, betas=config.get('betas', (0.9, 0.99)))
    elif opt_name == 'rlo_lambda_a':
        return RLO_LambdaA(params, lr=lr, weight_decay=wd)
    elif opt_name == 'smooth_lifted_rlo':
        return SmoothLiftedRLO(params, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


# =============================================================================
# 5. 训练循环
# =============================================================================

def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')

    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(images)

        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total


def train_one_epoch(
    model, train_loader, optimizer, scheduler, device, epoch,
    mixup=None, label_smoothing=0.0, rank=0
):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    step = 0

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler('cuda')

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mixup
        if mixup is not None:
            images, labels_mixed = mixup(images, labels)
            use_soft_labels = True
        else:
            labels_mixed = None
            use_soft_labels = False

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(images)
            if use_soft_labels:
                loss = -torch.sum(F.log_softmax(logits, dim=1) * labels_mixed, dim=1).mean()
            else:
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Update scheduler per step
        scheduler.step()

        # Metrics
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        step += 1

        # Print progress
        if rank == 0 and step % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Step {step}/{len(train_loader)}: loss={total_loss/step:.4f}, "
                  f"acc={100*correct/total:.1f}%, lr={lr:.2e}")

    return total_loss / step, 100.0 * correct / total


class CosineScheduler:
    """Cosine learning rate scheduler with warmup"""
    def __init__(self, optimizer, base_lr, total_steps, warmup_steps):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.base_lr * self.current_step / max(1, self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train(
    experiment: str,
    opt_name: str,
    data_path: Path,
    output_dir: Path,
    rank: int,
    world_size: int,
    local_rank: int,
):
    """Main training function"""
    device = torch.device(f'cuda:{local_rank}')
    config = EXPERIMENT_CONFIGS[experiment]
    opt_config = config['optimizers'][opt_name]

    # Adjust batch size for DDP
    batch_size_per_gpu = config['batch_size'] // world_size

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Experiment: {experiment}")
        print(f"Optimizer: {opt_name}")
        print(f"Config: lr={opt_config['lr']}, wd={opt_config['wd']}")
        print(f"Batch size: {config['batch_size']} (per GPU: {batch_size_per_gpu})")
        print(f"Epochs: {config['epochs']}")
        print(f"{'='*70}\n")

    # Data loaders
    train_loader, train_sampler = create_dataloader(
        data_path,
        batch_size=batch_size_per_gpu,
        is_train=True,
        distributed=(world_size > 1),
        num_workers=8,
        use_randaugment=config['use_randaugment'],
    )

    val_loader, _ = create_dataloader(
        data_path,
        batch_size=batch_size_per_gpu * 2,
        is_train=False,
        distributed=False,  # Validate on all data per GPU
        num_workers=8,
    )

    if rank == 0:
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Steps per epoch: {len(train_loader)}")

    # Model
    model = create_model(config['model'])
    model = model.to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Count params
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(f"Model params: {n_params/1e6:.1f}M")

    # Optimizer
    optimizer = create_optimizer(model, opt_name, opt_config)

    # Scheduler
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = len(train_loader) * config['warmup_epochs']
    scheduler = CosineScheduler(optimizer, opt_config['lr'], total_steps, warmup_steps)

    # Mixup
    mixup = Mixup() if config['use_mixup'] else None

    # Training
    best_acc = 0.0
    history = {'train_acc': [], 'val_acc': [], 'train_loss': []}

    for epoch in range(config['epochs']):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            mixup=mixup, label_smoothing=config['label_smoothing'], rank=rank
        )

        # Evaluate
        model_to_eval = model.module if hasattr(model, 'module') else model
        val_acc = evaluate(model_to_eval, val_loader, device)

        # Gather metrics across GPUs
        if world_size > 1:
            val_acc_tensor = torch.tensor([val_acc], device=device)
            dist.all_reduce(val_acc_tensor, op=dist.ReduceOp.SUM)
            val_acc = val_acc_tensor.item() / world_size

        epoch_time = time.time() - t0
        throughput = len(train_loader.dataset) / epoch_time

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        if rank == 0:
            print(f"Epoch {epoch+1}/{config['epochs']}: "
                  f"train_acc={train_acc:.2f}%, val_acc={val_acc:.2f}%{'*' if is_best else ''}, "
                  f"throughput={throughput:.0f} img/s")

            # Save checkpoint
            if is_best:
                save_path = output_dir / f"{experiment}_{opt_name}_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_eval.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'config': config,
                }, save_path)

    # Save final results
    if rank == 0:
        results = {
            'experiment': experiment,
            'optimizer': opt_name,
            'config': opt_config,
            'best_acc': best_acc,
            'final_acc': history['val_acc'][-1],
            'history': history,
        }
        with open(output_dir / f"{experiment}_{opt_name}_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Training complete!")
        print(f"Best accuracy: {best_acc:.2f}%")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*70}\n")

    return best_acc


def main():
    parser = argparse.ArgumentParser(description='RLO ImageNet Training')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['resnet50', 'vit_s16', 'vit_b16'],
                        help='Experiment to run')
    parser.add_argument('--optimizer', type=str, default='all',
                        choices=['all', 'adamw', 'lion', 'rlo', 'rlo_lambda_a', 'smooth_lifted_rlo'],
                        help='Optimizer to use')
    parser.add_argument('--data', type=str, default='/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder',
                        help='Path to ImageNet data')
    parser.add_argument('--output', type=str, default='./results',
                        help='Output directory')
    args = parser.parse_args()

    # Setup
    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # GPU optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    # Set seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Initialize distributed
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"World size: {world_size}")
        print(f"Data path: {data_path}")

    # Determine optimizers to run
    if args.optimizer == 'all':
        optimizers = ['adamw', 'lion', 'rlo', 'rlo_lambda_a', 'smooth_lifted_rlo']
    else:
        optimizers = [args.optimizer]

    # Run experiments
    results = {}
    for opt_name in optimizers:
        try:
            best_acc = train(
                experiment=args.experiment,
                opt_name=opt_name,
                data_path=data_path,
                output_dir=output_dir,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
            )
            results[opt_name] = best_acc
        except Exception as e:
            if rank == 0:
                print(f"Error training {opt_name}: {e}")
                import traceback
                traceback.print_exc()

    # Print summary
    if rank == 0 and results:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        for opt, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"  {opt:<20}: {acc:.2f}%")
        print("="*70)

    cleanup_distributed()


if __name__ == '__main__':
    main()
