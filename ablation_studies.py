#!/usr/bin/env python3
"""
Ablation Studies for RLO
========================

Studies include:
1. Belief coefficient (λ_b) sensitivity: [0.0, 0.05, 0.1, 0.2, 0.5]
2. Gamma (smoothness) sensitivity: [1.0, 3.0, 5.0, 10.0]
3. Learning rate sensitivity grid
4. Batch size effect: [64, 256, 1024, 4096]
5. Component analysis: RLO vs RLO without belief vs LION

Usage:
    torchrun --nproc_per_node=8 ablation_studies.py --study belief_coef
    torchrun --nproc_per_node=8 ablation_studies.py --study gamma
    torchrun --nproc_per_node=8 ablation_studies.py --study lr_grid
    torchrun --nproc_per_node=8 ablation_studies.py --study batch_size
    torchrun --nproc_per_node=8 ablation_studies.py --study all
"""

import os
import sys
import time
import math
import random
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
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
from torchvision.models import resnet50
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. Optimizers with Configurable Parameters
# =============================================================================

class RLO(Optimizer):
    """RLO with configurable belief_coef"""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1,
                 belief_coef=0.1, eps=1e-8):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                        belief_coef=belief_coef, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
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
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                c = beta1 * m + (1.0 - beta1) * g
                delta = g - m
                delta_norm = delta.norm(p=2).clamp(min=eps)
                d = c.sign() + belief * (delta / delta_norm)
                p.add_(d, alpha=-lr)
                m.mul_(beta2).add_(g, alpha=(1.0 - beta2))


class RLO_LambdaA(Optimizer):
    """RLO_LambdaA with configurable gamma and lambda_b"""
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.99, beta3=0.999,
                 weight_decay=0.1, lambda_b=0.1, eps=1e-8, gamma=5.0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, beta3=beta3,
                        weight_decay=weight_decay, lambda_b=lambda_b, eps=eps, gamma=gamma)
        super().__init__(params, defaults)
        self._init_sqrt_dim()

    def _init_sqrt_dim(self):
        total = sum(p.numel() for g in self.param_groups for p in g["params"])
        self.sqrt_dim = math.sqrt(total)

    @torch.no_grad()
    def step(self, closure=None):
        all_smooth_pre, all_belief, all_params = [], [], []
        for group in self.param_groups:
            eps, gamma = group["eps"], group["gamma"]
            beta1, beta2, beta3, lambda_b = group["beta1"], group["beta2"], group["beta3"], group["lambda_b"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["s"] = torch.zeros_like(p)
                m, s = state["m"], state["s"]
                s.mul_(beta3).addcmul_(g, g, value=(1.0 - beta3))
                c = beta1 * m + (1.0 - beta1) * g
                smooth = torch.tanh(gamma * c)
                smooth_pre = smooth / (s.sqrt() + eps)
                delta = g - m
                delta_norm = delta.norm(p=2).clamp(min=eps)
                belief = lambda_b * (delta / delta_norm)
                all_smooth_pre.append(smooth_pre)
                all_belief.append(belief)
                all_params.append((p, group))
        if not all_params:
            return
        s_norm = sum((sp * sp).sum() for sp in all_smooth_pre).sqrt().clamp(min=1e-8)
        scale = self.sqrt_dim / s_norm
        for (p, group), sp, b in zip(all_params, all_smooth_pre, all_belief):
            lr, wd, beta2 = group["lr"], group["weight_decay"], group["beta2"]
            d = scale * sp + b
            state = self.state[p]
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            p.add_(d, alpha=-lr)
            state["m"].mul_(beta2).add_(p.grad, alpha=(1.0 - beta2))


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
                    state['exp_avg'] = torch.zeros_like(p)
                m = state['exp_avg']
                beta1, beta2 = group['betas']
                update = (beta1 * m + (1 - beta1) * g).sign_()
                p.add_(update, alpha=-group['lr'])
                m.mul_(beta2).add_(g, alpha=1 - beta2)


# =============================================================================
# 2. ViT Model (for ablation studies)
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
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViT_Small(nn.Module):
    """ViT-S/16 for ablation studies"""
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
# 3. Training Utilities
# =============================================================================

def create_dataloader(data_path, batch_size, is_train, world_size, rank):
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    
    if is_train:
        transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.RandAugment(2, 9),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
        split = 'train'
    else:
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
        split = 'val'
    
    dataset = ImageFolder(data_path / split, transform)
    
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=is_train)
    else:
        sampler = None
    
    loader = DataLoader(
        dataset, batch_size=batch_size // world_size,
        shuffle=(sampler is None and is_train),
        sampler=sampler, num_workers=8, pin_memory=True, drop_last=is_train,
    )
    return loader, sampler


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast('cuda', torch.bfloat16):
            out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def train_one_config(
    model_fn,
    optimizer_fn,
    data_path: Path,
    epochs: int = 30,
    batch_size: int = 1024,
    warmup_epochs: int = 3,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
) -> Dict:
    """Train with one configuration and return results"""
    device = torch.device(f'cuda:{local_rank}')
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Data
    train_loader, train_sampler = create_dataloader(data_path, batch_size, True, world_size, rank)
    val_loader, _ = create_dataloader(data_path, batch_size, False, world_size, rank)
    
    # Model
    model = model_fn().to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # Optimizer
    base_model = model.module if hasattr(model, 'module') else model
    optimizer, base_lr = optimizer_fn(base_model.parameters())
    
    # Training
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * warmup_epochs
    scaler = torch.amp.GradScaler('cuda')
    
    history = {'train_acc': [], 'val_acc': []}
    global_step = 0
    
    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        model.train()
        correct, total = 0, 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # LR schedule
            if global_step < warmup_steps:
                lr = base_lr * global_step / warmup_steps
            else:
                progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
                lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
            
            for g in optimizer.param_groups:
                g['lr'] = lr
            
            optimizer.zero_grad()
            
            with torch.autocast('cuda', torch.bfloat16):
                out = model(x)
                loss = F.cross_entropy(out, y, label_smoothing=0.1)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
            global_step += 1
        
        train_acc = 100.0 * correct / total
        val_acc = evaluate(base_model, val_loader, device)
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if rank == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train={train_acc:.1f}%, val={val_acc:.1f}%")
    
    return {
        'best_val_acc': max(history['val_acc']),
        'final_val_acc': history['val_acc'][-1],
        'history': history,
    }


# =============================================================================
# 4. Ablation Study Functions
# =============================================================================

def ablation_belief_coef(data_path: Path, output_dir: Path, rank: int, world_size: int, local_rank: int):
    """Study 1: Belief coefficient sensitivity"""
    belief_coefs = [0.0, 0.05, 0.1, 0.2, 0.5]
    
    if rank == 0:
        print("\n" + "="*70)
        print("ABLATION STUDY 1: Belief Coefficient (λ_b)")
        print("="*70)
    
    results = {}
    
    for belief in belief_coefs:
        if rank == 0:
            print(f"\nTraining with belief_coef={belief}")
        
        def opt_fn(params):
            return RLO(params, lr=1e-4, weight_decay=1.0, belief_coef=belief), 1e-4
        
        result = train_one_config(
            ViT_Small, opt_fn, data_path, epochs=30, batch_size=1024,
            rank=rank, world_size=world_size, local_rank=local_rank
        )
        results[f"belief_{belief}"] = result
    
    if rank == 0:
        # Save results
        with open(output_dir / "ablation_belief_coef.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        beliefs = [float(k.split('_')[1]) for k in results.keys()]
        accs = [v['best_val_acc'] for v in results.values()]
        ax.plot(beliefs, accs, 'o-', lw=2, markersize=10)
        ax.set_xlabel('Belief Coefficient (λ_b)', fontsize=12)
        ax.set_ylabel('Best Validation Accuracy (%)', fontsize=12)
        ax.set_title('RLO: Effect of Belief Coefficient', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "ablation_belief_coef.png", dpi=150)
        plt.close()
        
        print("\nResults:")
        for belief, res in zip(beliefs, results.values()):
            print(f"  λ_b={belief}: {res['best_val_acc']:.2f}%")
    
    return results


def ablation_gamma(data_path: Path, output_dir: Path, rank: int, world_size: int, local_rank: int):
    """Study 2: Gamma (smoothness) sensitivity"""
    gammas = [1.0, 3.0, 5.0, 10.0, 20.0]
    
    if rank == 0:
        print("\n" + "="*70)
        print("ABLATION STUDY 2: Gamma (Smoothness Parameter)")
        print("="*70)
    
    results = {}
    
    for gamma in gammas:
        if rank == 0:
            print(f"\nTraining with gamma={gamma}")
        
        def opt_fn(params):
            return RLO_LambdaA(params, lr=1e-4, weight_decay=1.0, gamma=gamma), 1e-4
        
        result = train_one_config(
            ViT_Small, opt_fn, data_path, epochs=30, batch_size=1024,
            rank=rank, world_size=world_size, local_rank=local_rank
        )
        results[f"gamma_{gamma}"] = result
    
    if rank == 0:
        with open(output_dir / "ablation_gamma.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        gs = [float(k.split('_')[1]) for k in results.keys()]
        accs = [v['best_val_acc'] for v in results.values()]
        ax.plot(gs, accs, 'o-', lw=2, markersize=10, color='green')
        ax.set_xlabel('Gamma (γ)', fontsize=12)
        ax.set_ylabel('Best Validation Accuracy (%)', fontsize=12)
        ax.set_title('RLO-LambdaA: Effect of Gamma', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "ablation_gamma.png", dpi=150)
        plt.close()
        
        print("\nResults:")
        for g, res in zip(gs, results.values()):
            print(f"  γ={g}: {res['best_val_acc']:.2f}%")
    
    return results


def ablation_lr_grid(data_path: Path, output_dir: Path, rank: int, world_size: int, local_rank: int):
    """Study 3: Learning rate sensitivity grid"""
    lrs = [3e-5, 1e-4, 3e-4, 1e-3]
    wds = [0.1, 0.5, 1.0, 2.0]
    
    if rank == 0:
        print("\n" + "="*70)
        print("ABLATION STUDY 3: Learning Rate x Weight Decay Grid")
        print("="*70)
    
    results = {}
    
    for lr in lrs:
        for wd in wds:
            if rank == 0:
                print(f"\nTraining with lr={lr}, wd={wd}")
            
            def opt_fn(params, _lr=lr, _wd=wd):
                return RLO(params, lr=_lr, weight_decay=_wd), _lr
            
            result = train_one_config(
                ViT_Small, opt_fn, data_path, epochs=30, batch_size=1024,
                rank=rank, world_size=world_size, local_rank=local_rank
            )
            results[f"lr{lr}_wd{wd}"] = result
    
    if rank == 0:
        with open(output_dir / "ablation_lr_grid.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Heatmap
        acc_matrix = np.zeros((len(lrs), len(wds)))
        for i, lr in enumerate(lrs):
            for j, wd in enumerate(wds):
                acc_matrix[i, j] = results[f"lr{lr}_wd{wd}"]['best_val_acc']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(acc_matrix, cmap='YlGn')
        ax.set_xticks(range(len(wds)))
        ax.set_yticks(range(len(lrs)))
        ax.set_xticklabels([str(w) for w in wds])
        ax.set_yticklabels([f"{l:.0e}" for l in lrs])
        ax.set_xlabel('Weight Decay', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('RLO: LR x WD Sensitivity (Val Acc %)', fontsize=14)
        
        # Add text annotations
        for i in range(len(lrs)):
            for j in range(len(wds)):
                ax.text(j, i, f"{acc_matrix[i,j]:.1f}", ha='center', va='center')
        
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(output_dir / "ablation_lr_grid.png", dpi=150)
        plt.close()
        
        # Find best
        best_idx = np.unravel_index(np.argmax(acc_matrix), acc_matrix.shape)
        print(f"\nBest: lr={lrs[best_idx[0]]}, wd={wds[best_idx[1]]} -> {acc_matrix[best_idx]:.2f}%")
    
    return results


def ablation_batch_size(data_path: Path, output_dir: Path, rank: int, world_size: int, local_rank: int):
    """Study 4: Batch size effect"""
    batch_sizes = [256, 512, 1024, 2048, 4096]
    
    if rank == 0:
        print("\n" + "="*70)
        print("ABLATION STUDY 4: Batch Size Effect")
        print("="*70)
    
    results = {}
    
    for bs in batch_sizes:
        if rank == 0:
            print(f"\nTraining with batch_size={bs}")
        
        # Scale LR with batch size (linear scaling rule)
        scaled_lr = 1e-4 * (bs / 1024)
        
        def opt_fn(params, _lr=scaled_lr):
            return RLO(params, lr=_lr, weight_decay=1.0), _lr
        
        result = train_one_config(
            ViT_Small, opt_fn, data_path, epochs=30, batch_size=bs,
            rank=rank, world_size=world_size, local_rank=local_rank
        )
        results[f"bs_{bs}"] = result
    
    if rank == 0:
        with open(output_dir / "ablation_batch_size.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bs_list = [int(k.split('_')[1]) for k in results.keys()]
        accs = [v['best_val_acc'] for v in results.values()]
        ax.semilogx(bs_list, accs, 'o-', lw=2, markersize=10, color='red', base=2)
        ax.set_xlabel('Batch Size', fontsize=12)
        ax.set_ylabel('Best Validation Accuracy (%)', fontsize=12)
        ax.set_title('RLO: Effect of Batch Size', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "ablation_batch_size.png", dpi=150)
        plt.close()
        
        print("\nResults:")
        for bs, res in zip(bs_list, results.values()):
            print(f"  batch_size={bs}: {res['best_val_acc']:.2f}%")
    
    return results


def ablation_component_analysis(data_path: Path, output_dir: Path, rank: int, world_size: int, local_rank: int):
    """Study 5: Component analysis - what makes RLO better than LION?"""
    
    if rank == 0:
        print("\n" + "="*70)
        print("ABLATION STUDY 5: Component Analysis")
        print("="*70)
    
    results = {}
    
    # 1. LION (baseline)
    if rank == 0:
        print("\nTraining LION (baseline)")
    
    def lion_fn(params):
        return Lion(params, lr=1e-4, weight_decay=1.0), 1e-4
    
    results['lion'] = train_one_config(
        ViT_Small, lion_fn, data_path, epochs=30, batch_size=1024,
        rank=rank, world_size=world_size, local_rank=local_rank
    )
    
    # 2. RLO without belief (equivalent to LION)
    if rank == 0:
        print("\nTraining RLO (belief=0, should match LION)")
    
    def rlo_no_belief_fn(params):
        return RLO(params, lr=1e-4, weight_decay=1.0, belief_coef=0.0), 1e-4
    
    results['rlo_no_belief'] = train_one_config(
        ViT_Small, rlo_no_belief_fn, data_path, epochs=30, batch_size=1024,
        rank=rank, world_size=world_size, local_rank=local_rank
    )
    
    # 3. RLO with belief
    if rank == 0:
        print("\nTraining RLO (with belief correction)")
    
    def rlo_fn(params):
        return RLO(params, lr=1e-4, weight_decay=1.0, belief_coef=0.1), 1e-4
    
    results['rlo'] = train_one_config(
        ViT_Small, rlo_fn, data_path, epochs=30, batch_size=1024,
        rank=rank, world_size=world_size, local_rank=local_rank
    )
    
    # 4. RLO_LambdaA
    if rank == 0:
        print("\nTraining RLO_LambdaA (with adaptive preconditioning)")
    
    def rlo_lambda_fn(params):
        return RLO_LambdaA(params, lr=1e-4, weight_decay=1.0), 1e-4
    
    results['rlo_lambda_a'] = train_one_config(
        ViT_Small, rlo_lambda_fn, data_path, epochs=30, batch_size=1024,
        rank=rank, world_size=world_size, local_rank=local_rank
    )
    
    # 5. AdamW (reference)
    if rank == 0:
        print("\nTraining AdamW (reference)")
    
    def adamw_fn(params):
        return torch.optim.AdamW(params, lr=1e-3, weight_decay=0.1), 1e-3
    
    results['adamw'] = train_one_config(
        ViT_Small, adamw_fn, data_path, epochs=30, batch_size=1024,
        rank=rank, world_size=world_size, local_rank=local_rank
    )
    
    if rank == 0:
        with open(output_dir / "ablation_components.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = {'adamw': '#1f77b4', 'lion': '#ff7f0e', 'rlo_no_belief': '#d62728',
                  'rlo': '#2ca02c', 'rlo_lambda_a': '#9467bd'}
        
        for name, res in results.items():
            ax.plot(res['history']['val_acc'], label=f"{name} ({res['best_val_acc']:.1f}%)",
                   color=colors.get(name, 'gray'), lw=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
        ax.set_title('Component Analysis: What Makes RLO Better?', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "ablation_components.png", dpi=150)
        plt.close()
        
        print("\n" + "="*60)
        print("COMPONENT ANALYSIS RESULTS")
        print("="*60)
        for name, res in sorted(results.items(), key=lambda x: x[1]['best_val_acc'], reverse=True):
            print(f"  {name:<20}: {res['best_val_acc']:.2f}%")
        print("="*60)
        print("\nKey Finding:")
        belief_gain = results['rlo']['best_val_acc'] - results['rlo_no_belief']['best_val_acc']
        print(f"  Belief correction adds {belief_gain:+.2f}% over LION-equivalent")
    
    return results


# =============================================================================
# 5. Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', default='all', 
                       choices=['all', 'belief_coef', 'gamma', 'lr_grid', 'batch_size', 'components'])
    parser.add_argument('--data', default='/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder')
    parser.add_argument('--output', default='./results_ablation')
    args = parser.parse_args()
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
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
    
    studies = {
        'belief_coef': ablation_belief_coef,
        'gamma': ablation_gamma,
        'lr_grid': ablation_lr_grid,
        'batch_size': ablation_batch_size,
        'components': ablation_component_analysis,
    }
    
    if args.study == 'all':
        for name, fn in studies.items():
            try:
                fn(data_path, output_dir, rank, world_size, local_rank)
            except Exception as e:
                if rank == 0:
                    print(f"Error in {name}: {e}")
                    import traceback
                    traceback.print_exc()
    else:
        studies[args.study](data_path, output_dir, rank, world_size, local_rank)
    
    if rank == 0:
        print("\n" + "="*70)
        print("ALL ABLATION STUDIES COMPLETE")
        print(f"Results saved to: {output_dir}")
        print("="*70)
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
