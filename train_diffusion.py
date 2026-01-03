#!/usr/bin/env python3
"""
Section 4.3: Image Generation with Diffusion Models
====================================================

Following LION paper Section 4.3:
- U-Net based diffusion model
- ImageNet 64x64, 128x128, 256x256
- FID score comparison

Usage:
    torchrun --nproc_per_node=8 train_diffusion.py --resolution 64 --optimizer all
    torchrun --nproc_per_node=8 train_diffusion.py --resolution 128 --optimizer all
    torchrun --nproc_per_node=8 train_diffusion.py --resolution 256 --optimizer all
"""

import os
import sys
import time
import math
import random
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
from functools import partial

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
# 1. Optimizers
# =============================================================================

class RLO(Optimizer):
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


class SmoothLiftedRLO(Optimizer):
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.99, eta=0.3,
                 weight_decay=0.1, lambda_b=0.1, eps=1e-8, gamma=5.0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eta=eta,
                        weight_decay=weight_decay, lambda_b=lambda_b, eps=eps, gamma=gamma)
        super().__init__(params, defaults)
        self._init_sqrt_dim()

    def _init_sqrt_dim(self):
        total = sum(p.numel() for g in self.param_groups for p in g["params"])
        self.sqrt_dim = math.sqrt(total)

    @torch.no_grad()
    def step(self, closure=None):
        all_s, all_b, all_params = [], [], []
        for group in self.param_groups:
            eps, gamma, beta1, lambda_b = group["eps"], group["gamma"], group["beta1"], group["lambda_b"]
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
            return
        s_norm = sum((s * s).sum() for s in all_s).sqrt().clamp(min=1e-8)
        scale = self.sqrt_dim / s_norm
        for (p, group), s, b in zip(all_params, all_s, all_b):
            lr, wd, eta, beta2 = group["lr"], group["weight_decay"], group["eta"], group["beta2"]
            d = scale * s + b
            state = self.state[p]
            m, v = state["m"], state["v"]
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            v.mul_(1.0 - eta).add_(d, alpha=eta)
            p.add_(v, alpha=-lr)
            m.mul_(beta2).add_(p.grad, alpha=(1.0 - beta2))


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
# 2. Diffusion Utilities
# =============================================================================

def get_beta_schedule(schedule_type='linear', num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
    """Get noise schedule"""
    if schedule_type == 'linear':
        return torch.linspace(beta_start, beta_end, num_timesteps)
    elif schedule_type == 'cosine':
        steps = num_timesteps + 1
        s = 0.008
        t = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos(((t / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    else:
        raise ValueError(f"Unknown schedule: {schedule_type}")


class GaussianDiffusion:
    """DDPM-style Gaussian Diffusion"""
    def __init__(self, num_timesteps=1000, beta_schedule='cosine'):
        self.num_timesteps = num_timesteps
        
        betas = get_beta_schedule(beta_schedule, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('posterior_variance', 
                            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
    
    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)
    
    def to(self, device):
        for name in ['betas', 'alphas_cumprod', 'alphas_cumprod_prev', 
                     'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 
                     'posterior_variance']:
            setattr(self, name, getattr(self, name).to(device))
        return self
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas * x_start + sqrt_one_minus * noise
    
    def training_loss(self, model, x_start, t, noise=None, class_labels=None):
        """Training loss: predict noise"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = model(x_noisy, t, class_labels)
        
        loss = F.mse_loss(predicted_noise, noise)
        return loss


# =============================================================================
# 3. U-Net Architecture
# =============================================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h = h + self.time_mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.shortcut(x)


class Attention2D(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv.unbind(1)
        q = q.permute(0, 1, 3, 2)  # B, heads, HW, dim
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(attn)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, has_attn=False, num_heads=4):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, time_dim)
        self.attn = Attention2D(out_ch, num_heads) if has_attn else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t):
        x = self.res(x, t)
        x = self.attn(x)
        return self.down(x), x


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, has_attn=False, num_heads=4):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res = ResBlock(in_ch + out_ch, out_ch, time_dim)
        self.attn = Attention2D(out_ch, num_heads) if has_attn else nn.Identity()

    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t)
        return self.attn(x)


class UNet(nn.Module):
    """U-Net for diffusion models"""
    def __init__(self, in_channels=3, out_channels=3, base_dim=128, 
                 dim_mults=(1, 2, 4, 8), num_classes=1000, dropout=0.1):
        super().__init__()
        
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Class embedding
        self.class_emb = nn.Embedding(num_classes, time_dim)
        
        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, base_dim, 3, padding=1)
        
        # Downsampling
        dims = [base_dim] + [base_dim * m for m in dim_mults]
        self.downs = nn.ModuleList()
        for i in range(len(dim_mults)):
            has_attn = i >= 2  # Attention at lower resolutions
            self.downs.append(DownBlock(dims[i], dims[i+1], time_dim, has_attn))
        
        # Middle
        mid_dim = dims[-1]
        self.mid_res1 = ResBlock(mid_dim, mid_dim, time_dim)
        self.mid_attn = Attention2D(mid_dim)
        self.mid_res2 = ResBlock(mid_dim, mid_dim, time_dim)
        
        # Upsampling
        self.ups = nn.ModuleList()
        for i in reversed(range(len(dim_mults))):
            has_attn = i >= 2
            self.ups.append(UpBlock(dims[i+1], dims[i], time_dim, has_attn))
        
        # Final
        self.final = nn.Sequential(
            nn.GroupNorm(32, base_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, out_channels, 3, padding=1),
        )

    def forward(self, x, t, class_labels=None):
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Add class embedding
        if class_labels is not None:
            t_emb = t_emb + self.class_emb(class_labels)
        
        # Initial
        x = self.init_conv(x)
        
        # Downsampling
        skips = []
        for down in self.downs:
            x, skip = down(x, t_emb)
            skips.append(skip)
        
        # Middle
        x = self.mid_res1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_res2(x, t_emb)
        
        # Upsampling
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip, t_emb)
        
        return self.final(x)


# =============================================================================
# 4. Training
# =============================================================================

def create_optimizer(model, opt_name: str, lr: float, wd: float):
    params = model.parameters()
    if opt_name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif opt_name == 'lion':
        return Lion(params, lr=lr, weight_decay=wd)
    elif opt_name == 'rlo':
        return RLO(params, lr=lr, weight_decay=wd)
    elif opt_name == 'rlo_lambda_a':
        return RLO_LambdaA(params, lr=lr, weight_decay=wd)
    elif opt_name == 'smooth_lifted_rlo':
        return SmoothLiftedRLO(params, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def train_diffusion(
    opt_name: str,
    data_path: Path,
    output_dir: Path,
    resolution: int = 64,
    epochs: int = 100,
    batch_size: int = 256,
    warmup_steps: int = 5000,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
):
    """Train diffusion model"""
    device = torch.device(f'cuda:{local_rank}')
    
    # LION paper hyperparameters for diffusion
    configs = {
        'adamw': {'lr': 1e-4, 'wd': 0.0},
        'lion': {'lr': 1e-5, 'wd': 0.0},  # 10x smaller for Lion
        'rlo': {'lr': 1e-5, 'wd': 0.0},
        'rlo_lambda_a': {'lr': 1e-5, 'wd': 0.0},
        'smooth_lifted_rlo': {'lr': 1e-5, 'wd': 0.0},
    }
    config = configs[opt_name]
    batch_size_per_gpu = batch_size // world_size
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Diffusion Training: {opt_name}")
        print(f"Resolution: {resolution}x{resolution}")
        print(f"lr={config['lr']}, batch={batch_size}")
        print(f"{'='*60}")
    
    # Data
    transform = T.Compose([
        T.Resize(resolution),
        T.CenterCrop(resolution),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    dataset = ImageFolder(data_path / 'train', transform)
    
    if world_size > 1:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None
    
    loader = DataLoader(
        dataset, batch_size=batch_size_per_gpu, shuffle=(sampler is None),
        sampler=sampler, num_workers=8, pin_memory=True, drop_last=True,
    )
    
    # Model
    if resolution == 64:
        base_dim = 128
        dim_mults = (1, 2, 4, 8)
    elif resolution == 128:
        base_dim = 128
        dim_mults = (1, 2, 4, 8)
    else:  # 256
        base_dim = 128
        dim_mults = (1, 2, 4, 4, 8)
    
    model = UNet(base_dim=base_dim, dim_mults=dim_mults).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # Diffusion
    diffusion = GaussianDiffusion(num_timesteps=1000).to(device)
    
    # Optimizer
    optimizer = create_optimizer(model, opt_name, config['lr'], config['wd'])
    
    # Training
    scaler = torch.amp.GradScaler('cuda')
    global_step = 0
    history = {'loss': []}
    
    total_steps = len(loader) * epochs
    
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Warmup LR
            if global_step < warmup_steps:
                lr = config['lr'] * global_step / warmup_steps
            else:
                progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
                lr = config['lr'] * 0.5 * (1 + math.cos(math.pi * progress))
            
            for g in optimizer.param_groups:
                g['lr'] = lr
            
            # Sample timesteps
            t = torch.randint(0, diffusion.num_timesteps, (images.shape[0],), device=device)
            
            optimizer.zero_grad()
            
            with torch.autocast('cuda', torch.bfloat16):
                loss = diffusion.training_loss(model, images, t, class_labels=labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            global_step += 1
            
            if rank == 0 and global_step % 500 == 0:
                print(f"Step {global_step}: loss={loss.item():.4f}, lr={lr:.2e}")
        
        avg_loss = epoch_loss / len(loader)
        history['loss'].append(avg_loss)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                base_model = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch': epoch,
                    'model': base_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, output_dir / f"diffusion_{resolution}_{opt_name}_ep{epoch+1}.pt")
    
    # Save results
    if rank == 0:
        results = {
            'optimizer': opt_name,
            'resolution': resolution,
            'final_loss': history['loss'][-1],
            'history': history,
        }
        with open(output_dir / f"diffusion_{resolution}_{opt_name}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    return history['loss'][-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='all')
    parser.add_argument('--resolution', type=int, default=64, choices=[64, 128, 256])
    parser.add_argument('--data', default='/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder')
    parser.add_argument('--output', default='./results_diffusion')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
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
    
    optimizers = ['adamw', 'lion', 'rlo', 'rlo_lambda_a', 'smooth_lifted_rlo'] if args.optimizer == 'all' else [args.optimizer]
    
    results = {}
    for opt in optimizers:
        try:
            loss = train_diffusion(opt, data_path, output_dir, args.resolution, args.epochs,
                                  rank=rank, world_size=world_size, local_rank=local_rank)
            results[opt] = loss
        except Exception as e:
            if rank == 0:
                print(f"Error {opt}: {e}")
                import traceback
                traceback.print_exc()
    
    if rank == 0 and results:
        print("\n" + "="*60)
        print(f"Diffusion {args.resolution}x{args.resolution} Results (Final Loss)")
        for opt, loss in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {opt}: {loss:.4f}")
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
