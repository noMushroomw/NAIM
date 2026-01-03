#!/usr/bin/env python3
"""
Section 4.3: Image Generation with Diffusion Models
====================================================

Following LION Paper Section 4.3:
- U-Net based diffusion
- ImageNet 64x64 / 128x128

Usage:
    torchrun --nproc_per_node=8 train_diffusion.py --resolution 64 --optimizer all
    torchrun --nproc_per_node=8 train_diffusion.py --resolution 128 --optimizer all
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
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, belief_coef=0.1, eps=1e-8):
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
                 weight_decay=0.0, lambda_b=0.1, eps=1e-8, gamma=5.0):
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
# U-Net
# =============================================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.block1 = nn.Sequential(nn.GroupNorm(32, in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.block2 = nn.Sequential(nn.GroupNorm(32, out_ch), nn.SiLU(), nn.Dropout(dropout), nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.block1(x) + self.time_mlp(t)[:, :, None, None]
        return self.block2(h) + self.shortcut(x)


class Attention2D(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q, k, v = q.permute(0, 1, 3, 2), k.permute(0, 1, 3, 2), v.permute(0, 1, 3, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        return x + self.proj(attn.permute(0, 1, 3, 2).reshape(B, C, H, W))


class UNet(nn.Module):
    def __init__(self, in_ch=3, base_dim=128, dim_mults=(1, 2, 4, 8), num_classes=1000):
        super().__init__()
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(base_dim), nn.Linear(base_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
        self.class_emb = nn.Embedding(num_classes, time_dim)
        self.init_conv = nn.Conv2d(in_ch, base_dim, 3, padding=1)
        
        dims = [base_dim] + [base_dim * m for m in dim_mults]
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        for i in range(len(dim_mults)):
            self.downs.append(nn.ModuleList([
                ResBlock(dims[i], dims[i+1], time_dim),
                Attention2D(dims[i+1]) if i >= 2 else nn.Identity(),
                nn.Conv2d(dims[i+1], dims[i+1], 3, stride=2, padding=1)
            ]))
        
        self.mid_res1 = ResBlock(dims[-1], dims[-1], time_dim)
        self.mid_attn = Attention2D(dims[-1])
        self.mid_res2 = ResBlock(dims[-1], dims[-1], time_dim)
        
        for i in reversed(range(len(dim_mults))):
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose2d(dims[i+1], dims[i+1], 4, stride=2, padding=1),
                ResBlock(dims[i+1] + dims[i+1], dims[i], time_dim),
                Attention2D(dims[i]) if i >= 2 else nn.Identity()
            ]))
        
        self.final = nn.Sequential(nn.GroupNorm(32, base_dim), nn.SiLU(), nn.Conv2d(base_dim, in_ch, 3, padding=1))

    def forward(self, x, t, cls=None):
        t_emb = self.time_mlp(t)
        if cls is not None:
            t_emb = t_emb + self.class_emb(cls)
        
        x = self.init_conv(x)
        skips = []
        
        for res, attn, down in self.downs:
            x = res(x, t_emb)
            x = attn(x)
            skips.append(x)
            x = down(x)
        
        x = self.mid_res1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_res2(x, t_emb)
        
        for (up, res, attn), skip in zip(self.ups, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = res(x, t_emb)
            x = attn(x)
        
        return self.final(x)


# =============================================================================
# Diffusion
# =============================================================================

class GaussianDiffusion:
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_steps = num_steps
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus = torch.sqrt(1.0 - self.alphas_cumprod)

    def to(self, device):
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus = self.sqrt_one_minus.to(device)
        return self

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_a = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_1m = self.sqrt_one_minus[t][:, None, None, None]
        return sqrt_a * x0 + sqrt_1m * noise

    def loss(self, model, x0, t, cls=None):
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        pred = model(x_noisy, t, cls)
        return F.mse_loss(pred, noise)


# =============================================================================
# Training
# =============================================================================

CONFIGS = {
    'adamw': {'lr': 1e-4, 'wd': 0.0},
    'lion': {'lr': 1e-5, 'wd': 0.0},
    'rlo': {'lr': 1e-5, 'wd': 0.0},
    'rlo_lambda_a': {'lr': 1e-5, 'wd': 0.0},
    'smooth_lifted_rlo': {'lr': 1e-5, 'wd': 0.0},
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


def train_diffusion(opt_name, data_path, output_dir, resolution, epochs, rank, world_size, local_rank, logger):
    device = torch.device(f'cuda:{local_rank}')
    cfg = CONFIGS[opt_name]
    global_batch, per_gpu = 256, 256 // world_size

    logger.info("=" * 60)
    logger.info(f"Diffusion Training: {opt_name} @ {resolution}x{resolution}")
    logger.info(f"LR: {cfg['lr']}, Epochs: {epochs}")
    logger.info("=" * 60)

    # Data
    transform = T.Compose([
        T.Resize(resolution), T.CenterCrop(resolution), T.RandomHorizontalFlip(),
        T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3),
    ])
    dataset = ImageFolder(data_path / 'train', transform)
    sampler = DistributedSampler(dataset, world_size, rank) if world_size > 1 else None
    loader = DataLoader(dataset, per_gpu, shuffle=(sampler is None), sampler=sampler,
                       num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)

    logger.info(f"Train: {len(dataset)} samples, {len(loader)} steps/epoch")

    # Model
    dim_mults = (1, 2, 4, 8) if resolution <= 64 else (1, 2, 4, 4, 8)
    model = UNet(base_dim=128, dim_mults=dim_mults).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    diffusion = GaussianDiffusion().to(device)
    optimizer = create_optimizer(model.parameters(), opt_name, cfg['lr'], cfg['wd'])
    scaler = torch.amp.GradScaler('cuda')

    # Scheduler
    total_steps = len(loader) * epochs
    warmup_steps = 5000

    def get_lr(step):
        if step < warmup_steps:
            return cfg['lr'] * step / warmup_steps
        return cfg['lr'] * 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

    history = {'loss': [], 'lr': [], 'epoch_time': []}
    step = 0

    for epoch in range(epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        t0 = time.time()

        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            t = torch.randint(0, diffusion.num_steps, (imgs.size(0),), device=device)

            lr = get_lr(step)
            for g in optimizer.param_groups:
                g['lr'] = lr

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast('cuda', torch.bfloat16):
                loss = diffusion.loss(model, imgs, t, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()
            step += 1

            if rank == 0 and (batch_idx + 1) % 200 == 0:
                logger.debug(f"E{epoch+1} S{batch_idx+1}: loss={loss.item():.4f}")

        epoch_time = time.time() - t0
        avg_loss = loss_sum / len(loader)
        history['loss'].append(avg_loss)
        history['lr'].append(lr)
        history['epoch_time'].append(epoch_time)

        logger.info(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, time={epoch_time:.1f}s")

        # Save checkpoint
        if rank == 0 and (epoch + 1) % 20 == 0:
            base_model = model.module if hasattr(model, 'module') else model
            torch.save({'epoch': epoch, 'model': base_model.state_dict()}, 
                      output_dir / f"diffusion_{resolution}_{opt_name}_ep{epoch+1}.pt")

    # Save
    if rank == 0:
        results = {'optimizer': opt_name, 'resolution': resolution, 'config': cfg,
                  'final_loss': history['loss'][-1], 'total_time': sum(history['epoch_time']), 'history': history}
        with open(output_dir / f"diffusion_{resolution}_{opt_name}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Final loss: {history['loss'][-1]:.4f}")

    return history['loss'][-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='all')
    parser.add_argument('--resolution', type=int, default=64, choices=[64, 128])
    parser.add_argument('--data', default='/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder')
    parser.add_argument('--output', default='./results_diffusion')
    parser.add_argument('--epochs', type=int, default=100)
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
    logger = setup_logger(output_dir, rank, f"diffusion_{args.resolution}")

    opts = list(CONFIGS.keys()) if args.optimizer == 'all' else [args.optimizer]
    results = {}

    for opt in opts:
        try:
            loss = train_diffusion(opt, Path(args.data), output_dir, args.resolution, args.epochs, 
                                  rank, world_size, local_rank, logger)
            results[opt] = loss
        except Exception as e:
            logger.error(f"Error {opt}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    if rank == 0 and results:
        logger.info("=" * 60)
        logger.info(f"Diffusion {args.resolution}x{args.resolution} Results")
        for opt, loss in sorted(results.items(), key=lambda x: x[1]):
            logger.info(f"  {opt}: {loss:.4f}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
