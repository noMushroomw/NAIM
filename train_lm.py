#!/usr/bin/env python3
"""
Section 4.4: Autoregressive Language Modeling
==============================================

Following LION Paper Section 4.4:
- GPT-2 style transformer
- WikiText-103

Usage:
    torchrun --nproc_per_node=8 train_lm.py --optimizer all --epochs 10
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
# GPT Model
# =============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_len=1024):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, max_len=1024):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, max_len)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size=50257, embed_dim=768, num_heads=12, num_layers=12, max_len=1024):
        super().__init__()
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, max_len) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss


# =============================================================================
# Dataset
# =============================================================================

class TextDataset(Dataset):
    def __init__(self, data_path, seq_len=1024, split='train'):
        self.seq_len = seq_len
        text_file = data_path / f'{split}.txt'
        
        if text_file.exists():
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            # Simple character-level tokenization
            chars = sorted(set(text))
            self.char2idx = {c: i for i, c in enumerate(chars)}
            self.vocab_size = len(chars)
            self.tokens = torch.tensor([self.char2idx.get(c, 0) for c in text], dtype=torch.long)
        else:
            # Dummy data
            self.vocab_size = 256
            self.tokens = torch.randint(0, self.vocab_size, (500000,))
        
        self.n_samples = max(1, (len(self.tokens) - 1) // seq_len)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        if len(chunk) < self.seq_len + 1:
            chunk = F.pad(chunk, (0, self.seq_len + 1 - len(chunk)))
        return chunk[:-1], chunk[1:]


# =============================================================================
# Training
# =============================================================================

CONFIGS = {
    'adamw': {'lr': 6e-4, 'wd': 0.1},
    'lion': {'lr': 1e-4, 'wd': 1.0},
    'rlo': {'lr': 1e-4, 'wd': 1.0},
    'rlo_lambda_a': {'lr': 1e-4, 'wd': 1.0},
    'smooth_lifted_rlo': {'lr': 1e-4, 'wd': 1.0},
}


def create_optimizer(params, name, lr, wd):
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    elif name == 'lion':
        return Lion(params, lr=lr, weight_decay=wd)
    elif name == 'rlo':
        return RLO(params, lr=lr, weight_decay=wd)
    elif name == 'rlo_lambda_a':
        return RLO_LambdaA(params, lr=lr, weight_decay=wd)
    elif name == 'smooth_lifted_rlo':
        return SmoothLiftedRLO(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown: {name}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast('cuda', torch.bfloat16):
            _, loss = model(x, y)
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    return math.exp(total_loss / total_tokens)


def train_lm(opt_name, data_path, output_dir, epochs, rank, world_size, local_rank, logger):
    device = torch.device(f'cuda:{local_rank}')
    cfg = CONFIGS[opt_name]
    global_batch, per_gpu = 64, 64 // world_size
    seq_len = 1024

    logger.info("=" * 60)
    logger.info(f"LM Training: {opt_name}")
    logger.info(f"LR: {cfg['lr']}, WD: {cfg['wd']}, Seq len: {seq_len}")
    logger.info("=" * 60)

    # Data
    train_data = TextDataset(data_path, seq_len, 'train')
    val_data = TextDataset(data_path, seq_len, 'valid')
    
    sampler = DistributedSampler(train_data, world_size, rank) if world_size > 1 else None
    train_loader = DataLoader(train_data, per_gpu, shuffle=(sampler is None), sampler=sampler,
                             num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_data, per_gpu * 2, num_workers=4, pin_memory=True)

    logger.info(f"Train: {len(train_data)} samples, {len(train_loader)} steps/epoch")
    logger.info(f"Vocab size: {train_data.vocab_size}")

    # Model
    model = GPT(vocab_size=train_data.vocab_size, embed_dim=768, num_heads=12, num_layers=12, max_len=seq_len).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    optimizer = create_optimizer(model.parameters(), opt_name, cfg['lr'], cfg['wd'])
    scaler = torch.amp.GradScaler('cuda')

    # Scheduler
    total_steps = len(train_loader) * epochs
    warmup_steps = 2000

    def get_lr(step):
        if step < warmup_steps:
            return cfg['lr'] * step / warmup_steps
        return cfg['lr'] * 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

    history = {'train_loss': [], 'val_ppl': [], 'lr': [], 'epoch_time': []}
    best_ppl = float('inf')
    step = 0

    for epoch in range(epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        t0 = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            lr = get_lr(step)
            for g in optimizer.param_groups:
                g['lr'] = lr

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast('cuda', torch.bfloat16):
                _, loss = model(x, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()
            step += 1

            if rank == 0 and (batch_idx + 1) % 100 == 0:
                logger.debug(f"E{epoch+1} S{batch_idx+1}: loss={loss.item():.4f}")

        epoch_time = time.time() - t0
        avg_loss = loss_sum / len(train_loader)

        # Validation
        base_model = model.module if hasattr(model, 'module') else model
        val_ppl = evaluate(base_model, val_loader, device)

        history['train_loss'].append(avg_loss)
        history['val_ppl'].append(val_ppl)
        history['lr'].append(lr)
        history['epoch_time'].append(epoch_time)

        is_best = val_ppl < best_ppl
        best_ppl = min(val_ppl, best_ppl)

        logger.info(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, ppl={val_ppl:.2f}{'*' if is_best else ''}")

        if is_best and rank == 0:
            torch.save({'model': base_model.state_dict(), 'ppl': val_ppl}, output_dir / f"lm_{opt_name}_best.pt")

    # Save
    if rank == 0:
        results = {'optimizer': opt_name, 'config': cfg, 'best_ppl': best_ppl,
                  'final_ppl': history['val_ppl'][-1], 'total_time': sum(history['epoch_time']), 'history': history}
        with open(output_dir / f"lm_{opt_name}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Best PPL: {best_ppl:.2f}")

    return best_ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='all')
    parser.add_argument('--data', default='./wikitext-103')
    parser.add_argument('--output', default='./results_lm')
    parser.add_argument('--epochs', type=int, default=10)
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
    logger = setup_logger(output_dir, rank, "lm")

    opts = list(CONFIGS.keys()) if args.optimizer == 'all' else [args.optimizer]
    results = {}

    for opt in opts:
        try:
            ppl = train_lm(opt, Path(args.data), output_dir, args.epochs, rank, world_size, local_rank, logger)
            results[opt] = ppl
        except Exception as e:
            logger.error(f"Error {opt}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    if rank == 0 and results:
        logger.info("=" * 60)
        logger.info("LM Results (Perplexity)")
        for opt, ppl in sorted(results.items(), key=lambda x: x[1]):
            logger.info(f"  {opt}: {ppl:.2f}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
