#!/usr/bin/env python3
"""
Section 4.4: Autoregressive Language Modeling
==============================================

Following LION paper Section 4.4:
- GPT-2 style transformer
- WikiText-103 dataset
- Perplexity comparison

Usage:
    torchrun --nproc_per_node=8 train_lm.py --optimizer all
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
# 2. GPT Model
# =============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, max_seq_len=1024):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len))
                            .view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Causal attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.resid_dropout(self.proj(x))


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        return self.dropout(self.fc2(x))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, max_seq_len=1024):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, 4, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """GPT-2 style language model"""
    def __init__(self, vocab_size=50257, embed_dim=768, num_heads=12, 
                 num_layers=12, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout, max_seq_len)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Weight tying
        self.token_emb.weight = self.lm_head.weight
        
        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} > max {self.max_seq_len}"
        
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss


# Model configurations
GPT_CONFIGS = {
    'small': {'embed_dim': 768, 'num_heads': 12, 'num_layers': 12},    # 124M
    'medium': {'embed_dim': 1024, 'num_heads': 16, 'num_layers': 24},   # 350M
    'large': {'embed_dim': 1280, 'num_heads': 20, 'num_layers': 36},    # 774M
    'xl': {'embed_dim': 1600, 'num_heads': 25, 'num_layers': 48},       # 1.5B
}


# =============================================================================
# 3. Dataset (WikiText-103)
# =============================================================================

class TextDataset(Dataset):
    """Simple text dataset for language modeling"""
    def __init__(self, data_path: Path, seq_len: int = 1024, split: str = 'train'):
        self.seq_len = seq_len
        
        # Load or create vocabulary
        self.vocab = self._build_vocab(data_path, split)
        self.vocab_size = len(self.vocab)
        
        # Load and tokenize text
        self.tokens = self._load_and_tokenize(data_path, split)
        
        # Calculate number of samples
        self.n_samples = max(1, (len(self.tokens) - 1) // seq_len)

    def _build_vocab(self, data_path, split):
        """Build vocabulary from text"""
        vocab_path = data_path / 'vocab.json'
        
        if vocab_path.exists():
            import json
            with open(vocab_path) as f:
                return json.load(f)
        
        # Build from training data
        text_path = data_path / f'{split}.txt'
        if not text_path.exists():
            # Create dummy data for testing
            return {chr(i): i for i in range(256)}  # Character-level
        
        vocab = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
        idx = 3
        
        with open(text_path, 'r', encoding='utf-8') as f:
            for line in f:
                for word in line.strip().split():
                    if word not in vocab:
                        vocab[word] = idx
                        idx += 1
                        if idx >= 50000:  # Limit vocab size
                            break
                if idx >= 50000:
                    break
        
        # Save vocab
        import json
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)
        
        return vocab

    def _load_and_tokenize(self, data_path, split):
        """Load and tokenize text file"""
        text_path = data_path / f'{split}.txt'
        
        if not text_path.exists():
            # Create dummy data for testing
            print(f"Warning: {text_path} not found, using dummy data")
            return torch.randint(0, len(self.vocab), (100000,))
        
        tokens = []
        with open(text_path, 'r', encoding='utf-8') as f:
            for line in f:
                for word in line.strip().split():
                    tokens.append(self.vocab.get(word, self.vocab['<unk>']))
                tokens.append(self.vocab['<eos>'])
        
        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        
        if end > len(self.tokens):
            # Pad if necessary
            chunk = self.tokens[start:]
            chunk = F.pad(chunk, (0, end - len(self.tokens)), value=0)
        else:
            chunk = self.tokens[start:end]
        
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


# =============================================================================
# 4. Training
# =============================================================================

def create_optimizer(model, opt_name: str, lr: float, wd: float):
    params = model.parameters()
    if opt_name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    elif opt_name == 'lion':
        return Lion(params, lr=lr, weight_decay=wd, betas=(0.9, 0.99))
    elif opt_name == 'rlo':
        return RLO(params, lr=lr, weight_decay=wd, betas=(0.9, 0.99))
    elif opt_name == 'rlo_lambda_a':
        return RLO_LambdaA(params, lr=lr, weight_decay=wd)
    elif opt_name == 'smooth_lifted_rlo':
        return SmoothLiftedRLO(params, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate and return perplexity"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast('cuda', torch.bfloat16):
            _, loss = model(x, y)
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


def train_lm(
    opt_name: str,
    data_path: Path,
    output_dir: Path,
    model_size: str = 'small',
    epochs: int = 10,
    batch_size: int = 64,
    seq_len: int = 1024,
    warmup_steps: int = 2000,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
):
    """Train language model"""
    device = torch.device(f'cuda:{local_rank}')
    
    # LION paper hyperparameters
    configs = {
        'adamw': {'lr': 6e-4, 'wd': 0.1},
        'lion': {'lr': 1e-4, 'wd': 1.0},
        'rlo': {'lr': 1e-4, 'wd': 1.0},
        'rlo_lambda_a': {'lr': 1e-4, 'wd': 1.0},
        'smooth_lifted_rlo': {'lr': 1e-4, 'wd': 1.0},
    }
    config = configs[opt_name]
    batch_size_per_gpu = batch_size // world_size
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Language Model Training: {opt_name}")
        print(f"Model: GPT-{model_size}")
        print(f"lr={config['lr']}, wd={config['wd']}, batch={batch_size}")
        print(f"{'='*60}")
    
    # Data
    train_dataset = TextDataset(data_path, seq_len, 'train')
    val_dataset = TextDataset(data_path, seq_len, 'valid')
    
    if rank == 0:
        print(f"Train tokens: {len(train_dataset.tokens)}")
        print(f"Vocab size: {train_dataset.vocab_size}")
    
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_per_gpu, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size_per_gpu * 2, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    
    # Model
    model_config = GPT_CONFIGS[model_size]
    model = GPT(
        vocab_size=train_dataset.vocab_size,
        max_seq_len=seq_len,
        **model_config
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Model params: {n_params/1e6:.1f}M")
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # Optimizer
    optimizer = create_optimizer(model, opt_name, config['lr'], config['wd'])
    
    # Training
    scaler = torch.amp.GradScaler('cuda')
    global_step = 0
    total_steps = len(train_loader) * epochs
    
    history = {'train_loss': [], 'val_ppl': []}
    best_ppl = float('inf')
    
    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Learning rate schedule
            if global_step < warmup_steps:
                lr = config['lr'] * global_step / warmup_steps
            else:
                progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
                lr = config['lr'] * 0.5 * (1 + math.cos(math.pi * progress))
            
            for g in optimizer.param_groups:
                g['lr'] = lr
            
            optimizer.zero_grad()
            
            with torch.autocast('cuda', torch.bfloat16):
                _, loss = model(x, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            global_step += 1
            
            if rank == 0 and global_step % 100 == 0:
                print(f"Step {global_step}: loss={loss.item():.4f}, lr={lr:.2e}")
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Validation
        base_model = model.module if hasattr(model, 'module') else model
        val_ppl = evaluate(base_model, val_loader, device)
        
        history['train_loss'].append(avg_loss)
        history['val_ppl'].append(val_ppl)
        
        is_best = val_ppl < best_ppl
        best_ppl = min(val_ppl, best_ppl)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, val_ppl={val_ppl:.2f}{'*' if is_best else ''}")
            
            if is_best:
                torch.save({
                    'epoch': epoch,
                    'model': base_model.state_dict(),
                    'ppl': val_ppl,
                }, output_dir / f"lm_{opt_name}_best.pt")
    
    # Save results
    if rank == 0:
        results = {
            'optimizer': opt_name,
            'model_size': model_size,
            'best_ppl': best_ppl,
            'final_ppl': history['val_ppl'][-1],
            'history': history,
        }
        with open(output_dir / f"lm_{opt_name}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    return best_ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='all')
    parser.add_argument('--model_size', default='small', choices=['small', 'medium', 'large'])
    parser.add_argument('--data', default='./wikitext-103')
    parser.add_argument('--output', default='./results_lm')
    parser.add_argument('--epochs', type=int, default=10)
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
            ppl = train_lm(opt, data_path, output_dir, args.model_size, args.epochs,
                          rank=rank, world_size=world_size, local_rank=local_rank)
            results[opt] = ppl
        except Exception as e:
            if rank == 0:
                print(f"Error {opt}: {e}")
                import traceback
                traceback.print_exc()
    
    if rank == 0 and results:
        print("\n" + "="*60)
        print("Language Model Results (Perplexity, lower is better)")
        for opt, ppl in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {opt}: {ppl:.2f}")
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
