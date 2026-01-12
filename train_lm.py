#!/usr/bin/env python3
"""
Section 4.4: Autoregressive Language Modeling
==============================================
Following LION Paper Section 4.4 EXACTLY:

Model: GPT-2 Small (117M params)
- 12 layers, 768 dim, 12 heads
- Sequence length: 1024
- Batch size: 512 (global)
- Training: 100K steps

Hyperparameters:
- AdamW: lr=6e-4, β1=0.9, β2=0.95, wd=0.1
- Lion:  lr=1e-4, β1=0.9, β2=0.99, wd=1.0

Dataset: OpenWebText (proxy for C4)
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
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Optimizer
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
# GPT-2 Model (117M)
# =============================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(x.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT2(nn.Module):
    """GPT-2 Small: 12 layers, 768 dim, 12 heads = 117M params"""
    def __init__(self, vocab_size=50257, embed_dim=768, num_heads=12, num_layers=12, max_len=1024):
        super().__init__()
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight  # Weight tying
        self.apply(self._init_weights)
        # Scale residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight') or pn.endswith('mlp.2.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Dataset
# =============================================================================
class TextDataset(Dataset):
    """Load tokenized data or use character-level fallback"""
    def __init__(self, data_path, seq_len=1024, split='train'):
        self.seq_len = seq_len
        data_path = Path(data_path)
        
        # Try pre-tokenized binary
        token_file = data_path / f'{split}_tokens.bin'
        if token_file.exists():
            self.tokens = np.memmap(token_file, dtype=np.uint16, mode='r')
            self.vocab_size = 50257
        else:
            # Fallback: text file with character-level
            text_file = data_path / f'{split}.txt'
            if text_file.exists():
                with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                chars = sorted(set(text))
                self.char2idx = {c: i for i, c in enumerate(chars)}
                self.vocab_size = len(chars)
                self.tokens = np.array([self.char2idx.get(c, 0) for c in text], dtype=np.int64)
            else:
                # Synthetic for testing
                self.vocab_size = 50257
                self.tokens = np.random.randint(0, self.vocab_size, size=10_000_000, dtype=np.int64)
        
        self.n_samples = (len(self.tokens) - 1) // seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start:start + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


# =============================================================================
# Config - LION Paper Section 4.4 EXACT
# =============================================================================
CONFIGS = {
    # LION paper: AdamW β1=0.9, β2=0.95, wd=0.1
    'adamw': {'lr': 6e-4, 'wd': 0.1, 'betas': (0.9, 0.95)},
    # LION paper: Lion β1=0.9, β2=0.99, wd=1.0
    'lion': {'lr': 1e-4, 'wd': 1.0, 'betas': (0.9, 0.99)},
    'rlo': {'lr': 1e-4, 'wd': 1.0, 'betas': (0.9, 0.99), 'belief_coef': 0.1},
    'smooth_lifted_rlo': {'lr': 1e-4, 'wd': 1.0, 'lambda_b': 0.1, 'eta': 0.3},
}


def create_optimizer(model, name, cfg):
    # Separate decay/no-decay params (GPT-2/3 practice)
    decay, no_decay = [], []
    for pn, p in model.named_parameters():
        if p.requires_grad:
            if pn.endswith('bias') or 'ln' in pn or 'pos_emb' in pn:
                no_decay.append(p)
            else:
                decay.append(p)
    
    if name == 'adamw':
        return torch.optim.AdamW([{'params': decay, 'weight_decay': cfg['wd']},
                                  {'params': no_decay, 'weight_decay': 0.0}],
                                 lr=cfg['lr'], betas=cfg.get('betas', (0.9, 0.95)))
    elif name == 'lion':
        return Lion([{'params': decay, 'weight_decay': cfg['wd']},
                     {'params': no_decay, 'weight_decay': 0.0}],
                    lr=cfg['lr'], betas=cfg.get('betas', (0.9, 0.99)))
    elif name == 'rlo':
        return RLO([{'params': decay, 'weight_decay': cfg['wd']},
                    {'params': no_decay, 'weight_decay': 0.0}],
                   lr=cfg['lr'], betas=cfg.get('betas', (0.9, 0.99)), belief_coef=cfg.get('belief_coef', 0.1))
    elif name == 'smooth_lifted_rlo':
        return SmoothLiftedRLO([{'params': decay, 'weight_decay': cfg['wd']},
                                {'params': no_decay, 'weight_decay': 0.0}],
                               lr=cfg['lr'], lambda_b=cfg.get('lambda_b', 0.1), eta=cfg.get('eta', 0.3))
    raise ValueError(f"Unknown: {name}")


# =============================================================================
# Training
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, device, max_batches=50):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast('cuda', torch.bfloat16):
            _, loss = model(x, y)
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')


def train_lm(opt_name, data_path, output_dir, total_steps, rank, world_size, local_rank, logger):
    device = torch.device(f'cuda:{local_rank}')
    cfg = CONFIGS[opt_name]
    
    # LION paper: batch=512, seq=1024
    global_batch = 512
    seq_len = 1024
    per_gpu = global_batch // world_size
    # Gradient accumulation for memory
    micro_batch = min(8, per_gpu)
    grad_accum = per_gpu // micro_batch

    logger.info("=" * 70)
    logger.info(f"LM Training: {opt_name}")
    logger.info(f"Global batch: {global_batch} | Per-GPU: {per_gpu} | Micro: {micro_batch} | Accum: {grad_accum}")
    logger.info(f"LR: {cfg['lr']} | WD: {cfg['wd']} | Steps: {total_steps}")
    logger.info("=" * 70)

    # Data
    train_data = TextDataset(data_path, seq_len, 'train')
    val_data = TextDataset(data_path, seq_len, 'valid') if (Path(data_path) / 'valid.txt').exists() else train_data
    
    sampler = DistributedSampler(train_data, world_size, rank) if world_size > 1 else None
    train_loader = DataLoader(train_data, micro_batch, shuffle=(sampler is None), sampler=sampler,
                             num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_data, micro_batch * 2, num_workers=2, pin_memory=True)

    logger.info(f"Train: {len(train_data)} samples | Vocab: {train_data.vocab_size}")

    # Model
    model = GPT2(vocab_size=train_data.vocab_size, embed_dim=768, num_heads=12, num_layers=12, max_len=seq_len).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    base_model = model.module if hasattr(model, 'module') else model
    logger.info(f"Parameters: {base_model.num_params() / 1e6:.1f}M")

    optimizer = create_optimizer(base_model, opt_name, cfg)
    scaler = torch.amp.GradScaler('cuda')

    # Warmup + cosine decay
    warmup_steps = 2000
    def get_lr(step):
        if step < warmup_steps:
            return cfg['lr'] * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return cfg['lr'] * 0.1 + 0.9 * cfg['lr'] * 0.5 * (1 + math.cos(math.pi * progress))

    history = {'train_loss': [], 'val_ppl': [], 'lr': [], 'step': []}
    best_ppl = float('inf')
    step = 0
    epoch = 0
    train_iter = iter(train_loader)
    t0 = time.time()
    loss_accum = 0.0

    while step < total_steps:
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Gradient accumulation
        for _ in range(grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                epoch += 1
                if sampler:
                    sampler.set_epoch(epoch)
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            
            x, y = x.to(device), y.to(device)
            with torch.autocast('cuda', torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            loss_accum += loss.item()

        # Update
        lr = get_lr(step)
        for g in optimizer.param_groups:
            g['lr'] = lr
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        step += 1

        # Log every 100 steps
        if step % 100 == 0:
            tokens_per_sec = (100 * global_batch * seq_len) / (time.time() - t0)
            logger.info(f"Step {step}/{total_steps}: loss={loss_accum:.4f}, lr={lr:.2e}, {tokens_per_sec/1e6:.2f}M tok/s")
            history['train_loss'].append(loss_accum)
            history['lr'].append(lr)
            history['step'].append(step)
            loss_accum = 0.0
            t0 = time.time()

        # Validation every 2000 steps
        if step % 2000 == 0 or step == total_steps:
            val_ppl = evaluate(base_model, val_loader, device)
            history['val_ppl'].append(val_ppl)
            is_best = val_ppl < best_ppl
            best_ppl = min(val_ppl, best_ppl)
            logger.info(f"Step {step}: val_ppl={val_ppl:.2f}{'*' if is_best else ''}")
            
            if is_best and rank == 0:
                torch.save({'model': base_model.state_dict(), 'ppl': best_ppl, 'step': step},
                          output_dir / f"lm_{opt_name}_best.pt")

    # Save results
    if rank == 0:
        with open(output_dir / f"lm_{opt_name}_results.json", 'w') as f:
            json.dump({'optimizer': opt_name, 'config': cfg, 'best_ppl': best_ppl,
                      'total_steps': total_steps, 'history': history}, f, indent=2)
        logger.info(f"Done. Best PPL: {best_ppl:.2f}")

    return best_ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='all')
    parser.add_argument('--data', default='/blue/wdixon/wang.yixuan/rlo_results/openwebtext')
    parser.add_argument('--output', default='/blue/wdixon/wang.yixuan/rlo_results/lm')
    parser.add_argument('--steps', type=int, default=100000)  # LION paper: 100K
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
            gc.collect(); torch.cuda.empty_cache()
            results[opt] = train_lm(opt, Path(args.data), output_dir, args.steps, rank, world_size, local_rank, logger)
        except Exception as e:
            logger.error(f"Error {opt}: {e}")
            import traceback; logger.error(traceback.format_exc())

    if rank == 0 and results:
        logger.info("=" * 70)
        logger.info("LM Results (PPL, lower=better):")
        for opt, ppl in sorted(results.items(), key=lambda x: x[1]):
            logger.info(f"  {opt}: {ppl:.2f}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
