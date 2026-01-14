#!/usr/bin/env python3
"""
Section 4.4: Language Modeling - LION Paper Table 12 EXACT
GPT-2 117M, batch=512, seq=1024, 100K steps

CRITICAL from Table 12:
- AdamW: lr=3e-3, β=(0.9, 0.99)
- Lion: lr=3e-4, β=(0.95, 0.98)  # Different betas for LM!
"""
import sys
def _patch_dill():
    try:
        import dill
        if not hasattr(dill, 'extend'): dill.extend = lambda use_dill=True: None
    except ImportError: pass
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
import warnings; warnings.filterwarnings('ignore')

def setup_logger(out, rank, name):
    logger = logging.getLogger(name); logger.setLevel(logging.DEBUG); logger.handlers.clear()
    fh = logging.FileHandler(out / f"{name}_rank{rank}.log", mode='w'); fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s')); logger.addHandler(fh)
    if rank == 0:
        ch = logging.StreamHandler(); ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')); logger.addHandler(ch)
    return logger

class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))
    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g['params']:
                if p.grad is None: continue
                if g['weight_decay'] != 0: p.mul_(1 - g['lr'] * g['weight_decay'])
                s = self.state[p]
                if len(s) == 0: s['m'] = torch.zeros_like(p)
                m = s['m']; b1, b2 = g['betas']
                p.add_((b1 * m + (1 - b1) * p.grad).sign_(), alpha=-g['lr'])
                m.mul_(b2).add_(p.grad, alpha=1 - b2)

class RLO(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, belief_coef=0.1, eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay, belief_coef=belief_coef, eps=eps))
    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                if g["weight_decay"] != 0: p.mul_(1 - g["lr"] * g["weight_decay"])
                s = self.state[p]
                if len(s) == 0: s["m"] = torch.zeros_like(p)
                m, gr = s["m"], p.grad; b1, b2 = g["betas"]
                c = b1 * m + (1 - b1) * gr; delta = gr - m
                p.add_(c.sign() + g["belief_coef"] * (delta / delta.norm().clamp(min=g["eps"])), alpha=-g["lr"])
                m.mul_(b2).add_(gr, alpha=1 - b2)

class RLO_LambdaA(Optimizer):
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.99, beta3=0.999, weight_decay=0.0, lambda_b=0.1, eps=1e-8, gamma=5.0):
        super().__init__(params, dict(lr=lr, beta1=beta1, beta2=beta2, beta3=beta3, weight_decay=weight_decay, lambda_b=lambda_b, eps=eps, gamma=gamma))
        self.sqrt_dim = math.sqrt(sum(p.numel() for g in self.param_groups for p in g["params"]))
    @torch.no_grad()
    def step(self, closure=None):
        all_sp, all_b, all_p = [], [], []
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                gr, s = p.grad, self.state[p]
                if len(s) == 0: s["m"], s["s"] = torch.zeros_like(p), torch.zeros_like(p)
                m, ss = s["m"], s["s"]
                ss.mul_(g["beta3"]).addcmul_(gr, gr, value=1 - g["beta3"])
                c = g["beta1"] * m + (1 - g["beta1"]) * gr
                sp = torch.tanh(g["gamma"] * c) / (ss.sqrt() + g["eps"])
                delta = gr - m; b = g["lambda_b"] * (delta / delta.norm().clamp(min=g["eps"]))
                all_sp.append(sp); all_b.append(b); all_p.append((p, g, m, gr))
        if not all_p: return
        scale = self.sqrt_dim / sum((x * x).sum() for x in all_sp).sqrt().clamp(min=1e-8)
        for (p, g, m, gr), sp, b in zip(all_p, all_sp, all_b):
            if g["weight_decay"] != 0: p.mul_(1 - g["lr"] * g["weight_decay"])
            p.add_(scale * sp + b, alpha=-g["lr"]); m.mul_(g["beta2"]).add_(gr, alpha=1 - g["beta2"])

class SmoothLiftedRLO(Optimizer):
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.99, eta=0.3, weight_decay=0.0, lambda_b=0.1, eps=1e-8, gamma=5.0):
        super().__init__(params, dict(lr=lr, beta1=beta1, beta2=beta2, eta=eta, weight_decay=weight_decay, lambda_b=lambda_b, eps=eps, gamma=gamma))
        self.sqrt_dim = math.sqrt(sum(p.numel() for g in self.param_groups for p in g["params"]))
    @torch.no_grad()
    def step(self, closure=None):
        all_s, all_b, all_p = [], [], []
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                s = self.state[p]
                if len(s) == 0: s["m"], s["v"] = torch.zeros_like(p), torch.zeros_like(p)
                gr, m = p.grad, s["m"]
                c = g["beta1"] * m + (1 - g["beta1"]) * gr; ss = torch.tanh(g["gamma"] * c)
                delta = gr - m; b = g["lambda_b"] * (delta / delta.norm().clamp(min=g["eps"]))
                all_s.append(ss); all_b.append(b); all_p.append((p, g))
        if not all_p: return
        scale = self.sqrt_dim / sum((x * x).sum() for x in all_s).sqrt().clamp(min=1e-8)
        for (p, g), ss, b in zip(all_p, all_s, all_b):
            s = self.state[p]
            if g["weight_decay"] != 0: p.mul_(1 - g["lr"] * g["weight_decay"])
            d = scale * ss + b; s["v"].mul_(1 - g["eta"]).add_(d, alpha=g["eta"])
            p.add_(s["v"], alpha=-g["lr"]); s["m"].mul_(g["beta2"]).add_(p.grad, alpha=1 - g["beta2"])

class CausalAttn(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.h, self.hd = h, d // h
        self.qkv, self.proj = nn.Linear(d, d * 3), nn.Linear(d, d)
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        return self.proj(F.scaled_dot_product_attention(*qkv.unbind(0), is_causal=True).transpose(1, 2).reshape(B, T, C))

class Blk(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.n1, self.attn, self.n2 = nn.LayerNorm(d), CausalAttn(d, h), nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
    def forward(self, x): x = x + self.attn(self.n1(x)); return x + self.mlp(self.n2(x))

class GPT2(nn.Module):
    def __init__(self, vocab=50257, d=768, h=12, layers=12, maxlen=1024):
        super().__init__(); self.maxlen = maxlen
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(maxlen, d)
        self.blks, self.ln = nn.ModuleList([Blk(d, h) for _ in range(layers)]), nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False); self.tok.weight = self.head.weight
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith('proj.weight') or n.endswith('mlp.2.weight'): nn.init.normal_(p, 0, 0.02 / math.sqrt(2 * layers))
    def _init(self, m):
        if isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0, 0.02); m.bias is not None and nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, 0, 0.02)
    def forward(self, idx, tgt=None):
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1), device=idx.device))
        for b in self.blks: x = b(x)
        logits = self.head(self.ln(x))
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1)) if tgt is not None else None
    def num_params(self): return sum(p.numel() for p in self.parameters())

class TextDataset(Dataset):
    def __init__(self, path, seq=1024, split='train'):
        self.seq = seq; path = Path(path)
        tf = path / f'{split}_tokens.bin'
        if tf.exists(): self.tokens, self.vocab = np.memmap(tf, dtype=np.uint16, mode='r'), 50257
        else:
            txt = path / f'{split}.txt'
            if txt.exists():
                text = open(txt, 'r', errors='ignore').read(); chars = sorted(set(text))
                self.vocab = len(chars); self.tokens = np.array([{c: i for i, c in enumerate(chars)}.get(c, 0) for c in text], dtype=np.int64)
            else: self.vocab, self.tokens = 50257, np.random.randint(0, 50257, 10_000_000, dtype=np.int64)
        self.n = (len(self.tokens) - 1) // seq
    def __len__(self): return self.n
    def __getitem__(self, i):
        c = self.tokens[i * self.seq:(i + 1) * self.seq + 1]
        return torch.from_numpy(c[:-1].astype(np.int64)), torch.from_numpy(c[1:].astype(np.int64))

# ============= LION Paper Table 12 EXACT for LM =============
# CRITICAL: For LM, Lion uses β1=0.95, β2=0.98 (NOT 0.9, 0.99!)
CONFIGS = {
    # Table 12 Small & Medium (PG-19, C4): AdamW lr=3e-3 β=(0.9,0.99), Lion lr=3e-4 β=(0.95,0.98)
    'adamw': {'lr': 3e-3, 'wd': 0.0, 'betas': (0.9, 0.99)},
    'lion': {'lr': 3e-4, 'wd': 0.0, 'betas': (0.95, 0.98)},  # CRITICAL: β=(0.95, 0.98) for LM
    'rlo': {'lr': 3e-4, 'wd': 0.0, 'betas': (0.95, 0.98), 'belief_coef': 0.1},
    'rlo_lambda_a': {'lr': 3e-4, 'wd': 0.0, 'beta1': 0.95, 'beta2': 0.98, 'lambda_b': 0.1},
    'smooth_lifted_rlo': {'lr': 3e-4, 'wd': 0.0, 'beta1': 0.95, 'beta2': 0.98, 'lambda_b': 0.1, 'eta': 0.3},
}

def create_optimizer(model, name, cfg):
    d, nd = [], []
    for n, p in model.named_parameters():
        if p.requires_grad: (nd if n.endswith('bias') or 'ln' in n or 'pos' in n else d).append(p)
    pg = [{'params': d, 'weight_decay': cfg['wd']}, {'params': nd, 'weight_decay': 0.0}]
    if name == 'adamw': return torch.optim.AdamW(pg, lr=cfg['lr'], betas=cfg.get('betas', (0.9, 0.99)))
    if name == 'lion': return Lion(pg, lr=cfg['lr'], betas=cfg.get('betas', (0.95, 0.98)))
    if name == 'rlo': return RLO(pg, lr=cfg['lr'], betas=cfg.get('betas', (0.95, 0.98)), belief_coef=cfg.get('belief_coef', 0.1))
    if name == 'rlo_lambda_a': return RLO_LambdaA(pg, lr=cfg['lr'], beta1=cfg.get('beta1', 0.95), beta2=cfg.get('beta2', 0.98), lambda_b=cfg.get('lambda_b', 0.1))
    if name == 'smooth_lifted_rlo': return SmoothLiftedRLO(pg, lr=cfg['lr'], beta1=cfg.get('beta1', 0.95), beta2=cfg.get('beta2', 0.98), lambda_b=cfg.get('lambda_b', 0.1), eta=cfg.get('eta', 0.3))
    raise ValueError(name)

@torch.no_grad()
def evaluate(model, loader, dev, n=50):
    model.eval(); loss, tok = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= n: break
        x, y = x.to(dev), y.to(dev)
        with torch.autocast('cuda', torch.bfloat16): _, l = model(x, y)
        loss += l.item() * y.numel(); tok += y.numel()
    return math.exp(loss / tok) if tok else float('inf')

def train_lm(opt, data, out, steps, rank, ws, lr, logger):
    dev = torch.device(f'cuda:{lr}'); cfg = CONFIGS[opt]
    gb, seq, pg = 512, 1024, 512 // ws
    mb, ga = min(8, pg), pg // min(8, pg)
    logger.info(f"{'='*70}\nLM: {opt} | batch={gb} seq={seq} steps={steps}")
    logger.info(f"LR={cfg['lr']} betas={cfg.get('betas', cfg.get('beta1', 0.95))}\n{'='*70}")

    train_d = TextDataset(data, seq, 'train')
    val_d = TextDataset(data, seq, 'valid') if (data / 'valid.txt').exists() else train_d
    samp = DistributedSampler(train_d, ws, rank) if ws > 1 else None
    tl = DataLoader(train_d, mb, shuffle=samp is None, sampler=samp, num_workers=2, pin_memory=True, drop_last=True)
    vl = DataLoader(val_d, mb * 2, num_workers=2, pin_memory=True)
    logger.info(f"Train: {len(train_d)} | Vocab: {train_d.vocab}")

    model = GPT2(vocab=train_d.vocab, d=768, h=12, layers=12, maxlen=seq).to(dev)
    if ws > 1: model = DDP(model, device_ids=[lr])
    bm = model.module if hasattr(model, 'module') else model
    logger.info(f"Params: {bm.num_params()/1e6:.1f}M")

    optim = create_optimizer(bm, opt, cfg); scaler = torch.amp.GradScaler('cuda')
    warmup = 2000
    def get_lr(s): return cfg['lr'] * s / warmup if s < warmup else cfg['lr'] * 0.1 + 0.9 * cfg['lr'] * 0.5 * (1 + math.cos(math.pi * (s - warmup) / (steps - warmup)))

    hist, best, step, ep = {'loss': [], 'ppl': [], 'step': []}, float('inf'), 0, 0
    ti = iter(tl); t0, la = time.time(), 0.0

    while step < steps:
        model.train(); optim.zero_grad(set_to_none=True)
        for _ in range(ga):
            try: x, y = next(ti)
            except StopIteration: ep += 1; samp and samp.set_epoch(ep); ti = iter(tl); x, y = next(ti)
            x, y = x.to(dev), y.to(dev)
            with torch.autocast('cuda', torch.bfloat16): _, l = model(x, y); l = l / ga
            scaler.scale(l).backward(); la += l.item()
        lrn = get_lr(step)
        for g in optim.param_groups: g['lr'] = lrn
        scaler.unscale_(optim); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim); scaler.update(); step += 1

        if step % 100 == 0:
            logger.info(f"Step {step}/{steps}: loss={la:.4f} lr={lrn:.2e} {(100*gb*seq)/(time.time()-t0)/1e6:.2f}M tok/s")
            hist['loss'].append(la); hist['step'].append(step); la = 0; t0 = time.time()
        if step % 2000 == 0 or step == steps:
            ppl = evaluate(bm, vl, dev); hist['ppl'].append(ppl)
            ib = ppl < best; best = min(ppl, best)
            logger.info(f"Step {step}: ppl={ppl:.2f}{'*' if ib else ''}")
            if ib and rank == 0: torch.save({'model': bm.state_dict(), 'ppl': best}, out / f"lm_{opt}_best.pt")

    if rank == 0:
        with open(out / f"lm_{opt}_results.json", 'w') as f: json.dump({'opt': opt, 'config': cfg, 'best_ppl': best, 'hist': hist}, f, indent=2)
        logger.info(f"Best PPL: {best:.2f}")
    return best

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--optimizer', default='all')
    pa.add_argument('--data', default='/blue/wdixon/wang.yixuan/rlo_experiments/data/wikitext')
    pa.add_argument('--output', default='/blue/wdixon/wang.yixuan/rlo_experiments/lm')
    pa.add_argument('--steps', type=int, default=100000)
    args = pa.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    if 'RANK' in os.environ:
        rank, ws, lr = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(lr); dist.init_process_group('nccl')
    else: rank, ws, lr = 0, 1, 0

    out = Path(args.output); out.mkdir(exist_ok=True, parents=True)
    logger = setup_logger(out, rank, "lm")
    opts = list(CONFIGS.keys()) if args.optimizer == 'all' else [args.optimizer]
    res = {}
    for o in opts:
        try: gc.collect(); torch.cuda.empty_cache(); res[o] = train_lm(o, Path(args.data), out, args.steps, rank, ws, lr, logger)
        except Exception as e: logger.error(f"Error {o}: {e}"); import traceback; logger.error(traceback.format_exc())
    if rank == 0 and res:
        logger.info("=" * 70 + "\nLM Results (PPL):")
        for o, p in sorted(res.items(), key=lambda x: x[1]): logger.info(f"  {o}: {p:.2f}")
    dist.is_initialized() and dist.destroy_process_group()

if __name__ == '__main__': main()
