#!/usr/bin/env python3
"""
Section 4.2: Vision-Language (LiT) - LION Paper Table 12 EXACT

Table 12 "LiT-B/*-B": AdamW lr=1e-3, Lion lr=3e-4
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
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
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
                c = b1 * m + (1 - b1) * gr; d = gr - m
                p.add_(c.sign() + g["belief_coef"] * (d / d.norm().clamp(min=g["eps"])), alpha=-g["lr"])
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
                d = gr - m; b = g["lambda_b"] * (d / d.norm().clamp(min=g["eps"]))
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
                d = gr - m; b = g["lambda_b"] * (d / d.norm().clamp(min=g["eps"]))
                all_s.append(ss); all_b.append(b); all_p.append((p, g))
        if not all_p: return
        scale = self.sqrt_dim / sum((x * x).sum() for x in all_s).sqrt().clamp(min=1e-8)
        for (p, g), ss, b in zip(all_p, all_s, all_b):
            s = self.state[p]
            if g["weight_decay"] != 0: p.mul_(1 - g["lr"] * g["weight_decay"])
            dd = scale * ss + b; s["v"].mul_(1 - g["eta"]).add_(dd, alpha=g["eta"])
            p.add_(s["v"], alpha=-g["lr"]); s["m"].mul_(g["beta2"]).add_(p.grad, alpha=1 - g["beta2"])

class Attn(nn.Module):
    def __init__(self, d, h=8):
        super().__init__(); self.h, self.hd = h, d // h
        self.qkv, self.proj = nn.Linear(d, d * 3), nn.Linear(d, d)
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        a = (q @ k.transpose(-2, -1)) * (self.hd ** -0.5)
        if mask is not None: a = a.masked_fill(mask == 0, float('-inf'))
        return self.proj((a.softmax(-1) @ v).transpose(1, 2).reshape(B, N, C))

class Blk(nn.Module):
    def __init__(self, d, h, causal=False):
        super().__init__(); self.causal = causal
        self.n1, self.attn, self.n2 = nn.LayerNorm(d), Attn(d, h), nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
    def forward(self, x):
        mask = torch.tril(torch.ones(x.size(1), x.size(1), device=x.device)).unsqueeze(0).unsqueeze(0) if self.causal else None
        x = x + self.attn(self.n1(x), mask); return x + self.mlp(self.n2(x))

class ImgEnc(nn.Module):
    def __init__(self, d=768, depth=12, h=12, od=512):
        super().__init__()
        self.pe = nn.Conv2d(3, d, 32, 32)
        self.cls, self.pos = nn.Parameter(torch.zeros(1, 1, d)), nn.Parameter(torch.zeros(1, 50, d))
        self.blks = nn.ModuleList([Blk(d, h) for _ in range(depth)])
        self.n, self.proj = nn.LayerNorm(d), nn.Linear(d, od)
        nn.init.trunc_normal_(self.pos, std=0.02); nn.init.trunc_normal_(self.cls, std=0.02)
    def forward(self, x):
        B = x.shape[0]; x = self.pe(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(B, -1, -1), x], 1) + self.pos
        for b in self.blks: x = b(x)
        return self.proj(self.n(x[:, 0]))

class TxtEnc(nn.Module):
    def __init__(self, vocab=49408, d=512, depth=6, h=8, maxlen=77, od=512):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(maxlen, d)
        self.blks = nn.ModuleList([Blk(d, h, causal=True) for _ in range(depth)])
        self.n, self.proj = nn.LayerNorm(d), nn.Linear(d, od)
    def forward(self, t):
        B, T = t.shape; x = self.tok(t) + self.pos(torch.arange(T, device=t.device))
        for b in self.blks: x = b(x)
        return self.proj(self.n(x[torch.arange(B), t.argmax(-1)]))

class LiT(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.img, self.txt = ImgEnc(od=d), TxtEnc(od=d)
        self.scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        for p in self.img.parameters(): p.requires_grad = False
    def forward(self, img, tok):
        with torch.no_grad(): ie = F.normalize(self.img(img), dim=-1)
        te = F.normalize(self.txt(tok), dim=-1)
        s = self.scale.exp(); return s * ie @ te.t(), s * te @ ie.t()

class ImgTxtDS(Dataset):
    def __init__(self, folder, tf=None, maxlen=77):
        self.ds = ImageFolder(folder, tf); self.maxlen = maxlen
        self.v = {chr(i): i for i in range(256)}; self.vocab = 256
    def tok(self, t):
        tk = [self.v.get(c, 0) for c in t[:self.maxlen - 1]]
        return torch.tensor(tk + [0] * (self.maxlen - len(tk)), dtype=torch.long)
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        img, lbl = self.ds[i]
        return img, self.tok(f"a photo of {self.ds.classes[lbl].replace('_', ' ')}")

# ============= LION Paper Table 12 EXACT for LiT =============
# Table 12 "LiT-B/*-B": AdamW lr=1e-3, Lion lr=3e-4
CONFIGS = {
    'adamw': {'lr': 1e-3, 'wd': 0.0, 'betas': (0.9, 0.999)},
    'lion': {'lr': 3e-4, 'wd': 0.0, 'betas': (0.9, 0.99)},  # 0.3x lr
    'rlo': {'lr': 3e-4, 'wd': 0.0, 'betas': (0.9, 0.99), 'belief_coef': 0.1},
    'rlo_lambda_a': {'lr': 3e-4, 'wd': 0.0, 'lambda_b': 0.1},
    'smooth_lifted_rlo': {'lr': 3e-4, 'wd': 0.0, 'lambda_b': 0.1, 'eta': 0.3},
}

def create_optimizer(model, name, cfg):
    p = [pp for pp in model.parameters() if pp.requires_grad]
    if name == 'adamw': return torch.optim.AdamW(p, lr=cfg['lr'], weight_decay=cfg['wd'], betas=cfg.get('betas', (0.9, 0.999)))
    if name == 'lion': return Lion(p, lr=cfg['lr'], weight_decay=cfg['wd'], betas=cfg.get('betas', (0.9, 0.99)))
    if name == 'rlo': return RLO(p, lr=cfg['lr'], weight_decay=cfg['wd'], betas=cfg.get('betas', (0.9, 0.99)), belief_coef=cfg.get('belief_coef', 0.1))
    if name == 'rlo_lambda_a': return RLO_LambdaA(p, lr=cfg['lr'], weight_decay=cfg['wd'], lambda_b=cfg.get('lambda_b', 0.1))
    if name == 'smooth_lifted_rlo': return SmoothLiftedRLO(p, lr=cfg['lr'], weight_decay=cfg['wd'], lambda_b=cfg.get('lambda_b', 0.1), eta=cfg.get('eta', 0.3))
    raise ValueError(name)

def closs(li, lt):
    lbl = torch.arange(li.shape[0], device=li.device)
    return (F.cross_entropy(li, lbl) + F.cross_entropy(lt, lbl)) / 2

def train_lit(opt, data, out, epochs, rank, ws, lr, logger):
    dev = torch.device(f'cuda:{lr}'); cfg = CONFIGS[opt]
    gb, pg = 256, 256 // ws
    logger.info(f"{'='*70}\nLiT: {opt} | LR={cfg['lr']}\n{'='*70}")

    tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    ds = ImgTxtDS(data / 'train', tf)
    samp = DistributedSampler(ds, ws, rank) if ws > 1 else None
    tl = DataLoader(ds, pg, shuffle=samp is None, sampler=samp, num_workers=2, pin_memory=True, drop_last=True, persistent_workers=False)

    model = LiT().to(dev)
    if ws > 1: model = DDP(model, device_ids=[lr], find_unused_parameters=True)
    logger.info(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")

    optim = create_optimizer(model, opt, cfg); scaler = torch.amp.GradScaler('cuda')
    steps, warmup = len(tl) * epochs, len(tl) * 2
    def get_lr(s): return cfg['lr'] * s / max(1, warmup) if s < warmup else cfg['lr'] * 0.5 * (1 + math.cos(math.pi * (s - warmup) / (steps - warmup)))

    hist, best, step = {'loss': []}, float('inf'), 0
    for ep in range(epochs):
        if samp: samp.set_epoch(ep)
        model.train(); ls, t0 = 0.0, time.time()
        for img, tok in tl:
            img, tok = img.to(dev), tok.to(dev)
            for g in optim.param_groups: g['lr'] = get_lr(step)
            optim.zero_grad(set_to_none=True)
            with torch.autocast('cuda', torch.bfloat16):
                li, lt = model(img, tok); loss = closs(li, lt)
            scaler.scale(loss).backward(); scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update(); ls += loss.item(); step += 1

        al = ls / len(tl); hist['loss'].append(al)
        ib = al < best; best = min(al, best)
        logger.info(f"E{ep+1}: loss={al:.4f}{'*' if ib else ''} {time.time()-t0:.0f}s")
        if ib and rank == 0:
            bm = model.module if hasattr(model, 'module') else model
            torch.save({'model': bm.state_dict(), 'loss': best}, out / f"lit_{opt}_best.pt")

    if rank == 0:
        with open(out / f"lit_{opt}_results.json", 'w') as f: json.dump({'opt': opt, 'config': cfg, 'best_loss': best, 'hist': hist}, f, indent=2)
        logger.info(f"Best loss: {best:.4f}")
    return best

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--optimizer', default='all')
    pa.add_argument('--data', default='/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder')
    pa.add_argument('--output', default='/blue/wdixon/wang.yixuan/rlo_experiments/lit')
    pa.add_argument('--epochs', type=int, default=30)
    args = pa.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    if 'RANK' in os.environ:
        rank, ws, lr = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(lr); dist.init_process_group('nccl')
    else: rank, ws, lr = 0, 1, 0

    out = Path(args.output); out.mkdir(exist_ok=True, parents=True)
    logger = setup_logger(out, rank, "lit")
    opts = list(CONFIGS.keys()) if args.optimizer == 'all' else [args.optimizer]
    res = {}
    for o in opts:
        try: gc.collect(); torch.cuda.empty_cache(); res[o] = train_lit(o, Path(args.data), out, args.epochs, rank, ws, lr, logger)
        except Exception as e: logger.error(f"Error {o}: {e}"); import traceback; logger.error(traceback.format_exc())
    if rank == 0 and res:
        logger.info("=" * 70 + "\nLiT Results (loss):")
        for o, l in sorted(res.items(), key=lambda x: x[1]): logger.info(f"  {o}: {l:.4f}")
    dist.is_initialized() and dist.destroy_process_group()

if __name__ == '__main__': main()
