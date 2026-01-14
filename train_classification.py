#!/usr/bin/env python3
"""
Section 4.1: Image Classification - LION Paper Table 12 EXACT
ResNet-50/CIFAR-100, ViT-S/16, ViT-B/16/ImageNet

CRITICAL HYPERPARAMETERS from Table 12:
- Lion uses 0.1x LR of AdamW
- Lion uses 10x WD of AdamW
- Lion β2=0.99, AdamW β2=0.999
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
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Optimizer
from torch.utils.checkpoint import checkpoint
import torchvision.transforms as T
from torchvision.datasets import ImageFolder, CIFAR100
from torchvision.models import resnet50
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

class Attn(nn.Module):
    def __init__(self, d, h=8):
        super().__init__(); self.h, self.hd = h, d // h
        self.qkv, self.proj = nn.Linear(d, d * 3), nn.Linear(d, d)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        return self.proj(F.scaled_dot_product_attention(*qkv.unbind(0)).transpose(1, 2).reshape(B, N, C))

class Blk(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.n1, self.attn, self.n2 = nn.LayerNorm(d), Attn(d, h), nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
    def forward(self, x): x = x + self.attn(self.n1(x)); return x + self.mlp(self.n2(x))

class ViT(nn.Module):
    def __init__(self, d=384, depth=12, h=6, nc=1000):
        super().__init__()
        self.pe = nn.Conv2d(3, d, 16, 16)
        self.cls, self.pos = nn.Parameter(torch.zeros(1, 1, d)), nn.Parameter(torch.zeros(1, 197, d))
        self.blks = nn.ModuleList([Blk(d, h) for _ in range(depth)])
        self.n, self.head = nn.LayerNorm(d), nn.Linear(d, nc)
        nn.init.trunc_normal_(self.pos, std=0.02); nn.init.trunc_normal_(self.cls, std=0.02)
    def forward(self, x):
        B = x.shape[0]; x = self.pe(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(B, -1, -1), x], 1) + self.pos
        for b in self.blks: x = checkpoint(b, x, use_reentrant=False) if self.training else b(x)
        return self.head(self.n(x[:, 0]))

def create_model(name, nc=1000):
    if name == 'resnet50': return resnet50(weights=None, num_classes=nc)
    if name == 'vit_s16': return ViT(d=384, depth=12, h=6, nc=nc)
    if name == 'vit_b16': return ViT(d=768, depth=12, h=12, nc=nc)
    raise ValueError(name)

def create_cifar100_loader(path, bs, train, ws, rank):
    tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(),
                   T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]) if train else \
         T.Compose([T.ToTensor(), T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    ds = CIFAR100(root=path, train=train, download=True, transform=tf)
    samp = DistributedSampler(ds, ws, rank, shuffle=train) if ws > 1 else None
    return DataLoader(ds, bs, shuffle=(samp is None and train), sampler=samp, num_workers=4, pin_memory=True, drop_last=train), samp

def create_imagenet_loader(path, bs, train, ws, rank, aug=False):
    M, S = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if train:
        tfs = [T.RandomResizedCrop(224), T.RandomHorizontalFlip()]
        if aug: tfs.append(T.RandAugment(2, 15))
        tfs.extend([T.ToTensor(), T.Normalize(M, S)]); tf, sp = T.Compose(tfs), 'train'
    else: tf, sp = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(M, S)]), 'val'
    ds = ImageFolder(path / sp, tf)
    samp = DistributedSampler(ds, ws, rank, shuffle=train) if ws > 1 else None
    return DataLoader(ds, bs, shuffle=(samp is None and train), sampler=samp, num_workers=2, pin_memory=True, drop_last=train, persistent_workers=False), samp

class Mixup:
    def __init__(self, a=0.5, nc=1000): self.a, self.nc = a, nc
    def __call__(self, x, y):
        lam = np.random.beta(self.a, self.a); idx = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1 - lam) * x[idx], lam * F.one_hot(y, self.nc).float() + (1 - lam) * F.one_hot(y[idx], self.nc).float()

# ============= LION Paper Table 12 EXACT =============
CONFIGS = {
    'resnet50': {
        'dataset': 'cifar100', 'num_classes': 100, 'epochs': 200, 'global_batch': 128,
        'warmup_epochs': 5, 'augment': False, 'mixup': False, 'label_smoothing': 0.0,
        'optimizers': {
            'sgd': {'lr': 0.1, 'wd': 5e-4, 'momentum': 0.9},
            'rmsprop': {'lr': 1e-3, 'wd': 1e-4},
            'adam': {'lr': 1e-3, 'wd': 1e-4},
            # Table 12 ResNet-50: AdamW lr=3e-3 wd=0.1, Lion lr=3e-4 wd=1.0
            'adamw': {'lr': 3e-3, 'wd': 0.1, 'betas': (0.9, 0.999)},
            'lion': {'lr': 3e-4, 'wd': 1.0, 'betas': (0.9, 0.99)},
            'rlo': {'lr': 3e-4, 'wd': 1.0, 'betas': (0.9, 0.99), 'belief_coef': 0.1},
            'rlo_lambda_a': {'lr': 3e-4, 'wd': 1.0, 'lambda_b': 0.1},
            'smooth_lifted_rlo': {'lr': 3e-4, 'wd': 1.0, 'lambda_b': 0.1, 'eta': 0.3},
        },
    },
    'vit_s16': {
        'dataset': 'imagenet', 'num_classes': 1000, 'epochs': 300, 'global_batch': 1024,
        'warmup_epochs': 10, 'augment': True, 'mixup': True, 'label_smoothing': 0.1,
        'optimizers': {
            # Table 12 ViT-S/16 RandAug: AdamW lr=3e-3 wd=0.1, Lion lr=3e-4 wd=1.0
            'adamw': {'lr': 3e-3, 'wd': 0.1, 'betas': (0.9, 0.999)},
            'lion': {'lr': 3e-4, 'wd': 1.0, 'betas': (0.9, 0.99)},
            'rlo': {'lr': 3e-4, 'wd': 1.0, 'betas': (0.9, 0.99), 'belief_coef': 0.1},
            'rlo_lambda_a': {'lr': 3e-4, 'wd': 1.0, 'lambda_b': 0.1},
            'smooth_lifted_rlo': {'lr': 3e-4, 'wd': 1.0, 'lambda_b': 0.1, 'eta': 0.3},
        },
    },
    'vit_b16': {
        'dataset': 'imagenet', 'num_classes': 1000, 'epochs': 300, 'global_batch': 1024,
        'warmup_epochs': 10, 'augment': True, 'mixup': True, 'label_smoothing': 0.1,
        'optimizers': {
            # Table 12 ViT-B/16 RandAug: AdamW lr=1e-3 wd=1.0, Lion lr=1e-4 wd=10.0
            'adamw': {'lr': 1e-3, 'wd': 1.0, 'betas': (0.9, 0.999)},
            'lion': {'lr': 1e-4, 'wd': 10.0, 'betas': (0.9, 0.99)},
            'rlo': {'lr': 1e-4, 'wd': 10.0, 'betas': (0.9, 0.99), 'belief_coef': 0.1},
            'rlo_lambda_a': {'lr': 1e-4, 'wd': 10.0, 'lambda_b': 0.1},
            'smooth_lifted_rlo': {'lr': 1e-4, 'wd': 10.0, 'lambda_b': 0.1, 'eta': 0.3},
        },
    },
}

def create_optimizer(model, name, cfg):
    p = model.parameters(); lr, wd = cfg['lr'], cfg['wd']
    if name == 'sgd': return torch.optim.SGD(p, lr=lr, weight_decay=wd, momentum=cfg.get('momentum', 0.9))
    if name == 'rmsprop': return torch.optim.RMSprop(p, lr=lr, weight_decay=wd)
    if name == 'adam': return torch.optim.Adam(p, lr=lr, weight_decay=wd)
    if name == 'adamw': return torch.optim.AdamW(p, lr=lr, weight_decay=wd, betas=cfg.get('betas', (0.9, 0.999)))
    if name == 'lion': return Lion(p, lr=lr, weight_decay=wd, betas=cfg.get('betas', (0.9, 0.99)))
    if name == 'rlo': return RLO(p, lr=lr, weight_decay=wd, betas=cfg.get('betas', (0.9, 0.99)), belief_coef=cfg.get('belief_coef', 0.1))
    if name == 'rlo_lambda_a': return RLO_LambdaA(p, lr=lr, weight_decay=wd, lambda_b=cfg.get('lambda_b', 0.1))
    if name == 'smooth_lifted_rlo': return SmoothLiftedRLO(p, lr=lr, weight_decay=wd, lambda_b=cfg.get('lambda_b', 0.1), eta=cfg.get('eta', 0.3))
    raise ValueError(name)

@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); c, t = 0, 0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast('cuda', torch.bfloat16): c += (model(x).argmax(1) == y).sum().item()
        t += y.size(0)
    return 100.0 * c / t

def train_one(mname, oname, data, imnet, out, rank, ws, lr, logger):
    dev = torch.device(f'cuda:{lr}')
    cfg, ocfg = CONFIGS[mname], CONFIGS[mname]['optimizers'][oname]
    pg = cfg['global_batch'] // max(1, ws)
    logger.info(f"{'='*70}\n{mname} | {oname} | {cfg['dataset']}\nLR={ocfg['lr']} WD={ocfg['wd']}\n{'='*70}")

    if cfg['dataset'] == 'cifar100':
        tl, ts = create_cifar100_loader(data, pg, True, ws, rank)
        vl, _ = create_cifar100_loader(data, pg, False, 1, 0)
    else:
        tl, ts = create_imagenet_loader(imnet, pg, True, ws, rank, cfg['augment'])
        vl, _ = create_imagenet_loader(imnet, pg, False, 1, 0)

    model = create_model(mname, cfg['num_classes']).to(dev)
    if ws > 1: model = DDP(model, device_ids=[lr])
    logger.info(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    opt = create_optimizer(model, oname, ocfg)
    total, warmup = len(tl) * cfg['epochs'], len(tl) * cfg['warmup_epochs']
    def get_lr(s): return ocfg['lr'] * s / max(1, warmup) if s < warmup else ocfg['lr'] * 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total - warmup)))

    mixup = Mixup(nc=cfg['num_classes']) if cfg['mixup'] else None
    crit = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    scaler = torch.amp.GradScaler('cuda')
    hist, best, step = {'loss': [], 'acc': []}, 0.0, 0

    for ep in range(cfg['epochs']):
        if ts: ts.set_epoch(ep)
        model.train(); ls = 0.0
        for x, y in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            if mixup: x, ys = mixup(x, y); soft = True
            else: ys, soft = None, False
            for g in opt.param_groups: g['lr'] = get_lr(step)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', torch.bfloat16):
                logits = model(x)
                loss = -torch.sum(F.log_softmax(logits, 1) * ys, 1).mean() if soft else crit(logits, y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); ls += loss.item(); step += 1

        bm = model.module if hasattr(model, 'module') else model
        acc = evaluate(bm, vl, dev); hist['loss'].append(ls / len(tl)); hist['acc'].append(acc)
        ib = acc > best; best = max(acc, best)
        if (ep + 1) % 10 == 0 or ep == cfg['epochs'] - 1:
            logger.info(f"E{ep+1:3d}: loss={ls/len(tl):.4f} acc={acc:.2f}%{'*' if ib else ''}")
        if ib and rank == 0: torch.save({'model': bm.state_dict(), 'acc': best}, out / f"{mname}_{oname}_best.pt")

    if rank == 0:
        with open(out / f"{mname}_{oname}_results.json", 'w') as f:
            json.dump({'model': mname, 'opt': oname, 'config': ocfg, 'best_acc': best, 'hist': hist}, f, indent=2)
        logger.info(f"Best: {best:.2f}%")
    return best

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--model', required=True, choices=['resnet50', 'vit_s16', 'vit_b16'])
    pa.add_argument('--optimizer', default='all')
    pa.add_argument('--data', default='/blue/wdixon/wang.yixuan/rlo_experiments/data')
    pa.add_argument('--imagenet', default='/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder')
    pa.add_argument('--output', default='/blue/wdixon/wang.yixuan/rlo_experiments/classification')
    args = pa.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    if 'RANK' in os.environ:
        rank, ws, lr = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(lr); dist.init_process_group('nccl')
    else: rank, ws, lr = 0, 1, 0

    out = Path(args.output); out.mkdir(exist_ok=True, parents=True)
    logger = setup_logger(out, rank, f"cls_{args.model}")
    opts = list(CONFIGS[args.model]['optimizers'].keys()) if args.optimizer == 'all' else [args.optimizer]
    res = {}
    for o in opts:
        try: gc.collect(); torch.cuda.empty_cache(); res[o] = train_one(args.model, o, Path(args.data), Path(args.imagenet), out, rank, ws, lr, logger)
        except Exception as e: logger.error(f"Error {o}: {e}"); import traceback; logger.error(traceback.format_exc())
    if rank == 0 and res:
        logger.info("=" * 70)
        for o, a in sorted(res.items(), key=lambda x: x[1], reverse=True): logger.info(f"  {o}: {a:.2f}%")
    dist.is_initialized() and dist.destroy_process_group()

if __name__ == '__main__': main()
