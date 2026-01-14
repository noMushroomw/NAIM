#!/usr/bin/env python3
"""Ablation Studies on ResNet-50 / CIFAR-100: lr, batch size, belief, eta, component"""
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
import torchvision.transforms as T
from torchvision.datasets import CIFAR100
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

def create_loader(path, bs, train, ws, rank):
    tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(),
                   T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]) if train else \
         T.Compose([T.ToTensor(), T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    ds = CIFAR100(root=path, train=train, download=True, transform=tf)
    samp = DistributedSampler(ds, ws, rank, shuffle=train) if ws > 1 else None
    return DataLoader(ds, bs, shuffle=(samp is None and train), sampler=samp, num_workers=4, pin_memory=True, drop_last=train), samp

@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); c, t = 0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        with torch.autocast('cuda', torch.bfloat16): c += (model(x).argmax(1) == y).sum().item()
        t += y.size(0)
    return 100.0 * c / t

def train_one(cfg, data, out, rank, ws, lr, logger):
    dev = torch.device(f'cuda:{lr}')
    bs, epochs = cfg['batch_size'], cfg.get('epochs', 30)
    pg = bs // ws
    logger.info(f"Config: {cfg}")
    tl, ts = create_loader(data, pg, True, ws, rank)
    vl, _ = create_loader(data, pg, False, 1, 0)
    
    model = resnet50(weights=None, num_classes=100).to(dev)
    if ws > 1: model = DDP(model, device_ids=[lr])
    
    p = model.parameters()
    oname = cfg['optimizer']
    if oname == 'lion': opt = Lion(p, lr=cfg['lr'], weight_decay=cfg['wd'])
    elif oname == 'rlo': opt = RLO(p, lr=cfg['lr'], weight_decay=cfg['wd'], belief_coef=cfg.get('belief_coef', 0.1))
    elif oname == 'rlo_lambda_a': opt = RLO_LambdaA(p, lr=cfg['lr'], weight_decay=cfg['wd'], lambda_b=cfg.get('lambda_b', 0.1))
    elif oname == 'smooth_lifted_rlo': opt = SmoothLiftedRLO(p, lr=cfg['lr'], weight_decay=cfg['wd'], lambda_b=cfg.get('lambda_b', 0.1), eta=cfg.get('eta', 0.3))
    else: opt = torch.optim.AdamW(p, lr=cfg['lr'], weight_decay=cfg['wd'])
    
    total, warmup = len(tl) * epochs, len(tl) * 5
    def get_lr(s): return cfg['lr'] * s / max(1, warmup) if s < warmup else cfg['lr'] * 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total - warmup)))
    
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    best, step = 0.0, 0
    
    for ep in range(epochs):
        if ts: ts.set_epoch(ep)
        model.train()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            for g in opt.param_groups: g['lr'] = get_lr(step)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', torch.bfloat16): loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); step += 1
        
        bm = model.module if hasattr(model, 'module') else model
        acc = evaluate(bm, vl, dev)
        best = max(acc, best)
        if (ep + 1) % 10 == 0: logger.info(f"E{ep+1}: acc={acc:.2f}% best={best:.2f}%")
    
    return best

def run_study(study, data, out, rank, ws, lr, logger):
    results = {}
    
    if study == 'lr' or study == 'all':
        logger.info("=" * 70 + "\nStudy: Learning Rate Sensitivity")
        for lrv in [5e-5, 1e-4, 2e-4, 5e-4, 1e-3]:
            for opt in ['lion', 'rlo', 'smooth_lifted_rlo']:
                cfg = {'optimizer': opt, 'lr': lrv, 'wd': 1.0, 'batch_size': 128, 'epochs': 30}
                try:
                    gc.collect(); torch.cuda.empty_cache()
                    acc = train_one(cfg, data, out, rank, ws, lr, logger)
                    results[f'lr_{lrv}_{opt}'] = acc
                    logger.info(f"LR={lrv} {opt}: {acc:.2f}%")
                except Exception as e: logger.error(f"Error: {e}")
    
    if study == 'batch' or study == 'all':
        logger.info("=" * 70 + "\nStudy: Batch Size Sensitivity")
        base_lr = 3e-4
        for bs in [64, 128, 256, 512]:
            scaled_lr = base_lr * (bs / 128)
            for opt in ['lion', 'rlo', 'smooth_lifted_rlo']:
                cfg = {'optimizer': opt, 'lr': scaled_lr, 'wd': 1.0, 'batch_size': bs, 'epochs': 30}
                try:
                    gc.collect(); torch.cuda.empty_cache()
                    acc = train_one(cfg, data, out, rank, ws, lr, logger)
                    results[f'batch_{bs}_{opt}'] = acc
                    logger.info(f"Batch={bs} {opt}: {acc:.2f}%")
                except Exception as e: logger.error(f"Error: {e}")
    
    if study == 'belief' or study == 'all':
        logger.info("=" * 70 + "\nStudy: Belief Coefficient")
        for lb in [0.0, 0.05, 0.1, 0.2, 0.5]:
            cfg = {'optimizer': 'smooth_lifted_rlo', 'lr': 3e-4, 'wd': 1.0, 'batch_size': 128, 'epochs': 30, 'lambda_b': lb}
            try:
                gc.collect(); torch.cuda.empty_cache()
                acc = train_one(cfg, data, out, rank, ws, lr, logger)
                results[f'belief_{lb}'] = acc
                logger.info(f"lambda_b={lb}: {acc:.2f}%")
            except Exception as e: logger.error(f"Error: {e}")
    
    if study == 'eta' or study == 'all':
        logger.info("=" * 70 + "\nStudy: Fiber Contraction Rate")
        for eta in [0.1, 0.2, 0.3, 0.5, 1.0]:
            cfg = {'optimizer': 'smooth_lifted_rlo', 'lr': 3e-4, 'wd': 1.0, 'batch_size': 128, 'epochs': 30, 'eta': eta}
            try:
                gc.collect(); torch.cuda.empty_cache()
                acc = train_one(cfg, data, out, rank, ws, lr, logger)
                results[f'eta_{eta}'] = acc
                logger.info(f"eta={eta}: {acc:.2f}%")
            except Exception as e: logger.error(f"Error: {e}")
    
    if study == 'component' or study == 'all':
        logger.info("=" * 70 + "\nStudy: Component Analysis")
        configs = [
            ('lion_base', {'optimizer': 'lion', 'lr': 3e-4, 'wd': 1.0}),
            ('rlo_sign', {'optimizer': 'rlo', 'lr': 3e-4, 'wd': 1.0, 'belief_coef': 0.0}),
            ('rlo_belief', {'optimizer': 'rlo', 'lr': 3e-4, 'wd': 1.0, 'belief_coef': 0.1}),
            ('smooth_no_belief', {'optimizer': 'smooth_lifted_rlo', 'lr': 3e-4, 'wd': 1.0, 'lambda_b': 0.0}),
            ('smooth_full', {'optimizer': 'smooth_lifted_rlo', 'lr': 3e-4, 'wd': 1.0, 'lambda_b': 0.1}),
            ('rlo_lambda_a', {'optimizer': 'rlo_lambda_a', 'lr': 3e-4, 'wd': 1.0}),
        ]
        for name, cfg in configs:
            cfg['batch_size'] = 128; cfg['epochs'] = 30
            try:
                gc.collect(); torch.cuda.empty_cache()
                acc = train_one(cfg, data, out, rank, ws, lr, logger)
                results[f'comp_{name}'] = acc
                logger.info(f"{name}: {acc:.2f}%")
            except Exception as e: logger.error(f"Error: {e}")
    
    return results

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--study', default='all', choices=['all', 'lr', 'batch', 'belief', 'eta', 'component'])
    pa.add_argument('--data', default='/blue/wdixon/wang.yixuan/rlo_experiments/data')
    pa.add_argument('--output', default='/blue/wdixon/wang.yixuan/rlo_experiments/ablation')
    args = pa.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    if 'RANK' in os.environ:
        rank, ws, lr = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(lr); dist.init_process_group('nccl')
    else: rank, ws, lr = 0, 1, 0

    out = Path(args.output); out.mkdir(exist_ok=True, parents=True)
    logger = setup_logger(out, rank, "ablation")
    
    results = run_study(args.study, Path(args.data), out, rank, ws, lr, logger)
    
    if rank == 0:
        with open(out / "ablation_results.json", 'w') as f: json.dump(results, f, indent=2)
        logger.info("=" * 70 + "\nAll Results:")
        for k, v in sorted(results.items()): logger.info(f"  {k}: {v:.2f}%")
    
    dist.is_initialized() and dist.destroy_process_group()

if __name__ == '__main__': main()
