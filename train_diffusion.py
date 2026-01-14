#!/usr/bin/env python3
"""
Section 4.3: Diffusion Models with FID - LION Paper Table 12 EXACT

Table 12 "Image generation on ImageNet":
- AdamW: lr=3e-4, wd=0.01, β=(0.9, 0.999)
- Lion: lr=3e-5, wd=0.1, β=(0.9, 0.99)  # 0.1x lr, 10x wd
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

class SinEmb(nn.Module):
    def __init__(self, d): super().__init__(); self.d = d
    def forward(self, t):
        h = self.d // 2; e = math.log(10000) / (h - 1)
        e = torch.exp(torch.arange(h, device=t.device) * -e)
        e = t[:, None] * e[None, :]; return torch.cat([e.sin(), e.cos()], -1)

class ResBlk(nn.Module):
    def __init__(self, ic, oc, td):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(ic, oc, 3, padding=1), nn.Conv2d(oc, oc, 3, padding=1)
        self.tm = nn.Linear(td, oc); self.n1, self.n2 = nn.GroupNorm(8, ic), nn.GroupNorm(8, oc)
        self.skip = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()
    def forward(self, x, t):
        h = self.c1(F.silu(self.n1(x))); h = h + self.tm(F.silu(t))[:, :, None, None]
        return self.c2(F.silu(self.n2(h))) + self.skip(x)

class UNet(nn.Module):
    def __init__(self, ic=3, bc=128, mult=(1, 2, 4), td=256):
        super().__init__()
        self.tm = nn.Sequential(SinEmb(td), nn.Linear(td, td * 4), nn.GELU(), nn.Linear(td * 4, td))
        self.enc, self.dn, inc, chs = nn.ModuleList(), nn.ModuleList(), ic, [bc]
        for m in mult:
            oc = bc * m; self.enc.append(ResBlk(inc, oc, td)); self.dn.append(nn.Conv2d(oc, oc, 3, 2, 1))
            chs.append(oc); inc = oc
        self.mid = ResBlk(inc, inc, td)
        self.dec, self.up = nn.ModuleList(), nn.ModuleList()
        for m in reversed(mult):
            oc = bc * m; self.up.append(nn.ConvTranspose2d(inc, oc, 4, 2, 1))
            self.dec.append(ResBlk(oc + chs.pop(), oc, td)); inc = oc
        self.out = nn.Conv2d(inc, ic, 1)
    def forward(self, x, t):
        te = self.tm(t); hs = []
        for e, d in zip(self.enc, self.dn): x = e(x, te); hs.append(x); x = d(x)
        x = self.mid(x, te)
        for u, dc in zip(self.up, self.dec): x = u(x); x = torch.cat([x, hs.pop()], 1); x = dc(x, te)
        return self.out(x)

class GaussianDiffusion:
    def __init__(self, T=1000, bs=1e-4, be=0.02):
        self.T = T; b = torch.linspace(bs, be, T); a = 1 - b; ac = torch.cumprod(a, 0)
        self.buf = {'b': b, 'ac': ac, 'sac': ac.sqrt(), 'soac': (1 - ac).sqrt()}
    def to(self, dev):
        for k in self.buf: self.buf[k] = self.buf[k].to(dev)
        return self
    def q_sample(self, x0, t, n=None):
        n = torch.randn_like(x0) if n is None else n
        return self.buf['sac'][t][:, None, None, None] * x0 + self.buf['soac'][t][:, None, None, None] * n
    def p_losses(self, model, x0, t):
        n = torch.randn_like(x0); return F.mse_loss(model(self.q_sample(x0, t, n), t.float()), n)
    @torch.no_grad()
    def p_sample(self, model, x, t):
        tb = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        pn = model(x, tb.float()); a = 1 - self.buf['b'][t]
        mean = (1 / a.sqrt()) * (x - self.buf['b'][t] / self.buf['soac'][t] * pn)
        return mean + self.buf['b'][t].sqrt() * torch.randn_like(x) if t > 0 else mean
    @torch.no_grad()
    def sample(self, model, shape, dev):
        x = torch.randn(shape, device=dev)
        for t in reversed(range(self.T)): x = self.p_sample(model, x, t)
        return x

class InceptionFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            inc = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        except: from torchvision.models import inception_v3; inc = inception_v3(pretrained=True)
        self.blks = nn.Sequential(inc.Conv2d_1a_3x3, inc.Conv2d_2a_3x3, inc.Conv2d_2b_3x3, nn.MaxPool2d(3, 2),
            inc.Conv2d_3b_1x1, inc.Conv2d_4a_3x3, nn.MaxPool2d(3, 2), inc.Mixed_5b, inc.Mixed_5c, inc.Mixed_5d,
            inc.Mixed_6a, inc.Mixed_6b, inc.Mixed_6c, inc.Mixed_6d, inc.Mixed_6e, inc.Mixed_7a, inc.Mixed_7b,
            inc.Mixed_7c, nn.AdaptiveAvgPool2d((1, 1)))
        self.eval()
        for p in self.parameters(): p.requires_grad = False
    def forward(self, x):
        x = F.interpolate(x, (299, 299), mode='bilinear', align_corners=False)
        return self.blks((x - 0.5) / 0.5).flatten(1)

def calc_fid(rf, ff):
    m1, m2 = rf.mean(0), ff.mean(0); s1, s2 = torch.cov(rf.T), torch.cov(ff.T)
    d = m1 - m2; cm = torch.linalg.eigvalsh(s1 @ s2).clamp(min=0).sqrt().sum()
    return (d.dot(d) + s1.trace() + s2.trace() - 2 * cm).item()

@torch.no_grad()
def compute_fid(model, diff, loader, dev, n=500, sz=64):
    model.eval()
    try: inc = InceptionFeatures().to(dev)
    except: return float('nan')
    rf = []
    for x, _ in loader:
        if len(rf) * x.size(0) >= n: break
        rf.append(inc(x.to(dev)))
    rf = torch.cat(rf)[:n]
    ff = []
    while len(ff) * 32 < n:
        s = diff.sample(model, (32, 3, sz, sz), dev).clamp(-1, 1) * 0.5 + 0.5
        ff.append(inc(s))
    ff = torch.cat(ff)[:n]
    return calc_fid(rf, ff)

# ============= LION Paper Table 12 EXACT for Diffusion =============
# Table 12 "Image generation on ImageNet": AdamW lr=3e-4 wd=0.01, Lion lr=3e-5 wd=0.1
CONFIGS = {
    'adamw': {'lr': 3e-4, 'wd': 0.01, 'betas': (0.9, 0.999)},
    'lion': {'lr': 3e-5, 'wd': 0.1, 'betas': (0.9, 0.99)},  # 0.1x lr, 10x wd
    'rlo': {'lr': 3e-5, 'wd': 0.1, 'betas': (0.9, 0.99), 'belief_coef': 0.1},
    'rlo_lambda_a': {'lr': 3e-5, 'wd': 0.1, 'lambda_b': 0.1},
    'smooth_lifted_rlo': {'lr': 3e-5, 'wd': 0.1, 'lambda_b': 0.1, 'eta': 0.3},
}

def create_optimizer(model, name, cfg):
    p = model.parameters()
    if name == 'adamw': return torch.optim.AdamW(p, lr=cfg['lr'], weight_decay=cfg['wd'], betas=cfg.get('betas', (0.9, 0.999)))
    if name == 'lion': return Lion(p, lr=cfg['lr'], weight_decay=cfg['wd'], betas=cfg.get('betas', (0.9, 0.99)))
    if name == 'rlo': return RLO(p, lr=cfg['lr'], weight_decay=cfg['wd'], betas=cfg.get('betas', (0.9, 0.99)), belief_coef=cfg.get('belief_coef', 0.1))
    if name == 'rlo_lambda_a': return RLO_LambdaA(p, lr=cfg['lr'], weight_decay=cfg['wd'], lambda_b=cfg.get('lambda_b', 0.1))
    if name == 'smooth_lifted_rlo': return SmoothLiftedRLO(p, lr=cfg['lr'], weight_decay=cfg['wd'], lambda_b=cfg.get('lambda_b', 0.1), eta=cfg.get('eta', 0.3))
    raise ValueError(name)

def train_diff(opt, res, data, out, epochs, rank, ws, lr, logger):
    dev = torch.device(f'cuda:{lr}'); cfg = CONFIGS[opt]
    bs = 64 if res <= 64 else 32; pg = bs // ws
    logger.info(f"{'='*70}\nDiffusion {res}x{res} | {opt}\nLR={cfg['lr']} WD={cfg['wd']}\n{'='*70}")

    tf = T.Compose([T.Resize(res), T.CenterCrop(res), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    ds = ImageFolder(data / 'train', tf)
    samp = DistributedSampler(ds, ws, rank) if ws > 1 else None
    tl = DataLoader(ds, pg, shuffle=samp is None, sampler=samp, num_workers=2, pin_memory=True, drop_last=True, persistent_workers=False)
    vl = DataLoader(ds, pg * 2, num_workers=2, pin_memory=True)

    model = UNet(3, 128 if res <= 64 else 96).to(dev)
    if ws > 1: model = DDP(model, device_ids=[lr])
    logger.info(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    diff = GaussianDiffusion(1000).to(dev)
    optim = create_optimizer(model, opt, cfg); scaler = torch.amp.GradScaler('cuda')

    hist, best = {'loss': [], 'fid': []}, float('inf')
    for ep in range(epochs):
        if samp: samp.set_epoch(ep)
        model.train(); ls, nb, t0 = 0.0, 0, time.time()
        for x, _ in tl:
            x = x.to(dev); t = torch.randint(0, diff.T, (x.size(0),), device=dev)
            optim.zero_grad(set_to_none=True)
            with torch.autocast('cuda', torch.bfloat16): loss = diff.p_losses(model, x, t)
            scaler.scale(loss).backward(); scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update(); ls += loss.item(); nb += 1

        al = ls / nb; hist['loss'].append(al); fid = float('nan')
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            bm = model.module if hasattr(model, 'module') else model
            try: fid = compute_fid(bm, diff, vl, dev, 500, res)
            except: pass
            hist['fid'].append(fid)
            ib = fid < best if not math.isnan(fid) else False; best = min(fid, best) if not math.isnan(fid) else best
            logger.info(f"E{ep+1}: loss={al:.4f} FID={fid:.2f}{'*' if ib else ''} {time.time()-t0:.0f}s")
            if ib and rank == 0: torch.save({'model': bm.state_dict(), 'fid': best}, out / f"diff_{res}_{opt}_best.pt")
        else: logger.info(f"E{ep+1}: loss={al:.4f} {time.time()-t0:.0f}s")

    if rank == 0:
        with open(out / f"diff_{res}_{opt}_results.json", 'w') as f: json.dump({'opt': opt, 'config': cfg, 'res': res, 'best_fid': best, 'hist': hist}, f, indent=2)
        logger.info(f"Best FID: {best:.2f}")
    return best

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--resolution', type=int, default=64, choices=[64, 128])
    pa.add_argument('--optimizer', default='all')
    pa.add_argument('--data', default='/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder')
    pa.add_argument('--output', default='/blue/wdixon/wang.yixuan/rlo_experiments/diffusion')
    pa.add_argument('--epochs', type=int, default=100)
    args = pa.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    if 'RANK' in os.environ:
        rank, ws, lr = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(lr); dist.init_process_group('nccl')
    else: rank, ws, lr = 0, 1, 0

    out = Path(args.output); out.mkdir(exist_ok=True, parents=True)
    logger = setup_logger(out, rank, f"diff_{args.resolution}")
    opts = list(CONFIGS.keys()) if args.optimizer == 'all' else [args.optimizer]
    res = {}
    for o in opts:
        try: gc.collect(); torch.cuda.empty_cache(); res[o] = train_diff(o, args.resolution, Path(args.data), out, args.epochs, rank, ws, lr, logger)
        except Exception as e: logger.error(f"Error {o}: {e}"); import traceback; logger.error(traceback.format_exc())
    if rank == 0 and res:
        logger.info("=" * 70 + f"\nDiffusion {args.resolution}x{args.resolution} (FID):")
        for o, f in sorted(res.items(), key=lambda x: x[1] if not math.isnan(x[1]) else float('inf')): logger.info(f"  {o}: {f:.2f}")
    dist.is_initialized() and dist.destroy_process_group()

if __name__ == '__main__': main()
