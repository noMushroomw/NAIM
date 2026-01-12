#!/usr/bin/env python3
"""
Section 4.3: Image Generation with Diffusion Models
====================================================
Following LION paper setup + FID evaluation

Model: U-Net style diffusion
Dataset: ImageNet subsampled to 64x64 / 128x128
Metric: FID (Frechet Inception Distance)
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
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Optimizer
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
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
# U-Net for Diffusion
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
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(F.silu(t))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=128, ch_mult=(1, 2, 4), time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(time_dim), nn.Linear(time_dim, time_dim * 4),
                                      nn.GELU(), nn.Linear(time_dim * 4, time_dim))
        
        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = base_ch
        in_c = in_ch
        chs = [ch]
        for mult in ch_mult:
            out_c = base_ch * mult
            self.enc_blocks.append(ResBlock(in_c, out_c, time_dim))
            self.downs.append(nn.Conv2d(out_c, out_c, 3, stride=2, padding=1))
            chs.append(out_c)
            in_c = out_c
        
        # Middle
        self.mid = ResBlock(in_c, in_c, time_dim)
        
        # Decoder
        self.dec_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        for mult in reversed(ch_mult):
            out_c = base_ch * mult
            self.ups.append(nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1))
            self.dec_blocks.append(ResBlock(out_c + chs.pop(), out_c, time_dim))
            in_c = out_c
        
        self.final = nn.Conv2d(in_c, in_ch, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        # Encoder
        hs = []
        for enc, down in zip(self.enc_blocks, self.downs):
            x = enc(x, t_emb)
            hs.append(x)
            x = down(x)
        
        # Middle
        x = self.mid(x, t_emb)
        
        # Decoder
        for up, dec in zip(self.ups, self.dec_blocks):
            x = up(x)
            x = torch.cat([x, hs.pop()], dim=1)
            x = dec(x, t_emb)
        
        return self.final(x)


# =============================================================================
# Diffusion Process
# =============================================================================
class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer_dict = {
            'betas': betas,
            'alphas_cumprod': alphas_cumprod,
            'sqrt_alphas_cumprod': alphas_cumprod.sqrt(),
            'sqrt_one_minus_alphas_cumprod': (1 - alphas_cumprod).sqrt(),
        }
    
    def to(self, device):
        for k, v in self.register_buffer_dict.items():
            self.register_buffer_dict[k] = v.to(device)
        return self
    
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.register_buffer_dict['sqrt_alphas_cumprod'][t][:, None, None, None]
        sqrt_one_minus_alpha = self.register_buffer_dict['sqrt_one_minus_alphas_cumprod'][t][:, None, None, None]
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
    
    def p_losses(self, model, x0, t):
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        predicted = model(x_noisy, t.float())
        return F.mse_loss(predicted, noise)
    
    @torch.no_grad()
    def p_sample(self, model, x, t):
        betas = self.register_buffer_dict['betas']
        sqrt_one_minus_alpha = self.register_buffer_dict['sqrt_one_minus_alphas_cumprod']
        sqrt_alpha = self.register_buffer_dict['sqrt_alphas_cumprod']
        
        t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        pred_noise = model(x, t_batch.float())
        
        alpha = 1 - betas[t]
        alpha_cumprod = sqrt_alpha[t] ** 2
        
        mean = (1 / alpha.sqrt()) * (x - betas[t] / sqrt_one_minus_alpha[t] * pred_noise)
        
        if t > 0:
            noise = torch.randn_like(x)
            sigma = betas[t].sqrt()
            return mean + sigma * noise
        return mean
    
    @torch.no_grad()
    def sample(self, model, shape, device):
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(model, x, t)
        return x


# =============================================================================
# FID Calculation (simplified)
# =============================================================================
class InceptionV3Features(nn.Module):
    """Extract features from Inception V3 for FID calculation"""
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        except:
            from torchvision.models import inception_v3
            inception = inception_v3(pretrained=True)
        
        # Remove final layers
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3,
            nn.MaxPool2d(3, 2), inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(3, 2), inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d,
            inception.Mixed_6e, inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Resize to 299x299 and normalize
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = (x - 0.5) / 0.5  # Normalize to [-1, 1]
        return self.blocks(x).flatten(1)


def calculate_fid(real_features, fake_features):
    """Calculate FID between two sets of features"""
    mu1, mu2 = real_features.mean(0), fake_features.mean(0)
    sigma1 = torch.cov(real_features.T)
    sigma2 = torch.cov(fake_features.T)
    
    diff = mu1 - mu2
    
    # Matrix square root using eigendecomposition
    covmean = torch.linalg.eigvalsh(sigma1 @ sigma2).clamp(min=0).sqrt().sum()
    
    fid = diff.dot(diff) + sigma1.trace() + sigma2.trace() - 2 * covmean
    return fid.item()


@torch.no_grad()
def compute_fid(model, diffusion, real_loader, device, n_samples=1000, img_size=64):
    """Compute FID score"""
    model.eval()
    
    # Load Inception
    try:
        inception = InceptionV3Features().to(device)
    except Exception as e:
        print(f"Could not load Inception for FID: {e}")
        return float('nan')
    
    # Get real features
    real_features = []
    n_real = 0
    for x, _ in real_loader:
        if n_real >= n_samples:
            break
        x = x.to(device)
        feat = inception(x)
        real_features.append(feat)
        n_real += x.size(0)
    real_features = torch.cat(real_features)[:n_samples]
    
    # Generate fake samples and get features
    fake_features = []
    batch_size = 32
    n_gen = 0
    while n_gen < n_samples:
        samples = diffusion.sample(model, (batch_size, 3, img_size, img_size), device)
        samples = samples.clamp(-1, 1) * 0.5 + 0.5  # [-1,1] -> [0,1]
        feat = inception(samples)
        fake_features.append(feat)
        n_gen += batch_size
    fake_features = torch.cat(fake_features)[:n_samples]
    
    return calculate_fid(real_features, fake_features)


# =============================================================================
# Config
# =============================================================================
CONFIGS = {
    'adamw': {'lr': 2e-4, 'wd': 0.01, 'betas': (0.9, 0.999)},
    'lion': {'lr': 2e-5, 'wd': 0.1, 'betas': (0.9, 0.99)},
    'rlo': {'lr': 2e-5, 'wd': 0.1, 'betas': (0.9, 0.99), 'belief_coef': 0.1},
    'smooth_lifted_rlo': {'lr': 2e-5, 'wd': 0.1, 'lambda_b': 0.1, 'eta': 0.3},
}


def create_optimizer(model, name, cfg):
    params = model.parameters()
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['wd'], betas=cfg.get('betas', (0.9, 0.999)))
    elif name == 'lion':
        return Lion(params, lr=cfg['lr'], weight_decay=cfg['wd'], betas=cfg.get('betas', (0.9, 0.99)))
    elif name == 'rlo':
        return RLO(params, lr=cfg['lr'], weight_decay=cfg['wd'], betas=cfg.get('betas', (0.9, 0.99)),
                   belief_coef=cfg.get('belief_coef', 0.1))
    elif name == 'smooth_lifted_rlo':
        return SmoothLiftedRLO(params, lr=cfg['lr'], weight_decay=cfg['wd'],
                              lambda_b=cfg.get('lambda_b', 0.1), eta=cfg.get('eta', 0.3))
    raise ValueError(f"Unknown: {name}")


# =============================================================================
# Training
# =============================================================================
def train_diffusion(opt_name, resolution, data_path, output_dir, epochs, rank, world_size, local_rank, logger):
    device = torch.device(f'cuda:{local_rank}')
    cfg = CONFIGS[opt_name]
    batch_size = 64 if resolution <= 64 else 32
    per_gpu = batch_size // world_size

    logger.info("=" * 70)
    logger.info(f"Diffusion {resolution}x{resolution} | Opt: {opt_name}")
    logger.info(f"LR: {cfg['lr']} | WD: {cfg['wd']} | Epochs: {epochs}")
    logger.info("=" * 70)

    # Data
    transform = T.Compose([T.Resize(resolution), T.CenterCrop(resolution), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    dataset = ImageFolder(data_path / 'train', transform)
    sampler = DistributedSampler(dataset, world_size, rank) if world_size > 1 else None
    train_loader = DataLoader(dataset, per_gpu, shuffle=(sampler is None), sampler=sampler,
                             num_workers=2, pin_memory=True, drop_last=True, persistent_workers=False)
    
    val_loader = DataLoader(dataset, per_gpu * 2, num_workers=2, pin_memory=True)

    # Model
    model = UNet(in_ch=3, base_ch=128 if resolution <= 64 else 96).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    diffusion = GaussianDiffusion(timesteps=1000).to(device)
    optimizer = create_optimizer(model, opt_name, cfg)
    scaler = torch.amp.GradScaler('cuda')

    history = {'train_loss': [], 'fid': []}
    best_fid = float('inf')

    for epoch in range(epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        loss_sum, n_batches = 0.0, 0
        t0 = time.time()

        for x, _ in train_loader:
            x = x.to(device)
            t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=device)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', torch.bfloat16):
                loss = diffusion.p_losses(model, x, t)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            loss_sum += loss.item()
            n_batches += 1

        epoch_time = time.time() - t0
        avg_loss = loss_sum / n_batches
        history['train_loss'].append(avg_loss)

        # Compute FID every 10 epochs
        fid = float('nan')
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            base_model = model.module if hasattr(model, 'module') else model
            try:
                fid = compute_fid(base_model, diffusion, val_loader, device, n_samples=500, img_size=resolution)
            except Exception as e:
                logger.warning(f"FID computation failed: {e}")
            history['fid'].append(fid)
            
            is_best = fid < best_fid
            best_fid = min(fid, best_fid) if not math.isnan(fid) else best_fid
            
            logger.info(f"E{epoch+1:3d}: loss={avg_loss:.4f}, FID={fid:.2f}{'*' if is_best else ''}, {epoch_time:.0f}s")
            
            if is_best and rank == 0 and not math.isnan(fid):
                torch.save({'model': base_model.state_dict(), 'fid': best_fid},
                          output_dir / f"diff_{resolution}_{opt_name}_best.pt")
        else:
            logger.info(f"E{epoch+1:3d}: loss={avg_loss:.4f}, {epoch_time:.0f}s")

        if (epoch + 1) % 20 == 0:
            gc.collect(); torch.cuda.empty_cache()

    # Save results
    if rank == 0:
        with open(output_dir / f"diff_{resolution}_{opt_name}_results.json", 'w') as f:
            json.dump({'optimizer': opt_name, 'resolution': resolution, 'best_fid': best_fid,
                      'history': history}, f, indent=2)
        logger.info(f"Done. Best FID: {best_fid:.2f}")

    return best_fid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=64, choices=[64, 128])
    parser.add_argument('--optimizer', default='all')
    parser.add_argument('--data', default='/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder')
    parser.add_argument('--output', default='/blue/wdixon/wang.yixuan/rlo_results/diffusion')
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
    logger = setup_logger(output_dir, rank, f"diff_{args.resolution}")

    opts = list(CONFIGS.keys()) if args.optimizer == 'all' else [args.optimizer]
    results = {}

    for opt in opts:
        try:
            gc.collect(); torch.cuda.empty_cache()
            results[opt] = train_diffusion(opt, args.resolution, Path(args.data), output_dir, args.epochs,
                                           rank, world_size, local_rank, logger)
        except Exception as e:
            logger.error(f"Error {opt}: {e}")
            import traceback; logger.error(traceback.format_exc())

    if rank == 0 and results:
        logger.info("=" * 70)
        logger.info(f"Diffusion {args.resolution}x{args.resolution} Results (FID, lower=better):")
        for opt, fid in sorted(results.items(), key=lambda x: x[1] if not math.isnan(x[1]) else float('inf')):
            logger.info(f"  {opt}: {fid:.2f}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
