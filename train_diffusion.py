#!/usr/bin/env python3
"""
Diffusion Model Experiment - Following Lion Paper Figure 6
64x64 ImageNet diffusion, records FID (lower is better)
"""
import os, gc, time, math, json, logging, argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import inception_v3
import warnings; warnings.filterwarnings('ignore')

try:
    from optimizers import create_optimizer
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from optimizers import create_optimizer


def setup_logger(out, rank, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(out / f"{name}_rank{rank}.log", mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
    logger.addHandler(fh)
    if rank == 0:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(ch)
    return logger


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    """Residual block with time embedding - FIXED for any channel count."""
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        # Use GroupNorm with appropriate num_groups
        def get_norm(ch):
            # Use min of 32 or ch, and ensure ch is divisible by num_groups
            for g in [32, 16, 8, 4, 2, 1]:
                if ch >= g and ch % g == 0:
                    return nn.GroupNorm(g, ch)
            return nn.GroupNorm(1, ch)
        
        self.norm1 = get_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = get_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(F.silu(t))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """Simple U-Net for diffusion - FIXED to handle 3-channel input properly."""
    def __init__(self, in_ch=3, base_ch=128, ch_mults=(1, 2, 4), time_dim=256):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # CRITICAL FIX: Initial conv to expand 3 channels to base_ch
        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        curr_ch = base_ch
        channels = [base_ch]
        for mult in ch_mults:
            out_c = base_ch * mult
            self.encoder.append(ResBlock(curr_ch, out_c, time_dim))
            self.downsample.append(nn.Conv2d(out_c, out_c, 3, 2, 1))
            channels.append(out_c)
            curr_ch = out_c
        
        # Middle
        self.mid = ResBlock(curr_ch, curr_ch, time_dim)
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        
        for mult in reversed(ch_mults):
            out_c = base_ch * mult
            self.upsample.append(nn.ConvTranspose2d(curr_ch, out_c, 4, 2, 1))
            skip_c = channels.pop()
            self.decoder.append(ResBlock(out_c + skip_c, out_c, time_dim))
            curr_ch = out_c
        
        # Output conv back to 3 channels
        self.out_conv = nn.Conv2d(curr_ch, in_ch, 1)
    
    def forward(self, x, t):
        t = self.time_mlp(t)
        
        # Initial conv: 3 -> base_ch
        x = self.init_conv(x)
        
        # Encoder
        skips = [x]
        for enc, down in zip(self.encoder, self.downsample):
            x = enc(x, t)
            skips.append(x)
            x = down(x)
        
        # Middle
        x = self.mid(x, t)
        
        # Decoder
        for up, dec in zip(self.upsample, self.decoder):
            x = up(x)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = dec(x, t)
        
        # Output: base_ch -> 3
        return self.out_conv(x)


class GaussianDiffusion:
    """DDPM-style diffusion process."""
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.timesteps = timesteps
        self.device = device
        
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
    
    def loss(self, model, x0):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred_noise = model(x_t, t.float())
        return F.mse_loss(pred_noise, noise)
    
    @torch.no_grad()
    def sample(self, model, shape, steps=50):
        device = self.device
        x = torch.randn(shape, device=device)
        timesteps = torch.linspace(self.timesteps - 1, 0, steps, device=device).long()
        
        for i in range(len(timesteps) - 1):
            t = timesteps[i].expand(shape[0])
            t_next = timesteps[i + 1].expand(shape[0])
            
            pred_noise = model(x, t.float())
            
            alpha = self.alphas_cumprod[t][:, None, None, None]
            alpha_next = self.alphas_cumprod[t_next][:, None, None, None]
            
            x0_pred = (x - torch.sqrt(1 - alpha) * pred_noise) / torch.sqrt(alpha)
            x0_pred = x0_pred.clamp(-1, 1)
            x = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * pred_noise
        
        return x


@torch.no_grad()
def compute_fid(real_features, fake_features):
    """Compute FID between two feature sets."""
    mu1, sigma1 = real_features.mean(0), torch.cov(real_features.T)
    mu2, sigma2 = fake_features.mean(0), torch.cov(fake_features.T)
    
    diff = mu1 - mu2
    
    # Matrix square root
    eigvals, eigvecs = torch.linalg.eigh(sigma1 @ sigma2)
    eigvals = eigvals.clamp(min=1e-8)
    sqrt_sigma = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T
    
    fid = diff @ diff + torch.trace(sigma1 + sigma2 - 2 * sqrt_sigma)
    return fid.item()


class InceptionFeatureExtractor:
    """Extract features for FID computation."""
    def __init__(self, device):
        self.model = inception_v3(weights='DEFAULT', transform_input=False)
        self.model.fc = nn.Identity()
        self.model = self.model.to(device).eval()
        self.device = device
    
    @torch.no_grad()
    def extract(self, images):
        # Resize to 299x299 and normalize
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        images = (images - 0.5) / 0.5  # Normalize to [-1, 1]
        return self.model(images)


def train_diffusion(optimizer_name, resolution, data_path, output_dir, epochs, 
                   rank, world_size, device, logger):
    """Train diffusion model."""
    
    configs = {
        'adamw': {'lr': 3e-4, 'wd': 0.01, 'betas': (0.9, 0.99)},
        'lion': {'lr': 3e-5, 'wd': 0.1, 'betas': (0.9, 0.99)},
        'rlo': {'lr': 3e-5, 'wd': 0.1},
        'rlo_lambda_a': {'lr': 3e-5, 'wd': 0.1},
        'smooth_lifted_rlo': {'lr': 3e-5, 'wd': 0.1},
    }
    
    cfg = configs.get(optimizer_name, configs['adamw'])
    
    if rank == 0:
        logger.info("=" * 70)
        logger.info(f"Diffusion {resolution}x{resolution} | {optimizer_name}")
        logger.info(f"LR={cfg['lr']}, WD={cfg['wd']}")
        logger.info("=" * 70)
    
    # Data
    transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    train_path = data_path / 'train' if (data_path / 'train').exists() else data_path
    dataset = ImageFolder(train_path, transform)
    
    if rank == 0:
        logger.info(f"Samples: {len(dataset)}")
    
    sampler = DistributedSampler(dataset, world_size, rank) if world_size > 1 else None
    loader = DataLoader(dataset, batch_size=32, shuffle=(sampler is None),
                       sampler=sampler, num_workers=8, pin_memory=True,
                       drop_last=True, persistent_workers=True)
    
    # Model
    base_ch = 128 if resolution <= 64 else 256
    model = UNet(base_ch=base_ch).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    raw_model = model.module if hasattr(model, 'module') else model
    
    if rank == 0:
        params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"Parameters: {params:.1f}M")
    
    # Optimizer
    optimizer = create_optimizer(optimizer_name, model.parameters(),
                                lr=cfg['lr'], weight_decay=cfg['wd'],
                                betas=cfg.get('betas'))
    
    # Diffusion
    diffusion = GaussianDiffusion(device=device)
    
    # FID setup
    if rank == 0:
        fid_extractor = InceptionFeatureExtractor(device)
        
        # Precompute real features
        real_features = []
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(loader):
                if batch_idx >= 50:  # ~1600 samples
                    break
                images = images.to(device)
                images = (images + 1) / 2  # [-1,1] -> [0,1]
                feats = fid_extractor.extract(images)
                real_features.append(feats)
        real_features = torch.cat(real_features, 0)
        logger.info(f"Real features: {real_features.shape[0]} samples")
    
    # Training
    best_fid = float('inf')
    results = {'optimizer': optimizer_name, 'fid_history': []}
    scaler = torch.amp.GradScaler()
    
    for epoch in range(1, epochs + 1):
        model.train()
        if sampler:
            sampler.set_epoch(epoch)
        
        total_loss = 0
        num_batches = 0
        
        for images, _ in loader:
            images = images.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                loss = diffusion.loss(model, images)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Compute FID every 20 epochs
        if rank == 0 and epoch % 20 == 0:
            model.eval()
            raw_model.eval()
            
            fake_features = []
            num_samples = min(500, real_features.shape[0])
            batch_size = 16
            
            for i in range(0, num_samples, batch_size):
                n = min(batch_size, num_samples - i)
                samples = diffusion.sample(raw_model, (n, 3, resolution, resolution), steps=50)
                samples = (samples + 1) / 2  # [-1,1] -> [0,1]
                samples = samples.clamp(0, 1)
                feats = fid_extractor.extract(samples)
                fake_features.append(feats)
            
            fake_features = torch.cat(fake_features, 0)
            fid = compute_fid(real_features[:num_samples], fake_features)
            
            results['fid_history'].append({'epoch': epoch, 'fid': fid, 'loss': avg_loss})
            
            if fid < best_fid:
                best_fid = fid
                logger.info(f"E{epoch:3d}: loss={avg_loss:.4f} FID={fid:.2f}*")
            else:
                logger.info(f"E{epoch:3d}: loss={avg_loss:.4f} FID={fid:.2f}")
            
            model.train()
        elif rank == 0 and epoch % 10 == 0:
            logger.info(f"E{epoch:3d}: loss={avg_loss:.4f}")
    
    if rank == 0:
        results['best_fid'] = best_fid
        results['final_loss'] = avg_loss
        logger.info(f"Final: Best FID = {best_fid:.2f}")
    
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--optimizer', type=str, default='all')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    
    print(f"Starting diffusion training with args: {args}")
    
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if world_size > 1:
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
    
    device = torch.device(f'cuda:{local_rank}')
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(output_dir, rank, 'diffusion')
    
    if args.optimizer == 'all':
        optimizers = ['adamw', 'lion', 'rlo', 'rlo_lambda_a', 'smooth_lifted_rlo']
    else:
        optimizers = [args.optimizer]
    
    results = {}
    
    for opt in optimizers:
        try:
            results[opt] = train_diffusion(opt, args.resolution, Path(args.data),
                                          output_dir, args.epochs, rank, world_size,
                                          device, logger)
            
            if rank == 0:
                with open(output_dir / f'{opt}_results.json', 'w') as f:
                    json.dump(results[opt], f, indent=2)
        
        except Exception as e:
            if rank == 0:
                logger.error(f"Error with {opt}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    if rank == 0:
        logger.info("=" * 70)
        logger.info("Diffusion Results (FID, lower is better):")
        for opt in sorted(results.keys(), key=lambda x: results[x].get('best_fid', float('inf'))):
            fid = results[opt].get('best_fid', float('inf'))
            logger.info(f"  {opt}: {fid:.2f}")
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
