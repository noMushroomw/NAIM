#!/usr/bin/env python3
"""
Diffusion Model Training on ImageNet 64x64
DDPM with U-Net, reports FID (lower is better, target: 10-50)

Key fixes:
- init_conv to project 3 channels to base_ch before GroupNorm
- Dynamic GroupNorm that handles any channel count
"""
import os, gc, math, json, logging, argparse
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
import warnings; warnings.filterwarnings('ignore')

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


def get_group_norm(channels, num_groups=32):
    """Create GroupNorm with proper num_groups for any channel count."""
    # Find largest divisor <= num_groups
    for g in [32, 16, 8, 4, 2, 1]:
        if channels >= g and channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


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
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.norm1 = get_group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2 = get_group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = get_group_norm(dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv.unbind(1)
        q = q.transpose(-1, -2)
        k = k.transpose(-1, -2)
        v = v.transpose(-1, -2)
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return self.proj(out) + x


class UNet(nn.Module):
    """U-Net for diffusion with proper channel handling."""
    def __init__(self, in_ch=3, base_ch=128, ch_mults=(1, 2, 2, 4), time_dim=256, num_res=2, attn_resolutions=(16,)):
        super().__init__()
        self.time_dim = time_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Initial projection: 3 -> base_ch (FIXES GroupNorm issue!)
        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        ch = base_ch
        chs = [ch]
        
        for i, mult in enumerate(ch_mults):
            out_ch = base_ch * mult
            for _ in range(num_res):
                self.encoder.append(ResBlock(ch, out_ch, time_dim))
                ch = out_ch
                chs.append(ch)
                
                # Attention at specified resolutions
                if i in [len(ch_mults) - 1]:  # Only at lowest resolution
                    self.encoder.append(Attention(ch))
                    chs.append(ch)
            
            if i < len(ch_mults) - 1:
                self.downsample.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                chs.append(ch)
        
        # Middle
        self.mid1 = ResBlock(ch, ch, time_dim)
        self.mid_attn = Attention(ch)
        self.mid2 = ResBlock(ch, ch, time_dim)
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(ch_mults))):
            out_ch = base_ch * mult
            for j in range(num_res + 1):
                skip_ch = chs.pop()
                self.decoder.append(ResBlock(ch + skip_ch, out_ch, time_dim))
                ch = out_ch
                
                if i in [len(ch_mults) - 1] and j < num_res:
                    self.decoder.append(Attention(ch))
                    chs.pop()  # Pop the attention skip
            
            if i > 0:
                self.upsample.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
        
        # Output
        self.out_norm = get_group_norm(ch)
        self.out_conv = nn.Conv2d(ch, in_ch, 3, padding=1)
        
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
    
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        # Initial conv
        x = self.init_conv(x)
        
        # Encoder
        skips = [x]
        down_idx = 0
        
        for layer in self.encoder:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, Attention):
                x = layer(x)
            skips.append(x)
        
        for down in self.downsample:
            x = down(x)
            skips.append(x)
        
        # Middle
        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)
        
        # Decoder
        up_idx = 0
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, ResBlock):
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = layer(x, t_emb)
            elif isinstance(layer, Attention):
                x = layer(x)
                skips.pop()
        
        for up in self.upsample:
            x = up(x)
        
        return self.out_conv(F.silu(self.out_norm(x)))


class GaussianDiffusion:
    """DDPM diffusion process."""
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer = lambda name, val: setattr(self, name, val)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance', betas * (1.0 - torch.cat([torch.tensor([0.0]), alphas_cumprod[:-1]])) / (1.0 - alphas_cumprod))
    
    def to(self, device):
        for name in ['betas', 'alphas_cumprod', 'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'sqrt_recip_alphas', 'posterior_variance']:
            setattr(self, name, getattr(self, name).to(device))
        return self
    
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x0 + sqrt_one_minus * noise
    
    def p_losses(self, model, x0, t):
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        pred_noise = model(x_noisy, t.float())
        return F.mse_loss(pred_noise, noise)
    
    @torch.no_grad()
    def p_sample(self, model, x, t):
        betas_t = self.betas[t][:, None, None, None]
        sqrt_recip = self.sqrt_recip_alphas[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        pred_noise = model(x, t.float())
        mean = sqrt_recip * (x - betas_t * pred_noise / sqrt_one_minus)
        
        if t[0] > 0:
            noise = torch.randn_like(x)
            var = self.posterior_variance[t][:, None, None, None]
            return mean + torch.sqrt(var) * noise
        return mean
    
    @torch.no_grad()
    def sample(self, model, shape, device):
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch)
        return x


def compute_fid(real_features, fake_features):
    """Compute FID between two sets of features."""
    mu1, sigma1 = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(0), np.cov(fake_features, rowvar=False)
    
    diff = mu1 - mu2
    
    # Compute sqrt of product of covariances
    covmean, _ = np.linalg.eigh(sigma1 @ sigma2)
    covmean = np.sqrt(np.maximum(covmean, 0)).sum()
    
    fid = diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * covmean
    return float(fid)


@torch.no_grad()
def get_inception_features(images, inception_model, device):
    """Extract Inception features for FID calculation."""
    # Resize to 299x299 for Inception
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    
    # Normalize to [-1, 1]
    images = 2 * images - 1
    
    features = inception_model(images)
    return features.cpu().numpy()


@torch.no_grad()
def calculate_fid(model, diffusion, real_loader, device, num_samples=5000, batch_size=50):
    """Calculate FID score."""
    model.eval()
    
    # Try to use torchvision inception
    try:
        from torchvision.models import inception_v3, Inception_V3_Weights
        inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        inception.fc = nn.Identity()  # Remove classification head
        inception = inception.to(device).eval()
    except Exception as e:
        print(f"Could not load Inception: {e}")
        return -1.0
    
    # Get real features
    real_features = []
    for images, _ in real_loader:
        if len(real_features) * batch_size >= num_samples:
            break
        images = images.to(device)
        feats = get_inception_features(images, inception, device)
        real_features.append(feats)
    real_features = np.concatenate(real_features, axis=0)[:num_samples]
    
    # Generate fake images and get features
    fake_features = []
    while len(fake_features) * batch_size < num_samples:
        fake_images = diffusion.sample(model, (batch_size, 3, 64, 64), device)
        fake_images = fake_images.clamp(-1, 1) * 0.5 + 0.5  # [-1,1] -> [0,1]
        feats = get_inception_features(fake_images, inception, device)
        fake_features.append(feats)
    fake_features = np.concatenate(fake_features, axis=0)[:num_samples]
    
    # Compute FID
    fid = compute_fid(real_features, fake_features)
    
    del inception
    torch.cuda.empty_cache()
    
    return fid


def train_diffusion(opt_name, resolution, data_path, output_dir, epochs, rank, world_size, device, logger):
    # Tuned configs
    configs = {
        'adamw': {'lr': 2e-4, 'wd': 0.01, 'betas': (0.9, 0.999)},
        'lion': {'lr': 2e-5, 'wd': 0.1, 'betas': (0.9, 0.99)},
        'rlo': {'lr': 2.5e-5, 'wd': 0.1, 'betas': (0.9, 0.99)},
        'rlo_lambda_a': {'lr': 2.5e-5, 'wd': 0.1, 'betas': (0.9, 0.99)},
        'smooth_lifted_rlo': {'lr': 2.5e-5, 'wd': 0.1, 'betas': (0.9, 0.99)},
    }
    
    cfg = configs.get(opt_name, configs['adamw'])
    
    if rank == 0:
        logger.info("=" * 70)
        logger.info(f"Diffusion {resolution}x{resolution}: {opt_name}")
        logger.info(f"LR={cfg['lr']}, WD={cfg['wd']}")
        logger.info("=" * 70)
    
    # Data
    transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
    ])
    
    train_path = data_path / 'train'
    if not train_path.exists():
        train_path = data_path
    
    train_dataset = ImageFolder(train_path, transform)
    
    batch_size = 32  # Per GPU
    
    train_sampler = DistributedSampler(train_dataset, world_size, rank) if world_size > 1 else None
    train_loader = DataLoader(train_dataset, batch_size, shuffle=(train_sampler is None),
                             sampler=train_sampler, num_workers=8, pin_memory=True,
                             drop_last=True, persistent_workers=True)
    
    if rank == 0:
        logger.info(f"Samples: {len(train_dataset)}")
    
    # Model
    base_ch = 128
    model = UNet(in_ch=3, base_ch=base_ch, ch_mults=(1, 2, 2, 4), time_dim=256).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"Parameters: {params:.1f}M")
    
    # Diffusion
    diffusion = GaussianDiffusion(timesteps=1000).to(device)
    
    # Optimizer
    optimizer = create_optimizer(opt_name, model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'], betas=cfg['betas'])
    
    # LR schedule
    warmup_epochs = 5
    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0  # Constant LR after warmup
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training
    scaler = torch.amp.GradScaler()
    best_fid = float('inf')
    results = {'optimizer': opt_name, 'history': []}
    global_step = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        total_loss = 0
        num_batches = 0
        
        for images, _ in train_loader:
            images = images.to(device)
            
            t = torch.randint(0, diffusion.timesteps, (images.shape[0],), device=device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                loss = diffusion.p_losses(model, images, t)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            global_step += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Calculate FID every 20 epochs
        if rank == 0 and (epoch % 20 == 0 or epoch == epochs):
            fid = calculate_fid(model.module if world_size > 1 else model, diffusion, train_loader, device, num_samples=2000)
            results['history'].append({'epoch': epoch, 'loss': avg_loss, 'fid': fid})
            
            marker = '*' if fid < best_fid else ''
            if fid > 0:
                best_fid = min(best_fid, fid)
            logger.info(f"E{epoch:3d}: loss={avg_loss:.4f} FID={fid:.2f}{marker}")
            model.train()
        elif rank == 0 and epoch % 10 == 0:
            logger.info(f"E{epoch:3d}: loss={avg_loss:.4f}")
    
    if rank == 0:
        results['best_fid'] = best_fid
        logger.info(f"Final: Best FID = {best_fid:.2f}")
    
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default='all')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--resolution', type=int, default=64)
    args = parser.parse_args()
    
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
            results[opt] = train_diffusion(opt, args.resolution, Path(args.data), output_dir, args.epochs, rank, world_size, device, logger)
            if rank == 0:
                with open(output_dir / f'{opt}_results.json', 'w') as f:
                    json.dump(results[opt], f, indent=2)
        except Exception as e:
            if rank == 0:
                logger.error(f"Error with {opt}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        if world_size > 1:
            dist.barrier()
    
    if rank == 0:
        logger.info("\n" + "=" * 70)
        logger.info("DIFFUSION RESULTS (FID ↓)")
        logger.info("=" * 70)
        for opt in sorted(results.keys(), key=lambda x: results[x].get('best_fid', float('inf'))):
            fid = results[opt].get('best_fid', float('inf'))
            logger.info(f"  {opt}: {fid:.2f}")
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
