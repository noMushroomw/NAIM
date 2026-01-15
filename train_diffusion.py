#!/usr/bin/env python3
"""
Diffusion Model Experiment - Following Lion Paper Figure 6
Imagen-style text-to-image, 64x64 resolution, records FID

Lion Paper configs:
- AdamW: lr=3e-4, wd=0.01, β=(0.9, 0.999)
- Lion: lr=3e-5 (0.1x), wd=0.1 (10x), β=(0.9, 0.99)
"""
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
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import warnings; warnings.filterwarnings('ignore')

from optimizers import Lion, RLO, RLO_LambdaA, SmoothLiftedRLO, create_optimizer


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
    """Sinusoidal time embeddings."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    """Residual block with time conditioning."""
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
    """Simple U-Net for diffusion."""
    def __init__(self, in_ch=3, base_ch=128, ch_mults=(1, 2, 4), time_dim=256):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        in_c = in_ch
        channels = [base_ch]
        for mult in ch_mults:
            out_c = base_ch * mult
            self.encoder.append(ResBlock(in_c, out_c, time_dim))
            self.downsample.append(nn.Conv2d(out_c, out_c, 3, 2, 1))
            channels.append(out_c)
            in_c = out_c
        
        # Middle
        self.mid = ResBlock(in_c, in_c, time_dim)
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        
        for mult in reversed(ch_mults):
            out_c = base_ch * mult
            self.upsample.append(nn.ConvTranspose2d(in_c, out_c, 4, 2, 1))
            skip_c = channels.pop()
            self.decoder.append(ResBlock(out_c + skip_c, out_c, time_dim))
            in_c = out_c
        
        self.out = nn.Conv2d(in_c, in_ch, 1)
    
    def forward(self, x, t):
        t = self.time_mlp(t)
        
        # Encoder
        skips = []
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
        
        return self.out(x)


class GaussianDiffusion:
    """DDPM-style diffusion."""
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.timesteps = timesteps
        
        # Linear schedule
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer = lambda name, val: setattr(self, name, val)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
    
    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
    
    def p_losses(self, model, x0, t):
        """Training loss: predict noise."""
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred_noise = model(x_t, t.float())
        return F.mse_loss(pred_noise, noise)
    
    @torch.no_grad()
    def p_sample(self, model, x, t_idx):
        """Reverse diffusion step."""
        t = torch.full((x.shape[0],), t_idx, device=x.device, dtype=torch.long)
        
        pred_noise = model(x, t.float())
        
        alpha = self.alphas[t_idx]
        alpha_cumprod = self.alphas_cumprod[t_idx]
        beta = self.betas[t_idx]
        
        # Mean
        mean = (1 / alpha.sqrt()) * (x - beta / (1 - alpha_cumprod).sqrt() * pred_noise)
        
        if t_idx > 0:
            noise = torch.randn_like(x)
            return mean + beta.sqrt() * noise
        return mean
    
    @torch.no_grad()
    def sample(self, model, shape, device):
        """Generate samples."""
        x = torch.randn(shape, device=device)
        
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(model, x, t)
        
        return x


class InceptionV3Features(nn.Module):
    """Extract Inception features for FID computation."""
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        except:
            from torchvision.models import inception_v3
            inception = inception_v3(pretrained=True)
        
        # Use layers up to pool3
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(3, 2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(3, 2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
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


def compute_fid(real_features, fake_features):
    """Compute FID between two sets of features."""
    mu1, mu2 = real_features.mean(0), fake_features.mean(0)
    sigma1, sigma2 = torch.cov(real_features.T), torch.cov(fake_features.T)
    
    diff = mu1 - mu2
    
    # Matrix square root approximation using eigenvalues
    product = sigma1 @ sigma2
    eigenvalues = torch.linalg.eigvalsh(product)
    sqrt_product = eigenvalues.clamp(min=0).sqrt().sum()
    
    fid = diff.dot(diff) + sigma1.trace() + sigma2.trace() - 2 * sqrt_product
    return fid.item()


@torch.no_grad()
def evaluate_fid(model, diffusion, real_loader, device, num_samples=1000, img_size=64):
    """Compute FID score."""
    model.eval()
    
    try:
        inception = InceptionV3Features().to(device)
    except Exception as e:
        print(f"Failed to load Inception: {e}")
        return float('nan')
    
    # Get real features
    real_features = []
    for images, _ in real_loader:
        if len(real_features) * images.size(0) >= num_samples:
            break
        images = images.to(device)
        # Denormalize from [-1,1] to [0,1]
        images = images * 0.5 + 0.5
        feat = inception(images)
        real_features.append(feat)
    
    real_features = torch.cat(real_features)[:num_samples]
    
    # Generate fake samples and get features
    fake_features = []
    batch_size = 32
    
    while len(fake_features) * batch_size < num_samples:
        samples = diffusion.sample(model, (batch_size, 3, img_size, img_size), device)
        samples = samples.clamp(-1, 1) * 0.5 + 0.5
        feat = inception(samples)
        fake_features.append(feat)
    
    fake_features = torch.cat(fake_features)[:num_samples]
    
    return compute_fid(real_features, fake_features)


# ============= Lion Paper Configs =============
CONFIGS = {
    'adamw': {'lr': 3e-4, 'wd': 0.01, 'betas': (0.9, 0.999)},
    'lion': {'lr': 3e-5, 'wd': 0.1, 'betas': (0.9, 0.99)},  # 0.1x lr, 10x wd
    'rlo': {'lr': 3e-5, 'wd': 0.1, 'betas': (0.9, 0.99), 'belief_coef': 0.1},
    'rlo_lambda_a': {'lr': 3e-5, 'wd': 0.1, 'beta1': 0.9, 'beta2': 0.99, 'lambda_b': 0.1},
    'smooth_lifted_rlo': {'lr': 3e-5, 'wd': 0.1, 'beta1': 0.9, 'beta2': 0.99, 
                          'lambda_b': 0.1, 'eta': 0.3},
}


def train_diffusion(opt_name, resolution, data_path, output_path, epochs,
                   rank, world_size, local_rank, logger):
    device = torch.device(f'cuda:{local_rank}')
    cfg = CONFIGS[opt_name]
    
    # Batch size
    if resolution <= 64:
        batch_size = 64
    else:
        batch_size = 32
    
    per_gpu_batch = batch_size // world_size
    
    logger.info("=" * 70)
    logger.info(f"Diffusion {resolution}x{resolution} | {opt_name}")
    logger.info(f"LR={cfg['lr']}, WD={cfg['wd']}")
    logger.info("=" * 70)
    
    # Data
    transform = T.Compose([
        T.Resize(resolution),
        T.CenterCrop(resolution),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3)  # [-1, 1]
    ])
    
    dataset = ImageFolder(data_path / 'train', transform)
    sampler = DistributedSampler(dataset, world_size, rank) if world_size > 1 else None
    train_loader = DataLoader(dataset, per_gpu_batch, 
                             shuffle=(sampler is None),
                             sampler=sampler,
                             num_workers=12, pin_memory=True, drop_last=True,
                             persistent_workers=True, prefetch_factor=4)
    
    # For FID evaluation
    eval_loader = DataLoader(dataset, per_gpu_batch * 2, 
                            shuffle=False, num_workers=8, pin_memory=True,
                            prefetch_factor=4)
    
    logger.info(f"Samples: {len(dataset)}")
    
    # Model
    base_ch = 128 if resolution <= 64 else 96
    model = UNet(base_ch=base_ch).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Parameters: {num_params:.1f}M")
    
    # Diffusion
    diffusion = GaussianDiffusion(timesteps=1000, device=device)
    
    # Optimizer
    base_model = model.module if hasattr(model, 'module') else model
    optimizer = create_optimizer(base_model, opt_name, cfg)
    
    scaler = torch.amp.GradScaler('cuda')
    
    # Training
    history = {'loss': [], 'fid': [], 'epoch': []}
    best_fid = float('inf')
    
    for epoch in range(epochs):
        if sampler:
            sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        t0 = time.time()
        
        for images, _ in train_loader:
            images = images.to(device)
            t = torch.randint(0, diffusion.timesteps, (images.size(0),), device=device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast('cuda', torch.bfloat16):
                loss = diffusion.p_losses(model, images, t)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # Evaluate FID every 20 epochs
        fid = float('nan')
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            base_model = model.module if hasattr(model, 'module') else model
            fid = evaluate_fid(base_model, diffusion, eval_loader, device, 
                              num_samples=500, img_size=resolution)
            
            is_best = fid < best_fid if not math.isnan(fid) else False
            if not math.isnan(fid):
                best_fid = min(fid, best_fid)
            
            history['loss'].append(avg_loss)
            history['fid'].append(fid)
            history['epoch'].append(epoch + 1)
            
            logger.info(f"E{epoch+1:3d}: loss={avg_loss:.4f} FID={fid:.2f}{'*' if is_best else ''} "
                       f"time={time.time()-t0:.0f}s")
            
            if is_best and rank == 0:
                torch.save({
                    'model': base_model.state_dict(),
                    'fid': best_fid,
                    'epoch': epoch
                }, output_path / f"diffusion_{resolution}_{opt_name}_best.pt")
        else:
            logger.info(f"E{epoch+1:3d}: loss={avg_loss:.4f} time={time.time()-t0:.0f}s")
    
    # Save results
    if rank == 0:
        results = {
            'optimizer': opt_name,
            'config': cfg,
            'resolution': resolution,
            'best_fid': best_fid,
            'history': history
        }
        with open(output_path / f"diffusion_{resolution}_{opt_name}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Final: Best FID = {best_fid:.2f}")
    
    return best_fid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=64, choices=[64, 128])
    parser.add_argument('--optimizer', default='all')
    parser.add_argument('--data', default='/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder')
    parser.add_argument('--output', default='/blue/wdixon/wang.yixuan/rlo_experiments/diffusion')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    
    # Setup
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Distributed
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl')
    else:
        rank, world_size, local_rank = 0, 1, 0
    
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)
    
    logger = setup_logger(output_path, rank, f"diffusion_{args.resolution}")
    
    # Run experiments
    optimizers = list(CONFIGS.keys()) if args.optimizer == 'all' else [args.optimizer]
    results = {}
    
    for opt in optimizers:
        try:
            gc.collect()
            torch.cuda.empty_cache()
            results[opt] = train_diffusion(opt, args.resolution, Path(args.data), 
                                          output_path, args.epochs,
                                          rank, world_size, local_rank, logger)
        except Exception as e:
            logger.error(f"Error with {opt}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Summary
    if rank == 0 and results:
        logger.info("=" * 70)
        logger.info(f"Diffusion {args.resolution}x{args.resolution} Results (FID, lower is better):")
        for opt, fid in sorted(results.items(), key=lambda x: x[1] if not math.isnan(x[1]) else float('inf')):
            logger.info(f"  {opt}: {fid:.2f}")
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
