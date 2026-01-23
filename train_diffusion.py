#!/usr/bin/env python3
"""
Diffusion Model Training on ImageNet 64x64
DDPM with U-Net, reports FID (lower is better, target: 10-50)
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
    for g in [32, 16, 8, 4, 2, 1]:
        if channels >= g and channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.norm1 = get_group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = get_group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = get_group_norm(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return self.proj(out) + x


class UNet(nn.Module):
    """
    Simple but correct U-Net for diffusion.
    Resolution progression: 64 -> 32 -> 16 -> 8 (bottleneck) -> 16 -> 32 -> 64
    """
    def __init__(self, in_ch=3, base_ch=128, ch_mults=(1, 2, 2, 4), time_dim=256, num_res=2):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Initial conv
        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        ch = base_ch
        for i, mult in enumerate(ch_mults):
            out_ch = base_ch * mult
            blocks = nn.ModuleList()
            for _ in range(num_res):
                blocks.append(ResBlock(ch, out_ch, time_dim))
                ch = out_ch
            if i == len(ch_mults) - 1:  # Attention at bottleneck
                blocks.append(Attention(ch))
            self.down_blocks.append(blocks)
            
            if i < len(ch_mults) - 1:
                self.down_samples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            else:
                self.down_samples.append(None)
        
        # Middle
        self.mid_block1 = ResBlock(ch, ch, time_dim)
        self.mid_attn = Attention(ch)
        self.mid_block2 = ResBlock(ch, ch, time_dim)
        
        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(ch_mults))):
            out_ch = base_ch * mult
            blocks = nn.ModuleList()
            for j in range(num_res + 1):
                # First block receives skip connection (doubles channels)
                in_channels = ch + out_ch if j == 0 else ch
                blocks.append(ResBlock(in_channels, out_ch, time_dim))
                ch = out_ch
            if i == len(ch_mults) - 1:  # Attention at bottleneck level
                blocks.append(Attention(ch))
            self.up_blocks.append(blocks)
            
            if i > 0:
                self.up_samples.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(ch, ch, 3, padding=1)
                ))
            else:
                self.up_samples.append(None)
        
        # Output
        self.out_norm = get_group_norm(ch)
        self.out_conv = nn.Conv2d(ch, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
    
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        # Initial conv
        x = self.init_conv(x)
        
        # Encoder - save skip connections
        skips = []
        for blocks, down in zip(self.down_blocks, self.down_samples):
            for block in blocks:
                if isinstance(block, ResBlock):
                    x = block(x, t_emb)
                else:
                    x = block(x)
            skips.append(x)
            if down is not None:
                x = down(x)
        
        # Middle
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)
        
        # Decoder - use skip connections in reverse
        for blocks, up in zip(self.up_blocks, self.up_samples):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            for i, block in enumerate(blocks):
                if isinstance(block, ResBlock):
                    x = block(x, t_emb) if i == 0 else block(x, t_emb)
                else:
                    x = block(x)
            if up is not None:
                x = up(x)
        
        return self.out_conv(F.silu(self.out_norm(x)))


class GaussianDiffusion:
    """DDPM diffusion process."""
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
    
    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)
    
    def to(self, device):
        for attr in ['betas', 'alphas_cumprod', 'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self
    
    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    
    def p_losses(self, model, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        x_noisy = self.q_sample(x_0, t, noise)
        pred_noise = model(x_noisy, t.float())
        return F.mse_loss(pred_noise, noise)
    
    @torch.no_grad()
    def p_sample(self, model, x, t, t_idx):
        pred_noise = model(x, t)
        alpha = self.alphas_cumprod[t_idx]
        alpha_prev = self.alphas_cumprod[t_idx - 1] if t_idx > 0 else torch.tensor(1.0, device=x.device)
        beta = self.betas[t_idx]
        
        pred_x0 = (x - torch.sqrt(1 - alpha) * pred_noise) / torch.sqrt(alpha)
        pred_x0 = pred_x0.clamp(-1, 1)
        
        if t_idx > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta * (1 - alpha_prev) / (1 - alpha))
            x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev - sigma**2) * pred_noise + sigma * noise
        else:
            x = pred_x0
        return x
    
    @torch.no_grad()
    def sample(self, model, shape, device):
        x = torch.randn(shape, device=device)
        for t_idx in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), t_idx, device=device, dtype=torch.float)
            x = self.p_sample(model, x, t, t_idx)
        return x


def compute_fid(real_features, fake_features):
    """Compute FID between two sets of features."""
    from scipy import linalg
    
    mu1, sigma1 = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(0), np.cov(fake_features, rowvar=False)
    
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(max(fid, 0))


@torch.no_grad()
def get_inception_features(images, inception_model, device):
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    images = 2 * images - 1
    features = inception_model(images)
    return features.cpu().numpy()


@torch.no_grad()
def calculate_fid(model, diffusion, real_loader, device, num_samples=5000, batch_size=50):
    model.eval()
    
    try:
        from torchvision.models import inception_v3, Inception_V3_Weights
        inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        inception.fc = nn.Identity()
        inception = inception.to(device).eval()
    except Exception as e:
        print(f"Could not load Inception: {e}")
        return -1.0
    
    real_features = []
    for images, _ in real_loader:
        if len(real_features) * images.size(0) >= num_samples:
            break
        images = images.to(device)
        feats = get_inception_features(images, inception, device)
        real_features.append(feats)
    real_features = np.concatenate(real_features, axis=0)[:num_samples]
    
    fake_features = []
    while len(fake_features) * batch_size < num_samples:
        fake_images = diffusion.sample(model, (batch_size, 3, 64, 64), device)
        fake_images = fake_images.clamp(-1, 1) * 0.5 + 0.5
        feats = get_inception_features(fake_images, inception, device)
        fake_features.append(feats)
    fake_features = np.concatenate(fake_features, axis=0)[:num_samples]
    
    fid = compute_fid(real_features, fake_features)
    
    del inception
    torch.cuda.empty_cache()
    
    return fid


def train_diffusion(opt_name, resolution, data_path, output_dir, epochs, rank, world_size, device, logger):
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
    
    transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    train_path = data_path / 'train'
    if not train_path.exists():
        train_path = data_path
    
    train_dataset = ImageFolder(train_path, transform)
    
    if rank == 0:
        logger.info(f"Samples: {len(train_dataset)}")
    
    batch_size = 32
    
    train_sampler = DistributedSampler(train_dataset, world_size, rank) if world_size > 1 else None
    train_loader = DataLoader(train_dataset, batch_size, shuffle=(train_sampler is None),
                             sampler=train_sampler, num_workers=8, pin_memory=True,
                             drop_last=True, persistent_workers=True)
    
    model = UNet(in_ch=3, base_ch=128, ch_mults=(1, 2, 2, 4), time_dim=256, num_res=2).to(device)
    diffusion = GaussianDiffusion(timesteps=1000).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"Parameters: {params:.1f}M")
    
    optimizer = create_optimizer(opt_name, model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'], betas=cfg['betas'])
    
    warmup_epochs = 5
    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scaler = torch.amp.GradScaler()
    best_fid = float('inf')
    results = {'optimizer': opt_name, 'history': []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        total_loss = 0
        num_batches = 0
        
        for images, _ in train_loader:
            images = images.to(device)
            t = torch.randint(0, diffusion.timesteps, (images.size(0),), device=device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                loss = diffusion.p_losses(model, images, t)
            
            # CRITICAL: 在分布式训练中，必须同步 NaN 检查
            # 否则不同 rank 跳过不同数量的 batch 会导致 NCCL 死锁
            has_nan = torch.isnan(loss) or torch.isinf(loss)
            if world_size > 1:
                has_nan_tensor = torch.tensor([1.0 if has_nan else 0.0], device=device)
                dist.all_reduce(has_nan_tensor, op=dist.ReduceOp.MAX)
                has_nan = has_nan_tensor.item() > 0
            
            if has_nan:
                continue  # 所有 rank 一起跳过
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        if rank == 0 and epoch % 10 == 0:
            logger.info(f"E{epoch:3d}: loss={avg_loss:.4f}")
        
        if rank == 0 and (epoch % 25 == 0 or epoch == epochs):
            fid = calculate_fid(model.module if world_size > 1 else model, diffusion, train_loader, device, num_samples=5000)
            results['history'].append({'epoch': epoch, 'loss': avg_loss, 'fid': fid})
            
            marker = '*' if fid < best_fid else ''
            best_fid = min(best_fid, fid) if fid > 0 else best_fid
            logger.info(f"E{epoch:3d}: loss={avg_loss:.4f} FID={fid:.2f}{marker}")
    
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
