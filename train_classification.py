#!/usr/bin/env python3
"""
Image Classification Experiment - Following Lion Paper Table 2
ViT-S/16 or ViT-B/16 on ImageNet, 90 epochs
Records Top-1 Accuracy
"""
import os, gc, time, math, json, logging, argparse
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

try:
    from optimizers import create_optimizer
except ImportError:
    import sys
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


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)
    
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """Multi-head self-attention."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MLP(nn.Module):
    """MLP block."""
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
    
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Transformer block."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer for classification."""
    def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                 embed_dim=384, depth=12, num_heads=6, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        
        x = self.norm(x)
        return self.head(x[:, 0])


def create_vit(model_name='vit_s16', num_classes=1000, use_checkpoint=True):
    """Create ViT model by name."""
    configs = {
        'vit_s16': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
        'vit_b16': {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
    }
    cfg = configs.get(model_name, configs['vit_s16'])
    return ViT(num_classes=num_classes, use_checkpoint=use_checkpoint, **cfg)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return 100.0 * correct / total


def train_classification(optimizer_name, model_name, data_path, output_dir, epochs, 
                        rank, world_size, device, logger):
    """Train classification model with specified optimizer."""
    
    # Hyperparameters from Lion paper Table 2
    configs = {
        'adamw': {'lr': 3e-3, 'wd': 0.1, 'betas': (0.9, 0.99)},
        'lion': {'lr': 3e-4, 'wd': 1.0, 'betas': (0.9, 0.99)},  # 0.1x lr, 10x wd
        'rlo': {'lr': 3e-4, 'wd': 1.0},
        'rlo_lambda_a': {'lr': 3e-4, 'wd': 1.0},
        'smooth_lifted_rlo': {'lr': 3e-4, 'wd': 1.0},
    }
    
    cfg = configs.get(optimizer_name, configs['adamw'])
    
    if rank == 0:
        logger.info("=" * 70)
        logger.info(f"Classification {model_name} | {optimizer_name}")
        logger.info(f"LR={cfg['lr']}, WD={cfg['wd']}")
        logger.info("=" * 70)
    
    # Data augmentation (following Lion paper)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    train_path = data_path / 'train'
    val_path = data_path / 'val'
    
    if not train_path.exists():
        train_path = data_path
    if not val_path.exists():
        val_path = data_path
    
    train_dataset = ImageFolder(train_path, train_transform)
    val_dataset = ImageFolder(val_path, val_transform)
    
    # Batch size: 4096 total, micro-batch 64 per GPU
    total_batch = 4096
    micro_batch = 64
    accum_steps = total_batch // (micro_batch * world_size)
    
    train_sampler = DistributedSampler(train_dataset, world_size, rank) if world_size > 1 else None
    train_loader = DataLoader(train_dataset, micro_batch,
                             shuffle=(train_sampler is None),
                             sampler=train_sampler,
                             num_workers=12, pin_memory=True, drop_last=True,
                             persistent_workers=True, prefetch_factor=4)
    
    val_loader = DataLoader(val_dataset, micro_batch * 2,
                           shuffle=False, num_workers=8, pin_memory=True,
                           prefetch_factor=4)
    
    if rank == 0:
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Model
    model = create_vit(model_name, num_classes=1000, use_checkpoint=True).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"Parameters: {params:.1f}M")
    
    # Optimizer
    optimizer = create_optimizer(optimizer_name, model.parameters(),
                                lr=cfg['lr'], weight_decay=cfg['wd'],
                                betas=cfg.get('betas'))
    
    # LR scheduler: cosine with warmup
    warmup_epochs = 5
    total_steps = epochs * len(train_loader) // accum_steps
    warmup_steps = warmup_epochs * len(train_loader) // accum_steps
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training
    scaler = torch.amp.GradScaler()
    best_acc = 0
    results = {'optimizer': optimizer_name, 'model': model_name, 'history': []}
    global_step = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        total_loss = 0
        num_batches = 0
        optimizer.zero_grad()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels) / accum_steps
            
            scaler.scale(loss).backward()
            total_loss += loss.item() * accum_steps
            num_batches += 1
            
            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
        
        avg_loss = total_loss / num_batches
        
        # Evaluate every 10 epochs
        if rank == 0 and (epoch % 10 == 0 or epoch == epochs):
            acc = evaluate(model, val_loader, device)
            results['history'].append({'epoch': epoch, 'loss': avg_loss, 'acc': acc})
            
            if acc > best_acc:
                best_acc = acc
                logger.info(f"E{epoch:3d}: loss={avg_loss:.4f} acc={acc:.2f}%*")
            else:
                logger.info(f"E{epoch:3d}: loss={avg_loss:.4f} acc={acc:.2f}%")
        elif rank == 0 and epoch % 5 == 0:
            logger.info(f"E{epoch:3d}: loss={avg_loss:.4f}")
    
    if rank == 0:
        results['best_acc'] = best_acc
        logger.info(f"Final: Best Acc = {best_acc:.2f}%")
    
    # Cleanup
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default='all')
    parser.add_argument('--model', type=str, default='vit_s16', choices=['vit_s16', 'vit_b16'])
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=90)
    args = parser.parse_args()
    
    # Distributed setup
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if world_size > 1:
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
    
    device = torch.device(f'cuda:{local_rank}')
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(output_dir, rank, f'cls_{args.model}')
    
    # Optimizers to test
    if args.optimizer == 'all':
        optimizers = ['adamw', 'lion', 'rlo', 'rlo_lambda_a', 'smooth_lifted_rlo']
    else:
        optimizers = [args.optimizer]
    
    results = {}
    
    for opt in optimizers:
        try:
            results[opt] = train_classification(opt, args.model, Path(args.data),
                                               output_dir, args.epochs, rank, world_size,
                                               device, logger)
            
            if rank == 0:
                with open(output_dir / f'{opt}_{args.model}_results.json', 'w') as f:
                    json.dump(results[opt], f, indent=2)
        
        except Exception as e:
            if rank == 0:
                logger.error(f"Error with {opt}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    # Summary
    if rank == 0:
        logger.info("=" * 70)
        logger.info(f"Classification Results ({args.model}, Top-1 Accuracy %):")
        for opt in sorted(results.keys(), key=lambda x: -results[x].get('best_acc', 0)):
            acc = results[opt].get('best_acc', 0)
            logger.info(f"  {opt}: {acc:.2f}%")
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
