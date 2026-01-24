#!/usr/bin/env python3
"""
Image Classification: ViT-B/16 on ImageNet
90 epochs, reports Top-1 Accuracy
"""
import os, gc, math, json, logging, argparse
from pathlib import Path
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


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)
    
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)
        self.drop_path = drop_path
    
    def forward(self, x):
        if self.drop_path > 0 and self.training:
            x = x + self._drop_path(self.attn(self.norm1(x)))
            x = x + self._drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x
    
    def _drop_path(self, x):
        if self.drop_path == 0:
            return x
        keep = 1 - self.drop_path
        mask = (torch.rand(x.shape[0], 1, 1, device=x.device) < keep).float() / keep
        return x * mask


class ViTB16(nn.Module):
    """ViT-B/16: 768 dim, 12 layers, 12 heads (~86M params)."""
    def __init__(self, num_classes=1000, dropout=0.0, drop_path=0.1):
        super().__init__()
        dim = 768
        depth = 12
        num_heads = 12
        
        self.patch_embed = PatchEmbed(224, 16, 3, dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.pos_drop = nn.Dropout(dropout)
        
        dpr = [drop_path * i / (depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, 4.0, dropout, dpr[i]) for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x[:, 0])
        return self.head(x)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def train_classification(opt_name, data_path, output_dir, epochs, rank, world_size, device, logger):
    # Tuned configs following Lion paper Table 2
    configs = {
        'adamw': {'lr': 3e-3, 'wd': 0.3, 'betas': (0.9, 0.999)},
        'lion': {'lr': 3e-4, 'wd': 1.0, 'betas': (0.9, 0.99)},
        'rlo': {'lr': 3.5e-4, 'wd': 1.0, 'betas': (0.9, 0.99)},
        'rlo_lambda_a': {'lr': 3.5e-4, 'wd': 1.0, 'betas': (0.9, 0.99)},
        'smooth_lifted_rlo': {'lr': 3.5e-4, 'wd': 1.0, 'betas': (0.9, 0.99)},
    }
    
    cfg = configs.get(opt_name, configs['adamw'])
    
    if rank == 0:
        logger.info("=" * 70)
        logger.info(f"ViT-B/16 Classification: {opt_name}")
        logger.info(f"LR={cfg['lr']}, WD={cfg['wd']}")
        logger.info("=" * 70)
    
    # Data augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25),
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
    
    # Batch size: 1024 global, 128 per GPU for 8 GPUs
    batch_size = 128
    
    train_sampler = DistributedSampler(train_dataset, world_size, rank) if world_size > 1 else None
    train_loader = DataLoader(train_dataset, batch_size, shuffle=(train_sampler is None),
                             sampler=train_sampler, num_workers=8, pin_memory=True,
                             drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)
    
    if rank == 0:
        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model = ViTB16(num_classes=1000, dropout=0.0, drop_path=0.1).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"Parameters: {params:.1f}M")
    
    # Optimizer
    optimizer = create_optimizer(opt_name, model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'], betas=cfg['betas'])
    
    # LR schedule: warmup + cosine
    warmup_epochs = 5
    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training
    scaler = torch.amp.GradScaler()
    best_acc = 0
    results = {'optimizer': opt_name, 'history': []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        total_loss = 0
        num_batches = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels, label_smoothing=0.1)
            
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
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Evaluate
        if rank == 0 and (epoch % 10 == 0 or epoch == epochs):
            acc = evaluate(model, val_loader, device)
            results['history'].append({'epoch': epoch, 'loss': avg_loss, 'acc': acc})
            
            marker = '*' if acc > best_acc else ''
            best_acc = max(best_acc, acc)
            logger.info(f"E{epoch:3d}: loss={avg_loss:.4f} acc={acc:.2f}%{marker}")
        elif rank == 0 and epoch % 5 == 0:
            logger.info(f"E{epoch:3d}: loss={avg_loss:.4f}")
    
    if rank == 0:
        results['best_acc'] = best_acc
        logger.info(f"Final: Best Acc = {best_acc:.2f}%")
    
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default='all')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=90)
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
    
    logger = setup_logger(output_dir, rank, 'cls_vit_b16')
    
    if args.optimizer == 'all':
        optimizers = ['adamw', 'lion', 'rlo', 'rlo_lambda_a', 'smooth_lifted_rlo']
    else:
        optimizers = [args.optimizer]
    
    results = {}
    for opt in optimizers:
        try:
            results[opt] = train_classification(opt, Path(args.data), output_dir, args.epochs, rank, world_size, device, logger)
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
        logger.info("ViT-B/16 RESULTS (Top-1 Accuracy ↑)")
        logger.info("=" * 70)
        for opt in sorted(results.keys(), key=lambda x: -results[x].get('best_acc', 0)):
            acc = results[opt].get('best_acc', 0)
            logger.info(f"  {opt}: {acc:.2f}%")
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
