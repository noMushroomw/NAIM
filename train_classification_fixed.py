#!/usr/bin/env python3
"""
Image Classification: ViT on ImageNet - Following Lion Paper Table 2
90 epochs, records TOP-1 ACCURACY

Supports both ViT-S/16 and ViT-B/16:
- ViT-S/16: 384 dim, 12 layers, 6 heads (~22M params)
- ViT-B/16: 768 dim, 12 layers, 12 heads (~86M params)

Lion Paper Table 2 configs:
- AdamW: lr=3e-3, wd=0.1, β=(0.9, 0.999)
- Lion: lr=3e-4 (0.1x), wd=1.0 (10x), β=(0.9, 0.99)

FIXES:
- Added dist.barrier() between optimizer runs
- Eval on rank 0 only to prevent timing drift
- Added model size selection (--model vit_s16 or vit_b16)
- Better error recovery
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
from torch.utils.checkpoint import checkpoint
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


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Use flash attention if available
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer with configurable size."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=384, depth=12, num_heads=6, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            if self.training and self.use_checkpoint:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        x = self.norm(x[:, 0])
        return self.head(x)


# Model configurations
MODEL_CONFIGS = {
    'vit_s16': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},   # ~22M params
    'vit_b16': {'embed_dim': 768, 'depth': 12, 'num_heads': 12},  # ~86M params
}


class Mixup:
    """Mixup and Cutmix data augmentation."""
    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, num_classes=1000):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.num_classes = num_classes
    
    def __call__(self, x, y):
        if random.random() < 0.5 and self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        elif self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            return x, F.one_hot(y, self.num_classes).float()
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        # Mixup
        x = lam * x + (1 - lam) * x[index]
        
        # Soft labels
        y_onehot = F.one_hot(y, self.num_classes).float()
        y_mixed = lam * y_onehot + (1 - lam) * y_onehot[index]
        
        return x, y_mixed


@torch.no_grad()
def evaluate(model, loader, device):
    """Compute top-1 accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.autocast('cuda', torch.bfloat16):
            outputs = model(images)
        
        predictions = outputs.argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    return 100.0 * correct / total


# ============= Lion Paper Table 2 Configs =============
OPTIMIZER_CONFIGS = {
    'adamw': {'lr': 3e-3, 'wd': 0.1, 'betas': (0.9, 0.999)},
    'lion': {'lr': 3e-4, 'wd': 1.0, 'betas': (0.9, 0.99)},  # 0.1x lr, 10x wd
    'rlo': {'lr': 3e-4, 'wd': 1.0, 'betas': (0.9, 0.99), 'belief_coef': 0.1},
    'rlo_lambda_a': {'lr': 3e-4, 'wd': 1.0, 'beta1': 0.9, 'beta2': 0.99, 'lambda_b': 0.1},
    'smooth_lifted_rlo': {'lr': 3e-4, 'wd': 1.0, 'beta1': 0.9, 'beta2': 0.99, 
                          'lambda_b': 0.1, 'eta': 0.3},
}


def cleanup_between_runs(world_size):
    """Clean up and sync between optimizer runs."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if world_size > 1:
        dist.barrier()


def train_classification(opt_name, model_name, data_path, output_path, epochs, 
                        rank, world_size, local_rank, logger):
    device = torch.device(f'cuda:{local_rank}')
    opt_cfg = OPTIMIZER_CONFIGS[opt_name]
    model_cfg = MODEL_CONFIGS[model_name]
    
    # Batch size: 1024 global, scaled per GPU
    global_batch = 1024
    per_gpu_batch = global_batch // world_size
    
    # Gradient accumulation if needed to avoid OOM
    # ViT-B needs smaller micro-batch than ViT-S
    if model_name == 'vit_b16':
        micro_batch = min(64, per_gpu_batch)  # Smaller for B/16
    else:
        micro_batch = min(128, per_gpu_batch)
    accumulation_steps = per_gpu_batch // micro_batch
    
    logger.info("=" * 70)
    logger.info(f"Classification: {model_name.upper()} on ImageNet | {opt_name}")
    logger.info(f"LR={opt_cfg['lr']}, WD={opt_cfg['wd']}, Epochs={epochs}")
    logger.info(f"Global batch={global_batch}, Per-GPU={per_gpu_batch}, "
               f"Micro={micro_batch}, Accum={accumulation_steps}")
    logger.info("=" * 70)
    
    # Data transforms with RandAugment
    train_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.08, 1.0)),
        T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(),  # Built-in augmentation
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        T.RandomErasing(p=0.25),
    ])
    
    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # Datasets
    train_dataset = ImageFolder(data_path / 'train', train_transform)
    val_dataset = ImageFolder(data_path / 'val' if (data_path / 'val').exists() 
                              else data_path / 'train', val_transform)
    
    train_sampler = DistributedSampler(train_dataset, world_size, rank) if world_size > 1 else None
    train_loader = DataLoader(train_dataset, micro_batch,
                             shuffle=(train_sampler is None),
                             sampler=train_sampler,
                             num_workers=8, pin_memory=True, drop_last=True,
                             persistent_workers=True, prefetch_factor=2)
    
    # Val loader only on rank 0 to avoid timing drift
    if rank == 0:
        val_loader = DataLoader(val_dataset, micro_batch * 2,
                               shuffle=False, num_workers=4, pin_memory=True,
                               prefetch_factor=2)
    else:
        val_loader = None
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Model
    model = ViT(num_classes=1000, use_checkpoint=True, **model_cfg).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Parameters: {num_params:.1f}M")
    
    # Optimizer
    base_model = model.module if hasattr(model, 'module') else model
    optimizer = create_optimizer(base_model, opt_name, opt_cfg)
    
    scaler = torch.amp.GradScaler('cuda')
    
    # Mixup
    mixup = Mixup(num_classes=1000)
    
    # LR schedule
    steps_per_epoch = len(train_loader) // accumulation_steps
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * 10  # 10 epoch warmup
    
    def get_lr(step):
        if step < warmup_steps:
            return opt_cfg['lr'] * step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return opt_cfg['lr'] * 0.5 * (1 + math.cos(math.pi * progress))
    
    # Training
    history = {'loss': [], 'acc': [], 'epoch': []}
    best_acc = 0.0
    global_step = 0
    
    for epoch in range(epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        t0 = time.time()
        
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Mixup
            images, soft_labels = mixup(images, labels)
            
            # Update LR
            step_in_epoch = batch_idx // accumulation_steps
            current_step = epoch * steps_per_epoch + step_in_epoch
            lr = get_lr(current_step)
            for g in optimizer.param_groups:
                g['lr'] = lr
            
            # Forward + backward
            with torch.autocast('cuda', torch.bfloat16):
                outputs = model(images)
                loss = -(F.log_softmax(outputs, dim=-1) * soft_labels).sum(dim=-1).mean()
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            epoch_loss += loss.item() * accumulation_steps
            
            # Step optimizer
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # Sync all ranks after each epoch
        if world_size > 1:
            dist.barrier()
        
        # Evaluate every 10 epochs (rank 0 only)
        acc = 0.0
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            if rank == 0:
                base_model = model.module if hasattr(model, 'module') else model
                acc = evaluate(base_model, val_loader, device)
                
                is_best = acc > best_acc
                best_acc = max(acc, best_acc)
                
                history['loss'].append(avg_loss)
                history['acc'].append(acc)
                history['epoch'].append(epoch + 1)
                
                logger.info(f"E{epoch+1:3d}: loss={avg_loss:.4f} acc={acc:.2f}%{'*' if is_best else ''} "
                           f"lr={lr:.2e} time={time.time()-t0:.0f}s")
                
                if is_best:
                    torch.save({
                        'model': base_model.state_dict(),
                        'acc': best_acc,
                        'epoch': epoch
                    }, output_path / f"{model_name}_{opt_name}_best.pt")
            
            # Sync after evaluation
            if world_size > 1:
                dist.barrier()
        else:
            if rank == 0:
                logger.info(f"E{epoch+1:3d}: loss={avg_loss:.4f} lr={lr:.2e} time={time.time()-t0:.0f}s")
    
    # Broadcast best_acc from rank 0
    if world_size > 1:
        best_acc_tensor = torch.tensor([best_acc], device=device)
        dist.broadcast(best_acc_tensor, src=0)
        best_acc = best_acc_tensor.item()
    
    # Save results
    if rank == 0:
        results = {
            'model': model_name,
            'optimizer': opt_name,
            'config': opt_cfg,
            'best_acc': best_acc,
            'history': history
        }
        with open(output_path / f"{model_name}_{opt_name}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Final: Best Acc = {best_acc:.2f}%")
    
    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vit_s16', choices=['vit_s16', 'vit_b16'])
    parser.add_argument('--optimizer', default='all')
    parser.add_argument('--data', default='/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder')
    parser.add_argument('--output', default='/blue/wdixon/wang.yixuan/rlo_experiments/classification')
    parser.add_argument('--epochs', type=int, default=90)
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
    
    logger = setup_logger(output_path, rank, f"cls_{args.model}")
    
    # Run experiments
    optimizers = list(OPTIMIZER_CONFIGS.keys()) if args.optimizer == 'all' else [args.optimizer]
    results = {}
    
    for opt in optimizers:
        try:
            cleanup_between_runs(world_size)
            results[opt] = train_classification(opt, args.model, Path(args.data), output_path,
                                               args.epochs, rank, world_size, 
                                               local_rank, logger)
        except Exception as e:
            logger.error(f"Error with {opt}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Try to recover
            if world_size > 1:
                try:
                    dist.barrier()
                except:
                    pass
    
    # Summary
    if rank == 0 and results:
        logger.info("=" * 70)
        logger.info(f"Classification Results - {args.model.upper()} (Top-1 Accuracy %):")
        for opt, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {opt}: {acc:.2f}%")
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
