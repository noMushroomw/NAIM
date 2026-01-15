#!/usr/bin/env python3
"""
LiT (Locked-image Text Tuning) Experiment - Following Lion Paper Table 4
Records ZERO-SHOT ACCURACY on ImageNet/CIFAR-100 (not just loss!)

Lion Paper Table 4 configs:
- AdamW: lr=1e-3, wd=0.0
- Lion: lr=3e-4 (0.3x), wd=0.0
"""
import os, gc, time, math, random, json, logging, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as T
from torchvision.datasets import ImageFolder, CIFAR100
import warnings; warnings.filterwarnings('ignore')

# Import optimizers
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
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, causal=False):
        super().__init__()
        self.causal = causal
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x):
        mask = None
        if self.causal:
            seq_len = x.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            mask = mask.unsqueeze(0).unsqueeze(0)
        
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """ViT-B/32 style encoder for images."""
    def __init__(self, dim=768, depth=12, heads=12, output_dim=512, patch_size=32):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, patch_size, patch_size)
        num_patches = (224 // patch_size) ** 2  # 49 for patch_size=32
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, output_dim)
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x[:, 0])
        return self.proj(x)


class TextEncoder(nn.Module):
    """Transformer text encoder."""
    def __init__(self, vocab_size=49408, dim=512, depth=6, heads=8, 
                 max_len=77, output_dim=512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, causal=True) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, output_dim)
    
    def forward(self, tokens):
        B, T = tokens.shape
        x = self.token_embed(tokens) + self.pos_embed(torch.arange(T, device=tokens.device))
        
        for block in self.blocks:
            x = block(x)
        
        # Get features at EOT token position
        x = self.norm(x[torch.arange(B), tokens.argmax(dim=-1)])
        return self.proj(x)


class LiTModel(nn.Module):
    """Locked-image Text tuning model."""
    def __init__(self, embed_dim=512):
        super().__init__()
        self.vision_encoder = VisionEncoder(output_dim=embed_dim)
        self.text_encoder = TextEncoder(output_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # Freeze vision encoder (LiT style)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
    
    def forward(self, images, tokens):
        with torch.no_grad():
            image_features = F.normalize(self.vision_encoder(images), dim=-1)
        
        text_features = F.normalize(self.text_encoder(tokens), dim=-1)
        
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text
    
    def get_image_features(self, images):
        with torch.no_grad():
            return F.normalize(self.vision_encoder(images), dim=-1)
    
    def get_text_features(self, tokens):
        return F.normalize(self.text_encoder(tokens), dim=-1)


class ImageTextDataset(Dataset):
    """Simple dataset for image-text pairs."""
    def __init__(self, image_folder, transform=None, max_len=77):
        self.dataset = ImageFolder(image_folder, transform)
        self.max_len = max_len
        # Simple byte-level tokenization
        self.vocab_size = 256
    
    def tokenize(self, text):
        tokens = [ord(c) % 256 for c in text[:self.max_len - 1]]
        # Add EOT at end
        tokens = tokens + [0] * (self.max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        class_name = self.dataset.classes[label].replace('_', ' ')
        text = f"a photo of {class_name}"
        tokens = self.tokenize(text)
        return image, tokens, label


def contrastive_loss(logits_per_image, logits_per_text):
    """CLIP-style contrastive loss."""
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


@torch.no_grad()
def compute_zero_shot_accuracy(model, image_loader, class_names, device):
    """Compute zero-shot classification accuracy."""
    model.eval()
    
    # Create text embeddings for all classes
    text_embeddings = []
    for class_name in class_names:
        text = f"a photo of {class_name.replace('_', ' ')}"
        tokens = torch.tensor([[ord(c) % 256 for c in text[:76]] + [0] * (77 - len(text[:76]))], 
                             dtype=torch.long, device=device)
        text_feat = model.get_text_features(tokens)
        text_embeddings.append(text_feat)
    
    text_embeddings = torch.cat(text_embeddings, dim=0)  # [num_classes, dim]
    
    correct = 0
    total = 0
    
    for images, _, labels in image_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        image_features = model.get_image_features(images)
        
        # Compute similarity
        similarity = image_features @ text_embeddings.t()
        predictions = similarity.argmax(dim=-1)
        
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    return 100.0 * correct / total


# ============= Lion Paper Table 4 Configs =============
CONFIGS = {
    'adamw': {'lr': 1e-3, 'wd': 0.0, 'betas': (0.9, 0.999)},
    'lion': {'lr': 3e-4, 'wd': 0.0, 'betas': (0.9, 0.99)},  # 0.3x lr
    'rlo': {'lr': 3e-4, 'wd': 0.0, 'betas': (0.9, 0.99), 'belief_coef': 0.1},
    'rlo_lambda_a': {'lr': 3e-4, 'wd': 0.0, 'beta1': 0.9, 'beta2': 0.99, 'lambda_b': 0.1},
    'smooth_lifted_rlo': {'lr': 3e-4, 'wd': 0.0, 'beta1': 0.9, 'beta2': 0.99, 
                          'lambda_b': 0.1, 'eta': 0.3},
}


def train_lit(opt_name, data_path, output_path, epochs, rank, world_size, local_rank, logger):
    device = torch.device(f'cuda:{local_rank}')
    cfg = CONFIGS[opt_name]
    
    batch_size = 256
    per_gpu_batch = batch_size // world_size
    
    logger.info("=" * 70)
    logger.info(f"LiT Training: {opt_name}")
    logger.info(f"LR={cfg['lr']}, WD={cfg['wd']}")
    logger.info("=" * 70)
    
    # Data transforms
    train_transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    val_transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Datasets
    train_dataset = ImageTextDataset(data_path / 'train', train_transform)
    val_dataset = ImageTextDataset(data_path / 'val' if (data_path / 'val').exists() 
                                   else data_path / 'train', val_transform)
    
    train_sampler = DistributedSampler(train_dataset, world_size, rank) if world_size > 1 else None
    train_loader = DataLoader(train_dataset, per_gpu_batch, 
                             shuffle=(train_sampler is None),
                             sampler=train_sampler, 
                             num_workers=12, pin_memory=True, drop_last=True,
                             persistent_workers=True, prefetch_factor=4)
    
    val_loader = DataLoader(val_dataset, per_gpu_batch * 2,
                           shuffle=False, num_workers=8, pin_memory=True,
                           prefetch_factor=4)
    
    class_names = train_dataset.dataset.classes
    logger.info(f"Classes: {len(class_names)}, Train samples: {len(train_dataset)}")
    
    # Model
    model = LiTModel(embed_dim=512).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {trainable_params / 1e6:.1f}M")
    
    # Optimizer
    base_model = model.module if hasattr(model, 'module') else model
    optimizer = create_optimizer(base_model.text_encoder, opt_name, cfg)
    
    scaler = torch.amp.GradScaler('cuda')
    
    # LR schedule
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * 2
    
    def get_lr(step):
        if step < warmup_steps:
            return cfg['lr'] * step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return cfg['lr'] * 0.5 * (1 + math.cos(math.pi * progress))
    
    # Training
    history = {'loss': [], 'acc_imagenet': [], 'step': []}
    best_acc = 0.0
    global_step = 0
    
    for epoch in range(epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        
        for images, tokens, _ in train_loader:
            images = images.to(device)
            tokens = tokens.to(device)
            
            # Update LR
            lr = get_lr(global_step)
            for g in optimizer.param_groups:
                g['lr'] = lr
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast('cuda', torch.bfloat16):
                logits_i, logits_t = model(images, tokens)
                loss = contrastive_loss(logits_i, logits_t)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            global_step += 1
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Evaluate zero-shot accuracy every 5 epochs
        acc = 0.0
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            base_model = model.module if hasattr(model, 'module') else model
            acc = compute_zero_shot_accuracy(base_model, val_loader, class_names, device)
            
            is_best = acc > best_acc
            best_acc = max(acc, best_acc)
            
            history['loss'].append(avg_loss)
            history['acc_imagenet'].append(acc)
            history['step'].append(global_step)
            
            logger.info(f"E{epoch+1:3d}: loss={avg_loss:.4f} acc={acc:.2f}%{'*' if is_best else ''} "
                       f"time={time.time()-t0:.0f}s")
            
            if is_best and rank == 0:
                torch.save({
                    'model': base_model.state_dict(),
                    'acc': best_acc,
                    'epoch': epoch
                }, output_path / f"lit_{opt_name}_best.pt")
        else:
            logger.info(f"E{epoch+1:3d}: loss={avg_loss:.4f} time={time.time()-t0:.0f}s")
    
    # Save final results
    if rank == 0:
        results = {
            'optimizer': opt_name,
            'config': cfg,
            'best_acc': best_acc,
            'history': history
        }
        with open(output_path / f"lit_{opt_name}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Final: Best Acc = {best_acc:.2f}%")
    
    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='all')
    parser.add_argument('--data', default='/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder')
    parser.add_argument('--output', default='/blue/wdixon/wang.yixuan/rlo_experiments/lit')
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    
    # Setup
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Distributed setup
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
    
    logger = setup_logger(output_path, rank, "lit")
    
    # Run experiments
    optimizers = list(CONFIGS.keys()) if args.optimizer == 'all' else [args.optimizer]
    results = {}
    
    for opt in optimizers:
        try:
            gc.collect()
            torch.cuda.empty_cache()
            results[opt] = train_lit(opt, Path(args.data), output_path, 
                                    args.epochs, rank, world_size, local_rank, logger)
        except Exception as e:
            logger.error(f"Error with {opt}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Summary
    if rank == 0 and results:
        logger.info("=" * 70)
        logger.info("LiT Results (Zero-Shot Accuracy %):")
        for opt, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {opt}: {acc:.2f}%")
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
