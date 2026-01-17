#!/usr/bin/env python3
"""
LiT (Locked-image Text Tuning) Experiment - Following Lion Paper Table 4
Uses pre-trained CLIP vision encoder (frozen), trains text encoder
Records Zero-shot Classification Accuracy
"""
import os, gc, time, math, json, logging, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
import warnings; warnings.filterwarnings('ignore')

try:
    from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

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


# ============================================================================
# Simple ViT and Text Encoder for non-CLIP fallback
# ============================================================================

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=512):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, patch_size, patch_size)
    
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
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
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VisionEncoder(nn.Module):
    """Simple ViT for vision encoding."""
    def __init__(self, embed_dim=512, depth=12, num_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.proj(self.norm(x[:, 0]))


class TextEncoder(nn.Module):
    """Simple transformer for text encoding."""
    def __init__(self, vocab_size=50000, embed_dim=512, depth=6, num_heads=8, max_len=77):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, tokens):
        x = self.token_embed(tokens) + self.pos_embed[:, :tokens.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        # Use last token (EOT position) or mean pooling
        x = self.norm(x[:, -1])
        return self.proj(x)


class SimpleCLIP(nn.Module):
    """Simple CLIP-like model."""
    def __init__(self, embed_dim=512, freeze_vision=True):
        super().__init__()
        self.vision_encoder = VisionEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
    
    def forward(self, images, tokens):
        with torch.no_grad():
            image_features = F.normalize(self.vision_encoder(images), dim=-1)
        text_features = F.normalize(self.text_encoder(tokens), dim=-1)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        return logits_per_image, logits_per_image.t()
    
    def get_image_features(self, images):
        with torch.no_grad():
            return F.normalize(self.vision_encoder(images), dim=-1)
    
    def get_text_features(self, tokens):
        return F.normalize(self.text_encoder(tokens), dim=-1)


# ============================================================================
# Dataset
# ============================================================================

class ImageTextDataset(Dataset):
    """Dataset that creates image-text pairs from ImageFolder."""
    def __init__(self, image_folder, transform=None, tokenizer=None):
        self.dataset = ImageFolder(image_folder, transform)
        self.tokenizer = tokenizer
        self.max_len = 77
    
    def simple_tokenize(self, text):
        """Simple byte-level tokenization fallback."""
        tokens = [ord(c) % 50000 for c in text[:self.max_len - 1]]
        tokens = tokens + [0] * (self.max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        class_name = self.dataset.classes[label].replace('_', ' ')
        text = f"a photo of a {class_name}"
        
        if self.tokenizer:
            tokens = self.tokenizer(text, max_length=self.max_len, padding='max_length',
                                   truncation=True, return_tensors='pt')['input_ids'].squeeze(0)
        else:
            tokens = self.simple_tokenize(text)
        
        return image, tokens, label


def contrastive_loss(logits_per_image, logits_per_text):
    """CLIP-style symmetric contrastive loss."""
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


@torch.no_grad()
def compute_zero_shot_accuracy(model, image_loader, class_names, device, tokenizer=None):
    """Compute zero-shot classification accuracy."""
    model.eval()
    
    # Create text embeddings for all classes
    texts = [f"a photo of a {name.replace('_', ' ')}" for name in class_names]
    
    if tokenizer:
        text_tokens = tokenizer(texts, max_length=77, padding='max_length',
                               truncation=True, return_tensors='pt')['input_ids'].to(device)
    else:
        # Simple tokenization
        text_tokens = []
        for text in texts:
            tokens = [ord(c) % 50000 for c in text[:76]]
            tokens = tokens + [0] * (77 - len(tokens))
            text_tokens.append(tokens)
        text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=device)
    
    # Get text features
    if hasattr(model, 'module'):
        text_features = model.module.get_text_features(text_tokens)
    else:
        text_features = model.get_text_features(text_tokens)
    text_features = F.normalize(text_features, dim=-1)
    
    correct = 0
    total = 0
    
    for images, _, labels in image_loader:
        images, labels = images.to(device), labels.to(device)
        
        if hasattr(model, 'module'):
            image_features = model.module.get_image_features(images)
        else:
            image_features = model.get_image_features(images)
        image_features = F.normalize(image_features, dim=-1)
        
        # Compute similarity
        similarity = image_features @ text_features.t()
        predictions = similarity.argmax(dim=-1)
        
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    return 100.0 * correct / total


def train_lit(optimizer_name, data_path, output_dir, epochs, rank, world_size, device, logger):
    """Train LiT model with specified optimizer."""
    
    # Hyperparameters from Lion paper Table 4
    configs = {
        'adamw': {'lr': 1e-3, 'wd': 0.0, 'betas': (0.9, 0.99)},
        'lion': {'lr': 3e-4, 'wd': 0.0, 'betas': (0.9, 0.99)},  # 0.3x lr
        'rlo': {'lr': 3e-4, 'wd': 0.0},
        'rlo_lambda_a': {'lr': 3e-4, 'wd': 0.0},
        'smooth_lifted_rlo': {'lr': 3e-4, 'wd': 0.0},
    }
    
    cfg = configs.get(optimizer_name, configs['adamw'])
    
    if rank == 0:
        logger.info("=" * 70)
        logger.info(f"LiT Training: {optimizer_name}")
        logger.info(f"LR={cfg['lr']}, WD={cfg['wd']}")
        logger.info("=" * 70)
    
    # Data
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    train_path = data_path / 'train' if (data_path / 'train').exists() else data_path
    val_path = data_path / 'val' if (data_path / 'val').exists() else train_path
    
    # Try to use HuggingFace CLIP tokenizer
    tokenizer = None
    if HAS_TRANSFORMERS:
        try:
            tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')
        except Exception as e:
            if rank == 0:
                logger.warning(f"Could not load CLIP tokenizer: {e}")
    
    train_dataset = ImageTextDataset(train_path, transform, tokenizer)
    val_dataset = ImageTextDataset(val_path, val_transform, tokenizer)
    
    per_gpu_batch = 128
    
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
    
    if rank == 0:
        logger.info(f"Classes: {len(class_names)}, Train samples: {len(train_dataset)}")
    
    # Model - use pre-trained CLIP if available
    if HAS_TRANSFORMERS:
        try:
            clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16').to(device)
            # Freeze vision encoder (LiT style)
            for param in clip_model.vision_model.parameters():
                param.requires_grad = False
            model = clip_model
            use_clip = True
            if rank == 0:
                logger.info("Using pre-trained CLIP model")
        except Exception as e:
            if rank == 0:
                logger.warning(f"Could not load CLIP: {e}, using SimpleCLIP")
            model = SimpleCLIP(freeze_vision=True).to(device)
            use_clip = False
    else:
        model = SimpleCLIP(freeze_vision=True).to(device)
        use_clip = False
        if rank == 0:
            logger.info("Using SimpleCLIP (transformers not available)")
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Count trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        logger.info(f"Trainable params: {trainable_params/1e6:.1f}M")
    
    # Optimizer - only optimize trainable parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = create_optimizer(optimizer_name, trainable,
                                lr=cfg['lr'], weight_decay=cfg['wd'],
                                betas=cfg.get('betas'))
    
    # Training
    best_acc = 0
    results = {'optimizer': optimizer_name, 'history': []}
    scaler = torch.amp.GradScaler()
    
    for epoch in range(1, epochs + 1):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        total_loss = 0
        num_batches = 0
        
        for images, tokens, _ in train_loader:
            images, tokens = images.to(device), tokens.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                if use_clip:
                    outputs = model(pixel_values=images, input_ids=tokens)
                    logits_per_image = outputs.logits_per_image
                    logits_per_text = outputs.logits_per_text
                else:
                    logits_per_image, logits_per_text = model(images, tokens)
                
                loss = contrastive_loss(logits_per_image, logits_per_text)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Evaluate every 5 epochs
        if rank == 0 and (epoch % 5 == 0 or epoch == epochs):
            # Custom zero-shot evaluation
            model.eval()
            
            if use_clip:
                # Use CLIP's built-in methods
                raw_model = model.module if hasattr(model, 'module') else model
                
                # Create class text embeddings
                texts = [f"a photo of a {name.replace('_', ' ')}" for name in class_names]
                text_inputs = tokenizer(texts, max_length=77, padding='max_length',
                                       truncation=True, return_tensors='pt').to(device)
                
                with torch.no_grad():
                    text_features = raw_model.get_text_features(**text_inputs)
                    text_features = F.normalize(text_features, dim=-1)
                
                correct = 0
                total = 0
                
                for images, _, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    
                    with torch.no_grad():
                        image_features = raw_model.get_image_features(pixel_values=images)
                        image_features = F.normalize(image_features, dim=-1)
                    
                    similarity = image_features @ text_features.t()
                    predictions = similarity.argmax(dim=-1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
                
                acc = 100.0 * correct / total
            else:
                acc = compute_zero_shot_accuracy(model, val_loader, class_names, device, tokenizer)
            
            results['history'].append({'epoch': epoch, 'loss': avg_loss, 'acc': acc})
            
            if acc > best_acc:
                best_acc = acc
                logger.info(f"E{epoch:3d}: loss={avg_loss:.4f} acc={acc:.2f}%*")
            else:
                logger.info(f"E{epoch:3d}: loss={avg_loss:.4f} acc={acc:.2f}%")
        elif rank == 0:
            logger.info(f"E{epoch:3d}: loss={avg_loss:.4f}")
    
    if rank == 0:
        results['best_acc'] = best_acc
        logger.info(f"Final: Best Acc = {best_acc:.2f}%")
    
    # Cleanup
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default='all')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=30)
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
    
    logger = setup_logger(output_dir, rank, 'lit')
    
    # Optimizers to test
    if args.optimizer == 'all':
        optimizers = ['adamw', 'lion', 'rlo', 'rlo_lambda_a', 'smooth_lifted_rlo']
    else:
        optimizers = [args.optimizer]
    
    results = {}
    
    for opt in optimizers:
        try:
            results[opt] = train_lit(opt, Path(args.data), output_dir, args.epochs,
                                    rank, world_size, device, logger)
            
            if rank == 0:
                with open(output_dir / f'{opt}_results.json', 'w') as f:
                    json.dump(results[opt], f, indent=2)
        
        except Exception as e:
            if rank == 0:
                logger.error(f"Error with {opt}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    # Summary
    if rank == 0:
        logger.info("=" * 70)
        logger.info("LiT Results (Zero-Shot Accuracy %):")
        for opt in sorted(results.keys(), key=lambda x: -results[x].get('best_acc', 0)):
            acc = results[opt].get('best_acc', 0)
            logger.info(f"  {opt}: {acc:.2f}%")
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
