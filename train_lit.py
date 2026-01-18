#!/usr/bin/env python3
"""
LiT (Locked-image Tuning) on ImageNet
Uses pre-trained CLIP vision encoder (frozen) + trainable text encoder
Reports Zero-shot Top-1 Accuracy
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


# ============= Simple CLIP Implementation =============
class VisionEncoder(nn.Module):
    """ViT-B/16 vision encoder."""
    def __init__(self, embed_dim=512):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 768, 16, 16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, 768))  # 14x14 + 1
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(768, 12, 3072, dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(12)
        ])
        self.norm = nn.LayerNorm(768)
        self.proj = nn.Linear(768, embed_dim)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, 0])
        return F.normalize(self.proj(x), dim=-1)


class TextEncoder(nn.Module):
    """Transformer text encoder."""
    def __init__(self, vocab_size=49408, embed_dim=512, max_len=77):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, 512)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, 512))
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(512, 8, 2048, dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(12)
        ])
        self.norm = nn.LayerNorm(512)
        self.proj = nn.Linear(512, embed_dim)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, tokens):
        x = self.token_embed(tokens) + self.pos_embed[:, :tokens.shape[1]]
        
        # Causal mask
        mask = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device), diagonal=1).bool()
        
        for blk in self.blocks:
            x = blk(x, src_mask=mask, is_causal=True)
        
        x = self.norm(x)
        # Use EOT token position
        eot_idx = tokens.argmax(dim=-1)
        x = x[torch.arange(x.shape[0], device=x.device), eot_idx]
        return F.normalize(self.proj(x), dim=-1)


class SimpleCLIP(nn.Module):
    """Simple CLIP model for LiT training."""
    def __init__(self, embed_dim=512):
        super().__init__()
        self.visual = VisionEncoder(embed_dim)
        self.text = TextEncoder(embed_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
    
    def forward(self, images, tokens):
        image_features = self.visual(images)
        text_features = self.text(tokens)
        return image_features, text_features, self.logit_scale.exp()


class LiTModel(nn.Module):
    """LiT: Locked-image Tuning - freeze vision, train text."""
    def __init__(self, embed_dim=512, use_pretrained=True):
        super().__init__()
        self.use_hf = False
        
        if use_pretrained:
            try:
                from transformers import CLIPModel
                clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
                self.visual = clip.vision_model
                self.visual_proj = clip.visual_projection
                self.text = clip.text_model
                self.text_proj = clip.text_projection
                self.logit_scale = clip.logit_scale
                self.use_hf = True
                
                # Freeze vision encoder
                for p in self.visual.parameters():
                    p.requires_grad = False
                for p in self.visual_proj.parameters():
                    p.requires_grad = False
            except Exception as e:
                print(f"Could not load pretrained CLIP: {e}")
                use_pretrained = False
        
        if not use_pretrained:
            self.clip = SimpleCLIP(embed_dim)
            # Freeze vision
            for p in self.clip.visual.parameters():
                p.requires_grad = False
    
    def forward(self, images, tokens):
        if self.use_hf:
            with torch.no_grad():
                vis_out = self.visual(images)
                image_embeds = self.visual_proj(vis_out.pooler_output)
                image_embeds = F.normalize(image_embeds, dim=-1)
            
            text_out = self.text(tokens)
            text_embeds = self.text_proj(text_out.pooler_output)
            text_embeds = F.normalize(text_embeds, dim=-1)
            
            return image_embeds, text_embeds, self.logit_scale.exp()
        else:
            with torch.no_grad():
                image_embeds = self.clip.visual(images)
            text_embeds = self.clip.text(tokens)
            return image_embeds, text_embeds, self.clip.logit_scale.exp()


class SimpleTokenizer:
    """Simple tokenizer for CLIP."""
    def __init__(self, max_len=77):
        self.max_len = max_len
        self.sot_token = 49406
        self.eot_token = 49407
    
    def __call__(self, text):
        # Simple byte-pair encoding simulation
        tokens = [self.sot_token]
        for c in text.lower()[:self.max_len - 2]:
            tokens.append(ord(c) % 49406)
        tokens.append(self.eot_token)
        
        # Pad
        while len(tokens) < self.max_len:
            tokens.append(0)
        
        return torch.tensor(tokens[:self.max_len], dtype=torch.long)


def get_tokenizer():
    """Get CLIP tokenizer."""
    try:
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        def tokenize(text):
            return tokenizer(text, padding='max_length', max_length=77, truncation=True, return_tensors='pt')['input_ids'][0]
        return tokenize
    except:
        return SimpleTokenizer()


# ImageNet class names (subset for memory)
IMAGENET_CLASSES = None

def load_imagenet_classes():
    global IMAGENET_CLASSES
    if IMAGENET_CLASSES is not None:
        return IMAGENET_CLASSES
    
    # Load from file or use default
    try:
        import json
        # Try to load from standard location
        with open('/home/claude/imagenet_classes.json', 'r') as f:
            IMAGENET_CLASSES = json.load(f)
    except:
        # Fallback: will be populated from dataset
        IMAGENET_CLASSES = [f"class_{i}" for i in range(1000)]
    
    return IMAGENET_CLASSES


@torch.no_grad()
def zero_shot_evaluate(model, loader, device, tokenizer, class_names):
    """Zero-shot evaluation using text prompts."""
    model.eval()
    
    # Unwrap DDP model if necessary
    raw_model = model.module if hasattr(model, 'module') else model
    
    # Create text embeddings for all classes
    templates = [
        "a photo of a {}.",
        "a blurry photo of a {}.",
        "a photo of the {}.",
        "a rendition of a {}.",
        "a photo of the large {}.",
        "a photo of the small {}.",
    ]
    
    text_embeddings = []
    for class_name in class_names:
        class_embeds = []
        for template in templates[:2]:  # Use fewer templates for speed
            text = template.format(class_name.replace('_', ' '))
            tokens = tokenizer(text).unsqueeze(0).to(device)
            
            if hasattr(raw_model, 'use_hf') and raw_model.use_hf:
                text_out = raw_model.text(tokens)
                embed = raw_model.text_proj(text_out.pooler_output)
                embed = F.normalize(embed, dim=-1)
            elif hasattr(raw_model, 'clip'):
                embed = raw_model.clip.text(tokens)
            else:
                embed = raw_model.text(tokens)
            
            class_embeds.append(embed)
        
        class_embed = torch.stack(class_embeds).mean(0)
        class_embed = F.normalize(class_embed, dim=-1)
        text_embeddings.append(class_embed)
    
    text_embeddings = torch.cat(text_embeddings, dim=0)  # [num_classes, embed_dim]
    
    correct, total = 0, 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # Get image embeddings
        if hasattr(raw_model, 'use_hf') and raw_model.use_hf:
            vis_out = raw_model.visual(images)
            image_embeds = raw_model.visual_proj(vis_out.pooler_output)
            image_embeds = F.normalize(image_embeds, dim=-1)
        elif hasattr(raw_model, 'clip'):
            image_embeds = raw_model.clip.visual(images)
        else:
            image_embeds = raw_model.visual(images)
        
        # Compute similarities
        logits = image_embeds @ text_embeddings.T
        preds = logits.argmax(dim=-1)
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return 100.0 * correct / total


def train_lit(opt_name, data_path, output_dir, epochs, rank, world_size, device, logger):
    # Tuned configs for CLIP fine-tuning
    # Lower LRs for Lion/RLO to prevent NaN
    configs = {
        'adamw': {'lr': 5e-4, 'wd': 0.2, 'betas': (0.9, 0.98)},
        'lion': {'lr': 5e-5, 'wd': 0.2, 'betas': (0.9, 0.99)},
        'rlo': {'lr': 6e-5, 'wd': 0.2, 'betas': (0.9, 0.99)},
        'rlo_lambda_a': {'lr': 6e-5, 'wd': 0.2, 'betas': (0.9, 0.99)},
        'smooth_lifted_rlo': {'lr': 6e-5, 'wd': 0.2, 'betas': (0.9, 0.99)},
    }
    
    cfg = configs.get(opt_name, configs['adamw'])
    
    if rank == 0:
        logger.info("=" * 70)
        logger.info(f"LiT Training: {opt_name}")
        logger.info(f"LR={cfg['lr']}, WD={cfg['wd']}")
        logger.info("=" * 70)
    
    # Data
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])
    
    train_path = data_path / 'train'
    val_path = data_path / 'val'
    
    if not train_path.exists():
        train_path = data_path
    if not val_path.exists():
        val_path = data_path
    
    train_dataset = ImageFolder(train_path, train_transform)
    val_dataset = ImageFolder(val_path, val_transform)
    
    # Get class names
    class_names = train_dataset.classes
    
    batch_size = 128
    
    train_sampler = DistributedSampler(train_dataset, world_size, rank) if world_size > 1 else None
    train_loader = DataLoader(train_dataset, batch_size, shuffle=(train_sampler is None),
                             sampler=train_sampler, num_workers=8, pin_memory=True,
                             drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    if rank == 0:
        logger.info(f"Classes: {len(class_names)}, Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model = LiTModel(embed_dim=512, use_pretrained=True).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        total = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"Trainable: {trainable:.1f}M / {total:.1f}M params")
    
    # Tokenizer
    tokenizer = get_tokenizer()
    
    # Optimizer - only train text encoder
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = create_optimizer(opt_name, trainable_params, lr=cfg['lr'], weight_decay=cfg['wd'], betas=cfg['betas'])
    
    # LR schedule
    warmup_epochs = 2
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
            images = images.to(device)
            
            # Create text tokens for batch
            texts = [f"a photo of a {class_names[l]}." for l in labels]
            tokens = torch.stack([tokenizer(t) for t in texts]).to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                image_embeds, text_embeds, logit_scale = model(images, tokens)
                
                # Contrastive loss
                logits = logit_scale * image_embeds @ text_embeds.T
                labels_cl = torch.arange(len(images), device=device)
                loss = (F.cross_entropy(logits, labels_cl) + F.cross_entropy(logits.T, labels_cl)) / 2
            
            if torch.isnan(loss) or torch.isinf(loss):
                if rank == 0:
                    logger.warning(f"NaN/Inf loss at epoch {epoch}, skipping batch")
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Evaluate
        if rank == 0 and (epoch % 5 == 0 or epoch == epochs):
            acc = zero_shot_evaluate(model, val_loader, device, tokenizer, class_names)
            results['history'].append({'epoch': epoch, 'loss': avg_loss, 'acc': acc})
            
            marker = '*' if acc > best_acc else ''
            best_acc = max(best_acc, acc)
            logger.info(f"E{epoch:3d}: loss={avg_loss:.4f} acc={acc:.2f}%{marker}")
        elif rank == 0:
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
    parser.add_argument('--epochs', type=int, default=30)
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
    
    logger = setup_logger(output_dir, rank, 'lit')
    
    if args.optimizer == 'all':
        optimizers = ['adamw', 'lion', 'rlo', 'rlo_lambda_a', 'smooth_lifted_rlo']
    else:
        optimizers = [args.optimizer]
    
    results = {}
    for opt in optimizers:
        try:
            results[opt] = train_lit(opt, Path(args.data), output_dir, args.epochs, rank, world_size, device, logger)
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
        logger.info("LiT RESULTS (Zero-shot Accuracy ↑)")
        logger.info("=" * 70)
        for opt in sorted(results.keys(), key=lambda x: -results[x].get('best_acc', 0)):
            acc = results[opt].get('best_acc', 0)
            logger.info(f"  {opt}: {acc:.2f}%")
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
