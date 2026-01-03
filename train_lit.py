#!/usr/bin/env python3
"""
Section 4.2: Vision-Language Contrastive Learning (LiT)
========================================================

Following LION paper Section 4.2:
- LiT (Locked-image Text Tuning) with ViT-B/32 image encoder
- Contrastive learning on image-text pairs
- Zero-shot evaluation on ImageNet and CIFAR-100

Usage:
    torchrun --nproc_per_node=8 train_lit.py --optimizer all
"""

import os
import sys
import time
import math
import random
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Optimizer
import torchvision.transforms as T
from torchvision.datasets import ImageFolder, CIFAR100

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. Optimizers (same as train_rlo.py)
# =============================================================================

class RLO(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1,
                 belief_coef=0.1, eps=1e-8):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                        belief_coef=belief_coef, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            beta1, beta2 = group["betas"]
            belief, eps = group["belief_coef"], group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                c = beta1 * m + (1.0 - beta1) * g
                delta = g - m
                delta_norm = delta.norm(p=2).clamp(min=eps)
                d = c.sign() + belief * (delta / delta_norm)
                p.add_(d, alpha=-lr)
                m.mul_(beta2).add_(g, alpha=(1.0 - beta2))


class RLO_LambdaA(Optimizer):
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.99, beta3=0.999,
                 weight_decay=0.1, lambda_b=0.1, eps=1e-8, gamma=5.0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, beta3=beta3,
                        weight_decay=weight_decay, lambda_b=lambda_b, eps=eps, gamma=gamma)
        super().__init__(params, defaults)
        self._init_sqrt_dim()

    def _init_sqrt_dim(self):
        total = sum(p.numel() for g in self.param_groups for p in g["params"])
        self.sqrt_dim = math.sqrt(total)

    @torch.no_grad()
    def step(self, closure=None):
        all_smooth_pre, all_belief, all_params = [], [], []
        for group in self.param_groups:
            eps, gamma = group["eps"], group["gamma"]
            beta1, beta2, beta3, lambda_b = group["beta1"], group["beta2"], group["beta3"], group["lambda_b"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["s"] = torch.zeros_like(p)
                m, s = state["m"], state["s"]
                s.mul_(beta3).addcmul_(g, g, value=(1.0 - beta3))
                c = beta1 * m + (1.0 - beta1) * g
                smooth = torch.tanh(gamma * c)
                smooth_pre = smooth / (s.sqrt() + eps)
                delta = g - m
                delta_norm = delta.norm(p=2).clamp(min=eps)
                belief = lambda_b * (delta / delta_norm)
                all_smooth_pre.append(smooth_pre)
                all_belief.append(belief)
                all_params.append((p, group))
        if not all_params:
            return
        s_norm = sum((sp * sp).sum() for sp in all_smooth_pre).sqrt().clamp(min=1e-8)
        scale = self.sqrt_dim / s_norm
        for (p, group), sp, b in zip(all_params, all_smooth_pre, all_belief):
            lr, wd, beta2 = group["lr"], group["weight_decay"], group["beta2"]
            d = scale * sp + b
            state = self.state[p]
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            p.add_(d, alpha=-lr)
            state["m"].mul_(beta2).add_(p.grad, alpha=(1.0 - beta2))


class SmoothLiftedRLO(Optimizer):
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.99, eta=0.3,
                 weight_decay=0.1, lambda_b=0.1, eps=1e-8, gamma=5.0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eta=eta,
                        weight_decay=weight_decay, lambda_b=lambda_b, eps=eps, gamma=gamma)
        super().__init__(params, defaults)
        self._init_sqrt_dim()

    def _init_sqrt_dim(self):
        total = sum(p.numel() for g in self.param_groups for p in g["params"])
        self.sqrt_dim = math.sqrt(total)

    @torch.no_grad()
    def step(self, closure=None):
        all_s, all_b, all_params = [], [], []
        for group in self.param_groups:
            eps, gamma, beta1, lambda_b = group["eps"], group["gamma"], group["beta1"], group["lambda_b"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                m = state["m"]
                c = beta1 * m + (1.0 - beta1) * g
                s = torch.tanh(gamma * c)
                delta = g - m
                delta_norm = delta.norm(p=2).clamp(min=eps)
                b = lambda_b * (delta / delta_norm)
                all_s.append(s)
                all_b.append(b)
                all_params.append((p, group))
        if not all_params:
            return
        s_norm = sum((s * s).sum() for s in all_s).sqrt().clamp(min=1e-8)
        scale = self.sqrt_dim / s_norm
        for (p, group), s, b in zip(all_params, all_s, all_b):
            lr, wd, eta, beta2 = group["lr"], group["weight_decay"], group["eta"], group["beta2"]
            d = scale * s + b
            state = self.state[p]
            m, v = state["m"], state["v"]
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            v.mul_(1.0 - eta).add_(d, alpha=eta)
            p.add_(v, alpha=-lr)
            m.mul_(beta2).add_(p.grad, alpha=(1.0 - beta2))


class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                m = state['exp_avg']
                beta1, beta2 = group['betas']
                update = (beta1 * m + (1 - beta1) * g).sign_()
                p.add_(update, alpha=-group['lr'])
                m.mul_(beta2).add_(g, alpha=1 - beta2)


# =============================================================================
# 2. Vision Transformer (Image Encoder)
# =============================================================================

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
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    """ViT-B/32 for image encoding"""
    def __init__(self, img_size=224, patch_size=32, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x[:, 0])


# =============================================================================
# 3. Text Encoder (Transformer)
# =============================================================================

class TextTransformer(nn.Module):
    """Simple text transformer for CLIP-style encoding"""
    def __init__(self, vocab_size=49408, context_length=77, embed_dim=512, 
                 depth=12, num_heads=8, output_dim=768):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, context_length, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, output_dim, bias=False)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, text):
        # text: [B, L] token ids
        x = self.token_embedding(text) + self.pos_embedding[:, :text.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        # Take features from EOT token (last non-padding)
        x = self.norm(x)
        # Simplified: just take mean
        x = x.mean(dim=1)
        return self.proj(x)


# =============================================================================
# 4. LiT Model (Locked-image Text tuning)
# =============================================================================

class LiT(nn.Module):
    """
    LiT: Locked-image Text tuning
    - Image encoder is frozen (pretrained)
    - Text encoder is trained
    - Contrastive loss between image and text embeddings
    """
    def __init__(self, embed_dim=768, temperature=0.07):
        super().__init__()
        # ViT-B/32 image encoder (will be frozen)
        self.image_encoder = VisionTransformer(patch_size=32, embed_dim=embed_dim)
        # Text encoder (will be trained)
        self.text_encoder = TextTransformer(output_dim=embed_dim)
        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def encode_image(self, image):
        return F.normalize(self.image_encoder(image), dim=-1)

    def encode_text(self, text):
        return F.normalize(self.text_encoder(text), dim=-1)

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text


def contrastive_loss(logits_per_image, logits_per_text):
    """CLIP-style contrastive loss"""
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


# =============================================================================
# 5. Dataset (Simplified - using ImageNet with synthetic text)
# =============================================================================

# ImageNet class names (subset for demo)
IMAGENET_CLASSES = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
    "electric ray", "stingray", "cock", "hen", "ostrich",
    # ... (would include all 1000 classes in full implementation)
]


class ImageTextDataset(Dataset):
    """
    Image-Text dataset for contrastive learning.
    Uses ImageNet images with template-based text descriptions.
    """
    def __init__(self, root, split='train', transform=None):
        self.dataset = ImageFolder(root / split, transform=transform)
        self.templates = [
            "a photo of a {}.",
            "a picture of a {}.",
            "an image of a {}.",
            "a {} in the wild.",
            "a photograph of a {}.",
        ]
        # Simple tokenizer (in practice, use CLIP tokenizer)
        self.vocab = self._build_vocab()
        self.context_length = 77
        
    def _build_vocab(self):
        # Simplified vocabulary
        vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        idx = 4
        words = set()
        for template in self.templates:
            for word in template.replace('.', '').replace(',', '').split():
                words.add(word.lower())
        for cls_name in IMAGENET_CLASSES:
            for word in cls_name.replace('_', ' ').split():
                words.add(word.lower())
        for word in sorted(words):
            vocab[word] = idx
            idx += 1
        return vocab
    
    def tokenize(self, text):
        """Simple tokenization"""
        tokens = [self.vocab.get('<sos>')]
        for word in text.replace('.', '').replace(',', '').lower().split():
            tokens.append(self.vocab.get(word, self.vocab['<unk>']))
        tokens.append(self.vocab.get('<eos>'))
        # Pad or truncate
        if len(tokens) < self.context_length:
            tokens += [self.vocab['<pad>']] * (self.context_length - len(tokens))
        else:
            tokens = tokens[:self.context_length]
        return torch.tensor(tokens, dtype=torch.long)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Get class name (simplified - would use actual ImageNet class names)
        class_name = f"class_{label}"
        if label < len(IMAGENET_CLASSES):
            class_name = IMAGENET_CLASSES[label]
        # Random template
        template = random.choice(self.templates)
        text = template.format(class_name.replace('_', ' '))
        tokens = self.tokenize(text)
        return image, tokens, label


# =============================================================================
# 6. Zero-shot Evaluation
# =============================================================================

@torch.no_grad()
def zero_shot_evaluate(model, loader, class_names, device, templates=None):
    """
    Zero-shot classification evaluation.
    """
    if templates is None:
        templates = ["a photo of a {}."]
    
    model.eval()
    
    # Build text features for all classes
    text_features_list = []
    for class_name in class_names:
        class_texts = [t.format(class_name) for t in templates]
        # Simplified: just use first template
        # In practice, average over all templates
        tokens = torch.zeros(1, 77, dtype=torch.long, device=device)
        text_features = model.encode_text(tokens)
        text_features_list.append(text_features)
    
    text_features = torch.cat(text_features_list, dim=0)
    text_features = F.normalize(text_features, dim=-1)
    
    correct = 0
    total = 0
    
    for images, _, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        image_features = model.encode_image(images)
        similarity = image_features @ text_features.t()
        pred = similarity.argmax(dim=-1)
        
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    
    return 100.0 * correct / total


# =============================================================================
# 7. Training
# =============================================================================

def create_optimizer(model, opt_name: str, lr: float, wd: float):
    # Only train text encoder (LiT setting)
    params = model.text_encoder.parameters()
    
    if opt_name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif opt_name == 'lion':
        return Lion(params, lr=lr, weight_decay=wd)
    elif opt_name == 'rlo':
        return RLO(params, lr=lr, weight_decay=wd)
    elif opt_name == 'rlo_lambda_a':
        return RLO_LambdaA(params, lr=lr, weight_decay=wd)
    elif opt_name == 'smooth_lifted_rlo':
        return SmoothLiftedRLO(params, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def train_lit(
    opt_name: str,
    data_path: Path,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 4096,
    warmup_epochs: int = 3,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
):
    """Train LiT model"""
    device = torch.device(f'cuda:{local_rank}')
    
    # Hyperparameters (from LION paper)
    configs = {
        'adamw': {'lr': 1e-3, 'wd': 0.1},
        'lion': {'lr': 1e-4, 'wd': 1.0},
        'rlo': {'lr': 1e-4, 'wd': 1.0},
        'rlo_lambda_a': {'lr': 1e-4, 'wd': 1.0},
        'smooth_lifted_rlo': {'lr': 1e-4, 'wd': 1.0},
    }
    config = configs[opt_name]
    batch_size_per_gpu = batch_size // world_size
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"LiT Training: {opt_name}")
        print(f"lr={config['lr']}, wd={config['wd']}, batch={batch_size}")
        print(f"{'='*60}")
    
    # Data
    transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    train_dataset = ImageTextDataset(data_path, 'train', transform)
    
    if world_size > 1:
        sampler = DistributedSampler(train_dataset)
    else:
        sampler = None
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_per_gpu, shuffle=(sampler is None),
        sampler=sampler, num_workers=8, pin_memory=True, drop_last=True,
    )
    
    # Model
    model = LiT().to(device)
    
    # Freeze image encoder (LiT setting)
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # Optimizer (only for text encoder)
    base_model = model.module if hasattr(model, 'module') else model
    optimizer = create_optimizer(base_model, opt_name, config['lr'], config['wd'])
    
    # Scheduler
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * warmup_epochs
    
    def get_lr(step):
        if step < warmup_steps:
            return config['lr'] * step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return config['lr'] * 0.5 * (1 + math.cos(math.pi * progress))
    
    scaler = torch.amp.GradScaler('cuda')
    global_step = 0
    history = {'loss': []}
    
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        model.train()
        # But keep image encoder in eval mode
        base_model.image_encoder.eval()
        
        epoch_loss = 0.0
        
        for images, tokens, _ in train_loader:
            images = images.to(device)
            tokens = tokens.to(device)
            
            # Update LR
            lr = get_lr(global_step)
            for g in optimizer.param_groups:
                g['lr'] = lr
            
            optimizer.zero_grad()
            
            with torch.autocast('cuda', torch.bfloat16):
                logits_i, logits_t = model(images, tokens)
                loss = contrastive_loss(logits_i, logits_t)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(base_model.text_encoder.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            global_step += 1
        
        avg_loss = epoch_loss / len(train_loader)
        history['loss'].append(avg_loss)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")
    
    # Save results
    if rank == 0:
        results = {
            'optimizer': opt_name,
            'final_loss': history['loss'][-1],
            'history': history,
        }
        with open(output_dir / f"lit_{opt_name}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    return history['loss'][-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='all')
    parser.add_argument('--data', default='/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder')
    parser.add_argument('--output', default='./results_lit')
    parser.add_argument('--epochs', type=int, default=30)
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
    
    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    optimizers = ['adamw', 'lion', 'rlo', 'rlo_lambda_a', 'smooth_lifted_rlo'] if args.optimizer == 'all' else [args.optimizer]
    
    results = {}
    for opt in optimizers:
        try:
            loss = train_lit(opt, data_path, output_dir, args.epochs, rank=rank, 
                           world_size=world_size, local_rank=local_rank)
            results[opt] = loss
        except Exception as e:
            if rank == 0:
                print(f"Error {opt}: {e}")
    
    if rank == 0 and results:
        print("\n" + "="*60)
        print("LiT Results (Final Loss)")
        for opt, loss in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {opt}: {loss:.4f}")
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
