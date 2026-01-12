#!/usr/bin/env python3
"""
Section 4.2: Vision-Language Contrastive Learning (LiT)
=======================================================
LiT: Locked-image Text Tuning
- Freeze pretrained image encoder
- Train text encoder with contrastive loss
"""

import sys
def _patch_dill():
    try:
        import dill
        if not hasattr(dill, 'extend'):
            dill.extend = lambda use_dill=True: None
    except ImportError:
        pass
_patch_dill()

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
from torch.optim import Optimizer
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import warnings
warnings.filterwarnings('ignore')


def setup_logger(output_dir, rank, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(output_dir / f"{name}_rank{rank}.log", mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
    logger.addHandler(fh)
    if rank == 0:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(ch)
    return logger


# =============================================================================
# Optimizers
# =============================================================================
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
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p)
                m = state['m']
                beta1, beta2 = group['betas']
                p.add_((beta1 * m + (1 - beta1) * p.grad).sign_(), alpha=-group['lr'])
                m.mul_(beta2).add_(p.grad, alpha=1 - beta2)


class RLO(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, belief_coef=0.1, eps=1e-8):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, belief_coef=belief_coef, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                m, g = state["m"], p.grad
                beta1, beta2 = group["betas"]
                c = beta1 * m + (1 - beta1) * g
                delta = g - m
                update = c.sign() + group["belief_coef"] * (delta / delta.norm().clamp(min=group["eps"]))
                p.add_(update, alpha=-group["lr"])
                m.mul_(beta2).add_(g, alpha=1 - beta2)


class SmoothLiftedRLO(Optimizer):
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.99, eta=0.3,
                 weight_decay=0.0, lambda_b=0.1, eps=1e-8, gamma=5.0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eta=eta,
                        weight_decay=weight_decay, lambda_b=lambda_b, eps=eps, gamma=gamma)
        super().__init__(params, defaults)
        self.sqrt_dim = math.sqrt(sum(p.numel() for g in self.param_groups for p in g["params"]))

    @torch.no_grad()
    def step(self, closure=None):
        all_s, all_b, all_p = [], [], []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                g, m = p.grad, state["m"]
                c = group["beta1"] * m + (1 - group["beta1"]) * g
                s = torch.tanh(group["gamma"] * c)
                delta = g - m
                b = group["lambda_b"] * (delta / delta.norm().clamp(min=group["eps"]))
                all_s.append(s); all_b.append(b); all_p.append((p, group))
        if not all_p:
            return
        scale = self.sqrt_dim / sum((x * x).sum() for x in all_s).sqrt().clamp(min=1e-8)
        for (p, group), s, b in zip(all_p, all_s, all_b):
            state = self.state[p]
            if group["weight_decay"] != 0:
                p.mul_(1 - group["lr"] * group["weight_decay"])
            d = scale * s + b
            state["v"].mul_(1 - group["eta"]).add_(d, alpha=group["eta"])
            p.add_(state["v"], alpha=-group["lr"])
            state["m"].mul_(group["beta2"]).add_(p.grad, alpha=1 - group["beta2"])


# =============================================================================
# Models
# =============================================================================
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
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
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    def __init__(self, dim, num_heads, causal=False):
        super().__init__()
        self.causal = causal
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        mask = None
        if self.causal:
            N = x.size(1)
            mask = torch.tril(torch.ones(N, N, device=x.device)).unsqueeze(0).unsqueeze(0)
        x = x + self.attn(self.norm1(x), mask)
        return x + self.mlp(self.norm2(x))


class ViTImageEncoder(nn.Module):
    """ViT-B/32 Image Encoder"""
    def __init__(self, embed_dim=768, depth=12, num_heads=12, out_dim=512):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, 32, 32)  # B/32
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 50, embed_dim))  # 224/32=7, 7x7+1=50
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, out_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.proj(self.norm(x[:, 0]))


class TextTransformer(nn.Module):
    """Text Transformer Encoder"""
    def __init__(self, vocab_size=49408, embed_dim=512, depth=6, num_heads=8, max_len=77, out_dim=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, causal=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, tokens):
        B, T = tokens.shape
        x = self.token_emb(tokens) + self.pos_emb(torch.arange(T, device=tokens.device))
        for blk in self.blocks:
            x = blk(x)
        # Use last token (EOS position) for pooling
        x = self.norm(x[torch.arange(B), tokens.argmax(dim=-1)])  # Approximate EOS
        return self.proj(x)


class LiTModel(nn.Module):
    """LiT: Locked-image Text tuning"""
    def __init__(self, embed_dim=512):
        super().__init__()
        self.image_encoder = ViTImageEncoder(out_dim=embed_dim)
        self.text_encoder = TextTransformer(out_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # Freeze image encoder
        for p in self.image_encoder.parameters():
            p.requires_grad = False

    def forward(self, images, tokens):
        with torch.no_grad():
            image_features = self.image_encoder(images)
            image_features = F.normalize(image_features, dim=-1)
        
        text_features = self.text_encoder(tokens)
        text_features = F.normalize(text_features, dim=-1)
        
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text


# =============================================================================
# Dataset
# =============================================================================
class ImageTextDataset(Dataset):
    """Simple ImageNet with synthetic text (class names as proxy)"""
    def __init__(self, image_folder, transform=None, max_text_len=77):
        self.image_folder = ImageFolder(image_folder, transform)
        self.max_text_len = max_text_len
        # Simple tokenizer: character level
        self.vocab = {chr(i): i for i in range(256)}
        self.vocab_size = 256

    def tokenize(self, text):
        tokens = [self.vocab.get(c, 0) for c in text[:self.max_text_len - 1]]
        tokens = tokens + [0] * (self.max_text_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        image, label = self.image_folder[idx]
        # Use class folder name as text
        class_name = self.image_folder.classes[label]
        text = f"a photo of {class_name.replace('_', ' ')}"
        tokens = self.tokenize(text)
        return image, tokens


# =============================================================================
# Config
# =============================================================================
CONFIGS = {
    'adamw': {'lr': 1e-3, 'wd': 0.1, 'betas': (0.9, 0.999)},
    'lion': {'lr': 1e-4, 'wd': 1.0, 'betas': (0.9, 0.99)},
    'rlo': {'lr': 1e-4, 'wd': 1.0, 'betas': (0.9, 0.99), 'belief_coef': 0.1},
    'smooth_lifted_rlo': {'lr': 1e-4, 'wd': 1.0, 'lambda_b': 0.1, 'eta': 0.3},
}


def create_optimizer(model, name, cfg):
    # Only optimize text encoder (image encoder is frozen)
    params = [p for p in model.parameters() if p.requires_grad]
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['wd'], betas=cfg.get('betas', (0.9, 0.999)))
    elif name == 'lion':
        return Lion(params, lr=cfg['lr'], weight_decay=cfg['wd'], betas=cfg.get('betas', (0.9, 0.99)))
    elif name == 'rlo':
        return RLO(params, lr=cfg['lr'], weight_decay=cfg['wd'], betas=cfg.get('betas', (0.9, 0.99)),
                   belief_coef=cfg.get('belief_coef', 0.1))
    elif name == 'smooth_lifted_rlo':
        return SmoothLiftedRLO(params, lr=cfg['lr'], weight_decay=cfg['wd'],
                              lambda_b=cfg.get('lambda_b', 0.1), eta=cfg.get('eta', 0.3))
    raise ValueError(f"Unknown: {name}")


# =============================================================================
# Training
# =============================================================================
def contrastive_loss(logits_per_image, logits_per_text):
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


def train_lit(opt_name, data_path, output_dir, epochs, rank, world_size, local_rank, logger):
    device = torch.device(f'cuda:{local_rank}')
    cfg = CONFIGS[opt_name]
    global_batch = 256
    per_gpu = global_batch // world_size

    logger.info("=" * 70)
    logger.info(f"LiT Training: {opt_name}")
    logger.info(f"LR: {cfg['lr']} | WD: {cfg['wd']} | Epochs: {epochs}")
    logger.info("=" * 70)

    # Data
    transform = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    dataset = ImageTextDataset(data_path / 'train', transform)
    sampler = DistributedSampler(dataset, world_size, rank) if world_size > 1 else None
    train_loader = DataLoader(dataset, per_gpu, shuffle=(sampler is None), sampler=sampler,
                             num_workers=2, pin_memory=True, drop_last=True, persistent_workers=False)

    # Model
    model = LiTModel().to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable/1e6:.1f}M")

    optimizer = create_optimizer(model, opt_name, cfg)
    scaler = torch.amp.GradScaler('cuda')

    # Scheduler
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * 2
    
    def get_lr(step):
        if step < warmup_steps:
            return cfg['lr'] * step / max(1, warmup_steps)
        return cfg['lr'] * 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

    history = {'train_loss': []}
    best_loss = float('inf')
    step = 0

    for epoch in range(epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        t0 = time.time()

        for images, tokens in train_loader:
            images = images.to(device)
            tokens = tokens.to(device)
            
            lr = get_lr(step)
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
            
            loss_sum += loss.item()
            step += 1

        avg_loss = loss_sum / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        is_best = avg_loss < best_loss
        best_loss = min(avg_loss, best_loss)
        
        logger.info(f"E{epoch+1:3d}: loss={avg_loss:.4f}{'*' if is_best else ''}, {time.time()-t0:.0f}s")

        if is_best and rank == 0:
            base_model = model.module if hasattr(model, 'module') else model
            torch.save({'model': base_model.state_dict(), 'loss': best_loss},
                      output_dir / f"lit_{opt_name}_best.pt")

        if (epoch + 1) % 10 == 0:
            gc.collect(); torch.cuda.empty_cache()

    # Save results
    if rank == 0:
        with open(output_dir / f"lit_{opt_name}_results.json", 'w') as f:
            json.dump({'optimizer': opt_name, 'config': cfg, 'best_loss': best_loss,
                      'history': history}, f, indent=2)
        logger.info(f"Done. Best loss: {best_loss:.4f}")

    return best_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='all')
    parser.add_argument('--data', default='/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder')
    parser.add_argument('--output', default='/blue/wdixon/wang.yixuan/rlo_results/lit')
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    if 'RANK' in os.environ:
        rank, world_size, local_rank = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl')
    else:
        rank, world_size, local_rank = 0, 1, 0

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = setup_logger(output_dir, rank, "lit")

    opts = list(CONFIGS.keys()) if args.optimizer == 'all' else [args.optimizer]
    results = {}

    for opt in opts:
        try:
            gc.collect(); torch.cuda.empty_cache()
            results[opt] = train_lit(opt, Path(args.data), output_dir, args.epochs, rank, world_size, local_rank, logger)
        except Exception as e:
            logger.error(f"Error {opt}: {e}")
            import traceback; logger.error(traceback.format_exc())

    if rank == 0 and results:
        logger.info("=" * 70)
        logger.info("LiT Results (loss, lower=better):")
        for opt, loss in sorted(results.items(), key=lambda x: x[1]):
            logger.info(f"  {opt}: {loss:.4f}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
