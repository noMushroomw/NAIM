#!/usr/bin/env python3
"""
Language Modeling Experiment - Following Lion Paper Figure 7
GPT-2 style, records LOG PERPLEXITY (like Figure 7)

Lion Paper configs (LM):
- AdamW: lr=3e-3, β=(0.9, 0.99)
- Lion: lr=3e-4 (0.1x), β=(0.95, 0.98) - DIFFERENT BETAS FOR LM!
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


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""
    def __init__(self, dim, num_heads, max_len):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Register causal mask
        self.register_buffer('mask', torch.tril(torch.ones(max_len, max_len)))
    
    def forward(self, x):
        B, T, C = x.shape
        
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Use Flash Attention with causal mask
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x.transpose(1, 2).reshape(B, T, C)
        
        return self.proj(x)


class TransformerBlock(nn.Module):
    """GPT-2 style transformer block."""
    def __init__(self, dim, num_heads, max_len, mlp_ratio=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, max_len)
        self.ln2 = nn.LayerNorm(dim)
        
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2(nn.Module):
    """GPT-2 117M style model."""
    def __init__(self, vocab_size=50257, dim=768, num_heads=12, num_layers=12, max_len=1024):
        super().__init__()
        self.max_len = max_len
        
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, max_len) for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        # Weight tying
        self.token_embed.weight = self.head.weight
        
        # Initialize
        self.apply(self._init_weights)
        
        # Special initialization for residual projections
        for name, p in self.named_parameters():
            if name.endswith('proj.weight') or name.endswith('mlp.2.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        x = self.token_embed(idx) + self.pos_embed(torch.arange(T, device=idx.device))
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class TextDataset(Dataset):
    """Simple text dataset."""
    def __init__(self, data_path, seq_len=1024, split='train'):
        self.seq_len = seq_len
        data_path = Path(data_path)
        
        # Try to load pre-tokenized data
        token_file = data_path / f'{split}_tokens.bin'
        if token_file.exists():
            self.tokens = np.memmap(token_file, dtype=np.uint16, mode='r')
            self.vocab_size = 50257
        else:
            # Fall back to character-level
            text_file = data_path / f'{split}.txt'
            if text_file.exists():
                with open(text_file, 'r', errors='ignore') as f:
                    text = f.read()
                chars = sorted(set(text))
                self.vocab_size = len(chars)
                char_to_idx = {c: i for i, c in enumerate(chars)}
                self.tokens = np.array([char_to_idx.get(c, 0) for c in text], dtype=np.int64)
            else:
                # Random tokens for testing
                self.vocab_size = 50257
                self.tokens = np.random.randint(0, 50257, 10_000_000, dtype=np.int64)
        
        self.num_samples = (len(self.tokens) - 1) // seq_len
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start:start + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


@torch.no_grad()
def evaluate(model, loader, device, max_batches=50):
    """Compute perplexity on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        
        x = x.to(device)
        y = y.to(device)
        
        with torch.autocast('cuda', torch.bfloat16):
            _, loss = model(x, y)
        
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)
    
    return perplexity


# ============= Lion Paper Configs (LM) =============
# CRITICAL: LM uses different betas for Lion! β=(0.95, 0.98)
CONFIGS = {
    'adamw': {'lr': 3e-3, 'wd': 0.0, 'betas': (0.9, 0.99)},
    'lion': {'lr': 3e-4, 'wd': 0.0, 'betas': (0.95, 0.98)},  # Different betas!
    'rlo': {'lr': 3e-4, 'wd': 0.0, 'betas': (0.95, 0.98), 'belief_coef': 0.1},
    'rlo_lambda_a': {'lr': 3e-4, 'wd': 0.0, 'beta1': 0.95, 'beta2': 0.98, 'lambda_b': 0.1},
    'smooth_lifted_rlo': {'lr': 3e-4, 'wd': 0.0, 'beta1': 0.95, 'beta2': 0.98, 
                          'lambda_b': 0.1, 'eta': 0.3},
}


def train_lm(opt_name, data_path, output_path, total_steps,
            rank, world_size, local_rank, logger):
    device = torch.device(f'cuda:{local_rank}')
    cfg = CONFIGS[opt_name]
    
    # Batch size: global 512, seq_len 1024
    global_batch = 512
    seq_len = 1024
    per_gpu_batch = global_batch // world_size
    
    # Use gradient accumulation to avoid OOM
    micro_batch = min(per_gpu_batch, 8)  # 8 sequences per micro-batch
    accumulation_steps = per_gpu_batch // micro_batch
    
    logger.info("=" * 70)
    logger.info(f"Language Modeling | {opt_name}")
    logger.info(f"LR={cfg['lr']}, Betas={cfg.get('betas', (cfg.get('beta1'), cfg.get('beta2')))}")
    logger.info(f"Global batch={global_batch}, Steps={total_steps}")
    logger.info(f"Micro batch={micro_batch}, Accum={accumulation_steps}")
    logger.info("=" * 70)
    
    # Data
    train_dataset = TextDataset(data_path, seq_len, 'train')
    val_dataset = TextDataset(data_path, seq_len, 'valid') if (Path(data_path) / 'valid.txt').exists() else train_dataset
    
    train_sampler = DistributedSampler(train_dataset, world_size, rank) if world_size > 1 else None
    train_loader = DataLoader(train_dataset, micro_batch,
                             shuffle=(train_sampler is None),
                             sampler=train_sampler,
                             num_workers=8, pin_memory=True, drop_last=True,
                             persistent_workers=True, prefetch_factor=4)
    
    val_loader = DataLoader(val_dataset, micro_batch * 2,
                           shuffle=False, num_workers=4, pin_memory=True,
                           prefetch_factor=4)
    
    logger.info(f"Train samples: {len(train_dataset)}, Vocab: {train_dataset.vocab_size}")
    
    # Model
    model = GPT2(vocab_size=train_dataset.vocab_size, max_len=seq_len).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    base_model = model.module if hasattr(model, 'module') else model
    num_params = base_model.num_params() / 1e6
    logger.info(f"Parameters: {num_params:.1f}M")
    
    # Optimizer with weight decay on non-bias, non-layernorm params
    decay_params = []
    no_decay_params = []
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            if 'bias' in name or 'ln' in name or 'pos' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': cfg['wd']},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = create_optimizer(None, opt_name, cfg, param_groups)
    scaler = torch.amp.GradScaler('cuda')
    
    # LR schedule
    warmup_steps = 2000
    
    def get_lr(step):
        if step < warmup_steps:
            return cfg['lr'] * step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return cfg['lr'] * 0.1 + 0.9 * cfg['lr'] * 0.5 * (1 + math.cos(math.pi * progress))
    
    # Training
    history = {'loss': [], 'ppl': [], 'step': []}
    best_ppl = float('inf')
    
    global_step = 0
    epoch = 0
    train_iter = iter(train_loader)
    
    running_loss = 0.0
    t0 = time.time()
    
    while global_step < total_steps:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        for _ in range(accumulation_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                epoch += 1
                if train_sampler:
                    train_sampler.set_epoch(epoch)
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            
            x = x.to(device)
            y = y.to(device)
            
            with torch.autocast('cuda', torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            running_loss += loss.item() * accumulation_steps
        
        # Update LR
        lr = get_lr(global_step)
        for g in optimizer.param_groups:
            g['lr'] = lr
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        global_step += 1
        
        # Log every 100 steps
        if global_step % 100 == 0:
            tokens_per_sec = (100 * global_batch * seq_len) / (time.time() - t0)
            logger.info(f"Step {global_step:5d}/{total_steps}: loss={running_loss/100:.4f} "
                       f"lr={lr:.2e} {tokens_per_sec/1e6:.2f}M tok/s")
            running_loss = 0.0
            t0 = time.time()
        
        # Evaluate every 2000 steps
        if global_step % 2000 == 0 or global_step == total_steps:
            base_model = model.module if hasattr(model, 'module') else model
            ppl = evaluate(base_model, val_loader, device)
            
            is_best = ppl < best_ppl
            best_ppl = min(ppl, best_ppl)
            
            history['step'].append(global_step)
            history['ppl'].append(ppl)
            
            logger.info(f"Step {global_step}: PPL={ppl:.2f}{'*' if is_best else ''}")
            
            if is_best and rank == 0:
                torch.save({
                    'model': base_model.state_dict(),
                    'ppl': best_ppl,
                    'step': global_step
                }, output_path / f"lm_{opt_name}_best.pt")
    
    # Save results
    if rank == 0:
        results = {
            'optimizer': opt_name,
            'config': cfg,
            'best_ppl': best_ppl,
            'history': history
        }
        with open(output_path / f"lm_{opt_name}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Final: Best PPL = {best_ppl:.2f}")
    
    return best_ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='all')
    parser.add_argument('--data', default='/blue/wdixon/wang.yixuan/rlo_experiments/data/wikitext')
    parser.add_argument('--output', default='/blue/wdixon/wang.yixuan/rlo_experiments/lm')
    parser.add_argument('--steps', type=int, default=50000)  # Reduced from 100k
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
    
    logger = setup_logger(output_path, rank, "lm")
    
    # Run experiments
    optimizers = list(CONFIGS.keys()) if args.optimizer == 'all' else [args.optimizer]
    results = {}
    
    for opt in optimizers:
        try:
            gc.collect()
            torch.cuda.empty_cache()
            results[opt] = train_lm(opt, Path(args.data), output_path,
                                   args.steps, rank, world_size, local_rank, logger)
        except Exception as e:
            logger.error(f"Error with {opt}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Summary
    if rank == 0 and results:
        logger.info("=" * 70)
        logger.info("Language Model Results (Perplexity, lower is better):")
        for opt, ppl in sorted(results.items(), key=lambda x: x[1]):
            logger.info(f"  {opt}: {ppl:.2f}")
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
