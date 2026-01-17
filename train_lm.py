#!/usr/bin/env python3
"""
Language Modeling Experiment - Following Lion Paper Figure 7
GPT-2 style, records PERPLEXITY (lower is better)

Lion Paper configs (LM):
- AdamW: lr=3e-3, β=(0.9, 0.99)
- Lion: lr=3e-4 (0.1x), β=(0.95, 0.98) - DIFFERENT BETAS FOR LM!
"""
import os, gc, time, math, json, logging, argparse, sys
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

try:
    from optimizers import create_optimizer
except ImportError:
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


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""
    def __init__(self, dim, num_heads, max_len):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(x.transpose(1, 2).reshape(B, T, C))


class TransformerBlock(nn.Module):
    """GPT-2 style transformer block."""
    def __init__(self, dim, num_heads, max_len, mlp_ratio=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, max_len)
        self.ln2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT2(nn.Module):
    """GPT-2 style language model."""
    def __init__(self, vocab_size=50257, dim=768, num_heads=12, num_layers=12, max_len=1024):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, dim))
        self.blocks = nn.ModuleList([TransformerBlock(dim, num_heads, max_len) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.token_embed.weight
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, idx):
        B, T = idx.shape
        x = self.token_embed(idx) + self.pos_embed[:, :T]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


class TextDataset(Dataset):
    """Simple text dataset for language modeling."""
    def __init__(self, data_path, seq_len=1024, tokenizer=None):
        self.seq_len = seq_len
        
        # Try to load tokenizer
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            try:
                from transformers import GPT2Tokenizer
                self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            except Exception:
                self.tokenizer = None
        
        # Load text
        if isinstance(data_path, str):
            data_path = Path(data_path)
        
        # Find text file
        if data_path.is_file():
            text_file = data_path
        elif (data_path / 'train.txt').exists():
            text_file = data_path / 'train.txt'
        elif (data_path / 'wiki.train.raw').exists():
            text_file = data_path / 'wiki.train.raw'
        else:
            # Try to find any .txt file
            txt_files = list(data_path.glob('*.txt'))
            if txt_files:
                text_file = txt_files[0]
            else:
                raise FileNotFoundError(f"No text file found in {data_path}")
        
        print(f"Loading text from: {text_file}")
        
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize
        if self.tokenizer:
            self.tokens = self.tokenizer.encode(text)
        else:
            # Simple byte-level tokenization
            self.tokens = [ord(c) % 50257 for c in text]
        
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        print(f"Total tokens: {len(self.tokens):,}")
    
    def __len__(self):
        return max(1, (len(self.tokens) - 1) // self.seq_len)
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = min(start + self.seq_len + 1, len(self.tokens))
        chunk = self.tokens[start:end]
        
        # Pad if necessary
        if len(chunk) < self.seq_len + 1:
            chunk = F.pad(chunk, (0, self.seq_len + 1 - len(chunk)))
        
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


@torch.no_grad()
def evaluate(model, loader, device):
    """Compute perplexity on validation set."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum')
        total_loss += loss.item()
        total_tokens += y.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20))  # Clamp to avoid overflow
    return perplexity


def train_lm(optimizer_name, data_path, output_dir, total_steps, rank, world_size, device, logger):
    """Train language model with specified optimizer."""
    
    # Hyperparameters from Lion paper
    # IMPORTANT: Lion uses different betas for LM!
    configs = {
        'adamw': {'lr': 3e-3, 'wd': 0.1, 'betas': (0.9, 0.99)},
        'lion': {'lr': 3e-4, 'wd': 1.0, 'betas': (0.95, 0.98)},  # Different betas!
        'rlo': {'lr': 3e-4, 'wd': 1.0, 'betas': (0.95, 0.98)},
        'rlo_lambda_a': {'lr': 3e-4, 'wd': 1.0, 'betas': (0.95, 0.98)},
        'smooth_lifted_rlo': {'lr': 3e-4, 'wd': 1.0, 'betas': (0.95, 0.98)},
    }
    
    cfg = configs.get(optimizer_name, configs['adamw'])
    
    if rank == 0:
        logger.info("=" * 70)
        logger.info(f"Language Model Training: {optimizer_name}")
        logger.info(f"LR={cfg['lr']}, WD={cfg['wd']}, Betas={cfg['betas']}")
        logger.info("=" * 70)
    
    # Try to load GPT2 tokenizer
    tokenizer = None
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if rank == 0:
            logger.info("Using GPT2 tokenizer")
    except Exception as e:
        if rank == 0:
            logger.warning(f"Could not load GPT2 tokenizer: {e}")
            logger.info("Using byte-level tokenization")
    
    # Data
    seq_len = 1024
    batch_size = 8  # Per GPU
    
    try:
        train_dataset = TextDataset(data_path, seq_len, tokenizer)
    except Exception as e:
        if rank == 0:
            logger.error(f"Failed to load dataset: {e}")
        raise
    
    if rank == 0:
        logger.info(f"Dataset size: {len(train_dataset)} sequences")
    
    sampler = DistributedSampler(train_dataset, world_size, rank) if world_size > 1 else None
    loader = DataLoader(train_dataset, batch_size, shuffle=(sampler is None),
                       sampler=sampler, num_workers=4, pin_memory=True,
                       drop_last=True, persistent_workers=True)
    
    # Model: GPT-2 Small (117M params)
    model = GPT2(vocab_size=50257, dim=768, num_heads=12, num_layers=12, max_len=seq_len).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"Parameters: {params:.1f}M")
    
    # Optimizer with weight decay only on certain params
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if 'bias' in name or 'ln' in name or 'embed' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': cfg['wd']},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = create_optimizer(optimizer_name, param_groups,
                                lr=cfg['lr'], weight_decay=cfg['wd'],
                                betas=cfg['betas'])
    
    # LR scheduler: cosine with warmup
    warmup_steps = min(2000, total_steps // 10)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training
    scaler = torch.amp.GradScaler()
    best_ppl = float('inf')
    results = {'optimizer': optimizer_name, 'history': []}
    
    model.train()
    step = 0
    epoch = 0
    
    while step < total_steps:
        epoch += 1
        if sampler:
            sampler.set_epoch(epoch)
        
        for x, y in loader:
            if step >= total_steps:
                break
            
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            step += 1
            
            # Log every 2000 steps
            if rank == 0 and step % 2000 == 0:
                ppl = math.exp(min(loss.item(), 20))
                results['history'].append({'step': step, 'loss': loss.item(), 'ppl': ppl})
                
                if ppl < best_ppl:
                    best_ppl = ppl
                    logger.info(f"Step {step:6d}: loss={loss.item():.4f} ppl={ppl:.2f}*")
                else:
                    logger.info(f"Step {step:6d}: loss={loss.item():.4f} ppl={ppl:.2f}")
    
    if rank == 0:
        results['best_ppl'] = best_ppl
        results['final_step'] = step
        logger.info(f"Final: Best Perplexity = {best_ppl:.2f}")
    
    # Cleanup
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default='all')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--steps', type=int, default=50000)
    args = parser.parse_args()
    
    print(f"Starting LM training with args: {args}")
    
    # Distributed setup
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    print(f"Rank {rank}/{world_size}, local_rank={local_rank}")
    
    if world_size > 1:
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
    
    device = torch.device(f'cuda:{local_rank}')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(output_dir, rank, 'lm')
    
    if rank == 0:
        logger.info(f"Data path: {args.data}")
        logger.info(f"Output dir: {args.output}")
        logger.info(f"Total steps: {args.steps}")
    
    # Optimizers to test
    if args.optimizer == 'all':
        optimizers = ['adamw', 'lion', 'rlo', 'rlo_lambda_a', 'smooth_lifted_rlo']
    else:
        optimizers = [args.optimizer]
    
    results = {}
    
    for opt in optimizers:
        try:
            results[opt] = train_lm(opt, Path(args.data), output_dir, args.steps,
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
        logger.info("LM Results (Perplexity, lower is better):")
        for opt in sorted(results.keys(), key=lambda x: results[x].get('best_ppl', float('inf'))):
            ppl = results[opt].get('best_ppl', float('inf'))
            logger.info(f"  {opt}: {ppl:.2f}")
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
