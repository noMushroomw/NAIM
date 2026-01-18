#!/usr/bin/env python3
"""
Language Modeling: GPT-2 Small on WikiText-103
Reports PERPLEXITY (lower is better, target: 20-50)
"""
import os, gc, math, json, logging, argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
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


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = dropout
        self.resid_drop = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, 
                                           dropout_p=self.attn_drop if self.training else 0)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dropout)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2(nn.Module):
    """GPT-2 Small: ~124M params."""
    def __init__(self, vocab_size=50257, dim=768, num_heads=12, num_layers=12, max_len=1024, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dim, num_heads, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.token_embed.weight
        
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('fc2.weight') or pn.endswith('proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * num_layers))
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    
    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.token_embed(idx) + self.pos_embed(pos))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


class TextDataset(Dataset):
    def __init__(self, tokens, seq_len=1024):
        self.tokens = tokens
        self.seq_len = seq_len
    
    def __len__(self):
        return max(1, (len(self.tokens) - 1) // self.seq_len)
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = min(start + self.seq_len + 1, len(self.tokens))
        chunk = self.tokens[start:end]
        if len(chunk) < self.seq_len + 1:
            chunk = F.pad(chunk, (0, self.seq_len + 1 - len(chunk)))
        return chunk[:-1], chunk[1:]


def load_wikitext(data_path, split='train'):
    """Load WikiText-103 data."""
    data_path = Path(data_path)
    
    # Try different file names
    candidates = [
        data_path / f'wiki.{split}.raw',
        data_path / f'wiki.{split}.tokens',
        data_path / f'{split}.txt',
        data_path / f'{split}.raw',
    ]
    
    text_file = None
    for c in candidates:
        if c.exists():
            text_file = c
            break
    
    if text_file is None:
        # Try to find any text file
        txt_files = list(data_path.glob('*.txt')) + list(data_path.glob('*.raw'))
        if txt_files:
            text_file = txt_files[0]
        else:
            raise FileNotFoundError(f"No text file found in {data_path}")
    
    print(f"Loading {split} from: {text_file}")
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return text


def tokenize(text, tokenizer=None):
    """Tokenize text using GPT2 tokenizer or byte-level fallback."""
    if tokenizer is not None:
        return torch.tensor(tokenizer.encode(text), dtype=torch.long)
    else:
        # Byte-level fallback
        return torch.tensor([ord(c) % 50257 for c in text], dtype=torch.long)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum')
        total_loss += loss.item()
        total_tokens += y.numel()
    
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(min(avg_loss, 100))


def train_lm(opt_name, data_path, output_dir, total_steps, rank, world_size, device, logger):
    # Tuned configs for good perplexity
    # Lion paper uses different betas for LM: (0.95, 0.98)
    configs = {
        'adamw': {'lr': 6e-4, 'wd': 0.1, 'betas': (0.9, 0.95)},
        'lion': {'lr': 1e-4, 'wd': 0.1, 'betas': (0.95, 0.98)},
        'rlo': {'lr': 1.2e-4, 'wd': 0.1, 'betas': (0.95, 0.98)},
        'rlo_lambda_a': {'lr': 1.2e-4, 'wd': 0.1, 'betas': (0.95, 0.98)},
        'smooth_lifted_rlo': {'lr': 1.2e-4, 'wd': 0.1, 'betas': (0.95, 0.98)},
    }
    
    cfg = configs.get(opt_name, configs['adamw'])
    
    if rank == 0:
        logger.info("=" * 70)
        logger.info(f"Language Model: {opt_name}")
        logger.info(f"LR={cfg['lr']}, WD={cfg['wd']}, Betas={cfg['betas']}")
        logger.info("=" * 70)
    
    # Tokenizer
    tokenizer = None
    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        if rank == 0:
            logger.info("Using GPT2 tokenizer")
    except Exception as e:
        if rank == 0:
            logger.info(f"Using byte-level tokenization: {e}")
    
    # Load data
    train_text = load_wikitext(data_path, 'train')
    val_text = load_wikitext(data_path, 'valid')
    
    train_tokens = tokenize(train_text, tokenizer)
    val_tokens = tokenize(val_text, tokenizer)
    
    if rank == 0:
        logger.info(f"Train tokens: {len(train_tokens):,}, Val tokens: {len(val_tokens):,}")
    
    seq_len = 512
    batch_size = 16  # Per GPU
    
    train_dataset = TextDataset(train_tokens, seq_len)
    val_dataset = TextDataset(val_tokens, seq_len)
    
    sampler = DistributedSampler(train_dataset, world_size, rank) if world_size > 1 else None
    train_loader = DataLoader(train_dataset, batch_size, shuffle=(sampler is None),
                             sampler=sampler, num_workers=4, pin_memory=True,
                             drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)
    
    # Model
    model = GPT2(vocab_size=50257, dim=768, num_heads=12, num_layers=12, max_len=seq_len, dropout=0.1).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"Parameters: {params:.1f}M")
    
    # Separate decay/no-decay params
    decay_params = [p for n, p in model.named_parameters() if 'bias' not in n and 'ln' not in n and 'embed' not in n]
    no_decay_params = [p for n, p in model.named_parameters() if 'bias' in n or 'ln' in n or 'embed' in n]
    
    param_groups = [
        {'params': decay_params, 'weight_decay': cfg['wd']},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = create_optimizer(opt_name, param_groups, lr=cfg['lr'], weight_decay=cfg['wd'], betas=cfg['betas'])
    
    # LR schedule: warmup + cosine
    warmup_steps = min(2000, total_steps // 10)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training
    scaler = torch.amp.GradScaler()
    best_ppl = float('inf')
    results = {'optimizer': opt_name, 'history': []}
    
    model.train()
    step = 0
    epoch = 0
    
    while step < total_steps:
        epoch += 1
        if sampler:
            sampler.set_epoch(epoch)
        
        for x, y in train_loader:
            if step >= total_steps:
                break
            
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            if torch.isnan(loss) or torch.isinf(loss):
                if rank == 0:
                    logger.warning(f"Step {step}: NaN/Inf loss, skipping")
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            step += 1
            
            # Evaluate every 5000 steps
            if rank == 0 and step % 5000 == 0:
                ppl = evaluate(model, val_loader, device)
                results['history'].append({'step': step, 'ppl': ppl})
                
                marker = '*' if ppl < best_ppl else ''
                best_ppl = min(best_ppl, ppl)
                logger.info(f"Step {step:6d}: ppl={ppl:.2f}{marker} lr={scheduler.get_last_lr()[0]:.2e}")
                model.train()
    
    # Final eval
    if rank == 0:
        final_ppl = evaluate(model, val_loader, device)
        best_ppl = min(best_ppl, final_ppl)
        results['best_ppl'] = best_ppl
        logger.info(f"Final: Best PPL = {best_ppl:.2f}")
    
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default='all')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--steps', type=int, default=100000)
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
    
    logger = setup_logger(output_dir, rank, 'lm')
    
    if args.optimizer == 'all':
        optimizers = ['adamw', 'lion', 'rlo', 'rlo_lambda_a', 'smooth_lifted_rlo']
    else:
        optimizers = [args.optimizer]
    
    results = {}
    for opt in optimizers:
        try:
            results[opt] = train_lm(opt, Path(args.data), output_dir, args.steps, rank, world_size, device, logger)
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
        logger.info("LM RESULTS (Perplexity ↓)")
        logger.info("=" * 70)
        for opt in sorted(results.keys(), key=lambda x: results[x].get('best_ppl', float('inf'))):
            ppl = results[opt].get('best_ppl', float('inf'))
            logger.info(f"  {opt}: {ppl:.2f}")
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
