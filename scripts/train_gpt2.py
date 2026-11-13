import contextlib
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data_processing import TokenBin
from .models import GPT2, gpt2_param_groups
from .optimizers import build_optimizer
from .utils import (expand_runs, gpt2_lr, load_config, parse_args, resolve_device, run_name, save_json,
                    seed_everything, torch_generator)

OPT_KEYS = ('lambda_b', 'gamma', 'eta', 'field', 'use_global_norm', 'belief_norm', 'beta3',
            'scale_mode', 'scale_cap', 'eps')

DEFAULTS = dict(
    optimizer='adamw', lr=6e-4, weight_decay=0.1, betas=[0.9, 0.95], seed=42,
    max_steps=10000, batch_size=12, grad_accum=40, block_size=1024, warmup_steps=2000,
    min_lr_ratio=0.1, grad_clip=1.0, eval_mode='random', eval_interval=1000, eval_iters=200,
    warmup_indexing='table5', data_seed=1337, compile=True, data_dir='data/openwebtext',
    log_interval=100, name_keys=['optimizer', 'lr', 'warmup_steps', 'seed'],
)

def train(cfg, device):
    seed_everything(cfg['seed'])
    train_data = TokenBin(cfg['data_dir'], 'train', cfg['block_size'])
    val_data = TokenBin(cfg['data_dir'], 'val', cfg['block_size'])
    model = GPT2(block_size=cfg['block_size']).to(device)
    n_params = model.count_parameters()
    if cfg['compile']:
        try:
            import triton
            model = torch.compile(model)
        except ImportError:
            print('  Triton is not installed: running without torch.compile', flush=True)
    kw = {k: cfg[k] for k in OPT_KEYS if k in cfg}
    opt = build_optimizer(cfg['optimizer'], gpt2_param_groups(model, cfg['weight_decay']),
                          lr=cfg['lr'], weight_decay=cfg['weight_decay'], betas=cfg['betas'], **kw)
    use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    ctx = torch.autocast('cuda', dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda' and not use_bf16)

    fixed = cfg['eval_mode'] == 'fixed'
    train_gen = torch_generator(cfg['data_seed'], 'gpt2_train') if fixed else None
    if fixed:
        g = torch_generator(cfg['data_seed'], 'gpt2_val')
        val_ix = [val_data.indices(cfg['batch_size'], g) for _ in range(cfg['eval_iters'])]
        train_ix = [train_data.indices(cfg['batch_size'], g) for _ in range(cfg['eval_iters'])]

    @torch.no_grad()
    def estimate():
        model.eval()
        out = {}
        for split, data, ixs in (('train', train_data, train_ix if fixed else None),
                                 ('val', val_data, val_ix if fixed else None)):
            losses = []
            for k in range(cfg['eval_iters']):
                X, Y = data.batch(ixs[k], device) if fixed else data.sample(cfg['batch_size'], device)
                with ctx:
                    losses.append(float(model(X, Y)[1]))
            out[split] = float(np.mean(losses))
        model.train()
        return out

    hist = {'steps': [], 'train_loss': [], 'val_loss': [], 'lr': []}

    def record(step, lr):
        e = estimate()
        for k, v in (('steps', step), ('train_loss', e['train']), ('val_loss', e['val']), ('lr', lr)):
            hist[k].append(v)
        print(f"  step {step:6d}  train {e['train']:.4f}  val {e['val']:.4f}  "
              f"ppl {math.exp(e['val']):.2f}", flush=True)

    record(0, 0.0)
    model.train()
    running, t0 = 0.0, time.time()
    offset = 0 if cfg['warmup_indexing'] == 'table5' else 1
    for step in range(1, cfg['max_steps'] + 1):
        lr = cfg['lr'] * gpt2_lr(step - offset, cfg['warmup_steps'], cfg['max_steps'], cfg['min_lr_ratio'])
        for group in opt.param_groups:
            group['lr'] = lr
        opt.zero_grad(set_to_none=True)
        for _ in range(cfg['grad_accum']):
            X, Y = train_data.sample(cfg['batch_size'], device, train_gen)
            with ctx:
                loss = model(X, Y)[1] / cfg['grad_accum']
            running += loss.item()
            scaler.scale(loss).backward()
        if cfg['grad_clip'] > 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        scaler.step(opt)
        scaler.update()
        if step % cfg['log_interval'] == 0:
            tok = cfg['log_interval'] * cfg['grad_accum'] * cfg['batch_size'] * cfg['block_size']
            print(f'  step {step:6d}  loss {running / cfg["log_interval"]:.4f}  lr {lr:.2e}  '
                  f'{tok / (time.time() - t0):.0f} tok/s', flush=True)
            running, t0 = 0.0, time.time()
        if step % cfg['eval_interval'] == 0:
            record(step, lr)

    if fixed:
        if hist['steps'][-1] != cfg['max_steps']:
            record(cfg['max_steps'], lr)
        final = {'train': hist['train_loss'][-1], 'val': hist['val_loss'][-1]}
    else:
        final = estimate()
    best = min(hist['val_loss'][1:])
    metrics = {
        'n_params': n_params,
        'best_val_loss': best, 'best_val_ppl': math.exp(best),
        'final_val_loss': final['val'], 'final_val_ppl': math.exp(final['val']),
        'final_train_loss': final['train'], 'gap': final['val'] - final['train'],
    }
    return metrics, {'history': hist}

def main():
    args = parse_args('')
    cfg_file = load_config(args.config, args.set)
    runs = [{**DEFAULTS, **r} for r in expand_runs(cfg_file)]
    out_dir = Path(args.out) / cfg_file.get('name', Path(args.config).stem)
    device = resolve_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    for i, cfg in enumerate(runs):
        name = run_name(cfg, cfg['name_keys'], cfg_file.get('overrides'))
        if args.dry_run:
            print(f'[{i}] {name}')
            continue
        if args.only is not None and i != args.only:
            continue
        path = out_dir / f'{name}.json'
        if path.exists() and not args.force:
            print(f'[{i}] {name}: done, skipping')
            continue
        print(f'[{i + 1}/{len(runs)}] {name}', flush=True)
        t0 = time.time()
        metrics, extra = train(cfg, device)
        metrics['minutes'] = (time.time() - t0) / 60
        save_json({'config': cfg, 'metrics': metrics, **extra}, path)
        print(f'  -> {metrics}', flush=True)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
