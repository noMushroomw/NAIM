import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from .data_processing import imagenet_loaders
from .models import ResNet50, vit_b16, vit_s16
from .optimizers import build_optimizer
from .utils import cosine_warmup, load_config, parse_args, run_name, save_json, seed_everything
from .utils.config import parse_overrides

MODELS = {'resnet50': ResNet50, 'vit_s16': vit_s16, 'vit_b16': vit_b16}
OPT_KEYS = ('lambda_b', 'gamma', 'eta', 'field', 'use_global_norm', 'belief_norm', 'beta3',
            'scale_mode', 'scale_cap', 'eps')
DEFAULTS = dict(model='vit_b16', optimizer='adamw', epochs=90, batch_size=128, warmup_epochs=5,
                eval_every=10, num_workers=8, seed=0, data_root='data/imagenet',
                name_keys=['model', 'optimizer', 'lr', 'weight_decay'])

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16):
            correct += model(x).argmax(1).eq(y).sum().item()
        total += y.size(0)
    model.train()
    return 100.0 * correct / total

def main():
    args = parse_args('')
    cfg_file = load_config(args.config)
    over = parse_overrides(args.set)

    cfg = {**DEFAULTS, **cfg_file['base'], **over}
    cfg.update(cfg_file.get('hparams', {}).get(cfg['model'], {}).get(cfg['optimizer'], {}))
    cfg.update(over)

    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if world > 1:
        dist.init_process_group('nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    seed_everything(cfg['seed'] + rank)

    name = run_name(cfg, cfg['name_keys'], over)
    path = Path(args.out) / cfg_file.get('name', 'table4_imagenet') / f'{name}.json'
    if rank == 0:
        print(f'{name}: {cfg}', flush=True)
    if args.dry_run:
        return
    if path.exists() and not args.force:
        done = json.loads(path.read_text())
        if len(done.get('history', [])) >= cfg['epochs']:
            if rank == 0:
                print(f'{name}: finished run exists, skipping (use --force to rerun)')
            return

    train_loader, val_loader, sampler = imagenet_loaders(cfg['data_root'], cfg['batch_size'], rank,
                                                         world, cfg['num_workers'])
    model = MODELS[cfg['model']](num_classes=1000).to(device)
    if world > 1:
        model = DDP(model, device_ids=[local_rank])
    kw = {k: cfg[k] for k in OPT_KEYS if k in cfg}
    opt = build_optimizer(cfg['optimizer'], model.parameters(), lr=cfg['lr'],
                          weight_decay=cfg['weight_decay'], betas=cfg['betas'], **kw)
    steps = cfg['epochs'] * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, cosine_warmup(steps, cfg['warmup_epochs'] * len(train_loader), 0.01))
    scaler = torch.amp.GradScaler()
    history, t0 = [], time.time()
    for epoch in range(1, cfg['epochs'] + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        total, nb = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.autocast('cuda', dtype=torch.float16):
                loss = F.cross_entropy(model(x), y, label_smoothing=0.1)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            nb += 1
        if rank == 0:
            entry = {'epoch': epoch, 'loss': total / max(nb, 1)}
            if epoch % cfg['eval_every'] == 0 or epoch == cfg['epochs']:

                entry['acc'] = evaluate(model.module if world > 1 else model, val_loader, device)
            history.append(entry)
            print(f"epoch {epoch:3d}  loss {entry['loss']:.4f}"
                  + (f"  top1 {entry['acc']:.2f}%" if 'acc' in entry else '')
                  + f"  ({(time.time() - t0) / 3600:.1f} h)", flush=True)
            accs = [h['acc'] for h in history if 'acc' in h]
            save_json({'config': cfg, 'history': history,
                       'metrics': {'best_acc': max(accs) if accs else None,
                                   'final_acc': accs[-1] if accs else None}}, path)
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
