import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_processing import cifar10_train_test, cifar10_train_val_probe
from .models import cifar_resnet18
from .optimizers import MetricRLO, RLOLifted, build_metric, build_optimizer
from .utils import (cosine_warmup, expand_runs, load_config, parse_args, resolve_device, run_name,
                    save_json, seed_everything)

OPT_KEYS = ('lambda_b', 'gamma', 'eta', 'field', 'use_global_norm', 'belief_norm', 'beta3',
            'scale_mode', 'scale_cap', 'eps')

DEFAULTS = dict(
    protocol='recipe', optimizer='rlo_lifted', lr=1e-4, weight_decay=0.1, betas=[0.9, 0.99],
    seed=42, epochs=50, batch_size=128, eval_every=5, num_workers=4, data_root='data/cifar10',
    naim_interval=0, name_keys=['optimizer', 'lr', 'seed'],
)

def build(cfg, params):
    kw = {k: cfg[k] for k in OPT_KEYS if k in cfg}
    if cfg['optimizer'] == 'metric_rlo':
        metric = build_metric(cfg['metric'], **cfg.get('metric_kwargs', {}))
        kw = {k: v for k, v in kw.items() if k in ('gamma', 'lambda_b', 'eps')}
        return MetricRLO(params, metric=metric, lr=cfg['lr'], betas=tuple(cfg['betas']),
                         weight_decay=cfg['weight_decay'], **kw)
    if cfg['protocol'] in ('ablation', 'diagnostics'):
        return RLOLifted(params, lr=cfg['lr'], betas=tuple(cfg['betas']),
                         weight_decay=cfg['weight_decay'], naim_interval=cfg['naim_interval'],
                         collect_diagnostics=cfg.get('collect_diagnostics', False), **kw)
    return build_optimizer(cfg['optimizer'], params, lr=cfg['lr'], weight_decay=cfg['weight_decay'],
                           betas=cfg['betas'], **kw)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += model(x).argmax(1).eq(y).sum().item()
        total += y.size(0)
    model.train()
    return 100.0 * correct / total

@contextlib.contextmanager
def bn_batch_stats(model):

    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    saved = [(m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone()) for m in bns]
    for m in bns:
        m.train()
    try:
        yield
    finally:
        for m, (rm, rv, nb) in zip(bns, saved):
            m.running_mean.copy_(rm)
            m.running_var.copy_(rv)
            m.num_batches_tracked.copy_(nb)

@torch.no_grad()
def probe_loss(model, batches, device):
    model.eval()
    total = n = 0
    with bn_batch_stats(model):
        for x, y in batches:
            total += float(F.cross_entropy(model(x.to(device)), y.to(device))) * x.size(0)
            n += x.size(0)
    model.train()
    return total / n

def run_recipe(cfg, device):
    train_loader, test_loader = cifar10_train_test(cfg['data_root'], cfg['batch_size'],
                                                   cfg['num_workers'], drop_last=True,
                                                   test_batch_size=2 * cfg['batch_size'])
    seed_everything(cfg['seed'])
    model = cifar_resnet18().to(device)
    opt = build(cfg, model.parameters())
    steps = cfg['epochs'] * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_warmup(steps, 5 * len(train_loader), 0.01))
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)
    history = []
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        total, nb = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = F.cross_entropy(model(x), y, label_smoothing=0.1)
            if not torch.isfinite(loss):
                sched.step()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            nb += 1
        if epoch % cfg['eval_every'] == 0 or epoch == cfg['epochs']:
            acc = evaluate(model, test_loader, device)
            history.append({'epoch': epoch, 'loss': total / max(nb, 1), 'acc': acc})
            print(f"  epoch {epoch:3d}  loss {history[-1]['loss']:.4f}  test {acc:.2f}%", flush=True)
    accs = [h['acc'] for h in history]
    return {'best_acc': max(accs), 'final_acc': accs[-1]}, {'history': history}

def run_ablation(cfg, device):
    train_loader, test_loader = cifar10_train_test(cfg['data_root'], cfg['batch_size'], 0,
                                                   drop_last=False)
    seed_everything(cfg['seed'])
    model = cifar_resnet18().to(device)
    opt = build(cfg, model.parameters())
    step_losses, test_acc, epoch_diag = [], [], []
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            step_losses.append(loss.item())
        if cfg.get('collect_diagnostics'):
            epoch_diag.append({k: float(np.mean(v)) for k, v in opt.get_diagnostics().items()})
        test_acc.append(evaluate(model, test_loader, device))
        print(f'  epoch {epoch:3d}  loss {np.mean(step_losses[-len(train_loader):]):.4f}  '
              f'test {test_acc[-1]:.2f}%', flush=True)
    metrics = {'best_acc': max(test_acc), 'final_acc': test_acc[-1],
               'loss_std': float(np.std(step_losses))}
    extra = {'test_acc': test_acc}
    if epoch_diag:
        extra['epoch_diagnostics'] = epoch_diag
        metrics['final_fiber_residual'] = epoch_diag[-1]['fiber_residual']
    if cfg['naim_interval'] > 0:
        log = opt.pop_naim_log()
        for key in ('r_rel', 'cos_vd', 'q_perp_rel'):
            vals = [v for v in log[key] if v is not None and math.isfinite(v)]
            metrics[f'mean_{key}'] = float(np.mean(vals)) if vals else None
        extra['naim'] = log
    return metrics, extra

def run_diagnostics(cfg, device):
    train_loader, val_loader, probe = cifar10_train_val_probe(cfg['data_root'], cfg['seed'],
                                                              cfg['batch_size'], cfg['num_workers'])
    seed_everything(cfg['seed'])
    model = cifar_resnet18().to(device)
    opt = build(cfg, model.parameters())
    total_dim = sum(p.numel() for p in model.parameters())
    f_probe, val_acc = [], []
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        for x, y in train_loader:
            f_probe.append(probe_loss(model, probe, device))
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y, label_smoothing=0.1)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite loss at epoch {epoch}')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val_acc.append(evaluate(model, val_loader, device))
        print(f'  epoch {epoch:3d}  val {val_acc[-1]:.2f}%', flush=True)
    log = opt.pop_naim_log()
    z = np.asarray(log['z_norm'])
    drift = np.asarray([v for v in log['drift_norm'] if v is not None])
    v_hat = np.asarray(f_probe) + z ** 2 / total_dim
    dv = np.diff(v_hat)
    tail_v = v_hat[int(0.8 * len(v_hat)):]
    metrics = {
        'best_val_acc': max(val_acc), 'final_val_acc': val_acc[-1],
        'tail_z_norm': float(z[len(z) // 2:].mean()),
        'tail_delta_norm': float(drift[len(drift) // 2:].mean()),
        'mean_dV': float(dv[int(0.1 * len(dv)):].mean()),
        'tail_V_mean': float(tail_v.mean()), 'tail_V_max': float(tail_v.max()),
    }
    return metrics, {'val_acc': val_acc}

PROTOCOLS = {'recipe': run_recipe, 'ablation': run_ablation, 'diagnostics': run_diagnostics}

def select_lr(cfg, root):

    best = None
    for path in sorted((Path(root) / cfg['lr_from']).glob('*.json')):
        r = json.loads(path.read_text())
        rc = r['config']
        if rc['optimizer'] != cfg['optimizer'] or rc.get('metric') != cfg.get('metric'):
            continue
        if best is None or r['metrics']['best_acc'] > best[0]:
            best = (r['metrics']['best_acc'], rc['lr'])
    if best is None:
        raise FileNotFoundError(f"no search results for {cfg['optimizer']} in {root}/{cfg['lr_from']}; "
                                'run the search config first')
    return best[1]

def main():
    args = parse_args('')
    cfg_file = load_config(args.config, args.set)
    runs = [{**DEFAULTS, **r} for r in expand_runs(cfg_file)]
    out_dir = Path(args.out) / cfg_file.get('name', Path(args.config).stem)
    device = resolve_device(args.device)
    for i, cfg in enumerate(runs):
        if 'lr_from' in cfg and 'lr' not in cfg_file.get('overrides', {}):
            try:
                cfg['lr'] = select_lr(cfg, args.out)
            except FileNotFoundError:
                if not args.dry_run:
                    raise
                cfg['lr'] = f"<best of {cfg['lr_from']}>"
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
        metrics, extra = PROTOCOLS[cfg['protocol']](cfg, device)
        metrics['minutes'] = (time.time() - t0) / 60
        save_json({'config': cfg, 'metrics': metrics, **extra}, path)
        print(f'  -> {metrics}', flush=True)

if __name__ == '__main__':
    main()
