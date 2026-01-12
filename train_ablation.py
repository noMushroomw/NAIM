#!/usr/bin/env python3
"""
Ablation Studies
================
1. Learning Rate Sensitivity
2. Batch Size Sensitivity
3. Belief Coefficient (λ_b)
4. Fiber Contraction Rate (η)
5. Component Analysis (sign vs smooth, with/without belief)

Uses ResNet-50 on ImageNet for 30 epochs (quick ablation)
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
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Optimizer
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
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
    """RLO with configurable belief coefficient"""
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
                # Belief correction with configurable coefficient
                if group["belief_coef"] > 0:
                    update = c.sign() + group["belief_coef"] * (delta / delta.norm().clamp(min=group["eps"]))
                else:
                    update = c.sign()  # Pure sign (like Lion)
                p.add_(update, alpha=-group["lr"])
                m.mul_(beta2).add_(g, alpha=1 - beta2)


class SmoothLiftedRLO(Optimizer):
    """SmoothLiftedRLO with configurable parameters"""
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
                b = group["lambda_b"] * (delta / delta.norm().clamp(min=group["eps"])) if group["lambda_b"] > 0 else 0
                all_s.append(s); all_b.append(b); all_p.append((p, group))
        if not all_p:
            return
        scale = self.sqrt_dim / sum((x * x).sum() for x in all_s).sqrt().clamp(min=1e-8)
        for (p, group), s, b in zip(all_p, all_s, all_b):
            state = self.state[p]
            if group["weight_decay"] != 0:
                p.mul_(1 - group["lr"] * group["weight_decay"])
            d = scale * s + (b if isinstance(b, torch.Tensor) else 0)
            state["v"].mul_(1 - group["eta"]).add_(d, alpha=group["eta"])
            p.add_(state["v"], alpha=-group["lr"])
            state["m"].mul_(group["beta2"]).add_(p.grad, alpha=1 - group["beta2"])


# =============================================================================
# Data
# =============================================================================
def create_loader(data_path, batch_size, is_train, world_size, rank):
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if is_train:
        transform = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
        split = 'train'
    else:
        transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
        split = 'val'
    dataset = ImageFolder(data_path / split, transform)
    sampler = DistributedSampler(dataset, world_size, rank, shuffle=is_train) if world_size > 1 else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None and is_train),
                     sampler=sampler, num_workers=2, pin_memory=True, drop_last=is_train,
                     persistent_workers=False), sampler


# =============================================================================
# Training Function
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast('cuda', torch.bfloat16):
            correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def train_ablation(config, data_path, output_dir, rank, world_size, local_rank, logger):
    """Train with specific configuration for ablation"""
    device = torch.device(f'cuda:{local_rank}')
    
    global_batch = config.get('batch_size', 1024)
    lr = config.get('lr', 1e-4)
    wd = config.get('wd', 1.0)
    opt_type = config.get('optimizer', 'rlo')
    belief_coef = config.get('belief_coef', 0.1)
    eta = config.get('eta', 0.3)
    epochs = config.get('epochs', 30)
    name = config.get('name', 'ablation')
    
    per_gpu = global_batch // world_size

    logger.info(f">>> {name}: batch={global_batch}, lr={lr}, wd={wd}, opt={opt_type}, belief={belief_coef}, eta={eta}")

    # Data
    train_loader, train_sampler = create_loader(data_path, per_gpu, True, world_size, rank)
    val_loader, _ = create_loader(data_path, per_gpu, False, 1, 0)
    steps_per_epoch = len(train_loader)

    # Model
    model = resnet50(weights=None, num_classes=1000).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Optimizer
    params = model.parameters()
    if opt_type == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif opt_type == 'lion':
        optimizer = Lion(params, lr=lr, weight_decay=wd)
    elif opt_type == 'rlo':
        optimizer = RLO(params, lr=lr, weight_decay=wd, belief_coef=belief_coef)
    elif opt_type == 'smooth_lifted_rlo':
        optimizer = SmoothLiftedRLO(params, lr=lr, weight_decay=wd, lambda_b=belief_coef, eta=eta)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")

    # Scheduler
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * 5
    
    def get_lr(step):
        if step < warmup_steps:
            return lr * step / max(1, warmup_steps)
        return lr * 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    history = {'train_loss': [], 'val_acc': []}
    best_acc = 0.0
    step = 0

    for epoch in range(epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        loss_sum = 0.0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            cur_lr = get_lr(step)
            for g in optimizer.param_groups:
                g['lr'] = cur_lr

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', torch.bfloat16):
                loss = criterion(model(x), y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()
            step += 1

        base_model = model.module if hasattr(model, 'module') else model
        val_acc = evaluate(base_model, val_loader, device)
        history['train_loss'].append(loss_sum / steps_per_epoch)
        history['val_acc'].append(val_acc)
        best_acc = max(val_acc, best_acc)

        if (epoch + 1) % 5 == 0:
            logger.info(f"  E{epoch+1}: loss={loss_sum/steps_per_epoch:.4f}, val={val_acc:.2f}%")

    logger.info(f"  Best: {best_acc:.2f}%")
    return {'name': name, 'config': config, 'best_acc': best_acc, 'final_acc': val_acc, 'history': history}


# =============================================================================
# Ablation Studies
# =============================================================================
def run_lr_ablation(data_path, output_dir, rank, world_size, local_rank, logger):
    """Study 1: Learning rate sensitivity"""
    logger.info("=" * 70)
    logger.info("ABLATION 1: Learning Rate Sensitivity")
    logger.info("=" * 70)
    
    results = []
    for lr in [5e-5, 1e-4, 2e-4, 5e-4, 1e-3]:
        for opt in ['lion', 'rlo', 'smooth_lifted_rlo']:
            config = {
                'name': f'{opt}_lr{lr}',
                'optimizer': opt,
                'lr': lr,
                'wd': 1.0,
                'batch_size': 1024,
                'epochs': 30,
                'belief_coef': 0.1,
                'eta': 0.3,
            }
            result = train_ablation(config, data_path, output_dir, rank, world_size, local_rank, logger)
            results.append(result)
            gc.collect(); torch.cuda.empty_cache()
    
    return results


def run_batch_ablation(data_path, output_dir, rank, world_size, local_rank, logger):
    """Study 2: Batch size sensitivity"""
    logger.info("=" * 70)
    logger.info("ABLATION 2: Batch Size Sensitivity")
    logger.info("=" * 70)
    
    results = []
    for batch in [256, 512, 1024, 2048]:
        # Scale LR linearly with batch size (linear scaling rule)
        base_lr = 1e-4
        lr = base_lr * (batch / 1024)
        
        for opt in ['lion', 'rlo', 'smooth_lifted_rlo']:
            config = {
                'name': f'{opt}_batch{batch}',
                'optimizer': opt,
                'lr': lr,
                'wd': 1.0,
                'batch_size': batch,
                'epochs': 30,
                'belief_coef': 0.1,
                'eta': 0.3,
            }
            result = train_ablation(config, data_path, output_dir, rank, world_size, local_rank, logger)
            results.append(result)
            gc.collect(); torch.cuda.empty_cache()
    
    return results


def run_belief_ablation(data_path, output_dir, rank, world_size, local_rank, logger):
    """Study 3: Belief coefficient (λ_b) sensitivity"""
    logger.info("=" * 70)
    logger.info("ABLATION 3: Belief Coefficient (λ_b)")
    logger.info("=" * 70)
    
    results = []
    for belief in [0.0, 0.05, 0.1, 0.2, 0.5]:
        for opt in ['rlo', 'smooth_lifted_rlo']:
            config = {
                'name': f'{opt}_belief{belief}',
                'optimizer': opt,
                'lr': 1e-4,
                'wd': 1.0,
                'batch_size': 1024,
                'epochs': 30,
                'belief_coef': belief,
                'eta': 0.3,
            }
            result = train_ablation(config, data_path, output_dir, rank, world_size, local_rank, logger)
            results.append(result)
            gc.collect(); torch.cuda.empty_cache()
    
    return results


def run_eta_ablation(data_path, output_dir, rank, world_size, local_rank, logger):
    """Study 4: Fiber contraction rate (η) for lifted optimizer"""
    logger.info("=" * 70)
    logger.info("ABLATION 4: Fiber Contraction Rate (η)")
    logger.info("=" * 70)
    
    results = []
    for eta in [0.1, 0.2, 0.3, 0.5, 1.0]:
        config = {
            'name': f'smooth_lifted_eta{eta}',
            'optimizer': 'smooth_lifted_rlo',
            'lr': 1e-4,
            'wd': 1.0,
            'batch_size': 1024,
            'epochs': 30,
            'belief_coef': 0.1,
            'eta': eta,
        }
        result = train_ablation(config, data_path, output_dir, rank, world_size, local_rank, logger)
        results.append(result)
        gc.collect(); torch.cuda.empty_cache()
    
    return results


def run_component_ablation(data_path, output_dir, rank, world_size, local_rank, logger):
    """Study 5: Component analysis"""
    logger.info("=" * 70)
    logger.info("ABLATION 5: Component Analysis")
    logger.info("=" * 70)
    
    configs = [
        {'name': 'adamw_baseline', 'optimizer': 'adamw', 'lr': 1e-3, 'wd': 1e-4},
        {'name': 'lion_baseline', 'optimizer': 'lion', 'lr': 1e-4, 'wd': 1.0},
        {'name': 'rlo_no_belief', 'optimizer': 'rlo', 'lr': 1e-4, 'wd': 1.0, 'belief_coef': 0.0},
        {'name': 'rlo_with_belief', 'optimizer': 'rlo', 'lr': 1e-4, 'wd': 1.0, 'belief_coef': 0.1},
        {'name': 'smooth_lifted_no_belief', 'optimizer': 'smooth_lifted_rlo', 'lr': 1e-4, 'wd': 1.0, 'belief_coef': 0.0, 'eta': 0.3},
        {'name': 'smooth_lifted_full', 'optimizer': 'smooth_lifted_rlo', 'lr': 1e-4, 'wd': 1.0, 'belief_coef': 0.1, 'eta': 0.3},
    ]
    
    results = []
    for cfg in configs:
        cfg.setdefault('batch_size', 1024)
        cfg.setdefault('epochs', 30)
        cfg.setdefault('belief_coef', 0.1)
        cfg.setdefault('eta', 0.3)
        result = train_ablation(cfg, data_path, output_dir, rank, world_size, local_rank, logger)
        results.append(result)
        gc.collect(); torch.cuda.empty_cache()
    
    return results


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', default='all', choices=['all', 'lr', 'batch', 'belief', 'eta', 'component'])
    parser.add_argument('--data', default='/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder')
    parser.add_argument('--output', default='/blue/wdixon/wang.yixuan/rlo_results/ablation')
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
    logger = setup_logger(output_dir, rank, "ablation")
    data_path = Path(args.data)

    all_results = {}
    
    studies = {
        'lr': run_lr_ablation,
        'batch': run_batch_ablation,
        'belief': run_belief_ablation,
        'eta': run_eta_ablation,
        'component': run_component_ablation,
    }
    
    if args.study == 'all':
        for name, func in studies.items():
            try:
                all_results[name] = func(data_path, output_dir, rank, world_size, local_rank, logger)
            except Exception as e:
                logger.error(f"Error in {name}: {e}")
                import traceback; logger.error(traceback.format_exc())
    else:
        all_results[args.study] = studies[args.study](data_path, output_dir, rank, world_size, local_rank, logger)

    # Save all results
    if rank == 0:
        with open(output_dir / "ablation_results.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info("=" * 70)
        logger.info("ABLATION SUMMARY")
        logger.info("=" * 70)
        for study_name, results in all_results.items():
            logger.info(f"\n{study_name.upper()}:")
            if isinstance(results, list):
                for r in sorted(results, key=lambda x: x['best_acc'], reverse=True)[:5]:
                    logger.info(f"  {r['name']}: {r['best_acc']:.2f}%")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
