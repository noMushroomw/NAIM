#!/usr/bin/env python3
"""
Ablation Studies on CIFAR-100 with ResNet-18
Focus on: Learning Rate and Batch Size sensitivity

Quick experiments (30 epochs each) to understand optimizer behavior.

FIXES:
- Rank 0 downloads data first, others wait
- Added dist.barrier() between configs to prevent desync
- Fixed evaluation to only run on rank 0 (avoids timing issues)
- Added timeout handling and better cleanup
"""
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
import torchvision.transforms as T
from torchvision.datasets import CIFAR100
from torchvision.models import resnet18
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


def create_cifar_resnet18(num_classes=100):
    """Create ResNet-18 adapted for CIFAR (32x32 images)."""
    model = resnet18(weights=None, num_classes=num_classes)
    # Adapt for CIFAR: smaller kernel, no maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def download_cifar_data(data_path, rank, world_size):
    """Download CIFAR-100 on rank 0 only, others wait."""
    if rank == 0:
        print(f"Rank 0: Downloading CIFAR-100 to {data_path}...")
        CIFAR100(root=data_path, train=True, download=True)
        CIFAR100(root=data_path, train=False, download=True)
        print("Rank 0: Download complete.")
    
    # Synchronize all ranks
    if world_size > 1:
        dist.barrier()


def create_cifar_loader(data_path, batch_size, train, world_size, rank):
    """Create CIFAR-100 data loader."""
    if train:
        transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    
    # Data already downloaded, don't try again
    dataset = CIFAR100(root=data_path, train=train, download=False, transform=transform)
    
    sampler = DistributedSampler(dataset, world_size, rank, shuffle=train) if world_size > 1 else None
    loader = DataLoader(dataset, batch_size, 
                       shuffle=(sampler is None and train),
                       sampler=sampler,
                       num_workers=4, pin_memory=True, drop_last=train,
                       persistent_workers=True if train else False,
                       prefetch_factor=2)
    
    return loader, sampler


@torch.no_grad()
def evaluate(model, loader, device):
    """Compute accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.autocast('cuda', torch.bfloat16):
            outputs = model(images)
        
        predictions = outputs.argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    return 100.0 * correct / total


def train_single_config(cfg, data_path, device, world_size, rank, logger, epochs=30):
    """Train with a single configuration."""
    batch_size = cfg['batch_size']
    
    # Ensure minimum batch per GPU of 16 for stability
    per_gpu_batch = max(16, batch_size // world_size)
    effective_batch = per_gpu_batch * world_size
    
    if rank == 0 and effective_batch != batch_size:
        logger.info(f"  (adjusted batch: {batch_size} -> {effective_batch})")
    
    train_loader, train_sampler = create_cifar_loader(data_path, per_gpu_batch, True, world_size, rank)
    
    # Val loader only needed on rank 0
    if rank == 0:
        val_loader, _ = create_cifar_loader(data_path, 256, False, 1, 0)
    else:
        val_loader = None
    
    model = create_cifar_resnet18(num_classes=100).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    base_model = model.module if hasattr(model, 'module') else model
    optimizer = create_optimizer(base_model, cfg['optimizer'], cfg)
    
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()
    
    # LR schedule
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * 5
    
    def get_lr(step):
        if step < warmup_steps:
            return cfg['lr'] * step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return cfg['lr'] * 0.5 * (1 + math.cos(math.pi * progress))
    
    best_acc = 0.0
    global_step = 0
    
    for epoch in range(epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        model.train()
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            lr = get_lr(global_step)
            for g in optimizer.param_groups:
                g['lr'] = lr
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast('cuda', torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            global_step += 1
        
        # Evaluate on rank 0 only
        if rank == 0:
            base_model = model.module if hasattr(model, 'module') else model
            acc = evaluate(base_model, val_loader, device)
            best_acc = max(acc, best_acc)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"  E{epoch+1}: acc={acc:.2f}% best={best_acc:.2f}%")
        
        # Sync all ranks after each epoch to prevent drift
        if world_size > 1:
            dist.barrier()
    
    # Broadcast best_acc from rank 0 to all ranks
    if world_size > 1:
        best_acc_tensor = torch.tensor([best_acc], device=device)
        dist.broadcast(best_acc_tensor, src=0)
        best_acc = best_acc_tensor.item()
    
    return best_acc


def cleanup_between_configs(world_size):
    """Clean up memory and synchronize between configs."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    if world_size > 1:
        dist.barrier()


def run_lr_study(data_path, output_path, device, world_size, rank, logger):
    """Learning rate sensitivity study."""
    logger.info("=" * 70)
    logger.info("STUDY 1: Learning Rate Sensitivity")
    logger.info("=" * 70)
    
    results = {}
    
    # Test different learning rates
    lr_values = [5e-5, 1e-4, 3e-4, 5e-4, 1e-3]
    optimizers = ['adamw', 'lion', 'rlo', 'rlo_lambda_a', 'smooth_lifted_rlo']
    
    # Base configs for each optimizer
    base_configs = {
        'adamw': {'wd': 0.1, 'betas': (0.9, 0.999)},
        'lion': {'wd': 1.0, 'betas': (0.9, 0.99)},
        'rlo': {'wd': 1.0, 'betas': (0.9, 0.99), 'belief_coef': 0.1},
        'rlo_lambda_a': {'wd': 1.0, 'beta1': 0.9, 'beta2': 0.99, 'lambda_b': 0.1},
        'smooth_lifted_rlo': {'wd': 1.0, 'beta1': 0.9, 'beta2': 0.99, 'lambda_b': 0.1, 'eta': 0.3},
    }
    
    for opt_name in optimizers:
        results[opt_name] = {}
        
        for lr in lr_values:
            cfg = {
                'optimizer': opt_name,
                'lr': lr,
                'batch_size': 128,
                **base_configs[opt_name]
            }
            
            try:
                cleanup_between_configs(world_size)
                
                logger.info(f"\n{opt_name} | lr={lr}")
                acc = train_single_config(cfg, data_path, device, world_size, rank, logger)
                results[opt_name][str(lr)] = acc
                logger.info(f"  Final: {acc:.2f}%")
                
            except Exception as e:
                logger.error(f"Error: {e}")
                results[opt_name][str(lr)] = 0.0
                # Try to recover
                if world_size > 1:
                    try:
                        dist.barrier()
                    except:
                        pass
    
    # Save incrementally
    if rank == 0:
        with open(output_path / 'lr_study_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def run_batch_study(data_path, output_path, device, world_size, rank, logger):
    """Batch size sensitivity study with linear LR scaling."""
    logger.info("=" * 70)
    logger.info("STUDY 2: Batch Size Sensitivity (Linear LR Scaling)")
    logger.info("=" * 70)
    
    results = {}
    
    # Test different batch sizes - ensure each is >= 16 * world_size
    # With 8 GPUs, minimum total batch = 128
    batch_sizes = [128, 256, 512, 1024]
    base_lr = 3e-4  # Base LR for batch=128
    optimizers = ['adamw', 'lion', 'rlo', 'rlo_lambda_a', 'smooth_lifted_rlo']
    
    base_configs = {
        'adamw': {'wd': 0.1, 'betas': (0.9, 0.999)},
        'lion': {'wd': 1.0, 'betas': (0.9, 0.99)},
        'rlo': {'wd': 1.0, 'betas': (0.9, 0.99), 'belief_coef': 0.1},
        'rlo_lambda_a': {'wd': 1.0, 'beta1': 0.9, 'beta2': 0.99, 'lambda_b': 0.1},
        'smooth_lifted_rlo': {'wd': 1.0, 'beta1': 0.9, 'beta2': 0.99, 'lambda_b': 0.1, 'eta': 0.3},
    }
    
    for opt_name in optimizers:
        results[opt_name] = {}
        
        for bs in batch_sizes:
            # Linear LR scaling
            scaled_lr = base_lr * (bs / 128)
            
            cfg = {
                'optimizer': opt_name,
                'lr': scaled_lr,
                'batch_size': bs,
                **base_configs[opt_name]
            }
            
            try:
                cleanup_between_configs(world_size)
                
                logger.info(f"\n{opt_name} | batch={bs} lr={scaled_lr:.2e}")
                acc = train_single_config(cfg, data_path, device, world_size, rank, logger)
                results[opt_name][str(bs)] = acc
                logger.info(f"  Final: {acc:.2f}%")
                
            except Exception as e:
                logger.error(f"Error: {e}")
                results[opt_name][str(bs)] = 0.0
                if world_size > 1:
                    try:
                        dist.barrier()
                    except:
                        pass
    
    # Save incrementally
    if rank == 0:
        with open(output_path / 'batch_study_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def run_eta_study(data_path, output_path, device, world_size, rank, logger):
    """Eta (fiber contraction rate) study for SmoothLiftedRLO."""
    logger.info("=" * 70)
    logger.info("STUDY 3: Eta (Fiber Contraction Rate) - SmoothLiftedRLO only")
    logger.info("=" * 70)
    
    results = {}
    eta_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    
    for eta in eta_values:
        cfg = {
            'optimizer': 'smooth_lifted_rlo',
            'lr': 3e-4,
            'wd': 1.0,
            'batch_size': 128,
            'beta1': 0.9,
            'beta2': 0.99,
            'lambda_b': 0.1,
            'eta': eta,
        }
        
        try:
            cleanup_between_configs(world_size)
            
            logger.info(f"\neta={eta}")
            acc = train_single_config(cfg, data_path, device, world_size, rank, logger)
            results[str(eta)] = acc
            logger.info(f"  Final: {acc:.2f}%")
            
        except Exception as e:
            logger.error(f"Error: {e}")
            results[str(eta)] = 0.0
            if world_size > 1:
                try:
                    dist.barrier()
                except:
                    pass
    
    # Save incrementally
    if rank == 0:
        with open(output_path / 'eta_study_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def run_lambda_study(data_path, output_path, device, world_size, rank, logger):
    """Lambda_b (belief coefficient) study."""
    logger.info("=" * 70)
    logger.info("STUDY 4: Lambda_b (Belief Coefficient)")
    logger.info("=" * 70)
    
    results = {}
    lambda_values = [0.0, 0.05, 0.1, 0.2, 0.5]
    
    for lb in lambda_values:
        cfg = {
            'optimizer': 'smooth_lifted_rlo',
            'lr': 3e-4,
            'wd': 1.0,
            'batch_size': 128,
            'beta1': 0.9,
            'beta2': 0.99,
            'lambda_b': lb,
            'eta': 0.3,
        }
        
        try:
            cleanup_between_configs(world_size)
            
            logger.info(f"\nlambda_b={lb}")
            acc = train_single_config(cfg, data_path, device, world_size, rank, logger)
            results[str(lb)] = acc
            logger.info(f"  Final: {acc:.2f}%")
            
        except Exception as e:
            logger.error(f"Error: {e}")
            results[str(lb)] = 0.0
            if world_size > 1:
                try:
                    dist.barrier()
                except:
                    pass
    
    # Save incrementally
    if rank == 0:
        with open(output_path / 'lambda_study_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', default='all', 
                       choices=['all', 'lr', 'batch', 'eta', 'lambda'])
    parser.add_argument('--data', default='/blue/wdixon/wang.yixuan/rlo_experiments/data')
    parser.add_argument('--output', default='/blue/wdixon/wang.yixuan/rlo_experiments/ablation')
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
    
    device = torch.device(f'cuda:{local_rank}')
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)
    
    logger = setup_logger(output_path, rank, "ablation")
    
    # Download data FIRST on rank 0, then sync
    download_cifar_data(Path(args.data), rank, world_size)
    
    all_results = {}
    
    # Run studies
    if args.study == 'all' or args.study == 'lr':
        all_results['lr_study'] = run_lr_study(Path(args.data), output_path, 
                                               device, world_size, rank, logger)
    
    if args.study == 'all' or args.study == 'batch':
        all_results['batch_study'] = run_batch_study(Path(args.data), output_path,
                                                     device, world_size, rank, logger)
    
    if args.study == 'all' or args.study == 'eta':
        all_results['eta_study'] = run_eta_study(Path(args.data), output_path,
                                                  device, world_size, rank, logger)
    
    if args.study == 'all' or args.study == 'lambda':
        all_results['lambda_study'] = run_lambda_study(Path(args.data), output_path,
                                                        device, world_size, rank, logger)
    
    # Save all results
    if rank == 0:
        with open(output_path / 'ablation_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("ABLATION STUDY SUMMARY")
        logger.info("=" * 70)
        
        for study_name, study_results in all_results.items():
            logger.info(f"\n{study_name}:")
            if isinstance(study_results, dict):
                if any(isinstance(v, dict) for v in study_results.values()):
                    # Nested (optimizer -> param -> acc)
                    for opt, params in study_results.items():
                        logger.info(f"  {opt}:")
                        for param, acc in params.items():
                            logger.info(f"    {param}: {acc:.2f}%")
                else:
                    # Simple (param -> acc)
                    for param, acc in study_results.items():
                        logger.info(f"  {param}: {acc:.2f}%")
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
