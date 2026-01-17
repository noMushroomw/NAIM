#!/usr/bin/env python3
"""
Ablation Studies for RLO Optimizer Family
Studies:
1. Learning Rate Sensitivity
2. Batch Size with Linear LR Scaling
3. Eta (Fiber Contraction Rate) - SmoothLiftedRLO only
4. Lambda_b (Belief Coefficient)
"""
import os, gc, time, math, json, logging, argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.models import resnet18
import warnings; warnings.filterwarnings('ignore')

try:
    from optimizers import Lion, RLO, SmoothLiftedRLO, create_optimizer
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from optimizers import Lion, RLO, SmoothLiftedRLO, create_optimizer


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


def get_cifar_loaders(data_path, batch_size, rank, world_size):
    """Get CIFAR-100 data loaders."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    ])
    
    train_dataset = CIFAR100(root=data_path, train=True, download=True, transform=train_transform)
    val_dataset = CIFAR100(root=data_path, train=False, download=True, transform=val_transform)
    
    train_sampler = DistributedSampler(train_dataset, world_size, rank) if world_size > 1 else None
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=(train_sampler is None),
                             sampler=train_sampler, num_workers=4, pin_memory=True,
                             drop_last=True, persistent_workers=True)
    
    val_loader = DataLoader(val_dataset, batch_size * 2, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, train_sampler


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return 100.0 * correct / total


def train_single_run(model, optimizer, train_loader, val_loader, train_sampler,
                    epochs, device, rank, logger=None, log_prefix=""):
    """Train for specified epochs and return best accuracy."""
    scaler = torch.amp.GradScaler()
    best_acc = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # Evaluate
        if rank == 0 and epoch % 10 == 0:
            acc = evaluate(model, val_loader, device)
            if acc > best_acc:
                best_acc = acc
            if logger:
                logger.info(f"{log_prefix}  E{epoch}: acc={acc:.2f}% best={best_acc:.2f}%")
    
    # Final evaluation
    if rank == 0:
        acc = evaluate(model, val_loader, device)
        if acc > best_acc:
            best_acc = acc
        if logger:
            logger.info(f"{log_prefix}  Final: {best_acc:.2f}%")
    
    return best_acc


def study_learning_rate(data_path, output_dir, rank, world_size, device, logger):
    """Study 1: Learning rate sensitivity across optimizers."""
    if rank == 0:
        logger.info("=" * 70)
        logger.info("STUDY 1: Learning Rate Sensitivity")
        logger.info("=" * 70)
    
    learning_rates = [5e-5, 1e-4, 3e-4, 5e-4, 1e-3]
    optimizers_to_test = ['adamw', 'lion', 'rlo', 'smooth_lifted_rlo']
    epochs = 30
    batch_size = 128
    
    results = {}
    
    for opt_name in optimizers_to_test:
        results[opt_name] = {}
        
        for lr in learning_rates:
            if rank == 0:
                logger.info(f"\n{opt_name} | lr={lr:.2e}")
            
            # Fresh model and data loaders
            train_loader, val_loader, train_sampler = get_cifar_loaders(
                data_path, batch_size, rank, world_size)
            
            model = resnet18(num_classes=100).to(device)
            if world_size > 1:
                model = DDP(model, device_ids=[rank])
            
            # Create optimizer
            if opt_name == 'adamw':
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
            elif opt_name == 'lion':
                optimizer = Lion(model.parameters(), lr=lr, weight_decay=0.1)
            elif opt_name == 'rlo':
                optimizer = RLO(model.parameters(), lr=lr, weight_decay=0.1)
            elif opt_name == 'smooth_lifted_rlo':
                optimizer = SmoothLiftedRLO(model.parameters(), lr=lr, weight_decay=0.1)
            
            acc = train_single_run(model, optimizer, train_loader, val_loader,
                                  train_sampler, epochs, device, rank, logger)
            
            results[opt_name][str(lr)] = acc
            
            del model, optimizer, train_loader, val_loader
            gc.collect()
            torch.cuda.empty_cache()
    
    return results


def study_batch_size(data_path, output_dir, rank, world_size, device, logger):
    """Study 2: Batch size sensitivity with linear LR scaling."""
    if rank == 0:
        logger.info("=" * 70)
        logger.info("STUDY 2: Batch Size Sensitivity (Linear LR Scaling)")
        logger.info("=" * 70)
    
    batch_sizes = [64, 128, 256, 512]
    base_lr = 3e-4
    optimizers_to_test = ['adamw', 'lion', 'rlo', 'smooth_lifted_rlo']
    epochs = 30
    
    results = {}
    
    for opt_name in optimizers_to_test:
        results[opt_name] = {}
        
        for batch_size in batch_sizes:
            # Linear LR scaling
            lr = base_lr * (batch_size / 128)
            
            if rank == 0:
                logger.info(f"\n{opt_name} | batch={batch_size} lr={lr:.2e}")
            
            train_loader, val_loader, train_sampler = get_cifar_loaders(
                data_path, batch_size, rank, world_size)
            
            model = resnet18(num_classes=100).to(device)
            if world_size > 1:
                model = DDP(model, device_ids=[rank])
            
            if opt_name == 'adamw':
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
            elif opt_name == 'lion':
                optimizer = Lion(model.parameters(), lr=lr, weight_decay=0.1)
            elif opt_name == 'rlo':
                optimizer = RLO(model.parameters(), lr=lr, weight_decay=0.1)
            elif opt_name == 'smooth_lifted_rlo':
                optimizer = SmoothLiftedRLO(model.parameters(), lr=lr, weight_decay=0.1)
            
            acc = train_single_run(model, optimizer, train_loader, val_loader,
                                  train_sampler, epochs, device, rank, logger)
            
            results[opt_name][str(batch_size)] = acc
            
            del model, optimizer, train_loader, val_loader
            gc.collect()
            torch.cuda.empty_cache()
    
    return results


def study_eta(data_path, output_dir, rank, world_size, device, logger):
    """Study 3: Eta (fiber contraction rate) for SmoothLiftedRLO."""
    if rank == 0:
        logger.info("=" * 70)
        logger.info("STUDY 3: Eta (Fiber Contraction Rate) - SmoothLiftedRLO only")
        logger.info("=" * 70)
    
    eta_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    epochs = 30
    batch_size = 128
    lr = 3e-4
    
    results = {}
    
    for eta in eta_values:
        if rank == 0:
            logger.info(f"\neta={eta}")
        
        train_loader, val_loader, train_sampler = get_cifar_loaders(
            data_path, batch_size, rank, world_size)
        
        model = resnet18(num_classes=100).to(device)
        if world_size > 1:
            model = DDP(model, device_ids=[rank])
        
        optimizer = SmoothLiftedRLO(model.parameters(), lr=lr, weight_decay=0.1, eta=eta)
        
        acc = train_single_run(model, optimizer, train_loader, val_loader,
                              train_sampler, epochs, device, rank, logger)
        
        results[str(eta)] = acc
        
        del model, optimizer, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()
    
    return results


def study_lambda_b(data_path, output_dir, rank, world_size, device, logger):
    """Study 4: Lambda_b (belief coefficient) for RLO."""
    if rank == 0:
        logger.info("=" * 70)
        logger.info("STUDY 4: Lambda_b (Belief Coefficient)")
        logger.info("=" * 70)
    
    lambda_values = [0.0, 0.05, 0.1, 0.2, 0.5]
    epochs = 30
    batch_size = 128
    lr = 3e-4
    
    results = {}
    
    for lambda_b in lambda_values:
        if rank == 0:
            logger.info(f"\nlambda_b={lambda_b}")
        
        train_loader, val_loader, train_sampler = get_cifar_loaders(
            data_path, batch_size, rank, world_size)
        
        model = resnet18(num_classes=100).to(device)
        if world_size > 1:
            model = DDP(model, device_ids=[rank])
        
        optimizer = RLO(model.parameters(), lr=lr, weight_decay=0.1, lambda_b=lambda_b)
        
        acc = train_single_run(model, optimizer, train_loader, val_loader,
                              train_sampler, epochs, device, rank, logger)
        
        results[str(lambda_b)] = acc
        
        del model, optimizer, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=str, default='all',
                       choices=['all', 'lr', 'batch', 'eta', 'lambda'])
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    print(f"Starting ablation studies with args: {args}")
    
    # Distributed setup
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if world_size > 1:
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
    
    device = torch.device(f'cuda:{local_rank}')
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(output_dir, rank, 'ablation')
    
    all_results = {}
    
    # Run studies based on selection
    studies_to_run = []
    if args.study == 'all':
        studies_to_run = ['lr', 'batch', 'eta', 'lambda']
    else:
        studies_to_run = [args.study]
    
    for study in studies_to_run:
        if study == 'lr':
            all_results['lr_study'] = study_learning_rate(
                Path(args.data), output_dir, rank, world_size, device, logger)
        elif study == 'batch':
            all_results['batch_study'] = study_batch_size(
                Path(args.data), output_dir, rank, world_size, device, logger)
        elif study == 'eta':
            all_results['eta_study'] = study_eta(
                Path(args.data), output_dir, rank, world_size, device, logger)
        elif study == 'lambda':
            all_results['lambda_study'] = study_lambda_b(
                Path(args.data), output_dir, rank, world_size, device, logger)
        
        # Save intermediate results
        if rank == 0:
            with open(output_dir / 'ablation_results.json', 'w') as f:
                json.dump(all_results, f, indent=2)
    
    # Print summary
    if rank == 0:
        logger.info("=" * 70)
        logger.info("ABLATION STUDY SUMMARY")
        logger.info("=" * 70)
        
        for study_name, results in all_results.items():
            logger.info(f"\n{study_name}:")
            if isinstance(results, dict):
                # Check if it's nested (optimizer -> values) or flat (values)
                first_val = next(iter(results.values()))
                if isinstance(first_val, dict):
                    # Nested: optimizer -> {param: acc}
                    for opt_name, opt_results in results.items():
                        logger.info(f"  {opt_name}:")
                        for param, acc in opt_results.items():
                            logger.info(f"    {param}: {acc:.2f}%")
                else:
                    # Flat: {param: acc}
                    for param, acc in results.items():
                        logger.info(f"  {param}: {acc:.2f}%")
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
