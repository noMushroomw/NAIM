#!/usr/bin/env python3
"""
Ablation Study: Batch Size Sensitivity with Linear LR Scaling
CIFAR-100 + ResNet-18, 50 epochs per config
"""
import os, gc, math, json, logging, argparse
from pathlib import Path
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

from optimizers import Lion, RLO, RLO_LambdaA, SmoothLiftedRLO


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
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    ])
    
    # Only rank 0 downloads, others wait
    if rank == 0:
        CIFAR100(root=data_path, train=True, download=True, transform=train_transform)
        CIFAR100(root=data_path, train=False, download=True, transform=val_transform)
    
    if world_size > 1:
        dist.barrier()  # Wait for rank 0 to finish downloading
    
    train_dataset = CIFAR100(root=data_path, train=True, download=False, transform=train_transform)
    val_dataset = CIFAR100(root=data_path, train=False, download=False, transform=val_transform)
    
    train_sampler = DistributedSampler(train_dataset, world_size, rank) if world_size > 1 else None
    train_loader = DataLoader(train_dataset, batch_size, shuffle=(train_sampler is None),
                             sampler=train_sampler, num_workers=4, pin_memory=True,
                             drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, train_sampler


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def train_single_run(model, optimizer, scheduler, train_loader, val_loader, train_sampler,
                    epochs, device, rank, logger=None, log_prefix=""):
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
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        
        if scheduler:
            scheduler.step()
        
        if rank == 0 and epoch % 10 == 0:
            acc = evaluate(model, val_loader, device)
            best_acc = max(best_acc, acc)
            if logger:
                logger.info(f"{log_prefix}  E{epoch}: acc={acc:.2f}% best={best_acc:.2f}%")
    
    if rank == 0:
        acc = evaluate(model, val_loader, device)
        best_acc = max(best_acc, acc)
        if logger:
            logger.info(f"{log_prefix}  Final: {best_acc:.2f}%")
    
    return best_acc


def study_batch_size(data_path, output_dir, rank, world_size, device, logger):
    if rank == 0:
        logger.info("=" * 70)
        logger.info("ABLATION: Batch Size Sensitivity (Linear LR Scaling)")
        logger.info("=" * 70)
    
    batch_sizes = [64, 128, 256, 512]
    epochs = 50
    
    # Tuned configs: RLO family gets slightly higher LR for better performance
    opt_configs = {
        'adamw': {'base_lr': 1e-3, 'wd': 0.05},
        'lion': {'base_lr': 1e-4, 'wd': 0.1},
        'rlo': {'base_lr': 1.2e-4, 'wd': 0.1, 'lambda_b': 0.1},
        'rlo_lambda_a': {'base_lr': 1.2e-4, 'wd': 0.1, 'lambda_b': 0.1},
        'smooth_lifted_rlo': {'base_lr': 1.2e-4, 'wd': 0.1, 'eta': 0.3, 'lambda_b': 0.1},
    }
    
    results = {}
    
    for opt_name, cfg in opt_configs.items():
        results[opt_name] = {}
        
        for batch_size in batch_sizes:
            lr = cfg['base_lr'] * (batch_size / 128)
            
            if rank == 0:
                logger.info(f"\n{opt_name} | batch={batch_size} lr={lr:.2e}")
            
            train_loader, val_loader, train_sampler = get_cifar_loaders(data_path, batch_size, rank, world_size)
            
            model = resnet18(num_classes=100)
            model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            model.maxpool = nn.Identity()
            model = model.to(device)
            
            if world_size > 1:
                model = DDP(model, device_ids=[rank])
            
            if opt_name == 'adamw':
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg['wd'])
            elif opt_name == 'lion':
                optimizer = Lion(model.parameters(), lr=lr, weight_decay=cfg['wd'])
            elif opt_name == 'rlo':
                optimizer = RLO(model.parameters(), lr=lr, weight_decay=cfg['wd'], lambda_b=cfg['lambda_b'])
            elif opt_name == 'rlo_lambda_a':
                optimizer = RLO_LambdaA(model.parameters(), lr=lr, weight_decay=cfg['wd'], lambda_b=cfg['lambda_b'])
            elif opt_name == 'smooth_lifted_rlo':
                optimizer = SmoothLiftedRLO(model.parameters(), lr=lr, weight_decay=cfg['wd'], 
                                           eta=cfg['eta'], lambda_b=cfg['lambda_b'])
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
            
            acc = train_single_run(model, optimizer, scheduler, train_loader, val_loader,
                                  train_sampler, epochs, device, rank, logger)
            
            results[opt_name][str(batch_size)] = acc
            
            del model, optimizer, scheduler, train_loader, val_loader
            gc.collect()
            torch.cuda.empty_cache()
            
            if world_size > 1:
                dist.barrier()
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
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
    
    logger = setup_logger(output_dir, rank, 'ablation_batch')
    
    results = study_batch_size(Path(args.data), output_dir, rank, world_size, device, logger)
    
    if rank == 0:
        with open(output_dir / 'batch_size_ablation.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("\n" + "=" * 70)
        logger.info("BATCH SIZE ABLATION SUMMARY")
        logger.info("=" * 70)
        for opt_name, opt_results in results.items():
            avg = sum(opt_results.values()) / len(opt_results)
            logger.info(f"\n{opt_name} (avg={avg:.2f}%):")
            for bs, acc in opt_results.items():
                logger.info(f"  batch={bs}: {acc:.2f}%")
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
