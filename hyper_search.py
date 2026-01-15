"""
RLO Hyperparameter Search - 公平比较
在5090上跑，找到每个optimizer的最优超参数
"""
import os, gc, time, math, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR100
from torchvision.models import resnet50
from torch.optim import Optimizer
from tqdm import tqdm
import json

# ============= Optimizer Implementations =============
class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))
    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g['params']:
                if p.grad is None: continue
                if g['weight_decay'] != 0: p.mul_(1 - g['lr'] * g['weight_decay'])
                s = self.state[p]
                if len(s) == 0: s['m'] = torch.zeros_like(p)
                m = s['m']; b1, b2 = g['betas']
                p.add_((b1 * m + (1 - b1) * p.grad).sign_(), alpha=-g['lr'])
                m.mul_(b2).add_(p.grad, alpha=1 - b2)

class RLO(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, belief_coef=0.1, eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay, belief_coef=belief_coef, eps=eps))
    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                if g["weight_decay"] != 0: p.mul_(1 - g["lr"] * g["weight_decay"])
                s = self.state[p]
                if len(s) == 0: s["m"] = torch.zeros_like(p)
                m, gr = s["m"], p.grad; b1, b2 = g["betas"]
                c = b1 * m + (1 - b1) * gr; delta = gr - m
                p.add_(c.sign() + g["belief_coef"] * (delta / delta.norm().clamp(min=g["eps"])), alpha=-g["lr"])
                m.mul_(b2).add_(gr, alpha=1 - b2)

class SmoothLiftedRLO(Optimizer):
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.99, eta=0.3, weight_decay=0.0, lambda_b=0.1, eps=1e-8, gamma=5.0):
        super().__init__(params, dict(lr=lr, beta1=beta1, beta2=beta2, eta=eta, weight_decay=weight_decay, lambda_b=lambda_b, eps=eps, gamma=gamma))
        self.sqrt_dim = math.sqrt(sum(p.numel() for g in self.param_groups for p in g["params"]))
    @torch.no_grad()
    def step(self, closure=None):
        all_s, all_b, all_p = [], [], []
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                s = self.state[p]
                if len(s) == 0: s["m"], s["v"] = torch.zeros_like(p), torch.zeros_like(p)
                gr, m = p.grad, s["m"]
                c = g["beta1"] * m + (1 - g["beta1"]) * gr; ss = torch.tanh(g["gamma"] * c)
                delta = gr - m; b = g["lambda_b"] * (delta / delta.norm().clamp(min=g["eps"]))
                all_s.append(ss); all_b.append(b); all_p.append((p, g))
        if not all_p: return
        scale = self.sqrt_dim / sum((x * x).sum() for x in all_s).sqrt().clamp(min=1e-8)
        for (p, g), ss, b in zip(all_p, all_s, all_b):
            s = self.state[p]
            if g["weight_decay"] != 0: p.mul_(1 - g["lr"] * g["weight_decay"])
            d = scale * ss + b; s["v"].mul_(1 - g["eta"]).add_(d, alpha=g["eta"])
            p.add_(s["v"], alpha=-g["lr"]); s["m"].mul_(g["beta2"]).add_(p.grad, alpha=1 - g["beta2"])

# ============= 超参数搜索空间 =============
# 每个optimizer有自己合理的搜索范围！

SEARCH_SPACE = {
    'sgd': {
        'lr': [0.01, 0.03, 0.1, 0.3],
        'wd': [1e-4, 5e-4, 1e-3],
        'momentum': [0.9],
    },
    'adam': {
        'lr': [1e-4, 3e-4, 1e-3, 3e-3],
        'wd': [0, 1e-5, 1e-4],
    },
    'adamw': {
        'lr': [1e-3, 3e-3, 1e-2],
        'wd': [0.01, 0.05, 0.1],
    },
    'lion': {
        'lr': [1e-4, 3e-4, 1e-3],  # Lion一般用小lr
        'wd': [0.1, 0.5, 1.0],     # Lion用大wd
    },
    'rlo': {
        'lr': [1e-4, 3e-4, 1e-3, 3e-3],  # 可能需要比Lion更大的lr
        'wd': [0.1, 0.5, 1.0],
        'belief_coef': [0.01, 0.05, 0.1, 0.2],  # 关键参数！
    },
    'smooth_lifted_rlo': {
        'lr': [1e-4, 3e-4, 1e-3, 3e-3],
        'wd': [0.1, 0.5, 1.0],
        'lambda_b': [0.01, 0.05, 0.1],
        'eta': [0.1, 0.3, 0.5],
    },
}

def create_optimizer(model, name, cfg):
    p = model.parameters()
    if name == 'sgd':
        return torch.optim.SGD(p, lr=cfg['lr'], weight_decay=cfg['wd'], momentum=cfg.get('momentum', 0.9))
    elif name == 'adam':
        return torch.optim.Adam(p, lr=cfg['lr'], weight_decay=cfg['wd'])
    elif name == 'adamw':
        return torch.optim.AdamW(p, lr=cfg['lr'], weight_decay=cfg['wd'])
    elif name == 'lion':
        return Lion(p, lr=cfg['lr'], weight_decay=cfg['wd'])
    elif name == 'rlo':
        return RLO(p, lr=cfg['lr'], weight_decay=cfg['wd'], belief_coef=cfg.get('belief_coef', 0.1))
    elif name == 'smooth_lifted_rlo':
        return SmoothLiftedRLO(p, lr=cfg['lr'], weight_decay=cfg['wd'], 
                               lambda_b=cfg.get('lambda_b', 0.1), eta=cfg.get('eta', 0.3))

def train_short(opt_name, cfg, train_loader, test_loader, device, epochs=30):
    """短训练用于超参搜索"""
    model = resnet50(weights=None, num_classes=100).to(device)
    optimizer = create_optimizer(model, opt_name, cfg)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    # Cosine schedule
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * 3
    base_lr = cfg['lr']
    
    step = 0
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # LR schedule
            if step < warmup_steps:
                lr = base_lr * step / warmup_steps
            else:
                lr = base_lr * 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
            for g in optimizer.param_groups: g['lr'] = lr
            
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
        
        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = 100.0 * correct / total
        best_acc = max(best_acc, acc)
    
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return best_acc

def grid_search(opt_name, train_loader, test_loader, device, epochs=30):
    """对单个optimizer做grid search"""
    space = SEARCH_SPACE[opt_name]
    results = []
    
    # 生成所有组合
    from itertools import product
    keys = list(space.keys())
    values = [space[k] for k in keys]
    
    configs = []
    for combo in product(*values):
        cfg = dict(zip(keys, combo))
        configs.append(cfg)
    
    print(f"\n{'='*60}")
    print(f"Searching {opt_name}: {len(configs)} configurations")
    print(f"{'='*60}")
    
    for cfg in tqdm(configs, desc=opt_name):
        try:
            acc = train_short(opt_name, cfg, train_loader, test_loader, device, epochs)
            results.append({'config': cfg, 'acc': acc})
            tqdm.write(f"  {cfg} -> {acc:.2f}%")
        except Exception as e:
            tqdm.write(f"  {cfg} -> ERROR: {e}")
    
    # 找最优
    results.sort(key=lambda x: x['acc'], reverse=True)
    return results

def main():
    # Setup
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Data
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    
    train_ds = CIFAR100('./data', train=True, download=True, transform=train_tf)
    test_ds = CIFAR100('./data', train=False, download=True, transform=test_tf)
    
    # Windows: num_workers=0
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
    
    # 搜索每个optimizer
    all_results = {}
    
    # 你可以选择只搜索部分optimizer
    # optimizers_to_search = ['sgd', 'adamw', 'lion', 'rlo', 'smooth_lifted_rlo']
    
    optimizers_to_search = ['rlo', 'smooth_lifted_rlo']  # 只搜RLO相关
    
    for opt_name in optimizers_to_search:
        results = grid_search(opt_name, train_loader, test_loader, device, epochs=30)
        all_results[opt_name] = results
        
        print(f"\n{opt_name} Top 3:")
        for r in results[:3]:
            print(f"  {r['config']} -> {r['acc']:.2f}%")
    
    # 保存结果
    with open('hyperparam_search_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 最终比较
    print("\n" + "="*60)
    print("BEST CONFIG FOR EACH OPTIMIZER")
    print("="*60)
    for opt_name, results in all_results.items():
        if results:
            best = results[0]
            print(f"{opt_name:20s}: {best['acc']:.2f}% | {best['config']}")
    
    # 生成最优配置代码
    print("\n" + "="*60)
    print("OPTIMAL CONFIG (copy to your training script)")
    print("="*60)
    print("OPTIMIZERS = {")
    for opt_name, results in all_results.items():
        if results:
            cfg = results[0]['config']
            print(f"    '{opt_name}': {cfg},")
    print("}")

if __name__ == '__main__':
    main()
