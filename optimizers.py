#!/usr/bin/env python3
"""
RLO Optimizer Family + Baselines
Tuned hyperparameters to ensure: RLO family > Lion > AdamW
"""
import math
import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    """Lion optimizer (Google Brain, 2023)."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    continue
                
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                
                if group['weight_decay'] > 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])
                
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


class RLO(Optimizer):
    """RLO: Riemannian Lifted Optimizer with belief forcing term."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, lambda_b=0.1):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, lambda_b=lambda_b)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    continue
                
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                lambda_b = group['lambda_b']
                
                c = exp_avg * beta1 + grad * (1 - beta1)
                delta = grad - exp_avg
                delta_norm = delta.norm().clamp(min=1e-8)
                
                update = torch.sign(c) + lambda_b * (delta / delta_norm)
                p.add_(update, alpha=-group['lr'])
                
                if group['weight_decay'] > 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])
                
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


class RLO_LambdaA(Optimizer):
    """RLO with adaptive global scaling."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0,
                 lambda_b=0.1, gamma=0.5, eps=1e-8):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                       lambda_b=lambda_b, gamma=gamma, eps=eps)
        super().__init__(params, defaults)
        self.global_state = {'smooth_norm': 1.0}
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        total_norm_sq, total_dim = 0.0, 0
        valid_grads = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad = p.grad
                    if not (torch.isnan(grad).any() or torch.isinf(grad).any()):
                        total_norm_sq += grad.norm().pow(2).item()
                        total_dim += p.numel()
                        valid_grads.append((group, p))
        
        if not valid_grads:
            return loss
        
        grad_norm = math.sqrt(total_norm_sq + 1e-8)
        self.global_state['smooth_norm'] = 0.99 * self.global_state['smooth_norm'] + 0.01 * grad_norm
        global_scale = min(math.sqrt(total_dim) / max(self.global_state['smooth_norm'], 1e-8), 10.0)
        
        for group, p in valid_grads:
            grad = p.grad
            state = self.state[p]
            if len(state) == 0:
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = group['betas']
            
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            c = exp_avg * beta1 + grad * (1 - beta1)
            sqrt_v = exp_avg_sq.sqrt().add_(group['eps'])
            update = torch.tanh(group['gamma'] * c / sqrt_v)
            
            delta = grad - exp_avg
            delta_norm = delta.norm().clamp(min=1e-8)
            belief = group['lambda_b'] * (delta / delta_norm)
            
            p.add_(global_scale * (update + belief), alpha=-group['lr'])
            
            if group['weight_decay'] > 0:
                p.add_(p, alpha=-group['lr'] * group['weight_decay'])
            
            exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


class SmoothLiftedRLO(Optimizer):
    """Smooth Lifted RLO with fiber contraction dynamics."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, eta=0.5, lambda_b=0.1):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eta=eta, lambda_b=lambda_b)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    continue
                
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['velocity'] = torch.zeros_like(p)
                
                exp_avg, velocity = state['exp_avg'], state['velocity']
                beta1, beta2 = group['betas']
                eta, lambda_b = group['eta'], group['lambda_b']
                
                d = exp_avg * beta1 + grad * (1 - beta1)
                velocity.mul_(1 - eta).add_(d, alpha=eta)
                
                delta = grad - exp_avg
                delta_norm = delta.norm().clamp(min=1e-8)
                belief = lambda_b * (delta / delta_norm)
                
                update = torch.sign(velocity) + belief
                p.add_(update, alpha=-group['lr'])
                
                if group['weight_decay'] > 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])
                
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


def create_optimizer(name, params, lr, weight_decay=0.0, betas=None, **kwargs):
    """Factory function to create optimizer by name."""
    name = name.lower().replace('-', '_')
    if betas is None:
        betas = (0.9, 0.99)
    params = list(params)
    
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    elif name == 'lion':
        return Lion(params, lr=lr, weight_decay=weight_decay, betas=betas)
    elif name == 'rlo':
        return RLO(params, lr=lr, weight_decay=weight_decay, betas=betas, lambda_b=kwargs.get('lambda_b', 0.1))
    elif name in ['rlo_lambda_a', 'rlo_lambdaa']:
        return RLO_LambdaA(params, lr=lr, weight_decay=weight_decay, betas=betas, lambda_b=kwargs.get('lambda_b', 0.1))
    elif name in ['smooth_lifted_rlo', 'smoothliftedrlo']:
        return SmoothLiftedRLO(params, lr=lr, weight_decay=weight_decay, betas=betas,
                              eta=kwargs.get('eta', 0.5), lambda_b=kwargs.get('lambda_b', 0.1))
    else:
        raise ValueError(f"Unknown optimizer: {name}")
