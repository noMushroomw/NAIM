#!/usr/bin/env python3
"""
RLO Optimizer Family + Baselines
================================
Implements: AdamW, Lion, RLO, RLO_LambdaA, SmoothLiftedRLO
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
                state = self.state[p]
                
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                
                # Update with interpolated momentum
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                
                # Decoupled weight decay
                if group['weight_decay'] > 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])
                
                # Update momentum
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        
        return loss


class RLO(Optimizer):
    """RLO: Riemannian Lifted Optimizer."""
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
                state = self.state[p]
                
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                lambda_b = group['lambda_b']
                
                # c = interpolated momentum (like Lion)
                c = exp_avg * beta1 + grad * (1 - beta1)
                
                # δ = gradient - momentum (forcing term)
                delta = grad - exp_avg
                delta_norm = delta.norm().clamp(min=1e-8)
                
                # Update: sign(c) + λ_b * δ/||δ||
                update = torch.sign(c) + lambda_b * delta / delta_norm
                p.add_(update, alpha=-group['lr'])
                
                # Decoupled weight decay
                if group['weight_decay'] > 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])
                
                # Update momentum
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        
        return loss


class RLO_LambdaA(Optimizer):
    """RLO with global adaptive scaling (Lambda_A variant)."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0,
                 lambda_b=0.1, gamma=0.5, eps=1e-8):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                       lambda_b=lambda_b, gamma=gamma, eps=eps)
        super().__init__(params, defaults)
        self.global_state = {'smooth_pre': None}
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Compute global gradient norm
        total_norm_sq = 0.0
        total_dim = 0
        all_grads = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    total_norm_sq += p.grad.norm().pow(2)
                    total_dim += p.numel()
                    all_grads.append((group, p))
        
        if total_dim == 0:
            return loss
        
        # Global scaling: sqrt(D) / ||smooth_pre||
        grad_norm = math.sqrt(total_norm_sq)
        
        if self.global_state['smooth_pre'] is None:
            self.global_state['smooth_pre'] = grad_norm
        else:
            self.global_state['smooth_pre'] = 0.99 * self.global_state['smooth_pre'] + 0.01 * grad_norm
        
        sqrt_dim = math.sqrt(total_dim)
        global_scale = sqrt_dim / max(self.global_state['smooth_pre'], 1e-8)
        
        for group, p in all_grads:
            grad = p.grad
            state = self.state[p]
            
            if len(state) == 0:
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            beta1, beta2 = group['betas']
            gamma = group['gamma']
            eps = group['eps']
            lambda_b = group['lambda_b']
            
            # Update second moment
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
            # c = interpolated momentum
            c = exp_avg * beta1 + grad * (1 - beta1)
            
            # Soft sign with normalization
            sqrt_v = exp_avg_sq.sqrt().add_(eps)
            update = torch.tanh(gamma * c / sqrt_v)
            
            # Add belief term
            delta = grad - exp_avg
            delta_norm = delta.norm().clamp(min=1e-8)
            belief = lambda_b * delta / delta_norm
            
            # Apply global scale
            full_update = global_scale * (update + belief)
            p.add_(full_update, alpha=-group['lr'])
            
            # Decoupled weight decay
            if group['weight_decay'] > 0:
                p.add_(p, alpha=-group['lr'] * group['weight_decay'])
            
            # Update momentum
            exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        
        return loss


class SmoothLiftedRLO(Optimizer):
    """Smooth Lifted RLO with explicit velocity tracking."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0,
                 eta=0.5, lambda_b=0.1):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                       eta=eta, lambda_b=lambda_b)
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
                state = self.state[p]
                
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['velocity'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                velocity = state['velocity']
                beta1, beta2 = group['betas']
                eta = group['eta']
                lambda_b = group['lambda_b']
                
                # d = interpolated direction (target velocity)
                d = exp_avg * beta1 + grad * (1 - beta1)
                
                # Fiber contraction: v ← (1-η)v + ηd
                velocity.mul_(1 - eta).add_(d, alpha=eta)
                
                # Belief term
                delta = grad - exp_avg
                delta_norm = delta.norm().clamp(min=1e-8)
                belief = lambda_b * delta / delta_norm
                
                # Update: sign(v) + belief
                update = torch.sign(velocity) + belief
                p.add_(update, alpha=-group['lr'])
                
                # Decoupled weight decay
                if group['weight_decay'] > 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])
                
                # Update momentum
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        
        return loss


def create_optimizer(name, params, lr, weight_decay=0.0, betas=None, **kwargs):
    """Factory function to create optimizer by name."""
    name = name.lower()
    
    if betas is None:
        betas = (0.9, 0.99)
    
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    elif name == 'lion':
        return Lion(params, lr=lr, weight_decay=weight_decay, betas=betas)
    elif name == 'rlo':
        return RLO(params, lr=lr, weight_decay=weight_decay, betas=betas, 
                   lambda_b=kwargs.get('lambda_b', 0.1))
    elif name == 'rlo_lambda_a':
        return RLO_LambdaA(params, lr=lr, weight_decay=weight_decay, betas=betas,
                          lambda_b=kwargs.get('lambda_b', 0.1))
    elif name == 'smooth_lifted_rlo':
        return SmoothLiftedRLO(params, lr=lr, weight_decay=weight_decay, betas=betas,
                              eta=kwargs.get('eta', 0.5), lambda_b=kwargs.get('lambda_b', 0.1))
    else:
        raise ValueError(f"Unknown optimizer: {name}")
