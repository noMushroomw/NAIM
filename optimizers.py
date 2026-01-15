#!/usr/bin/env python3
"""Optimizers: Lion, RLO, RLO_LambdaA, SmoothLiftedRLO"""
import math
import torch
from torch.optim import Optimizer

class Lion(Optimizer):
    """Lion optimizer from Google Brain paper."""
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
                
                # Decoupled weight decay
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Initialize state
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                
                # Update step: sign(beta1*m + (1-beta1)*g)
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1).sign_()
                p.add_(update, alpha=-group['lr'])
                
                # Update EMA: m = beta2*m + (1-beta2)*g
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        
        return loss


class RLO(Optimizer):
    """RLO: sign + belief correction."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, 
                 belief_coef=0.1, eps=1e-8):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                       belief_coef=belief_coef, eps=eps)
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
                
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                belief = group['belief_coef']
                eps = group['eps']
                
                # c = beta1*m + (1-beta1)*g
                c = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                
                # delta = g - m
                delta = grad - exp_avg
                delta_norm = delta.norm().clamp(min=eps)
                
                # d = sign(c) + belief * delta/||delta||
                update = c.sign().add_(delta.div(delta_norm), alpha=belief)
                p.add_(update, alpha=-group['lr'])
                
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        
        return loss


class RLO_LambdaA(Optimizer):
    """RLO with Lambda preconditioner: tanh + sqrt(s) scaling + belief."""
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.99, beta3=0.999,
                 weight_decay=0.0, lambda_b=0.1, eps=1e-8, gamma=5.0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, beta3=beta3,
                       weight_decay=weight_decay, lambda_b=lambda_b, 
                       eps=eps, gamma=gamma)
        super().__init__(params, defaults)
        
        # Compute total dimension for global scaling
        total_dim = sum(p.numel() for group in self.param_groups 
                       for p in group['params'] if p.requires_grad)
        self.sqrt_dim = math.sqrt(total_dim)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # First pass: compute smooth_pre and belief for all params
        all_smooth_pre = []
        all_belief = []
        all_params = []
        
        for group in self.param_groups:
            eps = group['eps']
            gamma = group['gamma']
            beta1, beta3 = group['beta1'], group['beta3']
            lambda_b = group['lambda_b']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                # Update second moment: s = beta3*s + (1-beta3)*g^2
                exp_avg_sq.mul_(beta3).addcmul_(grad, grad, value=1 - beta3)
                
                # c = beta1*m + (1-beta1)*g
                c = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                
                # smooth_pre = tanh(gamma*c) / (sqrt(s) + eps)
                smooth_pre = torch.tanh(gamma * c) / (exp_avg_sq.sqrt() + eps)
                
                # belief = lambda_b * delta/||delta||
                delta = grad - exp_avg
                delta_norm = delta.norm().clamp(min=eps)
                belief = lambda_b * delta / delta_norm
                
                all_smooth_pre.append(smooth_pre)
                all_belief.append(belief)
                all_params.append((p, group, state, grad))
        
        if not all_params:
            return loss
        
        # Compute global scale: sqrt(D) / ||smooth_pre||
        smooth_norm_sq = sum((sp * sp).sum() for sp in all_smooth_pre)
        scale = self.sqrt_dim / smooth_norm_sq.sqrt().clamp(min=1e-8)
        
        # Second pass: apply updates
        for (p, group, state, grad), smooth_pre, belief in zip(all_params, all_smooth_pre, all_belief):
            if group['weight_decay'] != 0:
                p.mul_(1 - group['lr'] * group['weight_decay'])
            
            # d = scale * smooth_pre + belief
            update = scale * smooth_pre + belief
            p.add_(update, alpha=-group['lr'])
            
            # Update EMA
            state['exp_avg'].mul_(group['beta2']).add_(grad, alpha=1 - group['beta2'])
        
        return loss


class SmoothLiftedRLO(Optimizer):
    """Smooth Lifted RLO: maintains explicit velocity v with fiber contraction."""
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.99, eta=0.3,
                 weight_decay=0.0, lambda_b=0.1, eps=1e-8, gamma=5.0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eta=eta,
                       weight_decay=weight_decay, lambda_b=lambda_b,
                       eps=eps, gamma=gamma)
        super().__init__(params, defaults)
        
        total_dim = sum(p.numel() for group in self.param_groups 
                       for p in group['params'] if p.requires_grad)
        self.sqrt_dim = math.sqrt(total_dim)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # First pass: compute smooth and belief
        all_smooth = []
        all_belief = []
        all_params = []
        
        for group in self.param_groups:
            eps = group['eps']
            gamma = group['gamma']
            beta1 = group['beta1']
            lambda_b = group['lambda_b']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['velocity'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                
                # c = beta1*m + (1-beta1)*g
                c = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                
                # smooth = tanh(gamma*c)
                smooth = torch.tanh(gamma * c)
                
                # belief = lambda_b * delta/||delta||
                delta = grad - exp_avg
                delta_norm = delta.norm().clamp(min=eps)
                belief = lambda_b * delta / delta_norm
                
                all_smooth.append(smooth)
                all_belief.append(belief)
                all_params.append((p, group, state, grad))
        
        if not all_params:
            return loss
        
        # Global scale
        smooth_norm_sq = sum((s * s).sum() for s in all_smooth)
        scale = self.sqrt_dim / smooth_norm_sq.sqrt().clamp(min=1e-8)
        
        # Second pass: apply updates with fiber contraction
        for (p, group, state, grad), smooth, belief in zip(all_params, all_smooth, all_belief):
            if group['weight_decay'] != 0:
                p.mul_(1 - group['lr'] * group['weight_decay'])
            
            # d = scale * smooth + belief
            d = scale * smooth + belief
            
            # Fiber contraction: v = (1-eta)*v + eta*d
            eta = group['eta']
            velocity = state['velocity']
            velocity.mul_(1 - eta).add_(d, alpha=eta)
            
            # Update: theta = theta - lr * v
            p.add_(velocity, alpha=-group['lr'])
            
            # Update EMA
            state['exp_avg'].mul_(group['beta2']).add_(grad, alpha=1 - group['beta2'])
        
        return loss


def create_optimizer(model, name, cfg, param_groups=None):
    """Factory function for creating optimizers."""
    if param_groups is None:
        params = model.parameters()
    else:
        params = param_groups
    
    lr = cfg['lr']
    wd = cfg.get('wd', 0.0)
    
    if name == 'adamw':
        betas = cfg.get('betas', (0.9, 0.999))
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)
    elif name == 'lion':
        betas = cfg.get('betas', (0.9, 0.99))
        return Lion(params, lr=lr, weight_decay=wd, betas=betas)
    elif name == 'rlo':
        betas = cfg.get('betas', (0.9, 0.99))
        belief = cfg.get('belief_coef', 0.1)
        return RLO(params, lr=lr, weight_decay=wd, betas=betas, belief_coef=belief)
    elif name == 'rlo_lambda_a':
        beta1 = cfg.get('beta1', 0.9)
        beta2 = cfg.get('beta2', 0.99)
        lambda_b = cfg.get('lambda_b', 0.1)
        return RLO_LambdaA(params, lr=lr, weight_decay=wd, 
                          beta1=beta1, beta2=beta2, lambda_b=lambda_b)
    elif name == 'smooth_lifted_rlo':
        beta1 = cfg.get('beta1', 0.9)
        beta2 = cfg.get('beta2', 0.99)
        eta = cfg.get('eta', 0.3)
        lambda_b = cfg.get('lambda_b', 0.1)
        return SmoothLiftedRLO(params, lr=lr, weight_decay=wd,
                              beta1=beta1, beta2=beta2, eta=eta, lambda_b=lambda_b)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
