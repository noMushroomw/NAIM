import torch
from torch.optim import Optimizer

class Lion(Optimizer):

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
            beta1, beta2 = group['betas']
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
                p.add_(torch.sign(exp_avg * beta1 + grad * (1 - beta1)), alpha=-group['lr'])
                if group['weight_decay'] > 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss
