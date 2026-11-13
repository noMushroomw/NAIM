import torch
from torch.optim import Optimizer

class EuclideanMetric:

    name = 'euclidean'

    def init_state(self, p):
        return {}

    @torch.no_grad()
    def apply(self, p, grad, state):
        return grad

class AdamMetric:

    name = 'adam'

    def __init__(self, beta3=0.999, eps=1e-8):
        self.beta3, self.eps = float(beta3), float(eps)

    def init_state(self, p):
        return {'v': torch.zeros_like(p), 'step': 0}

    @torch.no_grad()
    def apply(self, p, grad, state):
        state['step'] += 1
        v = state['v']
        v.mul_(self.beta3).addcmul_(grad, grad, value=1.0 - self.beta3)
        denom = (v / (1.0 - self.beta3 ** state['step'])).sqrt().add_(self.eps)
        return grad / denom

class RMSPropMetric:

    name = 'rmsprop'

    def __init__(self, beta3=0.99, eps=1e-8):
        self.beta3, self.eps = float(beta3), float(eps)

    def init_state(self, p):
        return {'v': torch.zeros_like(p)}

    @torch.no_grad()
    def apply(self, p, grad, state):
        v = state['v']
        v.mul_(self.beta3).addcmul_(grad, grad, value=1.0 - self.beta3)
        return grad / v.sqrt().add_(self.eps)

class AdaGradMetric:

    name = 'adagrad'

    def __init__(self, eps=1e-10):
        self.eps = float(eps)

    def init_state(self, p):
        return {'G': torch.zeros_like(p)}

    @torch.no_grad()
    def apply(self, p, grad, state):
        G = state['G']
        G.addcmul_(grad, grad, value=1.0)
        return grad / G.sqrt().add_(self.eps)

class LayerwiseMetric:

    name = 'layerwise'

    def __init__(self, eps=1e-8):
        self.eps = float(eps)

    def init_state(self, p):
        return {}

    @torch.no_grad()
    def apply(self, p, grad, state):
        return grad * (1.0 / (grad.norm() + self.eps))

METRICS = ('euclidean', 'adam', 'rmsprop', 'adagrad', 'layerwise')

def build_metric(name, beta3_adam=0.999, beta3_rmsprop=0.99, eps=1e-8):
    if name == 'euclidean':
        return EuclideanMetric()
    if name == 'adam':
        return AdamMetric(beta3=beta3_adam, eps=eps)
    if name == 'rmsprop':
        return RMSPropMetric(beta3=beta3_rmsprop, eps=eps)
    if name == 'adagrad':
        return AdaGradMetric(eps=1e-10)
    if name == 'layerwise':
        return LayerwiseMetric(eps=eps)
    raise ValueError(f'unknown metric {name!r}; choose from {METRICS}')

class MetricRLO(Optimizer):

    def __init__(self, params, metric, lr=1e-4, betas=(0.9, 0.99), gamma=5.0,
                 lambda_b=0.1, eps=1e-8, weight_decay=0.0):
        if metric is None:
            raise ValueError('a metric object must be supplied')
        defaults = dict(lr=lr, betas=betas, gamma=gamma, lambda_b=lambda_b, eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.metric = metric

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
                g = p.grad
                if torch.isnan(g).any() or torch.isinf(g).any():
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p)
                    state['metric'] = self.metric.init_state(p)
                g_tilde = self.metric.apply(p, g, state['metric'])
                m = state['m']
                c = beta1 * m + (1.0 - beta1) * g_tilde
                delta = g_tilde - m
                direction = torch.tanh(group['gamma'] * c) + group['lambda_b'] * (delta / delta.norm().clamp(min=group['eps']))
                p.add_(direction, alpha=-group['lr'])
                if group['weight_decay'] > 0.0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])
                m.mul_(beta2).add_(g_tilde, alpha=1.0 - beta2)
        return loss
