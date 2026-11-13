import inspect

import torch

from .lion import Lion
from .metric_rlo import METRICS, MetricRLO, build_metric
from .rlo import RLO, RLOLambda, RLOLifted

__all__ = ['RLO', 'RLOLambda', 'RLOLifted', 'Lion', 'MetricRLO', 'build_metric', 'METRICS',
           'build_optimizer', 'OPTIMIZERS']

OPTIMIZERS = {
    'adamw': torch.optim.AdamW,
    'lion': Lion,
    'rlo': RLO,
    'rlo_lambda': RLOLambda,
    'rlo_lifted': RLOLifted,
}

def build_optimizer(name, params, lr, weight_decay=0.0, betas=(0.9, 0.99), **kwargs):

    key = name.lower().replace('-', '_')
    if key not in OPTIMIZERS:
        raise ValueError(f'unknown optimizer {name!r}; choose from {sorted(OPTIMIZERS)}')
    cls = OPTIMIZERS[key]
    accepted = inspect.signature(cls.__init__).parameters

    extra = {} if key == 'adamw' else {k: v for k, v in kwargs.items() if k in accepted}
    return cls(params, lr=lr, weight_decay=weight_decay, betas=tuple(betas), **extra)
