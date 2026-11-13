import json
import os
import random
from pathlib import Path

import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def resolve_device(device=None):

    device = device or os.environ.get('RLO_DEVICE')
    if device:
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)

_STREAMS = {'train_sampling': 1, 'validation': 2, 'probe_selection': 3, 'model_init': 4,
            'gpt2_train': 5, 'gpt2_val': 6}

def stream_seed(seed, stream):
    return int(seed) * 1000 + _STREAMS[stream]

def torch_generator(seed, stream):
    g = torch.Generator()
    g.manual_seed(stream_seed(seed, stream))
    return g

def numpy_state(seed, stream):
    return np.random.RandomState(stream_seed(seed, stream))

def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)
