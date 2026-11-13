from pathlib import Path

import numpy as np
import torch

class TokenBin:

    def __init__(self, data_dir, split, block_size):
        path = Path(data_dir) / f'{split}.bin'
        if not path.exists():
            raise FileNotFoundError(f'{path} not found; run `python prepare_openwebtext.py` first')
        self.data = np.memmap(str(path), dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return len(self.data)

    def indices(self, batch_size, generator=None):
        return torch.randint(len(self.data) - self.block_size, (batch_size,), generator=generator)

    def batch(self, ix, device):
        bs = self.block_size
        x = torch.stack([torch.from_numpy(self.data[i:i + bs].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1:i + 1 + bs].astype(np.int64)) for i in ix])
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    def sample(self, batch_size, device, generator=None):
        return self.batch(self.indices(batch_size, generator), device)
