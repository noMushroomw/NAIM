import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from ..utils.common import numpy_state, seed_worker, torch_generator

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

def _transforms():
    train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(),
                       T.Normalize(MEAN, STD)])
    test = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    return train, test

def cifar10_train_test(root, batch_size=128, num_workers=4, drop_last=True, test_batch_size=None):

    train_tf, test_tf = _transforms()
    train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=drop_last)
    test_loader = DataLoader(test, batch_size=test_batch_size or batch_size, shuffle=False,
                             num_workers=min(num_workers, 2), pin_memory=True)
    return train_loader, test_loader

def cifar10_train_val_probe(root, seed, batch_size=128, num_workers=4, split_seed=1234,
                            val_size=5000, probe_seed=0, probe_size=256):

    train_tf, test_tf = _transforms()
    train_aug = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    train_eval = torchvision.datasets.CIFAR10(root, train=True, download=False, transform=test_tf)
    perm = np.random.RandomState(int(split_seed)).permutation(len(train_aug))
    val_idx, train_idx = perm[:val_size].tolist(), perm[val_size:].tolist()

    order = numpy_state(probe_seed, 'probe_selection').permutation(len(train_idx))
    probe_idx = [train_idx[i] for i in order[:probe_size]]
    xs = torch.stack([train_eval[i][0] for i in probe_idx])
    ys = torch.tensor([int(train_eval[i][1]) for i in probe_idx])
    probe = [(xs[i:i + 256], ys[i:i + 256]) for i in range(0, probe_size, 256)]

    train_loader = DataLoader(Subset(train_aug, train_idx), batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              generator=torch_generator(seed, 'train_sampling'),
                              worker_init_fn=seed_worker)
    val_loader = DataLoader(Subset(train_eval, val_idx), batch_size=512, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, probe
