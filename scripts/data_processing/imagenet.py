from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def imagenet_loaders(root, batch_size, rank=0, world_size=1, num_workers=8):

    root = Path(root)
    train_tf = T.Compose([
        T.RandomResizedCrop(224, scale=(0.08, 1.0)), T.RandomHorizontalFlip(), T.TrivialAugmentWide(),
        T.ToTensor(), T.Normalize(MEAN, STD), T.RandomErasing(p=0.25)])
    val_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
    train = ImageFolder(root / 'train', train_tf)
    val = ImageFolder(root / 'val', val_tf)
    sampler = DistributedSampler(train, world_size, rank) if world_size > 1 else None
    train_loader = DataLoader(train, batch_size, shuffle=sampler is None, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val, batch_size * 2, shuffle=False, num_workers=max(num_workers // 2, 1),
                            pin_memory=True)
    return train_loader, val_loader, sampler
