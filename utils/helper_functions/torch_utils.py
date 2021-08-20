from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler


def weights_init(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv'):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('InstanceNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def set_seeds(seed: Optional[int]) -> None:
    """Sets random seeds"""
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def training_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_data_loader(ds, idx=None, batch_size=1, num_workers=4, shuffle=True):
    if idx is None:
        return DataLoader(ds, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)
    else:
        if shuffle:
            return DataLoader(ds, num_workers=num_workers, batch_size=batch_size, sampler=SubsetRandomSampler(idx))
        return DataLoader(torch.utils.data.Subset(ds, idx), num_workers=num_workers, batch_size=batch_size,
                          shuffle=False)
