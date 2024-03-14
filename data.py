import torch
from torch.utils.data import random_split

from torch_geometric import datasets, transforms
from torch_geometric.loader import DataLoader
from shot import SHOTTransform

from utils import PosAsX, SetTarget

RANDOM_TRANSFORM = transforms.Compose(
    [
        transforms.RandomScale([0.5, 2]),
        transforms.RandomRotate(180, 0),
        transforms.RandomRotate(180, 1),
        transforms.RandomRotate(180, 2),
        transforms.RandomFlip(0),
        transforms.RandomFlip(1),
        transforms.RandomFlip(2),
    ]
)


def qm9(
    data_path,
    label=0,
    train_ratio=0.5,
    random_transform=False,
    use_pos=False,
    use_shot=False,
    use_rops=False,
    **kwargs
):
    transform = [SetTarget(label)]
    if random_transform:
        transform.append(RANDOM_TRANSFORM)
    if use_pos:
        transform.append(PosAsX())
    if use_shot:
        transform.append(SHOTTransform())
    if use_rops:
        pass

    qm9 = datasets.QM9(root=data_path, transform=transforms.Compose(transform))
    train, val = random_split(qm9, [train_ratio, 1 - train_ratio])
    train_loader = DataLoader(train, shuffle=True, **kwargs)
    val_loader = DataLoader(val, **kwargs)

    return train_loader, val_loader
