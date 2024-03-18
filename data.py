import torch
from torch.utils.data import random_split

from torch_geometric import datasets, transforms
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from shot import SHOTTransform

from utils import PosAsX, SetTarget

RANDOM_TRANSFORM = transforms.Compose(
    [
        transforms.RandomScale([0.5, 2]),
        transforms.RandomRotate(180, 0),
        transforms.RandomRotate(180, 1),
        transforms.RandomRotate(180, 2),
    ]
)


def filter(data: Data):
    return len(data.x) >= 7


def qm9(
    data_path,
    label=0,
    train_ratio=0.5,
    random_transform=False,
    force_reload=False,
    batch_size=256,
    num_workers=4,
    pin_memory=True,
    **kwargs
):
    transform = [SetTarget(label)]
    if random_transform:
        transform.append(RANDOM_TRANSFORM)

    qm9 = datasets.QM9(
        root=data_path,
        transform=transforms.Compose(transform),
        pre_filter=filter,
        force_reload=force_reload,
    )
    kwargs.update(
        {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": pin_memory}
    )
    train, val = random_split(qm9, [train_ratio, 1 - train_ratio])
    train_loader = DataLoader(train, shuffle=True, **kwargs)
    val_loader = DataLoader(val, **kwargs)

    return train_loader, val_loader
