from torch.utils.data import random_split

from torch_geometric import datasets, transforms
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from utils import SetTarget

RANDOM_TRANSFORM = transforms.Compose(
    [
        transforms.RandomScale([0.5, 2]),
        transforms.RandomRotate(180, 0),
        transforms.RandomRotate(180, 1),
        transforms.RandomRotate(180, 2),
    ]
)


def filter(data: Data):
    return len(data.x) >= 9


def qm9(
    data_path,
    label=0,
    split_ratio=[0.7, 0.2, 0.1],
    random_transform=False,
    force_reload=False,
    batch_size=256,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
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
        {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
        }
    )
    splits = random_split(qm9, split_ratio)
    train_loader = DataLoader(splits[0], shuffle=True, **kwargs)
    val_loader = DataLoader(splits[1], **kwargs)
    loaders = [train_loader, val_loader]
    if len(splits) == 3:
        test_loader = DataLoader(splits[2], **kwargs)
        loaders.append(test_loader)

    return loaders
