import numpy as np
import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric import transforms


from shot import SHOTDescriptor

DATA_PATH = "/scratch/local/ssd/nhat/data/"

RANDOM_TRANSFORM = transforms.Compose(
    [
        transforms.RandomRotate(180, 0),
        transforms.RandomRotate(180, 1),
        transforms.RandomRotate(180, 2),
    ]
)


def test_shot_invariant(num_test=20):
    data = QM9(DATA_PATH)
    loader = DataLoader(data, batch_size=1, shuffle=True)
    trans = RANDOM_TRANSFORM

    for _ in range(num_test):
        batch = next(iter(loader))
        pos = batch.pos

        rad = 4
        shot = SHOTDescriptor(rad, 4)

        shot_orig = shot(pos)

        rand_transform = trans.forward(batch)
        shot_trans = shot(rand_transform.pos)

        assert torch.allclose(shot_orig, shot_trans)


if __name__ == "__main__":
    test_shot_invariant()
    print("All tests passed!")
