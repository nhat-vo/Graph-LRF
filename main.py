import matplotlib.pyplot as plt

import torch_geometric as G
import torch_geometric.nn as gnn
from torch_geometric import transforms


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import numpy as np
from tqdm import tqdm, trange

import lightning as L

from shot import SHOTTransform
from utils import PosAsX, SetTarget
from models import GAT, GCN
from training import RegressionModel, ClassificationModel
from data import qm9

DATA_PATH = "/scratch/local/ssd/nhat/data/"
TRAIN_PATH = "/scratch/local/ssd/nhat/out/lrf"
BATCH_SIZE = 256

torch.manual_seed(2024)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def main():
    L.seed_everything(2024)
    train_loader, val_loader = qm9(
        DATA_PATH,
        label=0,
        train_ratio=0.7,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        random_transform=False,
        use_pos=False,
        use_shot=True,
        use_rops=False,
    )

    # model = GCN(11, [32, 64, 128], 1)
    model = GAT(11, [32, 64, 128], 1)
    module = RegressionModel(model, nn.L1Loss(), lr=0.001)
    trainer = L.Trainer(default_root_dir=TRAIN_PATH, max_epochs=100)
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
