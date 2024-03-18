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
from training import RegressionModel, ClassificationModel, SHOTRegressionModel
from data import qm9

from argparse import ArgumentParser

DATA_PATH = "/scratch/local/ssd/nhat/data/"
TRAIN_PATH = "/scratch/local/ssd/nhat/out/lrf"
BATCH_SIZE = 256

torch.manual_seed(2024)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gat")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--profiler", type=str, default="none")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--lrf", type=str, default="none")
    args = parser.parse_args()

    L.seed_everything(2024)
    train_loader, val_loader = qm9(
        DATA_PATH,
        label=0,
        train_ratio=0.7,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        random_transform=False,
    )

    input_size = 11
    if args.lrf == "shot":
        input_size += 80

    if args.model == "gat":
        model = GAT(input_size, [32, 32, 32], 1)
    else:
        model = GCN(input_size, [32, 64, 128], 1)

    if args.lrf == "shot":
        module = SHOTRegressionModel(model, nn.L1Loss(), lr=args.lr)
    else:
        module = RegressionModel(model, nn.L1Loss(), lr=args.lr)

    if args.profiler == "none":
        args.profiler = None

    trainer = L.Trainer(
        default_root_dir=TRAIN_PATH,
        max_epochs=args.max_epochs,
        devices=[args.gpu_id],
        profiler=args.profiler,
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
