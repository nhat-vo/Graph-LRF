import torch
import torch.nn as nn

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from models import GAT, GCN
from training import RegressionModel, ClassificationModel, LRFRegressionModel
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--profiler", type=str, default="none")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--lrf", type=str, default="none")
    parser.add_argument("--task", type=str, default="regression")
    parser.add_argument("--n_neighbors", type=int, default=5)
    parser.add_argument("--label", type=int, default=0)
    args = parser.parse_args()

    L.seed_everything(2024)

    if args.task == "regression":
        train_loader, val_loader, test_loader = qm9(
            DATA_PATH,
            label=args.label,
            split_ratio=[0.7, 0.2, 0.1],
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True,
            random_transform=False,
        )

    # probably shouldn't hardcode
    input_size = 11
    if args.lrf == "shot":
        input_size += 80
    elif args.lrf == "rops":
        input_size += 2 * 3 * 4 * 3

    if args.model == "gat":
        model = GAT(input_size, [32, 64, 128], 1)
    else:
        model = GCN(input_size, [32, 64, 128], 1)

    if args.lrf != "none":
        if args.task == "regression":
            module = LRFRegressionModel(
                model,
                nn.L1Loss(),
                lr=args.lr,
                n_neighbors=args.n_neighbors,
                lrf_type=args.lrf,
            )
        else:
            raise NotImplementedError
    else:
        if args.task == "regression":
            module = RegressionModel(model, nn.L1Loss(), lr=args.lr)
        else:
            module = ClassificationModel(model, nn.CrossEntropyLoss(), lr=args.lr)

    if args.profiler == "none":
        args.profiler = None

    exp_name = f"{args.model}_{args.lrf}_{args.task}_k={args.n_neighbors}"
    logger = TensorBoardLogger(save_dir=TRAIN_PATH, name=exp_name)

    early_stopping = EarlyStopping("val_loss", patience=5, mode="min")

    trainer = L.Trainer(
        default_root_dir=TRAIN_PATH,
        max_epochs=args.max_epochs,
        devices=[args.gpu_id],
        check_val_every_n_epoch=5,
        callbacks=[early_stopping],
        profiler=args.profiler,
        logger=logger,
    )
    trainer.fit(module, train_loader, val_loader)
    trainer.test(module, test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
