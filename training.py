from collections import namedtuple
import lightning as L
import torch

from shot import SHOTDescriptor
from rops import ROPSDescriptor

Batch = namedtuple("Batch", ["x", "y", "edge_index", "batch"])


class RegressionModel(L.LightningModule):
    def __init__(self, model, loss_fn, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss_fn", "descriptor"])
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr

    def forward(self, *args):
        return self.model(*args)

    def _get_loss(self, batch):
        y_hat = self.model(batch.x, batch.edge_index, batch.batch)
        return self.loss_fn(y_hat, batch.y)

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("train_loss", loss.item(), batch_size=len(batch.y))
        return loss

    def validation_step(self, batch, batch_idx):
        self.log("val_loss", self._get_loss(batch).item(), batch_size=len(batch.y))

    def test_step(self, batch, batch_idx):
        self.log("test_loss", self._get_loss(batch).item(), batch_size=len(batch.y))

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


class LRFRegressionModel(RegressionModel):
    def __init__(self, model, loss_fn, lr=1e-3, n_neighbors=5, lrf_type="shot"):
        super().__init__(model, loss_fn, lr)
        self.n_neighbors = n_neighbors
        self.lrf_type = lrf_type
        if lrf_type == "shot":
            self.descriptor = SHOTDescriptor(n_neighbors)
        elif lrf_type == "rops":
            self.descriptor = ROPSDescriptor(n_neighbors)
        else:
            raise ValueError(f"Unknown LRF type: {lrf_type}")

    def _calculate_descriptor(self, batch):
        shot = self.descriptor(batch.pos, batch.batch).to(batch.x)
        return Batch(
            x=torch.cat([batch.x, shot], dim=-1),
            y=batch.y,
            edge_index=batch.edge_index,
            batch=batch.batch,
        )

    def training_step(self, batch, batch_idx):
        return super().training_step(self._calculate_descriptor(batch), batch_idx)

    def validation_step(self, batch, batch_idx):
        return super().validation_step(self._calculate_descriptor(batch), batch_idx)

    def test_step(self, batch, batch_idx):
        return super().test_step(self._calculate_descriptor(batch), batch_idx)


class ClassificationModel(L.LightningModule):
    def __init__(self, model, loss_fn, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr

    def forward(self, *args):
        return self.model(*args)

    def training_step(self, batch, batch_idx):
        y_hat = self.model(batch.x, batch.edge_index, batch.batch)
        loss = self.loss_fn(y_hat, batch.y)
        self.log("train_loss", loss.item())

        pred = y_hat.argmax(dim=1)
        correct = (pred == batch.y).sum().item()
        self.log("train_acc", correct / len(batch.y))

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch.x, batch.edge_index, batch.batch)
        loss = self.loss_fn(y_hat, batch.y)
        self.log("val_loss", loss.item())

        pred = y_hat.argmax(dim=1)
        correct = (pred == batch.y).sum().item()
        self.log("val_acc", correct / len(batch.y))

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
