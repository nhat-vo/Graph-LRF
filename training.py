import lightning as L
import torch


class RegressionModel(L.LightningModule):
    def __init__(self, model, loss_fn, lr=1e-3):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr

    def forward(self, *args):
        return self.model(*args)

    def training_step(self, batch, batch_idx):
        y_hat = self.model(batch.x, batch.edge_index, batch.batch)
        loss = self.loss_fn(y_hat, batch.y)
        self.log("train_loss", loss.item(), batch_size=len(batch.y))
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch.x, batch.edge_index, batch.batch)
        loss = self.loss_fn(y_hat, batch.y)
        self.log("val_loss", loss.item(), batch_size=len(batch.y))

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


class ClassificationModel(L.LightningModule):
    def __init__(self, model, loss_fn, lr=1e-3):
        super().__init__()
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
