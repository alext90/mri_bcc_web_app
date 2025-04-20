import pytorch_lightning as pl
import torch
from torchvision import models
import torch.nn as nn


class BCC_Model(pl.LightningModule):
    def __init__(self, num_classes, lr=1e-4):
        super().__init__()

        base_model = models.mobilenet_v2(pretrained=True)

        # Freeze all layers
        for param in base_model.features.parameters():
            param.requires_grad = False

        # Replace the classifier head for num_classes
        base_model.classifier[1] = nn.Linear(
            in_features=base_model.last_channel, out_features=num_classes
        )

        # Unfreeze the classifier head
        for param in base_model.classifier.parameters():
            param.requires_grad = True

        self.model = base_model
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
