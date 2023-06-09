from torch import nn, optim
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from typing import Optional
from pytorch_lightning.loggers import TensorBoardLogger


class System(LightningModule):

    model: nn.Module
    optimizer: optim.Optimizer
    loss_func: nn.Module

    def __init__(self, 
                 model: nn.Module, 
                 optimizer: optim.Optimizer, 
                 loss_func: nn.Module):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        X, target = batch
        y = self(X)
        loss = self.loss_func(y, target)
        self.log("loss", loss, logger=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        X, target = batch
        y = self(X)
        loss = self.loss_func(y, target)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        return self.optimizer

