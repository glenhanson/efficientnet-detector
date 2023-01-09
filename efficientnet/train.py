from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics.classification import BinaryF1Score

from .config import Config
from .dataset import EfficientNetDataset
from .model import EfficientNet
from .utils import Holdout


class EfficientNetTrainer(pl.LightningModule):
    """Pytorch Lightning trainer"""

    def __init__(self) -> None:
        super().__init__()
        self.model = EfficientNet()
        self.model.to(Config.device)
        self.binary_f1 = BinaryF1Score()
        self.learning_rate = Config.learning_rate
        self.batch_size = Config.batch_size
        self.n_workers = Config.n_workers

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Create loader for training holdout."""
        return torch.utils.data.DataLoader(
            EfficientNetDataset(Holdout.train),
            batch_size=self.batch_size,
            num_workers=self.n_workers,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Create loader for dev holdout."""
        return torch.utils.data.DataLoader(
            EfficientNetDataset(Holdout.dev),
            batch_size=self.batch_size,
            num_workers=self.n_workers,
        )

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        images = batch["image"].to(Config.device)
        targets = batch["label"].to(Config.device)
        result = self.model(images)
        loss = torch.nn.functional.binary_cross_entropy(result["score"], targets)
        self.log("loss", loss, on_step=True, logger=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], *args):
        images = batch["image"].to(Config.device)
        targets = batch["label"].to(Config.device)
        result = self.model(images)
        f1_score = self.binary_f1(result["score"], targets)
        self.log("f1score", f1_score, prog_bar=True, on_epoch=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers."""
        return torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)


def train_model(checkpoint_path: Optional[str] = None) -> None:
    """Train model."""
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=Config.max_epochs,
        callbacks=[EarlyStopping(monitor="f1score", mode="max", patience=10)],
    )
    lightning = EfficientNetTrainer()
    trainer.fit(lightning)
