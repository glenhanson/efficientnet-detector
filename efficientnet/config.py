from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Config:
    """Package configurations."""

    data_directory: Path = Path(os.getcwd()).parent / "data"
    dataset_directory: Path = data_directory / "dataset"
    image_directory: Path = Path("/home/ubuntu/data/output")
    annotation_directory: Path = dataset_directory / "annotations"
    models_directory: Path = data_directory / "models"

    device = torch.device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Model
    rgb_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    rgb_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Training
    max_epochs: int = 100
    gpus: int = 1
    learning_rate: float = 0.001
    batch_size: int = 16
    n_workers: int = 4
