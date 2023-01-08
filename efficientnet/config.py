import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Package configurations."""

    data_directory: Path = Path(os.getcwd()).parent / "data"

    # Model
    rgb_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    rgb_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
