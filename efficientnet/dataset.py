from __future__ import annotations

import json
from typing import Any, Optional

import imageio
import numpy as np
import torch
from PIL import Image

from .config import Config
from .utils import Holdout, get_holdout


def get_sample_dicts(holdout: Optional[Holdout] = None) -> list[dict[str, Any]]:
    """Get sample dicts for classification task"""
    samples = []
    for file in Config.annotation_directory.glob("*.json"):
        with open(file) as f:
            data = json.loads(f.read())
        slug = file.stem
        if holdout and get_holdout(slug) != holdout:
            continue

        image_path = data["image_path"]
        img = imageio.imread(image_path)
        samples.append(
            {
                "image_path": image_path,
                "image": img,
                "label": np.array(data["label"]).astype("float32"),
            }
        )
    return samples


class EfficientNetDataset(torch.utils.data.Dataset):
    """Creates dataset"""

    def __init__(self, holdout: Optional[Holdout] = None) -> None:
        self.samples = get_sample_dicts(holdout)
        self.transform = Config.img_transform

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Access the ith sample."""
        sample = self.samples[idx]
        x = self.transform(Image.fromarray(sample["image"]).convert("RGB"))
        return {
            "image": x,
            "image_path": str(sample["image_path"]),
            "label": sample["label"],
        }
