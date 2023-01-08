import json
from typing import Any, Optional

import imageio
import numpy as np

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
        img = imageio.v2.imread(image_path)
        samples.append(
            {
                "image_path": image_path,
                "image": img,
                "label": np.array(data["label"]).astype("float32"),
            }
        )
    return samples
