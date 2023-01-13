from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import torchvision

from .config import Config


class EfficientNet(torch.nn.Module):
    """EfficientNet Classifier Model."""

    def __init__(self) -> None:
        super().__init__()
        imagenet_model = torchvision.models.efficientnet_v2_l(
            weights="IMAGENET1K_V1", progress=False
        )
        backbone_nodes = list(imagenet_model.children())
        self.features = torch.nn.Sequential(
            torchvision.transforms.Normalize(Config.rgb_mean, Config.rgb_std),
            *backbone_nodes[:-1],
            torch.nn.Flatten(),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(imagenet_model.classifier[1].in_features, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward step."""
        features = self.features(x)
        score = self.classifier(features)

        return {
            "score": score,
        }

    def load(self, model_path: Union[Path, str]) -> None:
        """Load model weights."""
        weights = torch.load(model_path)
        self.load_state_dict(weights)


def export_model(input_shape: tuple[int, int, int] = (3, 1024, 1024)) -> None:
    """Export the model."""
    x = torch.randn(*input_shape)
    model = EfficientNet().eval()
    model.load(Config.trained_model_path)
    print(f"Input shape: {input_shape}")
    torch.onnx.export(
        model,
        x[None],
        Config.onnx_model_path,
        input_names=["input"],
        output_names=["scores"],
        opset_version=11,
    )
    print(f"model saved at {Config.onnx_model_path}")
