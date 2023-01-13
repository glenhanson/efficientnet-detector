import fire

from .pytorch_classifier.model import export_model
from .pytorch_classifier.train import train_model


def main():
    """Expose CLI."""
    fire.Fire(
        {
            "train": train_model,
            "export": export_model,
        }
    )
