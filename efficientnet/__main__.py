import fire

from .model import export_model
from .train import train_model


def main():
    """Expose CLI."""
    fire.Fire(
        {
            "train": train_model,
            "export": export_model,
        }
    )
