import fire
from .train import train_model

def main():
    """Expose CLI."""
    fire.Fire({
        "train": train_model,
    })
