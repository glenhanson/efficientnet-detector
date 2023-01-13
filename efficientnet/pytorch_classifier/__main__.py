"""Export to CLI."""

from __future__ import annotations

from typing import Callable

from .model import export_model
from .train import train_model


def commands() -> dict[str, Callable[..., None]]:
    """Return tracking subcommands for CLI."""
    return {
        "train": train_model,
        "export": export_model,
    }
