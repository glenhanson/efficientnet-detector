import fire

from .pytorch_classifier import commands as pytorch_commands


def main():
    """Expose CLI."""
    fire.Fire(
        {
            "pytorch": pytorch_commands(),
        }
    )
