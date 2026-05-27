"""Dataset acquisition helpers.

Large raw datasets should not be committed to git. This module records the public
benchmark targets and leaves actual downloading/copying as an explicit operator
step so that experiments stay reproducible.
"""

from src.data.registry import list_dataset_specs


def print_dataset_manifest() -> None:
    """Print registered datasets and expected local storage locations."""

    for spec in list_dataset_specs():
        print(f"{spec.name}\t{spec.domain}\t{spec.storage}")
