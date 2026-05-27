"""Walk-forward splitting governance stubs."""

from collections.abc import Iterator

import pandas as pd


def walk_forward_split(
    df: pd.DataFrame, train_size: int, val_size: int, test_size: int, step: int
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Yield chronological train, validation, and test splits."""

    raise NotImplementedError("Phase 2")
