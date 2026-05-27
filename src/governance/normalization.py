"""Normalization governance stubs."""

import pandas as pd


def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Apply date-wise z-score normalization."""

    raise NotImplementedError("Phase 2")


def cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Apply date-wise rank normalization."""

    raise NotImplementedError("Phase 2")


def rolling_zscore(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Apply rolling time-series z-score normalization."""

    raise NotImplementedError("Phase 2")
