"""Data cleaning governance stubs."""

import pandas as pd


def handle_missing(df: pd.DataFrame, strategy: str = "ffill") -> pd.DataFrame:
    """Handle missing values according to the selected strategy."""

    raise NotImplementedError("Phase 2")


def winsorize_cross_section(
    df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99
) -> pd.DataFrame:
    """Winsorize cross-sectional feature values by quantile bounds."""

    raise NotImplementedError("Phase 2")


def mad_clip(df: pd.DataFrame, n_mad: float = 5.0) -> pd.DataFrame:
    """Clip outliers using median absolute deviation."""

    raise NotImplementedError("Phase 2")
