"""Feature engineering governance stubs."""

import pandas as pd


def add_returns(df: pd.DataFrame, periods: list[int] = [1, 5, 20]) -> pd.DataFrame:
    """Add return features for the requested lookback periods."""

    raise NotImplementedError("Phase 2")


def add_volatility(df: pd.DataFrame, windows: list[int] = [5, 20, 60]) -> pd.DataFrame:
    """Add realized volatility features for the requested windows."""

    raise NotImplementedError("Phase 2")


def add_technical_indicators(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    """Add selected technical indicators."""

    raise NotImplementedError("Phase 2")
