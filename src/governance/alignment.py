"""Calendar alignment governance stubs."""

import pandas as pd


def align_calendar(df: pd.DataFrame, freq: str = "B") -> pd.DataFrame:
    """Align observations to a target calendar frequency."""

    raise NotImplementedError("Phase 2")


def handle_halts(df: pd.DataFrame, strategy: str = "mark") -> pd.DataFrame:
    """Handle trading halts using the selected strategy."""

    raise NotImplementedError("Phase 2")
