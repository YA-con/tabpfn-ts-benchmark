"""Dataset loading utilities."""

from pathlib import Path

import pandas as pd


def load_nixtla_csv(path: str | Path, unique_id: str, target_column: str = "close_price") -> pd.DataFrame:
    """Load one provided OHLCV CSV into Nixtla long format."""

    df = pd.read_csv(path)
    required = {"trade_date", target_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    output = df[["trade_date", target_column]].rename(
        columns={"trade_date": "ds", target_column: "y"}
    )
    output["unique_id"] = unique_id
    output["ds"] = pd.to_datetime(output["ds"])
    return output[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])
