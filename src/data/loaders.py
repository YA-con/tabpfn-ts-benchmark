"""Dataset loading utilities."""

from pathlib import Path

import pandas as pd

from src.data.schema import validate_nixtla_frame


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
    return validate_nixtla_frame(output[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"]))


def load_provided_market_dataset(
    data_root: str | Path,
    markets: list[str],
    target_column: str = "close_price",
    max_files: int | None = None,
    skip_invalid: bool = True,
) -> pd.DataFrame:
    """Load provided OHLCV market CSV files into one Nixtla-format panel.

    The source files stay outside the repository. This loader only materializes an
    in-memory long-format panel with columns [unique_id, ds, y].
    """

    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Provided market data root does not exist: {root}")

    frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    for market in markets:
        market_dir = root / market
        if not market_dir.exists():
            raise FileNotFoundError(f"Market data directory does not exist: {market_dir}")
        for path in sorted(market_dir.glob("*.csv")):
            if max_files is not None and len(frames) >= max_files:
                break
            try:
                frames.append(
                    load_nixtla_csv(
                        path=path,
                        unique_id=path.stem,
                        target_column=target_column,
                    )
                )
            except ValueError:
                if not skip_invalid:
                    raise
                skipped.append(str(path))
        if max_files is not None and len(frames) >= max_files:
            break

    if not frames:
        raise ValueError(f"No CSV files found under {root} for markets={markets}.")
    if skipped:
        print(f"Skipped {len(skipped)} invalid market files.")
    return validate_nixtla_frame(pd.concat(frames, ignore_index=True))


def load_public_benchmark_csv(
    path: str | Path,
    dataset_name: str,
    date_column: str = "date",
    target_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load a wide public benchmark CSV into Nixtla long format.

    Common benchmark files such as ETT, electricity, traffic, weather, and exchange
    rate are often stored as one timestamp column plus many target columns. This
    function melts that wide format into [unique_id, ds, y].
    """

    csv_path = Path(path).expanduser().resolve()
    df = pd.read_csv(csv_path)
    if date_column not in df.columns:
        raise ValueError(f"{csv_path} is missing date column {date_column!r}.")
    value_columns = target_columns or [column for column in df.columns if column != date_column]
    if not value_columns:
        raise ValueError(f"{csv_path} does not contain target columns.")
    missing = set(value_columns).difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing target columns: {sorted(missing)}")

    long_df = df[[date_column, *value_columns]].melt(
        id_vars=date_column,
        value_vars=value_columns,
        var_name="unique_id",
        value_name="y",
    )
    long_df["unique_id"] = dataset_name + "/" + long_df["unique_id"].astype(str)
    long_df = long_df.rename(columns={date_column: "ds"})
    long_df["ds"] = pd.to_datetime(long_df["ds"])
    return validate_nixtla_frame(long_df[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"]))
