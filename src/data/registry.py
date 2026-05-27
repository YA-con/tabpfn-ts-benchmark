"""Dataset registry for cross-domain benchmark planning."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loaders import load_provided_market_dataset, load_public_benchmark_csv
from src.data.schema import DatasetSpec, DatasetSummary, summarize_nixtla_frame


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "etth1": DatasetSpec(
        name="etth1",
        domain="energy",
        freq="h",
        horizon=24,
        context_length=168,
        loader="public_benchmark_csv",
        storage="data/raw/etth1/ETTh1.csv",
        path="data/raw/etth1/ETTh1.csv",
        date_column="date",
        notes="Public ETT hourly transformer temperature benchmark.",
    ),
    "electricity": DatasetSpec(
        name="electricity",
        domain="energy",
        freq="h",
        horizon=24,
        context_length=168,
        loader="public_benchmark_csv",
        storage="data/raw/electricity/electricity.csv",
        path="data/raw/electricity/electricity.csv",
        date_column="date",
        notes="Public electricity load benchmark.",
    ),
    "exchange_rate": DatasetSpec(
        name="exchange_rate",
        domain="economics",
        freq="d",
        horizon=7,
        context_length=365,
        loader="public_benchmark_csv",
        storage="data/raw/exchange_rate/exchange_rate.csv",
        path="data/raw/exchange_rate/exchange_rate.csv",
        date_column="date",
        notes="Public exchange-rate benchmark.",
    ),
    "traffic": DatasetSpec(
        name="traffic",
        domain="traffic",
        freq="h",
        horizon=24,
        context_length=168,
        loader="public_benchmark_csv",
        storage="data/raw/traffic/traffic.csv",
        path="data/raw/traffic/traffic.csv",
        date_column="date",
        notes="Public traffic occupancy benchmark.",
    ),
    "weather": DatasetSpec(
        name="weather",
        domain="weather",
        freq="10min",
        horizon=144,
        context_length=1008,
        loader="public_benchmark_csv",
        storage="data/raw/weather/weather.csv",
        path="data/raw/weather/weather.csv",
        date_column="date",
        notes="Public weather benchmark.",
    ),
    "stock_provided": DatasetSpec(
        name="stock_provided",
        domain="finance",
        freq="h",
        horizon=24,
        context_length=240,
        loader="provided_market_dataset",
        storage="../market_data_downloader",
        data_root="../market_data_downloader",
        target_column="close_price",
        markets=(
            "market_data_1h/A股上交所SS",
            "market_data_1h/港股HK",
            "market_data_1h/加密货币Crypto",
        ),
        notes="Existing provided OHLCV data, kept outside git.",
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    """Return a registered dataset specification by name."""

    try:
        return DATASET_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(DATASET_REGISTRY)}") from exc


def list_dataset_specs() -> list[DatasetSpec]:
    """Return all registered dataset specifications."""

    return list(DATASET_REGISTRY.values())


def load_dataset(name: str, project_root: str | Path = ".", max_files: int | None = None) -> pd.DataFrame:
    """Load a registered dataset into Nixtla long format."""

    spec = get_dataset_spec(name)
    root = Path(project_root).resolve()
    if spec.loader == "provided_market_dataset":
        if spec.data_root is None:
            raise ValueError(f"{name} requires data_root.")
        return load_provided_market_dataset(
            data_root=root / spec.data_root,
            markets=list(spec.markets),
            target_column=spec.target_column,
            max_files=max_files,
        )
    if spec.loader == "public_benchmark_csv":
        if spec.path is None:
            raise ValueError(f"{name} requires path.")
        path = root / spec.path
        if not path.exists():
            raise FileNotFoundError(
                f"{name} raw file is not present at {path}. "
                "Download or copy the benchmark file before loading."
            )
        return load_public_benchmark_csv(
            path=path,
            dataset_name=spec.name,
            date_column=spec.date_column,
        )
    raise ValueError(f"Unsupported loader {spec.loader!r} for dataset {name!r}.")


def summarize_dataset(name: str, project_root: str | Path = ".", max_files: int | None = None) -> DatasetSummary:
    """Load and summarize a registered dataset."""

    spec = get_dataset_spec(name)
    frame = load_dataset(name=name, project_root=project_root, max_files=max_files)
    return summarize_nixtla_frame(frame, spec)
