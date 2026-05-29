"""Dataset registry and loader tests."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.loaders import load_provided_market_dataset, load_public_benchmark_csv
from src.data.registry import get_dataset_spec, list_dataset_specs
from src.data.schema import validate_nixtla_frame


def test_registry_contains_cross_domain_specs() -> None:
    """Registry includes non-finance and finance datasets."""

    specs = {spec.name: spec for spec in list_dataset_specs()}
    assert {"etth1", "electricity", "exchange_rate", "traffic", "weather", "stock_provided"}.issubset(
        specs
    )
    assert specs["stock_provided"].domain == "finance"
    assert specs["weather"].domain == "weather"


def test_unknown_dataset_has_helpful_error() -> None:
    """Unknown registry keys fail loudly with available names."""

    with pytest.raises(KeyError, match="Available"):
        get_dataset_spec("not_a_dataset")


def test_validate_nixtla_frame_rejects_duplicates() -> None:
    """Schema validation catches duplicate series timestamps."""

    df = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": ["2024-01-01", "2024-01-01"],
            "y": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_nixtla_frame(df)


def test_load_provided_market_dataset(tmp_path: Path) -> None:
    """Provided OHLCV files load into Nixtla format."""

    market_dir = tmp_path / "market_data_1h" / "港股HK"
    market_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "trade_date": ["2024-01-01 09:30:00", "2024-01-01 10:30:00"],
            "open_price": [1.0, 2.0],
            "high_price": [1.2, 2.2],
            "low_price": [0.9, 1.9],
            "close_price": [1.1, 2.1],
            "volume": [100, 200],
        }
    ).to_csv(market_dir / "0700.HK_1h.csv", index=False)

    loaded = load_provided_market_dataset(
        data_root=tmp_path,
        markets=["market_data_1h/港股HK"],
    )

    assert list(loaded.columns) == ["unique_id", "ds", "y"]
    assert loaded["unique_id"].unique().tolist() == ["0700.HK_1h"]
    assert loaded["y"].tolist() == [1.1, 2.1]


def test_load_provided_market_dataset_skips_invalid_files(tmp_path: Path) -> None:
    """Invalid OHLCV files can be skipped while loading a market panel."""

    market_dir = tmp_path / "market_data_1h" / "港股HK"
    market_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "trade_date": ["2024-01-01 09:30:00"],
            "close_price": [1.1],
        }
    ).to_csv(market_dir / "good.csv", index=False)
    pd.DataFrame(
        {
            "trade_date": ["2024-01-01 09:30:00"],
            "close_price": [None],
        }
    ).to_csv(market_dir / "bad.csv", index=False)

    loaded = load_provided_market_dataset(
        data_root=tmp_path,
        markets=["market_data_1h/港股HK"],
        skip_invalid=True,
    )

    assert loaded["unique_id"].tolist() == ["good"]


def test_load_public_benchmark_csv(tmp_path: Path) -> None:
    """Wide public benchmark CSV files melt into Nixtla format."""

    path = tmp_path / "wide.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "target_a": [1.0, 2.0],
            "target_b": [3.0, 4.0],
        }
    ).to_csv(path, index=False)

    loaded = load_public_benchmark_csv(path, dataset_name="demo")

    assert len(loaded) == 4
    assert set(loaded["unique_id"]) == {"demo/target_a", "demo/target_b"}
    assert loaded["y"].sum() == pytest.approx(10.0)


def test_load_public_benchmark_csv_consolidates_duplicate_timestamps(tmp_path: Path) -> None:
    """Public benchmark mirrors can contain duplicate rows for the same timestamp."""

    path = tmp_path / "wide_with_duplicates.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "target_a": [1.0, 3.0, 5.0],
        }
    ).to_csv(path, index=False)

    loaded = load_public_benchmark_csv(path, dataset_name="demo")

    assert len(loaded) == 2
    assert loaded["y"].tolist() == [2.0, 5.0]
