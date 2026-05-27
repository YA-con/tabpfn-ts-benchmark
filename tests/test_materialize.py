"""Processed dataset materialization tests."""

from pathlib import Path

import pandas as pd

from src.data.materialize import load_processed_dataset, materialize_dataset
from src.data.registry import DATASET_REGISTRY
from src.data.schema import DatasetSpec


def test_materialize_public_benchmark_csv(tmp_path: Path) -> None:
    """A registered wide CSV can be materialized and loaded back."""

    raw_dir = tmp_path / "data" / "raw" / "demo"
    raw_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "a": [1.0, 2.0],
            "b": [3.0, 4.0],
        }
    ).to_csv(raw_dir / "demo.csv", index=False)

    DATASET_REGISTRY["demo_materialize"] = DatasetSpec(
        name="demo_materialize",
        domain="test",
        freq="d",
        horizon=1,
        context_length=2,
        loader="public_benchmark_csv",
        storage="data/raw/demo/demo.csv",
        path="data/raw/demo/demo.csv",
    )
    try:
        summary = materialize_dataset("demo_materialize", project_root=tmp_path)
        loaded = load_processed_dataset("demo_materialize", project_root=tmp_path)
    finally:
        DATASET_REGISTRY.pop("demo_materialize", None)

    assert summary.n_series == 2
    assert summary.n_observations == 4
    assert (tmp_path / "data" / "processed" / "demo_materialize" / "series.parquet").exists()
    assert loaded["y"].sum() == 10.0
