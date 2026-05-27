"""Materialize registered datasets into processed benchmark artifacts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.data.registry import get_dataset_spec, load_dataset
from src.data.schema import DatasetSummary, summarize_nixtla_frame, validate_nixtla_frame


def materialize_dataset(
    name: str,
    project_root: str | Path = ".",
    output_root: str | Path = "data/processed",
    max_files: int | None = None,
) -> DatasetSummary:
    """Load a registered dataset and write standardized parquet plus summary CSV."""

    root = Path(project_root).resolve()
    output_dir = (root / output_root / name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = get_dataset_spec(name)
    frame = validate_nixtla_frame(load_dataset(name=name, project_root=root, max_files=max_files))
    summary = summarize_nixtla_frame(frame, spec)

    frame.to_parquet(output_dir / "series.parquet", index=False)
    pd.DataFrame([summary.to_dict()]).to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame([asdict(spec)]).to_json(output_dir / "spec.json", orient="records", indent=2)
    return summary


def load_processed_dataset(
    name: str,
    project_root: str | Path = ".",
    output_root: str | Path = "data/processed",
) -> pd.DataFrame:
    """Load a previously materialized processed dataset."""

    path = Path(project_root).resolve() / output_root / name / "series.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    return validate_nixtla_frame(pd.read_parquet(path))
