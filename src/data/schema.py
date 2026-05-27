"""Shared dataset schema definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata needed to identify and load a benchmark dataset."""

    name: str
    domain: str
    freq: str
    horizon: int
    context_length: int
    loader: str
    storage: str
    target_column: str = "y"
    data_root: str | None = None
    path: str | None = None
    markets: tuple[str, ...] = ()
    date_column: str = "date"
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly representation."""

        return asdict(self)


@dataclass(frozen=True)
class DatasetSummary:
    """Small metadata summary for a loaded Nixtla-format dataset."""

    name: str
    domain: str
    n_series: int
    n_observations: int
    min_ds: str
    max_ds: str
    freq: str
    horizon: int
    context_length: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly representation."""

        return asdict(self)


def validate_nixtla_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a long-format frame with columns [unique_id, ds, y]."""

    required = ["unique_id", "ds", "y"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Nixtla frame is missing columns: {missing}")
    if df.empty:
        raise ValueError("Nixtla frame must not be empty.")

    output = df[required].copy()
    output["unique_id"] = output["unique_id"].astype(str)
    output["ds"] = pd.to_datetime(output["ds"])
    output["y"] = pd.to_numeric(output["y"], errors="coerce")
    if output["unique_id"].isna().any() or (output["unique_id"] == "").any():
        raise ValueError("unique_id must not contain null or empty values.")
    if output["ds"].isna().any():
        raise ValueError("ds must parse to datetimes without null values.")
    if output["y"].isna().any():
        raise ValueError("y must be numeric and non-null.")
    if output.duplicated(["unique_id", "ds"]).any():
        raise ValueError("Nixtla frame contains duplicate [unique_id, ds] rows.")

    return output.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def summarize_nixtla_frame(df: pd.DataFrame, spec: DatasetSpec) -> DatasetSummary:
    """Summarize a validated Nixtla-format dataset."""

    valid = validate_nixtla_frame(df)
    return DatasetSummary(
        name=spec.name,
        domain=spec.domain,
        n_series=int(valid["unique_id"].nunique()),
        n_observations=int(len(valid)),
        min_ds=str(valid["ds"].min()),
        max_ds=str(valid["ds"].max()),
        freq=spec.freq,
        horizon=spec.horizon,
        context_length=spec.context_length,
    )


def write_dataset_summary(summary: DatasetSummary, path: str | Path) -> None:
    """Write a dataset summary as a compact CSV row."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary.to_dict()]).to_csv(output_path, index=False)
