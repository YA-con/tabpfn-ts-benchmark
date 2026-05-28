"""Top-level experiment report orchestration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.reporting.collector import (
    collect_baseline_results,
    collect_environment_info,
    collect_experiment_results,
)
from src.reporting.comparison import compare_with_baselines, summarize_best_models
from src.reporting.figures import generate_matplotlib_figures
from src.reporting.html import render_html_report


def _summary_csv(metrics: pd.DataFrame | None, primary_metric: str) -> pd.DataFrame:
    """Build a compact model summary for report exports."""

    if metrics is None or metrics.empty or "model" not in metrics.columns:
        return pd.DataFrame()
    numeric = metrics.select_dtypes(include="number").columns.tolist()
    grouped = metrics.groupby("model")[numeric].mean(numeric_only=True).reset_index()
    if primary_metric in grouped.columns:
        grouped = grouped.sort_values(primary_metric)
    return grouped


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy NaN values into strict JSON-compatible nulls."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def generate_experiment_report(
    run_dir: str | Path,
    out_dir: str | Path,
    baseline_dirs: list[str | Path] | None = None,
    config_paths: list[str | Path] | None = None,
    primary_metric: str = "smape",
) -> Path:
    """Generate a reusable HTML report for an experiment run."""

    artifacts = collect_experiment_results(run_dir, config_paths=config_paths)
    baselines = collect_baseline_results(baseline_dirs)
    output = Path(out_dir).resolve()
    assets = output / "assets"
    output.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)

    baseline_pairs = [(item.run_dir.name, item.metrics) for item in baselines]
    comparison = compare_with_baselines(artifacts.metrics, baseline_pairs, metric=primary_metric)
    summaries = summarize_best_models(artifacts.metrics, primary_metric=primary_metric)
    figures = generate_matplotlib_figures(
        artifacts.metrics,
        artifacts.predictions,
        assets,
        primary_metric=primary_metric,
    )
    summary = _summary_csv(artifacts.metrics, primary_metric=primary_metric)
    summary.to_csv(output / "summary.csv", index=False)
    metadata = {
        "run_id": artifacts.run_id,
        "run_dir": str(artifacts.run_dir),
        "primary_metric": primary_metric,
        "warnings": artifacts.warnings + [w for baseline in baselines for w in baseline.warnings],
        "summary": summary.to_dict(orient="records"),
        "baseline_comparison": comparison.to_dict(orient="records"),
        "figures": figures,
    }
    (output / "metrics.json").write_text(
        json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    render_html_report(
        output / "index.html",
        run_id=artifacts.run_id,
        run_dir=artifacts.run_dir,
        summaries=summaries,
        comparison=comparison,
        metrics=artifacts.metrics,
        predictions=artifacts.predictions,
        figures=figures,
        config=artifacts.config,
        environment=collect_environment_info(),
        warnings=metadata["warnings"],
        primary_metric=primary_metric,
    )
    return output / "index.html"
