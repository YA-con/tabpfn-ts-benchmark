"""Metric aggregation and baseline comparison helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

LOWER_IS_BETTER = {
    "loss",
    "mse",
    "mae",
    "rmse",
    "smape",
    "wape",
    "mase",
    "runtime",
    "train_time_s",
    "inference_time_s",
    "gpu_memory_mb",
}


@dataclass(frozen=True)
class MetricSummary:
    """One best-model summary row."""

    metric: str
    model: str
    value: float


def numeric_metric_columns(metrics: pd.DataFrame | None) -> list[str]:
    """Return numeric metric columns excluding metadata-like fields."""

    if metrics is None or metrics.empty:
        return []
    excluded = {"horizon", "context_length", "seed"}
    return [
        col
        for col in metrics.select_dtypes(include="number").columns
        if col not in excluded and not col.endswith("_id")
    ]


def metric_direction(metric: str) -> int:
    """Return 1 when higher is better and -1 when lower is better."""

    return -1 if metric.lower() in LOWER_IS_BETTER else 1


def summarize_best_models(metrics: pd.DataFrame | None, primary_metric: str = "smape") -> list[MetricSummary]:
    """Summarize the best model for each metric."""

    if metrics is None or metrics.empty:
        return []
    cols = numeric_metric_columns(metrics)
    summaries: list[MetricSummary] = []
    for metric in cols:
        if "model" not in metrics.columns:
            continue
        grouped = metrics.groupby("model")[metric].mean(numeric_only=True).dropna()
        if grouped.empty:
            continue
        best_model = grouped.idxmin() if metric_direction(metric) < 0 else grouped.idxmax()
        summaries.append(MetricSummary(metric=metric, model=str(best_model), value=float(grouped[best_model])))
    summaries.sort(key=lambda item: (item.metric != primary_metric, item.metric))
    return summaries


def compare_with_baselines(
    current: pd.DataFrame | None,
    baselines: list[tuple[str, pd.DataFrame | None]],
    metric: str = "smape",
) -> pd.DataFrame:
    """Compare current best value with each baseline best value."""

    columns = [
        "baseline",
        "metric",
        "current_model",
        "current_value",
        "baseline_model",
        "baseline_value",
        "absolute_difference",
        "relative_improvement_pct",
    ]
    if current is None or current.empty or metric not in current.columns or "model" not in current.columns:
        return pd.DataFrame(columns=columns)
    current_group = current.groupby("model")[metric].mean(numeric_only=True).dropna()
    if current_group.empty:
        return pd.DataFrame(columns=columns)
    lower = metric_direction(metric) < 0
    current_model = current_group.idxmin() if lower else current_group.idxmax()
    current_value = float(current_group[current_model])
    rows = []
    for name, frame in baselines:
        if frame is None or frame.empty or metric not in frame.columns or "model" not in frame.columns:
            rows.append(
                {
                    "baseline": name,
                    "metric": metric,
                    "current_model": current_model,
                    "current_value": current_value,
                    "baseline_model": "N/A",
                    "baseline_value": math.nan,
                    "absolute_difference": math.nan,
                    "relative_improvement_pct": math.nan,
                }
            )
            continue
        base_group = frame.groupby("model")[metric].mean(numeric_only=True).dropna()
        if base_group.empty:
            continue
        baseline_model = base_group.idxmin() if lower else base_group.idxmax()
        baseline_value = float(base_group[baseline_model])
        absolute = baseline_value - current_value if lower else current_value - baseline_value
        relative = absolute / abs(baseline_value) * 100 if baseline_value != 0 else math.nan
        rows.append(
            {
                "baseline": name,
                "metric": metric,
                "current_model": current_model,
                "current_value": current_value,
                "baseline_model": baseline_model,
                "baseline_value": baseline_value,
                "absolute_difference": absolute,
                "relative_improvement_pct": relative,
            }
        )
    return pd.DataFrame(rows, columns=columns)
