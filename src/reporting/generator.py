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


def _build_conclusions(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    primary_metric: str,
) -> list[str]:
    """Create concise, data-grounded conclusions without fabricating missing signals."""

    conclusions: list[str] = []
    if not summary.empty and {"model", primary_metric}.issubset(summary.columns):
        best = summary.sort_values(primary_metric).iloc[0]
        conclusions.append(
            f"主指标 {primary_metric.upper()} 下，当前 run 的最佳模型是 "
            f"{best['model']}，平均值为 {float(best[primary_metric]):.4g}。"
        )
    if not comparison.empty:
        for _, row in comparison.iterrows():
            if pd.notna(row.get("relative_improvement_pct")):
                direction = "提升" if float(row["relative_improvement_pct"]) >= 0 else "下降"
                conclusions.append(
                    f"相对 baseline {row['baseline']}，当前最佳模型在 {primary_metric.upper()} 上"
                    f"{direction} {abs(float(row['relative_improvement_pct'])):.2f}% "
                    f"（absolute={float(row['absolute_difference']):+.4g}）。"
                )
    if not conclusions:
        conclusions.append("当前数据不足以生成自动结论；报告仅展示可用的真实指标与诊断图。")
    return conclusions


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
    summary = _summary_csv(artifacts.metrics, primary_metric=primary_metric)
    conclusions = _build_conclusions(summary, comparison, primary_metric)
    figures = generate_matplotlib_figures(
        artifacts.metrics,
        artifacts.predictions,
        assets,
        primary_metric=primary_metric,
        comparison=comparison,
    )
    summary.to_csv(output / "summary.csv", index=False)
    metadata = {
        "run_id": artifacts.run_id,
        "run_dir": str(artifacts.run_dir),
        "primary_metric": primary_metric,
        "warnings": artifacts.warnings + [w for baseline in baselines for w in baseline.warnings],
        "summary": summary.to_dict(orient="records"),
        "baseline_comparison": comparison.to_dict(orient="records"),
        "figures": figures,
        "conclusions": conclusions,
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
        conclusions=conclusions,
    )
    return output / "index.html"
