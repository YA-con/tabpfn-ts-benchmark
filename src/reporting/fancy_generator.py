"""Generate non-invasive fancy demo dashboards from existing experiment outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.reporting.collector import collect_baseline_results, collect_environment_info, collect_experiment_results
from src.reporting.comparison import compare_with_baselines
from src.reporting.fancy_figures import generate_fancy_figures
from src.reporting.fancy_html import render_fancy_dashboard
from src.reporting.generator import _summary_csv
from src.reporting.style_config import get_theme


def _json_safe(value: Any) -> Any:
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


def generate_fancy_demo_report(
    run_dir: str | Path,
    out_dir: str | Path,
    *,
    baseline_dirs: list[str | Path] | None = None,
    config_paths: list[str | Path] | None = None,
    primary_metric: str = "smape",
    theme_name: str = "dark_premium",
    is_demo: bool = True,
) -> Path:
    """Generate one fancy dashboard without touching training/evaluation logic."""

    artifacts = collect_experiment_results(run_dir, config_paths=config_paths)
    baselines = collect_baseline_results(baseline_dirs)
    baseline_pairs = [(item.run_dir.name, item.metrics) for item in baselines]
    comparison = compare_with_baselines(artifacts.metrics, baseline_pairs, metric=primary_metric)
    baseline_metrics = baselines[0].metrics if baselines else None
    output = Path(out_dir).resolve()
    assets = output / "assets"
    output.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    theme = get_theme(theme_name)
    figures, figure_warnings = generate_fancy_figures(
        artifacts.metrics,
        artifacts.predictions,
        baseline_metrics,
        comparison,
        assets,
        theme,
        primary_metric=primary_metric,
    )
    summary = _summary_csv(artifacts.metrics, primary_metric)
    summary.to_csv(output / "summary.csv", index=False)
    warnings = artifacts.warnings + [w for item in baselines for w in item.warnings] + figure_warnings
    metadata = {
        "report_type": "fancy_demo",
        "theme": theme_name,
        "run_id": artifacts.run_id,
        "run_dir": str(artifacts.run_dir),
        "primary_metric": primary_metric,
        "is_demo": is_demo,
        "demo_note": "训练曲线仅在缺少真实日志时作为视觉占位，不参与实验结论。",
        "warnings": warnings,
        "summary": summary.to_dict(orient="records"),
        "baseline_comparison": comparison.to_dict(orient="records"),
        "figures": figures,
    }
    (output / "metrics_demo.json").write_text(
        json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    render_fancy_dashboard(
        output / "index.html",
        theme=theme,
        run_id=artifacts.run_id,
        run_dir=artifacts.run_dir,
        metrics=artifacts.metrics,
        predictions=artifacts.predictions,
        comparison=comparison,
        figures=figures,
        config=artifacts.config,
        environment=collect_environment_info(),
        warnings=warnings,
        primary_metric=primary_metric,
        is_demo=is_demo,
    )
    return output / "index.html"
