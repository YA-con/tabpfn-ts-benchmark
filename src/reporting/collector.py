"""Collect experiment artifacts without assuming one training framework."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass
class ExperimentArtifacts:
    """Normalized view of one experiment run directory."""

    run_dir: Path
    run_id: str
    metrics: pd.DataFrame | None = None
    predictions: pd.DataFrame | None = None
    config: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _load_table(path: Path, warnings: list[str]) -> pd.DataFrame | None:
    """Load CSV or JSON tables while keeping report generation non-fatal."""

    if not path.exists():
        warnings.append(f"缺失文件: {path.name}")
        return None
    try:
        if path.suffix == ".csv":
            return pd.read_csv(path)
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return pd.DataFrame(data)
            if isinstance(data, dict):
                return pd.DataFrame([data])
    except Exception as exc:  # pragma: no cover - defensive reporting path
        warnings.append(f"无法读取 {path.name}: {exc}")
    return None


def _read_config_file(path: Path, warnings: list[str]) -> dict[str, Any]:
    """Read JSON/YAML config files with a small compatibility surface."""

    try:
        if path.suffix in {".yaml", ".yml"}:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        else:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            return loaded
        warnings.append(f"配置文件不是对象格式: {path}")
    except Exception as exc:  # pragma: no cover - defensive reporting path
        warnings.append(f"无法读取配置 {path}: {exc}")
    return {}


def collect_experiment_results(
    run_dir: str | Path,
    config_paths: list[str | Path] | None = None,
) -> ExperimentArtifacts:
    """Collect metrics, predictions, config, and warnings for one run."""

    root = Path(run_dir).resolve()
    warnings: list[str] = []
    metrics = _load_table(root / "forecast_metrics.csv", warnings)
    predictions = _load_table(root / "forecast_predictions.csv", warnings)
    if metrics is not None and metrics.empty:
        warnings.append("metrics 文件为空。")
    if predictions is not None and predictions.empty:
        warnings.append("predictions 文件为空。")

    config: dict[str, Any] = {}
    for candidate in [root / "config.yaml", root / "config.yml", root / "config.json"]:
        if candidate.exists():
            config[candidate.name] = _read_config_file(candidate, warnings)
    for path in config_paths or []:
        candidate = Path(path).resolve()
        if candidate.exists():
            config[str(candidate)] = _read_config_file(candidate, warnings)
        else:
            warnings.append(f"配置路径不存在: {candidate}")

    if metrics is not None and "run_id" in metrics.columns and not metrics["run_id"].dropna().empty:
        run_id = str(metrics["run_id"].dropna().iloc[0])
    else:
        run_id = root.name
        warnings.append("未找到 run_id，使用目录名作为 run_id。")

    return ExperimentArtifacts(
        run_dir=root,
        run_id=run_id,
        metrics=metrics,
        predictions=predictions,
        config=config,
        warnings=warnings,
    )


def collect_baseline_results(baseline_dirs: list[str | Path] | None) -> list[ExperimentArtifacts]:
    """Collect multiple baseline directories."""

    baselines: list[ExperimentArtifacts] = []
    for path in baseline_dirs or []:
        baselines.append(collect_experiment_results(path))
    return baselines


def collect_environment_info() -> dict[str, Any]:
    """Collect lightweight runtime environment metadata."""

    git_head = "N/A"
    git_status = "N/A"
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        git_status = subprocess.check_output(
            ["git", "status", "--short"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_head": git_head,
        "git_dirty": bool(git_status and git_status != "N/A"),
    }
