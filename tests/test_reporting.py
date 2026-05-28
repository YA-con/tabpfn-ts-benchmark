"""Reusable experiment report tests."""

from pathlib import Path

import pandas as pd

from src.reporting import generate_experiment_report


def _write_run(path: Path, run_id: str, smape: float) -> None:
    path.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "run_id": run_id,
                "model": "model_a",
                "dataset": "demo",
                "domain": "energy",
                "horizon": 2,
                "context_length": 8,
                "mse": smape / 10,
                "mae": smape / 20,
                "rmse": smape / 5,
                "smape": smape,
                "wape": smape / 2,
                "mase": smape / 3,
                "train_time_s": 0.1,
                "inference_time_s": 0.2,
                "gpu_memory_mb": 0.0,
            }
        ]
    ).to_csv(path / "forecast_metrics.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["demo", "demo", "demo"],
            "domain": ["energy", "energy", "energy"],
            "model": ["model_a", "model_a", "model_a"],
            "unique_id": ["x", "x", "x"],
            "ds": pd.date_range("2024-01-01", periods=3),
            "y": [1.0, 2.0, 3.0],
            "y_hat": [1.1, 1.8, 3.2],
        }
    ).to_csv(path / "forecast_predictions.csv", index=False)


def test_generate_experiment_report(tmp_path: Path) -> None:
    """Report generator writes HTML, metadata, summary, and figures."""

    run = tmp_path / "run"
    baseline = tmp_path / "baseline"
    out = tmp_path / "report"
    _write_run(run, "current", 5.0)
    _write_run(baseline, "baseline", 7.0)

    html = generate_experiment_report(run, out, baseline_dirs=[baseline])

    assert html.exists()
    assert (out / "metrics.json").exists()
    assert (out / "summary.csv").exists()
    assert list((out / "assets").glob("*.svg"))
    text = html.read_text(encoding="utf-8")
    assert "核心指标卡片" in text
    assert "Baseline 对比" in text
    assert "Matplotlib 可视化图表" in text
