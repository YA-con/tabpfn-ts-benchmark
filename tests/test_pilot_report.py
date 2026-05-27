"""Pilot report generation tests."""

from pathlib import Path

import pandas as pd

from src.visualization.report import build_pilot_report


def test_build_pilot_report(tmp_path: Path) -> None:
    """Pilot report writes a self-contained HTML artifact."""

    metrics = pd.DataFrame(
        [
            {
                "model": "dummy_mean",
                "dataset": "demo",
                "domain": "energy",
                "smape": 10.0,
                "mse": 1.0,
                "mae": 1.0,
                "rmse": 1.0,
                "wape": 1.0,
                "mase": 1.0,
            },
            {
                "model": "seasonal_naive",
                "dataset": "demo",
                "domain": "energy",
                "smape": 5.0,
                "mse": 0.5,
                "mae": 0.5,
                "rmse": 0.7,
                "wape": 0.5,
                "mase": 0.5,
            },
        ]
    )
    predictions = pd.DataFrame(
        {
            "dataset": ["demo", "demo"],
            "domain": ["energy", "energy"],
            "model": ["seasonal_naive", "seasonal_naive"],
            "unique_id": ["a", "a"],
            "ds": pd.date_range("2024-01-01", periods=2),
            "y": [1.0, 2.0],
            "y_hat": [1.1, 1.9],
        }
    )
    output = tmp_path / "report.html"

    build_pilot_report(metrics, predictions, output)

    text = output.read_text(encoding="utf-8")
    assert "TabPFN-TS Pilot Results" in text
    assert "<svg" in text
