"""Result schema tests."""

import pandas as pd
import pytest

from src.evaluation.results_schema import FORECAST_RESULT_COLUMNS, validate_result_frame
from src.visualization.specs import list_dashboard_figures


def test_validate_result_frame_orders_columns() -> None:
    """Result validation returns the canonical column order."""

    df = pd.DataFrame(
        [
            {
                "extra": "ignored",
                "run_id": "r1",
                "model": "m",
                "dataset": "d",
                "domain": "energy",
                "horizon": 24,
                "context_length": 168,
                "mse": 1.0,
                "mae": 1.0,
                "rmse": 1.0,
                "smape": 1.0,
                "wape": 1.0,
                "mase": 1.0,
                "train_time_s": 0.0,
                "inference_time_s": 0.1,
                "gpu_memory_mb": 0.0,
            }
        ]
    )

    validated = validate_result_frame(df, FORECAST_RESULT_COLUMNS)

    assert list(validated.columns) == FORECAST_RESULT_COLUMNS


def test_validate_result_frame_rejects_missing_columns() -> None:
    """Missing result columns fail loudly."""

    with pytest.raises(ValueError, match="missing"):
        validate_result_frame(pd.DataFrame({"run_id": ["r1"]}), FORECAST_RESULT_COLUMNS)


def test_dashboard_specs_have_required_columns() -> None:
    """Each planned figure has a name, section, purpose, and input schema."""

    figures = list_dashboard_figures()

    assert figures
    assert all(figure.name for figure in figures)
    assert all(figure.section for figure in figures)
    assert all(figure.purpose for figure in figures)
    assert all(figure.required_columns for figure in figures)
