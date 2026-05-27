"""Canonical result schemas for benchmark outputs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


FORECAST_RESULT_COLUMNS = [
    "run_id",
    "model",
    "dataset",
    "domain",
    "horizon",
    "context_length",
    "mse",
    "mae",
    "rmse",
    "smape",
    "wape",
    "mase",
    "train_time_s",
    "inference_time_s",
    "gpu_memory_mb",
]

FINANCE_RESULT_COLUMNS = [
    "run_id",
    "model",
    "dataset",
    "universe",
    "horizon",
    "ic",
    "rank_ic",
    "icir",
    "annual_return",
    "sharpe",
    "max_drawdown",
    "turnover",
    "transaction_cost_bps",
    "stability_pass",
]


@dataclass(frozen=True)
class ForecastResultRow:
    """One row of cross-domain forecasting benchmark metrics."""

    run_id: str
    model: str
    dataset: str
    domain: str
    horizon: int
    context_length: int
    mse: float
    mae: float
    rmse: float
    smape: float
    wape: float
    mase: float
    train_time_s: float
    inference_time_s: float
    gpu_memory_mb: float


@dataclass(frozen=True)
class FinanceResultRow:
    """One row of finance signal and backtest metrics."""

    run_id: str
    model: str
    dataset: str
    universe: str
    horizon: int
    ic: float
    rank_ic: float
    icir: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    turnover: float
    transaction_cost_bps: float
    stability_pass: bool


def validate_result_frame(df: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    """Validate a result frame against an ordered column schema."""

    missing = [column for column in expected_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Result frame is missing columns: {missing}")
    return df[expected_columns].copy()
