"""Quantitative finance metrics."""

import numpy as np
import pandas as pd


def _validate_vector(values: np.ndarray | pd.Series, name: str) -> np.ndarray:
    """Validate a one-dimensional finite numeric vector."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def information_coefficient(predictions: np.ndarray | pd.Series, returns: np.ndarray | pd.Series) -> float:
    """Pearson correlation between predictions and realized returns."""

    pred = _validate_vector(predictions, "predictions")
    ret = _validate_vector(returns, "returns")
    if pred.shape != ret.shape:
        raise ValueError("predictions and returns must have the same shape.")
    if np.std(pred) == 0 or np.std(ret) == 0:
        raise ValueError("Information coefficient is undefined for constant vectors.")
    return float(np.corrcoef(pred, ret)[0, 1])


def rank_ic(predictions: np.ndarray | pd.Series, returns: np.ndarray | pd.Series) -> float:
    """Spearman rank correlation between predictions and realized returns."""

    pred = pd.Series(_validate_vector(predictions, "predictions")).rank(method="average").to_numpy()
    ret = pd.Series(_validate_vector(returns, "returns")).rank(method="average").to_numpy()
    return information_coefficient(pred, ret)


def ic_ir(ic_series: np.ndarray | pd.Series) -> float:
    """Information ratio of an IC time series."""

    values = _validate_vector(ic_series, "ic_series")
    std = np.std(values, ddof=1)
    if std == 0:
        raise ValueError("IC IR is undefined when IC standard deviation is zero.")
    return float(np.mean(values) / std)


def sharpe_ratio(returns: np.ndarray | pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio with zero risk-free rate."""

    values = _validate_vector(returns, "returns")
    std = np.std(values, ddof=1)
    if std == 0:
        raise ValueError("Sharpe ratio is undefined when return standard deviation is zero.")
    return float(np.mean(values) / std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: np.ndarray | pd.Series) -> float:
    """Maximum drawdown as a positive fraction of peak equity."""

    equity = _validate_vector(equity_curve, "equity_curve")
    if np.any(equity <= 0):
        raise ValueError("equity_curve must be strictly positive.")
    running_peak = np.maximum.accumulate(equity)
    drawdowns = equity / running_peak - 1.0
    return float(abs(np.min(drawdowns)))
