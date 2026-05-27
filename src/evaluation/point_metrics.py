"""Point forecasting metrics."""

import numpy as np


def _validate_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce metric inputs."""

    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true {true.shape} vs y_pred {pred.shape}.")
    if true.size == 0:
        raise ValueError("Metric inputs must not be empty.")
    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
        raise ValueError("Metric inputs must contain only finite values.")
    return true, pred


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""

    true, pred = _validate_arrays(y_true, y_pred)
    return float(np.mean((true - pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""

    true, pred = _validate_arrays(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""

    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error in percent."""

    true, pred = _validate_arrays(y_true, y_pred)
    if np.any(true == 0):
        raise ValueError("MAPE is undefined when y_true contains zero.")
    return float(np.mean(np.abs((true - pred) / true)) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric mean absolute percentage error in percent."""

    true, pred = _validate_arrays(y_true, y_pred)
    denominator = np.abs(true) + np.abs(pred)
    if np.any(denominator == 0):
        raise ValueError("SMAPE is undefined when y_true and y_pred are both zero.")
    return float(np.mean(2.0 * np.abs(pred - true) / denominator) * 100.0)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted absolute percentage error in percent."""

    true, pred = _validate_arrays(y_true, y_pred)
    denominator = np.sum(np.abs(true))
    if denominator == 0:
        raise ValueError("WAPE is undefined when the absolute sum of y_true is zero.")
    return float(np.sum(np.abs(true - pred)) / denominator * 100.0)


def mase(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute scaled error using an in-sample naive one-step denominator."""

    true, pred = _validate_arrays(y_true, y_pred)
    if true.size < 2:
        raise ValueError("MASE requires at least two observations.")
    scale = np.mean(np.abs(np.diff(true)))
    if scale == 0:
        raise ValueError("MASE is undefined when the naive scale is zero.")
    return float(mae(true, pred) / scale)
