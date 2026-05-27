"""Metric correctness tests."""

import numpy as np
import pytest

from src.evaluation.point_metrics import mae, mape, mase, mse, rmse, smape, wape
from src.evaluation.quant_metrics import (
    ic_ir,
    information_coefficient,
    max_drawdown,
    rank_ic,
    sharpe_ratio,
)


def test_point_metrics_known_values() -> None:
    """Point metrics match hand-computable examples."""

    y_true = np.array([1.0, 2.0, 4.0, 8.0])
    y_pred = np.array([1.0, 3.0, 5.0, 7.0])

    assert mse(y_true, y_pred) == pytest.approx(0.75)
    assert mae(y_true, y_pred) == pytest.approx(0.75)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(0.75))
    assert mape(y_true, y_pred) == pytest.approx(21.875)
    expected_smape = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
    assert smape(y_true, y_pred) == pytest.approx(expected_smape)
    assert wape(y_true, y_pred) == pytest.approx(20.0)
    assert mase(y_true, y_pred) == pytest.approx(0.75 / (7.0 / 3.0))


def test_quant_metrics_known_values() -> None:
    """Quant metrics match deterministic examples."""

    predictions = np.array([0.1, 0.4, 0.2, 0.3])
    returns = np.array([0.0, 0.5, 0.1, 0.2])
    ic_values = np.array([0.1, 0.2, 0.3])
    strategy_returns = np.array([0.01, 0.02, -0.01, 0.03])
    equity = np.array([1.0, 1.2, 1.1, 1.5, 1.2])

    assert information_coefficient(predictions, returns) == pytest.approx(
        np.corrcoef(predictions, returns)[0, 1]
    )
    assert rank_ic(predictions, returns) == pytest.approx(1.0)
    assert ic_ir(ic_values) == pytest.approx(np.mean(ic_values) / np.std(ic_values, ddof=1))
    assert sharpe_ratio(strategy_returns) == pytest.approx(
        np.mean(strategy_returns) / np.std(strategy_returns, ddof=1) * np.sqrt(252)
    )
    assert max_drawdown(equity) == pytest.approx(0.2)
