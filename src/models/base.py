"""Shared forecasting model interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class ForecastInput:
    """Standardized input for any model.

    `series` is long-format with columns [unique_id, ds, y] following the Nixtla
    convention.
    """

    series: pd.DataFrame
    horizon: int
    context_length: int | None = None
    freq: str = "D"


@dataclass
class ForecastOutput:
    """Standardized output.

    `predictions` is long-format [unique_id, ds, y_hat] plus optional quantile
    columns y_hat_q{q}.
    """

    predictions: pd.DataFrame
    quantiles: list[float] | None = None
    inference_time_s: float = 0.0


class ForecastModel(ABC):
    """Abstract base class for all benchmark forecasting models."""

    name: str
    requires_training: bool

    @abstractmethod
    def fit(self, data: ForecastInput) -> None:
        """Fit the model on standardized forecasting input."""

    @abstractmethod
    def predict(self, data: ForecastInput) -> ForecastOutput:
        """Generate a standardized forecast output."""
