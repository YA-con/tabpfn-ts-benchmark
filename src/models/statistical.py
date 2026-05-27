"""Statistical and lightweight baseline models."""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from src.models.base import ForecastInput, ForecastModel, ForecastOutput


@dataclass
class DummyMeanModel(ForecastModel):
    """Predict each series with its in-sample mean."""

    name: str = "dummy_mean"
    requires_training: bool = True

    def fit(self, data: ForecastInput) -> None:
        self._means = data.series.groupby("unique_id")["y"].mean()

    def predict(self, data: ForecastInput) -> ForecastOutput:
        if not hasattr(self, "_means"):
            raise RuntimeError("DummyMeanModel must be fitted before predict().")

        start = time.perf_counter()
        rows: list[dict[str, object]] = []
        for unique_id, group in data.series.groupby("unique_id"):
            last_ds = pd.to_datetime(group["ds"]).max()
            future_dates = pd.date_range(last_ds, periods=data.horizon + 1, freq=data.freq)[1:]
            y_hat = float(self._means.loc[unique_id])
            rows.extend({"unique_id": unique_id, "ds": ds, "y_hat": y_hat} for ds in future_dates)
        predictions = pd.DataFrame(rows)
        return ForecastOutput(predictions=predictions, inference_time_s=time.perf_counter() - start)


@dataclass
class SeasonalNaiveModel(ForecastModel):
    """Seasonal naive baseline backed by statsforecast."""

    season_length: int = 1
    name: str = "seasonal_naive"
    requires_training: bool = False

    def fit(self, data: ForecastInput) -> None:
        """No-op for the statsforecast seasonal naive baseline."""

    def predict(self, data: ForecastInput) -> ForecastOutput:
        start = time.perf_counter()
        try:
            from statsforecast import StatsForecast
            from statsforecast.models import SeasonalNaive
        except ImportError as exc:
            raise ImportError("SeasonalNaiveModel requires statsforecast to be installed.") from exc

        model = StatsForecast(models=[SeasonalNaive(season_length=self.season_length)], freq=data.freq)
        forecast = model.forecast(df=data.series, h=data.horizon).reset_index()
        predictions = forecast.rename(columns={"SeasonalNaive": "y_hat"})[
            ["unique_id", "ds", "y_hat"]
        ]
        return ForecastOutput(predictions=predictions, inference_time_s=time.perf_counter() - start)
