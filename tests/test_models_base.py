"""Forecast model interface tests."""

import pandas as pd

from src.models.base import ForecastInput, ForecastModel, ForecastOutput
from src.models.statistical import DummyMeanModel


def test_dummy_mean_model_contract() -> None:
    """DummyMeanModel implements the ForecastModel interface."""

    series = pd.DataFrame(
        {
            "unique_id": ["a", "a", "a"],
            "ds": pd.date_range("2024-01-01", periods=3, freq="D"),
            "y": [1.0, 2.0, 3.0],
        }
    )
    model = DummyMeanModel()
    assert isinstance(model, ForecastModel)
    model.fit(ForecastInput(series=series, horizon=2, freq="D"))
    output = model.predict(ForecastInput(series=series, horizon=2, freq="D"))

    assert isinstance(output, ForecastOutput)
    assert list(output.predictions.columns) == ["unique_id", "ds", "y_hat"]
    assert output.predictions["y_hat"].tolist() == [2.0, 2.0]
