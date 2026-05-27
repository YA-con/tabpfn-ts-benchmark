"""Zero-shot smoke experiment on synthetic sine-wave data."""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.evaluation.point_metrics import mae, mse
from src.models.base import ForecastInput
from src.models.statistical import DummyMeanModel, SeasonalNaiveModel
from src.utils.seeding import set_seed


def _make_sine_data(n_series: int, n_observations: int, freq: str, seed: int) -> pd.DataFrame:
    """Create deterministic multi-series sine-wave data in Nixtla format."""

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_observations, freq=freq)
    rows: list[dict[str, object]] = []
    for idx in range(n_series):
        phase = idx * np.pi / 6.0
        signal = np.sin(np.arange(n_observations) * 2.0 * np.pi / 12.0 + phase)
        trend = 0.01 * np.arange(n_observations)
        noise = rng.normal(0.0, 0.03, size=n_observations)
        values = signal + trend + noise
        rows.extend(
            {"unique_id": f"sine_{idx}", "ds": ds, "y": float(y)}
            for ds, y in zip(dates, values, strict=True)
        )
    return pd.DataFrame(rows)


def _score_predictions(test: pd.DataFrame, predictions: pd.DataFrame, model_name: str) -> dict[str, float | str]:
    """Join predictions to held-out data and compute point metrics."""

    joined = test.merge(predictions, on=["unique_id", "ds"], how="inner")
    if len(joined) != len(test):
        raise ValueError(f"{model_name} produced {len(joined)} aligned rows for {len(test)} targets.")
    return {
        "model": model_name,
        "mse": mse(joined["y"].to_numpy(), joined["y_hat"].to_numpy()),
        "mae": mae(joined["y"].to_numpy(), joined["y_hat"].to_numpy()),
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the Phase 1 smoke experiment."""

    os.environ.setdefault("WANDB_MODE", str(cfg.wandb.mode))
    import wandb

    set_seed(int(cfg.seed))
    experiment_cfg = cfg.experiment
    data = _make_sine_data(
        n_series=int(experiment_cfg.n_series),
        n_observations=int(experiment_cfg.n_observations),
        freq=str(experiment_cfg.freq),
        seed=int(cfg.seed),
    )
    horizon = int(experiment_cfg.horizon)
    cutoff = data["ds"].sort_values().unique()[-horizon]
    train = data[data["ds"] < cutoff].copy()
    test = data[data["ds"] >= cutoff].copy()

    forecast_input = ForecastInput(
        series=train,
        horizon=horizon,
        context_length=int(cfg.dataset.context_length),
        freq=str(experiment_cfg.freq),
    )
    models = [
        DummyMeanModel(),
        SeasonalNaiveModel(season_length=int(experiment_cfg.season_length)),
    ]

    run = wandb.init(
        project=str(cfg.wandb.project),
        mode=str(cfg.wandb.mode),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    results: list[dict[str, float | str]] = []
    for model in models:
        model.fit(forecast_input)
        output = model.predict(forecast_input)
        row = _score_predictions(test, output.predictions, model.name)
        row["inference_time_s"] = output.inference_time_s
        results.append(row)
        wandb.log({f"{model.name}/mse": row["mse"], f"{model.name}/mae": row["mae"]})

    results_df = pd.DataFrame(results)
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / str(cfg.output_dir) / str(experiment_cfg.results_subdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "metrics.csv", index=False)
    run.finish()


if __name__ == "__main__":
    main()
