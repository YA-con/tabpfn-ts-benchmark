"""Pilot benchmark run that produces result CSVs and an HTML report."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.materialize import load_processed_dataset, materialize_dataset
from src.evaluation.point_metrics import mae, mase, mse, rmse, smape, wape
from src.visualization.report import build_pilot_report


def _synthetic_panel(name: str, domain: str, freq: str, n_series: int, n_obs: int) -> pd.DataFrame:
    """Create deterministic synthetic panels with domain-shaped dynamics."""

    dates = pd.date_range("2024-01-01", periods=n_obs, freq=freq)
    rows: list[dict[str, object]] = []
    x = np.arange(n_obs)
    for idx in range(n_series):
        if domain == "energy":
            y = 10 + idx + 2.5 * np.sin(2 * np.pi * x / 24) + 0.02 * x
        elif domain == "traffic":
            y = 30 + 8 * np.sin(2 * np.pi * x / 24) + 3 * np.sin(2 * np.pi * x / 168) + idx
        elif domain == "weather":
            y = 18 + 5 * np.sin(2 * np.pi * x / 144) + 0.5 * np.cos(2 * np.pi * x / 36) + idx
        else:
            y = 1.0 + 0.01 * x + 0.08 * np.sin(2 * np.pi * x / 7) + idx * 0.03
        rows.extend(
            {"unique_id": f"{name}_{idx}", "ds": ds, "y": float(value)}
            for ds, value in zip(dates, y, strict=True)
        )
    return pd.DataFrame(rows)


def _load_pilot_datasets(project_root: Path) -> list[tuple[str, str, str, int, int, pd.DataFrame]]:
    """Load pilot datasets as tuples of metadata plus panel."""

    datasets: list[tuple[str, str, str, int, int, pd.DataFrame]] = [
        ("synthetic_energy", "energy", "h", 24, 168, _synthetic_panel("energy", "energy", "h", 3, 240)),
        (
            "synthetic_traffic",
            "traffic",
            "h",
            24,
            168,
            _synthetic_panel("traffic", "traffic", "h", 3, 240),
        ),
        (
            "synthetic_weather",
            "weather",
            "10min",
            144,
            1008,
            _synthetic_panel("weather", "weather", "10min", 2, 1200),
        ),
        (
            "synthetic_exchange",
            "economics",
            "d",
            7,
            365,
            _synthetic_panel("exchange", "economics", "d", 3, 420),
        ),
    ]
    processed_stock = project_root / "data" / "processed" / "stock_provided" / "series.parquet"
    if not processed_stock.exists():
        materialize_dataset("stock_provided", project_root=project_root, max_files=5)
    stock = load_processed_dataset("stock_provided", project_root=project_root)
    datasets.append(("stock_provided_sample", "finance", "h", 24, 240, stock))
    return datasets


def _evaluate_dataset(
    dataset: str,
    domain: str,
    freq: str,
    horizon: int,
    context_length: int,
    frame: pd.DataFrame,
    run_id: str,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    """Run pilot baselines for one dataset."""

    sorted_frame = frame.sort_values(["unique_id", "ds"]).copy()
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in sorted_frame.groupby("unique_id"):
        if len(group) <= horizon:
            continue
        train_parts.append(group.iloc[:-horizon])
        test_parts.append(group.iloc[-horizon:])
    if not train_parts or not test_parts:
        raise ValueError(f"{dataset} does not contain enough observations for horizon={horizon}.")
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    season_length = max(1, min(horizon, 24))
    for model_name in ["dummy_mean", "seasonal_naive"]:
        start_fit = time.perf_counter()
        train_stats = train.groupby("unique_id")["y"].mean()
        train_time = time.perf_counter() - start_fit
        start_predict = time.perf_counter()
        prediction_parts: list[pd.DataFrame] = []
        for unique_id, test_group in test.groupby("unique_id"):
            history = train[train["unique_id"] == unique_id].sort_values("ds")
            pred_group = test_group[["unique_id", "ds", "y"]].copy()
            if model_name == "dummy_mean":
                pred_group["y_hat"] = float(train_stats.loc[unique_id])
            else:
                seasonal_values = history["y"].tail(season_length).to_numpy()
                repeats = int(np.ceil(len(pred_group) / len(seasonal_values)))
                pred_group["y_hat"] = np.tile(seasonal_values, repeats)[: len(pred_group)]
            prediction_parts.append(pred_group)
        joined = pd.concat(prediction_parts, ignore_index=True)
        inference_time_s = time.perf_counter() - start_predict
        if len(joined) != len(test):
            raise ValueError(f"{model_name} produced {len(joined)} aligned rows for {len(test)} targets.")
        y_true = joined["y"].to_numpy()
        y_pred = joined["y_hat"].to_numpy()
        metrics.append(
            {
                "run_id": run_id,
                "model": model_name,
                "dataset": dataset,
                "domain": domain,
                "horizon": horizon,
                "context_length": context_length,
                "mse": mse(y_true, y_pred),
                "mae": mae(y_true, y_pred),
                "rmse": rmse(y_true, y_pred),
                "smape": smape(y_true, y_pred),
                "wape": wape(y_true, y_pred),
                "mase": mase(y_true, y_pred),
                "train_time_s": train_time,
                "inference_time_s": inference_time_s,
                "gpu_memory_mb": 0.0,
            }
        )
        joined["model"] = model_name
        joined["dataset"] = dataset
        joined["domain"] = domain
        predictions.append(joined[["dataset", "domain", "model", "unique_id", "ds", "y", "y_hat"]])
    return metrics, predictions


def main() -> None:
    """Run the pilot experiment and generate visual outputs."""

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "results" / "pilot_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("pilot_%Y%m%d_%H%M%S")

    all_metrics: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    for dataset, domain, freq, horizon, context_length, frame in _load_pilot_datasets(project_root):
        metrics, predictions = _evaluate_dataset(
            dataset=dataset,
            domain=domain,
            freq=freq,
            horizon=horizon,
            context_length=context_length,
            frame=frame,
            run_id=run_id,
        )
        all_metrics.extend(metrics)
        all_predictions.extend(predictions)

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df.to_csv(output_dir / "forecast_metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "forecast_predictions.csv", index=False)
    build_pilot_report(metrics_df, predictions_df, output_dir / "report.html")
    print(f"Wrote {output_dir / 'report.html'}")


if __name__ == "__main__":
    main()
