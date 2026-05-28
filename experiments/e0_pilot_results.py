"""Pilot benchmark run that produces result CSVs and an HTML report."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.materialize import load_processed_dataset, materialize_dataset
from src.evaluation.point_metrics import mae, mase, mse, rmse, smape, wape
from src.reporting import generate_experiment_report
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
    for name, domain, freq, horizon, context_length in [
        ("etth1", "energy", "h", 24, 168),
        ("electricity", "energy", "h", 24, 168),
        ("exchange_rate", "economics", "d", 7, 365),
        ("traffic", "traffic", "h", 24, 168),
        ("weather", "weather", "10min", 144, 1008),
    ]:
        processed_path = project_root / "data" / "processed" / name / "series.parquet"
        if processed_path.exists():
            datasets.append(
                (
                    name,
                    domain,
                    freq,
                    horizon,
                    context_length,
                    load_processed_dataset(name, project_root=project_root),
                )
            )
    processed_stock = project_root / "data" / "processed" / "stock_provided" / "series.parquet"
    if not processed_stock.exists():
        materialize_dataset("stock_provided", project_root=project_root)
    stock = load_processed_dataset("stock_provided", project_root=project_root)
    datasets.append(("stock_provided", "finance", "h", 24, 240, stock))
    return datasets


def _lag_features(history: list[float], lags: list[int]) -> list[float]:
    """Build autoregressive features from normalized history."""

    values = [history[-lag] for lag in lags]
    values.append(float(np.mean(history[-min(3, len(history)) :])))
    values.append(float(np.mean(history[-min(max(lags), len(history)) :])))
    values.append(float(np.std(history[-min(max(lags), len(history)) :])))
    return values


def _fit_sklearn_autoregressor(
    train: pd.DataFrame,
    model_name: str,
    season_length: int,
    max_windows_per_series: int = 360,
) -> tuple[object, dict[str, tuple[float, float]], list[int]]:
    """Fit a lightweight global autoregressive sklearn model."""

    lags = sorted({1, 2, 3, max(1, season_length), max(2, season_length * 2)})
    max_lag = max(lags)
    features: list[list[float]] = []
    targets: list[float] = []
    scales: dict[str, tuple[float, float]] = {}
    for unique_id, group in train.groupby("unique_id"):
        values = group.sort_values("ds")["y"].to_numpy(dtype=float)
        if len(values) <= max_lag:
            continue
        center = float(np.mean(values))
        scale = float(np.std(values))
        if scale == 0.0:
            scale = 1.0
        scales[str(unique_id)] = (center, scale)
        normalized = ((values - center) / scale).tolist()
        candidate_idxs = list(range(max_lag, len(normalized)))
        for idx in candidate_idxs[-max_windows_per_series:]:
            history = normalized[:idx]
            features.append(_lag_features(history, lags))
            targets.append(float(normalized[idx]))
    if not features:
        raise ValueError(f"Not enough training windows for {model_name}.")
    if model_name == "ridge_ar":
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    elif model_name == "hist_gradient_boosting_ar":
        estimator = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=42,
        )
    elif model_name == "lightgbm_ar":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("lightgbm_ar requires lightgbm to be installed.") from exc
        estimator = LGBMRegressor(
            n_estimators=220,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=12,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbose=-1,
        )
    elif model_name == "xgboost_ar":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("xgboost_ar requires xgboost to be installed.") from exc
        estimator = XGBRegressor(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=2,
        )
    else:
        raise ValueError(f"Unknown sklearn model: {model_name}")
    estimator.fit(np.asarray(features), np.asarray(targets))
    return estimator, scales, lags


def _predict_sklearn_autoregressor(
    model: object,
    scales: dict[str, tuple[float, float]],
    lags: list[int],
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Recursively forecast each series with a fitted sklearn autoregressor."""

    prediction_parts: list[pd.DataFrame] = []
    for unique_id, test_group in test.groupby("unique_id"):
        history_values = train[train["unique_id"] == unique_id].sort_values("ds")["y"].to_numpy(dtype=float)
        if str(unique_id) not in scales or len(history_values) < max(lags):
            pred_group = test_group[["unique_id", "ds", "y"]].copy()
            pred_group["y_hat"] = float(np.mean(history_values))
            prediction_parts.append(pred_group)
            continue
        center, scale = scales[str(unique_id)]
        history = ((history_values - center) / scale).tolist()
        y_hat: list[float] = []
        for _ in range(len(test_group)):
            x = np.asarray([_lag_features(history, lags)])
            pred_norm = float(model.predict(x)[0])
            history.append(pred_norm)
            y_hat.append(pred_norm * scale + center)
        pred_group = test_group[["unique_id", "ds", "y"]].copy()
        pred_group["y_hat"] = y_hat
        prediction_parts.append(pred_group)
    return pd.concat(prediction_parts, ignore_index=True)


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
    model_names = [
        "dummy_mean",
        "seasonal_naive",
        "moving_average",
        "linear_trend",
        "ridge_ar",
        "hist_gradient_boosting_ar",
        "lightgbm_ar",
        "xgboost_ar",
    ]
    for model_name in model_names:
        start_fit = time.perf_counter()
        train_stats = train.groupby("unique_id")["y"].mean()
        sklearn_state = None
        if model_name in {"ridge_ar", "hist_gradient_boosting_ar", "lightgbm_ar", "xgboost_ar"}:
            sklearn_state = _fit_sklearn_autoregressor(train, model_name, season_length)
        train_time = time.perf_counter() - start_fit
        start_predict = time.perf_counter()
        prediction_parts: list[pd.DataFrame] = []
        if sklearn_state is not None:
            estimator, scales, lags = sklearn_state
            joined = _predict_sklearn_autoregressor(estimator, scales, lags, train, test)
        else:
            for unique_id, test_group in test.groupby("unique_id"):
                history = train[train["unique_id"] == unique_id].sort_values("ds")
                pred_group = test_group[["unique_id", "ds", "y"]].copy()
                if model_name == "dummy_mean":
                    pred_group["y_hat"] = float(train_stats.loc[unique_id])
                elif model_name == "seasonal_naive":
                    seasonal_values = history["y"].tail(season_length).to_numpy()
                    repeats = int(np.ceil(len(pred_group) / len(seasonal_values)))
                    pred_group["y_hat"] = np.tile(seasonal_values, repeats)[: len(pred_group)]
                elif model_name == "moving_average":
                    window = min(context_length, max(horizon, 24), len(history))
                    pred_group["y_hat"] = float(history["y"].tail(window).mean())
                elif model_name == "linear_trend":
                    window = min(context_length, len(history))
                    values = history["y"].tail(window).to_numpy(dtype=float)
                    x = np.arange(len(values), dtype=float)
                    slope, intercept = np.polyfit(x, values, deg=1)
                    future_x = np.arange(len(values), len(values) + len(pred_group), dtype=float)
                    pred_group["y_hat"] = intercept + slope * future_x
                else:
                    raise ValueError(f"Unknown pilot model: {model_name}")
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
    report_path = generate_experiment_report(
        run_dir=output_dir,
        out_dir=project_root / "reports" / output_dir.name,
        config_paths=[project_root / "configs" / "config.yaml"],
    )
    print(f"Wrote {output_dir / 'report.html'}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
