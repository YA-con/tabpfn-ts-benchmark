"""Small controlled TabPFN-TS pilot benchmark."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

from experiments.e0_pilot_results import _evaluate_dataset, _load_pilot_datasets
from src.evaluation.point_metrics import mae, mase, mse, rmse, smape, wape
from src.visualization.report import build_pilot_report


CHECKPOINT = Path(
    os.environ.get(
        "TABPFN_TS_CHECKPOINT",
        "/home/wanyi/zy_test/tabpfn_weights/tabpfn-v3-regressor-v3_20260506_timeseries.ckpt",
    )
)


def _limit_panel(frame: pd.DataFrame, max_series: int, max_context: int, horizon: int) -> pd.DataFrame:
    """Keep a small suffix panel for controlled TabPFN-TS runs."""

    parts: list[pd.DataFrame] = []
    for _, group in frame.sort_values(["unique_id", "ds"]).groupby("unique_id"):
        if len(parts) >= max_series:
            break
        window = group.tail(max_context + horizon)
        if len(window) > horizon:
            parts.append(window)
    if not parts:
        raise ValueError("No series survived the TabPFN-TS pilot limits.")
    return pd.concat(parts, ignore_index=True)


def _predict_tabpfn_ts(
    train: pd.DataFrame,
    test: pd.DataFrame,
    context_length: int,
    checkpoint: Path,
) -> tuple[pd.DataFrame, float]:
    """Predict with the local TabPFN-TS checkpoint."""

    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing TabPFN-TS checkpoint: {checkpoint}")

    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    import torch
    from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    context = train.rename(
        columns={"unique_id": "item_id", "ds": "timestamp", "y": "target"}
    )[["item_id", "timestamp", "target"]].copy()
    future = test.rename(columns={"unique_id": "item_id", "ds": "timestamp"})[
        ["item_id", "timestamp"]
    ].copy()

    start = time.perf_counter()
    pipeline = TabPFNTSPipeline(
        max_context_length=context_length,
        tabpfn_mode=TabPFNMode.LOCAL,
        tabpfn_model_config={
            "model_path": str(checkpoint),
            "device": device,
            "n_estimators": 1,
        },
    )
    raw = pipeline.predict_df(context, future_df=future, quantiles=[0.5])
    inference_time_s = time.perf_counter() - start
    predictions = raw.reset_index().rename(
        columns={"item_id": "unique_id", "timestamp": "ds", "target": "y_hat"}
    )
    joined = test.merge(predictions[["unique_id", "ds", "y_hat"]], on=["unique_id", "ds"])
    return joined, inference_time_s


def _evaluate_tabpfn_dataset(
    dataset: str,
    domain: str,
    horizon: int,
    context_length: int,
    frame: pd.DataFrame,
    run_id: str,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    """Evaluate existing pilot baselines plus TabPFN-TS on a limited panel."""

    baseline_metrics, baseline_predictions = _evaluate_dataset(
        dataset=dataset,
        domain=domain,
        freq="h",
        horizon=horizon,
        context_length=context_length,
        frame=frame,
        run_id=run_id,
    )
    sorted_frame = frame.sort_values(["unique_id", "ds"]).copy()
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in sorted_frame.groupby("unique_id"):
        train_parts.append(group.iloc[:-horizon])
        test_parts.append(group.iloc[-horizon:])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    joined, inference_time_s = _predict_tabpfn_ts(train, test, context_length, CHECKPOINT)
    y_true = joined["y"].to_numpy()
    y_pred = joined["y_hat"].to_numpy()
    metrics = baseline_metrics + [
        {
            "run_id": run_id,
            "model": "tabpfn_ts",
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
            "train_time_s": 0.0,
            "inference_time_s": inference_time_s,
            "gpu_memory_mb": 0.0,
        }
    ]
    joined["model"] = "tabpfn_ts"
    joined["dataset"] = dataset
    joined["domain"] = domain
    predictions = baseline_predictions + [
        joined[["dataset", "domain", "model", "unique_id", "ds", "y", "y_hat"]]
    ]
    return metrics, predictions


def main() -> None:
    """Run a small TabPFN-TS pilot and generate report artifacts."""

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / os.environ.get("TABPFN_TS_OUTPUT_DIR", "results/tabpfn_ts_pilot")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("tabpfn_ts_%Y%m%d_%H%M%S")
    max_series = int(os.environ.get("TABPFN_TS_MAX_SERIES", "1"))
    context_length = int(os.environ.get("TABPFN_TS_CONTEXT", "48"))
    horizon = int(os.environ.get("TABPFN_TS_HORIZON", "6"))
    keep_datasets = set(os.environ.get("TABPFN_TS_DATASETS", "synthetic_energy,stock_provided").split(","))

    all_metrics: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    for dataset, domain, _freq, _horizon, _context_length, frame in _load_pilot_datasets(project_root):
        if dataset not in keep_datasets:
            continue
        limited = _limit_panel(frame, max_series=max_series, max_context=context_length, horizon=horizon)
        metrics, predictions = _evaluate_tabpfn_dataset(
            dataset=dataset,
            domain=domain,
            horizon=horizon,
            context_length=context_length,
            frame=limited,
            run_id=run_id,
        )
        all_metrics.extend(metrics)
        all_predictions.extend(predictions)

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df.to_csv(output_dir / "forecast_metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "forecast_predictions.csv", index=False)
    build_pilot_report(metrics_df, predictions_df, output_dir / "report.html")
    print(metrics_df.groupby("model")["smape"].mean().sort_values())
    print(f"Wrote {output_dir / 'report.html'}")


if __name__ == "__main__":
    main()
