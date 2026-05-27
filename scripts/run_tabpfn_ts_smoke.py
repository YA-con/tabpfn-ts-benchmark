"""Run a minimal offline TabPFN-TS smoke test."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "TABPFN_TS_CHECKPOINT",
        "/home/wanyi/zy_test/tabpfn_weights/tabpfn-v3-regressor-v3_20260506_timeseries.ckpt",
    )
)


def _make_context(periods: int) -> pd.DataFrame:
    """Create a deterministic single-series forecasting context."""

    rng = np.random.default_rng(42)
    x = np.arange(periods)
    return pd.DataFrame(
        {
            "item_id": ["demo"] * periods,
            "timestamp": pd.date_range("2024-01-01", periods=periods, freq="h"),
            "target": np.sin(2 * np.pi * x / 12) + 0.02 * x + rng.normal(0, 0.02, periods),
        }
    )


def main() -> None:
    """Run a small TabPFN-TS prediction and write the output CSV."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--context-length", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("results/tabpfn_ts_smoke.csv"))
    args = parser.parse_args()

    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing TabPFN-TS checkpoint: {args.checkpoint}")

    import torch
    from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    context = _make_context(args.context_length)
    start = time.perf_counter()
    pipeline = TabPFNTSPipeline(
        max_context_length=args.context_length,
        tabpfn_mode=TabPFNMode.LOCAL,
        tabpfn_model_config={
            "model_path": str(args.checkpoint),
            "device": device,
            "n_estimators": 1,
        },
    )
    prediction = pipeline.predict_df(context, prediction_length=args.horizon, quantiles=[0.5])
    elapsed = time.perf_counter() - start

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction.reset_index().to_csv(args.output, index=False)
    print(f"device={device}")
    print(f"checkpoint={args.checkpoint}")
    print(f"rows={len(prediction)} columns={list(prediction.columns)} elapsed_s={elapsed:.3f}")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
