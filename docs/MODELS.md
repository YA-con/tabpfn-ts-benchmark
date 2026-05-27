# Model Setup Notes

## A00 execution

Run experiments on `A00` under:

```bash
cd /home/wanyi/zy_test/tabpfn-ts-benchmark
export PATH=/home/wanyi/zy_test/venv/bin:$PATH
export WANDB_MODE=offline
```

Always check GPU state before launching a model run:

```bash
nvidia-smi
```

CPU-only benchmark runs should set:

```bash
export CUDA_VISIBLE_DEVICES=""
```

## Current runnable models

The pilot benchmark currently runs these models end to end:

```text
Historical Mean
Seasonal Naive
Moving Average
Linear Trend
Ridge AR
Gradient Boosting AR
LightGBM AR
XGBoost AR
```

LightGBM and XGBoost use the same autoregressive feature window as the sklearn
models, so the comparison shares the same chronological split and metrics.

## TabPFN-TS status

`tabpfn-time-series==1.1.0` installs on A00 and imports successfully.

The time-series checkpoint should be:

```text
tabpfn-v3-regressor-v3_20260506_timeseries.ckpt
```

On A00 it is stored outside the git checkout:

```text
/home/wanyi/zy_test/tabpfn_weights/tabpfn-v3-regressor-v3_20260506_timeseries.ckpt
```

Do not commit checkpoint files to git.

Run the offline smoke test:

```bash
cd /home/wanyi/zy_test/tabpfn-ts-benchmark
export PATH=/home/wanyi/zy_test/venv/bin:$PATH
export TABPFN_DISABLE_TELEMETRY=1
export CUDA_VISIBLE_DEVICES=""
python scripts/run_tabpfn_ts_smoke.py --device cpu
```

This has been verified with CPU torch on A00. For GPU inference, the environment
needs a CUDA-enabled torch build; the current A00 venv reports `torch==2.5.1+cpu`.
