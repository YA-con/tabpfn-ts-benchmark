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

A00 is currently configured with CUDA torch:

```text
torch==2.5.1+cu124
CUDA runtime 12.4
```

If the venv is rebuilt, install torch with:

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install --force-reinstall numpy==1.26.4 fsspec==2026.2.0
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
export CUDA_VISIBLE_DEVICES=2
python scripts/run_tabpfn_ts_smoke.py --device cuda --output results/tabpfn_ts_smoke_gpu.csv
```

This has been verified on A00 with the local time-series checkpoint and GPU 2.
The script writes `results/tabpfn_ts_smoke_gpu.csv`.
