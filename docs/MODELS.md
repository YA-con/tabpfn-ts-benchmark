# Model Setup Notes

## Execution policy

Run experiments on the least busy available server among `A00` and `4080s`.
Before launching any GPU run, check both machines and pick the one with lower
GPU utilization, enough free memory, and no obvious conflict with other users:

```bash
scripts/check_experiment_hosts.sh
```

Keep project state aligned on both machines before and after substantial runs:

```bash
scripts/sync_experiment_hosts.sh
```

To sync only one side:

```bash
scripts/sync_experiment_hosts.sh A00
scripts/sync_experiment_hosts.sh 4080s
```

`4080s` currently resolves to:

```text
zhangyao@10.46.18.55
project: /home/zhangyao/zy_test/tabpfn-ts-benchmark
checkpoint: /home/zhangyao/zy_test/tabpfn_weights/tabpfn-v3-regressor-v3_20260506_timeseries.ckpt
```

`A00` currently uses:

```text
wanyi@114.214.255.146:23333
project: /home/wanyi/zy_test/tabpfn-ts-benchmark
checkpoint: /home/wanyi/zy_test/tabpfn_weights/tabpfn-v3-regressor-v3_20260506_timeseries.ckpt
```

Do not launch experiments if all candidate GPUs are actively occupied. In that
case, sync files only and wait for a clearer GPU.

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

## TabPFN-TS pilot benchmark

The first controlled benchmark with TabPFN-TS is intentionally small:

```bash
cd /home/wanyi/zy_test/tabpfn-ts-benchmark
export PATH=/home/wanyi/zy_test/venv/bin:$PATH
export TABPFN_DISABLE_TELEMETRY=1
export CUDA_VISIBLE_DEVICES=2
python -B -m experiments.e0_tabpfn_ts_pilot
```

Default limits:

```text
datasets: synthetic_energy, stock_provided
series per dataset: 1
context length: 48
horizon: 6
checkpoint: /home/wanyi/zy_test/tabpfn_weights/tabpfn-v3-regressor-v3_20260506_timeseries.ckpt
```

Outputs:

```text
results/tabpfn_ts_pilot/forecast_metrics.csv
results/tabpfn_ts_pilot/forecast_predictions.csv
results/tabpfn_ts_pilot/report.html
```

This pilot verifies the full path from local checkpoint to GPU inference and
HTML reporting before scaling TabPFN-TS to longer contexts and more series.

The current recommended expanded pilot uses five series when available:

```bash
export TABPFN_TS_MAX_SERIES=5
export TABPFN_TS_OUTPUT_DIR=results/tabpfn_ts_pilot_s5
python -B -m experiments.e0_tabpfn_ts_pilot
```

In this run TabPFN-TS ranked first on both `synthetic_energy` and
`stock_provided` under the controlled short-context setup.
