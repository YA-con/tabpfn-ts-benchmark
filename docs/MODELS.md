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

Local TabPFN-TS inference is currently blocked by the gated Hugging Face model
`Prior-Labs/tabpfn_3`. The smoke test reaches model loading, then requires:

```text
HF_TOKEN with access to Prior-Labs/tabpfn_3
```

or a prior command-line login:

```bash
hf auth login
```

A00 also showed temporary DNS failures when checking the gated model license, so
the next TabPFN-TS run should first verify both network access and authentication.
