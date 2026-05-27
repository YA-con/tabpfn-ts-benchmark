#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
python -m experiments.e0_pilot_results
