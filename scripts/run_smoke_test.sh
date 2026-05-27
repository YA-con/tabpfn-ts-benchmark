#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
WANDB_MODE=offline python -m experiments.e1_zeroshot
