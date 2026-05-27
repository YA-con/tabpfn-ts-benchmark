#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python scripts/materialize_dataset.py stock_provided --max-files "${1:-5}"
