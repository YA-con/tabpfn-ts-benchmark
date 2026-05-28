#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/wanyi/zy_test/tabpfn-ts-benchmark}"
PYTHON="${PYTHON:-/home/wanyi/zy_test/venv/bin/python}"
CHECKPOINT="${TABPFN_TS_CHECKPOINT:-/home/wanyi/zy_test/tabpfn_weights/tabpfn-v3-regressor-v3_20260506_timeseries.ckpt}"
RUN_GROUP="${RUN_GROUP:-full_benchmark_v1}"
DATASETS="${TABPFN_TS_DATASETS:-synthetic_energy,synthetic_traffic,synthetic_weather,synthetic_exchange,stock_provided}"
MAX_SERIES="${TABPFN_TS_MAX_SERIES:-2}"
CONTEXTS="${TABPFN_TS_CONTEXTS:-48 96 192}"
HORIZONS="${TABPFN_TS_HORIZONS:-12 24}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-20}"
MAX_GPU_MEM_MB="${MAX_GPU_MEM_MB:-10000}"
CHECK_INTERVAL_S="${CHECK_INTERVAL_S:-60}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

select_free_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, -v max_mem="${MAX_GPU_MEM_MB}" -v max_util="${MAX_GPU_UTIL}" '
      {
        idx=$1; mem=$2; util=$3;
        gsub(/ /, "", idx); gsub(/ /, "", mem); gsub(/ /, "", util);
        mem += 0; util += 0;
        if (mem <= max_mem && util <= max_util) {
          print idx, mem, util;
        }
      }
    ' | sort -k2,2n -k3,3n | head -n 1
}

wait_for_gpu() {
  local selected
  while true; do
    echo "[$(date '+%F %T')] GPU snapshot:" >&2
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits >&2
    selected="$(select_free_gpu || true)"
    if [[ -n "${selected}" ]]; then
      echo "[$(date '+%F %T')] Selected GPU: ${selected}" >&2
      echo "${selected}" | awk '{print $1}'
      return 0
    fi
    echo "No GPU under util<=${MAX_GPU_UTIL}% and mem<=${MAX_GPU_MEM_MB}MB. Sleep ${CHECK_INTERVAL_S}s." >&2
    sleep "${CHECK_INTERVAL_S}"
  done
}

for context in ${CONTEXTS}; do
  for horizon in ${HORIZONS}; do
    if (( context <= horizon * 2 )); then
      echo "Skip invalid context=${context}, horizon=${horizon}: context must be > 2*horizon for AR baselines."
      continue
    fi
    output_dir="results/${RUN_GROUP}_c${context}_h${horizon}_s${MAX_SERIES}"
    if [[ -f "${output_dir}/forecast_metrics.csv" && "${FORCE_RERUN:-0}" != "1" ]]; then
      echo "Skip existing ${output_dir}; set FORCE_RERUN=1 to rerun."
      continue
    fi
    gpu_id="$(wait_for_gpu)"
    log_file="${LOG_DIR}/${RUN_GROUP}_c${context}_h${horizon}_s${MAX_SERIES}.log"
    echo "[$(date '+%F %T')] Start context=${context} horizon=${horizon} max_series=${MAX_SERIES} on GPU ${gpu_id}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      export TABPFN_TS_CHECKPOINT="${CHECKPOINT}"
      export TABPFN_TS_CONTEXT="${context}"
      export TABPFN_TS_HORIZON="${horizon}"
      export TABPFN_TS_MAX_SERIES="${MAX_SERIES}"
      export TABPFN_TS_DATASETS="${DATASETS}"
      export TABPFN_TS_OUTPUT_DIR="${output_dir}"
      "${PYTHON}" -m experiments.e0_tabpfn_ts_pilot
      "${PYTHON}" scripts/generate_fancy_demo_report.py \
        --run-dir "${output_dir}" \
        --baseline-dir results/pilot_baseline \
        --config configs/config.yaml \
        --out-root "reports/${RUN_GROUP}_c${context}_h${horizon}_s${MAX_SERIES}"
    ) 2>&1 | tee "${log_file}"
    echo "[$(date '+%F %T')] Finished ${output_dir}. Log: ${log_file}"
  done
done
