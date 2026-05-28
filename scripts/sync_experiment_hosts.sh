#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHT="${TABPFN_TS_LOCAL_CHECKPOINT:-/Users/lioz/Desktop/组里/tabpfn_TS_26-5-26/tabpfn-v3-regressor-v3_20260506_timeseries.ckpt}"
LOCAL_HEAD="$(cd "${ROOT}" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
LOCAL_STATUS="$(cd "${ROOT}" && git status --short 2>/dev/null | wc -l | tr -d ' ')"

sync_host() {
  local host="$1"
  local project_dir="$2"
  local weight_dir="$3"

  echo "===== sync ${host} ====="
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${host}" "mkdir -p '${project_dir}' '${weight_dir}'"
  rsync -az \
    -e "ssh -o ServerAliveInterval=20 -o ServerAliveCountMax=6" \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    --exclude '.mypy_cache/' \
    --exclude '.DS_Store' \
    "${ROOT}/" "${host}:${project_dir}/"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${host}" "cat > '${project_dir}/.sync_state' <<'EOF'
synced_from_local_head=${LOCAL_HEAD}
local_dirty_files=${LOCAL_STATUS}
synced_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
EOF"
  if [ -f "${WEIGHT}" ]; then
    rsync -az --partial \
      -e "ssh -o ServerAliveInterval=20 -o ServerAliveCountMax=6" \
      "${WEIGHT}" "${host}:${weight_dir}/"
  else
    echo "checkpoint not found locally: ${WEIGHT}"
  fi
}

if [ "$#" -gt 0 ]; then
  TARGETS=("$@")
else
  TARGETS=("A00" "4080s")
fi

for target in "${TARGETS[@]}"; do
  case "${target}" in
    A00)
      sync_host "A00" "/home/wanyi/zy_test/tabpfn-ts-benchmark" "/home/wanyi/zy_test/tabpfn_weights" || echo "A00 sync failed"
      ;;
    4080s)
      sync_host "4080s" "/home/zhangyao/zy_test/tabpfn-ts-benchmark" "/home/zhangyao/zy_test/tabpfn_weights" || echo "4080s sync failed"
      ;;
    *)
      echo "unknown target: ${target}" >&2
      exit 2
      ;;
  esac
done
