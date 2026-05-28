#!/usr/bin/env bash
set -euo pipefail

HOSTS=("A00" "4080s")

for host in "${HOSTS[@]}"; do
  echo "===== ${host} ====="
  if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "${host}" '
    set -e
    echo "[host]"
    hostname
    date
    echo
    echo "[gpu]"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
      echo
      nvidia-smi
    else
      echo "nvidia-smi not found"
    fi
    echo
    echo "[project]"
    project=""
    for candidate in "$HOME/zy_test/tabpfn-ts-benchmark" "/home/wanyi/zy_test/tabpfn-ts-benchmark" "/home/zhangyao/zy_test/tabpfn-ts-benchmark"; do
      if [ -d "$candidate" ]; then
        project="$candidate"
        break
      fi
    done
    if [ -n "$project" ]; then
      cd "$project"
      pwd
      if [ -f .sync_state ]; then
        cat .sync_state
      fi
      if command -v git >/dev/null 2>&1 && [ -d .git ]; then
        git rev-parse --short HEAD
        git status --short --branch
      else
        echo "git metadata unavailable on this host"
      fi
    else
      echo "project missing"
    fi
  '; then
    echo "${host}: unreachable or command failed"
  fi
  echo
done
