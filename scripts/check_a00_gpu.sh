#!/usr/bin/env bash
set -euo pipefail

ssh A00 'hostname; date; nvidia-smi'
