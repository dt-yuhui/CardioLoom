#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 exec "${PYTHON:-python}" -m cardioloom.scripts.smoke
