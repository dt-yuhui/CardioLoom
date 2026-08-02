#!/usr/bin/env bash
set -euo pipefail

# All values can be overridden without editing this script. Example for an
# offline server:
#   TEXT_MODEL=/path/to/MedCPT-Query-Encoder \
#   TEXT_MODEL_LOCAL_FILES_ONLY=True bash scripts/run_clip.sh

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pretrain}"
TEXT_MODEL="${TEXT_MODEL:-ncbi/MedCPT-Query-Encoder}"
TEXT_MODEL_LOCAL_FILES_ONLY="${TEXT_MODEL_LOCAL_FILES_ONLY:-False}"
ECG_MODEL="${ECG_MODEL:-resnet18}"

case "${ECG_MODEL}" in
  resnet18)
    default_batch_size=768
    default_exp_name=ecg_clip_r18
    ;;
  resnet34)
    default_batch_size=700
    default_exp_name=ecg_clip_r34
    ;;
  resnet50)
    default_batch_size=512
    default_exp_name=ecg_clip_r50
    ;;
  *)
    echo "Unsupported public pretraining architecture: ${ECG_MODEL}" >&2
    exit 2
    ;;
esac

TRAIN_BATCH_SIZE_PER_GPU="${TRAIN_BATCH_SIZE_PER_GPU:-${default_batch_size}}"
EXP_NAME="${EXP_NAME:-${default_exp_name}}"

python -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE}" main.py \
    --epochs 20 \
    --train_batch_size_per_gpu "${TRAIN_BATCH_SIZE_PER_GPU}" \
    --use_fp16 True \
    --exp_name "${EXP_NAME}" \
    --ecg_model "${ECG_MODEL}" \
    --data_root "${DATA_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --text_model "${TEXT_MODEL}" \
    --text_model_local_files_only "${TEXT_MODEL_LOCAL_FILES_ONLY}"
