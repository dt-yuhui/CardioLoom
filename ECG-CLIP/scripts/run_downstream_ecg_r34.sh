#!/usr/bin/env bash
set -euo pipefail

# ECGCLIP-R34 fine-tuning for the ECG multi-label task at 100%, 10%, and 1%
# of the downstream training set. Override paths through environment variables.

DEVICE_ID="${DEVICE_ID:-0}"
DATA_ROOT="${DATA_ROOT:-data}"
PRETRAIN_OUTPUT_ROOT="${PRETRAIN_OUTPUT_ROOT:-outputs/pretrain}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/downstream}"
PRETRAINED_EXP_NAME="${PRETRAINED_EXP_NAME:-ecg_clip_r34}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"
TEST_ONLY="${TEST_ONLY:-False}"

export CUDA_VISIBLE_DEVICES="${DEVICE_ID}"

percentages=(100 10 1)
sample_ratios=(1.0 0.1 0.01)
train_indices_files=(
  "new_split_v2/downstream_train_indices.csv"
  "new_split_v2/downstream_train_indices_10_percent.csv"
  "new_split_v2/downstream_train_indices_1_percent.csv"
)

pretrained_args=()
if [[ -n "${PRETRAINED_CHECKPOINT}" ]]; then
  pretrained_args+=(--pretrained_checkpoint "${PRETRAINED_CHECKPOINT}")
else
  pretrained_args+=(
    --custom_pretrained_root "${PRETRAIN_OUTPUT_ROOT}"
    --pretrained_exp_name "${PRETRAINED_EXP_NAME}"
    --pretrained_metric acc
  )
fi

for index in "${!percentages[@]}"; do
  percentage="${percentages[$index]}"
  ratio="${sample_ratios[$index]}"
  train_file="${train_indices_files[$index]}"
  exp_name="clip-acc_resnet34_ecg_multi_label_sample-ratio-${percentage}%"

  python main_downstream.py \
    --finetune_type finetune \
    --ecg_model resnet34 \
    --sample_ratio "${ratio}" \
    --exp_name "${exp_name}" \
    --downstream_task ecg_multi_label \
    --num_classes 2 \
    --lr 5e-4 \
    --num_leads 8 \
    --test_only "${TEST_ONLY}" \
    --data_root "${DATA_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --train_indices_file_path "${train_file}" \
    "${pretrained_args[@]}"
done

