#!/usr/bin/env bash
set -euo pipefail

# Linear SVM evaluation for the five rare-disease tasks used in the paper.
# This entry point consumes pre-extracted ECGCLIP-R34 feature files.

DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/downstream}"
FEATURE_ROOT="${FEATURE_ROOT:-ecg_features_v3/zs/ecg_clip_r34}"
TEST_ONLY="${TEST_ONLY:-False}"
TOTAL_ITERATIONS="${TOTAL_ITERATIONS:-1000}"

rare_disease_ids=(63 65 71 73 74)

for label_id in "${rare_disease_ids[@]}"; do
  downstream_task="ecg_label_${label_id}"
  exp_name="clip-acc_resnet34_${downstream_task}_sample-ratio-100%"

  python main_downstream_svm.py \
    --finetune_type svm \
    --ecg_model resnet34 \
    --sample_ratio 1.0 \
    --exp_name "${exp_name}" \
    --downstream_task "${downstream_task}" \
    --num_classes 2 \
    --data_root "${DATA_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --train_offline_features_file "${FEATURE_ROOT}/train_features.h5" \
    --valid_offline_features_file "${FEATURE_ROOT}/valid_features.h5" \
    --test_offline_features_file "${FEATURE_ROOT}/test_features.h5" \
    --total_iterations "${TOTAL_ITERATIONS}" \
    --test_only "${TEST_ONLY}"
done
