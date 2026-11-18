# Inputs and setup

The archived experiments used Python 3.9 and PyTorch 2.3.0. `requirements.txt` records core package versions from the working research environment, not a fresh-install lock for every OS or CUDA version. Install the appropriate PyTorch build and the selected upstream project's dependencies in your own environment. Older Hydra/OmegaConf packaging may require pip <24.1. No third-party repositories or weights are vendored.

| Component | Source / revision | Environment variable |
|---|---|---|
| DeepECG-SSL toolkit | [DeepECG-SSL-finetune](https://github.com/HeartWise-AI/DeepECG-SSL-finetune), `0d37b4c0a90a2bdc8ef29b3d8f34808a0f296c6d` | `DEEPECG_FAIRSEQ` (directory containing `fairseq_signals`) |
| DeepECG SSL base model | Obtain the compatible pretrained checkpoint from its provider | `DEEPECG_CHECKPOINT` |
| Medical report encoder/tokenizer | [MedCPT-Query-Encoder](https://huggingface.co/ncbi/MedCPT-Query-Encoder), local Transformers directory | `MEDCPT_DIR` |
| Trained ECG–text model | Validation-selected pretraining checkpoint with a `model` state dict | `CARDIOLOOM_CHECKPOINT` |

Set `DATA_ROOT` to the input directory and optionally `CARDIOLOOM_WORK` to a new writable output root. Use absolute paths when setting these variables.

| Local input | Contract |
|---|---|
| `ecg_data_all.h5` | `ECGs` array `[N,8,5000]`, 10 s at 500 Hz, mV, leads I, II, V1, V2, V3, V4, V5, V6. CardioLoom reconstructs the four dependent leads and resamples to 250 Hz internally. Do not apply the conversion twice. |
| `ecg_labels_all.csv` | One row per HDF5 row in exactly the same order. `Filename` is a local record key. Synthetic or opaque keys suffice. |
| `all_information.csv` | `Filename,english_report`, one mapped English report per pretraining record. Prepare the English reports locally before training. |
| `all_information_v8.csv` | Unique `Filename`, binary task columns with IDs matching the configuration and `-1` for missing labels. Preserve training-time column order for multitask heads. |
| `new_split_v2/*.csv` | Zero-based HDF5 row numbers under `index`. Provide `pretrain_{train,valid}_indices.csv` and `downstream_{train,valid,test}_indices.csv`. Assign patients to partitions first, then expand to ECG rows. |

Filtering by task eligibility can change denominators. Multi-label training uses the eligible rows for that head; valve labels and other ECHO heads should not be assumed to share one denominator. The clinical taxonomy is a reporting order, not the model's output order. In particular the archived common ECG head also contains an output outside the 45 routine diagnoses. Use task IDs to select the reported targets.

The journal protocol records 20 pretraining epochs, 2 A100-80GB GPUs, global batch 1,400 and seed 42. The study selected epoch 16 for CardioLoom. Repeating on different data requires selection on your own validation set. Removing `--dry-run` launches real training.

## Sparse endpoints

Extract the frozen features once, then fit one linear SVM per endpoint. For example, cardiac amyloidosis (CA) uses `ecg_label_74`:

```bash
bash scripts/run.bash features
bash scripts/run.bash svm ecg_label_74
```

With the default output root, feature extraction writes `train_features.h5`, `valid_features.h5` and `test_features.h5` under `work/outputs/features/cardioloom/`. The SVM command reads those files and saves the selected CA classifier at `work/outputs/svm/cardioloom_ecg_label_74/best-checkpoint.pkl`. Pass this file to `predict --svm` to obtain one decision margin per ECG.

The sparse tasks are `PR`, `PS`, `TS`, `echo_label_{8,9,14}` and `ecg_label_{63,65,71,73,74}`. Pass one literal task ID at a time. The [task dictionary](../configs/tasks.csv) maps these IDs to diagnoses. The SVM returns a signed decision margin.
