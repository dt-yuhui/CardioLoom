#!/usr/bin/env python3
"""Run the unchanged ECGCLIP SVM score/threshold/metric logic on offline features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cardioloom.scripts.common import (
    CODEX_ROOT,
    PUBLIC_SOURCE,
    add_dataset_args,
    assert_under_codex,
    assert_new_output,
    dry_run_report,
    original_downstream_namespace,
    prepare_public_source_imports,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    add_dataset_args(result)
    result.add_argument("--exp-name", required=True)
    result.add_argument("--train-offline-features-file", required=True, type=Path)
    result.add_argument("--valid-offline-features-file", required=True, type=Path)
    result.add_argument("--test-offline-features-file", required=True, type=Path)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--num-workers", type=int, default=4)
    result.add_argument("--total-iterations", type=int, default=1000)
    result.add_argument("--val-iter", type=int, default=200)
    result.add_argument("--test-only", action="store_true")
    result.add_argument(
        "--svm-protocol",
        choices=("rawscore_v2",),
        default="rawscore_v2",
        help="Opt-in corrected raw decision-score protocol; historical remains reproducible.",
    )
    result.add_argument("--dry-run", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    args.finetune_type = "svm"
    for field in (
        "train_offline_features_file", "valid_offline_features_file", "test_offline_features_file"
    ):
        setattr(args, field, str(assert_under_codex(getattr(args, field))))
    output_dir = CODEX_ROOT / "outputs/svm" / args.exp_name
    if args.dry_run:
        dry_run_report(args, output_paths={"experiment": output_dir})
        print("offline_feature_files=")
        for field in (
            "train_offline_features_file", "valid_offline_features_file", "test_offline_features_file"
        ):
            path = Path(getattr(args, field))
            print(f"  {field}: path={path} exists={path.is_file()}")
        print(f"svm_protocol={args.svm_protocol}")
        print("raw_ecg_opened=false")
        return
    output_dir = assert_new_output(output_dir)

    prepare_public_source_imports()
    import main_downstream_svm as original

    if args.svm_protocol == "rawscore_v2":
        from cardioloom.models.corrected_svm import (
            build_corrected_classifier,
            get_rawscore_metrics,
            identity_decision_scores,
        )

        original.logits_to_probability = identity_decision_scores
        original.get_metrics = get_rawscore_metrics

        def corrected_classifier_factory(*positional, **keywords):
            return build_corrected_classifier(
                *positional, random_state=args.seed, **keywords
            )

        original.SGDClassifier = corrected_classifier_factory

    class FeatureOnlyInternalDataset(original.ECGDatasetDownstream):
        """Serve aligned offline features without reading the raw ECG array."""

        def __init__(
            self,
            h5_file,
            label_df,
            info_df,
            indices,
            task,
            ecg_or_echo="ecg",
            num_leads=8,
            downsample=False,
            exclude_label_ids=None,
            offline_features_file=None,
        ):
            import h5py
            import numpy as np

            del h5_file, num_leads, downsample
            if offline_features_file is None:
                raise ValueError("Feature-only SVM requires an offline feature file")
            self.task = task
            self.ecg_or_echo = ecg_or_echo
            self.exclude_label_ids = tuple(exclude_label_ids or ())
            prefix = "ecg_label" if ecg_or_echo == "ecg" else "echo_label"
            self.exclude_columns = [
                f"{prefix}_{label_id}" for label_id in self.exclude_label_ids
            ]
            self.offline_features_file = h5py.File(offline_features_file, "r")
            self.offline_features = self.offline_features_file["features"]
            feature_indices = self.offline_features_file["indices"][:]
            self.offline_index_lookup = {
                int(data_idx): position
                for position, data_idx in enumerate(feature_indices)
            }

            self.label_columns = self.get_task_label_columns(
                task, info_df.columns, self.exclude_columns
            )
            task_labels = self.get_task_labels(info_df, self.label_columns, task)
            if "multi_label" in task:
                for column in self.label_columns:
                    label_df[column] = label_df["Filename"].map(task_labels.get).apply(
                        lambda value: value.get(column, -1) if value else -1
                    )
                    label_df[column] = label_df[column].replace(np.nan, -1)
            else:
                label_df["label"] = label_df["Filename"].map(task_labels).fillna(-1)
            self.labels_df = label_df
            valid_indices = self.get_valid_indices(label_df, task)
            self.indices = list(set(indices).intersection(valid_indices))
            if task not in ["age", "ecg_multi_label", "echo_multi_label", "valve_multi_label"]:
                self.setup_class_weights_and_sample_weights(label_df)
            else:
                self.class_weights = self.sample_weights = None

        def __del__(self):
            handle = getattr(self, "offline_features_file", None)
            if handle is not None:
                handle.close()

        def __getitem__(self, position):
            import numpy as np
            import torch

            if self.offline_features is None:
                raise RuntimeError("Rare-disease SVM requires offline features")
            data_idx = self.indices[position]
            feature_position = self.offline_index_lookup.get(data_idx)
            if feature_position is None:
                raise RuntimeError(
                    "Offline feature missing for a selected row; identifier suppressed"
                )
            feature = np.asarray(self.offline_features[feature_position])
            if feature.ndim != 1:
                raise RuntimeError(f"Expected a vector feature, got {feature.shape}")

            row = self.labels_df.loc[data_idx]
            if "multi_label" in self.task:
                labels = torch.tensor(
                    np.asarray(row[self.label_columns].values).astype(float)
                ).long()
            else:
                label = row["label"]
                if self.task == "age":
                    label = max(0, min(100, label)) / 100.0
                labels = [torch.tensor(label)]

            return {
                "ecg_data": torch.empty(0, dtype=torch.float32),
                "filename": "identifier_suppressed",
                "data_idx": data_idx,
                "label": {
                    task_id: label.item()
                    for task_id, label in zip(self.label_columns, labels)
                },
                "ecg_feature": torch.as_tensor(feature).float(),
            }

    def create_feature_only_datasets(*positional, **keywords):
        import pandas as pd
        from sklearn.model_selection import train_test_split

        if positional:
            raise TypeError("Feature-only dataset factory requires keyword arguments")
        task = keywords["task"]
        if "multi_label" in task or task == "age":
            raise ValueError("Corrected rare-disease SVM expects one binary task")
        if float(keywords.get("sample_ratio", 1.0)) != 1.0:
            raise ValueError("Feature-only SVM requires the explicit 100% split")

        # The original factory reads every column of the large information
        # table for every SVM task.  Only Filename and the selected target are
        # required for a single-task offline-feature run.
        label_df = pd.read_csv(
            keywords["label_file"], usecols=["Filename"], low_memory=False
        )
        info_df = pd.read_csv(
            keywords["info_file"], usecols=["Filename", task], low_memory=False
        )
        train_indices = pd.read_csv(
            keywords["train_index_file"], usecols=["index"]
        )["index"].tolist()
        valid_indices = pd.read_csv(
            keywords["valid_index_file"], usecols=["index"]
        )["index"].tolist()
        test_indices = pd.read_csv(
            keywords["test_index_file"], usecols=["index"]
        )["index"].tolist()
        random_seed = keywords.get("random_split_seed")
        if random_seed is not None:
            train_indices, valid_indices = train_test_split(
                train_indices + valid_indices,
                test_size=2 / 9,
                random_state=random_seed,
            )

        common = {
            "h5_file": None,
            "label_df": label_df,
            "info_df": info_df,
            "task": task,
            "ecg_or_echo": keywords.get("ecg_or_echo", "ecg"),
            "exclude_label_ids": keywords.get("exclude_label_ids"),
        }
        datasets = tuple(
            FeatureOnlyInternalDataset(
                indices=indices,
                offline_features_file=feature_file,
                **common,
            )
            for indices, feature_file in (
                (train_indices, keywords["train_offline_features_file"]),
                (valid_indices, keywords["valid_offline_features_file"]),
                (test_indices, keywords["test_offline_features_file"]),
            )
        )
        print(f"train size: {len(datasets[0])}")
        print(f"val size: {len(datasets[1])}")
        print(f"test size: {len(datasets[2])}")
        return datasets

    original.ECGDatasetDownstream.create_datasets = staticmethod(
        create_feature_only_datasets
    )

    original_data_loader = original.DataLoader

    def cpu_feature_data_loader(*positional, **keywords):
        # SVM consumes CPU numpy features; CUDA-pinned host buffers add an
        # unnecessary copy thread and provide no transfer benefit.
        keywords["pin_memory"] = False
        return original_data_loader(*positional, **keywords)

    original.DataLoader = cpu_feature_data_loader

    original_inference = original.run_inference_and_save_logits

    def no_identifier_csv(*positional, **keywords):
        keywords["save_flag"] = False
        return original_inference(*positional, **keywords)

    original.run_inference_and_save_logits = no_identifier_csv
    runtime_args = original_downstream_namespace(args, output_dir)
    runtime_args.svm_protocol = args.svm_protocol
    original.main(runtime_args)


if __name__ == "__main__":
    main()
