#!/usr/bin/env python3
"""Extract split-aligned offline features without saving patient identifiers."""

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
    add_model_args,
    assert_new_output,
    dry_run_report,
    resolve_model_paths,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    add_model_args(result)
    add_dataset_args(result)
    result.add_argument("--exp-name", required=True)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--num-workers", type=int, default=4)
    result.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    result.add_argument(
        "--feature-selection",
        default="all_split",
        choices=("all_split", "task_valid"),
        help="all_split extracts every original split row once; task_valid preserves task-specific filtering.",
    )
    result.add_argument("--dry-run", action="store_true")
    return result


def extract_split(
    dataset,
    adapter,
    *,
    output_file: Path,
    batch_size: int,
    num_workers: int,
    device,
    model_name: str,
    split_name: str,
    downstream_task: str,
):
    import h5py
    import torch
    from torch.utils.data import DataLoader

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": False,
        "drop_last": False,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_options.update(persistent_workers=True, prefetch_factor=4)
    loader = DataLoader(dataset, **loader_options)
    with h5py.File(output_file, "x") as handle:
        features_ds = handle.create_dataset(
            "features", shape=(0, adapter.feature_dim), maxshape=(None, adapter.feature_dim), dtype="f4"
        )
        indices_ds = handle.create_dataset("indices", shape=(0,), maxshape=(None,), dtype="i8")
        handle.attrs["feature_dim"] = adapter.feature_dim
        handle.attrs["contains_patient_identifiers"] = False
        handle.attrs["model_name"] = model_name
        handle.attrs["checkpoint"] = adapter.checkpoint_path
        handle.attrs["split"] = split_name
        handle.attrs["downstream_task"] = downstream_task
        handle.attrs["source_tensor_shape"] = "[B,8,5000]"
        handle.attrs["source_sampling_rate_hz"] = adapter.source_sampling_rate
        handle.attrs["source_lead_order"] = ",".join(adapter.source_lead_order)
        handle.attrs["model_sampling_rate_hz"] = adapter.model_sampling_rate
        handle.attrs["model_samples"] = adapter.model_samples
        handle.attrs["preprocessing"] = adapter.preprocessing_summary
        row_count = 0
        with torch.inference_mode():
            for batch in loader:
                ecg = batch["ecg_data"].to(device, non_blocking=device.type == "cuda")
                features = adapter.forward_features(ecg).detach().cpu().float().numpy()
                indices = batch["data_idx"].detach().cpu().numpy()
                next_count = row_count + features.shape[0]
                features_ds.resize((next_count, adapter.feature_dim))
                indices_ds.resize((next_count,))
                features_ds[row_count:next_count] = features
                indices_ds[row_count:next_count] = indices
                row_count = next_count


def main() -> None:
    args = parser().parse_args()
    resolve_model_paths(args)
    output_dir = CODEX_ROOT / "outputs/features" / args.exp_name
    output_files = {name: output_dir / f"{name}_features.h5" for name in ("train", "valid", "test")}
    if args.dry_run:
        dry_run_report(args, output_paths=output_files)
        return
    output_dir = assert_new_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    import torch
    from cardioloom.scripts.common import build_adapter

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    adapter = build_adapter(args).to(device).eval()

    root = args.data_root.resolve()
    if args.feature_selection == "all_split":
        import h5py
        import pandas as pd
        from torch.utils.data import Dataset

        class SplitFeatureDataset(Dataset):
            def __init__(self, h5_path: Path, split_path: Path) -> None:
                self.h5_path = str(h5_path)
                self._h5_handle = None
                self.h5_data = None
                self.indices = pd.read_csv(split_path, usecols=["index"])["index"].tolist()

            def _waveforms(self):
                # Open independently inside each DataLoader worker. Sharing an
                # h5py handle created before fork is not reliably safe.
                if self.h5_data is None:
                    self._h5_handle = h5py.File(self.h5_path, "r")
                    self.h5_data = self._h5_handle["ECGs"]
                return self.h5_data

            def __len__(self) -> int:
                return len(self.indices)

            def __getitem__(self, position: int):
                data_idx = int(self.indices[position])
                return {
                    "ecg_data": torch.as_tensor(self._waveforms()[data_idx]).float(),
                    "data_idx": data_idx,
                }

            def __del__(self):
                handle = getattr(self, "_h5_handle", None)
                if handle is not None:
                    handle.close()

        datasets = tuple(
            SplitFeatureDataset(root / args.h5_file_path, root / split_file)
            for split_file in (
                args.train_indices_file_path,
                args.valid_indices_file_path,
                args.test_indices_file_path,
            )
        )
    else:
        sys.path.insert(0, str(PUBLIC_SOURCE))
        from datasets.dataset_downstream import ECGDatasetDownstream

        datasets = ECGDatasetDownstream.create_datasets(
            h5_file=str(root / args.h5_file_path),
            label_file=str(root / args.label_file_path),
            info_file=str(root / args.info_file_path),
            train_index_file=str(root / args.train_indices_file_path),
            valid_index_file=str(root / args.valid_indices_file_path),
            test_index_file=str(root / args.test_indices_file_path),
            task=args.downstream_task,
            ecg_or_echo=args.ecg_or_echo,
            num_leads=8,
            downsample=False,
            exclude_label_ids=args.exclude_label_ids,
            random_split_seed=args.random_split_seed,
            sample_ratio=args.sample_ratio,
        )
    for split_name, dataset in zip(("train", "valid", "test"), datasets):
        extract_split(
            dataset,
            adapter,
            output_file=output_files[split_name],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            model_name=args.model,
            split_name=split_name,
            downstream_task=args.downstream_task,
        )


if __name__ == "__main__":
    main()
