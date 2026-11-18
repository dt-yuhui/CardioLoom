"""Worker-safe, strictly read-only ECGCLIP pretraining dataset."""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cardioloom.models.ecg_preprocessing import reconstruct_standard_12_leads
from cardioloom.report_templates import (
    normalize_report_template,
    stable_report_template_id,
)


class ReadOnlyECGPretrainDataset(Dataset):
    """Preserve the original dataset contract with one HDF5 handle per worker."""

    def __init__(
        self,
        h5_file: str | Path,
        filenames: list[str],
        reports: list[str],
        indices: list[int],
        num_leads: int,
        include_report_template_ids: bool = False,
    ) -> None:
        if int(num_leads) not in (8, 12):
            raise ValueError("Transformer pretraining num_leads must be 8 or 12")
        self.h5_file = str(Path(h5_file).resolve())
        if not (len(filenames) == len(reports) == len(indices)):
            raise ValueError("filenames, reports and indices must have equal lengths")
        self.indices = [int(index) for index in indices]
        self.filenames = list(filenames)
        self.reports = [str(report) for report in reports]
        self.num_leads = int(num_leads)
        self.include_report_template_ids = bool(include_report_template_ids)
        self.report_template_ids: torch.Tensor | None = None
        self.report_template_frequencies: torch.Tensor | None = None
        self.report_template_counts: Counter[int] | None = None
        if self.include_report_template_ids:
            template_ids: list[int] = []
            collision_guard: dict[int, str] = {}
            for report in self.reports:
                normalized = normalize_report_template(report)
                if not normalized:
                    raise ValueError("A pretraining report is empty after normalization")
                template_id = stable_report_template_id(normalized)
                previous = collision_guard.setdefault(template_id, normalized)
                if previous != normalized:
                    raise RuntimeError("A report-template ID collision was detected")
                template_ids.append(template_id)
            self.report_template_ids = torch.tensor(template_ids, dtype=torch.long)
            self.report_template_counts = Counter(template_ids)
            self.report_template_frequencies = torch.tensor(
                [self.report_template_counts[value] for value in template_ids],
                dtype=torch.double,
            )
        self._h5_handle: h5py.File | None = None
        self._h5_data: Any | None = None
        self._owner_pid: int | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5_handle"] = None
        state["_h5_data"] = None
        state["_owner_pid"] = None
        return state

    def _close_h5(self) -> None:
        if self._h5_handle is not None:
            self._h5_handle.close()
        self._h5_handle = None
        self._h5_data = None
        self._owner_pid = None

    def _ensure_h5(self) -> Any:
        pid = os.getpid()
        if self._h5_handle is None or self._owner_pid != pid:
            self._close_h5()
            self._h5_handle = h5py.File(self.h5_file, "r")
            if "ECGs" not in self._h5_handle:
                self._close_h5()
                raise KeyError(f"HDF5 file has no 'ECGs' dataset: {self.h5_file}")
            self._h5_data = self._h5_handle["ECGs"]
            self._owner_pid = pid
        return self._h5_data

    def __getitem__(self, idx: int) -> dict[str, Any]:
        data_idx = self.indices[idx]
        ecg_data = self._ensure_h5()[data_idx]
        if ecg_data.shape != (8, 5000):
            raise RuntimeError(
                f"Expected stored ECG shape (8,5000), got {tuple(ecg_data.shape)}"
            )
        ecg_tensor = torch.as_tensor(ecg_data, dtype=torch.float32)
        if self.num_leads == 12:
            # Stored ECGCLIP data contain the eight independent leads in
            # [I,II,V1,V2,V3,V4,V5,V6] order. The four dependent limb leads
            # are reconstructed deterministically; no data file is changed.
            ecg_tensor = reconstruct_standard_12_leads(ecg_tensor.unsqueeze(0)).squeeze(0)
        expected = (self.num_leads, 5000)
        if tuple(ecg_tensor.shape) != expected:
            raise RuntimeError(
                f"Formatted ECG shape {tuple(ecg_tensor.shape)} != {expected}"
            )
        result = {
            "ecg_data": ecg_tensor,
            "filename": self.filenames[idx],
            "report": self.reports[idx],
        }
        if self.report_template_ids is not None:
            result["report_template_id"] = self.report_template_ids[idx]
        return result

    def get_report_sampling_weights(self, alpha: float) -> torch.Tensor:
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError(f"report sampling alpha must be in [0,1], got {alpha}")
        if (
            self.report_template_ids is None
            or self.report_template_counts is None
            or self.report_template_frequencies is None
        ):
            raise RuntimeError("Report template IDs were not enabled for this dataset")
        weights = self.report_template_frequencies.pow(-float(alpha))
        return weights / weights.mean()

    def report_sampling_summary(self, alpha: float) -> dict[str, float | int]:
        if self.report_template_counts is None:
            raise RuntimeError("Report template IDs were not enabled for this dataset")
        counts = torch.tensor(
            list(self.report_template_counts.values()), dtype=torch.double
        )
        masses = counts.pow(1.0 - float(alpha))
        probabilities = masses / masses.sum()
        return {
            "row_count": len(self),
            "unique_template_count": len(self.report_template_counts),
            "alpha": float(alpha),
            "top_template_probability": float(probabilities.max()),
            "same_template_collision_probability": float(probabilities.square().sum()),
        }

    def __del__(self) -> None:
        try:
            self._close_h5()
        except Exception:
            pass

    @classmethod
    def create_datasets(
        cls,
        h5_file: str | Path,
        label_file: str | Path,
        info_file: str | Path,
        train_index_file: str | Path,
        valid_index_file: str | Path,
        num_leads: int,
        include_report_template_ids: bool = False,
    ) -> tuple["ReadOnlyECGPretrainDataset", "ReadOnlyECGPretrainDataset"]:
        label_df = pd.read_csv(label_file, usecols=["Filename"], low_memory=False)
        info_columns = ["Filename", "english_report"]
        info_df = pd.read_csv(
            info_file,
            usecols=info_columns,
            low_memory=False,
        )
        train_indices = pd.read_csv(train_index_file)["index"].tolist()
        valid_indices = pd.read_csv(valid_index_file)["index"].tolist()
        info_filename_to_report = dict(
            zip(info_df["Filename"], info_df["english_report"])
        )
        reports = label_df["Filename"].map(info_filename_to_report)
        missing = int(reports.isna().sum())
        if missing:
            raise RuntimeError(f"Missing report mappings for {missing} ECG rows")

        def select(rows):
            return cls(h5_file, label_df.loc[rows, 'Filename'].astype(str).tolist(),
                       reports.loc[rows].astype(str).tolist(), rows, num_leads,
                       include_report_template_ids=include_report_template_ids)
        return select(train_indices), select(valid_indices)
