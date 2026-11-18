"""Read-only, semantics-preserving external ECG I/O helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


def read_shiyuan_vectorized(
    ecg_path: str | Path,
    *,
    bandpass_filtering: Callable,
    notch_filtering: Callable,
) -> np.ndarray:
    """Vectorized equivalent of the public Shiyuan binary decoder."""
    path = Path(ecg_path)
    file_size = path.stat().st_size
    waveform_base = file_size - 480000
    if waveform_base < 0:
        raise RuntimeError("Shiyuan binary file is shorter than its expected payload")

    leads = []
    with path.open("rb") as handle:
        for lead_index in range(8):
            offset = waveform_base + 50 * 2 * 500 + lead_index * 60000
            handle.seek(offset)
            raw = handle.read(5000 * 2)
            if len(raw) != 5000 * 2:
                raise RuntimeError("Shiyuan binary lead payload is incomplete")
            # ``struct.unpack('h')`` in the public implementation uses native
            # signed-int16 byte order. The current x86 containers are little-endian.
            lead = np.frombuffer(raw, dtype=np.dtype("=i2"), count=5000)
            leads.append(lead)

    ecg = np.vstack(leads) * 0.00244
    ecg = bandpass_filtering(ecg.T, 500, 0.5, 100)
    ecg = notch_filtering(ecg, 500, 50)
    ecg = np.asarray(ecg).T
    if ecg.shape != (8, 5000) or not np.isfinite(ecg).all():
        raise RuntimeError("Vectorized Shiyuan preprocessing returned an invalid tensor")
    return ecg


def read_wfdb_selected(
    wfdb_module,
    record_path: str | Path,
    *,
    channels: Sequence[int],
    target_samples: int = 5000,
    pad_short: bool = False,
) -> np.ndarray:
    """Read only requested channels/time and return ``[C,target_samples]``."""
    path = str(record_path)
    if pad_short:
        header = wfdb_module.rdheader(path)
        sampto = min(int(header.sig_len), int(target_samples))
    else:
        sampto = int(target_samples)
    signal, _ = wfdb_module.rdsamp(
        path,
        channels=list(channels),
        sampfrom=0,
        sampto=sampto,
    )
    signal = np.asarray(signal)
    if signal.ndim != 2 or signal.shape[1] != len(channels):
        raise RuntimeError("WFDB selective read returned unexpected channel axes")
    if signal.shape[0] < target_samples:
        if not pad_short:
            raise RuntimeError("WFDB record is shorter than the required input")
        pad_length = target_samples - signal.shape[0]
        left = int(np.floor(pad_length / 2))
        right = int(np.ceil(pad_length / 2))
        signal = np.pad(signal, ((left, right), (0, 0)), mode="constant")
    if signal.shape[0] != target_samples:
        raise RuntimeError("WFDB selective read did not produce the required length")
    return signal.T.copy()
