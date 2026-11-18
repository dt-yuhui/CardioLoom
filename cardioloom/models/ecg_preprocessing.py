"""Explicit, differentiable preprocessing shared by internal adapters."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


ECGCLIP_8_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
STANDARD_12_LEADS = (
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"
)


def validate_ecg_input(
    ecg: torch.Tensor,
    *,
    expected_leads: int,
    expected_samples: int,
    source_lead_order: Sequence[str],
) -> None:
    if ecg.ndim != 3:
        raise ValueError(f"Expected ECG [B,C,T], got shape={tuple(ecg.shape)}")
    if ecg.shape[1] != expected_leads:
        raise ValueError(f"Expected {expected_leads} leads, got {ecg.shape[1]}")
    if ecg.shape[2] != expected_samples:
        raise ValueError(f"Expected {expected_samples} samples, got {ecg.shape[2]}")
    if tuple(source_lead_order) != ECGCLIP_8_LEADS:
        raise ValueError(
            "The adapter only accepts explicit ECGCLIP lead order "
            f"{ECGCLIP_8_LEADS}; received {tuple(source_lead_order)}"
        )
    if not ecg.is_floating_point():
        raise TypeError(f"Expected floating-point ECG tensor, got {ecg.dtype}")
    if not torch.isfinite(ecg).all():
        raise ValueError("ECG input contains NaN or Inf")


def reconstruct_standard_12_leads(ecg_8: torch.Tensor) -> torch.Tensor:
    """Reconstruct III/aVR/aVL/aVF from I/II and return standard 12-lead order."""
    lead_i = ecg_8[:, 0, :]
    lead_ii = ecg_8[:, 1, :]
    lead_iii = lead_ii - lead_i
    lead_avr = -(lead_i + lead_ii) / 2.0
    lead_avl = lead_i - lead_ii / 2.0
    lead_avf = lead_ii - lead_i / 2.0
    return torch.stack(
        (
            lead_i,
            lead_ii,
            lead_iii,
            lead_avr,
            lead_avl,
            lead_avf,
            *tuple(ecg_8[:, 2:, :].unbind(dim=1)),
        ),
        dim=1,
    )


def standardize_each_lead(ecg: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-record, per-lead z-score; constant leads become exactly zero."""
    mean = ecg.mean(dim=-1, keepdim=True)
    std = ecg.std(dim=-1, keepdim=True, unbiased=False)
    standardized = (ecg - mean) / std.clamp_min(eps)
    return torch.where(std > eps, standardized, torch.zeros_like(standardized))


def linear_resample(ecg: torch.Tensor, target_samples: int) -> torch.Tensor:
    """Differentiable linear resampling of [B,C,T] to an explicit target length."""
    if target_samples <= 0:
        raise ValueError("target_samples must be positive")
    if ecg.shape[-1] == target_samples:
        return ecg
    return F.interpolate(ecg, size=target_samples, mode="linear", align_corners=False)


def masked_time_mean(tokens: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Expected token tensor [B,T,D], got {tuple(tokens.shape)}")
    if padding_mask is None:
        return tokens.mean(dim=1)
    if padding_mask.shape != tokens.shape[:2]:
        raise ValueError(
            f"padding_mask shape {tuple(padding_mask.shape)} does not match {tuple(tokens.shape[:2])}"
        )
    keep = (~padding_mask).to(tokens.dtype).unsqueeze(-1)
    denom = keep.sum(dim=1).clamp_min(1.0)
    return (tokens * keep).sum(dim=1) / denom
