"""Adapter for heartwise DeepECG-SSL SSL_pretrained.pt."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from .ecg_preprocessing import (
    ECGCLIP_8_LEADS,
    linear_resample,
    reconstruct_standard_12_leads,
    validate_ecg_input,
)
from .fairseq_backbone import (
    FairseqFeatureBackbone,
    enable_fairseq_transformer_gradient_checkpointing,
    load_official_fairseq_backbone,
)


class DeepECGSSLAdapter(FairseqFeatureBackbone):
    feature_dim = 768
    source_sampling_rate = 500
    source_samples = 5000
    model_sampling_rate = 250
    model_samples = 2500

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        fairseq_root: str | Path,
        input_scale: float,
        normalization_mode: str,
        source_lead_order: Sequence[str] = ECGCLIP_8_LEADS,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if input_scale <= 0:
            raise ValueError("input_scale must be explicitly set to a positive value")
        if normalization_mode not in {"fft_power_factor", "mv_scale"}:
            raise ValueError("normalization_mode must be 'fft_power_factor' or 'mv_scale'")
        self.input_scale = float(input_scale)
        self.normalization_mode = normalization_mode
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.source_lead_order = tuple(source_lead_order)
        self.preprocessing_summary = (
            "8-to-12 deterministic lead reconstruction; "
            f"{normalization_mode} scale={self.input_scale}; linear 500-to-250-Hz resampling; token mean"
        )
        self.model, self.checkpoint_cfg, self.load_report = load_official_fairseq_backbone(
            checkpoint_path=checkpoint_path,
            fairseq_root=fairseq_root,
            expected_feature_dim=self.feature_dim,
        )
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.checkpointed_encoder_layers = 0
        if self.gradient_checkpointing:
            self.checkpointed_encoder_layers = (
                enable_fairseq_transformer_gradient_checkpointing(self.model)
            )

    def forward_features(self, ecg: torch.Tensor) -> torch.Tensor:
        validate_ecg_input(
            ecg,
            expected_leads=8,
            expected_samples=self.source_samples,
            source_lead_order=self.source_lead_order,
        )
        ecg_12 = reconstruct_standard_12_leads(ecg)
        # Both official modes reduce to a fixed scalar here. For fft_power_factor,
        # the factor must have been computed from training data only; this adapter
        # never scans validation/test data to estimate it.
        ecg_12 = ecg_12 * self.input_scale
        ecg_12 = linear_resample(ecg_12, self.model_samples)
        features = self._encode_and_pool(ecg_12)
        if features.shape != (ecg.shape[0], self.feature_dim):
            raise RuntimeError(f"Unexpected DeepECG-SSL feature shape {tuple(features.shape)}")
        return features

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        return self.forward_features(ecg)
