"""Downstream adapter for DeepECG initialized, ECGCLIP-pretrained encoders."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from .deepecg_ssl_adapter import DeepECGSSLAdapter
from .ecg_preprocessing import ECGCLIP_8_LEADS


class ECGCLIPDeepECGAdapter(DeepECGSSLAdapter):
    """Load only the ECG encoder from a P1 ECGCLIP pretraining checkpoint."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        base_checkpoint_path: str | Path,
        fairseq_root: str | Path,
        input_scale: float,
        normalization_mode: str,
        source_lead_order: Sequence[str] = ECGCLIP_8_LEADS,
    ) -> None:
        p1_checkpoint = Path(checkpoint_path).resolve()
        super().__init__(
            checkpoint_path=base_checkpoint_path,
            fairseq_root=fairseq_root,
            input_scale=input_scale,
            normalization_mode=normalization_mode,
            source_lead_order=source_lead_order,
            gradient_checkpointing=False,
        )
        self.base_checkpoint_path = self.checkpoint_path
        self.base_load_report = self.load_report
        self.checkpoint_path = str(p1_checkpoint)
        self.preprocessing_summary = (
            "P1 ECGCLIP-pretrained DeepECG encoder; deterministic 8-to-12 lead "
            "reconstruction; fixed input scale; linear 500-to-250-Hz resampling; "
            "padding-aware token mean"
        )
        self.load_report = self._load_p1_encoder(p1_checkpoint)

    def _load_p1_encoder(self, checkpoint_path: Path) -> tuple[list[str], list[str]]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
            raise ValueError(
                "Expected a P1 checkpoint dictionary containing model state_dict"
            )
        saved_args = checkpoint.get("args")
        if saved_args is not None:
            if getattr(saved_args, "backbone_init", None) != "deepecg_ssl":
                raise ValueError("Checkpoint was not pretrained from a DeepECG backbone")
            if int(getattr(saved_args, "num_leads", -1)) != 8:
                raise ValueError("P1 checkpoint does not declare eight-lead dataset input")
            if int(getattr(saved_args, "encoder_dim", -1)) != self.feature_dim:
                raise ValueError(
                    "P1 checkpoint encoder dimension does not match the adapter"
                )

        prefixes = ("module.ecg_encoder.", "ecg_encoder.")
        encoder_state = {}
        for key, value in checkpoint["model"].items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    encoder_state[key[len(prefix):]] = value
                    break
        if not encoder_state:
            raise ValueError("P1 checkpoint contains no ecg_encoder parameters")
        if not all(key.startswith("model.") for key in encoder_state):
            invalid = sorted(key for key in encoder_state if not key.startswith("model."))
            raise ValueError(
                "P1 ECG encoder contains unexpected non-DeepECG components: "
                f"{invalid[:5]}"
            )

        incompatible = self.load_state_dict(encoder_state, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        if missing or unexpected:
            raise RuntimeError(
                "P1 DeepECG encoder checkpoint mismatch: "
                f"missing_keys={missing}; unexpected_keys={unexpected}"
            )
        return missing, unexpected
