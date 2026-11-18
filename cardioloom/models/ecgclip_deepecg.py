"""ECGCLIP initialized from the official DeepECG-SSL twelve-lead backbone."""

from __future__ import annotations

from typing import Any

from torch import nn

from .deepecg_ssl_adapter import DeepECGSSLAdapter
from .ecgclip_transformer import ECGCLIPTransformer


class ECGCLIPDeepECG(ECGCLIPTransformer):
    """Keep the ECGCLIP objective while replacing its ECG encoder initialization."""

    def __init__(
        self,
        args: Any,
        *,
        text_encoder: nn.Module | None = None,
        tokenizer: Any | None = None,
        load_text_encoder: bool = True,
    ) -> None:
        adapter = DeepECGSSLAdapter(
            checkpoint_path=args.deepecg_checkpoint,
            fairseq_root=args.fairseq_root,
            input_scale=float(args.deepecg_input_scale),
            normalization_mode=args.deepecg_normalization_mode,
            gradient_checkpointing=bool(args.gradient_checkpointing),
        )
        # Match the official diagnosis fine-tuning contract: keep the convolutional
        # feature extractor frozen while allowing the Transformer to adapt to CLIP.
        if not hasattr(adapter.model, "feature_grad_mult"):
            raise AttributeError("DeepECG backbone has no feature_grad_mult control")
        adapter.model.feature_grad_mult = 0.0
        load_report = adapter.load_report
        super().__init__(
            args,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            load_text_encoder=load_text_encoder,
            ecg_encoder=adapter,
        )
        self.ecg_model = "ecg_clip_deepecg_init"
        self.initial_backbone_load_report = load_report
