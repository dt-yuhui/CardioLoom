"""Exact ECG-FM/DeepECG fairseq-signals backbone adapted to ECGCLIP leads.

The two official checkpoints serialize the same encoder configuration:
four ``(256, 2, 2)`` convolution blocks, a 12-layer post-norm Transformer
with width 768, 12 heads and FFN width 3072, plus a grouped positional
convolution. This module directly reuses the official fairseq-signals
components. ``input_leads`` explicitly selects either eight independent leads
or the deterministically reconstructed standard twelve-lead representation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .ecg_preprocessing import ECGCLIP_8_LEADS, STANDARD_12_LEADS


class Wav2VecECGTransformer(nn.Module):
    """Official fairseq-signals ECG backbone for 8/12-lead ECGCLIP ablations."""

    input_leads = 8
    input_samples = 5000
    sampling_rate_hz = 500

    def __init__(
        self,
        *,
        fairseq_root: str | Path,
        input_leads: int = 8,
        conv_channels: int = 256,
        conv_layers: int = 4,
        encoder_dim: int = 768,
        encoder_layers: int = 12,
        encoder_heads: int = 12,
        encoder_ffn_dim: int = 3072,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        dropout_input: float = 0.1,
        positional_kernel: int = 128,
        positional_groups: int = 16,
        feature_grad_mult: float = 0.1,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if int(input_leads) not in (8, 12):
            raise ValueError(f"input_leads must be 8 or 12, got {input_leads}")
        self.input_leads = int(input_leads)
        self.input_lead_order = (
            ECGCLIP_8_LEADS if self.input_leads == 8 else STANDARD_12_LEADS
        )

        expected = {
            "conv_channels": (conv_channels, 256),
            "conv_layers": (conv_layers, 4),
            "encoder_dim": (encoder_dim, 768),
            "encoder_layers": (encoder_layers, 12),
            "encoder_heads": (encoder_heads, 12),
            "encoder_ffn_dim": (encoder_ffn_dim, 3072),
            "dropout": (dropout, 0.1),
            "attention_dropout": (attention_dropout, 0.1),
            "activation_dropout": (activation_dropout, 0.0),
            "dropout_input": (dropout_input, 0.1),
            "positional_kernel": (positional_kernel, 128),
            "positional_groups": (positional_groups, 16),
            "feature_grad_mult": (feature_grad_mult, 0.1),
        }
        mismatches = [
            f"{name}={actual} (expected {required})"
            for name, (actual, required) in expected.items()
            if actual != required
        ]
        if mismatches:
            raise ValueError(
                "Exact ECG-FM/DeepECG backbone configuration required: "
                + "; ".join(mismatches)
            )

        fairseq_root = Path(fairseq_root).resolve()
        if not (fairseq_root / "fairseq_signals").is_dir():
            raise FileNotFoundError(
                f"Missing official fairseq_signals package under {fairseq_root}"
            )
        loaded = sys.modules.get("fairseq_signals")
        if loaded is not None:
            loaded_path = Path(loaded.__file__).resolve()
            if fairseq_root not in loaded_path.parents:
                raise RuntimeError(
                    "A different fairseq_signals fork is already imported in this process"
                )
        elif str(fairseq_root) not in sys.path:
            sys.path.insert(0, str(fairseq_root))

        from fairseq_signals.modules import (
            ConvFeatureExtraction,
            ConvPositionalEncoding,
            GradMultiply,
            LayerNorm,
            TransformerEncoder,
        )

        self.feature_dim = int(encoder_dim)
        self.feature_grad_mult = float(feature_grad_mult)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self._grad_multiply = GradMultiply

        conv_spec = [(conv_channels, 2, 2)] * conv_layers
        self.feature_extractor = ConvFeatureExtraction(
            conv_layers=conv_spec,
            in_d=self.input_leads,
            dropout=0.0,
            mode="default",
            conv_bias=False,
        )
        self.feature_layer_norm = LayerNorm(conv_channels)
        self.post_extract_proj = nn.Linear(conv_channels, encoder_dim)
        self.dropout_input = nn.Dropout(dropout_input)

        cfg = SimpleNamespace(
            encoder_embed_dim=encoder_dim,
            encoder_ffn_embed_dim=encoder_ffn_dim,
            encoder_attention_heads=encoder_heads,
            encoder_layers=encoder_layers,
            layer_norm_first=False,
            saliency=False,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            encoder_layerdrop=0.0,
            conv_pos=positional_kernel,
            conv_pos_groups=positional_groups,
        )
        self.conv_pos = ConvPositionalEncoding(cfg)
        self.encoder = TransformerEncoder(cfg)

        self.finetuning_configuration: dict[str, float] | None = None

    def configure_for_finetuning(
        self,
        *,
        feature_grad_mult: float = 0.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.1,
        dropout_input: float = 0.0,
    ) -> dict[str, float]:
        """Apply the official fairseq-signals fine-tuning dropout contract.

        The pretrained weights are unchanged.  Only gradient flow through the
        convolutional feature extractor and the non-parameterized dropout
        probabilities are adjusted.  The defaults mirror the ECG diagnosis
        fine-tuning configs shipped with ECG-FM/DeepECG fairseq-signals.
        """

        values = {
            "feature_grad_mult": float(feature_grad_mult),
            "dropout": float(dropout),
            "attention_dropout": float(attention_dropout),
            "activation_dropout": float(activation_dropout),
            "dropout_input": float(dropout_input),
        }
        if not 0.0 <= values["feature_grad_mult"] <= 1.0:
            raise ValueError(
                "feature_grad_mult must be in [0, 1], got "
                f"{values['feature_grad_mult']}"
            )
        for name in ("dropout", "attention_dropout", "activation_dropout", "dropout_input"):
            value = values[name]
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}")

        self.feature_grad_mult = values["feature_grad_mult"]
        self.dropout_input.p = values["dropout_input"]
        self.encoder.dropout = values["dropout"]
        for layer in self.encoder.layers:
            layer.dropout = values["dropout"]
            layer.activation_dropout = values["activation_dropout"]
            layer.dropout1.p = values["dropout"]
            layer.dropout2.p = values["activation_dropout"]
            layer.dropout3.p = values["dropout"]
            layer.self_attn.dropout.p = values["attention_dropout"]

        self.finetuning_configuration = dict(values)
        return dict(values)

    def _validate_input(self, ecg: torch.Tensor) -> None:
        if not torch.is_tensor(ecg):
            raise TypeError("ECG input must be a torch.Tensor")
        if ecg.ndim != 3:
            raise ValueError(
                f"Expected ECG rank 3 [B,{self.input_leads},5000], got {tuple(ecg.shape)}"
            )
        if ecg.shape[1:] != (self.input_leads, self.input_samples):
            raise ValueError(
                f"Expected ECG [B,{self.input_leads},5000] in lead order "
                f"{list(self.input_lead_order)}, got {tuple(ecg.shape)}"
            )
        if not ecg.is_floating_point():
            raise TypeError(f"ECG input must be floating point, got {ecg.dtype}")

    @staticmethod
    def _checkpoint_layer(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        output, _ = layer(
            x,
            self_attn_padding_mask=None,
            self_attn_mask=None,
        )
        return output

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # Exact TransformerEncoder path for layer_norm_first=False and
        # encoder_layerdrop=0.0. Checkpointing only trades compute for memory.
        x = self.encoder.layer_norm(x)
        x = F.dropout(x, p=self.encoder.dropout, training=self.training)
        x = x.transpose(0, 1)
        for layer in self.encoder.layers:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                def layer_forward(value: torch.Tensor, current_layer: nn.Module = layer) -> torch.Tensor:
                    return self._checkpoint_layer(current_layer, value)

                x = checkpoint(layer_forward, x, use_reentrant=False)
            else:
                x, _ = layer(
                    x,
                    self_attn_padding_mask=None,
                    self_attn_mask=None,
                )
        return x.transpose(0, 1)

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        self._validate_input(ecg)
        if self.feature_grad_mult > 0.0:
            features = self.feature_extractor(ecg)
            if self.feature_grad_mult != 1.0:
                features = self._grad_multiply.apply(features, self.feature_grad_mult)
        else:
            # This is the official fairseq-signals behavior when the
            # fine-tuning config sets feature_grad_mult=0.0.
            with torch.no_grad():
                features = self.feature_extractor(ecg)
        features = self.feature_layer_norm(features.transpose(1, 2))
        tokens = self.post_extract_proj(features)
        tokens = self.dropout_input(tokens)
        tokens = tokens + self.conv_pos(tokens, channel_first=False)
        tokens = self._encode(tokens)
        expected = (ecg.shape[0], 312, self.feature_dim)
        if tokens.shape != expected:
            raise RuntimeError(f"Context feature shape {tuple(tokens.shape)} != {expected}")
        return tokens

    def forward_features(self, ecg: torch.Tensor) -> torch.Tensor:
        """Return the official aggregate-token representation ``[B,768]``."""
        return self.forward(ecg).mean(dim=1)
