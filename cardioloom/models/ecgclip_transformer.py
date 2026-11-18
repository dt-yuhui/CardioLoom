"""ECGCLIP with an 8/12-lead wav2vec-style Transformer ECG encoder."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .wav2vec_ecg_transformer import Wav2VecECGTransformer


class SyntheticTextEncoder(nn.Module):
    """Small dry-run-only replacement for the clinical text encoder."""

    def __init__(self, hidden_size: int = 768, vocab_size: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class ECGCLIPTransformer(nn.Module):
    """Output-compatible replacement for ``models.ECG_CLIP.ECGCLIP``."""

    def __init__(
        self,
        args: Any,
        *,
        text_encoder: nn.Module | None = None,
        tokenizer: Any | None = None,
        load_text_encoder: bool = True,
        ecg_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if int(args.num_leads) not in (8, 12):
            raise ValueError("ECGCLIPTransformer num_leads must be 8 or 12")

        self.ecg_model = "ecg_clip_transformer"
        self.num_leads = int(args.num_leads)
        self.feature_dim = int(args.encoder_dim)
        self.projection_dim = int(args.projection_size)
        if ecg_encoder is None:
            self.ecg_encoder = Wav2VecECGTransformer(
                fairseq_root=args.fairseq_root,
                input_leads=self.num_leads,
                conv_channels=int(args.conv_channels),
                conv_layers=int(args.conv_layers),
                encoder_dim=self.feature_dim,
                encoder_layers=int(args.encoder_layers),
                encoder_heads=int(args.encoder_heads),
                encoder_ffn_dim=int(args.encoder_ffn_dim),
                dropout=float(args.encoder_dropout),
                attention_dropout=float(args.attention_dropout),
                activation_dropout=float(args.activation_dropout),
                dropout_input=float(args.dropout_input),
                positional_kernel=int(args.positional_kernel),
                positional_groups=int(args.positional_groups),
                feature_grad_mult=float(args.feature_grad_mult),
                gradient_checkpointing=bool(args.gradient_checkpointing),
            )
        else:
            encoder_dim = int(getattr(ecg_encoder, "feature_dim", -1))
            if encoder_dim != self.feature_dim:
                raise ValueError(
                    f"Injected ECG encoder feature_dim={encoder_dim}, expected {self.feature_dim}"
                )
            self.ecg_encoder = ecg_encoder
        self._ecg_backbone_frozen = False

        hidden_dim = int(args.mlp_hidden_size)
        self.ecg_projector = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.projection_dim),
        )
        self.linear1 = nn.Linear(self.feature_dim, self.projection_dim, bias=False)
        self.linear2 = nn.Linear(self.feature_dim, self.projection_dim, bias=False)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

        if text_encoder is not None:
            self.lm_model = text_encoder
            self.tokenizer = tokenizer
        elif load_text_encoder:
            from transformers import AutoModel, AutoTokenizer

            self.lm_model = AutoModel.from_pretrained(
                args.text_model, local_files_only=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.text_model, local_files_only=True
            )
        else:
            raise ValueError("Provide text_encoder or set load_text_encoder=True")

        self.proj_t = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.projection_dim),
        )
        self.last_monitor: dict[str, torch.Tensor] = {}
        self._reset_projection_parameters()

    def _reset_projection_parameters(self) -> None:
        for module in [self.ecg_projector, self.linear1, self.linear2, self.proj_t]:
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    nn.init.normal_(child.weight, mean=0.0, std=0.02)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)

    def _tokenize(self, text: list[str]) -> Any:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is unavailable in synthetic dry-run mode")
        return self.tokenizer.batch_encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt",
        )

    def get_text_emb(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.lm_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]

    def forward_features(self, ecg: torch.Tensor) -> torch.Tensor:
        if self._ecg_backbone_frozen:
            with torch.no_grad():
                return self.ecg_encoder.forward_features(ecg)
        return self.ecg_encoder.forward_features(ecg)

    def set_ecg_backbone_frozen(self, frozen: bool) -> None:
        """Temporarily stop ECG-backbone gradients without changing DDP parameters."""
        self._ecg_backbone_frozen = bool(frozen)

    def ext_ecg_emb(self, ecg: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(ecg)
        return F.normalize(self.ecg_projector(features), dim=-1)

    def forward(
        self,
        ecg: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if self._ecg_backbone_frozen:
            with torch.no_grad():
                encoder_output = self.ecg_encoder(ecg)
        else:
            encoder_output = self.ecg_encoder(ecg)
        if encoder_output.ndim == 3:
            tokens = encoder_output
            features = tokens.mean(dim=1)
            encoder_mean = tokens.detach().float().mean()
            encoder_std = tokens.detach().float().std(unbiased=False)
        elif encoder_output.ndim == 2:
            # Foundation adapters already apply their model-native token pooling.
            features = encoder_output
            encoder_mean = features.detach().float().mean()
            encoder_std = features.detach().float().std(unbiased=False)
        else:
            raise RuntimeError(
                f"Expected ECG encoder output [B,T,D] or [B,D], got {tuple(encoder_output.shape)}"
            )
        proj_ecg_emb = F.normalize(self.ecg_projector(features), dim=-1)
        ecg_emb1 = self.dropout1(self.linear1(features))
        ecg_emb2 = self.dropout2(self.linear2(features))

        text_emb = self.get_text_emb(input_ids, attention_mask)
        if text_emb.ndim != 2 or text_emb.shape[-1] != 768:
            raise RuntimeError(f"Expected [B,768] text CLS features, got {tuple(text_emb.shape)}")
        proj_text_emb = F.normalize(self.proj_t(text_emb.contiguous()), dim=-1)

        # Batch-level aggregates only: no waveform, text, identifier, label, or
        # sample-level representation is retained or written to TensorBoard.
        with torch.no_grad():
            self.last_monitor = {
                "input_rms": ecg.detach().float().square().mean().sqrt(),
                "token_mean": encoder_mean,
                "token_std": encoder_std,
                "pooled_l2_mean": features.detach().float().norm(dim=-1).mean(),
                "proj_ecg_l2_mean": proj_ecg_emb.detach().float().norm(dim=-1).mean(),
                "proj_text_l2_mean": proj_text_emb.detach().float().norm(dim=-1).mean(),
            }

        expected = (ecg.shape[0], self.projection_dim)
        for name, value in {
            "proj_ecg_emb": proj_ecg_emb,
            "proj_text_emb": proj_text_emb,
            "ecg_emb1": ecg_emb1,
            "ecg_emb2": ecg_emb2,
        }.items():
            if value.shape != expected:
                raise RuntimeError(f"{name} shape {tuple(value.shape)} != {expected}")
        return {
            "ecg_emb": [ecg_emb1, ecg_emb2],
            "proj_ecg_emb": proj_ecg_emb,
            "proj_text_emb": proj_text_emb,
            # The downstream adapter consumes this exact pre-projector feature.
            # Exposing it here allows signal-side regularization without changing
            # the checkpoint state_dict or the public downstream interface.
            "backbone_features": features,
        }
