"""Thin downstream head that delegates feature extraction to a foundation adapter."""

from __future__ import annotations

import torch
from torch import nn


class FoundationFineTuner(nn.Module):
    """Signature-compatible replacement for ECG_CLIP.models.finetuner.FineTuner."""

    def __init__(
        self,
        backbone: nn.Module,
        in_features: int,
        dropout: float,
        task_info: dict[str, int],
        backbone_name: str = "foundation",
        hidden_dim: int | None = None,
        finetune_type: str = "finetune",
    ) -> None:
        super().__init__()
        del in_features, backbone_name
        self.backbone = backbone
        self.task_info = task_info
        self.task_names = list(task_info)
        self.feature_dim = int(backbone.feature_dim)
        if hidden_dim is not None:
            raise ValueError("hidden_dim is intentionally unsupported to match the existing linear head")
        if finetune_type in {"linear_prob", "svm"}:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        total_outputs = sum(task_info.values())
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_dim, total_outputs))
        nn.init.trunc_normal_(self.classifier[1].weight, std=0.02)
        nn.init.zeros_(self.classifier[1].bias)

    def forward_features(self, batch: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(batch)

    def forward_head(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        joined = self.classifier(features)
        logits: dict[str, torch.Tensor] = {}
        offset = 0
        for task_name, num_classes in self.task_info.items():
            logits[task_name] = joined[:, offset : offset + num_classes]
            offset += num_classes
        return logits

    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward_head(self.forward_features(batch))

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Never allow an external caller to silently accept checkpoint drift."""
        del strict
        return super().load_state_dict(state_dict, strict=True, assign=assign)
