"""Multi-positive symmetric ECG-text InfoNCE for repeated report templates."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.nn import functional as F


def distributed_concat_ids(template_ids: torch.Tensor) -> torch.Tensor:
    """Gather non-differentiable int64 template IDs across the DDP group."""

    if template_ids.ndim != 1 or template_ids.dtype != torch.long:
        raise ValueError("template_ids must be a one-dimensional torch.long tensor")
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return template_ids
    gathered = [torch.empty_like(template_ids) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, template_ids.contiguous())
    return torch.cat(gathered, dim=0)

def _any_positive_topk(
    logits: torch.Tensor, positive_mask: torch.Tensor, k: int
) -> torch.Tensor:
    k = min(int(k), logits.shape[1])
    predictions = logits.topk(k, dim=1, largest=True, sorted=True).indices
    hits = positive_mask.gather(1, predictions).any(dim=1)
    return hits.float().mean() * 100.0

def _direction_loss(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    if not bool(positive_mask.any(dim=1).all()):
        raise RuntimeError("Every contrastive anchor must have at least one positive")
    denominator = torch.logsumexp(logits, dim=1)
    positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
    numerator = torch.logsumexp(positive_logits, dim=1)
    return -(numerator - denominator).mean()

def multi_positive_clip_loss(
    ecg_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    template_ids: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return original-scale bidirectional loss, any-positive accuracy and stats."""

    if ecg_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("ECG/text embeddings must have shape [B,D]")
    if ecg_embeddings.shape != text_embeddings.shape:
        raise ValueError("ECG/text embedding shapes must match")
    if template_ids.ndim != 1 or len(template_ids) != len(ecg_embeddings):
        raise ValueError("template_ids must have one value per embedding")
    if template_ids.dtype != torch.long or template_ids.device != ecg_embeddings.device:
        raise ValueError("template_ids must be torch.long on the embedding device")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not torch.isfinite(ecg_embeddings).all() or not torch.isfinite(text_embeddings).all():
        raise ValueError("Contrastive embeddings contain NaN or Inf")

    ecg = F.normalize(ecg_embeddings.float(), dim=-1)
    text = F.normalize(text_embeddings.float(), dim=-1)
    logits = (ecg @ text.T) / float(temperature)
    positive_mask = template_ids[:, None].eq(template_ids[None, :])
    loss_ecg_to_text = _direction_loss(logits, positive_mask)
    loss_text_to_ecg = _direction_loss(logits.T, positive_mask.T)
    loss = loss_ecg_to_text + loss_text_to_ecg
    acc1 = (_any_positive_topk(logits, positive_mask, 1) + _any_positive_topk(logits.T, positive_mask.T, 1)) / 2.0
    acc5 = (_any_positive_topk(logits, positive_mask, 5) + _any_positive_topk(logits.T, positive_mask.T, 5)) / 2.0

    positive_counts = positive_mask.sum(dim=1).float()
    metrics = {
        "report_global_batch_size": template_ids.new_tensor(
            len(template_ids), dtype=torch.float32
        ),
        "report_positive_count_mean": positive_counts.mean(),
        "report_positive_count_max": positive_counts.max(),
        "report_duplicate_anchor_fraction": (positive_counts > 1).float().mean(),
        "report_unique_template_fraction": template_ids.new_tensor(
            torch.unique(template_ids).numel() / float(len(template_ids)),
            dtype=torch.float32,
        ),
        "loss_cma_ecg_to_text": loss_ecg_to_text.detach(),
        "loss_cma_text_to_ecg": loss_text_to_ecg.detach(),
    }
    return loss, acc1, acc5, metrics
