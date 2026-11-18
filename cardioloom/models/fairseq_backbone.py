"""CPU-safe checkpoint construction for the two official fairseq-signals forks."""

from __future__ import annotations

import inspect
import logging
import sys
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from torch.nn import functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .ecg_preprocessing import masked_time_mean


LOGGER = logging.getLogger(__name__)


def _checkpointed_transformer_extract_features(
    encoder: nn.Module,
    x: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact fairseq-signals encoder path with per-layer recomputation.

    DeepECG's ECG Transformer has no activation-checkpointing switch.  This
    function preserves its post-norm forward order and only changes how
    intermediate activations are retained for backward.
    """
    if padding_mask is not None:
        x[padding_mask] = 0

    if not encoder.layer_norm_first:
        x = encoder.layer_norm(x)
    x = F.dropout(x, p=encoder.dropout, training=encoder.training)
    x = x.transpose(0, 1)

    use_checkpointing = encoder.training and torch.is_grad_enabled()
    for layer in encoder.layers:
        if use_checkpointing:
            def layer_forward(
                value: torch.Tensor,
                current_layer: nn.Module = layer,
            ) -> torch.Tensor:
                output, _ = current_layer(
                    value,
                    self_attn_padding_mask=padding_mask,
                    self_attn_mask=attn_mask,
                    need_weights=False,
                )
                return output

            x = checkpoint(
                layer_forward,
                x,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            x, _ = layer(
                x,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                need_weights=False,
            )

    return x.transpose(0, 1)


def enable_fairseq_transformer_gradient_checkpointing(model: nn.Module) -> int:
    """Enable checkpointing without editing or reparenting third-party modules."""
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise AttributeError("DeepECG backbone has no Transformer encoder")
    required = ("layers", "layer_norm_first", "layer_norm", "dropout", "layerdrop")
    missing = [name for name in required if not hasattr(encoder, name)]
    if missing:
        raise AttributeError(
            f"DeepECG encoder is incompatible with checkpointing; missing={missing}"
        )
    if float(encoder.layerdrop) != 0.0:
        raise ValueError(
            "Exact DeepECG activation checkpointing currently requires encoder_layerdrop=0"
        )
    layer_count = len(encoder.layers)
    if layer_count <= 0:
        raise ValueError("DeepECG encoder has no Transformer layers")
    layer_parameters = inspect.signature(encoder.layers[0].forward).parameters
    if "need_weights" not in layer_parameters:
        raise TypeError(
            "DeepECG checkpointing requires the official DeepECG fairseq-signals "
            "fork; the loaded Transformer layer has an incompatible forward signature"
        )

    if not bool(getattr(encoder, "_codex_gradient_checkpointing", False)):
        encoder.extract_features = MethodType(
            _checkpointed_transformer_extract_features,
            encoder,
        )
        encoder._codex_gradient_checkpointing = True
    return layer_count


def _select_cfg(state: dict[str, Any], checkpoint_utils: Any) -> Any:
    if state.get("cfg") is not None:
        return state["cfg"]
    if state.get("args") is not None:
        return checkpoint_utils.convert_namespace_to_omegaconf(state["args"])
    raise RuntimeError(f"Checkpoint has neither 'cfg' nor 'args'; keys={sorted(state)}")


def _setup_task(tasks_module: Any, task_cfg: Any) -> Any:
    try:
        return tasks_module.setup_task(task_cfg, from_checkpoint=True)
    except TypeError:
        return tasks_module.setup_task(task_cfg)


def _load_checkpoint_cpu_compat(
    checkpoint_path: Path,
    checkpoint_utils: Any,
) -> dict[str, Any]:
    """Load official checkpoints plus Codex WCR continuation checkpoints.

    The old DeepECG fairseq-signals upgrader assumes that any checkpoint
    without ``optimizer_history`` is a much older fairseq checkpoint and then
    unconditionally reads top-level ``best_loss``/``optimizer``.  Our WCR
    continuation checkpoint intentionally stores ``cfg`` and ``model`` plus a
    modern optimizer state, so that legacy assumption is false.  Keep the
    official loading path first and use a narrowly marked CPU-only fallback.
    """
    try:
        return checkpoint_utils.load_checkpoint_to_cpu(str(checkpoint_path))
    except KeyError as error:
        if error.args != ("best_loss",):
            raise
        state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        extra_state = state.get("extra_state", {})
        if not bool(extra_state.get("codex_wcr_continued_pretraining", False)):
            raise
        if state.get("cfg") is None or state.get("model") is None:
            raise RuntimeError(
                "Codex WCR checkpoint must contain non-empty 'cfg' and 'model'"
            ) from error
        cfg = state["cfg"]
        if isinstance(cfg, dict):
            from omegaconf import OmegaConf

            cfg = OmegaConf.create(cfg)
            OmegaConf.set_struct(cfg, True)
            state["cfg"] = cfg
        LOGGER.warning(
            "Loaded marked Codex WCR checkpoint via compatibility path: %s",
            checkpoint_path,
        )
        return state


def load_official_fairseq_backbone(
    *, checkpoint_path: str | Path, fairseq_root: str | Path, expected_feature_dim: int
) -> tuple[nn.Module, Any, tuple[list[str], list[str]]]:
    checkpoint_path = Path(checkpoint_path).resolve()
    fairseq_root = Path(fairseq_root).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not (fairseq_root / "fairseq_signals").is_dir():
        raise FileNotFoundError(f"Missing fairseq_signals package under {fairseq_root}")

    loaded = sys.modules.get("fairseq_signals")
    if loaded is not None:
        loaded_path = Path(loaded.__file__).resolve()
        if fairseq_root not in loaded_path.parents:
            raise RuntimeError(
                "A different fairseq_signals fork is already imported in this process. "
                "Instantiate ECG-FM and DeepECG-SSL in separate processes."
            )
    elif str(fairseq_root) not in sys.path:
        sys.path.insert(0, str(fairseq_root))

    from fairseq_signals import tasks
    from fairseq_signals.utils import checkpoint_utils

    # Both paths use map_location=torch.device("cpu").
    state = _load_checkpoint_cpu_compat(checkpoint_path, checkpoint_utils)
    cfg = _select_cfg(state, checkpoint_utils)
    feature_dim = int(cfg.model.encoder_embed_dim)
    if feature_dim != expected_feature_dim:
        raise ValueError(
            f"Checkpoint feature dim {feature_dim} != adapter expectation {expected_feature_dim}"
        )

    task = _setup_task(tasks, cfg.task)
    build_spec = inspect.getfullargspec(task.build_model)
    if "from_checkpoint" in build_spec.args:
        model = task.build_model(cfg.model, from_checkpoint=True)
    else:
        model = task.build_model(cfg.model)

    try:
        incompatible = model.load_state_dict(state["model"], strict=False, model_cfg=cfg.model)
    except TypeError:
        incompatible = model.load_state_dict(state["model"], strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    LOGGER.warning("checkpoint missing_keys=%s", missing)
    LOGGER.warning("checkpoint unexpected_keys=%s", unexpected)

    if hasattr(model, "remove_pretraining_modules"):
        model.remove_pretraining_modules()
    return model, cfg, (missing, unexpected)


def build_random_fairseq_backbone_from_checkpoint_config(
    *, config_checkpoint_path: str | Path, fairseq_root: str | Path,
    expected_feature_dim: int,
) -> tuple[nn.Module, Any]:
    """Build the official architecture without loading any checkpoint weights.

    The checkpoint is read on CPU only to recover the exact task/model config.
    Its ``model`` state dict is deliberately never passed to ``load_state_dict``.
    Random initialization therefore follows the official model constructor and
    the caller-controlled PyTorch seed.
    """
    config_checkpoint_path = Path(config_checkpoint_path).resolve()
    fairseq_root = Path(fairseq_root).resolve()
    if not config_checkpoint_path.is_file():
        raise FileNotFoundError(config_checkpoint_path)
    if not (fairseq_root / "fairseq_signals").is_dir():
        raise FileNotFoundError(f"Missing fairseq_signals package under {fairseq_root}")

    loaded = sys.modules.get("fairseq_signals")
    if loaded is not None:
        loaded_path = Path(loaded.__file__).resolve()
        if fairseq_root not in loaded_path.parents:
            raise RuntimeError(
                "A different fairseq_signals fork is already imported in this process."
            )
    elif str(fairseq_root) not in sys.path:
        sys.path.insert(0, str(fairseq_root))

    from fairseq_signals import tasks
    from fairseq_signals.utils import checkpoint_utils

    state = _load_checkpoint_cpu_compat(config_checkpoint_path, checkpoint_utils)
    cfg = _select_cfg(state, checkpoint_utils)
    feature_dim = int(cfg.model.encoder_embed_dim)
    if feature_dim != expected_feature_dim:
        raise ValueError(
            f"Checkpoint config feature dim {feature_dim} != adapter expectation "
            f"{expected_feature_dim}"
        )
    task = _setup_task(tasks, cfg.task)
    build_spec = inspect.getfullargspec(task.build_model)
    if "from_checkpoint" in build_spec.args:
        model = task.build_model(cfg.model, from_checkpoint=True)
    else:
        model = task.build_model(cfg.model)
    # Explicitly discard the checkpoint state before returning the random model.
    del state
    if hasattr(model, "remove_pretraining_modules"):
        model.remove_pretraining_modules()
    return model, cfg


class FairseqFeatureBackbone(nn.Module):
    feature_dim: int

    def _encode_and_pool(self, source: torch.Tensor) -> torch.Tensor:
        result = self.model.extract_features(source=source, padding_mask=None, mask=False)
        tokens = result["x"]
        pooled = masked_time_mean(tokens, result.get("padding_mask"))
        if pooled.shape != (source.shape[0], self.feature_dim):
            raise RuntimeError(
                f"Backbone returned {tuple(pooled.shape)}, expected {(source.shape[0], self.feature_dim)}"
            )
        return pooled
