#!/usr/bin/env python3
"""Stable end-to-end downstream fine-tuning for Transformer ECG backbones.

The data, labels, split construction, class weights, threshold search, and
final metric implementation are imported from the original ECGCLIP pipeline.
Only the optimization policy is specialized for a wav2vec-style Transformer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cardioloom.scripts.common import (
    CODEX_ROOT,
    add_dataset_args,
    add_model_args,
    assert_new_output,
    dry_run_report,
    original_downstream_namespace,
    prepare_public_source_imports,
    resolve_model_paths,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    add_model_args(result)
    add_dataset_args(result)
    result.add_argument("--exp-name", required=True)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--num-workers", type=int, default=4)
    result.add_argument("--lr", type=float, default=5e-5)
    result.add_argument("--weight-decay", type=float, default=5e-3)
    result.add_argument("--adam-beta1", type=float, default=0.9)
    result.add_argument("--adam-beta2", type=float, default=0.98)
    result.add_argument("--adam-eps", type=float, default=1e-8)
    result.add_argument("--max-updates", type=int, default=4000)
    result.add_argument("--val-iter", type=int, default=200)
    result.add_argument("--early-stopping-patience", type=int, default=5)
    result.add_argument("--clip-grad-norm", type=float, default=5.0)
    result.add_argument(
        "--selection-metric",
        choices=("pr_auc", "roc_auc"),
        default="pr_auc",
        help="Validation metric used for checkpoint selection and early stopping.",
    )
    result.add_argument("--selection-min-delta", type=float, default=0.0)
    result.add_argument("--feature-grad-mult", type=float, default=0.0)
    result.add_argument("--encoder-dropout", type=float, default=0.0)
    result.add_argument("--attention-dropout", type=float, default=0.0)
    result.add_argument("--activation-dropout", type=float, default=0.1)
    result.add_argument("--dropout-input", type=float, default=0.0)
    result.add_argument(
        "--imbalance-recipe",
        choices=("legacy", "recipe_v3"),
        default="legacy",
        help="Opt-in class-imbalance recipe; legacy exactly preserves completed experiments.",
    )
    result.add_argument("--single-task-sampler-exponent", type=float, default=0.5)
    result.add_argument("--multitask-positive-weight-exponent", type=float, default=0.5)
    result.add_argument("--multitask-positive-weight-cap", type=float, default=10.0)
    result.add_argument("--dry-run", action="store_true")
    return result


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_updates": args.max_updates,
        "val_iter": args.val_iter,
        "early_stopping_patience": args.early_stopping_patience,
        "clip_grad_norm": args.clip_grad_norm,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"Expected positive stable fine-tuning values, got {invalid}")
    if args.weight_decay < 0 or args.adam_eps <= 0:
        raise ValueError("weight decay must be non-negative and Adam epsilon must be positive")
    if not (0 <= args.adam_beta1 < 1 and 0 <= args.adam_beta2 < 1):
        raise ValueError("Adam betas must be in [0, 1)")
    if args.selection_min_delta < 0:
        raise ValueError("selection-min-delta must be non-negative")
    if not 0 <= args.single_task_sampler_exponent <= 1:
        raise ValueError("single-task-sampler-exponent must be in [0, 1]")
    if not 0 <= args.multitask_positive_weight_exponent <= 1:
        raise ValueError("multitask-positive-weight-exponent must be in [0, 1]")
    if args.multitask_positive_weight_cap < 1:
        raise ValueError("multitask-positive-weight-cap must be at least 1")


def _stable_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "model": args.model,
        "initialization": "checkpoint",
        "optimizer": "AdamW",
        "lr_scheduler": "fixed",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "adam_betas": [args.adam_beta1, args.adam_beta2],
        "adam_eps": args.adam_eps,
        "batch_size": args.batch_size,
        "max_updates": args.max_updates,
        "validation_interval": args.val_iter,
        "early_stopping_patience": args.early_stopping_patience,
        "clip_grad_norm": args.clip_grad_norm,
        "multi_task_loss_reduction": "mean",
        "checkpoint_selection": args.selection_metric,
        "feature_grad_mult": args.feature_grad_mult,
        "encoder_dropout": args.encoder_dropout,
        "attention_dropout": args.attention_dropout,
        "activation_dropout": args.activation_dropout,
        "dropout_input": args.dropout_input,
        "imbalance_recipe": args.imbalance_recipe,
        "single_task_sampler_exponent": args.single_task_sampler_exponent,
        "multitask_positive_weight_exponent": args.multitask_positive_weight_exponent,
        "multitask_positive_weight_cap": args.multitask_positive_weight_cap,
    }


def _configure_official_foundation_adapter(adapter, args: argparse.Namespace) -> dict[str, float]:
    """Apply the same explicit fine-tuning dropout/freeze contract to official models."""
    model = adapter.model
    if not hasattr(model, "feature_grad_mult") or not hasattr(model, "encoder"):
        raise AttributeError("Official foundation backbone lacks fine-tuning controls")
    values = {
        "feature_grad_mult": float(args.feature_grad_mult),
        "dropout": float(args.encoder_dropout),
        "attention_dropout": float(args.attention_dropout),
        "activation_dropout": float(args.activation_dropout),
        "dropout_input": float(args.dropout_input),
    }
    model.feature_grad_mult = values["feature_grad_mult"]
    for name in ("dropout_input", "dropout_features"):
        module = getattr(model, name, None)
        if module is not None and hasattr(module, "p"):
            module.p = values["dropout_input"]
    model.encoder.dropout = values["dropout"]
    for layer in model.encoder.layers:
        layer.dropout = values["dropout"]
        layer.activation_dropout = values["activation_dropout"]
        layer.dropout1.p = values["dropout"]
        layer.dropout2.p = values["activation_dropout"]
        layer.dropout3.p = values["dropout"]
        layer.self_attn.dropout.p = values["attention_dropout"]
    return values


def _configure_adapter(adapter, args: argparse.Namespace) -> dict[str, float]:
    if hasattr(adapter, "backbone") and hasattr(
        adapter.backbone, "configure_for_finetuning"
    ):
        return adapter.backbone.configure_for_finetuning(
            feature_grad_mult=args.feature_grad_mult,
            dropout=args.encoder_dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            dropout_input=args.dropout_input,
        )
    if hasattr(adapter, "model"):
        return _configure_official_foundation_adapter(adapter, args)
    raise TypeError(f"Unsupported adapter type: {type(adapter).__name__}")


def _atomic_torch_save(torch_module, payload: dict[str, object], destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        torch_module.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _as_task_labels(torch_module, labels, task_names: list[str], device):
    if isinstance(labels, dict):
        return {name: labels[name].to(device).reshape(-1) for name in task_names}
    if len(task_names) != 1:
        raise RuntimeError("A tensor label is only valid for a single-task experiment")
    return {task_names[0]: labels.to(device).reshape(-1)}


def _classification_loss(torch_module, logits, labels, weight):
    return torch_module.nn.functional.cross_entropy(
        logits.float(), labels.long(), weight=weight, reduction="mean"
    )


def _imbalance_setup(
    *, np_module, torch_module, train_ds, task_info, args, device
):
    """Return sample weights, per-task CE weights and aggregate diagnostics.

    ``legacy`` preserves the completed pipeline exactly. ``recipe_v3`` uses
    tempered sampling without a weighted loss for one binary task, and uniform
    ECG-row sampling with capped task-specific positive weights for multi-label
    tasks.  Counts come from the training split only.
    """

    is_multitask = "multi" in args.downstream_task
    if args.downstream_task == "age":
        raise ValueError("The stable classification recipe does not support age regression")

    if args.imbalance_recipe == "legacy":
        sample_weights = train_ds.get_sample_weights()
        shared_weight = train_ds.get_class_weights()
        if shared_weight is not None:
            shared_weight = torch_module.as_tensor(
                shared_weight, dtype=torch_module.float32, device=device
            )
        task_weights = {task: shared_weight for task in task_info}
        return sample_weights, task_weights, {
            "recipe": "legacy",
            "sampling_unit": "ECG row",
            "sampler": "inverse class count" if sample_weights is not None else "uniform shuffle",
            "loss_weight": "inverse class count" if shared_weight is not None else "none",
            "double_compensation": bool(sample_weights is not None and shared_weight is not None),
            "tasks": {},
        }

    if is_multitask:
        task_weights = {}
        diagnostics = {}
        for task in task_info:
            values = train_ds.labels_df.loc[train_ds.indices, task].to_numpy()
            values = values[values != -1].astype(np_module.int64, copy=False)
            counts = np_module.bincount(values, minlength=2)
            if len(counts) != 2 or int(counts.min()) <= 0:
                raise ValueError(f"Training split must contain both classes for {task}: {counts}")
            negative, positive = int(counts[0]), int(counts[1])
            raw = (negative / positive) ** args.multitask_positive_weight_exponent
            positive_weight = min(raw, args.multitask_positive_weight_cap)
            task_weights[task] = torch_module.tensor(
                [1.0, positive_weight], dtype=torch_module.float32, device=device
            )
            diagnostics[task] = {
                "negative_ecgs": negative,
                "positive_ecgs": positive,
                "natural_positive_fraction": positive / (negative + positive),
                "expected_positive_per_batch": args.batch_size * positive / (negative + positive),
                "positive_loss_weight_raw": raw,
                "positive_loss_weight_applied": positive_weight,
            }
        return None, task_weights, {
            "recipe": "recipe_v3",
            "sampling_unit": "ECG row",
            "sampler": "uniform shuffle",
            "loss_weight": "per-task capped tempered positive weight",
            "sampler_exponent": None,
            "positive_weight_exponent": args.multitask_positive_weight_exponent,
            "positive_weight_cap": args.multitask_positive_weight_cap,
            "double_compensation": False,
            "tasks": diagnostics,
        }

    task = next(iter(task_info))
    values = train_ds.labels_df.loc[train_ds.indices, "label"].to_numpy()
    values = values.astype(np_module.int64, copy=False)
    counts = np_module.bincount(values, minlength=2)
    if len(counts) != 2 or int(counts.min()) <= 0:
        raise ValueError(f"Training split must contain both classes for {task}: {counts}")
    per_class = counts.astype(np_module.float64) ** (-args.single_task_sampler_exponent)
    sample_weights = per_class[values]
    positive_probability = float(sample_weights[values == 1].sum() / sample_weights.sum())
    diagnostics = {
        task: {
            "negative_ecgs": int(counts[0]),
            "positive_ecgs": int(counts[1]),
            "natural_positive_fraction": float(counts[1] / counts.sum()),
            "sampler_positive_probability": positive_probability,
            "expected_positive_per_batch": args.batch_size * positive_probability,
            "positive_loss_weight_applied": 1.0,
        }
    }
    return sample_weights, {task: None}, {
        "recipe": "recipe_v3",
        "sampling_unit": "ECG row",
        "sampler": "tempered inverse class-count sampling",
        "sampler_exponent": args.single_task_sampler_exponent,
        "loss_weight": "none",
        "double_compensation": False,
        "tasks": diagnostics,
    }


def _validation_pass(
    *,
    original,
    np_module,
    torch_module,
    model,
    loader,
    task_class_weights,
    task_info,
    device,
    writer,
    step: int,
    logger,
):
    model.eval()
    logits_dict = {task: [] for task in task_info}
    labels_dict = {task: [] for task in task_info}
    loss_dict = {task: [] for task in task_info}

    with torch_module.no_grad():
        for batch in loader:
            ecg = batch["ecg_data"].to(device, non_blocking=True)
            labels = _as_task_labels(torch_module, batch["label"], list(task_info), device)
            logits = model(ecg)
            for task in task_info:
                task_logits = logits[task].float()
                task_labels = labels[task].long()
                valid = task_labels != -1
                if not bool(valid.any()):
                    continue
                loss_dict[task].append(float(_classification_loss(
                    torch_module, task_logits[valid], task_labels[valid],
                    task_class_weights[task],
                ).item()))
                logits_dict[task].append(task_logits[valid].cpu().numpy())
                labels_dict[task].append(task_labels[valid].cpu().numpy())

    empty = [task for task in task_info if not logits_dict[task]]
    if empty:
        raise RuntimeError(f"Validation has no usable labels for tasks: {empty}")
    for task in task_info:
        logits_dict[task] = np_module.concatenate(logits_dict[task], axis=0)
        labels_dict[task] = np_module.concatenate(labels_dict[task], axis=0)

    results, thresholds = original.get_metrics(
        logits_dict=logits_dict,
        labels_dict=labels_dict,
        task_info=task_info,
        mode="valid",
    )
    averages: dict[str, float] = {}
    for task, metrics in results.items():
        metrics = dict(metrics)
        metrics["loss"] = float(np_module.mean(loss_dict[task]))
        for metric, value in metrics.items():
            writer.add_scalar(f"valid/{task}/{metric}", value, step)
            if metric != "best_threshold":
                averages[metric] = averages.get(metric, 0.0) + float(value)
    for metric in list(averages):
        averages[metric] /= len(results)
        writer.add_scalar(f"valid/average/{metric}", averages[metric], step)
    original.log_loss_to_file(
        {f"valid_{metric}": value for metric, value in sorted(averages.items())},
        logger=logger,
    )
    model.train()
    return averages, thresholds


def _run(args: argparse.Namespace, output_dir: Path) -> None:
    prepare_public_source_imports()
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, WeightedRandomSampler

    import main_downstream as original
    from datasets.dataset_downstream import ECGDatasetDownstream

    from cardioloom.models.finetuner import FoundationFineTuner
    from cardioloom.scripts.common import build_adapter

    # ``original_downstream_namespace`` preserves the legacy argument surface
    # and expects this flag even though the stable entry point intentionally
    # supports training only.
    args.test_only = False
    runtime = original_downstream_namespace(args, output_dir)
    runtime.batch_size = args.batch_size
    runtime.num_workers = args.num_workers
    runtime.total_iterations = args.max_updates
    runtime.lr = args.lr
    runtime.weight_decay = args.weight_decay
    runtime.logits_path = str(output_dir / "logits")

    (output_dir / "tfboard").mkdir(parents=True, exist_ok=False)
    (output_dir / "logits").mkdir(parents=True, exist_ok=False)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = output_dir / f"{timestamp}.log"
    logger = original.get_root_logger(
        log_file=str(log_file), name=f"internal_{args.model}_stable_v2"
    )
    original.log_args_to_file(runtime, "args", logger=logger)
    original.log_dict_to_logger(_stable_config(args), "stable_finetune", logger=logger)

    original.fix_random_seeds(args.seed)
    train_ds, valid_ds, test_ds = ECGDatasetDownstream.create_datasets(
        h5_file=os.path.join(runtime.data_root, runtime.h5_file_path),
        label_file=os.path.join(runtime.data_root, runtime.label_file_path),
        info_file=os.path.join(runtime.data_root, runtime.info_file_path),
        train_index_file=os.path.join(runtime.data_root, runtime.train_indices_file_path),
        valid_index_file=os.path.join(runtime.data_root, runtime.valid_indices_file_path),
        test_index_file=os.path.join(runtime.data_root, runtime.test_indices_file_path),
        task=runtime.downstream_task,
        ecg_or_echo=runtime.ecg_or_echo,
        exclude_label_ids=runtime.exclude_label_ids,
        random_split_seed=runtime.random_split_seed,
        sample_ratio=runtime.sample_ratio,
        num_leads=runtime.num_leads,
        downsample=runtime.downsample,
    )
    original.print_log(f"train dataset load complete: {len(train_ds)}", logger=logger)
    original.print_log(f"valid dataset load complete: {len(valid_ds)}", logger=logger)

    class WorkerLocalECGDataset(ECGDatasetDownstream):
        """Open the read-only waveform HDF5 independently in each worker."""

        def _waveforms(self):
            if self.h5_data is None:
                import h5py

                self._worker_h5_handle = h5py.File(self.h5_file, "r")
                self.h5_data = self._worker_h5_handle["ECGs"]
            return self.h5_data

        def __getitem__(self, position):
            self._waveforms()
            return super().__getitem__(position)

        def __del__(self):
            handle = getattr(self, "_worker_h5_handle", None)
            if handle is not None:
                handle.close()

    for dataset in (train_ds, valid_ds, test_ds):
        # The public dataset opens HDF5 before DataLoader workers are created.
        # Close that parent-process handle and lazily reopen it per worker.
        parent_dataset = getattr(dataset, "h5_data", None)
        if parent_dataset is not None:
            parent_dataset.file.close()
        dataset.h5_data = None
        dataset._worker_h5_handle = None
        dataset.__class__ = WorkerLocalECGDataset

    if "multi" in args.downstream_task:
        task_info = {name: 2 for name in train_ds.get_label_ids()}
    else:
        task_info = {args.downstream_task: args.num_classes}
    if not task_info:
        raise RuntimeError("No downstream tasks were constructed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime.cuda = device.type == "cuda"
    sample_weights, task_class_weights, imbalance_summary = _imbalance_setup(
        np_module=np,
        torch_module=torch,
        train_ds=train_ds,
        task_info=task_info,
        args=args,
        device=device,
    )
    sampler = (
        WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        if sample_weights is not None
        else None
    )
    loader_options = {
        "pin_memory": True,
        "drop_last": False,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    if args.num_workers > 0:
        loader_options.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(
        train_ds,
        sampler=sampler,
        shuffle=sampler is None,
        **loader_options,
    )
    valid_loader = DataLoader(
        valid_ds,
        shuffle=False,
        **loader_options,
    )

    original.print_log(f"Using device: {device}", logger=logger)
    adapter = build_adapter(args)
    if hasattr(adapter, "initialization_summary"):
        original.print_log(
            f"ECG encoder initialization: {adapter.initialization_summary}",
            logger=logger,
        )
    original.print_log(
        f"ecg encoder from {args.model}: "
        f"missing_keys={adapter.load_report[0]}; unexpected_keys={adapter.load_report[1]}",
        logger=logger,
    )
    configured = _configure_adapter(adapter, args)
    original.log_dict_to_logger(
        configured, "adapter_finetune_configuration", logger=logger
    )
    model = FoundationFineTuner(
        backbone=adapter,
        in_features=adapter.feature_dim,
        dropout=0.0,
        finetune_type="finetune",
        task_info=task_info,
    )
    nn.init.xavier_uniform_(model.classifier[1].weight)
    nn.init.zeros_(model.classifier[1].bias)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    writer = original.create_tfboard_on_master(str(output_dir / "tfboard"))
    if writer is None:
        raise RuntimeError("TensorBoard writer was not created")
    original.log_dict_to_logger(
        imbalance_summary, "imbalance_recipe", logger=logger
    )
    for task, statistics in imbalance_summary["tasks"].items():
        for name, value in statistics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"imbalance/{task}/{name}", float(value), 0)

    best_value = -float("inf")
    bad_validations = 0
    global_step = 0
    last_validation_step = -1
    checkpoint_path = output_dir / "best-checkpoint.pth"
    model.train()

    while global_step < args.max_updates and bad_validations < args.early_stopping_patience:
        for batch in train_loader:
            if global_step >= args.max_updates or bad_validations >= args.early_stopping_patience:
                break
            ecg = batch["ecg_data"].to(device, non_blocking=True)
            labels = _as_task_labels(torch, batch["label"], list(task_info), device)
            logits = model(ecg)
            losses = []
            for task in task_info:
                valid = labels[task] != -1
                if bool(valid.any()):
                    losses.append(_classification_loss(
                        torch, logits[task][valid], labels[task][valid],
                        task_class_weights[task],
                    ))
            if not losses:
                raise RuntimeError("Training batch contains no usable labels")
            total_loss = torch.stack(losses).mean()
            if not bool(torch.isfinite(total_loss)):
                raise FloatingPointError(f"Non-finite training loss at step {global_step}")

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(f"Non-finite gradient norm at step {global_step}")
            optimizer.step()
            global_step += 1

            writer.add_scalar("train/loss", float(total_loss.item()), global_step)
            writer.add_scalar("train/grad_norm_pre_clip", float(grad_norm), global_step)
            writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), global_step)
            if global_step == 1 or global_step % 50 == 0:
                original.print_log(
                    f"update={global_step}/{args.max_updates} loss={total_loss.item():.6g} "
                    f"grad_norm_pre_clip={float(grad_norm):.6g} lr={args.lr:.6g}",
                    logger=logger,
                )

            if global_step % args.val_iter == 0:
                averages, thresholds = _validation_pass(
                    original=original,
                    np_module=np,
                    torch_module=torch,
                    model=model,
                    loader=valid_loader,
                    task_class_weights=task_class_weights,
                    task_info=task_info,
                    device=device,
                    writer=writer,
                    step=global_step,
                    logger=logger,
                )
                last_validation_step = global_step
                selection_value = float(averages[args.selection_metric])
                if selection_value > best_value + args.selection_min_delta:
                    best_value = selection_value
                    bad_validations = 0
                    _atomic_torch_save(
                        torch,
                        {
                            "model": model.state_dict(),
                            "threshold": thresholds,
                            "task_names": list(task_info),
                            "global_step": global_step,
                            "selection_metric": args.selection_metric,
                            "selection_value": selection_value,
                            "stable_finetune_config": _stable_config(args),
                        },
                        checkpoint_path,
                    )
                    original.print_log(
                        f"Saved best checkpoint: step={global_step} "
                        f"{args.selection_metric}={selection_value:.6g}",
                        logger=logger,
                    )
                else:
                    bad_validations += 1
                    original.print_log(
                        f"No checkpoint improvement: {bad_validations}/"
                        f"{args.early_stopping_patience}",
                        logger=logger,
                    )
                writer.add_scalar("valid/selection/best", best_value, global_step)

    if last_validation_step < 0 or not checkpoint_path.is_file():
        raise RuntimeError("Training ended before a validation checkpoint was created")
    original.print_log(
        f"Training stopped at update={global_step}; best_{args.selection_metric}={best_value:.6g}",
        logger=logger,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    thresholds = checkpoint["threshold"]
    original.test_model(
        eval_ds=valid_ds,
        finetuner=model,
        task_info=task_info,
        split_name="valid",
        logger=logger,
        writer=writer,
        args=runtime,
        threshold=thresholds,
        save_flag=False,
    )
    original.test_model(
        eval_ds=test_ds,
        finetuner=model,
        task_info=task_info,
        split_name="internal_test",
        logger=logger,
        writer=writer,
        args=runtime,
        threshold=thresholds,
        save_flag=False,
    )
    writer.flush()
    writer.close()


def main() -> None:
    args = parser().parse_args()
    _validate_args(args)
    resolve_model_paths(args)
    output_dir = CODEX_ROOT / "outputs/finetune" / args.exp_name
    if args.dry_run:
        dry_run_report(args, output_paths={"experiment": output_dir})
        print(json.dumps({"stable_finetune": _stable_config(args)}, indent=2))
        return
    output_dir = assert_new_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    _run(args, output_dir)


if __name__ == "__main__":
    main()
