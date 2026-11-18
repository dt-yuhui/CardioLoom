"""Corrected linear-SVM protocol for offline ECG features.

This module is importable at pickle load time.  Historical ECGCLIP SVM
artifacts are left untouched; new experiments opt in with ``rawscore_v2``.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)


SVM_PROTOCOL = "rawscore_v2"
SVM_CLASSES = np.asarray([-1, 1], dtype=np.int64)


class CorrectedSGDClassifier(SGDClassifier):
    """Online linear hinge classifier with a stable two-class contract.

    The dataset sampler already balances the two classes.  The historical
    implementation additionally recomputed class weights independently for
    every mini-batch.  ``rawscore_v2`` deliberately ignores that second set of
    weights and always supplies both legal classes to ``partial_fit``.
    """

    svm_protocol_ = SVM_PROTOCOL
    score_space_ = "decision_function"
    class_balance_ = "weighted_sampler_only"

    def partial_fit(self, X, y, classes=None, sample_weight=None):  # noqa: N803
        del classes, sample_weight
        return super().partial_fit(X, y, classes=SVM_CLASSES, sample_weight=None)


def build_corrected_classifier(*args, random_state: int, **kwargs) -> CorrectedSGDClassifier:
    kwargs["random_state"] = int(random_state)
    model = CorrectedSGDClassifier(*args, **kwargs)
    return model


def identity_decision_scores(scores, scale_factor=None):
    """Return raw decision scores; retained signature matches legacy helper."""

    del scale_factor
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected one-dimensional SVM decision scores, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("SVM decision scores contain non-finite values")
    return values


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _positive_scores(logits: np.ndarray) -> np.ndarray:
    if logits.ndim == 1:
        scores = logits.astype(np.float64, copy=False)
    elif logits.ndim == 2 and logits.shape[1] == 2:
        stable = logits - np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(stable)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        scores = probabilities[:, 1].astype(np.float64, copy=False)
    else:
        raise ValueError(f"Expected [N] decision scores or [N,2] logits, got {logits.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("Metric input contains non-finite scores")
    return scores


def _binary_counts(labels: np.ndarray) -> dict[int, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {int(label): int(count) for label, count in zip(unique, counts)}


def _require_binary_split(labels: np.ndarray, *, task_name: str, mode: str) -> None:
    counts = _binary_counts(labels)
    if set(counts) != {0, 1}:
        raise ValueError(
            f"{mode} split for {task_name} must contain labels 0 and 1; counts={counts}"
        )


def search_best_raw_threshold(labels, scores) -> tuple[float, float]:
    """Choose the exact validation-score threshold that maximizes F1.

    ``precision_recall_curve`` returns thresholds in ascending score order.
    Ties therefore resolve deterministically to the smallest threshold, which
    matches the historical grid search's first-improvement behavior.
    """

    labels = _as_numpy(labels).reshape(-1).astype(np.int64, copy=False)
    scores = _as_numpy(scores).reshape(-1).astype(np.float64, copy=False)
    if labels.shape != scores.shape:
        raise ValueError(f"Label/score shape mismatch: {labels.shape} versus {scores.shape}")
    _require_binary_split(labels, task_name="threshold_search", mode="validation")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        raise RuntimeError("Validation threshold search returned no candidate thresholds")
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator > 0,
    )
    best_index = int(np.flatnonzero(f1_values == np.max(f1_values))[0])
    return float(thresholds[best_index]), float(f1_values[best_index])


def _classification_metrics(labels, scores, threshold: float) -> dict[str, float]:
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": specificity,
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        # Preserve ECGCLIP's historical trapezoidal PR-AUC definition.
        "pr_auc": float(auc(recall, precision)),
    }


def get_rawscore_metrics(
    logits_dict: Mapping[str, object],
    labels_dict: Mapping[str, object],
    task_info: Mapping[str, int],
    mode: str = "valid",
    threshold: Mapping[str, float] | None = None,
):
    """Drop-in replacement for ECGCLIP ``get_metrics`` for binary SVM tasks."""

    if mode not in {"train", "valid", "test"}:
        raise ValueError(f"Unsupported metric mode: {mode}")
    results: dict[str, dict[str, float]] = {}
    thresholds: dict[str, float] = {}

    for task_name, classes in task_info.items():
        if int(classes) != 2:
            raise ValueError(f"rawscore_v2 supports binary tasks only; {task_name} has {classes}")
        logits = _as_numpy(logits_dict[task_name])
        labels = _as_numpy(labels_dict[task_name]).reshape(-1).astype(np.int64, copy=False)
        valid = labels != -1
        labels = labels[valid]
        scores = _positive_scores(logits[valid])
        if labels.size == 0:
            raise ValueError(f"No usable labels for {task_name}")

        if mode == "train" and set(np.unique(labels)) != {0, 1}:
            # Training metrics are diagnostics only.  Avoid aborting a valid
            # online update when a sampled mini-batch happens to be single-class.
            predictions = (scores >= 0.0).astype(np.int64)
            tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
            results[task_name] = {
                "mcc": float(matthews_corrcoef(labels, predictions)),
                "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
                "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
                "f1_score": float(f1_score(labels, predictions, zero_division=0)),
                "roc_auc": 0.5,
                "pr_auc": float(np.mean(labels == 1)),
            }
            continue

        _require_binary_split(labels, task_name=task_name, mode=mode)
        if mode == "valid":
            selected_threshold, selected_f1 = search_best_raw_threshold(labels, scores)
            task_metrics = _classification_metrics(labels, scores, selected_threshold)
            task_metrics["best_threshold"] = selected_threshold
            task_metrics["f1_score"] = selected_f1
            thresholds[task_name] = selected_threshold
        elif mode == "test":
            if threshold is None or task_name not in threshold:
                raise ValueError(f"Missing validation threshold for test task {task_name}")
            selected_threshold = float(threshold[task_name])
            task_metrics = _classification_metrics(labels, scores, selected_threshold)
            task_metrics["threshold"] = selected_threshold
        else:
            task_metrics = _classification_metrics(labels, scores, 0.0)
        results[task_name] = task_metrics

    return (results, thresholds) if mode == "valid" else results


def assert_rawscore_checkpoint(payload: object) -> None:
    if not isinstance(payload, dict) or "model" not in payload or "threshold" not in payload:
        raise ValueError("Invalid SVM checkpoint payload")
    model = payload["model"]
    if getattr(model, "svm_protocol_", None) != SVM_PROTOCOL:
        raise ValueError("External rawscore_v2 evaluation requires a rawscore_v2 SVM checkpoint")
    thresholds = payload["threshold"]
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError("rawscore_v2 checkpoint has no validation thresholds")
    if not all(np.isfinite(float(value)) for value in thresholds.values()):
        raise ValueError("rawscore_v2 checkpoint contains a non-finite threshold")
