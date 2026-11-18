"""Evaluate a single CardioLoom endpoint using a validation-selected threshold."""
from __future__ import annotations
import argparse
import json
import math
import pandas as pd
import numpy as np
from cardioloom.models.corrected_svm import search_best_raw_threshold
from cardioloom.statistics import RankedBinary

def select_threshold(labels: np.ndarray, scores: np.ndarray, target: float) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Threshold selection requires aligned binary validation data")
    order = np.argsort(-scores, kind="stable")
    y = labels[order]
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
    tp = np.cumsum(y == 1)[ends]
    fp = np.cumsum(y == 0)[ends]
    n_positive = int((labels == 1).sum())
    n_negative = int((labels == 0).sum())
    specificity = 1.0 - fp / n_negative
    sensitivity = tp / n_positive
    valid = specificity >= target
    # The threshold above the maximum score always predicts no positives and
    # guarantees specificity 1.0; it is used only if no observed-score
    # candidate satisfies the target.
    if not valid.any():
        return {
            "threshold": float(np.nextafter(scores.max(), np.inf)),
            "validation_sensitivity": 0.0,
            "validation_specificity": 1.0,
        }
    best_sensitivity = sensitivity[valid].max()
    eligible = np.flatnonzero(valid & (sensitivity == best_sensitivity))
    # Scores are descending, so the last eligible index is the smallest
    # threshold among equal-sensitivity candidates.
    index = int(eligible[-1])
    return {
        "threshold": float(sorted_scores[ends[index]]),
        "validation_sensitivity": float(sensitivity[index]),
        "validation_specificity": float(specificity[index]),
    }


def test_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, object]:
    prediction = scores >= threshold
    tp = int(((labels == 1) & prediction).sum())
    fp = int(((labels == 0) & prediction).sum())
    tn = int(((labels == 0) & ~prediction).sum())
    fn = int(((labels == 1) & ~prediction).sum())
    n = len(labels)
    sensitivity = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    ppv = tp / (tp + fp) if tp + fp else None
    return {
        "n_ecgs": n,
        "positive_ecgs": int((labels == 1).sum()),
        "prevalence": float((labels == 1).mean()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": float(sensitivity),
        "test_specificity": float(specificity),
        "ppv": ppv,
        "nns": n / tp if tp else None,
        # Historical manuscript name retained in the output contract. This is
        # N/FP (screens per false positive), not a causal number needed to harm.
        "historical_nnh_n_over_fp": n / fp if fp else None,
        "false_positives_per_1000_ecgs": 1000 * fp / n,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('validation')
    parser.add_argument('test')
    parser.add_argument('--rule', choices=['f1', 'spec99'], default='f1')
    args = parser.parse_args()
    def load(path):
        frame = pd.read_csv(path, usecols=['label','score'])
        if set(frame.label.unique()) != {0,1} or not np.isfinite(frame.score).all():
            raise ValueError('Both binary classes and finite scores are required')
        return frame.label.to_numpy(), frame.score.to_numpy()
    vy, vs = load(args.validation)
    y, scores = load(args.test)
    threshold = (select_threshold(vy, vs, .99)['threshold'] if args.rule == 'spec99'
                 else search_best_raw_threshold(vy, vs)[0])
    auprc, auroc = RankedBinary(y, scores).metrics(np.ones(len(y)))
    print(json.dumps(dict(AUPRC=auprc, AUROC=auroc, threshold=threshold,
                          **test_metrics(y,scores,threshold)), indent=2, allow_nan=False))

if __name__=='__main__':
    main()
