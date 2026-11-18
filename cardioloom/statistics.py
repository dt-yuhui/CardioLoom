import numpy as np

class RankedBinary:
    """Weighted full-split AUROC and trapezoidal PR-AUC, with ties handled exactly.

    Sorting is cached once, so bootstrap never re-sorts the prediction vectors.
    Weighting by a patient's bootstrap multiplicity equals physically duplicating
    every ECG belonging to that patient, preserving the ECG-level estimand.
    """
    def __init__(self, labels, scores):
        self.order = np.argsort(-np.asarray(scores), kind="stable")
        self.y = np.asarray(labels, dtype=np.float64)[self.order]
        s = np.asarray(scores)[self.order]
        self.ends = np.r_[np.flatnonzero(np.diff(s)), len(s)-1]

    def metrics(self, weights):
        w = np.asarray(weights, dtype=np.float64)[self.order]
        tp = np.cumsum(w*self.y)[self.ends]
        fp = np.cumsum(w*(1-self.y))[self.ends]
        # Remove score thresholds supported only by zero-weight observations.
        keep = np.diff(np.r_[0., tp+fp]) > 0
        tp, fp = tp[keep], fp[keep]
        if not len(tp) or tp[-1] <= 0 or fp[-1] <= 0:
            return np.asarray([np.nan, np.nan])
        recall = np.r_[0., tp/tp[-1]]
        precision = np.r_[1., tp/(tp+fp)]
        fpr = np.r_[0., fp/fp[-1]]
        return np.asarray([np.trapz(precision, recall), np.trapz(recall, fpr)])

def paired_bootstrap(labels, scores, clusters, *, repeats, seed, universe_clusters=None):
    labels = np.asarray(labels)
    universe = np.unique(clusters if universe_clusters is None else universe_clusters)
    cluster_index = np.searchsorted(universe, clusters)
    if not np.array_equal(universe[cluster_index],clusters):
        raise ValueError("Cluster outside bootstrap universe")
    n = len(universe)
    ranked = [RankedBinary(labels, s) for s in scores]
    observed = np.asarray([r.metrics(np.ones(len(labels))) for r in ranked])
    if not np.isfinite(observed).all():
        raise ValueError("Both classes are required for observed ranking metrics")
    rng = np.random.default_rng(seed)
    samples = np.empty((repeats, len(scores), 2))
    for b in range(repeats):
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
        weights = counts[cluster_index]
        samples[b] = [r.metrics(weights) for r in ranked]
    return observed, samples

def bh_adjust(values):
    p = np.asarray(values, dtype=float)
    if not len(p):
        return []
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order]*len(p)/np.arange(1, len(p)+1))[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(adjusted, 1.)
    return result.tolist()
