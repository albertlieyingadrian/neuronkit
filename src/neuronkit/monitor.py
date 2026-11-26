"""Monitors in the neuron basis: sparse linear read-outs whose features are individual neurons.

A residual-stream probe reads one direction made of every coordinate at once, so asking "why did it fire?" has no
short answer. A neuron monitor is an L1-penalised logistic regression on neuron activations (or on their tail
indicators): the few neurons it keeps can each be described, traced with `subcircuit`, and ablated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from neuronkit.core import Neuron


def tail_features(X: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """+1 where a neuron fires above its upper percentile, -1 below its lower one, else 0 (NaN counts as 0)."""
    X = np.nan_to_num(X, nan=0.0)
    return (X > hi).astype(np.float32) - (X < lo).astype(np.float32)


@dataclass
class NeuronMonitor:
    neurons: list[Neuron]
    mu: np.ndarray
    sd: np.ndarray
    w: np.ndarray
    b: float
    C: float
    features: str = "raw"
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    cv: dict = field(default_factory=dict)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.features == "tails":
            assert self.lo is not None and self.hi is not None
            X = tail_features(X, self.lo, self.hi)
        return (np.nan_to_num(X) - self.mu) / self.sd

    def score(self, X: np.ndarray) -> np.ndarray:
        """Log-odds of the positive class."""
        return self.transform(X) @ self.w + self.b

    def top(self, k: int = 10) -> list[tuple[Neuron, float]]:
        """The k neurons with the largest |weight| (weights are per standardised feature)."""
        order = np.argsort(-np.abs(self.w))[:k]
        return [(self.neurons[i], float(self.w[i])) for i in order if self.w[i] != 0]

    @property
    def n_used(self) -> int:
        return int((self.w != 0).sum())


def _fit(Xs: np.ndarray, y: np.ndarray, C: float, seed: int):
    import sklearn
    from sklearn.linear_model import LogisticRegression

    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    l1 = {"l1_ratio": 1.0} if (major, minor) >= (1, 8) else {"penalty": "l1"}
    return LogisticRegression(C=C, solver="liblinear", max_iter=5000, random_state=seed, **l1).fit(Xs, y)


def neuron_monitor(
    X: np.ndarray,
    y: Sequence[int],
    neurons: Sequence[Neuron],
    groups: Sequence | None = None,
    Cs: Sequence[float] = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0),
    features: str = "raw",
    lo: np.ndarray | None = None,
    hi: np.ndarray | None = None,
    k: int = 5,
    seed: int = 0,
) -> NeuronMonitor:
    """Fit a sparse neuron monitor. C is chosen by k-fold CV AUROC (grouped when `groups` is given), then refit on all
    rows.

    X: (rows, neurons) activations at one reading (e.g. the last prompt token), columns matching `neurons`.
    features: "raw" (standardised activations) or "tails" (+1/-1/0 tail indicators from `lo` and `hi`, each
    (neurons,), e.g. a `Percentiles`' p3 and p97 for those neurons).
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    y = np.asarray(y)
    X = np.asarray(X, dtype=np.float64)
    if features == "tails":
        if lo is None or hi is None:
            raise ValueError("tail features need lo and hi thresholds")
        Xf = tail_features(X, lo, hi).astype(np.float64)
    elif features == "raw":
        Xf = np.nan_to_num(X)
    else:
        raise ValueError("features is 'raw' or 'tails'")
    if groups is not None and len(set(groups)) >= k:
        splits = list(GroupKFold(n_splits=k).split(Xf, y, groups))
    else:
        splits = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(Xf, y))
    cv = {}
    for C in Cs:
        oof = np.zeros(len(y))
        for tr, te in splits:
            mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-6
            clf = _fit((Xf[tr] - mu) / sd, y[tr], C, seed)
            oof[te] = clf.decision_function((Xf[te] - mu) / sd)
        cv[C] = float(roc_auc_score(y, oof)) if len(set(y)) > 1 else float("nan")
    best = max(Cs, key=lambda c: (cv[c], -c))
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    clf = _fit((Xf - mu) / sd, y, best, seed)
    return NeuronMonitor(list(neurons), mu, sd, clf.coef_[0], float(clf.intercept_[0]), best, features, lo, hi, cv)
