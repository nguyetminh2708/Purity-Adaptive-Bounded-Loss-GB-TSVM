"""Label-noise injection, four kinds. Applied to training labels only.

symmetric  random flip (NCAR)
asymmetric one-directional flip on the +1 class (NAR)
boundary   flip points nearest the decision boundary (NNAR)
far        flip points farthest from their own class centre (outliers)

Each returns (y_noisy, flipped_mask); y in {-1, +1}.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression

NOISE_KINDS = ("symmetric", "asymmetric", "boundary", "far")


def _flip(y, pick):
    y2 = y.copy(); y2[pick] = -y2[pick]
    m = np.zeros(len(y), bool); m[pick] = True
    return y2, m


def inject(X, y, rate, kind="symmetric", seed=0):
    assert kind in NOISE_KINDS, kind
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    n_flip = int(round(rate * len(y)))
    if n_flip == 0:
        return y.copy(), np.zeros(len(y), bool)

    if kind == "symmetric":
        pick = rng.choice(len(y), size=n_flip, replace=False)
    elif kind == "asymmetric":
        pos = np.flatnonzero(y == 1)
        k = min(int(round(rate * len(pos))), len(pos))
        pick = rng.choice(pos, size=k, replace=False)
    elif kind == "boundary":
        f = LogisticRegression(max_iter=1000).fit(X, y).decision_function(X)
        cand = np.argsort(np.abs(f))[: min(2 * n_flip, len(y))]
        pick = rng.choice(cand, size=min(n_flip, len(cand)), replace=False)
    else:  # far
        d = np.empty(len(y))
        for c in (-1, 1):
            m = y == c
            d[m] = np.linalg.norm(X[m] - X[m].mean(axis=0), axis=1)
        cand = np.argsort(-d)[: min(2 * n_flip, len(y))]
        pick = rng.choice(cand, size=min(n_flip, len(cand)), replace=False)

    return _flip(y, pick)
