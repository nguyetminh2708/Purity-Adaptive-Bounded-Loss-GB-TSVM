"""
The fixed path: reuse the original QP solvers, own the decision rule here.

Why not the thin adapter originally planned: bug L5 sits INSIDE LGBTSVM.predict
(the broadcast (m,1) + (m,) -> (m,m)), so wrapping from the outside cannot fix
it. But fit is fine — w1, b1, w2, b2 are all valid after fitting.

Hence the split:
    - fit  (solving the two QPs)      -> reused verbatim from Sourcecode/class*.py
    - decision rule + voting          -> written here, one implementation for all
                                         three variants

This also disposes of:
    L5  LGBTSVM's wrong decision function
    L6  GBTSVM/PinGBTSVM returning (1,k) instead of (k,)
    A5  OvR returning a multi-label matrix instead of labels -> margin argmax
and removes six duplicated OvO/OvR implementations from the original code.

Sourcecode/class*.py is NOT modified — it still runs, for comparison.
"""

from itertools import combinations
import io
import contextlib

import numpy as np

from .baselines import classGBTSVM as _G
from .baselines import classLGBTSVM as _L
from .baselines import classPinGBTSVM as _P

# variant name -> (original binary class, the parameters it accepts)
BINARY = {
    "GBTSVM": (_G.GBTSVM, ("d1", "d2", "eps1", "eps2")),
    "LGBTSVM": (_L.LGBTSVM, ("d1", "d2", "d3", "d4")),
    "PinGBTSVM": (_P.PinGBTSVM, ("d1", "d2", "eps1", "eps2", "tau")),
}

PARAMS = {k: v[1] for k, v in BINARY.items()}


class Degenerate(Exception):
    """The QP gave no usable solution (too few balls, singular matrix, ...)."""


def fit_binary(kind, params, ball_mat):
    """
    ball_mat: (m, d+2) = [center(d) | radius | label +/-1]
    Returns (w1, b1, w2, b2), flattened — w is (d,), b is a float.
    """
    cls, keys = BINARY[kind]
    kw = {k: params[k] for k in keys}

    lab = ball_mat[:, -1]
    if (lab == 1).sum() < 1 or (lab != 1).sum() < 1:
        raise Degenerate("one of the two classes is missing")

    m = cls(**kw)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            m.fit(ball_mat)
    except Exception as e:                       # cvxopt / linalg blew up
        raise Degenerate(f"{type(e).__name__}: {e}") from e

    W = []
    for w, b in ((m.w1, m.b1), (m.w2, m.b2)):
        w = np.asarray(w, dtype=float).ravel()
        b = float(np.asarray(b, dtype=float).ravel()[0])
        if not (np.all(np.isfinite(w)) and np.isfinite(b)):
            raise Degenerate("solution contains NaN/Inf")
        W.append((w, b))

    d = ball_mat.shape[1] - 2
    if W[0][0].shape[0] != d or W[1][0].shape[0] != d:
        raise Degenerate(f"w has {W[0][0].shape[0]} dims, expected {d}")

    return W[0][0], W[0][1], W[1][0], W[1][1]


def margin(W, X):
    """|y2| - |y1|. Positive means leaning towards class +1. X: (k, d)."""
    w1, b1, w2, b2 = W
    y1 = X @ w1 + b1
    y2 = X @ w2 + b2
    return np.abs(y2) - np.abs(y1)


def decide(W, X):
    """Labels +/-1: assign to whichever hyperplane is CLOSER."""
    return np.where(margin(W, X) >= 0, 1, -1)


# ----------------------------------------------------------------------------
# multi-class
# ----------------------------------------------------------------------------

def fit_multiclass(kind, params, ball_mat, strategy):
    """
    ball_mat: (m, d+2) carrying the original K class labels.

    strategy 'ovo': one model per pair of classes
    strategy 'ovr': one model per class, against the rest

    Balls are generated ONCE over all K classes and only then split per
    subproblem — exactly like the baseline. Phase B reverses that order
    (see fit_multiclass_per_subproblem).
    """
    labels = np.unique(ball_mat[:, -1])
    if len(labels) < 2:
        raise Degenerate("balls collapsed to a single class")

    models = {}
    if strategy == "ovo":
        for ci, cj in combinations(labels, 2):
            bm = ball_mat[np.isin(ball_mat[:, -1], (ci, cj))].copy()
            bm[:, -1] = np.where(bm[:, -1] == ci, 1, -1)
            models[(ci, cj)] = fit_binary(kind, params, bm)
    elif strategy == "ovr":
        for c in labels:
            bm = ball_mat.copy()
            bm[:, -1] = np.where(ball_mat[:, -1] == c, 1, -1)
            models[c] = fit_binary(kind, params, bm)
    else:
        raise ValueError(strategy)

    return labels, models


def fit_multiclass_per_subproblem(kind, params, train_raw, strategy,
                                  pur, num, seed, gen, to_matrix, viable):
    """
    PHASE B — generate balls SEPARATELY for each subproblem.

    Differs from fit_multiclass in exactly one place: gen_balls is called inside
    the subproblem loop, on raw data that already carries binary labels, so
    purity is computed over 2 labels instead of K.

    Included in phase A so that phase B is only a flag flip; unused in A.
    """
    X, y = train_raw[:, :-1], train_raw[:, -1]
    labels = np.unique(y)
    models, ball_counts = {}, []

    def _one(sub_raw):
        balls = gen(sub_raw, pur=pur, delbals=num, seed=seed)
        if not viable(balls):
            raise Degenerate("subproblem has too few balls")
        ball_counts.append(len(balls))
        return fit_binary(kind, params, to_matrix(balls))

    if strategy == "ovo":
        for ci, cj in combinations(labels, 2):
            m = (y == ci) | (y == cj)
            models[(ci, cj)] = _one(
                np.column_stack([X[m], np.where(y[m] == ci, 1, -1)]))
    elif strategy == "ovr":
        for c in labels:
            models[c] = _one(np.column_stack([X, np.where(y == c, 1, -1)]))
    else:
        raise ValueError(strategy)

    return labels, models, ball_counts


def predict_multiclass(labels, models, X, strategy):
    """X: (k, d) raw features. Returns original labels, shape (k,)."""
    labels = np.asarray(labels)

    if strategy == "ovo":
        votes = np.zeros((len(X), len(labels)))
        idx = {c: i for i, c in enumerate(labels)}
        for (ci, cj), W in models.items():
            p = decide(W, X)
            votes[p == 1, idx[ci]] += 1
            votes[p == -1, idx[cj]] += 1
        return labels[votes.argmax(axis=1)]

    if strategy == "ovr":
        # A5: argmax over margins, replacing the original multi-label matrix
        S = np.column_stack([margin(models[c], X) for c in labels])
        return labels[S.argmax(axis=1)]

    raise ValueError(strategy)
