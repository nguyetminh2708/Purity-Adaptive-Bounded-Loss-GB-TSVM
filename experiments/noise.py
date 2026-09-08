"""Tiem nhieu nhan - 4 kieu, CHI ap dung cho tap huan luyen (test giu nhan sach).

Kieu nhieu (theo taxonomy Frenay & Verleysen, IEEE TNNLS 2014):
  symmetric  - NCAR: lat nhan ngau nhien deu, ti le `rate`.
  asymmetric - NAR:  chi lat mot chieu (+1 -> -1), ti le `rate` tren lop +1.
  boundary   - NNAR: uu tien lat diem gan bien quyet dinh (margin nho nhat).
  far        - outlier sai nhan: lat cac diem XA tam lop cua chinh no nhat.

Moi ham tra ve (y_noisy, flipped_mask) - mask de doi chieu ground truth
trong thi nghiem "lambda_k co tuong quan voi ti le nhan lat that khong".
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression

NOISE_KINDS = ("symmetric", "asymmetric", "boundary", "far")


def _flip(y, pick):
    y2 = y.copy(); y2[pick] = -y2[pick]
    m = np.zeros(len(y), bool); m[pick] = True
    return y2, m


def inject(X: np.ndarray, y: np.ndarray, rate: float, kind: str = "symmetric",
           seed: int = 0):
    """y in {-1,+1}. rate trong [0, 0.5]. Tra ve (y_noisy, flipped_mask)."""
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
        # margin |f(x)| nho nhat = gan bien; lay ngau nhien trong nhom 2*n_flip gan nhat
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
