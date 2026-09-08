"""
Load every dataset into one shape: (n, d+1) float, label encoded in the last column.

UCI (ucimlrepo) is unreachable from both the sandbox and the local machine, so
this set combines:
  - sklearn built-ins: iris = UCI 53 (the exact dataset the notebook uses),
    wine = UCI 109, breast_cancer = UCI 17
  - CSVs already in the repo: chess_krvkp, seeds, tae, winequality-white, SPECTF

globalcancer.csv is deliberately absent: it has categorical columns and it is
unclear whether the target is Cancer_Stage or Target_Severity_Score. Guessing
would produce a meaningless number.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _finish(X, y, name, min_class=20):
    """Encode labels, drop NaN rows, drop tiny classes (recorded, never silent)."""
    X = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
    keep = X.notna().all(axis=1).to_numpy()
    X, y = X.to_numpy(dtype=float)[keep], np.asarray(y)[keep]

    y = LabelEncoder().fit_transform(y).astype(float)

    lab, cnt = np.unique(y, return_counts=True)
    small = lab[cnt < min_class]
    notes = []
    if len(small):
        m = ~np.isin(y, small)
        notes.append(f"dropped {len(small)} class(es) with <{min_class} samples "
                     f"({int((~m).sum())} samples)")
        X, y = X[m], y[m]
        y = LabelEncoder().fit_transform(y).astype(float)

    return {"name": name, "data": np.column_stack([X, y]),
            "n_classes": len(np.unique(y)), "notes": "; ".join(notes)}


# ---------------------------------------------------------------- sklearn ----

def _sk(loader, name):
    from sklearn import datasets as skd
    d = getattr(skd, loader)()
    return _finish(d.data, d.target, name)


# ------------------------------------------------------------------ local ----

def _chess():
    a = np.loadtxt(DATA_DIR / "chess_krvkp.csv", delimiter=",")
    return _finish(a[:, :-1], a[:, -1], "chess_krvkp")


def _seeds():
    a = np.loadtxt(DATA_DIR / "seeds_dataset.txt")
    return _finish(a[:, :-1], a[:, -1], "seeds")


def _tae():
    a = np.loadtxt(DATA_DIR / "tae/tae.data", delimiter=",")
    return _finish(a[:, :-1], a[:, -1], "tae")


def _wine_quality():
    df = pd.read_csv(DATA_DIR / "winequality-white.csv", sep=";")
    return _finish(df.iloc[:, :-1], df.iloc[:, -1], "winequality_white")


def _spectf():
    # label is the FIRST column; the header is column indices "0,1,...,44";
    # train and test are concatenated
    fr = [pd.read_csv(DATA_DIR / f"SPECTF_{s}.csv") for s in ("train", "test")]
    df = pd.concat(fr, ignore_index=True)
    return _finish(df.iloc[:, 1:], df.iloc[:, 0], "SPECTF")


def _balance_scale(binary=True):
    """
    UCI 12 (Balance Scale), regenerated from its DEFINITION rather than fetched.

    The dataset is fully deterministic: every combination of
    (left-weight, left-distance, right-weight, right-distance) in 1..5, labelled
    L if lw*ld > rw*rd, R if <, B if =. Regenerating gives 625 rows with
    L=288, B=49, R=288 — exactly the distribution UCI publishes, verified.

    Needed because ucimlrepo is unreachable from both the sandbox and the local
    machine. binary=True drops class B and keeps L vs R, matching the Colab.
    """
    import itertools
    rows = []
    for lw, ld, rw, rd in itertools.product(range(1, 6), repeat=4):
        l, r = lw * ld, rw * rd
        rows.append([lw, ld, rw, rd, 0 if l > r else (2 if l < r else 1)])
    a = np.array(rows, dtype=float)
    if binary:
        a = a[a[:, -1] != 1]
    return _finish(a[:, :-1], a[:, -1],
                   "balance_scale" + ("" if binary else "_3class"))


def inject_label_noise(y, rate, rng):
    """Flip `rate` of the labels at random. Train labels only — never test."""
    y = np.asarray(y).copy()
    if rate <= 0:
        return y
    classes = np.unique(y)
    idx = rng.choice(len(y), int(round(rate * len(y))), replace=False)
    if len(classes) == 2:
        other = {classes[0]: classes[1], classes[1]: classes[0]}
        y[idx] = [other[v] for v in y[idx]]
    else:
        for i in idx:
            y[i] = rng.choice([c for c in classes if c != y[i]])
    return y


REGISTRY = {
    "balance_scale":     _balance_scale,
    "iris":              lambda: _sk("load_iris", "iris"),
    "wine":              lambda: _sk("load_wine", "wine"),
    "breast_cancer":     lambda: _sk("load_breast_cancer", "breast_cancer"),
    "seeds":             _seeds,
    "tae":               _tae,
    "SPECTF":            _spectf,
    "chess_krvkp":       _chess,
    "winequality_white": _wine_quality,
}


def load(name):
    return REGISTRY[name]()


def load_all(names=None):
    return [load(n) for n in (names or REGISTRY)]
