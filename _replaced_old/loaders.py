"""Nap du lieu chuan hoa cho toan bo thi nghiem.

Moi bo tra ve (X, y) voi:
  - X: da chuan hoa z-score (StandardScaler), dtype float
  - y: nhan {-1, +1}

Thu tu uu tien nguon:
  1) CSV cache trong data/raw/<name>.csv  (cot cuoi = nhan) — tai lap, khong can mang
  2) scikit-learn built-in (chi 'wdbc') — luon co offline
  3) OpenML fetch_openml — can mang; tu dong cache lai ra data/raw/ sau khi tai

De tai lap ket qua GD1, nen tai san 12 bo mot lan (chay `download_all`) roi commit
DANH SACH ten (khong commit CSV — da gitignore data/raw/).
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import DATA_RAW, DATASETS


def _binarize(y: np.ndarray) -> np.ndarray:
    """Ep nhan ve {-1,+1}: lop xuat hien nhieu thu 2 -> +1 (on dinh giua cac nguon)."""
    y = np.asarray(y).ravel()
    vals, counts = np.unique(y, return_counts=True)
    assert len(vals) == 2, f"khong phai nhi phan: {vals}"
    pos = vals[np.argmax(counts)]           # lop da so -> +1 (quy uoc co dinh)
    return np.where(y == pos, 1, -1).astype(int)


def _finalize(X, y):
    X = StandardScaler().fit_transform(np.asarray(X, float))
    return X, _binarize(y)


def _from_cache(name):
    f = DATA_RAW / f"{name}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    return _finalize(df.iloc[:, :-1].values, df.iloc[:, -1].values)


def _from_sklearn(spec):
    if spec.get("sklearn") == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer()
        return _finalize(d.data, d.target)
    return None


def _from_openml(spec):
    from sklearn.datasets import fetch_openml
    oid, ver = spec["openml"]
    d = fetch_openml(oid, version=ver, as_frame=True)
    X = pd.get_dummies(d.data, drop_first=True).astype(float)  # xu ly cot phan loai
    Xy = X.copy(); Xy["__target__"] = np.asarray(d.target)
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    Xy.to_csv(DATA_RAW / f"{spec['name']}.csv", index=False)   # cache lai
    return _finalize(X.values, d.target)


def load_dataset(name: str):
    """Nap 1 bo theo ten trong config.DATASETS."""
    spec = next((d for d in DATASETS if d["name"] == name), None)
    if spec is None:
        raise KeyError(f"khong co bo '{name}' trong config.DATASETS")
    for src in (_from_cache(name), _from_sklearn(spec)):
        if src is not None:
            return src
    return _from_openml(spec)      # nem loi ro rang neu khong co mang & khong cache


def available(offline_only: bool = False):
    """Danh sach bo co the nap ngay: cache co san, hoac sklearn built-in."""
    out = []
    for spec in DATASETS:
        if (DATA_RAW / f"{spec['name']}.csv").exists() or spec.get("sklearn"):
            out.append(spec["name"])
        elif not offline_only:
            out.append(spec["name"])
    return out


def download_all(verbose: bool = True):
    """Tai va cache toan bo 12 bo (can mang). Chay 1 lan tren may co internet."""
    ok, fail = [], []
    for spec in DATASETS:
        try:
            X, y = load_dataset(spec["name"])
            ok.append(spec["name"])
            if verbose:
                print(f"OK  {spec['name']:12s} {X.shape}  pos={int((y==1).sum())} neg={int((y==-1).sum())}")
        except Exception as e:
            fail.append(spec["name"])
            if verbose:
                print(f"FAIL {spec['name']:12s} {str(e)[:70]}")
    return ok, fail


if __name__ == "__main__":
    download_all()
