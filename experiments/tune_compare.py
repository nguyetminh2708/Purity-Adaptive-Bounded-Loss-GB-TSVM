"""Nested CV for all five models, tuned per dataset.

    svm    self-coded C-SVM (cvxopt), tune C
    hard   GBTSVM, purity=1 balls, tune d1=d2
    binom  GBTSVM, binomial-stop balls, tune eta and d1=d2
    wave   Wave-GBTSVM on purity=1 balls, tune C, lam0, kappa
    wave2  Wave-GBTSVM on binomial-stop balls, tune eta, C, lam0, kappa

Outer 5-fold scores on the clean test fold; inner 3-fold picks params on the
noisy train fold only. Noise goes into train labels, never test. Balls are
generated once per inner fold and reused across the parameter grid.

    python experiments/tune_compare.py [--datasets ...] [--rates ...] --kernel linear|rbf

Rows are appended to the out CSV as each (dataset, rate) finishes, so a partial
run keeps its progress. The full frame is also logged via results/RUNS.md.
"""
from __future__ import annotations
import sys, pathlib, argparse, warnings, time
from itertools import product
from collections import Counter
warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from config import DATASETS_BINARY
from data.datasets import load, inject_label_noise
from gb.balls import gen_balls, to_matrix, viable
from models.registry import fit_binary, decide, Degenerate
from models.svm_cvxopt import SVM_CVXOPT
from models.wave_gbtsvm import WaveGBTSVM
from models.kernel import RBFMap
from runlog import save_run

SVM_C = [1.0, 10.0]
D_GRID = [0.01, 0.1, 1.0]
ETAS = [0.2, 0.3, 0.4]
WAVE = list(product([1.0, 10.0], [0.5, 1.0, 2.0, 4.0], [0.0, 2.0, 4.0]))
WAVE2 = list(product([10.0], [1.0, 2.0], [2.0, 4.0]))


def _arr(b):
    return (np.array([x.center for x in b]), np.array([x.radius for x in b]),
            np.array([x.label for x in b]), np.array([x.purity for x in b]),
            np.array([x.n for x in b]))


def _svm(Xtr, ytr, Xva, yva, C, kernel):
    return accuracy_score(yva, SVM_CVXOPT(C=C, kernel=kernel).fit(Xtr, ytr).predict(Xva))


def _gb(balls, Fva, yva, d):
    if not viable(balls):
        return np.nan
    try:
        W = fit_binary("GBTSVM", dict(d1=d, d2=d, eps1=0.05, eps2=0.05), to_matrix(balls))
        return accuracy_score(yva, decide(W, Fva))
    except Degenerate:
        return np.nan


def _wave(balls, Fva, yva, cfg, steps):
    if not viable(balls):
        return np.nan
    C, r, yb, p, n = _arr(balls)
    m = WaveGBTSVM(c1=cfg[0], c2=cfg[0], lam0=cfg[1], kappa=cfg[2],
                   adaptive_lambda=cfg[2] > 0, steps=steps).fit(C, r, yb, purity=p, size=n)
    return accuracy_score(yva, m.predict(Fva))


def _best(scores):
    return max(scores, key=lambda k: (np.nanmean(scores[k]) if np.any(~np.isnan(scores[k])) else -np.inf))


def nested(name, rate, kernel, seed=0, n_out=5, n_in=3, steps=300):
    ds = load(name); X, y = ds["data"][:, :-1], ds["data"][:, -1]
    cls = np.unique(y); to_pm = lambda v: np.where(v == cls[0], 1.0, -1.0)
    acc = {m: [] for m in ("svm", "hard", "binom", "wave", "wave2")}
    picks = {m: [] for m in acc}
    outer = StratifiedKFold(n_out, shuffle=True, random_state=seed)
    for tr, te in outer.split(X, y):
        yp = to_pm(inject_label_noise(y[tr], rate, np.random.default_rng(seed)))
        yte = to_pm(y[te])
        sc = MinMaxScaler((-1, 1)).fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        if kernel == "rbf":
            km = RBFMap(seed=seed).fit(Xtr); Ftr, Fte = km.transform(Xtr), km.transform(Xte)
        else:
            Ftr, Fte = Xtr, Xte

        s_svm = {c: [] for c in SVM_C}
        s_hard = {d: [] for d in D_GRID}
        s_wave = {c: [] for c in WAVE}
        s_binom = {(e, d): [] for e in ETAS for d in D_GRID}
        s_wave2 = {(e, c): [] for e in ETAS for c in WAVE2}
        inner = StratifiedKFold(n_in, shuffle=True, random_state=seed + 1)
        for itr, iva in inner.split(Ftr, yp):
            if len(np.unique(yp[itr])) < 2:
                continue
            for c in SVM_C:
                s_svm[c].append(_svm(Xtr[itr], yp[itr], Xtr[iva], yp[iva], c, kernel))
            sub = np.column_stack([Ftr[itr], yp[itr]])
            bp = gen_balls(sub, pur=1.0, delbals=1, seed=seed)
            for d in D_GRID:
                s_hard[d].append(_gb(bp, Ftr[iva], yp[iva], d))
            for c in WAVE:
                s_wave[c].append(_wave(bp, Ftr[iva], yp[iva], c, steps))
            for e in ETAS:
                be = gen_balls(sub, pur=1.0, delbals=1, seed=seed, eta=e, alpha=0.05)
                for d in D_GRID:
                    s_binom[(e, d)].append(_gb(be, Ftr[iva], yp[iva], d))
                for c in WAVE2:
                    s_wave2[(e, c)].append(_wave(be, Ftr[iva], yp[iva], c, steps))

        b_svm, b_hard, b_wave = _best(s_svm), _best(s_hard), _best(s_wave)
        b_binom, b_wave2 = _best(s_binom), _best(s_wave2)
        picks["svm"].append(b_svm); picks["hard"].append(b_hard); picks["wave"].append(b_wave)
        picks["binom"].append(b_binom); picks["wave2"].append(b_wave2)

        sub = np.column_stack([Ftr, yp])
        bp = gen_balls(sub, pur=1.0, delbals=1, seed=seed)
        be_b = gen_balls(sub, pur=1.0, delbals=1, seed=seed, eta=b_binom[0], alpha=0.05)
        be_w = gen_balls(sub, pur=1.0, delbals=1, seed=seed, eta=b_wave2[0], alpha=0.05)
        acc["svm"].append(_svm(Xtr, yp, Xte, yte, b_svm, kernel))
        acc["hard"].append(_gb(bp, Fte, yte, b_hard))
        acc["binom"].append(_gb(be_b, Fte, yte, b_binom[1]))
        acc["wave"].append(_wave(bp, Fte, yte, b_wave, steps))
        acc["wave2"].append(_wave(be_w, Fte, yte, b_wave2[1], steps))
    return acc, picks


def main():
    ap = argparse.ArgumentParser()
    ten = [d for d in DATASETS_BINARY if d != "chess_krvkp"]
    ap.add_argument("--datasets", nargs="*", default=ten)
    ap.add_argument("--rates", nargs="*", type=float, default=[0.2, 0.3, 0.4])
    ap.add_argument("--kernel", choices=["linear", "rbf"], default="linear")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or str(HERE.parent / "results" / f"_partial_tunecmp_{args.kernel}.csv")
    rows = []
    hdr = f"{'dataset':13s}{'rate':>5}" + "".join(f"{m:>8}" for m in ("svm", "hard", "binom", "wave", "wave2"))
    print(hdr, flush=True)
    for name in args.datasets:
        for rate in args.rates:
            t0 = time.time()
            try:
                acc, picks = nested(name, rate, args.kernel, steps=args.steps)
            except Exception as e:
                print(f"  skip {name} r={rate}: {type(e).__name__}: {str(e)[:60]}", flush=True)
                continue
            for m in acc:
                v = acc[m]
                top = Counter(map(tuple, ([p] if np.isscalar(p) else p for p in picks[m]))).most_common(1)
                rows.append(dict(dataset=name, rate=rate, model=m,
                                 acc=float(np.nanmean(v)), acc_std=float(np.nanstd(v)),
                                 pick=str(top[0][0]) if top else ""))
            pd.DataFrame(rows).to_csv(out, index=False)
            means = "".join(f"{np.nanmean(acc[m]):>8.3f}" for m in ("svm", "hard", "binom", "wave", "wave2"))
            print(f"{name:13s}{rate:>5.1f}{means}   ({time.time()-t0:.0f}s)", flush=True)
    fn = save_run(pd.DataFrame(rows), f"tunecmp_{args.kernel}",
                  dict(kernel=args.kernel, datasets=",".join(args.datasets), rates=args.rates,
                       nested="outer5 inner3", steps=args.steps,
                       grids="svmC[1,10] d[.01,.1,1] eta[.2,.3,.4] wave24 wave2_4"))
    print(f"saved {fn}")


if __name__ == "__main__":
    main()
