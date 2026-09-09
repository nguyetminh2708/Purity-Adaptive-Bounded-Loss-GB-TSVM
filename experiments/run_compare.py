"""Compare four granular-ball models under label noise.

    hard   GBTSVM, purity=1, hinge (original)
    binom  GBTSVM, binomial ball generation (oracle eta = true rate), hinge
    wave   Wave-GBTSVM on purity=1 balls
    wave2  Wave-GBTSVM on binomial balls (two-stage: adaptive stopping + bounded loss)

    python experiments/run_compare.py [--datasets ...] [--seeds N] [--rates ...]

Writes results/<ts>_compare.csv and logs the run in results/RUNS.md.
"""
from __future__ import annotations
import sys, pathlib, argparse, warnings
warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from models.svm_cvxopt import SVM_CVXOPT
from config import DATASETS_BINARY
from data.datasets import load, inject_label_noise
from gb.balls import gen_balls, to_matrix, viable
from models.registry import fit_binary, decide, Degenerate
from models.wave_gbtsvm import WaveGBTSVM
from models.kernel import RBFMap
from runlog import save_run

GB = dict(d1=0.1, d2=0.1, eps1=0.05, eps2=0.05)


def _arr(b):
    return (np.array([x.center for x in b]), np.array([x.radius for x in b]),
            np.array([x.label for x in b]), np.array([x.purity for x in b]),
            np.array([x.n for x in b]))


def _gbtsvm(balls, Xte, yte):
    if not viable(balls):
        return np.nan
    try:
        return accuracy_score(yte, decide(fit_binary("GBTSVM", GB, to_matrix(balls)), Xte))
    except Degenerate:
        return np.nan


def _wave(balls, Xte, yte, wave_kw):
    if not viable(balls):
        return np.nan
    C, r, yb, p, n = _arr(balls)
    m = WaveGBTSVM(**wave_kw).fit(C, r, yb, purity=p, size=n)
    return accuracy_score(yte, m.predict(Xte))


def run(name, seeds, rates, wave_kw, steps, kernel):
    ds = load(name); X, y = ds["data"][:, :-1], ds["data"][:, -1]
    cls = np.unique(y); to_pm = lambda v: np.where(v == cls[0], 1.0, -1.0)
    rows = []
    for rate in rates:
        acc = {"svm": [], "hard": [], "binom": [], "wave": [], "wave2": []}
        for seed in range(seeds):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
                ytr = inject_label_noise(y[tr], rate, np.random.default_rng(seed))
                if len(np.unique(ytr)) < 2:
                    continue
                sc = MinMaxScaler((-1, 1)).fit(X[tr])
                Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
                yp, yte = to_pm(ytr), to_pm(y[te])
                acc["svm"].append(accuracy_score(yte, SVM_CVXOPT(kernel=kernel).fit(Xtr, yp).predict(Xte)))
                if kernel == "rbf":
                    km = RBFMap(seed=seed).fit(Xtr); Ftr, Fte = km.transform(Xtr), km.transform(Xte)
                else:
                    Ftr, Fte = Xtr, Xte
                sub = np.column_stack([Ftr, yp])
                bh = gen_balls(sub, pur=1.0, delbals=1, seed=seed)
                bo = gen_balls(sub, pur=1.0, delbals=1, seed=seed, eta=max(rate, 1e-9), alpha=0.05)
                acc["hard"].append(_gbtsvm(bh, Fte, yte))
                acc["binom"].append(_gbtsvm(bo, Fte, yte))
                acc["wave"].append(_wave(bh, Fte, yte, wave_kw))
                acc["wave2"].append(_wave(bo, Fte, yte, wave_kw))
        for model, v in acc.items():
            rows.append(dict(dataset=name, rate=rate, model=model,
                             acc=float(np.nanmean(v)), acc_std=float(np.nanstd(v))))
        print(f"  {name:13s} rate={rate:.1f}  " +
              "  ".join(f"{m}={np.nanmean(acc[m]):.3f}" for m in acc), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS_BINARY)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rates", nargs="*", type=float, default=[0.0, 0.2, 0.4])
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--lam0", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--kernel", choices=["linear", "rbf"], default="linear")
    args = ap.parse_args()
    wave_kw = dict(c1=args.C, c2=args.C, lam0=args.lam0, kappa=args.kappa,
                   adaptive_lambda=True, steps=args.steps)
    rows = []
    for name in args.datasets:
        try:
            rows += run(name, args.seeds, args.rates, wave_kw, args.steps, args.kernel)
        except Exception as e:
            print(f"  skip {name}: {type(e).__name__}: {str(e)[:70]}")
    fn = save_run(pd.DataFrame(rows), f"compare_{args.kernel}",
                  dict(kernel=args.kernel, datasets=",".join(args.datasets), seeds=args.seeds,
                       rates=args.rates, C=args.C, lam0=args.lam0, kappa=args.kappa,
                       steps=args.steps))
    print(f"saved {fn} (+ logged in results/RUNS.md)")


if __name__ == "__main__":
    main()
