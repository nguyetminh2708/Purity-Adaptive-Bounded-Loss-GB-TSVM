"""Nested CV for Wave-GBTSVM: inner CV picks C x lam0 x kappa, outer CV scores.

Noise is injected into training labels only; inner CV tunes against the noisy
train labels (no clean labels at tuning time); the outer test fold stays clean.

    python experiments/tune_wave.py [--datasets ...] [--rates ...] [--kernel linear|rbf]

Writes results/<ts>_tunewave_<kernel>.csv and logs to results/RUNS.md.
"""
from __future__ import annotations
import sys, pathlib, argparse, warnings
from itertools import product
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
from gb.balls import gen_balls, viable
from models.wave_gbtsvm import WaveGBTSVM
from models.kernel import RBFMap
from runlog import save_run

GRID = {"C": [1.0, 10.0], "lam0": [0.5, 1.0, 2.0, 4.0], "kappa": [0.0, 2.0, 4.0]}
COMBOS = [dict(zip(GRID, v)) for v in product(*GRID.values())]


def _arr(b):
    return (np.array([x.center for x in b]), np.array([x.radius for x in b]),
            np.array([x.label for x in b]), np.array([x.purity for x in b]),
            np.array([x.n for x in b]))


def _wave_fit_score(Ftr, ytr, Fva, yva, cfg, seed, steps):
    b = gen_balls(np.column_stack([Ftr, ytr]), pur=1.0, delbals=1, seed=seed)
    if not viable(b):
        return np.nan
    C, r, yb, p, n = _arr(b)
    m = WaveGBTSVM(c1=cfg["C"], c2=cfg["C"], lam0=cfg["lam0"], kappa=cfg["kappa"],
                   adaptive_lambda=cfg["kappa"] > 0, steps=steps).fit(C, r, yb, purity=p, size=n)
    return accuracy_score(yva, m.predict(Fva))


def nested(name, rate, kernel, seed=0, n_out=5, n_in=3, steps=400):
    ds = load(name); X, y = ds["data"][:, :-1], ds["data"][:, -1]
    cls = np.unique(y); to_pm = lambda v: np.where(v == cls[0], 1.0, -1.0)
    tuned, default, picks = [], [], []
    outer = StratifiedKFold(n_out, shuffle=True, random_state=seed)
    for tr, te in outer.split(X, y):
        ytr = to_pm(inject_label_noise(y[tr], rate, np.random.default_rng(seed)))
        yte = to_pm(y[te])
        sc = MinMaxScaler((-1, 1)).fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        if kernel == "rbf":
            km = RBFMap(seed=seed).fit(Xtr); Ftr, Fte = km.transform(Xtr), km.transform(Xte)
        else:
            Ftr, Fte = Xtr, Xte
        # inner CV to pick a combo (scored on noisy inner-val)
        inner = StratifiedKFold(n_in, shuffle=True, random_state=seed + 1)
        best = (-np.inf, COMBOS[0])
        for cfg in COMBOS:
            sc_in = []
            for itr, iva in inner.split(Ftr, ytr):
                if len(np.unique(ytr[itr])) < 2:
                    continue
                sc_in.append(_wave_fit_score(Ftr[itr], ytr[itr], Ftr[iva], ytr[iva], cfg, seed, steps))
            m = np.nanmean(sc_in) if sc_in else np.nan
            if m > best[0]:
                best = (m, cfg)
        picks.append(best[1])
        tuned.append(_wave_fit_score(Ftr, ytr, Fte, yte, best[1], seed, steps))
        default.append(_wave_fit_score(Ftr, ytr, Fte, yte, dict(C=1.0, lam0=1.0, kappa=2.0), seed, steps))
    return np.nanmean(tuned), np.nanmean(default), picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["breast_cancer", "balance_scale", "sonar"])
    ap.add_argument("--rates", nargs="*", type=float, default=[0.2, 0.4])
    ap.add_argument("--kernel", choices=["linear", "rbf"], default="linear")
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()
    rows = []
    print(f"{'dataset':13s}{'rate':>6}{'Wave-tuned':>12}{'Wave-default':>13}   top pick")
    for name in args.datasets:
        for rate in args.rates:
            t, d, picks = nested(name, rate, args.kernel, steps=args.steps)
            from collections import Counter
            top = Counter((p["C"], p["lam0"], p["kappa"]) for p in picks).most_common(1)[0][0]
            rows.append(dict(dataset=name, rate=rate, wave_tuned=t, wave_default=d,
                             top_C=top[0], top_lam0=top[1], top_kappa=top[2]))
            print(f"{name:13s}{rate:>6.1f}{t:>12.3f}{d:>13.3f}   C={top[0]} lam0={top[1]} kappa={top[2]}", flush=True)
    fn = save_run(pd.DataFrame(rows), f"tunewave_{args.kernel}",
                  dict(kernel=args.kernel, datasets=",".join(args.datasets), rates=args.rates,
                       grid="C[1,10] lam0[.5,1,2,4] kappa[0,2,4]", steps=args.steps))
    print(f"saved {fn}")


if __name__ == "__main__":
    main()
