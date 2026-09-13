"""Step 3: re-run the GB family across all four noise kinds with the fixed code
(absolute-purity lambda, empty-side guard) and optional robust ball stats.

Compares against experiments A/B (old min-max lambda, plain ball stats) to see
whether the fixes change the picture -- especially whether robust balls + the new
lambda lift wave/wave2 or soften binom's asymmetric collapse.

    python experiments/exp_fixed.py [--robust] [--kernel linear|rbf]
"""
from __future__ import annotations
import sys, pathlib, argparse, warnings, time
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
from data.datasets import load
from gb.balls import gen_balls, to_matrix, arrays, viable
from models.registry import fit_binary, decide, Degenerate
from models.wave_gbtsvm import WaveGBTSVM, Degenerate as WDeg
from models.kernel import RBFMap
from runlog import save_run
from noise import inject, NOISE_KINDS
from tune_compare import _best

D_GRID = [0.01, 0.1, 1.0]
ETAS = [0.3, 0.4, 0.5]
WAVE = list(product([10.0], [0.5, 2.0, 4.0], [0.0, 2.0, 4.0]))


def _gb(balls, Fva, yva, d, robust):
    if not viable(balls):
        return np.nan
    try:
        return accuracy_score(yva, decide(fit_binary("GBTSVM",
            dict(d1=d, d2=d, eps1=0.05, eps2=0.05), to_matrix(balls, robust=robust)), Fva))
    except Degenerate:
        return np.nan


def _wave(balls, Fva, yva, cfg, steps, robust):
    if not viable(balls):
        return np.nan
    C, r, yb, p, n = arrays(balls, robust=robust)
    try:
        m = WaveGBTSVM(c1=cfg[0], c2=cfg[0], lam0=cfg[1], kappa=cfg[2],
                       adaptive_lambda=cfg[2] > 0, steps=steps).fit(C, r, yb, purity=p, size=n)
    except WDeg:
        return np.nan
    return accuracy_score(yva, m.predict(Fva))


def nested(name, rate, kind, kernel, robust, seed=0, n_out=5, n_in=3, steps=300):
    ds = load(name); X, y = ds["data"][:, :-1], ds["data"][:, -1]
    cls = np.unique(y); to_pm = lambda v: np.where(v == cls[0], 1.0, -1.0)
    acc = {m: [] for m in ("hard", "binom", "wave", "wave2")}
    outer = StratifiedKFold(n_out, shuffle=True, random_state=seed)
    for tr, te in outer.split(X, y):
        yn, _ = inject(X[tr], to_pm(y[tr]).astype(int), rate, kind, seed); yn = yn.astype(float)
        yte = to_pm(y[te])
        sc = MinMaxScaler((-1, 1)).fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        if kernel == "rbf":
            km = RBFMap(seed=seed).fit(Xtr); Ftr, Fte = km.transform(Xtr), km.transform(Xte)
        else:
            Ftr, Fte = Xtr, Xte

        s_hard = {d: [] for d in D_GRID}
        s_wave = {c: [] for c in WAVE}
        s_binom = {(e, d): [] for e in ETAS for d in D_GRID}
        s_wave2 = {(e, c): [] for e in ETAS for c in WAVE}
        inner = StratifiedKFold(n_in, shuffle=True, random_state=seed + 1)
        for itr, iva in inner.split(Ftr, yn):
            if len(np.unique(yn[itr])) < 2:
                continue
            sub = np.column_stack([Ftr[itr], yn[itr]])
            bp = gen_balls(sub, pur=1.0, delbals=1, seed=seed)
            for d in D_GRID:
                s_hard[d].append(_gb(bp, Ftr[iva], yn[iva], d, robust))
            for c in WAVE:
                s_wave[c].append(_wave(bp, Ftr[iva], yn[iva], c, steps, robust))
            for e in ETAS:
                be = gen_balls(sub, pur=1.0, delbals=1, seed=seed, eta=e, alpha=0.05)
                for d in D_GRID:
                    s_binom[(e, d)].append(_gb(be, Ftr[iva], yn[iva], d, robust))
                for c in WAVE:
                    s_wave2[(e, c)].append(_wave(be, Ftr[iva], yn[iva], c, steps, robust))

        bh, bw, bb, bw2 = _best(s_hard), _best(s_wave), _best(s_binom), _best(s_wave2)
        sub = np.column_stack([Ftr, yn])
        bp = gen_balls(sub, pur=1.0, delbals=1, seed=seed)
        beb = gen_balls(sub, pur=1.0, delbals=1, seed=seed, eta=bb[0], alpha=0.05)
        bew = gen_balls(sub, pur=1.0, delbals=1, seed=seed, eta=bw2[0], alpha=0.05)
        acc["hard"].append(_gb(bp, Fte, yte, bh, robust))
        acc["wave"].append(_wave(bp, Fte, yte, bw, steps, robust))
        acc["binom"].append(_gb(beb, Fte, yte, bb[1], robust))
        acc["wave2"].append(_wave(bew, Fte, yte, bw2[1], steps, robust))
    return acc


def main():
    ap = argparse.ArgumentParser()
    ten = [d for d in DATASETS_BINARY if d != "chess_krvkp"]
    ap.add_argument("--datasets", nargs="*", default=ten)
    ap.add_argument("--kinds", nargs="*", default=list(NOISE_KINDS))
    ap.add_argument("--rates", nargs="*", type=float, default=[0.3, 0.4, 0.5])
    ap.add_argument("--kernel", choices=["linear", "rbf"], default="linear")
    ap.add_argument("--robust", action="store_true")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = "robust" if args.robust else "plain"
    out = args.out or str(HERE.parent / "results" / f"_partial_fixed_{tag}_{args.kernel}.csv")
    rows = []
    print(f"[{tag}] {'dataset':13s}{'kind':>11}{'rate':>5}{'hard':>8}{'binom':>8}{'wave':>8}{'wave2':>8}", flush=True)
    for name in args.datasets:
        for kind in args.kinds:
            for rate in args.rates:
                t0 = time.time()
                try:
                    acc = nested(name, rate, kind, args.kernel, args.robust, steps=args.steps)
                except Exception as e:
                    print(f"  skip {name}/{kind}/{rate}: {type(e).__name__}: {str(e)[:50]}", flush=True); continue
                for m in acc:
                    v = np.asarray(acc[m], float)
                    rows.append(dict(dataset=name, kind=kind, rate=rate, model=m, robust=args.robust,
                                     acc=float(np.nanmean(v)), acc_std=float(np.nanstd(v)),
                                     n_valid=int(np.sum(~np.isnan(v))), n_total=int(v.size)))
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"{name:13s}{kind:>11}{rate:>5.1f}"
                      f"{np.nanmean(acc['hard']):>8.3f}{np.nanmean(acc['binom']):>8.3f}"
                      f"{np.nanmean(acc['wave']):>8.3f}{np.nanmean(acc['wave2']):>8.3f}"
                      f"   ({time.time()-t0:.0f}s)", flush=True)
    fn = save_run(pd.DataFrame(rows), f"fixed_{tag}_{args.kernel}",
                  dict(kernel=args.kernel, robust=args.robust, kinds=",".join(args.kinds),
                       rates=args.rates, steps=args.steps, note="new abs-lambda + empty guard"))
    print(f"saved {fn}")


if __name__ == "__main__":
    main()
