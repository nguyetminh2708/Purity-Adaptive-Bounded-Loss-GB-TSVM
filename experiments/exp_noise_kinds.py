"""Experiment A: does bounded loss pull ahead under heavy / structured noise?

Nested CV (outer 5 clean / inner 3 on noisy train) for hard, binom, wave across
the four noise kinds x rates 0.3/0.4/0.5. Question: does the wave-hard gap widen
at high noise and on asymmetric/far noise, where a bounded loss should help most.

    python experiments/exp_noise_kinds.py [--datasets ...] [--kernel linear|rbf]
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
from config import DATASETS_BINARY
from data.datasets import load
from gb.balls import gen_balls, viable
from models.kernel import RBFMap
from runlog import save_run
from noise import inject, NOISE_KINDS
from tune_compare import _gb, _wave, _best   # shared scoring helpers

D_GRID = [0.01, 0.1, 1.0]
ETAS = [0.3, 0.4, 0.5]
WAVE = list(product([1.0, 10.0], [0.5, 2.0, 4.0], [0.0, 2.0, 4.0]))


def nested(name, rate, kind, kernel, seed=0, n_out=5, n_in=3, steps=300):
    ds = load(name); X, y = ds["data"][:, :-1], ds["data"][:, -1]
    cls = np.unique(y); to_pm = lambda v: np.where(v == cls[0], 1.0, -1.0)
    acc = {m: [] for m in ("hard", "binom", "wave")}
    outer = StratifiedKFold(n_out, shuffle=True, random_state=seed)
    for tr, te in outer.split(X, y):
        yb = to_pm(y[tr]).astype(int)
        yn, _ = inject(X[tr], yb, rate, kind, seed)
        yn = yn.astype(float); yte = to_pm(y[te])
        sc = MinMaxScaler((-1, 1)).fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        if kernel == "rbf":
            km = RBFMap(seed=seed).fit(Xtr); Ftr, Fte = km.transform(Xtr), km.transform(Xte)
        else:
            Ftr, Fte = Xtr, Xte

        s_hard = {d: [] for d in D_GRID}
        s_wave = {c: [] for c in WAVE}
        s_binom = {(e, d): [] for e in ETAS for d in D_GRID}
        inner = StratifiedKFold(n_in, shuffle=True, random_state=seed + 1)
        for itr, iva in inner.split(Ftr, yn):
            if len(np.unique(yn[itr])) < 2:
                continue
            sub = np.column_stack([Ftr[itr], yn[itr]])
            bp = gen_balls(sub, pur=1.0, delbals=1, seed=seed)
            for d in D_GRID:
                s_hard[d].append(_gb(bp, Ftr[iva], yn[iva], d))
            for c in WAVE:
                s_wave[c].append(_wave(bp, Ftr[iva], yn[iva], c, steps))
            for e in ETAS:
                be = gen_balls(sub, pur=1.0, delbals=1, seed=seed, eta=e, alpha=0.05)
                for d in D_GRID:
                    s_binom[(e, d)].append(_gb(be, Ftr[iva], yn[iva], d))

        bh, bw, bb = _best(s_hard), _best(s_wave), _best(s_binom)
        sub = np.column_stack([Ftr, yn])
        bp = gen_balls(sub, pur=1.0, delbals=1, seed=seed)
        be = gen_balls(sub, pur=1.0, delbals=1, seed=seed, eta=bb[0], alpha=0.05)
        acc["hard"].append(_gb(bp, Fte, yte, bh))
        acc["wave"].append(_wave(bp, Fte, yte, bw, steps))
        acc["binom"].append(_gb(be, Fte, yte, bb[1]))
    return acc


def main():
    ap = argparse.ArgumentParser()
    ten = [d for d in DATASETS_BINARY if d != "chess_krvkp"]
    ap.add_argument("--datasets", nargs="*", default=ten)
    ap.add_argument("--kinds", nargs="*", default=list(NOISE_KINDS))
    ap.add_argument("--rates", nargs="*", type=float, default=[0.3, 0.4, 0.5])
    ap.add_argument("--kernel", choices=["linear", "rbf"], default="linear")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or str(HERE.parent / "results" / f"_partial_noisekinds_{args.kernel}.csv")
    rows = []
    print(f"{'dataset':13s}{'kind':>11}{'rate':>5}{'hard':>8}{'binom':>8}{'wave':>8}", flush=True)
    for name in args.datasets:
        for kind in args.kinds:
            for rate in args.rates:
                t0 = time.time()
                try:
                    acc = nested(name, rate, kind, args.kernel, steps=args.steps)
                except Exception as e:
                    print(f"  skip {name}/{kind}/{rate}: {type(e).__name__}: {str(e)[:50]}", flush=True); continue
                for m in acc:
                    rows.append(dict(dataset=name, kind=kind, rate=rate, model=m,
                                     acc=float(np.nanmean(acc[m])), acc_std=float(np.nanstd(acc[m]))))
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"{name:13s}{kind:>11}{rate:>5.1f}"
                      f"{np.nanmean(acc['hard']):>8.3f}{np.nanmean(acc['binom']):>8.3f}{np.nanmean(acc['wave']):>8.3f}"
                      f"   ({time.time()-t0:.0f}s)", flush=True)
    fn = save_run(pd.DataFrame(rows), f"noisekinds_{args.kernel}",
                  dict(kernel=args.kernel, kinds=",".join(args.kinds), rates=args.rates, steps=args.steps))
    print(f"saved {fn}")


if __name__ == "__main__":
    main()
