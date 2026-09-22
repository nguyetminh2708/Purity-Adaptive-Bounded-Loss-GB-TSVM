"""Multi-seed run at impure balls (purity<1) to test whether Purity-Adaptive
lambda (kappa>0) actually earns its keep once balls carry a real purity signal.

Three models, fixed params, many seeds (like exp_seeds.py):
  hard   : GBTSVM gốc (hinge)
  wavef  : Wave loss, lambda CỐ ĐỊNH  (kappa=0, adaptive_lambda=False)
  wavea  : Wave loss, lambda THÍCH NGHI theo purity (kappa>0, adaptive_lambda=True)

Balls generated with pur=--purity (default 0.8) so impure balls exist.
Output long CSV: dataset,kind,rate,seed,model,acc  (acc = mean over 5 folds).

    python experiments/exp_seeds_pur.py [--purity 0.8] [--kappa 2.0] [--seeds 5] [--kernel linear]
"""
from __future__ import annotations
import sys, pathlib, argparse, warnings, time
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
from noise import inject

GB = dict(d1=0.1, d2=0.1, eps1=0.05, eps2=0.05)
WAVE_BASE = dict(c1=10.0, c2=10.0, lam0=1.0)


def _hard(balls, F, y):
    if not viable(balls): return np.nan
    try: return accuracy_score(y, decide(fit_binary("GBTSVM", GB, to_matrix(balls)), F))
    except Degenerate: return np.nan


def _wave(balls, F, y, steps, kappa, adaptive):
    if not viable(balls): return np.nan
    C, r, yb, p, n = arrays(balls)
    try:
        m = WaveGBTSVM(steps=steps, kappa=kappa, adaptive_lambda=adaptive,
                       **WAVE_BASE).fit(C, r, yb, purity=p, size=n)
    except WDeg:
        return np.nan
    return accuracy_score(y, m.predict(F))


def cell(name, kind, rate, seed, kernel, steps, purity, kappa):
    ds = load(name); X, y = ds["data"][:, :-1], ds["data"][:, -1]
    cls = np.unique(y); pm = lambda v: np.where(v == cls[0], 1.0, -1.0)
    ah, af, aa = [], [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        yb = pm(y[tr]).astype(int)
        yn = yb.copy() if rate == 0 else inject(X[tr], yb, rate, kind, seed)[0]
        yn = yn.astype(float); yte = pm(y[te])
        sc = MinMaxScaler((-1, 1)).fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        if kernel == "rbf":
            km = RBFMap(seed=seed).fit(Xtr); Ftr, Fte = km.transform(Xtr), km.transform(Xte)
        else:
            Ftr, Fte = Xtr, Xte
        sub = np.column_stack([Ftr, yn])
        bp = gen_balls(sub, pur=purity, delbals=1, seed=seed)
        ah.append(_hard(bp, Fte, yte))
        af.append(_wave(bp, Fte, yte, steps, kappa=0.0, adaptive=False))
        aa.append(_wave(bp, Fte, yte, steps, kappa=kappa, adaptive=True))
    return np.nanmean(ah), np.nanmean(af), np.nanmean(aa)


def main():
    ap = argparse.ArgumentParser()
    ten = [d for d in DATASETS_BINARY if d != "chess_krvkp"]
    ap.add_argument("--datasets", nargs="*", default=ten)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--kernel", choices=["linear", "rbf"], default="linear")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--purity", type=float, default=0.8)
    ap.add_argument("--kappa", type=float, default=2.0)
    args = ap.parse_args()
    KINDS = ["symmetric", "asymmetric", "boundary", "far"]
    combos = [("(clean)", 0.0)] + [(k, r) for k in KINDS for r in (0.2, 0.3, 0.4, 0.5)]
    out = str(HERE.parent / "results" / f"_partial_seedspur_{args.kernel}.csv")
    rows = []
    t0 = time.time()
    print(f"pur={args.purity} kappa={args.kappa} kernel={args.kernel} seeds={args.seeds}", flush=True)
    for name in args.datasets:
        for kind, rate in combos:
            for seed in range(args.seeds):
                try:
                    h, wf, wa = cell(name, kind, rate, seed, args.kernel, args.steps,
                                     args.purity, args.kappa)
                except Exception as e:
                    print(f"  skip {name}/{kind}/{rate}/s{seed}: {type(e).__name__}", flush=True); continue
                for m, v in (("hard", h), ("wavef", wf), ("wavea", wa)):
                    rows.append(dict(dataset=name, kind=kind, rate=rate, seed=seed, model=m, acc=float(v)))
            pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  done {name}  ({time.time()-t0:.0f}s)", flush=True)
    fn = save_run(pd.DataFrame(rows), f"seedspur_{args.kernel}",
                  dict(kernel=args.kernel, seeds=args.seeds, steps=args.steps,
                       purity=args.purity, kappa=args.kappa,
                       params="hard d=.1; wavef lam0=1 kappa=0; wavea lam0=1 kappa=K adaptive"))
    print(f"saved {fn}")


if __name__ == "__main__":
    main()
