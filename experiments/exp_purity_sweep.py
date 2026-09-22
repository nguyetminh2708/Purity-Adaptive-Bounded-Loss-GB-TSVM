"""Quét độ thuần thấp: mỗi mức purity P chạy 3 model trên CÙNG bộ bóng pur=P.
  hard        : GBTSVM gốc (hinge) trên bóng pur=P, đè đa số
  wave        : Wave-GBTSVM trên bóng pur=P, đè đa số
  wave_rescue : Wave-GBTSVM trên bóng pur=P, hai-tâm rescue-only (floor=1)

Baseline pur=1 (hard@1, wave@1) KHÔNG chạy lại — lấy từ dữ liệu sheet 13.
Full 170×5 seed (10 bộ × 17 ô × 5 seed).

    python experiments/exp_purity_sweep.py --purities 0.7 0.6 [--kernel linear]
"""
from __future__ import annotations
import sys, pathlib, argparse, warnings, time
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
from data.datasets import load
from gb.balls import gen_balls, arrays, to_matrix, viable
from models.wave_gbtsvm import WaveGBTSVM, Degenerate as WDeg
from models.registry import fit_binary, decide, Degenerate
from models.kernel import RBFMap
from noise import inject

WAVE = dict(c1=10.0, c2=10.0, lam0=1.0, kappa=0.0, adaptive_lambda=False)
GB = dict(d1=0.1, d2=0.1, eps1=0.05, eps2=0.05)
ALL_LABELS = (1.0, -1.0)


def _radius(X, c):
    return float(np.sqrt(((X - c) ** 2).sum(1)).mean()) if len(X) else 0.0


def rescue_arrays(balls, floor=1):
    C = [b.center for b in balls]; r = [b.radius for b in balls]; y = [b.label for b in balls]
    cnt = Counter(b.label for b in balls)
    threatened = {c for c in ALL_LABELS if cnt.get(c, 0) < floor}
    if threatened:
        for b in balls:
            labs = b.data[:, -2]
            for c in threatened:
                if c == b.label:
                    continue
                Xc = b.X[labs == c]
                if len(Xc) > 0:
                    cc = Xc.mean(0)
                    C.append(cc); r.append(_radius(Xc, cc)); y.append(c)
    return np.array(C), np.array(r), np.array(y)


def _wave(C, r, y, F, yte, steps):
    if len(C) == 0 or len(np.unique(y)) < 2:
        return np.nan
    try:
        m = WaveGBTSVM(steps=steps, **WAVE).fit(C, r, y)
    except WDeg:
        return np.nan
    return accuracy_score(yte, m.predict(F))


def _hard(C, r, y, F, yte):
    if len(C) == 0 or len(np.unique(y)) < 2:
        return np.nan
    mat = np.column_stack([C, r, y])
    try:
        return accuracy_score(yte, decide(fit_binary("GBTSVM", GB, mat), F))
    except Degenerate:
        return np.nan


def cell(name, kind, rate, seed, kernel, steps, pur):
    ds = load(name); X, y = ds["data"][:, :-1], ds["data"][:, -1]
    cls = np.unique(y); pm = lambda v: np.where(v == cls[0], 1.0, -1.0)
    ah, aw, awr = [], [], []
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
        b = gen_balls(sub, pur=pur, delbals=1, seed=seed)
        C, r, yl, _, _ = arrays(b)
        Cr, rr, yr = rescue_arrays(b)
        ah.append(_hard(C, r, yl, Fte, yte))
        aw.append(_wave(C, r, yl, Fte, yte, steps))
        awr.append(_wave(Cr, rr, yr, Fte, yte, steps))
    return np.nanmean(ah), np.nanmean(aw), np.nanmean(awr)


def main():
    ap = argparse.ArgumentParser()
    ten = [d for d in DATASETS_BINARY if d != "chess_krvkp"]
    ap.add_argument("--datasets", nargs="*", default=ten)
    ap.add_argument("--purities", nargs="*", type=float, default=[0.7, 0.6])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--kernel", choices=["linear", "rbf"], default="linear")
    ap.add_argument("--steps", type=int, default=250)
    args = ap.parse_args()
    KINDS = ["symmetric", "asymmetric", "boundary", "far"]
    combos = [("(clean)", 0.0)] + [(k, r) for k in KINDS for r in (0.2, 0.3, 0.4, 0.5)]
    out = str(HERE.parent / "results" / f"_partial_puritysweep_{args.kernel}.csv")
    rows = []; t0 = time.time()
    print(f"purities={args.purities} kernel={args.kernel} seeds={args.seeds}", flush=True)
    for pur in args.purities:
        for name in args.datasets:
            for kind, rate in combos:
                for seed in range(args.seeds):
                    try:
                        vh, vw, vwr = cell(name, kind, rate, seed, args.kernel, args.steps, pur)
                    except Exception as e:
                        print(f"  skip p{pur}/{name}/{kind}/{rate}/s{seed}: {type(e).__name__}", flush=True); continue
                    for m, v in (("hard", vh), ("wave", vw), ("wave_rescue", vwr)):
                        rows.append(dict(purity=pur, dataset=name, kind=kind, rate=rate, seed=seed, model=m, acc=float(v)))
                pd.DataFrame(rows).to_csv(out, index=False)
            print(f"  [pur={pur}] done {name}  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    from runlog import save_run
    fn = save_run(df, f"puritysweep_{args.kernel}",
                  dict(kernel=args.kernel, seeds=args.seeds, steps=args.steps,
                       purities=",".join(map(str, args.purities)),
                       note="hard/wave/wave_rescue per purity, wave lam0=1 kappa=0"))
    for pur in args.purities:
        d = df[df.purity == pur]
        ov = {m: round(d[d.model == m].groupby(['dataset', 'kind', 'rate'])['acc'].mean().mean(), 4)
              for m in ['hard', 'wave', 'wave_rescue']}
        print(f"pur={pur} OVERALL:", ov, flush=True)
    print(f"saved {fn}", flush=True)


if __name__ == "__main__":
    main()
