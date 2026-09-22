"""Full protocol (giống sheet 7/9): 10 bộ × 17 ô ((clean)+4 nhiễu×4 mức) × 5 seed,
cho hai-tâm RESCUE-ONLY ở độ thuần pur=0.7 (Wave λ=1, kappa=0, linear).

Ba cột:
  p07        : pur=0.7, đè đa số (như cũ)
  p07_rescue : pur=0.7, hai-tâm CHỈ khi một lớp sắp mất bóng (floor=1)
  p10        : pur=1.0 (baseline)

    python experiments/exp_rescue_full.py [--seeds 5] [--kernel linear]
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
PUR = 0.7


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


def _wave_arr(C, r, y, F, yte, steps):
    if len(C) == 0 or len(np.unique(y)) < 2:
        return np.nan
    try:
        m = WaveGBTSVM(steps=steps, **WAVE).fit(C, r, y)
    except WDeg:
        return np.nan
    return accuracy_score(yte, m.predict(F))


def _hard(balls, F, yte):
    if not viable(balls):
        return np.nan
    try:
        return accuracy_score(yte, decide(fit_binary("GBTSVM", GB, to_matrix(balls)), F))
    except Degenerate:
        return np.nan


def cell(name, kind, rate, seed, kernel, steps):
    ds = load(name); X, y = ds["data"][:, :-1], ds["data"][:, -1]
    cls = np.unique(y); pm = lambda v: np.where(v == cls[0], 1.0, -1.0)
    ah, aw, a07, a07r = [], [], [], []
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
        b07 = gen_balls(sub, pur=PUR, delbals=1, seed=seed)
        b10 = gen_balls(sub, pur=1.0, delbals=1, seed=seed)
        C07, r07, y07, _, _ = arrays(b07)
        Crs, rrs, yrs = rescue_arrays(b07)
        C10, r10, y10, _, _ = arrays(b10)
        ah.append(_hard(b10, Fte, yte))                                        # GB thường (hinge, pur=1)
        aw.append(_wave_arr(C10, r10, y10, Fte, yte, steps) if viable(b10) else np.nan)  # GB wave (pur=1)
        a07.append(_wave_arr(C07, r07, y07, Fte, yte, steps) if viable(b07) else np.nan)  # wave pur=0.7 đè
        a07r.append(_wave_arr(Crs, rrs, yrs, Fte, yte, steps))                 # hai-tâm rescue pur=0.7
    return np.nanmean(ah), np.nanmean(aw), np.nanmean(a07), np.nanmean(a07r)


def main():
    ap = argparse.ArgumentParser()
    ten = [d for d in DATASETS_BINARY if d != "chess_krvkp"]
    ap.add_argument("--datasets", nargs="*", default=ten)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--kernel", choices=["linear", "rbf"], default="linear")
    ap.add_argument("--steps", type=int, default=250)
    args = ap.parse_args()
    KINDS = ["symmetric", "asymmetric", "boundary", "far"]
    combos = [("(clean)", 0.0)] + [(k, r) for k in KINDS for r in (0.2, 0.3, 0.4, 0.5)]
    out = str(HERE.parent / "results" / f"_partial_rescuefull_{args.kernel}.csv")
    rows = []; t0 = time.time()
    print(f"pur={PUR} kappa=0 kernel={args.kernel} seeds={args.seeds}", flush=True)
    MODELS = ["gb_thuong", "gb_wave", "p07", "p07_rescue"]
    for name in args.datasets:
        for kind, rate in combos:
            for seed in range(args.seeds):
                try:
                    vh, vw, v07, v07r = cell(name, kind, rate, seed, args.kernel, args.steps)
                except Exception as e:
                    print(f"  skip {name}/{kind}/{rate}/s{seed}: {type(e).__name__}", flush=True); continue
                for m, v in zip(MODELS, (vh, vw, v07, v07r)):
                    rows.append(dict(dataset=name, kind=kind, rate=rate, seed=seed, model=m, acc=float(v)))
            pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  done {name}  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    from runlog import save_run
    fn = save_run(df, f"rescuefull_{args.kernel}",
                  dict(kernel=args.kernel, seeds=args.seeds, steps=args.steps, purity=PUR,
                       note="gb_thuong(hinge p1) vs gb_wave(p1) vs p07 vs p07_rescue, wave lam0=1 kappa=0"))
    tb = df.groupby(['kind', 'rate', 'model'])['acc'].mean().unstack('model')[MODELS]
    print(tb.round(3).to_string(), flush=True)
    ov = {m: round(df[df.model == m].groupby(['dataset', 'kind', 'rate'])['acc'].mean().mean(), 4)
          for m in MODELS}
    print("\nOVERALL:", ov, flush=True)
    for k in KINDS:
        sub = df[df['kind'] == k]
        vc = {m: int(sub[sub.model == m]['acc'].notna().sum()) for m in ['p07', 'p07_rescue']}
        mn = {m: round(sub[sub.model == m]['acc'].mean(), 3) for m in ['p07', 'p07_rescue']}
        print(f"  {k:11s} p07 {mn['p07']} / rescue {mn['p07_rescue']}  | ô hợp lệ p07 {vc['p07']} / rescue {vc['p07_rescue']}", flush=True)
    print(f"saved {fn}", flush=True)


if __name__ == "__main__":
    main()
