"""Bóng bền (robust) ở độ thuần thấp: tâm lệch có phải thủ phạm làm pur=0.8 tệ?

So 3 cấu hình cho Wave-GBTSVM (λ cố định, hard params):
  p08        : sinh bóng pur=0.8, tâm/bán kính MEAN (thường)
  p08_robust : sinh bóng pur=0.8, tâm NHÃN-ĐA-SỐ + bán kính MEDIAN (robust=True)
  p10        : sinh bóng pur=1.0 (baseline tốt nhất hiện tại)

Giả thuyết: ở pur=0.8, tâm mean bị 20% điểm sai-nhãn kéo lệch → robust (tâm chỉ
tính trên điểm đa số) sẽ kéo p08 lại gần p10.

    python experiments/exp_robust.py [--seeds 3] [--kernel linear]
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
from gb.balls import gen_balls, arrays, viable
from models.wave_gbtsvm import WaveGBTSVM, Degenerate as WDeg
from models.kernel import RBFMap
from noise import inject

WAVE = dict(c1=10.0, c2=10.0, lam0=1.0, kappa=0.0, adaptive_lambda=False)


def _wave(balls, F, y, steps, robust):
    if not viable(balls):
        return np.nan
    C, r, yb, p, n = arrays(balls, robust=robust)
    try:
        m = WaveGBTSVM(steps=steps, **WAVE).fit(C, r, yb, purity=p, size=n)
    except WDeg:
        return np.nan
    return accuracy_score(y, m.predict(F))


def cell(name, kind, rate, seed, kernel, steps):
    ds = load(name); X, y = ds["data"][:, :-1], ds["data"][:, -1]
    cls = np.unique(y); pm = lambda v: np.where(v == cls[0], 1.0, -1.0)
    a08, a08r, a10 = [], [], []
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
        b08 = gen_balls(sub, pur=0.8, delbals=1, seed=seed)
        b10 = gen_balls(sub, pur=1.0, delbals=1, seed=seed)
        a08.append(_wave(b08, Fte, yte, steps, robust=False))
        a08r.append(_wave(b08, Fte, yte, steps, robust=True))
        a10.append(_wave(b10, Fte, yte, steps, robust=False))
    return np.nanmean(a08), np.nanmean(a08r), np.nanmean(a10)


def main():
    ap = argparse.ArgumentParser()
    ten = [d for d in DATASETS_BINARY if d != "chess_krvkp"]
    ap.add_argument("--datasets", nargs="*", default=ten)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--kernel", choices=["linear", "rbf"], default="linear")
    ap.add_argument("--steps", type=int, default=250)
    args = ap.parse_args()
    KINDS = ["symmetric", "asymmetric", "boundary", "far"]
    combos = [(k, r) for k in KINDS for r in (0.3, 0.4, 0.5)]
    out = str(HERE.parent / "results" / f"_partial_robust_{args.kernel}.csv")
    rows = []; t0 = time.time()
    print(f"kernel={args.kernel} seeds={args.seeds}", flush=True)
    for name in args.datasets:
        for kind, rate in combos:
            for seed in range(args.seeds):
                try:
                    v8, v8r, v10 = cell(name, kind, rate, seed, args.kernel, args.steps)
                except Exception as e:
                    print(f"  skip {name}/{kind}/{rate}/s{seed}: {type(e).__name__}", flush=True); continue
                for m, v in (("p08", v8), ("p08_robust", v8r), ("p10", v10)):
                    rows.append(dict(dataset=name, kind=kind, rate=rate, seed=seed, model=m, acc=float(v)))
            pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  done {name}  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    from runlog import save_run
    fn = save_run(df, f"robust_{args.kernel}",
                  dict(kernel=args.kernel, seeds=args.seeds, steps=args.steps,
                       note="p08 vs p08_robust vs p10, wave lam0=1 kappa=0"))
    # tóm tắt nhanh
    tb = df.groupby(['kind','rate','model'])['acc'].mean().unstack('model')
    tb = tb[['p10','p08','p08_robust']]
    print(tb.round(3).to_string(), flush=True)
    print("\nOVERALL:", {m: round(df[df.model==m].groupby(['dataset','kind','rate'])['acc'].mean().mean(),4)
                         for m in ['p10','p08','p08_robust']}, flush=True)
    print(f"saved {fn}", flush=True)


if __name__ == "__main__":
    main()
