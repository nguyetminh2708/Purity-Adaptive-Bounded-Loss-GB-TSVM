"""Hai-tâm cho bóng lẫn: giữ đại diện lớp thiểu số để chống suy biến mất nhãn.

Bóng THUẦN  -> một bóng như cũ (tâm đa số, nhãn đa số).
Bóng LẪN    -> HAI bóng: (tâm điểm đa số, nhãn đa số) + (tâm điểm thiểu số, nhãn NGƯỢC).
Nhờ vậy hễ còn điểm nhãn -1 ở đâu thì còn bóng -1 ở đó -> lớp thiểu số không bị nuốt.

So 3 cấu hình (Wave-GBTSVM, λ cố định, linear):
  p10     : pur=1.0 (baseline tốt nhất hiện tại)
  p08     : pur=0.8, bóng lẫn ĐÈ về nhãn đa số (như cũ)
  p08_2c  : pur=0.8, bóng lẫn tách HAI-TÂM

    python experiments/exp_twocent.py [--seeds 3] [--kernel linear]
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


def _radius(X, c):
    return float(np.sqrt(((X - c) ** 2).sum(1)).mean()) if len(X) else 0.0


def twocent_arrays(balls):
    """list[Ball] -> (C, r, y) với bóng lẫn tách thành hai tâm."""
    C, r, y = [], [], []
    for b in balls:
        labs = b.data[:, -2]
        maj = b.label
        Xmaj = b.X[labs == maj]
        cM = Xmaj.mean(0)
        C.append(cM); r.append(_radius(Xmaj, cM)); y.append(maj)
        Xmin = b.X[labs != maj]
        if len(Xmin) > 0:                       # bóng lẫn -> thêm tâm thiểu số
            cm = Xmin.mean(0)
            C.append(cm); r.append(_radius(Xmin, cm)); y.append(-maj)
    return np.array(C), np.array(r), np.array(y)


def _wave_arr(C, r, y, F, yte, steps):
    if len(C) == 0 or len(np.unique(y)) < 2:
        return np.nan
    try:
        m = WaveGBTSVM(steps=steps, **WAVE).fit(C, r, y)
    except WDeg:
        return np.nan
    return accuracy_score(yte, m.predict(F))


def cell(name, kind, rate, seed, kernel, steps):
    ds = load(name); X, y = ds["data"][:, :-1], ds["data"][:, -1]
    cls = np.unique(y); pm = lambda v: np.where(v == cls[0], 1.0, -1.0)
    a10, a08, a2c = [], [], []
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
        # p10 & p08: tâm mean / nhãn đa số
        C10, r10, y10, _, _ = arrays(b10)
        C08, r08, y08, _, _ = arrays(b08)
        C2c, r2c, y2c = twocent_arrays(b08)
        a10.append(_wave_arr(C10, r10, y10, Fte, yte, steps) if viable(b10) else np.nan)
        a08.append(_wave_arr(C08, r08, y08, Fte, yte, steps) if viable(b08) else np.nan)
        a2c.append(_wave_arr(C2c, r2c, y2c, Fte, yte, steps))
    return np.nanmean(a10), np.nanmean(a08), np.nanmean(a2c)


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
    out = str(HERE.parent / "results" / f"_partial_twocent_{args.kernel}.csv")
    rows = []; t0 = time.time()
    print(f"kernel={args.kernel} seeds={args.seeds}", flush=True)
    for name in args.datasets:
        for kind, rate in combos:
            for seed in range(args.seeds):
                try:
                    v10, v08, v2c = cell(name, kind, rate, seed, args.kernel, args.steps)
                except Exception as e:
                    print(f"  skip {name}/{kind}/{rate}/s{seed}: {type(e).__name__}", flush=True); continue
                for m, v in (("p10", v10), ("p08", v08), ("p08_2c", v2c)):
                    rows.append(dict(dataset=name, kind=kind, rate=rate, seed=seed, model=m, acc=float(v)))
            pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  done {name}  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    from runlog import save_run
    fn = save_run(df, f"twocent_{args.kernel}",
                  dict(kernel=args.kernel, seeds=args.seeds, steps=args.steps,
                       note="p10 vs p08 vs p08 two-centroid, wave lam0=1 kappa=0"))
    tb = df.groupby(['kind', 'rate', 'model'])['acc'].mean().unstack('model')[['p10', 'p08', 'p08_2c']]
    print(tb.round(3).to_string(), flush=True)
    ov = {m: round(df[df.model == m].groupby(['dataset', 'kind', 'rate'])['acc'].mean().mean(), 4)
          for m in ['p10', 'p08', 'p08_2c']}
    print("\nOVERALL:", ov, flush=True)
    # riêng asymmetric (chỗ suy biến hay xảy ra) + đếm ô hợp lệ
    for k in KINDS:
        sub = df[df['kind'] == k]
        vc = {m: int(sub[sub.model == m]['acc'].notna().sum()) for m in ['p08', 'p08_2c']}
        mn = {m: round(sub[sub.model == m]['acc'].mean(), 3) for m in ['p08', 'p08_2c']}
        print(f"  {k:11s} mean p08 {mn['p08']} / 2c {mn['p08_2c']}  | ô hợp lệ p08 {vc['p08']} / 2c {vc['p08_2c']}", flush=True)
    print(f"saved {fn}", flush=True)


if __name__ == "__main__":
    main()
