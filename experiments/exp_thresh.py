"""Quét ngưỡng ĐỘ THUẦN τ cho hai-tâm CHỦ ĐỘNG (per-ball), kèm rescue floor=1.
Mỗi bóng: luôn giữ tâm-đa-số; THÊM tâm-thiểu-số nếu
   (purity < τ)  HOẶC  (lớp thiểu số của nó đang bị đe dọa: < floor bóng).

τ = 0.7 (≤ mọi purity) → chỉ rescue khi suy biến (= bản hiện tại).
τ = 1.0 → tách mọi bóng lẫn (= full two-centroid + rescue).
τ trung gian → chỉ tách bóng lẫn NẶNG (purity thấp) + an toàn suy biến.

    python experiments/exp_thresh.py [--purity 0.7] [--seeds 3]
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
from gb.balls import gen_balls, arrays, viable
from models.wave_gbtsvm import WaveGBTSVM, Degenerate as WDeg
from models.kernel import RBFMap
from noise import inject

WAVE = dict(c1=10.0, c2=10.0, lam0=1.0, kappa=0.0, adaptive_lambda=False)
ALL = (1.0, -1.0)
TAUS = [0.7, 0.8, 0.9, 1.0]   # 0.7 = chỉ rescue; 1.0 = full
FLOOR = 1


def _radius(X, c):
    return float(np.sqrt(((X - c) ** 2).sum(1)).mean()) if len(X) else 0.0


def twocent_thresh(balls, tau, floor=FLOOR):
    C = [b.center for b in balls]; r = [b.radius for b in balls]; y = [b.label for b in balls]
    cnt = Counter(b.label for b in balls)
    threatened = {c for c in ALL if cnt.get(c, 0) < floor}
    for b in balls:
        if b.n_minor == 0:
            continue
        minlab = -b.label  # nhị phân
        # thêm tâm thiểu số nếu bóng lẫn nặng (purity<τ) HOẶC lớp thiểu số bị đe dọa
        if b.purity < tau or (minlab in threatened):
            labs = b.data[:, -2]
            Xmin = b.X[labs == minlab]
            if len(Xmin) > 0:
                cm = Xmin.mean(0)
                C.append(cm); r.append(_radius(Xmin, cm)); y.append(minlab)
    return np.array(C), np.array(r), np.array(y)


def _wave(C, r, y, F, yte, steps):
    if len(C) == 0 or len(np.unique(y)) < 2:
        return np.nan
    try:
        m = WaveGBTSVM(steps=steps, **WAVE).fit(C, r, y)
    except WDeg:
        return np.nan
    return accuracy_score(yte, m.predict(F))


def cell(name, kind, rate, seed, kernel, steps, pur):
    ds = load(name); X, y = ds["data"][:, :-1], ds["data"][:, -1]
    cls = np.unique(y); pm = lambda v: np.where(v == cls[0], 1.0, -1.0)
    out = {f"t{int(t*100)}": [] for t in TAUS}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        yb = pm(y[tr]).astype(int)
        yn = yb.copy() if rate == 0 else inject(X[tr], yb, rate, kind, seed)[0]
        yn = yn.astype(float); yte = pm(y[te])
        sc = MinMaxScaler((-1, 1)).fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        if kernel == "rbf":
            km = RBFMap(seed=seed).fit(Xtr); Ftr, Fte = km.transform(Xtr), km.transform(Xte)
        else:
            Ftr, Fte = Xtr, Xte
        b = gen_balls(np.column_stack([Ftr, yn]), pur=pur, delbals=1, seed=seed)
        for t in TAUS:
            C, r, yl = twocent_thresh(b, t)
            out[f"t{int(t*100)}"].append(_wave(C, r, yl, Fte, yte, steps))
    return {k: np.nanmean(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ten = [d for d in DATASETS_BINARY if d != "chess_krvkp"]
    ap.add_argument("--datasets", nargs="*", default=ten)
    ap.add_argument("--purity", type=float, default=0.7)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--kernel", choices=["linear", "rbf"], default="linear")
    ap.add_argument("--steps", type=int, default=250)
    args = ap.parse_args()
    KINDS = ["symmetric", "asymmetric", "boundary", "far"]
    combos = [(k, r) for k in KINDS for r in (0.3, 0.4, 0.5)]
    CFG = [f"t{int(t*100)}" for t in TAUS]
    out = str(HERE.parent / "results" / f"_partial_thresh_{args.kernel}.csv")
    rows = []; t0 = time.time()
    print(f"pur={args.purity} taus={TAUS} floor={FLOOR} seeds={args.seeds}", flush=True)
    for name in args.datasets:
        for kind, rate in combos:
            for seed in range(args.seeds):
                try:
                    d = cell(name, kind, rate, seed, args.kernel, args.steps, args.purity)
                except Exception as e:
                    print(f"  skip {name}/{kind}/{rate}/s{seed}: {type(e).__name__}", flush=True); continue
                for c in CFG:
                    rows.append(dict(dataset=name, kind=kind, rate=rate, seed=seed, config=c, acc=float(d[c])))
            pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  done {name} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    from runlog import save_run
    fn = save_run(df, f"thresh_{args.kernel}",
                  dict(kernel=args.kernel, seeds=args.seeds, purity=args.purity, taus=str(TAUS)))
    piv = df.groupby(['kind', 'rate', 'config'])['acc'].mean().unstack('config')[CFG]
    print(piv.round(3).to_string(), flush=True)
    ov = {c: round(df[df.config == c].groupby(['dataset', 'kind', 'rate'])['acc'].mean().mean(), 4) for c in CFG}
    print("OVERALL:", ov, flush=True)
    print(f"saved {fn}", flush=True)


if __name__ == "__main__":
    main()
