"""Đếm SỐ BÓNG theo protocol full (giống sheet 14) — chỉ sinh bóng, không huấn luyện.
Mỗi ô = số bóng TB trên 5 fold của một seed, cho các cấu hình:
  N (số điểm) · pur=1 · pur=0.7 · pur=0.7+R · pur=0.6 · pur=0.6+R
(+R = pur đó sau khi thêm bóng rescue-only khi một lớp sắp mất bóng)

    python experiments/exp_ballcount.py [--seeds 5] [--kernel linear]
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
from config import DATASETS_BINARY
from data.datasets import load
from gb.balls import gen_balls
from models.kernel import RBFMap
from noise import inject

ALL = (1.0, -1.0)


def rescue_added(balls):
    c = Counter(b.label for b in balls); add = 0
    threatened = {x for x in ALL if c.get(x, 0) < 1}
    if threatened:
        for b in balls:
            labs = b.data[:, -2]
            for x in threatened:
                if x != b.label and (labs == x).any():
                    add += 1
    return add


def cell(name, kind, rate, seed, kernel):
    ds = load(name); X, y = ds["data"][:, :-1], ds["data"][:, -1]
    cls = np.unique(y); pm = lambda v: np.where(v == cls[0], 1.0, -1.0)
    N, p1, p07, p07r, p06, p06r = [], [], [], [], [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        yb = pm(y[tr]).astype(int)
        yn = yb.copy() if rate == 0 else inject(X[tr], yb, rate, kind, seed)[0]
        yn = yn.astype(float)
        sc = MinMaxScaler((-1, 1)).fit(X[tr]); Ftr = sc.transform(X[tr])
        if kernel == "rbf":
            Ftr = RBFMap(seed=seed).fit(Ftr).transform(Ftr)
        sub = np.column_stack([Ftr, yn])
        b1 = gen_balls(sub, pur=1.0, delbals=1, seed=seed)
        b7 = gen_balls(sub, pur=0.7, delbals=1, seed=seed)
        b6 = gen_balls(sub, pur=0.6, delbals=1, seed=seed)
        N.append(len(Ftr)); p1.append(len(b1))
        p07.append(len(b7)); p07r.append(len(b7) + rescue_added(b7))
        p06.append(len(b6)); p06r.append(len(b6) + rescue_added(b6))
    return dict(N=np.mean(N), p1=np.mean(p1), p07=np.mean(p07), p07r=np.mean(p07r),
                p06=np.mean(p06), p06r=np.mean(p06r))


def main():
    ap = argparse.ArgumentParser()
    ten = [d for d in DATASETS_BINARY if d != "chess_krvkp"]
    ap.add_argument("--datasets", nargs="*", default=ten)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--kernel", choices=["linear", "rbf"], default="linear")
    args = ap.parse_args()
    KINDS = ["symmetric", "asymmetric", "boundary", "far"]
    combos = [("(clean)", 0.0)] + [(k, r) for k in KINDS for r in (0.2, 0.3, 0.4, 0.5)]
    out = str(HERE.parent / "results" / f"_partial_ballcount_{args.kernel}.csv")
    CFG = ["N", "p1", "p07", "p07r", "p06", "p06r"]
    rows = []; t0 = time.time()
    print(f"ballcount kernel={args.kernel} seeds={args.seeds}", flush=True)
    for name in args.datasets:
        for kind, rate in combos:
            for seed in range(args.seeds):
                d = cell(name, kind, rate, seed, args.kernel)
                for c in CFG:
                    rows.append(dict(dataset=name, kind=kind, rate=rate, seed=seed, config=c, nballs=float(d[c])))
            pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  done {name}  ({time.time()-t0:.0f}s)", flush=True)
    from runlog import save_run
    fn = save_run(pd.DataFrame(rows), f"ballcount_{args.kernel}",
                  dict(kernel=args.kernel, seeds=args.seeds, note="so bong / diem, pur 1/0.7/0.7R/0.6/0.6R"))
    print(f"saved {fn}", flush=True)


if __name__ == "__main__":
    main()
