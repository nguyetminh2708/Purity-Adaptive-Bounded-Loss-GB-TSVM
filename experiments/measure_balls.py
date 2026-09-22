import sys, pathlib
import pathlib as _pl
HERE=_pl.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"src")); sys.path.insert(0, str(HERE))
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from config import DATASETS_BINARY
from data.datasets import load
from gb.balls import gen_balls
from noise import inject

ALL = (1.0, -1.0)
def rescue_count(balls):
    """số bóng sau rescue-only (floor=1)."""
    n = len(balls)
    cnt = Counter(b.label for b in balls)
    threatened = {c for c in ALL if cnt.get(c, 0) < 1}
    if threatened:
        for b in balls:
            labs = b.data[:, -2]
            for c in threatened:
                if c != b.label and (labs == c).any():
                    n += 1
    return n

ten = [d for d in DATASETS_BINARY if d != "chess_krvkp"]
SEEDS = [0, 1, 2]
# đo ở 2 điều kiện: symmetric 0.3 (rescue ít khi bật) và asymmetric 0.4 (rescue hay bật)
CONDS = [("symmetric", 0.3), ("asymmetric", 0.4)]

print(f"{'dataset':13s}{'Ntrain':>8}{'p1':>7}{'0.7':>7}{'0.7+R':>7}{'0.6':>7}{'  N/0.7':>8}{'  N/p1':>7}   (đk)")
agg = {c: {k: [] for k in ['N','p1','p07','p07r','p06']} for c in CONDS}
for name in ten:
    ds = load(name); X, y = ds["data"][:, :-1], ds["data"][:, -1]
    cls = np.unique(y); pm = lambda v: np.where(v == cls[0], 1.0, -1.0)
    for kind, rate in CONDS:
        N=[]; b1=[]; b07=[]; b07r=[]; b06=[]
        for seed in SEEDS:
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
                yb = pm(y[tr]).astype(int)
                yn = inject(X[tr], yb, rate, kind, seed)[0].astype(float)
                sc = MinMaxScaler((-1,1)).fit(X[tr]); Ftr = sc.transform(X[tr])
                sub = np.column_stack([Ftr, yn])
                N.append(len(Ftr))
                b1.append(len(gen_balls(sub, pur=1.0, delbals=1, seed=seed)))
                bb = gen_balls(sub, pur=0.7, delbals=1, seed=seed)
                b07.append(len(bb)); b07r.append(rescue_count(bb))
                b06.append(len(gen_balls(sub, pur=0.6, delbals=1, seed=seed)))
        N,b1,b07,b07r,b06 = map(lambda z: float(np.mean(z)), (N,b1,b07,b07r,b06))
        for k,v in zip(['N','p1','p07','p07r','p06'],(N,b1,b07,b07r,b06)): agg[(kind,rate)][k].append(v)
        tag = "sym.3" if kind=="symmetric" else "asy.4"
        print(f"{name:13s}{N:>8.0f}{b1:>7.0f}{b07:>7.0f}{b07r:>7.0f}{b06:>7.0f}{N/b07:>8.1f}{N/b1:>7.1f}   {tag}")

print("\n=== TRUNG BÌNH 10 BỘ ===")
for c in CONDS:
    a = agg[c]; N=np.mean(a['N'])
    print(f"{c[0]:11s} {c[1]}: Ntrain {N:.0f} | bóng p1 {np.mean(a['p1']):.0f} "
          f"({N/np.mean(a['p1']):.1f}x) · 0.7 {np.mean(a['p07']):.0f} ({N/np.mean(a['p07']):.1f}x) "
          f"· 0.7+R {np.mean(a['p07r']):.0f} ({N/np.mean(a['p07r']):.1f}x) · 0.6 {np.mean(a['p06']):.0f} ({N/np.mean(a['p06']):.1f}x)")
