"""Kiểm soát pur=1: SVM có thắng nearest-centroid ở độ thuần cao không?
So hard (hinge GBTSVM) / wave / nearest-centroid trên CÙNG tâm bóng pur=1.
Nếu wave/hard > NC ở pur=1 -> SVM làm việc thật ở độ thuần cao (luận điểm chính vững).
"""
from __future__ import annotations
import sys, pathlib, argparse, warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from gb.balls import gen_balls, arrays
from gb.rescue import nearest_centroid_predict
from models.wave_gbtsvm import WaveGBTSVM, Degenerate as WDeg
from models.registry import fit_binary, decide, Degenerate
from noise import inject

WAVE = dict(c1=10.0, c2=10.0, lam0=1.0, kappa=0.0, adaptive_lambda=False)
GB = dict(d1=0.1, d2=0.1, eps1=0.05, eps2=0.05)


def wave_acc(C, r, y, F, yte):
    if len(C) == 0 or len(np.unique(y)) < 2: return np.nan
    try: return accuracy_score(yte, WaveGBTSVM(steps=250, **WAVE).fit(C, r, y).predict(F))
    except WDeg: return np.nan


def hard_acc(C, r, y, F, yte):
    if len(C) == 0 or len(np.unique(y)) < 2: return np.nan
    try: return accuracy_score(yte, decide(fit_binary("GBTSVM", GB, np.column_stack([C, r, y])), F))
    except Degenerate: return np.nan


def nc_acc(C, y, F, yte):
    if len(C) == 0 or len(np.unique(y)) < 2: return np.nan
    return accuracy_score(yte, nearest_centroid_predict(C, y, F))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["haberman", "heart", "breast_cancer", "ionosphere"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--kinds", nargs="*", default=["symmetric", "asymmetric"])
    ap.add_argument("--rates", nargs="*", type=float, default=[0.3, 0.4])
    args = ap.parse_args()
    from data.datasets import load
    rows = []; t0 = time.time()
    print(f"pur=1 control seeds={args.seeds} kinds={args.kinds} rates={args.rates}", flush=True)
    for name in args.datasets:
        ds = load(name); X, y = ds["data"][:, :-1], ds["data"][:, -1]
        cls = np.unique(y); pm = lambda v: np.where(v == cls[0], 1.0, -1.0)
        for kind in args.kinds:
            for rate in args.rates:
                for seed in range(args.seeds):
                    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
                        yb = pm(y[tr]).astype(int)
                        yn = inject(X[tr], yb, rate, kind, seed)[0].astype(float)
                        yte = pm(y[te])
                        sc = MinMaxScaler((-1, 1)).fit(X[tr]); Ftr, Fte = sc.transform(X[tr]), sc.transform(X[te])
                        b = gen_balls(np.column_stack([Ftr, yn]), pur=1.0, delbals=1, seed=seed)
                        C, r, yl, _, _ = arrays(b)
                        rows.append(dict(dataset=name, kind=kind, rate=rate, seed=seed, nballs=len(b),
                                         hard=hard_acc(C, r, yl, Fte, yte),
                                         wave=wave_acc(C, r, yl, Fte, yte),
                                         nc=nc_acc(C, yl, Fte, yte)))
        print(f"  done {name} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows); df.to_csv(HERE.parent / "results" / "pur1_nc.csv", index=False)
    print("\n==== pur=1: hard / wave / nearest-centroid ====", flush=True)
    print(f"TỔNG: hard={df.hard.mean():.3f}  wave={df.wave.mean():.3f}  NC={df.nc.mean():.3f}"
          f"  | Δ(wave-NC)={df.wave.mean()-df.nc.mean():+.3f}  bóng TB={df.nballs.mean():.0f}", flush=True)
    for d, s in df.groupby("dataset"):
        print(f"  {d:14s} hard={s.hard.mean():.3f} wave={s.wave.mean():.3f} NC={s.nc.mean():.3f}"
              f"  (bóng {s.nballs.mean():.0f})", flush=True)


if __name__ == "__main__":
    main()
