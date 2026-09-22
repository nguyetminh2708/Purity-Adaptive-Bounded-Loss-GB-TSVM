"""Tự động dò siêu tham số cho Wave-GBTSVM (λ CỐ ĐỊNH) bằng metaheuristic (mealpy),
so với grid-search trên cùng hàm mục tiêu.

linear: tune (C, λ, reg)          — 3 chiều
rbf   : tune (C, λ, reg, γ)       — 4 chiều (γ = bề rộng RBF; refit map mỗi lần chấm)

C=c1=c2 trọng số phạt · λ=lam0 trần phạt (cố định, kappa=0) · reg = điều chuẩn
Tikhonov (vai trò eps) · γ = bề rộng Gaussian của ánh xạ RBF.
Mục tiêu = độ chính xác CV vòng-trong (3-fold) trên nhãn NHIỄU; đánh giá cuối trên
tập test SẠCH.

    python experiments/tune_meta.py --kernel rbf [--datasets heart sonar]
        [--algo WOA|PSO] [--kind symmetric] [--rate 0.3] [--epoch 25] [--pop 20] [--steps 150]

Novelty: tầng tối ưu SIÊU THAM SỐ (vòng ngoài) — tiện ích auto-tuning, KHÔNG phải
đóng góp chính. Metaheuristic chỉ đáng khi không gian đủ lớn (grid nổ tổ hợp).
"""
from __future__ import annotations
import sys, pathlib, argparse, warnings, time, logging
from itertools import product
warnings.filterwarnings("ignore")
logging.disable(logging.INFO)   # tắt log INFO của mealpy
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from mealpy import FloatVar
from mealpy.swarm_based.WOA import OriginalWOA
from mealpy.swarm_based.PSO import OriginalPSO
from config import DATASETS_BINARY
from data.datasets import load
from gb.balls import gen_balls, arrays, viable
from models.wave_gbtsvm import WaveGBTSVM, Degenerate as WDeg
from models.kernel import RBFMap
from noise import inject

# lưới grid tham chiếu
GRID_C   = [1.0, 10.0]
GRID_LAM = [0.25, 0.5, 1.0, 2.0, 4.0]
GRID_REG = [0.01, 0.05]
GRID_G   = [0.01, 0.1, 1.0]          # chỉ dùng cho rbf
# biên metaheuristic: linear x=[C,λ,log10reg] ; rbf thêm log10γ
LB_LIN, UB_LIN = [0.1, 0.1, -4.0], [50.0, 8.0, -1.0]
LB_RBF, UB_RBF = [0.1, 0.1, -4.0, -3.0], [50.0, 8.0, -1.0, 1.0]


def _decode(x, kernel):
    C, lam, reg = x[0], x[1], 10.0 ** x[2]
    gamma = (10.0 ** x[3]) if kernel == "rbf" else None
    return C, lam, reg, gamma


def _feat(Xfit, Xother, kernel, gamma, seed):
    if kernel == "rbf":
        km = RBFMap(gamma=gamma, seed=seed).fit(Xfit)
        return km.transform(Xfit), km.transform(Xother)
    return Xfit, Xother


def _wave_cv(Xtr, ytr, C, lam, reg, gamma, kernel, steps, folds=3, seed=0):
    accs = []
    for itr, iva in StratifiedKFold(folds, shuffle=True, random_state=seed).split(Xtr, ytr):
        if len(np.unique(ytr[itr])) < 2:
            continue
        Fi, Fv = _feat(Xtr[itr], Xtr[iva], kernel, gamma, seed)
        sub = np.column_stack([Fi, ytr[itr]])
        b = gen_balls(sub, pur=1.0, delbals=1, seed=seed)
        if not viable(b):
            continue
        Cc, r, yb, p, n = arrays(b)
        try:
            m = WaveGBTSVM(c1=C, c2=C, reg=reg, lam0=lam, kappa=0.0,
                           adaptive_lambda=False, steps=steps).fit(Cc, r, yb, purity=p, size=n)
        except WDeg:
            continue
        accs.append(accuracy_score(ytr[iva], m.predict(Fv)))
    return float(np.mean(accs)) if accs else 0.0


def _wave_test(Xtr, ytr, Xte, yte, C, lam, reg, gamma, kernel, steps, seed=0):
    Ftr, Fte = _feat(Xtr, Xte, kernel, gamma, seed)
    sub = np.column_stack([Ftr, ytr])
    b = gen_balls(sub, pur=1.0, delbals=1, seed=seed)
    if not viable(b):
        return np.nan
    Cc, r, yb, p, n = arrays(b)
    try:
        m = WaveGBTSVM(c1=C, c2=C, reg=reg, lam0=lam, kappa=0.0,
                       adaptive_lambda=False, steps=steps).fit(Cc, r, yb, purity=p, size=n)
    except WDeg:
        return np.nan
    return accuracy_score(yte, m.predict(Fte))


def run_dataset(name, kind, rate, kernel, algo, epoch, pop, steps, seed=0):
    ds = load(name); X, y = ds["data"][:, :-1], ds["data"][:, -1]
    cls = np.unique(y); pm = lambda v: np.where(v == cls[0], 1.0, -1.0)
    (tr, te), = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))[:1]
    yb = pm(y[tr]).astype(int)
    yn = inject(X[tr], yb, rate, kind, seed)[0].astype(float)
    yte = pm(y[te])
    sc = MinMaxScaler((-1, 1)).fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])

    # ---- metaheuristic ----
    n_eval = {"c": 0}
    def obj(x):
        n_eval["c"] += 1
        C, lam, reg, gamma = _decode(x, kernel)
        return _wave_cv(Xtr, yn, C, lam, reg, gamma, kernel, steps, seed=seed)
    lb, ub = (LB_RBF, UB_RBF) if kernel == "rbf" else (LB_LIN, UB_LIN)
    problem = {"obj_func": obj, "bounds": FloatVar(lb=lb, ub=ub), "minmax": "max"}
    Model = {"WOA": OriginalWOA, "PSO": OriginalPSO}[algo]
    model = Model(epoch=epoch, pop_size=pop, log_to=None)
    t0 = time.time()
    g = model.solve(problem, seed=seed)
    t_meta = time.time() - t0
    mC, mlam, mreg, mg = _decode(g.solution, kernel)
    meta_test = _wave_test(Xtr, yn, Xte, yte, mC, mlam, mreg, mg, kernel, steps, seed)
    meta_par = (round(mC, 2), round(mlam, 2), round(mreg, 4)) + ((round(mg, 4),) if kernel == "rbf" else ())

    # ---- grid tham chiếu ----
    grid_G = GRID_G if kernel == "rbf" else [None]
    best = (-1, None); g_evals = 0
    for C, lam, reg, gamma in product(GRID_C, GRID_LAM, GRID_REG, grid_G):
        g_evals += 1
        cv = _wave_cv(Xtr, yn, C, lam, reg, gamma, kernel, steps, seed=seed)
        if cv > best[0]:
            best = (cv, (C, lam, reg, gamma))
    gC, glam, greg, gg = best[1]
    grid_test = _wave_test(Xtr, yn, Xte, yte, gC, glam, greg, gg, kernel, steps, seed)
    grid_par = (gC, glam, greg) + ((gg,) if kernel == "rbf" else ())

    return dict(name=name, meta_test=meta_test, meta_evals=n_eval["c"], meta_par=meta_par,
                t_meta=t_meta, grid_test=grid_test, grid_evals=g_evals, grid_par=grid_par)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["heart", "sonar"])
    ap.add_argument("--kernel", choices=["linear", "rbf"], default="linear")
    ap.add_argument("--algo", choices=["WOA", "PSO"], default="WOA")
    ap.add_argument("--kind", default="symmetric")
    ap.add_argument("--rate", type=float, default=0.3)
    ap.add_argument("--epoch", type=int, default=25)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()
    print(f"kernel={args.kernel} algo={args.algo} epoch={args.epoch} pop={args.pop} "
          f"seeds={args.seeds} | nhiễu {args.kind} {args.rate}\n", flush=True)
    if args.seeds == 1:
        print(f"{'dataset':11s}{'meta_test':>10}{'grid_test':>10}{'meta_evals':>11}{'grid_evals':>11}"
              f"{'t_meta(s)':>10}   meta_par | grid_par", flush=True)
        for name in args.datasets:
            r = run_dataset(name, args.kind, args.rate, args.kernel, args.algo,
                            args.epoch, args.pop, args.steps)
            print(f"{r['name']:11s}{r['meta_test']:>10.3f}{r['grid_test']:>10.3f}"
                  f"{r['meta_evals']:>11d}{r['grid_evals']:>11d}{r['t_meta']:>10.0f}   "
                  f"{r['meta_par']} | {r['grid_par']}", flush=True)
        return
    # ---- multi-seed: trung bình meta vs grid qua các seed ----
    for name in args.datasets:
        mts, gts, per = [], [], []
        for seed in range(args.seeds):
            try:
                r = run_dataset(name, args.kind, args.rate, args.kernel, args.algo,
                                args.epoch, args.pop, args.steps, seed=seed)
                mts.append(r["meta_test"]); gts.append(r["grid_test"])
                per.append(f"{r['meta_test']:.2f}/{r['grid_test']:.2f}")
                print(f"  {name:11s} seed{seed}: meta {r['meta_test']:.3f} | grid {r['grid_test']:.3f}"
                      f"  ({r['t_meta']:.0f}s)", flush=True)
            except Exception as e:
                print(f"  {name} seed{seed} ERR: {type(e).__name__}: {str(e)[:60]}", flush=True)
        if mts:
            mts, gts = np.array(mts), np.array(gts)
            wins = int((mts > gts).sum())
            print(f"==> {name:11s} meta_TB {np.nanmean(mts):.3f} | grid_TB {np.nanmean(gts):.3f}"
                  f" | Δ {np.nanmean(mts)-np.nanmean(gts):+.3f} | meta>grid {wins}/{len(mts)}\n", flush=True)


if __name__ == "__main__":
    main()
