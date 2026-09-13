"""Training/prediction speed of the five models, with ball counts.

Fixed (untuned) params -- a timing benchmark, not an accuracy run. For each
dataset, average over seeds x 5 folds. Reports ball-generation time and solver
time separately, plus predict time, and how many balls each stopping rule keeps.

    python experiments/bench_speed.py [--datasets ...] [--rate R] [--reps N] [--kernel linear|rbf]

Writes results/<ts>_bench_<kernel>.csv (per dataset x model) and a
results/<ts>_bench_<kernel>.png of train time vs training-set size.
"""
from __future__ import annotations
import sys, pathlib, argparse, warnings, time
warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from config import DATASETS_BINARY
from data.datasets import load, inject_label_noise
from gb.balls import gen_balls, to_matrix, viable
from models.registry import fit_binary, decide, Degenerate
from models.svm_cvxopt import SVM_CVXOPT
from models.wave_gbtsvm import WaveGBTSVM
from models.kernel import RBFMap

GB = dict(d1=0.1, d2=0.1, eps1=0.05, eps2=0.05)
WAVE = dict(c1=10.0, c2=10.0, lam0=1.0, kappa=2.0, adaptive_lambda=True)
MODELS = ("svm", "hard", "binom", "wave", "wave2")


def _arr(b):
    return (np.array([x.center for x in b]), np.array([x.radius for x in b]),
            np.array([x.label for x in b]), np.array([x.purity for x in b]),
            np.array([x.n for x in b]))


def _t(fn):
    t = time.perf_counter(); out = fn(); return out, time.perf_counter() - t


def bench(name, rate, reps, kernel, steps):
    ds = load(name); X, y = ds["data"][:, :-1], ds["data"][:, -1]
    cls = np.unique(y); pm = lambda v: np.where(v == cls[0], 1.0, -1.0)
    rec = {m: {"gen": [], "fit": [], "pred": [], "balls": []} for m in MODELS}
    ntr = []
    for s in range(reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            yp = pm(inject_label_noise(y[tr], rate, np.random.default_rng(s)))
            if len(np.unique(yp)) < 2:
                continue
            sc = MinMaxScaler((-1, 1)).fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            yte = pm(y[te]); ntr.append(len(tr))
            if kernel == "rbf":
                km = RBFMap(seed=s).fit(Xtr); Ftr, Fte = km.transform(Xtr), km.transform(Xte)
            else:
                Ftr, Fte = Xtr, Xte

            m, dt = _t(lambda: SVM_CVXOPT(kernel=kernel, C=10.0).fit(Xtr, yp))
            rec["svm"]["fit"].append(dt); rec["svm"]["gen"].append(0.0)
            _, pt = _t(lambda: m.predict(Xte)); rec["svm"]["pred"].append(pt); rec["svm"]["balls"].append(np.nan)

            sub = np.column_stack([Ftr, yp])
            bp, gp = _t(lambda: gen_balls(sub, pur=1.0, delbals=1, seed=s))
            be, ge = _t(lambda: gen_balls(sub, pur=1.0, delbals=1, seed=s, eta=rate, alpha=0.05))

            for tag, balls, gt in (("hard", bp, gp), ("binom", be, ge)):
                if not viable(balls):
                    continue
                try:
                    W, ft = _t(lambda: fit_binary("GBTSVM", GB, to_matrix(balls)))
                except Degenerate:
                    continue
                _, pt = _t(lambda: decide(W, Fte))
                rec[tag]["gen"].append(gt); rec[tag]["fit"].append(ft)
                rec[tag]["pred"].append(pt); rec[tag]["balls"].append(len(balls))

            for tag, balls, gt in (("wave", bp, gp), ("wave2", be, ge)):
                if not viable(balls):
                    continue
                C, r, yb, p, n = _arr(balls)
                mm, ft = _t(lambda: WaveGBTSVM(steps=steps, **WAVE).fit(C, r, yb, purity=p, size=n))
                _, pt = _t(lambda: mm.predict(Fte))
                rec[tag]["gen"].append(gt); rec[tag]["fit"].append(ft)
                rec[tag]["pred"].append(pt); rec[tag]["balls"].append(len(balls))

    rows = []
    for m in MODELS:
        r = rec[m]
        rows.append(dict(dataset=name, n_train=int(np.mean(ntr)), model=m,
                         gen_ms=1e3 * np.mean(r["gen"]) if r["gen"] else np.nan,
                         fit_ms=1e3 * np.mean(r["fit"]) if r["fit"] else np.nan,
                         train_ms=1e3 * (np.mean(r["gen"]) + np.mean(r["fit"])) if r["fit"] else np.nan,
                         predict_ms=1e3 * np.mean(r["pred"]) if r["pred"] else np.nan,
                         n_balls=float(np.nanmean(r["balls"])) if r["balls"] else np.nan))
    return rows


def main():
    from runlog import save_run
    ap = argparse.ArgumentParser()
    ten = [d for d in DATASETS_BINARY if d != "chess_krvkp"]
    ap.add_argument("--datasets", nargs="*", default=ten)
    ap.add_argument("--rate", type=float, default=0.3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--kernel", choices=["linear", "rbf"], default="linear")
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()

    rows = []
    print(f"{'dataset':13s}{'n':>6}  " + "".join(f"{m:>9}" for m in MODELS) + "   (train ms)")
    for name in args.datasets:
        try:
            r = bench(name, args.rate, args.reps, args.kernel, args.steps)
        except Exception as e:
            print(f"  skip {name}: {type(e).__name__}: {str(e)[:60]}"); continue
        rows += r
        d = {x["model"]: x for x in r}
        print(f"{name:13s}{d['svm']['n_train']:>6}  " +
              "".join(f"{d[m]['train_ms']:>9.1f}" for m in MODELS), flush=True)

    df = pd.DataFrame(rows)
    fn = save_run(df, f"bench_{args.kernel}",
                  dict(kernel=args.kernel, rate=args.rate, reps=args.reps, steps=args.steps,
                       params="svmC=10 GBd=.1 wave lam0=1 kappa=2"))
    ts = pathlib.Path(fn).name[:13]

    # train time vs n
    piv = df.pivot_table(index="n_train", columns="model", values="train_ms")[list(MODELS)].sort_index()
    col = {"svm": "#888", "hard": "#d62728", "binom": "#2ca02c", "wave": "#1f77b4", "wave2": "#9467bd"}
    plt.figure(figsize=(6.4, 4.4))
    for m in MODELS:
        plt.plot(piv.index, piv[m], marker="o", color=col[m], label=m)
    plt.yscale("log"); plt.xlabel("training-set size n"); plt.ylabel("train time (ms, log)")
    plt.title(f"Train time vs n  ({args.kernel}, noise {args.rate})")
    plt.grid(alpha=.3, which="both"); plt.legend(fontsize=9)
    png = str(HERE.parent / "results" / f"{ts}_bench_{args.kernel}.png")
    plt.tight_layout(); plt.savefig(png, dpi=130)
    print(f"saved {fn}\nsaved {png}")


if __name__ == "__main__":
    main()
