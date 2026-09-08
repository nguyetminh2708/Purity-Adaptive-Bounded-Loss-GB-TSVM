"""Diagnostic sweep: how GBTSVM degrades under label noise.

Writes results/diagnosis_balls.csv (ball counts/sizes) and
results/diagnosis_acc.csv (SVM / GBTSVM / PinGBTSVM accuracy), over the four
noise kinds and the configured rates. Plot with plot_diagnosis.py.

    python experiments/run_diagnosis.py [--quick] [--datasets ...]
"""
from __future__ import annotations
import sys, pathlib, argparse, warnings
from datetime import datetime
warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from config import SEEDS, NOISE_RATES, NOISE_KINDS, PURITY_DEFAULT, N_FOLDS, RESULTS, DATASETS_BINARY
from data.datasets import load
from gb.balls import gen_balls, to_matrix, viable
from models.registry import fit_binary, decide, Degenerate
from noise import inject

GB_PARAMS = dict(d1=0.1, d2=0.1, eps1=0.05, eps2=0.05)
PIN_PARAMS = dict(d1=0.1, d2=0.1, eps1=0.05, eps2=0.05, tau=0.5)


def ball_stats(balls):
    n = np.array([b.n for b in balls]); p = np.array([b.purity for b in balls])
    return dict(n_balls=len(balls), points_per_ball_mean=float(n.mean()),
                frac_singleton=float((n <= 2).mean()), purity_mean=float(p.mean()),
                radius_mean=float(np.mean([b.radius for b in balls])))


def _gb_acc(kind, params, Xtr, ytr, Xte, yte, seed):
    balls = gen_balls(np.column_stack([Xtr, ytr]), pur=PURITY_DEFAULT, delbals=1, seed=seed)
    if not viable(balls):
        return np.nan
    try:
        W = fit_binary(kind, params, to_matrix(balls))
    except Degenerate:
        return np.nan
    return accuracy_score(yte, decide(W, Xte))


def diagnose(name, seeds, rates):
    ds = load(name)
    X, y = ds["data"][:, :-1], ds["data"][:, -1]
    cls = np.unique(y)
    assert len(cls) == 2, f"{name} is not binary: {cls}"
    to_pm = lambda v: np.where(v == cls[0], 1.0, -1.0)
    y_pm = to_pm(y)

    ball_rows, acc_rows = [], []
    for kind in NOISE_KINDS:
        for rate in rates:
            if rate == 0.0 and kind != "symmetric":
                continue
            for seed in seeds:
                yn_all, flipped = inject(X, y_pm, rate, kind, seed)
                Xs = MinMaxScaler((-1, 1)).fit_transform(X)
                balls = gen_balls(np.column_stack([Xs, yn_all]), pur=PURITY_DEFAULT, delbals=1, seed=seed)
                if viable(balls):
                    st = ball_stats(balls)
                    st.update(dataset=name, noise_kind=kind, rate=rate, seed=seed,
                              n_flipped=int(flipped.sum()))
                    ball_rows.append(st)

                skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
                a = {"SVM": [], "GBTSVM": [], "PinGBTSVM": []}
                for tr, te in skf.split(X, y):
                    yn_tr, _ = inject(X[tr], y_pm[tr], rate, kind, seed)
                    if len(np.unique(yn_tr)) < 2:
                        continue
                    sc = MinMaxScaler((-1, 1)).fit(X[tr])
                    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
                    yte = y_pm[te]
                    a["SVM"].append(accuracy_score(yte, SVC(kernel="rbf").fit(Xtr, yn_tr).predict(Xte)))
                    a["GBTSVM"].append(_gb_acc("GBTSVM", GB_PARAMS, Xtr, yn_tr, Xte, yte, seed))
                    a["PinGBTSVM"].append(_gb_acc("PinGBTSVM", PIN_PARAMS, Xtr, yn_tr, Xte, yte, seed))
                for m, v in a.items():
                    acc_rows.append(dict(dataset=name, noise_kind=kind, rate=rate,
                                         seed=seed, model=m, acc=float(np.nanmean(v))))
            done = [r for r in acc_rows if r["dataset"] == name and r["noise_kind"] == kind and r["rate"] == rate]
            sv = np.nanmean([r["acc"] for r in done if r["model"] == "SVM"])
            gb = np.nanmean([r["acc"] for r in done if r["model"] == "GBTSVM"])
            print(f"  {name:13s} {kind:10s} rate={rate:.1f}  SVM={sv:.3f}  GBTSVM={gb:.3f}", flush=True)
    return ball_rows, acc_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--tag", default="", help="optional label appended to the run folder")
    args = ap.parse_args()
    seeds = SEEDS[:3] if args.quick else SEEDS
    rates = [0.0, 0.2, 0.4] if args.quick else NOISE_RATES
    names = args.datasets or (["balance_scale"] if args.quick else DATASETS_BINARY)
    print(f"datasets={names} seeds={len(seeds)} rates={rates}")

    ball_all, acc_all = [], []
    for name in names:
        try:
            b, a = diagnose(name, seeds, rates)
            ball_all += b; acc_all += a
        except Exception as e:
            print(f"  skip {name}: {type(e).__name__}: {str(e)[:70]}")
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    prefix = ts + (f"_{args.tag}" if args.tag else "")
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ball_all).to_csv(RESULTS / f"{prefix}_diagnosis_balls.csv", index=False)
    pd.DataFrame(acc_all).to_csv(RESULTS / f"{prefix}_diagnosis_acc.csv", index=False)
    print(f"wrote {prefix}_diagnosis_*.csv to results/")


if __name__ == "__main__":
    main()
