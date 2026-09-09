"""Binomial stopping rule vs hard purity for GBTSVM under label noise (no Wave yet).

Three ball-generation rules, same GBTSVM classifier and folds:
  hard    purity == 1 (original)
  oracle  binomial rule with eta set to the true injected noise rate (upper bound)
  eta<v>  binomial rule with a fixed assumed eta (realistic: rate unknown)

    python experiments/run_binomial.py [--datasets ...] [--seeds N] [--eta 0.2]

Writes results/<ts>_binomial.csv (accuracy and ball count per dataset/rate/rule).
"""
from __future__ import annotations
import sys, pathlib, argparse, warnings
from datetime import datetime
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from config import RESULTS, DATASETS_BINARY
from data.datasets import load, inject_label_noise
from gb.balls import gen_balls, to_matrix, viable
from models.registry import fit_binary, decide, Degenerate

GB = dict(d1=0.1, d2=0.1, eps1=0.05, eps2=0.05)


def gb_eval(cfg, sub, Xte, yte, seed):
    balls = gen_balls(sub, pur=1.0, delbals=1, seed=seed, **cfg)
    if not viable(balls):
        return np.nan, len(balls)
    try:
        W = fit_binary("GBTSVM", GB, to_matrix(balls))
    except Degenerate:
        return np.nan, len(balls)
    return accuracy_score(yte, decide(W, Xte)), len(balls)


def run(name, seeds, rates, eta_fixed):
    ds = load(name); X, y = ds["data"][:, :-1], ds["data"][:, -1]
    cls = np.unique(y); to_pm = lambda v: np.where(v == cls[0], 1.0, -1.0)
    rows = []
    for rate in rates:
        rules = {"hard": dict(eta=None),
                 "oracle": dict(eta=max(rate, 1e-9), alpha=0.05),
                 f"eta{eta_fixed}": dict(eta=eta_fixed, alpha=0.05)}
        agg = {k: ([], []) for k in rules}
        for seed in seeds:
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
                ytr = inject_label_noise(y[tr], rate, np.random.default_rng(seed))
                if len(np.unique(ytr)) < 2:
                    continue
                sc = MinMaxScaler((-1, 1)).fit(X[tr])
                sub = np.column_stack([sc.transform(X[tr]), to_pm(ytr)])
                Xte, yte = sc.transform(X[te]), to_pm(y[te])
                for k, cf in rules.items():
                    a, nb = gb_eval(cf, sub, Xte, yte, seed)
                    agg[k][0].append(a); agg[k][1].append(nb)
        for k in rules:
            rows.append(dict(dataset=name, rate=rate, rule=k,
                             acc=float(np.nanmean(agg[k][0])), balls=float(np.mean(agg[k][1]))))
        print(f"  {name:13s} rate={rate:.1f}  " +
              "  ".join(f"{k}={np.nanmean(agg[k][0]):.3f}({np.mean(agg[k][1]):.0f})" for k in rules), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS_BINARY)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rates", nargs="*", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--eta", type=float, default=0.2)
    args = ap.parse_args()
    rows = []
    for name in args.datasets:
        try:
            rows += run(name, range(args.seeds), args.rates, args.eta)
        except Exception as e:
            print(f"  skip {name}: {type(e).__name__}: {str(e)[:70]}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    out = RESULTS / f"{ts}_binomial.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
