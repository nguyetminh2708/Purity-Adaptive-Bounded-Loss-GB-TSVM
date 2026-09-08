"""GD1 - Thi nghiem chan doan co che: bong vo vun theo muc nhieu.

Chay:  python experiments/run_diagnosis.py
Ra:    results/diagnosis_balls.csv  (moi dong: dataset x kieu nhieu x muc x seed)

Day la bang chung dong co cua bai: neu so bong tang / diem-tren-bong giam
don dieu theo muc nhieu thi co che "nhan da so nuot nhieu" cua GBTSVM
khong con ton tai o nhieu cao.
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from gb.granular_ball import generate_balls, ball_stats
from noise import inject, NOISE_KINDS

SEEDS = range(10)
RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
PURITY_T = 1.0          # nhu GBTSVM goc; chay them 0.9 / 0.8 de doi chieu


def datasets():
    d = load_breast_cancer()
    X = StandardScaler().fit_transform(d.data)
    y = np.where(d.target == 1, 1, -1)
    yield "wdbc", X, y
    # TODO GD1: them Balance Scale, Haberman, Cleveland... (12 bo theo lo trinh)


def main():
    rows = []
    for name, X, y in datasets():
        for kind in NOISE_KINDS:
            for rate in RATES:
                if rate == 0.0 and kind != "symmetric":
                    continue
                for seed in SEEDS:
                    yn, flipped = inject(X, y, rate, kind, seed)
                    balls = generate_balls(X, yn, PURITY_T, seed=seed)
                    st = ball_stats(balls)
                    st.update(dataset=name, noise_kind=kind, rate=rate, seed=seed,
                              n_points=len(y), n_flipped=int(flipped.sum()))
                    rows.append(st)
                r = [x for x in rows if x["dataset"] == name and x["noise_kind"] == kind and x["rate"] == rate]
                m = np.mean([x["n_balls"] for x in r])
                print(f"{name:8s} {kind:10s} rate={rate:.1f}  n_balls={m:7.1f}")
    out = pathlib.Path(__file__).resolve().parents[1] / "results" / "diagnosis_balls.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("->", out)


if __name__ == "__main__":
    main()
