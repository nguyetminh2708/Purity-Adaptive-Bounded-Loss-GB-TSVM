"""Plot the two mechanism figures for one diagnostic run.

    python experiments/plot_diagnosis.py [--prefix 2026-09-08_14-30-05]

Defaults to the most recent <timestamp>_diagnosis_*.csv in results/. Figures are
saved with the same prefix.
"""
from __future__ import annotations
import sys, pathlib, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import RESULTS

KIND = "symmetric"


def latest_prefix():
    files = sorted(RESULTS.glob("*_diagnosis_balls.csv"))
    if not files:
        sys.exit("no *_diagnosis_balls.csv in results/ - run run_diagnosis.py first")
    return files[-1].name[: -len("_diagnosis_balls.csv")]


def fig_ball_collapse(prefix):
    df = pd.read_csv(RESULTS / f"{prefix}_diagnosis_balls.csv")
    df = df[df.noise_kind == KIND]
    x = sorted(df.rate.unique())
    g = df.groupby("rate")
    ppb = g["points_per_ball_mean"].mean().reindex(x)
    sing = g["frac_singleton"].mean().reindex(x)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(x, ppb.values, "o-", color="#3B4FD8", lw=2)
    ax1.set_xlabel("label noise rate"); ax1.set_ylabel("points per ball", color="#3B4FD8")
    ax1.tick_params(axis="y", labelcolor="#3B4FD8")
    ax2 = ax1.twinx()
    ax2.plot(x, 100 * sing.values, "s--", color="#CE3556", lw=2)
    ax2.set_ylabel("% singleton balls (<=2 pts)", color="#CE3556")
    ax2.tick_params(axis="y", labelcolor="#CE3556")
    ax1.set_title(f"ball fragmentation vs noise ({KIND}, {df.dataset.nunique()} sets)")
    fig.tight_layout(); fig.savefig(RESULTS / f"{prefix}_fig_ball_collapse.png", dpi=150)


def fig_acc_collapse(prefix):
    df = pd.read_csv(RESULTS / f"{prefix}_diagnosis_acc.csv")
    df = df[df.noise_kind == KIND]
    x = sorted(df.rate.unique())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"SVM": "#3B4FD8", "PinGBTSVM": "#0F9B87", "GBTSVM": "#CE3556"}
    for m, c in colors.items():
        sub = df[df.model == m].groupby("rate")["acc"]
        mu = sub.mean().reindex(x); sd = sub.std().reindex(x).fillna(0)
        ax.plot(x, mu.values, "o-", color=c, lw=2, label=m)
        ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.12)
    ax.set_xlabel("label noise rate"); ax.set_ylabel("accuracy (clean test)")
    ax.set_title(f"accuracy vs noise ({KIND}, {df.dataset.nunique()} sets)")
    ax.legend(); fig.tight_layout(); fig.savefig(RESULTS / f"{prefix}_fig_acc_collapse.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=None)
    args = ap.parse_args()
    prefix = args.prefix or latest_prefix()
    fig_ball_collapse(prefix)
    fig_acc_collapse(prefix)
    print(f"figures written for {prefix}")
