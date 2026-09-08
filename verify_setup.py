"""Sanity checks for the ported stack. Runs offline.

    python verify_setup.py
"""
import sys, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from config import SEEDS, NOISE_RATES, NOISE_KINDS, DATASETS_BINARY, PURITY_DEFAULT
from loss.wave import numerical_gradient_check, wave_loss, lambda_adaptive
from gb.balls import gen_balls, to_matrix, viable, should_split, Ball
from data.datasets import load
from models.registry import fit_binary, decide
from noise import inject


def check(name, fn):
    try:
        info = fn()
        print(f"  ok    {name}" + (f"  {info}" if info else "")); return True
    except Exception as e:
        print(f"  FAIL  {name}  {type(e).__name__}: {e}"); return False


def main():
    print(f"binary sets {DATASETS_BINARY}, {len(SEEDS)} seeds, rates {NOISE_RATES}")
    print(f"noise kinds {NOISE_KINDS}")
    oks = []

    oks.append(check("wave gradient",
                     lambda: f"max err {max(numerical_gradient_check(lam=0.7), numerical_gradient_check(lam=np.full(64,2.0),a=1.5)):.1e}"))
    oks.append(check("wave bounded (u=50 -> 1/lam)",
                     lambda: f"{np.allclose(wave_loss(np.array([50.0]),0.5)[0],2.0,atol=1e-2)}"))

    ds = load("balance_scale"); X, y = ds["data"][:, :-1], ds["data"][:, -1]
    cls = np.unique(y); to_pm = lambda v: np.where(v == cls[0], 1.0, -1.0)
    oks.append(check("load balance_scale", lambda: f"X{X.shape} classes {cls}"))
    oks.append(check("load breast_cancer", lambda: f"X{load('breast_cancer')['data'].shape}"))

    def balls_lambda():
        sub = np.column_stack([MinMaxScaler((-1, 1)).fit_transform(X), to_pm(y)])
        balls = gen_balls(sub, pur=PURITY_DEFAULT, delbals=1, seed=0)
        p = np.array([b.purity for b in balls]); n = np.array([b.n for b in balls])
        r = np.array([b.radius for b in balls])
        lam = lambda_adaptive(p, n, r)
        return f"{len(balls)} balls, lambda in [{lam.min():.2f},{lam.max():.2f}]"
    oks.append(check("gen_balls + lambda_adaptive", balls_lambda))

    def binom_rule():
        b = Ball(np.column_stack([X[:50], to_pm(y[:50]), np.arange(50)]))
        return f"hard={should_split(b, pur=1.0, eta=None)} binomial={should_split(b, eta=0.1, alpha=0.05)}"
    oks.append(check("binomial stopping rule", binom_rule))

    oks.append(check("noise injection",
                     lambda: {k: int(inject(X, to_pm(y), 0.2, k, 0)[1].sum()) for k in NOISE_KINDS}))

    def gbtsvm_real():
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        tr, te = next(iter(skf.split(X, y)))
        sc = MinMaxScaler((-1, 1)).fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        balls = gen_balls(np.column_stack([Xtr, to_pm(y[tr])]), pur=1.0, delbals=1, seed=0)
        W = fit_binary("GBTSVM", dict(d1=0.1, d2=0.1, eps1=0.05, eps2=0.05), to_matrix(balls))
        return f"acc {accuracy_score(to_pm(y[te]), decide(W, Xte)):.3f}"
    oks.append(check("GBTSVM fit/predict", gbtsvm_real))

    print(f"{sum(oks)}/{len(oks)} ok")
    return 0 if all(oks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
