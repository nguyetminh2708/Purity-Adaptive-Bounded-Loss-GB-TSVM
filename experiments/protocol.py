"""
Evaluation protocol: nested CV 5x3 + random search.

Three things plug the holes left by grid_search_optimization.ipynb:

  L2  (pur, num) travel TOGETHER with the model parameters inside one candidate,
      so they are selected in the INNER loop instead of by comparing test_score.
  L3  gen_balls is called INSIDE each fold, on that fold's training part only.
  L4  MinMaxScaler is fitted on that fold's training part only.

Plus: StratifiedKFold instead of a positional slice, and one row returned per
outer fold carrying that fold's own theta — this is where std comes from.

sklearn Pipeline cannot express this: a sklearn transformer is not allowed to
change the number of rows, and ball generation turns n samples into m balls
(iris: 120 -> ~11).
"""

import time

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score

import sys, pathlib as _pl
_here = _pl.Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent / "src"))
from gb import balls as B
from models import registry as M

PUR_IDX = (1, 2, 5, 7, 10)      # actual purity = 1 - 0.015 * idx, as in the notebook
NUM_VALS = (1, 2, 3, 4, 5)


def purity_of(idx):
    return 1.0 - 0.015 * idx


# ----------------------------------------------------------------------------
# random search
# ----------------------------------------------------------------------------

def _logu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_candidates(kind, n, seed, pur_choices=PUR_IDX, num_choices=NUM_VALS):
    """
    Sample log-uniformly over a continuous range instead of snapping to 7
    discrete values.

    A full 1225-cell grid still only tries 7 distinct values of d1; n=200 points
    try 200 distinct values, at roughly 1/6 of the cost (Bergstra & Bengio 2012).

    pur_choices/num_choices can be narrowed for large datasets: ball-generation
    cost scales with the NUMBER OF (pur, num) COMBINATIONS, not with n, because
    the cache is keyed on (fold, pur, num). Any narrowing must be recorded in
    the results rather than silently applied.
    """
    rng = np.random.RandomState(seed)
    keys = M.PARAMS[kind]
    out = []
    for _ in range(n):
        p = {"d1": _logu(rng, 1e-4, 1e1), "d2": _logu(rng, 1e-4, 1e1)}
        if "d3" in keys:                       # LGBTSVM
            p["d3"] = _logu(rng, 1e-4, 1e1)
            p["d4"] = _logu(rng, 1e-4, 1e1)
        if "eps1" in keys:
            p["eps1"] = _logu(rng, 1e-4, 1e0)
            p["eps2"] = _logu(rng, 1e-4, 1e0)
        if "tau" in keys:                      # Pin-GBTSVM
            p["tau"] = float(rng.uniform(0.05, 1.0))
        out.append({
            "params": p,
            "pur_idx": int(rng.choice(pur_choices)),
            "num": int(rng.choice(num_choices)),
        })
    return out


# ----------------------------------------------------------------------------
# cache: balls depend on (fold, pur, num) only — NOT on d1/d2/eps
# ----------------------------------------------------------------------------

class Cache:
    """Without this, 200 candidates regenerate the same ball set 200 times."""

    def __init__(self):
        self.scal, self.ball = {}, {}
        self.hits = self.miss = 0

    def scaled(self, fold_key, train_raw, eval_raw):
        if fold_key not in self.scal:
            sc = MinMaxScaler((-1, 1)).fit(train_raw[:, :-1])
            self.scal[fold_key] = sc
        sc = self.scal[fold_key]
        tr = np.column_stack([sc.transform(train_raw[:, :-1]), train_raw[:, -1]])
        ev = np.column_stack([sc.transform(eval_raw[:, :-1]), eval_raw[:, -1]])
        return tr, ev

    def balls(self, key, tr, pur, num, seed):
        if key in self.ball:
            self.hits += 1
            return self.ball[key]
        self.miss += 1
        t0 = time.perf_counter()
        bl = B.gen_balls(tr, pur=pur, delbals=num, seed=seed)
        v = (bl if B.viable(bl) else None, time.perf_counter() - t0)
        self.ball[key] = v
        return v


# ----------------------------------------------------------------------------
# one fit + score
# ----------------------------------------------------------------------------

def fit_eval(kind, strategy, cand, train_raw, eval_raw, seed,
             fold_key, cache, per_subproblem=False):
    """
    train_raw, eval_raw: (., d+1) RAW samples, label in the last column.
    Returns a dict of metrics, or None if the config is unusable.
    """
    pur, num = purity_of(cand["pur_idx"]), cand["num"]
    tr, ev = cache.scaled(fold_key, train_raw, eval_raw)
    ev_X, ev_y = ev[:, :-1], ev[:, -1]

    try:
        if per_subproblem:                                   # phase B
            t0 = time.perf_counter()
            labels, mdl, counts = M.fit_multiclass_per_subproblem(
                kind, cand["params"], tr, strategy, pur, num, seed,
                B.gen_balls, B.to_matrix, B.viable)
            t_fit, n_balls, t_gen = time.perf_counter() - t0, float(np.mean(counts)), None
        else:                                                # phase A
            bl, t_gen = cache.balls((fold_key, cand["pur_idx"], num), tr, pur, num, seed)
            if bl is None:
                return None
            n_balls = len(bl)
            t0 = time.perf_counter()
            labels, mdl = M.fit_multiclass(kind, cand["params"], B.to_matrix(bl), strategy)
            t_fit = time.perf_counter() - t0
    except M.Degenerate:
        return None
    except Exception:                       # unexpected linalg/cvxopt failure
        return None

    y_pred = M.predict_multiclass(labels, mdl, ev_X, strategy)
    return {
        "acc": float(accuracy_score(ev_y, y_pred)),
        "macro_f1": float(f1_score(ev_y, y_pred, average="macro", zero_division=0)),
        "n_balls": float(n_balls),
        "t_gen": t_gen,
        "t_fit": float(t_fit),
    }


# ----------------------------------------------------------------------------
# nested CV
# ----------------------------------------------------------------------------

def nested_cv(kind, strategy, data, n_out=5, n_in=3, n_iter=200, seed=0,
              per_subproblem=False, verbose=False, cache=None,
              pur_choices=PUR_IDX, num_choices=NUM_VALS):
    """
    Returns a list of n_out dicts, one per outer fold:
        fold, theta, cv (inner-loop score), test, macro_f1, n_balls, t_*
    The outer fold's test set takes part in NO selection whatsoever.

    `cache` may be shared across variants for the SAME (data, seed, n_out, n_in):
    the ball key is (outer_fold, inner_fold, pur_idx, num), so it does not depend
    on d1/d2/eps. Sharing cuts ball-generation cost by roughly 6x across
    3 variants x 2 strategies.
    """
    X, y = data[:, :-1], data[:, -1]
    outer = StratifiedKFold(n_out, shuffle=True, random_state=seed)
    cache, rows = (cache if cache is not None else Cache()), []

    for k, (tr, te) in enumerate(outer.split(X, y)):
        d_tr = data[tr]
        inner = StratifiedKFold(n_in, shuffle=True, random_state=seed + k)
        splits = list(inner.split(d_tr[:, :-1], d_tr[:, -1]))

        best, n_ok = (-np.inf, None), 0
        for cand in sample_candidates(kind, n_iter, seed + 1000 * k,
                                      pur_choices, num_choices):
            sc = []
            for j, (itr, iva) in enumerate(splits):
                r = fit_eval(kind, strategy, cand, d_tr[itr], d_tr[iva], seed,
                             (k, j), cache, per_subproblem)
                if r is None:
                    break            # a candidate must survive EVERY inner fold
                sc.append(r["acc"])
            if len(sc) == len(splits):
                n_ok += 1
                if np.mean(sc) > best[0]:
                    best = (float(np.mean(sc)), cand)

        if best[1] is None:
            rows.append({"fold": k, "theta": None, "cv": None, "test": None,
                         "n_viable": 0})
            continue

        out = fit_eval(kind, strategy, best[1], d_tr, data[te], seed,
                       (k, "final"), cache, per_subproblem)
        rows.append({
            "fold": k, "theta": best[1], "cv": best[0], "n_viable": n_ok,
            "test": None if out is None else out["acc"],
            "macro_f1": None if out is None else out["macro_f1"],
            "n_balls": None if out is None else out["n_balls"],
            "t_fit": None if out is None else out["t_fit"],
        })
        if verbose:
            print(f"  fold {k}: cv={best[0]:.4f} test={rows[-1]['test']} "
                  f"usable candidates={n_ok}/{n_iter}")

    return rows


def summarize(rows):
    t = [r["test"] for r in rows if r.get("test") is not None]
    f = [r["macro_f1"] for r in rows if r.get("macro_f1") is not None]
    b = [r["n_balls"] for r in rows if r.get("n_balls") is not None]
    return {
        "n_folds_ok": len(t),
        "acc_mean": float(np.mean(t)) if t else None,
        "acc_std": float(np.std(t)) if t else None,
        "f1_mean": float(np.mean(f)) if f else None,
        "balls_mean": float(np.mean(b)) if b else None,
    }
