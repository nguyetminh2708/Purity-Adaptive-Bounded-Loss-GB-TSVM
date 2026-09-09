"""Empirical RBF kernel map: X -> [K(x, L_1), ..., K(x, L_m)] over landmarks L.

Lets the linear twin models (GBTSVM, Wave-GBTSVM) run in an RBF feature space,
matching how the essay and the Tanveer kernel variants kernelize twin models,
so they can be compared fairly against an RBF SVM.
"""
from __future__ import annotations
import numpy as np


class RBFMap:
    def __init__(self, n_landmarks=300, gamma="scale", seed=0):
        self.m, self.gamma, self.seed = n_landmarks, gamma, seed

    def fit(self, X):
        X = np.asarray(X, float)
        self.g = (1.0 / (X.shape[1] * X.var()) if self.gamma == "scale" and X.var() > 0
                  else (self.gamma if self.gamma != "scale" else 1.0 / X.shape[1]))
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(X), min(self.m, len(X)), replace=False)
        self.L = X[idx]
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        sq = (X**2).sum(1)[:, None] + (self.L**2).sum(1)[None, :] - 2 * X @ self.L.T
        return np.exp(-self.g * np.clip(sq, 0, None))

    def fit_transform(self, X):
        return self.fit(X).transform(X)
