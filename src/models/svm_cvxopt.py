"""Soft-margin C-SVM solved with the same CVXOPT QP backend as the twin models.

Dual: min 0.5 a'Qa - 1'a  s.t. 0<=a<=C, y'a=0, with Q = (yy') .* K.
Kernel: 'linear' or 'rbf'. Used to compare fairly against a self-coded SVM
instead of a tuned library SVC.
"""
from __future__ import annotations
import numpy as np
from cvxopt import matrix, solvers
solvers.options["show_progress"] = False


def _kernel(A, B, kind, gamma):
    if kind == "linear":
        return A @ B.T
    sq = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-gamma * np.clip(sq, 0, None))


class SVM_CVXOPT:
    def __init__(self, C=1.0, kernel="rbf", gamma="scale"):
        self.C, self.kernel, self.gamma = C, kernel, gamma

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float).ravel()
        n = len(y)
        g = self.gamma
        if g == "scale":
            g = 1.0 / (X.shape[1] * X.var()) if X.var() > 0 else 1.0 / X.shape[1]
        self._g = g
        K = _kernel(X, X, self.kernel, g)
        Q = np.outer(y, y) * K
        P = matrix(Q + 1e-8 * np.eye(n)); q = matrix(-np.ones(n))
        G = matrix(np.vstack([-np.eye(n), np.eye(n)]))
        h = matrix(np.hstack([np.zeros(n), self.C * np.ones(n)]))
        A = matrix(y.reshape(1, -1)); b = matrix(0.0)
        a = np.array(solvers.qp(P, q, G, h, A, b)["x"]).ravel()
        sv = a > 1e-5
        self.a, self.sv_X, self.sv_y = a[sv], X[sv], y[sv]
        free = sv & (a < self.C - 1e-5)
        if free.any():
            Kf = _kernel(X[free], self.sv_X, self.kernel, g)
            self.b = np.mean(y[free] - (self.a * self.sv_y) @ Kf.T)
        else:
            self.b = 0.0
        return self

    def decision_function(self, X):
        K = _kernel(np.asarray(X, float), self.sv_X, self.kernel, self._g)
        return (self.a * self.sv_y) @ K.T + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)
