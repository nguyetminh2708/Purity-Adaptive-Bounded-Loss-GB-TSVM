"""Wave-GBTSVM: GBTSVM with the hinge penalty replaced by Wave loss.

Because Wave loss is smooth, the two twin subproblems are unconstrained and are
solved by gradient descent (Adam) instead of a QP - no (A^T A)^-1, no cvxopt.

Plane 1 fits the +1 ball centres and pushes the -1 centres (with their radii)
past the margin; plane 2 is symmetric. For ball j on the opposing side the
violation is u_j = 1 + r_j +/- (c_j . w + b); Wave loss L(u_j; lam_j) caps how
hard that ball can pull the hyperplane. lam_j is either a shared scalar or, when
per-ball trust info is given, adaptive (bounded_loss = high ceiling for clean
balls, low ceiling for suspect ones).
"""
from __future__ import annotations
import numpy as np
from loss.wave import wave_loss, wave_grad, lambda_adaptive


def _adam(grad, z0, lr=0.05, steps=800, b1=0.9, b2=0.999, eps=1e-8):
    z = z0.copy(); m = np.zeros_like(z); v = np.zeros_like(z)
    for t in range(1, steps + 1):
        g = grad(z)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        z -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    return z


class WaveGBTSVM:
    def __init__(self, c1=1.0, c2=1.0, reg=1e-3, a=1.0, lam0=1.0, kappa=2.0,
                 adaptive_lambda=True, lr=0.05, steps=800):
        self.c1, self.c2, self.reg, self.a = c1, c2, reg, a
        self.lam0, self.kappa, self.adaptive_lambda = lam0, kappa, adaptive_lambda
        self.lr, self.steps = lr, steps

    def _lam(self, purity, size, radius):
        if self.adaptive_lambda and purity is not None:
            return lambda_adaptive(purity, size, radius, self.lam0, self.kappa)
        return self.lam0

    def _solve(self, Hown, Gopp, r_opp, c, lam, sign):
        # J = 0.5||Hown z||^2 + 0.5 reg||z||^2 + c * sum L(1 + r_opp + sign*(Gopp z))
        HtH = Hown.T @ Hown
        def grad(z):
            u = 1.0 + r_opp + sign * (Gopp @ z)
            Lp = wave_grad(u, lam, self.a)
            return HtH @ z + self.reg * z + c * sign * (Gopp.T @ Lp)
        z0 = np.zeros(Hown.shape[1])
        return _adam(grad, z0, self.lr, self.steps)

    def fit(self, C, r, y, purity=None, size=None):
        C = np.asarray(C, float); r = np.asarray(r, float); y = np.asarray(y).ravel()
        pos, neg = y == 1, y == -1
        C1, C2, r1, r2 = C[pos], C[neg], r[pos], r[neg]
        e1 = np.ones((len(C1), 1)); e2 = np.ones((len(C2), 1))
        H1 = np.hstack([C1, e1]); H2 = np.hstack([C2, e2])
        G1 = np.hstack([C1, e1]); G2 = np.hstack([C2, e2])
        p = (np.asarray(purity, float) if purity is not None else None)
        s = (np.asarray(size, float) if size is not None else None)
        lam2 = self._lam(p[neg] if p is not None else None, s[neg] if s is not None else None, r2)
        lam1 = self._lam(p[pos] if p is not None else None, s[pos] if s is not None else None, r1)
        # plane 1: fit +1, push -1 with +sign; plane 2: fit -1, push +1 with -sign
        self.z1 = self._solve(H1, G2, r2, self.c1, lam2, +1.0)
        self.z2 = self._solve(H2, G1, r1, self.c2, lam1, -1.0)
        return self

    def _f(self, X, z):
        return X @ z[:-1] + z[-1]

    def predict(self, X):
        X = np.asarray(X, float)
        d1 = np.abs(self._f(X, self.z1)); d2 = np.abs(self._f(X, self.z2))
        return np.where(d1 <= d2, 1, -1)
