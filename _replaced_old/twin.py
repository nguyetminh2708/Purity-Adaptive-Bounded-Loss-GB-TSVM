"""Baseline twin SVM tuyen tinh: TWSVM va GBTSVM goc (Quadir, Sajid, Tanveer 2025).

Muc dich: lam MOC so sanh cho GD1 (chan doan) va cho bang ket qua. Day la ban
cai dat rieng, gon, giai dual QP - KHONG sao chep code cua nhom Tanveer.

Twin SVM giai 2 bai toan, moi bai cho mot lop:
  min 0.5||H z||^2 + c e'xi   s.t.  -G z + xi >= rhs,  xi >= 0
voi H=[X_own e], G=[X_opp e], z=[w;b]. Dual:
  max rhs'a - 0.5 a'(G (H'H+eI)^-1 G')a,  0 <= a <= c
  z = -(H'H+eI)^-1 G' a.

GBTSVM = TWSVM tren TAM bong, va rhs = e + r_opp (ban kinh bong lop doi) — ca
mat cau ban kinh r phai vuot le.
"""
from __future__ import annotations
import numpy as np

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except Exception:
    _HAS_CVXPY = False


def _solve_dual(H, G, rhs, c, reg=1e-6):
    """Tra ve z=[w;b] cho mot bai toan twin."""
    HtH = H.T @ H + reg * np.eye(H.shape[1])
    HtH_inv = np.linalg.inv(HtH)
    Q = G @ HtH_inv @ G.T
    Q = 0.5 * (Q + Q.T)
    m = G.shape[0]
    if _HAS_CVXPY:
        a = cp.Variable(m)
        obj = cp.Maximize(rhs @ a - 0.5 * cp.quad_form(a, cp.psd_wrap(Q)))
        cp.Problem(obj, [a >= 0, a <= c]).solve(solver=cp.OSQP, verbose=False)
        av = np.asarray(a.value).ravel()
    else:                                   # fallback: projected gradient ascent
        av = np.zeros(m); L = np.linalg.eigvalsh(Q).max() + 1e-9
        for _ in range(2000):
            av = np.clip(av + (rhs - Q @ av) / L, 0, c)
    return -(HtH_inv @ G.T @ av)


class TwinSVM:
    """TWSVM tuyen tinh chuan (Jayadeva 2007)."""
    def __init__(self, c1=1.0, c2=1.0, reg=1e-6):
        self.c1, self.c2, self.reg = c1, c2, reg

    def _fit_planes(self, Xp, Xn):
        Hp = np.hstack([Xp, np.ones((len(Xp), 1))])
        Hn = np.hstack([Xn, np.ones((len(Xn), 1))])
        # plane lop +1: own=+, opp=-  ; rhs = e (le = 1)
        self.z1 = _solve_dual(Hp, Hn, np.ones(len(Xn)), self.c1, self.reg)
        self.z2 = _solve_dual(Hn, Hp, np.ones(len(Xp)), self.c2, self.reg)
        return self

    def fit(self, X, y):
        return self._fit_planes(X[y == 1], X[y == -1])

    def _dist(self, X, z):
        w, b = z[:-1], z[-1]
        return np.abs(X @ w + b) / (np.linalg.norm(w) + 1e-12)

    def predict(self, X):
        return np.where(self._dist(X, self.z1) <= self._dist(X, self.z2), 1, -1)


class GBTSVM(TwinSVM):
    """GBTSVM goc: twin tren tam bong, rhs = e + r_opp."""
    def fit_balls(self, C, r, y):
        """C: tam bong (m,d); r: ban kinh (m,); y: nhan bong {-1,+1}."""
        Cp, Cn = C[y == 1], C[y == -1]
        rp, rn = r[y == 1], r[y == -1]
        Hp = np.hstack([Cp, np.ones((len(Cp), 1))])
        Hn = np.hstack([Cn, np.ones((len(Cn), 1))])
        self.z1 = _solve_dual(Hp, Hn, 1.0 + rn, self.c1, self.reg)   # +r bong lop -1
        self.z2 = _solve_dual(Hn, Hp, 1.0 + rp, self.c2, self.reg)
        return self
