"""Wave loss (Akhtar, Tanveer, Arshad, Pattern Recognition 2024) and a per-ball variant.

    L(u) = (1/lam) * (1 - 1/(1 + lam * u^2 * exp(a*u)))

Bounded by 1/lam, smooth, redescending. For small u it is ~ u^2 exp(a*u),
independent of lam, so lam only sets the saturation ceiling.
`lam` may be a scalar or a per-ball vector.
"""
from __future__ import annotations
import numpy as np


def wave_loss(u, lam, a=1.0):
    u = np.asarray(u, dtype=float)
    t = u * u * np.exp(np.clip(a * u, -50.0, 50.0))
    return t / (1.0 + lam * t)


def wave_grad(u, lam, a=1.0):
    u = np.asarray(u, dtype=float)
    e = np.exp(np.clip(a * u, -50.0, 50.0))
    t = u * u * e
    tp = (2.0 * u + a * u * u) * e
    return tp / (1.0 + lam * t) ** 2


def lambda_adaptive(purity, size, radius, lam0=1.0, kappa=2.0,
                    w_p=1.0, w_n=0.5, w_r=0.5):
    """Per-ball ceiling lam_k = lam0 * exp(-kappa * s_k).

    s_k in [0,1] rises with purity and size, falls with radius; each term is
    min-max scaled across the current ball set. Trusted ball -> small lam -> high ceiling.
    """
    def mm(x):
        x = np.asarray(x, dtype=float)
        lo, hi = x.min(), x.max()
        return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)

    s = (w_p * mm(purity) + w_n * mm(np.log1p(size)) + w_r * (1.0 - mm(radius)))
    s = s / (w_p + w_n + w_r)
    return lam0 * np.exp(-kappa * s)


def numerical_gradient_check(a=1.0, lam=0.7, n=64, eps=1e-6, seed=0, tol=1e-5):
    rng = np.random.default_rng(seed)
    u = rng.uniform(-3, 6, size=n)
    num = (wave_loss(u + eps, lam, a) - wave_loss(u - eps, lam, a)) / (2 * eps)
    err = float(np.max(np.abs(num - wave_grad(u, lam, a))))
    assert err < tol, f"gradient check failed: {err:.2e}"
    return err


if __name__ == "__main__":
    e = max(numerical_gradient_check(lam=0.7),
            numerical_gradient_check(lam=np.full(64, 2.0), a=1.5))
    print(f"gradient check ok ({e:.1e})")
    for lam in (0.5, 2.0):
        print(f"lam={lam}: L(5)={wave_loss(np.array([5.0]), lam)[0]:.3f} ceiling={1/lam:.2f}")
