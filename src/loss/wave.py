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


def lambda_adaptive(purity, size=None, radius=None, lam0=1.0, kappa=2.0,
                    p0=0.85, sharp=8.0):
    """Per-ball ceiling from an ABSOLUTE purity map (no min-max over the set).

        lam_k = lam0 * (1 + kappa * sigmoid(sharp * (p0 - purity_k)))

    A ball below p0 gets a larger lam -> lower ceiling (1/lam) -> its influence is
    capped harder; a clean ball keeps lam ~ lam0. Absolute so it does not degrade
    when every ball has the same purity (e.g. purity=1), unlike a min-max scaling
    that stretches tiny differences across [0,1]. size/radius are accepted for
    signature compatibility but unused: they diluted the trust signal. Returns a
    scalar lam0 if purity is missing or empty.
    """
    if purity is None:
        return lam0
    purity = np.asarray(purity, dtype=float)
    if purity.size == 0:
        return lam0
    s = 1.0 / (1.0 + np.exp(-np.clip(sharp * (p0 - purity), -30.0, 30.0)))
    return lam0 * (1.0 + kappa * s)


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
