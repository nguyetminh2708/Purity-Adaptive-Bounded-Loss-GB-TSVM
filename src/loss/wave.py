"""Wave loss (Akhtar, Tanveer, Arshad - Pattern Recognition 2024) va bien the per-ball.

    L(u) = (1/lam) * (1 - 1/(1 + lam * u^2 * exp(a*u)))
         = t / (1 + lam*t),  voi t = u^2 * exp(a*u)

Tinh chat dung trong PA-BL-GBTSVM:
  - Bi chan:    L(u) < 1/lam voi moi u; gradient -> 0 khi u lon (redescending).
  - Tron:       kha vi moi noi -> giai bang Adam/NAG, khong can QP.
  - Vung sach:  u nho => L ~ u^2 exp(a*u), KHONG phu thuoc lam
                => lam chi dieu khien "tran phan doi", khong doi hanh vi vung sach.

`lam` nhan scalar hoac vector (lambda_k rieng tung bong) - broadcast theo u.
"""
from __future__ import annotations
import numpy as np


def wave_loss(u: np.ndarray, lam, a: float = 1.0) -> np.ndarray:
    """Gia tri Wave loss. u: (m,), lam: scalar hoac (m,)."""
    u = np.asarray(u, dtype=float)
    # clip mu de tranh overflow exp; voi u lon L ~ 1/lam nen sai so khong dang ke
    t = u * u * np.exp(np.clip(a * u, -50.0, 50.0))
    return t / (1.0 + lam * t)


def wave_grad(u: np.ndarray, lam, a: float = 1.0) -> np.ndarray:
    """dL/du = t'(u) / (1 + lam*t)^2,  t'(u) = (2u + a*u^2) * exp(a*u)."""
    u = np.asarray(u, dtype=float)
    e = np.exp(np.clip(a * u, -50.0, 50.0))
    t = u * u * e
    tp = (2.0 * u + a * u * u) * e
    return tp / (1.0 + lam * t) ** 2


def lambda_adaptive(purity: np.ndarray, size: np.ndarray, radius: np.ndarray,
                    lam0: float = 1.0, kappa: float = 2.0,
                    w_p: float = 1.0, w_n: float = 0.5, w_r: float = 0.5) -> np.ndarray:
    """lambda_k = lam0 * exp(-kappa * s_k) — tran phat rieng tung bong.

    s_k: diem tin cay trong [0,1], tang theo purity va kich thuoc, giam theo ban kinh.
    Cac thanh phan duoc chuan hoa min-max trong tap bong hien tai.
    """
    def mm(x):
        x = np.asarray(x, dtype=float)
        lo, hi = x.min(), x.max()
        return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)

    s = (w_p * mm(purity) + w_n * mm(np.log1p(size)) + w_r * (1.0 - mm(radius)))
    s = s / (w_p + w_n + w_r)
    return lam0 * np.exp(-kappa * s)   # s cao (dang tin) -> lambda nho -> tran cao


def numerical_gradient_check(a: float = 1.0, lam=0.7, n: int = 64, eps: float = 1e-6,
                             seed: int = 0, tol: float = 1e-5) -> float:
    """Kiem tra wave_grad bang sai phan trung tam. Tra ve sai so lon nhat."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-3, 6, size=n)
    num = (wave_loss(u + eps, lam, a) - wave_loss(u - eps, lam, a)) / (2 * eps)
    err = float(np.max(np.abs(num - wave_grad(u, lam, a))))
    assert err < tol, f"gradient check FAILED: {err:.2e}"
    return err


if __name__ == "__main__":
    err1 = numerical_gradient_check(lam=0.7)
    err2 = numerical_gradient_check(lam=np.full(64, 2.0), a=1.5)
    print(f"gradient check OK  (max err {max(err1, err2):.2e})")
    # minh hoa tran bao hoa
    for lam in (0.5, 2.0):
        print(f"lam={lam}:  L(5)={wave_loss(np.array([5.0]), lam)[0]:.3f}  (tran 1/lam = {1/lam:.2f})")
