"""Sinh granular ball theo chuan GBTSVM goc (Quadir, Sajid, Tanveer - IEEE TNNLS 2025):
chia de quy bang 2-means cho toi khi moi bong dat purity >= T.

Moi bong: center = trung binh cac diem, radius = khoang cach trung binh toi tam,
label = nhan da so, purity = ti le nhan da so.

Ham `ball_stats` phuc vu thi nghiem chan doan GD1: do so bong / kich thuoc bong
theo muc nhieu — bang chung co che "bong vo vun".
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans


@dataclass
class GranularBall:
    center: np.ndarray
    radius: float
    label: int
    purity: float
    n: int
    idx: np.ndarray  # chi so diem goc trong bong


def _make_ball(X, y, idx) -> GranularBall:
    c = X.mean(axis=0)
    r = float(np.linalg.norm(X - c, axis=1).mean()) if len(X) > 1 else 0.0
    labels, counts = np.unique(y, return_counts=True)
    k = int(np.argmax(counts))
    return GranularBall(c, r, int(labels[k]), counts[k] / len(y), len(y), idx)


def generate_balls(X: np.ndarray, y: np.ndarray, purity_threshold: float = 1.0,
                   min_size: int = 2, seed: int = 0) -> list[GranularBall]:
    """Sinh bong bang 2-means de quy, dung khi purity >= T hoac bong qua nho."""
    X = np.asarray(X, float); y = np.asarray(y).astype(int)
    out: list[GranularBall] = []
    stack = [np.arange(len(y))]
    rng_state = seed
    while stack:
        idx = stack.pop()
        ball = _make_ball(X[idx], y[idx], idx)
        if ball.purity >= purity_threshold or len(idx) <= min_size:
            out.append(ball)
            continue
        km = KMeans(n_clusters=2, n_init=4, random_state=rng_state).fit(X[idx])
        rng_state += 1
        a, b = idx[km.labels_ == 0], idx[km.labels_ == 1]
        if len(a) == 0 or len(b) == 0:      # 2-means khong tach duoc -> dung
            out.append(ball)
        else:
            stack += [a, b]
    return out


def ball_stats(balls: list[GranularBall]) -> dict:
    """Thong ke cho thi nghiem chan doan: so bong, diem/bong, purity, ban kinh."""
    n = np.array([b.n for b in balls]); p = np.array([b.purity for b in balls])
    return {
        "n_balls": len(balls),
        "points_per_ball_mean": float(n.mean()),
        "points_per_ball_median": float(np.median(n)),
        "purity_mean": float(p.mean()),
        "frac_singleton": float((n <= 2).mean()),
        "radius_mean": float(np.mean([b.radius for b in balls])),
    }


def balls_to_arrays(balls: list[GranularBall]):
    """(C, r, y, n, p) de dua thang vao bai toan twin."""
    return (np.stack([b.center for b in balls]),
            np.array([b.radius for b in balls]),
            np.array([b.label for b in balls]),
            np.array([b.n for b in balls]),
            np.array([b.purity for b in balls]))
