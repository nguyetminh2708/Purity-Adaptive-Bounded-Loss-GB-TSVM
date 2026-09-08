"""
Granular ball generation — seeded, reproducible version.

This is a fork of Sourcecode/gen_ball_mix.py with EXACTLY ONE change:
KMeans now receives a random_state. The algorithm itself is unchanged:
  - center = mean of features
  - r      = MEAN Euclidean distance from the center (kept as in the original;
             switching to a quantile is improvement #4, not part of phase A)
  - label  = majority label
  - split  = 2-means, repeated until purity >= pur or n <= 1

The original never set random_state, so its results were not reproducible
across runs.
"""

from collections import Counter

import numpy as np
from scipy.stats import binom
from sklearn.cluster import KMeans


class Ball:
    """A single granular ball. data: (n, d+2) = [features(d) | label | index]."""

    __slots__ = ("data", "X", "n", "dim", "center", "label", "purity", "radius",
                 "n_major", "n_minor")

    def __init__(self, data):
        self.data = data
        self.X = data[:, :-2]
        self.n, self.dim = self.X.shape
        self.center = self.X.mean(axis=0)

        cnt = Counter(data[:, -2])
        self.label = max(cnt, key=cnt.get)
        # exact counts, not purity*n rounded back — the binomial rule needs an
        # integer k and float round-trips can be off by one on large balls
        self.n_major = int(cnt[self.label])
        self.n_minor = int(self.n - self.n_major)
        self.purity = self.n_major / self.n
        self.radius = float(np.sqrt(((self.X - self.center) ** 2).sum(axis=1)).mean())

    def split(self, seed):
        """
        Split into two balls with 2-means. Returns (ball1, ball2).

        n_init=1 is deliberate, for two reasons:
          - The original calls KMeans(n_clusters=2) without n_init, i.e. it takes
            sklearn's default 'auto' == 1 for k-means++. Setting 10 would be a
            deviation from the original.
          - Measured on chess_krvkp (1700 samples): n_init=10 costs 91.6s per
            ball generation, n_init=1 costs 8.7s, while the ball count differs
            only 297 vs 310 (~4%).
        """
        lab = KMeans(n_clusters=2, random_state=seed, n_init=1).fit(self.X).labels_
        if (lab == 0).any() and (lab == 1).any():
            return Ball(self.data[lab == 0]), Ball(self.data[lab == 1])
        # 2-means could not separate (all points identical) — peel one off,
        # same fallback as the original.
        return Ball(self.data[:1]), Ball(self.data[1:])

    def __repr__(self):
        return f"Ball(label={self.label}, purity={self.purity:.2f}, n={self.n})"


def should_split(b, pur=1.0, eta=None, alpha=0.05):
    """
    Stopping rule for the splitting recursion.

    eta is None  -> the original rule: split while purity < pur.

    eta is set   -> the statistical rule. Treat purity as an ESTIMATE from a
    finite sample rather than a fact, and ask whether the observed impurity is
    explainable by label noise alone:

        H0 : the ball is truly pure, and all k minority labels are noise
             => k ~ Binomial(n, eta)
        split iff  P(K >= k | Bin(n, eta)) < alpha

    Why this matters. Under label noise a genuinely pure region NEVER reaches a
    hard purity threshold, so the original rule keeps splitting on features that
    carry no label signal until the balls are near-singletons. At that point the
    compression is gone AND so is the majority-vote noise absorption that is the
    whole point of granular ball computing — the method degenerates into plain
    TSVM exactly when noise makes it most valuable.

    The binomial rule is n-adaptive, which no fixed threshold can be: the same
    purity of 0.6 stops a ball of 5 (p=0.081, indistinguishable from noise) and
    splits a ball of 50 (p~0, real structure). That n-adaptivity is the point.

    eta = 0 recovers the hard purity rule at T = 1, so the original method is a
    special case inside the search space.
    """
    if b.n <= 1:
        return False
    if eta is None:
        return b.purity < pur
    if b.n_minor == 0:
        return False
    if eta <= 0:
        return True
    return float(binom.sf(b.n_minor - 1, b.n, eta)) < alpha


def gen_balls(data, pur=1.0, delbals=0, seed=0, eta=None, alpha=0.05):
    """
    data: (n, d+1) = [features | label].  Returns list[Ball].

    pur           : purity threshold (used only when eta is None)
    delbals       : drop balls holding fewer than `delbals` samples
    eta, alpha    : statistical stopping rule; see should_split
    """
    n = data.shape[0]
    d = np.hstack([data, np.arange(n).reshape(-1, 1)])

    balls = [Ball(d)]
    i = 0
    while i < len(balls):
        b = balls[i]
        if should_split(b, pur, eta, alpha):
            b1, b2 = b.split(seed)
            balls[i] = b1
            balls.append(b2)
        else:
            i += 1

    return [b for b in balls if b.n >= delbals]


def to_matrix(balls):
    """list[Ball] -> (m, d+2) = [center(d) | radius | label].

    This is the ONLY shape allowed into fit. See contract.py.
    """
    return np.column_stack([
        np.array([b.center for b in balls]),
        np.array([b.radius for b in balls]),
        np.array([b.label for b in balls]),
    ])


def viable(balls, min_per_class=1, min_classes=2):
    """
    Whether a ball-generation config is usable at all.

    min_per_class=1 is correct, not 2: a linearly separable class collapses into
    EXACTLY ONE pure ball (iris setosa does) — that is the ideal outcome of
    granular ball computing, not a failure. It is also enough for the QP:
    HH1 = H1'H1 + eps*I is invertible thanks to eps regardless of m1, and the
    QP runs over m2 variables, so m1 >= 1 and m2 >= 1 suffices.

    What must be blocked is LOSING a class entirely (min_classes), because then
    fit_binary has nothing to separate and cvxopt returns garbage instead of
    raising.
    """
    if not balls:
        return False
    cnt = Counter(b.label for b in balls)
    return len(cnt) >= min_classes and min(cnt.values()) >= min_per_class


def compression(balls, n_samples):
    """Compression ratio n samples -> m balls. Used by the phase B metrics."""
    return n_samples / max(len(balls), 1)
