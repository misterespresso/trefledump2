"""Small statistics helpers, kept separate so they can be unit-tested offline."""

from __future__ import annotations

import math
from typing import Iterable

from scipy.stats import binomtest, fisher_exact


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion. Returns (low, high)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def binom_p(successes: int, n: int, p0: float) -> float:
    """Two-sided binomial test p-value against p0."""
    if n == 0:
        return float("nan")
    return float(binomtest(successes, n, p0, alternative="two-sided").pvalue)


def fisher_2x2(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Fisher's exact test on [[a, b], [c, d]]. Returns (odds_ratio, two-sided p)."""
    odds, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    return float(odds), float(p)


def entropy(ps: Iterable[float]) -> float:
    return -sum(p * math.log(p) for p in ps if p > 0)


def normalised_entropy(ps: list[float]) -> float:
    k = len(ps)
    return entropy(ps) / math.log(k) if k > 1 else 0.0


def fmt_ci(low: float, high: float, pct: bool = True) -> str:
    if math.isnan(low):
        return "n/a"
    return f"{low * 100:.2f}%-{high * 100:.2f}%" if pct else f"{low:.4f}-{high:.4f}"


def fmt_p(p: float) -> str:
    if p != p:
        return "n/a"
    if p < 1e-12:
        return "<1e-12"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"
