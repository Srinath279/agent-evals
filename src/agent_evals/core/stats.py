"""Statistical rigor (master plan §8): bootstrap CIs for gating on
significant regressions rather than point noise, and agreement measures
for judge calibration. stdlib-only."""

from __future__ import annotations

import math
import random
from typing import Sequence


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_mean_diff_ci(
    current: Sequence[float],
    baseline: Sequence[float],
    iterations: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """CI for mean(current) - mean(baseline). Regression is 'significant'
    when the upper bound is below 0."""
    if not current or not baseline:
        raise ValueError("both samples must be non-empty")
    rng = random.Random(seed)
    diffs = []
    for _ in range(iterations):
        cur = [current[rng.randrange(len(current))] for _ in range(len(current))]
        base = [baseline[rng.randrange(len(baseline))] for _ in range(len(baseline))]
        diffs.append(mean(cur) - mean(base))
    diffs.sort()
    lo = diffs[int((alpha / 2) * iterations)]
    hi = diffs[min(int((1 - alpha / 2) * iterations), iterations - 1)]
    return lo, hi


def cohens_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Agreement between two binary raters (judge vs human), chance-corrected.
    Target for judge calibration: kappa > 0.8 (master plan §7)."""
    if len(a) != len(b) or not a:
        raise ValueError("samples must be equal-length and non-empty")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    expected = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("samples must be equal-length with n >= 2")
    ma, mb = mean(a), mean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    var_a = math.sqrt(sum((x - ma) ** 2 for x in a))
    var_b = math.sqrt(sum((y - mb) ** 2 for y in b))
    if var_a == 0 or var_b == 0:
        return 0.0
    return cov / (var_a * var_b)
