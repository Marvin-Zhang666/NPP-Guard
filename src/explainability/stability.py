"""Deterministic ranking metrics for explanation validation."""

from __future__ import annotations

import numpy as np


def jaccard(left: list[str], right: list[str]) -> float:
    a, b = set(left), set(right)
    return 1.0 if not a and not b else len(a & b) / max(1, len(a | b))


def rank_correlation(left: list[str], right: list[str]) -> float:
    common = [item for item in left if item in set(right)]
    if len(common) < 2:
        return float("nan")
    left_rank = {item: index for index, item in enumerate(left)}
    right_rank = {item: index for index, item in enumerate(right)}
    x = np.asarray([left_rank[item] for item in common], dtype=float)
    y = np.asarray([right_rank[item] for item in common], dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
