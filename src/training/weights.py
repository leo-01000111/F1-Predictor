"""
Training sample-weight utilities.
"""
from __future__ import annotations

import numpy as np


def compute_recency_sample_weight(
    years: np.ndarray,
    *,
    min_weight: float = 0.65,
    max_weight: float = 1.35,
) -> np.ndarray:
    """
    Compute monotonic recency weights by season.

    Oldest season gets min_weight, most recent gets max_weight, with linear interpolation.
    """
    years = np.asarray(years)
    if years.size == 0:
        return np.array([], dtype=float)

    uniq = sorted({int(y) for y in years.tolist()})
    if len(uniq) <= 1:
        return np.ones(len(years), dtype=float)

    denom = float(len(uniq) - 1)
    mapping: dict[int, float] = {}
    for i, y in enumerate(uniq):
        frac = float(i) / denom
        mapping[y] = float(min_weight + frac * (max_weight - min_weight))

    weights = np.array([mapping[int(y)] for y in years], dtype=float)
    mean_w = float(np.mean(weights)) if len(weights) else 1.0
    if mean_w > 0:
        weights = weights / mean_w
    return weights
