"""
Calibration utilities.
"""
from __future__ import annotations

import numpy as np


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if probs.size == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bucket = np.digitize(probs, bins) - 1
    ece = 0.0

    for b in range(n_bins):
        mask = bucket == b
        if not np.any(mask):
            continue
        conf = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        weight = float(mask.sum()) / float(len(probs))
        ece += abs(acc - conf) * weight

    return float(ece)
