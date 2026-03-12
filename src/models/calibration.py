"""
Probability calibration utilities.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class ProbabilityCalibrator:
    """
    Fits an isotonic or sigmoid calibrator per target (p1, p2, p3).
    """

    def __init__(self, method: str = "isotonic"):
        assert method in ("isotonic", "sigmoid"), "method must be 'isotonic' or 'sigmoid'"
        self.method = method
        self._calibrators: dict[str, IsotonicRegression | LogisticRegression] = {}

    def fit(
        self,
        raw_probs: dict[str, np.ndarray],
        y_true: dict[str, np.ndarray],
    ) -> "ProbabilityCalibrator":
        for target in ["p1", "p2", "p3"]:
            probs = raw_probs[target]
            labels = y_true[target]

            if self.method == "isotonic":
                cal = IsotonicRegression(out_of_bounds="clip")
                cal.fit(probs, labels)
            else:
                log_odds = np.log(np.clip(probs, 1e-8, 1 - 1e-8) / np.clip(1 - probs, 1e-8, 1.0)).reshape(-1, 1)
                cal = LogisticRegression(C=1.0, max_iter=1000)
                cal.fit(log_odds, labels)

            self._calibrators[target] = cal
            logger.info(f"Calibrator fitted for {target}")

        return self

    def calibrate(self, raw_probs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        calibrated = {}
        for target in ["p1", "p2", "p3"]:
            probs = raw_probs[target]
            cal = self._calibrators[target]

            if self.method == "isotonic":
                calibrated[target] = cal.transform(probs)
            else:
                log_odds = np.log(np.clip(probs, 1e-8, 1 - 1e-8) / np.clip(1 - probs, 1e-8, 1.0)).reshape(-1, 1)
                calibrated[target] = cal.predict_proba(log_odds)[:, 1]

        return calibrated

    def save(self, name: str = "calibrator") -> None:
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved calibrator -> {path}")

    @classmethod
    def load(cls, name: str = "calibrator") -> "ProbabilityCalibrator":
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Loaded calibrator <- {path}")
        return obj


def normalize_position_probs(probs: np.ndarray) -> np.ndarray:
    """
    Normalize per-position probabilities to sum to 1 across drivers.

    This is used for P1/P2/P3 targets, each of which has exactly one positive
    outcome per race.
    """
    arr = np.asarray(probs, dtype=float)
    if arr.size == 0:
        return arr

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)

    total = float(arr.sum())
    if total <= 0:
        return np.full(arr.shape, 1.0 / float(len(arr)), dtype=float)
    return arr / total


def normalize_win_probs(probs: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for win-probability normalization."""
    return normalize_position_probs(probs)


def normalize_position_prob_dict(
    probs: dict[str, np.ndarray],
    targets: tuple[str, ...] = ("p1", "p2", "p3"),
) -> dict[str, np.ndarray]:
    """Return a copy with selected targets normalized across drivers."""
    out = dict(probs)
    for target in targets:
        if target in out:
            out[target] = normalize_position_probs(out[target])
    return out

def get_calibration_stats(raw_probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    fraction_of_pos, mean_predicted = calibration_curve(
        y_true,
        raw_probs,
        n_bins=n_bins,
        strategy="uniform",
    )
    return {
        "fraction_of_positives": fraction_of_pos.tolist(),
        "mean_predicted_value": mean_predicted.tolist(),
    }


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    """
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


def fit_result_doc_calibrator(
    result_docs: list[dict[str, Any]],
    *,
    target: str,
    method: str = "isotonic",
    min_samples: int = 120,
    min_positives: int = 6,
) -> Optional[IsotonicRegression | LogisticRegression]:
    """
    Fit a lightweight seasonal calibrator from saved result documents.

    Result docs are expected to include `predictions` rows with probability fields
    and `actual_finish` values.
    """
    assert target in {"p1", "p2", "p3"}
    prob_key = {"p1": "p_win", "p2": "p_p2", "p3": "p_p3"}[target]
    finish_pos = {"p1": 1, "p2": 2, "p3": 3}[target]

    probs: list[float] = []
    labels: list[int] = []

    for doc in result_docs:
        for row in doc.get("predictions", []) or []:
            p = row.get(prob_key)
            af = row.get("actual_finish")
            if p is None or af is None:
                continue
            try:
                p_f = float(p)
                af_i = int(af)
            except Exception:
                continue
            if not np.isfinite(p_f):
                continue
            probs.append(float(np.clip(p_f, 0.0, 1.0)))
            labels.append(1 if af_i == finish_pos else 0)

    if len(probs) < min_samples:
        return None
    if int(np.sum(labels)) < min_positives:
        return None

    x = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=int)

    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(x, y)
        return cal

    logits = np.log(np.clip(x, 1e-8, 1 - 1e-8) / np.clip(1 - x, 1e-8, 1.0)).reshape(-1, 1)
    cal_lr = LogisticRegression(C=1.0, max_iter=1000)
    cal_lr.fit(logits, y)
    return cal_lr


def apply_external_calibrator(
    probs: np.ndarray,
    calibrator: IsotonicRegression | LogisticRegression,
) -> np.ndarray:
    x = np.asarray(probs, dtype=float)
    if isinstance(calibrator, IsotonicRegression):
        return calibrator.transform(x)
    logits = np.log(np.clip(x, 1e-8, 1 - 1e-8) / np.clip(1 - x, 1e-8, 1.0)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]

