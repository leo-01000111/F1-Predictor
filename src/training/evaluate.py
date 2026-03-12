"""
Model evaluation metrics.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import brier_score_loss, log_loss, precision_score, recall_score, roc_auc_score

from src.models.calibration import expected_calibration_error


def evaluate_predictions(
    probs: dict[str, np.ndarray],
    y_true: dict[str, np.ndarray],
    driver_names: Optional[list[str]] = None,
    feature_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Compute metrics for p1/p2/p3.

    If feature_df is provided and aligned with probs/y_true, race-level winner/podium
    metrics are computed via evaluate_by_race.
    """
    metrics: dict = {}

    for target in ["p1", "p2", "p3"]:
        if target not in probs or target not in y_true:
            continue

        p = probs[target]
        y = y_true[target]

        if len(np.unique(y)) < 2:
            logger.warning(f"Degenerate labels for {target}; skipping metrics")
            continue

        brier = brier_score_loss(y, p)
        ll = log_loss(y, np.column_stack([1 - p, p]))
        auc = roc_auc_score(y, p)
        precision = precision_score(y, (p >= 0.5).astype(int), zero_division=0)
        recall = recall_score(y, (p >= 0.5).astype(int), zero_division=0)
        ece = expected_calibration_error(np.asarray(p, dtype=float), np.asarray(y, dtype=int), n_bins=10)

        metrics[target] = {
            "brier_score": brier,
            "log_loss": ll,
            "roc_auc": auc,
            "precision_at_05": precision,
            "recall_at_05": recall,
            "ece": ece,
        }

    if feature_df is not None and all(col in feature_df.columns for col in ["year", "round", "driver", "p1", "podium"]):
        if len(feature_df) != len(y_true.get("p1", [])):
            logger.warning("feature_df length mismatch; falling back to global top-1/podium logic")
        else:
            race_results = evaluate_by_race(feature_df, probs)
            if len(race_results) > 0:
                metrics["top1_accuracy"] = float(race_results["winner_correct"].mean())
                metrics["podium_accuracy"] = float(race_results["podium_overlap"].mean() / 3.0)
                if "ece_p1" in race_results.columns:
                    metrics["ece_p1_race_mean"] = float(race_results["ece_p1"].mean())
            else:
                metrics["top1_accuracy"] = 0.0
                metrics["podium_accuracy"] = 0.0
            _log_metrics(metrics)
            return metrics

    if "p1" in probs and "p1" in y_true:
        metrics["top1_accuracy"] = _top_n_accuracy(probs["p1"], y_true["p1"])

    if all(t in probs for t in ["p1", "p2", "p3"]):
        metrics["podium_accuracy"] = _podium_set_accuracy(probs, y_true)

    _log_metrics(metrics)
    return metrics


def _top_n_accuracy(probs: np.ndarray, y_true: np.ndarray) -> float:
    if len(probs) == 0:
        return 0.0
    return float(y_true[np.argmax(probs)])


def _podium_set_accuracy(probs: dict[str, np.ndarray], y_true: dict[str, np.ndarray]) -> float:
    combined = probs["p1"] + probs["p2"] + probs["p3"]
    top3_pred = set(np.argsort(combined)[-3:])
    actual_podium = set(np.where(y_true.get("podium") == 1)[0]) if "podium" in y_true else set()
    if not actual_podium:
        return 0.0
    return len(top3_pred & actual_podium) / 3.0


def evaluate_by_race(feature_df: pd.DataFrame, probs: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Per-race metrics including winner correctness, podium overlap, and ECE(P1).
    """
    df = feature_df.copy()
    df["pred_p1"] = probs["p1"]
    df["pred_p2"] = probs["p2"]
    df["pred_p3"] = probs["p3"]

    results = []
    for (year, rnd), group in df.groupby(["year", "round"]):
        pred_winner = group.loc[group["pred_p1"].idxmax(), "driver"]
        actual_winner = group.loc[group["p1"] == 1, "driver"]
        actual_winner = actual_winner.iloc[0] if not actual_winner.empty else "Unknown"

        group = group.copy()
        group["podium_score"] = group["pred_p1"] + group["pred_p2"] + group["pred_p3"]
        pred_podium = set(group.nlargest(3, "podium_score")["driver"].tolist())
        actual_podium = set(group[group["podium"] == 1]["driver"].tolist())

        ece_p1 = expected_calibration_error(
            group["pred_p1"].to_numpy(dtype=float),
            group["p1"].to_numpy(dtype=int),
            n_bins=10,
        )

        results.append(
            {
                "year": year,
                "round": rnd,
                "circuit": (
                    group["circuit_key"].iloc[0]
                    if "circuit_key" in group.columns
                    else (group["circuit"].iloc[0] if "circuit" in group.columns else "Unknown")
                ),
                "predicted_winner": pred_winner,
                "actual_winner": actual_winner,
                "winner_correct": pred_winner == actual_winner,
                "podium_overlap": len(pred_podium & actual_podium),
                "pred_podium": list(pred_podium),
                "actual_podium": list(actual_podium),
                "ece_p1": float(ece_p1),
            }
        )

    out = pd.DataFrame(results)
    if out.empty:
        return out

    out = out.sort_values(["year", "round"]).reset_index(drop=True)
    out["ece_p1_running"] = out.groupby("year")["ece_p1"].expanding().mean().reset_index(level=0, drop=True)
    return out


def _log_metrics(metrics: dict) -> None:
    for target, m in metrics.items():
        if isinstance(m, dict):
            logger.info(
                f"[{target}] Brier={m.get('brier_score', 0):.4f} | "
                f"LogLoss={m.get('log_loss', 0):.4f} | AUC={m.get('roc_auc', 0):.4f} | "
                f"ECE={m.get('ece', 0):.4f}"
            )
        else:
            logger.info(f"[{target}] = {m:.4f}")
