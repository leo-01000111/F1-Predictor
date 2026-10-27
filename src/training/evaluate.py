"""
Model evaluation metrics (v2 — regression + Plackett-Luce).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import brier_score_loss

from src.models.calibration import expected_calibration_error


def evaluate_race_predictions(
    feature_df: pd.DataFrame,
    pred_positions: np.ndarray,
    sampler: "Any",
) -> tuple[dict, pd.DataFrame]:
    """
    Evaluate a regression + Plackett-Luce model on a held-out set.

    Parameters
    ----------
    feature_df : DataFrame with columns year, round, driver, p1, podium,
                 finish_position (and optionally circuit_key).
    pred_positions : np.ndarray, shape (n_rows,) — predicted finish positions
                     aligned row-for-row with feature_df.
    sampler : PlackettLuceSampler instance.

    Returns
    -------
    metrics : dict with winner_accuracy, podium_accuracy, brier_p1,
              mean_spearman_r, mean_mae_position, n_holdout_races.
    race_results : DataFrame — one row per race, compatible with the
                   holdout_race_results.parquet schema.
    """
    from scipy.stats import spearmanr

    df = feature_df.copy().reset_index(drop=True)
    df["_pred_pos"] = np.asarray(pred_positions, dtype=float)
    df["_p1_pred"] = 0.0

    results = []

    for (year, rnd), group in df.groupby(["year", "round"]):
        race_pred_pos = group["_pred_pos"].values
        prob_matrix = sampler.simulate(race_pred_pos)  # (n_drivers, n_drivers)

        p1_probs = prob_matrix[:, 0]
        p2_probs = prob_matrix[:, 1]
        p3_probs = prob_matrix[:, 2]

        df.loc[group.index, "_p1_pred"] = p1_probs

        pred_winner = group.iloc[int(np.argmax(p1_probs))]["driver"]
        actual_winner_rows = group[group["p1"] == 1]["driver"]
        actual_winner = actual_winner_rows.iloc[0] if not actual_winner_rows.empty else "Unknown"

        podium_score = p1_probs + p2_probs + p3_probs
        pred_podium = set(group.iloc[np.argsort(-podium_score)[:3]]["driver"].tolist())
        actual_podium = set(group[group["podium"] == 1]["driver"].tolist())

        actual_pos = pd.to_numeric(group.get("finish_position"), errors="coerce").values
        valid = np.isfinite(actual_pos)
        if valid.sum() >= 3:
            corr, _ = spearmanr(race_pred_pos[valid], actual_pos[valid])
            corr = float(corr)
        else:
            corr = None

        mae = float(np.abs(race_pred_pos[valid] - actual_pos[valid]).mean()) if valid.sum() > 0 else None

        ece_p1 = expected_calibration_error(p1_probs, group["p1"].values, n_bins=10)

        # Winner log loss: -log P(actual winner), the per-race proper score
        win_mask = group["p1"].values == 1
        log_loss_winner = (
            float(-np.log(max(p1_probs[win_mask][0], 1e-6))) if win_mask.any() else None
        )

        # Grid baselines: pole sitter wins, grid top 3 = podium.
        # Grid 0 is a pit-lane start, so it counts as last.
        grid_pole_correct = None
        grid_podium_overlap = None
        if "grid_position" in group.columns:
            grid = pd.to_numeric(group["grid_position"], errors="coerce").values
            if np.isfinite(grid).any():
                grid = np.where(np.isfinite(grid) & (grid > 0), grid, np.inf)
                order = np.argsort(grid, kind="stable")
                grid_pole_correct = group.iloc[order[0]]["driver"] == actual_winner
                grid_podium_overlap = len(set(group.iloc[order[:3]]["driver"]) & actual_podium)

        circuit = (
            group["circuit_key"].iloc[0] if "circuit_key" in group.columns
            else (group["circuit"].iloc[0] if "circuit" in group.columns else "unknown")
        )

        results.append({
            "year": year,
            "round": rnd,
            "circuit": circuit,
            "predicted_winner": pred_winner,
            "actual_winner": actual_winner,
            "winner_correct": pred_winner == actual_winner,
            "podium_overlap": len(pred_podium & actual_podium),
            "pred_podium": list(pred_podium),
            "actual_podium": list(actual_podium),
            "ece_p1": float(ece_p1),
            "spearman_r": corr,
            "mae_position": mae,
            "log_loss_winner": log_loss_winner,
            "grid_pole_correct": grid_pole_correct,
            "grid_podium_overlap": grid_podium_overlap,
        })

    out = pd.DataFrame(results)
    if out.empty:
        return {
            "winner_accuracy": 0.0,
            "podium_accuracy": 0.0,
            "brier_p1": None,
            "mean_spearman_r": None,
            "mean_mae_position": None,
            "n_holdout_races": 0,
        }, out

    out = out.sort_values(["year", "round"]).reset_index(drop=True)
    out["ece_p1_running"] = (
        out.groupby("year")["ece_p1"].expanding().mean().reset_index(level=0, drop=True)
    )

    all_p1_preds = df["_p1_pred"].values
    all_p1_true = df["p1"].values
    brier_p1 = float(brier_score_loss(all_p1_true, np.clip(all_p1_preds, 0.0, 1.0)))

    metrics = {
        "winner_accuracy": float(out["winner_correct"].mean()),
        "podium_accuracy": float(out["podium_overlap"].mean() / 3.0),
        "brier_p1": brier_p1,
        "mean_spearman_r": (
            float(out["spearman_r"].dropna().mean()) if out["spearman_r"].notna().any() else None
        ),
        "mean_mae_position": (
            float(out["mae_position"].dropna().mean()) if out["mae_position"].notna().any() else None
        ),
        "n_holdout_races": len(out),
        "mean_log_loss_winner": (
            float(out["log_loss_winner"].dropna().mean()) if out["log_loss_winner"].notna().any() else None
        ),
    }

    # Baselines the model has to beat. Same races, same grid the model sees.
    has_grid = out["grid_pole_correct"].notna()
    if has_grid.any():
        baselines = {
            "winner_accuracy_pole": float(out.loc[has_grid, "grid_pole_correct"].astype(float).mean()),
            "podium_accuracy_grid_top3": float(out.loc[has_grid, "grid_podium_overlap"].astype(float).mean() / 3.0),
            "n_races": int(has_grid.sum()),
        }
        metrics["baselines"] = baselines
        metrics["beats_baselines"] = bool(
            metrics["winner_accuracy"] > baselines["winner_accuracy_pole"]
            and metrics["podium_accuracy"] > baselines["podium_accuracy_grid_top3"]
        )

    _log_race_metrics(metrics)
    return metrics, out


def _log_race_metrics(metrics: dict) -> None:
    logger.info(
        f"Winner accuracy : {metrics['winner_accuracy']:.1%} over {metrics['n_holdout_races']} races"
    )
    logger.info(f"Podium accuracy : {metrics['podium_accuracy']:.1%}")
    if metrics.get("brier_p1") is not None:
        logger.info(f"Brier P1        : {metrics['brier_p1']:.4f}")
    if metrics.get("mean_spearman_r") is not None:
        logger.info(f"Mean Spearman r : {metrics['mean_spearman_r']:.3f}")
    if metrics.get("mean_mae_position") is not None:
        logger.info(f"Mean MAE pos    : {metrics['mean_mae_position']:.2f} places")
    if metrics.get("mean_log_loss_winner") is not None:
        logger.info(f"Winner log loss : {metrics['mean_log_loss_winner']:.3f}")
    b = metrics.get("baselines")
    if b:
        logger.info(
            f"Baseline pole wins   : {b['winner_accuracy_pole']:.1%}  "
            f"(model {metrics['winner_accuracy']:.1%})"
        )
        logger.info(
            f"Baseline grid top 3  : {b['podium_accuracy_grid_top3']:.1%}  "
            f"(model {metrics['podium_accuracy']:.1%})"
        )
        if not metrics["beats_baselines"]:
            logger.warning("Model does NOT beat the grid baselines on this holdout")
