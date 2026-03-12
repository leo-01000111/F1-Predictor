"""
Leave-one-season-out cross-validation.
Critical for temporal correctness: never train on future data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.features.build_features import FEATURE_COLS, get_X_y
from src.training.evaluate import evaluate_predictions, evaluate_by_race


def leave_season_out_cv(
    feature_df: pd.DataFrame,
    model_factory,  # callable() -> model with .fit() and .predict_proba()
    holdout_seasons: list[int] = [2023, 2024],
    min_train_seasons: int = 3,
) -> dict:
    """
    Perform leave-one-season-out cross-validation.

    For each season (excluding holdout_seasons):
      - Train on all seasons before it
      - Evaluate on that season

    Returns aggregated metrics dict and per-race results DataFrame.
    """
    all_seasons = sorted(feature_df["year"].unique())
    cv_seasons = [s for s in all_seasons if s not in holdout_seasons]

    all_race_results = []
    all_metrics = []

    for eval_season in cv_seasons:
        train_seasons = [s for s in all_seasons if s < eval_season and s not in holdout_seasons]

        if len(train_seasons) < min_train_seasons:
            logger.info(f"Skipping {eval_season}: only {len(train_seasons)} train seasons available")
            continue

        logger.info(f"CV: train on {train_seasons}, eval on {eval_season}")

        train_df = feature_df[feature_df["year"].isin(train_seasons)]
        val_df = feature_df[feature_df["year"] == eval_season]

        # Prepare inputs
        X_train = train_df[FEATURE_COLS].values
        X_val = val_df[FEATURE_COLS].values

        y_train = {
            "p1": train_df["p1"].values,
            "p2": train_df["p2"].values,
            "p3": train_df["p3"].values,
        }
        y_val = {
            "p1": val_df["p1"].values,
            "p2": val_df["p2"].values,
            "p3": val_df["p3"].values,
        }

        # Train
        model = model_factory()
        driver_idx_train = train_df["driver_idx"].values if "driver_idx" in train_df.columns else np.zeros(len(train_df), dtype=int)
        team_idx_train = train_df["team_idx"].values if "team_idx" in train_df.columns else np.zeros(len(train_df), dtype=int)
        circuit_idx_train = train_df["circuit_key_idx"].values if "circuit_key_idx" in train_df.columns else np.zeros(len(train_df), dtype=int)

        driver_idx_val = val_df["driver_idx"].values if "driver_idx" in val_df.columns else np.zeros(len(val_df), dtype=int)
        team_idx_val = val_df["team_idx"].values if "team_idx" in val_df.columns else np.zeros(len(val_df), dtype=int)
        circuit_idx_val = val_df["circuit_key_idx"].values if "circuit_key_idx" in val_df.columns else np.zeros(len(val_df), dtype=int)

        try:
            model.fit(
                X_train, driver_idx_train, team_idx_train, circuit_idx_train,
                y_train,
            )
            probs = model.predict_proba(
                X_val, driver_idx_val, team_idx_val, circuit_idx_val,
            )
        except TypeError:
            # Fallback for XGBoost-only (no embedding indices)
            model.fit(pd.DataFrame(X_train, columns=FEATURE_COLS), y_train)
            probs = model.predict_proba(pd.DataFrame(X_val, columns=FEATURE_COLS))

        metrics = evaluate_predictions(probs, y_val, feature_df=val_df)
        metrics["season"] = eval_season
        all_metrics.append(metrics)

        race_results = evaluate_by_race(val_df.reset_index(drop=True), probs)
        race_results["cv_season"] = eval_season
        all_race_results.append(race_results)

    race_results_df = pd.concat(all_race_results, ignore_index=True) if all_race_results else pd.DataFrame()

    # Aggregate metrics across seasons
    agg = _aggregate_cv_metrics(all_metrics)
    logger.info(f"CV Summary: {agg}")

    return {"per_season_metrics": all_metrics, "aggregated": agg, "race_results": race_results_df}


def _aggregate_cv_metrics(metrics_list: list[dict]) -> dict:
    """Average numeric metrics across seasons."""
    agg: dict = {}
    for key in ["p1", "p2", "p3"]:
        vals = [m.get(key, {}) for m in metrics_list if key in m]
        if vals:
            for metric in ["brier_score", "log_loss", "roc_auc"]:
                scores = [v.get(metric) for v in vals if v.get(metric) is not None]
                if scores:
                    agg[f"{key}_{metric}_mean"] = float(np.mean(scores))
                    agg[f"{key}_{metric}_std"] = float(np.std(scores))
    for key in ["top1_accuracy", "podium_accuracy"]:
        scores = [m.get(key) for m in metrics_list if m.get(key) is not None]
        if scores:
            agg[f"{key}_mean"] = float(np.mean(scores))
    return agg

