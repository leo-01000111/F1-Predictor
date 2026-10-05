"""
Optuna hyperparameter tuning for XGBRacePredictor.

Searches over XGBoost parameters using a held-out evaluation window.
Optimises for winner accuracy (primary) with Brier score as tiebreaker.

Usage:
    python scripts/tune_hyperparams.py
    python scripts/tune_hyperparams.py --n-trials 100 --timeout 1800
    python scripts/tune_hyperparams.py --holdout-years 2024 2025

Results saved to models/optuna_best_params.json.
To apply: python scripts/quick_train.py --use-tuned-params
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger


def main() -> None:
    import argparse

    import numpy as np
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=60,
                        help="Number of Optuna trials (default: 60)")
    parser.add_argument("--timeout", type=int, default=900,
                        help="Timeout in seconds (default: 900)")
    parser.add_argument("--holdout-years", type=int, nargs="+", default=None,
                        help="Years to hold out (default: last 2 in data)")
    args = parser.parse_args()

    try:
        import optuna
    except ImportError:
        logger.error("optuna not installed. Run: pip install optuna")
        raise SystemExit(1)

    from src.data_collection.f1_historical import (
        load_lap_extras,
        load_pit_stops,
        load_quali_results,
        load_race_results,
        load_sprint_results,
    )
    from src.data_collection.weather import load_weather_history
    from src.features.build_features import FEATURE_COLS, build_feature_matrix
    from src.models.plackett_luce import PlackettLuceSampler
    from src.models.xgb_model import XGBRacePredictor
    from src.training.evaluate import evaluate_race_predictions
    from src.training.weights import compute_recency_sample_weight

    # ------------------------------------------------------------------ #
    # Load data (reuse cached feature matrix if available, else rebuild)
    # ------------------------------------------------------------------ #
    MODELS_DIR = ROOT / "models"
    FM_PATH = ROOT / "data" / "processed" / "feature_matrix.parquet"

    if FM_PATH.exists():
        logger.info(f"Loading cached feature matrix from {FM_PATH}")
        fm = pd.read_parquet(FM_PATH)
    else:
        logger.info("Feature matrix not found — rebuilding...")
        race_df = load_race_results()
        quali_df = load_quali_results()
        try:
            weather_df = load_weather_history()
        except FileNotFoundError:
            weather_df = None
        try:
            lap_extras_df = load_lap_extras()
        except FileNotFoundError:
            lap_extras_df = None
        try:
            pit_df = load_pit_stops()
        except FileNotFoundError:
            pit_df = None
        try:
            sprint_df = load_sprint_results()
        except FileNotFoundError:
            sprint_df = None
        fm = build_feature_matrix(race_df, quali_df, weather_df, None, lap_extras_df, pit_df, sprint_df)

    if "target_position" not in fm.columns:
        logger.error("target_position missing from feature matrix. Run build_features first.")
        raise SystemExit(1)

    all_years = sorted(fm["year"].unique())
    holdout_years = args.holdout_years or all_years[-2:]
    train_years = [y for y in all_years if y not in holdout_years]

    train_df = fm[fm["year"].isin(train_years)].reset_index(drop=True)
    test_df = fm[fm["year"].isin(holdout_years)].reset_index(drop=True)

    logger.info(f"Train: {len(train_df)} rows {sorted(train_df['year'].unique())}")
    logger.info(f"Test : {len(test_df)} rows {holdout_years}")

    X_train = pd.DataFrame(train_df[FEATURE_COLS].values, columns=FEATURE_COLS)
    y_train = train_df["target_position"].values
    train_weight = compute_recency_sample_weight(train_df["year"].values)

    X_test = pd.DataFrame(test_df[FEATURE_COLS].values, columns=FEATURE_COLS)

    # Faster sampler for tuning (5k sims instead of 10k)
    sampler = PlackettLuceSampler(temperature=1.0, n_simulations=5_000, seed=42)

    # ------------------------------------------------------------------ #
    # Objective
    # ------------------------------------------------------------------ #
    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "objective": "reg:absoluteerror",
            "n_jobs": -1,
            "random_state": 42,
        }

        model = XGBRacePredictor(params=params)
        model.fit(X_train, y_train, sample_weight=train_weight)

        pred_positions = model.predict(X_test)
        metrics, _ = evaluate_race_predictions(test_df, pred_positions, sampler)

        # Primary: maximise winner_accuracy; secondary: minimise brier_p1
        # Combine into single minimisation target: -acc + 0.1 * brier
        brier = metrics["brier_p1"] or 0.0
        return -metrics["winner_accuracy"] + 0.1 * brier

    # ------------------------------------------------------------------ #
    # Run study
    # ------------------------------------------------------------------ #
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )

    logger.info(f"Starting Optuna study: {args.n_trials} trials, timeout={args.timeout}s")
    logger.info("This will take a while — each trial trains a full XGB model.")

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_acc = -study.best_value  # approximate (includes brier term)

    logger.info("=" * 55)
    logger.info("Optuna study complete")
    logger.info("=" * 55)
    logger.info(f"Trials completed : {len(study.trials)}")
    logger.info(f"Best params      :")
    for k, v in best_params.items():
        logger.info(f"  {k:<25} = {v}")

    # Re-evaluate best params cleanly (without brier mix)
    best_model = XGBRacePredictor(params={
        **best_params,
        "objective": "reg:absoluteerror",
        "n_jobs": -1,
        "random_state": 42,
    })
    best_model.fit(X_train, y_train, sample_weight=train_weight)
    pred_positions = best_model.predict(X_test)
    final_metrics, _ = evaluate_race_predictions(test_df, pred_positions, sampler)

    logger.info(f"Winner accuracy  : {final_metrics['winner_accuracy']:.1%}")
    logger.info(f"Podium accuracy  : {final_metrics['podium_accuracy']:.1%}")
    logger.info(f"Brier P1         : {final_metrics['brier_p1']:.4f}")

    # Compare to defaults
    default_model = XGBRacePredictor()
    default_model.fit(X_train, y_train, sample_weight=train_weight)
    default_pred = default_model.predict(X_test)
    default_metrics, _ = evaluate_race_predictions(test_df, default_pred, sampler)
    delta = final_metrics["winner_accuracy"] - default_metrics["winner_accuracy"]
    logger.info(f"vs DEFAULT params: {default_metrics['winner_accuracy']:.1%} "
                f"({'+'if delta >= 0 else ''}{delta:.1%})")

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    output = {
        "best_winner_accuracy": float(final_metrics["winner_accuracy"]),
        "best_podium_accuracy": float(final_metrics["podium_accuracy"]),
        "best_brier_p1": float(final_metrics["brier_p1"] or 0),
        "default_winner_accuracy": float(default_metrics["winner_accuracy"]),
        "holdout_years": list(holdout_years),
        "n_trials_completed": len(study.trials),
        "best_params": best_params,
    }
    def _json_default(obj):
        if hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        raise TypeError(f"Not serializable: {type(obj)}")

    out_path = MODELS_DIR / "optuna_best_params.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=_json_default)
    logger.info(f"Saved -> {out_path}")
    logger.info("To use: python scripts/quick_train.py --use-tuned-params")


if __name__ == "__main__":
    main()
