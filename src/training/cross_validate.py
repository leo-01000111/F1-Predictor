"""
Leave-one-season-out (LOSO) cross-validation for XGBRacePredictor.

For each season in [min_year, max_year], trains on all other seasons and
evaluates on that held-out season. Produces honest per-season metrics and
mean +/- std confidence intervals — a more reliable picture of model
performance than a single held-out window.

Usage:
    python -m src.training.cross_validate
    python -m src.training.cross_validate --min-year 2019 --max-year 2024
    python -m src.training.cross_validate --use-tuned-params

Outputs:
    models/loso_cv_results.parquet   — one row per fold
    models/loso_cv_summary.json      — mean +/- std across folds
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger


def main() -> None:
    import argparse

    import numpy as np
    import pandas as pd

    parser = argparse.ArgumentParser(description="Leave-one-season-out cross-validation")
    parser.add_argument("--min-year", type=int, default=2019,
                        help="Earliest season to use as a holdout fold (default: 2019)")
    parser.add_argument("--max-year", type=int, default=None,
                        help="Latest season to use as holdout (default: last in data)")
    parser.add_argument("--use-tuned-params", action="store_true",
                        help="Load params from models/optuna_best_params.json")
    args = parser.parse_args()

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

    MODELS_DIR = ROOT / "models"
    FM_PATH = ROOT / "data" / "processed" / "feature_matrix.parquet"

    # ------------------------------------------------------------------ #
    # Load feature matrix
    # ------------------------------------------------------------------ #
    if FM_PATH.exists():
        logger.info(f"Loading feature matrix from {FM_PATH}")
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
        logger.error("target_position column missing. Run build_features first.")
        raise SystemExit(1)

    all_years = sorted(fm["year"].unique())
    max_year = args.max_year or all_years[-1]
    fold_years = [y for y in all_years if args.min_year <= y <= max_year]

    if not fold_years:
        logger.error(f"No years in [{args.min_year}, {max_year}]. Check --min-year / --max-year.")
        raise SystemExit(1)

    logger.info(f"LOSO CV folds ({len(fold_years)}): {fold_years}")

    # ------------------------------------------------------------------ #
    # Load hyperparameters
    # ------------------------------------------------------------------ #
    xgb_params: dict | None = None
    if args.use_tuned_params:
        tuned_path = MODELS_DIR / "optuna_best_params.json"
        if tuned_path.exists():
            with open(tuned_path, encoding="utf-8") as f:
                data = json.load(f)
            raw = data["best_params"]
            xgb_params = {
                **XGBRacePredictor.DEFAULT_PARAMS,
                **raw,
                "objective": "reg:absoluteerror",
                "n_jobs": -1,
                "random_state": 42,
            }
            logger.info(f"Using tuned params from {tuned_path}")
        else:
            logger.warning("optuna_best_params.json not found — using DEFAULT_PARAMS")

    sampler = PlackettLuceSampler(temperature=1.0, n_simulations=5_000, seed=42)

    # ------------------------------------------------------------------ #
    # Run folds
    # ------------------------------------------------------------------ #
    fold_results = []

    for fold_year in fold_years:
        train_df = fm[fm["year"] != fold_year].reset_index(drop=True)
        test_df = fm[fm["year"] == fold_year].reset_index(drop=True)

        if len(train_df) < 100:
            logger.warning(f"Fold {fold_year}: only {len(train_df)} training rows, skipping")
            continue
        if len(test_df) == 0:
            logger.warning(f"Fold {fold_year}: no test rows, skipping")
            continue

        X_train = pd.DataFrame(train_df[FEATURE_COLS].values, columns=FEATURE_COLS)
        y_train = train_df["target_position"].values
        train_weight = compute_recency_sample_weight(train_df["year"].values)

        X_test = pd.DataFrame(test_df[FEATURE_COLS].values, columns=FEATURE_COLS)

        model = XGBRacePredictor(params=xgb_params)
        model.fit(X_train, y_train, sample_weight=train_weight)

        pred_positions = model.predict(X_test)
        metrics, race_df_out = evaluate_race_predictions(test_df, pred_positions, sampler)

        fold_results.append({
            "year": fold_year,
            "n_races": metrics["n_holdout_races"],
            "winner_accuracy": metrics["winner_accuracy"],
            "podium_accuracy": metrics["podium_accuracy"],
            "brier_p1": metrics["brier_p1"],
            "mean_spearman_r": metrics.get("mean_spearman_r"),
            "mean_mae_position": metrics.get("mean_mae_position"),
        })

        logger.info(
            f"  {fold_year}  winner={metrics['winner_accuracy']:.1%}  "
            f"podium={metrics['podium_accuracy']:.1%}  "
            f"brier={metrics['brier_p1']:.4f}  "
            f"spearman={metrics.get('mean_spearman_r', float('nan')):.3f}  "
            f"mae_pos={metrics.get('mean_mae_position', float('nan')):.2f}"
        )

    if not fold_results:
        logger.error("No folds completed successfully.")
        return

    results_df = pd.DataFrame(fold_results)

    # ------------------------------------------------------------------ #
    # Summary statistics
    # ------------------------------------------------------------------ #
    def _stats(series: pd.Series) -> dict:
        s = series.dropna()
        return {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
        }

    summary = {
        "n_folds": len(fold_results),
        "fold_years": [int(r["year"]) for r in fold_results],
        "params_source": "optuna" if args.use_tuned_params and xgb_params else "default",
        "winner_accuracy": _stats(results_df["winner_accuracy"]),
        "podium_accuracy": _stats(results_df["podium_accuracy"]),
        "brier_p1": _stats(results_df["brier_p1"]),
        "mean_spearman_r": _stats(results_df["mean_spearman_r"]),
        "mean_mae_position": _stats(results_df["mean_mae_position"]),
    }

    logger.info("=" * 55)
    logger.info("LOSO CV Summary")
    logger.info("=" * 55)
    logger.info(f"Folds            : {summary['n_folds']} ({fold_years[0]}-{fold_years[-1]})")
    logger.info(
        f"Winner accuracy  : {summary['winner_accuracy']['mean']:.1%} "
        f"+/- {summary['winner_accuracy']['std']:.1%}  "
        f"[{summary['winner_accuracy']['min']:.1%} - {summary['winner_accuracy']['max']:.1%}]"
    )
    logger.info(
        f"Podium accuracy  : {summary['podium_accuracy']['mean']:.1%} "
        f"+/- {summary['podium_accuracy']['std']:.1%}"
    )
    logger.info(
        f"Brier P1         : {summary['brier_p1']['mean']:.4f} "
        f"+/- {summary['brier_p1']['std']:.4f}"
    )
    logger.info(
        f"Spearman r       : {summary['mean_spearman_r']['mean']:.3f} "
        f"+/- {summary['mean_spearman_r']['std']:.3f}"
    )
    logger.info(
        f"MAE position     : {summary['mean_mae_position']['mean']:.2f} "
        f"+/- {summary['mean_mae_position']['std']:.2f}"
    )

    # ------------------------------------------------------------------ #
    # Save outputs
    # ------------------------------------------------------------------ #
    results_df.to_parquet(MODELS_DIR / "loso_cv_results.parquet", index=False)
    with open(MODELS_DIR / "loso_cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved -> models/loso_cv_results.parquet")
    logger.info("Saved -> models/loso_cv_summary.json")


if __name__ == "__main__":
    main()
