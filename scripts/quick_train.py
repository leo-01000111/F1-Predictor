"""
Quick training script — v2 (regression + Plackett-Luce).

Trains a single XGBRacePredictor to predict finish_position.
Evaluation uses PlackettLuceSampler to convert predicted positions to
coherent podium probabilities before computing metrics.

Two-pass strategy:
  Pass 1 (evaluation): honest holdout metrics + dashboard artifacts.
  Pass 2 (production): retrain on recency-maximised split, overwrite active artifacts.

Usage:
    python scripts/quick_train.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models.model_registry import (  # noqa: E402
    write_active_model_metadata,
    write_holdout_metrics,
)


def _load_tuned_params(use_tuned: bool) -> "dict | None":
    """Return tuned XGB params dict if --use-tuned-params and file exists, else None."""
    if not use_tuned:
        return None
    import json
    from src.models.xgb_model import XGBRacePredictor
    tuned_path = ROOT / "models" / "optuna_best_params.json"
    if not tuned_path.exists():
        logger.warning("--use-tuned-params: models/optuna_best_params.json not found, using defaults")
        return None
    with open(tuned_path, encoding="utf-8") as f:
        data = json.load(f)
    params = {
        **XGBRacePredictor.DEFAULT_PARAMS,
        **data["best_params"],
        "objective": "reg:absoluteerror",
        "n_jobs": -1,
        "random_state": 42,
    }
    logger.info(f"Using tuned params (holdout acc={data.get('best_winner_accuracy', '?'):.1%})")
    return params


def main() -> None:
    import argparse
    import numpy as np
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--use-tuned-params", action="store_true",
                        help="Load XGB params from models/optuna_best_params.json")
    args = parser.parse_args()

    from src.data_collection.f1_historical import (
        load_lap_extras,
        load_pit_stops,
        load_practice_results,
        load_quali_results,
        load_race_results,
        load_sprint_results,
    )
    from src.data_collection.weather import load_weather_history
    from src.features.build_features import FEATURE_COLS, build_feature_matrix, save_feature_matrix
    from src.models.plackett_luce import PlackettLuceSampler
    from src.models.xgb_model import XGBRacePredictor
    from src.training.evaluate import evaluate_race_predictions
    from src.training.weights import compute_recency_sample_weight

    # ------------------------------------------------------------------ #
    # Load raw data
    # ------------------------------------------------------------------ #
    logger.info("Loading raw data...")
    race_df = load_race_results()
    quali_df = load_quali_results()
    logger.info(f"Race data: {len(race_df)} rows, years {sorted(race_df['year'].unique())}")

    try:
        practice_df = load_practice_results()
        logger.info(f"Practice data: {len(practice_df)} rows")
    except FileNotFoundError:
        logger.warning("No practice data; FP features will be zero-filled")
        practice_df = None

    try:
        weather_df = load_weather_history()
    except FileNotFoundError:
        logger.warning("No weather data; weather features will be zero-filled")
        weather_df = None

    try:
        lap_extras_df = load_lap_extras()
        logger.info(f"Lap extras: {len(lap_extras_df)} rows")
    except FileNotFoundError:
        logger.info("No lap_extras; driver_start_delta_avg/start_compound will be zero-filled")
        lap_extras_df = None

    try:
        pit_df = load_pit_stops()
        logger.info(f"Pit stop data: {len(pit_df)} rows")
    except FileNotFoundError:
        logger.info("No pit_stops.parquet; team_pit_delta_s will be NaN")
        pit_df = None

    try:
        sprint_df = load_sprint_results()
        logger.info(f"Sprint data: {len(sprint_df)} rows")
    except FileNotFoundError:
        logger.info("No sprint_results.parquet; sprint_position_rel will be 0.5")
        sprint_df = None

    # ------------------------------------------------------------------ #
    # Build feature matrix
    # ------------------------------------------------------------------ #
    logger.info("Building feature matrix...")
    fm = build_feature_matrix(race_df, quali_df, weather_df, practice_df, lap_extras_df, pit_df, sprint_df)
    save_feature_matrix(fm)
    logger.info(f"Feature matrix: {len(fm)} rows x {len(FEATURE_COLS)} features")

    if "target_position" not in fm.columns:
        logger.error("target_position column missing from feature matrix. Rebuild required.")
        raise SystemExit(1)

    all_years = sorted(fm["year"].unique())
    logger.info(f"Available years: {all_years}")

    # ------------------------------------------------------------------ #
    # PASS 1 — evaluation split
    # ------------------------------------------------------------------ #
    if len(all_years) >= 8:
        holdout_years = all_years[-2:]
        cal_years = [all_years[-3]]
    elif len(all_years) >= 5:
        holdout_years = [all_years[-1]]
        cal_years = [all_years[-2]]
    elif len(all_years) >= 3:
        holdout_years = []
        cal_years = [all_years[-1]]
    else:
        holdout_years = []
        cal_years = []

    excluded = set(holdout_years + cal_years)
    train_df = fm[~fm["year"].isin(excluded)].reset_index(drop=True)
    test_df = fm[fm["year"].isin(holdout_years)].reset_index(drop=True)

    logger.info(f"Train : {len(train_df)} rows ({sorted(train_df['year'].unique())})")
    logger.info(f"Test  : {len(test_df)} rows ({holdout_years})")

    if len(train_df) < 100:
        logger.error("Not enough training data. Run data collection first.")
        raise SystemExit(1)

    X_train = pd.DataFrame(train_df[FEATURE_COLS].values, columns=FEATURE_COLS)
    y_train = train_df["target_position"].values
    train_weight = compute_recency_sample_weight(train_df["year"].values)
    logger.info(
        f"Recency weights: min={train_weight.min():.3f} "
        f"max={train_weight.max():.3f} mean={train_weight.mean():.3f}"
    )

    xgb_params = _load_tuned_params(args.use_tuned_params)

    logger.info("Training evaluation XGBRacePredictor...")
    model = XGBRacePredictor(params=xgb_params)
    model.fit(X_train, y_train, sample_weight=train_weight)
    model.save("xgb_race")

    MODELS_DIR = ROOT / "models"
    winner_acc_for_manifest: float | None = None

    sampler = PlackettLuceSampler(temperature=1.0, n_simulations=10_000)

    if len(test_df) > 0:
        X_test = pd.DataFrame(test_df[FEATURE_COLS].values, columns=FEATURE_COLS)
        pred_positions = model.predict(X_test)

        metrics, race_res = evaluate_race_predictions(test_df, pred_positions, sampler)
        winner_acc_for_manifest = metrics["winner_accuracy"]

        metrics["holdout_years"] = holdout_years
        metrics["cal_years"] = cal_years
        metrics["train_years"] = sorted(int(y) for y in train_df["year"].unique())

        metric_paths = write_holdout_metrics(metrics, model_family="xgb_v2")
        logger.info(f"Holdout metrics saved -> {metric_paths['generic']}")
        logger.info(f"Model-specific metrics -> {metric_paths['family']}")

        race_res.to_parquet(MODELS_DIR / "holdout_race_results.parquet", index=False)
        logger.info(f"Saved holdout_race_results.parquet ({len(race_res)} races)")

        # Calibration curve for P(P1) — derived from Plackett-Luce probs
        # Build per-driver P(P1) predictions across entire test set for calibration curve
        all_p1_preds = np.zeros(len(test_df))
        for (year, rnd), group in test_df.reset_index(drop=True).groupby(["year", "round"]):
            idx = group.index
            race_pred = pred_positions[idx]
            mat = sampler.simulate(race_pred)
            all_p1_preds[idx] = mat[:, 0]

        p1_true = test_df["p1"].values
        bins = np.linspace(0, 1, 11)
        bucket_idx = np.digitize(all_p1_preds, bins) - 1
        calib_rows = []
        for b_i in range(10):
            mask = bucket_idx == b_i
            if mask.sum() > 0:
                calib_rows.append({
                    "prob_bin_mid": float(bins[b_i] + 0.05),
                    "mean_predicted": float(all_p1_preds[mask].mean()),
                    "mean_actual": float(p1_true[mask].mean()),
                    "n": int(mask.sum()),
                })
        calib_df = pd.DataFrame(calib_rows)
        if not calib_df.empty:
            calib_df.to_parquet(MODELS_DIR / "calibration_curve.parquet", index=False)
            logger.info("Saved calibration_curve.parquet")

        # SHAP values
        logger.info("Computing SHAP values (regressor)...")
        try:
            import shap

            explainer = shap.TreeExplainer(model._model)
            shap_vals = explainer.shap_values(X_test)
            shap_df = pd.DataFrame(shap_vals, columns=FEATURE_COLS)
            shap_df["driver"] = test_df["driver"].values
            shap_df["year"] = test_df["year"].values
            shap_df["round"] = test_df["round"].values
            shap_df["p1"] = test_df["p1"].values
            shap_df["pred_p1"] = all_p1_preds
            shap_df["pred_position"] = pred_positions
            shap_df.to_parquet(MODELS_DIR / "shap_values_p1.parquet", index=False)
            logger.info(f"Saved shap_values_p1.parquet ({len(shap_df)} rows)")
        except Exception as exc:
            logger.warning(f"SHAP computation failed: {exc}")
    else:
        logger.warning("No holdout data; skipping evaluation")

    # ------------------------------------------------------------------ #
    # PASS 2 — production retrain
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("PASS 2 - Production retrain")
    logger.info("=" * 60)

    if len(all_years) >= 2:
        prod_train_years = all_years[:-1]
        prod_cal_years = [all_years[-1]]
    else:
        prod_train_years = all_years
        prod_cal_years = []

    prod_train_df = fm[fm["year"].isin(prod_train_years)].reset_index(drop=True)
    logger.info(f"Prod train: {len(prod_train_df)} rows, years {sorted(prod_train_df['year'].unique())}")

    X_prod = pd.DataFrame(prod_train_df[FEATURE_COLS].values, columns=FEATURE_COLS)
    y_prod = prod_train_df["target_position"].values
    prod_weight = compute_recency_sample_weight(prod_train_df["year"].values)

    prod_model = XGBRacePredictor(params=xgb_params)
    prod_model.fit(X_prod, y_prod, sample_weight=prod_weight)
    prod_model.save("xgb_race")
    logger.info("Production XGBRacePredictor saved -> models/xgb_race.pkl")

    write_active_model_metadata(
        model_family="xgb_v2",
        model_artifact="xgb_race.pkl",
        calibrator_artifact=None,
        train_years=[int(y) for y in sorted(prod_train_df["year"].unique())],
        calibration_years=[int(y) for y in prod_cal_years],
        holdout_years=[int(y) for y in holdout_years],
        winner_accuracy=winner_acc_for_manifest,
        source="quick_train.py",
        feature_count=len(FEATURE_COLS),
        feature_cols=FEATURE_COLS,
    )
    logger.info("Active model manifest saved -> models/active_model.json")
    logger.info("Quick training complete. Ready for inference.")


if __name__ == "__main__":
    main()
