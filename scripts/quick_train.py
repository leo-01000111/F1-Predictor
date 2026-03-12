"""
Quick training script for race-day use (XGBoost only).

Two-pass strategy:
Pass 1 (evaluation): write holdout metrics and dashboard artifacts.
Pass 2 (production): retrain on recency-maximized split and overwrite active artifacts.

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


def main() -> None:
    import numpy as np
    import pandas as pd

    from src.data_collection.f1_historical import (
        load_practice_results,
        load_quali_results,
        load_race_results,
        load_pit_stops,
        load_sprint_results,
    )
    from src.data_collection.weather import load_weather_history
    from src.features.build_features import FEATURE_COLS, build_feature_matrix, save_feature_matrix
    from src.models.calibration import ProbabilityCalibrator
    from src.models.xgb_model import XGBPodiumPredictor
    from src.training.evaluate import evaluate_by_race, evaluate_predictions
    from src.training.weights import compute_recency_sample_weight

    logger.info("Loading raw data...")
    race_df = load_race_results()
    quali_df = load_quali_results()
    logger.info(f"Race data: {len(race_df)} rows, years {sorted(race_df['year'].unique())}")

    try:
        practice_df = load_practice_results()
        logger.info(f"Practice data: {len(practice_df)} rows")
    except FileNotFoundError:
        logger.warning("No practice data found; FP features will be zero-filled")
        practice_df = None

    try:
        weather_df = load_weather_history()
    except FileNotFoundError:
        logger.warning("No weather data found; weather features will be zero-filled")
        weather_df = None

    try:
        from src.data_collection.f1_historical import load_lap_extras
        lap_extras_df = load_lap_extras()
        logger.info(f"Lap extras loaded: {len(lap_extras_df)} rows")
    except FileNotFoundError:
        logger.info("No lap_extras found; driver_start_delta_avg/start_compound will be zero-filled")
        lap_extras_df = None

    try:
        pit_df = load_pit_stops()
        logger.info(f"Pit stop data loaded: {len(pit_df)} rows")
    except FileNotFoundError:
        logger.info("No pit_stops.parquet; team_pit_delta_s will be NaN. "
                    "Run: python scripts/backfill_pit_sprint.py --pit")
        pit_df = None

    try:
        sprint_df = load_sprint_results()
        logger.info(f"Sprint data loaded: {len(sprint_df)} rows")
    except FileNotFoundError:
        logger.info("No sprint_results.parquet; sprint_position_rel will be 0.5. "
                    "Run: python scripts/backfill_pit_sprint.py --sprint")
        sprint_df = None

    logger.info("Building feature matrix...")
    fm = build_feature_matrix(race_df, quali_df, weather_df, practice_df, lap_extras_df, pit_df, sprint_df)
    save_feature_matrix(fm)
    logger.info(f"Feature matrix: {len(fm)} rows x {len(FEATURE_COLS)} features")

    all_years = sorted(fm["year"].unique())
    logger.info(f"Available years: {all_years}")

    # PASS 1 split (evaluation)
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
    cal_df = fm[fm["year"].isin(cal_years)].reset_index(drop=True)
    test_df = fm[fm["year"].isin(holdout_years)].reset_index(drop=True)

    logger.info(f"Train: {len(train_df)} rows ({sorted(train_df['year'].unique())})")
    logger.info(f"Cal:   {len(cal_df)} rows ({cal_years})")
    logger.info(f"Test:  {len(test_df)} rows ({holdout_years})")

    if len(train_df) < 100:
        logger.error("Not enough training data. Run data collection first.")
        raise SystemExit(1)

    X_train = pd.DataFrame(train_df[FEATURE_COLS].values, columns=FEATURE_COLS)
    y_train = {
        "p1": train_df["p1"].values,
        "p2": train_df["p2"].values,
        "p3": train_df["p3"].values,
    }

    train_weight = compute_recency_sample_weight(train_df["year"].values)
    logger.info(
        f"Evaluation recency weights: min={train_weight.min():.3f}, "
        f"max={train_weight.max():.3f}, mean={train_weight.mean():.3f}"
    )

    logger.info("Training evaluation XGBoost...")
    model = XGBPodiumPredictor()
    model.fit(X_train, y_train, sample_weight=train_weight)
    model.save("xgb_podium")

    calibrator = None
    if len(cal_df) > 0:
        X_cal = pd.DataFrame(cal_df[FEATURE_COLS].values, columns=FEATURE_COLS)
        y_cal = {
            "p1": cal_df["p1"].values,
            "p2": cal_df["p2"].values,
            "p3": cal_df["p3"].values,
        }
        raw_cal = model.predict_proba(X_cal)
        calibrator = ProbabilityCalibrator(method="isotonic")
        calibrator.fit(raw_cal, y_cal)
        calibrator.save("calibrator")
        logger.info("Evaluation calibrator saved")
    else:
        logger.warning("No evaluation calibration data; skipping")

    MODELS_DIR = ROOT / "models"
    winner_acc_for_eval: float | None = None

    if len(test_df) > 0:
        X_test = pd.DataFrame(test_df[FEATURE_COLS].values, columns=FEATURE_COLS)
        y_test = {
            "p1": test_df["p1"].values,
            "p2": test_df["p2"].values,
            "p3": test_df["p3"].values,
        }
        raw_test = model.predict_proba(X_test)
        probs = calibrator.calibrate(raw_test) if calibrator else raw_test

        metrics = evaluate_predictions(probs, y_test, feature_df=test_df)
        race_res = evaluate_by_race(test_df, probs)
        winner_acc = race_res["winner_correct"].mean() if len(race_res) > 0 else 0.0
        winner_acc_for_eval = float(winner_acc)

        logger.info(f"Holdout winner accuracy: {winner_acc:.1%} over {len(race_res)} races")
        if "p1" in metrics:
            logger.info(f"Holdout P1 Brier score: {metrics['p1']['brier_score']:.4f}")

        # Calibration curve buckets
        p1_probs = probs["p1"]
        p1_true = y_test["p1"]
        bins = np.linspace(0, 1, 11)
        bucket_idx = np.digitize(p1_probs, bins) - 1
        calib_rows = []
        for b_i in range(10):
            mask = bucket_idx == b_i
            if mask.sum() > 0:
                calib_rows.append(
                    {
                        "prob_bin_mid": float(bins[b_i] + 0.05),
                        "mean_predicted": float(p1_probs[mask].mean()),
                        "mean_actual": float(p1_true[mask].mean()),
                        "n": int(mask.sum()),
                    }
                )
        calib_df = pd.DataFrame(calib_rows)

        # Per-year summary
        year_metrics = []
        for yr, grp in test_df.groupby("year"):
            idx = grp.index.to_numpy()
            base = test_df.index[0]
            local = idx - base
            y_yr = y_test["p1"][local]
            p_yr = probs["p1"][local]
            from sklearn.metrics import brier_score_loss

            year_metrics.append({"year": int(yr), "brier_p1": float(brier_score_loss(y_yr, p_yr))})

        metrics["holdout_years"] = holdout_years
        metrics["cal_years"] = cal_years
        metrics["train_years"] = sorted([int(y) for y in train_df["year"].unique()])
        metrics["winner_accuracy"] = float(winner_acc)
        metrics["n_holdout_races"] = int(len(race_res))
        metrics["year_metrics"] = year_metrics

        metric_paths = write_holdout_metrics(metrics, model_family="xgb")
        logger.info(f"Holdout metrics saved -> {metric_paths['generic']}")
        logger.info(f"Model-specific holdout metrics saved -> {metric_paths['family']}")

        race_res.to_parquet(MODELS_DIR / "holdout_race_results.parquet", index=False)
        logger.info(f"Saved holdout_race_results.parquet ({len(race_res)} races)")

        if not calib_df.empty:
            calib_df.to_parquet(MODELS_DIR / "calibration_curve.parquet", index=False)
            logger.info("Saved calibration_curve.parquet")

        logger.info("Computing SHAP values (P1 classifier)...")
        try:
            import shap

            xgb_p1 = model._models["p1"]
            explainer = shap.TreeExplainer(xgb_p1)
            shap_vals = explainer.shap_values(X_test)
            shap_df = pd.DataFrame(shap_vals, columns=FEATURE_COLS)
            shap_df["driver"] = test_df["driver"].values
            shap_df["year"] = test_df["year"].values
            shap_df["round"] = test_df["round"].values
            shap_df["p1"] = y_test["p1"]
            shap_df["pred_p1"] = probs["p1"]
            shap_df.to_parquet(MODELS_DIR / "shap_values_p1.parquet", index=False)
            logger.info(f"Saved shap_values_p1.parquet ({len(shap_df)} rows)")
        except Exception as exc:
            logger.warning(f"SHAP computation failed: {exc}")
    else:
        logger.warning("No holdout data to evaluate on")

    # PASS 2 split (production)
    logger.info("=" * 60)
    logger.info("PASS 2 - Production retrain")
    logger.info("=" * 60)

    if len(all_years) >= 2:
        prod_cal_years = [all_years[-1]]
        prod_train_years = all_years[:-1]
    else:
        prod_cal_years = []
        prod_train_years = all_years

    prod_train_df = fm[fm["year"].isin(prod_train_years)].reset_index(drop=True)
    prod_cal_df = fm[fm["year"].isin(prod_cal_years)].reset_index(drop=True)

    logger.info(f"Prod train : {len(prod_train_df)} rows - years {sorted(prod_train_df['year'].unique())}")
    logger.info(f"Prod cal   : {len(prod_cal_df)} rows - years {prod_cal_years}")

    X_prod = pd.DataFrame(prod_train_df[FEATURE_COLS].values, columns=FEATURE_COLS)
    y_prod = {
        "p1": prod_train_df["p1"].values,
        "p2": prod_train_df["p2"].values,
        "p3": prod_train_df["p3"].values,
    }

    prod_weight = compute_recency_sample_weight(prod_train_df["year"].values)
    logger.info(
        f"Production recency weights: min={prod_weight.min():.3f}, "
        f"max={prod_weight.max():.3f}, mean={prod_weight.mean():.3f}"
    )

    prod_model = XGBPodiumPredictor()
    prod_model.fit(X_prod, y_prod, sample_weight=prod_weight)
    prod_model.save("xgb_podium")
    logger.info("Production XGBoost saved")

    if len(prod_cal_df) > 0:
        X_pcal = pd.DataFrame(prod_cal_df[FEATURE_COLS].values, columns=FEATURE_COLS)
        y_pcal = {
            "p1": prod_cal_df["p1"].values,
            "p2": prod_cal_df["p2"].values,
            "p3": prod_cal_df["p3"].values,
        }
        raw_pcal = prod_model.predict_proba(X_pcal)
        prod_calibrator = ProbabilityCalibrator(method="isotonic")
        prod_calibrator.fit(raw_pcal, y_pcal)
        prod_calibrator.save("calibrator")
        logger.info("Production calibrator saved")
    else:
        logger.warning("No production calibration data; skipping")

    write_active_model_metadata(
        model_family="xgb",
        model_artifact="xgb_podium.pkl",
        calibrator_artifact="calibrator.pkl",
        train_years=[int(y) for y in sorted(prod_train_df["year"].unique())],
        calibration_years=[int(y) for y in prod_cal_years],
        holdout_years=[int(y) for y in holdout_years],
        winner_accuracy=winner_acc_for_eval,
        source="quick_train.py",
        feature_count=len(FEATURE_COLS),
        feature_cols=FEATURE_COLS,
    )
    logger.info(f"Active model manifest saved -> {MODELS_DIR / 'active_model.json'}")
    logger.info(
        f"Production model active: trained {sorted(prod_train_df['year'].unique())}, "
        f"calibrated on {prod_cal_years}"
    )
    logger.info("Quick training complete. Ready for inference.")


if __name__ == "__main__":
    main()




