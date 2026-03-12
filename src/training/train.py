"""
Full training pipeline (XGBoost).

Two-pass strategy:
Pass 1 (evaluation): train/calibrate/evaluate with holdout years for honest metrics.
Pass 2 (production): retrain on most recent data split and overwrite active artifacts.

Usage:
    python -m src.training.train
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.models.model_registry import (
    write_active_model_metadata,
    write_holdout_metrics,
)

ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train() -> None:
    from src.features.build_features import FEATURE_COLS, load_feature_matrix
    from src.models.calibration import ProbabilityCalibrator
    from src.models.xgb_model import XGBPodiumPredictor
    from src.training.evaluate import evaluate_by_race, evaluate_predictions
    from src.training.weights import compute_recency_sample_weight

    logger.info("Loading feature matrix...")
    fm = load_feature_matrix()

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
    holdout_df = fm[fm["year"].isin(holdout_years)].reset_index(drop=True)

    logger.info(
        "PASS 1 - Evaluation splits:\n"
        f"  Train:   {len(train_df)} rows ({sorted(train_df['year'].unique())})\n"
        f"  Cal:     {len(cal_df)} rows ({cal_years})\n"
        f"  Holdout: {len(holdout_df)} rows ({holdout_years})"
    )

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

    eval_model = XGBPodiumPredictor()
    eval_model.fit(X_train, y_train, sample_weight=train_weight)
    eval_model.save("xgb_podium")

    calibrator = None
    if len(cal_df) > 0:
        X_cal = pd.DataFrame(cal_df[FEATURE_COLS].values, columns=FEATURE_COLS)
        y_cal = {
            "p1": cal_df["p1"].values,
            "p2": cal_df["p2"].values,
            "p3": cal_df["p3"].values,
        }
        calibrator = ProbabilityCalibrator(method="isotonic")
        calibrator.fit(eval_model.predict_proba(X_cal), y_cal)
        calibrator.save("calibrator")
        logger.info("Evaluation calibrator saved")
    else:
        logger.warning("No calibration data in pass 1; skipping")

    winner_acc_for_eval: float | None = None
    if len(holdout_df) > 0:
        X_holdout = pd.DataFrame(holdout_df[FEATURE_COLS].values, columns=FEATURE_COLS)
        y_holdout = {
            "p1": holdout_df["p1"].values,
            "p2": holdout_df["p2"].values,
            "p3": holdout_df["p3"].values,
        }
        raw_probs = eval_model.predict_proba(X_holdout)
        probs = calibrator.calibrate(raw_probs) if calibrator else raw_probs

        metrics = evaluate_predictions(probs, y_holdout, feature_df=holdout_df)
        race_results = evaluate_by_race(holdout_df, probs)

        winner_acc = race_results["winner_correct"].mean() if len(race_results) > 0 else 0.0
        winner_acc_for_eval = float(winner_acc)
        logger.info(f"Holdout winner accuracy: {winner_acc:.1%} over {len(race_results)} races")
        if "p1" in metrics:
            logger.info(f"Holdout P1 Brier score: {metrics['p1']['brier_score']:.4f}")

        metrics["holdout_years"] = holdout_years
        metrics["cal_years"] = cal_years
        metrics["train_years"] = sorted([int(y) for y in train_df["year"].unique()])
        metrics["winner_accuracy"] = float(winner_acc)
        metrics["n_holdout_races"] = int(len(race_results))

        metric_paths = write_holdout_metrics(metrics, model_family="xgb")
        logger.info(f"Holdout metrics saved -> {metric_paths['generic']}")

        race_results.to_parquet(MODELS_DIR / "holdout_race_results.parquet", index=False)
        logger.info(f"Race-by-race holdout results saved ({len(race_results)} races)")

        logger.info("Computing SHAP values (P1 classifier)...")
        try:
            import shap

            explainer = shap.TreeExplainer(eval_model._models["p1"])
            shap_vals = explainer.shap_values(X_holdout)
            shap_df = pd.DataFrame(shap_vals, columns=FEATURE_COLS)
            shap_df["driver"] = holdout_df["driver"].values
            shap_df["year"] = holdout_df["year"].values
            shap_df["round"] = holdout_df["round"].values
            shap_df["p1"] = y_holdout["p1"]
            shap_df["pred_p1"] = probs["p1"]
            shap_df.to_parquet(MODELS_DIR / "shap_values_p1.parquet", index=False)
            logger.info(f"SHAP values saved ({len(shap_df)} rows)")
        except Exception as exc:
            logger.warning(f"SHAP computation failed: {exc}")
    else:
        logger.warning("No holdout data found; skipping evaluation artifacts")

    # PASS 2 (production)
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

    logger.info(f"Prod train: {len(prod_train_df)} rows, years {sorted(prod_train_df['year'].unique())}")
    logger.info(f"Prod cal:   {len(prod_cal_df)} rows, years {prod_cal_years}")

    X_prod = pd.DataFrame(prod_train_df[FEATURE_COLS].values, columns=FEATURE_COLS)
    y_prod = {
        "p1": prod_train_df["p1"].values,
        "p2": prod_train_df["p2"].values,
        "p3": prod_train_df["p3"].values,
    }
    prod_weight = compute_recency_sample_weight(prod_train_df["year"].values)

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
        prod_calibrator = ProbabilityCalibrator(method="isotonic")
        prod_calibrator.fit(prod_model.predict_proba(X_pcal), y_pcal)
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
        source="train.py",
        feature_count=len(FEATURE_COLS),
        feature_cols=FEATURE_COLS,
    )
    logger.info(f"Active model manifest saved -> {MODELS_DIR / 'active_model.json'}")
    logger.info("Training complete. Ready for inference.")


if __name__ == "__main__":
    train()
