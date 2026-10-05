"""
XGBoost race predictor (v2).
Single regression model predicting finish_position per driver.
Outputs are consumed by PlackettLuceSampler to produce coherent podium probabilities.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)



# --------------------------------------------------------------------------- #
# V2: single regression model for finish-position prediction
# --------------------------------------------------------------------------- #

class XGBRacePredictor(BaseEstimator, RegressorMixin):
    """
    Single XGBoost regressor that predicts finish_position for each driver.

    Target: target_position (1 = win, n_starters+1 = DNF).
    Uses MAE objective (reg:absoluteerror) for robustness to the DNF spike.

    Outputs are raw predicted positions (floats). These are fed into
    PlackettLuceSampler to produce coherent P(finish_k) probabilities.
    """

    DEFAULT_PARAMS = {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "reg:absoluteerror",
        "n_jobs": -1,
        "random_state": 42,
    }

    def __init__(self, params: Optional[dict] = None) -> None:
        self.params = params or self.DEFAULT_PARAMS
        self._model: Optional[xgb.XGBRegressor] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "XGBRacePredictor":
        """
        Fit the regressor.

        y: 1-D array of target_position values (float).
        Early stopping: probe on a 10% internal split to find optimal n_estimators,
        then refit on full data at that depth.
        """
        from sklearn.model_selection import train_test_split

        y_arr = np.asarray(y, dtype=float)
        w_arr = np.asarray(sample_weight, dtype=float) if sample_weight is not None else None

        if w_arr is not None and len(w_arr) != len(X):
            raise ValueError(
                f"sample_weight length mismatch: got {len(w_arr)}, expected {len(X)}"
            )

        best_n = self.params["n_estimators"]
        try:
            if w_arr is not None:
                X_tr, X_es, y_tr, y_es, w_tr, _ = train_test_split(
                    X, y_arr, w_arr, test_size=0.1, random_state=42
                )
            else:
                X_tr, X_es, y_tr, y_es = train_test_split(
                    X, y_arr, test_size=0.1, random_state=42
                )
                w_tr = None

            probe_params = dict(self.params, early_stopping_rounds=40)
            probe = xgb.XGBRegressor(**probe_params)
            probe_kwargs: dict = {"eval_set": [(X_es, y_es)], "verbose": False}
            if w_tr is not None:
                probe_kwargs["sample_weight"] = w_tr
            probe.fit(X_tr, y_tr, **probe_kwargs)
            best_n = probe.best_iteration + 1
            logger.info(f"XGBRacePredictor best n_estimators (early stopping): {best_n}")
        except Exception as exc:
            logger.warning(f"Early stopping probe failed ({exc}), using n_estimators={best_n}")

        params = dict(self.params, n_estimators=best_n)
        self._model = xgb.XGBRegressor(**params)
        fit_kwargs: dict = {}
        if w_arr is not None:
            fit_kwargs["sample_weight"] = w_arr
        self._model.fit(X, y_arr, **fit_kwargs)
        logger.info(f"XGBRacePredictor trained on {len(X)} rows, {best_n} trees")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return predicted finish position (float) for each driver.

        Forward-compatible: extra columns not seen during training are dropped.
        Missing trained columns are filled with 0.0.
        """
        check_is_fitted(self, "_model")
        X_in = X
        if hasattr(self._model, "feature_names_in_"):
            trained_cols = list(self._model.feature_names_in_)
            missing = [c for c in trained_cols if c not in X_in.columns]
            if missing:
                X_in = X_in.copy()
                for c in missing:
                    X_in[c] = 0.0
            X_in = X_in[trained_cols]
        return self._model.predict(X_in)

    def feature_importances(self) -> pd.DataFrame:
        check_is_fitted(self, "_model")
        return pd.DataFrame(
            {
                "feature": self._model.feature_names_in_,
                "importance": self._model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

    def save(self, name: str = "xgb_race") -> None:
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved XGBRacePredictor -> {path}")
        try:
            from src.models.model_registry import snapshot_model
            snap = snapshot_model(name, keep_last=5)
            if snap:
                logger.info(f"Snapshot -> {snap.name}")
        except Exception as exc:
            logger.warning(f"Snapshot failed (non-fatal): {exc}")

    @classmethod
    def load(cls, name: str = "xgb_race") -> "XGBRacePredictor":
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Loaded XGBRacePredictor <- {path}")
        return obj







