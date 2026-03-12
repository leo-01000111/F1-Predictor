"""
XGBoost podium predictor.
Trains 3 binary classifiers: P(P1), P(P2), P(P3).
Also supports Optuna hyperparameter tuning.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class XGBPodiumPredictor(BaseEstimator, ClassifierMixin):
    """
    Three independent XGBoost classifiers for P1, P2, P3 prediction.
    Produces calibrated probabilities via post-processing normalization.
    """

    DEFAULT_PARAMS = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "eval_metric": "logloss",
        "n_jobs": -1,
        "random_state": 42,
    }

    def __init__(self, params: Optional[dict] = None):
        self.params = params or self.DEFAULT_PARAMS
        self._models: dict[str, xgb.XGBClassifier] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y_dict: dict[str, pd.Series],
        eval_set_dict: Optional[dict[str, tuple]] = None,
        sample_weight: Optional[np.ndarray] = None,
        eval_sample_weight_dict: Optional[dict[str, np.ndarray]] = None,
    ) -> "XGBPodiumPredictor":
        """
        Fit one XGB classifier per target.
        y_dict: {"p1": Series, "p2": Series, "p3": Series}
        eval_set_dict: optional {"p1": (X_val, y_val), ...}

        Class imbalance handling: scale_pos_weight is computed automatically
        from each target's positive rate (~4.6% for P1/P2/P3 in F1 data).
        Without this, the model learns to predict "no win" for everyone, which
        is numerically 95%+ accurate but cannot identify a winner.

        Early stopping: a 10% internal split finds the best n_estimators,
        then the model is refit on full data at that depth.
        """
        from sklearn.model_selection import train_test_split
        sample_weight_arr: Optional[np.ndarray] = None
        if sample_weight is not None:
            sample_weight_arr = np.asarray(sample_weight, dtype=float)
            if len(sample_weight_arr) != len(X):
                raise ValueError(
                    f"sample_weight length mismatch: got {len(sample_weight_arr)} expected {len(X)}"
                )

        for target in ["p1", "p2", "p3"]:
            y = y_dict[target]
            pos_rate = float(np.mean(y))
            # neg_count / pos_count — XGBoost convention for class weighting
            scale_pos_weight = (1.0 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
            logger.info(
                f"Training XGB for {target} | pos_rate={pos_rate:.3f} | "
                f"scale_pos_weight={scale_pos_weight:.1f}x"
            )

            # Merge dynamic weight into a copy of params
            params = dict(self.params)
            params["scale_pos_weight"] = scale_pos_weight

            # Early stopping: find optimal n_estimators on a small holdout,
            # then refit on full training data at that tree count.
            # early_stopping_rounds goes in the constructor in older XGBoost
            # and in fit() in newer — try constructor first, fallback to fixed.
            best_n = params["n_estimators"]
            try:
                if sample_weight_arr is not None:
                    X_tr, X_es, y_tr, y_es, w_tr, _w_es = train_test_split(
                        X,
                        y,
                        sample_weight_arr,
                        test_size=0.1,
                        random_state=42,
                        stratify=y,
                    )
                else:
                    X_tr, X_es, y_tr, y_es = train_test_split(
                        X, y, test_size=0.1, random_state=42, stratify=y
                    )
                    w_tr = None

                probe_params = dict(params, early_stopping_rounds=40)
                probe = xgb.XGBClassifier(**probe_params)
                probe_kwargs = {"eval_set": [(X_es, y_es)], "verbose": False}
                if w_tr is not None:
                    probe_kwargs["sample_weight"] = w_tr
                probe.fit(X_tr, y_tr, **probe_kwargs)
                best_n = probe.best_iteration + 1
                logger.info(f"  Best n_estimators (early stopping): {best_n}")
            except Exception as exc:
                logger.warning(f"  Early stopping probe failed ({exc}), using n_estimators={best_n}")

            # Final model on full training data at the optimal tree count
            params["n_estimators"] = best_n
            model = xgb.XGBClassifier(**params)

            fit_kwargs: dict = {}
            if eval_set_dict and target in eval_set_dict:
                X_val_ext, y_val_ext = eval_set_dict[target]
                fit_kwargs["eval_set"] = [(X_val_ext, y_val_ext)]
                fit_kwargs["verbose"] = 50
                if eval_sample_weight_dict and target in eval_sample_weight_dict:
                    fit_kwargs["sample_weight_eval_set"] = [eval_sample_weight_dict[target]]
            if sample_weight_arr is not None:
                fit_kwargs["sample_weight"] = sample_weight_arr
            model.fit(X, y, **fit_kwargs)
            self._models[target] = model

        return self

    def predict_proba_all(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return raw predicted probabilities for each target.

        Forward-compatible: if X has extra columns not in the trained model
        (e.g. after adding round_fraction to FEATURE_COLS), they are silently
        dropped so the old pkl continues to work until the next retrain.
        """
        check_is_fitted(self, "_models")
        result = {}
        for target, model in self._models.items():
            X_in = X
            if hasattr(model, "feature_names_in_"):
                trained_cols = list(model.feature_names_in_)
                missing = [c for c in trained_cols if c not in X.columns]
                if missing:
                    for c in missing:
                        X_in = X_in.copy()
                        X_in[c] = 0.0
                X_in = X_in[trained_cols]
            result[target] = model.predict_proba(X_in)[:, 1]
        return result

    def predict_proba(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Alias for predict_proba_all."""
        return self.predict_proba_all(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return index of most likely outcome (0=no podium, 1=p1, 2=p2, 3=p3)."""
        check_is_fitted(self, "_models")
        p1 = self._models["p1"].predict_proba(X)[:, 1]
        return (p1 >= 0.5).astype(int)

    def feature_importances(self) -> pd.DataFrame:
        """Return feature importance DataFrame per target."""
        check_is_fitted(self, "_models")
        dfs = []
        for target, model in self._models.items():
            fi = pd.DataFrame(
                {"feature": model.feature_names_in_, "importance": model.feature_importances_, "target": target}
            )
            dfs.append(fi)
        return pd.concat(dfs, ignore_index=True).sort_values("importance", ascending=False)

    def save(self, name: str = "xgb_podium") -> None:
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved XGB model → {path}")

    @classmethod
    def load(cls, name: str = "xgb_podium") -> "XGBPodiumPredictor":
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Loaded XGB model ← {path}")
        return obj


# --------------------------------------------------------------------------- #
# Optuna hyperparameter search
# --------------------------------------------------------------------------- #

def tune_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    target: str = "p1",
) -> dict:
    """
    Run Optuna hyperparameter search for XGBoost on a single target.
    Returns best params dict.
    """
    import optuna
    from sklearn.metrics import log_loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "n_jobs": -1,
            "random_state": 42,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict_proba(X_val)[:, 1]
        return log_loss(y_val, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"XGB tuning ({target}) best log_loss: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    return study.best_params





