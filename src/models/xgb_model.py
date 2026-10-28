"""
XGBoost race models.
  v2 XGBRacePredictor: regression on finish_position.
  v3 XGBRaceRanker: learning-to-rank per race.
  v4 TwoStageRaceModel: finishers-only ranker + finish probability (production).
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


# --------------------------------------------------------------------------- #
# V3: learning-to-rank model (production since 2026-10-28)
# --------------------------------------------------------------------------- #

class XGBRaceRanker(XGBRacePredictor):
    """
    XGBoost ranker with one query group per race (NDCG objective).

    Trained on the finishing order directly instead of regressing the position:
    walk-forward 2018-2026 (191 races) top-3 PL log loss 4.861 vs 5.160 for the
    v2 regressor and 5.296 for the grid (scripts/backtest.py, ndcg_all).

    predict() returns -score / temperature: a "predicted position" (lower is
    better) already on the Plackett-Luce scale, so callers keep using
    PlackettLuceSampler(temperature=1.0). The temperature is fitted on a season
    the model was not trained on (fit_temperature).
    """

    DEFAULT_PARAMS = {
        "objective": "rank:ndcg",
        "lambdarank_pair_method": "topk",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "n_jobs": -1,
        "random_state": 42,
    }

    def __init__(self, params: Optional[dict] = None, n_seeds: int = 1) -> None:
        super().__init__(params=params or self.DEFAULT_PARAMS)
        self.temperature = 1.0
        # n_seeds > 1: average the scores of rankers that differ only in their
        # random seed (row/column subsampling), which lowers the variance
        self.n_seeds = n_seeds
        self._seed_models: list = []

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        qid: Optional[np.ndarray] = None,
    ) -> "XGBRaceRanker":
        """
        y: target_position (1 = win, n_starters + 1 = DNF).
        qid: race id per row (e.g. year * 100 + round). Required.
        sample_weight: per row; the first row of each race gives the race weight.
        Relevance = n_starters + 1 - target_position, so the winner has the
        highest label and DNFs get 0.
        """
        if qid is None:
            raise ValueError("XGBRaceRanker.fit needs qid (one id per race)")
        qid = np.asarray(qid)
        order = np.argsort(qid, kind="stable")
        X_s = X.iloc[order].reset_index(drop=True)
        q_s = qid[order]
        y_s = np.asarray(y, dtype=float)[order]
        n = pd.Series(q_s).map(pd.Series(q_s).value_counts()).to_numpy()
        rel = np.clip(n + 1 - y_s, 0, None)

        fit_kwargs: dict = {"qid": q_s}
        if sample_weight is not None:
            w_s = np.asarray(sample_weight, dtype=float)[order]
            first = np.r_[True, q_s[1:] != q_s[:-1]]
            fit_kwargs["sample_weight"] = w_s[first]

        seed0 = self.params.get("random_state", 42)
        self._seed_models = []
        for i in range(getattr(self, "n_seeds", 1)):
            m = xgb.XGBRanker(**dict(self.params, random_state=seed0 + i))
            m.fit(X_s, rel, **fit_kwargs)
            self._seed_models.append(m)
        self._model = self._seed_models[0]   # feature names / SHAP use the first
        logger.info(f"XGBRaceRanker trained on {len(X_s)} rows, {len(np.unique(q_s))} races, "
                    f"{len(self._seed_models)} seed(s)")
        return self

    def _raw_score(self, X: pd.DataFrame) -> np.ndarray:
        models = getattr(self, "_seed_models", None) or [self._model]
        scores = []
        for m in models:
            self._model = m
            scores.append(super().predict(X))
        self._model = models[0]
        return np.mean(scores, axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return -self._raw_score(X) / self.temperature

    def predict_finish(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """P(finish) per car; None for a pace-only model (every car assumed to finish)."""
        return None

    def fit_temperature(self, X: pd.DataFrame, races: pd.DataFrame) -> float:
        """Fit the PL temperature on out-of-sample races (year, round, finish_position)."""
        from src.training.backtest import fit_temperature

        self.temperature = 1.0
        self.temperature = fit_temperature(self.predict(X), races.reset_index(drop=True),
                                           self.predict_finish(X))
        logger.info(f"{type(self).__name__} temperature = {self.temperature}")
        return self.temperature


# --------------------------------------------------------------------------- #
# V4: two-stage model (production since 2026-10-28)
# --------------------------------------------------------------------------- #

class TwoStageRaceModel(XGBRaceRanker):
    """
    Pace ranker trained on finishers only + a finish-probability model.

    Walk-forward 2018-2026 (191 races) top-3 log loss: 4.636 vs 4.861 for the
    v3 ranker (-0.225, 95% CI -0.351..-0.114) and 5.296 for the grid.

    predict() is the pace part (-score / T, as v3). predict_finish() gives
    P(finish). Probabilities: src.models.two_stage.simulate(pred, q, 1.0).

    Settings from scripts/tune_ranker.py (2026-10-28): picked on 2018-2022
    (depth 3, 300 trees, learning rate 0.03), then on 2023-2026 with 5 seeds
    top-3 log loss -0.095 [-0.175, -0.021] vs the v3 ranker's settings.
    """

    DEFAULT_PARAMS = dict(XGBRaceRanker.DEFAULT_PARAMS, max_depth=3, n_estimators=300, learning_rate=0.03)
    DEFAULT_SEEDS = 5

    def __init__(self, params: Optional[dict] = None, n_seeds: Optional[int] = None) -> None:
        super().__init__(params=params or self.DEFAULT_PARAMS, n_seeds=n_seeds or self.DEFAULT_SEEDS)
        self.finish_model = None

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        qid: Optional[np.ndarray] = None,
        dnf: Optional[np.ndarray] = None,
    ) -> "TwoStageRaceModel":
        """dnf: 1 for retired cars. The ranker sees finishers only; the finish model sees all."""
        from src.models.two_stage import FinishModel

        if dnf is None:
            raise ValueError("TwoStageRaceModel.fit needs dnf")
        dnf = np.asarray(dnf, dtype=float)
        fin = dnf == 0
        X = X.reset_index(drop=True)
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        super().fit(X[fin].reset_index(drop=True), np.asarray(y)[fin],
                    sample_weight=None if w is None else w[fin], qid=np.asarray(qid)[fin])
        self.finish_model = FinishModel().fit(X.assign(dnf=dnf), sample_weight=w)
        return self

    def predict_finish(self, X: pd.DataFrame) -> np.ndarray:
        return self.finish_model.predict(X)







