"""
Stacking ensemble: combines XGBoost + NN predictions via a logistic meta-learner.
Uses out-of-fold predictions for stacking features to prevent leakage.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class EnsemblePredictor:
    """
    Two-layer stacking ensemble.
    Layer 1: XGBoost + NN (base learners)
    Layer 2: Logistic Regression meta-learner

    For each of P1, P2, P3:
      - Stack predicted probabilities from both base learners
      - Meta-learner combines them
    """

    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self._meta_models: dict[str, LogisticRegression] = {}
        self._scalers: dict[str, StandardScaler] = {}
        self._xgb_model = None
        self._nn_model = None

    def _temporal_group_folds(self, groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Build strict forward-in-time folds from group labels.

        For each validation group g, training uses only groups < g.
        The earliest group is skipped because it has no past data.
        """
        unique_groups = sorted(pd.Series(groups).dropna().unique().tolist())
        folds: list[tuple[np.ndarray, np.ndarray]] = []

        for val_group in unique_groups[1:]:
            train_idx = np.where(groups < val_group)[0]
            val_idx = np.where(groups == val_group)[0]
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            folds.append((train_idx, val_idx))

        if len(folds) > self.cv_folds:
            folds = folds[-self.cv_folds :]

        return folds

    def fit(
        self,
        X_num: np.ndarray,
        driver_idx: np.ndarray,
        team_idx: np.ndarray,
        circuit_idx: np.ndarray,
        y_dict: dict[str, np.ndarray],  # {"p1": arr, "p2": arr, "p3": arr}
        feature_names: Optional[list[str]] = None,
        sample_weight: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> "EnsemblePredictor":
        """
        Fit the full stacking ensemble via out-of-fold training.

        This method:
        1. Generates OOF predictions from XGB and NN
        2. Trains the meta-learner on OOF predictions
        3. Retrains XGB and NN on full data (stored for inference)
        """
        from src.models.xgb_model import XGBPodiumPredictor
        from src.models.nn_model import NNPodiumPredictor
        from src.features.build_features import FEATURE_COLS

        N = len(X_num)

        sample_weight_arr: Optional[np.ndarray] = None
        if sample_weight is not None:
            sample_weight_arr = np.asarray(sample_weight, dtype=float)
            if len(sample_weight_arr) != N:
                raise ValueError(
                    f"sample_weight length mismatch: got {len(sample_weight_arr)} expected {N}"
                )

        group_arr: Optional[np.ndarray] = None
        if groups is not None:
            group_arr = np.asarray(groups)
            if len(group_arr) != N:
                raise ValueError(f"groups length mismatch: got {len(group_arr)} expected {N}")

        # Storage for OOF predictions: (N,) per target per model
        oof_xgb = {t: np.zeros(N) for t in ["p1", "p2", "p3"]}
        oof_nn = {t: np.zeros(N) for t in ["p1", "p2", "p3"]}
        oof_mask = np.zeros(N, dtype=bool)

        feature_names = feature_names or FEATURE_COLS
        X_df = pd.DataFrame(X_num, columns=feature_names[:X_num.shape[1]])

        folds: list[tuple[np.ndarray, np.ndarray]]
        if group_arr is not None and len(np.unique(group_arr)) >= 2:
            folds = self._temporal_group_folds(group_arr)
            if folds:
                logger.info(
                    "Stacking CV: using strict temporal group folds "
                    f"({len(folds)} folds, groups={sorted(np.unique(group_arr).tolist())})"
                )
            else:
                logger.warning("Could not build temporal folds; falling back to StratifiedKFold")
                splitter = StratifiedKFold(
                    n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
                )
                folds = list(splitter.split(X_num, y_dict["p1"]))
        else:
            splitter = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
            folds = list(splitter.split(X_num, y_dict["p1"]))

        n_drivers = int(driver_idx.max()) + 2
        n_teams = int(team_idx.max()) + 2
        n_circuits = int(circuit_idx.max()) + 2

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            logger.info(f"Stacking fold {fold_idx + 1}/{len(folds)}")

            X_tr, X_vl = X_df.iloc[train_idx], X_df.iloc[val_idx]
            d_tr, d_vl = driver_idx[train_idx], driver_idx[val_idx]
            t_tr, t_vl = team_idx[train_idx], team_idx[val_idx]
            c_tr, c_vl = circuit_idx[train_idx], circuit_idx[val_idx]

            y_tr = {k: v[train_idx] for k, v in y_dict.items()}
            train_weight = sample_weight_arr[train_idx] if sample_weight_arr is not None else None

            # ---- XGBoost fold ----
            xgb_fold = XGBPodiumPredictor()
            xgb_fold.fit(X_tr, y_tr, sample_weight=train_weight)
            xgb_preds = xgb_fold.predict_proba_all(X_vl)
            for t in ["p1", "p2", "p3"]:
                oof_xgb[t][val_idx] = xgb_preds[t]

            # ---- NN fold ----
            y_tr_stack = np.column_stack([y_tr["p1"], y_tr["p2"], y_tr["p3"]])
            y_vl_stack = np.column_stack(
                [y_dict["p1"][val_idx], y_dict["p2"][val_idx], y_dict["p3"][val_idx]]
            )

            nn_fold = NNPodiumPredictor(
                num_numeric_features=X_num.shape[1],
                n_drivers=n_drivers,
                n_teams=n_teams,
                n_circuits=n_circuits,
            )
            nn_fold.fit(
                X_num[train_idx],
                d_tr,
                t_tr,
                c_tr,
                y_tr_stack,
                X_val_num=X_num[val_idx],
                driver_val=d_vl,
                team_val=t_vl,
                circuit_val=c_vl,
                y_val=y_vl_stack,
                sample_weight=train_weight,
            )
            nn_preds = nn_fold.predict_proba(X_num[val_idx], d_vl, t_vl, c_vl)
            for i, t in enumerate(["p1", "p2", "p3"]):
                oof_nn[t][val_idx] = nn_preds[:, i]

            oof_mask[val_idx] = True

        if not np.any(oof_mask):
            raise RuntimeError("No OOF validation rows were generated for meta-learner training")

        covered = int(oof_mask.sum())
        logger.info(f"Meta-learner training rows from OOF folds: {covered}/{N}")

        # ---- Train meta-learners on OOF predictions ----
        for target in ["p1", "p2", "p3"]:
            meta_X = np.column_stack([oof_xgb[target], oof_nn[target]])[oof_mask]
            meta_y = y_dict[target][oof_mask]
            meta_w = sample_weight_arr[oof_mask] if sample_weight_arr is not None else None

            scaler = StandardScaler()
            meta_X_scaled = scaler.fit_transform(meta_X)

            # C=0.01: heavy L2 regularization prevents over-confident weights
            # from noisy OOF folds.
            meta = LogisticRegression(C=0.01, random_state=self.random_state, max_iter=500)
            if meta_w is not None:
                meta.fit(meta_X_scaled, meta_y, sample_weight=meta_w)
            else:
                meta.fit(meta_X_scaled, meta_y)

            # Safety clamp: keep NN coefficient within 2x XGB coefficient.
            xgb_w = abs(meta.coef_[0][0])
            nn_w = abs(meta.coef_[0][1])
            if xgb_w > 0 and nn_w > 2.0 * xgb_w:
                meta.coef_[0][1] *= (2.0 * xgb_w / nn_w)
                logger.warning(
                    f"Meta-learner {target}: clamped NN weight "
                    f"{nn_w:.4f} -> {meta.coef_[0][1]:.4f} (ratio was {nn_w / xgb_w:.1f}x)"
                )

            self._meta_models[target] = meta
            self._scalers[target] = scaler
            logger.info(
                f"Meta-learner {target}: XGB={meta.coef_[0][0]:.4f} "
                f"NN={meta.coef_[0][1]:.4f}"
            )

        # ---- Retrain base models on full data ----
        logger.info("Retraining XGB on full dataset...")
        self._xgb_model = XGBPodiumPredictor()
        self._xgb_model.fit(X_df, y_dict, sample_weight=sample_weight_arr)

        logger.info("Retraining NN on full dataset...")
        y_full = np.column_stack([y_dict["p1"], y_dict["p2"], y_dict["p3"]])
        self._nn_model = NNPodiumPredictor(
            num_numeric_features=X_num.shape[1],
            n_drivers=n_drivers,
            n_teams=n_teams,
            n_circuits=n_circuits,
        )
        self._nn_model.fit(
            X_num,
            driver_idx,
            team_idx,
            circuit_idx,
            y_full,
            sample_weight=sample_weight_arr,
        )

        return self

    def predict_proba(
        self,
        X_num: np.ndarray,
        driver_idx: np.ndarray,
        team_idx: np.ndarray,
        circuit_idx: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> dict[str, np.ndarray]:
        """
        Returns dict of probabilities: {"p1": arr, "p2": arr, "p3": arr}
        """
        from src.features.build_features import FEATURE_COLS

        feature_names = feature_names or FEATURE_COLS
        X_df = pd.DataFrame(X_num, columns=feature_names[:X_num.shape[1]])

        xgb_preds = self._xgb_model.predict_proba_all(X_df)
        nn_preds = self._nn_model.predict_proba(X_num, driver_idx, team_idx, circuit_idx)

        result = {}
        for i, target in enumerate(["p1", "p2", "p3"]):
            meta_X = np.column_stack([xgb_preds[target], nn_preds[:, i]])
            meta_X_scaled = self._scalers[target].transform(meta_X)
            result[target] = self._meta_models[target].predict_proba(meta_X_scaled)[:, 1]

        return result

    def save(self, name: str = "ensemble") -> None:
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved ensemble -> {path}")

    @classmethod
    def load(cls, name: str = "ensemble") -> "EnsemblePredictor":
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Loaded ensemble <- {path}")
        return obj
