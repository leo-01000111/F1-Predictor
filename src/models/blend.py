"""
V6: blend of the two-stage pace ranker (v5) and a linear Plackett-Luce model.

Log-strength of driver i in a race (higher = faster):
    s_i = w_pace * score_i(TwoStageRaceModel ranker) + w_glm * x_i . beta
predict() returns -s / temperature (temperature stays 1: the weights carry the
scale), so every caller keeps using two_stage.simulate(pred, q, 1.0).
P(finish) comes from the two-stage model's finish model unchanged.

Linear model: Plackett-Luce on standardised inputs (FEATURE_COLS subset, log grid,
the driver/car strength ratings: callers pass model_inputs(model, df)),
fitted by exact top-3 likelihood. A different model family from the tree
ranker; it leans on grid and quali gap where the ranker leans on reputation.

Weights: fitted on out-of-sample predictions of earlier seasons (walk-forward:
each season predicted by both components trained on the seasons before it),
by the production two-stage top-3 likelihood (uses P(finish)).

Backtest (scripts/blend_backtest.py --two-stage-fit, 2019-2026, 171 races) vs
v5 under the same protocol: top-3 log loss +0.002 [-0.052, +0.058] (same),
winner hit 53.2% -> 59.6% (+6.4 pp [+2.3, +11.1]). NOT shipped (2026-11-08): by era
the top-3 log loss is better in 2019-22 but worse in 2023-26 (+0.054), and the
pass-1 holdout 2025-26 had it below v5. Kept as an experiment; production is v5.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize

GLM_FEATURES = ["log_grid", "quali_gap_relative", "driver_elo", "driver_form_ewm", "team_form_5",
                "car_pace_delta_pct", "driver_rating", "car_rating", "teammate_quali_delta"]
# Inputs beyond FEATURE_COLS (strength_ratings.py, computed in build_features and predict.py)
EXTRA_INPUTS = ["driver_rating", "car_rating"]
GLM_L2 = 1e-3
WEIGHT_FIT_FIRST_SEASON = 2018   # first season whose OOS predictions fit the blend weights


def _grid_slot(grid: pd.Series) -> np.ndarray:
    """Grid slot; pit-lane / unknown starts go behind the last car (as backtest._grid_score)."""
    g = pd.to_numeric(grid, errors="coerce").to_numpy(dtype=float)
    last = np.nanmax(np.where(g > 0, g, np.nan)) + 1 if np.any(g > 0) else len(g)
    return np.where(np.isfinite(g) & (g > 0), g, last)


def race_tops(df: pd.DataFrame) -> list[tuple[np.ndarray, list[int]]]:
    """(row positions, indices of P1-P3 in finishing order) per race with a full podium."""
    out = []
    fin = pd.to_numeric(df["finish_position"], errors="coerce").to_numpy()
    for _, idx in df.groupby(["year", "round"]).indices.items():
        f = fin[idx]
        top = [int(np.where(f == p)[0][0]) for p in (1, 2, 3) if (f == p).any()]
        if len(top) == 3:
            out.append((idx, top))
    return out


def _pl_nll_grad(w: np.ndarray, races: list, l2: float) -> tuple[float, np.ndarray]:
    nll, grad = 0.0, np.zeros_like(w)
    for X, top in races:
        s = X @ w
        alive = np.ones(len(s), bool)
        for i in top:
            sa = s[alive]
            m = sa.max()
            e = np.exp(sa - m)
            z = e.sum()
            nll -= s[i] - m - np.log(z)
            grad -= X[i] - (e / z) @ X[alive]
            alive[i] = False
    n = max(len(races), 1)
    return nll / n + l2 * (w @ w), grad / n + 2 * l2 * w


class LinearPLModel:
    """Linear Plackett-Luce ranker on standardised features (exact top-3 likelihood)."""

    def __init__(self, features: Optional[list[str]] = None, l2: float = GLM_L2) -> None:
        self.features = list(features or GLM_FEATURES)
        self.l2 = l2
        self.mu: Optional[pd.Series] = None
        self.sd: Optional[pd.Series] = None
        self.coef: Optional[np.ndarray] = None

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        M = X.reindex(columns=[f for f in self.features if f != "log_grid"]).apply(pd.to_numeric, errors="coerce")
        if "log_grid" in self.features:
            M.insert(0, "log_grid", np.log(_grid_slot(X["grid_position"])))
        M = M[self.features]
        if self.mu is None:
            self.mu, self.sd = M.mean(), M.std().replace(0, 1).fillna(1)
        return ((M - self.mu) / self.sd).fillna(0.0).to_numpy(float)

    def fit(self, df: pd.DataFrame) -> "LinearPLModel":
        """df: training rows with the features, grid_position, year, round, finish_position."""
        df = df.reset_index(drop=True)
        self.mu = self.sd = None
        X = self._matrix(df)
        races = [(X[idx], top) for idx, top in race_tops(df)]
        r = minimize(_pl_nll_grad, np.zeros(X.shape[1]), args=(races, self.l2), jac=True, method="L-BFGS-B")
        self.coef = r.x
        logger.info("LinearPLModel: " + ", ".join(f"{f}={c:+.3f}" for f, c in zip(self.features, self.coef)))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Log-strength (higher = faster) per row."""
        return self._matrix(X) @ self.coef


class BlendedRaceModel:
    """v6 production model: w_pace * ranker score + w_glm * linear PL score."""

    def __init__(self, pace, glm: LinearPLModel, weights: tuple[float, float],
                 pace_cols: list[str]) -> None:
        self.pace = pace                  # TwoStageRaceModel (temperature unused)
        self.pace_cols = list(pace_cols)  # FEATURE_COLS at training time
        self.input_cols = self.pace_cols + [c for c in EXTRA_INPUTS if c not in self.pace_cols]
        self.glm = glm
        self.weights = (float(weights[0]), float(weights[1]))
        self.temperature = 1.0            # the weights carry the Plackett-Luce scale

    # Attributes other code reads from the production pickle
    @property
    def _model(self):                     # first seed booster (SHAP / feature importance)
        return self.pace._model

    @property
    def params(self) -> dict:
        return self.pace.params

    @property
    def n_seeds(self) -> int:
        return self.pace.n_seeds

    def log_strength(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reset_index(drop=True)
        return (self.weights[0] * self.pace._raw_score(X[self.pace_cols])
                + self.weights[1] * self.glm.score(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return -self.log_strength(X) / self.temperature

    def predict_finish(self, X: pd.DataFrame) -> np.ndarray:
        return self.pace.predict_finish(X.reset_index(drop=True)[self.pace_cols])


def model_inputs(model, df: pd.DataFrame) -> pd.DataFrame:
    """The columns `model` predicts from (FEATURE_COLS for v5, + ratings for the blend)."""
    from src.features.build_features import FEATURE_COLS

    cols = list(getattr(model, "input_cols", FEATURE_COLS))
    return pd.DataFrame(df[cols].values, columns=cols).apply(pd.to_numeric, errors="coerce")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def _fit_components(train: pd.DataFrame, cols: list[str]):
    from src.models.xgb_model import TwoStageRaceModel
    from src.training.weights import compute_recency_sample_weight

    train = train.reset_index(drop=True)
    pace = TwoStageRaceModel()
    pace.fit(pd.DataFrame(train[cols].values, columns=cols), train["target_position"].values,
             sample_weight=compute_recency_sample_weight(train["year"].values),
             qid=(train["year"] * 100 + train["round"]).to_numpy(),
             dnf=pd.to_numeric(train["dnf"], errors="coerce").fillna(0).to_numpy())
    glm = LinearPLModel().fit(train)
    return pace, glm


def _per_race_matrix(model_pace, glm: LinearPLModel, test: pd.DataFrame, cols: list[str]):
    """Component log-strengths [n, 2] and P(finish) for test rows (any number of races)."""
    test = test.reset_index(drop=True)
    X = pd.DataFrame(test[cols].values, columns=cols)
    pace = model_pace._raw_score(X)
    return np.column_stack([pace, glm.score(test)]), np.asarray(model_pace.predict_finish(X), float)


def fit_weights(oos: list[tuple[pd.DataFrame, np.ndarray, np.ndarray]], n_samples: int = 300) -> tuple[float, float]:
    """Blend weights minimising the two-stage top-3 log loss over OOS seasons
    [(rows, components [n, 2], P(finish))]. Start: exact-PL fit (no DNFs)."""
    from src.models.two_stage import top_k_nll_logw

    data = [(C[idx], q[idx], top) for df, C, q in oos for idx, top in race_tops(df.reset_index(drop=True))]
    exact = [(C, top) for C, _, top in data]
    w0 = minimize(_pl_nll_grad, np.array([0.5, 0.5]), args=(exact, 0.0), jac=True, method="L-BFGS-B").x

    def loss(w):
        return float(np.mean([top_k_nll_logw(C @ w, q, top, n_samples=n_samples) for C, q, top in data]))
    w = minimize(loss, w0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 200}).x
    logger.info(f"Blend weights on {len(data)} OOS races: pace={w[0]:.3f}, glm={w[1]:.3f} "
                f"(exact-PL start {w0.round(3).tolist()}, two-stage ll_top3 {loss(w):.4f})")
    return float(w[0]), float(w[1])


def fit_blend(fm: pd.DataFrame, train_years: list[int], weight_years: list[int],
              cols: list[str]) -> BlendedRaceModel:
    """Production blend: components trained on train_years; weights fitted on the
    walk-forward OOS predictions of weight_years (each season predicted by
    components trained on all seasons before it)."""
    oos = []
    for y in weight_years:
        tr = fm[fm["year"] < y]
        te = fm[fm["year"] == y].reset_index(drop=True)
        if tr.empty or te.empty:
            continue
        p, g = _fit_components(tr, cols)
        C, q = _per_race_matrix(p, g, te, cols)
        oos.append((te, C, q))
        logger.info(f"Blend OOS season {y}: {te[['year', 'round']].drop_duplicates().shape[0]} races")
    weights = fit_weights(oos)
    pace, glm = _fit_components(fm[fm["year"].isin(train_years)], cols)
    return BlendedRaceModel(pace, glm, weights, cols)
