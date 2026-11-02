"""
Promotion gate for the post-race retrain (scripts/quick_train.py, pass 2).

quick_train writes the retrained production model to models/xgb_race_candidate.pkl
(gitignored). It replaces models/xgb_race.pkl only if this gate says so; otherwise
production stays as it is and the reason is recorded.

Evaluation set: every raced round of the seasons after the last season EITHER
model trained on. Production trains on all seasons but the latest, so in-season
the candidate and the current model trained on the same seasons and the set is
the current season so far, which neither has seen. After a season rollover the
candidate has one more training season, so the set shrinks to the new season's
first round(s): fewer than MIN_EVAL_RACES is not enough to judge and the
candidate is promoted (new data must get in). The 42-race pass-1 holdout is not
used: the production candidate trains on part of it.

Temperature: each model's stored T was fitted on (most of) these races, the
candidate's on one race more. Both are re-scored the same way instead: race i
is scored at the T (backtest.T_GRID) minimising mean ll_top3 over the OTHER eval
races (leave-one-race-out), so no race is scored with a T fitted on itself. The
grid baseline (backtest.grid_predict) gets the same treatment.

Metric: per-race ll_top3 (Plackett-Luce / two-stage log loss of the ordered top
3, as scripts/backtest.py), compared with paired_bootstrap (mean difference and
95% interval). Lower is better. The candidate is refused only when it is
clearly worse, i.e. the mean difference exceeds the tolerance AND the interval
excludes 0:
  vs current : candidate - current > MAX_WORSE_VS_CURRENT and lower bound > 0
  vs grid    : candidate - grid    > MAX_WORSE_VS_GRID    and lower bound > 0
Differences within noise promote.

Promoted without scoring: no production model yet, or the production model was
trained on a different feature list than FEATURE_COLS (inference builds
FEATURE_COLS, so the old model could not be used anyway).

On a gate error production is KEPT: the current model already passed a gate (or
a manual run), and in-season the candidate differs from it only by the refitted
temperature, so keeping it for one more race costs little, while promoting an
unchecked model can publish broken probabilities. The error is logged
(ci_log `promotion` event, holdout_metrics.json); `quick_train.py --no-gate`
promotes by hand.
"""
from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"
PRODUCTION_NAME = "xgb_race"
PRODUCTION_PATH = MODELS_DIR / f"{PRODUCTION_NAME}.pkl"
CANDIDATE_PATH = MODELS_DIR / f"{PRODUCTION_NAME}_candidate.pkl"

# Mean ll_top3 per race (nats). Typical per-race spread is ~1-2 nats, the model
# beats the grid by ~0.5 in the walk-forward backtest.
MAX_WORSE_VS_CURRENT = 0.05   # tolerated mean degradation vs the current model
MAX_WORSE_VS_GRID = 0.0       # the candidate must not be clearly worse than the grid
MIN_EVAL_RACES = 5            # fewer races: within noise by definition, promote
N_SAMPLES_FIT = 500           # top_k_nll samples per T on the grid (as fit_temperature)
N_SAMPLES_SCORE = 2000        # ... for the final score (as score_races)

THRESHOLDS = {"max_worse_vs_current": MAX_WORSE_VS_CURRENT, "max_worse_vs_grid": MAX_WORSE_VS_GRID,
              "min_eval_races": MIN_EVAL_RACES}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_candidate(model: Any) -> Path:
    """Write the retrained model next to production (no snapshot: it may never be used)."""
    with open(CANDIDATE_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Candidate model saved -> {CANDIDATE_PATH}")
    return CANDIDATE_PATH


def promote_candidate() -> None:
    """Replace the production pickle with the candidate (atomic) and snapshot it."""
    os.replace(CANDIDATE_PATH, PRODUCTION_PATH)
    logger.info(f"Candidate promoted -> {PRODUCTION_PATH}")
    try:
        from src.models.model_registry import snapshot_model
        snap = snapshot_model(PRODUCTION_NAME, keep_last=5)
        if snap:
            logger.info(f"Snapshot -> {snap.name}")
    except Exception as exc:
        logger.warning(f"Snapshot failed (non-fatal): {exc}")


def _loss_matrix(pred: np.ndarray, q: Optional[np.ndarray], races: list) -> np.ndarray:
    """ll_top3 per [T in T_GRID, race] for predictions at T = 1."""
    from src.training.backtest import T_GRID, _nll

    return np.array([[_nll(pred[idx], None if q is None else q[idx], top, t, n_samples=N_SAMPLES_FIT)
                      for _, idx, top in races] for t in T_GRID])


def loo_scores(pred: np.ndarray, q: Optional[np.ndarray], races: list) -> tuple[np.ndarray, np.ndarray]:
    """Per-race ll_top3, each race at the T fitted on the other races. Returns (scores, T per race)."""
    from src.training.backtest import T_GRID, _nll

    mat = _loss_matrix(pred, q, races)
    total = mat.sum(axis=1)
    scores, temps = [], []
    for j, (_, idx, top) in enumerate(races):
        t = float(T_GRID[int(np.argmin(total - mat[:, j]))])
        temps.append(t)
        scores.append(_nll(pred[idx], None if q is None else q[idx], top, t, n_samples=N_SAMPLES_SCORE))
    return np.array(scores), np.array(temps)


def _model_pred(model: Any, X: pd.DataFrame) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Predicted positions at T = 1 (predict() divides by the stored T) and P(finish)."""
    pred = np.asarray(model.predict(X), dtype=float) * float(getattr(model, "temperature", 1.0))
    q = model.predict_finish(X)
    return pred, (None if q is None else np.asarray(q, dtype=float))


def _decision(decision: str, reason: str, **data: Any) -> dict[str, Any]:
    out = {"decision": decision, "reason": reason, "checked_at": now_utc(), "thresholds": THRESHOLDS}
    out.update(data)
    log = logger.info if decision == "promote" else logger.warning
    log(f"Promotion gate: {decision.upper()} ({reason})")
    return out


def evaluate(candidate: Any, fm: pd.DataFrame, candidate_train_years: list[int],
             current_meta: Optional[dict] = None) -> dict[str, Any]:
    """Score candidate vs current production vs grid and decide. May raise (see run_gate)."""
    from src.features.build_features import FEATURE_COLS
    from src.training.backtest import _race_orders, grid_predict, paired_bootstrap

    if not PRODUCTION_PATH.exists():
        return _decision("promote", "no production model yet")
    current_meta = current_meta or {}
    cur_cols = current_meta.get("feature_cols")
    if cur_cols is not None and list(cur_cols) != list(FEATURE_COLS):
        return _decision("promote", "production model was trained on a different feature list")

    cur_train = current_meta.get("train_years") or candidate_train_years
    last_train = max(max(candidate_train_years), max(int(y) for y in cur_train))
    eval_df = fm[fm["year"] > last_train].reset_index(drop=True)
    races = _race_orders(eval_df)
    eval_years = sorted(int(y) for y in eval_df["year"].unique())
    base = {"eval_years": eval_years, "n_races": len(races)}
    if len(races) < MIN_EVAL_RACES:
        return _decision("promote", f"only {len(races)} unseen races (< {MIN_EVAL_RACES}), within noise", **base)

    X = pd.DataFrame(eval_df[FEATURE_COLS].values, columns=FEATURE_COLS)
    cand_ll, cand_t = loo_scores(*_model_pred(candidate, X), races)
    try:
        with open(PRODUCTION_PATH, "rb") as f:
            current = pickle.load(f)
        cur_pred = _model_pred(current, X)
    except Exception as exc:
        # The production model cannot run on the current features: inference would fail too
        return _decision("promote", f"production model cannot predict on current features: {exc}"[:300], **base)
    cur_ll, cur_t = loo_scores(*cur_pred, races)
    grid_ll, grid_t = loo_scores(grid_predict(eval_df, eval_df), None, races)

    vs_cur = paired_bootstrap(pd.Series(cand_ll), pd.Series(cur_ll))
    vs_grid = paired_bootstrap(pd.Series(cand_ll), pd.Series(grid_ll))
    metrics = {
        **base,
        "ll_top3": {"candidate": round(float(cand_ll.mean()), 4), "current": round(float(cur_ll.mean()), 4),
                    "grid": round(float(grid_ll.mean()), 4)},
        "median_temperature": {"candidate": float(np.median(cand_t)), "current": float(np.median(cur_t)),
                               "grid": float(np.median(grid_t))},
        "candidate_minus_current": [round(v, 4) for v in vs_cur],
        "candidate_minus_grid": [round(v, 4) for v in vs_grid],
    }
    logger.info(f"Promotion gate on {len(races)} races {eval_years}: ll_top3 candidate "
                f"{cand_ll.mean():.4f}, current {cur_ll.mean():.4f}, grid {grid_ll.mean():.4f}; "
                f"cand-current {vs_cur[0]:+.4f} [{vs_cur[1]:+.4f},{vs_cur[2]:+.4f}], "
                f"cand-grid {vs_grid[0]:+.4f} [{vs_grid[1]:+.4f},{vs_grid[2]:+.4f}]")

    if vs_cur[0] > MAX_WORSE_VS_CURRENT and vs_cur[1] > 0:
        return _decision("keep", f"candidate worse than current by {vs_cur[0]:+.3f} "
                         f"(> {MAX_WORSE_VS_CURRENT}, 95% CI excludes 0)", **metrics)
    if vs_grid[0] > MAX_WORSE_VS_GRID and vs_grid[1] > 0:
        return _decision("keep", f"candidate worse than the grid by {vs_grid[0]:+.3f} "
                         f"(> {MAX_WORSE_VS_GRID}, 95% CI excludes 0)", **metrics)
    return _decision("promote", f"candidate vs current {vs_cur[0]:+.3f}, vs grid {vs_grid[0]:+.3f}: "
                     "not clearly worse", **metrics)


def run_gate(candidate: Any, fm: pd.DataFrame, candidate_train_years: list[int],
             current_meta: Optional[dict] = None) -> dict[str, Any]:
    """evaluate() that never raises: on error production is kept (see module docstring)."""
    try:
        return evaluate(candidate, fm, candidate_train_years, current_meta)
    except Exception as exc:
        logger.exception("Promotion gate failed")
        return _decision("keep", "gate error, production kept", error=f"{type(exc).__name__}: {exc}"[:300])


def record(gate: dict[str, Any], model_family: str) -> None:
    """Store the gate result as "promotion" in holdout_metrics(.json/_<family>.json) and
    active_model.json (when production is kept, the manifest otherwise unchanged)."""
    from src.models.model_registry import ACTIVE_MODEL_PATH

    paths = [MODELS_DIR / "holdout_metrics.json", MODELS_DIR / f"holdout_metrics_{model_family}.json"]
    if gate.get("decision") != "promote":
        paths.append(ACTIVE_MODEL_PATH)
    for path in paths:
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            doc["promotion"] = gate
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2)
        except Exception as exc:
            logger.warning(f"Could not record the promotion gate in {path.name}: {exc}")
