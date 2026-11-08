"""
Per-driver explanations of a race prediction ("why this driver is here").

The production ranker's score is a sum of per-feature contributions (XGBoost
pred_contribs = exact TreeSHAP for trees), averaged over the seed models. Only
differences between drivers matter for the order, so each contribution is taken
relative to the field average for that feature in this race, divided by the
Plackett-Luce temperature (log-strength units: +0.69 = twice the strength), and
summed into the feature groups the site uses (FEATURE_GROUPS).

For each driver the group values sum to its log-strength minus the field
average. They explain the pace order; P(finish) comes from the separate finish
model and is not included.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_GROUPS: dict[str, list[str]] = {
    "qualifying": ["grid_position", "quali_gap_relative", "grid_position_rel", "teammate_quali_delta",
                   "grid_penalty_places", "circuit_grid_conversion"],
    "car": ["car_pace_delta_pct", "constructor_standing", "team_cum_points", "team_form_5"],
    "driver": ["driver_form_5", "driver_form_10", "driver_form_ewm", "driver_elo", "driver_experience_norm",
               "driver_wet_avg_finish"],
    "championship": ["championship_position", "cum_points_before", "championship_pressure", "round_fraction"],
    "circuit": ["circuit_avg_finish", "circuit_podiums", "circuit_appearances", "is_street_circuit",
                "circuit_safety_car_rate"],
    "reliability": ["dnf_rate", "team_reliability_rate"],
    "weather": ["temp_c_mean", "precip_mm_total", "windspeed_ms_mean", "humidity_pct_mean", "is_wet_race"],
    "sprint": ["sprint_position_rel"],
    "practice": ["fp_longrun_gap_pct", "team_longrun_gap_pct"],
}


def feature_contributions(model, X: pd.DataFrame) -> pd.DataFrame:
    """Per-row, per-feature contribution to the ranker score (higher = faster), seed-averaged."""
    import xgboost as xgb

    models = getattr(model, "_seed_models", None) or [model._model]
    cols = list(X.columns)
    dm = xgb.DMatrix(X.astype(float), feature_names=cols)
    contribs = np.mean([m.get_booster().predict(dm, pred_contribs=True) for m in models], axis=0)
    return pd.DataFrame(contribs[:, :-1], columns=cols)    # last column is the bias


def explain_race(model, X: pd.DataFrame) -> list[dict[str, float]]:
    """One dict {group: log-strength vs field average} per row of X (one race)."""
    c = feature_contributions(model, X.reset_index(drop=True))
    rel = (c - c.mean(axis=0)) / float(getattr(model, "temperature", 1.0) or 1.0)
    out = []
    for _, row in rel.iterrows():
        out.append({g: round(float(row.reindex(cols).fillna(0).sum()), 3)
                    for g, cols in FEATURE_GROUPS.items()})
    return out
