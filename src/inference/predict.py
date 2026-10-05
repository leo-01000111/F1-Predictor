"""
Inference engine: assembles features for an upcoming race and runs the model.
Called by race_weekend_inference.py after qualifying concludes.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.models.model_registry import load_active_model_metadata

ROOT = Path(__file__).parent.parent.parent
PREDICTIONS_DIR = ROOT / "data" / "predictions"
RESULTS_DIR = ROOT / "data" / "results"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)



def run_inference(
    year: int,
    round_num: int,
    quali_df: pd.DataFrame,
    race_df_history: pd.DataFrame,
    weather_dict: dict,
    practice_df: Optional[pd.DataFrame] = None,
    pit_df: Optional[pd.DataFrame] = None,
    sprint_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Build features and generate race predictions.

    Returns a prediction document and writes data/predictions/{year}_R{round}.json.
    """
    from src.features.build_features import FEATURE_COLS, load_feature_matrix
    from src.features.driver_features import (
        add_circuit_history,
        add_championship_standing,
        add_dnf_rate,
        add_driver_form,
        add_teammate_quali_delta,
        add_driver_start_delta,
        add_elo_ratings,
        add_championship_pressure,
        add_driver_experience,
    )
    from src.features.circuit_features import add_grid_conversion
    from src.features.practice_features import merge_practice_features
    from src.features.team_features import (
        add_constructor_standing,
        add_pit_stop_performance,
        add_team_reliability,
        add_car_pace_delta,
        add_team_rolling_form,
    )
    from src.features.build_features import (
        STREET_CIRCUITS,
        CIRCUIT_SC_RATE,
        CIRCUIT_TYPICAL_COMPOUND,
        COMPOUND_MAP,
        _DEFAULT_SC_RATE,
        _DEFAULT_COMPOUND,
        _DEFAULT_TYPICAL_COMPOUND,
    )
    from src.features.weather_features import merge_weather
    from src.models.plackett_luce import PlackettLuceSampler

    logger.info(f"Running inference for {year} R{round_num}")

    circuit_key = quali_df["circuit_key"].iloc[0] if "circuit_key" in quali_df.columns else "unknown"
    race_date = quali_df["race_date"].iloc[0] if "race_date" in quali_df.columns else ""
    drivers = quali_df["driver_abbr"].tolist()
    teams = quali_df["team"].tolist()

    # 1. Build synthetic upcoming race rows
    is_wet = int(bool(weather_dict.get("is_wet_race", 0)))
    upcoming_rows = []
    for _, row in quali_df.iterrows():
        qp = row.get("quali_position", None)
        upcoming_rows.append(
            {
                "year": year,
                "round": round_num,
                "circuit": row.get("circuit", circuit_key),
                "circuit_key": circuit_key,
                "race_date": race_date,
                "driver": row.get("driver_abbr", row.get("driver", "")),
                "team": row.get("team", "Unknown"),
                "grid_position": qp,
                "quali_position_from_q": qp,
                "finish_position": None,
                "dnf": 0,
                "status": "",
                "points": 0.0,
                "p1": 0,
                "p2": 0,
                "p3": 0,
                "podium": 0,
                "quali_gap_to_pole_s": row.get("quali_gap_to_pole_s", None),
                "start_compound": row.get("start_compound", None),
                "is_wet_race": is_wet,
            }
        )

    upcoming_df = pd.DataFrame(upcoming_rows)

    combined = pd.concat([race_df_history, upcoming_df], ignore_index=True)
    combined = combined.sort_values(["year", "round"])

    combined = add_driver_form(combined)
    combined = add_championship_standing(combined)
    combined = add_dnf_rate(combined)
    combined = add_circuit_history(combined)
    combined = add_driver_start_delta(combined)
    combined = add_elo_ratings(combined)
    combined = add_championship_pressure(combined)
    combined = add_driver_experience(combined)
    combined = add_grid_conversion(combined)
    combined = add_constructor_standing(combined)
    combined = add_team_reliability(combined)
    combined = add_team_rolling_form(combined)
    combined = add_pit_stop_performance(combined, pit_df=pit_df)

    feat_df = combined[(combined["year"] == year) & (combined["round"] == round_num)].copy()

    # Teammate delta
    quali_temp = feat_df[["year", "round", "driver", "team", "quali_gap_to_pole_s"]].copy()
    quali_with_delta = add_teammate_quali_delta(quali_temp)
    feat_df = feat_df.merge(
        quali_with_delta[["year", "round", "driver", "teammate_quali_delta"]],
        on=["year", "round", "driver"],
        how="left",
    )

    # Car pace delta (constructor-level qualifying pace)
    feat_df = add_car_pace_delta(feat_df)

    # 1b. Grid penalty (quali_position_from_q vs grid_position)
    if "quali_position_from_q" in feat_df.columns:
        qp = pd.to_numeric(feat_df["quali_position_from_q"], errors="coerce")
        gp = pd.to_numeric(feat_df["grid_position"], errors="coerce")
        feat_df["grid_penalty_places"] = (gp - qp).clip(lower=0).fillna(0.0)
    else:
        feat_df["grid_penalty_places"] = 0.0

    # 1c. Sprint result
    if sprint_df is not None and not sprint_df.empty:
        sprint_sub = sprint_df[
            (sprint_df["year"] == year) & (sprint_df["round"] == round_num)
        ][["year", "round", "driver", "sprint_position", "n_sprint_starters"]].copy()
        if not sprint_sub.empty:
            feat_df = feat_df.merge(sprint_sub, on=["year", "round", "driver"], how="left")
            sp = pd.to_numeric(feat_df.get("sprint_position"), errors="coerce")
            ns = pd.to_numeric(feat_df.get("n_sprint_starters"), errors="coerce").fillna(20)
            feat_df["sprint_position_rel"] = (sp / ns).fillna(0.5)
            feat_df = feat_df.drop(columns=["sprint_position", "n_sprint_starters"], errors="ignore")
        else:
            feat_df["sprint_position_rel"] = 0.5
    else:
        feat_df["sprint_position_rel"] = 0.5

    # 2. Weather
    weather_row = pd.DataFrame([{"year": year, "round": round_num, **weather_dict}])
    feat_df = merge_weather(feat_df, weather_row)

    # 2b. Wet-race driver performance
    # raw race_df_history doesn't have is_wet_race, so we load the pre-computed
    # feature matrix and use the latest driver_wet_avg_finish value per driver.
    try:
        hist_fm = load_feature_matrix()
        drv_wet = (
            hist_fm.sort_values(["driver", "year", "round"])
            .groupby("driver")["driver_wet_avg_finish"]
            .last()
        )
        feat_df["driver_wet_avg_finish"] = feat_df["driver"].map(drv_wet)
    except Exception:
        feat_df["driver_wet_avg_finish"] = np.nan

    # 3. Practice (sprint_df passed so sprint weekends get real pace data instead of median)
    feat_df = merge_practice_features(feat_df, practice_df, sprint_df=sprint_df)

    # 3b. Derived quali and season features
    feat_df["grid_position"] = pd.to_numeric(feat_df.get("grid_position"), errors="coerce")
    feat_df["quali_gap_to_pole_s"] = pd.to_numeric(feat_df.get("quali_gap_to_pole_s"), errors="coerce")
    feat_df["quali_gap_to_pole_s"] = feat_df["quali_gap_to_pole_s"].clip(upper=5.0)

    max_gap = feat_df["quali_gap_to_pole_s"].max()
    if pd.notna(max_gap) and max_gap >= 0.01:
        feat_df["quali_gap_relative"] = feat_df["quali_gap_to_pole_s"] / max_gap
    else:
        feat_df["quali_gap_relative"] = 0.0

    max_grid = feat_df["grid_position"].max()
    if pd.notna(max_grid) and max_grid > 1:
        feat_df["grid_position_rel"] = (feat_df["grid_position"] - 1) / (max_grid - 1)
    else:
        feat_df["grid_position_rel"] = 0.0

    known_max = race_df_history[race_df_history["year"] == year]["round"].max() if year in race_df_history["year"].values else None

    season_max_candidates: list[int] = [int(round_num)]
    if pd.notna(known_max):
        season_max_candidates.append(int(known_max))

    # Prefer the official season length when FastF1 schedule is available.
    try:
        import fastf1

        schedule = fastf1.get_event_schedule(year, include_testing=False)
        if "RoundNumber" in schedule.columns:
            sched_max = pd.to_numeric(schedule["RoundNumber"], errors="coerce").max()
            if pd.notna(sched_max):
                season_max_candidates.append(int(sched_max))
    except Exception:
        pass

    if race_df_history is not None and not race_df_history.empty:
        hist_max = pd.to_numeric(race_df_history["round"], errors="coerce").max()
        if pd.notna(hist_max):
            season_max_candidates.append(int(hist_max))

    season_max = max(season_max_candidates)
    feat_df["round_fraction"] = (round_num - 1) / max(season_max - 1, 1)

    # 3c. Circuit type and safety car rate
    ck = str(circuit_key).lower()
    feat_df["is_street_circuit"] = float(ck in STREET_CIRCUITS)
    feat_df["circuit_safety_car_rate"] = float(CIRCUIT_SC_RATE.get(ck, _DEFAULT_SC_RATE))

    # 3d. Tire strategy features
    # start_compound_code: use qualifying info if available, otherwise Medium (1)
    if "start_compound" in feat_df.columns:
        feat_df["start_compound_code"] = (
            feat_df["start_compound"].str.upper().map(COMPOUND_MAP).fillna(_DEFAULT_COMPOUND)
        )
    else:
        feat_df["start_compound_code"] = float(_DEFAULT_COMPOUND)
        logger.info("start_compound not available; defaulting to Medium (1) for all drivers")

    typical_compound = float(CIRCUIT_TYPICAL_COMPOUND.get(ck, _DEFAULT_TYPICAL_COMPOUND))
    feat_df["compound_aggression_delta"] = feat_df["start_compound_code"] - typical_compound

    # 4. Ensure features and impute
    for col in FEATURE_COLS:
        if col not in feat_df.columns:
            feat_df[col] = 0.0
        if feat_df[col].isna().any():
            med = feat_df[col].median()
            feat_df[col] = feat_df[col].fillna(med if not feat_df[col].isna().all() else 0.0)

    # Categorical encoding with explicit unknown=0 index
    try:
        hist_fm = load_feature_matrix()
        for col in ["driver", "team", "circuit_key"]:
            cat = pd.Categorical(hist_fm[col])
            raw_codes = pd.Categorical(feat_df[col], categories=cat.categories).codes
            feat_df[f"{col}_idx"] = np.where(raw_codes < 0, 0, raw_codes + 1)
    except Exception:
        for col in ["driver", "team", "circuit_key"]:
            feat_df[f"{col}_idx"] = 0

    X = feat_df[FEATURE_COLS].values
    driver_idx = feat_df["driver_idx"].values.astype(int)
    team_idx = feat_df["team_idx"].values.astype(int)
    circuit_idx = feat_df["circuit_key_idx"].values.astype(int)

    # 5. Model inference (v2: single regressor + Plackett-Luce)
    from src.models.xgb_model import XGBRacePredictor

    model = XGBRacePredictor.load("xgb_race")
    pred_positions = model.predict(pd.DataFrame(X, columns=FEATURE_COLS))
    logger.info(
        f"Predicted positions: {dict(zip(feat_df['driver'].tolist(), pred_positions.round(2).tolist()))}"
    )

    # 6. Plackett-Luce simulation → coherent position probabilities
    sampler = PlackettLuceSampler(temperature=1.0, n_simulations=10_000)
    prob_matrix = sampler.simulate(pred_positions)  # (n_drivers, n_drivers)
    n_pos = prob_matrix.shape[1]
    probs: dict[str, np.ndarray] = {
        "p1": prob_matrix[:, 0],
        "p2": prob_matrix[:, 1],
        "p3": prob_matrix[:, 2],
        "p_top6": prob_matrix[:, :min(6, n_pos)].sum(axis=1),
        "p_top10": prob_matrix[:, :min(10, n_pos)].sum(axis=1),
    }

    # 7. Format output
    predictions = []
    for i, (_, row) in enumerate(feat_df.iterrows()):
        predictions.append(
            {
                "driver": row["driver"],
                "team": row["team"],
                "grid_position": int(row["grid_position"]) if pd.notna(row.get("grid_position")) else None,
                "quali_gap_to_pole_s": float(row.get("quali_gap_to_pole_s", 0) or 0),
                "fp_best_gap_s": float(round(float(row.get("fp_best_gap_s", 0) or 0), 4)),
                "fp2_best_gap_s": float(round(float(row.get("fp2_best_gap_s", 0) or 0), 4)),
                "fp_long_run_delta": float(round(float(row.get("fp_long_run_delta", 0) or 0), 4)),
                "fp_total_laps": float(round(float(row.get("fp_total_laps", 0) or 0), 4)),
                "p_win": float(round(probs["p1"][i], 4)),
                "p_p2": float(round(probs["p2"][i], 4)),
                "p_p3": float(round(probs["p3"][i], 4)),
                "p_podium": float(round(min(probs["p1"][i] + probs["p2"][i] + probs["p3"][i], 1.0), 4)),
                "p_top6": float(round(min(probs["p_top6"][i], 1.0), 4)),
                "p_top10": float(round(min(probs["p_top10"][i], 1.0), 4)),
            }
        )

    predictions = sorted(predictions, key=lambda x: x["p_win"], reverse=True)

    active_model = load_active_model_metadata() or {}
    result = {
        "year": year,
        "round": round_num,
        "circuit_key": circuit_key,
        "race_date": race_date,
        "generated_at": datetime.utcnow().isoformat(),
        "weather": weather_dict,
        "drivers": drivers,
        "teams": teams,
        "model": {
            "family": "xgb_v2",
            "artifact": "xgb_race.pkl",
            "sampler": f"plackett_luce(T={sampler.temperature}, n_sim={sampler.n_simulations})",
            "active_manifest": active_model,
        },
        "predictions": predictions,
    }

    path = PREDICTIONS_DIR / f"{year}_R{round_num:02d}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Predictions saved -> {path}")
    logger.info(f"{'Driver':<12} {'P(Win)':>8} {'P(P2)':>8} {'P(P3)':>8} {'P(Podium)':>10}")
    for p in predictions[:5]:
        logger.info(
            f"{p['driver']:<12} {p['p_win']:>8.1%} {p['p_p2']:>8.1%} {p['p_p3']:>8.1%} {p['p_podium']:>10.1%}"
        )

    return result



