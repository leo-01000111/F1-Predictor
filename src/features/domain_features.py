"""
Domain-knowledge features: cross-row structure a tree model on per-driver rows
cannot build by itself (each row is scored alone by the ranker).

All history-based values use earlier races only (shift(1) within the entity).
Input: the merged race frame (one row per driver per race, sorted any way) with
year, round, circuit_key, driver, team, grid_position, finish_position, dnf,
driver_form_ewm, driver_elo, car_pace_delta_pct. Output: the same frame plus
DOMAIN_COLS.

Groups
  gain       driver_gain_ewm / team_gain_ewm: EWM of places gained (grid - finish)
             by classified finishers, minus the usual gain from that grid slot
             (earlier seasons). Race-craft and race-vs-quali car (tyre wear).
  teammate   teammate_finish_delta_ewm: EWM of finish minus teammate's finish
             (both classified): driver skill with the car taken out.
             teammate_quali_delta_ewm: EWM of past teammate quali deltas.
  traffic    who starts around you, by pre-race strength (Elo):
             stronger_behind_n: stronger cars in the 5 grid slots behind (threats),
             weaker_ahead_n: weaker cars in the 5 slots ahead (easy passes),
             elo_rank_gap: grid position minus Elo rank in the field (+ = starts
             behind where its strength says it belongs).
  track type circuit taxonomy (power / high-downforce / fast-flowing / traction):
             team_type_form_delta: team's mean best finish at this circuit type
             (last 6 such races) minus its mean best finish over its last 6 races.
             Negative = this kind of track suits the car.
  trend      car_pace_trend: this weekend's car_pace_delta_pct minus the team's EWM
             of earlier weekends (+ = slower than usual, - = upgrade / good fit).
  sprint     is_sprint_weekend: 1 if the weekend had a sprint (one practice session,
             parc ferme from Friday).
  regs       reg_era_progress: races since the last big regulation reset
             (2014, 2017, 2022, 2026), /44, clipped to 1. Early in an era,
             last season's form says less.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GROUPS = {
    "gain": ["driver_gain_ewm", "team_gain_ewm"],
    "teammate": ["teammate_finish_delta_ewm", "teammate_quali_delta_ewm"],
    "traffic": ["stronger_behind_n", "weaker_ahead_n", "elo_rank_gap"],
    "track_type": ["team_type_form_delta"],
    "trend": ["car_pace_trend"],
    "regs": ["reg_era_progress"],
    "sprint": ["is_sprint_weekend"],
}
DOMAIN_COLS = [c for cols in GROUPS.values() for c in cols]

CIRCUIT_TYPE = {
    # low drag, long straights, braking zones
    **dict.fromkeys(["monza", "spa", "baku", "jeddah", "las_vegas", "sochi", "montreal", "mexico_city"], "power"),
    # slow, twisty, max downforce, track position is king
    **dict.fromkeys(["monaco", "hungaroring", "singapore", "zandvoort", "madrid"], "high_downforce"),
    # fast sweeping corners, aero efficiency
    **dict.fromkeys(["silverstone", "suzuka", "losail", "cota", "istanbul", "mugello", "portimao",
                     "barcelona"], "fast_flowing"),
}
# everything else (bahrain, abu_dhabi, red_bull_ring, interlagos, shanghai, ...): "traction"
REG_RESETS = (2014, 2017, 2022, 2026)
SPAN = 8


def _ewm_prev(s: pd.Series, span: int = SPAN) -> pd.Series:
    return s.shift(1).ewm(span=span, min_periods=1).mean()


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_order"] = np.arange(len(df))
    df = df.sort_values(["year", "round", "grid_position"], kind="stable")
    race = ["year", "round"]
    grid = pd.to_numeric(df["grid_position"], errors="coerce")
    fin = pd.to_numeric(df["finish_position"], errors="coerce")
    classified = fin.notna() & (pd.to_numeric(df.get("dnf", 0), errors="coerce").fillna(0) == 0)

    # --- gain ------------------------------------------------------------- #
    # Places gained beyond what the grid slot usually yields: expected finish per
    # slot from EARLIER seasons only (backmarkers "gain" by default as cars ahead retire)
    raw_gain = (grid - fin).where(classified)
    slot = grid.clip(upper=22).round()
    df["_gain"] = np.nan
    for yr in sorted(df["year"].unique()):
        past = (df["year"] < yr).to_numpy()
        src = past if past.sum() >= 500 else (df["year"] == yr).to_numpy()
        curve = raw_gain[src].groupby(slot[src]).mean()
        cur = (df["year"] == yr).to_numpy()
        df.loc[cur, "_gain"] = (raw_gain[cur] - slot[cur].map(curve)).to_numpy()
    # EWM over the driver's past races where he was classified (NaN rows skipped by ewm)
    df["driver_gain_ewm"] = df.groupby("driver")["_gain"].transform(
        lambda s: s.shift(1).ewm(span=SPAN, min_periods=3, ignore_na=True).mean())
    tg = df.groupby(["team", *race], sort=False)["_gain"].mean().reset_index().sort_values(race)
    tg["team_gain_ewm"] = tg.groupby("team")["_gain"].transform(
        lambda s: s.shift(1).ewm(span=SPAN, min_periods=3, ignore_na=True).mean())
    df = df.merge(tg[["team", *race, "team_gain_ewm"]], on=["team", *race], how="left")
    grid = pd.to_numeric(df["grid_position"], errors="coerce")
    fin = pd.to_numeric(df["finish_position"], errors="coerce")
    classified = fin.notna() & (pd.to_numeric(df.get("dnf", 0), errors="coerce").fillna(0) == 0)

    # --- teammate ----------------------------------------------------------- #
    fc = fin.where(classified)
    team_sum = fc.groupby([df["team"], df["year"], df["round"]]).transform("sum")
    team_cnt = fc.groupby([df["team"], df["year"], df["round"]]).transform("count")
    df["_tm_delta"] = np.where((team_cnt == 2) & fc.notna(), fc - (team_sum - fc), np.nan)
    df["teammate_finish_delta_ewm"] = df.groupby("driver")["_tm_delta"].transform(
        lambda s: s.shift(1).ewm(span=10, min_periods=3, ignore_na=True).mean())
    if "teammate_quali_delta" in df.columns:
        df["teammate_quali_delta_ewm"] = df.groupby("driver")["teammate_quali_delta"].transform(
            lambda s: s.shift(1).ewm(span=10, min_periods=3, ignore_na=True).mean())
    else:
        df["teammate_quali_delta_ewm"] = np.nan

    # --- traffic (within race, uses this race's grid + pre-race Elo) --------- #
    elo = pd.to_numeric(df.get("driver_elo"), errors="coerce")
    df["_elo"] = elo
    sb, wa, rg = [], [], []
    for _, idx in df.groupby(race, sort=False).indices.items():
        g = grid.to_numpy()[idx]
        e = df["_elo"].to_numpy()[idx]
        e_rank = pd.Series(-e).rank(method="average").to_numpy()
        for k in range(len(idx)):
            behind = (g > g[k]) & (g <= g[k] + 5)
            ahead = (g < g[k]) & (g >= g[k] - 5)
            sb.append((idx[k], int(np.sum(behind & (e > e[k] + 25)))))
            wa.append((idx[k], int(np.sum(ahead & (e < e[k] - 25)))))
            rg.append((idx[k], g[k] - e_rank[k]))
    pos = df.index
    for name, vals in (("stronger_behind_n", sb), ("weaker_ahead_n", wa), ("elo_rank_gap", rg)):
        arr = np.full(len(df), np.nan)
        for i, v in vals:
            arr[i] = v
        df[name] = pd.Series(arr, index=pos)

    # --- track type --------------------------------------------------------- #
    df["_ctype"] = df["circuit_key"].map(CIRCUIT_TYPE).fillna("traction")
    tb = (df.assign(_fin=fc).groupby(["team", *race, "_ctype"], sort=False)["_fin"].min()
          .reset_index().sort_values(race))
    tb["_overall"] = tb.groupby("team")["_fin"].transform(
        lambda s: s.shift(1).rolling(6, min_periods=3).mean())
    tb["_type"] = tb.groupby(["team", "_ctype"])["_fin"].transform(
        lambda s: s.shift(1).rolling(6, min_periods=2).mean())
    tb["team_type_form_delta"] = tb["_type"] - tb["_overall"]
    df = df.merge(tb[["team", *race, "team_type_form_delta"]], on=["team", *race], how="left")

    # --- trend -------------------------------------------------------------- #
    if "car_pace_delta_pct" in df.columns:
        tp = df.groupby(["team", *race], sort=False)["car_pace_delta_pct"].mean().reset_index().sort_values(race)
        tp["_prev"] = tp.groupby("team")["car_pace_delta_pct"].transform(lambda s: _ewm_prev(s, 5))
        tp["car_pace_trend"] = tp["car_pace_delta_pct"] - tp["_prev"]
        df = df.merge(tp[["team", *race, "car_pace_trend"]], on=["team", *race], how="left")
    else:
        df["car_pace_trend"] = np.nan

    # --- regs --------------------------------------------------------------- #
    rounds = df.groupby("year")["round"].max()
    yrs = sorted(rounds.index)
    start_of = {}
    for yr in yrs:
        rr = max(r for r in REG_RESETS if r <= yr) if yr >= REG_RESETS[0] else yr
        start_of[yr] = sum(int(rounds[v]) for v in yrs if rr <= v < yr)
    df["reg_era_progress"] = ((df["year"].map(start_of) + df["round"] - 1) / 44.0).clip(upper=1.0)

    # --- sprint weekend (failure analysis 2026-11-08: worse in every season 2021-26) --- #
    if "sprint_position_rel" in df.columns:
        sp = pd.to_numeric(df["sprint_position_rel"], errors="coerce").fillna(0.5)
        df["is_sprint_weekend"] = (sp != 0.5).groupby([df["year"], df["round"]]).transform("any").astype(float)
    else:
        df["is_sprint_weekend"] = 0.0

    df = df.sort_values("_order").drop(columns=["_order", "_gain", "_tm_delta", "_elo", "_ctype"])
    return df.reset_index(drop=True)
