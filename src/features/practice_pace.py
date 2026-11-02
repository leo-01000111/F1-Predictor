"""
Practice long-run pace (ROADMAP idea 2), from OpenF1 practice laps (2023 on).

Friday long runs on race fuel show race pace in a way a single qualifying lap
does not. Per practice session:

  long run   a stint with >= MIN_LAPS clean laps: timed, not an out-lap, not
             the stint's first lap, within 107% of the stint median
  pace       the median of those clean laps
  gap        driver's best long run vs the median long run of the session, in %
             (negative = faster than the field); clipped to +-CLIP_PCT

Features per (year, round, driver), averaged over the weekend's sessions:
  fp_longrun_gap_pct     the driver's own long-run gap
  team_longrun_gap_pct   the faster of the team's two cars (race pace is mostly car)
  fp_longrun_laps        clean long-run laps the driver did (how much evidence)

Seasons before 2023 have no data (NaN); XGBoost treats that as missing.

Optional refinements (ROADMAP Phase 3b, scripts/practice_compound_backtest.py),
all off by default (V5 = the production method):
  clean            drop stints that are not steady runs: lap-time IQR > STEADY_IQR_PCT
                   of the stint median (push / cool-down alternation) or a median more
                   than SLOW_PCT over the session's fastest lap
  adjust_age       correct every lap to tyre age 0 with the session's within-stint
                   slope per compound (tyre wear net of fuel burn), clipped to 0-0.4 %/lap
  adjust_compound  normalise each stint to MEDIUM: per session, OLS of log pace on
                   compound dummies + the team's qualifying pace (car_pace_delta_pct,
                   so who ran which tyre does not bias it), ridge-shrunk toward
                   COMPOUND_PRIOR (within-driver soft/hard vs medium deltas, 2023 only)
  session_weights  weighted mean over sessions instead of the plain mean
  deg              fills fp_longrun_deg_pct: the driver's within-stint slope minus the
                   session's slope for the same compound, %/lap
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_LAPS = 5
CLIP_PCT = 5.0
PACE_COLS = ["fp_longrun_gap_pct", "team_longrun_gap_pct"]
OUT_COLS = ["fp_longrun_gap_pct", "team_longrun_gap_pct", "fp_longrun_laps", "fp_longrun_deg_pct"]

STEADY_IQR_PCT = 3.0
SLOW_PCT = 12.0
MAX_SLOPE_PCT = 0.4
COMPOUND_PRIOR = {"SOFT": -0.3, "HARD": 0.2}   # % vs MEDIUM
PRIOR_WEIGHT = 3.0                              # pseudo-runs per compound
DEG_CLIP = 0.5

V5: dict = {}
# Pre-registered primary variant of the Phase 3b test (2026-11-02). Not shipped:
# walk-forward 2024-2026 top-3 log loss +0.025 [-0.001, +0.051] vs V5
# (models/experiments/practice_compound_backtest.json)
V6: dict = {"clean": True, "adjust_age": True, "adjust_compound": True,
            "session_weights": {"FP1": 0.5, "FP2": 1.0, "FP3": 0.5}}

_KEY = ["year", "round", "session", "driver", "stint_number"]
_SESS = ["year", "round", "session"]


def long_run_laps(laps: pd.DataFrame, clean: bool = False) -> pd.DataFrame:
    """Lap-level rows of every long-run stint (the laps behind stint_paces)."""
    df = laps.dropna(subset=["lap_duration", "stint_number"]).copy()
    df = df[~df["is_pit_out_lap"]]
    first = df.groupby(_KEY)["lap_number"].transform("min")
    df = df[df["lap_number"] > first]
    med = df.groupby(_KEY)["lap_duration"].transform("median")
    df = df[df["lap_duration"] <= med * 1.07]
    df = df[df.groupby(_KEY)["lap_duration"].transform("size") >= MIN_LAPS]
    if clean and not df.empty:
        g = df.groupby(_KEY)["lap_duration"]
        med = g.transform("median")
        iqr = (g.transform(lambda s: s.quantile(0.75) - s.quantile(0.25)) / med) * 100
        best = laps.groupby(_SESS)["lap_duration"].transform("min").reindex(df.index)
        slow = (med / best - 1) * 100
        df = df[(iqr <= STEADY_IQR_PCT) & ~(slow > SLOW_PCT)]
    return df


def _age(df: pd.DataFrame) -> pd.Series:
    """Tyre age per lap; laps into the stint where OpenF1 has no age."""
    into = (df["lap_number"] - df.groupby(_KEY)["lap_number"].transform("min")).astype(float)
    if "tyre_age" not in df.columns:
        return into
    return pd.to_numeric(df["tyre_age"], errors="coerce").fillna(into)


def _slopes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Within-stint lap-time slope (s/lap) per stint and pooled per session x compound."""
    x = df["_age"] - df.groupby(_KEY)["_age"].transform("mean")
    y = df["lap_duration"] - df.groupby(_KEY)["lap_duration"].transform("mean")
    d = df[_KEY + ["compound"]].assign(xy=x * y, xx=x * x)
    d["compound"] = d["compound"].fillna("UNKNOWN")
    st = d.groupby(_KEY + ["compound"])[["xy", "xx"]].sum().reset_index()
    st["slope"] = st["xy"] / st["xx"].replace(0, np.nan)
    sc = st.groupby(_SESS + ["compound"]).agg(xy=("xy", "sum"), xx=("xx", "sum"), n=("xy", "size")).reset_index()
    sess = st.groupby(_SESS)[["xy", "xx"]].sum()
    sess_slope = (sess["xy"] / sess["xx"].replace(0, np.nan)).rename("sess_slope").reset_index()
    sc = sc.merge(sess_slope, on=_SESS, how="left")
    sc["comp_slope"] = np.where(sc["n"] >= 3, sc["xy"] / sc["xx"].replace(0, np.nan), sc["sess_slope"])
    return st[_KEY + ["slope"]], sc[_SESS + ["compound", "comp_slope"]]


def _compound_offsets(best: pd.DataFrame) -> pd.Series:
    """% offset vs MEDIUM per row of `best` (one row per driver x session x compound)."""
    off = pd.Series(0.0, index=best.index)
    k = np.sqrt(PRIOR_WEIGHT)
    for _, g in best.groupby(_SESS):
        g = g[g["compound"].isin(["SOFT", "MEDIUM", "HARD"])]
        if g.empty:
            continue
        y = np.log(g["pace_s"].to_numpy(dtype=float)) * 100
        if "car_pace_delta_pct" in g.columns:
            car = pd.to_numeric(g["car_pace_delta_pct"], errors="coerce")
            car = car.fillna(car.mean()).fillna(0.0).to_numpy(dtype=float)
        else:
            car = np.zeros(len(g))
        soft = (g["compound"] == "SOFT").to_numpy(dtype=float)
        hard = (g["compound"] == "HARD").to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(g)), car, soft, hard])
        Xp = np.array([[0.0, 0.0, k, 0.0], [0.0, 0.0, 0.0, k]])
        yp = np.array([k * COMPOUND_PRIOR["SOFT"], k * COMPOUND_PRIOR["HARD"]])
        coef = np.linalg.lstsq(np.vstack([X, Xp]), np.concatenate([y, yp]), rcond=None)[0]
        off.loc[g.index] = soft * coef[2] + hard * coef[3]
    return off


def stint_paces(laps: pd.DataFrame, clean: bool = False, adjust_age: bool = False) -> pd.DataFrame:
    """One row per long-run stint: year, round, session, driver, stint_number, pace_s, n_laps,
    compound, deg_pct (the stint's slope minus its session x compound slope, %/lap)."""
    df = long_run_laps(laps, clean=clean)
    if df.empty:
        return pd.DataFrame(columns=_KEY + ["pace_s", "n_laps", "compound", "deg_pct"])
    df["compound"] = df["compound"].fillna("UNKNOWN")
    df["_age"] = _age(df)
    st_slope, sc_slope = _slopes(df)
    if adjust_age:
        df = df.merge(sc_slope, on=_SESS + ["compound"], how="left")
        med = df.groupby(_KEY)["lap_duration"].transform("median")
        slope = df["comp_slope"].fillna(0.0).clip(lower=0.0, upper=med * MAX_SLOPE_PCT / 100)
        df["lap_duration"] = df["lap_duration"] - slope * df["_age"]
    out = df.groupby(_KEY).agg(pace_s=("lap_duration", "median"), n_laps=("lap_duration", "size"),
                               compound=("compound", "first")).reset_index()
    out = out.merge(st_slope, on=_KEY, how="left").merge(sc_slope, on=_SESS + ["compound"], how="left")
    out["deg_pct"] = (out["slope"] - out["comp_slope"]) / out["pace_s"] * 100
    return out.drop(columns=["slope", "comp_slope"])


def _wmean(df: pd.DataFrame, by: list[str], col: str, weights: dict | None) -> pd.Series:
    """Mean of `col` per group; with session weights, the weighted mean."""
    if not weights:
        return df.groupby(by)[col].mean()
    w = df["session"].map(weights).fillna(1.0)
    d = df.assign(_wx=df[col] * w, _w=w.where(df[col].notna(), 0.0))
    s = d.groupby(by)[["_wx", "_w"]].sum(min_count=1)
    return s["_wx"] / s["_w"].replace(0, np.nan)


def practice_pace(laps: pd.DataFrame, teams: pd.DataFrame, clean: bool = False, adjust_age: bool = False,
                  adjust_compound: bool = False, session_weights: dict | None = None,
                  deg: bool = False) -> pd.DataFrame:
    """Weekend long-run features per (year, round, driver).

    teams: year, round, driver, team (the race entry list) for the team feature;
    its car_pace_delta_pct is used when adjust_compound is on.
    """
    cols = ["year", "round", "driver"] + OUT_COLS
    if laps is None or laps.empty:
        return pd.DataFrame(columns=cols)
    st = stint_paces(laps, clean=clean, adjust_age=adjust_age)
    if st.empty:
        return pd.DataFrame(columns=cols)
    tcols = ["year", "round", "driver", "team"] + (["car_pace_delta_pct"] if "car_pace_delta_pct" in teams else [])
    t = teams[tcols].drop_duplicates(["year", "round", "driver"])
    if adjust_compound:
        bc = st.groupby(_SESS + ["driver", "compound"]).agg(pace_s=("pace_s", "min")).reset_index()
        bc = bc.merge(t, on=["year", "round", "driver"], how="left")
        bc["offset"] = _compound_offsets(bc)
        st = st.merge(bc[_SESS + ["driver", "compound", "offset"]], on=_SESS + ["driver", "compound"], how="left")
        st["pace_s"] = st["pace_s"] * np.exp(-st["offset"].fillna(0.0) / 100)
    best = st.groupby(_SESS + ["driver"]).agg(pace_s=("pace_s", "min"), n_laps=("n_laps", "sum")).reset_index()
    dg = st.assign(_wd=st["deg_pct"] * st["n_laps"], _n=st["n_laps"].where(st["deg_pct"].notna(), 0))
    dg = dg.groupby(_SESS + ["driver"])[["_wd", "_n"]].sum()
    best = best.merge((dg["_wd"] / dg["_n"].replace(0, np.nan)).rename("deg").reset_index(),
                      on=_SESS + ["driver"], how="left")
    field = best.groupby(_SESS)["pace_s"].transform("median")
    best["gap"] = ((best["pace_s"] / field - 1) * 100).clip(-CLIP_PCT, CLIP_PCT)
    best["deg"] = best["deg"].clip(-DEG_CLIP, DEG_CLIP)
    # Sessions with only a handful of long runs say little about the field
    n_runs = best.groupby(_SESS)["driver"].transform("size")
    best = best[n_runs >= 6]
    drv = pd.DataFrame({
        "fp_longrun_gap_pct": _wmean(best, ["year", "round", "driver"], "gap", session_weights),
        "fp_longrun_laps": best.groupby(["year", "round", "driver"])["n_laps"].sum(),
        "fp_longrun_deg_pct": _wmean(best, ["year", "round", "driver"], "deg", session_weights),
    }).reset_index()
    if not deg:
        drv["fp_longrun_deg_pct"] = np.nan

    best = best.merge(t[["year", "round", "driver", "team"]], on=["year", "round", "driver"], how="left")
    team_sess = best.groupby(_SESS + ["team"])["gap"].min().reset_index()
    team = _wmean(team_sess, ["year", "round", "team"], "gap", session_weights)
    team = team.rename("team_longrun_gap_pct").reset_index()
    out = t[["year", "round", "driver", "team"]].merge(drv, on=["year", "round", "driver"], how="left").merge(
        team, on=["year", "round", "team"], how="left")
    out = out[out[["fp_longrun_gap_pct", "team_longrun_gap_pct"]].notna().any(axis=1)]
    return out[cols]


def add_practice_pace(df: pd.DataFrame, laps: pd.DataFrame | None, opts: dict | None = None) -> pd.DataFrame:
    """Merge the practice features into a race-level frame (NaN where no data).

    opts: practice_pace() options; None = the production method.
    """
    opts = V5 if opts is None else opts
    df = df.drop(columns=[c for c in OUT_COLS if c in df.columns])
    if laps is None or laps.empty:
        for c in OUT_COLS:
            df[c] = np.nan
        return df
    pace = practice_pace(laps, df, **opts)
    return df.merge(pace, on=["year", "round", "driver"], how="left")
