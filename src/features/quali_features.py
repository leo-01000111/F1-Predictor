"""
Qualifying reliability ("quali distrust") features.

How much should the model trust this weekend's qualifying order? A rain-hit
session decided by one lap on a drying (or flooding) track says less about
race pace than a normal dry session. Computed from the qualifying
classification only (position + each driver's time from the last segment they
reached, as stored in quali_results), so training and live inference use the
same function on the same columns.

Per race:
  quali_spread_pct     median Q3 time of P2-P10 vs P1, %. ~1% dry, 3-20% wet.
  quali_segment_order  Spearman(position, last-segment time). ~1 when Q3 was
                       faster than Q2 than Q1 (normal track evolution); low or
                       negative when conditions turned between segments.
  quali_shuffle        mean |quali_vs_prior| over the field: how far the order
                       moved from each driver's recent qualifying form.
Per driver:
  quali_vs_prior       quali position (0 = pole, 1 = last) minus the driver's
                       EWM of past qualifying positions. Positive = qualified
                       worse than usual (a fast car stuck at the back).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QUALI_RELIABILITY_COLS = [
    "quali_spread_pct",
    "quali_segment_order",
    "quali_shuffle",
    "quali_vs_prior",
]

PRIOR_SPAN = 5          # EWM span of past qualifying positions
PRIOR_MIN_RACES = 2     # fewer past qualis: fall back to the team's prior


def _race_stats(d: pd.DataFrame) -> pd.Series:
    d = d.sort_values("quali_position")
    t = pd.to_numeric(d["quali_best_time_s"], errors="coerce")
    pos = d["quali_position"]
    spread = np.nan
    top = t[(pos <= 10).to_numpy()].dropna()
    p1 = t[(pos == pos.min()).to_numpy()]
    if len(top) >= 5 and len(p1) and pd.notna(p1.iloc[0]) and p1.iloc[0] > 0:
        rest = top.iloc[1:]
        spread = float((rest.median() / p1.iloc[0] - 1.0) * 100.0)
    ok = t.notna().to_numpy()
    order = np.nan
    if ok.sum() >= 10:
        order = float(pd.Series(pos.to_numpy()[ok]).corr(pd.Series(t.to_numpy()[ok]), method="spearman"))
    return pd.Series({"quali_spread_pct": spread, "quali_segment_order": order})


def add_quali_reliability(quali: pd.DataFrame) -> pd.DataFrame:
    """quali: year, round, driver, team, quali_position, quali_best_time_s for
    every race up to and including the one(s) to featurise (history drives the
    priors). Returns year, round, driver + QUALI_RELIABILITY_COLS.
    """
    q = quali[["year", "round", "driver", "team", "quali_position", "quali_best_time_s"]].copy()
    q["quali_position"] = pd.to_numeric(q["quali_position"], errors="coerce")
    q = q.dropna(subset=["quali_position"]).sort_values(["year", "round", "quali_position"])
    q = q.drop_duplicates(["year", "round", "driver"])
    n = q.groupby(["year", "round"])["quali_position"].transform("max")
    q["qrel"] = (q["quali_position"] - 1) / (n - 1).clip(lower=1)

    # Driver prior: EWM of earlier qualis only (shift 1 within the driver's history)
    g = q.groupby("driver")["qrel"]
    q["_prior"] = g.transform(lambda s: s.shift(1).ewm(span=PRIOR_SPAN, min_periods=PRIOR_MIN_RACES).mean())
    # Team prior for rookies: EWM of the team's earlier race means
    team_race = q.groupby(["team", "year", "round"], sort=False)["qrel"].mean().reset_index()
    team_race = team_race.sort_values(["year", "round"])
    team_race["_tprior"] = team_race.groupby("team")["qrel"].transform(
        lambda s: s.shift(1).ewm(span=PRIOR_SPAN, min_periods=1).mean())
    q = q.merge(team_race[["team", "year", "round", "_tprior"]], on=["team", "year", "round"], how="left")
    prior = q["_prior"].fillna(q["_tprior"]).fillna(0.5)
    q["quali_vs_prior"] = q["qrel"] - prior
    q["quali_shuffle"] = q.groupby(["year", "round"])["quali_vs_prior"].transform(lambda s: s.abs().mean())

    stats = q.groupby(["year", "round"]).apply(_race_stats, include_groups=False).reset_index()
    q = q.merge(stats, on=["year", "round"], how="left")
    q["quali_spread_pct"] = q["quali_spread_pct"].clip(lower=0, upper=10)
    return q[["year", "round", "driver", *QUALI_RELIABILITY_COLS]]
