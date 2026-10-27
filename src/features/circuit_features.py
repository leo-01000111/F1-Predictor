"""Circuit-level feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _grid_bin(pos) -> str:
    """Map a grid position integer to a discrete bin label."""
    try:
        p = int(pos)
    except (TypeError, ValueError):
        return "P16+"
    if p == 1:
        return "P1"
    if p == 2:
        return "P2"
    if p == 3:
        return "P3"
    if p <= 6:
        return "P4-6"
    if p <= 10:
        return "P7-10"
    if p <= 15:
        return "P11-15"
    return "P16+"


def add_grid_conversion(race_df: pd.DataFrame) -> pd.DataFrame:
    """
    Historical average finish position from each grid-position bin at each circuit.

    For each (circuit_key, grid_bin) group, averages finish_position over
    earlier races only. Aggregating per race first matters: a row-level
    shift(1) let drivers sharing a bin in the same race (P4-6, P7-10, ...)
    see each other's finish, a target leak that inference never has.

    When no history exists for a (circuit, bin) pair, falls back to the
    average finish for that grid bin across all circuits in earlier races.

    Feature: circuit_grid_conversion — lower is better (reflects a strong
    conversion at this grid slot on this track).
    """
    df = race_df.copy()
    df["_grid_bin"] = df["grid_position"].apply(_grid_bin)
    finish = pd.to_numeric(df["finish_position"], errors="coerce")
    df["_finish"] = finish

    def _prior_mean(keys: list[str]) -> pd.Series:
        """Mean finish over races before the current one, per `keys` group."""
        race_keys = keys + ["year", "round"]
        per_race = (
            df.groupby(race_keys)["_finish"].agg(["sum", "count"])
            .reset_index()
            .sort_values(race_keys)
        )
        prior_sum = per_race.groupby(keys)["sum"].cumsum() - per_race["sum"]
        prior_cnt = per_race.groupby(keys)["count"].cumsum() - per_race["count"]
        per_race["_mean"] = prior_sum / prior_cnt.where(prior_cnt > 0)
        lookup = per_race.set_index(race_keys)["_mean"]
        idx = pd.MultiIndex.from_frame(df[race_keys])
        return pd.Series(lookup.reindex(idx).values, index=df.index)

    df["circuit_grid_conversion"] = _prior_mean(["circuit_key", "_grid_bin"])
    df["circuit_grid_conversion"] = df["circuit_grid_conversion"].fillna(_prior_mean(["_grid_bin"]))

    df = df.drop(columns=["_grid_bin", "_finish"])
    return df
