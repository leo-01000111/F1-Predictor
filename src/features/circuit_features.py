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

    For each (circuit_key, grid_bin) group, computes an expanding mean of
    finish_position with shift(1) so the current race is never included.

    When no history exists for a (circuit, bin) pair, falls back to the
    global average finish for that grid bin across all circuits.

    Feature: circuit_grid_conversion — lower is better (reflects a strong
    conversion at this grid slot on this track).
    """
    df = race_df.copy()

    df["_grid_bin"] = df["grid_position"].apply(_grid_bin)

    # Sort so expanding window processes races chronologically within each group
    df = df.sort_values(["circuit_key", "_grid_bin", "year", "round"])

    df["circuit_grid_conversion"] = (
        df.groupby(["circuit_key", "_grid_bin"])["finish_position"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )

    # Impute missing values with global avg finish per grid bin
    global_avg = df.groupby("_grid_bin")["finish_position"].mean()
    df["circuit_grid_conversion"] = df["circuit_grid_conversion"].fillna(
        df["_grid_bin"].map(global_avg)
    )

    df = df.drop(columns=["_grid_bin"])
    return df
