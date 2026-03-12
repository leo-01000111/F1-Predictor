"""
Free Practice feature engineering.
Merges FP1/FP2/FP3 pace data into the feature matrix.

Key features:
  fp_best_gap_s       : best gap to session fastest across FP1/FP2/FP3 (lowest = fastest)
  fp2_best_gap_s      : FP2-specific gap (best race-sim day proxy)
  fp_long_run_delta   : best long-run pace delta across FP1/FP2/FP3 (lower = better race pace)
  fp_total_laps       : total laps completed across all FP sessions (reliability proxy)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def merge_practice_features(base_df: pd.DataFrame, practice_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge FP features into base_df (one row per driver per race).
    base_df must have: year, round, driver
    practice_df must have: year, round, driver, fp1_best_gap_s, fp2_best_gap_s, fp3_best_gap_s, etc.
    """
    if practice_df is None or practice_df.empty:
        for col in ["fp_best_gap_s", "fp2_best_gap_s", "fp_long_run_delta", "fp_total_laps"]:
            base_df[col] = np.nan
        return base_df

    fp = practice_df.copy()

    # Best gap to fastest across all FP sessions (minimum = closest to fastest)
    gap_cols = [c for c in fp.columns if c.endswith("_best_gap_s")]
    if gap_cols:
        fp["fp_best_gap_s"] = fp[gap_cols].min(axis=1)
    else:
        fp["fp_best_gap_s"] = np.nan

    # FP2 gap specifically (race-sim day, best long run proxy)
    if "fp2_best_gap_s" not in fp.columns:
        fp["fp2_best_gap_s"] = np.nan

    # Best long-run pace delta (minimum delta = best relative race pace)
    long_run_cols = [c for c in fp.columns if c.endswith("_long_run_delta")]
    if long_run_cols:
        fp["fp_long_run_delta"] = fp[long_run_cols].min(axis=1)
    else:
        fp["fp_long_run_delta"] = np.nan

    # Total laps across FP1/FP2/FP3
    laps_cols = [c for c in fp.columns if c.endswith("_laps") and c.startswith("fp")]
    if laps_cols:
        fp["fp_total_laps"] = fp[laps_cols].sum(axis=1, min_count=1)
    else:
        fp["fp_total_laps"] = np.nan

    keep_cols = ["year", "round", "driver", "fp_best_gap_s", "fp2_best_gap_s", "fp_long_run_delta", "fp_total_laps"]
    fp_sub = fp[[c for c in keep_cols if c in fp.columns]]

    merged = base_df.merge(fp_sub, on=["year", "round", "driver"], how="left")

    # Fill missing FP data with column median (some rounds have no FP e.g. sprint weekends)
    for col in ["fp_best_gap_s", "fp2_best_gap_s", "fp_long_run_delta", "fp_total_laps"]:
        if col in merged.columns:
            median = merged[col].median()
            merged[col] = merged[col].fillna(median if not np.isnan(median) else 0.0)


    # Clip remaining outliers to physically plausible bounds.
    if "fp_best_gap_s" in merged.columns:
        merged["fp_best_gap_s"] = merged["fp_best_gap_s"].clip(lower=0.0, upper=6.0)
    if "fp2_best_gap_s" in merged.columns:
        merged["fp2_best_gap_s"] = merged["fp2_best_gap_s"].clip(lower=0.0, upper=6.0)
    if "fp_long_run_delta" in merged.columns:
        merged["fp_long_run_delta"] = merged["fp_long_run_delta"].clip(lower=-8.0, upper=8.0)
    if "fp_total_laps" in merged.columns:
        merged["fp_total_laps"] = merged["fp_total_laps"].clip(lower=0.0, upper=200.0)
    return merged

