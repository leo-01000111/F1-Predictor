"""Merge Reddit sentiment scores into the feature matrix."""
from __future__ import annotations

import pandas as pd


def merge_sentiment(base_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge driver and team sentiment scores onto base_df.

    base_df must have: year, round, driver, team
    sentiment_df must have: year, round, entity_type, entity, sentiment_score
    """
    if sentiment_df is None or sentiment_df.empty:
        base_df["driver_sentiment_7d"] = 0.0
        base_df["team_sentiment_7d"] = 0.0
        return base_df

    # Driver sentiment
    driver_sent = sentiment_df[sentiment_df["entity_type"] == "driver"][
        ["year", "round", "entity", "sentiment_score"]
    ].rename(columns={"entity": "driver", "sentiment_score": "driver_sentiment_7d"})

    # Team sentiment — map team names (align with race_df team naming)
    team_sent = sentiment_df[sentiment_df["entity_type"] == "team"][
        ["year", "round", "entity", "sentiment_score"]
    ].rename(columns={"entity": "team", "sentiment_score": "team_sentiment_7d"})

    merged = base_df.merge(driver_sent, on=["year", "round", "driver"], how="left")
    merged = merged.merge(team_sent, on=["year", "round", "team"], how="left")

    # Fill missing sentiment with 0 (neutral)
    merged["driver_sentiment_7d"] = merged["driver_sentiment_7d"].fillna(0.0)
    merged["team_sentiment_7d"] = merged["team_sentiment_7d"].fillna(0.0)

    return merged
