"""
Reddit sentiment collector using PRAW.
Targets r/formula1 and r/f1technical.

For each historical race: pulls posts from the 7-day window before the race,
scores sentiment per driver/team mention using a RoBERTa model from HuggingFace,
and aggregates weighted sentiment scores.

Outputs:
  - data/raw/sentiment_history.parquet — sentiment scores per driver/team per race

At inference time: pulls the most recent 7 days.

Setup:
  1. Create a Reddit app at https://www.reddit.com/prefs/apps (script type)
  2. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import praw
from loguru import logger

ROOT = Path(__file__).parent.parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SUBREDDITS = ["formula1", "f1technical"]
SENTIMENT_WINDOW_DAYS = 7
MIN_POST_SCORE = 10
MAX_POSTS_PER_SUBREDDIT = 100
TOP_COMMENTS_PER_POST = 10

# Known driver abbreviations and common full-name variants
# Format: abbr -> [search patterns]
DRIVER_ALIASES: dict[str, list[str]] = {
    "VER": ["verstappen", "max verstappen", "max ver", "super max"],
    "NOR": ["norris", "lando norris", "lando"],
    "LEC": ["leclerc", "charles leclerc", "charles"],
    "HAM": ["hamilton", "lewis hamilton", "lewis"],
    "SAI": ["sainz", "carlos sainz", "carlos"],
    "RUS": ["russell", "george russell", "george"],
    "PIA": ["piastri", "oscar piastri", "oscar"],
    "ALO": ["alonso", "fernando alonso", "fernando"],
    "STR": ["stroll", "lance stroll", "lance"],
    "PER": ["perez", "sergio perez", "checo"],
    "GAS": ["gasly", "pierre gasly", "pierre"],
    "OCO": ["ocon", "esteban ocon", "esteban"],
    "TSU": ["tsunoda", "yuki tsunoda", "yuki"],
    "ALB": ["albon", "alex albon", "alex"],
    "HUL": ["hulkenberg", "nico hulkenberg", "nico"],
    "MAG": ["magnussen", "kevin magnussen", "kevin"],
    "BOT": ["bottas", "valtteri bottas", "valtteri"],
    "ZHO": ["zhou", "guanyu zhou", "zhou guanyu"],
    "SAR": ["sargeant", "logan sargeant", "logan"],
    "LAW": ["lawson", "liam lawson", "liam"],
    "BEA": ["bearman", "oliver bearman", "ollie bearman"],
    "ANT": ["antonelli", "andrea kimi antonelli", "kimi antonelli"],
    "DOO": ["doohan", "jack doohan"],
    "HAD": ["hadjar", "isack hadjar"],
    "BOR": ["bortoleto", "gabriel bortoleto"],
}

TEAM_ALIASES: dict[str, list[str]] = {
    "Red Bull": ["red bull", "redbull", "rb racing"],
    "Ferrari": ["ferrari", "scuderia ferrari", "the prancing horse"],
    "Mercedes": ["mercedes", "mercedes-amg", "mercs"],
    "McLaren": ["mclaren", "woking"],
    "Aston Martin": ["aston martin", "aston", "amr"],
    "Alpine": ["alpine", "renault"],
    "Williams": ["williams"],
    "RB": ["rb", "racing bulls", "alphatauri", "alpha tauri", "faenza"],
    "Kick Sauber": ["sauber", "kick sauber", "alfa romeo"],
    "Haas": ["haas"],
}


# --------------------------------------------------------------------------- #
# Sentiment model
# --------------------------------------------------------------------------- #

class SentimentScorer:
    """
    Lazy-loaded HuggingFace RoBERTa sentiment model.
    Returns score in [-1, 1] (negative → positive).
    """

    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self._pipeline = None

    def _load(self) -> None:
        if self._pipeline is None:
            from transformers import pipeline as hf_pipeline
            logger.info(f"Loading sentiment model: {self.model_name}")
            self._pipeline = hf_pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
                truncation=True,
                max_length=512,
            )
            logger.info("Sentiment model loaded.")

    def score(self, texts: list[str]) -> list[float]:
        """
        Score a list of texts. Returns float in [-1, 1] per text.
        Positive label score - Negative label score.
        """
        self._load()
        results = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Handle empty strings
            batch = [t if t.strip() else "neutral" for t in batch]
            try:
                preds = self._pipeline(batch)
                for pred in preds:
                    # pred is a list of {label, score} dicts
                    label_scores = {p["label"].lower(): p["score"] for p in pred}
                    pos = label_scores.get("positive", label_scores.get("label_2", 0.0))
                    neg = label_scores.get("negative", label_scores.get("label_0", 0.0))
                    results.append(float(pos - neg))
            except Exception as exc:
                logger.warning(f"Sentiment scoring failed for batch: {exc}")
                results.extend([0.0] * len(batch))

        return results


# Global scorer instance (lazy)
_scorer = SentimentScorer()


# --------------------------------------------------------------------------- #
# Reddit client
# --------------------------------------------------------------------------- #

def _build_reddit_client() -> praw.Reddit:
    """Build PRAW Reddit client from environment / .env file."""
    from dotenv import load_dotenv
    import os

    load_dotenv(ROOT / ".env")

    client_id = os.getenv("REDDIT_CLIENT_ID", "")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
    user_agent = os.getenv("REDDIT_USER_AGENT", "f1_prediction_bot/1.0")

    if not client_id or not client_secret:
        raise EnvironmentError(
            "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set in .env\n"
            "See .env.example for instructions."
        )

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        read_only=True,
    )


# --------------------------------------------------------------------------- #
# Text extraction from Reddit
# --------------------------------------------------------------------------- #

def _fetch_posts_in_window(
    reddit: praw.Reddit,
    subreddit_name: str,
    start_ts: float,
    end_ts: float,
    limit: int = MAX_POSTS_PER_SUBREDDIT,
) -> list[dict]:
    """
    Fetch posts from a subreddit within a Unix timestamp window.
    PRAW's search supports before/after as timestamp strings.
    """
    subreddit = reddit.subreddit(subreddit_name)
    posts = []

    try:
        # PRAW search with timestamp filter
        query = f"timestamp:{int(start_ts)}..{int(end_ts)}"
        for submission in subreddit.search(
            query,
            sort="top",
            syntax="cloudsearch",
            limit=limit,
        ):
            if submission.score < MIN_POST_SCORE:
                continue
            posts.append(
                {
                    "id": submission.id,
                    "title": submission.title,
                    "selftext": submission.selftext[:2000],  # cap length
                    "score": submission.score,
                    "created_utc": submission.created_utc,
                    "subreddit": subreddit_name,
                    "comments": _get_top_comments(submission),
                }
            )
    except Exception as exc:
        logger.warning(f"Search failed for r/{subreddit_name}: {exc}. Trying 'new' fallback.")
        # Fallback: iterate recent posts and filter by timestamp
        try:
            for submission in subreddit.new(limit=200):
                if submission.created_utc < start_ts or submission.created_utc > end_ts:
                    continue
                if submission.score < MIN_POST_SCORE:
                    continue
                posts.append(
                    {
                        "id": submission.id,
                        "title": submission.title,
                        "selftext": submission.selftext[:2000],
                        "score": submission.score,
                        "created_utc": submission.created_utc,
                        "subreddit": subreddit_name,
                        "comments": _get_top_comments(submission),
                    }
                )
                if len(posts) >= limit:
                    break
        except Exception as exc2:
            logger.error(f"Fallback also failed for r/{subreddit_name}: {exc2}")

    return posts


def _get_top_comments(submission, n: int = TOP_COMMENTS_PER_POST) -> list[str]:
    """Get top-level comment bodies, ranked by score."""
    try:
        submission.comment_sort = "top"
        submission.comments.replace_more(limit=0)
        top = sorted(submission.comments.list(), key=lambda c: getattr(c, "score", 0), reverse=True)
        return [c.body[:500] for c in top[:n] if hasattr(c, "body")]
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# Mention detection + sentiment aggregation
# --------------------------------------------------------------------------- #

def _detect_mentions(text: str, aliases: list[str]) -> bool:
    """Return True if any alias appears in the lowercased text."""
    text_lower = text.lower()
    return any(alias in text_lower for alias in aliases)


def _collect_texts_mentioning(posts: list[dict], aliases: list[str]) -> list[tuple[str, float]]:
    """
    Return list of (text, weight) for all text units mentioning any alias.
    Weight = log(1 + post_score) to dampen extremes.
    """
    import math

    results = []
    for post in posts:
        weight = math.log1p(max(post["score"], 1))
        title = post["title"]
        body = post["selftext"]

        # Combine title + body for mention check
        full_text = f"{title} {body}"
        if _detect_mentions(full_text, aliases):
            results.append((full_text[:512], weight))

        # Also check individual comments
        for comment in post["comments"]:
            if _detect_mentions(comment, aliases):
                results.append((comment, weight * 0.5))  # comments weighted less

    return results


def _compute_weighted_sentiment(texts_weights: list[tuple[str, float]]) -> Optional[float]:
    """Compute weighted average sentiment score."""
    if not texts_weights:
        return None

    texts = [tw[0] for tw in texts_weights]
    weights = [tw[1] for tw in texts_weights]

    scores = _scorer.score(texts)

    total_weight = sum(weights)
    if total_weight == 0:
        return None

    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    return weighted_sum / total_weight


# --------------------------------------------------------------------------- #
# High-level collection
# --------------------------------------------------------------------------- #

def collect_sentiment_for_race(
    reddit: praw.Reddit,
    race_date: str,
    year: int,
    round_num: int,
    circuit_key: str,
) -> list[dict]:
    """
    Collect and score sentiment for all drivers and teams for a given race.
    Returns list of dicts with driver/team sentiment scores.
    """
    race_dt = datetime.strptime(race_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_ts = race_dt.timestamp()
    start_ts = (race_dt - timedelta(days=SENTIMENT_WINDOW_DAYS)).timestamp()

    logger.info(f"Fetching Reddit posts for {year} R{round_num} ({circuit_key})")
    all_posts = []
    for sub in SUBREDDITS:
        posts = _fetch_posts_in_window(reddit, sub, start_ts, end_ts)
        all_posts.extend(posts)
        time.sleep(1.0)  # be polite to Reddit API

    logger.info(f"  Found {len(all_posts)} relevant posts")

    rows = []

    # Driver sentiment
    for driver_abbr, aliases in DRIVER_ALIASES.items():
        texts_weights = _collect_texts_mentioning(all_posts, aliases)
        score = _compute_weighted_sentiment(texts_weights)
        rows.append(
            {
                "year": year,
                "round": round_num,
                "circuit_key": circuit_key,
                "entity_type": "driver",
                "entity": driver_abbr,
                "sentiment_score": score,
                "mention_count": len(texts_weights),
            }
        )

    # Team sentiment
    for team_name, aliases in TEAM_ALIASES.items():
        texts_weights = _collect_texts_mentioning(all_posts, aliases)
        score = _compute_weighted_sentiment(texts_weights)
        rows.append(
            {
                "year": year,
                "round": round_num,
                "circuit_key": circuit_key,
                "entity_type": "team",
                "entity": team_name,
                "sentiment_score": score,
                "mention_count": len(texts_weights),
            }
        )

    return rows


def collect_historical_sentiment(race_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collect sentiment for all historical races.
    race_df must have: year, round, circuit_key, race_date
    """
    reddit = _build_reddit_client()
    races = race_df[["year", "round", "circuit_key", "race_date"]].drop_duplicates()

    all_rows = []
    for _, row in races.iterrows():
        try:
            rows = collect_sentiment_for_race(
                reddit,
                race_date=str(row["race_date"]),
                year=int(row["year"]),
                round_num=int(row["round"]),
                circuit_key=str(row["circuit_key"]),
            )
            all_rows.extend(rows)
        except Exception as exc:
            logger.error(f"Sentiment failed for {row['year']} R{row['round']}: {exc}")
        time.sleep(2.0)  # rate limit

    return pd.DataFrame(all_rows)


def collect_live_sentiment(circuit_key: str, race_date: str, year: int, round_num: int) -> pd.DataFrame:
    """Collect current pre-race sentiment for inference."""
    reddit = _build_reddit_client()
    rows = collect_sentiment_for_race(reddit, race_date, year, round_num, circuit_key)
    return pd.DataFrame(rows)


def save_sentiment(df: pd.DataFrame) -> None:
    path = RAW_DIR / "sentiment_history.parquet"
    df.to_parquet(path, index=False)
    logger.info(f"Saved {len(df)} sentiment rows → {path}")


def load_sentiment_history() -> pd.DataFrame:
    path = RAW_DIR / "sentiment_history.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run sentiment collection first. Expected: {path}")
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse
    from src.data_collection.f1_historical import load_race_results

    parser = argparse.ArgumentParser(description="Collect Reddit sentiment for historical F1 races")
    parser.add_argument("--start-year", type=int, default=2021, help="Start year (Reddit history limited)")
    parser.add_argument("--end-year", type=int, default=2024)
    args = parser.parse_args()

    race_df = load_race_results()
    race_df = race_df[
        (race_df["year"] >= args.start_year) & (race_df["year"] <= args.end_year)
    ]

    sentiment_df = collect_historical_sentiment(race_df)
    save_sentiment(sentiment_df)
    logger.info("Done.")
