# Archived: Sentiment Pipeline

**Archived 2026.** Reddit API sentiment was collected but added no predictive signal
in offline tests. Kept for reference only.

## Files

- `sentiment.py` — Reddit/PRAW data collection + HuggingFace RoBERTa scoring
- `sentiment_features.py` — Merge sentiment scores into feature matrix

## Why archived

- Both files were completely disconnected from every active code path
- Signal did not survive cross-validation against tabular pace/form features
- Reddit API rate limits and model latency made it impractical for race-day use

## Credentials

Reddit API credentials (`REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`) are preserved in
`.env` / `.env.example`. The data collection logic is intact here if you want to
revisit with a larger dataset or different NLP approach.

## Revisit conditions

Consider re-enabling if:
- Training set grows beyond ~15k rows
- Sentiment data is sourced from a higher-signal channel (e.g., team radio, press transcripts)
