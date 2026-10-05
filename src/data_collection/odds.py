"""
Bookmaker odds fetcher via The Odds API (https://the-odds-api.com).

Free tier: 500 requests/month — well within budget at ~1 fetch per race weekend.
Fetches race winner market (h2h) for the next F1 event and returns average
decimal odds per driver, keyed by 3-letter abbreviation.

Sign up at https://the-odds-api.com to get a free API key, then add to .env:
    ODDS_API_KEY=your_key_here
"""
from __future__ import annotations

import os
from difflib import SequenceMatcher
from typing import Optional

import requests
from loguru import logger

_BASE = "https://api.the-odds-api.com/v4"

# These are checked in order — first match wins.
_F1_SPORT_CANDIDATES = [
    "motorsport_formula_one",
    "motorsport_f1",
    "formula_one",
    "motorsport_f1_race_winner",
    "formula_1",
]


def _find_f1_sport_key(api_key: str) -> str | None:
    """
    Query /v4/sports/ and return the first key that looks like F1.
    Falls back to trying _F1_SPORT_CANDIDATES directly if the list call fails.
    """
    try:
        resp = requests.get(
            f"{_BASE}/sports/",
            params={"apiKey": api_key, "all": "false"},
            timeout=10,
        )
        if resp.ok:
            sports = resp.json()
            for s in sports:
                key = s.get("key", "").lower()
                title = s.get("title", "").lower()
                if "formula" in key or "formula" in title or ("f1" in key and "motor" in key):
                    logger.info(f"Found F1 sport key: '{s['key']}' (title: '{s.get('title')}')")
                    return s["key"]
            logger.warning("No F1 sport found in available sports list.")
        else:
            logger.warning(f"Sports list call returned {resp.status_code}")
    except requests.RequestException as exc:
        logger.warning(f"Could not fetch sports list: {exc}")
    return None

# Full driver name (lowercase) → 3-letter abbreviation.
# Updated for 2025/2026 grid. Add new drivers here as needed.
DRIVER_NAME_TO_ABBR: dict[str, str] = {
    # 2025/2026 grid
    "max verstappen":        "VER",
    "liam lawson":           "LAW",
    "lando norris":          "NOR",
    "oscar piastri":         "PIA",
    "charles leclerc":       "LEC",
    "lewis hamilton":        "HAM",
    "george russell":        "RUS",
    "andrea kimi antonelli": "ANT",
    "kimi antonelli":        "ANT",
    "fernando alonso":       "ALO",
    "lance stroll":          "STR",
    "pierre gasly":          "GAS",
    "jack doohan":           "DOO",
    "alexander albon":       "ALB",
    "carlos sainz":          "SAI",
    "nico hulkenberg":       "HUL",
    "gabriel bortoleto":     "BOR",
    "yuki tsunoda":          "TSU",
    "isack hadjar":          "HAD",
    "oliver bearman":        "BEA",
    "esteban ocon":          "OCO",
    # Previous seasons (for historical data if ever fetched)
    "sergio perez":          "PER",
    "valtteri bottas":       "BOT",
    "zhou guanyu":           "ZHO",
    "kevin magnussen":       "MAG",
    "valterri bottas":       "BOT",   # common typo in data sources
    "daniel ricciardo":      "RIC",
    "nyck de vries":         "DEV",
    "logan sargeant":        "SAR",
    "franco colapinto":      "COL",
    "mick schumacher":       "MSC",
    "sebastian vettel":      "VET",
    "robert kubica":         "KUB",
    "nikita mazepin":        "MAZ",
    "nicholas latifi":       "LAT",
    "guanyu zhou":           "ZHO",
}


def _resolve_name(full_name: str, threshold: float = 0.72) -> str | None:
    """
    Map a full driver name to a 3-letter abbreviation.
    First tries exact lookup, then falls back to fuzzy match.
    """
    key = full_name.strip().lower()
    if key in DRIVER_NAME_TO_ABBR:
        return DRIVER_NAME_TO_ABBR[key]

    # Fuzzy fallback — handles minor spelling variations from different bookmakers
    best_score, best_abbr = 0.0, None
    for name, abbr in DRIVER_NAME_TO_ABBR.items():
        score = SequenceMatcher(None, key, name).ratio()
        if score > best_score:
            best_score, best_abbr = score, abbr

    if best_score >= threshold:
        logger.debug(f"Fuzzy matched '{full_name}' -> '{best_abbr}' (score={best_score:.2f})")
        return best_abbr

    logger.warning(f"Could not map odds driver name '{full_name}' to abbreviation (best={best_score:.2f})")
    return None


def fetch_f1_race_winner_odds(
    api_key: Optional[str] = None,
    regions: str = "eu",
    bookmakers: Optional[list[str]] = None,
) -> dict[str, float]:
    """
    Fetch average race winner decimal odds for the next F1 event.

    Parameters
    ----------
    api_key : str, optional
        The Odds API key. Falls back to ODDS_API_KEY env var.
    regions : str
        Comma-separated region filter: 'eu', 'uk', 'us', 'au'.
        'eu' gives access to Pinnacle, Betfair, Unibet, etc.
    bookmakers : list[str], optional
        If set, restrict to these bookmaker keys (e.g. ['pinnacle', 'betfair_ex_eu']).
        When None, averages across all available bookmakers.

    Returns
    -------
    dict[str, float]
        {driver_abbr: avg_decimal_odds} for the next race event.
        Empty dict if API key missing, no events found, or request fails.
    """
    key = api_key or os.getenv("ODDS_API_KEY", "")
    if not key:
        logger.info("ODDS_API_KEY not set — skipping bookmaker odds fetch.")
        return {}

    sport_key = _find_f1_sport_key(key)
    if sport_key is None:
        # Try candidates blindly as last resort
        for candidate in _F1_SPORT_CANDIDATES:
            logger.info(f"Trying sport key candidate: '{candidate}'")
            sport_key = candidate
            break

    params: dict = {
        "apiKey": key,
        "regions": regions,
        "markets": "h2h",
        "oddsFormat": "decimal",
    }
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)

    try:
        resp = requests.get(
            f"{_BASE}/sports/{sport_key}/odds/",
            params=params,
            timeout=10,
        )
    except requests.RequestException as exc:
        logger.warning(f"Odds API request failed: {exc}")
        return {}

    remaining = resp.headers.get("x-requests-remaining", "?")
    logger.info(f"Odds API: {resp.status_code} | requests remaining this month: {remaining}")

    if resp.status_code == 401:
        logger.error("Odds API: invalid or missing API key.")
        return {}
    if resp.status_code == 404:
        logger.warning(f"Odds API: sport key '{sport_key}' not found or no upcoming events.")
        return {}
    if resp.status_code == 422:
        logger.warning("Odds API: sport not available or no upcoming events.")
        return {}
    if not resp.ok:
        logger.warning(f"Odds API: unexpected status {resp.status_code} — {resp.text[:200]}")
        return {}

    events = resp.json()
    if not events:
        logger.info("Odds API: no upcoming F1 events found.")
        return {}

    # Use the first (nearest) event
    event = events[0]
    logger.info(
        f"Odds API: found event '{event.get('sport_title', '')}' "
        f"id={event.get('id', '')} commence={event.get('commence_time', '')}"
    )

    # Collect all odds per driver across bookmakers
    driver_odds: dict[str, list[float]] = {}
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                name = outcome.get("name", "")
                price = outcome.get("price")
                if not name or price is None:
                    continue
                abbr = _resolve_name(name)
                if abbr is None:
                    continue
                driver_odds.setdefault(abbr, []).append(float(price))

    if not driver_odds:
        logger.warning("Odds API: no driver outcomes parsed from event data.")
        return {}

    # Average across bookmakers
    avg_odds = {abbr: round(sum(prices) / len(prices), 3) for abbr, prices in driver_odds.items()}
    logger.info(f"Odds API: parsed winner odds for {len(avg_odds)} drivers from {len(event.get('bookmakers', []))} bookmakers.")
    return avg_odds
