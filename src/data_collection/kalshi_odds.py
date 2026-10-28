"""
Pre-race market prices from Kalshi (ROADMAP Phase 4).

Kalshi is a regulated prediction exchange with free, unauthenticated market-data
endpoints. F1 coverage starts in 2025:
  KXF1RACE        race winner, one binary market per driver
  KXF1RACEPODIUM  podium finisher, one binary market per driver

For each race we read every driver's price at two moments:
  price_pred  when our model predicts (qualifying start + 2h45, as ci_tick)
  price_start at the race start
Price = midpoint of the best bid and ask at the end of the last hourly candle
before that moment; the last traded price if the book is empty or very wide;
missing if neither exists (price_from_quote). Raw bid/ask/last are stored too.
Prices are not normalised: de-vig in the analysis.

Output: data/raw/kalshi_odds.parquet
  year, round, market (win|podium), driver, name, bid/ask/last/price _pred and
  _start, result
Usage:
  python -m src.data_collection.kalshi_odds            # all settled + open races
"""
from __future__ import annotations

import time
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import fastf1
import pandas as pd
import requests
from loguru import logger

ROOT = Path(__file__).parent.parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = RAW_DIR / "kalshi_odds.parquet"
BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = {"win": "KXF1RACE", "podium": "KXF1RACEPODIUM"}
PRED_DELAY = timedelta(hours=2, minutes=45)   # quali start + ~1 h session + ci_tick's 1h45
MAX_SPREAD = 0.15


def _get(path: str, params: Optional[dict] = None) -> dict:
    for attempt in range(5):
        r = requests.get(BASE + path, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Kalshi rate limit on {path}")


def _paged(path: str, key: str, params: dict) -> list[dict]:
    out, cursor = [], None
    while True:
        p = dict(params, limit=200)
        if cursor:
            p["cursor"] = cursor
        d = _get(path, p)
        out += d.get(key, [])
        cursor = d.get("cursor")
        if not cursor:
            return out


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    words = s.replace("-", " ").replace(".", " ").split()
    return " ".join(w for w in words if w not in ("jr", "sr"))


def _driver_codes(year: int) -> dict[str, str]:
    """Full name (and family name) -> three-letter code, from Jolpica."""
    d = requests.get(f"https://api.jolpi.ca/ergast/f1/{year}/drivers.json?limit=100", timeout=30).json()
    out: dict[str, str] = {}
    for x in d["MRData"]["DriverTable"]["Drivers"]:
        code = x.get("code") or x["familyName"][:3].upper()
        out[_norm(f"{x['givenName']} {x['familyName']}")] = code
        out.setdefault(_norm(x["familyName"]), code)
    return out


def _code(name: str, codes: dict[str, str]) -> Optional[str]:
    n = _norm(name)
    if n in codes:
        return codes[n]
    last = n.split()[-1] if n else ""
    return codes.get(last)


def _schedule(year: int) -> pd.DataFrame:
    s = fastf1.get_event_schedule(year, include_testing=False)
    return pd.DataFrame({
        "round": s["RoundNumber"].astype(int),
        "quali_utc": pd.to_datetime(s["Session4DateUtc"], utc=True),
        "race_utc": pd.to_datetime(s["Session5DateUtc"], utc=True),
    })


def _match_round(occurrence: datetime, sched: pd.DataFrame) -> Optional[int]:
    """Round whose race start is within 36 h of the market's occurrence time."""
    diff = (sched["race_utc"] - occurrence).abs()
    i = diff.idxmin()
    return int(sched.loc[i, "round"]) if diff[i] <= timedelta(hours=36) else None


def price_from_quote(bid: Optional[float], ask: Optional[float], last: Optional[float]) -> Optional[float]:
    """Mid of a usable book; else the last trade; else None.

    An empty book (bid 0, ask ~1) gives no information: its midpoint (~0.5)
    must never be used as a price.
    """
    if bid is not None and ask is not None and bid > 0 and 0 < ask - bid <= MAX_SPREAD:
        return (bid + ask) / 2
    if last is not None and last > 0:
        return last
    return None


def _quote_at(series: str, ticker: str, t: datetime, historical: bool) -> dict:
    """Best bid, best ask and last trade at the end of the last hourly candle before t.
    Archived (settled, older) markets live under /historical, whose candles name
    fields "close" instead of "close_dollars"."""
    end = int(t.timestamp())
    path = (f"/historical/markets/{ticker}/candlesticks" if historical
            else f"/series/{series}/markets/{ticker}/candlesticks")
    d = _get(path, {"start_ts": end - 4 * 86400, "end_ts": end, "period_interval": 60})
    candles = [c for c in d.get("candlesticks", []) if c["end_period_ts"] <= end]
    if not candles:
        return {"bid": None, "ask": None, "last": None}
    c = candles[-1]

    def f(block: str, field: str) -> Optional[float]:
        b = c.get(block) or {}
        v = b.get(f"{field}_dollars", b.get(field))
        return float(v) if v not in (None, "") else None

    return {"bid": f("yes_bid", "close"), "ask": f("yes_ask", "close"),
            "last": f("price", "close") or f("price", "previous")}


def collect(existing: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """All Kalshi F1 race events. Races already in `existing` with a result are skipped."""
    done, done_events = set(), set()
    if existing is not None and not existing.empty:
        settled = existing[existing["result"].isin(["yes", "no"])]
        done = set(zip(settled["year"], settled["round"], settled["market"]))
        # events whose every market is settled: skip before any request
        all_settled = existing.groupby("event")["result"].agg(lambda r: r.isin(["yes", "no"]).all())
        done_events = set(all_settled[all_settled].index)
    schedules: dict[int, pd.DataFrame] = {}
    codes: dict[int, dict[str, str]] = {}
    rows = []
    for market, series in SERIES.items():
        for ev in _paged("/events", "events", {"series_ticker": series}):
            if ev["event_ticker"] in done_events:
                continue
            markets = _paged("/markets", "markets", {"event_ticker": ev["event_ticker"]})
            historical = not markets
            if historical:
                markets = _paged("/historical/markets", "markets", {"event_ticker": ev["event_ticker"]})
            if not markets:
                continue
            occ_s = markets[0].get("occurrence_datetime") or markets[0].get("expected_expiration_time")
            occ = datetime.fromisoformat(occ_s.replace("Z", "+00:00"))
            year = occ.year if occ.month > 1 else occ.year - 1
            if year not in schedules:
                schedules[year] = _schedule(year)
                codes[year] = _driver_codes(year)
            rnd = _match_round(occ, schedules[year])
            if rnd is None or (year, rnd, market) in done:
                continue
            s = schedules[year].set_index("round").loc[rnd]
            t_pred = s["quali_utc"].to_pydatetime() + PRED_DELAY
            t_start = s["race_utc"].to_pydatetime()
            now = datetime.now(timezone.utc)
            empty = {"bid": None, "ask": None, "last": None}
            for m in markets:
                name = m.get("yes_sub_title") or m.get("no_sub_title") or ""
                qp = _quote_at(series, m["ticker"], t_pred, historical) if t_pred < now else empty
                qs = _quote_at(series, m["ticker"], t_start, historical) if t_start < now else empty
                rows.append({
                    "year": year, "round": rnd, "market": market, "event": ev["event_ticker"],
                    "ticker": m["ticker"], "name": name, "driver": _code(name, codes[year]),
                    "bid_pred": qp["bid"], "ask_pred": qp["ask"], "last_pred": qp["last"],
                    "price_pred": price_from_quote(**qp),
                    "bid_start": qs["bid"], "ask_start": qs["ask"], "last_start": qs["last"],
                    "price_start": price_from_quote(**qs),
                    "result": m.get("result") or "",
                })
            logger.info(f"{market:6s} {year} R{rnd:<2d} {ev['event_ticker']}: {len(markets)} drivers")
    new = pd.DataFrame(rows)
    if existing is None or existing.empty:
        return new
    if new.empty:
        return existing
    keys = set(zip(new["year"], new["round"], new["market"]))
    keep = existing[[k not in keys for k in zip(existing["year"], existing["round"], existing["market"])]]
    return pd.concat([keep, new], ignore_index=True)


def load_kalshi_odds() -> pd.DataFrame:
    return pd.read_parquet(OUT_PATH)


if __name__ == "__main__":
    existing = pd.read_parquet(OUT_PATH) if OUT_PATH.exists() else None
    df = collect(existing).sort_values(["year", "round", "market", "driver"]).reset_index(drop=True)
    df.to_parquet(OUT_PATH, index=False)
    unmatched = df[df["driver"].isna()]["name"].unique()
    logger.info(f"Saved {len(df)} rows ({df.groupby(['year','round']).ngroups} races) -> {OUT_PATH}")
    if len(unmatched):
        logger.warning(f"Unmatched driver names: {list(unmatched)}")
