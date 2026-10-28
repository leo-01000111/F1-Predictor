"""
Free-practice laps from OpenF1 (free, 2023 onwards), for long-run pace features.

FastF1's lap data now needs an F1 TV login; OpenF1 publishes the same timing
for free from 2023. For each practice session we store every lap with its
stint, compound and tyre age, so the pace features (src/features/practice_pace.py)
can be computed and changed offline.

Output: data/raw/practice_laps.parquet
  year, round, session (FP1/FP2/FP3), driver, lap_number, lap_duration,
  is_pit_out_lap, stint_number, compound, tyre_age, date_start

OpenF1 rate-limits bursts (429 after a few quick requests), so requests are
paced at ~1 per second with back-off.

Usage:
  python -m src.data_collection.practice_openf1                 # 2023 to now, merge
  python -m src.data_collection.practice_openf1 --years 2026 --round 19
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from loguru import logger

ROOT = Path(__file__).parent.parent.parent
OUT_PATH = ROOT / "data" / "raw" / "practice_laps.parquet"
BASE = "https://api.openf1.org/v1"
SESSIONS = {"Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3"}
FIRST_YEAR = 2023
PACE_S = 1.0


def _get(endpoint: str, params: dict) -> list[dict]:
    for attempt in range(6):
        time.sleep(PACE_S)
        resp = requests.get(f"{BASE}/{endpoint}", params=params, timeout=60)
        if resp.status_code == 429:
            time.sleep(5 * 2 ** attempt)
            continue
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"OpenF1 {endpoint} {params}: still rate-limited")


def _rounds(year: int) -> dict[int, datetime]:
    """Round -> race start (UTC) from the FastF1 calendar."""
    import fastf1
    from src.data_collection import f1_historical  # noqa: F401  (enables the FastF1 cache)

    sched = fastf1.get_event_schedule(year, include_testing=False)
    out = {}
    for _, ev in sched.iterrows():
        for i in range(1, 6):
            if str(ev.get(f"Session{i}")) == "Race" and str(ev.get(f"Session{i}DateUtc")) != "NaT":
                out[int(ev["RoundNumber"])] = ev[f"Session{i}DateUtc"].to_pydatetime().replace(tzinfo=timezone.utc)
    return out


def session_laps(session: dict, year: int, rnd: int) -> pd.DataFrame:
    """Every lap of one practice session with its stint info."""
    sk = session["session_key"]
    laps = pd.DataFrame(_get("laps", {"session_key": sk}))
    if laps.empty:
        return laps
    stints = pd.DataFrame(_get("stints", {"session_key": sk}))
    drivers = {int(d["driver_number"]): d.get("name_acronym", "") for d in _get("drivers", {"session_key": sk})
               if d.get("driver_number") is not None}
    laps["stint_number"] = pd.NA
    laps["compound"] = None
    laps["tyre_age"] = pd.NA
    for _, s in stints.iterrows():
        if pd.isna(s.get("lap_start")) or pd.isna(s.get("lap_end")):
            continue
        m = (laps["driver_number"] == s["driver_number"]) & laps["lap_number"].between(s["lap_start"], s["lap_end"])
        laps.loc[m, "stint_number"] = s["stint_number"]
        laps.loc[m, "compound"] = s.get("compound")
        age0 = s.get("tyre_age_at_start")
        if pd.notna(age0):
            laps.loc[m, "tyre_age"] = age0 + laps.loc[m, "lap_number"] - s["lap_start"]
    laps["driver"] = laps["driver_number"].map(lambda n: drivers.get(int(n), str(n)))
    laps["year"], laps["round"], laps["session"] = year, rnd, SESSIONS[session["session_name"]]
    cols = ["year", "round", "session", "driver", "lap_number", "lap_duration", "is_pit_out_lap",
            "stint_number", "compound", "tyre_age", "date_start"]
    out = laps[cols].copy()
    for c in ("lap_number", "stint_number", "tyre_age"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["lap_duration"] = pd.to_numeric(out["lap_duration"], errors="coerce")
    out["is_pit_out_lap"] = out["is_pit_out_lap"].fillna(False).astype(bool)
    return out


def collect(years: list[int], only_round: Optional[int] = None,
            existing: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Collect practice laps; (year, round, session) already in `existing` are skipped."""
    have = set()
    if existing is not None and not existing.empty:
        have = set(zip(existing["year"], existing["round"], existing["session"]))
    frames = [existing] if existing is not None else []
    for year in years:
        rounds = _rounds(year)
        sessions = [s for s in _get("sessions", {"year": year}) if s.get("session_name") in SESSIONS]
        for s in sessions:
            start = datetime.fromisoformat(s["date_start"])
            # The practice belongs to the round whose race is 0-4 days later
            rnd = next((r for r, t in rounds.items() if 0 <= (t - start).total_seconds() / 86400 <= 4), None)
            if rnd is None or (only_round is not None and rnd != only_round):
                continue
            key = (year, rnd, SESSIONS[s["session_name"]])
            if key in have or start > datetime.now(timezone.utc):
                continue
            try:
                df = session_laps(s, year, rnd)
            except Exception as exc:
                logger.warning(f"{year} R{rnd} {key[2]}: {exc}")
                continue
            logger.info(f"{year} R{rnd:02d} {key[2]}: {len(df)} laps")
            if not df.empty:
                frames.append(df)
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_practice_laps() -> pd.DataFrame:
    if not OUT_PATH.exists():
        raise FileNotFoundError(f"{OUT_PATH} missing: python -m src.data_collection.practice_openf1")
    return pd.read_parquet(OUT_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect OpenF1 practice laps")
    parser.add_argument("--years", type=int, nargs="*",
                        default=list(range(FIRST_YEAR, datetime.now(timezone.utc).year + 1)))
    parser.add_argument("--round", type=int, default=None)
    args = parser.parse_args()
    existing = pd.read_parquet(OUT_PATH) if OUT_PATH.exists() else None
    out = collect(args.years, args.round, existing)
    if out.empty:
        logger.info("No practice laps collected")
        return
    out = out.drop_duplicates(["year", "round", "session", "driver", "lap_number"], keep="last")
    out = out.sort_values(["year", "round", "session", "driver", "lap_number"]).reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    logger.info(f"Saved {len(out)} laps -> {OUT_PATH}")


if __name__ == "__main__":
    main()
