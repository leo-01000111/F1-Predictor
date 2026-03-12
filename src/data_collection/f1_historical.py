"""
FastF1-based historical data collector.
Pulls Race, Qualifying, and Free Practice sessions for all rounds.
Outputs:
  - data/raw/race_results.parquet
  - data/raw/quali_results.parquet
  - data/raw/practice_results.parquet
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import fastf1
import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = ROOT / "data" / "cache"
RAW_DIR = ROOT / "data" / "raw"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))

LONG_RUN_MIN_LAPS = 5


def _load_session_safe(
    year: int,
    round_num: int,
    session_type: str,
    load_laps: bool = False,
) -> Optional[fastf1.core.Session]:
    try:
        session = fastf1.get_session(year, round_num, session_type)
        session.load(laps=load_laps, telemetry=False, weather=False, messages=False)
        return session
    except Exception as exc:
        logger.warning(f"Failed to load {year} R{round_num} {session_type}: {exc}")
        return None


def _extract_race_result(session: fastf1.core.Session, year: int) -> pd.DataFrame:
    results = session.results
    if results is None or results.empty:
        return pd.DataFrame()

    rows = []
    for _, row in results.iterrows():
        position = row.get("Position", None)
        status = row.get("Status", "")
        dnf = int(
            status not in ("Finished", "+1 Lap", "+2 Laps", "+3 Laps", "+4 Laps", "+5 Laps")
            and status != ""
            and not str(status).startswith("+")
        )
        grid = row.get("GridPosition", None)
        points = row.get("Points", 0.0)
        rows.append({
            "year": year,
            "round": session.event.RoundNumber,
            "circuit": session.event.Location,
            "circuit_key": session.event.EventName.lower().replace(" ", "_").replace("grand_prix", "gp"),
            "race_date": str(session.event.EventDate.date()),
            "driver": row.get("Abbreviation", ""),
            "team": row.get("TeamName", ""),
            "grid_position": int(grid) if pd.notna(grid) else None,
            "finish_position": int(position) if pd.notna(position) else None,
            "dnf": dnf,
            "status": status,
            "points": float(points) if pd.notna(points) else 0.0,
            "p1": int(position == 1) if pd.notna(position) else 0,
            "p2": int(position == 2) if pd.notna(position) else 0,
            "p3": int(position == 3) if pd.notna(position) else 0,
            "podium": int(position in [1, 2, 3]) if pd.notna(position) else 0,
        })
    return pd.DataFrame(rows)


def _extract_quali_result(session: fastf1.core.Session, year: int) -> pd.DataFrame:
    results = session.results
    if results is None or results.empty:
        return pd.DataFrame()

    q_times = []
    for _, row in results.iterrows():
        best = None
        for col in ["Q3", "Q2", "Q1"]:
            t = row.get(col)
            if t is not None and pd.notna(t):
                best = t.total_seconds() if hasattr(t, "total_seconds") else float(t)
                break
        q_times.append(best)

    pole_time = min((t for t in q_times if t is not None), default=None)

    rows = []
    for idx, (_, row) in enumerate(results.iterrows()):
        best_time = q_times[idx]
        gap_to_pole = (best_time - pole_time) if (best_time is not None and pole_time is not None) else None
        rows.append({
            "year": year,
            "round": session.event.RoundNumber,
            "circuit": session.event.Location,
            "circuit_key": session.event.EventName.lower().replace(" ", "_").replace("grand_prix", "gp"),
            "driver": row.get("Abbreviation", ""),
            "team": row.get("TeamName", ""),
            "quali_position": int(row.get("Position")) if pd.notna(row.get("Position")) else None,
            "quali_best_time_s": best_time,
            "quali_gap_to_pole_s": gap_to_pole,
        })
    return pd.DataFrame(rows)


def _extract_practice_result(session: fastf1.core.Session, year: int, fp_num: int) -> pd.DataFrame:
    """
    Extract per-driver FP pace metrics.
    fp{N}_best_gap_s     : best lap gap to session fastest
    fp{N}_long_run_delta : avg long-run pace vs session median (race sim)
    fp{N}_laps           : total laps completed
    """
    try:
        laps = session.laps
    except Exception:
        return pd.DataFrame()

    if laps is None or laps.empty:
        return pd.DataFrame()

    clean_laps = laps.pick_quicklaps(threshold=1.07)

    session_best = None
    session_median = None
    if not clean_laps.empty and "LapTime" in clean_laps.columns:
        times = clean_laps["LapTime"].dropna()
        if not times.empty:
            session_best = times.min().total_seconds()
            session_median = times.median().total_seconds()

    rows = []
    for driver in laps["Driver"].unique():
        driver_laps = laps[laps["Driver"] == driver]
        driver_clean = clean_laps[clean_laps["Driver"] == driver]
        total_laps = len(driver_laps)

        best_gap = None
        if not driver_clean.empty and session_best is not None:
            lap_times = driver_clean["LapTime"].dropna()
            if not lap_times.empty:
                best_gap = lap_times.min().total_seconds() - session_best

        long_run_pace = None
        if not driver_laps.empty and "Stint" in driver_laps.columns and session_median is not None:
            stint_sizes = driver_laps.groupby("Stint").size()
            long_stints = stint_sizes[stint_sizes >= LONG_RUN_MIN_LAPS].index
            if len(long_stints) > 0:
                long_run_laps = driver_laps[driver_laps["Stint"].isin(long_stints)]
                long_run_times = long_run_laps["LapTime"].dropna()
                if not long_run_times.empty:
                    avg_s = long_run_times.apply(lambda t: t.total_seconds()).mean()
                    long_run_pace = avg_s - session_median

        rows.append({
            "year": year,
            "round": session.event.RoundNumber,
            "circuit_key": session.event.EventName.lower().replace(" ", "_").replace("grand_prix", "gp"),
            "driver": driver,
            f"fp{fp_num}_best_gap_s": best_gap,
            f"fp{fp_num}_long_run_delta": long_run_pace,
            f"fp{fp_num}_laps": total_laps,
        })
    return pd.DataFrame(rows)


def _extract_lap_extras(session: fastf1.core.Session, year: int) -> pd.DataFrame:
    """Extract lap-1 position and start compound (tire) for each driver.

    lap1_position  : position at the end of lap 1 (start performance signal)
    start_compound : tire compound used at race start (strategy signal)

    Requires the session to be loaded with laps=True.
    """
    try:
        laps = session.laps
    except Exception:
        return pd.DataFrame()

    if laps is None or laps.empty:
        return pd.DataFrame()

    lap1 = laps[pd.to_numeric(laps["LapNumber"], errors="coerce") == 1]
    if lap1.empty:
        return pd.DataFrame()

    circuit_key = (
        session.event.EventName.lower()
        .replace(" ", "_")
        .replace("grand_prix", "gp")
    )

    rows = []
    for _, row in lap1.iterrows():
        driver = str(row.get("Driver", "")).strip()
        if not driver:
            continue
        pos = row.get("Position", None)
        compound = row.get("Compound", None)
        rows.append({
            "year": year,
            "round": int(session.event.RoundNumber),
            "circuit_key": circuit_key,
            "driver": driver,
            "lap1_position": int(float(pos)) if pd.notna(pos) else None,
            "start_compound": str(compound).upper() if compound and pd.notna(compound) and str(compound) != "nan" else None,
        })

    return pd.DataFrame(rows)


def collect_lap_extras(
    start_year: int = 2014,
    end_year: int = 2024,
    sleep_between_sessions: float = 2.0,
    merge_existing: bool = False,
) -> pd.DataFrame:
    """Collect lap-1 position and start compound for all historical races.

    Loads race sessions with laps=True (heavier than R+Q-only; ~2-4s per round).
    Saves incrementally to data/raw/lap_extras.parquet after each year.

    Args:
        merge_existing: if True, skip rounds already in the parquet and only
                        fetch the newly requested years.
    """
    lap_extras_path = RAW_DIR / "lap_extras.parquet"
    existing = pd.DataFrame()

    if merge_existing and lap_extras_path.exists():
        existing = pd.read_parquet(lap_extras_path)
        logger.info(
            f"Loaded existing lap_extras: {len(existing)} rows, "
            f"years {sorted(existing['year'].unique().tolist())}"
        )

    existing_keys: set[tuple[int, int]] = set()
    if not existing.empty:
        existing_keys = set(zip(existing["year"].tolist(), existing["round"].tolist()))

    all_rows: list[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as exc:
            logger.error(f"Could not fetch schedule for {year}: {exc}")
            continue

        rounds = schedule["RoundNumber"].dropna().astype(int).tolist()
        logger.info(f"{year}: collecting lap extras for {len(rounds)} rounds")

        for rnd in rounds:
            if (year, rnd) in existing_keys:
                logger.debug(f"  {year} R{rnd}: already in lap_extras, skipping")
                continue

            race_session = _load_session_safe(year, rnd, "R", load_laps=True)
            if race_session is not None:
                df = _extract_lap_extras(race_session, year)
                if not df.empty:
                    all_rows.append(df)
                    logger.debug(f"  {year} R{rnd}: {len(df)} drivers collected")
                else:
                    logger.warning(f"  {year} R{rnd}: no lap-1 data found")
            time.sleep(sleep_between_sessions)

        # Save after each year so progress is never lost
        if all_rows:
            new_df = pd.concat(all_rows, ignore_index=True)
            combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(
                subset=["year", "round", "driver"], keep="last"
            )
            combined.to_parquet(lap_extras_path, index=False)
            logger.info(f"  Saved lap_extras progress after {year} ({len(combined)} total rows)")

    if not all_rows:
        if not existing.empty:
            return existing
        return pd.DataFrame()

    new_df = pd.concat(all_rows, ignore_index=True)
    combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(
        subset=["year", "round", "driver"], keep="last"
    )
    combined = combined.sort_values(["year", "round", "driver"]).reset_index(drop=True)
    combined.to_parquet(lap_extras_path, index=False)
    logger.info(f"Lap extras collection complete: {len(combined)} rows saved to {lap_extras_path}")
    return combined


def _parse_pit_duration(dur_str: str) -> float | None:
    """Parse an Ergast/Jolpica pit stop duration string to seconds."""
    try:
        s = str(dur_str).strip()
        if ":" in s:
            parts = s.split(":")
            return float(parts[0]) * 60 + float(parts[1])
        return float(s)
    except (ValueError, AttributeError):
        return None


def _jolpica_get(url: str, sleep_between: float, max_retries: int = 6) -> dict:
    """
    GET a Jolpica/Ergast URL with exponential back-off on 429 rate-limit responses.

    Raises on non-retryable HTTP errors. Waits up to ~4 minutes total before giving up.
    """
    import requests

    backoff = max(sleep_between, 2.0)  # initial back-off on 429
    for attempt in range(max_retries):
        resp = requests.get(url, timeout=15)
        if resp.status_code == 429:
            wait = backoff * (2 ** attempt)
            logger.warning(f"  429 rate-limited on {url} — waiting {wait:.0f}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        time.sleep(sleep_between)
        return resp.json()
    raise RuntimeError(f"Exceeded {max_retries} retries on {url}")


def collect_pit_data(
    start_year: int = 2014,
    end_year: int = 2025,
    sleep_between: float = 1.5,
    merge_existing: bool = False,
) -> pd.DataFrame:
    """
    Fetch pit stop durations via the Jolpica (Ergast-compatible) API.

    Stores per-driver average pit duration per race (year, round, driver, avg_pit_s)
    so that team-level deltas can be computed at feature-build time.

    Endpoint: https://api.jolpi.ca/ergast/f1/{year}/{round}/pitstops.json
    Driver mapping: https://api.jolpi.ca/ergast/f1/{year}/{round}/results.json

    Rate limit: Jolpica allows ~4 req/s burst, ~500 req/min sustained.
    sleep_between=1.5 is conservative (~0.67 req/s) to stay safely under limits.
    On 429, _jolpica_get() backs off exponentially up to ~4 min.
    """
    pit_path = RAW_DIR / "pit_stops.parquet"
    existing = pd.DataFrame()

    if merge_existing and pit_path.exists():
        existing = pd.read_parquet(pit_path)
        logger.info(f"Loaded existing pit data: {len(existing)} rows")

    existing_keys: set[tuple[int, int]] = set()
    if not existing.empty:
        existing_keys = set(zip(existing["year"].tolist(), existing["round"].tolist()))

    BASE = "https://api.jolpi.ca/ergast/f1"
    new_rows: list[dict] = []

    for year in range(start_year, end_year + 1):
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as exc:
            logger.error(f"Could not fetch schedule for {year}: {exc}")
            continue

        rounds = schedule["RoundNumber"].dropna().astype(int).tolist()
        logger.info(f"{year}: collecting pit data for {len(rounds)} rounds")

        for rnd in rounds:
            if (year, rnd) in existing_keys:
                logger.debug(f"  {year} R{rnd}: already collected, skipping")
                continue

            try:
                # Get driver code -> constructor mapping from results
                r_url = f"{BASE}/{year}/{rnd}/results.json?limit=30"
                r_data = _jolpica_get(r_url, sleep_between)
                r_races = r_data.get("MRData", {}).get("RaceTable", {}).get("Races", [])

                if not r_races:
                    continue

                # driver_code (e.g. "VER") keyed by Ergast driverId
                code_map: dict[str, str] = {}
                for res in r_races[0].get("Results", []):
                    did = res.get("Driver", {}).get("driverId", "")
                    code = res.get("Driver", {}).get("code", did.upper()[:3])
                    code_map[did] = code

                # Fetch pit stops
                p_url = f"{BASE}/{year}/{rnd}/pitstops.json?limit=200"
                p_data = _jolpica_get(p_url, sleep_between)
                p_races = p_data.get("MRData", {}).get("RaceTable", {}).get("Races", [])

                if not p_races:
                    continue

                # Aggregate per driver: median pit duration
                driver_pits: dict[str, list[float]] = {}
                for pit in p_races[0].get("PitStops", []):
                    did = pit.get("driverId", "")
                    code = code_map.get(did, did.upper()[:3])
                    dur = _parse_pit_duration(pit.get("duration", ""))
                    if dur is None or dur < 1.5 or dur > 120:
                        continue
                    driver_pits.setdefault(code, []).append(dur)

                for drv, times in driver_pits.items():
                    new_rows.append({
                        "year": year,
                        "round": rnd,
                        "driver": drv,
                        "avg_pit_s": round(float(np.median(times)), 3),
                        "n_stops": len(times),
                    })

                logger.info(f"  {year} R{rnd}: {len(driver_pits)} drivers collected")

            except Exception as exc:
                logger.warning(f"  {year} R{rnd}: pit fetch failed: {exc}")

        # Save after each year
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(
                subset=["year", "round", "driver"], keep="last"
            )
            combined.to_parquet(pit_path, index=False)
            logger.info(f"  Pit data saved after {year} ({len(combined)} total rows)")

    if not new_rows:
        return existing

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(
        subset=["year", "round", "driver"], keep="last"
    )
    combined = combined.sort_values(["year", "round", "driver"]).reset_index(drop=True)
    combined.to_parquet(pit_path, index=False)
    logger.info(f"Pit data collection complete: {len(combined)} rows -> {pit_path}")
    return combined


def load_pit_stops() -> pd.DataFrame:
    """Load the pit_stops parquet (avg_pit_s per driver per race)."""
    path = RAW_DIR / "pit_stops.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "No pit stop data found. Run:\n"
            "  python scripts/backfill_pit_sprint.py --pit --start 2014 --end 2025 --merge"
        )
    return pd.read_parquet(path)


def _extract_sprint_result(session: fastf1.core.Session, year: int) -> pd.DataFrame:
    """Extract sprint race finishing positions."""
    results = session.results
    if results is None or results.empty:
        return pd.DataFrame()

    n_starters = len(results)
    rows = []
    for _, row in results.iterrows():
        position = row.get("Position", None)
        rows.append({
            "year": year,
            "round": int(session.event.RoundNumber),
            "circuit_key": session.event.EventName.lower().replace(" ", "_").replace("grand_prix", "gp"),
            "driver": row.get("Abbreviation", ""),
            "sprint_position": int(float(position)) if pd.notna(position) else None,
            "n_sprint_starters": n_starters,
        })
    return pd.DataFrame(rows)


def collect_sprint_results(
    start_year: int = 2021,
    end_year: int = 2025,
    sleep_between_sessions: float = 1.5,
    merge_existing: bool = False,
) -> pd.DataFrame:
    """
    Collect sprint race results via FastF1 (sprints started in 2021).

    Saves to data/raw/sprint_results.parquet with columns:
        year, round, circuit_key, driver, sprint_position, n_sprint_starters
    """
    sprint_path = RAW_DIR / "sprint_results.parquet"
    existing = pd.DataFrame()

    if merge_existing and sprint_path.exists():
        existing = pd.read_parquet(sprint_path)
        logger.info(f"Loaded existing sprint data: {len(existing)} rows")

    existing_keys: set[tuple[int, int]] = set()
    if not existing.empty:
        existing_keys = set(zip(existing["year"].tolist(), existing["round"].tolist()))

    new_rows: list[pd.DataFrame] = []

    for year in range(max(start_year, 2021), end_year + 1):
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as exc:
            logger.error(f"Could not fetch schedule for {year}: {exc}")
            continue

        # Only sprint weekends
        sprint_rounds = []
        for _, ev in schedule.iterrows():
            rnd = ev.get("RoundNumber")
            fmt = str(ev.get("EventFormat", "")).lower()
            if pd.notna(rnd) and "sprint" in fmt:
                sprint_rounds.append(int(rnd))

        logger.info(f"{year}: {len(sprint_rounds)} sprint rounds detected")

        for rnd in sprint_rounds:
            if (year, rnd) in existing_keys:
                logger.debug(f"  {year} R{rnd}: sprint already collected, skipping")
                continue

            # Try both FastF1 sprint session identifiers
            session = None
            for stype in ["Sprint", "S"]:
                session = _load_session_safe(year, rnd, stype)
                if session is not None:
                    break

            if session is None:
                logger.warning(f"  {year} R{rnd}: could not load sprint session")
                time.sleep(sleep_between_sessions)
                continue

            df = _extract_sprint_result(session, year)
            if not df.empty:
                new_rows.append(df)
                logger.debug(f"  {year} R{rnd}: {len(df)} sprint results collected")
            time.sleep(sleep_between_sessions)

        # Save after each year
        if new_rows:
            new_df = pd.concat(new_rows, ignore_index=True)
            combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(
                subset=["year", "round", "driver"], keep="last"
            )
            combined.to_parquet(sprint_path, index=False)
            logger.info(f"  Sprint data saved after {year} ({len(combined)} total rows)")

    if not new_rows:
        return existing

    new_df = pd.concat(new_rows, ignore_index=True)
    combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(
        subset=["year", "round", "driver"], keep="last"
    )
    combined = combined.sort_values(["year", "round", "driver"]).reset_index(drop=True)
    combined.to_parquet(sprint_path, index=False)
    logger.info(f"Sprint collection complete: {len(combined)} rows -> {sprint_path}")
    return combined


def load_sprint_results() -> pd.DataFrame:
    """Load sprint_results.parquet (sprint position per driver per sprint round)."""
    path = RAW_DIR / "sprint_results.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "No sprint data found. Run:\n"
            "  python scripts/backfill_pit_sprint.py --sprint --start 2021 --end 2025 --merge"
        )
    return pd.read_parquet(path)


def load_lap_extras() -> pd.DataFrame:
    """Load the lap_extras parquet (lap1_position + start_compound per driver per race)."""
    path = RAW_DIR / "lap_extras.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No lap_extras data found. Run the backfill first:\n"
            f"  python scripts/backfill_lap_extras.py --start 2014 --end 2025 --merge"
        )
    return pd.read_parquet(path)


def collect_historical(
    start_year: int = 2014,
    end_year: int = 2024,
    sleep_between_sessions: float = 1.0,
    include_fp: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pull Race, Qualifying, and (optionally) FP1/FP2/FP3 for all rounds.
    Returns (race_df, quali_df, practice_df).

    Set include_fp=False for a fast R+Q-only collection (~15 min vs ~90 min).
    FP features will be median-imputed during feature engineering if absent.
    """
    all_race_rows: list[pd.DataFrame] = []
    all_quali_rows: list[pd.DataFrame] = []
    all_practice_rows: list[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as exc:
            logger.error(f"Could not fetch schedule for {year}: {exc}")
            continue

        rounds = schedule["RoundNumber"].dropna().astype(int).tolist()
        logger.info(f"{year}: {len(rounds)} rounds to collect (fp={'yes' if include_fp else 'no'})")

        for rnd in rounds:
            race_session = _load_session_safe(year, rnd, "R")
            if race_session is not None:
                df = _extract_race_result(race_session, year)
                if not df.empty:
                    all_race_rows.append(df)
            time.sleep(sleep_between_sessions)

            quali_session = _load_session_safe(year, rnd, "Q")
            if quali_session is not None:
                df = _extract_quali_result(quali_session, year)
                if not df.empty:
                    all_quali_rows.append(df)
            time.sleep(sleep_between_sessions)

            if include_fp:
                fp_dfs: list[pd.DataFrame] = []
                for fp_num, fp_key in [(1, "FP1"), (2, "FP2"), (3, "FP3")]:
                    fp_session = _load_session_safe(year, rnd, fp_key, load_laps=True)
                    if fp_session is not None:
                        df = _extract_practice_result(fp_session, year, fp_num)
                        if not df.empty:
                            fp_dfs.append(df)
                    time.sleep(sleep_between_sessions)

                if fp_dfs:
                    fp_merged = fp_dfs[0]
                    for fp_df in fp_dfs[1:]:
                        merge_keys = ["year", "round", "circuit_key", "driver"]
                        fp_merged = fp_merged.merge(fp_df, on=merge_keys, how="outer")
                    all_practice_rows.append(fp_merged)
                    logger.debug(f"  {year} R{rnd} Practice: {len(fp_merged)} drivers")

        # Save after each year so progress is never lost on interrupt
        _save_incremental(all_race_rows, all_quali_rows, all_practice_rows)
        logger.info(f"  Saved progress after {year}")

    race_df = pd.concat(all_race_rows, ignore_index=True) if all_race_rows else pd.DataFrame()
    quali_df = pd.concat(all_quali_rows, ignore_index=True) if all_quali_rows else pd.DataFrame()
    practice_df = pd.concat(all_practice_rows, ignore_index=True) if all_practice_rows else pd.DataFrame()

    return race_df, quali_df, practice_df


def _save_incremental(
    race_rows: list,
    quali_rows: list,
    practice_rows: list,
) -> None:
    """Save current accumulated rows — called after each year to preserve progress."""
    if race_rows:
        pd.concat(race_rows, ignore_index=True).to_parquet(RAW_DIR / "race_results.parquet", index=False)
    if quali_rows:
        pd.concat(quali_rows, ignore_index=True).to_parquet(RAW_DIR / "quali_results.parquet", index=False)
    if practice_rows:
        pd.concat(practice_rows, ignore_index=True).to_parquet(RAW_DIR / "practice_results.parquet", index=False)


def save_raw(race_df: pd.DataFrame, quali_df: pd.DataFrame, practice_df: Optional[pd.DataFrame] = None) -> None:
    if not race_df.empty:
        path = RAW_DIR / "race_results.parquet"
        race_df.to_parquet(path, index=False)
        logger.info(f"Saved {len(race_df)} race rows -> {path}")
    if not quali_df.empty:
        path = RAW_DIR / "quali_results.parquet"
        quali_df.to_parquet(path, index=False)
        logger.info(f"Saved {len(quali_df)} quali rows -> {path}")
    if practice_df is not None and not practice_df.empty:
        path = RAW_DIR / "practice_results.parquet"
        practice_df.to_parquet(path, index=False)
        logger.info(f"Saved {len(practice_df)} practice rows -> {path}")


def load_race_results() -> pd.DataFrame:
    path = RAW_DIR / "race_results.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run collect_historical() first. Expected: {path}")
    return pd.read_parquet(path)


def load_quali_results() -> pd.DataFrame:
    path = RAW_DIR / "quali_results.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run collect_historical() first. Expected: {path}")
    return pd.read_parquet(path)


def load_practice_results() -> pd.DataFrame:
    path = RAW_DIR / "practice_results.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run collect_historical() first. Expected: {path}")
    return pd.read_parquet(path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect F1 historical data via FastF1")
    parser.add_argument("--start", type=int, default=2014)
    parser.add_argument("--end", type=int, default=2024)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--no-fp", action="store_true", help="Skip FP sessions (R+Q only, ~4x faster)")
    parser.add_argument(
        "--merge",
        action="store_true",
        help=(
            "Load existing parquet files and only collect years not already present, "
            "then merge and save. Avoids re-downloading cached data."
        ),
    )
    parser.add_argument(
        "--lap-extras",
        action="store_true",
        help=(
            "Collect lap-1 position and start compound for each race. "
            "Requires loading race laps (~2s per round). "
            "Outputs data/raw/lap_extras.parquet."
        ),
    )
    args = parser.parse_args()

    if args.lap_extras:
        collect_lap_extras(args.start, args.end, args.sleep, merge_existing=args.merge)
        import sys as _sys
        _sys.exit(0)

    if args.merge:
        # --merge: load existing parquet, collect only args.start..args.end years,
        # then merge new data into existing, deduplicating on year+round+driver.
        # Always collects the full args.start..args.end range so partial years
        # (e.g. 2020 with only 6/17 rounds) are properly completed.
        existing_race = pd.DataFrame()
        existing_quali = pd.DataFrame()
        existing_practice = pd.DataFrame()

        race_path = RAW_DIR / "race_results.parquet"
        quali_path = RAW_DIR / "quali_results.parquet"
        practice_path = RAW_DIR / "practice_results.parquet"

        if race_path.exists():
            existing_race = pd.read_parquet(race_path)
            logger.info(f"Loaded existing race data: {len(existing_race)} rows, years {sorted(existing_race['year'].unique().tolist())}")
        if quali_path.exists():
            existing_quali = pd.read_parquet(quali_path)
            logger.info(f"Loaded existing quali data: {len(existing_quali)} rows")
        if practice_path.exists():
            existing_practice = pd.read_parquet(practice_path)
            logger.info(f"Loaded existing practice data: {len(existing_practice)} rows")

        # Back up existing data — incremental saves inside collect_historical
        # overwrite the parquet with only the new years, so we need the backup
        # to restore the pre-existing years at the final merge step.
        backup_race_path = RAW_DIR / "_race_results_backup.parquet"
        backup_quali_path = RAW_DIR / "_quali_results_backup.parquet"
        if not existing_race.empty:
            existing_race.to_parquet(backup_race_path, index=False)
            logger.info(f"Backed up {len(existing_race)} race rows to {backup_race_path}")
        if not existing_quali.empty:
            existing_quali.to_parquet(backup_quali_path, index=False)

        logger.info(f"Collecting years {args.start}-{args.end} (fp={'no' if args.no_fp else 'yes'})")
        new_race, new_quali, new_practice = collect_historical(
            args.start, args.end, args.sleep, include_fp=not args.no_fp
        )

        # Merge existing (pre-start years) + new, deduplicate on year+round+driver
        def _merge_dfs(existing: pd.DataFrame, new: pd.DataFrame, key_cols: list) -> pd.DataFrame:
            if existing.empty:
                return new
            if new.empty:
                return existing
            # Drop existing rows that fall in the newly collected year range
            # so fresh data takes precedence for those years
            existing_outside = existing[~existing["year"].isin(range(args.start, args.end + 1))]
            combined = pd.concat([existing_outside, new], ignore_index=True)
            combined = combined.drop_duplicates(subset=key_cols, keep="last")
            return combined.sort_values(["year", "round"]).reset_index(drop=True)

        race_df = _merge_dfs(existing_race, new_race, ["year", "round", "driver"])
        quali_df = _merge_dfs(existing_quali, new_quali, ["year", "round", "driver"])
        if not existing_practice.empty or not new_practice.empty:
            practice_df = _merge_dfs(existing_practice, new_practice, ["year", "round", "driver"])
        else:
            practice_df = pd.DataFrame()

        save_raw(race_df, quali_df, practice_df if not practice_df.empty else None)
        logger.info(f"Merged dataset: {len(race_df)} race rows, {len(quali_df)} quali rows, years {sorted(race_df['year'].unique().tolist())}")

        # Clean up backup files
        backup_race_path.unlink(missing_ok=True)
        backup_quali_path.unlink(missing_ok=True)
    else:
        logger.info(f"Collecting F1 data {args.start}-{args.end} (fp={'no' if args.no_fp else 'yes'})")
        race_df, quali_df, practice_df = collect_historical(
            args.start, args.end, args.sleep, include_fp=not args.no_fp
        )
        save_raw(race_df, quali_df, practice_df)

    logger.info("Done.")
