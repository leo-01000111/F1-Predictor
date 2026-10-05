"""
OpenF1 API client for live/current qualifying data.
Used during inference (after qualifying concludes) to pull the latest
quali results and FP1/FP2/FP3 pace data for the upcoming race.

OpenF1 docs: https://openf1.org/
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import requests
from loguru import logger

OPENF1_BASE = "https://api.openf1.org/v1"
LONG_RUN_MIN_LAPS = 5
FP_LAP_MIN_S = 50.0
FP_LAP_MAX_S = 150.0
FP_DRIVER_OUTLIER_WINDOW_S = 12.0

def _get(endpoint: str, params: dict, retries: int = 4) -> list[dict]:
    """GET request to OpenF1, returns list of result objects. Retries on 429."""
    import time as _time
    url = f"{OPENF1_BASE}/{endpoint}"
    for attempt in range(retries):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            wait = 2 ** attempt * 3  # 3, 6, 12, 24 seconds
            logger.debug(f"OpenF1 429 rate limit on {endpoint}, waiting {wait}s (attempt {attempt + 1}/{retries})")
            _time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()  # final raise if still 429
    return []


def _session_candidates(session_type: str) -> list[str]:
    session_type = str(session_type)
    candidates = [session_type]
    upper = session_type.upper().replace(" ", "")
    if upper.startswith("FP") and len(upper) > 2:
        idx = upper[2:]
        if idx.isdigit():
            candidates.extend([
                f"Practice {idx}",
                f"Practice{idx}",
            ])
    return candidates


def _fp_session_index(session_type: str) -> Optional[int]:
    """Return 0-based index for FP sessions (FP1ďż˝0, FP2ďż˝1, FP3ďż˝2), or None if not FP."""
    upper = str(session_type).upper().replace(" ", "")
    if upper.startswith("FP") and len(upper) > 2:
        idx = upper[2:]
        if idx.isdigit():
            return int(idx) - 1
    return None


def get_latest_session(session_type: str = "Qualifying") -> Optional[dict]:
    """
    Return the most recent session of the given type.
    session_type: 'Qualifying', 'Race', 'Sprint', etc.
    """
    data = _get("sessions", {"session_type": session_type})
    if not data:
        logger.warning(f"No sessions of type '{session_type}' found via OpenF1")
        return None
    # Sessions are returned in chronological order; take the last
    return data[-1]


def get_session_by_round(
    year: int,
    round_number: int,
    session_type: str = "Qualifying",
) -> Optional[dict]:
    """
    Return a specific session by year and round number.
    Falls back to FastF1 schedule cross-reference when OpenF1 lacks round_number.
    """
    # Primary: direct round_number query (may 404 for newer seasons without round_number)
    for candidate_type in _session_candidates(session_type):
        try:
            data = _get(
                "sessions",
                {
                    "year": year,
                    "session_type": candidate_type,
                    "round_number": round_number,
                },
            )
            if data:
                return data[0]
        except Exception:
            pass  # Fall through to fallback logic below

    # Fallback A: use FastF1 schedule to find event date/location
    event_date = None
    location = None
    try:
        import fastf1

        schedule = fastf1.get_event_schedule(year, include_testing=False)
        event_rows = schedule[schedule["RoundNumber"] == round_number]
        if not event_rows.empty:
            event = event_rows.iloc[0]
            event_date = str(event["EventDate"].date())
            location = str(event["Location"])
    except Exception as exc:
        logger.debug(f"FastF1 schedule lookup failed: {exc}")

    event_dt = None
    if event_date:
        try:
            from datetime import datetime as _dt
            event_dt = _dt.fromisoformat(event_date).date()
        except Exception:
            event_dt = None

    # Fallback B: query all OpenF1 sessions for year, filter by session type, rank by date
    try:
        all_sessions = _get("sessions", {"year": year})
    except Exception:
        all_sessions = []

    if all_sessions:
        wanted_types = [t.lower() for t in _session_candidates(session_type)]

        def _type_match(s: dict) -> bool:
            """Match by session_type or session_name."""
            stype = str(s.get("session_type", "") or "").lower()
            sname = str(s.get("session_name", "") or "").lower()
            for w in wanted_types:
                if w in stype or stype in w:
                    return True
                if w in sname or sname in w:
                    return True
            return False

        type_sessions = [s for s in all_sessions if _type_match(s)]

        if type_sessions:
            from datetime import datetime as _dt

            def _start_date(s):
                d = str(s.get("date_start", "") or "")[:10]
                try:
                    return _dt.fromisoformat(d)
                except Exception:
                    return _dt(9999, 1, 1)

            # Keep only sessions in this venue if possible, then sort by date
            if location:
                loc_matches = [
                    s for s in type_sessions
                    if location.lower() in str(s.get("location", "")).lower()
                    or str(s.get("location", "")).lower() in location.lower()
                ]
                if loc_matches:
                    type_sessions = loc_matches

            # Keep only sessions close to event day if we know it (typically 1-3 days window)
            if event_dt is not None:
                nearby = [
                    s for s in type_sessions
                    if (
                        sd := _start_date(s)
                    ) != _dt(9999, 1, 1)
                    and abs((sd.date() - event_dt).days) <= 3
                ]
                if nearby:
                    type_sessions = nearby

            # In qualifying weekends with Sprint, prefer standard Qualifying
            if session_type.lower() == "qualifying":
                plain_qual = []
                for s in type_sessions:
                    stype = str(s.get("session_type", "")).lower()
                    sname = str(s.get("session_name", "")).lower()
                    is_qual = "qualifying" in stype or "qualifying" in sname
                    is_sprint = "sprint" in stype or "sprint" in sname
                    if is_qual and not is_sprint:
                        plain_qual.append(s)

                if plain_qual:
                    type_sessions = plain_qual

            # Sort by date
            type_sessions.sort(key=_start_date)

            fp_idx = _fp_session_index(session_type)
            if fp_idx is not None:
                if round_number <= len(type_sessions):
                    session = type_sessions[round_number - 1]
                    logger.info(
                        f"Resolved {year} R{round_number} {session_type} via FP-ordinal fallback: "
                        f"session_key={session.get('session_key')} location={session.get('location')}"
                    )
                    return session
            elif round_number <= len(type_sessions):
                session = type_sessions[round_number - 1]
                logger.info(
                    f"Resolved {year} R{round_number} {session_type} via ordinal fallback: "
                    f"session_key={session.get('session_key')} location={session.get('location')}"
                )
                return session

            # As a last fallback, return the closest-date matching session
            if len(type_sessions) > 0:
                if event_dt is not None:
                    session = sorted(
                        type_sessions,
                        key=lambda s: abs((_start_date(s).date() - event_dt).days)
                        if _start_date(s) != _dt(9999, 1, 1)
                        else 99999,
                    )[0]
                else:
                    session = type_sessions[0]

                logger.info(
                    f"Resolved {year} R{round_number} {session_type} via nearest-date fallback: "
                    f"session_key={session.get('session_key')} location={session.get('location')}"
                )
                return session

    logger.warning(f"Could not resolve session for {year} R{round_number} {session_type}")
    return None


def _stint_column(cols: list) -> Optional[str]:
    for col in ["Stint", "stint", "stint_number"]:
        if col in cols:
            return col
    return None


def _to_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def get_quali_results(session_key: int) -> pd.DataFrame:
    """
    Pull qualifying position data for a given session_key.
    Returns DataFrame with columns: driver_number, driver_abbr, team, position, best_lap_time_s
    """
    # Fetch drivers in this session
    drivers_raw = _get("drivers", {"session_key": session_key})
    drivers = {d["driver_number"]: d for d in drivers_raw}

    # Fetch laps for this qualifying session
    laps_raw = _get("laps", {"session_key": session_key})
    laps_df = pd.DataFrame(laps_raw)

    if laps_df.empty:
        logger.warning(f"No lap data found for session_key={session_key}")
        return pd.DataFrame()

    # Best lap per driver (from lap data)
    laps_df["lap_duration"] = _to_float_series(laps_df.get("lap_duration", pd.Series(dtype=float)))
    best_laps = (
        laps_df.groupby("driver_number")["lap_duration"]
        .min()
        .reset_index()
        .rename(columns={"lap_duration": "best_lap_time_s"})
    )

    # Include all registered drivers — even those with no timed lap (DNF/DNS in quali)
    timed_numbers = set(best_laps["driver_number"].tolist())
    untimed_rows = [
        {"driver_number": num, "best_lap_time_s": float("nan")}
        for num, info in drivers.items()
        if num not in timed_numbers and info.get("name_acronym")
    ]
    if untimed_rows:
        best_laps = pd.concat([best_laps, pd.DataFrame(untimed_rows)], ignore_index=True)
        logger.info(
            f"session_key={session_key}: added {len(untimed_rows)} driver(s) with no timed lap"
        )

    # Merge driver info
    best_laps["driver_abbr"] = best_laps["driver_number"].map(
        lambda n: drivers.get(n, {}).get("name_acronym", str(int(n)) if pd.notna(n) else "")
    )
    best_laps["team"] = best_laps["driver_number"].map(
        lambda n: drivers.get(n, {}).get("team_name", "Unknown")
    )

    # Rank by best lap time — timed drivers first (NaN sorts to end automatically)
    best_laps = best_laps.sort_values("best_lap_time_s", na_position="last").reset_index(drop=True)
    best_laps["quali_position"] = best_laps.index + 1

    # Gap to pole (NaN for untimed drivers)
    pole_time = best_laps["best_lap_time_s"].iloc[0]
    best_laps["quali_gap_to_pole_s"] = best_laps["best_lap_time_s"] - pole_time

    return best_laps[["driver_number", "driver_abbr", "team", "quali_position", "best_lap_time_s", "quali_gap_to_pole_s"]]


def get_practice_results(
    session_key: int,
    fp_num: int,
    year: int,
    round_num: int,
    circuit_key: str,
    circuit: str,
    race_date: str,
) -> pd.DataFrame:
    """
    Pull per-driver FP metrics for one free-practice session.
    Returns columns compatible with historical FP rows.
    """
    drivers_raw = _get("drivers", {"session_key": session_key})
    drivers = {d.get("driver_number"): d for d in drivers_raw}

    laps_raw = _get("laps", {"session_key": session_key})
    laps_df = pd.DataFrame(laps_raw)

    if laps_df.empty:
        logger.warning(f"No lap data found for session_key={session_key}")
        return pd.DataFrame()

    if "driver_number" not in laps_df.columns or "lap_duration" not in laps_df.columns:
        logger.warning(f"Practice session {session_key} missing driver/lap columns")
        return pd.DataFrame()

    laps_df["driver_number"] = pd.to_numeric(laps_df["driver_number"], errors="coerce")
    laps_df["lap_duration"] = _to_float_series(laps_df["lap_duration"])
    laps_df = laps_df.dropna(subset=["driver_number", "lap_duration"])
    if laps_df.empty:
        return pd.DataFrame()

    # Remove clearly invalid laps (out/in laps and telemetry artifacts).
    before = len(laps_df)
    laps_df = laps_df[
        (laps_df["lap_duration"] >= FP_LAP_MIN_S)
        & (laps_df["lap_duration"] <= FP_LAP_MAX_S)
    ]
    if laps_df.empty:
        logger.warning(f"All FP laps filtered as invalid for session_key={session_key}")
        return pd.DataFrame()

    # Per-driver robust filter around median pace.
    med = laps_df.groupby("driver_number")["lap_duration"].transform("median")
    laps_df = laps_df[(laps_df["lap_duration"] - med).abs() <= FP_DRIVER_OUTLIER_WINDOW_S]
    if laps_df.empty:
        logger.warning(f"All FP laps removed by outlier filter for session_key={session_key}")
        return pd.DataFrame()

    removed = before - len(laps_df)
    if removed > 0:
        logger.debug(f"Filtered {removed} FP laps as outliers for session_key={session_key}")

    # Use overall cleanest/lows as reference pace for gap calculations
    session_best = laps_df["lap_duration"].min()
    session_median = laps_df["lap_duration"].median()

    stint_col = _stint_column(list(laps_df.columns))

    rows = []
    for driver_num in laps_df["driver_number"].dropna().astype(int).unique():
        driver_laps = laps_df[laps_df["driver_number"] == driver_num]
        if driver_laps.empty:
            continue

        driver_info = drivers.get(driver_num, {})
        driver_abbr = driver_info.get("name_acronym", str(driver_num))
        team = driver_info.get("team_name", "Unknown")

        best_gap = np.nan
        if pd.notna(session_best):
            best_time = driver_laps["lap_duration"].dropna().min()
            if pd.notna(best_time):
                best_gap = float(best_time - session_best)

        long_run_delta = np.nan
        if pd.notna(session_median) and stint_col is not None:
            stint_sizes = driver_laps.groupby(stint_col).size()
            long_stints = stint_sizes[stint_sizes >= LONG_RUN_MIN_LAPS].index
            if len(long_stints) > 0:
                long_run_laps = driver_laps[driver_laps[stint_col].isin(long_stints)]["lap_duration"].dropna()
                if not long_run_laps.empty:
                    long_run_delta = float(long_run_laps.mean() - session_median)

        rows.append({
            "year": year,
            "round": round_num,
            "circuit": circuit,
            "circuit_key": circuit_key,
            "race_date": race_date,
            "driver": driver_abbr,
            f"fp{fp_num}_best_gap_s": float(best_gap) if pd.notna(best_gap) else np.nan,
            f"fp{fp_num}_long_run_delta": float(long_run_delta) if pd.notna(long_run_delta) else np.nan,
            f"fp{fp_num}_laps": int(len(driver_laps)),
        })

    return pd.DataFrame(rows)


def get_live_quali_dataframe(year: int, round_number: int) -> pd.DataFrame:
    """
    High-level function: given a year and round, return a clean quali DataFrame
    suitable for feature engineering at inference time.
    """
    session = get_session_by_round(year, round_number, "Qualifying")
    if session is None:
        logger.error(f"Could not find qualifying session for {year} round {round_number}")
        return pd.DataFrame()

    session_key = session["session_key"]
    circuit = session.get("location", "Unknown")
    circuit_key = session.get("circuit_short_name", "unknown").lower().replace(" ", "_")
    race_date = session.get("date_end", "")[:10] if session.get("date_end") else ""

    df = get_quali_results(session_key)
    if df.empty:
        return df

    # Add context columns
    df["year"] = year
    df["round"] = round_number
    df["circuit"] = circuit
    df["circuit_key"] = circuit_key
    df["race_date"] = race_date

    logger.info(f"Fetched live quali for {year} R{round_number} ({circuit}): {len(df)} drivers")
    return df


def get_live_practice_dataframe(year: int, round_number: int) -> pd.DataFrame:
    """
    Fetch live FP1/FP2/FP3 metrics for a given weekend and return
    one row per driver with FP1/FP2/FP3 columns.
    """
    fp_frames: list[pd.DataFrame] = []

    for fp_num in [1, 2, 3]:
        session = get_session_by_round(year, round_number, f"FP{fp_num}")
        if session is None:
            logger.warning(f"No FP{fp_num} session found for {year} R{round_number}")
            continue

        session_key = session.get("session_key")
        if session_key is None:
            continue

        circuit = session.get("location", "Unknown")
        circuit_key = session.get("circuit_short_name", "unknown").lower().replace(" ", "_")
        race_date = session.get("date_end", "")[:10] if session.get("date_end") else ""

        fp_df = get_practice_results(
            session_key=session_key,
            fp_num=fp_num,
            year=year,
            round_num=round_number,
            circuit_key=circuit_key,
            circuit=circuit,
            race_date=race_date,
        )
        if not fp_df.empty:
            fp_frames.append(fp_df)

    if not fp_frames:
        return pd.DataFrame()

    # Merge on stable driver identity only to avoid session-level naming drifts.
    merge_cols = ["year", "round", "driver"]

    fp_frames_clean: list[pd.DataFrame] = []
    meta_frames: list[pd.DataFrame] = []
    for df in fp_frames:
        metric_cols = [c for c in df.columns if c.startswith("fp")]
        fp_frames_clean.append(df[[c for c in merge_cols + metric_cols if c in df.columns]].copy())
        meta_cols = [c for c in ["year", "round", "driver", "circuit", "circuit_key", "race_date"] if c in df.columns]
        meta_frames.append(df[meta_cols].copy())

    merged = fp_frames_clean[0]
    for fp_df in fp_frames_clean[1:]:
        merged = merged.merge(fp_df, on=merge_cols, how="outer")

    if meta_frames:
        meta = pd.concat(meta_frames, ignore_index=True)
        if "race_date" in meta.columns:
            meta = meta.sort_values("race_date")
        meta = meta.drop_duplicates(subset=merge_cols, keep="last")
        merged = merged.merge(meta, on=merge_cols, how="left")

    return merged.sort_values("driver").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# In-season race results (FastF1)
# --------------------------------------------------------------------------- #

def get_completed_season_race_results(year: int, max_round: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch completed race results for the given year (up to max_round) via FastF1.
    Returns a DataFrame with the same schema as the historical race results table:
      year, round, circuit, circuit_key, race_date, driver, team,
      grid_position, finish_position, dnf, status, points, p1, p2, p3, podium,
      quali_gap_to_pole_s

    Called during race-weekend inference so rolling form features
    (driver_form_ewm, championship_standing, etc.) reflect all races already
    run in the current season rather than freezing at end-of-previous-season.
    """
    import fastf1

    rows = []
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as exc:
        logger.warning(f"FastF1 schedule fetch failed for {year}: {exc}")
        return pd.DataFrame()

    from datetime import date as _date
    today = _date.today()

    for _, event in schedule.iterrows():
        rnd = int(event.get("RoundNumber", 0))
        if rnd < 1:
            continue
        if max_round is not None and rnd >= max_round:
            break  # stop before the current (upcoming) round

        # Skip events that haven't occurred yet
        event_date = event.get("EventDate")
        try:
            if hasattr(event_date, "date"):
                event_date = event_date.date()
            elif isinstance(event_date, str):
                from datetime import datetime as _dt
                event_date = _dt.fromisoformat(event_date).date()
            if event_date > today:
                continue
        except Exception:
            pass

        try:
            session = fastf1.get_session(year, rnd, "R")
            session.load(laps=False, telemetry=False, weather=False, messages=False)
        except Exception as exc:
            logger.debug(f"Could not load {year} R{rnd} race session: {exc}")
            continue

        results = session.results
        if results is None or results.empty:
            continue

        circuit_key = str(event.get("OfficialEventName", "unknown")).lower().replace(" ", "_")[:32]
        circuit = str(event.get("Location", event.get("EventName", "unknown")))
        race_date = ""
        try:
            race_date = str(session.date.date())
        except Exception:
            pass

        for _, r in results.iterrows():
            driver = str(r.get("Abbreviation", "")).strip()
            team = str(r.get("TeamName", "Unknown")).strip()
            grid_pos = r.get("GridPosition", None)
            finish_pos = r.get("Position", None)
            status = str(r.get("Status", "")).strip()

            dnf = 1 if (
                status.lower() not in ("finished", "") and
                not str(finish_pos).replace(".0", "").isdigit()
            ) else 0

            try:
                finish_pos_int = int(float(finish_pos)) if pd.notna(finish_pos) else None
            except Exception:
                finish_pos_int = None

            try:
                grid_pos_int = int(float(grid_pos)) if pd.notna(grid_pos) else None
            except Exception:
                grid_pos_int = None

            points = float(r.get("Points", 0) or 0)
            p1 = 1 if finish_pos_int == 1 else 0
            p2 = 1 if finish_pos_int == 2 else 0
            p3 = 1 if finish_pos_int == 3 else 0
            podium = 1 if finish_pos_int in (1, 2, 3) else 0

            rows.append({
                "year": year,
                "round": rnd,
                "circuit": circuit,
                "circuit_key": circuit_key,
                "race_date": race_date,
                "driver": driver,
                "team": team,
                "grid_position": grid_pos_int,
                "finish_position": finish_pos_int,
                "dnf": dnf,
                "status": status,
                "points": points,
                "p1": p1,
                "p2": p2,
                "p3": p3,
                "podium": podium,
                "quali_gap_to_pole_s": None,  # not available from race results
            })

        logger.info(f"Fetched {year} R{rnd} ({circuit}): {len(results)} drivers")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["year", "round"]).reset_index(drop=True)
    logger.info(f"In-season results: {len(df)} rows across {df['round'].nunique()} completed rounds of {year}")
    return df


def get_live_sprint_pace_dataframe(year: int, round_num: int) -> pd.DataFrame:
    """
    Fetch sprint race lap-based pace metrics from OpenF1 for the current weekend.

    Returns a DataFrame with columns:
        year, round, driver, sprint_best_lap_gap_s, sprint_pace_gap_s

    sprint_best_lap_gap_s : driver best lap vs session best lap (gap to fastest)
    sprint_pace_gap_s     : driver median mid-race pace vs session median (race pace delta)

    Returns empty DataFrame when no sprint session exists (normal weekends) or
    when the sprint has not happened yet.
    """
    session = get_session_by_round(year, round_num, "Sprint")
    if session is None:
        logger.debug(f"No sprint session found for {year} R{round_num} (normal weekend)")
        return pd.DataFrame()

    session_key = session.get("session_key")
    if session_key is None:
        return pd.DataFrame()

    try:
        laps_raw = _get("laps", {"session_key": session_key})
    except Exception as exc:
        logger.warning(f"Could not fetch sprint laps for {year} R{round_num}: {exc}")
        return pd.DataFrame()

    if not laps_raw:
        logger.debug(f"No sprint lap data yet for {year} R{round_num} (sprint not finished?)")
        return pd.DataFrame()

    laps_df = pd.DataFrame(laps_raw)
    if "lap_duration" not in laps_df.columns or "driver_number" not in laps_df.columns:
        return pd.DataFrame()

    laps_df["lap_s"] = pd.to_numeric(laps_df["lap_duration"], errors="coerce")
    laps_df = laps_df[laps_df["lap_s"].between(40.0, 200.0)].copy()
    if laps_df.empty:
        return pd.DataFrame()

    # Map driver_number → abbreviation
    try:
        drivers_raw = _get("drivers", {"session_key": session_key})
        driver_map: dict[int, str] = {
            int(d["driver_number"]): str(d.get("name_acronym", ""))
            for d in drivers_raw
            if d.get("driver_number") is not None
        }
    except Exception:
        driver_map = {}

    laps_df["driver"] = pd.to_numeric(laps_df["driver_number"], errors="coerce").map(driver_map)
    laps_df = laps_df[laps_df["driver"].notna() & (laps_df["driver"] != "")]
    if laps_df.empty:
        return pd.DataFrame()

    session_best = laps_df["lap_s"].min()

    # Mid-race laps: skip lap 1 (standing start) and last 2 (push laps / safety car)
    lap_num_col = "lap_number" if "lap_number" in laps_df.columns else None
    if lap_num_col:
        max_lap = laps_df[lap_num_col].max()
        mid = laps_df[(laps_df[lap_num_col] > 1) & (laps_df[lap_num_col] < max_lap - 1)]
    else:
        mid = laps_df.copy()
    session_mid_median = mid["lap_s"].median() if not mid.empty else laps_df["lap_s"].median()

    rows = []
    for drv, grp in laps_df.groupby("driver"):
        best_gap = float(np.clip(grp["lap_s"].min() - session_best, 0.0, 10.0))
        drv_mid = mid[mid["driver"] == drv]["lap_s"] if not mid.empty else pd.Series(dtype=float)
        pace_gap = float(np.clip(
            (drv_mid.median() if not drv_mid.empty else grp["lap_s"].median()) - session_mid_median,
            -5.0, 10.0,
        ))
        rows.append({
            "year": year,
            "round": round_num,
            "driver": drv,
            "sprint_best_lap_gap_s": best_gap,
            "sprint_pace_gap_s": pace_gap,
        })

    result = pd.DataFrame(rows)
    logger.info(
        f"Sprint pace data fetched for {year} R{round_num}: {len(result)} drivers | "
        f"best gap range [{result['sprint_best_lap_gap_s'].min():.2f}, "
        f"{result['sprint_best_lap_gap_s'].max():.2f}]s"
    )
    return result


def get_start_compounds(year: int, round_num: int) -> dict[str, str]:
    """
    Fetch the start tire compound for each driver via OpenF1 stints.

    Returns {driver_abbr: compound_name} e.g. {"VER": "SOFT", "NOR": "MEDIUM"}.
    Falls back to {} when unavailable (e.g. pre-race, OpenF1 lag).

    NOTE: Stint data for a race session is only available AFTER the race starts.
    For pre-race inference, this will return {} and the caller should default to
    Medium (1). It is most useful for retroactive data collection.
    """
    session = get_session_by_round(year, round_num, "Race")
    if not session:
        logger.debug(f"No race session found for {year} R{round_num} -- start compounds unavailable")
        return {}

    session_key = session.get("session_key")
    if not session_key:
        return {}

    try:
        # Fetch stint-1 data for all drivers
        stints = _get("stints", {"session_key": session_key, "stint_number": 1})
        if not stints:
            logger.debug(f"No stint data for {year} R{round_num} (session_key={session_key})")
            return {}

        # Build driver_number -> abbreviation mapping
        drivers_raw = _get("drivers", {"session_key": session_key})
        driver_map: dict[int, str] = {
            int(d["driver_number"]): str(d.get("name_acronym", ""))
            for d in drivers_raw
            if d.get("driver_number") is not None
        }

        result: dict[str, str] = {}
        for stint in stints:
            if int(stint.get("stint_number", 0)) != 1:
                continue
            raw_num = stint.get("driver_number")
            if raw_num is None:
                continue
            try:
                driver_num = int(raw_num)
            except (ValueError, TypeError):
                continue
            abbr = driver_map.get(driver_num, "")
            compound = str(stint.get("compound") or "MEDIUM").upper()
            if abbr:
                result[abbr] = compound

        logger.info(
            f"Fetched start compounds for {year} R{round_num}: {len(result)} drivers"
        )
        return result

    except Exception as exc:
        logger.warning(f"Could not fetch start compounds for {year} R{round_num}: {exc}")
        return {}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch live F1 qualifying and practice data via OpenF1")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args()

    quali_df = get_live_quali_dataframe(args.year, args.round)
    fp_df = get_live_practice_dataframe(args.year, args.round)

    print(quali_df.to_string())
    print("\n" + "=" * 80)
    print(fp_df.to_string())





