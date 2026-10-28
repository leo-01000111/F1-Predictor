"""
Availability probes for the race-weekend sources, logged by every ci_tick.

Each probe asks one source what it has for one round right now and logs a
`probe` event (src/ci_log.py): row count, a compact payload where it matters
(the starting grid order, so later changes show up) and the error if any.
Over a weekend the hourly ticks give a timeline of when each piece of data
appeared, which the pipeline's timing assumptions can be checked against
(QUALI_DELAY, RESULTS_DELAY, when the post-penalty grid is published).
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Callable

import requests

from src import ci_log

OPENF1 = "https://api.openf1.org/v1"
JOLPICA = "https://api.jolpi.ca/ergast/f1"


def _get(url: str, params: dict | None = None) -> Any:
    for attempt in range(4):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            time.sleep(3 * 2 ** attempt)
            continue
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return []


def _probe(rnd: int, source: str, item: str, fn: Callable[[], tuple[int, Any]]) -> tuple[int, Any]:
    t0 = time.monotonic()
    try:
        n, detail = fn()
        ci_log.event("probe", round=rnd, source=source, item=item, n=n, detail=detail,
                     ms=round((time.monotonic() - t0) * 1000))
        return n, detail
    except Exception as exc:
        ci_log.event("probe", round=rnd, source=source, item=item, n=None, error=str(exc)[:300],
                     ms=round((time.monotonic() - t0) * 1000))
        return 0, None


def _jolpica(year: int, rnd: int, what: str, key: str) -> tuple[int, Any]:
    races = _get(f"{JOLPICA}/{year}/{rnd}/{what}.json", {"limit": 100})["MRData"]["RaceTable"]["Races"]
    rows = races[0].get(key, []) if races else []
    top = [r["Driver"].get("code") for r in rows[:3]]
    return len(rows), {"top3": top} if rows else None


def _meeting(year: int, race_time: datetime) -> list[dict]:
    """OpenF1 sessions of the meeting whose race starts on race_time's date (+-1 day)."""
    sessions = _get(f"{OPENF1}/sessions", {"year": year})
    races = [s for s in sessions if s.get("session_name") == "Race"
             and abs((datetime.fromisoformat(s["date_start"]).date() - race_time.date()).days) <= 1]
    if not races:
        return []
    mk = races[0]["meeting_key"]
    return [s for s in sessions if s.get("meeting_key") == mk]


def probe_weekend(year: int, rnd: int, race_time: datetime) -> None:
    """Everything the prediction needs for an upcoming race."""
    meeting: list[dict] = []

    def sessions():
        nonlocal meeting
        meeting = _meeting(year, race_time)
        return len(meeting), [{"name": s["session_name"], "key": s["session_key"],
                               "start": s["date_start"], "end": s.get("date_end")} for s in meeting]

    _probe(rnd, "openf1", "sessions", sessions)
    by_name = {s["session_name"]: s for s in meeting}
    quali = by_name.get("Qualifying")
    sprint = by_name.get("Sprint")

    names: dict[int, str] = {}
    if quali:
        sk = quali["session_key"]

        def drivers():
            names.update({int(d["driver_number"]): d.get("name_acronym", "") for d in
                          _get(f"{OPENF1}/drivers", {"session_key": sk}) if d.get("driver_number") is not None})
            return len(names), None

        _probe(rnd, "openf1", "quali_drivers", drivers)
        # What race_weekend_inference builds the quali table from
        _probe(rnd, "openf1", "quali_laps", lambda: (len(_get(f"{OPENF1}/laps", {"session_key": sk})), None))

        def quali_result():
            res = sorted(_get(f"{OPENF1}/session_result", {"session_key": sk}),
                         key=lambda r: r.get("position") or 99)
            return len(res), {"order": [names.get(int(r["driver_number"]), r["driver_number"]) for r in res]}

        _probe(rnd, "openf1", "quali_result", quali_result)

        def grid():
            g = sorted(_get(f"{OPENF1}/starting_grid", {"meeting_key": quali["meeting_key"]}),
                       key=lambda r: r.get("position") or 99)
            return len(g), {"order": [names.get(int(r["driver_number"]), r["driver_number"]) for r in g]}

        _probe(rnd, "openf1", "starting_grid", grid)
    _probe(rnd, "jolpica", "qualifying", lambda: _jolpica(year, rnd, "qualifying", "QualifyingResults"))
    if sprint:
        _probe(rnd, "openf1", "sprint_result",
               lambda: (len(_get(f"{OPENF1}/session_result", {"session_key": sprint["session_key"]})), None))
        _probe(rnd, "jolpica", "sprint", lambda: _jolpica(year, rnd, "sprint", "SprintResults"))


def probe_results(year: int, rnd: int, race_time: datetime) -> None:
    """What the post-race ingest needs: the classification on each source."""
    _probe(rnd, "jolpica", "results", lambda: _jolpica(year, rnd, "results", "Results"))
    _probe(rnd, "jolpica", "qualifying", lambda: _jolpica(year, rnd, "qualifying", "QualifyingResults"))

    def race_result():
        race = [s for s in _meeting(year, race_time) if s.get("session_name") == "Race"]
        if not race:
            return 0, None
        return len(_get(f"{OPENF1}/session_result", {"session_key": race[0]["session_key"]})), None

    _probe(rnd, "openf1", "race_result", race_result)
