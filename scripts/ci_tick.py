"""
One tick of the race-weekend automation (.github/workflows/race-weekend.yml).

Works out from the FastF1 calendar and the files in data/ what is due, does
it, and exits. Every step checks the files first, so running it more often
than needed is harmless.

  1. ingest   A race ended >= RESULTS_DELAY ago and its round is missing from
              race_results.parquet: collect race + quali, weather, pit stops
              and sprints for the season, Kalshi market prices (logged only),
              record the result against the
              prediction (if one was made), then retrain with quick_train.
  2. predict  Qualifying for the next race ended >= QUALI_DELAY ago, the race
              has not started, and there is no prediction file: run
              race_weekend_inference.
  3. export   Always: write site/f1-predictions.json for the website.

Lap-1 extras are not collected: FastF1 lap data now needs an F1 login (401).

A failed step is logged and the tick carries on; the exit code is 1 if any
step failed, so the workflow run shows red while still committing the steps
that worked.

Usage:
    python scripts/ci_tick.py [--now 2026-10-31T23:30:00Z] [--dry-run]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

QUALI_LENGTH = timedelta(hours=1)
QUALI_DELAY = timedelta(minutes=45)    # OpenF1 needs a little time to publish laps
RESULTS_DELAY = timedelta(hours=4)     # race length + Jolpica/FastF1 results publishing

PRED_DIR = ROOT / "data" / "predictions"
RES_DIR = ROOT / "data" / "results"


def _run(args: list[str], dry_run: bool) -> bool:
    cmd = [sys.executable, *args]
    print(f"$ {' '.join(args)}", flush=True)
    if dry_run:
        return True
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        print(f"!! failed with exit code {proc.returncode}: {' '.join(args)}", flush=True)
    return proc.returncode == 0


def _calendar(year: int) -> list[dict]:
    import fastf1
    from src.data_collection import f1_historical  # noqa: F401  (enables the FastF1 cache)

    sched = fastf1.get_event_schedule(year, include_testing=False)
    events = []
    for _, ev in sched.iterrows():
        times = {}
        for i in range(1, 6):
            name, t = ev.get(f"Session{i}"), ev.get(f"Session{i}DateUtc")
            if name and t is not None and str(t) != "NaT":
                times[str(name)] = t.to_pydatetime().replace(tzinfo=timezone.utc)
        if "Qualifying" in times and "Race" in times:
            events.append({
                "round": int(ev["RoundNumber"]),
                "name": str(ev["EventName"]),
                "quali": times["Qualifying"],
                "race": times["Race"],
            })
    return events


def _ingested_rounds(year: int) -> set[int]:
    from src.data_collection.f1_historical import load_race_results

    try:
        r = load_race_results()
    except FileNotFoundError:
        return set()
    return set(r.loc[r["year"] == year, "round"].astype(int))


def ingest(year: int, done_races: list[dict], dry_run: bool) -> bool:
    before = _ingested_rounds(year)
    missing = [e for e in done_races if e["round"] not in before]
    if not missing:
        print(f"ingest: nothing to do ({len(before)} {year} rounds in race_results)")
        return True
    print(f"ingest: rounds {[e['round'] for e in missing]} missing, collecting {year}")

    y = str(year)
    ok = _run(["-m", "src.data_collection.f1_historical", "--start", y, "--end", y, "--merge"], dry_run)
    if not ok:
        return False
    after = _ingested_rounds(year) if not dry_run else before | {e["round"] for e in missing}
    new_rounds = sorted(after - before)
    if not new_rounds:
        print("ingest: results not published yet, will retry next tick")
        return True

    ok &= _run(["-m", "src.data_collection.weather", "--years", y], dry_run)
    ok &= _run(["scripts/backfill_pit_sprint.py", "--pit", "--start", y, "--end", y, "--merge"], dry_run)
    ok &= _run(["scripts/backfill_pit_sprint.py", "--sprint", "--start", y, "--end", y, "--merge"], dry_run)
    # Market prices (Phase 4): logged for the model-vs-market comparison, not used by
    # the model, so a Kalshi outage must not fail the tick
    _run(["-m", "src.data_collection.kalshi_odds"], dry_run)

    for rnd in new_rounds:
        pred = PRED_DIR / f"{year}_R{rnd:02d}.json"
        res = RES_DIR / f"{year}_R{rnd:02d}_actual.json"
        if pred.exists() and not res.exists():
            ok &= _run(["scripts/update_race_result.py", "--year", y, "--round", str(rnd)], dry_run)

    ok &= _run(["scripts/quick_train.py"], dry_run)
    return ok


def predict(year: int, events: list[dict], now: datetime, dry_run: bool) -> bool:
    upcoming = [e for e in events if e["race"] > now]
    if not upcoming:
        print("predict: no upcoming race this season")
        return True
    ev = upcoming[0]
    pred = PRED_DIR / f"{year}_R{ev['round']:02d}.json"
    ready = ev["quali"] + QUALI_LENGTH + QUALI_DELAY
    if pred.exists():
        print(f"predict: R{ev['round']} {ev['name']} already predicted")
        return True
    if now < ready:
        print(f"predict: R{ev['round']} {ev['name']} waits for qualifying (ready {ready:%Y-%m-%d %H:%M} UTC)")
        return True
    print(f"predict: R{ev['round']} {ev['name']}")
    return _run(["scripts/race_weekend_inference.py", "--year", str(year), "--round", str(ev["round"])], dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="One tick of the race-weekend automation")
    parser.add_argument("--now", help="Pretend the current UTC time is this (ISO 8601), for testing")
    parser.add_argument("--dry-run", action="store_true", help="Print the steps without running them")
    args = parser.parse_args()

    now = (
        datetime.fromisoformat(args.now.replace("Z", "+00:00")) if args.now
        else datetime.now(timezone.utc)
    )
    year = now.year
    print(f"ci_tick at {now:%Y-%m-%d %H:%M} UTC, season {year}")

    try:
        events = _calendar(year)
    except Exception as exc:
        print(f"!! could not load the {year} calendar: {exc}")
        events = []

    ok = True
    if events:
        done = [e for e in events if e["race"] + RESULTS_DELAY <= now]
        ok &= ingest(year, done, args.dry_run)
        ok &= predict(year, events, now, args.dry_run)
    ok &= _run(["scripts/export_site_json.py", "--year", str(year)], args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
