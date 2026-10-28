"""
One tick of the race-weekend automation (.github/workflows/race-weekend.yml).

Works out from the FastF1 calendar and the files in data/ what is due, does
it, and exits. Every step checks the files first, so running it more often
than needed is harmless.

  0. probe    During a race weekend (3 days before to 2 days after the race):
              log what OpenF1 and Jolpica have for it right now (sessions,
              qualifying, starting grid, sprint, results). Logging only.
  1. ingest   A race ended >= RESULTS_DELAY ago and its round is missing from
              race_results.parquet: collect race + quali, weather, pit stops
              and sprints for the season, Kalshi market prices (logged only),
              record the result against the
              prediction (if one was made), retrain with quick_train, then
              update the championship odds (title_odds).
  2. predict  Qualifying for the next race ended >= QUALI_DELAY ago, the race
              has not started, and there is no prediction file: run
              race_weekend_inference.
     regrid   There is a prediction and the race has not started: re-predict
              if the official starting grid (grid penalties) now differs from
              the grid the prediction used. Penalties are often confirmed after
              the first prediction, and training sees the post-penalty grid.
     market   From qualifying to lights out: snapshot the live Kalshi win and
              podium books for the race (data/raw/kalshi_snapshots.parquet).
              The last one before the start is the exact pre-start price.
  3. export   Always: write site/f1-predictions.json for the website.

Lap-1 extras are not collected: FastF1 lap data now needs an F1 login (401).

A failed step is logged and the tick carries on; the exit code is 1 if any
step failed, so the workflow run shows red while still committing the steps
that worked.

Every tick also writes data/logs/ci_<season>.jsonl (src/ci_log.py, committed
by the workflow): start time and cron delay, each decision, each command with
its exit code, duration and output tail, data before/after, what a prediction
used, and the probes. Read it with scripts/ci_log_report.py.

Usage:
    python scripts/ci_tick.py [--now 2026-10-31T23:30:00Z] [--dry-run] [--no-probe]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src import ci_log  # noqa: E402

QUALI_LENGTH = timedelta(hours=1)
QUALI_DELAY = timedelta(minutes=45)    # OpenF1 needs a little time to publish laps
RESULTS_DELAY = timedelta(hours=4)     # race length + Jolpica/FastF1 results publishing
PROBE_BEFORE = timedelta(days=3)       # probe the sources from 3 days before a race
PROBE_AFTER = timedelta(days=2)        # ... until 2 days after it

PRED_DIR = ROOT / "data" / "predictions"
RES_DIR = ROOT / "data" / "results"
RAW_DIR = ROOT / "data" / "raw"
SITE_JSON = ROOT / "site" / "f1-predictions.json"
TAIL_LINES = 25
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _decide(step: str, msg: str, **data) -> None:
    print(f"{step}: {msg}", flush=True)
    ci_log.event("decision", step=step, msg=msg, **data)


def _run(args: list[str], dry_run: bool, step: str = "") -> bool:
    """Run a script, streaming its output; log exit code, duration and output tail."""
    cmd = [sys.executable, *args]
    print(f"$ {' '.join(args)}", flush=True)
    if dry_run:
        ci_log.event("step", step=step, cmd=" ".join(args), dry_run=True)
        return True
    t0 = time.monotonic()
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tail: collections.deque[str] = collections.deque(maxlen=TAIL_LINES)
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        print(line, end="", flush=True)
        clean = _ANSI.sub("", line.rstrip())
        if clean and "| DEBUG" not in clean:
            tail.append(clean[:300])
    rc = proc.wait()
    ci_log.event("step", step=step, cmd=" ".join(args), started=started, exit_code=rc,
                 seconds=round(time.monotonic() - t0, 1), tail=list(tail))
    if rc != 0:
        print(f"!! failed with exit code {rc}: {' '.join(args)}", flush=True)
    return rc == 0


def _digest(path: Path) -> str | None:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12] if path.exists() else None


def _load_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _raw_counts(year: int) -> dict:
    """Rows and rounds per raw file for the season: what an ingest added."""
    import pandas as pd

    out: dict = {}
    for name in ("race_results", "quali_results", "sprint_results", "pit_stops", "weather_history",
                 "practice_laps", "kalshi_odds"):
        path = RAW_DIR / f"{name}.parquet"
        if not path.exists():
            out[name] = None
            continue
        try:
            df = pd.read_parquet(path)
            df = df[df["year"] == year] if "year" in df.columns else df
            rounds = sorted(int(r) for r in df["round"].dropna().unique()) if "round" in df.columns else []
            out[name] = {"rows": len(df), "rounds": rounds}
        except Exception as exc:
            out[name] = {"error": str(exc)[:200]}
    return out


def _log_model() -> None:
    m = _load_json(ROOT / "models" / "holdout_metrics.json")
    a = _load_json(ROOT / "models" / "active_model.json")
    ci_log.event("data", what="model", trained_at=a.get("generated_at"), family=a.get("model_family"),
                 train_years=a.get("train_years"),
                 holdout={k: m.get(k) for k in ("holdout_years", "n_holdout_races", "winner_accuracy",
                                                 "podium_accuracy", "mean_log_loss_winner", "temperature")})


def _log_title(year: int) -> None:
    t = _load_json(PRED_DIR / f"{year}_title.json")
    if t:
        ci_log.event("data", what="title_odds", after_round=t.get("after_round"),
                     drivers=[(d.get("driver"), d.get("p_champion")) for d in t.get("drivers", [])[:3]])


def _log_prediction(path: Path, reason: str) -> None:
    """What a prediction file says it used, and its top picks."""
    d = _load_json(path)
    if not d:
        return
    preds = d.get("predictions", [])
    ci_log.event("prediction", reason=reason, round=d.get("round"), file=path.name,
                 generated_at=d.get("generated_at"), grid_source=d.get("grid_source"),
                 inputs=d.get("inputs"), weather=d.get("weather"),
                 grid=[(p["driver"], p.get("grid_position")) for p in preds],
                 top=[(p["driver"], round(p.get("p_win", 0), 3), round(p.get("p_podium", 0), 3))
                      for p in preds[:5]])


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
                "sessions": {k: v.strftime("%Y-%m-%dT%H:%M:%SZ") for k, v in times.items()},
            })
    return events


def probe(year: int, events: list[dict], now: datetime) -> None:
    """Log what each source has for the current race weekend (or the race just run)."""
    from src.data_collection import source_probe

    for ev in events:
        if not (ev["race"] - PROBE_BEFORE <= now <= ev["race"] + PROBE_AFTER):
            continue
        ci_log.event("weekend", round=ev["round"], name=ev["name"], sessions=ev["sessions"])
        try:
            if now < ev["race"]:
                source_probe.probe_weekend(year, ev["round"], ev["race"])
            else:
                source_probe.probe_results(year, ev["round"], ev["race"])
        except Exception as exc:
            ci_log.event("probe", round=ev["round"], source="probe", item="all", n=None, error=str(exc)[:300])


def _ingested_rounds(year: int) -> set[int]:
    from src.data_collection.f1_historical import load_race_results

    try:
        r = load_race_results()
    except FileNotFoundError:
        return set()
    return set(r.loc[r["year"] == year, "round"].astype(int))


def market_snapshot(year: int, events: list[dict], now: datetime) -> None:
    """From qualifying to lights out: save the live Kalshi books of the next race.

    The last snapshot before the start is the pre-start market price for the
    model/market blend (ROADMAP idea 3). Logging only: never fails the tick.
    """
    ev = next((e for e in events if e["quali"] <= now < e["race"]), None)
    if ev is None:
        return
    try:
        from src.data_collection.kalshi_odds import snapshot

        summary = snapshot(year, ev["round"], ev["race"], now)
        ci_log.event("market_snapshot", round=ev["round"], **summary)
        print(f"market: R{ev['round']} {summary.get('rows')} Kalshi quotes, "
              f"{summary.get('minutes_to_race')} min to the start", flush=True)
    except Exception as exc:
        ci_log.event("market_snapshot", round=ev["round"], error=str(exc)[:300])
        print(f"market: snapshot failed: {exc}", flush=True)


def ingest(year: int, done_races: list[dict], now: datetime, dry_run: bool) -> bool:
    before = _ingested_rounds(year)
    missing = [e for e in done_races if e["round"] not in before]
    if not missing:
        _decide("ingest", f"nothing to do ({len(before)} {year} rounds in race_results)")
        return True
    _decide("ingest", f"rounds {[e['round'] for e in missing]} missing, collecting {year}",
            missing=[e["round"] for e in missing],
            hours_since_race={e["round"]: round((now - e["race"]).total_seconds() / 3600, 2) for e in missing})
    counts_before = _raw_counts(year)

    y = str(year)
    ok = _run(["-m", "src.data_collection.f1_historical", "--start", y, "--end", y, "--merge"], dry_run, "ingest")
    if not ok:
        return False
    after = _ingested_rounds(year) if not dry_run else before | {e["round"] for e in missing}
    new_rounds = sorted(after - before)
    if not new_rounds:
        _decide("ingest", "results not published yet, will retry next tick")
        return True

    ok &= _run(["-m", "src.data_collection.weather", "--years", y], dry_run, "ingest")
    ok &= _run(["scripts/backfill_pit_sprint.py", "--pit", "--start", y, "--end", y, "--merge"], dry_run, "ingest")
    ok &= _run(["scripts/backfill_pit_sprint.py", "--sprint", "--start", y, "--end", y, "--merge"], dry_run, "ingest")
    # Practice laps for the long-run pace features (OpenF1)
    ok &= _run(["-m", "src.data_collection.practice_openf1", "--years", y], dry_run, "ingest")
    # Market prices (Phase 4): logged for the model-vs-market comparison, not used by
    # the model, so a Kalshi outage must not fail the tick
    _run(["-m", "src.data_collection.kalshi_odds"], dry_run, "ingest")
    ci_log.event("data", what="raw_files", new_rounds=new_rounds, before=counts_before, after=_raw_counts(year))

    for rnd in new_rounds:
        pred = PRED_DIR / f"{year}_R{rnd:02d}.json"
        res = RES_DIR / f"{year}_R{rnd:02d}_actual.json"
        if pred.exists() and not res.exists():
            ok &= _run(["scripts/update_race_result.py", "--year", y, "--round", str(rnd)], dry_run, "ingest")
            ci_log.event("data", what="result_vs_prediction", round=rnd, accuracy=_load_json(res).get("accuracy"))
        elif not pred.exists():
            _decide("ingest", f"R{rnd} has no prediction file, nothing to score", round=rnd)

    ok &= _run(["scripts/quick_train.py"], dry_run, "train")
    _log_model()
    # Championship odds from the new standings (Phase 5)
    ok &= _run(["scripts/title_odds.py", "--year", y], dry_run, "title_odds")
    _log_title(year)
    return ok


def predict(year: int, events: list[dict], now: datetime, dry_run: bool) -> bool:
    upcoming = [e for e in events if e["race"] > now]
    if not upcoming:
        _decide("predict", "no upcoming race this season")
        return True
    ev = upcoming[0]
    rnd = ev["round"]
    pred = PRED_DIR / f"{year}_R{rnd:02d}.json"
    ready = ev["quali"] + QUALI_LENGTH + QUALI_DELAY
    to_race_h = round((ev["race"] - now).total_seconds() / 3600, 2)
    if pred.exists():
        _decide("predict", f"R{rnd} {ev['name']} already predicted, checking the starting grid",
                round=rnd, hours_to_race=to_race_h)
        old = _digest(pred)
        ok = _run(["scripts/race_weekend_inference.py", "--year", str(year), "--round", str(rnd),
                   "--only-if-grid-changed"], dry_run, "regrid")
        if _digest(pred) != old:
            _log_prediction(pred, "grid changed")
        else:
            _decide("regrid", f"R{rnd} prediction unchanged", round=rnd)
        return ok
    if now < ready:
        _decide("predict", f"R{rnd} {ev['name']} waits for qualifying (ready {ready:%Y-%m-%d %H:%M} UTC)",
                round=rnd, ready=ready, hours_to_race=to_race_h)
        return True
    _decide("predict", f"R{rnd} {ev['name']}", round=rnd,
            hours_after_quali_start=round((now - ev["quali"]).total_seconds() / 3600, 2), hours_to_race=to_race_h)
    ok = _run(["scripts/race_weekend_inference.py", "--year", str(year), "--round", str(rnd)], dry_run, "predict")
    if pred.exists():
        _log_prediction(pred, "first prediction")
    return ok


def _cron_delay(cron: str | None, now: datetime) -> float | None:
    """Minutes between the cron slot that fired this run and now (GitHub often runs late)."""
    if not cron:
        return None
    minute, hour = cron.split()[:2]
    if not minute.isdigit():
        return None
    slot = now.replace(minute=int(minute), second=0, microsecond=0)
    if hour.isdigit():
        slot = slot.replace(hour=int(hour))
        while slot > now:
            slot -= timedelta(days=1)
    elif slot > now:
        slot -= timedelta(hours=1)
    return round((now - slot).total_seconds() / 60, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="One tick of the race-weekend automation")
    parser.add_argument("--now", help="Pretend the current UTC time is this (ISO 8601), for testing")
    parser.add_argument("--dry-run", action="store_true", help="Print the steps without running them")
    parser.add_argument("--no-probe", action="store_true", help="Skip the source availability probes")
    args = parser.parse_args()
    # Event names like "São Paulo" must not crash a tick on a non-UTF-8 console
    sys.stdout.reconfigure(errors="replace")

    real_now = datetime.now(timezone.utc)
    now = datetime.fromisoformat(args.now.replace("Z", "+00:00")) if args.now else real_now
    year = now.year
    print(f"ci_tick at {now:%Y-%m-%d %H:%M} UTC, season {year}")
    t0 = time.monotonic()
    info = ci_log.start(year, now if args.now else None)
    ci_log.event("tick_start", **info, cron_delay_min=_cron_delay(info.get("cron"), real_now), dry_run=args.dry_run)

    try:
        events = _calendar(year)
    except Exception as exc:
        print(f"!! could not load the {year} calendar: {exc}")
        ci_log.event("decision", step="calendar", msg=f"could not load the {year} calendar: {exc}")
        events = []

    ok = True
    if events:
        if not args.no_probe:
            probe(year, events, now)
        done = [e for e in events if e["race"] + RESULTS_DELAY <= now]
        ok &= ingest(year, done, now, args.dry_run)
        ok &= predict(year, events, now, args.dry_run)
        # Live books only make sense at the real time: not in dry runs or simulated ticks
        if not args.dry_run and not args.now:
            market_snapshot(year, events, now)
    old_site = _digest(SITE_JSON)
    ok &= _run(["scripts/export_site_json.py", "--year", str(year)], args.dry_run, "export")
    ci_log.event("data", what="site_json", changed=_digest(SITE_JSON) != old_site)
    ci_log.event("tick_end", ok=bool(ok), seconds=round(time.monotonic() - t0, 1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
