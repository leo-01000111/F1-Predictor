"""
Event log for the race-weekend automation (scripts/ci_tick.py).

Every tick appends JSON lines to data/logs/ci_<season>.jsonl, which the
workflow commits, so after a weekend we can read exactly what ran, when, what
it decided and what data each source had at that moment:

  tick_start  actual start, the cron slot it was scheduled for and the delay
  decision    what ci_tick chose to do (or not) and why
  probe       one source checked for one round: how many rows it had
              (starting grid, qualifying, sprint, results ...), errors included
  step        a command: exit code, duration and the tail of its output
  data        row counts / files before and after a step
  prediction  what a prediction used (grid source, sprint, weather) and its top picks
  tick_end    overall result

scripts/ci_log_report.py turns the log into a per-round timeline
("starting grid first seen at ...").

Set CI_LOG_PATH to write elsewhere (tests); CI_LOG_PATH=off disables logging.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
LOG_DIR = ROOT / "data" / "logs"

_context: dict[str, Any] = {}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _path(season: int) -> Path | None:
    override = os.environ.get("CI_LOG_PATH")
    if override == "off":
        return None
    return Path(override) if override else LOG_DIR / f"ci_{season}.jsonl"


def _scheduled_cron() -> str | None:
    """The cron expression that triggered this run (schedule events only)."""
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        return None
    try:
        with open(event_path, encoding="utf-8") as f:
            return json.load(f).get("schedule")
    except (OSError, json.JSONDecodeError):
        return None


def start(season: int, simulated_now: datetime | None = None) -> dict:
    """Set the run context every later event carries; returns the tick_start fields."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        sha = None
    _context.clear()
    _context.update({
        "season": season,
        "run_id": os.environ.get("GITHUB_RUN_ID") or f"local-{_now()}",
        "runner": "github" if os.environ.get("GITHUB_ACTIONS") else "local",
    })
    info = {
        "event": os.environ.get("GITHUB_EVENT_NAME") or "manual",
        "cron": _scheduled_cron(),
        "sha": sha,
        "simulated_now": simulated_now.strftime("%Y-%m-%dT%H:%M:%SZ") if simulated_now else None,
    }
    return info


def event(kind: str, **data: Any) -> None:
    """Append one event. Never raises: logging must not break a tick."""
    season = _context.get("season") or datetime.now(timezone.utc).year
    path = _path(season)
    if path is None:
        return
    rec = {"ts": _now(), "kind": kind, "run_id": _context.get("run_id", "local"),
           "runner": _context.get("runner", "local"), **data}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        print(f"!! ci_log write failed: {exc}")


def read(season: int) -> list[dict]:
    path = _path(season)
    if path is None or not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
