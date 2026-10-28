"""
Timeline of the race-weekend automation from data/logs/ci_<season>.jsonl.

For one round: every tick (when it ran, cron delay, result), when each source
first had data and when it changed (starting grid order, qualifying order),
the decisions and failed steps, and what each prediction used.

Usage:
    python scripts/ci_log_report.py --year 2026 --round 19
    python scripts/ci_log_report.py --year 2026 --round 19 --ticks   # list every tick too
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src import ci_log  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Race-weekend automation timeline")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", type=int, required=True, dest="rnd")
    parser.add_argument("--ticks", action="store_true", help="List every tick")
    args = parser.parse_args()

    log = ci_log.read(args.year)
    if not log:
        print(f"No log for {args.year}")
        return
    runs = {e["run_id"] for e in log if e.get("round") == args.rnd}
    rows = [e for e in log if e["run_id"] in runs]

    weekend = next((e for e in reversed(rows) if e["kind"] == "weekend" and e.get("round") == args.rnd), None)
    if weekend:
        print(f"R{args.rnd} {weekend['name']}: scheduled sessions (UTC)")
        for name, t in weekend["sessions"].items():
            print(f"  {t}  {name}")

    starts = [e for e in rows if e["kind"] == "tick_start"]
    ends = {e["run_id"]: e for e in rows if e["kind"] == "tick_end"}
    delays = [e["cron_delay_min"] for e in starts if e.get("cron_delay_min") is not None]
    print(f"\n{len(starts)} ticks touched R{args.rnd}"
          + (f"; cron delay min/median/max {min(delays)}/{sorted(delays)[len(delays) // 2]}/{max(delays)} min"
             if delays else ""))
    if args.ticks:
        for s in starts:
            end = ends.get(s["run_id"], {})
            print(f"  {s['ts']}  run {s['run_id']}  delay {s.get('cron_delay_min')} min  "
                  f"ok={end.get('ok')}  {end.get('seconds')} s  ({s['runner']})")

    # Source availability: first time each probe returned rows, and every change of payload
    print("\nSource availability (first seen, then changes):")
    seen: dict[tuple, tuple] = {}
    for e in rows:
        if e["kind"] != "probe" or e.get("round") != args.rnd:
            continue
        key = (e["source"], e["item"])
        state = (e.get("n"), str(e.get("detail")))
        if e.get("error"):
            print(f"  {e['ts']}  {key[0]:8} {key[1]:15} ERROR {e['error'][:120]}")
            continue
        if key not in seen:
            print(f"  {e['ts']}  {key[0]:8} {key[1]:15} n={e.get('n')}  (first probe)")
        elif seen[key] != state:
            detail = e.get("detail") or {}
            order = detail.get("order") if isinstance(detail, dict) else None
            print(f"  {e['ts']}  {key[0]:8} {key[1]:15} n={e.get('n')}"
                  + (f"  order {' '.join(map(str, order))}" if order else ""))
        seen[key] = state

    print("\nDecisions and steps:")
    for e in rows:
        if e["kind"] == "decision" and e.get("round") in (None, args.rnd):
            print(f"  {e['ts']}  {e['step']}: {e['msg']}")
        elif e["kind"] == "step" and not e.get("dry_run"):
            flag = "OK " if e.get("exit_code") == 0 else f"EXIT {e.get('exit_code')}"
            print(f"  {e['ts']}  {flag} {e['seconds']:>6}s  {e['cmd']}")
            if e.get("exit_code") != 0:
                for line in e.get("tail", [])[-8:]:
                    print(f"        | {line}")

    print("\nPredictions:")
    for e in rows:
        if e["kind"] == "prediction" and e.get("round") == args.rnd:
            inp = e.get("inputs") or {}
            print(f"  {e['ts']}  {e['reason']}: generated {e['generated_at']}, grid from {e.get('grid_source')}, "
                  f"sprint {(inp.get('sprint') or {}).get('source')}, weather {(inp.get('weather') or {}).get('source')}")
            print("        top: " + ", ".join(f"{d} {w:.0%}/{p:.0%}" for d, w, p in e.get("top", [])))
    for e in rows:
        if e["kind"] == "data" and e.get("what") == "result_vs_prediction" and e.get("round") == args.rnd:
            print(f"  {e['ts']}  scored against the result: {e.get('accuracy')}")

    print("\nKalshi snapshots (win market top 3):")
    for e in rows:
        if e["kind"] == "market_snapshot" and e.get("round") == args.rnd:
            if e.get("error"):
                print(f"  {e['ts']}  ERROR {e['error'][:120]}")
                continue
            top = (e.get("win") or {}).get("top3", [])
            print(f"  {e['ts']}  {e.get('minutes_to_race'):>7} min to start  "
                  + ", ".join(f"{d} {p:.0%}" for d, p in top))


if __name__ == "__main__":
    main()
