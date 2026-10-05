# -*- coding: utf-8 -*-
"""
Backfill Pit Stop and Sprint Race Data
=======================================
Collects two new data sources needed for Tier 2 features:

  1. Pit stop durations via Jolpica (Ergast-compatible) API
     -> data/raw/pit_stops.parquet
     -> used to compute team_pit_delta_s feature

  2. Sprint race results via FastF1
     -> data/raw/sprint_results.parquet
     -> used to compute sprint_position_rel feature
     -> sprints started in 2021; only sprint-format weekends collected

Usage:
    # Collect both (recommended first run):
    python scripts/backfill_pit_sprint.py --start 2014 --end 2025 --merge

    # Pit stops only:
    python scripts/backfill_pit_sprint.py --pit --start 2014 --end 2025 --merge

    # Sprint results only:
    python scripts/backfill_pit_sprint.py --sprint --start 2021 --end 2025 --merge

After running, rebuild the feature matrix and retrain:
    python -m src.features.build_features
    python scripts/quick_train.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill pit stop and sprint race data")
    parser.add_argument("--start", type=int, default=2014, help="Start year for pit stops (default 2014)")
    parser.add_argument("--end", type=int, default=2025, help="End year (default 2025)")
    parser.add_argument("--sleep", type=float, default=1.5,
                        help="Sleep between API calls in seconds (default 1.5). "
                             "Jolpica rate-limits at ~4 req/s; 1.5s is conservative. "
                             "On 429 the script backs off automatically.")
    parser.add_argument("--pit", action="store_true", help="Collect pit stop data only")
    parser.add_argument("--sprint", action="store_true", help="Collect sprint results only")
    parser.add_argument("--merge", action="store_true", help="Skip rounds already in parquet")
    args = parser.parse_args()

    # Default: collect both if neither flag specified
    do_pit = args.pit or (not args.pit and not args.sprint)
    do_sprint = args.sprint or (not args.pit and not args.sprint)

    from src.data_collection.f1_historical import collect_pit_data, collect_sprint_results

    if do_pit:
        logger.info(f"Collecting pit stop data {args.start}-{args.end}...")
        pit_df = collect_pit_data(
            start_year=args.start,
            end_year=args.end,
            sleep_between=args.sleep,
            merge_existing=args.merge,
        )
        logger.info(f"Pit stop collection done: {len(pit_df)} rows")

    if do_sprint:
        sprint_start = max(args.start, 2021)
        logger.info(f"Collecting sprint results {sprint_start}-{args.end} (with lap-pace data)...")
        sprint_df = collect_sprint_results(
            start_year=sprint_start,
            end_year=args.end,
            sleep_between_sessions=args.sleep * 3,
            merge_existing=args.merge,
            include_laps=True,
        )
        has_pace = "sprint_best_lap_gap_s" in sprint_df.columns
        filled = sprint_df["sprint_best_lap_gap_s"].notna().sum() if has_pace else 0
        logger.info(f"Sprint collection done: {len(sprint_df)} rows | pace data: {filled} driver-rounds")

    logger.info("Done. Now rebuild features: python -m src.features.build_features")


if __name__ == "__main__":
    main()
