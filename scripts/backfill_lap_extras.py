"""
Backfill lap-1 position and start compound for all historical races.

These two data points activate two previously zero-filled features:
  - driver_start_delta_avg  : rolling start performance (needs lap1_position)
  - start_compound_code     : race start tire (needs start_compound)

NOTE: FastF1 lap-by-lap data is only available from 2018 onwards.
Sessions before 2018 will print "API not supported" warnings and return no data.
Always use --start 2018 (the default).

This script loads each race session with lap data (heavier than R+Q-only),
so budget ~2-4s per round = 10-15 minutes for a 2018-2025 backfill.
Progress is saved after each year so you can interrupt and resume.

Usage:
    # Full backfill (2018-2025, ~12 min):
    python scripts/backfill_lap_extras.py

    # Resume / add new year only:
    python scripts/backfill_lap_extras.py --start 2025 --end 2025 --merge

    # After backfill, rebuild features and retrain:
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
from src.data_collection.f1_historical import collect_lap_extras, load_lap_extras


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill lap-1 position and start compound for all historical races"
    )
    parser.add_argument("--start", type=int, default=2018, help="Start year (default 2018 — FastF1 lap data unavailable before 2018)")
    parser.add_argument("--end", type=int, default=2025, help="End year inclusive (default 2025)")
    parser.add_argument("--sleep", type=float, default=8.0,
                        help="Sleep seconds between sessions (default 8.0 — FastF1 API is 500 calls/h)")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Skip rounds already present in lap_extras.parquet (resume mode)",
    )
    args = parser.parse_args()

    logger.info(f"Backfilling lap extras: {args.start}-{args.end} (merge={args.merge})")
    df = collect_lap_extras(
        start_year=args.start,
        end_year=args.end,
        sleep_between_sessions=args.sleep,
        merge_existing=args.merge,
    )

    if df.empty:
        logger.warning("No lap extras collected.")
        return

    logger.info(f"Done. {len(df)} rows, years {sorted(df['year'].unique().tolist())}")

    has_lap1 = df["lap1_position"].notna().sum()
    has_compound = df["start_compound"].notna().sum()
    logger.info(f"  lap1_position filled: {has_lap1}/{len(df)} rows ({has_lap1/len(df):.1%})")
    logger.info(f"  start_compound filled: {has_compound}/{len(df)} rows ({has_compound/len(df):.1%})")

    logger.info("\nNext steps:")
    logger.info("  python -m src.features.build_features")
    logger.info("  python scripts/quick_train.py")


if __name__ == "__main__":
    main()
