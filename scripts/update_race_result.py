"""
Fetch actual race result and compare with prediction file.

Usage:
    python scripts/update_race_result.py --year 2026 --round 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.actions import InlineContext, PipelineActionRunner


def main(year: int, round_num: int) -> None:
    runner = PipelineActionRunner(root=ROOT)

    def on_progress(value: int, message: str) -> None:
        print(f"[{value:3d}%] {message}")

    def on_log(message: str) -> None:
        print(f"  {message}")

    result = runner.record_actual_result(
        year=year,
        round_num=round_num,
        context=InlineContext(progress_cb=on_progress, log_cb=on_log),
    )

    print("\nSummary")
    print("-------")
    print(f"Result file: {result.get('result_file')}")
    print(f"Winner hit: {result.get('winner_hit')}")
    print(f"Podium overlap: {result.get('podium_overlap')}/3")
    print(f"ECE P1 (round): {result.get('ece_p1_round')}")
    print(f"ECE P1 (season mean): {result.get('ece_p1_season_mean')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch actual race result and compare to predictions")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", type=int, required=True, dest="round_num")
    args = parser.parse_args()
    main(year=args.year, round_num=args.round_num)
