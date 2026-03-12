#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Post-rate-limit pipeline: collect 2020-2024 data, update weather,
retrain model, re-run inference for 2026 R1.
Sleeps until 14:46:00 local time before starting.
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
LOG = ROOT / "data" / "pipeline_log.txt"


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    safe_line = line.encode("cp1250", errors="replace").decode("cp1250")
    print(safe_line, flush=True)
    with open(LOG, "a", encoding="utf-8", errors="replace") as f:
        f.write(line + "\n")


def run(cmd, description):
    log(f"START: {description}")
    log(f"CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.stdout:
        log(f"STDOUT:\n{result.stdout[-3000:]}")
    if result.stderr:
        log(f"STDERR:\n{result.stderr[-2000:]}")
    if result.returncode != 0:
        log(f"FAILED with exit code {result.returncode}")
        sys.exit(result.returncode)
    log(f"DONE: {description}")
    return result


def main():
    # Wait until 14:46:00 (rate limit from last run expires ~14:44:40)
    target = datetime(2026, 3, 7, 14, 46, 0)
    now = datetime.now()
    wait = (target - now).total_seconds()
    if wait > 0:
        log(f"Sleeping {wait:.0f}s until {target.strftime('%H:%M:%S')} for rate limit reset...")
        time.sleep(wait)
    else:
        log("Rate limit already expired, starting immediately.")

    log("=== PIPELINE START ===")

    # Step 1: Collect 2020-2024 data (merge with existing 2014-2019)
    run(
        [sys.executable, "-m", "src.data_collection.f1_historical",
         "--start", "2020", "--end", "2024", "--no-fp", "--sleep", "0.3", "--merge"],
        "F1 data collection 2020-2024 (merge mode)"
    )

    # Step 2: Weather for all races (re-runs for everything including new years)
    run(
        [sys.executable, "-m", "src.data_collection.weather"],
        "Weather collection"
    )

    # Step 3: Retrain model
    run(
        [sys.executable, "scripts/quick_train.py"],
        "Model training"
    )

    # Step 4: Re-run inference for 2026 R1
    run(
        [sys.executable, "scripts/race_weekend_inference.py",
         "--year", "2026", "--round", "1", "--xgb-only"],
        "Inference 2026 R1 Australian GP"
    )

    log("=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()