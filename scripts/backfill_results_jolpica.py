# -*- coding: utf-8 -*-
"""
Backfill missing race and qualifying results from Jolpica
==========================================================
FastF1 left holes in data/raw: 2023 stopped after round 11, and several 2022-23
rounds have rows with no grid/finish (race) or no position/times (quali).
This script finds rounds that are missing or empty and refills only those from
the Jolpica (Ergast-compatible) API. Complete rounds are not touched.

It also relabels `dnf` on every race row from `status` with `is_dnf()`, which
fixes finishers marked "Lapped" (Jolpica since late 2023) being counted as DNF.

Usage:
    python scripts/backfill_results_jolpica.py --start 2022 --end 2023
    python scripts/backfill_results_jolpica.py --start 2014 --end 2026 --dry-run

After running, rebuild the feature matrix and retrain:
    python -m src.data_collection.weather --years 2023
    python scripts/quick_train.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import fastf1
import pandas as pd
from loguru import logger

from src.data_collection.f1_historical import RAW_DIR, _jolpica_get, is_dnf
from src.features.teams import canonical_team

BASE = "https://api.jolpi.ca/ergast/f1"



def _lap_time_s(t: str | None) -> float | None:
    """'1:16.609' -> 76.609; empty -> None."""
    if not t:
        return None
    try:
        if ":" in t:
            m, s = t.split(":")
            return int(m) * 60 + float(s)
        return float(t)
    except ValueError:
        return None


def _team(constructor: dict) -> str:
    return canonical_team(constructor.get("constructorId") or constructor.get("name", ""))


def _driver(d: dict) -> str:
    return d.get("code") or d.get("familyName", "")[:3].upper()


def _event_info(schedule: pd.DataFrame, rnd: int) -> dict:
    """circuit / circuit_key / race_date in the same format f1_historical uses."""
    ev = schedule[schedule["RoundNumber"] == rnd].iloc[0]
    return {
        "circuit": ev["Location"],
        "circuit_key": ev["EventName"].lower().replace(" ", "_").replace("grand_prix", "gp"),
        "race_date": str(ev["EventDate"].date()),
    }


def fetch_race(year: int, rnd: int, info: dict, sleep: float) -> pd.DataFrame:
    data = _jolpica_get(f"{BASE}/{year}/{rnd}/results.json?limit=100", sleep)
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        return pd.DataFrame()
    rows = []
    for r in races[0]["Results"]:
        pos = int(r["position"])
        grid = int(r["grid"])
        rows.append({
            "year": year,
            "round": rnd,
            **info,
            "driver": _driver(r["Driver"]),
            "team": _team(r["Constructor"]),
            # Jolpica uses 0 for a pit-lane start; FastF1 kept that too
            "grid_position": grid,
            "finish_position": pos,
            "dnf": is_dnf(r["status"]),
            "status": r["status"],
            "points": float(r["points"]),
            "p1": int(pos == 1),
            "p2": int(pos == 2),
            "p3": int(pos == 3),
            "podium": int(pos <= 3),
            "quali_gap_to_pole_s": None,
        })
    return pd.DataFrame(rows)


def fetch_quali(year: int, rnd: int, info: dict, sleep: float) -> pd.DataFrame:
    data = _jolpica_get(f"{BASE}/{year}/{rnd}/qualifying.json?limit=100", sleep)
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        return pd.DataFrame()
    results = races[0]["QualifyingResults"]
    # Best time = latest session the driver reached (Q3, else Q2, else Q1),
    # same rule as f1_historical._extract_quali_result
    best = [
        next((_lap_time_s(r.get(q)) for q in ("Q3", "Q2", "Q1") if _lap_time_s(r.get(q)) is not None), None)
        for r in results
    ]
    pole = min((t for t in best if t is not None), default=None)
    rows = []
    for r, t in zip(results, best):
        rows.append({
            "year": year,
            "round": rnd,
            "circuit": info["circuit"],
            "circuit_key": info["circuit_key"],
            "driver": _driver(r["Driver"]),
            "team": _team(r["Constructor"]),
            "quali_position": int(r["position"]),
            "quali_best_time_s": t,
            "quali_gap_to_pole_s": (t - pole) if (t is not None and pole is not None) else None,
        })
    return pd.DataFrame(rows)


def _complete_rounds(df: pd.DataFrame, year: int, cols: list[str]) -> set[int]:
    """Rounds of `year` where every value column has at least one non-null."""
    y = df[df["year"] == year]
    ok = y.groupby("round")[cols].apply(lambda g: g.notna().any().all())
    return set(ok[ok].index.astype(int))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--start", type=int, default=2022)
    parser.add_argument("--end", type=int, default=2023)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true", help="Only list the rounds that would be refilled")
    args = parser.parse_args()

    race_path = RAW_DIR / "race_results.parquet"
    quali_path = RAW_DIR / "quali_results.parquet"
    race_df = pd.read_parquet(race_path)
    quali_df = pd.read_parquet(quali_path)

    new_race, new_quali = [], []
    for year in range(args.start, args.end + 1):
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        today = pd.Timestamp.today().normalize()
        rounds = [
            int(r) for r, d in zip(schedule["RoundNumber"], schedule["EventDate"])
            if r >= 1 and pd.Timestamp(d) < today
        ]
        race_ok = _complete_rounds(race_df, year, ["grid_position", "finish_position"])
        quali_ok = _complete_rounds(quali_df, year, ["quali_position"])
        race_todo = [r for r in rounds if r not in race_ok]
        quali_todo = [r for r in rounds if r not in quali_ok]
        logger.info(f"{year}: race rounds to refill {race_todo}")
        logger.info(f"{year}: quali rounds to refill {quali_todo}")
        if args.dry_run:
            continue

        for rnd in sorted(set(race_todo) | set(quali_todo)):
            info = _event_info(schedule, rnd)
            if rnd in race_todo:
                df = fetch_race(year, rnd, info, args.sleep)
                logger.info(f"  {year} R{rnd} race: {len(df)} rows")
                if not df.empty:
                    new_race.append(df)
            if rnd in quali_todo:
                df = fetch_quali(year, rnd, info, args.sleep)
                logger.info(f"  {year} R{rnd} quali: {len(df)} rows")
                if not df.empty:
                    new_quali.append(df)

    if args.dry_run:
        n = int((race_df["status"].eq("Lapped") & race_df["dnf"].eq(1)).sum())
        logger.info(f"Dry run: nothing written. {n} 'Lapped' rows currently labelled dnf=1.")
        return

    def _replace(old: pd.DataFrame, new: list[pd.DataFrame]) -> pd.DataFrame:
        if not new:
            return old
        new_df = pd.concat(new, ignore_index=True)
        keys = set(zip(new_df["year"], new_df["round"]))
        keep = old[[(y, r) not in keys for y, r in zip(old["year"], old["round"])]]
        out = pd.concat([keep, new_df], ignore_index=True)
        return out.sort_values(["year", "round"]).reset_index(drop=True)

    race_df = _replace(race_df, new_race)
    quali_df = _replace(quali_df, new_quali)

    before = int(race_df["dnf"].sum())
    race_df["dnf"] = race_df["status"].map(is_dnf).astype(int)
    logger.info(f"DNF relabel: {before} -> {int(race_df['dnf'].sum())} DNF rows")

    race_df.to_parquet(race_path, index=False)
    quali_df.to_parquet(quali_path, index=False)
    logger.info(
        f"Saved {len(race_df)} race rows, {len(quali_df)} quali rows. "
        f"Rounds per year: {race_df.groupby('year')['round'].nunique().to_dict()}"
    )


if __name__ == "__main__":
    main()
