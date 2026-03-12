"""
Season accuracy summary.

Reads all recorded actual results from data/results/ and prints a per-round
accuracy table plus season-level calibration metrics. Use this to monitor
whether model accuracy and calibration are improving as the season progresses
and the seasonal recalibrator accumulates data.

Usage:
    python scripts/season_accuracy.py --year 2026
    python scripts/season_accuracy.py --year 2025   # review past season
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _brier(probs: list[float], labels: list[int]) -> float:
    if not probs:
        return float("nan")
    return float(np.mean([(p - y) ** 2 for p, y in zip(probs, labels)]))


def _ece(probs: list[float], labels: list[int], n_bins: int = 10) -> float:
    if not probs:
        return float("nan")
    p = np.array(probs)
    y = np.array(labels, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    bucket = np.digitize(p, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = bucket == b
        if not np.any(mask):
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        ece += abs(acc - conf) * float(mask.sum()) / len(p)
    return float(ece)


def load_results(year: int, results_dir: Path) -> list[dict]:
    docs = []
    for path in sorted(results_dir.glob(f"{year}_R*_actual.json")):
        try:
            with open(path, encoding="utf-8") as f:
                docs.append(json.load(f))
        except Exception:
            continue
    return docs


def summarise(year: int, results_dir: Path) -> None:
    docs = load_results(year, results_dir)
    if not docs:
        print(f"No recorded results found for {year} in {results_dir}")
        print(f"After each race run: python scripts/update_race_result.py --year {year} --round N")
        return

    print(f"\n{'='*70}")
    print(f"  Season accuracy summary -- {year}  ({len(docs)} race(s) recorded)")
    print(f"{'='*70}")
    print(f"{'Rnd':<5} {'Circuit':<22} {'Winner':^10} {'Pod':^5} {'Brier P1':^10} {'ECE P1':^10}")
    print(f"{'-'*5} {'-'*22} {'-'*10} {'-'*5} {'-'*10} {'-'*10}")

    all_p1_probs: list[float] = []
    all_p1_labels: list[int] = []
    winner_hits = 0

    for doc in docs:
        rnd = int(doc.get("round", 0))
        circuit = str(doc.get("circuit_key", "")).replace("_", " ").title()[:22]
        acc = doc.get("accuracy", {}) or {}
        winner_hit = bool(acc.get("winner_hit", False))
        podium_overlap = int(acc.get("podium_drivers_in_top3_predicted", 0))
        brier_p1 = acc.get("brier_score_p1", float("nan"))
        ece_p1 = acc.get("ece_p1_round", float("nan"))

        winner_hits += int(winner_hit)

        # Collect driver-level probs for cumulative stats
        for pred_row in doc.get("predictions", []):
            p = pred_row.get("p_win")
            af = pred_row.get("actual_finish")
            if p is not None and af is not None:
                try:
                    all_p1_probs.append(float(np.clip(float(p), 0, 1)))
                    all_p1_labels.append(1 if int(af) == 1 else 0)
                except Exception:
                    pass

        winner_str = "YES" if winner_hit else "no"
        brier_str = f"{brier_p1:.4f}" if isinstance(brier_p1, float) and not np.isnan(brier_p1) else "  n/a  "
        ece_str = f"{ece_p1:.4f}" if isinstance(ece_p1, float) and not np.isnan(ece_p1) else "  n/a  "
        print(f"{rnd:<5} {circuit:<22} {winner_str:^10} {podium_overlap:^5} {brier_str:^10} {ece_str:^10}")

    print(f"{'-'*70}")
    n = len(docs)
    winner_acc = winner_hits / n if n > 0 else 0.0
    cumulative_brier = _brier(all_p1_probs, all_p1_labels)
    cumulative_ece = _ece(all_p1_probs, all_p1_labels)
    print(f"\nSeason totals ({n} race(s)):")
    print(f"  Winner accuracy        : {winner_hits}/{n} = {winner_acc:.1%}")
    print(f"  Cumulative Brier P1    : {cumulative_brier:.4f}  (lower is better; random=0.0476 for 21-car field)")
    print(f"  Cumulative ECE P1      : {cumulative_ece:.4f}  (lower is better; 0=perfectly calibrated)")

    # Seasonal recalibration check
    print(f"\nSeasonal recalibration status:")
    min_samples_needed = 22 * 4   # approx 4 races with 22-driver grid
    if len(all_p1_probs) >= min_samples_needed:
        print(f"  ACTIVE -- {len(all_p1_probs)} driver-race samples available (threshold: {min_samples_needed})")
    else:
        remaining = min_samples_needed - len(all_p1_probs)
        races_remaining = int(np.ceil(remaining / 22))
        print(
            f"  PENDING -- {len(all_p1_probs)}/{min_samples_needed} samples; "
            f"recalibrator activates after ~{races_remaining} more race(s)"
        )

    # Trend (if enough data)
    if len(docs) >= 3:
        early_docs = docs[:len(docs)//2]
        late_docs = docs[len(docs)//2:]

        def _winner_acc_for(ds: list[dict]) -> float:
            hits = sum(1 for d in ds if d.get("accuracy", {}).get("winner_hit", False))
            return hits / len(ds) if ds else 0.0

        early_acc = _winner_acc_for(early_docs)
        late_acc = _winner_acc_for(late_docs)
        trend = "improving" if late_acc > early_acc else ("flat" if late_acc == early_acc else "declining")
        print(f"\nAccuracy trend:")
        print(f"  First half : {early_acc:.1%}  ({len(early_docs)} race(s))")
        print(f"  Second half: {late_acc:.1%}  ({len(late_docs)} race(s))")
        print(f"  Trend      : {trend}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Season accuracy summary for F1 predictions")
    parser.add_argument("--year", type=int, default=2026, help="Season year (default 2026)")
    args = parser.parse_args()

    results_dir = ROOT / "data" / "results"
    summarise(args.year, results_dir)


if __name__ == "__main__":
    main()
