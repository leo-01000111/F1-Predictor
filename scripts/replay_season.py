"""
Replay a season's race predictions with the current production setup.

The 2026 record on the website was made in one batch on 2026-10-27 by the
archived v2 regressor, so it said nothing about the current model. This
re-predicts every round that has a result the way the live pipeline would
have at the time:

  model        TwoStageRaceModel with the production settings, trained on all
               seasons before `year` (the live model is retrained after each race
               but always on complete seasons only)
  temperature  fitted on season year-1, predicted by a model trained before it
               (as the walk-forward backtest does), so no 2026 result is used
  features     the race's row in the feature matrix: everything is computed
               from races before it, with the actual starting grid and sprint

Files are written in the live format with "replay": true; the old ones are
moved to data/predictions/archive/<family>/. Then each round is scored again
(update_race_result) and the site JSON can be exported.

Usage:
    python scripts/replay_season.py --year 2026
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.features.build_features import FEATURE_COLS, load_feature_matrix  # noqa: E402
from src.models import two_stage  # noqa: E402
from src.models.xgb_model import TwoStageRaceModel  # noqa: E402
from src.training.weights import compute_recency_sample_weight  # noqa: E402

PRED_DIR = ROOT / "data" / "predictions"
RES_DIR = ROOT / "data" / "results"


def _fit(train: pd.DataFrame, params: dict | None, n_seeds: int) -> TwoStageRaceModel:
    model = TwoStageRaceModel(params=params, n_seeds=n_seeds)
    model.fit(train[FEATURE_COLS], train["target_position"].values,
              sample_weight=compute_recency_sample_weight(train["year"].values),
              qid=(train["year"] * 100 + train["round"]).to_numpy(),
              dnf=pd.to_numeric(train["dnf"], errors="coerce").fillna(0).to_numpy())
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a season with the current model")
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()
    year = args.year

    manifest = json.load(open(ROOT / "models" / "active_model.json", encoding="utf-8"))
    params = manifest.get("ranker_params")
    n_seeds = int(manifest.get("n_seeds", 1))
    family = manifest.get("model_family", "two_stage")

    fm = load_feature_matrix()
    # Temperature: model trained before year-1, scored on year-1
    prev = fm[fm["year"] == year - 1].reset_index(drop=True)
    t_model = _fit(fm[fm["year"] < year - 1].reset_index(drop=True), params, n_seeds)
    t_model.fit_temperature(prev[FEATURE_COLS], prev)
    model = _fit(fm[fm["year"] < year].reset_index(drop=True), params, n_seeds)
    model.temperature = t_model.temperature
    print(f"Replay {year}: trained on <{year}, T={model.temperature} (fitted on {year - 1})")

    rounds = sorted(int(p.stem.split("_R")[1].split("_")[0]) for p in RES_DIR.glob(f"{year}_R*_actual.json"))
    season = fm[fm["year"] == year]
    for rnd in rounds:
        race = season[season["round"] == rnd].reset_index(drop=True)
        if race.empty:
            print(f"R{rnd}: no feature rows, skipped")
            continue
        X = race[FEATURE_COLS]
        pred, q = model.predict(X), model.predict_finish(X)
        pm = two_stage.simulate(pred, q, 1.0, n_sims=10_000)
        n = pm.shape[1]
        rows = []
        for i, r in race.iterrows():
            p1, p2, p3 = pm[i, 0], pm[i, 1], pm[i, 2]
            rows.append({
                "driver": r["driver"], "team": r["team"],
                "grid_position": int(r["grid_position"]) if pd.notna(r["grid_position"]) else None,
                "quali_gap_to_pole_s": float(r.get("quali_gap_to_pole_s", 0) or 0),
                "p_win": round(float(p1), 4), "p_p2": round(float(p2), 4), "p_p3": round(float(p3), 4),
                "p_podium": round(float(min(p1 + p2 + p3, 1.0)), 4),
                "p_top6": round(float(min(pm[i, :min(6, n)].sum(), 1.0)), 4),
                "p_top10": round(float(min(pm[i, :min(10, n)].sum(), 1.0)), 4),
                "p_finish": round(float(q[i]), 4),
            })
        path = PRED_DIR / f"{year}_R{rnd:02d}.json"
        old = json.load(open(path, encoding="utf-8")) if path.exists() else None
        if old and not old.get("replay"):
            arch = PRED_DIR / "archive" / (old.get("model", {}).get("family") or "unknown")
            arch.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, arch / path.name)
        doc = {
            "year": year, "round": rnd, "circuit_key": race["circuit_key"].iloc[0],
            "race_date": str(race["race_date"].iloc[0])[:10],
            "generated_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "replay": True,
            "replay_note": (f"Predicted after the fact with the {family} setup: trained on seasons before "
                            f"{year}, temperature fitted on {year - 1}, pre-race features only "
                            "(actual starting grid and sprint)."),
            "weather": {"is_wet_race": int(race["is_wet_race"].iloc[0])},
            "grid_source": "starting_grid",
            "drivers": race["driver"].tolist(), "teams": race["team"].tolist(),
            "model": {"family": family, "artifact": "replay", "sampler": f"two_stage(T={model.temperature})",
                      "ranker_params": params, "n_seeds": n_seeds},
            "predictions": sorted(rows, key=lambda x: -x["p_win"]),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        top = doc["predictions"][:3]
        print(f"R{rnd:02d}: " + ", ".join(f"{p['driver']} {p['p_win']:.0%}" for p in top))
        # Score it again against the stored result
        res = RES_DIR / f"{year}_R{rnd:02d}_actual.json"
        backup = res.with_suffix(".json.bak")
        shutil.move(res, backup)
        proc = subprocess.run([sys.executable, "scripts/update_race_result.py", "--year", str(year),
                               "--round", str(rnd)], cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0 or not res.exists():
            shutil.move(backup, res)
            raise SystemExit(f"R{rnd}: scoring failed, old result restored\n{proc.stdout[-2000:]}{proc.stderr[-2000:]}")
        backup.unlink()


if __name__ == "__main__":
    main()
