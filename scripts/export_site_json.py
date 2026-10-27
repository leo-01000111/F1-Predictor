"""
Export a compact JSON for the website (leongorecki.eu/projects/f1predictor.html).

Reads data/predictions/, data/results/, models/active_model.json and
models/holdout_metrics.json and writes site/f1-predictions.json:

  season          current season
  model           when the active model was trained + its holdout metrics
  next            the latest prediction without a result yet, or, if there
                  is none, the next race on the calendar (awaiting qualifying)
  last            the latest race with a result: prediction vs actual finish
  record          one row per predicted race with a result (season so far)

The website's update-f1.yml workflow copies this file into the site repo.

Usage:
    python scripts/export_site_json.py [--year 2026]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PRED_DIR = ROOT / "data" / "predictions"
RES_DIR = ROOT / "data" / "results"
MODELS_DIR = ROOT / "models"
OUT_PATH = ROOT / "site" / "f1-predictions.json"


def _load(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _calendar(year: int) -> dict[int, dict]:
    """Round -> {name, location, quali_utc, race_utc} from the FastF1 calendar."""
    try:
        import fastf1
        from src.data_collection import f1_historical  # noqa: F401  (enables the FastF1 cache)

        sched = fastf1.get_event_schedule(year, include_testing=False)
    except Exception:
        return {}
    out: dict[int, dict] = {}
    for _, ev in sched.iterrows():
        sessions = {}
        for i in range(1, 6):
            name, t = ev.get(f"Session{i}"), ev.get(f"Session{i}DateUtc")
            if name and t is not None and str(t) != "NaT":
                sessions[str(name)] = t.strftime("%Y-%m-%dT%H:%M:%SZ")
        out[int(ev["RoundNumber"])] = {
            "name": str(ev["EventName"]),
            "location": str(ev["Location"]),
            "quali_utc": sessions.get("Qualifying"),
            "race_utc": sessions.get("Race"),
        }
    return out


def _driver_rows(pred: dict) -> list[dict]:
    rows = sorted(pred.get("predictions", []), key=lambda p: -p.get("p_win", 0))
    return [
        {
            "driver": p["driver"],
            "team": p.get("team"),
            "grid": p.get("grid_position"),
            "p_win": round(float(p.get("p_win", 0)), 4),
            "p_podium": round(float(p.get("p_podium", 0)), 4),
            "p_top10": round(float(p.get("p_top10", 0)), 4),
        }
        for p in rows
    ]


def build(year: int) -> dict:
    cal = _calendar(year)
    preds = {int(p.stem.split("_R")[1]): p for p in sorted(PRED_DIR.glob(f"{year}_R*.json"))}
    results = {int(p.stem.split("_R")[1].split("_")[0]): p for p in sorted(RES_DIR.glob(f"{year}_R*_actual.json"))}

    def race_meta(rnd: int, doc: dict | None = None) -> dict:
        c = cal.get(rnd, {})
        return {
            "round": rnd,
            "name": c.get("name") or (doc or {}).get("circuit_key", "").replace("_", " ").title(),
            "location": c.get("location"),
            "quali_utc": c.get("quali_utc"),
            "race_utc": c.get("race_utc"),
        }

    # Season record: predicted races that have a result
    record = []
    for rnd in sorted(results):
        res = _load(results[rnd]) or {}
        acc = res.get("accuracy") or {}
        record.append({
            **{k: v for k, v in race_meta(rnd, res).items() if k in ("round", "name")},
            "winner_predicted": acc.get("winner_predicted"),
            "winner_actual": acc.get("winner_actual"),
            "winner_hit": bool(acc.get("winner_hit")),
            "podium_overlap": acc.get("podium_drivers_in_top3_predicted"),
            "brier_p1": acc.get("brier_score_p1"),
        })

    # Next: newest prediction without a result; else the next race on the calendar
    open_rounds = [r for r in preds if r not in results]
    nxt: dict
    if open_rounds:
        rnd = max(open_rounds)
        doc = _load(preds[rnd]) or {}
        nxt = {
            "status": "predicted",
            **race_meta(rnd, doc),
            "generated_at": doc.get("generated_at"),
            "is_wet_forecast": bool((doc.get("weather") or {}).get("is_wet_race", 0)),
            "drivers": _driver_rows(doc),
        }
    else:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        upcoming = [r for r, c in sorted(cal.items()) if c.get("race_utc") and c["race_utc"] > now]
        nxt = {"status": "awaiting_qualifying", **race_meta(upcoming[0])} if upcoming else {"status": "season_over"}

    # Last: latest race with a result, prediction next to the actual finish
    last = None
    if results:
        rnd = max(results)
        res = _load(results[rnd]) or {}
        finish = {a["driver"]: a.get("finish_position") for a in res.get("actual_results", [])}
        acc = res.get("accuracy") or {}
        drivers = _driver_rows(res)
        # The result file stores both podiums alphabetically: rank them instead
        podium_pred = [d["driver"] for d in sorted(drivers, key=lambda d: -d["p_podium"])[:3]]
        podium_act = [d for d, f in sorted(finish.items(), key=lambda kv: kv[1] or 99) if f and f <= 3]
        last = {
            **race_meta(rnd, res),
            "winner_hit": bool(acc.get("winner_hit")),
            "podium_predicted": podium_pred,
            "podium_actual": podium_act,
            "podium_overlap": acc.get("podium_drivers_in_top3_predicted"),
            "drivers": [{**d, "finish": finish.get(d["driver"])} for d in drivers],
        }

    active = _load(MODELS_DIR / "active_model.json") or {}
    metrics = _load(MODELS_DIR / "holdout_metrics.json") or {}
    n = len(record)
    return {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season": year,
        "model": {
            "trained_at": active.get("generated_at"),
            "train_years": [min(active.get("train_years", [0])), max(active.get("train_years", [0]))],
            "holdout_years": [int(y) for y in metrics.get("holdout_years", [])],
            "holdout_races": metrics.get("n_holdout_races"),
            "winner_accuracy": metrics.get("winner_accuracy"),
            "podium_accuracy": metrics.get("podium_accuracy"),
            "spearman": metrics.get("mean_spearman_r"),
        },
        "next": nxt,
        "last": last,
        "record": record,
        "record_summary": {
            "races": n,
            "winners": sum(r["winner_hit"] for r in record),
            "podium_overlap_avg": round(sum((r["podium_overlap"] or 0) for r in record) / n, 2) if n else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export site/f1-predictions.json for the website")
    parser.add_argument("--year", type=int, default=datetime.now(timezone.utc).year)
    args = parser.parse_args()
    doc = build(args.year)
    # Leave the file untouched when only the timestamp would change, so the
    # hourly CI tick does not commit (and the website does not redeploy) for nothing
    old = _load(OUT_PATH)
    if old is not None and {**old, "updated_at": None} == {**doc, "updated_at": None}:
        print(f"{OUT_PATH.relative_to(ROOT)} unchanged")
        return
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    print(f"Wrote {OUT_PATH.relative_to(ROOT)} (next: {doc['next'].get('status')}, record: {len(doc['record'])} races)")


if __name__ == "__main__":
    main()
