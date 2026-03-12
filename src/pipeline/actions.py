from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from src.models.calibration import expected_calibration_error


class ActionContext(Protocol):
    def progress(self, value: int, message: str = "") -> None: ...

    def log(self, message: str) -> None: ...

    def check_cancelled(self) -> None: ...


@dataclass(slots=True)
class InlineContext:
    """Simple context for synchronous/non-cancellable callers."""

    progress_cb: Any = None
    log_cb: Any = None

    def progress(self, value: int, message: str = "") -> None:
        if callable(self.progress_cb):
            self.progress_cb(int(max(0, min(value, 100))), message)

    def log(self, message: str) -> None:
        if callable(self.log_cb):
            self.log_cb(message)

    def check_cancelled(self) -> None:
        return


class PipelineActionRunner:
    """Shared in-process wrappers for pipeline operations."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parent.parent.parent
        self.predictions_dir = self.root / "data" / "predictions"
        self.results_dir = self.root / "data" / "results"
        self.models_dir = self.root / "models"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_inference(self, *, year: int, round_num: int, context: ActionContext, **_kwargs) -> dict[str, Any]:
        from src.data_collection.f1_historical import load_race_results
        from src.data_collection.f1_live import (
            get_completed_season_race_results,
            get_live_practice_dataframe,
            get_live_quali_dataframe,
        )
        from src.data_collection.weather import get_race_day_forecast
        from src.inference.predict import run_inference

        context.progress(5, "Loading historical race data...")
        race_df_history = load_race_results()
        context.log(f"Loaded {len(race_df_history)} historical rows.")
        context.check_cancelled()

        if round_num > 1:
            context.progress(12, f"Fetching completed {year} races before round {round_num}...")
            inseason_df = get_completed_season_race_results(year, max_round=round_num)
            if not inseason_df.empty:
                existing_keys = set(zip(race_df_history["year"], race_df_history["round"]))
                inseason_new = inseason_df[
                    ~inseason_df.apply(lambda r: (r["year"], r["round"]) in existing_keys, axis=1)
                ]
                if not inseason_new.empty:
                    race_df_history = pd.concat([race_df_history, inseason_new], ignore_index=True).sort_values(
                        ["year", "round"]
                    )
                    context.log(
                        f"Appended {len(inseason_new)} in-season rows from {inseason_new['round'].nunique()} rounds."
                    )
        context.check_cancelled()

        context.progress(25, f"Fetching qualifying for {year} R{round_num}...")
        quali_df = get_live_quali_dataframe(year, round_num)
        if quali_df.empty:
            raise RuntimeError("No qualifying data returned. Is qualifying finished for this round?")
        context.log(f"Qualifying rows: {len(quali_df)} drivers.")
        context.check_cancelled()

        context.progress(45, "Fetching FP1/FP2/FP3 data...")
        practice_df = get_live_practice_dataframe(year, round_num)
        if practice_df.empty:
            context.log("No practice data found. Proceeding with median-imputed FP features.")
        else:
            context.log(f"Practice rows: {len(practice_df)} ({practice_df['driver'].nunique()} drivers).")
        context.check_cancelled()

        context.progress(62, "Fetching race-day weather...")
        circuit_key = quali_df["circuit_key"].iloc[0] if "circuit_key" in quali_df.columns else ""
        race_date = quali_df["race_date"].iloc[0] if "race_date" in quali_df.columns else ""
        weather_dict = get_race_day_forecast(circuit_key, race_date)
        if not weather_dict:
            weather_dict = {
                "temp_c_mean": 20.0,
                "precip_mm_total": 0.0,
                "windspeed_ms_mean": 5.0,
                "humidity_pct_mean": 50.0,
                "is_wet_race": 0,
            }
            context.log("Weather unavailable. Falling back to defaults.")
        else:
            context.log(
                "Weather "
                f"T={weather_dict.get('temp_c_mean', 0):.1f}C "
                f"rain={weather_dict.get('precip_mm_total', 0):.1f}mm "
                f"wind={weather_dict.get('windspeed_ms_mean', 0):.1f}m/s"
            )
        context.check_cancelled()

        context.progress(78, "Loading supplemental data (pit stops, sprints)...")
        pit_df = None
        sprint_df = None
        try:
            from src.data_collection.f1_historical import load_pit_stops
            pit_df = load_pit_stops()
            context.log(f"Pit stop data: {len(pit_df)} rows.")
        except FileNotFoundError:
            context.log("No pit_stops.parquet; team_pit_delta_s will be NaN.")
        try:
            from src.data_collection.f1_historical import load_sprint_results
            sprint_df = load_sprint_results()
            context.log(f"Sprint data: {len(sprint_df)} rows.")
        except FileNotFoundError:
            context.log("No sprint_results.parquet; sprint_position_rel will be 0.5.")

        context.progress(80, "Running model inference...")
        result = run_inference(
            year=year,
            round_num=round_num,
            quali_df=quali_df,
            race_df_history=race_df_history,
            weather_dict=weather_dict,
            practice_df=practice_df,
            pit_df=pit_df,
            sprint_df=sprint_df,
        )
        context.check_cancelled()

        path = self.predictions_dir / f"{year}_R{round_num:02d}.json"
        top = result["predictions"][0] if result.get("predictions") else {}
        context.log(f"Saved prediction file: {path}")
        context.log(f"Top pick: {top.get('driver', '?')} ({top.get('p_win', 0):.1%} win).")
        context.progress(100, "Inference completed.")
        return {
            "prediction_file": str(path),
            "top_driver": top.get("driver"),
            "top_p_win": top.get("p_win", 0.0),
            "model_family": (result.get("model") or {}).get("family"),
        }

    def record_actual_result(self, *, year: int, round_num: int, context: ActionContext) -> dict[str, Any]:
        import fastf1

        pred_path = self.predictions_dir / f"{year}_R{round_num:02d}.json"
        if not pred_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {pred_path}")

        context.progress(10, "Loading FastF1 race session...")
        session = fastf1.get_session(year, round_num, "R")
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        context.check_cancelled()

        context.progress(35, "Extracting official race results...")
        results = session.results
        if results is None or results.empty:
            raise RuntimeError(f"No race results found for {year} R{round_num}.")

        rows = []
        for _, row in results.iterrows():
            driver = str(row.get("Abbreviation", "")).strip()
            team = str(row.get("TeamName", "Unknown")).strip()
            pos = row.get("Position")
            try:
                finish_position = int(float(pos)) if pd.notna(pos) else None
            except Exception:
                finish_position = None
            rows.append({"driver": driver, "team": team, "finish_position": finish_position})
        actual_df = pd.DataFrame(rows).sort_values("finish_position").reset_index(drop=True)
        context.log(f"Actual winner: {actual_df.iloc[0]['driver'] if not actual_df.empty else 'N/A'}")
        context.check_cancelled()

        context.progress(60, "Comparing with prediction file...")
        with open(pred_path, encoding="utf-8") as f:
            pred_data = json.load(f)

        predictions = pred_data.get("predictions", [])
        actual_lookup = {r["driver"]: r["finish_position"] for _, r in actual_df.iterrows()}
        winner_actual_arr = actual_df[actual_df["finish_position"] == 1]["driver"].values
        winner_actual = winner_actual_arr[0] if len(winner_actual_arr) else None
        podium_actual = set(actual_df[actual_df["finish_position"].isin([1, 2, 3])]["driver"].tolist())
        pred_winner = predictions[0]["driver"] if predictions else None
        pred_podium = {
            p["driver"]
            for p in sorted(
                predictions,
                key=lambda x: float(
                    (x.get("p_win", 0.0) or 0.0)
                    + (x.get("p_p2", 0.0) or 0.0)
                    + (x.get("p_p3", 0.0) or 0.0)
                ),
                reverse=True,
            )[:3]
        }

        winner_hit = pred_winner == winner_actual
        podium_overlap = len(pred_podium & podium_actual)

        p1_probs = []
        p1_labels = []
        brier_scores = []
        for p in predictions:
            p_win = float(p.get("p_win", 0.0) or 0.0)
            actual_win = 1 if actual_lookup.get(p["driver"]) == 1 else 0
            brier_scores.append((p_win - actual_win) ** 2)
            p1_probs.append(p_win)
            p1_labels.append(actual_win)
            p["actual_finish"] = actual_lookup.get(p["driver"])

        brier_p1 = float(np.mean(brier_scores)) if brier_scores else None
        ece_p1_round = (
            expected_calibration_error(np.asarray(p1_probs, dtype=float), np.asarray(p1_labels, dtype=int), n_bins=10)
            if p1_probs
            else None
        )

        summary = {
            "winner_predicted": pred_winner,
            "winner_actual": winner_actual,
            "winner_hit": winner_hit,
            "podium_predicted": sorted(pred_podium),
            "podium_actual": sorted(podium_actual),
            "podium_drivers_in_top3_predicted": podium_overlap,
            "brier_score_p1": round(brier_p1, 4) if brier_p1 is not None else None,
            "ece_p1_round": round(float(ece_p1_round), 4) if ece_p1_round is not None else None,
        }

        result_doc = {**pred_data, "actual_results": actual_df.to_dict(orient="records"), "accuracy": summary}

        context.progress(85, "Saving comparison output...")
        out_path = self.results_dir / f"{year}_R{round_num:02d}_actual.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result_doc, f, indent=2)

        season_ece = self._season_ece_summary(year)
        if season_ece:
            summary["ece_p1_season_mean"] = season_ece.get("mean_ece")
            summary["ece_p1_by_round"] = season_ece.get("by_round")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result_doc, f, indent=2)

        context.log(f"Saved result file: {out_path}")
        context.log(
            f"Winner {'hit' if winner_hit else 'miss'} | "
            f"Podium overlap {podium_overlap}/3 | "
            f"Brier P1 {summary['brier_score_p1']} | "
            f"ECE P1 {summary['ece_p1_round']}"
        )
        context.progress(100, "Actual result recorded.")
        return {
            "result_file": str(out_path),
            "winner_hit": winner_hit,
            "podium_overlap": podium_overlap,
            "ece_p1_round": summary["ece_p1_round"],
            "ece_p1_season_mean": summary.get("ece_p1_season_mean"),
        }

    def quick_retrain(self, *, context: ActionContext) -> dict[str, Any]:
        from scripts.quick_train import main as quick_train_main

        context.progress(3, "Starting quick retrain (XGBoost only)...")
        context.log("Training can take several minutes.")
        context.check_cancelled()
        try:
            quick_train_main()
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise RuntimeError(f"quick_train exited with code {exc.code}") from None
        context.progress(100, "Quick retrain completed.")
        context.log("Updated models/xgb_podium.pkl and models/calibrator.pkl")
        return {"mode": "quick_retrain", "status": "ok"}

    def full_retrain(self, *, context: ActionContext) -> dict[str, Any]:
        from src.training.train import train as full_train

        context.progress(3, "Starting full retrain (XGBoost)...")
        context.log("Training can take several minutes.")
        context.check_cancelled()
        full_train()
        context.progress(100, "Full retrain completed.")
        context.log("Updated models/xgb_podium.pkl and models/calibrator.pkl")
        return {"mode": "full_retrain", "status": "ok"}

    def _season_ece_summary(self, year: int) -> dict[str, Any] | None:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.results_dir.glob(f"{year}_R*_actual.json")):
            try:
                with open(path, encoding="utf-8") as f:
                    doc = json.load(f)
            except Exception:
                continue

            round_num = int(doc.get("round", 0) or 0)
            preds = doc.get("predictions", []) or []
            p = []
            y = []
            for row in preds:
                if row.get("actual_finish") is None:
                    continue
                try:
                    p.append(float(row.get("p_win", 0.0) or 0.0))
                    y.append(1 if int(row.get("actual_finish")) == 1 else 0)
                except Exception:
                    continue
            if not p:
                continue
            ece = expected_calibration_error(np.asarray(p, dtype=float), np.asarray(y, dtype=int), n_bins=10)
            rows.append({"round": round_num, "ece_p1": float(round(ece, 4))})

        if not rows:
            return None

        rows = sorted(rows, key=lambda x: x["round"])
        mean_ece = float(round(np.mean([r["ece_p1"] for r in rows]), 4))
        return {"mean_ece": mean_ece, "by_round": rows}

