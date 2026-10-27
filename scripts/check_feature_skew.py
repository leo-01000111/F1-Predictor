"""Training/serving skew check: features for one past race built two ways.

A) build_feature_matrix() over all raw data  -> the training row for (year, round)
B) run_inference() with history < round and an OpenF1-shaped quali_df
   (model.predict is stubbed to capture X and abort before any file is written)

Every remaining difference is a bug unless explained (grid_penalty_places always
differs when the race had grid penalties: inference only knows qualifying order).

Usage: python scripts/check_feature_skew.py <year> <round> <openf1_circuit_short_name>
   e.g. python scripts/check_feature_skew.py 2026 18 austin
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
YEAR, RND, SHORT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]

import pandas as pd
from loguru import logger
logger.remove()

from src.data_collection.f1_historical import (
    load_race_results, load_quali_results, load_lap_extras, load_pit_stops, load_sprint_results,
)
from src.data_collection.weather import load_weather_history
from src.features.build_features import FEATURE_COLS, build_feature_matrix
import src.models.xgb_model as xm
from src.inference.predict import run_inference

race, quali = load_race_results(), load_quali_results()
weather, lap, pit, sprint = load_weather_history(), load_lap_extras(), load_pit_stops(), load_sprint_results()

fm = build_feature_matrix(race, quali, weather_df=weather, lap_extras_df=lap, pit_df=pit, sprint_df=sprint)
train_rows = fm[(fm.year == YEAR) & (fm["round"] == RND)].set_index("driver")[FEATURE_COLS]

r = race[(race.year == YEAR) & (race["round"] == RND)]
hist = race[(race.year < YEAR) | ((race.year == YEAR) & (race["round"] < RND))]
quali_df = pd.DataFrame({
    "driver_abbr": r.driver.values, "team": r.team.values,
    "quali_position": r.grid_position.values,           # training uses race grid
    # gaps come from the qualifying file, as OpenF1 supplies them at inference
    "quali_gap_to_pole_s": r.driver.map(
        quali[(quali.year == YEAR) & (quali["round"] == RND)].set_index("driver")["quali_gap_to_pole_s"]
    ).values,
    "circuit": r.circuit.values, "circuit_key": SHORT, "race_date": r.race_date.values,
})
w = weather[(weather.year == YEAR) & (weather["round"] == RND)]
weather_dict = w.drop(columns=["year", "round", "circuit_key", "race_date"], errors="ignore").iloc[0].to_dict()


class _Captured(Exception):
    pass


class _Stub:
    def predict(self, X):
        _Stub.X = X
        raise _Captured


xm.XGBRacePredictor.load = classmethod(lambda cls, name: _Stub())
try:
    run_inference(YEAR, RND, quali_df, hist, weather_dict, practice_df=None, pit_df=pit, sprint_df=sprint)
except _Captured:
    pass
# X loses the driver column and predict.py reorders rows: align on grid position
assert quali_df.quali_position.is_unique, "grid positions not unique, cannot align"
grid_to_driver = dict(zip(quali_df.quali_position.astype(float), quali_df.driver_abbr))
inf_rows = _Stub.X.copy()
inf_rows.index = [grid_to_driver[g] for g in inf_rows["grid_position"].astype(float)]
inf_rows = inf_rows[FEATURE_COLS]

common = train_rows.index.intersection(inf_rows.index)
diff = (train_rows.loc[common] - inf_rows.loc[common]).abs()
bad = diff.max()[diff.max() > 1e-6].sort_values(ascending=False)
print(f"{YEAR} R{RND}: {len(common)} drivers, {len(FEATURE_COLS)} features, {len(bad)} differ")
for col in bad.index:
    d = diff[col].idxmax()
    print(f"  {col:28s} max|diff|={bad[col]:.4f}  e.g. {d}: train={train_rows.loc[d, col]:.4f} infer={inf_rows.loc[d, col]:.4f}")
