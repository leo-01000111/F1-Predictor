"""
Model gym: a reproducible search over combinations of model toggles.

A toggle is a named switch on the production model (v5, TwoStageRaceModel with
FEATURE_COLS). Two kinds:

  model    needs retraining: feature columns added / removed, model options
           (monotone constraints, recency half-life, finish model on/off).
  posthoc  applied to cached out-of-sample predictions in seconds: how the
           Plackett-Luce temperature is fitted, rating blend, overtaking-scaled
           T, safety-car mixture, Benter exponents (src/models/race_calibration.py).

Evaluation is the production walk-forward (src/training/backtest.py,
make_two_stage_predict with TwoStageRaceModel.DEFAULT_PARAMS / DEFAULT_SEEDS):
each test season is predicted by a model trained only on earlier seasons, and
every calibration parameter (T and the post-hoc layers) is fitted on earlier
test seasons' out-of-sample predictions. One difference from scripts/backtest.py:
the season before the first scored season is predicted as a warm-up, so the
first scored season also gets a fitted T (backtest.py uses T = 1 for 2018).
`warmup=False` reproduces backtest.py exactly.

Per-driver OOS predictions (pred, q) are cached per test season under
models/experiments/gym/cache/<key>/<year>.parquet, keyed by a hash of the
resolved model spec (columns in order, ranker params, seeds, half-life, finish
model inputs), the xgboost version, GYM_CACHE_VERSION and a fingerprint of the
feature-matrix columns the spec reads. Change any of them and the key changes.

Adding a toggle: call register() with a Toggle, e.g. for new feature columns
    register(add_columns_toggle("add_compound_fp", ["fp_compound_gap_pct"],
                                "compound-adjusted long runs", first_test_season=2024))
A toggle whose columns are missing from the feature matrix is listed as
unavailable and skipped. first_test_season: the first test season in which the
toggle can change anything (practice data starts 2023, so a model trained on
it is first tested in 2024); toggles that cannot act on the dev seasons are
left out of the dev search, since their dev "effect" would be pure noise
(column subsampling draws change with the column set).

Excluded by design (see EXCLUDED): in-season retraining / current_boost,
anything needing FastF1 lap data, market blend.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.features.build_features import FEATURE_COLS
from src.models.kalman_strength import KALMAN_COLS
from src.models.race_calibration import RaceCalibration, race_contexts
from src.models.two_stage import FINISH_COLS
from src.training.backtest import _race_orders, fit_temperature, make_two_stage_predict, paired_bootstrap

ROOT = Path(__file__).resolve().parent.parent.parent
GYM_DIR = ROOT / "models" / "experiments" / "gym"
CACHE_DIR = GYM_DIR / "cache"

GYM_CACHE_VERSION = 1

# A flip is accepted only if it lowers dev mean top-3 log loss by more than this
# (nats per race). Smallest shipped effect so far: practice features -0.029; the
# paired SE of a dev-mean difference over ~130 races is ~0.01, so 0.01 keeps the
# search from walking on noise while still catching effects of that size.
MIN_IMPROVEMENT = 0.010
MAX_STEPS = 8
N_BOOT = 2000
SIM_N = 5000

# Feature groups as on the site (scripts/export_site_json.py FEATURE_GROUPS)
FEATURE_GROUPS: dict[str, list[str]] = {
    "qualifying": ["grid_position", "quali_gap_relative", "grid_position_rel", "teammate_quali_delta",
                   "grid_penalty_places", "circuit_grid_conversion"],
    "car": ["car_pace_delta_pct", "constructor_standing", "team_cum_points", "team_form_5"],
    "driver": ["driver_form_5", "driver_form_10", "driver_form_ewm", "driver_elo", "driver_experience_norm",
               "driver_wet_avg_finish"],
    "championship": ["championship_position", "cum_points_before", "championship_pressure", "round_fraction"],
    "circuit": ["circuit_avg_finish", "circuit_podiums", "circuit_appearances", "is_street_circuit",
                "circuit_safety_car_rate"],
    "reliability": ["dnf_rate", "team_reliability_rate"],
    "weather": ["temp_c_mean", "precip_mm_total", "windspeed_ms_mean", "humidity_pct_mean", "is_wet_race"],
    "sprint": ["sprint_position_rel"],
    "practice": ["fp_longrun_gap_pct", "team_longrun_gap_pct"],
}
# scripts/backtest.py / grid_diagnostics.py "no_reputation"
REPUTATION = ["driver_form_5", "driver_form_10", "driver_form_ewm", "driver_elo",
              "driver_experience_norm", "driver_wet_avg_finish", "circuit_avg_finish",
              "circuit_podiums", "circuit_appearances", "championship_position",
              "cum_points_before", "championship_pressure"]
RATINGS = ["driver_rating", "team_rating", "car_rating"]
# Ranker score is higher = better, so a worse grid slot / bigger gap may only lower it
MONOTONE_RANKER = {"grid_position": -1, "grid_position_rel": -1, "quali_gap_relative": -1}
PRACTICE_FIRST_TEST = 2024   # OpenF1 practice laps start 2023

EXCLUDED = {
    "current_boost / in-season retraining": (
        "no-op in the per-season walk-forward (training never contains the test season); it needs "
        "walk_forward_by_round (retrain before every race, T from the last 22 races), a different "
        "protocol that the per-season cache and post-hoc fits do not fit, ~20x the cost. Tested "
        "2026-10-28: 2x boost -0.030 [-0.112, +0.058]."),
    "FastF1 lap-data features (start delta, compounds, FastF1 practice)": (
        "lap data needs a paid F1 TV subscription; no longer collectable."),
    "market blend": "Kalshi prices only from 2025: no dev seasons.",
    "compound-adjusted practice features": (
        "being built by another agent; register with add_columns_toggle once the columns exist."),
    "ranker hyperparameters (depth, trees, lr, seeds)": "tuned separately (scripts/tune_ranker.py).",
}


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

@dataclass
class ModelSpec:
    """Resolved model-level config: what is trained."""
    cols: list
    params: dict
    n_seeds: int
    half_life: Optional[float] = None
    use_finish_model: bool = True
    monotone: bool = False
    finish_cols: list = field(default_factory=lambda: list(FINISH_COLS))

    def ranker_params(self) -> dict:
        p = dict(self.params)
        if self.monotone:
            mono = {c: v for c, v in MONOTONE_RANKER.items() if c in self.cols}
            if mono:
                p["monotone_constraints"] = mono
        return p


@dataclass
class Toggle:
    name: str
    kind: str                      # "model" or "posthoc"
    description: str
    values: tuple = (False, True)  # values[0] = production (v5) setting
    apply: Optional[Callable[[ModelSpec, Any], ModelSpec]] = None   # model: spec -> spec
    phase: int = 0                 # model: order of application (adds 0, drops 1)
    requires: tuple = ()           # feature-matrix columns the toggle needs
    first_test_season: Optional[int] = None
    in_search: bool = True         # False: ablation map only (see --include-singles)
    fit_params: tuple = ()         # posthoc: RaceCalibration parameters fitted when on

    @property
    def default(self):
        return self.values[0]


REGISTRY: dict[str, Toggle] = {}


def register(t: Toggle) -> Toggle:
    if t.kind not in ("model", "posthoc"):
        raise ValueError(f"toggle kind must be model/posthoc: {t.kind}")
    REGISTRY[t.name] = t
    return t


def add_columns_toggle(name: str, cols: list[str], description: str, **kw) -> Toggle:
    def _apply(spec: ModelSpec, on) -> ModelSpec:
        return replace(spec, cols=spec.cols + [c for c in cols if c not in spec.cols]) if on else spec
    return Toggle(name, "model", description, apply=_apply, phase=0, requires=tuple(cols), **kw)


def drop_columns_toggle(name: str, cols: list[str], description: str, **kw) -> Toggle:
    def _apply(spec: ModelSpec, on) -> ModelSpec:
        return replace(spec, cols=[c for c in spec.cols if c not in cols]) if on else spec
    return Toggle(name, "model", description, apply=_apply, phase=1, **kw)


def _register_defaults() -> None:
    register(add_columns_toggle("add_pit", ["team_pit_delta_s"],
                                "+ team_pit_delta_s (dropped 2026-10-27)"))
    register(add_columns_toggle("add_ratings", RATINGS,
                                "+ driver/team/car strength ratings (strength_ratings.py)"))
    # Kalman strength filter (kalman_strength.py); columns merged by scripts/gym.py --with-kalman.
    # Its hyperparameters were tuned on 2014-2020 results: score on 2021+ only.
    register(add_columns_toggle("add_kalman", ["kalman_pace_postq"],
                                "+ Kalman car+driver pace after this weekend's qualifying"))
    register(add_columns_toggle("add_kalman_all", KALMAN_COLS,
                                "+ Kalman post-quali pace, pre-weekend sd, car and driver"))
    register(add_columns_toggle("add_overtaking", ["circuit_overtaking"],
                                "+ circuit overtaking index as a ranker input"))
    register(drop_columns_toggle("drop_reputation", REPUTATION,
                                 "- 12 reputation features (form, Elo, experience, circuit history, standing);"
                                 " with add_ratings = ratings_swap"))
    for g, cols in FEATURE_GROUPS.items():
        fts = PRACTICE_FIRST_TEST if g == "practice" else None
        register(drop_columns_toggle(f"drop_{g}", cols, f"- feature group '{g}' ({len(cols)})",
                                     first_test_season=fts))
    register(Toggle("mono", "model", "monotone constraints: worse grid / bigger quali gap -> lower score",
                    apply=lambda s, on: replace(s, monotone=bool(on)), phase=2))
    register(Toggle("half_life", "model", "exponential recency weights, half-life in years (None = linear 0.65-1.35)",
                    values=(None, 2.0, 4.0, 8.0), apply=lambda s, v: replace(s, half_life=v), phase=2))
    register(Toggle("no_finish_model", "model", "finish model off: every car finishes (q = 1)",
                    apply=lambda s, on: replace(s, use_finish_model=not on), phase=2))
    # Single-feature drops: ablation map only by default (35 more candidates per
    # step would make the search pick noise; --include-singles to search them)
    for c in FEATURE_COLS:
        fts = PRACTICE_FIRST_TEST if c in FEATURE_GROUPS["practice"] else None
        register(drop_columns_toggle(f"drop:{c}", [c], f"- {c}", first_test_season=fts, in_search=False))

    # Post-hoc layers
    register(Toggle("temp_pooled", "posthoc",
                    "fit T (and layers) on ALL earlier test seasons pooled, not only the previous one"))
    register(Toggle("rating_blend", "posthoc", "blend car_rating into the score: weight fitted, or fixed 0.5",
                    values=(None, "fit", 0.5), fit_params=("rating_weight",)))
    register(Toggle("overtaking_t", "posthoc", "per-race T scaled by the circuit overtaking index",
                    fit_params=("overtaking_slope",)))
    register(Toggle("safety_car", "posthoc", "safety-car mixture with a flatter order",
                    fit_params=("sc_scale", "sc_flatten")))
    register(Toggle("benter", "posthoc", "Benter exponents for places 2 and 3",
                    fit_params=("lambda_2", "lambda_3")))


_register_defaults()


def normalize(cfg: dict) -> dict:
    """Drop entries at their production value; sorted by registry order."""
    out = {}
    for name in REGISTRY:
        if name in cfg and cfg[name] != REGISTRY[name].default:
            out[name] = cfg[name]
    unknown = set(cfg) - set(REGISTRY)
    if unknown:
        raise KeyError(f"unknown toggles: {sorted(unknown)}")
    return out


def split_cfg(cfg: dict) -> tuple[dict, dict]:
    cfg = normalize(cfg)
    return ({k: v for k, v in cfg.items() if REGISTRY[k].kind == "model"},
            {k: v for k, v in cfg.items() if REGISTRY[k].kind == "posthoc"})


def label(cfg: dict) -> str:
    cfg = normalize(cfg)
    if not cfg:
        return "v5"
    return " ".join(k if v is True else f"{k}={v}" for k, v in cfg.items())


def toggle_status(t: Toggle, fm: pd.DataFrame, last_dev: int) -> Optional[str]:
    """None if the toggle can be searched on dev, else the reason it cannot."""
    missing = [c for c in t.requires if c not in fm.columns]
    if missing:
        return f"columns missing from the feature matrix: {missing}"
    if t.first_test_season is not None and t.first_test_season > last_dev:
        return f"cannot act before test season {t.first_test_season} (dev ends {last_dev})"
    return None


# --------------------------------------------------------------------------- #
# Model level: spec, cache, walk-forward predictions
# --------------------------------------------------------------------------- #

def build_spec(model_cfg: dict, overrides: Optional[dict] = None, n_seeds: Optional[int] = None) -> ModelSpec:
    from src.models.xgb_model import TwoStageRaceModel

    params = dict(TwoStageRaceModel.DEFAULT_PARAMS, **(overrides or {}))
    spec = ModelSpec(cols=list(FEATURE_COLS), params=params, n_seeds=n_seeds or TwoStageRaceModel.DEFAULT_SEEDS)
    cfg = normalize(model_cfg)
    for name in sorted(cfg, key=lambda k: (REGISTRY[k].phase, list(REGISTRY).index(k))):
        t = REGISTRY[name]
        if t.kind == "model":
            spec = t.apply(spec, cfg[name])
    return spec


META_COLS = ["year", "round", "driver", "target_position", "dnf", "finish_position"]


def spec_key(spec: ModelSpec, fm: pd.DataFrame) -> str:
    import xgboost

    used = list(dict.fromkeys(META_COLS + spec.cols + spec.finish_cols))
    fp = hashlib.sha1(pd.util.hash_pandas_object(fm[used], index=False).values.tobytes()).hexdigest()
    payload = {"v": GYM_CACHE_VERSION, "xgb": xgboost.__version__, "spec": asdict(spec), "data": fp}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def season_predictions(fm: pd.DataFrame, spec: ModelSpec, years: list[int],
                       cache_dir: Path = CACHE_DIR) -> dict[int, tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
    """{year: (test rows, pred, q)} OOS per season, trained on earlier seasons. Cached."""
    key = spec_key(spec, fm)
    d = cache_dir / key
    d.mkdir(parents=True, exist_ok=True)
    meta = d / "spec.json"
    if not meta.exists():
        meta.write_text(json.dumps(asdict(spec), indent=1, default=str), encoding="utf-8")
    predict = make_two_stage_predict(spec.cols, use_finish_model=spec.use_finish_model,
                                     half_life=spec.half_life, params=spec.ranker_params(),
                                     n_seeds=spec.n_seeds, finish_cols=spec.finish_cols)
    out = {}
    for year in years:
        test = fm[fm["year"] == year].reset_index(drop=True)
        path = d / f"{year}.parquet"
        if path.exists():
            c = pd.read_parquet(path)
            if (len(c) == len(test) and (c["driver"].to_numpy() == test["driver"].to_numpy()).all()
                    and (c["round"].to_numpy() == test["round"].to_numpy()).all()):
                out[year] = (test, c["pred"].to_numpy(), c["q"].to_numpy())
                continue
        train = fm[fm["year"] < year].reset_index(drop=True)
        res = predict(train, test)
        pred, q = res if isinstance(res, tuple) else (res, np.ones(len(test)))
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        pd.DataFrame({"year": test["year"], "round": test["round"], "driver": test["driver"],
                      "pred": pred, "q": q}).to_parquet(tmp, index=False)
        tmp.replace(path)
        out[year] = (test, np.asarray(pred, float), np.asarray(q, float))
    return out


# --------------------------------------------------------------------------- #
# Post-hoc layer and scoring
# --------------------------------------------------------------------------- #

def _races(test: pd.DataFrame, pred: np.ndarray, q: np.ndarray):
    tops = {key: top for key, _, top in _race_orders(test)}
    return [(key, ctx, tops[key]) for key, _, ctx in race_contexts(test, -pred, q) if key in tops]


def fit_calibration(prev: list[tuple[pd.DataFrame, np.ndarray, np.ndarray]], posthoc_cfg: dict) -> RaceCalibration:
    """Calibration fitted on earlier seasons' OOS predictions. T as in backtest.py
    (fit_temperature over T_GRID), then the enabled layers by coordinate descent."""
    if not prev:
        return RaceCalibration()
    pred = np.concatenate([p for _, p, _ in prev])
    q = np.concatenate([qq for _, _, qq in prev])
    df = pd.concat([t for t, _, _ in prev], ignore_index=True)
    cal = RaceCalibration(temperature=fit_temperature(pred, df, q))
    params: list[str] = []
    for name, v in normalize(posthoc_cfg).items():
        t = REGISTRY[name]
        if name == "rating_blend" and v != "fit":
            cal = replace(cal, rating_weight=float(v))   # fixed weight, not fitted
            continue
        params += list(t.fit_params)
    if params or cal.rating_weight:
        races = [(ctx, top) for s in prev for _, ctx, top in _races(*s)]
        cal = cal.fit(races, params)
    return cal


def score_season(test: pd.DataFrame, pred: np.ndarray, q: np.ndarray, cal: RaceCalibration) -> pd.DataFrame:
    rows = []
    for key, ctx, top in _races(test, pred, q):
        m = cal.simulate(ctx, n_sims=SIM_N, seed=0)
        on = np.zeros(len(ctx.raw))
        on[top] = 1.0
        p_pod = m[:, :3].sum(axis=1)
        podium = np.argsort(-p_pod, kind="stable")[:3]
        rows.append({
            "year": int(key[0]), "round": int(key[1]),
            "ll_top3": cal.nll(ctx, top),
            "ll_winner": cal.nll(ctx, top[:1]),
            "winner_hit": bool(int(np.argmax(m[:, 0])) == top[0]),
            "podium_overlap": len(set(podium.tolist()) & set(top)),
            "brier_podium": float(((p_pod - on) ** 2).sum()),
        })
    return pd.DataFrame(rows)


def summarize(races: pd.DataFrame) -> dict:
    return {
        "n_races": int(len(races)),
        "ll_top3": float(races["ll_top3"].mean()),
        "ll_winner": float(races["ll_winner"].mean()),
        "brier_podium": float(races["brier_podium"].mean()),
        "winner_accuracy": float(races["winner_hit"].mean()),
        "podium_accuracy": float(races["podium_overlap"].mean() / 3),
        "ll_top3_by_season": {int(y): round(float(v), 4) for y, v in races.groupby("year")["ll_top3"].mean().items()},
    }


def compare(a: pd.DataFrame, b: pd.DataFrame, seed: int = 0) -> dict:
    """a - b per race (paired bootstrap 95% CI) for each metric."""
    m = a.merge(b, on=["year", "round"], suffixes=("_a", "_b"))
    out = {}
    for k in ("ll_top3", "ll_winner", "brier_podium", "winner_hit"):
        d, lo, hi = paired_bootstrap(m[f"{k}_a"].astype(float), m[f"{k}_b"].astype(float), n=N_BOOT, seed=seed)
        out[k] = [round(d, 5), round(lo, 5), round(hi, 5)]
    return out


# --------------------------------------------------------------------------- #
# The gym: trial evaluation + logging
# --------------------------------------------------------------------------- #

class Gym:
    def __init__(self, fm: pd.DataFrame, out_dir: Path = GYM_DIR, cache_dir: Path = CACHE_DIR,
                 overrides: Optional[dict] = None, n_seeds: Optional[int] = None,
                 warmup: bool = True, run_id: Optional[str] = None) -> None:
        self.fm = fm.reset_index(drop=True)
        self.out_dir = Path(out_dir)
        self.cache_dir = Path(cache_dir)
        self.overrides = overrides or {}
        self.n_seeds = n_seeds
        self.warmup = warmup
        self.run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.trials_path = self.out_dir / "trials.jsonl"
        self._scores: dict = {}

    def spec(self, model_cfg: dict) -> ModelSpec:
        return build_spec(model_cfg, self.overrides, self.n_seeds)

    def evaluate(self, cfg: dict, years: list[int]) -> pd.DataFrame:
        """Per-race scores of config `cfg` (model + post-hoc toggles) on `years`."""
        model_cfg, posthoc_cfg = split_cfg(cfg)
        spec = self.spec(model_cfg)
        key = spec_key(spec, self.fm)
        first = years[0] - 1 if self.warmup else years[0]
        all_years = list(range(max(first, int(self.fm["year"].min()) + 1), years[-1] + 1))
        need = [y for y in years if (key, json.dumps(posthoc_cfg, sort_keys=True, default=str), y) not in self._scores]
        if need:
            preds = season_predictions(self.fm, spec, all_years, self.cache_dir)
            pooled = bool(posthoc_cfg.get("temp_pooled"))
            for y in need:
                earlier = [preds[p] for p in all_years if p < y]
                prev = earlier if pooled else earlier[-1:]
                cal = fit_calibration(prev, posthoc_cfg)
                sc = score_season(*preds[y], cal)
                sc["cal"] = json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                        for k, v in cal.__dict__.items() if k != "fitted"})
                self._scores[(key, json.dumps(posthoc_cfg, sort_keys=True, default=str), y)] = sc
        return pd.concat([self._scores[(key, json.dumps(posthoc_cfg, sort_keys=True, default=str), y)]
                          for y in years], ignore_index=True)

    def trial(self, stage: str, cfg: dict, years: list[int], ref: Optional[pd.DataFrame] = None,
              ref_label: str = "", extra: Optional[dict] = None) -> tuple[pd.DataFrame, dict]:
        t0 = time.time()
        races = self.evaluate(cfg, years)
        rec = {
            "run_id": self.run_id, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": stage,
            "label": label(cfg), "config": normalize(cfg),
            "model_key": spec_key(self.spec(split_cfg(cfg)[0]), self.fm),
            "years": [years[0], years[-1]], "metrics": summarize(races),
            "runtime_s": round(time.time() - t0, 1),
            "overrides": self.overrides, "n_seeds": self.n_seeds, "warmup": self.warmup,
        }
        if ref is not None:
            rec["vs"] = ref_label
            rec["delta"] = compare(races, ref)
        if extra:
            rec.update(extra)
        with open(self.trials_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        d = rec.get("delta", {}).get("ll_top3")
        logger.info(f"[{stage}] {rec['label']:40s} ll_top3 {rec['metrics']['ll_top3']:.4f}"
                    + (f"  d {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]" if d else "")
                    + f"  ({rec['runtime_s']:.0f}s)")
        return races, rec

    # ------------------------------------------------------------------ #

    def flips(self, cfg: dict, kind: str, include_singles: bool, last_dev: int) -> list[tuple[str, Any]]:
        out = []
        for t in REGISTRY.values():
            if t.kind != kind or (not t.in_search and not include_singles):
                continue
            if toggle_status(t, self.fm, last_dev) is not None:
                continue
            cur = cfg.get(t.name, t.default)
            out += [(t.name, v) for v in t.values if v != cur]
        return out

    def stepwise(self, start: dict, kind: str, dev: list[int], min_improvement: float = MIN_IMPROVEMENT,
                 max_steps: int = MAX_STEPS, include_singles: bool = False,
                 ablation_singles: bool = True) -> dict:
        """Forward/backward stepwise over `kind` toggles from `start` on the dev seasons.

        Every step tries each single flip of the current config (turning a toggle
        on, off or to another value, so earlier accepts can be undone) and accepts
        the best one if it lowers dev mean ll_top3 by more than min_improvement.
        Step 0 is the ablation map (plus single-feature drops if ablation_singles).
        """
        cur = normalize(start)
        cur_races, cur_rec = self.trial(f"{kind}_start", cur, dev)
        path = [{"step": 0, "config": cur, "label": label(cur), "metrics": cur_rec["metrics"]}]
        ablation = []
        for step in range(1, max_steps + 1):
            singles = include_singles or (step == 1 and ablation_singles and kind == "model")
            cands = []
            for name, v in self.flips(cur, kind, singles, dev[-1]):
                cfg = normalize({**cur, name: v})
                races, rec = self.trial(f"{kind}_step{step}", cfg, dev, ref=cur_races, ref_label=label(cur),
                                        extra={"flip": [name, v]})
                in_search = REGISTRY[name].in_search or include_singles
                cands.append((rec["delta"]["ll_top3"][0], name, v, cfg, races, rec, in_search))
                if step == 1:
                    ablation.append({"flip": name if v is True else f"{name}={v}",
                                     "in_search": in_search, "delta": rec["delta"],
                                     "ll_top3": rec["metrics"]["ll_top3"],
                                     "ll_top3_by_season": rec["metrics"]["ll_top3_by_season"]})
            eligible = [c for c in cands if c[6]]
            if not eligible:
                break
            best = min(eligible, key=lambda c: c[0])
            d, name, v, cfg, races, rec, _ = best
            logger.info(f"step {step}: best flip {name}={v} d ll_top3 {d:+.4f} "
                        f"[{rec['delta']['ll_top3'][1]:+.4f},{rec['delta']['ll_top3'][2]:+.4f}]")
            if d >= -min_improvement:
                path.append({"step": step, "accepted": False, "best_flip": [name, v], "delta": rec["delta"],
                             "reason": f"best d ll_top3 {d:+.4f} not below -{min_improvement}"})
                break
            cur, cur_races = cfg, races
            path.append({"step": step, "accepted": True, "flip": [name, v], "config": cur, "label": label(cur),
                         "delta": rec["delta"], "metrics": rec["metrics"]})
        ablation.sort(key=lambda a: a["delta"]["ll_top3"][0])
        return {"final": cur, "final_races": cur_races, "path": path, "ablation": ablation}
