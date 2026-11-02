# -*- coding: utf-8 -*-
"""
Reliability diagrams for P(win) and P(podium) over the walk-forward backtest
===========================================================================
Asks whether the published probabilities mean what they say: of the drivers we
gave ~30% to win, did ~30% win?

Probabilities are out of sample and computed the way production computes them:
the two-stage model (FEATURE_COLS, TwoStageRaceModel DEFAULT_PARAMS/DEFAULT_SEEDS)
for season S is trained on seasons before S, its temperature T fitted on season
S-1's out-of-sample predictions (a warm-up season is predicted only to fit T for
the first test season), and probabilities come from two_stage.simulate(pred, q, T)
with the finish model's q, which is what simulate(-score/T, q, 1.0) does in
src/inference/predict.py. The grid baseline is PL on grid position with T fitted
on all earlier races (as scripts/backtest.py and scripts/market_backtest.py).

Targets:
  win     P(P1)           vs finished 1st
  podium  P(P1+P2+P3)     vs finished top 3
  p23     P(P2)+P(P3)     vs finished 2nd or 3rd (the part of the podium PL's
          later stages decide; Benter / Harville bias shows up here)

Per target: fixed bins (finer at the low end, counts shown), per-bin mean
predicted vs observed with Wilson 95% intervals, Brier score with its
reliability / resolution / uncertainty decomposition (Murphy, on those bins),
ECE (count-weighted |observed - predicted|), log loss, and calibration
slope / intercept from a logistic regression of the outcome on logit(p)
(slope < 1: overconfident, too spread out; > 1: underconfident). Intervals on
slope / intercept and the model-minus-grid Brier come from a race bootstrap.
Also by favourite rank (the model's 1st, 2nd, 3rd... most likely winner).

Usage:
    python scripts/reliability.py                  # test 2018-latest
    python scripts/reliability.py --reuse          # reuse saved per-driver probabilities
    python scripts/reliability.py --start 2021 --n-sims 20000

Writes models/experiments/reliability.json, reliability.png and the per-driver
probabilities reliability_drivers.parquet. No model files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression

from src.features.build_features import FEATURE_COLS, load_feature_matrix
from src.models.two_stage import simulate
from src.training.backtest import _grid_score, fit_temperature, grid_predict, make_two_stage_predict

OUT_DIR = ROOT / "models" / "experiments"
DRIVERS_PATH = OUT_DIR / "reliability_drivers.parquet"

# Fixed bin edges: most drivers sit near 0, so the low end is split finely and
# the top end (favourites) gets its own bins. Counts are reported per bin.
BINS = {
    "win": [0, 0.01, 0.03, 0.06, 0.10, 0.15, 0.22, 0.30, 0.40, 0.55, 0.70, 1.0],
    "podium": [0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.85, 0.93, 1.0],
    "p23": [0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 1.0],
}
TARGETS = {"win": "won", "podium": "podium", "p23": "p23"}
N_BOOT = 500


# ---------------------------------------------------------------- probabilities

def _rows(g: pd.DataFrame, m: np.ndarray, prefix: str) -> pd.DataFrame:
    return pd.DataFrame({
        "year": g["year"].values, "round": g["round"].values, "driver": g["driver"].values,
        f"p_win_{prefix}": m[:, 0], f"p_podium_{prefix}": m[:, :3].sum(axis=1),
        f"p_p23_{prefix}": m[:, 1:3].sum(axis=1),
    })


def model_probs(fm: pd.DataFrame, years: list[int], n_sims: int) -> pd.DataFrame:
    """Out-of-sample two-stage probabilities, production configuration."""
    predict = make_two_stage_predict(FEATURE_COLS)
    rows, temps, prev = [], {}, None
    for year in [years[0] - 1] + years:
        train = fm[fm["year"] < year].reset_index(drop=True)
        test = fm[fm["year"] == year].reset_index(drop=True)
        pred, q = predict(train, test)
        if year in years:
            t = fit_temperature(*prev) if prev is not None else 1.0
            temps[year] = t
            for _, idx in test.groupby(["year", "round"]).indices.items():
                m = simulate(pred[idx], q[idx], t, n_sims=n_sims, seed=0)
                rows.append(_rows(test.iloc[idx], m, "model").assign(q_finish=q[idx]))
            logger.info(f"model {year}: T={t}")
        prev = (pred, test, q)
    return pd.concat(rows, ignore_index=True).assign(t_model=lambda d: d["year"].map(temps))


def grid_probs(fm: pd.DataFrame, years: list[int], n_sims: int) -> pd.DataFrame:
    """Grid as a probability model: PL on grid position, T fitted on earlier races."""
    rows, temps = [], {}
    for year in years:
        train = fm[fm["year"] < year].reset_index(drop=True)
        test = fm[fm["year"] == year].reset_index(drop=True)
        t = fit_temperature(grid_predict(train, train), train)
        temps[year] = t
        for _, idx in test.groupby(["year", "round"]).indices.items():
            g = test.iloc[idx]
            m = simulate(_grid_score(g["grid_position"]), np.ones(len(g)), t, n_sims=n_sims, seed=0)
            rows.append(_rows(g, m, "grid"))
        logger.info(f"grid  {year}: T={t}")
    return pd.concat(rows, ignore_index=True).assign(t_grid=lambda d: d["year"].map(temps))


# ---------------------------------------------------------------- statistics

def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def cal_slope(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration y ~ a + b * logit(p). Returns (intercept a, slope b)."""
    lr = LogisticRegression(penalty=None, max_iter=1000)
    lr.fit(_logit(p).reshape(-1, 1), y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def bin_table(p: np.ndarray, y: np.ndarray, edges: list[float]) -> list[dict]:
    b = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    out = []
    for i in range(len(edges) - 1):
        m = b == i
        n = int(m.sum())
        k = float(y[m].sum()) if n else 0.0
        lo, hi = wilson(k, n)
        out.append({
            "lo": edges[i], "hi": edges[i + 1], "n": n, "n_pos": int(k),
            "mean_pred": float(p[m].mean()) if n else None,
            "observed": k / n if n else None,
            "ci95": [lo, hi] if n else None,
            "obs_minus_pred": (k / n - float(p[m].mean())) if n else None,
        })
    return out


def summary(p: np.ndarray, y: np.ndarray, edges: list[float]) -> dict:
    bins = [b for b in bin_table(p, y, edges) if b["n"]]
    n = len(p)
    base = y.mean()
    rel = sum(b["n"] * (b["mean_pred"] - b["observed"]) ** 2 for b in bins) / n
    res = sum(b["n"] * (b["observed"] - base) ** 2 for b in bins) / n
    ece = sum(b["n"] * abs(b["observed"] - b["mean_pred"]) for b in bins) / n
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    a, s = cal_slope(p, y)
    return {
        "n": int(n), "n_pos": int(y.sum()), "base_rate": float(base),
        "mean_pred": float(p.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "brier_reliability": float(rel), "brier_resolution": float(res),
        "brier_uncertainty": float(base * (1 - base)),
        "ece": float(ece),
        "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
        "cal_intercept": a, "cal_slope": s,
    }


def race_bootstrap(df: pd.DataFrame, target: str, seed: int = 0) -> dict:
    """95% race-bootstrap intervals: slope/intercept per source, Brier model - grid."""
    y = df[TARGETS[target]].to_numpy(float)
    pm = df[f"p_{target}_model"].to_numpy(float)
    pg = df[f"p_{target}_grid"].to_numpy(float)
    groups = list(df.groupby(["year", "round"]).indices.values())
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        am, sm = cal_slope(pm[idx], y[idx])
        ag, sg = cal_slope(pg[idx], y[idx])
        d = np.mean((pm[idx] - y[idx]) ** 2) - np.mean((pg[idx] - y[idx]) ** 2)
        stats.append((am, sm, ag, sg, d))
    lo, hi = np.percentile(np.array(stats), [2.5, 97.5], axis=0)
    names = ["model_intercept", "model_slope", "grid_intercept", "grid_slope", "brier_model_minus_grid"]
    out = {k: [float(a), float(b)] for k, a, b in zip(names, lo, hi)}
    out["brier_model_minus_grid_point"] = float(np.mean((pm - y) ** 2) - np.mean((pg - y) ** 2))
    return out


def by_favourite_rank(df: pd.DataFrame, src: str, max_rank: int = 6) -> list[dict]:
    """Mean predicted vs observed, grouped by the source's win-probability rank in the race."""
    rank = df.groupby(["year", "round"])[f"p_win_{src}"].rank(ascending=False, method="first").astype(int)
    out = []
    for r in range(1, max_rank + 1):
        g = df[rank == r]
        row = {"rank": r, "n": int(len(g))}
        for t in TARGETS:
            k, n = float(g[TARGETS[t]].sum()), len(g)
            row[t] = {"mean_pred": float(g[f"p_{t}_{src}"].mean()), "observed": k / n,
                      "ci95": list(wilson(k, n))}
        out.append(row)
    return out


# ---------------------------------------------------------------- plot

def plot(df: pd.DataFrame, result: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"model": "#d62728", "grid": "#7f7f7f"}
    titles = {"win": "P(win)", "podium": "P(podium)", "p23": "P(2nd or 3rd)"}
    targets = list(TARGETS)
    fig, axes = plt.subplots(2, len(targets), figsize=(5.2 * len(targets), 7.4),
                             gridspec_kw={"height_ratios": [3.2, 1]}, sharex="col")
    for j, t in enumerate(targets):
        ax, hx = axes[0, j], axes[1, j]
        ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls="--", zorder=0)
        edges = BINS[t]
        for k, src in enumerate(("grid", "model")):
            bins = [b for b in result["sources"][src][t]["bins"] if b["n"]]
            x = np.array([b["mean_pred"] for b in bins])
            yv = np.array([b["observed"] for b in bins])
            lo = np.array([b["ci95"][0] for b in bins])
            hi = np.array([b["ci95"][1] for b in bins])
            s = result["sources"][src][t]["summary"]
            ax.errorbar(x, yv, yerr=[yv - lo, hi - yv], fmt="o-", ms=4, lw=1.2, capsize=2,
                        color=colors[src], alpha=0.9 if src == "model" else 0.7,
                        label=f"{src}: Brier {s['brier']:.4f}, ECE {s['ece']:.4f}, "
                              f"slope {s['cal_slope']:.2f}")
            if src == "model":
                for b in bins:
                    ax.annotate(str(b["n"]), (b["mean_pred"], b["observed"]), fontsize=6,
                                xytext=(3, -8), textcoords="offset points", color=colors[src])
            hx.hist(df[f"p_{t}_{src}"], bins=edges, histtype="step", lw=1.4, color=colors[src],
                    label=src)
        top = edges[-2] + 0.15 if t == "p23" else 1.0
        ax.set_xlim(0, top)
        ax.set_ylim(0, top if t == "p23" else 1.0)
        ax.set_title(f"{titles[t]}: observed vs predicted")
        ax.set_ylabel("observed frequency (Wilson 95%)")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.25)
        hx.set_yscale("log")
        hx.set_xlabel("predicted probability")
        hx.set_ylabel("drivers (log)")
        hx.grid(alpha=0.25)
    yrs = result["test_years"]
    fig.suptitle(f"Walk-forward reliability {yrs[0]}-{yrs[-1]}: {result['n_races']} races, "
                 f"{result['n_drivers']} driver-races (two-stage v5 vs grid; numbers = drivers per bin)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description="Reliability diagrams over the walk-forward backtest")
    parser.add_argument("--start", type=int, default=2018, help="First test season")
    parser.add_argument("--end", type=int, default=None, help="Last test season (default: latest)")
    parser.add_argument("--n-sims", type=int, default=20_000, help="Monte Carlo races per prediction")
    parser.add_argument("--reuse", action="store_true", help=f"Reuse {DRIVERS_PATH.name} if present")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", filter=lambda r: r["name"] == "__main__")

    fm = load_feature_matrix().reset_index(drop=True)
    end = args.end or int(fm["year"].max())
    years = list(range(args.start, end + 1))

    if args.reuse and DRIVERS_PATH.exists():
        df = pd.read_parquet(DRIVERS_PATH)
        df = df[df["year"].isin(years)].reset_index(drop=True)
        logger.info(f"reused {DRIVERS_PATH.name}: {len(df)} rows")
    else:
        base = fm[fm["year"].isin(years)][["year", "round", "driver", "p1", "podium", "finish_position"]]
        base = base.rename(columns={"p1": "won"})
        fin = pd.to_numeric(base["finish_position"], errors="coerce")
        base["p23"] = fin.isin([2, 3]).astype(int)
        df = (base.merge(model_probs(fm, years, args.n_sims), on=["year", "round", "driver"])
                  .merge(grid_probs(fm, years, args.n_sims), on=["year", "round", "driver"]))
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(DRIVERS_PATH, index=False)
    for c in ("won", "podium", "p23"):
        df[c] = df[c].astype(int)

    result: dict = {
        "test_years": years,
        "n_races": int(df.groupby(["year", "round"]).ngroups),
        "n_drivers": int(len(df)),
        "n_sims": args.n_sims,
        "method": ("Out-of-sample walk-forward. Model: two-stage v5 (FEATURE_COLS, DEFAULT_PARAMS, "
                   "DEFAULT_SEEDS), T fitted on previous season's OOS predictions, probabilities "
                   "from two_stage.simulate with finish model. Grid: PL on grid, T fitted on earlier "
                   "races. Fixed bins; Wilson 95%; Brier decomposition on the bins; ECE count-weighted; "
                   "calibration slope/intercept from unpenalised logistic regression of outcome on "
                   "logit(p) (p clipped to [1e-4, 1-1e-4]); intervals from a race bootstrap."),
        "temperatures": {
            "model": {int(y): float(t) for y, t in df.groupby("year")["t_model"].first().items()},
            "grid": {int(y): float(t) for y, t in df.groupby("year")["t_grid"].first().items()},
        },
        "bin_edges": BINS,
        "sources": {},
        "bootstrap_95": {},
        "by_favourite_rank": {},
    }
    for src in ("model", "grid"):
        result["sources"][src] = {}
        for t, ycol in TARGETS.items():
            p = df[f"p_{t}_{src}"].to_numpy(float)
            y = df[ycol].to_numpy(float)
            result["sources"][src][t] = {"summary": summary(p, y, BINS[t]),
                                         "bins": bin_table(p, y, BINS[t])}
        result["by_favourite_rank"][src] = by_favourite_rank(df, src)
    for t in TARGETS:
        result["bootstrap_95"][t] = race_bootstrap(df, t)

    with open(OUT_DIR / "reliability.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    plot(df, result, OUT_DIR / "reliability.png")

    # ---- console report (ASCII only)
    print(f"\nWalk-forward {years[0]}-{years[-1]}: {result['n_races']} races, {result['n_drivers']} driver-races\n")
    for t in TARGETS:
        print(f"== {t} ==")
        print(f"{'src':6s} {'brier':>8s} {'rel':>8s} {'res':>8s} {'ece':>7s} {'logloss':>8s} "
              f"{'icpt':>6s} {'slope':>6s}")
        for src in ("model", "grid"):
            s = result["sources"][src][t]["summary"]
            print(f"{src:6s} {s['brier']:8.5f} {s['brier_reliability']:8.5f} {s['brier_resolution']:8.5f} "
                  f"{s['ece']:7.4f} {s['log_loss']:8.4f} {s['cal_intercept']:6.2f} {s['cal_slope']:6.2f}")
        bs = result["bootstrap_95"][t]
        print(f"  model slope 95% {bs['model_slope']}, grid slope 95% {bs['grid_slope']}")
        print(f"  Brier model-grid {bs['brier_model_minus_grid_point']:+.5f} 95% {bs['brier_model_minus_grid']}")
        print(f"  {'bin':>13s} {'n':>5s} {'pred':>6s} {'obs':>6s} {'ci95':>15s}")
        for b in result["sources"]["model"][t]["bins"]:
            if b["n"]:
                print(f"  [{b['lo']:.2f},{b['hi']:.2f}) {b['n']:5d} {b['mean_pred']:6.3f} {b['observed']:6.3f} "
                      f"[{b['ci95'][0]:.3f},{b['ci95'][1]:.3f}]")
        print()
    print("By model favourite rank (pred / obs):")
    for r in result["by_favourite_rank"]["model"]:
        print(f"  rank {r['rank']}: n={r['n']} " + "  ".join(
            f"{t} {r[t]['mean_pred']:.3f}/{r[t]['observed']:.3f}" for t in TARGETS))
    print(f"\nSaved {OUT_DIR / 'reliability.json'} and reliability.png")


if __name__ == "__main__":
    main()
