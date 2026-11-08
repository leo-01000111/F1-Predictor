# -*- coding: utf-8 -*-
"""
Exhaustive sweep over feature-group toggles (+ the quali-scaled temperature)
============================================================================
Every combination of the MODEL toggles below (2^8 = 256 column sets), each with
and without the post-hoc `quali_t` layer, is scored with the gym's production
walk-forward on the DEV seasons (top-3 log loss, paired bootstrap vs v5).
The config with the lowest dev ll_top3 is then scored ONCE on the LOCKBOX
seasons vs v5. With 512 configs the best dev score is optimistic by
construction (selection bias); only the lockbox result decides adoption.

Usage:
    python scripts/feature_sweep.py                       # dev 2018-2023, lockbox 2024-2026
    python scripts/feature_sweep.py --no-posthoc          # model toggles only (256)

Writes models/experiments/sweep/sweep_<run>.csv (every config) and
sweep_<run>.json (best, top 10, lockbox). Shares the gym prediction cache.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from loguru import logger

from src.features.build_features import load_feature_matrix
from src.training import gym as G

MODEL_TOGGLES = ["q_race", "q_prior", "dom_gain", "dom_teammate", "dom_traffic",
                 "dom_track_type", "dom_trend", "dom_regs"]
POSTHOC = ["quali_t"]
OUT_DIR = ROOT / "models" / "experiments" / "sweep"


def _years(s: str) -> list[int]:
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="2018-2023")
    ap.add_argument("--lockbox", default="2024-2026")
    ap.add_argument("--no-posthoc", action="store_true")
    args = ap.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    dev, lock = _years(args.dev), _years(args.lockbox)
    gym = G.Gym(load_feature_matrix(), out_dir=OUT_DIR)
    posthoc = [] if args.no_posthoc else POSTHOC
    toggles = MODEL_TOGGLES + posthoc
    base, _ = gym.trial("sweep_v5", {}, dev)

    rows = []
    t0 = time.time()
    combos = list(itertools.product([False, True], repeat=len(toggles)))
    # model-toggle-major order: the posthoc variants reuse the cached model predictions
    for i, bits in enumerate(combos):
        cfg = {t: True for t, on in zip(toggles, bits) if on}
        if not cfg:
            continue
        _, rec = gym.trial("sweep", cfg, dev, ref=base, ref_label="v5")
        d = rec["delta"]
        rows.append({"config": "+".join(cfg), "n_on": len(cfg), **{t: int(t in cfg) for t in toggles},
                     "ll_top3": rec["metrics"]["ll_top3"], "d_ll_top3": d["ll_top3"][0],
                     "d_lo": d["ll_top3"][1], "d_hi": d["ll_top3"][2],
                     "d_ll_winner": d["ll_winner"][0], "d_winner_hit": d["winner_hit"][0]})
        if (i + 1) % 16 == 0:
            print(f"  {i + 1}/{len(combos)} configs, {time.time() - t0:.0f}s", flush=True)

    res = pd.DataFrame(rows).sort_values("d_ll_top3").reset_index(drop=True)
    run = gym.run_id
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_DIR / f"sweep_{run}.csv", index=False)

    print(f"\nDev {dev[0]}-{dev[-1]}: {len(res)} configs vs v5 (ll_top3 {base['ll_top3'].mean():.4f})")
    print(f"  beat v5 on dev: {(res['d_ll_top3'] < 0).sum()}, with CI below 0: {(res['d_hi'] < 0).sum()}")
    print(res.head(15)[["config", "d_ll_top3", "d_lo", "d_hi", "d_ll_winner", "d_winner_hit"]].to_string())
    # Marginal effect of each toggle: mean d_ll_top3 with it on minus off
    marg = {t: float(res.loc[res[t] == 1, "d_ll_top3"].mean() - res.loc[res[t] == 0, "d_ll_top3"].mean())
            for t in toggles}
    print("\nMean effect of switching each toggle on (dev d_ll_top3, all other combos averaged):")
    for t, v in sorted(marg.items(), key=lambda kv: kv[1]):
        print(f"  {t:16s} {v:+.4f}")

    best_cfg = {t: True for t in res.loc[0, "config"].split("+")}
    v5_lb, _ = gym.trial("sweep_lockbox_v5", {}, lock)
    _, lb = gym.trial("sweep_lockbox_best", best_cfg, lock, ref=v5_lb, ref_label="v5")
    d = lb["delta"]
    print(f"\nLOCKBOX {lock[0]}-{lock[-1]} ({lb['metrics']['n_races']} races), best dev config "
          f"{res.loc[0, 'config']} vs v5:")
    for k, v in d.items():
        print(f"  d {k:13s} {v[0]:+.4f} [{v[1]:+.4f},{v[2]:+.4f}]")
    confirmed = d["ll_top3"][2] < 0
    print(f"  confirmed (ll_top3 CI below 0): {confirmed}")

    with open(OUT_DIR / f"sweep_{run}.json", "w", encoding="utf-8") as f:
        json.dump({"run_id": run, "dev": [dev[0], dev[-1]], "lockbox": [lock[0], lock[-1]],
                   "n_configs": len(res), "toggles": toggles, "marginal_effects": marg,
                   "top10": res.head(10).to_dict("records"),
                   "best": res.loc[0, "config"], "lockbox_delta": d, "lockbox_metrics": lb["metrics"],
                   "confirmed": confirmed}, f, indent=2, default=str)
    print(f"Saved {OUT_DIR / f'sweep_{run}.json'}")


if __name__ == "__main__":
    main()
