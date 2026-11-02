# -*- coding: utf-8 -*-
"""
Model gym: stepwise search over model toggles with a lockbox
============================================================
Finds combinations of toggles (features added / removed, model options,
post-hoc calibration layers) that help together even if each alone did not.
Registry and evaluation: src/training/gym.py.

Search (--search stepwise):
  1. Start from production v5. Forward/backward stepwise over MODEL toggles on
     the DEV seasons only: each step tries every single flip of the current
     config and accepts the best if dev mean top-3 log loss drops by more than
     MIN_IMPROVEMENT (paired bootstrap CI reported). Step 0's trials are the
     ablation map (every flip from v5, incl. single-feature drops).
  2. Same stepwise over POST-HOC layers on the final config's cached
     predictions (still dev; every fitted parameter is out of sample per season).
  3. The chosen config is scored ONCE on the LOCKBOX seasons vs v5.
--search ablation: step 0 only (the ablation map), no lockbox.

Practice features exist from 2023 only: a model trained on them is first tested
in 2024. With --dev 2018-2023 the practice toggles cannot act on dev and are left
out of the search (frozen at v5 = on). --dev 2018-2024 --lockbox 2025-2026 lets
them act on one dev season (2024), at the cost of a 43-race lockbox.

Usage:
    python scripts/gym.py --toggles
    python scripts/gym.py --only add_ratings drop_reputation half_life=8 benter
    python scripts/gym.py --search stepwise --dev 2018-2023 --lockbox 2024-2026
    python scripts/gym.py --search stepwise --n-estimators 20 --seeds 1 --out-dir <tmp>   # smoke test

Writes models/experiments/gym/trials.jsonl (every trial) and
stepwise_<timestamp>.json (path, ablation map, lockbox). Cache (gitignored):
models/experiments/gym/cache/. No model files are touched.
"""
from __future__ import annotations

import argparse
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


def _years(s: str) -> list[int]:
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def _parse_flip(s: str) -> tuple[str, object]:
    name, eq, val = s.partition("=")
    if name not in G.REGISTRY:
        raise SystemExit(f"unknown toggle {name!r}; see --toggles")
    t = G.REGISTRY[name]
    if not eq:
        return name, True
    for v in t.values:
        if str(v) == val or (isinstance(v, float) and val.replace(".", "", 1).isdigit() and float(val) == v):
            return name, v
    raise SystemExit(f"{name}: value {val!r} not in {t.values}")


def list_toggles(fm, last_dev: int) -> None:
    print(f"{'toggle':34s} {'kind':8s} {'values':22s} {'search':7s} description / status (dev ends {last_dev})")
    for t in G.REGISTRY.values():
        st = G.toggle_status(t, fm, last_dev)
        print(f"{t.name:34s} {t.kind:8s} {str(t.values):22s} {'yes' if t.in_search else 'abl.':7s} "
              f"{t.description}" + (f"  [UNAVAILABLE: {st}]" if st else ""))
    print("\nExcluded by design:")
    for k, v in G.EXCLUDED.items():
        print(f"  {k}: {v}")


def _fmt(d) -> str:
    return f"{d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description="Stepwise toggle search with a lockbox")
    ap.add_argument("--toggles", action="store_true", help="List the registered toggles and exit")
    ap.add_argument("--only", nargs="+", metavar="TOGGLE[=VALUE]",
                    help="Score v5 and v5 + these flips on dev, then exit")
    ap.add_argument("--search", choices=["stepwise", "ablation"], default="stepwise")
    ap.add_argument("--dev", default="2018-2023")
    ap.add_argument("--lockbox", default="2024-2026")
    ap.add_argument("--min-improvement", type=float, default=G.MIN_IMPROVEMENT)
    ap.add_argument("--max-steps", type=int, default=G.MAX_STEPS)
    ap.add_argument("--include-singles", action="store_true",
                    help="Search single-feature drops too (default: ablation map only)")
    ap.add_argument("--no-singles-ablation", action="store_true", help="Skip single-feature drops at step 0")
    ap.add_argument("--no-warmup", action="store_true",
                    help="First dev season scored with T=1 (as scripts/backtest.py)")
    ap.add_argument("--n-estimators", type=int, default=None, help="Smoke tests only")
    ap.add_argument("--seeds", type=int, default=None, help="Seed-ensemble size (smoke tests only)")
    ap.add_argument("--with-kalman", action="store_true",
                    help="Merge Kalman filter columns (models/experiments/kalman_params.json) into the matrix")
    ap.add_argument("--out-dir", default=str(G.GYM_DIR))
    ap.add_argument("--cache-dir", default=str(G.CACHE_DIR))
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", filter=lambda r: r["name"] in ("src.training.gym", "__main__"))

    fm = load_feature_matrix()
    if args.with_kalman:
        from src.models.kalman_strength import KalmanParams, kalman_features
        params = KalmanParams(**json.loads((ROOT / "models" / "experiments" / "kalman_params.json").read_text()))
        raw = ROOT / "data" / "raw"
        kf = kalman_features(pd.read_parquet(raw / "quali_results.parquet"),
                             pd.read_parquet(raw / "race_results.parquet"), params)
        fm = fm.merge(kf, on=["year", "round", "driver"], how="left")
        logger.info(f"Kalman columns merged: {fm['kalman_pace_postq'].notna().mean():.1%} of rows")
    dev = _years(args.dev)
    lock = _years(args.lockbox)
    if args.toggles:
        list_toggles(fm, dev[-1])
        return
    if lock[0] <= dev[-1]:
        raise SystemExit("lockbox must come after dev")
    overrides = {"n_estimators": args.n_estimators} if args.n_estimators else {}
    gym = G.Gym(fm, out_dir=Path(args.out_dir), cache_dir=Path(args.cache_dir), overrides=overrides,
                n_seeds=args.seeds, warmup=not args.no_warmup)
    t0 = time.time()

    if args.only:
        cfg = G.normalize(dict(_parse_flip(s) for s in args.only))
        base, _ = gym.trial("only_base", {}, dev)
        _, rec = gym.trial("only", cfg, dev, ref=base, ref_label="v5")
        print(f"\nDev {dev[0]}-{dev[-1]} ({rec['metrics']['n_races']} races): {rec['label']} vs v5")
        for k, d in rec["delta"].items():
            print(f"  d {k:13s} {_fmt(d)}")
        print(f"  ll_top3 by season: {rec['metrics']['ll_top3_by_season']}")
        return

    unavailable = {t.name: G.toggle_status(t, fm, dev[-1]) for t in G.REGISTRY.values()
                   if G.toggle_status(t, fm, dev[-1])}
    logger.info(f"dev {dev[0]}-{dev[-1]}, lockbox {lock[0]}-{lock[-1]}; not searchable on dev: {sorted(unavailable)}")

    model = gym.stepwise({}, "model", dev, args.min_improvement,
                         1 if args.search == "ablation" else args.max_steps,
                         include_singles=args.include_singles, ablation_singles=not args.no_singles_ablation)
    out = {
        "run_id": gym.run_id, "args": vars(args), "dev": [dev[0], dev[-1]], "lockbox": [lock[0], lock[-1]],
        "warmup": gym.warmup, "min_improvement": args.min_improvement,
        "metric": "per-race top-3 Plackett-Luce log loss (ll_top3), lower is better; deltas = config - reference, "
                  "95% paired bootstrap over races",
        "unavailable_on_dev": unavailable, "excluded_by_design": G.EXCLUDED,
        "ablation_map_vs_v5": model["ablation"], "model_path": model["path"],
    }
    if args.search == "stepwise":
        post = gym.stepwise(model["final"], "posthoc", dev, args.min_improvement, args.max_steps)
        final = post["final"]
        out["posthoc_path"] = post["path"]
        out["posthoc_ablation_on_final_model"] = post["ablation"]
        out["final_config"] = final
        out["final_label"] = G.label(final)

        # Lockbox: scored once
        v5_lb, v5_rec = gym.trial("lockbox_v5", {}, lock)
        fin_lb, fin_rec = gym.trial("lockbox_final", final, lock, ref=v5_lb, ref_label="v5")
        lockbox = {"v5": v5_rec["metrics"], "final": fin_rec["metrics"], "final_minus_v5": fin_rec["delta"]}
        mcfg = G.split_cfg(final)[0]
        if mcfg != final and mcfg:
            # Diagnostic only (not a selection): the model-level part alone
            _, m_rec = gym.trial("lockbox_model_only", mcfg, lock, ref=v5_lb, ref_label="v5")
            lockbox["diagnostic_model_only_minus_v5"] = m_rec["delta"]
        d = fin_rec["delta"]["ll_top3"]
        lockbox["confirmed"] = bool(final) and d[2] < 0
        out["lockbox_result"] = lockbox

    out["runtime_min"] = round((time.time() - t0) / 60, 1)
    path = Path(args.out_dir) / f"stepwise_{gym.run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\nAblation map (dev {dev[0]}-{dev[-1]}, d ll_top3 vs v5, top 12):")
    for a in model["ablation"][:12]:
        print(f"  {a['flip']:34s} {_fmt(a['delta']['ll_top3'])}{'' if a['in_search'] else '  (ablation only)'}")
    print("\nModel path:")
    for p in model["path"]:
        print(f"  start: {p['label']} ll_top3 {p['metrics']['ll_top3']:.4f}" if p["step"] == 0 else
              f"  step {p['step']}: {'ACCEPT ' + str(p['flip']) if p.get('accepted') else 'stop (' + p['reason'] + ')'}")
    if args.search == "stepwise":
        print("Post-hoc path:")
        for p in post["path"][1:]:
            print(f"  step {p['step']}: {'ACCEPT ' + str(p['flip']) if p.get('accepted') else 'stop (' + p['reason'] + ')'}")
        lb = out["lockbox_result"]
        print(f"\nFinal: {out['final_label']}")
        print(f"Lockbox {lock[0]}-{lock[-1]} ({lb['v5']['n_races']} races): v5 {lb['v5']['ll_top3']:.4f}, "
              f"final {lb['final']['ll_top3']:.4f}")
        for k, dd in lb["final_minus_v5"].items():
            print(f"  d {k:13s} {_fmt(dd)}")
        print(f"  confirmed (CI upper < 0): {lb['confirmed']}")
    print(f"\nSaved {path} ({out['runtime_min']} min)")


if __name__ == "__main__":
    main()
