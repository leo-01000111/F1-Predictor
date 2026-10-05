"""
Model artifact registry helpers.

This module centralizes how training jobs persist:
  - holdout metrics (generic + model-family-specific files)
  - active production model metadata
  - model snapshots for rollback
"""
from __future__ import annotations

import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOTS_DIR = MODELS_DIR / "snapshots"

ACTIVE_MODEL_PATH = MODELS_DIR / "active_model.json"


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return float(obj.item())
    return obj


def write_holdout_metrics(metrics: dict[str, Any], model_family: str) -> dict[str, Path]:
    """
    Persist holdout metrics to:
      - models/holdout_metrics.json
      - models/holdout_metrics_<model_family>.json
    """
    family = str(model_family).strip().lower()
    generic_path = MODELS_DIR / "holdout_metrics.json"
    family_path = MODELS_DIR / f"holdout_metrics_{family}.json"

    with open(generic_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    with open(family_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    return {"generic": generic_path, "family": family_path}


def write_active_model_metadata(
    *,
    model_family: str,
    model_artifact: str,
    calibrator_artifact: str,
    train_years: list[int],
    calibration_years: list[int],
    holdout_years: Optional[list[int]] = None,
    winner_accuracy: Optional[float] = None,
    source: str = "",
    feature_count: Optional[int] = None,
    feature_cols: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Persist metadata for the currently active production model.
    """
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_family": str(model_family).strip().lower(),
        "model_artifact": model_artifact,
        "calibrator_artifact": calibrator_artifact,
        "train_years": [int(y) for y in train_years],
        "calibration_years": [int(y) for y in calibration_years],
        "holdout_years": [int(y) for y in (holdout_years or [])],
        "winner_accuracy_holdout": float(winner_accuracy) if winner_accuracy is not None else None,
        "source": source,
    }

    if feature_count is not None:
        payload["feature_count"] = int(feature_count)
    if feature_cols is not None:
        payload["feature_cols"] = list(feature_cols)

    with open(ACTIVE_MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


def load_active_model_metadata() -> Optional[dict[str, Any]]:
    if not ACTIVE_MODEL_PATH.exists():
        return None
    with open(ACTIVE_MODEL_PATH, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Snapshot / rollback helpers
# --------------------------------------------------------------------------- #

def snapshot_model(name: str, keep_last: int = 5) -> Optional[Path]:
    """
    Copy models/{name}.pkl to models/snapshots/{name}_{timestamp}.pkl.
    Keeps the most recent `keep_last` snapshots and deletes older ones.
    Returns the snapshot path, or None if source does not exist.
    """
    src = MODELS_DIR / f"{name}.pkl"
    if not src.exists():
        return None

    SNAPSHOTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dst = SNAPSHOTS_DIR / f"{name}_{timestamp}.pkl"
    shutil.copy2(src, dst)

    # Prune: keep only the most recent `keep_last` snapshots for this name
    existing = sorted(SNAPSHOTS_DIR.glob(f"{name}_*.pkl"))
    for old in existing[:-keep_last]:
        old.unlink(missing_ok=True)

    return dst


def list_snapshots(name: str) -> list[Path]:
    """Return available snapshots for a model, newest first."""
    if not SNAPSHOTS_DIR.exists():
        return []
    return sorted(SNAPSHOTS_DIR.glob(f"{name}_*.pkl"), reverse=True)


def load_snapshot(name: str, index: int = 0) -> Any:
    """
    Load a snapshot by recency index (0 = newest, 1 = second newest, ...).
    Raises FileNotFoundError if no snapshot exists at that index.
    """
    snaps = list_snapshots(name)
    if not snaps or index >= len(snaps):
        raise FileNotFoundError(f"No snapshot at index {index} for model '{name}'")
    with open(snaps[index], "rb") as f:
        return pickle.load(f)
