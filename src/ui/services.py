from __future__ import annotations

import inspect
import json
import sys
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import requests
import streamlit as st

from config.settings import settings
from src.models.model_registry import load_active_model_metadata
from src.pipeline.actions import ActionContext, InlineContext, PipelineActionRunner

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PREDICTIONS_DIR = ROOT / "data" / "predictions"
RESULTS_DIR = ROOT / "data" / "results"
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
TASK_HISTORY_PATH = RESULTS_DIR / "ui_task_history.jsonl"

NAV_PAGES = ["Race Desk", "SHAP Analysis", "Historical", "Control Panel"]
RIGHT_RAIL_SECTIONS = ["driver", "tasks", "actions"]
STATUS_TONES = {
    "queued": "warn",
    "running": "warn",
    "completed": "good",
    "failed": "bad",
    "cancelled": "neutral",
}

TEAM_COLORS = {
    "Red Bull Racing": "#3671C6",
    "Red Bull": "#3671C6",
    "Ferrari": "#E8002D",
    "Mercedes": "#27F4D2",
    "McLaren": "#FF8000",
    "Aston Martin": "#229971",
    "Alpine": "#0093CC",
    "Williams": "#64C4FF",
    "RB": "#6692FF",
    "Racing Bulls": "#6692FF",
    "AlphaTauri": "#6692FF",
    "Kick Sauber": "#52E252",
    "Sauber": "#52E252",
    "Audi": "#52E252",
    "Haas": "#B6BABD",
    "Haas F1 Team": "#B6BABD",
    "Cadillac": "#6B7280",
}
DEFAULT_COLOR = "#8F96A3"

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --f1-bg: #0d1117;
  --f1-surface: #161b22;
  --f1-surface2: #1c2128;
  --f1-card: #1c2128;
  --f1-border: #30363d;
  --f1-ink: #e6edf3;
  --f1-subtle: #8b949e;
  --f1-accent: #e8002d;
  --f1-accent2: #ff6b35;
  --f1-good: #3fb950;
  --f1-warn: #d29922;
  --f1-bad: #f85149;
  --f1-blue: #58a6ff;
  --f1-radius: 10px;
}

html, body, [class*="css"] {
  font-family: "Inter", "Segoe UI", system-ui, sans-serif;
  background-color: var(--f1-bg) !important;
  color: var(--f1-ink);
}

.stApp {
  background-color: var(--f1-bg) !important;
  color: var(--f1-ink);
  font-size: var(--f1-font-base, 0.94rem);
}

.stApp [data-testid="stHeader"] { background: transparent; }
.stApp [data-testid="stToolbar"],
.stApp [data-testid="stStatusWidget"],
.stApp [data-testid="stDecoration"] {
  visibility: hidden; height: 0; position: fixed;
}

.block-container {
  padding-top: 1rem !important;
  padding-bottom: 1rem !important;
  max-width: 100% !important;
  padding-left: 1.2rem !important;
  padding-right: 1.2rem !important;
}

.stApp h1 {
  color: var(--f1-ink);
  font-size: 1.6rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin-bottom: 0.3rem;
}

.stApp h2, .stApp h3 {
  color: var(--f1-ink);
  letter-spacing: -0.01em;
}

.stApp p, .stApp label, .stApp li,
.stApp [data-testid="stMarkdownContainer"] {
  color: var(--f1-ink);
}

/* Inputs */
.stApp [data-baseweb="input"] input,
.stApp [data-baseweb="select"] > div,
.stApp textarea {
  background-color: var(--f1-surface2) !important;
  color: var(--f1-ink) !important;
  border-color: var(--f1-border) !important;
}

.stApp [data-baseweb="select"] [data-testid="stSelectbox"] {
  background-color: var(--f1-surface2) !important;
}

/* Buttons */
.stApp .stButton > button {
  border-radius: 8px;
  border: 1px solid var(--f1-border);
  background: var(--f1-surface2);
  color: var(--f1-ink);
  font-weight: 600;
  font-size: 0.88rem;
  transition: border-color 0.15s, background 0.15s;
}

.stApp .stButton > button:hover {
  border-color: var(--f1-blue);
  background: #1f2937;
}

.stApp .stButton > button[kind="primary"] {
  background: var(--f1-accent);
  border-color: var(--f1-accent);
  color: #fff;
}

.stApp .stButton > button[kind="primary"]:hover {
  background: #c91a28;
  border-color: #c91a28;
}

/* Tabs */
.stApp [data-testid="stTabs"] [role="tablist"] {
  gap: 0.25rem;
  border-bottom: 1px solid var(--f1-border);
  padding-bottom: 0;
}

.stApp [data-testid="stTabs"] [role="tab"] {
  background: transparent;
  border: none;
  color: var(--f1-subtle);
  font-weight: 500;
  font-size: 0.9rem;
  padding: 0.5rem 0.9rem;
  border-radius: 6px 6px 0 0;
  transition: color 0.15s;
}

.stApp [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  color: var(--f1-ink);
  border-bottom: 2px solid var(--f1-accent);
  background: transparent;
}

/* Metrics */
.stApp [data-testid="stMetricLabel"] { color: var(--f1-subtle) !important; font-size: 0.78rem !important; }
.stApp [data-testid="stMetricValue"] { color: var(--f1-ink) !important; font-size: 1.35rem !important; font-weight: 700 !important; }
.stApp [data-testid="stCaptionContainer"] { color: var(--f1-subtle) !important; font-size: 0.79rem !important; }

/* DataFrames */
.stApp [data-testid="stDataFrame"] * { font-size: 0.88rem !important; color: var(--f1-ink) !important; }
.stApp [data-testid="stDataFrame"] th { background-color: var(--f1-surface) !important; color: var(--f1-subtle) !important; font-weight: 600 !important; text-transform: uppercase; font-size: 0.75rem !important; }

/* Radio */
.stApp [role="radiogroup"] > label {
  background: var(--f1-surface2);
  border: 1px solid var(--f1-border);
  border-radius: 8px;
  padding: 0.22rem 0.5rem;
  color: var(--f1-subtle);
}

.stApp [role="radiogroup"] > label:has(input:checked) {
  background: #1f2d44;
  border-color: var(--f1-blue);
  color: var(--f1-ink);
}

/* Expander */
.stApp [data-testid="stExpander"] {
  background: var(--f1-surface2) !important;
  border: 1px solid var(--f1-border) !important;
  border-radius: var(--f1-radius) !important;
}

/* ---- Custom classes ---- */

.f1-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.5rem 0 0.75rem 0;
  border-bottom: 1px solid var(--f1-border);
  margin-bottom: 0.75rem;
}

.f1-wordmark {
  font-weight: 800;
  font-size: 1.05rem;
  letter-spacing: 0.06em;
  color: var(--f1-accent);
  white-space: nowrap;
}

.f1-section-title {
  font-size: 0.78rem;
  font-weight: 700;
  color: var(--f1-subtle);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin: 0 0 0.5rem 0;
}

.f1-subtitle { color: var(--f1-subtle); font-size: 0.85rem; }
.f1-muted { color: var(--f1-subtle); font-size: 0.85rem; }

.f1-chip {
  display: inline-flex;
  align-items: center;
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  border: 1px solid var(--f1-border);
  background: var(--f1-surface2);
  color: var(--f1-subtle);
  font-size: 0.75rem;
  font-weight: 600;
  margin: 0 0.3rem 0.3rem 0;
  letter-spacing: 0.01em;
}

.f1-chip-good { border-color: #1e4620; background: #0d2218; color: var(--f1-good); }
.f1-chip-warn { border-color: #4a3500; background: #1f1800; color: var(--f1-warn); }
.f1-chip-bad  { border-color: #4a1616; background: #200d0d; color: var(--f1-bad); }

.f1-card {
  background: var(--f1-card);
  border: 1px solid var(--f1-border);
  border-radius: var(--f1-radius);
  padding: var(--f1-card-pad, 0.9rem);
  box-shadow: 0 1px 3px rgba(0,0,0,0.4);
}

.f1-accent-card {
  border-left: 4px solid var(--f1-accent);
}

.f1-podium-card {
  background: var(--f1-card);
  border: 1px solid var(--f1-border);
  border-radius: var(--f1-radius);
  padding: 1rem;
  height: 100%;
}

.f1-divider { border-top: 1px solid var(--f1-border); margin: 0.6rem 0; }

.f1-kpi-question {
  color: var(--f1-subtle);
  font-size: 0.85rem;
  margin-bottom: 0.6rem;
}

.f1-pipeline-banner {
  background: linear-gradient(90deg, #1a2540 0%, #1c2128 100%);
  border: 1px solid #2d4a7a;
  border-left: 3px solid var(--f1-blue);
  border-radius: var(--f1-radius);
  padding: 0.55rem 0.9rem;
  margin-bottom: 0.75rem;
  font-size: 0.85rem;
  color: var(--f1-blue);
}

.f1-pipeline-banner-warn {
  border-left-color: var(--f1-warn);
  color: var(--f1-warn);
  border-color: #4a3500;
  background: linear-gradient(90deg, #1a1600 0%, #1c2128 100%);
}

.f1-status-dot {
  display: inline-block;
  width: 7px; height: 7px;
  border-radius: 999px;
  margin-right: 5px;
}
.f1-status-dot-good   { background: var(--f1-good); }
.f1-status-dot-warn   { background: var(--f1-warn); }
.f1-status-dot-bad    { background: var(--f1-bad); }
.f1-status-dot-neutral { background: var(--f1-subtle); }

.f1-race-selector {
  display: flex; align-items: center; gap: 0.5rem;
}
</style>
"""


def get_ui_config() -> dict[str, Any]:
    density = str(getattr(settings, "ui_density_default", "dense") or "dense").strip().lower()
    if density not in {"dense", "comfort"}:
        density = "dense"

    task_poll_seconds = int(getattr(settings, "ui_task_poll_seconds", 5) or 5)
    task_poll_seconds = max(3, min(task_poll_seconds, 30))

    max_visible_blocks = int(getattr(settings, "ui_max_visible_blocks", 6) or 6)
    max_visible_blocks = max(4, min(max_visible_blocks, 10))

    live_enabled = bool(getattr(settings, "ui_enable_live_widgets", False))

    return {
        "density_default": density,
        "task_poll_seconds": task_poll_seconds,
        "max_visible_blocks": max_visible_blocks,
        "live_widgets_enabled": live_enabled,
    }


def _density_css_vars(density: str) -> str:
    if str(density).strip().lower() == "comfort":
        return "--f1-gap: 1.0rem; --f1-card-pad: 1.1rem; --f1-font-base: 1.0rem;"
    return "--f1-gap: 0.65rem; --f1-card-pad: 0.85rem; --f1-font-base: 0.94rem;"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _utc_now().isoformat()


def _parse_iso(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def inject_theme() -> None:
    density = str(st.session_state.get("layout_density", get_ui_config()["density_default"]))
    dynamic_css = f"""
<style>
:root {{ {_density_css_vars(density)} }}
.block-container {{
  padding-top: 1.2rem;
  padding-bottom: 0.9rem;
  max-width: 100% !important;
  padding-left: 0.9rem;
  padding-right: 0.9rem;
}}
</style>
"""
    st.markdown(APP_CSS + dynamic_css, unsafe_allow_html=True)


def get_team_color(team: str) -> str:
    team_norm = str(team or "").lower()
    for key, color in TEAM_COLORS.items():
        key_norm = key.lower()
        if key_norm in team_norm or team_norm in key_norm:
            return color
    return DEFAULT_COLOR


def chip(label: str, tone: str = "neutral") -> str:
    cls = "f1-chip"
    if tone == "good":
        cls += " f1-chip-good"
    elif tone == "warn":
        cls += " f1-chip-warn"
    elif tone == "bad":
        cls += " f1-chip-bad"
    return f"<span class='{cls}'>{label}</span>"


def task_status_chip(status: str) -> str:
    tone = STATUS_TONES.get(str(status).lower(), "neutral")
    return chip(str(status).upper(), tone=tone)


def tone_dot(tone: str) -> str:
    t = str(tone or "neutral")
    return f"<span class='f1-status-dot f1-status-dot-{t}'></span>"


def race_label(year: int, round_num: int, circuit: str | None = None) -> str:
    if circuit:
        circuit_txt = str(circuit).replace("_", " ").title()
        return f"{year} R{round_num:02d} - {circuit_txt}"
    return f"{year} R{round_num:02d}"


def _safe_json_load(path: Path) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data(ttl=60)
def load_prediction_index() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(PREDICTIONS_DIR.glob("*.json")):
        doc = _safe_json_load(path)
        if doc is None:
            continue
        rows.append(
            {
                "year": int(doc.get("year", 0) or 0),
                "round": int(doc.get("round", 0) or 0),
                "circuit": str(doc.get("circuit_key", "") or ""),
                "race_date": str(doc.get("race_date", "") or ""),
                "generated_at": str(doc.get("generated_at", "") or ""),
                "path": str(path),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["year", "round", "circuit", "race_date", "generated_at", "path"])

    return pd.DataFrame(rows).sort_values(["year", "round"]).reset_index(drop=True)


@st.cache_data(ttl=60)
def load_prediction(year: int, round_num: int) -> Optional[dict]:
    path = PREDICTIONS_DIR / f"{int(year)}_R{int(round_num):02d}.json"
    if not path.exists():
        return None
    return _safe_json_load(path)


@st.cache_data(ttl=60)
def load_latest_prediction() -> Optional[dict]:
    idx = load_prediction_index()
    if idx.empty:
        return None
    row = idx.iloc[-1]
    return load_prediction(int(row["year"]), int(row["round"]))


@st.cache_data(ttl=60)
def load_prior_prediction(year: Optional[int] = None, round_num: Optional[int] = None) -> Optional[dict]:
    idx = load_prediction_index()
    if idx.empty:
        return None

    if year is None or round_num is None:
        if len(idx) < 2:
            return None
        row = idx.iloc[-2]
        return load_prediction(int(row["year"]), int(row["round"]))

    sub = idx[(idx["year"] < year) | ((idx["year"] == year) & (idx["round"] < round_num))]
    if sub.empty:
        return None
    row = sub.iloc[-1]
    return load_prediction(int(row["year"]), int(row["round"]))


@st.cache_data(ttl=60)
def load_all_predictions() -> list[dict]:
    docs: list[dict] = []
    idx = load_prediction_index()
    for _, row in idx.iterrows():
        doc = load_prediction(int(row["year"]), int(row["round"]))
        if doc is not None:
            docs.append(doc)
    return docs


@st.cache_data(ttl=60)
def load_actual_results() -> list[dict]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for p in sorted(RESULTS_DIR.glob("*_actual.json")):
        doc = _safe_json_load(p)
        if doc is not None:
            rows.append(doc)
    return rows


@st.cache_data(ttl=60)
def load_actual_result(year: int, round_num: int) -> Optional[dict]:
    path = RESULTS_DIR / f"{int(year)}_R{int(round_num):02d}_actual.json"
    if not path.exists():
        return None
    return _safe_json_load(path)


@st.cache_data
def load_holdout_results() -> Optional[pd.DataFrame]:
    path = MODELS_DIR / "holdout_race_results.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


@st.cache_data
def load_calibration_curve() -> Optional[pd.DataFrame]:
    path = MODELS_DIR / "calibration_curve.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


@st.cache_data
def load_shap_values() -> Optional[pd.DataFrame]:
    path = MODELS_DIR / "shap_values_p1.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


@st.cache_data
def load_holdout_metrics() -> Optional[dict]:
    active = load_active_model_metadata() or {}
    family = str(active.get("model_family", "")).strip().lower()

    candidates = []
    if family:
        candidates.append(MODELS_DIR / f"holdout_metrics_{family}.json")
    candidates.append(MODELS_DIR / "holdout_metrics.json")

    for path in candidates:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return None


@st.cache_data
def load_active_model() -> Optional[dict]:
    return load_active_model_metadata()


@st.cache_data(ttl=60)
def build_search_catalog() -> list[dict[str, str]]:
    catalog: dict[str, dict[str, str]] = {}
    for doc in load_all_predictions():
        year = int(doc.get("year", 0) or 0)
        round_num = int(doc.get("round", 0) or 0)
        circuit = str(doc.get("circuit_key", "") or "")
        race_id = f"race:{year}:{round_num}"
        catalog[race_id] = {
            "id": race_id,
            "type": "race",
            "label": f"Race  {race_label(year, round_num, circuit)}",
            "search": f"race {year} round {round_num} {circuit}",
            "year": str(year),
            "round": str(round_num),
        }

        if circuit:
            circuit_id = f"circuit:{circuit.lower()}"
            catalog[circuit_id] = {
                "id": circuit_id,
                "type": "circuit",
                "label": f"Circuit  {circuit.replace('_', ' ').title()}",
                "search": f"circuit {circuit}",
                "year": str(year),
                "round": str(round_num),
            }

        for row in doc.get("predictions", []):
            driver = str(row.get("driver", "") or "").strip()
            team = str(row.get("team", "") or "").strip()
            if driver:
                drv_id = f"driver:{driver}"
                catalog[drv_id] = {
                    "id": drv_id,
                    "type": "driver",
                    "label": f"Driver  {driver} ({team or 'Unknown team'})",
                    "search": f"driver {driver} {team}",
                    "year": str(year),
                    "round": str(round_num),
                }
            if team:
                team_id = f"team:{team.lower()}"
                catalog[team_id] = {
                    "id": team_id,
                    "type": "team",
                    "label": f"Team  {team}",
                    "search": f"team {team}",
                    "year": str(year),
                    "round": str(round_num),
                }
    return list(catalog.values())


def suggest_search_targets(query: str, limit: int = 8) -> list[dict[str, str]]:
    q = str(query or "").strip().lower()
    catalog = build_search_catalog()
    if not catalog:
        return []

    if not q:
        races = [c for c in catalog if c["type"] == "race"]
        drivers = [c for c in catalog if c["type"] == "driver"][:3]
        return (races[-3:] + drivers)[:limit]

    scored = []
    for item in catalog:
        text = item.get("search", "").lower()
        label = item.get("label", "").lower()
        if q in text or q in label:
            score = 1.0
        else:
            score = max(SequenceMatcher(None, q, text).ratio(), SequenceMatcher(None, q, label).ratio())
        if score >= 0.35:
            scored.append((score, item))

    scored.sort(key=lambda x: (x[0], x[1].get("type") == "race"), reverse=True)
    return [item for _, item in scored[:limit]]


def get_prediction_freshness(pred: Optional[dict]) -> dict[str, Any]:
    if not pred:
        return {"tone": "bad", "label": "No prediction", "age_hours": None}

    generated = _parse_iso(pred.get("generated_at"))
    if generated is None:
        return {"tone": "warn", "label": "Unknown generation time", "age_hours": None}

    age_hours = (_utc_now() - generated).total_seconds() / 3600.0
    if age_hours <= 6:
        tone = "good"
        label = f"Fresh ({age_hours:.1f}h old)"
    elif age_hours <= 24:
        tone = "warn"
        label = f"Aging ({age_hours:.1f}h old)"
    else:
        tone = "bad"
        label = f"Stale ({age_hours:.1f}h old)"
    return {"tone": tone, "label": label, "age_hours": age_hours}


def get_actual_status(year: int, round_num: int) -> dict[str, Any]:
    doc = load_actual_result(year, round_num)
    if doc is None:
        return {"exists": False, "tone": "warn", "label": "Actual result not recorded"}

    acc = doc.get("accuracy", {})
    winner_hit = bool(acc.get("winner_hit", False))
    overlap = acc.get("podium_drivers_in_top3_predicted", 0)
    return {
        "exists": True,
        "winner_hit": winner_hit,
        "podium_overlap": overlap,
        "tone": "good" if winner_hit else "warn",
        "label": f"Actual recorded  Winner hit: {'yes' if winner_hit else 'no'}  Podium overlap: {overlap}/3",
    }


def driver_prediction_row(pred: Optional[dict], driver: str) -> Optional[dict]:
    if not pred:
        return None
    for row in pred.get("predictions", []):
        if str(row.get("driver", "")).strip().upper() == str(driver).strip().upper():
            return row
    return None


def build_driver_trend(driver: str, year: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for doc in load_all_predictions():
        if int(doc.get("year", 0) or 0) != int(year):
            continue
        for row in doc.get("predictions", []):
            if str(row.get("driver", "")).strip().upper() != str(driver).strip().upper():
                continue
            rows.append(
                {
                    "year": int(doc.get("year", 0) or 0),
                    "round": int(doc.get("round", 0) or 0),
                    "driver": str(row.get("driver", "")),
                    "team": str(row.get("team", "")),
                    "p_win": float(row.get("p_win", 0.0) or 0.0),
                    "p_podium": float(row.get("p_podium", 0.0) or 0.0),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["year", "round", "driver", "team", "p_win", "p_podium"])
    return pd.DataFrame(rows).sort_values(["year", "round"]).reset_index(drop=True)


def get_latest_race_hint() -> tuple[int, int]:
    idx = load_prediction_index()
    if not idx.empty:
        row = idx.iloc[-1]
        return int(row["year"]), int(row["round"])
    now = datetime.now()
    return int(now.year), 1


def _touch_context_version() -> int:
    version = int(st.session_state.get("global_context_version", 0)) + 1
    st.session_state["global_context_version"] = version
    return version


def init_ui_state() -> None:
    cfg = get_ui_config()
    default_year, default_round = get_latest_race_hint()
    st.session_state.setdefault("nav_page", NAV_PAGES[0])
    st.session_state.setdefault("selected_year", default_year)
    st.session_state.setdefault("selected_round", default_round)
    st.session_state.setdefault("selected_driver", "")
    st.session_state.setdefault("selected_team_filter", "All")
    st.session_state.setdefault("search_query", "")
    st.session_state.setdefault("search_pick_label", "")
    st.session_state.setdefault("task_notice_seen", {})
    st.session_state.setdefault("control_model_choice", "XGB-only (recommended)")
    st.session_state.setdefault("layout_density", cfg["density_default"])
    st.session_state.setdefault("right_rail_section", RIGHT_RAIL_SECTIONS[0])
    st.session_state.setdefault("live_widgets_enabled", cfg["live_widgets_enabled"])
    st.session_state.setdefault("task_poll_active", False)
    st.session_state.setdefault("global_context_version", 1)


def set_nav_page(page: str) -> None:
    if page in NAV_PAGES and page != st.session_state.get("nav_page"):
        st.session_state["nav_page"] = page
        _touch_context_version()


def set_selected_race(year: int, round_num: int) -> None:
    y = int(year)
    r = int(round_num)
    if y != int(st.session_state.get("selected_year", y)) or r != int(st.session_state.get("selected_round", r)):
        st.session_state["selected_year"] = y
        st.session_state["selected_round"] = r
        _touch_context_version()


def set_selected_driver(driver: str) -> None:
    value = str(driver or "")
    if value != str(st.session_state.get("selected_driver", "")):
        st.session_state["selected_driver"] = value
        _touch_context_version()


def set_selected_team(team: str) -> None:
    value = str(team or "All")
    if value != str(st.session_state.get("selected_team_filter", "All")):
        st.session_state["selected_team_filter"] = value
        _touch_context_version()


def set_density_mode(mode: str) -> None:
    value = str(mode or "dense").strip().lower()
    if value not in {"dense", "comfort"}:
        value = "dense"
    if value != str(st.session_state.get("layout_density", "dense")):
        st.session_state["layout_density"] = value
        _touch_context_version()


def set_right_rail_section(section: str) -> None:
    value = str(section or RIGHT_RAIL_SECTIONS[0]).strip().lower()
    if value not in RIGHT_RAIL_SECTIONS:
        value = RIGHT_RAIL_SECTIONS[0]
    if value != str(st.session_state.get("right_rail_section", RIGHT_RAIL_SECTIONS[0])):
        st.session_state["right_rail_section"] = value
        _touch_context_version()


def set_live_widgets_enabled(enabled: bool) -> None:
    value = bool(enabled)
    if value != bool(st.session_state.get("live_widgets_enabled", False)):
        st.session_state["live_widgets_enabled"] = value
        _touch_context_version()


def get_selected_race() -> tuple[int, int]:
    return int(st.session_state.get("selected_year", get_latest_race_hint()[0])), int(
        st.session_state.get("selected_round", get_latest_race_hint()[1])
    )


def load_xgb_feature_importance() -> Optional[pd.DataFrame]:
    try:
        from src.models.xgb_model import XGBRacePredictor

        model = XGBRacePredictor.load("xgb_race")
        return model.feature_importances()
    except Exception:
        return None


def invalidate_data_caches() -> None:
    st.cache_data.clear()
    _touch_context_version()

def _runner_context(progress_cb: Callable[[int, str], None], log_cb: Callable[[str], None]) -> InlineContext:
    return InlineContext(progress_cb=progress_cb, log_cb=log_cb)


def run_inference_action(
    *,
    year: int,
    round_num: int,
    progress_cb: Callable[[int, str], None],
    log_cb: Callable[[str], None],
    **_kwargs,
) -> dict:
    runner = PipelineActionRunner(root=ROOT)
    ctx = _runner_context(progress_cb, log_cb)
    return runner.run_inference(year=year, round_num=round_num, context=ctx)


def record_actual_action(
    *,
    year: int,
    round_num: int,
    progress_cb: Callable[[int, str], None],
    log_cb: Callable[[str], None],
) -> dict:
    runner = PipelineActionRunner(root=ROOT)
    ctx = _runner_context(progress_cb, log_cb)
    return runner.record_actual_result(year=year, round_num=round_num, context=ctx)


def quick_retrain_action(progress_cb: Callable[[int, str], None], log_cb: Callable[[str], None]) -> dict:
    runner = PipelineActionRunner(root=ROOT)
    ctx = _runner_context(progress_cb, log_cb)
    return runner.quick_retrain(context=ctx)


def full_retrain_action(progress_cb: Callable[[int, str], None], log_cb: Callable[[str], None]) -> dict:
    runner = PipelineActionRunner(root=ROOT)
    ctx = _runner_context(progress_cb, log_cb)
    return runner.full_retrain(context=ctx)


class TaskCancelledError(RuntimeError):
    pass


class CancellableActionContext(ActionContext):
    def __init__(
        self,
        progress_cb: Callable[[int, str], None],
        log_cb: Callable[[str], None],
        cancel_event: threading.Event,
    ) -> None:
        self._progress_cb = progress_cb
        self._log_cb = log_cb
        self._cancel_event = cancel_event

    def progress(self, value: int, message: str = "") -> None:
        self._progress_cb(int(max(0, min(value, 100))), str(message or ""))

    def log(self, message: str) -> None:
        self._log_cb(str(message))

    def check_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise TaskCancelledError("Cancellation requested.")


@dataclass(slots=True)
class UiTaskRecord:
    task_id: str
    kind: str
    payload: dict[str, Any]
    status: str
    progress: int
    message: str
    created_at: str
    updated_at: str
    finished_at: str = ""
    logs: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "payload": self.payload,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
            "log_tail": self.logs[-40:],
        }


_TASK_LOCK = threading.Lock()
_TASK_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="streamlit-ui-task")
_TASKS: dict[str, UiTaskRecord] = {}
_TASK_RUNTIME: dict[str, dict[str, Any]] = {}
_TASK_ORDER: list[str] = []


def _set_task_fields(task_id: str, **kwargs: Any) -> None:
    with _TASK_LOCK:
        task = _TASKS.get(task_id)
        if task is None:
            return
        for key, value in kwargs.items():
            setattr(task, key, value)
        task.updated_at = _iso_now()


def _append_task_log(task_id: str, message: str) -> None:
    line = str(message or "").strip()
    if not line:
        return
    with _TASK_LOCK:
        task = _TASKS.get(task_id)
        if task is None:
            return
        task.logs.append(line)
        if len(task.logs) > 400:
            task.logs = task.logs[-400:]
        task.updated_at = _iso_now()


def _persist_task_history(task: UiTaskRecord) -> None:
    TASK_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = task.summary()
    row["log_tail"] = task.logs[-60:]
    with open(TASK_HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _task_signature(kind: str, payload: dict[str, Any]) -> str:
    return f"{kind}|{json.dumps(payload, sort_keys=True, separators=(',', ':'))}"


def _find_active_duplicate_task(kind: str, payload: dict[str, Any]) -> Optional[str]:
    sig = _task_signature(kind, payload)
    for task_id in reversed(_TASK_ORDER):
        task = _TASKS.get(task_id)
        if task is None:
            continue
        if task.status not in {"queued", "running"}:
            continue
        if _task_signature(task.kind, task.payload) == sig:
            return task.task_id
    return None


def _run_task(task_id: str) -> None:
    with _TASK_LOCK:
        task = _TASKS.get(task_id)
        runtime = _TASK_RUNTIME.get(task_id)
        if task is None or runtime is None:
            return
        cancel_event = runtime["cancel_event"]
        task.status = "running"
        task.progress = max(task.progress, 1)
        task.message = "Task started."
        task.updated_at = _iso_now()
        kind = task.kind
        payload = dict(task.payload)

    runner = PipelineActionRunner(root=ROOT)

    def on_progress(value: int, message: str) -> None:
        _set_task_fields(task_id, progress=value, message=message or "Running")

    def on_log(message: str) -> None:
        _append_task_log(task_id, message)

    context = CancellableActionContext(on_progress, on_log, cancel_event)
    result: dict[str, Any] | None = None
    error_msg = ""
    status = "completed"

    try:
        if kind == "run_inference":
            result = runner.run_inference(
                year=int(payload["year"]),
                round_num=int(payload["round_num"]),
                context=context,
            )
        elif kind == "record_actual":
            result = runner.record_actual_result(
                year=int(payload["year"]),
                round_num=int(payload["round_num"]),
                context=context,
            )
        elif kind == "quick_retrain":
            result = runner.quick_retrain(context=context)
        elif kind == "full_retrain":
            result = runner.full_retrain(context=context)
        elif kind == "post_race_retrain":
            result = runner.post_race_retrain(
                year=int(payload.get("year", 2026)),
                round_num=int(payload.get("round_num", 1)),
                context=context,
            )
        else:
            raise RuntimeError(f"Unsupported task kind: {kind}")
    except TaskCancelledError:
        status = "cancelled"
        error_msg = "Task cancelled by user."
    except Exception as exc:
        status = "failed"
        error_msg = str(exc)

    now = _iso_now()
    with _TASK_LOCK:
        task = _TASKS.get(task_id)
        if task is None:
            return
        task.status = status
        task.progress = 100 if status == "completed" else task.progress
        task.message = "Completed" if status == "completed" else "Cancelled" if status == "cancelled" else (error_msg or "Failed")
        task.result = result
        task.error = error_msg
        task.finished_at = now
        task.updated_at = now
        _persist_task_history(task)


def _new_task_id() -> str:
    return uuid.uuid4().hex[:12]


def submit_background_task(kind: str, payload: Optional[dict[str, Any]] = None) -> str:
    payload = payload or {}

    with _TASK_LOCK:
        dup = _find_active_duplicate_task(kind, payload)
        if dup:
            return dup

    task_id = _new_task_id()
    now = _iso_now()
    task = UiTaskRecord(
        task_id=task_id,
        kind=kind,
        payload=payload,
        status="queued",
        progress=0,
        message="Queued",
        created_at=now,
        updated_at=now,
    )

    with _TASK_LOCK:
        _TASKS[task_id] = task
        _TASK_ORDER.append(task_id)
        cancel_event = threading.Event()
        _TASK_RUNTIME[task_id] = {"cancel_event": cancel_event, "future": None}

    future = _TASK_EXECUTOR.submit(_run_task, task_id)
    with _TASK_LOCK:
        runtime = _TASK_RUNTIME.get(task_id)
        if runtime is not None:
            runtime["future"] = future
    return task_id


def cancel_background_task(task_id: str) -> bool:
    with _TASK_LOCK:
        task = _TASKS.get(task_id)
        runtime = _TASK_RUNTIME.get(task_id)
        if task is None or runtime is None:
            return False

        future: Future | None = runtime.get("future")
        cancel_event: threading.Event = runtime["cancel_event"]

        if task.status == "queued" and future is not None and future.cancel():
            task.status = "cancelled"
            task.message = "Cancelled before start."
            task.updated_at = _iso_now()
            task.finished_at = task.updated_at
            _persist_task_history(task)
            return True

        if task.status == "running":
            cancel_event.set()
            task.message = "Cancellation requested..."
            task.updated_at = _iso_now()
            return True

    return False


def get_background_tasks(limit: int = 50) -> list[dict[str, Any]]:
    with _TASK_LOCK:
        ids = _TASK_ORDER[-int(max(1, limit)) :]
        rows = [_TASKS[i].summary() for i in ids if i in _TASKS]
    rows.sort(key=lambda x: x["created_at"], reverse=True)
    return rows


def get_background_task(task_id: str) -> Optional[dict[str, Any]]:
    with _TASK_LOCK:
        task = _TASKS.get(task_id)
        return task.summary() if task else None


def load_task_history(limit: int = 80) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if TASK_HISTORY_PATH.exists():
        with open(TASK_HISTORY_PATH, encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                try:
                    rows.append(json.loads(text))
                except Exception:
                    continue

    active = get_background_tasks(limit=120)
    by_id = {r["task_id"]: r for r in rows}
    for task in active:
        by_id[task["task_id"]] = task

    merged = list(by_id.values())
    merged.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return merged[: int(max(1, limit))]


def has_prediction(year: int, round_num: int) -> bool:
    return (PREDICTIONS_DIR / f"{int(year)}_R{int(round_num):02d}.json").exists()


def has_actual_result(year: int, round_num: int) -> bool:
    return (RESULTS_DIR / f"{int(year)}_R{int(round_num):02d}_actual.json").exists()


def recommended_next_action(year: int, round_num: int) -> str:
    if not has_prediction(year, round_num):
        return "run_inference"
    if not has_actual_result(year, round_num):
        return "record_actual"
    return "post_race_retrain"


def runtime_estimate_label(kind: str) -> str:
    labels = {
        "run_inference": "~1-3 min",
        "record_actual": "~30-90 sec",
        "quick_retrain": "~3-8 min",
        "full_retrain": "~10-30+ min",
        "post_race_retrain": "~5-12 min",
    }
    return labels.get(kind, "varies")

def build_task_history_summary(limit: int = 200) -> dict[str, Any]:
    rows = load_task_history(limit=limit)
    by_status: dict[str, int] = {}
    by_kind: dict[str, int] = {}

    for row in rows:
        status = str(row.get("status", "unknown"))
        kind = str(row.get("kind", "unknown"))
        by_status[status] = by_status.get(status, 0) + 1
        by_kind[kind] = by_kind.get(kind, 0) + 1

    return {
        "generated_at": _iso_now(),
        "total_rows": len(rows),
        "status_counts": by_status,
        "kind_counts": by_kind,
        "recent": rows[:25],
    }


def get_preflight_report(action: str, year: int, round_num: int) -> dict[str, Any]:
    kind = str(action or "").strip().lower()
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    valid_year = 2018 <= int(year) <= 2035
    valid_round = 1 <= int(round_num) <= 30

    checks.append({"name": "year_range", "ok": valid_year, "detail": f"year={year}"})
    checks.append({"name": "round_range", "ok": valid_round, "detail": f"round={round_num}"})

    has_pred = has_prediction(year, round_num)
    has_act = has_actual_result(year, round_num)

    if kind == "run_inference":
        if has_pred:
            warnings.append("Prediction file already exists for this race context.")
    elif kind == "record_actual":
        if not has_pred:
            errors.append("Prediction file is missing. Run inference before recording actual results.")
        if has_act:
            warnings.append("Actual result file already exists for this race context.")
    elif kind in {"quick_retrain", "full_retrain"}:
        feature_matrix = PROCESSED_DIR / "feature_matrix.parquet"
        if not feature_matrix.exists():
            errors.append("Feature matrix missing at data/processed/feature_matrix.parquet.")
    elif kind == "post_race_retrain":
        raw_race = ROOT / "data" / "raw" / "race_results.parquet"
        if not raw_race.exists():
            errors.append("race_results.parquet missing. Run f1_historical data collection first.")
        if not has_actual_result(year, round_num):
            warnings.append("Actual result not yet recorded for this round. Run 'record_actual' first.")
    else:
        errors.append(f"Unsupported action kind: {kind}")

    if not valid_year:
        errors.append("Year is outside supported UI range (2018-2035).")
    if not valid_round:
        errors.append("Round is outside supported UI range (1-30).")

    ready = len(errors) == 0
    status = "ok" if ready and not warnings else "warn" if ready else "error"
    return {
        "action": kind,
        "year": int(year),
        "round": int(round_num),
        "ready": ready,
        "status": status,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }


def get_shell_snapshot(year: int, round_num: int) -> dict[str, Any]:
    pred = load_prediction(year, round_num)
    prior = load_prior_prediction(year, round_num)
    freshness = get_prediction_freshness(pred)
    actual = get_actual_status(year, round_num)
    active_model = load_active_model() or {}

    tasks = get_background_tasks(limit=80)
    running = [t for t in tasks if str(t.get("status", "")).lower() in {"queued", "running"}]
    task_poll_active = bool(running)
    st.session_state["task_poll_active"] = task_poll_active

    return {
        "year": int(year),
        "round": int(round_num),
        "race_label": race_label(int(year), int(round_num), str(pred.get("circuit_key", "") if pred else "")),
        "prediction": pred,
        "prior_prediction": prior,
        "freshness": freshness,
        "actual": actual,
        "active_model": active_model,
        "tasks": tasks,
        "running_tasks": running,
        "layout_density": str(st.session_state.get("layout_density", get_ui_config()["density_default"])),
        "right_rail_section": str(st.session_state.get("right_rail_section", RIGHT_RAIL_SECTIONS[0])),
        "live_widgets_enabled": bool(st.session_state.get("live_widgets_enabled", get_ui_config()["live_widgets_enabled"])),
        "task_poll_active": task_poll_active,
        "global_context_version": int(st.session_state.get("global_context_version", 1)),
    }


def get_right_rail_state(year: int, round_num: int) -> dict[str, Any]:
    pred = load_prediction(year, round_num)
    driver = str(st.session_state.get("selected_driver", "") or "")

    if pred and not driver and pred.get("predictions"):
        driver = str(pred["predictions"][0].get("driver", ""))
        if driver:
            st.session_state["selected_driver"] = driver

    driver_row = driver_prediction_row(pred, driver) if driver else None
    tasks = get_background_tasks(limit=120)
    active_tasks = [t for t in tasks if str(t.get("status", "")).lower() in {"queued", "running"}]
    recent = sorted(tasks, key=lambda t: t.get("updated_at", ""), reverse=True)[:8]

    return {
        "selected_driver": driver,
        "driver_row": driver_row,
        "active_tasks": active_tasks,
        "recent_tasks": recent,
        "recommended_action": recommended_next_action(year, round_num),
        "preflight_run_inference": get_preflight_report("run_inference", year, round_num),
    }


def _openf1_get(endpoint: str, params: dict[str, Any], timeout_s: float = 0.8) -> list[dict[str, Any]]:
    try:
        resp = requests.get(f"https://api.openf1.org/v1/{endpoint}", params=params, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _latest_openf1_session(year: int, round_num: int) -> Optional[dict[str, Any]]:
    sessions = _openf1_get("sessions", {"year": int(year), "round_number": int(round_num)}, timeout_s=0.8)
    if not sessions:
        return None

    def key_fn(item: dict[str, Any]) -> str:
        return str(item.get("date_start") or item.get("date_end") or "")

    sessions = sorted(sessions, key=key_fn)
    return sessions[-1] if sessions else None

def _openf1_widget_group(session_key: Any) -> dict[str, list[dict[str, Any]]]:
    fetch_plan = {
        "laps": ("laps", {"session_key": session_key}),
        "intervals": ("intervals", {"session_key": session_key}),
        "weather": ("weather", {"session_key": session_key}),
    }

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="openf1-widget") as pool:
        futures = {
            name: pool.submit(_openf1_get, endpoint, params, 0.8)
            for name, (endpoint, params) in fetch_plan.items()
        }

        payload: dict[str, list[dict[str, Any]]] = {}
        for name, fut in futures.items():
            try:
                payload[name] = fut.result(timeout=1.0)
            except Exception:
                payload[name] = []

    return payload


def _live_widgets_from_prediction(pred: Optional[dict], prior: Optional[dict]) -> dict[str, Any]:
    if not pred:
        return {
            "source": "none",
            "available": False,
            "message": "No live data and no prediction artifact available.",
            "widgets": {},
        }

    weather = pred.get("weather", {}) or {}
    top = (pred.get("predictions", []) or [{}])[0]
    prior_map = {}
    if prior:
        for row in prior.get("predictions", []):
            prior_map[str(row.get("driver", ""))] = float(row.get("p_win", 0.0) or 0.0)
    top_driver = str(top.get("driver", "-"))
    top_delta = float(top.get("p_win", 0.0) or 0.0) - prior_map.get(top_driver, float(top.get("p_win", 0.0) or 0.0))

    return {
        "source": "artifact",
        "available": True,
        "message": "Using latest persisted artifacts (live feed unavailable).",
        "widgets": {
            "session_state": {
                "title": "Session State",
                "status": "Artifact mode",
                "detail": race_label(int(pred.get("year", 0) or 0), int(pred.get("round", 0) or 0), str(pred.get("circuit_key", ""))),
            },
            "recent_lap_delta": {
                "title": "Recent Lap Delta",
                "value": f"{top_driver} {top_delta:+.1%}",
                "detail": "Delta vs prior-round win probability",
            },
            "interval_snapshot": {
                "title": "Interval Snapshot",
                "value": f"Top P(Win): {float(top.get('p_win', 0.0) or 0.0):.1%}",
                "detail": "Model-derived contender gap",
            },
            "track_condition": {
                "title": "Track Condition Summary",
                "value": f"Temp {float(weather.get('temp_c_mean', 0.0) or 0.0):.1f} C | Rain {float(weather.get('precip_mm_total', 0.0) or 0.0):.1f} mm",
                "detail": "From forecast artifact",
            },
        },
    }


@st.cache_data(ttl=8)
def get_live_widget_state(year: int, round_num: int) -> dict[str, Any]:
    cfg = get_ui_config()
    enabled = bool(st.session_state.get("live_widgets_enabled", cfg["live_widgets_enabled"]))
    if not enabled:
        return {
            "enabled": False,
            "source": "disabled",
            "available": False,
            "message": "Live widgets disabled by UI configuration.",
            "widgets": {},
        }

    pred = load_prediction(year, round_num)
    prior = load_prior_prediction(year, round_num)
    session = _latest_openf1_session(year, round_num)

    if not session or session.get("session_key") is None:
        fallback = _live_widgets_from_prediction(pred, prior)
        return {
            "enabled": True,
            "source": fallback["source"],
            "available": fallback["available"],
            "message": fallback["message"],
            "widgets": fallback["widgets"],
            "entitled": False,
        }

    session_key = session.get("session_key")
    live_group = _openf1_widget_group(session_key)
    laps = live_group.get("laps", [])
    intervals = live_group.get("intervals", [])
    weather = live_group.get("weather", [])

    if not laps and not intervals and not weather:
        fallback = _live_widgets_from_prediction(pred, prior)
        return {
            "enabled": True,
            "source": fallback["source"],
            "available": fallback["available"],
            "message": fallback["message"],
            "widgets": fallback["widgets"],
            "entitled": False,
        }

    lap_delta_txt = "n/a"
    if laps:
        lap_df = pd.DataFrame(laps)
        if {"driver_number", "lap_duration"}.issubset(lap_df.columns):
            lap_df["lap_duration"] = pd.to_numeric(lap_df["lap_duration"], errors="coerce")
            lap_df = lap_df.dropna(subset=["lap_duration"])
            if not lap_df.empty:
                best = lap_df.groupby("driver_number", as_index=False)["lap_duration"].min().sort_values("lap_duration")
                if len(best) >= 2:
                    lead = float(best.iloc[0]["lap_duration"])
                    gap = float(best.iloc[1]["lap_duration"]) - lead
                    lap_delta_txt = f"Gap P2 to leader: {gap:.3f}s"

    interval_txt = "n/a"
    if intervals:
        int_df = pd.DataFrame(intervals)
        if not int_df.empty and "interval" in int_df.columns:
            vals = pd.to_numeric(int_df["interval"], errors="coerce").dropna()
            if not vals.empty:
                interval_txt = f"Median interval: {float(vals.median()):.2f}s"

    track_txt = "n/a"
    if weather:
        wdf = pd.DataFrame(weather)
        if not wdf.empty:
            t = pd.to_numeric(wdf.get("air_temperature"), errors="coerce").dropna()
            r = pd.to_numeric(wdf.get("rainfall"), errors="coerce").dropna()
            temp_txt = f"{float(t.iloc[-1]):.1f} C" if not t.empty else "n/a"
            rain_txt = f"{float(r.iloc[-1]):.2f} mm" if not r.empty else "n/a"
            track_txt = f"Temp {temp_txt} | Rain {rain_txt}"

    widgets = {
        "session_state": {
            "title": "Session State",
            "status": str(session.get("session_name") or session.get("session_type") or "Live session"),
            "detail": f"Session key {session_key}",
        },
        "recent_lap_delta": {
            "title": "Recent Lap Delta",
            "value": lap_delta_txt,
            "detail": "From OpenF1 lap stream",
        },
        "interval_snapshot": {
            "title": "Interval Snapshot",
            "value": interval_txt,
            "detail": "From OpenF1 intervals",
        },
        "track_condition": {
            "title": "Track Condition Summary",
            "value": track_txt,
            "detail": "From OpenF1 weather",
        },
    }

    return {
        "enabled": True,
        "source": "openf1",
        "available": True,
        "message": "Live widgets are using OpenF1 realtime feed.",
        "widgets": widgets,
        "entitled": True,
    }


def streamlit_button_supports_shortcut() -> bool:
    try:
        sig = inspect.signature(st.button)
        return "shortcut" in sig.parameters
    except Exception:
        return False


@st.cache_data(ttl=3600)
def get_season_schedule(year: int) -> list[dict]:
    """Fetch F1 season schedule via FastF1. Returns list of {round, event, quali_dt, race_dt}."""
    try:
        import fastf1
        import pandas as pd
        ROOT_LOCAL = Path(__file__).parent.parent.parent
        cache_dir = ROOT_LOCAL / "data" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        rows = []
        for _, ev in schedule.iterrows():
            quali_dt = pd.to_datetime(ev.get("Session4Date", None), utc=True, errors="coerce")
            race_dt = pd.to_datetime(ev.get("Session5Date", None), utc=True, errors="coerce")
            rows.append({
                "round": int(ev.get("RoundNumber", 0) or 0),
                "event": str(ev.get("EventName", "") or ""),
                "country": str(ev.get("Country", "") or ""),
                "quali_dt": quali_dt if not pd.isna(quali_dt) else None,
                "race_dt": race_dt if not pd.isna(race_dt) else None,
            })
        return [r for r in rows if r["round"] > 0]
    except Exception:
        return []


def get_next_unprocessed_race(year: int | None = None) -> tuple[int, int] | None:
    """Return (year, round) for the most recent quali-completed race with no prediction file."""
    import pandas as pd
    if year is None:
        year = datetime.now().year
    schedule = get_season_schedule(year)
    now = datetime.now(timezone.utc)
    for row in sorted(schedule, key=lambda x: x["round"]):
        round_num = row["round"]
        quali_dt = row.get("quali_dt")
        if quali_dt is None:
            continue
        try:
            if pd.isna(quali_dt):
                continue
        except Exception:
            pass
        if quali_dt > now:
            break
        if not has_prediction(year, round_num):
            return year, round_num
    return None


def auto_trigger_inference() -> str | None:
    """
    Called once per session on app launch.
    If there is a qualifying-completed race with no prediction file, auto-submits inference.
    Returns task_id if submitted, None otherwise.
    """
    if st.session_state.get("_auto_infer_checked"):
        return None
    st.session_state["_auto_infer_checked"] = True

    result = get_next_unprocessed_race()
    if result is None:
        return None

    year, round_num = result
    task_id = submit_background_task(
        "run_inference",
        {"year": year, "round_num": round_num},
    )
    st.session_state["_auto_infer_task_id"] = task_id
    st.session_state["_auto_infer_race"] = f"{year} R{round_num:02d}"
    return task_id









