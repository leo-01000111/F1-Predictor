"""
F1 Race Prediction Dashboard
Run with: streamlit run src/ui/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui import services
from src.ui.views import (
    render_control_panel,
    render_historical_accuracy,
    render_race_prediction,
    render_shap_analysis,
)

st.set_page_config(
    page_title="F1 Predictor",
    page_icon=":checkered_flag:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

POLL_SECONDS = services.get_ui_config()["task_poll_seconds"]


def _render_auto_pipeline_banner() -> None:
    task_id = st.session_state.get("_auto_infer_task_id")
    race_label = st.session_state.get("_auto_infer_race", "")
    if not task_id:
        return
    tasks = {t["task_id"]: t for t in services.get_background_tasks(limit=60)}
    task = tasks.get(task_id)
    if task is None:
        return
    status = str(task.get("status", ""))
    progress = int(task.get("progress", 0))
    msg = str(task.get("message", ""))
    if status in {"queued", "running"}:
        st.markdown(
            f"<div class='f1-pipeline-banner'>"
            f"<b>Auto-inference running</b> &mdash; {race_label} &nbsp; {progress}%"
            f"{(' &mdash; ' + msg) if msg else ''}"
            f"</div>",
            unsafe_allow_html=True,
        )
    elif status == "completed":
        st.markdown(
            f"<div class='f1-pipeline-banner'>"
            f"Inference complete for <b>{race_label}</b>. Predictions ready."
            f"</div>",
            unsafe_allow_html=True,
        )
        services.invalidate_data_caches()
    elif status == "failed":
        err = str(task.get("error", ""))
        st.markdown(
            f"<div class='f1-pipeline-banner f1-pipeline-banner-warn'>"
            f"Auto-inference failed for {race_label}: {err[:120]}"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_task_toasts() -> None:
    tasks = services.get_background_tasks(limit=40)
    seen: dict[str, str] = st.session_state.get("_task_notice_seen", {})
    dirty = False
    for task in tasks:
        task_id = task["task_id"]
        status = str(task.get("status", ""))
        if status not in {"completed", "failed"}:
            continue
        if seen.get(task_id) == status:
            continue
        kind = task.get("kind", "task")
        if hasattr(st, "toast"):
            if status == "completed":
                st.toast(f"{kind} completed", icon="✅")
            else:
                st.toast(f"{kind} failed: {task.get('error', '')[:60]}", icon="❌")
        seen[task_id] = status
        dirty = True
    if dirty:
        st.session_state["_task_notice_seen"] = seen
        services.invalidate_data_caches()


def _render_header(year: int, round_num: int) -> tuple[int, int]:
    idx = services.load_prediction_index()

    col_title, col_race, col_status, col_actions = st.columns([1, 3, 2, 1.2], gap="small")

    with col_title:
        st.markdown("<div class='f1-wordmark' style='padding-top:0.35rem;'>F1 PREDICTOR</div>", unsafe_allow_html=True)

    with col_race:
        if idx.empty:
            c1, c2 = st.columns(2)
            with c1:
                year = int(st.number_input("Year", min_value=2018, max_value=2035, value=year, step=1, label_visibility="collapsed"))
            with c2:
                round_num = int(st.number_input("Round", min_value=1, max_value=30, value=round_num, step=1, label_visibility="collapsed"))
        else:
            race_options = []
            race_lookup: dict[str, tuple[int, int]] = {}
            for _, row in idx.sort_values(["year", "round"], ascending=[False, False]).iterrows():
                label = services.race_label(int(row["year"]), int(row["round"]), str(row.get("circuit", "")))
                race_options.append(label)
                race_lookup[label] = (int(row["year"]), int(row["round"]))
            default_ix = 0
            for i, label in enumerate(race_options):
                y, r = race_lookup[label]
                if y == year and r == round_num:
                    default_ix = i
                    break
            picked = st.selectbox("Race", race_options, index=default_ix, label_visibility="collapsed")
            year, round_num = race_lookup[picked]
        services.set_selected_race(year, round_num)

    with col_status:
        shell = services.get_shell_snapshot(year, round_num)
        freshness = shell.get("freshness", {}) or {}
        active = shell.get("active_model", {}) or {}
        model_fam = str(active.get("model_family") or "XGB").upper()
        f_tone = str(freshness.get("tone", "neutral"))
        f_label = str(freshness.get("label", "No prediction"))
        chips = services.chip(f_label, tone=f_tone) + services.chip(f"Model: {model_fam}", tone="neutral")
        st.markdown(f"<div style='padding-top:0.3rem;'>{chips}</div>", unsafe_allow_html=True)

    with col_actions:
        if st.button("Refresh", key="btn_header_refresh", use_container_width=True):
            services.invalidate_data_caches()
            st.rerun()

    st.markdown("<div class='f1-divider'></div>", unsafe_allow_html=True)
    return year, round_num


def main() -> None:
    services.init_ui_state()
    services.inject_theme()

    # Auto-pipeline: once per session, check if inference is needed
    services.auto_trigger_inference()

    year, round_num = services.get_selected_race()
    _render_task_toasts()

    year, round_num = _render_header(year, round_num)
    shell = services.get_shell_snapshot(year, round_num)

    _render_auto_pipeline_banner()

    tab_labels = ["Race Desk", "SHAP Analysis", "Historical", "Control Panel"]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        render_race_prediction(year=year, round_num=round_num, shell=shell)
    with tabs[1]:
        render_shap_analysis(year=year, round_num=round_num, shell=shell)
    with tabs[2]:
        render_historical_accuracy(year=year, round_num=round_num, shell=shell)
    with tabs[3]:
        render_control_panel(year=year, round_num=round_num, shell=shell)


if __name__ == "__main__":
    main()
