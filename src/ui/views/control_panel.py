from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.ui import services


def _submit_task(kind: str, payload: dict) -> str:
    task_id = services.submit_background_task(kind, payload)
    if hasattr(st, "toast"):
        st.toast(f"{kind} queued", icon="⏳")
    st.success(f"Queued `{kind}` -- task `{task_id[:8]}`")
    return task_id


def _render_preflight(report: dict) -> None:
    status = str(report.get("status", "unknown"))
    st.markdown(services.task_status_chip(status), unsafe_allow_html=True)
    checks = report.get("checks", []) or []
    if checks:
        st.dataframe(pd.DataFrame(checks), width="stretch", hide_index=True)
    for e in report.get("errors", []) or []:
        st.error(str(e))
    for w in report.get("warnings", []) or []:
        st.warning(str(w))


def render(*, year: int, round_num: int, shell: dict | None = None) -> None:
    context = shell or services.get_shell_snapshot(year, round_num)
    active = context.get("active_model") if isinstance(context, dict) else None
    if not isinstance(active, dict):
        active = services.load_active_model() or {}

    # ── Model info ────────────────────────────────────────────────────────
    st.markdown("<div class='f1-section-title'>Active Model</div>", unsafe_allow_html=True)
    if active:
        a1, a2, a3 = st.columns(3)
        a1.metric("Family", str(active.get("model_family", "?")).upper())
        a2.metric("Features", active.get("feature_count", "?"))
        a3.metric("Winner Acc (holdout)", f"{float(active.get('winner_accuracy_holdout', 0.0) or 0.0):.1%}")
        st.caption(f"Train years: {active.get('train_years', [])}  |  Cal: {active.get('calibration_years', [])}")
    else:
        st.caption("No active model manifest found.")

    st.markdown("<div class='f1-divider'></div>", unsafe_allow_html=True)

    # ── Context + action ──────────────────────────────────────────────────
    st.markdown("<div class='f1-section-title'>Run Actions</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        cp_year = int(st.number_input("Year", min_value=2018, max_value=2035, value=int(year), step=1, key="cp_year"))
    with c2:
        cp_round = int(st.number_input("Round", min_value=1, max_value=30, value=int(round_num), step=1, key="cp_round"))
    services.set_selected_race(cp_year, cp_round)

    has_pred = services.has_prediction(cp_year, cp_round)
    has_actual = services.has_actual_result(cp_year, cp_round)
    recommended = services.recommended_next_action(cp_year, cp_round)

    st.markdown(
        services.chip(f"Pred: {'present' if has_pred else 'missing'}", tone="good" if has_pred else "warn")
        + services.chip(f"Actual: {'present' if has_actual else 'missing'}", tone="good" if has_actual else "warn")
        + services.chip(f"Recommended: {recommended}", tone="neutral"),
        unsafe_allow_html=True,
    )

    action_choice = st.radio(
        "Action",
        ["run_inference", "record_actual", "post_race_retrain", "quick_retrain", "full_retrain"],
        horizontal=True, key="cp_action",
    )

    report = services.get_preflight_report(action_choice, cp_year, cp_round)
    _render_preflight(report)

    if action_choice in ("run_inference", "record_actual", "post_race_retrain"):
        payload = {"year": cp_year, "round_num": cp_round}
    else:
        payload = {}

    btn_col, _ = st.columns([1, 3])
    with btn_col:
        if st.button(
            action_choice.replace("_", " ").title(),
            type="primary",
            disabled=not bool(report.get("ready", False)),
            width="stretch",
            key="cp_queue_btn",
        ):
            _submit_task(action_choice, payload)

        if not bool(report.get("ready", False)):
            reason = "; ".join(str(e) for e in (report.get("errors", []) or [])[:2]) or "Preflight must pass."
            st.caption(f"Blocked: {reason}")

    st.markdown("<div class='f1-divider'></div>", unsafe_allow_html=True)

    # ── Task monitor ──────────────────────────────────────────────────────
    st.markdown("<div class='f1-section-title'>Task Queue</div>", unsafe_allow_html=True)
    active_tasks = services.get_background_tasks(limit=120)
    if active_tasks:
        task_rows = [
            {
                "ID": t["task_id"][:8],
                "Kind": t.get("kind", ""),
                "Status": t.get("status", ""),
                "Progress": f"{int(t.get('progress', 0))}%",
                "Message": t.get("message", "")[:60],
                "Updated": t.get("updated_at", "")[:19],
            }
            for t in active_tasks
        ]
        st.dataframe(pd.DataFrame(task_rows), width="stretch", height=220, hide_index=True)
    else:
        st.caption("No tasks.")

    history_rows = services.load_task_history(limit=300)
    if history_rows:
        st.markdown("<div class='f1-section-title' style='margin-top:0.6rem;'>Inspect Task</div>", unsafe_allow_html=True)
        ids = [f"{t['task_id'][:8]} | {t['kind']} | {t['status']}" for t in history_rows]
        selected = st.selectbox("Task", ids, key="cp_task_pick", label_visibility="collapsed")
        selected_id = selected.split("|")[0].strip()
        task = next((t for t in history_rows if t["task_id"].startswith(selected_id)), None)
        if task:
            t1, t2 = st.columns(2)
            t1.metric("Status", str(task.get("status", "-")))
            t2.metric("Progress", f"{int(task.get('progress', 0))}%")
            if task.get("error"):
                st.error(str(task["error"]))
            logs = task.get("log_tail", []) or []
            if logs:
                with st.expander("Logs", expanded=False):
                    st.code("\n".join(logs[-60:]))
            if st.button("Cancel Task", disabled=task.get("status") not in {"queued", "running"}, key="cp_cancel"):
                ok = services.cancel_background_task(task["task_id"])
                st.warning("Cancellation requested." if ok else "Could not cancel.")

    summary = services.build_task_history_summary(limit=300)
    st.download_button(
        "Export Task History",
        data=json.dumps(summary, indent=2),
        file_name=f"task_history_{cp_year}_R{cp_round:02d}.json",
        mime="application/json",
        key="cp_export_history",
    )
