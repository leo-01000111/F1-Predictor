from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.features.build_features import FEATURE_COLS, load_feature_matrix
from src.ui import services


def render(*, year: int, round_num: int, shell: dict | None = None) -> None:
    st.title("Feature Explorer")
    st.markdown("<div class='f1-kpi-question'>Primary question: which features are driving winner probability and where is model behavior shifting?</div>", unsafe_allow_html=True)

    context = shell or services.get_shell_snapshot(year, round_num)
    pred = context.get("prediction") if isinstance(context, dict) else None
    selected_driver = str(st.session_state.get("selected_driver", "") or "")

    chips = [services.chip(f"Context: {services.race_label(year, round_num)}", tone="neutral")]
    if selected_driver:
        chips.append(services.chip(f"Driver: {selected_driver}", tone="good"))
    st.markdown("".join(chips), unsafe_allow_html=True)

    if pred is not None and selected_driver:
        row = services.driver_prediction_row(pred, selected_driver)
        if row:
            team = str(row.get("team", "Unknown"))
            color = services.get_team_color(team)
            card = f"""
            <div class='f1-card f1-accent-card' style='border-left-color: {color}'>
              <div class='f1-muted'>Selected Driver Snapshot</div>
              <div style='font-size:1.15rem; font-weight:700'>{selected_driver}</div>
              <div class='f1-muted'>{team}</div>
              <div style='margin-top:0.35rem;'>Grid: <b>{int(row.get('grid_position', 0) or 0)}</b></div>
              <div>Gap to Pole: <b>{float(row.get('quali_gap_to_pole_s', 0.0) or 0.0):.3f}s</b></div>
              <div>FP Best Gap: <b>{float(row.get('fp_best_gap_s', 0.0) or 0.0):.3f}s</b></div>
              <div>P(Win): <b>{float(row.get('p_win', 0.0) or 0.0):.1%}</b>  |  P(Podium): <b>{float(row.get('p_podium', 0.0) or 0.0):.1%}</b></div>
            </div>
            """
            st.markdown(card, unsafe_allow_html=True)

    st.markdown("### Feature Impact")
    shap_df = services.load_shap_values()
    if shap_df is None or shap_df.empty:
        st.info("No SHAP data found. Retrain a model from Control Panel to generate it.")
    else:
        ignored = {"driver", "year", "round", "p1", "pred_p1"}
        cols = [c for c in shap_df.columns if c not in ignored]

        if cols:
            mean_abs = shap_df[cols].abs().mean().sort_values(ascending=False).head(20)
            fig = px.bar(
                x=mean_abs.values,
                y=mean_abs.index,
                orientation="h",
                title="Global Feature Impact | Mean |SHAP| (Top 20)",
                labels={"x": "Mean |SHAP|", "y": "Feature"},
            )
            fig.update_layout(height=520, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch")

        if selected_driver and "driver" in shap_df.columns:
            sub = shap_df[shap_df["driver"].astype(str).str.upper() == selected_driver.upper()]
            if not sub.empty:
                cols = [c for c in sub.columns if c not in {"driver", "year", "round", "p1", "pred_p1"}]
                if cols:
                    local = sub[cols].abs().mean().sort_values(ascending=False).head(12)
                    fig_local = px.bar(
                        x=local.values,
                        y=local.index,
                        orientation="h",
                        title=f"{selected_driver} | Mean |SHAP| (Top 12)",
                        labels={"x": "Mean |SHAP|", "y": "Feature"},
                    )
                    fig_local.update_layout(height=400, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig_local, width="stretch")

    st.markdown("### Model Feature Importance")
    imp_df = services.load_xgb_feature_importance()
    if imp_df is not None and not imp_df.empty:
        target = st.selectbox("Target", ["p1", "p2", "p3"], index=0, key="xgb_target")
        sub = imp_df[imp_df["target"] == target].sort_values("importance", ascending=False).head(18)
        fig_imp = px.bar(sub, x="importance", y="feature", orientation="h", title=f"{target.upper()} Importance")
        fig_imp.update_layout(height=400, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_imp, width="stretch")
    else:
        st.info("XGBoost feature importance unavailable.")

    st.markdown("### Correlation with P1")
    try:
        fm = load_feature_matrix()
        corr = (
            fm[FEATURE_COLS + ["p1"]]
            .corr()["p1"]
            .drop("p1")
            .sort_values(key=abs, ascending=False)
            .head(20)
        )
        fig_corr = go.Figure(
            go.Bar(
                x=corr.values,
                y=corr.index,
                orientation="h",
                marker_color=["#d6242c" if x < 0 else "#2a72c9" for x in corr.values],
            )
        )
        fig_corr.add_vline(x=0, line_width=1, line_dash="dash", line_color="#8697ae")
        fig_corr.update_layout(height=520, yaxis=dict(autorange="reversed"), xaxis_title="Pearson r with P1", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_corr, width="stretch")
    except Exception as exc:
        st.warning(f"Could not compute correlation view: {exc}")

    model = pred.get("model", {}) if isinstance(pred, dict) else {}
    seasonal = model.get("seasonal_recalibration", {}) if isinstance(model, dict) else {}
    st.markdown("### Performance Cockpit Annotations")
    if seasonal:
        st.success("Seasonal recalibration metadata present in prediction artifact.")
    else:
        st.warning("Seasonal recalibration metadata missing or empty for this context.")

