from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.features.build_features import FEATURE_COLS, load_feature_matrix
from src.ui import services

_PLOTLY_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#8b949e",
)


def _dark_axes() -> dict:
    return dict(gridcolor="#21262d", zerolinecolor="#30363d", color="#8b949e")


def render(*, year: int, round_num: int, shell: dict | None = None) -> None:
    context = shell or services.get_shell_snapshot(year, round_num)
    pred = context.get("prediction") if isinstance(context, dict) else None
    selected_driver = str(st.session_state.get("selected_driver", "") or "")

    shap_df = services.load_shap_values()
    imp_df = services.load_xgb_feature_importance()

    # ── Section 1: Global feature importance ─────────────────────────────
    st.markdown("<div class='f1-section-title'>Global Feature Impact -- Mean |SHAP| (P1)</div>", unsafe_allow_html=True)

    if shap_df is not None and not shap_df.empty:
        ignored = {"driver", "year", "round", "p1", "pred_p1"}
        feat_cols = [c for c in shap_df.columns if c not in ignored]

        if feat_cols:
            mean_abs = shap_df[feat_cols].abs().mean().sort_values(ascending=True).tail(20)
            fig_global = go.Figure(go.Bar(
                x=mean_abs.values,
                y=mean_abs.index,
                orientation="h",
                marker=dict(
                    color=mean_abs.values,
                    colorscale=[[0, "#1e3a5f"], [0.5, "#2563eb"], [1, "#e8002d"]],
                    showscale=False,
                ),
            ))
            fig_global.update_layout(
                height=480,
                xaxis=dict(title="Mean |SHAP|", **_dark_axes()),
                yaxis=dict(title="", **_dark_axes()),
                margin=dict(l=8, r=8, t=8, b=8),
                **_PLOTLY_DARK,
            )
            st.plotly_chart(fig_global, use_container_width=True)
    else:
        st.markdown(
            "<div class='f1-card f1-muted' style='padding:1.5rem; text-align:center;'>"
            "No SHAP data found. Retrain the model to generate SHAP values."
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='f1-divider'></div>", unsafe_allow_html=True)

    # ── Section 2: P1 / P2 / P3 XGB importance comparison ────────────────
    st.markdown("<div class='f1-section-title'>XGBoost Feature Importance -- P1 / P2 / P3</div>", unsafe_allow_html=True)

    if imp_df is not None and not imp_df.empty:
        top_feats = (
            imp_df.groupby("feature")["importance"].mean()
            .sort_values(ascending=False)
            .head(15)
            .index.tolist()
        )
        sub_imp = imp_df[imp_df["feature"].isin(top_feats)]

        fig_compare = px.bar(
            sub_imp.sort_values("importance", ascending=True),
            x="importance",
            y="feature",
            color="target",
            orientation="h",
            barmode="group",
            color_discrete_map={"p1": "#e8002d", "p2": "#f59e0b", "p3": "#3b82f6"},
        )
        fig_compare.update_layout(
            height=420,
            xaxis=dict(title="Importance", **_dark_axes()),
            yaxis=dict(title="", **_dark_axes()),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=8, r=8, t=8, b=8),
            **_PLOTLY_DARK,
        )
        st.plotly_chart(fig_compare, use_container_width=True)
    else:
        st.caption("XGBoost feature importance unavailable (model not loaded).")

    st.markdown("<div class='f1-divider'></div>", unsafe_allow_html=True)

    # ── Section 3: Driver waterfall for current context ───────────────────
    st.markdown("<div class='f1-section-title'>Driver SHAP Profile</div>", unsafe_allow_html=True)

    if pred is not None:
        driver_options = []
        for p in pred.get("predictions", []):
            d = str(p.get("driver", ""))
            if d:
                driver_options.append(d)
        if driver_options:
            driver_pick = st.selectbox(
                "Select driver",
                driver_options,
                index=driver_options.index(selected_driver) if selected_driver in driver_options else 0,
                key="shap_driver_pick",
                label_visibility="collapsed",
            )
            services.set_selected_driver(driver_pick)
        else:
            driver_pick = selected_driver
    else:
        driver_pick = selected_driver

    col_waterfall, col_context = st.columns([2, 1])

    with col_waterfall:
        if shap_df is not None and not shap_df.empty and driver_pick:
            ignored = {"driver", "year", "round", "p1", "pred_p1"}
            feat_cols = [c for c in shap_df.columns if c not in ignored]

            # Try to get race-specific SHAP first, fall back to driver average
            race_sub = shap_df[
                (shap_df.get("year", pd.Series(dtype=int)) == year) &
                (shap_df.get("round", pd.Series(dtype=int)) == round_num) &
                (shap_df["driver"].astype(str).str.upper() == driver_pick.upper())
            ] if {"year", "round"}.issubset(shap_df.columns) else pd.DataFrame()

            driver_sub = shap_df[shap_df["driver"].astype(str).str.upper() == driver_pick.upper()]

            if not race_sub.empty:
                profile = race_sub[feat_cols].iloc[0]
                subtitle = f"Race-specific SHAP -- {year} R{round_num:02d}"
            elif not driver_sub.empty:
                profile = driver_sub[feat_cols].mean()
                subtitle = "Driver historical mean SHAP (all races)"
            else:
                profile = None
                subtitle = ""

            if profile is not None and feat_cols:
                signed = profile.sort_values(ascending=True)
                signed_top = signed.abs().sort_values(ascending=True).tail(15).index
                signed = signed[signed_top]

                colors = ["#e8002d" if v < 0 else "#3b82f6" for v in signed.values]
                fig_wf = go.Figure(go.Bar(
                    x=signed.values,
                    y=signed.index,
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.4f}" for v in signed.values],
                    textposition="outside",
                    textfont=dict(size=10, color="#8b949e"),
                ))
                fig_wf.add_vline(x=0, line_width=1, line_dash="dash", line_color="#30363d")
                fig_wf.update_layout(
                    title=dict(text=subtitle, font=dict(size=11, color="#8b949e"), x=0),
                    height=420,
                    xaxis=dict(title="SHAP value (+ = raises P(Win))", **_dark_axes()),
                    yaxis=dict(title="", **_dark_axes()),
                    margin=dict(l=8, r=8, t=30, b=8),
                    **_PLOTLY_DARK,
                )
                st.plotly_chart(fig_wf, use_container_width=True)
            else:
                st.caption(f"No SHAP data found for {driver_pick}.")
        else:
            st.caption("Select a driver and ensure SHAP data is available.")

    with col_context:
        if pred and driver_pick:
            row = services.driver_prediction_row(pred, driver_pick)
            if row:
                team = str(row.get("team", "Unknown"))
                color = services.get_team_color(team)
                st.markdown(
                    f"<div class='f1-card' style='border-top: 3px solid {color}; margin-top:1.5rem;'>"
                    f"<div style='font-size:1.1rem; font-weight:700;'>{driver_pick}</div>"
                    f"<div class='f1-muted'>{team}</div>"
                    f"<div style='margin-top:0.8rem;'>"
                    f"<div class='f1-muted' style='font-size:0.72rem;'>P(WIN)</div>"
                    f"<div style='font-size:1.4rem; font-weight:700;'>{float(row.get('p_win', 0.0) or 0.0):.1%}</div>"
                    f"</div>"
                    f"<div style='margin-top:0.5rem;'>"
                    f"<div class='f1-muted' style='font-size:0.72rem;'>P(PODIUM)</div>"
                    f"<div style='font-size:1.1rem; font-weight:600;'>{float(row.get('p_podium', 0.0) or 0.0):.1%}</div>"
                    f"</div>"
                    f"<div style='margin-top:0.5rem;'>"
                    f"<div class='f1-muted' style='font-size:0.72rem;'>GRID</div>"
                    f"<div style='font-weight:600;'>{int(row.get('grid_position', 0) or 0)}</div>"
                    f"</div>"
                    f"<div style='margin-top:0.5rem;'>"
                    f"<div class='f1-muted' style='font-size:0.72rem;'>GAP TO POLE</div>"
                    f"<div style='font-weight:600;'>{float(row.get('quali_gap_to_pole_s', 0.0) or 0.0):.3f}s</div>"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<div class='f1-divider'></div>", unsafe_allow_html=True)

    # ── Section 4: Feature correlation ────────────────────────────────────
    st.markdown("<div class='f1-section-title'>Feature Correlation with P1 Win</div>", unsafe_allow_html=True)
    try:
        fm = load_feature_matrix()
        available = [c for c in FEATURE_COLS if c in fm.columns] + (["p1"] if "p1" in fm.columns else [])
        corr = (
            fm[available]
            .corr()["p1"]
            .drop("p1")
            .sort_values(key=abs, ascending=True)
            .tail(20)
        )
        bar_colors = ["#e8002d" if v < 0 else "#3b82f6" for v in corr.values]
        fig_corr = go.Figure(go.Bar(
            x=corr.values,
            y=corr.index,
            orientation="h",
            marker_color=bar_colors,
        ))
        fig_corr.add_vline(x=0, line_width=1, line_dash="dash", line_color="#30363d")
        fig_corr.update_layout(
            height=420,
            xaxis=dict(title="Pearson r with P(Win)", **_dark_axes()),
            yaxis=dict(title="", **_dark_axes()),
            margin=dict(l=8, r=8, t=8, b=8),
            **_PLOTLY_DARK,
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    except Exception as exc:
        st.caption(f"Correlation view unavailable: {exc}")
