from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ui import services

_PLOTLY_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#8b949e",
)


def _dark_axes() -> dict:
    return dict(gridcolor="#21262d", zerolinecolor="#30363d", color="#8b949e")


def render(*, year: int, round_num: int, shell: dict | None = None) -> None:
    metrics = services.load_holdout_metrics()
    race_df = services.load_holdout_results()
    calib_df = services.load_calibration_curve()

    # ── Holdout metrics KPI row ───────────────────────────────────────────
    st.markdown("<div class='f1-section-title'>Holdout Performance</div>", unsafe_allow_html=True)
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Winner Accuracy", f"{metrics.get('winner_accuracy', metrics.get('top1_accuracy', 0.0)):.1%}")
        c2.metric("Podium Accuracy", f"{metrics.get('podium_accuracy', 0.0):.1%}")
        c3.metric("P1 Brier", f"{metrics.get('p1', {}).get('brier_score', 0.0):.4f}")
        c4.metric("Holdout Races", int(metrics.get("n_holdout_races", 0)))
        st.caption(
            f"Train: {metrics.get('train_years', [])}  |  "
            f"Cal: {metrics.get('cal_years', [])}  |  "
            f"Holdout: {metrics.get('holdout_years', [])}"
        )
    else:
        st.caption("No holdout metrics found. Run training to generate.")

    st.markdown("<div class='f1-divider'></div>", unsafe_allow_html=True)

    # ── Reliability ledger ────────────────────────────────────────────────
    st.markdown("<div class='f1-section-title'>Race Ledger</div>", unsafe_allow_html=True)
    if race_df is not None and not race_df.empty:
        race_df = race_df.copy()
        if "year" in race_df.columns:
            years = sorted(int(y) for y in race_df["year"].dropna().unique())
            selected_years = st.multiselect("Year", years, default=years, key="hist_year_filter", label_visibility="collapsed")
            if selected_years:
                race_df = race_df[race_df["year"].isin(selected_years)]

        display_cols = [c for c in ["year", "round", "circuit", "predicted_winner", "actual_winner", "winner_correct", "podium_overlap", "ece_p1", "ece_p1_running"] if c in race_df.columns]
        if display_cols:
            ledger = race_df[display_cols].sort_values(["year", "round"] if {"year", "round"}.issubset(race_df.columns) else display_cols[:1]).copy()
            if {"year", "round"}.issubset(ledger.columns):
                ledger["context"] = (ledger["year"].astype(int) == year) & (ledger["round"].astype(int) == round_num)
                st.dataframe(ledger[["context"] + display_cols], width="stretch", height=300, hide_index=True)
            else:
                st.dataframe(ledger, width="stretch", height=300, hide_index=True)
    else:
        st.caption("No holdout race ledger available.")

    st.markdown("<div class='f1-divider'></div>", unsafe_allow_html=True)

    # ── Trend charts ──────────────────────────────────────────────────────
    st.markdown("<div class='f1-section-title'>Accuracy Trends</div>", unsafe_allow_html=True)
    t1, t2 = st.columns(2)

    with t1:
        if race_df is not None and not race_df.empty and "winner_correct" in race_df.columns and "year" in race_df.columns:
            by_year = race_df.groupby("year", as_index=False)["winner_correct"].mean().rename(columns={"winner_correct": "Winner Acc"})
            overall = float(by_year["Winner Acc"].mean()) if not by_year.empty else 0.0
            fig = px.bar(by_year, x="year", y="Winner Acc", color_discrete_sequence=["#e8002d"])
            fig.add_hline(y=overall, line_dash="dot", line_color="#444c56", annotation_text=f"Mean {overall:.0%}", annotation_font_color="#8b949e")
            fig.update_layout(
                title=dict(text="Winner Accuracy by Year", font=dict(size=12, color="#8b949e"), x=0),
                yaxis_tickformat=".0%", height=300,
                xaxis=_dark_axes(), yaxis=_dark_axes(),
                margin=dict(l=8, r=8, t=35, b=8), **_PLOTLY_DARK,
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("Winner accuracy trend unavailable.")

    with t2:
        if race_df is not None and not race_df.empty and "ece_p1_running" in race_df.columns and "year" in race_df.columns:
            fig_ece = go.Figure()
            for yr, grp in race_df.sort_values(["year", "round"]).groupby("year"):
                fig_ece.add_trace(go.Scatter(x=grp["round"], y=grp["ece_p1_running"], mode="lines+markers", name=str(int(yr))))
            fig_ece.update_layout(
                title=dict(text="Running ECE (P1)", font=dict(size=12, color="#8b949e"), x=0),
                height=300,
                xaxis=dict(title="Round", **_dark_axes()),
                yaxis=dict(title="ECE", **_dark_axes()),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=8, r=8, t=35, b=8), **_PLOTLY_DARK,
            )
            st.plotly_chart(fig_ece, width="stretch")
        else:
            st.caption("ECE trend unavailable.")

    st.markdown("<div class='f1-divider'></div>", unsafe_allow_html=True)

    # ── Calibration curve ────────────────────────────────────────────────
    st.markdown("<div class='f1-section-title'>Calibration Curve</div>", unsafe_allow_html=True)
    if calib_df is not None and not calib_df.empty and {"mean_predicted", "mean_actual"}.issubset(calib_df.columns):
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dash", color="#444c56")))
        fig_cal.add_trace(go.Scatter(x=calib_df["mean_predicted"], y=calib_df["mean_actual"], mode="lines+markers", name="Model", line=dict(color="#3b82f6", width=2), marker=dict(size=6)))
        fig_cal.update_layout(
            height=360,
            xaxis=dict(title="Mean Predicted", **_dark_axes()),
            yaxis=dict(title="Observed Frequency", **_dark_axes()),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=8, r=8, t=8, b=8), **_PLOTLY_DARK,
        )
        st.plotly_chart(fig_cal, width="stretch")
    else:
        st.caption("Calibration curve unavailable.")

    # ── Current season actuals ────────────────────────────────────────────
    actual_docs = services.load_actual_results()
    if actual_docs:
        st.markdown("<div class='f1-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='f1-section-title'>Current Season Results</div>", unsafe_allow_html=True)
        rows = []
        for doc in actual_docs:
            acc = doc.get("accuracy", {})
            rows.append({
                "Year": doc.get("year"), "Round": doc.get("round"),
                "Winner Hit": bool(acc.get("winner_hit", False)),
                "Podium Overlap": acc.get("podium_drivers_in_top3_predicted", 0),
                "Brier P1": acc.get("brier_score_p1"),
                "ECE P1": acc.get("ece_p1_round"),
            })
        st.dataframe(
            pd.DataFrame(rows).sort_values(["Year", "Round"], ascending=[False, False]),
            width="stretch", hide_index=True,
        )
