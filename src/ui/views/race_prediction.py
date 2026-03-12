from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.features.build_features import CIRCUIT_SC_RATE, STREET_CIRCUITS, _DEFAULT_SC_RATE
from src.ui import services


def _fmt_pct(value: float) -> str:
    return f"{float(value or 0.0):.1%}"


def _build_prediction_df(pred: dict) -> pd.DataFrame:
    df = pd.DataFrame(pred.get("predictions", []))
    if df.empty:
        return df
    for col in ["p_win", "p_p2", "p_p3", "p_podium", "grid_position", "quali_gap_to_pole_s"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["team"] = df.get("team", pd.Series(["Unknown"] * len(df))).fillna("Unknown")
    df["driver"] = df.get("driver", pd.Series(["UNK"] * len(df))).fillna("UNK")
    df["color"] = df["team"].map(services.get_team_color)
    return df


def render(*, year: int, round_num: int, shell: dict | None = None) -> None:
    context = shell or services.get_shell_snapshot(year, round_num)
    pred = context.get("prediction") if isinstance(context, dict) else None
    prior = context.get("prior_prediction") if isinstance(context, dict) else None

    if pred is None:
        st.markdown(
            "<div class='f1-card' style='text-align:center; padding:2rem;'>"
            "<div style='font-size:1.5rem; margin-bottom:0.5rem;'>No prediction found</div>"
            "<div class='f1-muted'>Run inference from the Control Panel after qualifying.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    prior_pwin: dict[str, float] = {}
    if prior and prior.get("predictions"):
        for p in prior["predictions"]:
            prior_pwin[str(p.get("driver", ""))] = float(p.get("p_win", 0.0) or 0.0)

    df = _build_prediction_df(pred)
    if df.empty:
        st.info("Prediction file has no driver rows.")
        return

    ranked = df.sort_values("p_win", ascending=False).reset_index(drop=True)
    selected_driver = str(st.session_state.get("selected_driver", "") or "")
    if selected_driver not in ranked["driver"].tolist():
        selected_driver = str(ranked.iloc[0]["driver"])
        services.set_selected_driver(selected_driver)

    # ── Race header row ──────────────────────────────────────────────────
    top_pick = ranked.iloc[0]
    top_pwin = float(top_pick.get("p_win", 0.0) or 0.0)
    ck = str(pred.get("circuit_key", "")).lower()
    is_street = ck in STREET_CIRCUITS
    sc_rate = CIRCUIT_SC_RATE.get(ck, _DEFAULT_SC_RATE)
    weather = pred.get("weather", {}) or {}
    precip = float(weather.get("precip_mm_total", 0.0) or 0.0)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Season", int(pred.get("year", year)))
    m2.metric("Round", int(pred.get("round", round_num)))
    m3.metric("Circuit", str(pred.get("circuit_key", "")).replace("_", " ").title())
    m4.metric("Top Pick", str(top_pick.get("driver", "-")))
    m5.metric("P(Win)", _fmt_pct(top_pwin))

    # circuit / weather chips
    circ_label = "Street" if is_street else "Permanent"
    st.markdown(
        services.chip(circ_label, tone="neutral")
        + services.chip(f"SC rate {sc_rate:.0%}", tone="neutral")
        + services.chip(f"Precip {precip:.1f} mm", tone="warn" if precip > 2 else "neutral")
        + services.chip("High variance" if top_pwin < 0.25 else "Clear favourite", tone="warn" if top_pwin < 0.25 else "good"),
        unsafe_allow_html=True,
    )

    st.caption("Win probabilities are normalised to sum to 100% across all drivers.")

    st.markdown("<div class='f1-divider'></div>", unsafe_allow_html=True)

    # ── Podium cards ─────────────────────────────────────────────────────
    st.markdown("<div class='f1-section-title'>Predicted Podium</div>", unsafe_allow_html=True)
    top3 = ranked.head(3)
    medals = ["P1 - Win", "P2", "P3"]
    card_cols = st.columns(3)
    for i, (_, row) in enumerate(top3.iterrows()):
        prior_val = prior_pwin.get(str(row["driver"]))
        delta_str = "n/a" if prior_val is None else f"{(float(row['p_win']) - prior_val):+.1%}"
        color = str(row.get("color", services.DEFAULT_COLOR))
        card_cols[i].markdown(
            f"<div class='f1-podium-card' style='border-top: 3px solid {color};'>"
            f"<div class='f1-muted' style='font-size:0.75rem; text-transform:uppercase; letter-spacing:0.07em;'>{medals[i]}</div>"
            f"<div style='font-size:1.25rem; font-weight:700; margin:0.2rem 0;'>{row['driver']}</div>"
            f"<div class='f1-muted'>{row['team']}</div>"
            f"<div style='margin-top:0.6rem; display:flex; gap:0.8rem;'>"
            f"  <div><div class='f1-muted' style='font-size:0.72rem;'>P(WIN)</div><div style='font-weight:700; font-size:1.1rem;'>{_fmt_pct(row['p_win'])}</div></div>"
            f"  <div><div class='f1-muted' style='font-size:0.72rem;'>P(PODIUM)</div><div style='font-weight:700; font-size:1.1rem;'>{_fmt_pct(row['p_podium'])}</div></div>"
            f"</div>"
            f"<div class='f1-muted' style='margin-top:0.4rem; font-size:0.75rem;'>vs prior: {delta_str}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='f1-divider'></div>", unsafe_allow_html=True)

    # ── Leaderboard ──────────────────────────────────────────────────────
    st.markdown("<div class='f1-section-title'>Full Grid</div>", unsafe_allow_html=True)
    team_options = ["All"] + sorted(df["team"].dropna().unique().tolist())
    picked_team = st.selectbox(
        "Team",
        team_options,
        index=team_options.index(str(st.session_state.get("selected_team_filter", "All")))
        if str(st.session_state.get("selected_team_filter", "All")) in team_options else 0,
        label_visibility="collapsed",
        key="race_team_filter",
    )
    services.set_selected_team(picked_team)
    filtered = ranked if picked_team == "All" else ranked[ranked["team"] == picked_team].reset_index(drop=True)

    if not filtered.empty:
        table_cols = [c for c in ["driver", "team", "grid_position", "quali_gap_to_pole_s", "p_win", "p_p2", "p_p3", "p_podium"] if c in filtered.columns]
        table = filtered[table_cols].copy()
        table.insert(0, "Select", table["driver"].eq(selected_driver))
        rename_map = {
            "driver": "Driver", "team": "Team", "grid_position": "Grid",
            "quali_gap_to_pole_s": "Gap to Pole",
            "p_win": "P(Win)", "p_p2": "P(P2)", "p_p3": "P(P3)", "p_podium": "P(Podium)",
        }
        table.rename(columns=rename_map, inplace=True)

        edited = st.data_editor(
            table,
            hide_index=True,
            use_container_width=True,
            key=f"driver_table_{year}_{round_num}",
            disabled=[c for c in table.columns if c != "Select"],
            column_config={
                "Select": st.column_config.CheckboxColumn(required=False),
                "P(Win)": st.column_config.NumberColumn(format="%.3f"),
                "P(P2)": st.column_config.NumberColumn(format="%.3f"),
                "P(P3)": st.column_config.NumberColumn(format="%.3f"),
                "P(Podium)": st.column_config.NumberColumn(format="%.3f"),
                "Gap to Pole": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        picks = edited[edited["Select"] == True]["Driver"].tolist()  # noqa: E712
        if picks and str(picks[0]) != selected_driver:
            services.set_selected_driver(str(picks[0]))
            st.rerun()

        st.download_button(
            "Export CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"{year}_R{round_num:02d}_grid.csv",
            mime="text/csv",
            key="export_grid_csv",
        )

    st.markdown("<div class='f1-divider'></div>", unsafe_allow_html=True)

    # ── Season trend ─────────────────────────────────────────────────────
    st.markdown("<div class='f1-section-title'>Season P(Win) Trend</div>", unsafe_allow_html=True)
    trend_rows = []
    for doc in services.load_all_predictions():
        if int(doc.get("year", 0) or 0) != int(year):
            continue
        for p in doc.get("predictions", []):
            trend_rows.append({
                "round": int(doc.get("round", 0) or 0),
                "driver": str(p.get("driver", "")),
                "team": str(p.get("team", "")),
                "p_win": float(p.get("p_win", 0.0) or 0.0),
            })

    trend_df = pd.DataFrame(trend_rows)
    if trend_df.empty:
        st.caption("No season trend data yet.")
    else:
        latest_round = int(trend_df["round"].max())
        latest_drivers = trend_df[trend_df["round"] == latest_round].sort_values("p_win", ascending=False)["driver"].tolist()
        defaults = [selected_driver] + [d for d in latest_drivers if d != selected_driver][:4]
        selected_for_trend = st.multiselect(
            "Drivers",
            options=latest_drivers,
            default=[d for d in defaults if d in latest_drivers],
            key=f"trend_drivers_{year}_{round_num}",
            label_visibility="collapsed",
        )

        col_chart, col_sel = st.columns([3, 1])
        with col_chart:
            fig = go.Figure()
            n_drivers = len(latest_drivers) if latest_drivers else 20
            uniform = 1.0 / n_drivers
            fig.add_hline(y=uniform, line_dash="dot", line_color="#444c56", annotation_text=f"Uniform 1/{n_drivers}", annotation_font_color="#8b949e")
            for driver in selected_for_trend:
                sub = trend_df[trend_df["driver"] == driver].sort_values("round")
                if sub.empty:
                    continue
                color = services.get_team_color(str(sub["team"].iloc[0]))
                fig.add_trace(go.Scatter(
                    x=sub["round"], y=sub["p_win"],
                    mode="lines+markers", name=driver,
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                ))
            fig.update_layout(
                height=300,
                yaxis_tickformat=".0%",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#8b949e",
                xaxis=dict(gridcolor="#21262d", title="Round"),
                yaxis=dict(gridcolor="#21262d", title="P(Win)"),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=8, r=8, t=8, b=8),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_sel:
            row = services.driver_prediction_row(pred, selected_driver)
            if row:
                st.metric("P(Win)", _fmt_pct(row.get("p_win", 0.0)))
                st.metric("P(Podium)", _fmt_pct(row.get("p_podium", 0.0)))
                st.caption(f"Grid: {row.get('grid_position', '-')}")
                st.caption(f"Team: {row.get('team', '-')}")

    st.markdown("<div class='f1-divider'></div>", unsafe_allow_html=True)

    # ── Post-race ─────────────────────────────────────────────────────────
    st.markdown("<div class='f1-section-title'>Post-Race Result</div>", unsafe_allow_html=True)
    race_actual = services.load_actual_result(year, round_num)
    if race_actual:
        acc = race_actual.get("accuracy", {})
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Predicted Winner", str(acc.get("winner_predicted", "-")))
        a2.metric("Actual Winner", str(acc.get("winner_actual", "-")))
        a3.metric("Winner Correct", "Yes" if bool(acc.get("winner_hit", False)) else "No")
        a4.metric("Podium Overlap", f"{acc.get('podium_drivers_in_top3_predicted', 0)}/3")
    else:
        st.caption("Actual result not yet recorded for this round.")

    actual_docs = services.load_actual_results()
    if actual_docs:
        rows = []
        for doc in actual_docs:
            acc = doc.get("accuracy", {})
            rows.append({
                "Year": doc.get("year"), "Round": doc.get("round"),
                "Winner Hit": bool(acc.get("winner_hit", False)),
                "Podium Overlap": acc.get("podium_drivers_in_top3_predicted", 0),
                "Brier P1": acc.get("brier_score_p1"),
            })
        results_df = pd.DataFrame(rows).sort_values(["Year", "Round"], ascending=[False, False])
        with st.expander("All recorded results", expanded=False):
            st.dataframe(results_df, use_container_width=True, hide_index=True)
