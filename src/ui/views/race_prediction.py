from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.features.build_features import CIRCUIT_SC_RATE, STREET_CIRCUITS, _DEFAULT_SC_RATE
from src.ui import services

_BM_SESSION_KEY = "bm_odds_df"


def _fmt_pct(value: float) -> str:
    return f"{float(value or 0.0):.1%}"


def _build_prediction_df(pred: dict) -> pd.DataFrame:
    df = pd.DataFrame(pred.get("predictions", []))
    if df.empty:
        return df
    for col in ["p_win", "p_p2", "p_p3", "p_podium", "p_top6", "p_top10",
                "grid_position", "quali_gap_to_pole_s"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["team"] = df.get("team", pd.Series(["Unknown"] * len(df))).fillna("Unknown")
    df["driver"] = df.get("driver", pd.Series(["UNK"] * len(df))).fillna("UNK")
    df["color"] = df["team"].map(services.get_team_color)
    return df


def _min_odds(p: float) -> float:
    """Break-even decimal odds = 1 / probability."""
    return round(1.0 / max(float(p), 0.001), 2)


# Kelly-based flag thresholds (full Kelly fraction against fair odds):
#   FADE  : kelly <= -0.08   bookmaker has strong edge over model
#   NO    : -0.08 < kelly <= 0   no edge, skip
#   SLIM  : 0 < kelly <= 0.03   marginal, transaction cost likely kills it
#   VALUE : 0.03 < kelly <= 0.08  clear edge, worth a quarter-Kelly stake
#   BET   : kelly > 0.08          strong edge
_FLAG_STYLES: dict[str, str] = {
    "BET":   "background-color: #166534; color: #4ade80; font-weight: 800",
    "VALUE": "background-color: #15803d; color: #bbf7d0; font-weight: 700",
    "SLIM":  "background-color: #854d0e; color: #fef08a; font-weight: 600",
    "NO":    "background-color: #21262d; color: #484f58; font-weight: 400",
    "FADE":  "background-color: #7f1d1d; color: #fca5a5; font-weight: 700",
}


def _kelly_flag(kelly_full: float) -> str:
    if kelly_full > 0.08:
        return "BET"
    if kelly_full > 0.03:
        return "VALUE"
    if kelly_full > 0.0:
        return "SLIM"
    if kelly_full >= -0.08:
        return "NO"
    return "FADE"


def _style_flag_table(df: pd.DataFrame, flag_cols: list[str]) -> pd.io.formats.style.Styler:
    """Apply flag colours to the flag columns only."""
    def _row_style(row: pd.Series) -> list[str]:
        styles = [""] * len(row)
        for col in flag_cols:
            if col not in row.index:
                continue
            val = row[col]
            if pd.notna(val) and val in _FLAG_STYLES:
                styles[row.index.get_loc(col)] = _FLAG_STYLES[val]
        return styles

    return df.style.apply(_row_style, axis=1)


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

    circ_label = "Street" if is_street else "Permanent"
    st.markdown(
        services.chip(circ_label, tone="neutral")
        + services.chip(f"SC rate {sc_rate:.0%}", tone="neutral")
        + services.chip(f"Precip {precip:.1f} mm", tone="warn" if precip > 2 else "neutral")
        + services.chip("High variance" if top_pwin < 0.25 else "Clear favourite",
                        tone="warn" if top_pwin < 0.25 else "good"),
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
        p_win = float(row.get("p_win", 0.0) or 0.0)
        p_pod = float(row.get("p_podium", 0.0) or 0.0)
        drv = str(row["driver"])

        with card_cols[i]:
            st.markdown(
                f"<div class='f1-podium-card' style='border-top: 3px solid {color};'>"
                f"<div class='f1-muted' style='font-size:0.75rem; text-transform:uppercase; letter-spacing:0.07em;'>{medals[i]}</div>"
                f"<div style='font-size:1.25rem; font-weight:700; margin:0.2rem 0;'>{drv}</div>"
                f"<div class='f1-muted'>{row['team']}</div>"
                f"<div style='margin-top:0.6rem; display:flex; gap:0.8rem;'>"
                f"  <div><div class='f1-muted' style='font-size:0.72rem;'>P(WIN)</div>"
                f"  <div style='font-weight:700; font-size:1.1rem;'>{_fmt_pct(p_win)}</div></div>"
                f"  <div><div class='f1-muted' style='font-size:0.72rem;'>P(PODIUM)</div>"
                f"  <div style='font-weight:700; font-size:1.1rem;'>{_fmt_pct(p_pod)}</div></div>"
                f"</div>"
                f"<div class='f1-muted' style='margin-top:0.4rem; font-size:0.75rem;'>vs prior: {delta_str}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='f1-divider'></div>", unsafe_allow_html=True)

    # ── Full Grid ─────────────────────────────────────────────────────────
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
        # Build min_odds columns from model probabilities
        filt = filtered.copy()
        filt["min_1-1"] = filt["p_win"].apply(
            lambda p: _min_odds(p) if pd.notna(p) else np.nan
        )
        filt["min_1-3"] = filt["p_podium"].apply(
            lambda p: _min_odds(p) if pd.notna(p) else np.nan
        )
        filt["min_1-6"] = filt["p_top6"].apply(
            lambda p: _min_odds(p) if pd.notna(p) else np.nan
        ) if "p_top6" in filt.columns else np.nan
        filt["min_1-10"] = filt["p_top10"].apply(
            lambda p: _min_odds(p) if pd.notna(p) else np.nan
        ) if "p_top10" in filt.columns else np.nan

        table_cols = [c for c in [
            "driver", "team", "grid_position", "quali_gap_to_pole_s",
            "p_win", "p_p2", "p_p3", "p_podium",
            "min_1-1", "min_1-3", "min_1-6", "min_1-10",
        ] if c in filt.columns]
        table = filt[table_cols].copy()
        table.insert(0, "Select", table["driver"].eq(selected_driver))

        rename_map = {
            "driver": "Driver", "team": "Team", "grid_position": "Grid",
            "quali_gap_to_pole_s": "Gap to Pole",
            "p_win": "P(Win)", "p_p2": "P(P2)", "p_p3": "P(P3)", "p_podium": "P(Podium)",
            "min_1-1": "Min Win", "min_1-3": "Min 1-3", "min_1-6": "Min 1-6", "min_1-10": "Min 1-10",
        }
        table.rename(columns=rename_map, inplace=True)

        edited = st.data_editor(
            table,
            hide_index=True,
            width="stretch",
            key=f"driver_table_{year}_{round_num}",
            disabled=[c for c in table.columns if c != "Select"],
            column_config={
                "Select": st.column_config.CheckboxColumn(required=False),
                "P(Win)": st.column_config.NumberColumn(format="%.3f"),
                "P(P2)": st.column_config.NumberColumn(format="%.3f"),
                "P(P3)": st.column_config.NumberColumn(format="%.3f"),
                "P(Podium)": st.column_config.NumberColumn(format="%.3f"),
                "Gap to Pole": st.column_config.NumberColumn(format="%.3f"),
                "Min Win": st.column_config.NumberColumn(format="%.2f", help="Min odds needed for win bet (EV > 0)"),
                "Min 1-3": st.column_config.NumberColumn(format="%.2f", help="Min odds needed for top-3 (EV > 0)"),
                "Min 1-6": st.column_config.NumberColumn(format="%.2f", help="Min odds needed for top-6 (EV > 0)"),
                "Min 1-10": st.column_config.NumberColumn(format="%.2f", help="Min odds needed for top-10 (EV > 0)"),
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

    # ── Bookmaker EV ─────────────────────────────────────────────────────
    st.markdown("<div class='f1-section-title'>Bookmaker EV Analysis</div>", unsafe_allow_html=True)
    st.caption(
        "Upload a CSV: `driver, bm_win, bm_1-3, bm_1-6, bm_1-10` (driver abbreviations). "
        "Vig is removed per market using proportional normalisation across the full field. "
        "Kelly fractions use blended model/market probabilities to account for model uncertainty."
    )

    uploaded = st.file_uploader(
        "Bookmaker odds CSV",
        type="csv",
        key=f"bm_upload_{year}_{round_num}",
        label_visibility="collapsed",
    )
    if uploaded is not None:
        try:
            bm_raw = pd.read_csv(io.StringIO(uploaded.read().decode("utf-8")))
            bm_raw.columns = [c.strip().lower().replace(" ", "_") for c in bm_raw.columns]
            for col in ["bm_win", "bm_1-3", "bm_1-6", "bm_1-10"]:
                if col in bm_raw.columns:
                    bm_raw[col] = pd.to_numeric(bm_raw[col], errors="coerce")
            st.session_state[f"{_BM_SESSION_KEY}_{year}_{round_num}"] = bm_raw
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")

    bm_df: pd.DataFrame | None = st.session_state.get(f"{_BM_SESSION_KEY}_{year}_{round_num}")

    if bm_df is not None and not filtered.empty:
        # ── Controls ──────────────────────────────────────────────────────
        ctrl1, ctrl2, ctrl3 = st.columns(3)
        with ctrl1:
            _ALPHA_OPTIONS = {
                "Sceptical — 30% model": 0.30,
                "Balanced — 50% model":  0.50,
                "Confident — 70% model": 0.70,
            }
            alpha_label = st.selectbox(
                "Model confidence",
                list(_ALPHA_OPTIONS.keys()),
                index=1,
                key=f"ev_alpha_{year}_{round_num}",
                help=(
                    "How much weight the model gets vs the market's fair probability. "
                    "Balanced (50/50) recommended until the model has a verified track record. "
                    "Current holdout winner accuracy: "
                    f"{float((services.load_active_model() or {}).get('winner_accuracy_holdout', 0) or 0):.1%}."
                ),
            )
            alpha = _ALPHA_OPTIONS[alpha_label]

        with ctrl2:
            _KELLY_OPTIONS = {
                "Quarter Kelly (0.25×)": 0.25,
                "Half Kelly (0.50×)":   0.50,
                "Full Kelly (1.00×)":   1.00,
            }
            kelly_label = st.selectbox(
                "Kelly fraction",
                list(_KELLY_OPTIONS.keys()),
                index=0,
                key=f"ev_kelly_{year}_{round_num}",
                help=(
                    "Fraction of Kelly criterion to stake. Quarter Kelly halves variance "
                    "while retaining ~75% of long-run growth — standard for unvalidated models."
                ),
            )
            kelly_mult = _KELLY_OPTIONS[kelly_label]

        with ctrl3:
            bankroll = st.number_input(
                "Bankroll",
                min_value=10.0, max_value=1_000_000.0, value=500.0, step=50.0,
                key=f"ev_bankroll_{year}_{round_num}",
                help="Total betting bankroll (any currency). Used to calculate recommended stake amounts.",
            )

        # ── Market config: bm_col, label, model_p_col, k (places paid) ───
        _MARKET_CFG = [
            ("bm_win",  "Win",    "p_win",    1),
            ("bm_1-3",  "Top 3",  "p_podium", 3),
            ("bm_1-6",  "Top 6",  "p_top6",   6),
            ("bm_1-10", "Top 10", "p_top10",  10),
        ]

        # ── Per-market overround ──────────────────────────────────────────
        # For a k-place market: overround = sum(1/odds_i) / k
        # A fair market would have sum(implied_p) = k (exactly k drivers finish top-k).
        # Overround > 1 means bookmaker margin; fair odds = bm_odds * overround.
        market_overrounds: dict[str, float] = {}
        for bm_col, mkt_label, _pcol, k in _MARKET_CFG:
            if bm_col not in bm_df.columns:
                continue
            raw = pd.to_numeric(bm_df[bm_col], errors="coerce").dropna()
            raw = raw[raw > 1.0]
            if raw.empty:
                continue
            market_overrounds[bm_col] = float((1.0 / raw).sum() / k)

        # Market efficiency chips
        if market_overrounds:
            chips_html = ""
            for bm_col, mkt_label, _, _ in _MARKET_CFG:
                if bm_col not in market_overrounds:
                    continue
                o = market_overrounds[bm_col]
                margin_pct = (1.0 - 1.0 / o) * 100.0
                tone = "bad" if margin_pct > 14 else "warn" if margin_pct > 8 else "neutral"
                chips_html += services.chip(
                    f"{mkt_label}: {o:.1%} overround ({margin_pct:.1f}% margin)", tone=tone
                )
            st.markdown(chips_html, unsafe_allow_html=True)
            st.caption(
                "Your model edge must exceed the margin to generate a positive-EV bet. "
                "EV is calculated against fair odds (after vig removal), not raw bookmaker odds."
            )

        # ── Build analysis rows ───────────────────────────────────────────
        p_cols_avail = [c for c in ["p_win", "p_podium", "p_top6", "p_top10"] if c in filtered.columns]
        base_df = filtered[["driver", "team", "grid_position"] + p_cols_avail].copy()
        bm_merge_cols = ["driver"] + [m[0] for m in _MARKET_CFG if m[0] in bm_df.columns]
        base_df = base_df.merge(bm_df[bm_merge_cols], on="driver", how="left")

        analysis_rows: list[dict] = []
        for bm_col, mkt_label, p_col, k in _MARKET_CFG:
            if bm_col not in base_df.columns or p_col not in base_df.columns:
                continue
            if bm_col not in market_overrounds:
                continue
            o = market_overrounds[bm_col]

            for _, row in base_df.iterrows():
                bm_odds = row.get(bm_col)
                p_model = row.get(p_col)
                if pd.isna(bm_odds) or pd.isna(p_model) or float(bm_odds) <= 1.0:
                    continue
                bm_odds = float(bm_odds)
                p_model = float(p_model)

                # Fair probability: proportional vig removal.
                # p_fair = (1/bm_odds) / o   →  sum(p_fair) = k (correct for k-place markets)
                p_fair = float(np.clip((1.0 / bm_odds) / o, 1e-6, 1.0 - 1e-6))
                # Fair decimal odds (what bookie would offer with zero margin)
                d_fair = 1.0 / p_fair  # = bm_odds * o

                # Blended probability: geometric mean of model and market fair probability.
                # alpha=0.5 means equal trust; prevents overbetting on stale/overfit model estimates.
                p_blend = float(np.clip(
                    (p_model ** alpha) * (p_fair ** (1.0 - alpha)),
                    1e-6, 1.0 - 1e-6,
                ))

                # Kelly fraction against fair odds using blended probability.
                # f* = (b*p - (1-p)) / b  where b = d_fair - 1
                b_fair = d_fair - 1.0
                kelly_full = float((b_fair * p_blend - (1.0 - p_blend)) / b_fair) if b_fair > 0 else 0.0

                # Stake: fractional Kelly, hard-capped at 5% of bankroll
                kelly_frac = kelly_full * kelly_mult if kelly_full > 0 else 0.0
                stake = min(kelly_frac * bankroll, 0.05 * bankroll) if kelly_frac > 0 else 0.0

                # Edge: how much blended model probability exceeds fair market probability
                edge_pct = (p_blend - p_fair) * 100.0

                analysis_rows.append({
                    "driver":     str(row["driver"]),
                    "team":       str(row["team"]),
                    "market":     mkt_label,
                    "bm_odds":    round(bm_odds, 2),
                    "fair_odds":  round(d_fair, 2),
                    "model_p":    round(p_model, 4),
                    "fair_p":     round(p_fair, 4),
                    "blend_p":    round(p_blend, 4),
                    "edge_pct":   round(edge_pct, 1),
                    "kelly_full": round(kelly_full, 4),
                    "stake":      round(stake, 2),
                    "flag":       _kelly_flag(kelly_full),
                })

        analysis_df = pd.DataFrame(analysis_rows)

        if not analysis_df.empty:
            # ── Recommended bets (VALUE + BET only) ──────────────────────
            st.markdown(
                "<div class='f1-section-title' style='margin-top:0.75rem;'>Recommended Bets</div>",
                unsafe_allow_html=True,
            )
            recs = (
                analysis_df[analysis_df["kelly_full"] >= 0.03]
                .sort_values("kelly_full", ascending=False)
                .copy()
            )

            if recs.empty:
                st.caption(
                    "No bets meet the minimum edge threshold (Kelly ≥ 3%) for this race. "
                    "See the full EV table below for marginal or negative-value bets."
                )
            else:
                rec_disp = recs[["driver", "market", "bm_odds", "fair_odds",
                                 "edge_pct", "kelly_full", "stake", "flag"]].copy()
                rec_disp.columns = ["Driver", "Market", "BM Odds", "Fair Odds",
                                    "Edge%", "Kelly", "Stake", "Flag"]
                rec_disp["Edge%"]  = rec_disp["Edge%"].apply(lambda v: f"+{v:.1f}%" if v >= 0 else f"{v:.1f}%")
                rec_disp["Kelly"]  = rec_disp["Kelly"].apply(lambda v: f"{v:.1%}")
                rec_disp["Stake"]  = rec_disp["Stake"].apply(lambda v: f"{v:.2f}" if v > 0 else "—")
                st.dataframe(
                    _style_flag_table(rec_disp, ["Flag"]).format(
                        {"BM Odds": "{:.2f}", "Fair Odds": "{:.2f}"}, na_rep="—"
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption(
                    f"{kelly_label} on blended probability "
                    f"({int(alpha * 100)}% model / {int((1 - alpha) * 100)}% market), "
                    f"bankroll {bankroll:,.0f}, hard cap 5% per bet.  "
                    "BET = Kelly > 8%  ·  VALUE = Kelly 3–8%"
                )

                # Correlation warnings: same driver appearing in multiple recommended markets
                driver_counts = recs.groupby("driver").size()
                for drv in driver_counts[driver_counts > 1].index:
                    drv_recs = recs[recs["driver"] == drv]
                    total_stake = drv_recs["stake"].sum()
                    cap = 0.02 * bankroll
                    mkts = " + ".join(drv_recs["market"].tolist())
                    tone = "warn" if total_stake <= cap * 1.5 else "bad"
                    st.markdown(
                        services.chip(
                            f"Correlated — {drv} ({mkts}): "
                            f"combined stake {total_stake:.2f} vs 2% cap {cap:.2f}",
                            tone=tone,
                        ),
                        unsafe_allow_html=True,
                    )

            # ── Full EV table ─────────────────────────────────────────────
            with st.expander("Full EV table — all drivers, all markets", expanded=False):
                full_disp = (
                    analysis_df[["driver", "market", "bm_odds", "fair_odds",
                                 "model_p", "fair_p", "edge_pct", "kelly_full", "flag"]]
                    .sort_values(["market", "kelly_full"], ascending=[True, False])
                    .copy()
                )
                full_disp.columns = ["Driver", "Market", "BM Odds", "Fair Odds",
                                     "Model P", "Fair P", "Edge%", "Kelly", "Flag"]
                full_disp["Edge%"] = full_disp["Edge%"].apply(lambda v: f"+{v:.1f}%" if v >= 0 else f"{v:.1f}%")
                full_disp["Kelly"] = full_disp["Kelly"].apply(lambda v: f"{v:.1%}")
                st.dataframe(
                    _style_flag_table(full_disp, ["Flag"]).format(
                        {"BM Odds": "{:.2f}", "Fair Odds": "{:.2f}",
                         "Model P": "{:.3f}", "Fair P": "{:.3f}"},
                        na_rep="—",
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
        else:
            st.caption("No matching drivers found between model predictions and uploaded odds.")

    elif bm_df is None:
        st.caption("Upload a bookmaker odds CSV above to see the EV analysis.")

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
            fig.add_hline(y=uniform, line_dash="dot", line_color="#444c56",
                          annotation_text=f"Uniform 1/{n_drivers}", annotation_font_color="#8b949e")
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
            st.plotly_chart(fig, width="stretch")

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
