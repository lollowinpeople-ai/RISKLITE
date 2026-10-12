"""RiskLite: Streamlit front end. All calculations live in the `risklite` package."""
from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from risklite import __version__
from risklite import contrib, overlap, stress
from risklite.data import (SYNTHETIC_ASSETS, SYNTHETIC_PORTFOLIO, load_prices_csv,
                           load_prices_yfinance, normalise_weights, synthetic_prices)
from risklite.metrics import drawdown_series
from risklite.report import markdown_report, run_analysis

ROOT = Path(__file__).parent
DATA = ROOT / "data"

st.set_page_config(page_title="RiskLite: portfolio risk analytics", page_icon="📉", layout="wide")

# ----------------------------------------------------------------------------- chart style
INK, MUTED, GRID = "#111827", "#6b7280", "#e5e7eb"
BLUE, ORANGE, RED, GREEN, GREY = "#1d4ed8", "#ea580c", "#b91c1c", "#15803d", "#9ca3af"
SERIES = [BLUE, ORANGE, "#0f766e", "#7c3aed", "#be185d", "#a16207", "#475569", "#0369a1"]
pio.templates["risklite"] = go.layout.Template(layout=go.Layout(
    font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=13, color=INK),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=SERIES,
    xaxis=dict(showgrid=False, linecolor=MUTED, ticks="outside", tickcolor=MUTED),
    yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=MUTED),
    margin=dict(l=10, r=10, t=70, b=10), hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    title=dict(x=0, xanchor="left", y=0.98, yanchor="top", font=dict(size=15)),
))
pio.templates.default = "risklite"


def show(fig: go.Figure, height: int = 380) -> None:
    fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def pct(x: float, d: int = 2) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.{d}%}"


def eur(x: float) -> str:
    return "n/a" if x is None or pd.isna(x) else f"€{x:,.0f}"


# ----------------------------------------------------------------------------- cached loaders
@st.cache_data(show_spinner="Downloading prices from Yahoo Finance…", ttl=6 * 3600)
def cached_yahoo(tickers: tuple[str, ...], start: str):
    return load_prices_yfinance(list(tickers), start=start)


@st.cache_data(show_spinner=False)
def cached_synthetic():
    return synthetic_prices()


@st.cache_data(show_spinner="Running the analysis…")
def cached_analysis(prices: pd.DataFrame, weights: tuple, alpha: float, horizon: int, rf: float,
                    bench: pd.Series | None, classes: tuple, window: int, cov_method: str):
    return run_analysis(prices, dict(weights), alpha=alpha, horizon=horizon, rf_annual=rf,
                        benchmark=bench, asset_classes=dict(classes), window=window,
                        cov_method=cov_method)


def default_portfolio(source: str) -> pd.DataFrame:
    if source == "Demo data":
        return pd.DataFrame({"ticker": list(SYNTHETIC_PORTFOLIO),
                             "weight": [round(w * 100, 2) for w in SYNTHETIC_PORTFOLIO.values()],
                             "asset_class": [SYNTHETIC_ASSETS[t][2] for t in SYNTHETIC_PORTFOLIO]})
    df = pd.read_csv(DATA / "sample_portfolio.csv")
    return df[["ticker", "weight", "asset_class"]]


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### RiskLite")
    st.caption("Portfolio risk analytics. Analytical tool, not investment advice.")
    source = st.radio("Price data", ["Demo data", "Yahoo Finance", "Upload CSV"],
                      help="Demo data is simulated and works offline. Yahoo Finance needs internet "
                           "access. CSV: first column dates, one column of prices per ticker.")
    upload_prices = None
    if source == "Upload CSV":
        upload_prices = st.file_uploader("Prices CSV", type="csv")

    st.markdown("**Settings**")
    start = st.date_input("History from", dt.date(2007, 1, 1), min_value=dt.date(1990, 1, 1))
    c1, c2 = st.columns(2)
    alpha = c1.selectbox("Confidence", [0.99, 0.975, 0.95], format_func=lambda a: f"{a:.1%}".replace(".0%", "%"))
    horizon = c2.selectbox("Horizon (days)", [1, 5, 10, 21])
    value = st.number_input("Portfolio value (€)", min_value=0.0, value=250_000.0, step=10_000.0)
    bench_ticker = st.text_input("Benchmark ticker (optional)",
                                 "SYN_WORLD_EQ" if source == "Demo data" else "SWDA.MI")
    with st.expander("Advanced"):
        rf = st.number_input("Risk-free rate (annual %)", value=2.0, step=0.25) / 100
        window = st.select_slider("Backtest window (days)", [125, 250, 500, 750], value=250)
        cov_method = st.radio("Covariance for risk contributions", ["sample", "ewma"], horizontal=True,
                              help="EWMA (λ = 0.94) weights recent days more, so it reacts to new regimes.")

# ----------------------------------------------------------------------------- portfolio editor
st.title("Portfolio risk report")
with st.expander("Portfolio: edit holdings and weights", expanded=False):
    up_port = st.file_uploader("Load a portfolio CSV (columns: ticker, weight, asset_class)", type="csv",
                               key="port_upload")
    if up_port is not None:
        base = pd.read_csv(up_port)
        base.columns = [c.lower().strip() for c in base.columns]
        if "asset_class" not in base:
            base["asset_class"] = "Equity"
        base = base[["ticker", "weight", "asset_class"]]
    else:
        base = default_portfolio(source)
    port_df = st.data_editor(
        base, num_rows="dynamic", width="stretch", key=f"editor_{source}_{up_port is not None}",
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", required=True),
            "weight": st.column_config.NumberColumn("Weight %", min_value=0.0, step=0.5, format="%.1f"),
            "asset_class": st.column_config.SelectboxColumn("Asset class", options=stress.ASSET_CLASSES,
                                                            default="Equity"),
        })
    st.caption("Weights are rescaled to 100%. Yahoo Finance tickers for Milan listings end in .MI "
               "(e.g. ENI.MI). Asset class drives the hypothetical stress tests.")

# ----------------------------------------------------------------------------- load data
try:
    port_df = port_df.dropna(subset=["ticker"])
    port_df["ticker"] = port_df["ticker"].astype(str).str.strip()
    weights = normalise_weights(dict(zip(port_df["ticker"], port_df["weight"])))
    classes = {t: (c if isinstance(c, str) else "Other") for t, c in zip(port_df["ticker"], port_df["asset_class"])}
except ValueError as e:
    st.error(str(e)); st.stop()

try:
    if source == "Demo data":
        data = cached_synthetic()
    elif source == "Yahoo Finance":
        tick = tuple(sorted(set(weights) | ({bench_ticker} if bench_ticker else set())))
        data = cached_yahoo(tick, str(start))
    else:
        if upload_prices is None:
            st.info("Upload a prices CSV in the sidebar, or switch to Demo data."); st.stop()
        data = load_prices_csv(upload_prices)
except Exception as e:  # network errors, bad files
    st.error(f"Could not load prices: {e}"); st.stop()

prices = data.prices.loc[pd.Timestamp(start):] if source != "Yahoo Finance" else data.prices
missing = [t for t in weights if t not in prices.columns]
if missing:
    st.warning(f"No prices for {', '.join(missing)}; they are excluded and the other weights rescaled.")
    weights = {t: w for t, w in weights.items() if t in prices.columns}
    if not weights:
        st.stop()
    weights = normalise_weights(weights)
bench = prices[bench_ticker] if bench_ticker and bench_ticker in prices.columns else None

try:
    a = cached_analysis(prices, tuple(weights.items()), alpha, horizon, rf, bench,
                        tuple(classes.items()), window, cov_method)
except ValueError as e:
    st.error(str(e)); st.stop()

# ----------------------------------------------------------------------------- header
st.caption(f"{data.source} · {a.port.index[0]:%d %b %Y} to {a.port.index[-1]:%d %b %Y} · "
           f"{len(a.port):,} trading days · {len(weights)} positions")
if "Synthetic" in data.source:
    st.info("Showing **simulated demo prices**, not market data. Choose Yahoo Finance in the sidebar "
            "for real prices.", icon="ℹ️")
for n in data.notes:
    if "simulated" not in n:
        st.caption(n)

hv = a.var_table.loc["Historical", "VaR"]; he = a.var_table.loc["Historical", "ES"]
s = a.summary
m = st.columns(5)
m[0].metric("Annual return", pct(s["annual_return"]))
m[1].metric("Volatility", pct(s["annual_volatility"]))
m[2].metric("Max drawdown", pct(s["max_drawdown"]))
m[3].metric(f"VaR {alpha:.0%}, {horizon}d", pct(hv), eur(hv * value), delta_color="off", delta_arrow="off")
m[4].metric(f"ES {alpha:.0%}, {horizon}d", pct(he), eur(he * value), delta_color="off", delta_arrow="off")

tabs = st.tabs(["Overview", "VaR & ES", "Backtest", "Stress tests", "Risk contribution",
                "Fund overlap", "Report"])

# ----------------------------------------------------------------------------- overview
with tabs[0]:
    left, right = st.columns([3, 2])
    with left:
        wealth = (1 + a.port).cumprod() * value
        fig = go.Figure(go.Scatter(x=wealth.index, y=wealth, name="Portfolio", line=dict(color=BLUE, width=1.6)))
        if bench is not None:
            b = bench.loc[wealth.index[0]:]
            fig.add_scatter(x=b.index, y=b / b.iloc[0] * value, name=f"Benchmark ({bench_ticker})",
                            line=dict(color=GREY, width=1.2))
        fig.update_layout(title=f"Value of €{value:,.0f} invested", yaxis_tickprefix="€")
        show(fig, 340)
        dd = drawdown_series(a.port)
        fig = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", line=dict(color=RED, width=1),
                                   fillcolor="rgba(185,28,28,0.18)", name="Drawdown"))
        fig.update_layout(title="Drawdown from previous peak", yaxis_tickformat=".0%", showlegend=False)
        show(fig, 220)
    with right:
        st.markdown("#### What the numbers say")
        for f in a.findings:
            st.markdown(f"- {f}")
        st.markdown("#### Statistics")
        stats_tbl = pd.DataFrame({
            "Value": [pct(s["annual_return"]), pct(s["annual_volatility"]), f"{s['sharpe']:.2f}",
                      f"{s['sortino']:.2f}", f"{s['calmar']:.2f}", f"{s['skewness']:.2f}",
                      f"{s['excess_kurtosis']:.2f}", pct(s["worst_day"]), pct(s["best_day"]),
                      f"{a.concentration['effective_positions']:.1f}"]},
            index=["Annual return", "Annual volatility", "Sharpe", "Sortino", "Calmar", "Skewness",
                   "Excess kurtosis", "Worst day", "Best day", "Effective no. of positions"])
        if a.benchmark:
            b = a.benchmark
            stats_tbl.loc["Beta vs benchmark"] = f"{b['beta']:.2f}"
            stats_tbl.loc["Correlation vs benchmark"] = f"{b['correlation']:.2f}"
            stats_tbl.loc["Tracking error"] = pct(b["tracking_error"])
        st.dataframe(stats_tbl, width="stretch", height=35 * (len(stats_tbl) + 1) + 3)
        if a.drawdown.peak is not None:
            rec = (f"recovered on {a.drawdown.recovery:%d %b %Y}" if a.drawdown.recovery is not None
                   else "not yet recovered")
            st.caption(f"Max drawdown: peak {a.drawdown.peak:%d %b %Y}, trough {a.drawdown.trough:%d %b %Y}, {rec}.")

# ----------------------------------------------------------------------------- VaR
with tabs[1]:
    st.markdown(f"**{horizon}-day loss at {alpha:.0%} confidence**, four ways. VaR: the loss exceeded "
                f"on only {1 - alpha:.1%} of periods. ES: the average loss when VaR is exceeded.")
    vt = a.var_table.copy()
    tbl = pd.DataFrame({"VaR %": vt["VaR"].map(pct), "VaR €": (vt["VaR"] * value).map(eur),
                        "ES %": vt["ES"].map(pct), "ES €": (vt["ES"] * value).map(eur)})
    tbl = tbl.replace("n/a", "–")
    tbl.index = tbl.index.str.replace(" (skew/kurtosis)", "", regex=False)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(tbl, width="stretch")
        st.caption("Historical: past returns as they were. Parametric: assumes a normal distribution. "
                   "Cornish-Fisher: normal corrected for skew and fat tails. Monte Carlo: 20,000 "
                   "resampled scenarios that keep each day's cross-asset moves together.")
    with c2:
        r = a.port
        fig = go.Figure(go.Histogram(x=r, nbinsx=120, marker_color=GREY, name="Daily returns",
                                     hovertemplate="%{x:.2%}<extra></extra>"))
        one_day = {k: vt.loc[k, "VaR"] / np.sqrt(horizon) for k in ["Historical", "Parametric (normal)"]}
        for (k, x), c in zip(one_day.items(), [RED, BLUE]):
            fig.add_vline(x=-x, line=dict(color=c, dash="dash", width=1.5),
                          annotation_text=f"{k.split(' ')[0]} 1d VaR", annotation_font_color=c,
                          annotation_position="top left" if c == RED else "top right")
        fig.update_layout(title="Distribution of daily returns", xaxis_tickformat=".0%",
                          showlegend=False, bargap=0.02, hovermode="closest")
        show(fig, 360)
    ex = a.summary["excess_kurtosis"]
    st.caption(f"Excess kurtosis {ex:.1f}: {'fatter tails than a normal distribution, so the normal model tends to understate extreme losses.' if ex > 1 else 'tails close to normal.'}"
               + (" Horizons above 1 day use square-root-of-time scaling." if horizon > 1 else ""))

# ----------------------------------------------------------------------------- backtest
with tabs[2]:
    if not a.backtests:
        st.warning(f"Need more than {window} days of history to backtest.")
    else:
        names = {"historical": "Historical", "parametric": "Parametric (normal)", "ewma": "EWMA (RiskMetrics)"}
        summary_rows = [{"Model": names[k], "Days": r.n_obs, "Exceptions": r.n_exc,
                         "Expected": round(r.expected_exc, 1), "Rate": pct(r.exception_rate),
                         "Kupiec p": round(r.kupiec_p, 3), "Christoffersen p": round(r.christoffersen_p, 3),
                         "Zone": r.zone, "Verdict": r.verdict} for k, r in a.backtests.items()]
        st.markdown(f"Every day, a 1-day {alpha:.0%} VaR is forecast from the previous {window} days and "
                    "compared with what actually happened. A good model is breached about "
                    f"{1 - alpha:.1%} of the time, and not in clusters.")
        st.dataframe(pd.DataFrame(summary_rows).set_index("Model"), width="stretch")
        choice = st.radio("Show model", list(a.backtests), format_func=names.get, horizontal=True)
        r = a.backtests[choice]
        ret = a.port.loc[r.forecast.index]
        exc = r.exceptions[r.exceptions].index
        fig = go.Figure()
        fig.add_bar(x=ret.index, y=ret, marker_color=GREY, name="Daily return", marker_line_width=0)
        fig.add_scatter(x=r.forecast.index, y=-r.forecast, line=dict(color=BLUE, width=1.3), name="−VaR forecast")
        fig.add_scatter(x=exc, y=a.port.loc[exc], mode="markers", marker=dict(color=RED, size=6),
                        name=f"Exceptions ({r.n_exc})")
        fig.update_layout(title=f"{names[choice]} VaR vs realised returns: Basel zone {r.zone}",
                          yaxis_tickformat=".0%", bargap=0)
        show(fig, 400)
        st.caption("Kupiec tests whether the number of exceptions matches the confidence level. "
                   "Christoffersen tests whether exceptions bunch together, a sign the model reacts "
                   "too slowly to volatility. p < 0.05 rejects the model. Basel zones from the "
                   "cumulative binomial probability (green < 95%, red ≥ 99.99%).")

# ----------------------------------------------------------------------------- stress
with tabs[3]:
    st.markdown("#### Historical replay")
    st.caption("Today's weights run through the actual daily returns of past crises.")
    hs = a.historical_stress
    rows = [{"Scenario": x.name, "Window": f"{x.start} → {x.end}", "What happened": x.description,
             "Return": pct(x.total_return) if x.covered else "no data",
             "Loss €": eur(-x.total_return * value) if x.covered else "",
             "Max drawdown": pct(x.max_drawdown) if x.covered else ""} for x in hs]
    st.dataframe(pd.DataFrame(rows).set_index("Scenario"), width="stretch")
    covered = [x for x in hs if x.covered]
    if covered:
        fig = go.Figure()
        for x, c in zip(covered, SERIES):
            path = x.path.reset_index(drop=True)
            fig.add_scatter(x=path.index, y=path - 1, name=x.name, line=dict(color=c, width=1.6))
        fig.update_layout(title="Cumulative return through each crisis", xaxis_title="Trading days from start",
                          yaxis_tickformat=".0%", hovermode="x")
        show(fig, 340)
    if any(not x.covered for x in hs):
        st.caption("Scenarios marked *no data* fall before the first date with prices for every holding "
                   "(many ETFs were launched after 2008). Use the hypothetical shocks below for those.")

    st.markdown("#### Hypothetical shocks by asset class")
    st.caption("Illustrative assumptions for discussion, not forecasts. Edit any cell.")
    shocks = pd.DataFrame(stress.HYPOTHETICAL_SCENARIOS).T[stress.ASSET_CLASSES] * 100
    shocks = st.data_editor(shocks, width="stretch", key="shocks",
                            column_config={c: st.column_config.NumberColumn(c, format="%.0f%%")
                                           for c in stress.ASSET_CLASSES})
    hyp = stress.hypothetical_stress(weights, classes, (shocks / 100).T.to_dict())
    imp = hyp["Portfolio impact"].sort_values()
    fig = go.Figure(go.Bar(x=imp, y=imp.index, orientation="h",
                           marker_color=[RED if x < 0 else GREEN for x in imp],
                           text=[f"{x:.1%}  ({eur(x * value)})" for x in imp], textposition="outside",
                           cliponaxis=False))
    fig.update_layout(title="Instant portfolio impact", xaxis_tickformat=".0%", hovermode="closest",
                      xaxis_range=[min(imp.min() * 1.5, -0.05), max(imp.max() * 1.5, 0.05)])
    show(fig, 260)

    st.markdown("#### Worst periods in the sample")
    wp = a.worst_periods.copy()
    wp["Length (days)"] = wp["Length (days)"].map({1: "1 day", 5: "1 week (5d)", 21: "1 month (21d)"})
    wp["Loss €"] = (-wp["Return"] * value).map(eur)
    wp["Return"] = wp["Return"].map(pct)
    st.dataframe(wp, width="stretch", hide_index=True)

# ----------------------------------------------------------------------------- contributions
with tabs[4]:
    rc = a.risk_contrib.sort_values("Share of risk")
    st.markdown("Weight is where the money is; **share of risk** is where the volatility comes from "
                "(Euler decomposition, adds up to 100%).")
    fig = go.Figure()
    fig.add_bar(y=rc.index, x=rc["Weight"], orientation="h", name="Weight", marker_color=GRID,
                marker_line=dict(color=MUTED, width=1))
    fig.add_bar(y=rc.index, x=rc["Share of risk"], orientation="h", name="Share of risk", marker_color=BLUE)
    fig.update_layout(barmode="group", xaxis_tickformat=".0%", title="Capital vs risk", hovermode="y unified")
    show(fig, 120 + 42 * len(rc))
    es_c = a.es_contrib
    tbl = pd.DataFrame({
        "Weight": rc["Weight"].map(lambda x: pct(x, 1)),
        "Share of volatility": rc["Share of risk"].map(lambda x: pct(x, 1)),
        "Risk / weight": rc["Risk / weight"].map(lambda x: f"{x:.2f}×"),
        f"Component VaR {alpha:.0%} (€)": (rc["Component VaR"] * value).map(eur),
        "Share of ES (tail days)": (es_c / es_c.sum()).reindex(rc.index).map(lambda x: pct(x, 1)),
    }).iloc[::-1]
    st.dataframe(tbl, width="stretch")
    st.caption(f"Diversification ratio {a.diversification_ratio:.2f}: the weighted average of the "
               "holdings' own volatilities divided by portfolio volatility. Higher means more benefit "
               "from diversification. Share of ES shows who lost money on the portfolio's worst days.")
    corr = a.correlation
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                               colorscale=[[0, "#2563eb"], [0.5, "#f9fafb"], [1, "#dc2626"]],
                               text=np.round(corr.values, 2), texttemplate="%{text}",
                               hovertemplate="%{x} / %{y}: %{z:.2f}<extra></extra>"))
    fig.update_layout(title="Correlation of daily returns", hovermode="closest", yaxis_autorange="reversed")
    show(fig, 120 + 45 * len(corr))

# ----------------------------------------------------------------------------- overlap
with tabs[5]:
    st.markdown("Opens up ETFs and funds to show what you really own. Upload holdings as a CSV with "
                "columns `fund, holding, weight` (download them from each fund issuer's website). "
                "Use the portfolio tickers in `fund` and, for direct stocks, in `holding`.")
    up_h = st.file_uploader("Fund holdings CSV", type="csv", key="holdings")
    if up_h is not None:
        holdings = overlap.read_holdings_csv(up_h)
        st.caption("Using your uploaded holdings.")
    else:
        holdings = overlap.read_holdings_csv(DATA / "sample_holdings.csv")
        st.caption("Using the bundled sample for SWDA.MI and CSSPX.MI: **illustrative weights**, top "
                   "holdings only. Replace with the issuer's current file for real analysis.")
    funds_in_port = [f for f in sorted(set(holdings["fund"])) if f in weights]
    if not funds_in_port:
        st.info("None of the portfolio's positions match a fund in the holdings file. "
                "Switch the price source to Yahoo Finance to use the sample portfolio, or upload holdings.")
    else:
        lt = overlap.look_through(weights, holdings)
        c1, c2 = st.columns([3, 2])
        with c1:
            top = lt[~lt.index.str.endswith("(other holdings)")].head(15).iloc[::-1]
            fig = go.Figure(go.Bar(x=top["Exposure"], y=top.index, orientation="h",
                                   marker_color=[ORANGE if n > 1 else BLUE for n in top["Sources"]],
                                   customdata=top["Held via"],
                                   hovertemplate="%{y}: %{x:.2%}<br>via %{customdata}<extra></extra>"))
            fig.update_layout(title="Largest underlying exposures (orange = held through 2+ routes)",
                              xaxis_tickformat=".1%", hovermode="closest")
            show(fig, 520)
        with c2:
            if len(funds_in_port) > 1:
                ov = overlap.pairwise_overlap(holdings, funds_in_port)
                fig = go.Figure(go.Heatmap(z=ov.values, x=ov.columns, y=ov.index, zmin=0, zmax=1,
                                           colorscale=[[0, "#f9fafb"], [1, ORANGE]],
                                           text=[[f"{v:.0%}" for v in row] for row in ov.values],
                                           texttemplate="%{text}", showscale=False))
                fig.update_layout(title="Fund-to-fund overlap", hovermode="closest", yaxis_autorange="reversed")
                show(fig, 280)
                st.caption("Overlap = sum over shared holdings of the smaller of the two weights.")
            hc = overlap.hidden_concentration(weights, holdings)
            if not hc.empty:
                st.markdown("**Held through more than one route**")
                st.dataframe(hc.assign(Exposure=hc["Exposure"].map(lambda x: pct(x, 2)))[["Exposure", "Held via"]],
                             width="stretch")
        st.download_button("Download look-through (CSV)", lt.to_csv().encode(), "look_through.csv", "text/csv")

# ----------------------------------------------------------------------------- report
with tabs[6]:
    md = markdown_report(a, data.source, value)
    st.download_button("Download report (Markdown)", md.encode(), "risklite_report.md", "text/markdown",
                       type="primary")
    buf = io.StringIO()
    a.var_table.to_csv(buf); buf.write("\n"); a.risk_contrib.to_csv(buf)
    st.download_button("Download tables (CSV)", buf.getvalue().encode(), "risklite_tables.csv", "text/csv")
    with st.expander("Preview report", expanded=True):
        st.markdown(md)

st.divider()
st.caption(f"RiskLite {__version__} · Descriptive analysis of historical data, not investment advice. "
           "Past risk does not guarantee future risk. Methodology: docs/METHODOLOGY.md")
