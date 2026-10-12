"""Generate the case study: a Markdown report plus charts for the sample Italian portfolio.

    python -m risklite.casestudy --source yahoo       # real prices, needs internet
    python -m risklite.casestudy --source synthetic   # offline demo data

Outputs go to case_study/ (CASE_STUDY.md and PNG charts). Every number in the report is
computed by the package; nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402
import pandas as pd  # noqa: E402

from . import __version__  # noqa: E402
from .data import (SYNTHETIC_PORTFOLIO, load_prices_yfinance, read_asset_classes,  # noqa: E402
                   read_portfolio_csv, synthetic_asset_classes, synthetic_prices)
from .metrics import drawdown_series  # noqa: E402
from .report import Analysis, run_analysis, to_markdown_tables  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INK, MUTED, ACCENT, ALERT, GRID = "#1f2937", "#6b7280", "#1d4ed8", "#c2410c", "#e5e7eb"


def _style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_wealth(a: Analysis, path: Path) -> None:
    wealth = (1 + a.port).cumprod()
    dd = drawdown_series(a.port)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(wealth.index, wealth, color=ACCENT, linewidth=1.3)
    _style(ax1, "Growth of 1 (constant-mix portfolio)")
    ax2.fill_between(dd.index, dd, 0, color=ALERT, alpha=0.35, linewidth=0)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    _style(ax2, "Drawdown from previous peak")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def chart_backtest(a: Analysis, method: str, path: Path) -> None:
    r = a.backtests[method]
    ret = a.port.loc[r.forecast.index]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(ret.index, ret, width=1.0, color=MUTED, alpha=0.6, linewidth=0, label="Daily return")
    ax.plot(r.forecast.index, -r.forecast, color=ACCENT, linewidth=1.1, label=f"-VaR {a.alpha:.0%}")
    exc = r.exceptions[r.exceptions]
    ax.scatter(exc.index, a.port.loc[exc.index], color=ALERT, s=14, zorder=3,
               label=f"Exceptions ({r.n_exc} vs {r.expected_exc:.0f} expected)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    _style(ax, f"VaR backtest, {method} method: Basel zone {r.zone}")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def chart_contrib(a: Analysis, path: Path) -> None:
    rc = a.risk_contrib.sort_values("Share of risk")
    y = range(len(rc))
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rc) + 1.2))
    ax.barh([i + 0.2 for i in y], rc["Weight"], height=0.38, color=GRID, edgecolor=MUTED, label="Weight")
    ax.barh([i - 0.2 for i in y], rc["Share of risk"], height=0.38, color=ACCENT, label="Share of volatility")
    ax.set_yticks(list(y)); ax.set_yticklabels(rc.index)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    _style(ax, "Capital vs risk: weight and share of portfolio volatility")
    ax.grid(axis="x", color=GRID); ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def chart_stress(a: Analysis, path: Path) -> None:
    rows = [(x.name, x.total_return) for x in a.historical_stress if x.covered]
    rows += [(f"{n} (hypothetical)", v) for n, v in a.hypothetical_stress["Portfolio impact"].items()]
    df = pd.DataFrame(rows, columns=["Scenario", "Return"]).sort_values("Return", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(df) + 1.2))
    colors = [ALERT if v < 0 else ACCENT for v in df["Return"]]
    ax.barh(df["Scenario"], df["Return"], color=colors)
    for i, v in enumerate(df["Return"]):
        ax.text(v, i, f" {v:.1%} ", va="center", ha="right" if v < 0 else "left", fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    lo, hi = min(df["Return"].min(), 0), max(df["Return"].max(), 0)
    ax.set_xlim(lo - 0.18 * (hi - lo) - 0.01, hi + 0.18 * (hi - lo) + 0.01)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    _style(ax, "Stress scenarios: portfolio return")
    ax.grid(axis="x", color=GRID); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def write_report(a: Analysis, out: Path, source: str, notes: list[str], missing: list[str]) -> Path:
    t = to_markdown_tables(a)
    s = a.summary
    w = pd.Series(a.weights).sort_values(ascending=False)
    weights_md = "\n".join(f"| {k} | {v:.1%} |" for k, v in w.items())
    best_bt = "historical" if "historical" in a.backtests else next(iter(a.backtests), None)
    lines = [
        "# RiskLite case study: a typical Italian retail portfolio",
        "",
        f"*Generated {dt.date.today()} by RiskLite {__version__}. Data source: {source}. "
        f"Sample: {a.port.index[0].date()} to {a.port.index[-1].date()} ({len(a.port)} trading days).*",
        "",
    ]
    if "Synthetic" in source:
        lines += ["> **These results use simulated prices, not market data.** Run "
                  "`python -m risklite.casestudy --source yahoo` on a machine with internet access "
                  "to produce the real case study.", ""]
    lines += [
        "This report is descriptive analysis of historical risk. It is not investment advice.",
        "",
        "## Portfolio",
        "",
        "A home-biased portfolio of the kind often seen in Italian households: large Italian "
        "blue chips (energy, banking, utilities, insurance) next to global equity ETFs, a bond "
        "ETF and gold.",
        "",
        "| Position | Weight |", "|---|---|", weights_md, "",
    ]
    if missing:
        lines += [f"Tickers with no data (excluded): {', '.join(missing)}.", ""]
    for n in notes:
        lines += [f"Note: {n}", ""]
    lines += [
        "## Key findings", "",
        *[f"- {f}" for f in a.findings], "",
        "## Performance and drawdown", "",
        f"| Annual return | Annual volatility | Sharpe | Sortino | Max drawdown | Skewness | Excess kurtosis |",
        "|---|---|---|---|---|---|---|",
        f"| {s['annual_return']:.2%} | {s['annual_volatility']:.2%} | {s['sharpe']:.2f} | {s['sortino']:.2f} | "
        f"{s['max_drawdown']:.2%} | {s['skewness']:.2f} | {s['excess_kurtosis']:.2f} |", "",
        "![Growth and drawdown](fig_wealth.png)", "",
        f"## Value at Risk and Expected Shortfall ({a.horizon}-day, {a.alpha:.0%})", "",
        "Losses as a fraction of portfolio value.", "", t["var"], "",
        "## Does the VaR model work? Backtest", "",
        "Rolling 250-day one-day-ahead forecasts compared with the next day's realised return. "
        "Kupiec tests the exception frequency, Christoffersen tests whether exceptions cluster.", "",
        t["backtest"], "",
    ]
    if best_bt:
        lines += ["![Backtest](fig_backtest.png)", ""]
    lines += [
        "## Where the risk comes from", "",
        "Euler decomposition: contributions add up exactly to portfolio volatility.", "",
        t["contrib"], "", "![Risk contributions](fig_contrib.png)", "",
        "## Stress tests", "", "Historical replay of the current weights:", "", t["stress"], "",
        "Hypothetical instantaneous shocks by asset class (illustrative assumptions, editable in the app):",
        "", t["hypothetical"], "", "![Stress](fig_stress.png)", "",
        "Worst rolling periods in the sample:", "", t["worst"], "",
        "## Limitations", "",
        "- History is a sample, not a forecast. Risk measured over calm years understates crisis risk.",
        "- Constant-mix weights: the portfolio is assumed to be rebalanced daily to target weights.",
        "- Multi-day VaR uses square-root-of-time scaling, which ignores volatility clustering.",
        "- Hypothetical shocks are round-number assumptions for discussion, not calibrated forecasts.",
        "- Prices are adjusted closes from the stated source; data errors propagate into every figure.",
        "", "See docs/METHODOLOGY.md for formulas and references.",
    ]
    path = out / "CASE_STUDY.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description="Generate the RiskLite case study.")
    p.add_argument("--source", choices=["yahoo", "synthetic"], default="synthetic")
    p.add_argument("--portfolio", default=str(ROOT / "data" / "sample_portfolio.csv"))
    p.add_argument("--start", default="2007-01-01")
    p.add_argument("--benchmark", default=None, help="Benchmark ticker, e.g. SWDA.MI")
    p.add_argument("--alpha", type=float, default=0.99)
    p.add_argument("--out", default=str(ROOT / "case_study"))
    args = p.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    bench = None
    if args.source == "yahoo":
        weights = read_portfolio_csv(args.portfolio)
        classes = read_asset_classes(args.portfolio)
        tickers = list(weights) + ([args.benchmark] if args.benchmark else [])
        data = load_prices_yfinance(tickers, start=args.start)
        if args.benchmark and args.benchmark in data.prices:
            bench = data.prices[args.benchmark]
        weights = {k: v for k, v in weights.items() if k in data.prices.columns}
        total = sum(weights.values()); weights = {k: v / total for k, v in weights.items()}
    else:
        data = synthetic_prices()
        weights, classes = SYNTHETIC_PORTFOLIO, synthetic_asset_classes()
        bench = data.prices["SYN_WORLD_EQ"]

    a = run_analysis(data.prices, weights, alpha=args.alpha, benchmark=bench, asset_classes=classes)
    chart_wealth(a, out / "fig_wealth.png")
    if a.backtests:
        chart_backtest(a, "historical" if "historical" in a.backtests else next(iter(a.backtests)),
                       out / "fig_backtest.png")
    chart_contrib(a, out / "fig_contrib.png")
    chart_stress(a, out / "fig_stress.png")
    path = write_report(a, out, data.source, data.notes, data.missing)
    print(f"Case study written to {path}")
    return path


if __name__ == "__main__":
    main()
