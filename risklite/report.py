"""Run the whole analysis in one call and turn the numbers into plain-language findings.

The findings are descriptive ("the two bank-heavy positions produce 38% of the risk"),
never advice ("sell X"). RiskLite is an analytical tool, not a recommendation engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import backtest as bt
from . import contrib, metrics, stress
from . import var as v
from .data import to_returns


@dataclass
class Analysis:
    weights: dict[str, float]
    returns: pd.DataFrame
    port: pd.Series
    summary: dict[str, float]
    drawdown: metrics.DrawdownInfo
    concentration: dict[str, float]
    var_table: pd.DataFrame
    backtests: dict[str, bt.BacktestResult]
    historical_stress: list[stress.ScenarioResult]
    hypothetical_stress: pd.DataFrame
    worst_periods: pd.DataFrame
    risk_contrib: pd.DataFrame
    es_contrib: pd.Series
    diversification_ratio: float
    correlation: pd.DataFrame
    benchmark: dict[str, float] = field(default_factory=dict)
    alpha: float = 0.99
    horizon: int = 1
    findings: list[str] = field(default_factory=list)


def run_analysis(prices: pd.DataFrame, weights: dict[str, float], alpha: float = 0.99,
                 horizon: int = 1, rf_annual: float = 0.0, benchmark: pd.Series | None = None,
                 asset_classes: dict[str, str] | None = None, window: int = 250,
                 n_sims: int = 20_000, cov_method: str = "sample") -> Analysis:
    rets = to_returns(prices[list(weights)])
    port = metrics.portfolio_returns(rets, weights)
    cov = contrib.ewma_cov(rets) if cov_method == "ewma" else contrib.sample_cov(rets)
    rc = contrib.euler_vol_contributions(weights, cov)
    rc["Component VaR"] = contrib.component_var(weights, cov, alpha) * (horizon ** 0.5)
    bts = {}
    for m in ("historical", "parametric", "ewma"):
        try:
            bts[m] = bt.backtest(port, alpha, window, m)
        except ValueError:
            pass
    bench_stats = {}
    if benchmark is not None:
        bench_stats = metrics.benchmark_stats(port, benchmark.pct_change().dropna())
    a = Analysis(
        weights=weights, returns=rets, port=port,
        summary=metrics.summary(port, rf_annual), drawdown=metrics.max_drawdown(port),
        concentration=metrics.concentration(weights),
        var_table=v.var_table(port, rets, weights, alpha, horizon, n_sims),
        backtests=bts,
        historical_stress=stress.historical_stress(rets, weights),
        hypothetical_stress=stress.hypothetical_stress(weights, asset_classes or {}),
        worst_periods=stress.worst_periods(port),
        risk_contrib=rc, es_contrib=contrib.historical_es_contributions(rets, weights, alpha),
        diversification_ratio=contrib.diversification_ratio(weights, cov),
        correlation=contrib.correlation_matrix(rets), benchmark=bench_stats,
        alpha=alpha, horizon=horizon)
    a.findings = findings(a)
    return a


def _pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def findings(a: Analysis) -> list[str]:
    """Plain-language observations generated from the numbers."""
    out = []
    s, vt = a.summary, a.var_table
    h = "1-day" if a.horizon == 1 else f"{a.horizon}-day"
    out.append(
        f"Annualised volatility is {_pct(s['annual_volatility'])} and the worst peak-to-trough "
        f"fall in the sample was {_pct(s['max_drawdown'])}"
        + (f" ({a.drawdown.peak.date()} to {a.drawdown.trough.date()})." if a.drawdown.peak is not None else "."))
    hv, pv = vt.loc["Historical", "VaR"], vt.loc["Parametric (normal)", "VaR"]
    he = vt.loc["Historical", "ES"]
    out.append(f"{h} {a.alpha:.0%} VaR is {_pct(hv, 2)} (historical) against {_pct(pv, 2)} under a normal "
               f"distribution; on the days beyond VaR the average loss (ES) is {_pct(he, 2)}.")
    if hv > pv * 1.1:
        out.append(f"The normal model understates tail risk by about {_pct(hv / pv - 1, 0)}: returns are "
                   f"fat-tailed (excess kurtosis {s['excess_kurtosis']:.1f}).")
    rc = a.risk_contrib.sort_values("Share of risk", ascending=False)
    top = rc.iloc[0]
    out.append(f"{rc.index[0]} is {_pct(top['Weight'], 0)} of the portfolio but {_pct(top['Share of risk'], 0)} "
               f"of its volatility.")
    over = rc[rc["Risk / weight"] > 1.3]
    if len(over) > 1:
        out.append(f"{len(over)} positions contribute at least 30% more risk than their weight: "
                   f"{', '.join(over.index)} ({_pct(over['Share of risk'].sum(), 0)} of risk from "
                   f"{_pct(over['Weight'].sum(), 0)} of capital).")
    out.append(f"Diversification ratio is {a.diversification_ratio:.2f} (1.00 would mean no diversification "
               f"benefit); the portfolio behaves like {a.concentration['effective_positions']:.1f} "
               f"equal-weight positions.")
    for name, r in a.backtests.items():
        out.append(f"Backtest ({name} VaR): {r.n_exc} exceptions in {r.n_obs} days vs {r.expected_exc:.1f} "
                   f"expected; Kupiec p = {r.kupiec_p:.3f}, Basel zone {r.zone}. {r.verdict}.")
    covered = [x for x in a.historical_stress if x.covered]
    if covered:
        worst = min(covered, key=lambda x: x.total_return)
        out.append(f"Worst historical scenario replayed: {worst.name}, {_pct(worst.total_return)} over "
                   f"{worst.days} trading days.")
    missing = [x.name for x in a.historical_stress if not x.covered]
    if missing:
        out.append(f"Not enough price history to replay: {', '.join(missing)}.")
    return out


def to_markdown_tables(a: Analysis) -> dict[str, str]:
    """Key tables as Markdown strings (used by the case-study writer and the app download)."""
    fmt = lambda df: df.to_markdown(floatfmt=".4f")  # noqa: E731
    hs = pd.DataFrame([{"Scenario": x.name, "Window": f"{x.start} to {x.end}",
                        "Return": x.total_return if x.covered else None,
                        "Max drawdown": x.max_drawdown if x.covered else None,
                        "Covered": "yes" if x.covered else "no data"} for x in a.historical_stress])
    bts = pd.DataFrame([{"Method": k, "Days": r.n_obs, "Exceptions": r.n_exc,
                         "Expected": round(r.expected_exc, 1), "Kupiec p": r.kupiec_p,
                         "Christoffersen p": r.christoffersen_p, "Zone": r.zone, "Verdict": r.verdict}
                        for k, r in a.backtests.items()])
    return {"var": fmt(a.var_table), "contrib": fmt(a.risk_contrib), "stress": hs.to_markdown(index=False, floatfmt=".4f"),
            "hypothetical": fmt(a.hypothetical_stress), "backtest": bts.to_markdown(index=False, floatfmt=".3f"),
            "worst": a.worst_periods.to_markdown(index=False, floatfmt=".4f")}


def markdown_report(a: Analysis, source: str, portfolio_value: float | None = None) -> str:
    """A self-contained Markdown report of the analysis (the app's download button)."""
    t = to_markdown_tables(a)
    s = a.summary
    money = ""
    if portfolio_value:
        hv = a.var_table.loc["Historical", "VaR"] * portfolio_value
        he = a.var_table.loc["Historical", "ES"] * portfolio_value
        money = (f"\nOn a portfolio of EUR {portfolio_value:,.0f}: historical VaR EUR {hv:,.0f}, "
                 f"ES EUR {he:,.0f}.\n")
    w = "\n".join(f"| {k} | {x:.1%} |" for k, x in sorted(a.weights.items(), key=lambda kv: -kv[1]))
    return "\n".join([
        "# RiskLite portfolio risk report", "",
        f"Data: {source}. Sample {a.port.index[0].date()} to {a.port.index[-1].date()} "
        f"({len(a.port)} trading days). Confidence {a.alpha:.0%}, horizon {a.horizon} day(s).", "",
        "Descriptive analysis of historical risk; not investment advice.", "",
        "## Portfolio", "", "| Position | Weight |", "|---|---|", w, "",
        "## Findings", "", *[f"- {f}" for f in a.findings], "",
        "## Summary", "",
        f"Annual return {s['annual_return']:.2%}, volatility {s['annual_volatility']:.2%}, "
        f"Sharpe {s['sharpe']:.2f}, max drawdown {s['max_drawdown']:.2%}.", money,
        "## VaR and ES", "", t["var"], "", "## Backtest", "", t["backtest"], "",
        "## Risk contributions", "", t["contrib"], "",
        "## Stress tests", "", t["stress"], "", t["hypothetical"], "", "## Worst periods", "", t["worst"], "",
    ])
