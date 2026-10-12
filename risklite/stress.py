"""Stress testing: what would have happened, and what could happen.

Two complementary approaches:

* Historical replay: apply the actual daily returns of the current holdings over a past
  crisis window. Exact, but only possible when every holding has prices for that window.
* Hypothetical shocks: an instantaneous move per asset class (e.g. equities -30%,
  government bonds +3%). Always available. The default numbers are illustrative round
  figures for discussion, not calibrated forecasts, and every one is editable in the app.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import drawdown_series

HISTORICAL_SCENARIOS: dict[str, tuple[str, str, str]] = {
    # name: (start, end, short description)
    "Global Financial Crisis": ("2008-09-12", "2009-03-09", "Lehman collapse to the equity market low"),
    "Euro sovereign crisis": ("2011-07-01", "2011-11-25", "Italian and Spanish spreads widen, banks sell off"),
    "COVID crash": ("2020-02-19", "2020-03-23", "Pandemic sell-off, fastest bear market on record"),
    "2022 rate shock": ("2022-01-03", "2022-10-12", "Inflation and rate hikes hit stocks and bonds together"),
}

# Illustrative instantaneous shocks by asset class. Editable; not forecasts.
HYPOTHETICAL_SCENARIOS: dict[str, dict[str, float]] = {
    "Equity crash": {"Equity": -0.30, "Government bond": 0.03, "Corporate bond": -0.05,
                     "Commodity": 0.05, "Cash": 0.0, "Other": -0.10},
    "Rates +200bp": {"Equity": -0.10, "Government bond": -0.12, "Corporate bond": -0.10,
                     "Commodity": -0.03, "Cash": 0.0, "Other": -0.05},
    "Stagflation": {"Equity": -0.20, "Government bond": -0.08, "Corporate bond": -0.10,
                    "Commodity": 0.15, "Cash": 0.0, "Other": -0.08},
    "Italian spread blow-out": {"Equity": -0.15, "Government bond": -0.06, "Corporate bond": -0.08,
                                "Commodity": 0.05, "Cash": 0.0, "Other": -0.06},
}
ASSET_CLASSES = ["Equity", "Government bond", "Corporate bond", "Commodity", "Cash", "Other"]


@dataclass
class ScenarioResult:
    name: str
    start: str
    end: str
    description: str
    covered: bool
    total_return: float = float("nan")
    max_drawdown: float = float("nan")
    worst_day: float = float("nan")
    days: int = 0
    path: pd.Series | None = None  # cumulative wealth index, starting at 1
    asset_returns: pd.Series | None = None  # each holding's return over the window


def replay_scenario(returns: pd.DataFrame, weights: dict[str, float], name: str,
                    start: str, end: str, description: str = "",
                    min_coverage: float = 0.9) -> ScenarioResult:
    """Replay the daily returns of a past window on the current constant-mix portfolio.

    The scenario counts as covered when the data has trading days for at least
    `min_coverage` of the business days in the window.
    """
    w = pd.Series(weights, dtype=float)
    window = returns.loc[start:end, w.index].dropna()
    expected = len(pd.bdate_range(start, end))
    # business days include exchange holidays, so a fully covered window has ~96-98% of them
    if expected == 0 or len(window) < min_coverage * expected:
        return ScenarioResult(name, start, end, description, covered=False)
    port = (window * w).sum(axis=1)
    wealth = (1 + port).cumprod()
    asset_tot = (1 + window).prod() - 1
    return ScenarioResult(name, start, end, description, True,
                          total_return=float(wealth.iloc[-1] - 1),
                          max_drawdown=float(drawdown_series(port).min()),
                          worst_day=float(port.min()), days=len(port),
                          path=pd.concat([pd.Series([1.0], index=[window.index[0] - pd.Timedelta(days=1)]), wealth]),
                          asset_returns=asset_tot)


def historical_stress(returns: pd.DataFrame, weights: dict[str, float],
                      scenarios: dict[str, tuple[str, str, str]] | None = None) -> list[ScenarioResult]:
    scenarios = scenarios or HISTORICAL_SCENARIOS
    return [replay_scenario(returns, weights, n, s, e, d) for n, (s, e, d) in scenarios.items()]


def hypothetical_stress(weights: dict[str, float], asset_classes: dict[str, str],
                        scenarios: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    """Portfolio P&L for instantaneous shocks by asset class.

    Holdings without a known class are treated as "Other". Returns one row per scenario
    with the total impact and the contribution of each asset class.
    """
    scenarios = scenarios or HYPOTHETICAL_SCENARIOS
    w = pd.Series(weights, dtype=float)
    cls = pd.Series({t: asset_classes.get(t, "Other") for t in w.index})
    class_w = w.groupby(cls).sum()
    rows = {}
    for name, shocks in scenarios.items():
        contrib = {c: class_w.get(c, 0.0) * shocks.get(c, 0.0) for c in class_w.index}
        rows[name] = {"Portfolio impact": sum(contrib.values()), **contrib}
    return pd.DataFrame(rows).T.fillna(0.0)


def worst_periods(r: pd.Series, lengths: tuple[int, ...] = (1, 5, 21), top: int = 3) -> pd.DataFrame:
    """The worst non-overlapping rolling periods of each length, compounded.

    1, 5 and 21 trading days are roughly a day, a week and a month.
    """
    r = r.dropna()
    rows = []
    for n in lengths:
        roll = (1 + r).rolling(n).apply(np.prod, raw=True) - 1
        roll = roll.dropna().sort_values()
        used: list[pd.Timestamp] = []
        for end, val in roll.items():
            pos = r.index.get_loc(end)
            if any(abs(pos - r.index.get_loc(u)) < n for u in used):
                continue
            used.append(end)
            rows.append({"Length (days)": n, "Start": r.index[pos - n + 1].date(),
                         "End": end.date(), "Return": float(val)})
            if len(used) == top:
                break
    return pd.DataFrame(rows)
