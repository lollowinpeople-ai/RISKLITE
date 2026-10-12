"""Performance, drawdown and benchmark-relative statistics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .data import TRADING_DAYS


def portfolio_returns(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Daily returns of a constant-mix portfolio (rebalanced to target weights every day).

    Constant-mix is the standard assumption for risk measurement: it describes the risk of
    the allocation, not of a particular drift path.
    """
    missing = [t for t in weights if t not in returns.columns]
    if missing:
        raise KeyError(f"No return data for: {', '.join(missing)}")
    w = pd.Series(weights, dtype=float)
    return (returns[w.index] * w).sum(axis=1).rename("Portfolio")


def annualised_return(r: pd.Series) -> float:
    """Geometric (compound) annual growth rate."""
    r = r.dropna()
    if r.empty:
        return float("nan")
    growth = float((1 + r).prod())
    years = len(r) / TRADING_DAYS
    return growth ** (1 / years) - 1 if growth > 0 else -1.0


def annualised_vol(r: pd.Series) -> float:
    return float(r.dropna().std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(r: pd.Series, rf_annual: float = 0.0) -> float:
    vol = annualised_vol(r)
    return (annualised_return(r) - rf_annual) / vol if vol > 0 else float("nan")


def sortino_ratio(r: pd.Series, rf_annual: float = 0.0) -> float:
    """Excess return over downside deviation (target = daily risk-free rate)."""
    r = r.dropna()
    target = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    downside = np.minimum(r - target, 0)
    dd = float(np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS))
    return (annualised_return(r) - rf_annual) / dd if dd > 0 else float("nan")


def drawdown_series(r: pd.Series) -> pd.Series:
    """Percentage below the running peak of the cumulative wealth index (0 or negative)."""
    wealth = (1 + r.fillna(0)).cumprod()
    return (wealth / wealth.cummax() - 1).rename("Drawdown")


@dataclass
class DrawdownInfo:
    max_drawdown: float
    peak: pd.Timestamp | None
    trough: pd.Timestamp | None
    recovery: pd.Timestamp | None  # None if not yet recovered

    @property
    def days_to_recover(self) -> int | None:
        if self.recovery is None or self.trough is None:
            return None
        return int(np.busday_count(self.trough.date(), self.recovery.date()))


def max_drawdown(r: pd.Series) -> DrawdownInfo:
    dd = drawdown_series(r)
    if dd.empty or dd.min() == 0:
        return DrawdownInfo(0.0, None, None, None)
    trough = dd.idxmin()
    wealth = (1 + r.fillna(0)).cumprod()
    peak = wealth.loc[:trough].idxmax()
    after = wealth.loc[trough:]
    recovered = after[after >= wealth.loc[peak]]
    recovery = recovered.index[0] if not recovered.empty else None
    return DrawdownInfo(float(dd.min()), peak, trough, recovery)


def benchmark_stats(r: pd.Series, bench: pd.Series) -> dict[str, float]:
    """Beta, correlation, tracking error and information ratio versus a benchmark."""
    df = pd.concat([r, bench], axis=1, join="inner").dropna()
    if len(df) < 30:
        return {"beta": np.nan, "correlation": np.nan, "tracking_error": np.nan,
                "information_ratio": np.nan}
    p, b = df.iloc[:, 0], df.iloc[:, 1]
    beta = float(np.cov(p, b, ddof=1)[0, 1] / np.var(b, ddof=1))
    active = p - b
    te = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))
    ir = (annualised_return(p) - annualised_return(b)) / te if te > 0 else np.nan
    return {"beta": beta, "correlation": float(p.corr(b)), "tracking_error": te,
            "information_ratio": float(ir)}


def summary(r: pd.Series, rf_annual: float = 0.0) -> dict[str, float]:
    r = r.dropna()
    mdd = max_drawdown(r)
    ann_ret = annualised_return(r)
    return {
        "annual_return": ann_ret,
        "annual_volatility": annualised_vol(r),
        "sharpe": sharpe_ratio(r, rf_annual),
        "sortino": sortino_ratio(r, rf_annual),
        "max_drawdown": mdd.max_drawdown,
        "calmar": ann_ret / abs(mdd.max_drawdown) if mdd.max_drawdown < 0 else np.nan,
        "skewness": float(stats.skew(r, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(r, fisher=True, bias=False)),
        "best_day": float(r.max()),
        "worst_day": float(r.min()),
        "observations": float(len(r)),
    }


def concentration(weights: dict[str, float]) -> dict[str, float]:
    """Herfindahl index of weights and the equivalent number of equal-weight positions."""
    w = np.array(list(weights.values()), dtype=float)
    hhi = float((w ** 2).sum())
    return {"hhi": hhi, "effective_positions": 1 / hhi if hhi > 0 else np.nan,
            "largest_weight": float(w.max())}
