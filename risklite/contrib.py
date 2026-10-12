"""Where does the risk come from? Covariance estimates and risk decomposition.

A position's weight and its share of risk are different things: a 12% bank stock can
drive 25% of portfolio volatility. Euler decomposition splits a risk measure into parts
that add up exactly to the total, so each holding gets a fair share.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .data import TRADING_DAYS


def sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.dropna().cov(ddof=1)


def ewma_cov(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """RiskMetrics exponentially weighted covariance (recent days weigh more).

    Uses zero-mean returns, the RiskMetrics convention for daily data.
    """
    if not 0 < lam < 1:
        raise ValueError("lambda must be between 0 and 1")
    x = returns.dropna().to_numpy()
    n = len(x)
    wts = (1 - lam) * lam ** np.arange(n - 1, -1, -1)
    wts = wts / wts.sum()
    cov = (x * wts[:, None]).T @ x
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def euler_vol_contributions(weights: dict[str, float], cov: pd.DataFrame,
                            annualise: bool = True) -> pd.DataFrame:
    """Decompose portfolio volatility: contribution_i = w_i * (Σw)_i / σ_p. Sums to σ_p."""
    w = pd.Series(weights, dtype=float)
    S = cov.loc[w.index, w.index].to_numpy()
    wv = w.to_numpy()
    sigma = float(np.sqrt(wv @ S @ wv))
    if sigma == 0:
        raise ValueError("Portfolio variance is zero.")
    marginal = S @ wv / sigma
    contrib = wv * marginal
    scale = np.sqrt(TRADING_DAYS) if annualise else 1.0
    out = pd.DataFrame({"Weight": wv, "Marginal vol": marginal * scale,
                        "Vol contribution": contrib * scale,
                        "Share of risk": contrib / sigma}, index=w.index)
    out["Risk / weight"] = out["Share of risk"] / out["Weight"]
    return out


def component_var(weights: dict[str, float], cov: pd.DataFrame, alpha: float = 0.99) -> pd.Series:
    """Parametric (normal, zero-mean) component VaR. Sums to the portfolio parametric VaR."""
    vc = euler_vol_contributions(weights, cov, annualise=False)["Vol contribution"]
    return (-stats.norm.ppf(1 - alpha) * vc).rename("Component VaR")


def historical_es_contributions(returns: pd.DataFrame, weights: dict[str, float],
                                alpha: float = 0.99) -> pd.Series:
    """Each holding's average loss on the portfolio's tail days. Sums to historical ES.

    This is the Euler allocation of historical ES: it shows who actually lost money on the
    worst days, with no normality assumption.
    """
    w = pd.Series(weights, dtype=float)
    R = returns[w.index].dropna()
    port = R @ w
    q = np.quantile(port, 1 - alpha)
    tail = R[port <= q]
    return (-(tail * w).mean()).rename("ES contribution")


def diversification_ratio(weights: dict[str, float], cov: pd.DataFrame) -> float:
    """Weighted average of stand-alone vols divided by portfolio vol (1 = no diversification)."""
    w = pd.Series(weights, dtype=float)
    S = cov.loc[w.index, w.index].to_numpy()
    wv = w.to_numpy()
    return float(wv @ np.sqrt(np.diag(S)) / np.sqrt(wv @ S @ wv))


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.dropna().corr()
