"""VaR backtesting: does the model's predicted loss frequency match reality?

A 99% VaR should be exceeded on about 1% of days. Too many exceptions means the model
underestimates risk; too few means it is over-conservative. Exceptions should also be
scattered over time rather than clustered, otherwise the model is slow to react.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .var import historical_var, parametric_var


def rolling_var(r: pd.Series, window: int = 250, alpha: float = 0.99,
                method: str = "historical", ewma_lambda: float = 0.94) -> pd.Series:
    """One-day-ahead VaR forecasts. The forecast dated t only uses returns up to t-1.

    method: "historical", "parametric" (rolling window), or "ewma" (RiskMetrics volatility
    with decay `ewma_lambda`, which reacts faster to volatility changes).
    """
    r = r.dropna()
    if len(r) <= window:
        raise ValueError(f"Need more than {window} observations for a {window}-day rolling VaR.")
    if method == "historical":
        f = r.rolling(window).quantile(1 - alpha, interpolation="linear")
        out = -f
    elif method == "parametric":
        out = -(r.rolling(window).std(ddof=1) * stats.norm.ppf(1 - alpha))
    elif method == "ewma":
        var = np.empty(len(r))
        var[0] = r.iloc[:window].var(ddof=1)
        x = r.to_numpy()
        for t in range(1, len(r)):
            var[t] = ewma_lambda * var[t - 1] + (1 - ewma_lambda) * x[t - 1] ** 2
        # var[t] already uses information up to t-1, so no extra shift below
        out = pd.Series(-np.sqrt(var) * stats.norm.ppf(1 - alpha), index=r.index)
        out.iloc[:window] = np.nan
        return out.rename(f"VaR {alpha:.0%} ({method})")
    else:
        raise ValueError("method must be 'historical', 'parametric' or 'ewma'")
    return out.shift(1).rename(f"VaR {alpha:.0%} ({method})")


def exceptions(r: pd.Series, var_forecast: pd.Series) -> pd.Series:
    """True on days when the realised loss exceeded the forecast VaR."""
    df = pd.concat([r, var_forecast], axis=1, join="inner").dropna()
    return (df.iloc[:, 0] < -df.iloc[:, 1]).rename("exception")


def kupiec_pof(n_obs: int, n_exc: int, alpha: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures test. Returns (LR statistic, p-value).

    H0: the true exception probability equals 1 - alpha. LR ~ chi-square(1).
    """
    p = 1 - alpha
    x, n = n_exc, n_obs
    if n <= 0:
        return float("nan"), float("nan")
    phat = x / n

    def ll(prob: float) -> float:
        # log-likelihood of x exceptions in n Bernoulli trials; 0*log(0) treated as 0
        a = (n - x) * np.log(1 - prob) if n - x > 0 else 0.0
        b = x * np.log(prob) if x > 0 else 0.0
        return a + b

    lr = -2 * (ll(p) - ll(phat)) if 0 < phat < 1 else -2 * (ll(p) - 0.0)
    return float(lr), float(1 - stats.chi2.cdf(lr, df=1))


def christoffersen_independence(exc: pd.Series) -> tuple[float, float]:
    """Christoffersen (1998) independence test. Returns (LR statistic, p-value).

    H0: an exception today does not make an exception tomorrow more likely. LR ~ chi-square(1).
    """
    e = exc.astype(int).to_numpy()
    if len(e) < 2:
        return float("nan"), float("nan")
    prev, curr = e[:-1], e[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum()); n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum()); n11 = int(((prev == 1) & (curr == 1)).sum())
    pi0 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def term(n: int, p: float) -> float:
        return n * np.log(p) if n > 0 and p > 0 else 0.0

    ll_null = term(n00 + n10, 1 - pi) + term(n01 + n11, pi)
    ll_alt = term(n00, 1 - pi0) + term(n01, pi0) + term(n10, 1 - pi1) + term(n11, pi1)
    lr = -2 * (ll_null - ll_alt)
    return float(lr), float(1 - stats.chi2.cdf(lr, df=1))


def traffic_light(n_obs: int, n_exc: int, alpha: float) -> str:
    """Basel-style zone from the cumulative binomial probability of the exception count.

    Green below 95% cumulative probability, yellow below 99.99%, red otherwise. For 250
    days at 99% this reproduces the familiar 0-4 green, 5-9 yellow, 10+ red bands.
    """
    cum = stats.binom.cdf(n_exc, n_obs, 1 - alpha)
    if cum < 0.95:
        return "Green"
    if cum < 0.9999:
        return "Yellow"
    return "Red"


@dataclass
class BacktestResult:
    method: str
    alpha: float
    window: int
    n_obs: int
    n_exc: int
    expected_exc: float
    kupiec_lr: float
    kupiec_p: float
    christoffersen_lr: float
    christoffersen_p: float
    cc_lr: float
    cc_p: float
    zone: str
    forecast: pd.Series
    exceptions: pd.Series

    @property
    def exception_rate(self) -> float:
        return self.n_exc / self.n_obs if self.n_obs else float("nan")

    @property
    def verdict(self) -> str:
        if self.kupiec_p < 0.05 and self.exception_rate > (1 - self.alpha):
            return "Underestimates risk: too many exceptions"
        if self.kupiec_p < 0.05:
            return "Too conservative: too few exceptions"
        if self.christoffersen_p < 0.05:
            return "Right frequency, but exceptions cluster in time"
        return "Passes: frequency and independence consistent with the model"


def backtest(r: pd.Series, alpha: float = 0.99, window: int = 250,
             method: str = "historical") -> BacktestResult:
    f = rolling_var(r, window, alpha, method)
    exc = exceptions(r, f)
    n, x = len(exc), int(exc.sum())
    k_lr, k_p = kupiec_pof(n, x, alpha)
    c_lr, c_p = christoffersen_independence(exc)
    cc_lr = k_lr + c_lr
    cc_p = float(1 - stats.chi2.cdf(cc_lr, df=2))
    return BacktestResult(method, alpha, window, n, x, n * (1 - alpha), k_lr, k_p, c_lr, c_p,
                          cc_lr, cc_p, traffic_light(n, x, alpha), f.dropna(), exc)
