"""Value at Risk and Expected Shortfall.

Conventions used throughout:
  * `alpha` is the confidence level (0.95 or 0.99).
  * VaR and ES are reported as POSITIVE numbers meaning a loss, as a fraction of portfolio
    value. VaR 99% = 0.021 means "on 99% of days the loss should not exceed 2.1%".
  * Horizons longer than one day use the square-root-of-time rule. It assumes independent,
    identically distributed returns and understates risk when volatility clusters; the
    methodology note explains this limitation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _check_alpha(alpha: float) -> None:
    if not 0.5 < alpha < 1:
        raise ValueError("alpha must be between 0.5 and 1, e.g. 0.95 or 0.99")


def historical_var(r: pd.Series, alpha: float = 0.99, horizon: int = 1) -> float:
    """Loss not exceeded on `alpha` of past days: minus the (1-alpha) empirical quantile."""
    _check_alpha(alpha)
    q = float(np.quantile(r.dropna(), 1 - alpha))
    return -q * np.sqrt(horizon)


def historical_es(r: pd.Series, alpha: float = 0.99, horizon: int = 1) -> float:
    """Average loss on the days at or beyond the VaR threshold (Expected Shortfall / CVaR)."""
    _check_alpha(alpha)
    x = r.dropna().to_numpy()
    q = np.quantile(x, 1 - alpha)
    tail = x[x <= q]
    return -float(tail.mean()) * np.sqrt(horizon)


def parametric_var(r: pd.Series, alpha: float = 0.99, horizon: int = 1,
                   include_mean: bool = False) -> float:
    """Normal (variance-covariance) VaR. Zero-mean by default, the usual short-horizon choice."""
    _check_alpha(alpha)
    x = r.dropna()
    mu = float(x.mean()) * horizon if include_mean else 0.0
    sigma = float(x.std(ddof=1)) * np.sqrt(horizon)
    return -(mu + sigma * stats.norm.ppf(1 - alpha))


def parametric_es(r: pd.Series, alpha: float = 0.99, horizon: int = 1,
                  include_mean: bool = False) -> float:
    """Normal ES: sigma * pdf(z) / (1 - alpha) - mu."""
    _check_alpha(alpha)
    x = r.dropna()
    mu = float(x.mean()) * horizon if include_mean else 0.0
    sigma = float(x.std(ddof=1)) * np.sqrt(horizon)
    z = stats.norm.ppf(1 - alpha)
    return sigma * stats.norm.pdf(z) / (1 - alpha) - mu


def cornish_fisher_var(r: pd.Series, alpha: float = 0.99, horizon: int = 1,
                       include_mean: bool = False) -> float:
    """Modified VaR: adjusts the normal quantile for the sample's skewness and kurtosis.

    Useful when returns are fat-tailed or skewed, which equity returns usually are.
    """
    _check_alpha(alpha)
    x = r.dropna()
    s = float(stats.skew(x, bias=False))
    k = float(stats.kurtosis(x, fisher=True, bias=False))
    z = stats.norm.ppf(1 - alpha)
    z_cf = (z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * s ** 2 / 36)
    mu = float(x.mean()) * horizon if include_mean else 0.0
    sigma = float(x.std(ddof=1)) * np.sqrt(horizon)
    return -(mu + sigma * z_cf)


def monte_carlo_var_es(asset_returns: pd.DataFrame, weights: dict[str, float],
                       alpha: float = 0.99, horizon: int = 1, n_sims: int = 20_000,
                       method: str = "bootstrap", seed: int | None = 42) -> tuple[float, float]:
    """Simulated VaR and ES for the portfolio.

    method="bootstrap" resamples whole historical days (keeps real fat tails and the
    cross-asset dependence of each day); method="normal" draws from a multivariate normal
    with the sample covariance. Multi-day horizons compound `horizon` independent draws.
    """
    _check_alpha(alpha)
    rng = np.random.default_rng(seed)
    w = pd.Series(weights, dtype=float)
    R = asset_returns[w.index].dropna().to_numpy()
    wv = w.to_numpy()
    if method == "bootstrap":
        idx = rng.integers(0, len(R), size=(n_sims, horizon))
        sims = R[idx]  # (n_sims, horizon, n_assets)
    elif method == "normal":
        mu, cov = R.mean(axis=0), np.cov(R, rowvar=False)
        sims = rng.multivariate_normal(mu, cov, size=(n_sims, horizon))
    else:
        raise ValueError("method must be 'bootstrap' or 'normal'")
    port_daily = sims @ wv  # (n_sims, horizon)
    port = np.prod(1 + port_daily, axis=1) - 1
    q = np.quantile(port, 1 - alpha)
    return -float(q), -float(port[port <= q].mean())


def var_table(port_returns: pd.Series, asset_returns: pd.DataFrame, weights: dict[str, float],
              alpha: float = 0.99, horizon: int = 1, n_sims: int = 20_000) -> pd.DataFrame:
    """All methods side by side, as fractions of portfolio value."""
    mc_var, mc_es = monte_carlo_var_es(asset_returns, weights, alpha, horizon, n_sims)
    rows = [
        ("Historical", historical_var(port_returns, alpha, horizon), historical_es(port_returns, alpha, horizon)),
        ("Parametric (normal)", parametric_var(port_returns, alpha, horizon), parametric_es(port_returns, alpha, horizon)),
        ("Cornish-Fisher (skew/kurtosis)", cornish_fisher_var(port_returns, alpha, horizon), np.nan),
        ("Monte Carlo (bootstrap)", mc_var, mc_es),
    ]
    return pd.DataFrame(rows, columns=["Method", "VaR", "ES"]).set_index("Method")
