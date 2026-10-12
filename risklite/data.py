"""Price loading, cleaning and return calculation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class PriceData:
    """Cleaned prices plus a record of anything that was dropped or filled."""

    prices: pd.DataFrame
    source: str
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- portfolio
def normalise_weights(weights: dict[str, float]) -> dict[str, float]:
    """Return weights rescaled to sum to 1. Rejects empty, negative-sum or NaN input."""
    clean = {str(k).strip(): float(v) for k, v in weights.items()
             if str(k).strip() and v is not None and not pd.isna(v) and float(v) != 0.0}
    if not clean:
        raise ValueError("The portfolio has no positions with a non-zero weight.")
    total = sum(clean.values())
    if total <= 0:
        raise ValueError("Portfolio weights must sum to a positive number.")
    return {k: v / total for k, v in clean.items()}


def read_portfolio_csv(path_or_buffer) -> dict[str, float]:
    """Read a portfolio file with columns `ticker` and `weight` (percent or fraction)."""
    df = pd.read_csv(path_or_buffer)
    cols = {c.lower().strip(): c for c in df.columns}
    if "ticker" not in cols or "weight" not in cols:
        raise ValueError("Portfolio CSV needs the columns 'ticker' and 'weight'.")
    return normalise_weights(dict(zip(df[cols["ticker"]], df[cols["weight"]])))


def read_asset_classes(path_or_buffer) -> dict[str, str]:
    """Optional `asset_class` column of a portfolio CSV (used by hypothetical stress tests)."""
    df = pd.read_csv(path_or_buffer)
    cols = {c.lower().strip(): c for c in df.columns}
    if "ticker" not in cols or "asset_class" not in cols:
        return {}
    return {str(t).strip(): str(c).strip() for t, c in zip(df[cols["ticker"]], df[cols["asset_class"]])
            if not pd.isna(c)}


# --------------------------------------------------------------------------- loaders
def load_prices_yfinance(tickers: Iterable[str], start: str, end: str | None = None) -> PriceData:
    """Download adjusted daily closes from Yahoo Finance.

    Tickers that return no data are reported in `missing` instead of raising, so a single
    bad symbol does not break the whole analysis.
    """
    import yfinance as yf  # imported lazily so the rest of the package works offline

    tickers = list(dict.fromkeys(t.strip() for t in tickers if t and t.strip()))
    if not tickers:
        raise ValueError("No tickers to download.")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False,
                      group_by="column", threads=True)
    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance returned no data. Check the tickers and your connection.")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", axis=1, level=-1)
    else:  # a single ticker comes back with flat columns
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    close = close.reindex(columns=tickers)
    missing = [t for t in tickers if t not in close.columns or close[t].dropna().empty]
    data = clean_prices(close.drop(columns=missing, errors="ignore"))
    data.source, data.missing = "Yahoo Finance (adjusted close)", missing + data.missing
    return data


def load_prices_csv(path_or_buffer) -> PriceData:
    """Read a wide CSV: first column is the date, every other column is a price series."""
    df = pd.read_csv(path_or_buffer)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    data = clean_prices(df)
    data.source = "Uploaded CSV"
    return data


def clean_prices(prices: pd.DataFrame, max_ffill: int = 5) -> PriceData:
    """Align series on common dates.

    Holiday gaps of up to `max_ffill` days are forward-filled (different exchanges close on
    different days); series that are entirely empty are dropped; the history starts on the
    first date on which every remaining series has a price.
    """
    notes: list[str] = []
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    empty = [c for c in prices.columns if prices[c].dropna().empty]
    prices = prices.drop(columns=empty)
    nonpos = [c for c in prices.columns if (prices[c].dropna() <= 0).any()]
    if nonpos:
        raise ValueError(f"Prices must be positive; check: {', '.join(map(str, nonpos))}")
    prices = prices.ffill(limit=max_ffill)
    first_full = prices.dropna().index.min()
    if first_full is None or pd.isna(first_full):
        raise ValueError("The series never overlap: there is no date on which every asset has a price.")
    if first_full > prices.index.min():
        notes.append(f"History starts on {first_full.date()}, the first date with prices for every asset.")
    prices = prices.loc[first_full:].dropna()
    if len(prices) < 60:
        raise ValueError("Fewer than 60 common trading days: not enough history for risk estimates.")
    return PriceData(prices=prices, source="", missing=empty, notes=notes)


def to_returns(prices: pd.DataFrame, kind: str = "simple") -> pd.DataFrame:
    """Daily simple or log returns. Simple returns aggregate correctly across assets."""
    if kind == "simple":
        return prices.pct_change().dropna(how="all")
    if kind == "log":
        return np.log(prices / prices.shift(1)).dropna(how="all")
    raise ValueError("kind must be 'simple' or 'log'")


# --------------------------------------------------------------------------- synthetic demo
SYNTHETIC_ASSETS = {
    # name: (annual drift, annual vol, asset class)
    "SYN_ITA_ENERGY":  (0.06, 0.26, "Equity"),
    "SYN_ITA_BANK":    (0.07, 0.34, "Equity"),
    "SYN_ITA_UTILITY": (0.05, 0.19, "Equity"),
    "SYN_ITA_INSUR":   (0.06, 0.22, "Equity"),
    "SYN_WORLD_EQ":    (0.08, 0.16, "Equity"),
    "SYN_US_EQ":       (0.09, 0.18, "Equity"),
    "SYN_GLOBAL_BOND": (0.015, 0.05, "Government bond"),
    "SYN_GOLD":        (0.04, 0.15, "Commodity"),
}
SYNTHETIC_PORTFOLIO = {"SYN_ITA_ENERGY": 0.12, "SYN_ITA_BANK": 0.12, "SYN_ITA_UTILITY": 0.08,
                       "SYN_ITA_INSUR": 0.08, "SYN_WORLD_EQ": 0.20, "SYN_US_EQ": 0.15,
                       "SYN_GLOBAL_BOND": 0.20, "SYN_GOLD": 0.05}

# Calendar windows in the synthetic history where a stress regime is injected, so that the
# backtests and stress tools have something to find. Dates match well-known real episodes
# purely to make the demo readable; the numbers inside them are simulated.
# (start, end, volatility multiplier, extra daily equity drift). Each sell-off is followed by
# a partial rebound so that the long-run path stays plausible.
_SYNTHETIC_CRISES = [("2015-08-17", "2015-09-29", 2.0, -0.003),
                     ("2015-09-30", "2015-12-31", 1.2, 0.0012),
                     ("2020-02-20", "2020-03-23", 3.2, -0.012),
                     ("2020-03-24", "2020-12-31", 1.4, 0.0022),
                     ("2022-01-03", "2022-10-12", 1.5, -0.0012),
                     ("2022-10-13", "2023-07-31", 1.1, 0.0009)]


def synthetic_prices(start: str = "2012-01-02", end: str = "2026-09-30", seed: int = 7) -> PriceData:
    """Generate realistic-looking but entirely simulated daily prices for the demo.

    Features: fat tails (multivariate Student-t, 5 degrees of freedom), volatility clustering
    (a GARCH(1,1)-style common variance factor), a fixed correlation structure, and injected
    stress regimes. It exists so the app and tests run without internet access. It is not
    market data and must never be presented as such.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n, names = len(dates), list(SYNTHETIC_ASSETS)
    k = len(names)
    corr = np.full((k, k), 0.55)
    eq = [i for i, a in enumerate(names) if SYNTHETIC_ASSETS[a][2] == "Equity"]
    ita = [names.index(a) for a in names if a.startswith("SYN_ITA")]
    for i in ita:
        for j in ita:
            corr[i, j] = 0.72
    us_world = [names.index("SYN_WORLD_EQ"), names.index("SYN_US_EQ")]
    corr[us_world[0], us_world[1]] = corr[us_world[1], us_world[0]] = 0.93
    b, g = names.index("SYN_GLOBAL_BOND"), names.index("SYN_GOLD")
    for i in range(k):
        corr[i, b] = corr[b, i] = -0.15 if i in eq else 0.25
        corr[i, g] = corr[g, i] = 0.05 if i in eq else 0.25
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)

    nu = 5.0
    z = rng.standard_normal((n, k)) @ chol.T
    w = rng.chisquare(nu, size=(n, 1)) / nu
    eps = z / np.sqrt(w) * np.sqrt((nu - 2) / nu)  # unit-variance multivariate t

    # GARCH(1,1)-style common variance multiplier with long-run mean 1
    alpha, beta = 0.08, 0.90
    omega = 1 - alpha - beta
    h = np.ones(n)
    for t in range(1, n):
        h[t] = omega + alpha * eps[t - 1, eq].mean() ** 2 / 0.6 + beta * h[t - 1]
    h = h / h.mean()

    drift = np.zeros(n)
    for s, e, vol_mult, d in _SYNTHETIC_CRISES:
        mask = (dates >= s) & (dates <= e)
        h[mask] *= vol_mult ** 2
        drift[mask] += d

    # the table gives compound (geometric) growth; add back the volatility drag
    vol = np.array([SYNTHETIC_ASSETS[a][1] for a in names])
    mu = (np.array([SYNTHETIC_ASSETS[a][0] for a in names]) + 0.5 * vol ** 2) / TRADING_DAYS
    sig = vol / np.sqrt(TRADING_DAYS)
    is_eq = np.array([i in eq for i in range(k)], dtype=float)
    is_bond = np.zeros(k); is_bond[b] = 1.0
    r = mu + np.sqrt(h)[:, None] * sig * eps + drift[:, None] * (is_eq - 0.15 * is_bond)
    r = np.clip(r, -0.45, 0.45)
    # Pin each asset's compound growth over the full sample to its target, so a single
    # unlucky random draw cannot make the demo portfolio lose money for 15 years. This shifts
    # every day's log return by the same constant and leaves volatility and tails unchanged.
    lr = np.log1p(r)
    target = np.log1p(np.array([SYNTHETIC_ASSETS[a][0] for a in names])) / TRADING_DAYS
    r = np.expm1(lr + (target - lr.mean(axis=0)))
    prices = pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=dates, columns=names)
    data = clean_prices(prices)
    data.source = "Synthetic demo data (simulated, not market prices)"
    data.notes.append("These prices are simulated for demonstration. They are not market data.")
    return data


def synthetic_asset_classes() -> dict[str, str]:
    return {a: v[2] for a, v in SYNTHETIC_ASSETS.items()}
