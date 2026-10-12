"""RiskLite: portfolio risk analytics for small advisors and individual investors.

The package is split into small modules so that every number shown in the app can be
traced back to one documented function:

    data      loading prices (Yahoo Finance, CSV, synthetic demo data) and returns
    metrics   performance, drawdown and benchmark-relative statistics
    var       Value at Risk and Expected Shortfall (historical, parametric,
              Cornish-Fisher, Monte Carlo)
    backtest  rolling VaR forecasts, Kupiec and Christoffersen tests, Basel zones
    stress    historical scenario replay, hypothetical shocks, worst periods
    contrib   covariance (sample / EWMA) and risk contributions
    overlap   fund look-through and pairwise overlap
    report    a single function that runs the full analysis
"""

__version__ = "1.0.0"
