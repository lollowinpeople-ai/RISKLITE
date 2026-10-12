"""Checks every risk calculation against values computed by hand or from closed-form results."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from risklite import backtest as bt
from risklite import contrib, metrics, overlap, stress
from risklite import var as v
from risklite.data import (SYNTHETIC_PORTFOLIO, clean_prices, normalise_weights, synthetic_prices,
                           to_returns)
from risklite.report import run_analysis


@pytest.fixture(scope="module")
def demo():
    data = synthetic_prices()
    return data.prices, to_returns(data.prices)


def series(vals, start="2020-01-01"):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)), dtype=float)


# ------------------------------------------------------------------ data
def test_normalise_weights_rescales_and_drops_zeros():
    w = normalise_weights({"A": 60, "B": 40, "C": 0, "": 5})
    assert w == pytest.approx({"A": 0.6, "B": 0.4})


def test_normalise_weights_rejects_empty():
    with pytest.raises(ValueError):
        normalise_weights({"A": 0})


def test_clean_prices_starts_at_first_full_date():
    idx = pd.bdate_range("2020-01-01", periods=100)
    df = pd.DataFrame({"A": np.linspace(100, 110, 100), "B": np.linspace(50, 60, 100)}, index=idx)
    df.iloc[:10, 1] = np.nan
    out = clean_prices(df)
    assert out.prices.index[0] == idx[10] and out.notes


def test_clean_prices_rejects_negative():
    idx = pd.bdate_range("2020-01-01", periods=100)
    with pytest.raises(ValueError):
        clean_prices(pd.DataFrame({"A": np.r_[-1, np.ones(99)]}, index=idx))


# ------------------------------------------------------------------ metrics
def test_portfolio_return_is_weighted_sum():
    r = pd.DataFrame({"A": [0.01, -0.02], "B": [0.03, 0.01]})
    p = metrics.portfolio_returns(r, {"A": 0.25, "B": 0.75})
    assert p.tolist() == pytest.approx([0.025, 0.0025])


def test_max_drawdown_known_path():
    # wealth: 1.1, 1.21, 0.605, 0.6655, 1.331 -> peak 1.21, trough 0.605 (-50%), recovered day 5
    r = series([0.10, 0.10, -0.50, 0.10, 1.00])
    dd = metrics.max_drawdown(r)
    assert dd.max_drawdown == pytest.approx(-0.5)
    assert dd.peak == r.index[1] and dd.trough == r.index[2] and dd.recovery == r.index[4]


def test_annualised_return_of_constant_growth():
    r = series([0.0002] * 252)
    assert metrics.annualised_return(r) == pytest.approx(1.0002 ** 252 - 1)


def test_beta_of_scaled_series_is_scale():
    rng = np.random.default_rng(0)
    b = series(rng.normal(0, 0.01, 500))
    assert metrics.benchmark_stats(1.5 * b, b)["beta"] == pytest.approx(1.5)


def test_concentration_equal_weights():
    c = metrics.concentration({k: 0.25 for k in "ABCD"})
    assert c["hhi"] == pytest.approx(0.25) and c["effective_positions"] == pytest.approx(4)


# ------------------------------------------------------------------ VaR / ES
def test_historical_var_and_es_on_uniform_losses():
    r = series(-np.arange(1, 101) / 1000)  # losses 0.1% .. 10%
    assert v.historical_var(r, 0.95) == pytest.approx(-np.quantile(r, 0.05))
    tail = r[r <= np.quantile(r, 0.05)]
    assert v.historical_es(r, 0.95) == pytest.approx(-tail.mean())
    assert v.historical_es(r, 0.95) >= v.historical_var(r, 0.95)


def test_parametric_var_es_match_closed_form():
    rng = np.random.default_rng(1)
    r = series(rng.normal(0, 0.01, 2000))
    s = r.std(ddof=1)
    assert v.parametric_var(r, 0.99) == pytest.approx(2.326348 * s, rel=1e-5)
    assert v.parametric_es(r, 0.99) == pytest.approx(s * stats.norm.pdf(2.326348) / 0.01, rel=1e-5)


def test_sqrt_time_scaling():
    rng = np.random.default_rng(2)
    r = series(rng.normal(0, 0.01, 1000))
    assert v.parametric_var(r, 0.99, 10) == pytest.approx(np.sqrt(10) * v.parametric_var(r, 0.99))


def test_cornish_fisher_equals_normal_without_skew_or_kurtosis():
    # a symmetric sample with exactly normal quantiles has ~zero skew and excess kurtosis
    x = stats.norm.ppf((np.arange(1, 20001) - 0.5) / 20000) * 0.01
    r = series(x)
    assert v.cornish_fisher_var(r, 0.99) == pytest.approx(v.parametric_var(r, 0.99), rel=0.02)


def test_monte_carlo_normal_close_to_parametric(demo):
    _, rets = demo
    w = SYNTHETIC_PORTFOLIO
    p = metrics.portfolio_returns(rets, w)
    mc_var, mc_es = v.monte_carlo_var_es(rets, w, 0.99, method="normal", n_sims=200_000, seed=3)
    assert mc_var == pytest.approx(v.parametric_var(p, 0.99), rel=0.03)
    assert mc_es > mc_var


def test_alpha_validation():
    with pytest.raises(ValueError):
        v.historical_var(series([0.01, -0.01]), 99)


# ------------------------------------------------------------------ backtest
def test_kupiec_zero_lr_when_rate_matches():
    lr, p = bt.kupiec_pof(1000, 10, 0.99)
    assert lr == pytest.approx(0.0, abs=1e-9) and p == pytest.approx(1.0)


def test_kupiec_known_value():
    # x=20, n=1000, p=0.01: LR = -2[980 ln .99 + 20 ln .01 - 980 ln .98 - 20 ln .02]
    expected = -2 * (980 * np.log(.99) + 20 * np.log(.01) - 980 * np.log(.98) - 20 * np.log(.02))
    lr, p = bt.kupiec_pof(1000, 20, 0.99)
    assert lr == pytest.approx(expected) and p < 0.01


def test_kupiec_zero_exceptions():
    lr, _ = bt.kupiec_pof(250, 0, 0.99)
    assert lr == pytest.approx(-2 * 250 * np.log(0.99))


def test_christoffersen_detects_clustering():
    clustered = pd.Series([0] * 480 + [1] * 10 + [0] * 510).astype(bool)
    scattered = pd.Series(([0] * 99 + [1]) * 10).astype(bool)
    assert bt.christoffersen_independence(clustered)[1] < 0.01
    assert bt.christoffersen_independence(scattered)[1] > 0.05


def test_traffic_light_basel_bands():
    assert [bt.traffic_light(250, x, 0.99) for x in (4, 5, 9, 10)] == ["Green", "Yellow", "Yellow", "Red"]


def test_rolling_var_has_no_lookahead():
    rng = np.random.default_rng(4)
    r = series(rng.normal(0, 0.01, 400))
    f1 = bt.rolling_var(r, 250, 0.99, "historical")
    r2 = r.copy(); r2.iloc[-1] = -0.5  # a shock on the last day must not change that day's forecast
    f2 = bt.rolling_var(r2, 250, 0.99, "historical")
    assert f1.iloc[-1] == f2.iloc[-1]
    assert f1.iloc[-1] == pytest.approx(-np.quantile(r.iloc[-251:-1], 0.01))


def test_ewma_forecast_uses_only_past():
    rng = np.random.default_rng(5)
    r = series(rng.normal(0, 0.01, 400))
    r2 = r.copy(); r2.iloc[-1] = -0.5
    assert bt.rolling_var(r, method="ewma").iloc[-1] == bt.rolling_var(r2, method="ewma").iloc[-1]


def test_backtest_on_true_normal_model_passes():
    rng = np.random.default_rng(6)
    r = series(rng.normal(0, 0.01, 3000))
    res = bt.backtest(r, 0.99, 250, "parametric")
    assert res.kupiec_p > 0.05 and res.zone in ("Green", "Yellow")


# ------------------------------------------------------------------ contributions
def test_euler_contributions_sum_to_total(demo):
    _, rets = demo
    cov = contrib.sample_cov(rets)
    rc = contrib.euler_vol_contributions(SYNTHETIC_PORTFOLIO, cov)
    p = metrics.portfolio_returns(rets, SYNTHETIC_PORTFOLIO)
    assert rc["Vol contribution"].sum() == pytest.approx(metrics.annualised_vol(p), rel=1e-6)
    assert rc["Share of risk"].sum() == pytest.approx(1.0)


def test_component_var_sums_to_parametric_var(demo):
    _, rets = demo
    cv = contrib.component_var(SYNTHETIC_PORTFOLIO, contrib.sample_cov(rets), 0.99)
    p = metrics.portfolio_returns(rets, SYNTHETIC_PORTFOLIO)
    assert cv.sum() == pytest.approx(v.parametric_var(p, 0.99), rel=1e-6)


def test_es_contributions_sum_to_historical_es(demo):
    _, rets = demo
    esc = contrib.historical_es_contributions(rets, SYNTHETIC_PORTFOLIO, 0.99)
    p = metrics.portfolio_returns(rets, SYNTHETIC_PORTFOLIO)
    assert esc.sum() == pytest.approx(v.historical_es(p, 0.99), rel=1e-9)


def test_ewma_cov_weights_recent_data():
    a = np.r_[np.full(500, 0.001), np.full(50, 0.03)]
    df = pd.DataFrame({"A": a, "B": a})
    assert contrib.ewma_cov(df).iloc[0, 0] > contrib.sample_cov(df).iloc[0, 0]


def test_diversification_ratio_is_one_for_perfect_correlation():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 0.01, 500)
    df = pd.DataFrame({"A": x, "B": 2 * x})
    assert contrib.diversification_ratio({"A": .5, "B": .5}, df.cov()) == pytest.approx(1.0)


# ------------------------------------------------------------------ stress and overlap
def test_replay_compounds_returns():
    idx = pd.bdate_range("2020-02-19", "2020-03-23")
    rets = pd.DataFrame({"A": -0.01, "B": 0.0}, index=idx)
    res = stress.replay_scenario(rets, {"A": .5, "B": .5}, "x", "2020-02-19", "2020-03-23")
    assert res.covered and res.total_return == pytest.approx(0.995 ** len(idx) - 1)


def test_replay_flags_missing_history():
    rets = pd.DataFrame({"A": [0.0] * 10}, index=pd.bdate_range("2021-01-01", periods=10))
    assert not stress.replay_scenario(rets, {"A": 1}, "GFC", "2008-09-12", "2009-03-09").covered


def test_hypothetical_shock_is_weighted_sum():
    out = stress.hypothetical_stress({"E": .6, "B": .4}, {"E": "Equity", "B": "Government bond"},
                                     {"s": {"Equity": -0.3, "Government bond": 0.05}})
    assert out.loc["s", "Portfolio impact"] == pytest.approx(0.6 * -0.3 + 0.4 * 0.05)


def test_worst_periods_non_overlapping():
    r = series([0.0] * 10 + [-0.05, -0.04] + [0.0] * 10 + [-0.03] + [0.0] * 10)
    wp = stress.worst_periods(r, lengths=(1,), top=3)
    assert wp["Return"].tolist() == pytest.approx([-0.05, -0.04, -0.03])


def test_look_through_and_overlap():
    h = pd.DataFrame({"fund": ["F1", "F1", "F2", "F2"], "holding": ["Apple", "Nvidia", "Apple", "Eni"],
                      "weight": [0.5, 0.5, 0.2, 0.8]})
    lt = overlap.look_through({"F1": 0.5, "F2": 0.25, "Apple": 0.25}, h)
    assert lt.loc["Apple", "Exposure"] == pytest.approx(0.5 * 0.5 + 0.25 * 0.2 + 0.25)
    assert lt["Exposure"].sum() == pytest.approx(1.0)
    assert lt.loc["Apple", "Sources"] == 3
    assert overlap.pairwise_overlap(h).loc["F1", "F2"] == pytest.approx(0.2)


def test_holdings_csv_accepts_percentages(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("fund,holding,weight\nF,A,60\nF,B,40\n")
    assert overlap.read_holdings_csv(p)["weight"].tolist() == pytest.approx([0.6, 0.4])


# ------------------------------------------------------------------ end to end
def test_full_analysis_runs(demo):
    prices, _ = demo
    a = run_analysis(prices, SYNTHETIC_PORTFOLIO, n_sims=5000)
    assert set(a.backtests) == {"historical", "parametric", "ewma"}
    assert a.var_table.loc["Historical", "ES"] >= a.var_table.loc["Historical", "VaR"]
    assert any(s.covered for s in a.historical_stress) and a.findings
