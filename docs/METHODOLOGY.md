# RiskLite methodology

Every figure in the app comes from one documented function in the `risklite` package. This
note gives the formulas, the assumptions behind them and where they break down.

## 1. Data and returns

* **Prices**: adjusted daily closes (dividends and splits included) from Yahoo Finance, an
  uploaded CSV, or simulated demo data.
* **Cleaning** (`data.clean_prices`): dates are sorted and de-duplicated. Holiday gaps of up
  to 5 days are forward-filled, since Milan, Xetra and New York close on different days.
  The sample starts on the first date on which every holding has a price, and at least 60
  common days are required.
* **Returns**: simple daily returns $r_t = P_t / P_{t-1} - 1$. Simple returns are used
  because they aggregate across assets: $r_{p,t} = \sum_i w_i r_{i,t}$.
* **Portfolio**: constant-mix, which means it is rebalanced to the target weights every day.
  This measures the risk of the *allocation* rather than of one particular drift path.

## 2. Performance statistics (`metrics`)

| Measure | Formula |
|---|---|
| Annualised return | $\left(\prod (1+r_t)\right)^{252/N} - 1$ (geometric) |
| Annualised volatility | $\sigma_{daily}\sqrt{252}$ |
| Sharpe | $(R - r_f)/\sigma$ |
| Sortino | $(R - r_f)/\sigma_{down}$, with downside deviation measured against the daily risk-free rate |
| Max drawdown | $\min_t \left(W_t / \max_{s\le t} W_s - 1\right)$, $W$ = wealth index |
| Calmar | $R / \lvert MDD \rvert$ |
| Beta | $\mathrm{Cov}(r_p, r_b)/\mathrm{Var}(r_b)$ |
| Tracking error | $\sigma(r_p - r_b)\sqrt{252}$ |
| Concentration | Herfindahl $H = \sum w_i^2$; effective number of positions $1/H$ |

## 3. Value at Risk and Expected Shortfall (`var`)

VaR at confidence $\alpha$ is the loss exceeded with probability $1-\alpha$. ES (also called
CVaR) is the average loss *given* that VaR is exceeded. Both are reported as positive
fractions of portfolio value. ES is a coherent risk measure (Acerbi and Tasche, 2002) and has
been the Basel FRTB standard since 2019.

| Method | VaR | ES |
|---|---|---|
| Historical | $-q_{1-\alpha}(r)$, the empirical quantile | $-\mathbb{E}[r \mid r \le q_{1-\alpha}]$ |
| Parametric (normal) | $-(\mu + \sigma z_{1-\alpha})$ | $\sigma\,\varphi(z_{1-\alpha})/(1-\alpha) - \mu$ |
| Cornish-Fisher | $-(\mu + \sigma z_{cf})$ | n/a |
| Monte Carlo | quantile of 20,000 simulated outcomes | mean of the simulated tail |

The mean $\mu$ is set to zero by default, the usual convention at short horizons where the
mean is small and noisily estimated.

**Cornish-Fisher** adjusts the normal quantile for sample skewness $s$ and excess kurtosis $k$:

$$z_{cf} = z + \frac{(z^2-1)s}{6} + \frac{(z^3-3z)k}{24} - \frac{(2z^3-5z)s^2}{36}$$

**Monte Carlo** resamples whole historical days (bootstrap) by default. This keeps the fat
tails and the cross-asset co-movement of each day. A multivariate-normal option exists for
comparison. Multi-day horizons compound $h$ independent draws.

**Horizon scaling**: the closed-form methods scale 1-day figures by $\sqrt{h}$. This
assumes i.i.d. returns and understates risk when volatility clusters.

## 4. Backtesting (`backtest`)

1. **Rolling forecasts.** Each day $t$, a 1-day VaR is estimated from the previous $W$ days
   (default 250), using only information up to $t-1$; the tests check that there is no look-ahead.
   * *Historical*: rolling empirical quantile.
   * *Parametric*: rolling standard deviation times $z$.
   * *EWMA* (RiskMetrics, 1996): $\sigma_t^2 = \lambda\sigma_{t-1}^2 + (1-\lambda)r_{t-1}^2$,
     with $\lambda = 0.94$. It reacts faster to new volatility regimes.
2. **Exceptions.** An exception is a day when $r_t < -\mathrm{VaR}_t$. With $x$ exceptions in
   $n$ days, the expected count is $n(1-\alpha)$.
3. **Kupiec (1995) proportion-of-failures test.** With $p = 1-\alpha$ and $\hat p = x/n$:
   $$LR_{POF} = -2\ln\frac{(1-p)^{n-x}p^x}{(1-\hat p)^{n-x}\hat p^x} \sim \chi^2_1$$
4. **Christoffersen (1998) independence test.** It compares the probability of an exception
   after a quiet day, $\pi_0 = n_{01}/(n_{00}+n_{01})$, with the probability after an
   exception day, $\pi_1 = n_{11}/(n_{10}+n_{11})$. Under independence the two are equal,
   and $LR_{ind} \sim \chi^2_1$. Clustered exceptions mean the model adapts too slowly.
   **Conditional coverage**: $LR_{cc} = LR_{POF} + LR_{ind} \sim \chi^2_2$.
5. **Basel traffic light** (Basel Committee, 1996). The zone comes from the cumulative
   binomial probability of the observed count:
   * green below 95%;
   * yellow from 95% to below 99.99%;
   * red at 99.99% or above.

   For 250 days at 99% this gives the familiar bands: 0–4 green, 5–9 yellow, 10 or more red.

## 5. Stress testing (`stress`)

* **Historical replay.** The current weights are applied to the actual daily returns of each
  window, and the result is compounded:
  * Global Financial Crisis: 12 Sep 2008 to 9 Mar 2009;
  * Euro sovereign crisis: 1 Jul to 25 Nov 2011;
  * COVID: 19 Feb to 23 Mar 2020;
  * 2022 rate shock: 3 Jan to 12 Oct 2022.

  A window counts as covered only if prices exist for at least 90% of its business days.
  Many UCITS ETFs launched after 2008, so a real portfolio often cannot replay the GFC. The
  app says so instead of filling the gap silently.
* **Hypothetical shocks.** Instantaneous moves by asset class, applied as
  $\Delta V = \sum_c w_c\, s_c$. The defaults are round illustrative numbers for discussion.
  They are not calibrated forecasts, and every one can be edited in the app.
* **Worst periods.** The worst non-overlapping rolling 1-, 5- and 21-day compounded returns.

## 6. Risk contributions (`contrib`)

With covariance matrix $\Sigma$, portfolio volatility is $\sigma_p = \sqrt{w^\top\Sigma w}$.
By Euler's theorem for homogeneous functions (Tasche, 2000):

$$\sigma_p = \sum_i w_i\frac{(\Sigma w)_i}{\sigma_p}, \qquad \text{share}_i = \frac{w_i(\Sigma w)_i}{\sigma_p^2}$$

Contributions add up exactly to the total, so each holding gets an additive share of risk.
The following follow from the same allocation:

* **Component VaR** (normal): $-z_{1-\alpha}\cdot w_i(\Sigma w)_i/\sigma_p$. It sums to
  parametric VaR.
* **Historical ES contribution**: $-\mathbb{E}[w_i r_{i,t} \mid r_{p,t} \le q_{1-\alpha}]$.
  It sums to historical ES and needs no normality assumption.
* **Diversification ratio**: $\sum_i w_i\sigma_i / \sigma_p$. It equals 1 when all holdings
  are perfectly correlated.
* **Covariance estimate**: either the sample estimate, or EWMA with $\lambda = 0.94$ and zero
  mean (RiskMetrics).

## 7. Fund look-through (`overlap`)

Exposure to an underlying name $k$ is

$$E_k = \sum_{f} w_f\, h_{f,k} + w_k^{direct}$$

where $h_{f,k}$ is the weight of name $k$ in fund $f$. Holdings beyond those supplied appear
as "(other holdings)", so total exposure always sums to 100%. Pairwise fund overlap is
$\sum_k \min(h_{a,k}, h_{b,k})$. The bundled sample holdings are illustrative top-holdings
lists only. Real analysis needs the issuer's full holdings file.

## 8. Limitations

* The past is a sample, not a forecast. A sample without a crisis understates crisis risk.
* Normal-based figures understate tail losses when returns are fat-tailed, which the app
  reports through excess kurtosis.
* $\sqrt{h}$ scaling and constant-mix weights are simplifications.
* Data quality depends on the source. Yahoo Finance data is free and generally reliable, but
  it is not audited.
* The demo data is simulated: a multivariate Student-t with 5 degrees of freedom, a
  GARCH-style common variance and injected stress windows. It is not market data.
* RiskLite describes historical risk. It does not give investment advice.

## References

* Acerbi, C. and Tasche, D. (2002). On the coherence of Expected Shortfall. *Journal of
  Banking & Finance* 26(7).
* Basel Committee on Banking Supervision (1996). *Supervisory framework for the use of
  "backtesting" in conjunction with the internal models approach to market risk capital
  requirements.*
* Christoffersen, P. (1998). Evaluating interval forecasts. *International Economic Review*
  39(4).
* J.P. Morgan/Reuters (1996). *RiskMetrics Technical Document*, 4th ed.
* Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement models.
  *Journal of Derivatives* 3(2).
* Tasche, D. (2000). Risk contributions and performance measurement. Working paper, TU München.
* Zangari, P. (1996). A VaR methodology for portfolios that include options. *RiskMetrics
  Monitor*, Q1 (Cornish-Fisher VaR).
