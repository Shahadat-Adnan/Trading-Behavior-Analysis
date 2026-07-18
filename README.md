# Trading Behavior Analysis — Market Sentiment vs Trader Performance

Analysis of ~211k crypto trades against the Crypto Fear & Greed Index, testing
whether trader behavior and profitability shift with market sentiment, then
using clustering and a classifier to profile traders and predict trade outcomes.

## Data

- `historical_data.csv` — raw trade-level data (account, coin, side, size,
  execution price, fee, closed PnL, timestamp). ~211,224 rows.
- `fear_greed_index.csv` — daily Fear & Greed Index value and classification
  (Extreme Fear, Fear, Neutral, Greed, Extreme Greed), 2018–present.

## Notebooks

Run in this order (same kernel session, or restart-and-run-all on each):

1. **`EDA.ipynb`** — loads and cleans both datasets, standardizes column names,
   parses timestamps, builds per-trader aggregate metrics (`trader_metrics`),
   and covers univariate/bivariate exploration (PnL distribution, trade
   side split, top assets, hourly activity, correlation heatmap).

2. **`quantitative behavioral analysis.ipynb`** — merges trades with daily
   sentiment, then runs the statistical testing and modeling:

   - Sentiment regime distribution and PnL/volume breakdown by regime
   - **One-way ANOVA** — mean `closed_pnl` across Fear / Greed / Extreme Greed / Neutral
   - **Kruskal-Wallis** — non-parametric check on the same groups
   - **Shapiro-Wilk** (sampled) + **Levene's test** — checks the normality
     and equal-variance assumptions ANOVA depends on
   - **Eta squared** + **Tukey HSD** — effect size and pairwise post-hoc
     comparison, since a significant ANOVA only says "some group differs,"
     not which one
   - **Pearson correlation** — Fear/Greed index value vs `closed_pnl`
   - **Welch's t-test** — trade size (`size_usd`) in Greed-regime trades vs
     Fear-regime trades
   - **Cohen's d** + **95% confidence interval** — effect size for the t-test above
   - **Chi-square test** + **Cramer's V** — win rate (profitable vs not) across
     sentiment regimes
   - **A/A test** — sanity check: splits one regime (Neutral) into two random
     halves and re-runs the t-test; used to confirm the testing pipeline
     doesn't throw false positives on its own
   - **K-Means clustering** (elbow method, 5 clusters) + PCA projection to
     profile trader types
   - **Random Forest classifier** predicting whether a trade closes profitable,
     with accuracy/precision/recall/F1, confusion matrix, and feature importance

## Key statistical findings

- Both **Shapiro-Wilk and Levene's test reject their null** (p ≈ 0 for both),
  meaning PnL is neither normally distributed nor equal-variance across
  regimes — this is why the Kruskal-Wallis result is the one to trust over
  the ANOVA p-value.
- The ANOVA/Tukey HSD result is statistically significant but the **eta
  squared is ~0.0003** — sentiment regime explains a negligible share of
  PnL variance. Statistically real, practically small.
- The Greed-vs-Fear trade size difference is significant (Welch's t-test)
  but **Cohen's d ≈ -0.07**, a small effect by conventional thresholds.
- Win rate does differ significantly by regime (chi-square), but **Cramer's
  V ≈ 0.06** — again a weak association in practical terms.
- The **A/A test came back non-significant (p ≈ 0.07)**, which is the
  expected result and validates that the testing pipeline isn't
  systematically producing false positives.

The overall takeaway the tests support: sentiment-PnL relationships in this
dataset are statistically detectable at this sample size, but the effect
sizes are consistently small — useful to state explicitly in a portfolio
write-up rather than over-claiming predictive power from the p-values alone.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scipy
statsmodels
scikit-learn
```
