# Calibrated probabilities: the machine-learning layer

`oracle3.ml` learns how resolved Kalshi and Polymarket contracts actually turned out given what was
observable while they traded, and serves that model to agents through the `model_probability`
MCP tool. The benchmark compares four learners against the market price and the probit-shift model
of [Yang (2026)](https://ssrn.com/abstract=6468338), out of sample.

**Results:** [evidence/ml_calibration/REPORT.md](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/blob/main/evidence/ml_calibration/REPORT.md)
(machine-readable: `results.json` in the same directory).

## Data

`python -m oracle3.ml.collect kalshi|polymarket --out DIR` downloads resolved contracts and their
price histories into a resumable cache. Every request is a read-only GET; Kalshi requests are signed
when `KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY_PATH` are set (or given in `--env-file`), which only
raises the rate limit.

- **Kalshi:** settled events of every series whose date falls in the window, up to six per series
  spread evenly over time, capped per category; up to four markets per event, chosen by a seeded hash
  of the ticker. Bars are the venue's candlesticks (best bid and ask, last trade, volume): one-minute
  bars for contracts scheduled to live up to two days, hourly up to 120 days, daily beyond.
- **Polymarket:** closed events from the Gamma API, sampled the same way; binary markets with a
  clean 1/0 payout. Price history comes from the Data API v2 at its finest permanent grain
  (3 hours), so contracts scheduled to live less than a day, and the auto-generated "Up or Down"
  series, are left out. Polymarket publishes no time-stamped spread or volume history.

Contracts are chosen before any price or outcome is looked at; lifetime volume, final prices and
results play no part in the sample.

## Snapshots and features

`python -m oracle3.ml.dataset --cache DIR --out snapshots.parquet` makes one row per contract at 25%,
50% and 75% of its *scheduled* life, kept only if the market was still trading then. Each row's
features use bars completed by the prediction time and nothing later:

| Feature | Meaning |
|---|---|
| `price`, `logit_price`, `extremity` | displayed price: bid/ask midpoint if the spread is at most 10¢, else last trade |
| `log_hours_to_end`, `log_life_hours`, `elapsed_fraction` | time left until and since the scheduled end and open |
| `dp_life05`, `dp_life20`, `dp_1h`, `dp_24h` | price change over the last 5% and 20% of the scheduled life, 1 hour and 24 hours |
| `vol_logit_life20`, `hours_since_change` | volatility of log-odds changes, hours since the price last moved |
| `spread`, `log_volume_to_date`, `log_volume_life10`, `active_share_life10` | Kalshi only: spread, traded volume to date and recently, share of recent bars with trades |
| `is_kalshi`, `cat_*` | venue and one of nine shared categories |

Leakage controls, each covered by a test in `tests/test_ml_layer.py`: a bar counts only after it has
ended; appending extreme future bars changes no feature; time features use the scheduled end, not the
actual close (which moves earlier when an outcome happens early); Polymarket's settlement tick, which
the API appends to resolved series, is removed.

## Validation

`python -m oracle3.ml.benchmark --snapshots S.parquet --evidence DIR --bundle DIR`

- **Event-grouped, purged walk-forward cross-validation.** Events are ordered by their first
  prediction time and cut into consecutive blocks; each fold tests on the next block and trains only on
  events whose outcomes were known at least 24 hours before it begins. An event's contracts never sit on
  both sides.
- **Tuning on training data only.** Inside each training window, three purged inner walk-forward blocks
  are each predicted by models fitted on earlier training events; hyperparameters are chosen by the
  Brier score of those out-of-fold predictions, and LightGBM early-stops on them.
- **Isotonic recalibration** is fitted to the same inner out-of-fold predictions, then applied to the
  test predictions of the model refitted on the whole training window.
- **Learners:** L2-penalized logistic regression, LightGBM, random forest, a two-layer MLP.
  **Baselines:** the market price, and the paper's model (`WangMLE`), both one λ per venue and the
  hierarchical form with the paper's covariates.
- **Scores** are computed on test folds only: Brier score, log loss and expected calibration error
  (10 equal-width bins), with 95% intervals from resampling whole events.

## Serving

The best learner is refitted on all snapshots in the same way and saved with its isotonic map and a
manifest of features, hyperparameters, data range, out-of-sample scores and file hashes (checked on
load). `model_probability(venue, market_id)` rebuilds the features for an open market from public data
at call time and returns the model probability next to the market price:

```json
{"model_probability": 0.231, "market_price": 0.25, "model_minus_market": -0.019,
 "model": "...", "out_of_sample": {"vs_market": {"brier_reduction_pct": "..."}}, "features": {"...": "..."}}
```

Install the extra with `pip install 'oracle3[ml]'`. Set `ORACLE3_ML_BUNDLE` to serve a bundle you
trained yourself.

The output is a calibrated probability estimate. It does not account for fees, spreads, depth or
resolution-rule details, and a better-calibrated probability is not by itself a profitable trade.
