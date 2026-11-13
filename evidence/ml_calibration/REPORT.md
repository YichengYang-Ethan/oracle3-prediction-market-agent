# Calibrated probabilities for Kalshi and Polymarket contracts: out-of-sample benchmark

Generated 2026-11-13T00:02:22Z from commit `251ec0c24f`; snapshot table SHA-256 `996aed429af7fe93…`. Every number below is computed on test folds only.

## Data

41,505 resolved contracts (22,163 Kalshi, 19,342 Polymarket) in 14,126 events, 111,488 prediction snapshots at 25%, 50% and 75% of each contract's scheduled life (kept only while the market still traded). Prediction times span 2022-07-25T20:00:00Z to 2026-11-10T20:37:30Z.

| Venue | Contracts | Events | Snapshots | YES rate |
|---|---:|---:|---:|---:|
| kalshi | 22,163 | 6,644 | 57,943 | 0.340 |
| polymarket | 19,342 | 7,482 | 53,545 | 0.246 |

Features (all observed at or before the prediction time): `price`, `logit_price`, `extremity`, `log_hours_to_end`, `log_life_hours`, `elapsed_fraction`, `dp_life05`, `dp_life20`, `dp_1h`, `dp_24h`, `vol_logit_life20`, `hours_since_change`, `spread`, `log_volume_to_date`, `log_volume_life10`, `active_share_life10`, `is_kalshi`, `cat_sports`, `cat_politics`, `cat_economics`, `cat_finance`, `cat_crypto`, `cat_weather`, `cat_culture`, `cat_science_tech`, `cat_other`. Spread and volume features exist only for Kalshi; Polymarket rows carry them as missing.

## Validation design

- event-grouped purged walk-forward, expanding window: 5 folds. An event's contracts are never split between training and test; a training event must have resolved at least 24 hours before the test block begins (later-resolving events are purged).
- Inside each training window, 3 purged inner walk-forward blocks are each predicted by models fitted only on earlier training events. Hyperparameters are chosen by out-of-fold Brier over the inner walk-forward blocks; LightGBM early-stops on each inner block.
- isotonic, fitted on the inner out-of-fold predictions; the chosen configuration is then refitted on the full training window.
- Intervals: 1000 event-cluster resamples, 95% percentile intervals.

| Fold | Test window (prediction time) | Train events | Test events | Test contracts | Test snapshots | Purged |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 2026-01-09 – 2026-04-08 | 1,889 | 2,355 | 6,442 | 17,448 | 466 |
| 2 | 2026-04-08 – 2026-06-21 | 4,022 | 2,354 | 6,979 | 18,548 | 688 |
| 3 | 2026-06-21 – 2026-08-21 | 6,138 | 2,354 | 7,249 | 20,185 | 926 |
| 4 | 2026-08-21 – 2026-10-02 | 8,375 | 2,354 | 7,697 | 20,406 | 1,043 |
| 5 | 2026-10-02 – 2026-11-08 | 10,898 | 2,354 | 7,613 | 19,901 | 874 |

## Out-of-sample scores (all test folds pooled)

| Method | Brier | Log loss | ECE (10 bins) | Brier vs market | 95% CI | ECE vs market | 95% CI | Folds better on Brier |
|---|---:|---:|---:|---:|---|---:|---|---:|
| Market price | 0.11765 | 0.3647 | 0.0416 | — | — | — | — | — |
| Wang (paper), λ per venue | 0.11488 | 0.3585 | 0.0187 | +2.4% | [+1.9%, +2.8%] | +54.9% | [+47.3%, +61.0%] | 5/5 |
| Wang (paper), hierarchical λ | 0.11416 | 0.3573 | 0.0160 | +3.0% | [+2.3%, +3.6%] | +61.5% | [+53.5%, +67.5%] | 5/5 |
| Penalized logistic + isotonic | 0.11385 | 0.3622 | 0.0109 | +3.2% | [+2.5%, +3.9%] | +73.8% | [+63.9%, +78.7%] | 4/5 |
| LightGBM + isotonic | 0.11229 | 0.3570 | 0.0100 | +4.6% | [+3.8%, +5.2%] | +76.0% | [+69.4%, +82.3%] | 5/5 |
| Random forest + isotonic | 0.11351 | 0.3579 | 0.0129 | +3.5% | [+2.7%, +4.3%] | +68.9% | [+60.4%, +75.6%] | 4/5 |
| MLP + isotonic | 0.11521 | 0.3631 | 0.0048 | +2.1% | [+1.2%, +2.9%] | +88.4% | [+78.3%, +91.2%] | 4/5 |
| Penalized logistic (no recalibration) | 0.11327 | 0.3554 | 0.0106 | +3.7% | [+3.0%, +4.3%] | +74.6% | [+65.3%, +79.8%] | 5/5 |
| LightGBM (no recalibration) | 0.11176 | 0.3510 | 0.0051 | +5.0% | [+4.3%, +5.7%] | +87.7% | [+77.8%, +90.0%] | 5/5 |
| Random forest (no recalibration) | 0.11205 | 0.3526 | 0.0072 | +4.8% | [+4.0%, +5.4%] | +82.6% | [+73.8%, +86.9%] | 5/5 |
| MLP (no recalibration) | 0.11502 | 0.3622 | 0.0152 | +2.2% | [+1.4%, +3.0%] | +63.3% | [+56.1%, +70.5%] | 4/5 |

Positive percentages are reductions relative to the market price (lower Brier and ECE are better).

**Best learner: LightGBM + isotonic.** Out-of-sample Brier 0.11229 vs 0.11765 for market prices (+4.6%, 95% CI [+3.8%, +5.2%]); ECE 0.0100 vs 0.0416 (+76.0%, 95% CI [+69.4%, +82.3%]).

## Per fold (Brier)

| Fold | Market price | Wang (paper), λ per venue | Wang (paper), hierarchical λ | Penalized logistic + isotonic | LightGBM + isotonic | Random forest + isotonic | MLP + isotonic |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.11220 | 0.11050 | 0.11034 | 0.11373 | 0.11073 | 0.11342 | 0.11493 |
| 2 | 0.11231 | 0.11063 | 0.11110 | 0.10947 | 0.10954 | 0.11195 | 0.10971 |
| 3 | 0.11546 | 0.11346 | 0.11207 | 0.11181 | 0.11108 | 0.11130 | 0.11331 |
| 4 | 0.12126 | 0.11809 | 0.11796 | 0.11652 | 0.11420 | 0.11464 | 0.12089 |
| 5 | 0.12591 | 0.12082 | 0.11860 | 0.11738 | 0.11547 | 0.11612 | 0.11671 |

## By platform (test folds)

| Platform | Snapshots | Market Brier | Wang Brier | LightGBM + isotonic Brier | Market ECE | LightGBM + isotonic ECE |
|---|---:|---:|---:|---:|---:|---:|
| kalshi | 51,786 | 0.11476 | 0.11438 | 0.11241 | 0.0256 | 0.0152 |
| polymarket | 44,702 | 0.12099 | 0.11545 | 0.11214 | 0.0636 | 0.0158 |

## By category (test folds)

| Category | Snapshots | Market Brier | Wang Brier | LightGBM + isotonic Brier | Market ECE | LightGBM + isotonic ECE |
|---|---:|---:|---:|---:|---:|---:|
| sports | 19,403 | 0.17782 | 0.16996 | 0.16383 | 0.0822 | 0.0106 |
| finance | 18,279 | 0.12737 | 0.12991 | 0.12827 | 0.0246 | 0.0238 |
| weather | 14,596 | 0.08150 | 0.07608 | 0.07384 | 0.0591 | 0.0228 |
| economics | 12,713 | 0.09965 | 0.09913 | 0.09823 | 0.0267 | 0.0139 |
| culture | 12,076 | 0.10601 | 0.10087 | 0.09733 | 0.0561 | 0.0150 |
| politics | 10,744 | 0.09830 | 0.09871 | 0.09768 | 0.0220 | 0.0109 |
| science_tech | 4,867 | 0.09604 | 0.09651 | 0.09864 | 0.0261 | 0.0166 |
| crypto | 3,663 | 0.07837 | 0.07295 | 0.06922 | 0.0498 | 0.0198 |
| other | 147 | 0.17640 | 0.17446 | 0.17480 | 0.0748 | 0.0609 |

## By liquidity (test folds)

| Liquidity | Snapshots | Market Brier | Wang Brier | LightGBM + isotonic Brier | Market ECE | LightGBM + isotonic ECE |
|---|---:|---:|---:|---:|---:|---:|
| polymarket: price moved in last 24h | 37,644 | 0.12031 | 0.11539 | 0.11247 | 0.0608 | 0.0149 |
| kalshi: spread <= 5c | 25,461 | 0.07156 | 0.07308 | 0.07271 | 0.0117 | 0.0124 |
| kalshi: spread 5-20c | 14,094 | 0.11956 | 0.12105 | 0.12164 | 0.0229 | 0.0182 |
| kalshi: spread > 20c | 12,231 | 0.19915 | 0.19268 | 0.18441 | 0.0880 | 0.0281 |
| polymarket: no move in last 24h | 7,058 | 0.12465 | 0.11578 | 0.11041 | 0.0839 | 0.0273 |

## Reliability (test folds, 10 equal-width bins)

| Bin | Market: mean price | outcome rate | n | LightGBM + isotonic: mean prediction | outcome rate | n |
|---|---:|---:|---:|---:|---:|---:|
| 0.0–0.1 | 0.030 | 0.021 | 32,585 | 0.027 | 0.028 | 35,191 |
| 0.1–0.2 | 0.144 | 0.113 | 9,215 | 0.142 | 0.134 | 11,499 |
| 0.2–0.3 | 0.246 | 0.188 | 7,997 | 0.254 | 0.240 | 14,994 |
| 0.3–0.4 | 0.347 | 0.266 | 7,197 | 0.335 | 0.356 | 8,926 |
| 0.4–0.5 | 0.464 | 0.302 | 12,163 | 0.456 | 0.481 | 3,794 |
| 0.5–0.6 | 0.525 | 0.480 | 7,960 | 0.557 | 0.610 | 2,546 |
| 0.6–0.7 | 0.645 | 0.650 | 3,270 | 0.650 | 0.637 | 4,739 |
| 0.7–0.8 | 0.746 | 0.750 | 3,370 | 0.756 | 0.778 | 2,330 |
| 0.8–0.9 | 0.849 | 0.850 | 3,703 | 0.854 | 0.839 | 3,564 |
| 0.9–1.0 | 0.962 | 0.959 | 9,028 | 0.963 | 0.967 | 8,905 |

## Ablation: LightGBM on nested feature sets (same folds and procedure)

| Features | Brier vs market (no recalibration) | 95% CI | Brier vs market (isotonic) | 95% CI | ECE vs market (no recalibration) | ECE vs market (isotonic) |
|---|---:|---|---:|---|---:|---:|
| Price only (price, logit, extremity) | +3.3% | [+2.6%, +3.8%] | +1.7% | [+0.9%, +2.5%] | +81.4% | +64.5% |
| + venue and category | +4.0% | [+3.4%, +4.6%] | +2.4% | [+1.6%, +3.2%] | +82.3% | +67.6% |
| + time, price-path and liquidity features (all) | +5.0% | [+4.3%, +5.7%] | +4.6% | [+3.8%, +5.2%] | +87.7% | +76.0% |

Recalibrating the price alone recovers part of the gain; venue, category, timing, price-path and liquidity features add the rest. The isotonic step, fitted on inner out-of-fold predictions of models trained on less data, lowered the scores slightly in every row.


## Served model

`model_probability` (MCP tool) serves LightGBM + isotonic, refitted on all snapshots with the same inner tuning and isotonic step; parameters `{"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 100, "n_estimators": 157}`. File hashes are recorded in its manifest and checked on load.


## Limits

- Contracts were sampled per series (evenly over the window) and capped per category and per event, so the mix is not the venues' volume mix; scores are for this sample.
- Polymarket history is only available at 3-hour grain for older markets, so Polymarket contracts scheduled to live less than a day are excluded, and its spread and volume history are not public.
- Snapshots of one event (and of one contract) are correlated; intervals resample whole events.
- Scheduled end dates are read as the venues report them after resolution; a date that was revised while the market traded would place that snapshot using the revised date.
- A lower Brier score than the market price is a statement about calibration on resolved contracts, not about tradable profit after fees, spreads and limited depth.
