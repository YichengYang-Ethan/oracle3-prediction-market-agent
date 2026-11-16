# Calibrated probabilities for Kalshi and Polymarket contracts: out-of-sample benchmark

Generated 2026-11-13T06:42:52Z from commit `79918237b2`; snapshot table SHA-256 `e880deea4583b185…`. Every number below is computed on test folds only.

## Data

349,297 resolved contracts (190,414 Kalshi, 158,883 Polymarket) in 112,401 events, 964,103 prediction snapshots at 25%, 50% and 75% of each contract's scheduled life (kept only while the market still traded). Prediction times span 2022-07-25T20:00:00Z to 2026-11-10T20:37:30Z.

| Venue | Contracts | Events | Snapshots | YES rate |
|---|---:|---:|---:|---:|
| kalshi | 190,414 | 59,158 | 518,418 | 0.321 |
| polymarket | 158,883 | 53,243 | 445,685 | 0.222 |

Features (all observed at or before the prediction time): `price`, `logit_price`, `extremity`, `log_hours_to_end`, `log_life_hours`, `elapsed_fraction`, `dp_life05`, `dp_life20`, `dp_1h`, `dp_24h`, `vol_logit_life20`, `hours_since_change`, `spread`, `log_volume_to_date`, `log_volume_life10`, `active_share_life10`, `is_kalshi`, `cat_sports`, `cat_politics`, `cat_economics`, `cat_finance`, `cat_crypto`, `cat_weather`, `cat_culture`, `cat_science_tech`, `cat_other`. Spread and volume features exist only for Kalshi; Polymarket rows carry them as missing.

## Validation design

- event-grouped purged walk-forward, expanding window: 5 folds. An event's contracts are never split between training and test; a training event must have resolved at least 24 hours before the test block begins (later-resolving events are purged).
- Inside each training window, 3 purged inner walk-forward blocks are each predicted by models fitted only on earlier training events. Hyperparameters are chosen by out-of-fold Brier over the inner walk-forward blocks; the three boosted-tree libraries early-stop on each inner block.
- The ensembles combine the eight learners: the mean of their recalibrated probabilities, and a logistic regression on their log-odds fitted to the same inner out-of-fold predictions.
- isotonic, fitted on the inner out-of-fold predictions; the chosen configuration is then refitted on the full training window.
- Intervals: 1000 event-cluster resamples, 95% percentile intervals.

| Fold | Test window (prediction time) | Train events | Test events | Test contracts | Test snapshots | Purged |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 2025-12-11 – 2026-04-24 | 17,176 | 18,734 | 57,090 | 153,917 | 1,558 |
| 2 | 2026-04-24 – 2026-07-15 | 34,940 | 18,734 | 63,317 | 172,954 | 2,528 |
| 3 | 2026-07-15 – 2026-08-31 | 53,866 | 18,733 | 61,614 | 175,752 | 2,336 |
| 4 | 2026-08-31 – 2026-10-07 | 71,370 | 18,733 | 62,170 | 173,258 | 3,565 |
| 5 | 2026-10-07 – 2026-11-08 | 90,653 | 18,733 | 57,473 | 157,134 | 3,015 |

## Out-of-sample scores (all test folds pooled)

| Method | Brier | Log loss | ECE (10 bins) | Brier vs market | 95% CI | ECE vs market | 95% CI | Folds better on Brier |
|---|---:|---:|---:|---:|---|---:|---|---:|
| Market price | 0.11851 | 0.3612 | 0.0413 | — | — | — | — | — |
| Market price + isotonic (price-only recalibration) | 0.11406 | 0.3509 | 0.0022 | +3.8% | [+3.6%, +3.9%] | +94.6% | [+91.8%, +96.0%] | 5/5 |
| Wang (paper), λ per venue | 0.11587 | 0.3552 | 0.0188 | +2.2% | [+2.1%, +2.4%] | +54.4% | [+52.5%, +56.1%] | 5/5 |
| Wang (paper), hierarchical λ | 0.11493 | 0.3531 | 0.0131 | +3.0% | [+2.8%, +3.2%] | +68.3% | [+66.3%, +70.3%] | 5/5 |
| Penalized logistic | 0.11453 | 0.3517 | 0.0080 | +3.4% | [+3.2%, +3.5%] | +80.7% | [+78.3%, +82.7%] | 5/5 |
| Spline GAM (penalized logistic on spline bases) | 0.11406 | 0.3508 | 0.0080 | +3.8% | [+3.6%, +3.9%] | +80.6% | [+78.1%, +82.7%] | 5/5 |
| Random forest | 0.11228 | 0.3458 | 0.0066 | +5.3% | [+5.1%, +5.5%] | +84.1% | [+81.5%, +85.5%] | 5/5 |
| LightGBM | 0.11200 | 0.3444 | 0.0060 | +5.5% | [+5.3%, +5.7%] | +85.5% | [+82.8%, +87.2%] | 5/5 |
| LightGBM, monotone in price | 0.11213 | 0.3449 | 0.0055 | +5.4% | [+5.2%, +5.6%] | +86.7% | [+84.1%, +88.5%] | 5/5 |
| XGBoost | 0.11210 | 0.3447 | 0.0057 | +5.4% | [+5.2%, +5.6%] | +86.2% | [+83.4%, +87.8%] | 5/5 |
| CatBoost | 0.11229 | 0.3453 | 0.0064 | +5.2% | [+5.0%, +5.5%] | +84.4% | [+82.2%, +86.0%] | 5/5 |
| MLP | 0.11310 | 0.3477 | 0.0089 | +4.6% | [+4.3%, +4.8%] | +78.5% | [+76.0%, +81.0%] | 5/5 |
| Ensemble: mean of the eight learners | 0.11211 | 0.3447 | 0.0071 | +5.4% | [+5.2%, +5.6%] | +82.7% | [+80.2%, +84.2%] | 5/5 |
| Ensemble: logistic stack of the eight learners | 0.11199 | 0.3442 | 0.0066 | +5.5% | [+5.3%, +5.7%] | +83.9% | [+81.3%, +85.0%] | 5/5 |

Positive percentages are reductions relative to the market price (lower Brier and ECE are better). Each learner's recalibration (isotonic or none) is chosen per fold on training data; both variants are in the recalibration table below.

**Best single learner: LightGBM.** Out-of-sample Brier 0.11200 vs 0.11851 for market prices (+5.5%, 95% CI [+5.3%, +5.7%]); ECE 0.0060 vs 0.0413 (+85.5%, 95% CI [+82.8%, +87.2%]).

**Best of all methods: Ensemble: logistic stack of the eight learners.** Out-of-sample Brier 0.11199 vs 0.11851 for market prices (+5.5%, 95% CI [+5.3%, +5.7%]); ECE 0.0066 vs 0.0413 (+83.9%, 95% CI [+81.3%, +85.0%]).

## Per fold (Brier)

| Fold | Market price | Market price + isotonic (price-only recalibration) | Wang (paper), hierarchical λ | LightGBM | Ensemble: logistic stack of the eight learners |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.12012 | 0.11683 | 0.11751 | 0.11466 | 0.11466 |
| 2 | 0.09913 | 0.09685 | 0.09668 | 0.09476 | 0.09481 |
| 3 | 0.12593 | 0.12135 | 0.12264 | 0.12057 | 0.12036 |
| 4 | 0.12717 | 0.12117 | 0.12273 | 0.11904 | 0.11921 |
| 5 | 0.12043 | 0.11431 | 0.11524 | 0.11103 | 0.11096 |

## Recalibration check (Brier, test folds)

Isotonic recalibration is kept for a learner in a fold only if, fitted on the earlier inner blocks of the training window, it lowers the Brier score of the later ones.

| Learner | No recalibration | Always isotonic | As chosen on training data | Folds that kept isotonic |
|---|---:|---:|---:|---:|
| Penalized logistic | 0.11465 | 0.11463 | 0.11453 | 2/5 |
| Spline GAM (penalized logistic on spline bases) | 0.11387 | 0.11418 | 0.11406 | 1/5 |
| Random forest | 0.11231 | 0.11228 | 0.11228 | 2/5 |
| LightGBM | 0.11202 | 0.11203 | 0.11200 | 2/5 |
| LightGBM, monotone in price | 0.11212 | 0.11217 | 0.11213 | 2/5 |
| XGBoost | 0.11207 | 0.11216 | 0.11210 | 2/5 |
| CatBoost | 0.11225 | 0.11233 | 0.11229 | 2/5 |
| MLP | 0.11338 | 0.11310 | 0.11310 | 3/5 |

## By platform (test folds)

| Platform | Snapshots | Market Brier | Wang Brier | LightGBM Brier | Market ECE | LightGBM ECE |
|---|---:|---:|---:|---:|---:|---:|
| kalshi | 450,306 | 0.11398 | 0.11330 | 0.10895 | 0.0247 | 0.0077 |
| polymarket | 382,709 | 0.12384 | 0.11889 | 0.11559 | 0.0616 | 0.0215 |

## By category (test folds)

| Category | Snapshots | Market Brier | Wang Brier | LightGBM Brier | Market ECE | LightGBM ECE |
|---|---:|---:|---:|---:|---:|---:|
| sports | 277,107 | 0.19063 | 0.18615 | 0.18144 | 0.0627 | 0.0139 |
| weather | 267,417 | 0.06972 | 0.06798 | 0.06563 | 0.0308 | 0.0087 |
| finance | 103,188 | 0.11347 | 0.11427 | 0.11202 | 0.0325 | 0.0114 |
| culture | 52,537 | 0.11043 | 0.10319 | 0.09296 | 0.0715 | 0.0083 |
| crypto | 51,778 | 0.05409 | 0.05172 | 0.04500 | 0.0307 | 0.0031 |
| politics | 35,902 | 0.09831 | 0.09730 | 0.09647 | 0.0262 | 0.0086 |
| economics | 35,028 | 0.07313 | 0.07220 | 0.06877 | 0.0354 | 0.0131 |
| science_tech | 9,881 | 0.08300 | 0.08267 | 0.08077 | 0.0255 | 0.0115 |
| other | 177 | 0.15353 | 0.15115 | 0.14922 | 0.0663 | 0.0440 |

## By liquidity (test folds)

| Liquidity | Snapshots | Market Brier | Wang Brier | LightGBM Brier | Market ECE | LightGBM ECE |
|---|---:|---:|---:|---:|---:|---:|
| polymarket: price moved in last 24h | 337,060 | 0.11710 | 0.11287 | 0.10998 | 0.0556 | 0.0207 |
| kalshi: spread <= 5c | 255,835 | 0.07558 | 0.07680 | 0.07586 | 0.0086 | 0.0044 |
| kalshi: spread > 20c | 99,738 | 0.20176 | 0.19359 | 0.17814 | 0.0932 | 0.0191 |
| kalshi: spread 5-20c | 94,733 | 0.12529 | 0.12732 | 0.12547 | 0.0165 | 0.0096 |
| polymarket: no move in last 24h | 45,649 | 0.17355 | 0.16333 | 0.15705 | 0.1059 | 0.0287 |

## Reliability (test folds, 10 equal-width bins)

| Bin | Market: mean price | outcome rate | n | LightGBM: mean prediction | outcome rate | n |
|---|---:|---:|---:|---:|---:|---:|
| 0.0–0.1 | 0.025 | 0.016 | 296,155 | 0.021 | 0.021 | 320,760 |
| 0.1–0.2 | 0.146 | 0.121 | 72,939 | 0.155 | 0.151 | 97,934 |
| 0.2–0.3 | 0.246 | 0.204 | 73,578 | 0.261 | 0.249 | 89,106 |
| 0.3–0.4 | 0.348 | 0.269 | 69,278 | 0.340 | 0.316 | 106,778 |
| 0.4–0.5 | 0.466 | 0.318 | 115,010 | 0.441 | 0.431 | 59,883 |
| 0.5–0.6 | 0.520 | 0.472 | 78,370 | 0.551 | 0.548 | 28,977 |
| 0.6–0.7 | 0.644 | 0.655 | 22,871 | 0.648 | 0.649 | 25,414 |
| 0.7–0.8 | 0.746 | 0.751 | 20,317 | 0.751 | 0.755 | 18,740 |
| 0.8–0.9 | 0.849 | 0.855 | 19,368 | 0.849 | 0.850 | 17,166 |
| 0.9–1.0 | 0.970 | 0.971 | 65,129 | 0.975 | 0.975 | 68,257 |

## Ablation: LightGBM on nested feature sets (same folds and procedure)

| Features | Brier vs market (no recalibration) | 95% CI | Brier vs market (isotonic) | 95% CI | ECE vs market (no recalibration) | ECE vs market (isotonic) |
|---|---:|---|---:|---|---:|---:|
| Price only (price, logit, extremity) | +3.6% | [+3.5%, +3.9%] | +3.5% | [+3.3%, +3.7%] | +90.9% | +87.9% |
| + venue and category | +4.0% | [+3.8%, +4.2%] | +3.9% | [+3.7%, +4.1%] | +80.8% | +87.4% |
| + time, price-path and liquidity features (all) | +5.5% | [+5.3%, +5.7%] | +5.5% | [+5.3%, +5.7%] | +80.5% | +84.2% |

Recalibrating the price alone recovers part of the gain; venue, category, timing, price-path and liquidity features add the rest. Always applying the isotonic map (fitted on inner out-of-fold predictions of models trained on less data) raised the Brier score in 3 of 3 rows.


## Served model

`model_probability` (MCP tool) serves LightGBM: among learners whose library is in the base `oracle3[ml]` install, those within 0.1% of the best Brier score count as tied and the smallest of them is served. It is refitted on all snapshots with the same inner tuning and recalibration check (recalibration: isotonic); parameters `{"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 100, "n_estimators": 229}`. File hashes are recorded in its manifest and checked on load.


## Limits

- Contracts were sampled per series (evenly over the window) and capped per category and per event, so the mix is not the venues' volume mix; scores are for this sample.
- Polymarket history is only available at 3-hour grain for older markets, so Polymarket contracts scheduled to live less than a day are excluded, and its spread and volume history are not public.
- Snapshots of one event (and of one contract) are correlated; intervals resample whole events.
- Most sampled events on both venues resolve in 2026, when far more were listed, so the pooled scores weigh 2026 most; Kalshi's public event listings also leave out events from before its move to `KX` tickers.
- Scheduled end dates are read as the venues report them after resolution; a date that was revised while the market traded would place that snapshot using the revised date.
- A lower Brier score than the market price is a statement about calibration on resolved contracts, not about tradable profit after fees, spreads and limited depth.
