# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `oracle3.ml`, a machine-learning layer for calibrated probabilities: resumable read-only collection of resolved Kalshi and Polymarket contracts with price histories, leakage-safe snapshot features (displayed price, time to the scheduled end, recent price changes and volatility, Kalshi spread and volume, category), event-grouped purged walk-forward cross-validation with tuning and isotonic recalibration on training folds only, and a benchmark of penalized logistic regression, LightGBM, random forest and an MLP against market prices and the paper's Wang model. Results are in `evidence/ml_calibration/` and [docs/ml.md](docs/ml.md).
- MCP tool `model_probability` (14 tools in total): the served model's calibrated probability for an open market next to its market price, with its out-of-sample scores. Requires the new `ml` extra: `pip install 'oracle3[ml]'`.

## [1.2.2] - 2026-09-30

### Changed

- Every MCP tool now documents each of its parameters in the input schema (37 of 37, previously none) and says what it does, when to use it instead of a similar tool, whether it calls the network or writes local files, and what it returns. Tool behavior is unchanged.

### Added

- `glama.json` maintainer metadata for the Glama MCP directory.
- `oracle3 calibrate --csv FILE`: offline Wang Transform MLE from a resolved-contract CSV, with platform warm starts, optional hierarchical covariates and JSON output (#53, thanks @xyjk0511).
- `oracle3 dashboard --snapshot-json PATH --snapshot-interval SECONDS`: periodic, atomically written JSON snapshots of positions, cash and last prices (#55, thanks @MayurK-cmd).
- Documentation of the Solana pre-flight risk flow: local limits, `simulateTransaction`, Jito fallback and failure modes (#56, thanks @lntutor).

## [1.2.1] - 2026-09-28

### Fixed

- The MCP server failed to start in fresh installs: `openai-agents` now allows `mcp` 2.x, where `FastMCP` was renamed `MCPServer`. The server supports both SDK versions and a CI job tests against the newest SDK.
- Tool errors (venue failures, out-of-range inputs, unsupported fee schedules) are raised as `ToolError`, so their messages reach the agent under mcp 2.x instead of a generic "Error executing tool".
- `oracle3 --version` printed a hard-coded 1.0.0; it now reports the package version.
- `mcp` (>=1.19,<3) is now a core dependency instead of arriving only through `openai-agents`. The short-lived `mcp` extra is gone; `pip install oracle3` includes the server.
- If a client cached 1.2.0, run `uvx --refresh oracle3 mcp` once to pick up the fix.

## [1.2.0] - 2026-09-28

### Added

- **MCP server** (`oracle3 mcp`, `oracle3-mcp`) with 13 tools: market search, market details with resolution rules, order books, quotes with each market's fee schedule, offline and live no-arbitrage checks, fee calculator, Wang-transform fair value, local relation store, and a local paper-trading ledger. No tool can place a real order. Listed in the MCP Registry as `io.github.YichengYang-Ethan/oracle3` via `server.json`.
- **`oracle3.fees`**: Kalshi (effective July 7, 2026) and Polymarket taker and maker fee schedules as published, with per-market schedules read from each venue's API.
- **`oracle3.arbitrage`**: static checks for implication, exclusivity, complement, same-event and event-sum relations, returning the cheapest guaranteed-payoff basket with per-leg fees and the edge before and after fees.
- **Research note** `docs/research/fee-frontier.md` and `scripts/fee_frontier.py`: break-even violations under the published fee schedules.
- `AGENTS.md`, `llms.txt`, `codemeta.json`, a `pm-constraint-arbitrage` agent skill, and a docs page for the MCP server.

### Changed

- Repository renamed to `oracle3-prediction-market-agent`; the PyPI package and import name stay `oracle3`.
- Project description unified across the README, PyPI, GitHub, `CITATION.cff`, `.zenodo.json` and the docs site.
- `oracle3 market list/search/info` read Kalshi's `*_dollars` price fields and Polymarket's search endpoint. Kalshi prices had been returning 0 since the move to dollar fields, and Polymarket search only scanned the top 100 events.
- Agent skills translated to English and checked against the current CLI (`research features`, `research labels`, `research discover-alpha`, `trade get-state` and `trade swap-strategy` no longer exist).
- `run_full_simulation.py` translated to English; its docstring now states which inputs are simulated.
- Strategy comments no longer call the flat 0.005 per-side fee "conservative"; it is below the venues' taker fees at mid prices.
- **Documentation accuracy pass** — corrected claims to match the code exactly:
  - "peer-reviewed pricing model" → the Wang Transform (Wang 2000) calibrated in the author's working paper (Yang 2026, SSRN). The calibration is an unrefereed working paper; only the transform itself is from the peer-reviewed Wang (2000).
  - Strategy count restated as **6 constraint-based + 2 statistical-arbitrage + 2 model-driven** (10 total), replacing the ambiguous "8 constraint-based arbitrage strategies" phrasing in the README, paper, `CITATION.cff`, and `.zenodo.json`.
  - Removed an inaccurate "vectorized NumPy/PyTorch module" description of the pricing engine from the preprint — the deployed path is dependency-light pure Python; NumPy/SciPy are used only by the optional batch MLE estimator.
  - Softened `SpreadExecutor` ("being wired into the multi-leg strategies") and MEV-protection ("best-effort, with public-RPC fallback") descriptions to match their implemented status.
  - Corrected the hierarchical `D` covariate description from "days remaining to expiry" to total contract duration (hours), matching the estimated model.
- Added a verified / not-demonstrated section to the README (paper-traded status, fee economics, roadmap).

### Fixed

- **Removed fabricated profit accounting** from the flash-loan prototype, which previously booked a placeholder `amount * 0.005` profit on a Memo-only transaction.

### Moved

- Relocated the flash-loan arbitrage prototype to `oracle3/experimental/` and labeled the multi-agent pipeline and on-chain reputation modules as non-production prototypes.

## [1.1.0] - 2026-03-28

### Added

- **Wang Transform MLE** (`oracle3.pricing.wang_mle`): full maximum likelihood estimator implementing the core model from Yang (2026), "Pricing Prediction Markets: Risk Premiums, Incomplete Markets, and a Decomposition Framework"
  - Pooled and hierarchical estimation with analytic gradients
  - Three SE estimators: Fisher, sandwich robust, Liang-Zeger clustered sandwich
  - Numerical Hessian with eigenvalue regularization and BLAS chunk-size safety
  - Model comparison (LR tests, AIC/BIC, pseudo-R²)
  - Empirical priors: Polymarket λ=0.166, Kalshi=0.187, Metaculus=0.287, Manifold=-0.218

- **Probabilistic fair value engine** (`oracle3.pricing`): full pricing module grounded in Yang (2026)
  - Three distortion families: probit (Wang 2000), dual power (Denneberg 1994), proportional hazard
  - **Exact empirical coefficients**: λ_i = 0.259 - 0.072·ln(1+V) + 0.143·ln(1+D) - 0.477·|p-0.5|
  - **Time-varying model**: γ₁=-0.156·τ + γ₂=0.074·τ², half-life 33-77% of contract lifetime
  - **Volume-stratified alpha targeting**: >$10K volume → λ≈0 (premium competed away); $500-$10K = sweet spot
  - Hybrid calibrator: batch MLE (historical data) + streaming EWMA (live trading) with hierarchical shrinkage
  - Premium lifecycle tracker with polynomial decay fitting and optimal entry timing
  - Contract microstructure scorer (volume, spread, duration, extremity, book depth)

- **Model sensitivities (Greeks)** (`oracle3.pricing.greeks`): analytic derivatives of the Wang model
  - dp/dλ, dp/dp*, premium ratio, Kelly fraction, edge decay rate
  - Favorite-longshot bias proven as theorem: overpricing ratio monotonically decreasing in p*
  - Batch computation for portfolio-level risk decomposition

- **Model-informed Kelly sizing** (`oracle3.trading.sizing`): position sizing with Wang-derived edge
  - Kelly criterion with model edge, confidence scaling, and volume-tier gating
  - Automatic skip of very-high-volume markets where premium is already competed away
  - Inspired by three-tier sizing approaches in agent-native prediction market systems

- **Edge-weighted capital allocator** (`oracle3.trading.allocator`): multi-strategy budget allocation
  - Risk-adjusted scoring (PnL / |drawdown|) with 30-day exponential time decay
  - Premium-alpha strategy bonus, reserve capital, per-strategy caps
  - Graceful degradation: performance-weighted → equal → minimum budgets

- **Correlation-aware risk manager** (`oracle3.risk.correlation_risk_manager`): correlated exposure limits
  - EWMA rolling correlation estimation on price returns
  - Effective exposure via correlation matrix quadratic form
  - Concentration ratio gating, stale correlation decay

- **Fair value divergence strategy v2** (`FairValueStrategy`): model-driven alpha with exact coefficients
  - Uses Yang (2026) hierarchical model for per-contract λ estimation
  - Kelly-optimal sizing from model Greeks
  - Volume-tier targeting: focuses on medium-liquidity alpha sweet spot

- **Premium decay strategy** (`PremiumDecayStrategy`): timing-based premium lifecycle alpha

- **Distortion-based validation tools**: `estimate_risk_premium`, `cross_platform_premium_test`, `favorite_longshot_test` in `oracle3.market.validation`

- **79 new tests** (total: 633) covering MLE recovery, Greeks, sizing, allocation, strategies

## [1.0.0] - 2026-03-09

### Added

- **8 constraint-based & statistical arbitrage strategies**: cross-market, exclusivity, implication, conditional, event-sum, structural, cointegration spread, and lead-lag — each with formal invariant, fee-aware edge, cooldown windows, and audit trail
- **Market relation graph**: persistent knowledge graph (`~/.oracle3/relations.json`) with lifecycle management (discovered → validated → deployed → retired) and quantitative validation (Engle-Granger cointegration, ADF stationarity, OLS hedge ratio, OU half-life, Pearson correlation, lead-lag detection)
- **SpreadExecutor**: safe multi-leg execution with automatic LIFO unwind on partial fills — no naked positions
- **Engine control server**: Unix socket runtime control (pause/resume/stop/killswitch) without process restart
- **Strategy portfolio registry**: lifecycle tracking (paper → live → retired), health checks, Kelly capital allocation
- **8 on-chain agent capabilities**: cross-market arbitrage, on-chain risk manager, on-chain signal source, MEV protection (Jito), agent reputation, multi-agent pipeline, flash loan arbitrage, atomic multi-leg trader
- **AI-powered trading** with OpenAI Agents SDK, LiteLLM multi-provider support, and 8 built-in agent tools
- **Solana integration**: native transaction signing, on-chain trade logging via Memo program, Jito bundle submission, Solana Blinks
- **Multi-exchange support**: Solana/DFlow (SPL tokens), Polymarket (CLOB API), Kalshi (REST API)
- **Live trading dashboard** at `/live` with 8 feature cards, equity chart, execution pipeline animation, and pause/resume/e-stop controls
- **Classic terminal dashboard** at `/` for headless environments
- **Risk management**: dual-layer validation (local limits + Solana `simulateTransaction`), max drawdown monitoring, daily loss limits, kill switch
- **Backtesting engine** with DFlow episode replay (parquet format)
- **Coinjure matching pipeline**: cross-platform market relation discovery (implication, exclusivity, complementary) with resolution filter, volume filter, keyphrase pre-filter, confidence sizing, and tag coverage
- **CLI** (`oracle3`) with commands for market browsing, paper/live trading, engine control, reputation, blinks, trade logs
- **CI/CD**: pytest (553 tests), ruff, mypy, codespell, MkDocs documentation site
- **Interactive demo script** (`demo.sh`)

[1.0.0]: https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/releases/tag/v1.0.0
