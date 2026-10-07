# AGENTS.md

Instructions for AI coding agents working in this repository. Humans may find it useful too.

## What this project is

Oracle3 is a trading engine (live on Kalshi, Polymarket and Solana, or on paper) and an MCP server for prediction markets. It checks whether prices of related event contracts violate probability bounds after each venue's fees, and paper-trades the baskets that survive under pre-trade risk limits. See [README.md](README.md).

## Setup and checks

```bash
poetry install --all-extras                     # Python 3.10+
poetry run pytest                               # tests (coverage report included)
ruff check . && ruff format .                   # lint and format (single quotes, 88 columns)
poetry run mypy --config-file pyproject.toml .  # type check
codespell -c pyproject.toml                     # spelling
```

CI runs all four on every push to `main`. New code needs tests; venue APIs must be mocked in tests (see `tests/test_mcp_server.py` for an `httpx.MockTransport` example).

## Layout

| Path | Contents |
|---|---|
| `oracle3/fees.py` | Kalshi and Polymarket fee schedules as published; per-market schedules from the venue APIs |
| `oracle3/arbitrage.py` | Static no-arbitrage checks: implication, exclusivity, complement, same event, event sum |
| `oracle3/mcp_server/` | MCP server (`oracle3 mcp` or `oracle3-mcp`), public venue adapters, paper ledger |
| `oracle3/strategy/` | Strategies; `contrib/` holds the constraint, statistical and model-driven ones |
| `oracle3/pricing/` | Wang-transform fair-value engine and estimators |
| `oracle3/market/` | Relation store and statistical validation of relations |
| `oracle3/trader/` | Paper and live traders, spread executor |
| `oracle3/cli/` | Click CLI (`oracle3 ...`) |
| `oracle3/experimental/` | Prototypes that are not on the trading path |
| `coinjure/` | Bundled package from the original U Lab codebase (MIT; see NOTICE) |
| `skills/` | Agent skills, mirrored to `.claude/skills/` |
| `docs/` | MkDocs site, including `docs/research/fee-frontier.md` |

Integrations with third-party services (wallets, data providers, venues) and experimental features belong in [oracle3-extras](https://github.com/YichengYang-Ethan/oracle3-extras), which mirrors these namespaces. Propose them there rather than here.

## Safety rules

- **Default to paper.** Never run `oracle3 live run` or any command that places real orders unless the user explicitly asks for live trading in this session.
- **Commands that only read data** (safe to run): `oracle3 market ...`, `oracle3 news fetch`, `oracle3 trade status|state`, and every MCP tool except the `paper_*` ones. `oracle3 research ...` reads local history files and writes only to output paths you pass.
- **Commands with local side effects:** `oracle3 paper run`, `oracle3 data record`, `oracle3 trade pause|resume|swap|stop|killswitch`, and the MCP `paper_order`/`paper_reset` tools (local JSON ledger only).
- **Order-placing commands:** `oracle3 live run` and the live traders in `oracle3/trader/`. The MCP server imports none of them.
- Never print, commit or log credentials (`KALSHI_*`, `POLYMARKET_*`, Solana keypairs).

## Conventions

- English only in code, comments, docs and commit messages.
- Fees come from `oracle3.fees`; do not hard-code new fee constants.
- Do not describe paper or backtest results as live performance, and do not add performance claims to the README without a reproducible script.
- Keep `server.json`, `pyproject.toml`, `oracle3/__init__.py` and `CITATION.cff` versions in sync (commitizen bumps all four).
