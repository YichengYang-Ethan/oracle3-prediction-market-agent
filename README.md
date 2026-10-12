<!-- mcp-name: io.github.YichengYang-Ethan/oracle3 -->

# Oracle3

**Build prediction-market agents that discover markets, analyze opportunities, and test strategies.**

Connect AI agents to Kalshi and Polymarket through MCP. Give them reusable Skills for market research and strategy development, then build on a Python engine for paper and live execution.

**13 MCP tools · 6 agent Skills · Kalshi & Polymarket data · Python 3.10+**

[![Tests](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/actions/workflows/pytest.yml/badge.svg)](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/actions/workflows/pytest.yml)
[![GitHub stars](https://img.shields.io/github/stars/YichengYang-Ethan/oracle3-prediction-market-agent?style=social)](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/stargazers)
[![PyPI](https://img.shields.io/pypi/v/oracle3)](https://pypi.org/project/oracle3/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20062548.svg)](https://doi.org/10.5281/zenodo.20062548)

[Quickstart](#quickstart) · [MCP tools](docs/mcp.md) · [Agent Skills](#reusable-agent-skills) · [Project and contributions](docs/research-review.md) · [Documentation](https://yichengyang-ethan.github.io/oracle3-prediction-market-agent/)

## Why Oracle3

- **A connected research workflow.** Discover contracts, inspect their rules and order books, compare related outcomes, and track simulated positions.
- **Calculations that include trading costs.** Five relationship checks return candidate baskets, per-leg fees, and calculated net edge.
- **Tools and workflows for agents.** Use MCP for structured access and Skills for repeatable research, strategy authoring, and operations.
- **An engine you can extend.** Build strategies with Python interfaces, risk controls, and paper/live venue adapters.
- **A foundation for experiments.** Reproducible software checks, automated tests, and installed-package validation make the implementation inspectable.

Public Kalshi and Polymarket market reads need no exchange credentials. MCP provides market data, calculations, and a local paper ledger; real-order execution uses the separate CLI/Python engine.

## Quickstart

Install the current development checkout:

```bash
git clone https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent.git
cd oracle3-prediction-market-agent
python -m venv .venv
.venv/bin/python -m pip install -e .
```

Connect it to Claude Code from this directory:

```bash
claude mcp add oracle3 -- "$PWD/.venv/bin/oracle3-mcp"
```

For other MCP clients, configure the absolute path to `.venv/bin/oracle3-mcp` as the command with an empty argument list. On Windows, use `.venv\Scripts\python.exe` and `.venv\Scripts\oracle3-mcp.exe`.

[Client configuration and first tool call](docs/agent-quickstart.md) · [Published-package installation](docs/mcp.md)

### Give your agent a first task

> Find Kalshi and Polymarket contracts about the next Fed rate decision. Compare their outcome definitions, resolution rules, and order-book quotes. Summarize the markets with their IDs, prices, and the assumptions needed to compare them.

For a calculation that works offline after installation:

> Use Oracle3 to check an implication: A implies B. A has a YES bid of $0.60 and B has a YES ask of $0.55. Use the default Kalshi taker fee schedule and 100 contracts per leg. Show the basket, gross edge, fees, and net edge.

The synthetic example returns **$5.00 gross edge − $3.4125 fees = $1.5875 calculated net edge**, conditional on the supplied implication and quotes. [See exact inputs and output](docs/agent-quickstart.md#first-call-a-deterministic-fee-check).

## What agents can do

| Task | Tools | Result |
|---|---|---|
| Discover markets | `search_markets`, `get_market` | Contracts, metadata, and available resolution rules |
| Inspect market conditions | `get_quote`, `get_orderbook` | Prices, fee information, and displayed depth |
| Analyze related contracts | `check_constraint`, `check_constraint_live` | Baskets and net edge under a stated relationship |
| Calculate fees and model values | `trading_fee`, `fair_value` | Fee estimates and parameterized model outputs |
| Try a paper entry | `paper_order`, `paper_portfolio` | Simulated fills and holdings in a local ledger |

Relationship checks cover **implication, exclusivity, complement, same event, and event sum**. They expose assumptions with their results, giving agents and researchers a common calculation layer.

[All 13 MCP tools](docs/mcp.md) · [Workflow examples](docs/agent-quickstart.md)

## Reusable agent Skills

The repository includes six workflows that help agents move from individual tool calls to complete tasks:

| Skill | Use it to |
|---|---|
| [Data discovery](skills/pm-data-discovery/SKILL.md) | Find markets and organize research inputs |
| [Constraint arbitrage](skills/pm-constraint-arbitrage/SKILL.md) | Examine related contracts and calculate fee-aware baskets |
| [Quant strategy authoring](skills/pm-quant-strategy-authoring/SKILL.md) | Develop strategies with Oracle3's Python interfaces |
| [Agent strategy authoring](skills/pm-agent-strategy-authoring/SKILL.md) | Build agent-driven decision workflows |
| [Paper operations](skills/pm-paper-trade-ops/SKILL.md) | Run and inspect simulated trading sessions |
| [Live operations](skills/pm-live-trade-ops/SKILL.md) | Configure and operate the separate live engine |

Claude Code project copies are included in [`.claude/skills/`](.claude/skills/). Open the checkout as your project, or use your client's supported Skill installation mechanism. [Skill setup](docs/agent-quickstart.md#agent-skills).

**For AI agents:** start with [`llms.txt`](llms.txt) for task routing, the [MCP reference](docs/mcp.md) for tool schemas, and [`AGENTS.md`](AGENTS.md) for contributing code.

## Build with Python and the CLI

Use the same project for research scripts, agent strategies, and engine operations:

```bash
.venv/bin/oracle3 market search --exchange kalshi --query "fed" --json
.venv/bin/oracle3 market search --exchange polymarket --query "fed decision" --json
```

| Component | Build with it |
|---|---|
| [MCP server](oracle3/mcp_server/) | Structured market access and local paper operations for agents |
| [Relations and fees](oracle3/arbitrage.py) | Shared calculations for related-contract strategies |
| [Strategies](oracle3/strategy/) | Logical-relation, statistical, and model-driven approaches |
| [Trading engine](oracle3/trader/) | Paper/live adapters for Kalshi, Polymarket, and Solana via DFlow |
| [Pricing](oracle3/pricing/) | Wang-transform models and estimators for research |
| [oracle3-extras](https://github.com/YichengYang-Ethan/oracle3-extras) | Optional venue, data-provider, and wallet integrations |

[CLI guide](docs/CLI_MONITORING.md) · [Integration guide](docs/extras.md) · [Execution reference](docs/runtime-integrity.md)

## Built for experimentation

Oracle3 combines a concrete decision environment with structured tools, observable state, and a software validation harness. This supports work on agent interfaces, workflow guidance, and reproducible evaluation.

[CI](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/actions/workflows/pytest.yml) runs automated tests and checks the built package through real MCP sessions on SDK 1.x and 2.x. The [harness documentation](docs/agent-harness.md) explains the reproducible checks. Model-level agent evaluation is a next research direction.

[Project overview and Yicheng Yang's contributions](docs/research-review.md) · [Fee-frontier study](docs/research/fee-frontier.md) · [Companion pricing research](https://github.com/YichengYang-Ethan/prediction-market-pricing)

## Contribute

Help build better agent workflows, labeled evaluation tasks, strategy examples, and client integrations. New venue, data-provider, and wallet integrations start in [oracle3-extras](https://github.com/YichengYang-Ethan/oracle3-extras).

[Contributing guide](.github/CONTRIBUTING.md) · [Open problems](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/issues?q=is%3Aissue+label%3Aopen-problem) · [Discussions](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/discussions)

## Origin and attribution

Oracle3 began as `ulab-uiuc/oracle3`, developed by Yicheng Yang and Haofei Yu at U Lab (University of Illinois Urbana-Champaign). It also bundles the lab's [Coinjure](https://github.com/ulab-uiuc/coinjure) package. Yicheng Yang continues to develop and maintain the extensions in this repository. [Read the project story and contributions](docs/research-review.md).

We acknowledge the foundational work of the Coinjure team at the University of Illinois Urbana-Champaign—Haofei Yu, Yicheng Yang, Yuxiang Liu, and Prof. Jiaxuan You—on which Oracle3 builds.

For software citation, use [CITATION.cff](CITATION.cff) and [Zenodo](https://doi.org/10.5281/zenodo.20062548). For the pricing model, see the [working paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6468338).

## License

Apache 2.0; see [LICENSE](LICENSE). Original U Lab and Coinjure portions retain their MIT license in [NOTICE](NOTICE).

*This software is for research and education. Trading involves financial risk.*
