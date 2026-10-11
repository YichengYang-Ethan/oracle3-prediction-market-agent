<!-- mcp-name: io.github.YichengYang-Ethan/oracle3 -->

# Oracle3

**Agent tools and execution infrastructure for prediction markets.**

Give your agent Kalshi and Polymarket market data, fee-aware checks on related contracts, and a local paper-trading ledger through MCP. Build strategies with the Python API and trading engine, and replay tool errors and incomplete execution with an offline contract harness.

[![Tests](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/actions/workflows/pytest.yml/badge.svg)](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/actions/workflows/pytest.yml)
[![PyPI](https://img.shields.io/pypi/v/oracle3)](https://pypi.org/project/oracle3/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20062548.svg)](https://doi.org/10.5281/zenodo.20062548)

[Agent quickstart](docs/agent-quickstart.md) · [MCP tools](docs/mcp.md) · [Documentation](https://yichengyang-ethan.github.io/oracle3-prediction-market-agent/) · [Machine-readable index](llms.txt) · [Contributing](.github/CONTRIBUTING.md)

## When to use Oracle3

| Your task | Start here | What you get |
|---|---|---|
| Find Kalshi or Polymarket contracts and inspect their metadata | `search_markets` → `get_market` | Market identifiers, prices and available rule fields |
| Check whether a related-contract price gap survives fees | `check_constraint` or `check_constraint_live` | Candidate baskets, per-leg fees, gross/net edge and calculation assumptions |
| Simulate a trade with current order-book data | `paper_order` → `paper_portfolio` | Fills and positions in a local JSON ledger |
| Test how a tool or executor handles failure | [Contract harness](docs/agent-harness.md) | Frozen cases for errors, partial fills, recovery and unresolved exposure |
| Build a strategy or operate a trading engine | [Python modules](#architecture-and-extensions), [CLI](#cli-and-agent-skills) | Strategy interfaces, paper/live adapters and risk controls |

Public MCP market reads need no exchange credentials. Pure calculations and the contract harness run without network access after installation. MCP tools do not place real orders; live trading is a separate CLI/Python workflow.

## Connect an agent

Requires Python 3.10+ and an MCP client. With [uv](https://docs.astral.sh/uv/getting-started/installation/) installed, add the **released PyPI package** to Claude Code:

```bash
claude mcp add oracle3 -- uvx --from oracle3 oracle3-mcp
```

For clients supporting an `mcpServers` configuration:

```json
{
  "mcpServers": {
    "oracle3": { "command": "uvx", "args": ["--from", "oracle3", "oracle3-mcp"] }
  }
}
```

The client launches the stdio server. Running `oracle3 mcp` manually waits for protocol input; it is not an interactive trading prompt. If you use pip instead, install `oracle3` in a virtual environment and point your client at that environment's `oracle3-mcp` executable.

**First task:** ask your agent to call `list_relation_types`, then calculate an `implication` check for synthetic contracts A and B: A has YES bid 0.60, B has YES ask 0.55, both use the default Kalshi schedule, and the size is 100 contracts. Request the basket, fees, net edge and assumptions. This uses no market API or ledger. See the [exact tool arguments and expected result](docs/agent-quickstart.md).

**Release versus main:** the install above uses a published package. This README also describes development code: the explicit `assessment` fields and offline harness were added after the 1.2.2 release. Use the checkout below for those examples; installing a release does not install the repository's `scripts/` or `skills/` directories. The [changelog](CHANGELOG.md) records release history.

## Try a reproducible check

Install the development checkout in a virtual environment:

```bash
git clone https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent.git
cd oracle3-prediction-market-agent
python -m venv .venv
.venv/bin/python -m pip install -e .
```

On Windows, use `.venv\Scripts\python.exe`. To run this checkout through MCP, configure the absolute path to `.venv/bin/oracle3-mcp` (`.venv\Scripts\oracle3-mcp.exe` on Windows), with an empty `args` list. See [client setup](docs/agent-quickstart.md).

Run the following with the environment's Python. A and B are synthetic identifiers; the implication is an input assumption.

```python
import json

from oracle3.arbitrage import Quote, check_constraint
from oracle3.fees import KalshiSchedule

result = check_constraint(
    'implication',
    [
        Quote('A', yes_bid=0.60, schedule=KalshiSchedule()),
        Quote('B', yes_ask=0.55, schedule=KalshiSchedule()),
    ],
    contracts=100,
).to_dict()

best = result['best']
assert best is not None
print(json.dumps({
    'basket': best['description'],
    'gross_edge': best['gross_edge'],
    'fees': best['fees'],
    'net_edge': best['net_edge'],
    'assessment': result['assessment'],
}, indent=2))
```

Expected output, in dollars for 100 contracts per leg:

```json
{
  "basket": "NO on A + YES on B",
  "gross_edge": 5.0,
  "fees": 3.4125,
  "net_edge": 1.5875,
  "assessment": {
    "relation_verified": false,
    "execution_verified": false,
    "conditional_on_relation": true
  }
}
```

The positive edge is conditional on the supplied relationship and full fills at the supplied prices. It is a worked calculation, not an observed trading return. A missing executable quote can produce `best: null`; callers must inspect `missing_quotes` before using a basket.

## From market discovery to an informed decision

For a live-data research task, an agent can:

1. **Discover:** call `search_markets` on Kalshi and Polymarket for candidate contracts.
2. **Read the rules:** call `get_market` for each candidate. Kalshi responses include rule fields; the current Polymarket adapter does not expose full resolution text, so obtain it from the venue's market page. Compare outcome definitions, dates, settlement sources and any void/cancellation terms. If the rules are unavailable, leave the relationship unverified.
3. **State the relationship:** choose `implication`, `exclusivity`, `complement`, `same_event` or `event_sum`, with the evidence supporting that choice. Event sums require mutually exclusive outcomes covering every possible result.
4. **Calculate:** call `check_constraint_live` for current quotes and fee schedules. Inspect `best`, `missing_quotes`, per-leg costs and net edge; use `get_orderbook` to inspect available size.
5. **Report:** return market IDs, the assumed relationship, the quoted prices, fees, size and unresolved limitations. A positive check does not verify settlement equivalence or execution.

Paper trading is a separate, explicit next step: `paper_order` writes to a local ledger using live book data; `paper_portfolio` reads it. The MCP ledger is separate from the trading engine's paper state. `paper_reset` erases that ledger and requires `confirm=true`.

[All 13 MCP tools and arguments](docs/mcp.md) · [Agent task recipes](docs/agent-quickstart.md)

## Reliability you can inspect

The agent-facing contract harness runs nine frozen scenarios through the real MCP dispatcher, constraint checker and sequential spread executor, using mocked venue responses and a scripted trader.

| Scenario | Behavior checked |
|---|---|
| Invalid input or venue error | An error remains visible to the caller |
| A thin price gap | Fees can remove the apparent opportunity |
| An unverified relationship | The output preserves conditional assumptions |
| Partial execution | Incomplete fills do not count as a completed basket |
| Failed or uncertain recovery | Residual exposure and reconciliation requirements remain visible |

From the checkout above:

```bash
.venv/bin/python scripts/check_agent_contracts.py --output agent-contracts.json
```

The runner blocks socket connections, needs no credentials, writes JSON evidence, and exits nonzero on a failed case. [CI](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/actions/workflows/pytest.yml) runs it with the locked dependencies and latest supported MCP SDK and uploads the reports. The wider suite covers fees, constraint calculations, venue adapters and execution behavior; see the [tests](tests/).

This is deterministic regression coverage for the tool environment. LLM reasoning, Skill effectiveness and trading profitability require separate evaluations. `SpreadExecutor` submits sequential orders, exposes fills and recovery attempts, and retains uncertainty; it does not provide atomic multi-venue transactions or durable recovery after process loss. Its integration into the strategy collection is still in progress.

[Case definitions, output contract and migration notes](docs/agent-harness.md)

## CLI and agent skills

After installing the package, query markets from scripts or a terminal:

```bash
oracle3 market search --exchange kalshi --query "fed" --json
oracle3 market search --exchange polymarket --query "fed decision" --json
```

The CLI also operates the paper/live engine and inspects running sessions. See [command access and side effects](AGENTS.md) and the [CLI monitoring guide](docs/CLI_MONITORING.md). Live trading requires venue credentials, a configured strategy and explicit user authorization; it is not part of the MCP quickstart.

Six [repository skills](skills/) cover constraint checks, data discovery, quantitative strategy authoring, agent strategy authoring, paper operations and live operations. Claude Code project copies are included in [`.claude/skills/`](.claude/skills/); open the checkout as your project to use them. For another client, use its supported skill installation mechanism or load the selected `SKILL.md` as workflow context. [Choose a skill and set it up](docs/agent-quickstart.md#agent-skills).

## Architecture and extensions

| Component | Responsibility | Current boundary |
|---|---|---|
| [`oracle3.mcp_server`](oracle3/mcp_server/) | Public Kalshi/Polymarket data, calculations and local paper ledger | No real-money order tool; Solana is not an MCP venue |
| [`oracle3.arbitrage`](oracle3/arbitrage.py) + [`oracle3.fees`](oracle3/fees.py) | Five relationship checks with per-leg fee schedules | The caller supplies the relation; checks do not prove rules, depth or fills |
| [`oracle3.strategy`](oracle3/strategy/) + [`oracle3.trader`](oracle3/trader/) | Strategy interfaces, paper/live traders for Kalshi, Polymarket and Solana (DFlow), spread execution | Fee unification and spread-executor integration across strategies are incomplete |
| [`oracle3.pricing`](oracle3/pricing/) | Optional Wang-transform models and estimators | Model-implied probabilities depend on assumptions; a fitted relation is not a logical constraint |
| [`oracle3-extras`](https://github.com/YichengYang-Ethan/oracle3-extras) | Optional venue/data/wallet integrations and experiments | Separate installation and configuration; matched-market candidates still need rule review |

The strategy collection includes logical-relation, fitted/statistical and model-driven approaches. Their assumptions and integration status differ. The pricing module's independent-forecast comparison is described in the [migration notes](docs/agent-harness.md#model-comparisons); the underlying research is in the [working paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6468338) and [replication package](https://github.com/YichengYang-Ethan/prediction-market-pricing).

Fees are central to the checks: a visible price gap may disappear after both legs' costs. The [fee-frontier study](docs/research/fee-frontier.md) includes assumptions and a reproducible script. Live checks fetch market-specific schedules; synthetic checks use the schedules you supply or the documented defaults.

New venue, data-provider and wallet integrations start in [oracle3-extras](https://github.com/YichengYang-Ethan/oracle3-extras), which includes Kairos candidate pairs and a MetaMask Agent Wallet adapter. See [extension setup and boundaries](docs/extras.md).

## Roadmap and contributing

The following are planned work, not current guarantees:

- **Agent reliability:** expand replayable failure cases, add durable execution traces and reconciliation, and test adapters against common behavior contracts.
- **Agent evaluation:** define labeled end-to-end tasks, model/Skill baselines and metrics for task completion, tool errors and unsupported conclusions.
- **Execution integration:** route strategy signals through shared fee schedules and connect multi-leg strategies to explicit fill/recovery states.
- **Market evidence:** measure fee-surviving gaps with timestamped data and publish reproducible forward observations with execution assumptions.

Useful contributions include a reproducible tool failure, a venue-adapter fixture, a client setup correction or a task with a checkable expected outcome. Start with the [contributing guide](.github/CONTRIBUTING.md), [open problems](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/issues?q=is%3Aissue+label%3Aopen-problem) or [Discussions](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/discussions). Coding agents should read [AGENTS.md](AGENTS.md).

## Related work and citation

- [Coinjure / U Lab prediction-market-cli](https://github.com/ulab-uiuc/prediction-market-cli): upstream research and code; see attribution below.
- [PMXT](https://github.com/pmxt-dev/pmxt): a unified prediction-market API.
- [Prediction Market Analysis](https://github.com/Jon-Becker/prediction-market-analysis): data collection and analysis.
- [Prediction Market Pricing](https://github.com/YichengYang-Ethan/prediction-market-pricing): companion pricing research and replication.

For software citation, use [CITATION.cff](CITATION.cff) and [Zenodo](https://doi.org/10.5281/zenodo.20062548). For the pricing model, cite the [working paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6468338).

## Origin and attribution

Oracle3 began as `ulab-uiuc/oracle3`, developed by Yicheng Yang and Haofei Yu at U Lab (University of Illinois Urbana-Champaign) under the MIT License, and it bundles the [`coinjure`](https://github.com/ulab-uiuc/prediction-market-cli) package from the same lab. The strategy, pricing, risk, dashboard, and test layers in this repository were added on top of that base; see [NOTICE](NOTICE) for the retained license text.

We acknowledge the foundational work of the Coinjure team at the University of Illinois Urbana-Champaign—Haofei Yu, Yicheng Yang, Yuxiang Liu, and Prof. Jiaxuan You—on which Oracle3 builds.

## License

Apache 2.0; see [LICENSE](LICENSE). Portions from the original U Lab code remain under the MIT License reproduced in [NOTICE](NOTICE).

*This software is for research and education. Trading involves financial risk.*
