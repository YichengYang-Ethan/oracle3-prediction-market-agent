# Oracle3

**Agent tools and execution infrastructure for prediction markets.**

Give your agent Kalshi and Polymarket data, fee-aware checks on related
contracts, and a local paper-trading ledger through MCP. Build strategies
with Python and the trading engine, and inspect tool errors and incomplete
execution with an offline contract harness.

[Agent quickstart](agent-quickstart.md) · [MCP tool reference](mcp.md) ·
[Machine-readable index](llms.txt) ·
[GitHub](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent)

## Choose a task

| Task | Entry point | Result |
|---|---|---|
| Find a market and inspect its metadata | `search_markets`, `get_market` | Identifiers, prices and available rule fields |
| Check a related-contract price gap after fees | `check_constraint`, `check_constraint_live` | Candidate baskets, per-leg fees and conditional net edge |
| Simulate buying with live order-book data | `paper_order`, `paper_portfolio` | Fills and positions in a local paper ledger |
| Reproduce tool or execution failures | [Contract harness](agent-harness.md) | Frozen cases with structured pass/fail evidence |
| Build a strategy or integrate a venue | [Source layout](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/blob/main/AGENTS.md#layout), [extras](extras.md) | Python interfaces and extension guidance |

Public market reads need no exchange credentials. Pure calculations and the
harness run offline after installation. MCP exposes no real-money order tool;
the separate engine supports paper/live adapters for Kalshi, Polymarket and
Solana (DFlow).

## Connect an MCP client

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
For Claude Code, using the released PyPI package:

```bash
claude mcp add oracle3 -- uvx --from oracle3 oracle3-mcp
```

For clients that support an `mcpServers` configuration:

```json
{
  "mcpServers": {
    "oracle3": { "command": "uvx", "args": ["--from", "oracle3", "oracle3-mcp"] }
  }
}
```

The client launches the server as a stdio subprocess. Ask the agent to call
`list_relation_types`, then try the
[offline constraint example](agent-quickstart.md). It uses synthetic quotes,
needs no API keys and does not write a ledger.

!!! info "Published package and development checkout"
    These docs follow `main`. Explicit `assessment` fields and the offline
    contract harness were added after release 1.2.2. Follow the
    [source installation](agent-quickstart.md) for those behaviors and the
    repository's scripts and skills. A PyPI install alone does not provide
    the `scripts/` or `skills/` directories.

## Understand the result

A check answers whether a basket has a positive calculated edge under the
relationship, prices and fee schedules supplied. It does not prove settlement
equivalence, available depth or full execution. Current main exposes
`relation_verified: false`, `execution_verified: false` and
`conditional_on_relation: true` in its `assessment` object.

Read contract rules and outcome mappings before using a relationship.
Polymarket market details expose `rules` and `resolution_source` when the
upstream API supplies them; check the venue's market page if fields are missing
or ambiguous. Event sums require mutually exclusive, exhaustive outcomes.
The MCP paper ledger is independent of the engine's paper state.

## Inspect the evidence

The [agent contract harness](agent-harness.md) replays nine frozen scenarios:
invalid inputs, venue errors, fee-erased gaps, conditional relations, full and
partial fills, failed recovery and unresolved orders. It calls real components
with mocked responses, blocks socket connections and produces JSON evidence.
[CI reports](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/actions/workflows/pytest.yml)
cover locked dependencies and the latest supported MCP SDK.

This checks tool and execution behavior; agent reasoning, Skill effectiveness
and profitability need separate evaluations. The spread executor is sequential,
and integration across strategies remains in progress.

## Read next

- [Agent quickstart](agent-quickstart.md): exact inputs, expected outputs, client setup and skills.
- [MCP server](mcp.md): all 13 tools, side effects and relation semantics.
- [Agent contract harness](agent-harness.md): cases, output contract and migration notes.
- [oracle3-extras](extras.md): optional data, venue and wallet integrations.
- [Fee-frontier study](research/fee-frontier.md): reproducible fee calculations and assumptions.
- [CLI monitoring](CLI_MONITORING.md): inspecting and controlling engine sessions.
- [Solana execution architecture](architecture.md): pre-flight checks and submission behavior.
- [Origin and attribution](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent#origin-and-attribution): the Coinjure / U Lab foundation.
