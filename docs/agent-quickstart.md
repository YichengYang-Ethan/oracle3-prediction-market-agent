# Use Oracle3 from an AI agent

Use Oracle3 to find Kalshi or Polymarket contracts, inspect quotes and fees, and
calculate a basket's price gap under a stated relationship between its outcomes.
Public market reads need no exchange account or API key. The MCP server also
supports a local paper ledger; no MCP tool places real orders.

## Choose the code you run

`pip install oracle3` and `uvx --from oracle3 oracle3-mcp` run the **published
package**. The **main checkout** can contain changes that are not released yet.
The `assessment` fields, paper retry handling and regression runner below describe main; do not infer
their availability from a package version badge alone.

For current main, clone and install a checkout:

```bash
git clone --branch main https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent.git
cd oracle3-prediction-market-agent
python -m venv .venv
.venv/bin/python -m pip install -e .
git rev-parse HEAD
```

Record that commit when sharing a result. On Windows, use
`.venv\Scripts\python.exe` for the Python commands. Installation needs network
access; the first calculation below does not.

In a client that accepts an `mcpServers` configuration, replace the example path
with the **absolute path to this checkout's executable**:

```json
{
  "mcpServers": {
    "oracle3": {
      "command": "/absolute/path/oracle3-prediction-market-agent/.venv/bin/oracle3-mcp",
      "args": []
    }
  }
}
```

For Windows, use the absolute path to `.venv\Scripts\oracle3-mcp.exe`, escaping
backslashes in JSON. For Claude Code the equivalent is:

```bash
claude mcp add oracle3 -- /absolute/path/oracle3-prediction-market-agent/.venv/bin/oracle3-mcp
```

Choose one configuration for `oracle3` and reload the connection. The client
starts the stdio server; it is not an interactive shell. See the [MCP
reference](mcp.md) for published-package configuration and other transports.

## First call: a deterministic fee check

After the client discovers the tools, call **`check_constraint`** with these
arguments. `A` and `B` are fictional labels; this is a calculation example,
not an observed trading opportunity.

```json
{
  "relation": "implication",
  "quotes": [
    {"market_id": "A", "venue": "kalshi", "yes_bid": 0.60, "kalshi_multiplier": 1},
    {"market_id": "B", "venue": "kalshi", "yes_ask": 0.55, "kalshi_multiplier": 1}
  ],
  "contracts": 100,
  "maker": false
}
```

The supplied premise is **A implies B**. The checker derives the NO ask on A
as `1 - 0.60` and prices 100 NO contracts on A plus 100 YES contracts on B.
On current main, this is a subset of the returned data:

```json
{
  "violated": true,
  "profitable_after_fees": true,
  "best": {
    "description": "NO on A + YES on B",
    "cost": 95.0,
    "fees": 3.4125,
    "gross_edge": 5.0,
    "net_edge": 1.5875
  },
  "assessment": {
    "relation_verified": false,
    "execution_verified": false,
    "conditional_on_relation": true
  }
}
```

Amounts are dollars for the requested basket size. `violated` reports a positive
gross gap; `profitable_after_fees` applies the calculated fees. Neither verifies
the event relationship, liquidity for the requested size, or simultaneous fills.
Read `assumptions`, `missing_quotes`, and every leg. If no basket can be priced,
`best` is `null`. Invalid inputs and venue failures are tool errors, not
successful empty results.

## Match the task to a tool

| Task | Tool | Inputs that matter |
|---|---|---|
| Find candidates | `search_markets` | `venue`, `query`, `limit`; Kalshi also supports `series_ticker` |
| Inspect metadata and available rules | `get_market` | `venue`, `market_id` |
| Read prices and fee terms | `get_quote` | `venue`, `market_id` |
| Inspect displayed depth | `get_orderbook` | `venue`, `market_id`, `depth` |
| Calculate on supplied quotes | `check_constraint` | `relation`, ordered `quotes`, `contracts`, `maker` |
| Calculate with live quotes | `check_constraint_live` | `relation`, ordered `markets`, `contracts`, `maker` |
| Inspect supported labels / saved relations | `list_relation_types`, `list_relations` | Saved relations remain claims to review |
| Simulate a buy / inspect local holdings | `paper_order`, `paper_portfolio` | See the local paper section below |

MCP venues are `kalshi` and `polymarket`. Kalshi's `market_id` is its ticker;
Polymarket's is its Gamma market ID, not a CLOB token ID. On Polymarket, the
tool's `yes` and `no` mean the first and second outcome tokens; inspect the
outcome labels, especially for head-to-head markets.

`trading_fee` is a pure single-fill calculation. `fair_value` returns a
model-implied probability under a chosen Wang-transform parameter; it does not
independently verify an event probability.

## Research a live pair

1. Call `search_markets` separately with
   `{"venue":"kalshi","query":"fed","limit":10}` and
   `{"venue":"polymarket","query":"fed decision","limit":10}`.
2. Pass returned IDs to `get_market`, for example
   `{"venue":"kalshi","market_id":"<returned ticker>"}`. Compare outcomes,
   deadlines, and settlement sources. Kalshi responses include `rules_primary`
   and `rules_secondary`; Polymarket responses include `rules` and
   `resolution_source` when supplied by the venue. If fields are missing or
   ambiguous, check the venue's contract rules before assuming equivalence.
3. State the proposed relation and supporting rules. `event_sum` requires a
   complete, mutually exclusive outcome set; a shared event ID is insufficient.
   If rules are missing or incompatible, report the unresolved question.
4. Inspect each `get_orderbook` for the requested size. For a proposed same-event
   pair, call `check_constraint_live` with this object, replacing both placeholders:

```json
{
  "relation": "same_event",
  "markets": [
    {"venue": "kalshi", "market_id": "<returned Kalshi ticker>"},
    {"venue": "polymarket", "market_id": "<returned Gamma market ID>"}
  ],
  "contracts": 10,
  "maker": false
}
```

Report the declared relation, gross edge, fee on each leg, net edge, displayed
depth, and assumptions. Live quotes are snapshots; `check_constraint_live` does
not certify depth or fills. Public reads require network access, not trading
credentials. Review the returned fee terms instead of assuming every market
uses the same defaults.

## Local paper side effects

For paper execution, `paper_order` accepts arguments such as
`{"venue":"kalshi","market_id":"<returned ticker>","side":"yes","contracts":1,"limit_price":0.55,"request_id":"experiment-01-leg-1"}`.
It walks a live-book snapshot and writes `~/.oracle3/mcp_paper_ledger.json`, or
the path in `ORACLE3_MCP_LEDGER`. Inspect `status` and, for a fill, `requested`,
`filled`, `fees` and `levels`; a partial fill is not a completed basket. Each leg
is a separate simulation call. Quantities must be finite and positive, and the
limit finite and strictly between 0 and 1; invalid order inputs fail before HTTP.

Use an optional nonblank `request_id` for retries. Once a result is saved, the
same key with identical venue, market, side, quantity and limit returns it without
another fill or venue request; changing the intent is an error. Unfilled and
rejected results are remembered too. Use a new key for a deliberate new attempt
and a distinct key for each leg. Without a key, every call is a new order.

`paper_portfolio` takes `{}` and reads that ledger. It supports buys, carries
holdings at cost, and does not track resolution or P&L. This ledger is separate
from the CLI paper engine. `paper_reset` with `{"confirm":true}` erases paper
holdings, fills and retry history; without confirmation it changes nothing.
Reads, orders and resets lock across cooperating processes on a local
filesystem and use atomic writes. Sharing a ledger path shares one account;
use separate paths for experiments that need separate state.

Valid older ledgers load automatically. Invalid JSON, nonfinite values or
inconsistent accounting produce an error and preserve the file, including when
reset is requested. Preserve a copy and restore a valid backup, or point
`ORACLE3_MCP_LEDGER` at a fresh path and restart the server. Do not edit the
ledger or delete its `.lock` sidecar while servers are using it.

## Agent skills

MCP supplies tools; skills supply workflow instructions. Installing Python
packages or configuring MCP does **not** install these skills into a client.

| Skill source | Workflow |
|---|---|
| [pm-constraint-arbitrage](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/blob/main/skills/pm-constraint-arbitrage/SKILL.md) | Review a relation and its fee-adjusted basket through MCP |
| [pm-data-discovery](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/blob/main/skills/pm-data-discovery/SKILL.md) | Discover markets and collect research samples |
| [pm-quant-strategy-authoring](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/blob/main/skills/pm-quant-strategy-authoring/SKILL.md) | Author a parameterized quantitative strategy |
| [pm-agent-strategy-authoring](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/blob/main/skills/pm-agent-strategy-authoring/SKILL.md) | Author an LLM- or tool-driven strategy |
| [pm-paper-trade-ops](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/blob/main/skills/pm-paper-trade-ops/SKILL.md) | Run and inspect the CLI paper engine |
| [pm-live-trade-ops](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/blob/main/skills/pm-live-trade-ops/SKILL.md) | Operate the separate live CLI when explicitly requested |

Inside this checkout, Claude Code loads the `.claude/skills/` mirror; Cursor
also supports that directory. For Codex, copy the selected skill folder from
`skills/` into your project's `.agents/skills/`. Outside this checkout, copy the
folder to the client-supported project or user skills directory, preserving its
`SKILL.md` and supporting files. Check for an existing same-named skill before
copying; keep the MCP connection configured separately. CLI-based skills also
need the checkout's `.venv/bin` on the agent's executable search path (Windows:
`.venv\Scripts`).

Start with `pm-constraint-arbitrage`, confirm it appears in the client's skill
list, and ask the client to use it for the first-call example. Current discovery
instructions: [Claude Code](https://code.claude.com/docs/en/skills),
[Cursor](https://cursor.com/docs/skills), and
[Codex](https://learn.chatgpt.com/docs/build-skills).

## Replay the tool contracts

From the installed main checkout:

```bash
.venv/bin/python scripts/check_agent_contracts.py --output agent-contracts.json
```

This offline runner exercises frozen MCP errors, conditional checks, and
sequential execution failures against mocks. It emits structured evidence and
fails on regressions. It evaluates tool and execution behavior, not LLM
reasoning, skill effectiveness, or strategy returns. See
[agent contract regression](agent-harness.md) for cases and migration notes.
