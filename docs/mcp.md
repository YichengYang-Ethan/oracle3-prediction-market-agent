# MCP server

Oracle3 ships a [Model Context Protocol](https://modelcontextprotocol.io) server that gives AI agents read access to Kalshi and Polymarket, the venues' fee schedules, fee-aware no-arbitrage checks, and a local paper-trading ledger. No tool can place a real order.

## Run it

For a released package, use the dedicated server entry point:

```bash
pip install oracle3
oracle3-mcp                              # stdio, for desktop and IDE clients
```

Run this as a client-managed subprocess; stdio waits for MCP protocol input.
`oracle3 mcp` is an equivalent CLI entry point; it also accepts
`--transport streamable-http` for HTTP. Use a virtual environment for pip
installs and configure the absolute path to its server executable.

This page describes current `main`. The explicit `assessment` fields, paper
retry handling and execution contracts were added after release 1.2.2. For those behaviors,
follow the source installation in the [agent quickstart](agent-quickstart.md).
The quickstart also provides an offline first call with exact arguments and
expected output.

## Configure a client

Claude Code:

```bash
claude mcp add oracle3 -- uvx --from oracle3 oracle3-mcp
```

Claude Desktop, Cursor and other clients that read an `mcpServers` block:

```json
{
  "mcpServers": {
    "oracle3": { "command": "uvx", "args": ["--from", "oracle3", "oracle3-mcp"] }
  }
}
```

The server is listed in the MCP Registry as `io.github.YichengYang-Ethan/oracle3`.

## Tools

| Tool | Arguments | Returns | Side effects |
|---|---|---|---|
| `search_markets` | `venue`, `query`, `limit`, `series_ticker` | Open markets with best prices | read-only |
| `get_market` | `venue`, `market_id` | Prices, volume, timing and venue-provided rules; Polymarket resolution source | read-only |
| `get_orderbook` | `venue`, `market_id`, `depth` | YES and NO bids and asks, best first | read-only |
| `get_quote` | `venue`, `market_id` | Best bid and ask on both sides, the market's fee schedule | read-only |
| `check_constraint_live` | `relation`, `markets`, `contracts`, `maker` | Every basket's cost, fees and edge, with live quotes | read-only |
| `check_constraint` | `relation`, `quotes`, `contracts`, `maker` | The same, on quotes you supply | none |
| `trading_fee` | `venue`, `price`, `contracts`, `maker`, schedule parameters | Fee for one fill | none |
| `fair_value` | `market_price`, `lam` | Probability implied under the Wang transform | none |
| `list_relation_types` | | Supported relations and their bounds | none |
| `list_relations` | `market_id`, `spread_type`, `status` | Relations saved locally, for example by `oracle3-extras kairos sync` | reads `~/.oracle3/relations.json` |
| `paper_order` | `venue`, `market_id`, `side`, `contracts`, `limit_price`, optional `request_id` | Fill against the live book, with fees; original result on retry | writes the paper ledger |
| `paper_portfolio` | | Cash, positions, fill count | reads the paper ledger |
| `paper_reset` | `confirm` | Fresh ledger | erases the paper ledger |

Prices are dollars per contract. On Kalshi, `market_id` is the ticker; on Polymarket it is the Gamma market id, "yes" is the first outcome token and "no" the second. The paper ledger lives at `~/.oracle3/mcp_paper_ledger.json` unless `ORACLE3_MCP_LEDGER` points elsewhere.

Polymarket market details return `rules` from the upstream `description` and
`resolution_source` from `resolutionSource`. Obtain missing or ambiguous rules
from the venue's market page and verify the outcome mapping before assuming a
relationship. The MCP paper ledger is separate from the
trading engine's paper state. Use your client's `tools/list` response for the
installed server's complete JSON input schemas; older releases may differ.

## Local paper orders and retries

Call `paper_order` with a returned market ID and a distinct retry key for each
intended order, for example:

```json
{
  "venue": "kalshi",
  "market_id": "<returned ticker>",
  "side": "yes",
  "contracts": 1,
  "limit_price": 0.55,
  "request_id": "experiment-01-leg-1"
}
```

`contracts` must be finite and positive; `limit_price` must be finite and
strictly between 0 and 1. Invalid order inputs fail before venue requests.
Once a result is saved, the same nonblank `request_id` with identical venue,
market, side, quantity and limit returns it without another fill or venue request.
Changing that intent is an error. Unfilled and rejected results are also
remembered: use a new key for a deliberate new attempt. Without `request_id`,
each call is a new order. `paper_reset` with `{"confirm":true}` clears positions,
fills and retry history; without confirmation it changes nothing.

Ledger reads, orders and resets use an interprocess lock and atomic file
replacement. This assumes cooperating processes using this implementation on
a local filesystem; the same path shares one paper account. Use separate paths
for separate experiments. Positions remain at cost: this ledger does not track
resolution or P&L, and each basket leg is a separate simulation.

Valid older JSON ledgers load without a manual migration. Malformed JSON,
nonfinite values or inconsistent accounting cause an error and leave the file
intact; even `paper_reset` refuses to overwrite it. Preserve a copy, then
restore a valid backup or configure a fresh `ORACLE3_MCP_LEDGER` path and restart
the server. Do not edit the ledger or remove its `.lock` sidecar while servers
are using it.

## Relations

| Relation | Bound | Markets, in order | Basket |
|---|---|---|---|
| `implication` | P(A) ≤ P(B) when A implies B | A, B | NO on A + YES on B, pays at least 1 |
| `exclusivity` | probabilities sum to at most 1 | every outcome | NO on every outcome, pays at least n − 1 |
| `complement` | P(A) + P(B) = 1 | A, B | YES + YES or NO + NO, pays 1 |
| `same_event` | P(A) = P(B), one event on two venues | A, B | YES on one + NO on the other, pays 1 |
| `event_sum` | probabilities sum to 1 for mutually exclusive, exhaustive outcomes | every outcome | YES on all (pays 1) or NO on all (pays n − 1) |

Whether a relation holds is a judgment about the contracts' wording and resolution rules. Read both markets with `get_market` and obtain any missing rules before relying on a result.

Both check tools return `best` (or `null` when no candidate can be priced),
`missing_quotes`, `assumptions` and, on current main, `assessment`:

```json
{
  "relation_verified": false,
  "execution_verified": false,
  "conditional_on_relation": true
}
```

`profitable_after_fees` describes conditional arithmetic. The tools do not verify
settlement equivalence, order-book depth or full fills. `check_constraint_live`
fetches current quotes and fee terms but retains the same assessment boundary.

## Fees

Each quote carries the fee schedule the venue reports for that market:

- **Kalshi:** taker round_up(0.07 · M · C · P · (1 − P)), maker round_up(0.0175 · M · C · P · (1 − P)) only on series with maker fees; M is the series `fee_multiplier`.
- **Polymarket:** taker C · rate · p · (1 − p) with the market's `feeSchedule.rate`; makers pay nothing.

The derivation of what these fees mean for arbitrage is in [Do prediction-market arbitrage edges survive fees?](research/fee-frontier.md).

## Example

Ask an agent: *"Is the Fed-hold contract on Kalshi priced consistently with the same contract on Polymarket after fees?"* A typical tool sequence:

1. `search_markets(venue="kalshi", series_ticker="KXFEDDECISION")` and `search_markets(venue="polymarket", query="fed decision")`
2. `get_market` on the two candidates; compare their settlement rules and outcome mappings, consulting venue pages when returned fields are missing or ambiguous
3. `check_constraint_live(relation="same_event", markets=[{"venue": "kalshi", "market_id": "<ticker>"}, {"venue": "polymarket", "market_id": "<id>"}])`

The result lists both baskets (YES on Kalshi + NO on Polymarket, and the reverse) with the fee on each leg and the edge left after fees.
