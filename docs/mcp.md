# MCP server

Oracle3 ships a [Model Context Protocol](https://modelcontextprotocol.io) server that gives AI agents read access to Kalshi and Polymarket, the venues' fee schedules, fee-aware no-arbitrage checks, and a local paper-trading ledger. No tool can place a real order.

## Run it

```bash
pip install oracle3
oracle3 mcp                              # stdio, for desktop and IDE clients
oracle3 mcp --transport streamable-http  # HTTP
```

`oracle3-mcp` is an equivalent entry point.

## Configure a client

Claude Code:

```bash
claude mcp add oracle3 -- uvx oracle3 mcp
```

Claude Desktop, Cursor and other clients that read an `mcpServers` block:

```json
{
  "mcpServers": {
    "oracle3": { "command": "uvx", "args": ["oracle3", "mcp"] }
  }
}
```

The server is listed in the MCP Registry as `io.github.YichengYang-Ethan/oracle3`.

## Tools

| Tool | Arguments | Returns | Side effects |
|---|---|---|---|
| `search_markets` | `venue`, `query`, `limit`, `series_ticker` | Open markets with best prices | read-only |
| `get_market` | `venue`, `market_id` | Prices, volume, close time, resolution rules | read-only |
| `get_orderbook` | `venue`, `market_id`, `depth` | YES and NO bids and asks, best first | read-only |
| `get_quote` | `venue`, `market_id` | Best bid and ask on both sides, the market's fee schedule | read-only |
| `check_constraint_live` | `relation`, `markets`, `contracts`, `maker` | Every basket's cost, fees and edge, with live quotes | read-only |
| `check_constraint` | `relation`, `quotes`, `contracts`, `maker` | The same, on quotes you supply | none |
| `trading_fee` | `venue`, `price`, `contracts`, `maker`, schedule parameters | Fee for one fill | none |
| `fair_value` | `market_price`, `lam` | Probability implied under the Wang transform | none |
| `list_relation_types` | | Supported relations and their bounds | none |
| `list_relations` | `market_id`, `spread_type`, `status` | Relations saved locally, for example by `oracle3-extras kairos sync` | reads `~/.oracle3/relations.json` |
| `paper_order` | `venue`, `market_id`, `side`, `contracts`, `limit_price` | Fill against the live book, with fees | writes the paper ledger |
| `paper_portfolio` | | Cash, positions, fill count | reads the paper ledger |
| `paper_reset` | `confirm` | Fresh ledger | erases the paper ledger |

Prices are dollars per contract. On Kalshi, `market_id` is the ticker; on Polymarket it is the Gamma market id, "yes" is the first outcome token and "no" the second. The paper ledger lives at `~/.oracle3/mcp_paper_ledger.json` unless `ORACLE3_MCP_LEDGER` points elsewhere.

## Relations

| Relation | Bound | Markets, in order | Basket |
|---|---|---|---|
| `implication` | P(A) ≤ P(B) when A implies B | A, B | NO on A + YES on B, pays at least 1 |
| `exclusivity` | probabilities sum to at most 1 | every outcome | NO on every outcome, pays at least n − 1 |
| `complement` | P(A) + P(B) = 1 | A, B | YES + YES or NO + NO, pays 1 |
| `same_event` | P(A) = P(B), one event on two venues | A, B | YES on one + NO on the other, pays 1 |
| `event_sum` | probabilities sum to 1 | every outcome | YES on all (pays 1) or NO on all (pays n − 1) |

Whether a relation holds is a judgment about the contracts' wording and resolution rules. Read both markets with `get_market` before relying on a result.

## Fees

Each quote carries the fee schedule the venue reports for that market:

- **Kalshi:** taker round_up(0.07 · M · C · P · (1 − P)), maker round_up(0.0175 · M · C · P · (1 − P)) only on series with maker fees; M is the series `fee_multiplier`.
- **Polymarket:** taker C · rate · p · (1 − p) with the market's `feeSchedule.rate`; makers pay nothing.

The derivation of what these fees mean for arbitrage is in [Do prediction-market arbitrage edges survive fees?](research/fee-frontier.md).

## Example

Ask an agent: *"Is the Fed-hold contract on Kalshi priced consistently with the same contract on Polymarket after fees?"* A typical tool sequence:

1. `search_markets(venue="kalshi", series_ticker="KXFEDDECISION")` and `search_markets(venue="polymarket", query="fed decision")`
2. `get_market` on the two candidates to compare their resolution rules
3. `check_constraint_live(relation="same_event", markets=[{"venue": "kalshi", "market_id": "<ticker>"}, {"venue": "polymarket", "market_id": "<id>"}])`

The result lists both baskets (YES on Kalshi + NO on Polymarket, and the reverse) with the fee on each leg and the edge left after fees.
