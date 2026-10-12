# Runtime integrity and migration

These changes are on the repository's main branch. They are not a claim of live
profitability or a replacement for venue-account reconciliation. A published
package only includes them after its release contains this commit.

## Accounting

The engine's performance ledger uses cash plus marked inventory. Entry cash
outflow is not a trading loss. Entry and exit fees are allocated to FIFO closing
lots; `total_trades` counts those matched closing lots and `total_fills` counts
fills. One sale can close several lots. This count is not a round-trip count.

Reports expose `realized_pnl`, `unrealized_pnl`, `total_fees`, `cash`,
`inventory_value`, `valuation_complete`, and `missing_marks`. A missing quote
carries the previous mark or opening cost, while marking valuation incomplete.
A genuine zero bid is a zero mark. A quote's presence does not certify freshness,
executable depth, or a liquidation price for the entire position. Completeness
describes the current snapshot; drawdown uses observed/carried marks and cannot
reconstruct unobserved price changes between observations.

`sharpe_ratio=0` is a compatibility placeholder when `sharpe_available=false`;
it is not an estimate. Irregular fill observations are not annualized as daily
returns. Research selection excludes unavailable metrics; selecting on marked
PnL is exploratory and does not establish predictive performance.

The analyzer supports explicit `seed_position`, `mark_to_market`, and binary
`settle` events. It does not automatically fetch or authenticate venue settlement
outcomes. Engine restart restores a paired position/accounting checkpoint from
`runtime.json`. Legacy positions without that checkpoint establish a new session
cost baseline, not reconstructed historical performance. Inconsistent or corrupt
recovery data raises an error. The checkpoint is single-writer and is not a
transaction spanning a remote venue and the local intent journal.

## Orders and interruptions

Kalshi and Polymarket adapters retain client and venue order identifiers. Live
CLI sessions use a durable SQLite intent journal scoped to public account
identity. Direct Python callers must supply `journal_path` and a suitable
`journal_namespace`; the constructor default is an in-memory journal. Run one live
writer per account/namespace: SQLite reserves identical keys atomically, but it
is not a distributed account/inventory lock.

Reuse a `client_order_id` only for the same intent. A reserved ID is never
resubmitted, including after an ambiguous submission failure. A duplicate is an
explicit result, not a second order. Confirmed incremental fills enter accounting
once. Pending orders, contradictory reports, missing costs/fees, and unknown
outcomes require reconciliation and block further order submission. The engine
polls pending orders while the feed is idle and suspends strategy execution while
reconciliation or checkpoint persistence remains unresolved.

Kalshi accounting uses reported fill costs and fees. Polymarket matching is not
settlement finality: associated trades must confirm the actual quantities and
prices. Fee-bearing executions without verified net cash/share amounts remain
unresolved. This conservative limitation can prevent unattended continuation on
fee-bearing markets; it does not imply those markets are unsupported by the
venue itself.

A restart with previously applied fills or unresolved intents remains blocked
until account inventory and journal state are independently reconciled. There is
currently no automatic account-wide reconciliation/resume command. Do not delete
the journal or clear a guard to retry an uncertain order. An order ID that cannot
be recovered requires account-level investigation. A saved cash balance alone
cannot certify positions. The implementation does not cancel outstanding orders
automatically or guarantee exactly-once execution across a remote venue.

The existing portfolio risk breach and emergency kill switch stop all order
submission, including sales. They are stop controls, not liquidation routines.

## Multi-leg strategies

Implication, exclusivity, conditional, and structural strategies use shared fee
schedules and explicit basket execution outcomes. They track quantities filled
by their own strategy instance and never close unrelated account inventory.
Failed exits retain known residual exposure. An uncertain order stops subsequent
legs and automatic unwind; a terminal, confirmed partial execution can attempt
recovery of known fills. Recovery itself can fail.

`execution_state()` exposes the basket state, owned quantities, and most recent
execution evidence. Ownership is local to the strategy instance; it is not a
durable allocation ledger across restarts. Restarting with open strategy baskets
requires independent position attribution before resuming. Sequential orders
are not atomic transactions.

Configuration changes:

- `fee_rate=None` uses venue schedules through `oracle3.fees`; an explicitly
  supplied legacy rate is a flat per-contract override. `market_fee_terms`
  supplies known market-specific schedules. Missing rates in a present fee
  schedule are errors; they never mean zero fees.
- `min_edge` is measured after entry fees using executable basket quotes.
  Future exit fees and slippage are additional costs.
- Opaque Polymarket token IDs require explicit `yes_token_ids` mappings.
- Conditional baskets use the appropriate bound's hedge direction and weight.
  Structural baskets use the fitted slope's sign and magnitude. These are
  assumption-dependent expectation/convergence trades, not guaranteed-payoff
  arbitrages. Estimated relationships can fail.

Other contributed strategies and the Solana live adapter have not acquired all
of these lifecycle guarantees. Extending a venue or strategy requires separate
contract tests.

## Agent queues and installed packages

`AgentStrategy.queue_trade` stages an intent; it is not an execution receipt.
`execute_pending_trades` drains that queue once and returns observed outcomes.
A caller with custom execution must explicitly drain via `take_pending_trades`.
Rebuilding tools with an unhandled queue raises instead of silently discarding it.
The related-market example prioritizes queued tool requests over a duplicate
final JSON action and uses one execution path.

CI builds a wheel, installs it outside the checkout, and verifies actual stdio
MCP initialization, tool listing, a deterministic tool call, and shutdown with
supported MCP 1.x and 2.x versions. The contract harness records source/dependency
provenance. These are software regressions, not evaluations of LLM reasoning or
Skill effectiveness. See the [agent contract harness](agent-harness.md) and
[MCP ledger retry contract](mcp.md).
