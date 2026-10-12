# Agent-facing contract regression

The deterministic runner checks the tool and execution behavior an agent sees.
It calls the real MCP dispatcher, fee-aware constraint checker, and sequential
spread executor with frozen inputs, mocked venue HTTP responses, and a scripted
trader. It does not call an LLM or place live orders. Socket connections are
blocked, and no credentials or exchange accounts are needed.

## Run from a fresh checkout

Use Python 3.10 or later, from the repository root:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python scripts/check_agent_contracts.py
```

On Windows, use `.venv\Scripts\python.exe` instead of `.venv/bin/python`.
Installing dependencies needs network access; executing the runner does not.
The runner prints JSON to stdout and exits with status 0 when all cases pass,
or 1 if any case fails. To save the report:

```bash
.venv/bin/python scripts/check_agent_contracts.py --output /tmp/oracle3-contracts.json
```

The output records Oracle3/MCP and installed dependency versions, source Git
revision and dirty status, a source-content fingerprint, fixed case IDs, pass/fail
status, and structured evidence. It has no timestamps or random inputs. Repeated runs
with the same code and dependencies produce the same JSON. MCP error wording can
differ between SDK versions; compare case IDs and behavioral outcomes across
versions rather than expecting byte-identical reports.

## Frozen cases

| Case ID | Required behavior |
|---|---|
| `mcp.reject_negative_quantity` | Reject a negative contract count and identify the invalid input. |
| `mcp.reject_unknown_relation` | Reject a relation outside the supported schema. |
| `mcp.preserve_venue_error` | Surface a mocked HTTP 404 as a tool error, not a successful empty result. |
| `mcp.fees_remove_thin_edge` | Distinguish a positive gross price gap from a negative edge after fees. |
| `relation.conditional_not_verified` | Explicitly mark the relation and execution as unverified. |
| `execution.exact_full_fill` | Recognize two terminal fills for their full requested quantities. |
| `execution.partial_fill_is_incomplete` | Reject basket success for a terminal partial fill; report the exact remaining exposure. |
| `execution.failed_unwind_keeps_exposure` | Report an unsuccessful reverse order and retained exposure without claiming a hedge. |
| `execution.open_remainder_requires_reconciliation` | Preserve known residuals and skip automatic unwind while an order may still fill. |

The relation case deliberately supplies an incomplete outcome set while labeling
it `event_sum`. The price checker cannot discover an omitted outcome from those
quotes. The conditional price calculation can be positive even though the basket
pays zero if that omitted outcome occurs. Passing this case means the output
exposes its assumptions and assessment flags; it does **not** mean Oracle3 has
verified the settlement relationship.

Execution evidence includes requested legs, observed order results, reverse-order
attempts, known residual quantities, and uncertainty requiring reconciliation.
These are simulated exchange responses. The executor is sequential and does not
provide atomic transactions across venues.

## What the report establishes

This is a small regression suite for tool and environment behavior, not a
research benchmark. It does not measure an agent's reasoning, skill triggering,
relationship detection, live fill probability, or trading performance. It does
not establish profitability, production readiness, or superiority over another
project. Separate LLM or skill evaluations need labeled tasks, fixed model and
prompt configurations, baselines, and an explicit evaluation protocol.

The cases live in `scripts/check_agent_contracts.py`. Add a stable case ID when a
new externally visible failure is fixed, and update this table with the behavior
it protects. The runner's own tests check repeatability, failure reporting,
nonzero failure exit status, and the network block.

```bash
.venv/bin/python -m pip install pytest pytest-asyncio
.venv/bin/python -m pytest -o addopts='' tests/test_agent_contract_harness.py
```

## Migrating callers

### Spread execution

`all_filled` now requires nonempty requested legs and exact terminal fill
confirmations. An empty result is not a completed spread. `hedged` is true only
when reverse fills neutralize the attempt's known position changes and there is
no unresolved execution state. An attempted unwind alone does not establish this.

Consume `to_payload()` for requested orders, observed fills, `unwind_results`,
signed `residual_exposures`, `execution_uncertain`, and
`requires_reconciliation`. Residuals describe this attempt, not total account
positions. An empty residual list with uncertain execution still requires venue
reconciliation: an outstanding order could fill later. The executor does not
cancel orders or infer finality after an adapter exception. Uncertain execution
skips automatic unwind; confirmed terminal partial fills may attempt recovery.
This in-memory helper does not provide durable recovery after process loss or
coroutine cancellation; those interruptions require independent reconciliation.

### Event-sum strategies

`EventSumArbStrategy` does not trade until configured with a complete partition:

```json
{
  "event_id": "verified-event",
  "outcome_market_ids": ["market-a", "market-b"],
  "yes_token_ids": {"market-a": "yes-token-a", "market-b": "yes-token-b"},
  "outcomes_are_exhaustive": true,
  "outcomes_are_mutually_exclusive": true
}
```

These are caller assertions that require rule review, not a semantic proof.
All declared markets and the relevant quote sizes must be available. An
unexpected market invalidates the declaration. `trade_size` means equal shares
per leg. Fees default to `oracle3.fees.PolymarketSchedule`; supply current
per-market `market_fee_rates` where known. The optional legacy `fee_rate` is
an explicit flat per-share override, not a venue fee schedule. This strategy
stops after an incomplete leg but does not automatically neutralize earlier
fills; its orders remain sequential.

### Model comparisons

`FairValueEstimator.estimate(..., reference_probability=...)` accepts an
independent probability forecast available at the decision time. Its
`mispricing_signal` is now a signed price residual in dollars per contract:
observed price minus the model-adjusted forecast. Without a reference it is
exactly zero and excluded from opportunity ranking; this means no independent
comparison was made, not that the market has been proved correctly priced.
The decomposition-only `fair_value` and `risk_premium` remain available.

### Constraint checks

`check_constraint` and `check_constraint_live` reject nonfinite quantities,
out-of-range quotes, blank market identities, and duplicate contracts. Equal
IDs on different venues are distinct. The additive `assessment` object exposes
that rules and execution have not been verified; a positive conditional edge
must not be interpreted as an unconditional trade instruction.

## Runtime and installed-package checks

See [runtime integrity and migration](runtime-integrity.md) for accounting
semantics, order journaling, retry behavior, strategy ownership and recovery
limits. CI additionally builds and installs a wheel outside the checkout and
checks a real MCP stdio session on supported SDK 1.x and 2.x versions.
