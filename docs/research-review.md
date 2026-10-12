# Oracle3: agent environment reliability

Oracle3 extends the U Lab / Coinjure codebase with agent tools and execution
infrastructure for prediction markets. Its most inspectable contribution is the
contract between a tool call and the state change that actually occurred:
queued is different from executed, accepted is different from filled, and an
attempted recovery is different from a verified recovery.

**Current status:** an engineering artifact with reproducible failure cases.
Model-level usefulness is a research question, not an established result.

## A five-minute review

1. Read the three before/after cases below.
2. Inspect the [recorded evidence](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/blob/main/evidence/runtime-regressions.json), or replay both source revisions with one command.
3. Read the [contribution boundaries](#origin-and-contribution-boundaries) and the
   [next experiment](#next-experiment-not-yet-run).

Prediction markets provide a concrete stateful environment: actions change
inventory, some effects are delayed or partial, and retries can duplicate work.
These properties also occur in other agent tool environments. Whether the
findings transfer to those environments still needs testing.

## Three failures, reproduced on actual source revisions

Both sides run identical offline probes against committed Oracle3 snapshots.
The baseline is Oracle3 before its runtime-integrity repair, **not Coinjure**.
The agent output and venue responses are scripted; no LLM or live venue is used.

| Fixture | Earlier Oracle3 behavior | Repaired Oracle3 behavior | Invariant being checked |
|---|---|---|---|
| An agent queues a ten-share buy, then returns `HOLD` | Zero orders; the queued intent remains unexecuted | One order fills ten shares; queue is drained | An accepted queued intent has an explicit execution outcome |
| First leg fills ten shares; second leg fills one and remains open | Four submissions, including two reverse orders while the remainder can still fill | Stops after two submissions; exposes A=10 and B=1 as known residuals and requires reconciliation | Uncertain outstanding effects prevent automatic recovery claims or additional orders |
| Two paper-ledger instances start from the same state and each acknowledge a buy | One acknowledged fill disappears after the second write | Both fills remain; cash and inventory reflect both | A successful persisted write cannot be silently overwritten by a stale writer |

The queue example has an explicit contract: tool-queued requests take precedence
over the final free-form action text. This fixture tests that contract, not an
LLM's interpretation of `HOLD`.

The older executor already reported uncertainty in the second case. The repair
changes its behavior under that uncertainty; it is not a claim that the older
version reported the basket as hedged. The ledger fixture is a deterministic
stale-writer interleaving. Separate tests cover independent processes.

These are selected regressions found during development, not a representative
benchmark sample. They establish specific behavioral changes, not a general
reliability percentage or superiority over another project. A normal full-fill
control accompanies the failure cases.

## Reproduce the comparison

From a full Git checkout, after installing dependencies:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python scripts/replay_runtime_regressions.py --output /tmp/oracle3-regressions.json
```

Use `.venv\Scripts\python.exe` on Windows. The command uses temporary copies of
these immutable commits, without changing the working tree:

- Earlier: [`ec0f1a5d6bde7e9fe5b92663ec90aea496f74082`](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/commit/ec0f1a5d6bde7e9fe5b92663ec90aea496f74082).
- Repaired: [`586bd89b1ee5039488183d10744d389264e2bed1`](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/commit/586bd89b1ee5039488183d10744d389264e2bed1), merged in [PR #67](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/pull/67).

The report records source revisions, probe identity, environment versions, raw
observations, and expected invariants. Dependencies must be installed first;
replay itself blocks network access and uses temporary paper ledgers. An active
global trading kill switch is reported as a blocked environment and is left
unchanged; it must not be interpreted as a software-regression result. A shallow
clone needs the two commits fetched before replay. Dependency versions can affect
serialization, so compare semantic observations alongside the recorded versions.

The [nine-case contract harness](agent-harness.md) checks current tool behavior
separately. [CI](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/actions/workflows/pytest.yml)
also tests installed-wheel MCP sessions. At the repaired revision, the full
software suite passed 1,222 tests; this count is software coverage evidence, not
1,222 agent tasks or independent research observations.

## Origin and contribution boundaries

[Coinjure](https://github.com/ulab-uiuc/coinjure) lists Haofei Yu, Yicheng Yang,
Yuxiang Liu, and Jiaxuan You. Oracle3 retains attribution and the upstream MIT
notice in [NOTICE](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/blob/main/NOTICE).
The inherited code and original research are not presented as new independent
Oracle3 contributions. Acknowledgment does not imply that the upstream authors
supervise or endorse subsequent extensions.

| Layer | What to inspect | Boundary |
|---|---|---|
| U Lab / Coinjure foundation | [Upstream code and citation](https://github.com/ulab-uiuc/coinjure), bundled `coinjure/`, `NOTICE` | Existing team contribution; agent-native tooling predates these repairs |
| Agent-facing interfaces | [MCP reference](mcp.md), [agent quickstart](agent-quickstart.md), `AgentStrategy` | Integration and API contracts; MCP or Skills alone are not claimed as a research invention |
| Oracle3 reliability work | [PR #64](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/pull/64), [PR #67](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/pull/67), comparison above | Inspectable implementation changes and regression evidence in Yicheng Yang's maintained extension |
| Reproducibility | Frozen cases, structured outcomes, source/dependency provenance, installed-package checks | Deterministic software validation; model behavior has not been evaluated |
| Pricing research | [Separate replication repository](https://github.com/YichengYang-Ethan/prediction-market-pricing) | A separate contribution with separate assumptions and citation |

## Next experiment: not yet run

**Question:** Do explicit execution states and verified recovery outcomes help
an LLM agent complete stateful tasks correctly, beyond improving deterministic
software behavior?

A useful first study would keep the current execution backend fixed and compare
three information conditions: concise tool receipts, explicit state receipts,
and explicit state receipts plus the workflow Skill. Faults would be injected
into a simulator, without removing backend guards or using real accounts.

Before running models, freeze a separate task set, outcome oracle, prompts,
model versions, tool schemas, token/tool/time budgets, retry policy, and seeds.
Pair the same tasks and seeds across conditions. Include normal successful
workflows and cases where waiting or abstaining is the correct action, so a
system that always refuses cannot appear successful. Keep the known regressions
above as development examples, not a held-out evaluation set.

Report task completion from final state, unsupported success claims, unnecessary
or duplicate action attempts, recovery correctness, and tool/token cost.
Inspect traces and report uncertainty across tasks and repeated trials; do not
choose the best seed or infer profitability from task completion. Distinguish
model-provider failures from failures introduced by the simulator.

This direction is related to failure diagnosis and rerun verification in
[AgentDebugX](https://arxiv.org/abs/2607.18754), and to sequential evaluation in
[LiveTradeBench](https://arxiv.org/abs/2511.03628). Oracle3 is not integrated with
those systems, and no comparison against them is claimed.

## Remaining limits

Automatic account-wide restart recovery, complete fee-bearing Polymarket net
settlement accounting, and durable strategy ownership remain incomplete. The
relation checker relies on caller-supplied semantics, and sequential baskets
are not atomic. See [runtime integrity and migration](runtime-integrity.md).
These limits matter when choosing tasks for a future agent study.
