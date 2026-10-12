# Oracle3

**Build prediction-market agents that discover markets, analyze opportunities, and test strategies.**

Connect agents to Kalshi and Polymarket through MCP, guide their work with
reusable Skills, and develop strategies with a Python execution engine.

**13 MCP tools · 6 agent Skills · Kalshi & Polymarket data · Python 3.10+**

[Agent quickstart](agent-quickstart.md) · [MCP tool reference](mcp.md) ·
[Project and contributions](research-review.md) ·
[GitHub](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent)

## Explore the toolkit

| Build | What Oracle3 provides |
|---|---|
| Market research agents | Contract discovery, resolution rules, quotes, and order books |
| Related-contract analysis | Five relationship checks, shared fees, and structured basket calculations |
| Paper-trading workflows | Simulated entries and local holdings through MCP |
| Python strategies | Strategy interfaces, risk controls, and paper/live venue adapters |
| Agent experiments | Reusable Skills, observable state, and reproducible software validation |

Public market reads need no exchange credentials. MCP provides data,
calculations, and a local paper ledger. Real-order execution uses the separate
CLI/Python engine.

## Connect your agent

Install the current development checkout:

```bash
git clone https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent.git
cd oracle3-prediction-market-agent
python -m venv .venv
.venv/bin/python -m pip install -e .
claude mcp add oracle3 -- "$PWD/.venv/bin/oracle3-mcp"
```

For other MCP clients and Windows setup, follow the
[agent quickstart](agent-quickstart.md). Published-package installation is
documented in the [MCP reference](mcp.md).

Then ask your agent:

> Find Kalshi and Polymarket contracts about the next Fed rate decision.
> Compare their outcome definitions, resolution rules, and order-book quotes.
> Summarize the markets with their IDs, prices, and comparison assumptions.

Prefer an offline first task? Try the
[worked fee-aware calculation](agent-quickstart.md#first-call-a-deterministic-fee-check).

## Tools, Skills, and an execution environment

Use individual MCP tools for structured data and calculations, or load one of
the six [agent Skills](agent-quickstart.md#agent-skills) for a complete workflow.
The Python interfaces support strategy development and engine operations;
[oracle3-extras](extras.md) hosts optional data, venue, and wallet integrations.

For agent developers, [llms.txt](llms.txt) provides a machine-readable guide to
tasks, tools, and project resources.

## From U Lab research to an open-source project

Yicheng Yang worked with Haofei Yu on Coinjure at U Lab and co-developed the
original U Lab Oracle3 codebase. He continues to extend Oracle3 with MCP
integration, reusable agent workflows, and infrastructure for experimentation.

Read the [project overview and contributions](research-review.md) for the
background, current capabilities, and direction for agent research.

## Go deeper

- [MCP reference](mcp.md): all 13 tools and their inputs and outputs.
- [CLI monitoring](CLI_MONITORING.md): operate and inspect engine sessions.
- [Agent harness](agent-harness.md): reproducible software validation.
- [Execution reference](runtime-integrity.md): accounting and execution semantics.
- [Fee-frontier study](research/fee-frontier.md): a reproducible cost analysis.
- [Extension guide](extras.md): connect optional data, venues, and wallets.
