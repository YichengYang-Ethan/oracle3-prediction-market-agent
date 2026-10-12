# Oracle3: from U Lab research to reusable agent infrastructure

**Project development and maintenance: Yicheng Yang**

Oracle3 builds on prediction-market research and software from U Lab at the
University of Illinois Urbana-Champaign. It brings market data, structured
tools, reusable agent workflows, and an execution environment together in an
open-source Python project.

[GitHub repository](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent)
· [Connect an agent](agent-quickstart.md)
· [MCP tool reference](mcp.md)

## Work at U Lab

Yicheng Yang worked directly with Haofei Yu on
[Coinjure](https://github.com/ulab-uiuc/coinjure), contributing to the lab's
prediction-market software. He also co-developed the original U Lab Oracle3
codebase with Haofei. This work provided the foundation for his continued
development of Oracle3.

Coinjure established an agent-driven workflow spanning market discovery,
strategy development, and trading operations. Oracle3 extends that foundation
with additional interfaces, reusable workflows, and infrastructure for building
and evaluating agents.

## Yicheng Yang's subsequent development

Yicheng develops and maintains the Oracle3 extensions in this repository. The
work focuses on making the underlying system easier to connect to, build on,
and evaluate:

| Area | What Oracle3 provides |
|---|---|
| MCP integration | Thirteen tools for Kalshi and Polymarket market discovery, market details, order books, relationship checks, and local paper trading |
| Reusable agent Skills | Six documented workflows covering data discovery, constraint checks, strategy authoring, and paper/live operations |
| Market reasoning tools | Shared fee calculations and five relationship checks that return structured baskets, costs, and assumptions |
| Execution infrastructure | Python strategy interfaces, paper/live adapters, risk controls, and inspectable execution state; MCP paper operations use a separate local ledger |
| Evaluation infrastructure | A reproducible software harness, automated tests, and installed-package checks across MCP SDK versions |
| Open-source access | A packaged CLI and MCP server, client setup guides, examples, documentation, and a machine-readable project index |

These layers support a practical workflow: an agent can discover contracts,
inspect their rules and order books, calculate a fee-aware comparison, and
record a simulated position. Developers can use the Python interfaces to build
their own strategies and experiments. Real-order execution belongs to the
separate trading engine; the MCP interface provides market reads, calculations,
and paper operations.

## Why this is useful for agent research

Prediction markets provide a concrete environment for sequential decisions:
agents observe changing information, use tools, manage state, and evaluate the
consequences of their actions. Oracle3 makes these components available through
explicit interfaces that can be reused across agent implementations.

The broader development direction is a shared environment for studying how
agents use tools and follow workflows. Three questions motivate further work:

- **Tool design:** How do structured inputs, outputs, and state information
  affect task completion and the number of tool calls?
- **Workflow guidance:** What does a reusable Skill add beyond a tool schema
  and a task prompt?
- **Evaluation harnesses:** How can the same tasks and environment be used to
  compare agent implementations reproducibly?

A planned model-level evaluation would hold the environment and task budgets
fixed while comparing tool interfaces and Skill configurations. It would
measure completion from the resulting state, alongside tool use and token
cost. The current harness validates software behavior; model-level experiments
are the next research step.

## Explore the project

- [Agent quickstart](agent-quickstart.md): connect an MCP client and run a
  worked task.
- [MCP reference](mcp.md): inspect the tools, arguments, and returned data.
- [Source code](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent):
  explore the strategy, market, and execution modules.
- [Automated validation](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/actions/workflows/pytest.yml):
  inspect the software test and package-compatibility results.

## Origin and contribution boundaries

We acknowledge the foundational work of the Coinjure team at the University of Illinois Urbana-Champaign—Haofei Yu, Yicheng Yang, Yuxiang Liu, and Prof. Jiaxuan You—on which Oracle3 builds.

Oracle3 retains the original U Lab and Coinjure attribution and MIT license
notices in [NOTICE](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/blob/main/NOTICE).
Yicheng Yang maintains the subsequent extensions described above. Software
citation information is available in
[CITATION.cff](https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent/blob/main/CITATION.cff).
