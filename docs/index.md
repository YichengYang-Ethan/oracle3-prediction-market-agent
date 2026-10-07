# Oracle3

**Oracle3 is an open-source trading engine and MCP server for prediction markets.** It maps the logical relations between event contracts, finds prices that break the axioms of probability after each venue's fees, and trades them live on Kalshi, Polymarket and Solana, or on paper, under pre-trade risk limits.

!!! tip "Trades live on Kalshi, Polymarket and Solana"
    `oracle3 live run` executes with the same engine that runs paper trading, behind pre-trade risk limits and a kill switch. AI agents plug in through the MCP server.

!!! info "Extensions live in oracle3-extras"
    [oracle3-extras](https://github.com/YichengYang-Ethan/oracle3-extras) adds Kairos cross-venue pairs, live no-arbitrage scans and Polymarket execution through MetaMask Agent Wallet. New integrations and experimental features start there. See [oracle3-extras](extras.md).

## Install

```bash
pip install oracle3
```

## Find markets

```bash
oracle3 market search --exchange kalshi --query "fed" --json
oracle3 market search --exchange polymarket --query "fed decision" --json
```

## Check a relation between two contracts

```python
from oracle3.arbitrage import Quote, check_constraint
from oracle3.fees import KalshiSchedule

# A implies B, but A is bid at 0.60 while B is offered at 0.55.
result = check_constraint(
    "implication",
    [Quote("A", yes_bid=0.60, schedule=KalshiSchedule()),
     Quote("B", yes_ask=0.55, schedule=KalshiSchedule())],
)
print(result.best.gross_edge, result.best.fees, result.best.net_edge)
# 0.05 0.0342 0.0158
```

## Use it from an AI agent

```bash
claude mcp add oracle3 -- uvx oracle3 mcp
```

The [MCP server](mcp.md) exposes 13 tools: market search, quotes, order books, fee-aware constraint checks and a local paper ledger.

## Read next

- [MCP server](mcp.md): tools, client configuration and a worked example
- [oracle3-extras](extras.md): the companion package for integrations and experimental features
- [Do prediction-market arbitrage edges survive fees?](research/fee-frontier.md): break-even violations under the published fee schedules
- [CLI quick start](CLI_QUICK_START.md) and [monitoring](CLI_MONITORING.md)
- [Solana pre-flight risk and submission](architecture.md): local limits, transaction simulation, Jito fallback and failure handling
- [Project specification](PROJECT_SPECIFICATION.md): module reference
