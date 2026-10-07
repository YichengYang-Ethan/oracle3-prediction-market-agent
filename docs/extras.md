# oracle3-extras

[oracle3-extras](https://github.com/YichengYang-Ethan/oracle3-extras) is Oracle3's companion package, the way [pymc-extras](https://github.com/pymc-devs/pymc-extras) relates to PyMC. It holds integrations with wallets, data providers and venues, and features that need real-world use before they belong in Oracle3.

## What it adds

| Feature | Module | What it does |
|---|---|---|
| Kairos cross-venue pairs | `oracle3_extras.market.kairos` | Reads [Kairos](https://kairos.trade)'s public matched-market catalog, lines up the outcomes of every pair across Kalshi, Polymarket, Predict.fun and Hyperliquid, and saves the Kalshi–Polymarket ones as Oracle3 `same_event` or `complement` relations |
| Live no-arbitrage scans | `oracle3_extras.arbitrage` | Runs `check_constraint` over thousands of relations with batched quotes, then sizes the survivors against the order books |
| MetaMask Agent Wallet | `oracle3_extras.trader.metamask` | A `Trader` that executes on Polymarket through Agent Wallet, so Oracle3 never holds a private key (testnet by default) |

## Install and try it

```bash
pip install git+https://github.com/YichengYang-Ethan/oracle3-extras.git
oracle3-extras kairos pairs --rejected   # aligned pairs, and why the others were left out
oracle3-extras kairos sync               # save them where list_relations reads
oracle3-extras kairos scan               # read-only: cross-venue edges after fees
```

After `kairos sync`, the [MCP server](mcp.md) sees the pairs: `list_relations` lists them, and each relation's markets can be passed straight to `check_constraint_live`.

## How it relates to Oracle3

- **Same namespaces.** `oracle3_extras.market`, `oracle3_extras.arbitrage` and `oracle3_extras.trader` mirror `oracle3.market`, `oracle3.arbitrage` and `oracle3.trader`, and build on their classes (`MarketRelation`, `check_constraint`, `Trader`).
- **One-way dependency.** oracle3-extras depends on one Oracle3 minor series; Oracle3 does not depend on it.
- **Graduation.** A feature that people use, that is tested to Oracle3's standard and that needs no partner account moves into Oracle3 at the same path, and the extras version warns for one release before it is removed.

## Contributing

New integrations and experimental features start in oracle3-extras. Its [contributing guide](https://github.com/YichengYang-Ethan/oracle3-extras/blob/main/CONTRIBUTING.md) lists what every integration needs: tests without network access, safe defaults, a transparent attribution section and a named maintainer.
