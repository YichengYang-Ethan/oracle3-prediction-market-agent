"""Main CLI entry point for Oracle3."""

from importlib import import_module
from typing import Any

import click

from oracle3 import __version__

# Keep discovery and MCP startup independent of authenticated traders, wallet
# SDKs and strategy providers. Import a command module only when it is selected.
_LAZY_COMMANDS = {
    'backtest': ('oracle3.cli.agent_commands', 'Run a strategy on historical data.'),
    'data': ('oracle3.cli.data_commands', 'Record and inspect market data.'),
    'live': ('oracle3.cli.agent_commands', 'Run live trading with explicit approval.'),
    'market': ('oracle3.cli.market_commands', 'Query public market data.'),
    'monitor': ('oracle3.cli.monitor', 'Monitor the trading engine.'),
    'news': ('oracle3.cli.news_commands', 'Fetch market news.'),
    'paper': ('oracle3.cli.agent_commands', 'Run a paper-trading strategy.'),
    'research': (
        'oracle3.cli.research_commands',
        'Inspect and evaluate market relations.',
    ),
    'strategy': ('oracle3.cli.agent_commands', 'Create and validate strategies.'),
    'trade': ('oracle3.cli.trade_commands', 'Inspect and control a running engine.'),
}


class _LazyGroup(click.Group):
    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(set(super().list_commands(ctx)) | set(_LAZY_COMMANDS))

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        command = super().get_command(ctx, cmd_name)
        if command is None and cmd_name in _LAZY_COMMANDS:
            module, _ = _LAZY_COMMANDS[cmd_name]
            command = getattr(import_module(module), cmd_name)
            self.add_command(command, cmd_name)
        return command

    def format_commands(
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        # Click's default formatter resolves every command, defeating lazy imports.
        rows = []
        for name in self.list_commands(ctx):
            command = self.commands.get(name)
            if command is not None:
                if not command.hidden:
                    rows.append((name, command.get_short_help_str()))
            else:
                rows.append((name, _LAZY_COMMANDS[name][1]))
        if rows:
            with formatter.section('Commands'):
                formatter.write_dl(rows)


@click.group(cls=_LazyGroup)
@click.version_option(version=__version__)
def cli() -> None:
    """Oracle3 - paper-trading engine for prediction markets on Kalshi, Polymarket, and Solana."""
    pass


@cli.command()
@click.option('--host', default='0.0.0.0', show_default=True, help='Bind address')
@click.option('--port', default=8080, show_default=True, type=int, help='Server port')
def blinks(host: str, port: int) -> None:
    """Start the Solana Blinks (Actions) server."""
    from oracle3.blinks.server import run_server

    click.echo(f'Starting Oracle3 Blinks server on {host}:{port}...')
    run_server(host=host, port=port)


@cli.command()
@click.option(
    '--port', default=3000, show_default=True, type=int, help='Dashboard server port'
)
@click.option(
    '--exchange',
    type=click.Choice(['solana', 'polymarket', 'kalshi']),
    default='solana',
    show_default=True,
)
@click.option(
    '--duration', type=float, default=None, help='Seconds to run (default: forever)'
)
@click.option('--initial-capital', default='10000', show_default=True)
@click.option(
    '--strategy-ref',
    default=None,
    help='Strategy ref: module:Class or /path/file.py:Class. If omitted, run in idle mode.',
)
@click.option(
    '--strategy-kwargs-json',
    default=None,
    help='JSON object for strategy constructor kwargs.',
)
@click.option(
    '--episode-dir',
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help='Replay DFlow episode directory containing dflow_events.parquet (Solana backtest mode).',
)
@click.option(
    '--max-events', default=None, type=int, help='Limit events for episode replay.'
)
@click.option(
    '--snapshot-json',
    default=None,
    type=str,
    help='Write periodic JSON state snapshot to PATH',
)
@click.option(
    '--snapshot-interval',
    default=5,
    type=click.IntRange(min=1),
    show_default=True,
    help='Seconds between snapshot writes',
)
def dashboard(  # noqa: C901
    port: int,
    exchange: str,
    duration: float | None,
    initial_capital: str,
    strategy_ref: str | None,
    strategy_kwargs_json: str | None,
    episode_dir: str | None,
    max_events: int | None,
    snapshot_json: str | None,
    snapshot_interval: int,
) -> None:
    """Launch the web dashboard with paper trading (open browser to http://localhost:PORT)."""
    import asyncio
    import json as json_lib
    import os
    from decimal import Decimal
    from pathlib import Path

    from oracle3.cli.agent_commands import _build_news_augmented_source, _IdleStrategy
    from oracle3.dashboard.server import DashboardServer
    from oracle3.strategy.loader import load_strategy_class

    # Build strategy
    if strategy_ref:
        kwargs = json_lib.loads(strategy_kwargs_json) if strategy_kwargs_json else {}
        strategy_cls = load_strategy_class(strategy_ref)
        strategy_obj = strategy_cls(**kwargs)
    else:
        strategy_obj = _IdleStrategy()

    capital = Decimal(initial_capital)
    is_backtest = episode_dir is not None

    # Build data source: episode replay or live
    if episode_dir:
        from oracle3.data.backtest.kalshi_replay_data_source import (
            SolanaReplayDataSource,
        )

        data_source = SolanaReplayDataSource(episode_dir, max_events=max_events)
        exchange = 'solana'  # force Solana for DFlow episodes
    else:
        data_source = _build_news_augmented_source(exchange)

    async def _run() -> None:  # noqa: C901
        from oracle3.cli.control import ControlServer
        from oracle3.core.trading_engine import TradingEngine
        from oracle3.data.market_data_manager import MarketDataManager
        from oracle3.position.position_manager import Position, PositionManager
        from oracle3.risk.risk_manager import NoRiskManager
        from oracle3.ticker.ticker import CashTicker
        from oracle3.trader.paper_trader import PaperTrader

        market_data = MarketDataManager(
            spread=Decimal('0') if is_backtest else Decimal('0.01'),
            max_history_per_ticker=None,
            max_timeline_events=None,
        )
        position_manager = PositionManager()

        # Set up initial cash based on exchange
        if exchange in ('solana', 'dflow'):
            cash_ticker = CashTicker.DFLOW_USDC
        elif exchange == 'kalshi':
            cash_ticker = CashTicker.KALSHI_USD
        else:
            cash_ticker = CashTicker.POLYMARKET_USDC

        position_manager.update_position(
            Position(
                ticker=cash_ticker,
                quantity=capital,
                average_cost=Decimal('0'),
                realized_pnl=Decimal('0'),
            )
        )

        # --- Feature 2: On-Chain Risk Manager ---
        risk_manager: Any = NoRiskManager()
        try:
            from oracle3.risk.onchain_risk_manager import OnChainRiskManager

            risk_manager = OnChainRiskManager(
                position_manager=position_manager,
                market_data=market_data,
                initial_capital=capital,
                enable_simulation=False,  # paper mode — skip real RPC sim
            )
            click.echo('  [✓] On-Chain Risk Manager loaded')
        except Exception as exc:
            click.echo(f'  [–] On-Chain Risk Manager skipped: {exc}')

        trader = PaperTrader(
            market_data=market_data,
            risk_manager=risk_manager,
            position_manager=position_manager,
            min_fill_rate=Decimal('0.8'),
            max_fill_rate=Decimal('1.0'),
            commission_rate=Decimal('0.0'),
        )

        # --- Feature 4: MEV Protection (Jito) ---
        jito_submitter = None
        try:
            from oracle3.trader.jito_submitter import JitoSubmitter

            jito_submitter = JitoSubmitter(keypair=None)
            trader._jito_submitter = jito_submitter  # type: ignore[attr-defined]
            click.echo('  [✓] Jito MEV Protection loaded')
        except Exception as exc:
            click.echo(f'  [–] Jito MEV Protection skipped: {exc}')

        # --- Feature 3: On-Chain Signal Source ---
        if not is_backtest and exchange in ('solana', 'dflow'):
            try:
                from oracle3.data.live.onchain_signal_source import (
                    OnChainSignalSource,
                    WatchedWallet,
                )

                signal_source = OnChainSignalSource(
                    watched_wallets=[
                        WatchedWallet(
                            address='7RQ3YL4cLNbQbwAUHBP6GzdRbG6NRng8qBcHbiDrf8Ae',
                            label='oracle3-agent',
                        ),
                    ],
                    polling_interval=60.0,
                )
                # Add to composite data source
                if hasattr(data_source, 'sources'):
                    data_source.sources.append(signal_source)
                # Bootstrap "monitoring started" signal so Feature #3 card is active
                import time as _time

                from oracle3.data.live.onchain_signal_source import OnChainSignal

                signal_source._signals.append(
                    OnChainSignal(
                        signal_type='wallet_monitor',
                        wallet='7RQ3YL4cLNbQbwAUHBP6GzdRbG6NRng8qBcHbiDrf8Ae',
                        amount=0.0,
                        direction='monitoring',
                        token='USDC',
                        timestamp=_time.time(),
                        label='oracle3-agent',
                    )
                )
                click.echo('  [✓] On-Chain Signal Source loaded')
            except Exception as exc:
                click.echo(f'  [–] On-Chain Signal Source skipped: {exc}')

        engine = TradingEngine(
            data_source=data_source,
            strategy=strategy_obj,
            trader=trader,
            continuous=not is_backtest,
        )

        # --- Feature 5: Agent Reputation + On-Chain Logger ---
        onchain_logger = None
        keypair_file = Path.home() / '.oracle3' / 'keypair.json'
        try:
            if keypair_file.exists():
                from oracle3.onchain.logger import OnChainLogger
                from oracle3.trader.solana_trader import _load_keypair

                kp = _load_keypair(keypair_path=str(keypair_file))
                onchain_logger = OnChainLogger(
                    keypair=kp,
                    rpc_url='https://api.devnet.solana.com',
                )
                click.echo(
                    f'  [✓] Solana keypair loaded ({str(kp.pubkey())[:8]}…, devnet)'
                )
        except Exception as exc:
            click.echo(f'  [–] Solana keypair skipped: {exc}')

        try:
            from oracle3.onchain.reputation import ReputationManager

            rep_mgr = ReputationManager(on_chain_logger=onchain_logger)
            engine._reputation_manager = rep_mgr
            if hasattr(strategy_obj, 'reputation_manager'):
                strategy_obj.reputation_manager = rep_mgr
            if hasattr(strategy_obj, '_onchain_logger'):
                strategy_obj._onchain_logger = onchain_logger
            click.echo('  [✓] Agent Reputation loaded')
        except Exception as exc:
            click.echo(f'  [–] Agent Reputation skipped: {exc}')

        # --- Feature 6: Multi-Agent Pipeline ---
        coordinator = None
        try:
            from oracle3.agent.coordinator import AgentCoordinator, RiskAgent

            coordinator = AgentCoordinator(
                risk_agent=RiskAgent(risk_manager=risk_manager),
            )
            if hasattr(strategy_obj, 'coordinator'):
                strategy_obj.coordinator = coordinator
            click.echo('  [✓] Multi-Agent Pipeline loaded')
        except Exception as exc:
            click.echo(f'  [–] Multi-Agent Pipeline skipped: {exc}')

        # --- Feature 7: Flash Loan Arbitrage ---
        try:
            from oracle3.experimental.flash_loan import FlashLoanArbitrage

            kp_for_features = onchain_logger._keypair if onchain_logger else None
            flash_loan = FlashLoanArbitrage(
                keypair=kp_for_features,
                jito_submitter=jito_submitter,
                max_borrow=float(capital),
                rpc_url='https://api.devnet.solana.com',
            )
            engine._flash_loan = flash_loan
            # Also link to strategy if it supports it
            if hasattr(strategy_obj, 'flash_loan_handler'):
                strategy_obj.flash_loan_handler = flash_loan
            click.echo('  [✓] Flash Loan Arbitrage loaded')
        except Exception as exc:
            click.echo(f'  [–] Flash Loan Arbitrage skipped: {exc}')

        # --- Feature 8: Atomic Multi-Leg Trader ---
        try:
            from oracle3.trader.atomic_trader import AtomicTrader

            atomic_trader = AtomicTrader(
                keypair=kp_for_features,
                jito_submitter=jito_submitter,
                rpc_url='https://api.devnet.solana.com',
            )
            engine._atomic_trader = atomic_trader
            if hasattr(strategy_obj, 'atomic_trader'):
                strategy_obj.atomic_trader = atomic_trader
            click.echo('  [✓] Atomic Multi-Leg Trader loaded')
        except Exception as exc:
            click.echo(f'  [–] Atomic Multi-Leg Trader skipped: {exc}')

        # Start dashboard server (background thread)
        dash = DashboardServer(engine, port=port)
        dash.start()

        # Start control server (Unix socket; gracefully skipped on Windows)
        ctrl = ControlServer(engine)
        try:
            await ctrl.start()
        except AttributeError:
            click.echo(
                '  [–] Control server skipped (Windows — no Unix socket support)'
            )

        if coordinator is not None:
            await coordinator.start()

        # --- Snapshot writer (background) ---
        snapshot_task: asyncio.Task[None] | None = None
        if snapshot_json:
            from datetime import datetime, timezone

            async def _write_snapshot() -> None:
                """Periodically write a JSON state snapshot to PATH (atomically)."""
                while True:
                    await asyncio.sleep(snapshot_interval)

                    # Build non-cash positions list
                    positions_list = [
                        {
                            'ticker': pos.ticker.symbol,
                            'quantity': pos.quantity,
                            'average_cost': pos.average_cost,
                            'realized_pnl': pos.realized_pnl,
                        }
                        for pos in position_manager.positions.values()
                        if not isinstance(pos.ticker, CashTicker)
                    ]

                    # Cash balance
                    cash_positions = position_manager.get_cash_positions()
                    cash = (
                        cash_positions[0].quantity if cash_positions else Decimal('0')
                    )

                    # Last prices (mid-price from order books, YES-side only)
                    last_prices: dict[str, Decimal] = {}
                    for ticker in sorted(
                        market_data.order_books, key=lambda t: t.symbol
                    ):
                        if isinstance(ticker, CashTicker):
                            continue
                        if getattr(ticker, 'is_no_side', False):
                            continue
                        best_bid = market_data.get_best_bid(ticker)
                        best_ask = market_data.get_best_ask(ticker)
                        if best_bid is not None and best_ask is not None:
                            last_prices[ticker.symbol] = (
                                best_bid.price + best_ask.price
                            ) / Decimal('2')
                        elif best_bid is not None:
                            last_prices[ticker.symbol] = best_bid.price
                        elif best_ask is not None:
                            last_prices[ticker.symbol] = best_ask.price

                    snapshot = {
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'exchange': exchange,
                        'positions': positions_list,
                        'cash': cash,
                        'last_prices': last_prices,
                    }

                    # Atomic write: tmp → replace
                    tmp_path = f'{snapshot_json}.tmp'
                    with open(tmp_path, 'w') as f:
                        json_lib.dump(snapshot, f, default=str, indent=2)
                    os.replace(tmp_path, snapshot_json)

            snapshot_task = asyncio.create_task(_write_snapshot())
            click.echo(
                f'  [✓] JSON snapshots every {snapshot_interval}s → {snapshot_json}'
            )

        mode_label = 'Solana Backtest' if is_backtest else 'Paper Trading'
        click.echo(f'{mode_label} Dashboard running at http://localhost:{port}')
        click.echo('Press Ctrl+C to stop.\n')

        try:
            if duration:
                await asyncio.wait_for(engine.start(), timeout=duration)
            else:
                await engine.start()

            if is_backtest:
                click.echo('\nBacktest complete. Dashboard still running for review.')
                click.echo('Press Ctrl+C to stop.\n')
                while True:
                    await asyncio.sleep(1)
        except asyncio.TimeoutError:
            click.echo(f'\nDuration reached ({duration}s). Stopping...')
        except asyncio.CancelledError:
            pass
        finally:
            if snapshot_task is not None:
                snapshot_task.cancel()
            if coordinator is not None:
                await coordinator.stop()
            await engine.stop()
            try:
                await ctrl.stop()
            except Exception:
                pass
            dash.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        click.echo('\nDashboard stopped.')


@cli.command('trade-log')
@click.option(
    '--limit', default=20, show_default=True, type=int, help='Number of entries'
)
@click.option(
    '--keypair-path',
    default=None,
    help='Solana keypair JSON file (or SOLANA_KEYPAIR_PATH)',
)
@click.option(
    '--rpc-url',
    default='https://api.mainnet-beta.solana.com',
    show_default=True,
    help='Solana RPC URL',
)
@click.option('--json', 'as_json', is_flag=True, default=False, help='Output as JSON')
def trade_log(
    limit: int, keypair_path: str | None, rpc_url: str, as_json: bool
) -> None:
    """Show on-chain trade log from Solana Memo transactions."""
    import asyncio
    import json as json_lib

    from oracle3.onchain.logger import OnChainLogger
    from oracle3.trader.solana_trader import _load_keypair

    try:
        kp = _load_keypair(keypair_path=keypair_path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    log = OnChainLogger(keypair=kp, rpc_url=rpc_url)
    trades = asyncio.run(log.get_trade_log(limit=limit))

    if as_json:
        click.echo(json_lib.dumps({'trades': trades, 'count': len(trades)}))
        return

    if not trades:
        click.echo('No Oracle3 trade log entries found.')
        return

    click.echo(f'On-chain trade log ({len(trades)} entries):\n')
    for i, t in enumerate(trades, 1):
        market = t.get('market', '?')
        side = t.get('side', '?')
        price = t.get('price', '?')
        qty = t.get('qty', '?')
        ts = t.get('ts', '')
        sig = t.get('signature', '')[:16]
        click.echo(
            f'  [{i}] {market} {side.upper()} x{qty} @ {price}  {ts}  tx:{sig}...'
        )
    click.echo()


def _empty_calibration_rows() -> dict[str, list[float]]:
    return {
        'price': [],
        'outcome': [],
        'volume': [],
        'duration_hours': [],
        'spread': [],
    }


def _parse_calibration_row(row: dict[str, str], row_number: int) -> dict[str, float]:
    try:
        price = float(row['price'])
        outcome = float(row['outcome'])
    except (TypeError, ValueError) as exc:
        raise click.ClickException(
            f'Invalid price/outcome at CSV row {row_number}'
        ) from exc

    if not 0 < price < 1:
        raise click.ClickException(
            f'Invalid price at CSV row {row_number}: expected value in (0, 1)'
        )
    if outcome not in (0.0, 1.0):
        raise click.ClickException(
            f'Invalid outcome at CSV row {row_number}: expected 0 or 1'
        )

    parsed = {'price': price, 'outcome': float(outcome)}
    for optional in ('volume', 'duration_hours', 'spread'):
        value = row.get(optional)
        if value:
            try:
                parsed[optional] = float(value)
            except ValueError as exc:
                raise click.ClickException(
                    f'Invalid {optional} at CSV row {row_number}'
                ) from exc
    return parsed


def _load_calibration_csv(csv_path: str) -> dict[str, list[float]]:
    import csv

    rows = _empty_calibration_rows()

    with open(csv_path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise click.ClickException('CSV file has no header row')

        fieldnames = set(reader.fieldnames)
        required = {'price', 'outcome'}
        missing = sorted(required - fieldnames)
        if missing:
            raise click.ClickException(
                f'CSV is missing required column(s): {", ".join(missing)}'
            )

        for row_number, row in enumerate(reader, start=2):
            parsed = _parse_calibration_row(row, row_number)
            rows['price'].append(parsed['price'])
            rows['outcome'].append(parsed['outcome'])
            for optional in ('volume', 'duration_hours', 'spread'):
                if optional in parsed:
                    rows[optional].append(parsed[optional])

    if not rows['price']:
        raise click.ClickException('CSV file has no data rows')

    return rows


def _mle_result_to_json_dict(result: Any) -> dict[str, Any]:
    return {
        'beta': result.beta,
        'lambda_hat': result.lambda_hat,
        'se_fisher': result.se_fisher,
        'se_robust': result.se_robust,
        'se_cluster': result.se_cluster,
        'log_likelihood': result.log_likelihood,
        'n_obs': result.n_obs,
        'n_params': result.n_params,
        'converged': result.converged,
        'aic': result.aic,
        'bic': result.bic,
        'pseudo_r2': result.pseudo_r2,
        'covariate_names': result.covariate_names,
    }


@cli.command()
@click.option(
    '--csv',
    'csv_path',
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help='Resolved-contract CSV with price and outcome columns.',
)
@click.option(
    '--platform',
    type=click.Choice(['polymarket', 'kalshi', 'metaculus', 'manifold', 'default']),
    default='default',
    show_default=True,
    help='Platform prior used as the optimizer warm start.',
)
@click.option(
    '--hierarchical',
    is_flag=True,
    default=False,
    help='Fit a covariate model using volume, duration_hours, and spread columns.',
)
@click.option('--json', 'as_json', is_flag=True, default=False, help='Output as JSON')
def calibrate(
    csv_path: str,
    platform: str,
    hierarchical: bool,
    as_json: bool,
) -> None:
    """Estimate Wang Transform lambda from a resolved-contract CSV."""
    import json as json_lib

    from oracle3.pricing.wang_mle import (
        LAMBDA_KALSHI,
        LAMBDA_MANIFOLD,
        LAMBDA_METACULUS,
        LAMBDA_POLYMARKET,
        LAMBDA_POOLED,
        WangMLE,
    )

    platform_priors = {
        'polymarket': LAMBDA_POLYMARKET,
        'kalshi': LAMBDA_KALSHI,
        'metaculus': LAMBDA_METACULUS,
        'manifold': LAMBDA_MANIFOLD,
        'default': LAMBDA_POOLED,
    }
    import importlib.util

    missing = [m for m in ('numpy', 'scipy') if importlib.util.find_spec(m) is None]
    if missing:
        raise click.ClickException(
            f'oracle3 calibrate needs {" and ".join(missing)}: '
            f'pip install {" ".join(missing)}'
        )

    rows = _load_calibration_csv(csv_path)
    mle = WangMLE()

    covariates = None
    covariate_names = None
    initial_beta: list[float] = [platform_priors[platform]]
    if hierarchical:
        optional_columns = ('volume', 'duration_hours', 'spread')
        missing = [
            name for name in optional_columns if len(rows[name]) != len(rows['price'])
        ]
        if missing:
            raise click.ClickException(
                '--hierarchical requires complete CSV column(s): ' + ', '.join(missing)
            )
        covariates = mle.build_design_matrix(
            volumes=rows['volume'],
            durations_hours=rows['duration_hours'],
            prices=rows['price'],
            spreads=rows['spread'],
        )
        covariate_names = [
            'constant',
            'ln(1+volume)',
            'ln(1+duration)',
            '|p-0.5|',
            'spread',
        ]
        initial_beta.extend([0.0] * (len(covariate_names) - 1))

    result = mle.fit(
        prices=rows['price'],
        outcomes=rows['outcome'],
        covariates=covariates,
        covariate_names=covariate_names,
        initial_beta=initial_beta,
    )

    if as_json:
        click.echo(json_lib.dumps(_mle_result_to_json_dict(result)))
        return

    click.echo(result.summary_table())


@cli.command()
@click.option(
    '--keypair-path',
    default=None,
    help='Solana keypair JSON file (or SOLANA_KEYPAIR_PATH)',
)
@click.option(
    '--rpc-url',
    default='https://api.mainnet-beta.solana.com',
    show_default=True,
    help='Solana RPC URL',
)
@click.option(
    '--wallet', default=None, help='Wallet address to check (defaults to own wallet)'
)
@click.option('--json', 'as_json', is_flag=True, default=False, help='Output as JSON')
def reputation(
    keypair_path: str | None, rpc_url: str, wallet: str | None, as_json: bool
) -> None:
    """Show agent reputation score from on-chain trading history."""
    import json as json_lib

    from oracle3.onchain.reputation import ReputationManager

    # Build reputation manager
    rep_mgr: ReputationManager
    if keypair_path or not wallet:
        try:
            from oracle3.onchain.logger import OnChainLogger
            from oracle3.trader.solana_trader import _load_keypair

            kp = _load_keypair(keypair_path=keypair_path)
            logger = OnChainLogger(keypair=kp, rpc_url=rpc_url)
            rep_mgr = ReputationManager(on_chain_logger=logger)
        except (ValueError, Exception) as exc:
            if wallet:
                rep_mgr = ReputationManager()
            else:
                raise click.ClickException(str(exc)) from exc
    else:
        rep_mgr = ReputationManager()

    target_wallet = wallet or rep_mgr.wallet
    if not target_wallet:
        raise click.ClickException(
            'No wallet specified. Pass --wallet or configure a keypair.'
        )

    rep = rep_mgr.get_agent_reputation(target_wallet)

    if as_json:
        click.echo(json_lib.dumps(rep))
        return

    click.echo(f'Agent Reputation: {target_wallet[:8]}...{target_wallet[-4:]}')
    click.echo(f'  Score:       {rep.get("score", 0):.1f} / 100')
    click.echo(f'  Win Rate:    {rep.get("win_rate", 0):.2%}')
    click.echo(f'  Sharpe:      {rep.get("sharpe", 0):.4f}')
    click.echo(f'  Total Trades: {rep.get("total_trades", 0)}')
    click.echo(f'  Consistency: {rep.get("consistency", 0):.4f}')


@cli.command('mcp')
@click.option(
    '--transport',
    type=click.Choice(['stdio', 'streamable-http', 'sse']),
    default='stdio',
    show_default=True,
)
def mcp_server(transport: str) -> None:
    """Run the MCP server: market data, fee-aware no-arbitrage checks, paper trading."""
    from oracle3.mcp_server.server import main

    main(['--transport', transport])


if __name__ == '__main__':
    cli()
