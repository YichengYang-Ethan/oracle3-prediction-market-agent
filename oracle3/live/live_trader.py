from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from oracle3.core.trading_engine import TradingEngine
from oracle3.data.live.dflow_data_source import DFlowDataSource
from oracle3.data.live.kalshi_data_source import LiveKalshiDataSource
from oracle3.data.live.live_data_source import (
    LiveNewsDataSource,
    LivePolyMarketDataSource,
    LiveRSSNewsDataSource,
)
from oracle3.data.market_data_manager import MarketDataManager
from oracle3.position.position_manager import Position, PositionManager
from oracle3.risk.risk_manager import (
    NoRiskManager,
    RiskManager,
    StandardRiskManager,
)
from oracle3.strategy.strategy import Strategy
from oracle3.ticker.ticker import CashTicker
from oracle3.trader.kalshi_trader import KalshiTrader
from oracle3.trader.paper_trader import PaperTrader
from oracle3.trader.polymarket_trader import PolymarketTrader
from oracle3.trader.solana_trader import SolanaTrader
from oracle3.trader.trader import Trader

if TYPE_CHECKING:
    from oracle3.alerts.alerter import Alerter
    from oracle3.storage.state_store import StateStore

logger = logging.getLogger(__name__)


def _live_journal_config(
    venue: str, public_account: str, state_store: StateStore | None
) -> dict:
    """Account-scoped durable intent log; never persist a credential as an ID."""
    if not public_account:
        raise ValueError('Public account identity required for durable order recovery')
    identity = hashlib.sha256(public_account.lower().encode()).hexdigest()[:24]
    namespace = f'{venue}-{identity}'
    directory = Path(
        getattr(state_store, 'data_dir', Path.home() / '.oracle3' / 'orders')
    )
    return {
        'journal_path': directory / f'{namespace}.sqlite3',
        'journal_namespace': namespace,
    }


def _emit_stdout(message: str, *, emit_text: bool) -> None:
    if emit_text:
        print(message)


async def run_live_trading(
    data_source: LivePolyMarketDataSource
    | LiveNewsDataSource
    | LiveRSSNewsDataSource
    | LiveKalshiDataSource
    | DFlowDataSource,
    strategy: Strategy,
    trader: Trader,
    duration: float | None = None,
    state_store: StateStore | None = None,
    alerter: Alerter | None = None,
    continuous: bool = True,
    drawdown_alert_pct: Decimal | None = None,
    monitor: bool = False,
    exchange_name: str = '',
    emit_text: bool = True,
) -> None:
    """
    Run live trading with the given data source, strategy, and trader.

    Args:
        data_source: The live data source to use (Polymarket, News API, or RSS)
        strategy: The trading strategy to execute
        trader: The trader implementation (Paper or Polymarket)
        duration: Optional duration in seconds to run. If None, runs indefinitely.
        state_store: Optional state store for persistence.
        alerter: Optional alerter for notifications.
        continuous: Keep engine running when the data source is temporarily idle.
        drawdown_alert_pct: Optional drawdown alert threshold as a decimal (0.1 = 10%).
        monitor: Enable the Textual TUI dashboard and ControlServer.
        exchange_name: Exchange name shown in the monitor header.
    """
    engine = TradingEngine(
        data_source=data_source,
        strategy=strategy,
        trader=trader,
        state_store=state_store,
        alerter=alerter,
        continuous=continuous,
        drawdown_alert_pct=drawdown_alert_pct,
    )

    # NOTE: data_source.start() is called by engine.start() internally
    # (guarded by _ds_started flag). Do NOT call it here to avoid double-starting.

    if monitor:
        from oracle3.cli.utils import add_monitoring_to_engine

        monitored = add_monitoring_to_engine(
            engine,
            watch=True,
            exchange_name=exchange_name,
        )
        if duration:
            try:
                await asyncio.wait_for(monitored.start(), timeout=duration)
            except asyncio.TimeoutError:
                await monitored.stop()
                _emit_stdout(
                    f'Live trading stopped after {duration} seconds.',
                    emit_text=emit_text,
                )
        else:
            await monitored.start()
    else:
        # In headless mode, still run the control socket server so
        # `oracle3 monitor` and `oracle3 trade` can attach to this process.
        from oracle3.cli.utils import add_monitoring_to_engine

        monitored = add_monitoring_to_engine(
            engine,
            watch=False,
            exchange_name=exchange_name,
        )
        if duration:
            try:
                await asyncio.wait_for(monitored.start(), timeout=duration)
            except asyncio.TimeoutError:
                await monitored.stop()
                _emit_stdout(
                    f'Live trading stopped after {duration} seconds.',
                    emit_text=emit_text,
                )
        else:
            await monitored.start()

    _emit_stdout('Live trading session ended.', emit_text=emit_text)


async def run_live_paper_trading(
    data_source: LivePolyMarketDataSource
    | LiveNewsDataSource
    | LiveRSSNewsDataSource
    | DFlowDataSource,
    strategy: Strategy,
    initial_capital: Decimal,
    risk_manager: RiskManager | None = None,
    duration: float | None = None,
    state_store: StateStore | None = None,
    alerter: Alerter | None = None,
    continuous: bool = True,
    drawdown_alert_pct: Decimal | None = None,
    monitor: bool = False,
    exchange_name: str = '',
    emit_text: bool = True,
) -> None:
    """
    Run live paper trading (simulated) with the given configuration.

    Args:
        data_source: The live data source to use
        strategy: The trading strategy to execute
        initial_capital: Starting capital in USDC
        risk_manager: Optional risk manager (defaults to NoRiskManager)
        duration: Optional duration in seconds to run
        state_store: Optional state store for persistence and state recovery.
        alerter: Optional alerter for notifications.
        continuous: Keep engine running when the data source is temporarily idle.
        drawdown_alert_pct: Optional drawdown alert threshold as a decimal (0.1 = 10%).
    """
    market_data = MarketDataManager()
    position_manager = PositionManager()

    # State recovery: load saved positions if available.
    saved_positions = state_store.load_positions() if state_store else []
    if saved_positions:
        logger.info('Restoring %d positions from state store', len(saved_positions))
        for pos in saved_positions:
            position_manager.update_position(pos)
    else:
        position_manager.update_position(
            Position(
                ticker=CashTicker.POLYMARKET_USDC,
                quantity=initial_capital,
                average_cost=Decimal('0'),
                realized_pnl=Decimal('0'),
            )
        )

    if risk_manager is None:
        risk_manager = NoRiskManager()

    trader = PaperTrader(
        market_data=market_data,
        risk_manager=risk_manager,
        position_manager=position_manager,
        min_fill_rate=Decimal('0.5'),
        max_fill_rate=Decimal('1.0'),
        commission_rate=Decimal('0.0'),
        alerter=alerter,
    )

    await run_live_trading(
        data_source,
        strategy,
        trader,
        duration,
        state_store,
        alerter,
        continuous,
        drawdown_alert_pct,
        monitor=monitor,
        exchange_name=exchange_name,
        emit_text=emit_text,
    )

    # Print final portfolio status
    _emit_stdout('\n--- Final Portfolio Status ---', emit_text=emit_text)
    _emit_stdout(
        f'Cash positions: {position_manager.get_cash_positions()}',
        emit_text=emit_text,
    )
    _emit_stdout(
        f'Non-cash positions: {position_manager.get_non_cash_positions()}',
        emit_text=emit_text,
    )
    _emit_stdout(
        f'Total realized PnL: {position_manager.get_total_realized_pnl()}',
        emit_text=emit_text,
    )


async def run_live_polymarket_trading(
    data_source: LivePolyMarketDataSource,
    strategy: Strategy,
    wallet_private_key: str,
    signature_type: int,
    funder: str | None = None,
    risk_manager: RiskManager | None = None,
    duration: float | None = None,
    max_position_size: Decimal = Decimal('1000'),
    max_total_exposure: Decimal = Decimal('10000'),
    state_store: StateStore | None = None,
    alerter: Alerter | None = None,
    continuous: bool = True,
    drawdown_alert_pct: Decimal | None = None,
    monitor: bool = False,
    exchange_name: str = '',
    emit_text: bool = True,
) -> None:
    """
    Run live trading on Polymarket with real orders.

    Args:
        data_source: The Polymarket live data source
        strategy: The trading strategy to execute
        wallet_private_key: Private key for the trading wallet
        signature_type: Signature type for Polymarket API
        funder: Optional funder address for safe wallets
        risk_manager: Optional risk manager (defaults to StandardRiskManager)
        duration: Optional duration in seconds to run
        max_position_size: Maximum position size per trade
        max_total_exposure: Maximum total portfolio exposure
        state_store: Optional state store for persistence and state recovery.
        alerter: Optional alerter for notifications.
        continuous: Keep engine running when the data source is temporarily idle.
        drawdown_alert_pct: Optional drawdown alert threshold as a decimal (0.1 = 10%).
    """
    market_data = MarketDataManager()
    position_manager = PositionManager()

    if risk_manager is None:
        risk_manager = StandardRiskManager(
            position_manager=position_manager,
            market_data=market_data,
            max_position_size=max_position_size,
            max_total_exposure=max_total_exposure,
            max_single_trade_size=max_position_size / Decimal('2'),
            max_drawdown_pct=Decimal('0.2'),
        )

    if funder:
        public_account = funder
    else:
        from eth_account import Account

        public_account = Account.from_key(wallet_private_key).address
    trader = PolymarketTrader(
        market_data=market_data,
        risk_manager=risk_manager,
        position_manager=position_manager,
        wallet_private_key=wallet_private_key,
        signature_type=signature_type,
        funder=funder,
        alerter=alerter,
        **_live_journal_config('polymarket', public_account, state_store),
    )

    # Initialize USDC position: prefer live exchange balance for reconciliation.
    from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

    balance_info = trader.clob_client.get_balance_allowance(
        params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
    )
    live_balance = Decimal(balance_info['balance']) / Decimal('1000000')

    # State recovery: load saved positions if available, then reconcile cash.
    saved_positions = state_store.load_positions() if state_store else []
    if saved_positions:
        logger.info('Restoring %d positions from state store', len(saved_positions))
        for pos in saved_positions:
            position_manager.update_position(pos)
        # Reconcile cash position against live exchange balance.
        saved_cash = position_manager.get_position(CashTicker.POLYMARKET_USDC)
        if saved_cash is not None:
            diff_pct = abs(live_balance - saved_cash.quantity) / max(
                live_balance, Decimal('1')
            )
            if diff_pct > Decimal('0.01'):
                logger.warning(
                    'Saved cash %.4f differs from live balance %.4f by %.1f%% — using live value',
                    saved_cash.quantity,
                    live_balance,
                    float(diff_pct) * 100,
                )
                position_manager.update_position(
                    Position(
                        ticker=CashTicker.POLYMARKET_USDC,
                        quantity=live_balance,
                        average_cost=Decimal('0'),
                        realized_pnl=saved_cash.realized_pnl,
                    )
                )
        else:
            position_manager.update_position(
                Position(
                    ticker=CashTicker.POLYMARKET_USDC,
                    quantity=live_balance,
                    average_cost=Decimal('0'),
                    realized_pnl=Decimal('0'),
                )
            )
    else:
        position_manager.update_position(
            Position(
                ticker=CashTicker.POLYMARKET_USDC,
                quantity=live_balance,
                average_cost=Decimal('0'),
                realized_pnl=Decimal('0'),
            )
        )

    print(f'Starting live Polymarket trading with balance: {live_balance} USDC')

    await run_live_trading(
        data_source,
        strategy,
        trader,
        duration,
        state_store,
        alerter,
        continuous,
        drawdown_alert_pct,
        monitor=monitor,
        exchange_name=exchange_name,
    )

    # Print final portfolio status
    print('\n--- Final Portfolio Status ---')
    print(f'Cash positions: {position_manager.get_cash_positions()}')
    print(f'Non-cash positions: {position_manager.get_non_cash_positions()}')
    print(f'Total realized PnL: {position_manager.get_total_realized_pnl()}')


async def run_live_kalshi_paper_trading(
    data_source: LiveKalshiDataSource,
    strategy: Strategy,
    initial_capital: Decimal,
    risk_manager: RiskManager | None = None,
    duration: float | None = None,
    state_store: StateStore | None = None,
    alerter: Alerter | None = None,
    continuous: bool = True,
    drawdown_alert_pct: Decimal | None = None,
    monitor: bool = False,
    exchange_name: str = '',
    emit_text: bool = True,
) -> None:
    """
    Run live paper trading on Kalshi markets (simulated).

    Args:
        data_source: The Kalshi live data source
        strategy: The trading strategy to execute
        initial_capital: Starting capital in USD
        risk_manager: Optional risk manager (defaults to NoRiskManager)
        duration: Optional duration in seconds to run
        state_store: Optional state store for persistence and state recovery.
        alerter: Optional alerter for notifications.
        continuous: Keep engine running when the data source is temporarily idle.
        drawdown_alert_pct: Optional drawdown alert threshold as a decimal (0.1 = 10%).
    """
    market_data = MarketDataManager()
    position_manager = PositionManager()

    saved_positions = state_store.load_positions() if state_store else []
    if saved_positions:
        logger.info('Restoring %d positions from state store', len(saved_positions))
        for pos in saved_positions:
            position_manager.update_position(pos)
    else:
        position_manager.update_position(
            Position(
                ticker=CashTicker.KALSHI_USD,
                quantity=initial_capital,
                average_cost=Decimal('0'),
                realized_pnl=Decimal('0'),
            )
        )

    if risk_manager is None:
        risk_manager = NoRiskManager()

    trader = PaperTrader(
        market_data=market_data,
        risk_manager=risk_manager,
        position_manager=position_manager,
        min_fill_rate=Decimal('0.5'),
        max_fill_rate=Decimal('1.0'),
        commission_rate=Decimal('0.0'),
        alerter=alerter,
    )

    await run_live_trading(
        data_source,
        strategy,
        trader,
        duration,
        state_store,
        alerter,
        continuous,
        drawdown_alert_pct,
        monitor=monitor,
        exchange_name=exchange_name,
        emit_text=emit_text,
    )

    _emit_stdout('\n--- Final Portfolio Status ---', emit_text=emit_text)
    _emit_stdout(
        f'Cash positions: {position_manager.get_cash_positions()}',
        emit_text=emit_text,
    )
    _emit_stdout(
        f'Non-cash positions: {position_manager.get_non_cash_positions()}',
        emit_text=emit_text,
    )
    _emit_stdout(
        f'Total realized PnL: {position_manager.get_total_realized_pnl()}',
        emit_text=emit_text,
    )


async def run_live_kalshi_trading(
    data_source: LiveKalshiDataSource,
    strategy: Strategy,
    api_key_id: str | None = None,
    private_key_path: str | None = None,
    risk_manager: RiskManager | None = None,
    duration: float | None = None,
    max_position_size: Decimal = Decimal('1000'),
    max_total_exposure: Decimal = Decimal('10000'),
    state_store: StateStore | None = None,
    alerter: Alerter | None = None,
    continuous: bool = True,
    drawdown_alert_pct: Decimal | None = None,
    monitor: bool = False,
    exchange_name: str = '',
) -> None:
    """
    Run live trading on Kalshi with real orders.

    Args:
        data_source: The Kalshi live data source
        strategy: The trading strategy to execute
        api_key_id: Kalshi API key ID (or set KALSHI_API_KEY_ID env)
        private_key_path: Path to RSA private key PEM file (or set KALSHI_PRIVATE_KEY_PATH env)
        risk_manager: Optional risk manager (defaults to StandardRiskManager)
        duration: Optional duration in seconds to run
        max_position_size: Maximum position size per trade
        max_total_exposure: Maximum total portfolio exposure
        state_store: Optional state store for persistence and state recovery.
        alerter: Optional alerter for notifications.
        continuous: Keep engine running when the data source is temporarily idle.
        drawdown_alert_pct: Optional drawdown alert threshold as a decimal (0.1 = 10%).
    """
    market_data = MarketDataManager()
    position_manager = PositionManager()

    if risk_manager is None:
        risk_manager = StandardRiskManager(
            position_manager=position_manager,
            market_data=market_data,
            max_position_size=max_position_size,
            max_total_exposure=max_total_exposure,
            max_single_trade_size=max_position_size / Decimal('2'),
            max_drawdown_pct=Decimal('0.2'),
        )

    trader = KalshiTrader(
        market_data=market_data,
        risk_manager=risk_manager,
        position_manager=position_manager,
        api_key_id=api_key_id,
        private_key_path=private_key_path,
        alerter=alerter,
        **_live_journal_config(
            'kalshi', api_key_id or os.environ.get('KALSHI_API_KEY_ID', ''), state_store
        ),
    )

    # Fetch initial balance from Kalshi (via the trader's portfolio API)
    balance_response = await asyncio.to_thread(
        lambda: trader._portfolio_api.get_balance()
    )
    # Balance is in cents
    initial_balance = Decimal(str(balance_response.balance)) / Decimal('100')

    saved_positions = state_store.load_positions() if state_store else []
    if saved_positions:
        logger.info('Restoring %d positions from state store', len(saved_positions))
        for pos in saved_positions:
            position_manager.update_position(pos)
        saved_cash = position_manager.get_position(CashTicker.KALSHI_USD)
        if saved_cash is not None:
            diff_pct = abs(initial_balance - saved_cash.quantity) / max(
                initial_balance, Decimal('1')
            )
            if diff_pct > Decimal('0.01'):
                logger.warning(
                    'Saved cash %.2f differs from live balance %.2f by %.1f%% — using live value',
                    saved_cash.quantity,
                    initial_balance,
                    float(diff_pct) * 100,
                )
                position_manager.update_position(
                    Position(
                        ticker=CashTicker.KALSHI_USD,
                        quantity=initial_balance,
                        average_cost=Decimal('0'),
                        realized_pnl=saved_cash.realized_pnl,
                    )
                )
        else:
            position_manager.update_position(
                Position(
                    ticker=CashTicker.KALSHI_USD,
                    quantity=initial_balance,
                    average_cost=Decimal('0'),
                    realized_pnl=Decimal('0'),
                )
            )
    else:
        position_manager.update_position(
            Position(
                ticker=CashTicker.KALSHI_USD,
                quantity=initial_balance,
                average_cost=Decimal('0'),
                realized_pnl=Decimal('0'),
            )
        )

    logger.info('Starting live Kalshi trading with balance: $%s', initial_balance)

    await run_live_trading(
        data_source,
        strategy,
        trader,
        duration,
        state_store,
        alerter,
        continuous,
        drawdown_alert_pct,
        monitor=monitor,
        exchange_name=exchange_name,
    )

    logger.info('--- Final Portfolio Status ---')
    logger.info('Cash positions: %s', position_manager.get_cash_positions())
    logger.info('Non-cash positions: %s', position_manager.get_non_cash_positions())
    logger.info('Total realized PnL: %s', position_manager.get_total_realized_pnl())


async def run_live_solana_paper_trading(
    data_source: DFlowDataSource,
    strategy: Strategy,
    initial_capital: Decimal,
    risk_manager: RiskManager | None = None,
    duration: float | None = None,
    state_store: StateStore | None = None,
    alerter: Alerter | None = None,
    continuous: bool = True,
    drawdown_alert_pct: Decimal | None = None,
    monitor: bool = False,
    exchange_name: str = '',
    emit_text: bool = True,
) -> None:
    """Run live paper trading on Solana/DFlow markets (simulated)."""
    market_data = MarketDataManager()
    position_manager = PositionManager()

    saved_positions = state_store.load_positions() if state_store else []
    if saved_positions:
        logger.info('Restoring %d positions from state store', len(saved_positions))
        for pos in saved_positions:
            position_manager.update_position(pos)
    else:
        position_manager.update_position(
            Position(
                ticker=CashTicker.DFLOW_USDC,
                quantity=initial_capital,
                average_cost=Decimal('0'),
                realized_pnl=Decimal('0'),
            )
        )

    if risk_manager is None:
        risk_manager = NoRiskManager()

    trader = PaperTrader(
        market_data=market_data,
        risk_manager=risk_manager,
        position_manager=position_manager,
        min_fill_rate=Decimal('0.5'),
        max_fill_rate=Decimal('1.0'),
        commission_rate=Decimal('0.0'),
        alerter=alerter,
    )

    await run_live_trading(
        data_source,
        strategy,
        trader,
        duration,
        state_store,
        alerter,
        continuous,
        drawdown_alert_pct,
        monitor=monitor,
        exchange_name=exchange_name,
        emit_text=emit_text,
    )

    _emit_stdout('\n--- Final Portfolio Status ---', emit_text=emit_text)
    _emit_stdout(
        f'Cash positions: {position_manager.get_cash_positions()}',
        emit_text=emit_text,
    )
    _emit_stdout(
        f'Non-cash positions: {position_manager.get_non_cash_positions()}',
        emit_text=emit_text,
    )
    _emit_stdout(
        f'Total realized PnL: {position_manager.get_total_realized_pnl()}',
        emit_text=emit_text,
    )


async def run_live_solana_trading(
    data_source: DFlowDataSource,
    strategy: Strategy,
    keypair_path: str | None = None,
    rpc_url: str = 'https://api.mainnet-beta.solana.com',
    risk_manager: RiskManager | None = None,
    duration: float | None = None,
    max_position_size: Decimal = Decimal('1000'),
    max_total_exposure: Decimal = Decimal('10000'),
    state_store: StateStore | None = None,
    alerter: Alerter | None = None,
    continuous: bool = True,
    drawdown_alert_pct: Decimal | None = None,
    monitor: bool = False,
    exchange_name: str = '',
    initial_capital: Decimal = Decimal('1000'),
    emit_text: bool = True,
) -> None:
    """Run live trading on Solana/DFlow with real orders."""
    market_data = MarketDataManager()
    position_manager = PositionManager()

    if risk_manager is None:
        risk_manager = StandardRiskManager(
            position_manager=position_manager,
            market_data=market_data,
            max_position_size=max_position_size,
            max_total_exposure=max_total_exposure,
            max_single_trade_size=max_position_size / Decimal('2'),
            max_drawdown_pct=Decimal('0.2'),
        )

    trader = SolanaTrader(
        market_data=market_data,
        risk_manager=risk_manager,
        position_manager=position_manager,
        keypair_path=keypair_path,
        rpc_url=rpc_url,
        alerter=alerter,
    )

    # Reconcile on-chain USDC balance with internal position manager.
    live_balance = await trader.reconcile_balance()

    # Initialize USDC position
    saved_positions = state_store.load_positions() if state_store else []
    if saved_positions:
        logger.info('Restoring %d positions from state store', len(saved_positions))
        for pos in saved_positions:
            position_manager.update_position(pos)
        # Reconcile cash position against live on-chain balance.
        saved_cash = position_manager.get_position(CashTicker.DFLOW_USDC)
        if saved_cash is not None:
            diff_pct = abs(live_balance - saved_cash.quantity) / max(
                live_balance, Decimal('1')
            )
            if diff_pct > Decimal('0.01'):
                logger.warning(
                    'Saved cash %.4f differs from live balance %.4f by %.1f%% — using live value',
                    saved_cash.quantity,
                    live_balance,
                    float(diff_pct) * 100,
                )
                position_manager.update_position(
                    Position(
                        ticker=CashTicker.DFLOW_USDC,
                        quantity=live_balance,
                        average_cost=Decimal('0'),
                        realized_pnl=saved_cash.realized_pnl,
                    )
                )
        else:
            position_manager.update_position(
                Position(
                    ticker=CashTicker.DFLOW_USDC,
                    quantity=live_balance,
                    average_cost=Decimal('0'),
                    realized_pnl=Decimal('0'),
                )
            )
    else:
        position_manager.update_position(
            Position(
                ticker=CashTicker.DFLOW_USDC,
                quantity=live_balance,
                average_cost=Decimal('0'),
                realized_pnl=Decimal('0'),
            )
        )

    logger.info(
        'Starting live Solana trading with wallet: %s (balance: %s USDC)',
        trader.public_key,
        live_balance,
    )

    await run_live_trading(
        data_source,
        strategy,
        trader,
        duration,
        state_store,
        alerter,
        continuous,
        drawdown_alert_pct,
        monitor=monitor,
        exchange_name=exchange_name,
        emit_text=emit_text,
    )

    _emit_stdout('\n--- Final Portfolio Status ---', emit_text=emit_text)
    _emit_stdout(
        f'Cash positions: {position_manager.get_cash_positions()}',
        emit_text=emit_text,
    )
    _emit_stdout(
        f'Non-cash positions: {position_manager.get_non_cash_positions()}',
        emit_text=emit_text,
    )
    _emit_stdout(
        f'Total realized PnL: {position_manager.get_total_realized_pnl()}',
        emit_text=emit_text,
    )


if __name__ == '__main__':
    from oracle3.strategy.test_strategy import TestStrategy

    async def main():
        # Example: Run paper trading with RSS news data
        data_source = LiveRSSNewsDataSource(
            polling_interval=60.0,
            max_articles_per_poll=5,
        )

        await run_live_paper_trading(
            data_source=data_source,
            strategy=TestStrategy(),
            initial_capital=Decimal('10000'),
            duration=300,  # Run for 5 minutes
        )

    asyncio.run(main())
