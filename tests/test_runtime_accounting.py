from datetime import datetime, timezone
from decimal import Decimal as D

import pytest

from oracle3.core.trading_engine import TradingEngine
from oracle3.data.data_source import DataSource
from oracle3.data.market_data_manager import MarketDataManager
from oracle3.events.events import NewsEvent
from oracle3.order.order_book import Level, OrderBook
from oracle3.position.position_manager import Position, PositionManager
from oracle3.risk.risk_manager import NoRiskManager
from oracle3.storage.state_store import StateStore
from oracle3.strategy.strategy import Strategy
from oracle3.ticker.ticker import CashTicker, PolyMarketTicker, SolanaTicker
from oracle3.trader.paper_trader import PaperTrader
from oracle3.trader.types import TradeSide

TICKER = PolyMarketTicker(symbol='RUNTIME', token_id='runtime')


class Source(DataSource):
    async def get_next_event(self):
        return None


class Idle(Strategy):
    def __init__(self):
        self.events = 0

    async def process_event(self, event, trader):
        self.events += 1


def setup_trader(positions=None):
    pm = PositionManager()
    for p in positions or [Position(CashTicker.POLYMARKET_USDC, D('100'), D(0), D(0))]:
        pm.update_position(p)
    md = MarketDataManager()
    book = OrderBook()
    book.update(bids=[Level(D('.5'), D(100))], asks=[Level(D('.5'), D(100))])
    md.order_books[TICKER] = book
    return PaperTrader(
        md,
        NoRiskManager(),
        pm,
        min_fill_rate=D(1),
        max_fill_rate=D(1),
        commission_rate=D(0),
    )


@pytest.mark.asyncio
async def test_engine_marks_nav_then_closes_losing_roundtrip_and_resumes(tmp_path):
    trader = setup_trader()
    store = StateStore(tmp_path)
    engine = TradingEngine(Source(), Idle(), trader, state_store=store)
    engine._event_timestamp = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
    await trader.place_order(TradeSide.BUY, TICKER, D('.5'), D(10))
    await engine._sync_trades()
    engine._mark_performance()
    first = engine.get_snapshot()
    assert first.cash == D(95) and first.equity == D(100)
    assert first.total_pnl == 0 and first.valuation_complete
    assert not first.sharpe_available
    trader.market_data.order_books[TICKER].update(
        bids=[Level(D('.4'), D(100))], asks=[]
    )
    engine._mark_performance()
    assert engine.get_snapshot().total_pnl == D(-1)
    await trader.place_order(TradeSide.SELL, TICKER, D('.4'), D(10))
    await engine._sync_trades()
    engine._mark_performance()
    stats = engine._perf.get_stats()
    assert stats.total_fills == 2 and stats.total_trades == 1
    assert stats.winning_trades == 0 and stats.losing_trades == 1
    assert stats.total_pnl == D(-1) and stats.cash == D(99)
    await engine.stop()
    resumed = TradingEngine(
        Source(), Idle(), setup_trader(store.load_positions()), state_store=store
    )
    assert resumed._perf.get_stats() == stats
    assert resumed._perf.equity_curve[1].timestamp == engine._event_timestamp


@pytest.mark.asyncio
async def test_opening_inventory_is_not_a_fill_and_missing_mark_is_not_zero(tmp_path):
    positions = [
        Position(CashTicker.POLYMARKET_USDC, D(95), D(0), D(0)),
        Position(TICKER, D(10), D('.5'), D(0)),
    ]
    trader = setup_trader(positions)
    trader.market_data.order_books.clear()
    engine = TradingEngine(Source(), Idle(), trader)
    engine._mark_performance()
    assert engine.get_snapshot().equity == D(100)
    assert not engine.get_snapshot().valuation_complete
    assert engine.get_snapshot().missing_marks == ('RUNTIME',)
    assert engine._perf.get_stats().total_fills == 0
    trader.market_data.order_books[TICKER] = OrderBook()
    trader.market_data.order_books[TICKER].update(
        bids=[Level(D('.4'), D(100))], asks=[]
    )
    await trader.place_order(TradeSide.SELL, TICKER, D('.4'), D(10))
    await engine._sync_trades()
    engine._mark_performance()
    assert engine._perf.get_stats().total_pnl == D(-1)


@pytest.mark.asyncio
async def test_reconciliation_blocks_strategy_but_keeps_polling():
    trader = setup_trader()
    strategy = Idle()
    engine = TradingEngine(Source(), strategy, trader)
    polls = []

    async def fail_reconcile():
        polls.append(True)
        raise RuntimeError('offline')

    trader.reconcile_pending_orders = fail_reconcile
    await engine._process_one_event(NewsEvent(news='one'))
    assert strategy.events == 0 and engine.get_snapshot().reconciliation_required
    engine._last_reconciliation_poll = float('-inf')
    await engine.start()  # Empty feed still gives reconciliation a chance.
    assert len(polls) == 2

    async def success_reconcile():
        return []

    trader.reconcile_pending_orders = success_reconcile
    engine._last_reconciliation_poll = float('-inf')
    await engine._process_one_event(NewsEvent(news='two'))
    assert strategy.events == 1 and not engine.get_snapshot().reconciliation_required


@pytest.mark.asyncio
async def test_checkpoint_mismatch_and_corruption_fail_closed(tmp_path):
    store = StateStore(tmp_path)
    trader = setup_trader()
    engine = TradingEngine(Source(), Idle(), trader, state_store=store)
    await engine.stop()
    trader.position_manager.get_position(CashTicker.POLYMARKET_USDC).quantity -= D(1)
    with pytest.raises(ValueError, match='differ'):
        TradingEngine(Source(), Idle(), trader, state_store=store)
    (tmp_path / 'runtime.json').write_text('{broken')
    with pytest.raises(ValueError, match='Cannot recover'):
        store.load_positions()
    assert (tmp_path / 'runtime.json').read_text() == '{broken'


def test_solana_position_serialization_preserves_outcome_mints(tmp_path):
    ticker = SolanaTicker(symbol='CHAIN', yes_mint='yes', no_mint='no', is_no_side=True)
    pm = PositionManager()
    pm.update_position(Position(ticker, D(2), D('.6'), D(0)))
    store = StateStore(tmp_path)
    store.save_positions(pm)
    assert store.load_positions()[0].ticker == ticker


@pytest.mark.asyncio
async def test_cancelled_notification_does_not_replay_accounted_fill():
    import asyncio

    class CancelAlert:
        async def on_trade(self, trade):
            raise asyncio.CancelledError

    trader = setup_trader()
    engine = TradingEngine(Source(), Idle(), trader, alerter=CancelAlert())
    await trader.place_order(TradeSide.BUY, TICKER, D('.5'), D(10))
    with pytest.raises(asyncio.CancelledError):
        await engine._sync_trades()
    engine._alerter = None
    await engine._sync_trades()
    assert engine._perf.get_stats().total_fills == 1
    assert engine._perf.cash == D(95)
    assert engine._perf.position_quantity(TICKER) == D(10)


@pytest.mark.asyncio
async def test_failed_checkpoint_retries_without_new_order(tmp_path, monkeypatch):
    store = StateStore(tmp_path)
    trader = setup_trader()
    engine = TradingEngine(Source(), Idle(), trader, state_store=store)
    save = store.save_all
    attempts = []

    def flaky_save(*args):
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError('disk full')
        return save(*args)

    monkeypatch.setattr(store, 'save_all', flaky_save)
    await trader.place_order(TradeSide.BUY, TICKER, D('.5'), D(10))
    assert not await engine._reconcile_orders()
    assert engine._checkpoint_dirty
    engine._last_reconciliation_poll = float('-inf')
    assert await engine._reconcile_orders()
    assert len(attempts) == 2 and not engine._checkpoint_dirty
    assert store.load_performance().get_stats().total_fills == 1


def test_idle_valuation_does_not_append_duplicate_history():
    engine = TradingEngine(Source(), Idle(), setup_trader())
    engine._mark_performance()
    count = len(engine._perf.equity_curve)
    for _ in range(100):
        engine._mark_performance()
    assert len(engine._perf.equity_curve) == count


def test_journal_is_durable_and_scoped_to_public_account(tmp_path):
    from oracle3.live.live_trader import _live_journal_config

    store = StateStore(tmp_path)
    first = _live_journal_config('kalshi', 'public-id-1', store)
    second = _live_journal_config('kalshi', 'public-id-2', store)
    assert first == _live_journal_config('kalshi', 'public-id-1', store)
    assert first != second
    assert first['journal_path'].parent == tmp_path
    assert 'public-id-1' not in str(first)


@pytest.mark.asyncio
async def test_paired_checkpoint_preserves_public_read_api_and_rejects_split_writes(
    tmp_path,
):
    trader = setup_trader()
    store = StateStore(tmp_path)
    engine = TradingEngine(Source(), Idle(), trader, state_store=store)
    await trader.place_order(TradeSide.BUY, TICKER, D('.5'), D(10))
    trade = trader.orders[-1].trades[0]
    trade.execution_id, trade.client_order_id, trade.venue_order_id = (
        'fill-1',
        'intent-1',
        'venue-1',
    )
    trade.fee_source = 'fixture'
    await engine._sync_trades()
    restored = store.load_performance()
    assert restored.trades[0].to_payload() == trade.to_payload()
    assert store.load_equity_curve() == engine._perf.equity_curve
    with pytest.raises(ValueError, match='use save_all'):
        store.save_positions(trader.position_manager)
    with pytest.raises(ValueError, match='use save_all'):
        store.save_equity_curve([])


def test_pending_journal_order_visible_without_an_accounted_fill(tmp_path):
    from dataclasses import asdict

    trader = setup_trader()
    trader._init_order_journal(tmp_path / 'orders.sqlite3', 'test')
    trader.order_journal.reserve(
        'intent-pending',
        {
            'ticker_type': 'PolyMarketTicker',
            'ticker': asdict(TICKER),
            'side': 'buy',
            'limit_price': '.5',
            'quantity': '10',
        },
    )
    trader.order_journal.update(
        'intent-pending',
        venue_order_id='venue-pending',
        result={
            'status': 'placed',
            'filled_quantity': '0',
        },
    )
    snapshot = TradingEngine(Source(), Idle(), trader).get_snapshot()
    assert snapshot.reconciliation_required
    assert snapshot.reconciliation_reason
    assert len(snapshot.active_orders) == 1
    pending = snapshot.active_orders[0]
    assert pending.client_order_id == 'intent-pending'
    assert pending.venue_order_id == 'venue-pending'
    assert pending.filled_quantity == 0 and pending.quantity == 10
    assert pending.status == 'placed'
