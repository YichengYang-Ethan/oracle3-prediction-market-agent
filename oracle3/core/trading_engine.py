"""Trading engine with snapshot support for live monitoring."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from oracle3.analytics.performance_analyzer import PerformanceAnalyzer
from oracle3.data.data_source import DataSource
from oracle3.events.events import (
    Event,
    NewsEvent,
    OnChainSignalEvent,
    OrderBookEvent,
    PriceChangeEvent,
)
from oracle3.risk.risk_manager import StandardRiskManager
from oracle3.strategy.strategy import Strategy
from oracle3.ticker.ticker import CashTicker, Ticker
from oracle3.trader.trader import Trader
from oracle3.trader.types import OrderStatus

if TYPE_CHECKING:
    from oracle3.alerts.alerter import Alerter
    from oracle3.storage.state_store import StateStore

logger = logging.getLogger(__name__)

# Maximum consecutive ``None`` events before the engine warns (continuous mode).
_MAX_CONSECUTIVE_NONE = 120  # ~2 min at 1 s timeout per poll


# ---------------------------------------------------------------------------
# Snapshot data-classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionSnapshot:
    ticker_symbol: str
    ticker_name: str
    quantity: Decimal
    average_cost: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal


@dataclass(frozen=True)
class OrderBookSnapshot:
    ticker_symbol: str
    bids: list[tuple[Decimal, Decimal]]  # [(price, size), ...]
    asks: list[tuple[Decimal, Decimal]]


@dataclass(frozen=True)
class TradeSnapshot:
    time: str
    side: str
    ticker_symbol: str
    ticker_name: str
    price: Decimal
    quantity: Decimal
    status: str


@dataclass(frozen=True)
class OrderSnapshot:
    side: str
    ticker_symbol: str
    ticker_name: str
    limit_price: Decimal
    quantity: Decimal
    filled_quantity: Decimal
    status: str
    venue_order_id: str = ''
    client_order_id: str = ''


@dataclass(frozen=True)
class EngineSnapshot:
    """Immutable point-in-time copy of the engine / portfolio state."""

    # ---- header metrics ----
    equity: Decimal
    cash: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_pnl: Decimal
    sharpe: Decimal
    max_drawdown_pct: Decimal
    current_drawdown_pct: Decimal
    exposure: Decimal
    exposure_pct: float
    uptime: str
    event_count: int
    engine_running: bool

    # ---- collections ----
    positions: list[PositionSnapshot]
    orderbooks: list[OrderBookSnapshot]
    recent_trades: list[TradeSnapshot]
    active_orders: list[OrderSnapshot]
    news_headlines: list[dict[str, str]]  # [{timestamp,title,source,url}, ...]
    valuation_complete: bool = True
    missing_marks: tuple[str, ...] = ()
    sharpe_available: bool = False
    reconciliation_required: bool = False
    reconciliation_reason: str = ''


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TradingEngine:
    def __init__(
        self,
        data_source: DataSource,
        strategy: Strategy,
        trader: Trader,
        *,
        initial_capital: Decimal = Decimal('10000'),
        continuous: bool = False,
        state_store: StateStore | None = None,
        alerter: Alerter | None = None,
        drawdown_alert_pct: Decimal | None = None,
    ) -> None:
        self.data_source = data_source
        self.strategy = strategy
        self.trader = trader
        self.market_data = trader.market_data
        self.running = False

        # When *continuous* is True the engine keeps looping even when
        # ``get_next_event()`` returns ``None`` (live data sources use
        # ``None`` to signal "nothing right now", not "data exhausted").
        self._continuous = continuous

        # --- persistence & alerting ---
        self._state_store = state_store
        self._alerter = alerter
        self._drawdown_alert_pct = drawdown_alert_pct
        self._drawdown_alerted: bool = False  # avoid repeated alerts
        self._SAVE_INTERVAL = 100  # save every N events
        self._save_event_counter: int = 0
        self._consecutive_processing_errors: int = 0
        self._MAX_CONSECUTIVE_ERRORS: int = 5
        self._degraded_read_only: bool = False

        # --- monitoring helpers ---
        self._start_time: datetime | None = None
        self._event_count: int = 0
        self._last_orders_idx: int = len(trader.orders)
        self._last_order_trade_idx = 0
        self._checkpoint_dirty = False
        self._order_times: list[str] = []
        self._perf = self._initialize_performance(initial_capital)
        self._event_timestamp: int | None = None
        self._reconciliation_blocked = False
        self._last_reconciliation_poll = float('-inf')
        self._last_mark_signature: tuple | None = None
        self._news: deque[dict[str, str]] = deque(maxlen=300)
        self._activity_log: deque[tuple[str, str]] = deque(maxlen=100)
        self._last_decisions_count: int = 0

        # [H3] Guard: prevent calling data_source.start() more than once.
        self._ds_started: bool = False

        # [M4] Periodic stale order book pruning interval (by event count).
        self._last_prune_event: int = 0
        self._PRUNE_INTERVAL = 500

        # [P1] Deferred news events: collected during market-event draining.
        self._deferred_news: deque[NewsEvent] = deque(maxlen=500)

        # [D1] Data-flow pause flag: when True the event loop sleeps instead
        # of calling data_source.get_next_event().  Set by ControlServer.
        self._data_paused: bool = False

        # Reputation manager (optional, set externally)
        self._reputation_manager: Any | None = None

        # Flash loan and atomic trader instances (optional, set externally)
        self._flash_loan: Any | None = None
        self._atomic_trader: Any | None = None

    def _initialize_performance(self, initial_capital: Decimal) -> PerformanceAnalyzer:
        pm = self.trader.position_manager
        cash_positions = pm.get_cash_positions()
        cash = sum((p.quantity for p in cash_positions), Decimal('0'))
        positions = [p for p in pm.get_non_cash_positions() if p.quantity != 0]
        restored = self._state_store.load_performance() if self._state_store else None
        if restored is not None:
            held = {p.ticker: p.quantity for p in positions}
            tracked = {t.ticker for t in restored.trades} | set(held)
            # Marks include seeded holdings which may not have any fill records.
            tracked.update(restored.inventory_tickers())
            if restored.cash != cash or any(
                restored.position_quantity(t) != held.get(t, Decimal('0'))
                for t in tracked
            ):
                raise ValueError(
                    'Saved accounting and account positions differ; reconcile before restart'
                )
            return restored
        if cash_positions or positions:
            initial_capital = cash + sum(
                (p.quantity * p.average_cost for p in positions), Decimal('0')
            )
        perf = PerformanceAnalyzer(initial_capital=initial_capital)
        for pos in positions:
            perf.seed_position(pos.ticker, pos.quantity, pos.average_cost)
        if positions:
            logger.warning(
                'No accounting checkpoint: opening holdings establish a new session cost baseline'
            )
        return perf

    def _mark_performance(self) -> None:
        prices: dict[Ticker, Decimal] = {}
        for ticker in self._perf.inventory_tickers():
            bid = self.market_data.get_best_bid(ticker)
            ask = self.market_data.get_best_ask(ticker) if bid is None else None
            level = bid if bid is not None else ask
            if level is not None:
                prices[ticker] = level.price
        signature = (
            self._event_timestamp,
            self._perf.cash,
            frozenset(
                (t, self._perf.position_quantity(t))
                for t in self._perf.inventory_tickers()
            ),
            frozenset(prices.items()),
        )
        if signature != self._last_mark_signature:
            self._perf.mark_to_market(prices, timestamp=self._event_timestamp)
            self._last_mark_signature = signature

    async def _reconcile_orders(self) -> bool:
        """Poll pending orders even with no market events; never guess fills."""
        try:
            now = time.monotonic()
            if now - self._last_reconciliation_poll >= 1:
                self._last_reconciliation_poll = now
                await self.trader.reconcile_pending_orders()
                self._reconciliation_blocked = False
            await self._sync_trades()
            blocked = (
                self._reconciliation_blocked or self.trader.reconciliation_required
            )
        except Exception:
            logger.exception(
                'Order reconciliation failed; strategy execution suspended'
            )
            blocked = True
        if blocked != self._reconciliation_blocked:
            self._activity_log.append(
                (
                    datetime.now().strftime('%H:%M:%S'),
                    'Order reconciliation required'
                    if blocked
                    else 'Order reconciliation complete',
                )
            )
        self._reconciliation_blocked = blocked
        return not blocked

    def _drain_backtest_timestamp_batch(self, event: Event) -> list[Event]:
        """Return all queued backtest events that share a timestamp."""
        if self._continuous:
            return [event]

        drain = getattr(self.data_source, 'drain_same_timestamp_events', None)
        if not callable(drain):
            return [event]

        try:
            drained = drain(event)
        except Exception:
            logger.debug(
                'Failed to drain same-timestamp backtest events',
                exc_info=True,
            )
            return [event]

        if not drained:
            return [event]
        return [event, *drained]

    def _apply_market_event(self, event: object) -> None:
        if isinstance(event, OrderBookEvent):
            self.market_data.process_orderbook_event(event)
        elif isinstance(event, PriceChangeEvent):
            self.market_data.process_price_change_event(event)
            logger.debug('Price %s → %s', event.ticker.symbol, event.price)

    def _log_event_milestone(self) -> None:
        if self._event_count in (1, 10, 50, 100) or self._event_count % 500 == 0:
            now_str = datetime.now().strftime('%H:%M:%S')
            ob_count = len(self.market_data.order_books)
            self._activity_log.append(
                (
                    now_str,
                    f'Milestone: {self._event_count} events processed, {ob_count} order books',
                )
            )

    async def _process_one_event(
        self,
        event: Event,
        *,
        market_data_already_applied: bool = False,
    ) -> None:
        self._event_count += 1
        self._log_event_milestone()
        timestamp = getattr(event, 'timestamp', None)
        self._event_timestamp = (
            int(timestamp.timestamp()) if isinstance(timestamp, datetime) else None
        )

        try:
            now_str = datetime.now().strftime('%H:%M:%S')

            if not market_data_already_applied:
                self._apply_market_event(event)
            self._mark_performance()
            if not await self._reconcile_orders():
                self._mark_performance()
                return

            if isinstance(event, OnChainSignalEvent):
                self._activity_log.append(
                    (
                        now_str,
                        f'OnChainSignal: {event.signal_type} {event.direction} {event.amount} ({event.wallet[:8]}...)',
                    )
                )
                logger.info(
                    'OnChainSignalEvent: %s %s %s',
                    event.signal_type,
                    event.direction,
                    event.amount,
                )

            if isinstance(event, NewsEvent):
                headline = event.title or event.news[:100]
                source = getattr(event, 'source', '') or ''
                url = getattr(event, 'url', '') or ''
                self.trader.record_news(
                    timestamp=now_str,
                    title=headline,
                    source=source,
                    url=url,
                )
                self._news.append(
                    {
                        'timestamp': now_str,
                        'title': headline,
                        'source': source,
                        'url': url,
                    }
                )
                self._activity_log.append(
                    (now_str, f'News [{source[:15]}] "{headline[:55]}"')
                )
                logger.info('NewsEvent: %s', headline)

            prev_orders = len(self.trader.orders)
            self.strategy.bind_context(event, self.trader)
            await self.strategy.process_event(event, self.trader)

            decisions = self.strategy.get_decisions()
            stats = self.strategy.get_decision_stats()
            total_d = int(stats.get('decisions', len(decisions)))
            if total_d > self._last_decisions_count:
                new_count = total_d - self._last_decisions_count
                start_idx = max(0, len(decisions) - new_count)
                for i in range(start_idx, len(decisions)):
                    d = decisions[i]
                    exec_mark = 'TRADED' if d.executed else 'no trade'
                    signals = d.signal_values or {}
                    signal_pairs = list(signals.items())[:2]
                    signal_str = (
                        ' '.join(
                            f'{k}={v:.3f}'
                            if isinstance(v, (int, float))
                            else f'{k}={v}'
                            for k, v in signal_pairs
                        )
                        if signal_pairs
                        else ''
                    )
                    name = d.ticker_name[:30]
                    self._activity_log.append(
                        (now_str, f'{d.action} {signal_str} [{exec_mark}] "{name}"')
                    )
                self._last_decisions_count = total_d

            await self._sync_trades()
            self._mark_performance()
            await self._maybe_write_reputation()
            new_orders = self.trader.orders[prev_orders:]
            for order in new_orders:
                status = order.status.value.upper()
                side = order.side.value.upper()
                ticker_name = (
                    getattr(order.ticker, 'name', '') or order.ticker.symbol[:25]
                )
                self._activity_log.append(
                    (
                        now_str,
                        f'{side} {order.filled_quantity} @ ${order.average_price:.4f} → {status} "{ticker_name[:25]}"',
                    )
                )
                if hasattr(order.ticker, 'token_id'):
                    if order.side.value.upper() == 'BUY':
                        watch = getattr(self.data_source, 'watch_token', None)
                        if watch:
                            watch(order.ticker.token_id)
                            no_token_id = getattr(order.ticker, 'no_token_id', '')
                            if no_token_id:
                                watch(no_token_id)
                    elif order.side.value.upper() == 'SELL':
                        pos = self.trader.position_manager.get_position(order.ticker)
                        if pos is None or pos.quantity <= 0:
                            unwatch = getattr(self.data_source, 'unwatch_token', None)
                            if unwatch:
                                unwatch(order.ticker.token_id)
        except Exception as exc:
            now_str = datetime.now().strftime('%H:%M:%S')
            self._activity_log.append(
                (now_str, f'Error processing event #{self._event_count}')
            )
            logger.exception('Error processing event #%d', self._event_count)
            self._consecutive_processing_errors += 1
            if self._alerter:
                try:
                    await self._alerter.on_error(exc)
                except Exception:
                    logger.debug('alerter.on_error() failed', exc_info=True)
            await self._auto_degrade_if_needed()
        else:
            self._consecutive_processing_errors = 0

        self._save_event_counter += 1
        if self._save_event_counter >= self._SAVE_INTERVAL:
            self._save_event_counter = 0
            if self._state_store:
                try:
                    self._state_store.save_all(self.trader.position_manager, self._perf)
                except Exception:
                    logger.debug(
                        'periodic state_store.save_all() failed',
                        exc_info=True,
                    )
            await self._check_drawdown_alert()
            await self._check_portfolio_health()

        if self._event_count - self._last_prune_event >= self._PRUNE_INTERVAL:
            self._last_prune_event = self._event_count
            known = getattr(self.data_source, '_known_tickers', None)
            if known is None:
                for src in getattr(self.data_source, 'sources', []):
                    known = getattr(src, '_known_tickers', None)
                    if known is not None:
                        break
            if known is not None:
                removed = self.market_data.prune_stale_tickers(set(known.keys()))
                if removed:
                    logger.info(
                        'Pruned %d stale order books from MarketDataManager',
                        removed,
                    )

    # ------------------------------------------------------------------ #
    # Main loop                                                           #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:  # noqa: C901
        self._start_time = datetime.now()
        self.running = True

        # [H3] Start the data source exactly once.
        if not self._ds_started:
            self._ds_started = True
            await self.data_source.start()

        if self._alerter:
            try:
                await self._alerter.on_engine_start()
            except Exception:
                logger.debug('alerter.on_engine_start() failed', exc_info=True)

        logger.info('TradingEngine started (continuous=%s)', self._continuous)

        # [M1] Track consecutive None events so we can warn on prolonged
        # silence instead of silently spinning.
        consecutive_none = 0

        while self.running:
            # [D1] When data flow is paused, sleep instead of polling the source.
            if self._data_paused:
                await self._reconcile_orders()
                self._mark_performance()
                await asyncio.sleep(0.5)
                continue

            try:
                event = await self.data_source.get_next_event()
            except Exception:
                logger.exception('Error fetching next event')
                # [C1] Back off on repeated fetch errors to avoid CPU spin.
                await asyncio.sleep(1.0)
                continue

            if event is None:
                await self._reconcile_orders()
                self._mark_performance()
                # [P1] Process deferred news when the queue is idle.
                if self._deferred_news:
                    event = self._deferred_news.popleft()
                elif self._continuous:
                    await asyncio.sleep(0.1)
                    consecutive_none += 1
                    # [M1] Warn periodically on prolonged silence.
                    if consecutive_none == _MAX_CONSECUTIVE_NONE:
                        logger.warning(
                            'No events received for %d consecutive polls — '
                            'data source may be disconnected',
                            consecutive_none,
                        )
                    elif (
                        consecutive_none > _MAX_CONSECUTIVE_NONE
                        and consecutive_none % _MAX_CONSECUTIVE_NONE == 0
                    ):
                        logger.warning(
                            'Still no events (%d consecutive Nones)',
                            consecutive_none,
                        )
                    continue
                else:
                    self.running = False
                    logger.info('Data source exhausted — engine stopping')
                    break

            # Got a real event — reset the silence counter.
            consecutive_none = 0

            # [P1] Before processing a NewsEvent (slow, LLM call), drain all
            # pending OrderBookEvents so market data is up-to-date. This
            # prevents the scenario where order books stay empty for minutes
            # while the engine processes a burst of news events one by one.
            # Only in continuous (live) mode — backtesting preserves event order.
            if self._continuous and isinstance(event, NewsEvent):
                drained = 0
                while True:
                    try:
                        peek = await asyncio.wait_for(
                            self.data_source.get_next_event(),
                            timeout=0.05,
                        )
                    except (asyncio.TimeoutError, Exception):
                        break
                    if peek is None:
                        break
                    self._event_count += 1
                    if isinstance(peek, OrderBookEvent):
                        self.market_data.process_orderbook_event(peek)
                        drained += 1
                    elif isinstance(peek, PriceChangeEvent):
                        self.market_data.process_price_change_event(peek)
                        drained += 1
                    elif isinstance(peek, NewsEvent):
                        # Put non-market events into a buffer to process later
                        self._deferred_news.append(peek)
                if drained > 0:
                    logger.debug('Drained %d market events before NewsEvent', drained)

            batch = self._drain_backtest_timestamp_batch(event)
            batch_has_prefetched_market_updates = len(batch) > 1
            if batch_has_prefetched_market_updates:
                for batch_event in batch:
                    self._apply_market_event(batch_event)

            for batch_event in batch:
                await self._process_one_event(
                    batch_event,
                    market_data_already_applied=batch_has_prefetched_market_updates,
                )

        logger.info('TradingEngine stopped  (events=%d)', self._event_count)
        if self._state_store:
            self._state_store.save_all(self.trader.position_manager, self._perf)

    async def stop(self) -> None:
        """Stop the engine **and** its data source (async-safe)."""
        self.running = False
        # [C2] Propagate shutdown to the data source so it can cancel
        # any background polling tasks it may have launched.
        try:
            await self.data_source.stop()
        except Exception:
            logger.debug('data_source.stop() error (ignored)', exc_info=True)

        if self._state_store:
            try:
                self._state_store.save_all(self.trader.position_manager, self._perf)
            except Exception:
                logger.debug('state_store.save_all() on stop failed', exc_info=True)

        if self._alerter:
            try:
                await self._alerter.on_engine_stop('stopped')
            except Exception:
                logger.debug('alerter.on_engine_stop() failed', exc_info=True)

    def request_stop(self) -> None:
        """Non-async flag-flip — safe to call from any context."""
        self.running = False

    # ------------------------------------------------------------------ #
    # Trade tracking                                                      #
    # ------------------------------------------------------------------ #

    async def _sync_trades(self) -> None:
        """Feed newly-appeared orders into the performance analyser.

        Also persists new trades/orders to the state store and fires
        alerter.on_trade() for filled orders.
        """
        # [M3] Now that Trader base-class declares ``orders``, we can
        # access it directly instead of using getattr().
        orders = self.trader.orders
        now_str = datetime.now().strftime('%H:%M:%S')
        while self._last_orders_idx < len(orders):
            order = orders[self._last_orders_idx]
            if self._last_order_trade_idx == 0:
                self._order_times.append(now_str)
            while self._last_order_trade_idx < len(order.trades):
                trade = order.trades[self._last_order_trade_idx]
                self._perf.add_trade(trade, timestamp=self._event_timestamp)
                # Advance before any await: cancellation must not replay this fill.
                self._last_order_trade_idx += 1
                self._checkpoint_dirty = True
                if self._state_store:
                    try:
                        self._state_store.append_trade(trade)
                    except Exception:
                        logger.debug('state_store.append_trade() failed', exc_info=True)
                if self._alerter and order.status in (
                    OrderStatus.FILLED,
                    OrderStatus.PARTIALLY_FILLED,
                ):
                    try:
                        await self._alerter.on_trade(trade)
                    except Exception:
                        logger.debug('alerter.on_trade() failed', exc_info=True)
            if self._state_store:
                try:
                    self._state_store.append_order(order)
                except Exception:
                    logger.debug('state_store.append_order() failed', exc_info=True)
            self._last_orders_idx += 1
            self._last_order_trade_idx = 0
        if self._state_store and self._checkpoint_dirty:
            self._state_store.save_all(self.trader.position_manager, self._perf)
            self._checkpoint_dirty = False

    async def _maybe_write_reputation(self) -> None:
        """Write reputation summary if the manager is set and interval reached."""
        rm = self._reputation_manager
        if rm is None:
            return
        try:
            await rm.maybe_write_summary()
        except Exception:
            logger.debug('_maybe_write_reputation() failed', exc_info=True)

    async def _check_drawdown_alert(self) -> None:
        """Fire a drawdown alert if the current drawdown exceeds the threshold."""
        if not self._alerter or self._drawdown_alert_pct is None:
            return
        rm = self.trader.risk_manager
        if not isinstance(rm, StandardRiskManager):
            return
        try:
            current_dd = rm.get_current_drawdown()
            if current_dd >= self._drawdown_alert_pct and not self._drawdown_alerted:
                self._drawdown_alerted = True
                await self._alerter.on_drawdown_alert(
                    current_dd, self._drawdown_alert_pct
                )
            elif current_dd < self._drawdown_alert_pct:
                # Reset so we can alert again if drawdown recovers then worsens
                self._drawdown_alerted = False
        except Exception:
            logger.debug('_check_drawdown_alert() failed', exc_info=True)

    async def _check_portfolio_health(self) -> None:
        """Post-trade hard risk gate. Breaches force read-only degradation."""
        rm = self.trader.risk_manager
        if not isinstance(rm, StandardRiskManager):
            return
        try:
            ok, reason = rm.check_portfolio_health()
        except Exception:
            logger.debug('check_portfolio_health() failed', exc_info=True)
            return
        if ok or self._degraded_read_only:
            return

        now_str = datetime.now().strftime('%H:%M:%S')
        self._activity_log.append((now_str, f'Risk breach: {reason} -> READ-ONLY'))
        logger.error(
            'Risk breach detected (%s). Switching trader to read-only mode.', reason
        )
        self.trader.set_read_only(True)
        self.strategy.set_paused(True)
        self._degraded_read_only = True
        if self._alerter:
            try:
                await self._alerter.on_risk_limit_hit(reason)
            except Exception:
                logger.debug('alerter.on_risk_limit_hit() failed', exc_info=True)

    async def _auto_degrade_if_needed(self) -> None:
        """Pause strategy and force read-only mode on repeated processing errors."""
        if self._consecutive_processing_errors < self._MAX_CONSECUTIVE_ERRORS:
            return
        if self._degraded_read_only:
            return
        msg = (
            f'Consecutive processing errors={self._consecutive_processing_errors} '
            '-> auto-degrade READ-ONLY'
        )
        now_str = datetime.now().strftime('%H:%M:%S')
        self._activity_log.append((now_str, msg))
        logger.error(msg)
        self.trader.set_read_only(True)
        self.strategy.set_paused(True)
        self._degraded_read_only = True
        if self._alerter:
            try:
                await self._alerter.on_risk_limit_hit('auto_degrade_error_storm')
            except Exception:
                logger.debug('alerter.on_risk_limit_hit() failed', exc_info=True)

    # ------------------------------------------------------------------ #
    # Snapshot (non-blocking, no awaits)                                  #
    # ------------------------------------------------------------------ #

    def get_snapshot(self) -> EngineSnapshot:  # noqa: C901
        """Return an immutable snapshot of current engine + portfolio state.

        This method is intentionally **synchronous** and never yields, so it
        is safe to call from the UI refresh loop between ``await`` points.
        All mutable collections (``order_books``, ``orders``, …) are
        copied via ``list()`` to avoid issues if a background task mutates
        them concurrently.
        """
        pm = self.trader.position_manager
        md = self.trader.market_data
        rm = self.trader.risk_manager

        # One accounting source: cash + marked inventory, with explicit missing marks.
        stats = self._perf.get_stats()
        cash = stats.cash
        equity = cash + stats.inventory_value
        realized_pnl = stats.realized_pnl
        unrealized_pnl = stats.unrealized_pnl
        total_pnl = stats.total_pnl
        sharpe = stats.sharpe_ratio
        max_dd = stats.max_drawdown

        # ---- current drawdown from risk manager -------------------------
        current_dd = Decimal('0')
        if isinstance(rm, StandardRiskManager):
            try:
                current_dd = rm.get_current_drawdown()
            except Exception:
                logger.debug('get_current_drawdown error', exc_info=True)

        # ---- exposure ---------------------------------------------------
        market_value = max(equity - cash, Decimal('0'))
        exposure_pct = float(market_value / equity * 100) if equity > 0 else 0.0

        # ---- positions --------------------------------------------------
        pos_snaps: list[PositionSnapshot] = []
        for pos in pm.get_non_cash_positions():
            if pos.quantity <= 0:
                continue
            cur_price = pos.average_cost
            u_pnl = Decimal('0')
            try:
                bid = md.get_best_bid(pos.ticker)
                if bid is not None:
                    cur_price = bid.price
                else:
                    ask = md.get_best_ask(pos.ticker)
                    if ask is not None:
                        cur_price = ask.price
                u_pnl = (cur_price - pos.average_cost) * pos.quantity
            except (KeyError, AttributeError):
                pass
            pos_snaps.append(
                PositionSnapshot(
                    ticker_symbol=pos.ticker.symbol,
                    ticker_name=getattr(pos.ticker, 'name', '')
                    or pos.ticker.symbol[:30],
                    quantity=pos.quantity,
                    average_cost=pos.average_cost,
                    current_price=cur_price,
                    unrealized_pnl=u_pnl,
                    realized_pnl=pos.realized_pnl,
                )
            )

        # ---- order books ------------------------------------------------
        # [H1] Snapshot the dict to avoid RuntimeError if it's mutated.
        # [M2] Sort by symbol for stable UI ordering.
        ob_snaps: list[OrderBookSnapshot] = []
        for ticker, ob in sorted(  # noqa: C414
            list(md.order_books.items()),  # snapshot to avoid RuntimeError
            key=lambda kv: kv[0].symbol,
        ):
            if isinstance(ticker, CashTicker):
                continue
            bids = [(lv.price, lv.size) for lv in ob.get_bids(5)]
            asks = [(lv.price, lv.size) for lv in ob.get_asks(5)]
            if bids or asks:
                ob_snaps.append(
                    OrderBookSnapshot(
                        ticker_symbol=ticker.symbol,
                        bids=bids,
                        asks=asks,
                    )
                )

        # ---- recent trades / active orders ------------------------------
        # [M3] Direct attribute access now that Trader declares ``orders``.
        all_orders = list(self.trader.orders)  # snapshot the list

        trade_snaps: list[TradeSnapshot] = []
        active_snaps: list[OrderSnapshot] = []

        for idx in range(len(all_orders) - 1, max(len(all_orders) - 20, -1) - 1, -1):
            if idx < 0:
                break
            order = all_orders[idx]
            side_str = order.side.value.upper()
            status_str = order.status.value.upper()
            ts = self._order_times[idx] if idx < len(self._order_times) else ''

            if order.status in (
                OrderStatus.FILLED,
                OrderStatus.PARTIALLY_FILLED,
            ):
                trade_snaps.append(
                    TradeSnapshot(
                        time=ts,
                        side=side_str,
                        ticker_symbol=order.ticker.symbol,
                        ticker_name=getattr(order.ticker, 'name', '')
                        or order.ticker.symbol[:30],
                        price=(
                            order.average_price
                            if order.average_price > 0
                            else order.limit_price
                        ),
                        quantity=order.filled_quantity,
                        status=status_str,
                    )
                )
            if (
                order.status
                in (
                    OrderStatus.PLACED,
                    OrderStatus.PARTIALLY_FILLED,
                    OrderStatus.UNKNOWN,
                )
                or order.execution_uncertain
            ):
                active_snaps.append(
                    OrderSnapshot(
                        side=side_str,
                        ticker_symbol=order.ticker.symbol,
                        ticker_name=getattr(order.ticker, 'name', '')
                        or order.ticker.symbol[:30],
                        limit_price=order.limit_price,
                        quantity=order.remaining + order.filled_quantity,
                        filled_quantity=order.filled_quantity,
                        status=status_str,
                    )
                )

        # Live order history contains accounting deltas, not pending intents.
        journal = getattr(self.trader, 'order_journal', None)
        if journal is not None:
            active_snaps = []
            try:
                for row in journal.records(pending_only=True):
                    request = row['request']
                    observed = row.get('result') or {}
                    active_snaps.append(
                        OrderSnapshot(
                            side=request['side'],
                            ticker_symbol=request['ticker']['symbol'],
                            ticker_name=request['ticker'].get('name', ''),
                            limit_price=Decimal(request['limit_price']),
                            quantity=Decimal(request['quantity']),
                            filled_quantity=Decimal(
                                observed.get('filled_quantity', '0')
                            ),
                            status=observed.get('status', 'unknown'),
                            venue_order_id=row['venue_order_id'],
                            client_order_id=row['client_order_id'],
                        )
                    )
            except Exception:
                logger.exception('Cannot display pending order journal')

        # ---- uptime -----------------------------------------------------
        uptime = '00:00:00'
        if self._start_time:
            secs = int((datetime.now() - self._start_time).total_seconds())
            h, rem = divmod(secs, 3600)
            m, s = divmod(rem, 60)
            uptime = f'{h:02d}:{m:02d}:{s:02d}'

        return EngineSnapshot(
            equity=equity,
            cash=cash,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_pnl=total_pnl,
            sharpe=sharpe,
            max_drawdown_pct=max_dd,
            current_drawdown_pct=current_dd,
            exposure=market_value,
            exposure_pct=exposure_pct,
            uptime=uptime,
            event_count=self._event_count,
            engine_running=self.running,
            positions=pos_snaps,
            orderbooks=ob_snaps,
            recent_trades=trade_snaps[:10],
            active_orders=active_snaps,
            news_headlines=list(self._news),
            valuation_complete=stats.valuation_complete,
            missing_marks=stats.missing_marks,
            sharpe_available=stats.sharpe_available,
            reconciliation_required=self._reconciliation_blocked
            or self.trader.reconciliation_required,
            reconciliation_reason=self.trader.reconciliation_reason
            if self.trader.reconciliation_required
            else (
                'checkpoint or venue reconciliation failed'
                if self._reconciliation_blocked
                else ''
            ),
        )
