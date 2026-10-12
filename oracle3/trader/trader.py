from __future__ import annotations

import asyncio
import copy
import os
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from oracle3.data.market_data_manager import MarketDataManager
from oracle3.position.position_manager import PositionManager
from oracle3.risk.risk_manager import RiskManager
from oracle3.ticker.ticker import KalshiTicker, PolyMarketTicker, Ticker
from oracle3.trader.order_journal import OrderJournal
from oracle3.trader.types import (
    Order,
    OrderFailureReason,
    OrderStatus,
    PlaceOrderResult,
    Trade,
    TradeSide,
)

if TYPE_CHECKING:
    from oracle3.alerts.alerter import Alerter


class Trader(ABC):
    def __init__(
        self,
        market_data: MarketDataManager,
        risk_manager: RiskManager,
        position_manager: PositionManager,
        alerter: Alerter | None = None,
    ):
        self.market_data = market_data
        self.risk_manager = risk_manager
        self.position_manager = position_manager
        self.alerter = alerter
        self.orders: list[Order] = []
        self.read_only: bool = False
        self._seen_client_order_ids: set[str] = set()
        self._seen_client_order_queue: deque[str] = deque()
        self._max_seen_client_order_ids: int = 5000
        self._recent_news: deque[dict[str, str]] = deque(maxlen=200)
        self._allowed_ticker_symbols: set[str] | None = None

    @abstractmethod
    async def place_order(
        self,
        side: TradeSide,
        ticker: Ticker,
        limit_price: Decimal,
        quantity: Decimal,
        client_order_id: str | None = None,
    ) -> PlaceOrderResult:
        """Place an order."""
        pass

    def set_read_only(self, enabled: bool) -> None:
        """Enable/disable read-only mode (blocks new orders when enabled)."""
        self.read_only = enabled

    def set_allowed_tickers(self, tickers: list[Ticker | str] | None) -> None:
        """Restrict trading to a known set of ticker symbols."""
        if tickers is None:
            self._allowed_ticker_symbols = None
            return

        allowed: set[str] = set()
        for ticker in tickers:
            if isinstance(ticker, str):
                symbol = ticker.strip()
            else:
                symbol = ticker.symbol.strip()
            if symbol:
                allowed.add(symbol)
        self._allowed_ticker_symbols = allowed

    def is_ticker_tradable(self, ticker: Ticker) -> bool:
        allowed = self._allowed_ticker_symbols
        return allowed is None or ticker.symbol in allowed

    def _kill_switch_active(self) -> bool:
        """Global kill-switch that can be toggled outside the process.

        Either:
        - PRED_MARKET_CLI_KILL_SWITCH=1
        - PRED_MARKET_CLI_KILL_SWITCH_FILE points to a file that exists
        """
        if os.environ.get('PRED_MARKET_CLI_KILL_SWITCH', '').strip() == '1':
            return True
        kill_file = os.environ.get('PRED_MARKET_CLI_KILL_SWITCH_FILE', '').strip()
        if kill_file:
            return os.path.exists(kill_file)
        default_kill_file = Path.home() / '.oracle3' / 'kill.switch'
        return default_kill_file.exists()

    def _check_order_guard(
        self, client_order_id: str | None
    ) -> OrderFailureReason | None:
        """Validate global trade guards and idempotency keys."""
        if self.read_only or self._kill_switch_active():
            return OrderFailureReason.TRADING_DISABLED
        if client_order_id:
            if client_order_id in self._seen_client_order_ids:
                return OrderFailureReason.DUPLICATE_ORDER
            self._seen_client_order_ids.add(client_order_id)
            self._seen_client_order_queue.append(client_order_id)
            while len(self._seen_client_order_queue) > self._max_seen_client_order_ids:
                stale = self._seen_client_order_queue.popleft()
                self._seen_client_order_ids.discard(stale)
        return None

    def _init_order_journal(
        self, path: str | Path | None = None, namespace: str = 'default'
    ) -> None:
        """Enable live-order journaling; None deliberately creates no disk files."""
        self.order_journal = OrderJournal(path, namespace)
        self._live_order_lock = asyncio.Lock()
        self._session_order_ids: set[str] = set()
        self._integrity_error = ''
        records = self.order_journal.records()
        self._startup_inventory_unverified = any(
            row['needs_reconciliation'] or Decimal(row['applied_quantity']) != 0
            for row in records
        )

    @property
    def reconciliation_required(self) -> bool:
        journal = getattr(self, 'order_journal', None)
        if journal is None:
            return False
        if self._integrity_error or self._startup_inventory_unverified:
            return True
        try:
            return bool(journal.records(pending_only=True))
        except Exception:
            self._integrity_error = 'order journal unavailable'
            return True

    @property
    def reconciliation_reason(self) -> str:
        if getattr(self, '_integrity_error', ''):
            return self._integrity_error
        if getattr(self, '_startup_inventory_unverified', False):
            return 'restart inventory is unverified; reconcile account positions and journal before resuming'
        return (
            'outstanding or uncertain live order'
            if self.reconciliation_required
            else ''
        )

    @staticmethod
    def _valid_live_inputs(side: TradeSide, price: Decimal, quantity: Decimal) -> bool:
        return (
            isinstance(side, TradeSide)
            and isinstance(price, Decimal)
            and price.is_finite()
            and 0 < price <= 1
            and isinstance(quantity, Decimal)
            and quantity.is_finite()
            and quantity > 0
        )

    @staticmethod
    def _unknown_order(
        side: TradeSide,
        ticker: Ticker,
        price: Decimal,
        quantity: Decimal,
        client_id: str,
        venue_id: str = '',
        reason: str = 'venue outcome unknown',
    ) -> Order:
        return Order(
            OrderStatus.UNKNOWN,
            side,
            ticker,
            price,
            Decimal('0'),
            Decimal('0'),
            [],
            quantity,
            Decimal('0'),
            venue_order_id=venue_id,
            client_order_id=client_id,
            execution_uncertain=True,
            reconciliation_reason=reason,
            accounting_confirmed=False,
        )

    async def _submit_live_order(
        self,
        side: TradeSide,
        ticker: Ticker,
        price: Decimal,
        quantity: Decimal,
        client_order_id: str,
    ) -> dict[str, Any]:
        raise NotImplementedError('live adapter must implement submission')

    @staticmethod
    def _response_order_id(response: dict[str, Any]) -> str:
        raise NotImplementedError('live adapter must implement order identity')

    @staticmethod
    def _recovery_response(venue_id: str) -> dict[str, Any]:
        raise NotImplementedError('live adapter must implement recovery response')

    async def _process_order_response(
        self,
        response: dict[str, Any],
        side: TradeSide,
        ticker: Ticker,
        limit_price: Decimal,
        quantity: Decimal,
    ) -> Order:
        raise NotImplementedError('live adapter must validate execution reports')

    async def _place_live_order(  # noqa: C901
        self,
        side: TradeSide,
        ticker: Ticker,
        limit_price: Decimal,
        quantity: Decimal,
        client_order_id: str | None = None,
    ) -> PlaceOrderResult:
        """Reserve before submission and serialize account mutations within a process."""
        if not self._valid_live_inputs(side, limit_price, quantity):
            return PlaceOrderResult(failure_reason=OrderFailureReason.INVALID_ORDER)
        if not self.is_ticker_tradable(ticker):
            return PlaceOrderResult(
                failure_reason=OrderFailureReason.MARKET_NOT_ALLOWED
            )
        if client_order_id is not None and (
            not isinstance(client_order_id, str) or not client_order_id.strip()
        ):
            return PlaceOrderResult(failure_reason=OrderFailureReason.INVALID_ORDER)
        async with self._live_order_lock:
            key = client_order_id or str(uuid.uuid4())
            try:
                old = self.order_journal.get(key)
                if old is not None:
                    return PlaceOrderResult(
                        failure_reason=OrderFailureReason.DUPLICATE_ORDER,
                        client_order_id=key,
                        venue_order_id=old['venue_order_id'],
                        reconciliation_required=old['needs_reconciliation']
                        or self._startup_inventory_unverified,
                    )
                if self.reconciliation_required:
                    return PlaceOrderResult(
                        failure_reason=OrderFailureReason.RECONCILIATION_REQUIRED,
                        client_order_id=key,
                        reconciliation_required=True,
                    )
                guard = self._check_order_guard(None)
                if guard is not None:
                    return PlaceOrderResult(failure_reason=guard, client_order_id=key)
                if side == TradeSide.SELL:
                    position = self.position_manager.get_position(ticker)
                    if position is None or position.quantity < quantity:
                        return PlaceOrderResult(
                            failure_reason=OrderFailureReason.INVALID_ORDER,
                            client_order_id=key,
                        )
                else:
                    cash = self.position_manager.get_position(ticker.collateral)
                    if cash is None or cash.quantity < quantity * limit_price:
                        return PlaceOrderResult(
                            failure_reason=OrderFailureReason.INSUFFICIENT_CASH,
                            client_order_id=key,
                        )
                if not await self.risk_manager.check_trade(
                    ticker, side, quantity, limit_price
                ):
                    return PlaceOrderResult(
                        failure_reason=OrderFailureReason.RISK_CHECK_FAILED,
                        client_order_id=key,
                    )
                # A kill switch may have changed while the risk manager awaited I/O.
                guard = self._check_order_guard(None)
                if guard is not None:
                    return PlaceOrderResult(failure_reason=guard, client_order_id=key)
                request = {
                    'side': side.value,
                    'ticker_type': type(ticker).__name__,
                    'ticker': asdict(ticker),
                    'limit_price': str(limit_price),
                    'quantity': str(quantity),
                }
                if not self.order_journal.reserve(key, request):
                    return PlaceOrderResult(
                        failure_reason=OrderFailureReason.DUPLICATE_ORDER,
                        client_order_id=key,
                        reconciliation_required=True,
                    )
                self._session_order_ids.add(key)
            except Exception:
                self._integrity_error = 'order journal or pre-trade validation failed'
                return PlaceOrderResult(
                    failure_reason=OrderFailureReason.RECONCILIATION_REQUIRED,
                    client_order_id=key,
                    reconciliation_required=True,
                )

            venue_id = ''
            try:
                response = await self._submit_live_order(
                    side, ticker, limit_price, quantity, key
                )
                venue_id = self._response_order_id(response)
                if venue_id:
                    self.order_journal.update(key, venue_order_id=venue_id)
                order = await self._process_order_response(
                    response, side, ticker, limit_price, quantity
                )
                order.client_order_id = key
                order.venue_order_id = venue_id or order.venue_order_id
                return self._record_live_snapshot(key, order)
            except Exception as exc:
                order = self._unknown_order(
                    side,
                    ticker,
                    limit_price,
                    quantity,
                    key,
                    venue_id,
                    f'{type(exc).__name__} after intent reservation; never resubmit blindly',
                )
                return self._record_live_snapshot(key, order)

    def _record_live_snapshot(self, key: str, order: Order) -> PlaceOrderResult:
        """Apply only validated cumulative deltas; journal failure leaves positions unchanged."""
        try:
            row = self.order_journal.get(key)
            if row is None:
                raise ValueError('missing intent')
            if key not in self._session_order_ids:
                self._startup_inventory_unverified = True
                order = replace(
                    order,
                    accounting_confirmed=False,
                    execution_uncertain=True,
                    reconciliation_reason='restart inventory is unverified; no fills replayed',
                )
            if not order.accounting_confirmed or order.execution_uncertain:
                self.order_journal.update(
                    key,
                    venue_order_id=order.venue_order_id or None,
                    result=order.to_payload(),
                    needs_reconciliation=True,
                )
                return PlaceOrderResult(
                    order=order,
                    failure_reason=OrderFailureReason.UNKNOWN,
                    client_order_id=key,
                    venue_order_id=order.venue_order_id,
                    reconciliation_required=True,
                )
            old_q, old_n, old_f = (
                Decimal(row[k])
                for k in ('applied_quantity', 'applied_notional', 'applied_fee')
            )
            new_q = order.filled_quantity
            new_n = new_q * order.average_price
            new_f = order.commission
            dq, dn, df = new_q - old_q, new_n - old_n, new_f - old_f
            if min(dq, dn, df) < 0 or (dq == 0 and (dn != 0 or df != 0)):
                raise ValueError(
                    'non-monotonic execution totals; manual reconciliation required'
                )
            positions = copy.deepcopy(self.position_manager.positions)
            delta_order = None
            if dq > 0:
                actual_price = dn / dq
                if not actual_price.is_finite() or not 0 < actual_price <= 1:
                    raise ValueError('invalid execution price')
                trade = Trade(
                    order.side,
                    order.ticker,
                    actual_price,
                    dq,
                    df,
                    venue_order_id=order.venue_order_id,
                    client_order_id=key,
                    execution_id=f'{key}:{new_q}',
                    fee_source='venue execution report',
                )
                staged = PositionManager()
                staged.positions = positions
                staged.apply_trade(trade)
                delta_order = replace(
                    order,
                    filled_quantity=dq,
                    average_price=actual_price,
                    trades=[trade],
                    commission=df,
                )
            self.order_journal.update(
                key,
                venue_order_id=order.venue_order_id or None,
                result=order.to_payload(),
                needs_reconciliation=order.requires_reconciliation,
                applied_quantity=str(new_q),
                applied_notional=str(new_n),
                applied_fee=str(new_f),
            )
            self.position_manager.positions = positions
            if delta_order is not None:
                self.orders.append(delta_order)
            return PlaceOrderResult(
                order=order,
                client_order_id=key,
                venue_order_id=order.venue_order_id,
                failure_reason=OrderFailureReason.INVALID_ORDER
                if order.status == OrderStatus.REJECTED
                else None,
            )
        except Exception:
            self._integrity_error = 'live accounting or journal update failed; verify venue and local inventory'
            return PlaceOrderResult(
                order=replace(
                    order,
                    execution_uncertain=True,
                    accounting_confirmed=False,
                    reconciliation_reason=self._integrity_error,
                ),
                failure_reason=OrderFailureReason.UNKNOWN,
                client_order_id=key,
                venue_order_id=order.venue_order_id,
                reconciliation_required=True,
            )

    async def reconcile_pending_orders(self) -> list[PlaceOrderResult]:
        """Read outstanding venue orders; never cancel or submit an order here.

        Restarted sessions remain gated: the journal and an external inventory
        snapshot cannot be assumed atomic. This hook does not certify inventory.
        """
        if not hasattr(self, 'order_journal'):
            return []
        results: list[PlaceOrderResult] = []
        async with self._live_order_lock:
            try:
                rows = self.order_journal.records(pending_only=True)
            except Exception:
                self._integrity_error = (
                    'order journal unavailable during reconciliation'
                )
                return [
                    PlaceOrderResult(
                        failure_reason=OrderFailureReason.RECONCILIATION_REQUIRED,
                        reconciliation_required=True,
                    )
                ]
            for row in rows:
                request = row['request']
                ticker_type = {
                    'KalshiTicker': KalshiTicker,
                    'PolyMarketTicker': PolyMarketTicker,
                }[request['ticker_type']]
                ticker = ticker_type(**request['ticker'])
                side = TradeSide(request['side'])
                price, quantity = (
                    Decimal(request['limit_price']),
                    Decimal(request['quantity']),
                )
                key, venue_id = row['client_order_id'], row['venue_order_id']
                try:
                    if not venue_id:
                        raise ValueError(
                            'venue order ID unavailable; account-level lookup required'
                        )
                    response = self._recovery_response(venue_id)
                    order = await self._process_order_response(
                        response, side, ticker, price, quantity
                    )
                    order.client_order_id, order.venue_order_id = key, venue_id
                except Exception as exc:
                    order = self._unknown_order(
                        side,
                        ticker,
                        price,
                        quantity,
                        key,
                        venue_id,
                        f'reconciliation {type(exc).__name__}; no order resubmitted',
                    )
                results.append(self._record_live_snapshot(key, order))
        return results

    def record_news(
        self,
        *,
        timestamp: str,
        title: str,
        source: str = '',
        url: str = '',
    ) -> None:
        """Store recent news items so all strategy types can inspect them."""
        self._recent_news.append(
            {
                'timestamp': timestamp,
                'title': title,
                'source': source,
                'url': url,
            }
        )

    def get_recent_news(self, limit: int | None = None) -> list[dict[str, str]]:
        news = list(self._recent_news)
        if limit is not None:
            if limit <= 0:
                return []
            return news[-limit:]
        return news
