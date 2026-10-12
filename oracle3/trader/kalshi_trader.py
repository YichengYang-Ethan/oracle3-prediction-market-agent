from __future__ import annotations

import asyncio
import json
import logging
import os
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from oracle3.data.market_data_manager import MarketDataManager
from oracle3.position.position_manager import PositionManager
from oracle3.risk.risk_manager import RiskManager
from oracle3.ticker.ticker import KalshiTicker, Ticker
from oracle3.trader.trader import Trader
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

logger = logging.getLogger(__name__)


class KalshiTrader(Trader):
    def __init__(
        self,
        market_data: MarketDataManager,
        risk_manager: RiskManager,
        position_manager: PositionManager,
        api_key_id: str | None = None,
        private_key_path: str | None = None,
        commission_rate: Decimal = Decimal('0.0'),
        alerter: Alerter | None = None,
        journal_path: str | Path | None = None,
        journal_namespace: str = 'kalshi',
    ):
        super().__init__(market_data, risk_manager, position_manager, alerter=alerter)
        self.commission_rate = (
            commission_rate  # Legacy argument; actual fees come from fills.
        )
        self._init_order_journal(journal_path, journal_namespace)

        from kalshi_python import Configuration
        from kalshi_python.api.portfolio_api import PortfolioApi
        from kalshi_python.api_client import ApiClient

        config = Configuration(host='https://api.elections.kalshi.com/trade-api/v2')

        key_id = api_key_id or os.environ.get('KALSHI_API_KEY_ID')
        pk_path = private_key_path or os.environ.get('KALSHI_PRIVATE_KEY_PATH')

        if not key_id or not pk_path:
            raise ValueError(
                'Kalshi API credentials required. '
                'Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH env vars '
                'or pass api_key_id and private_key_path.'
            )

        self._api_client = ApiClient(configuration=config)
        self._api_client.set_kalshi_auth(key_id, pk_path)
        self._portfolio_api = PortfolioApi(self._api_client)
        self.orders: list[Order] = []

    async def _submit_order(
        self,
        action: str,
        side: str,
        ticker: KalshiTicker,
        price_cents: int,
        count: int,
        client_order_id: str,
    ) -> dict[str, Any]:
        """Submit an order to Kalshi API."""
        from kalshi_python import CreateOrderRequest

        kwargs: dict[str, Any] = {
            'ticker': ticker.market_ticker,
            'action': action,
            'side': side,
            'type': 'limit',
            'count': count,
            'client_order_id': client_order_id,
        }
        if side == 'no':
            kwargs['no_price'] = price_cents
        else:
            kwargs['yes_price'] = price_cents

        request = CreateOrderRequest(**kwargs)

        response = await asyncio.to_thread(
            lambda: self._portfolio_api.create_order(create_order_request=request)
        )

        if hasattr(response, 'to_dict'):
            return response.to_dict()
        return response if isinstance(response, dict) else {'order': response}

    @staticmethod
    def _response_order_id(response: dict[str, Any]) -> str:
        data = response.get('order', response)
        return str(data.get('order_id') or '')

    @staticmethod
    def _recovery_response(venue_id: str) -> dict[str, Any]:
        return {'order': {'order_id': venue_id}}

    async def _submit_live_order(
        self,
        side: TradeSide,
        ticker: Ticker,
        price: Decimal,
        quantity: Decimal,
        client_order_id: str,
    ) -> dict[str, Any]:
        if not isinstance(ticker, KalshiTicker):
            raise ValueError('invalid Kalshi ticker')
        return await self._submit_order(
            action=side.value,
            side='no' if ticker.is_no_side else 'yes',
            ticker=ticker,
            price_cents=int(price * 100),
            count=int(quantity),
            client_order_id=client_order_id,
        )

    async def _get_order_status(self, order_id: str) -> dict[str, Any]:
        # The installed generated model can discard modern *_fp / *_dollars
        # execution fields. Read raw JSON through the authenticated SDK instead.
        def fetch() -> dict[str, Any]:
            response = self._portfolio_api.get_order_without_preload_content(order_id)
            try:
                return json.loads(response.read())
            finally:
                response.release_conn()

        return await asyncio.to_thread(fetch)

    @staticmethod
    def _decimal(value: Any, name: str) -> Decimal:
        if value is None or isinstance(value, bool):
            raise ValueError(f'missing or invalid {name}')
        parsed = Decimal(str(value))
        if not parsed.is_finite() or parsed < 0:
            raise ValueError(f'invalid {name}')
        return parsed

    async def _process_order_response(  # noqa: C901
        self,
        response: dict[str, Any],
        side: TradeSide,
        ticker: Ticker,
        limit_price: Decimal,
        quantity: Decimal,
    ) -> Order:
        if not isinstance(ticker, KalshiTicker):
            raise ValueError('invalid Kalshi ticker')
        venue_id = self._response_order_id(response)
        if not venue_id:
            return self._unknown_order(
                side,
                ticker,
                limit_price,
                quantity,
                '',
                reason='submission lacks venue order ID',
            )
        details = await self._get_order_status(venue_id)
        data = details.get('order', details)
        if data.get('order_id') != venue_id:
            raise ValueError('order ID mismatch')
        if data.get('ticker') != ticker.market_ticker:
            raise ValueError('market mismatch')
        if data.get('action') != side.value:
            raise ValueError('action mismatch')
        expected_side = 'no' if ticker.is_no_side else 'yes'
        if data.get('side') != expected_side:
            raise ValueError('outcome mismatch')
        remaining = self._decimal(
            data.get('remaining_count_fp', data.get('remaining_count')),
            'remaining_count',
        )
        initial_value = data.get('initial_count_fp', data.get('count'))
        initial = self._decimal(initial_value, 'initial_count')
        if initial != quantity:
            raise ValueError('requested quantity mismatch')
        fill_value = data.get('fill_count_fp', data.get('fill_count'))
        # Cancellation can make remaining zero without filling the order.
        # Never infer executed contracts from original minus remaining.
        filled = self._decimal(fill_value, 'fill_count')
        if not 0 <= filled <= quantity or not 0 <= remaining <= quantity:
            raise ValueError('invalid filled or remaining quantity')
        status = str(data.get('status', '')).lower()
        if status in {'resting', 'pending'}:
            if filled + remaining != quantity:
                raise ValueError('inconsistent open order quantities')
            internal_status = (
                OrderStatus.PARTIALLY_FILLED if filled else OrderStatus.PLACED
            )
        elif status == 'executed':
            if filled != quantity or remaining != 0:
                raise ValueError('executed order is not fully filled')
            internal_status = OrderStatus.FILLED
        elif status in {'canceled', 'cancelled'}:
            internal_status = OrderStatus.CANCELLED
        elif status == 'rejected' and filled == 0:
            internal_status = OrderStatus.REJECTED
        else:
            return self._unknown_order(
                side,
                ticker,
                limit_price,
                quantity,
                '',
                venue_id,
                'unrecognized venue order state',
            )
        price, fees, trades = Decimal('0'), Decimal('0'), []
        if filled:
            # These are cumulative actual execution amounts, not limit-price estimates.
            cost = sum(
                (
                    self._decimal(data.get(key), key)
                    for key in ('maker_fill_cost_dollars', 'taker_fill_cost_dollars')
                ),
                Decimal('0'),
            )
            fees = sum(
                (
                    self._decimal(data.get(key), key)
                    for key in ('maker_fees_dollars', 'taker_fees_dollars')
                ),
                Decimal('0'),
            )
            price = cost / filled
            if (
                not 0 < price <= 1
                or (side == TradeSide.BUY and price > limit_price)
                or (side == TradeSide.SELL and price < limit_price)
            ):
                raise ValueError(
                    'execution price violates limit or binary price bounds'
                )
            trades = [
                Trade(
                    side,
                    ticker,
                    price,
                    filled,
                    fees,
                    venue_order_id=venue_id,
                    fee_source='venue cumulative execution amounts',
                )
            ]
        return Order(
            internal_status,
            side,
            ticker,
            limit_price,
            filled,
            price,
            trades,
            remaining,
            fees,
            venue_order_id=venue_id,
        )

    async def place_order(
        self,
        side: TradeSide,
        ticker: Ticker,
        limit_price: Decimal,
        quantity: Decimal,
        client_order_id: str | None = None,
    ) -> PlaceOrderResult:
        if not isinstance(ticker, KalshiTicker) or not ticker.market_ticker:
            return PlaceOrderResult(failure_reason=OrderFailureReason.INVALID_ORDER)
        if not self._valid_live_inputs(side, limit_price, quantity):
            return PlaceOrderResult(failure_reason=OrderFailureReason.INVALID_ORDER)
        # This SDK's submission schema is integer contracts and integer cents.
        # Reject unsupported precision rather than silently truncating the request.
        if (
            quantity != quantity.to_integral_value()
            or limit_price * 100 != (limit_price * 100).to_integral_value()
        ):
            return PlaceOrderResult(failure_reason=OrderFailureReason.INVALID_ORDER)
        return await self._place_live_order(
            side, ticker, limit_price, quantity, client_order_id
        )
