from __future__ import annotations

import asyncio
import math
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType, TradeParams
from py_clob_client.constants import POLYGON
from py_clob_client.order_builder.constants import BUY, SELL
from py_order_utils.model import EOA

from oracle3.data.market_data_manager import MarketDataManager
from oracle3.position.position_manager import PositionManager
from oracle3.risk.risk_manager import RiskManager
from oracle3.ticker.ticker import PolyMarketTicker, Ticker
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


class PolymarketTrader(Trader):
    """FOK submission with confirmed, zero-fee execution accounting.

    A nonzero fee rate alone does not establish the net collateral and tokens
    delivered by settlement. Those executions deliberately require external
    reconciliation rather than inventing an actual fee from a price estimate.
    """

    def __init__(
        self,
        market_data: MarketDataManager,
        risk_manager: RiskManager,
        position_manager: PositionManager,
        wallet_private_key: str,
        signature_type: int = EOA,
        funder: str | None = None,
        clob_api_url: str = 'https://clob.polymarket.com',
        chain_id: int = POLYGON,
        commission_rate: Decimal = Decimal('0.0'),
        alerter: Alerter | None = None,
        journal_path: str | Path | None = None,
        journal_namespace: str = 'polymarket',
    ):
        super().__init__(market_data, risk_manager, position_manager, alerter=alerter)
        # Retain constructor compatibility; estimates are never booked as actual fees.
        self.commission_rate = commission_rate
        self._init_order_journal(journal_path, journal_namespace)
        self.clob_client = ClobClient(
            clob_api_url,
            key=wallet_private_key,
            chain_id=chain_id,
            signature_type=signature_type,
            funder=funder,
        )
        self.clob_client.set_api_creds(self.clob_client.create_or_derive_api_creds())

    async def _submit_fok_order(
        self, side: TradeSide, ticker: PolyMarketTicker, price: Decimal, size: Decimal
    ) -> dict[str, Any]:
        # OrderArgs.size is conditional-token shares, not a collateral budget.
        order_args = OrderArgs(
            price=float(price),
            size=float(size),
            side=BUY if side == TradeSide.BUY else SELL,
            token_id=ticker.token_id,
        )

        def submit() -> dict[str, Any]:
            order = self.clob_client.create_order(order_args)
            return self.clob_client.post_order(order, orderType=OrderType.FOK)

        return await asyncio.to_thread(submit)

    async def _submit_live_order(
        self,
        side: TradeSide,
        ticker: Ticker,
        price: Decimal,
        quantity: Decimal,
        client_order_id: str,
    ) -> dict[str, Any]:
        if not isinstance(ticker, PolyMarketTicker):
            raise ValueError('invalid Polymarket ticker')
        # This SDK does not expose a venue client ID. The journal reserves our
        # key before this call and never automatically resubmits an ambiguous call.
        return await self._submit_fok_order(side, ticker, price, quantity)

    @staticmethod
    def _response_order_id(response: dict[str, Any]) -> str:
        return str(response.get('orderID') or '')

    @staticmethod
    def _recovery_response(venue_id: str) -> dict[str, Any]:
        return {'success': True, 'orderID': venue_id}

    async def _get_order_status(self, order_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self.clob_client.get_order, order_id)

    async def _get_order_trades(self, trade_ids: list[str]) -> list[dict[str, Any]]:
        result = []
        for trade_id in trade_ids:
            trades = await asyncio.to_thread(
                self.clob_client.get_trades, TradeParams(id=trade_id)
            )
            matches = [item for item in trades if item.get('id') == trade_id]
            if len(matches) != 1:
                raise ValueError('missing or duplicate associated trade')
            result.append(matches[0])
        return result

    @staticmethod
    def _decimal(value: Any, name: str) -> Decimal:
        if value is None or isinstance(value, bool):
            raise ValueError(f'missing or invalid {name}')
        number = Decimal(str(value))
        if not number.is_finite() or number < 0:
            raise ValueError(f'invalid {name}')
        return number

    async def _process_order_response(
        self,
        response: dict[str, Any],
        side: TradeSide,
        ticker: Ticker,
        limit_price: Decimal,
        quantity: Decimal,
    ) -> Order:
        if not isinstance(ticker, PolyMarketTicker):
            raise ValueError('invalid Polymarket ticker')
        venue_id = self._response_order_id(response)
        if not venue_id:
            # A missing acknowledgement is not evidence that no order exists.
            return self._unknown_order(
                side,
                ticker,
                limit_price,
                quantity,
                '',
                reason='submission did not provide a venue order ID',
            )
        data = await self._get_order_status(venue_id)
        if data.get('id') != venue_id or data.get('asset_id') != ticker.token_id:
            raise ValueError('order identity does not match submission')
        if str(data.get('side', '')).upper() != side.value.upper():
            raise ValueError('order side does not match submission')
        initial = self._decimal(data.get('original_size'), 'original_size')
        filled = self._decimal(data.get('size_matched'), 'size_matched')
        if initial != quantity or filled > quantity:
            raise ValueError('order quantities do not match requested shares')
        status = str(data.get('status', '')).upper()
        if status == 'MATCHED':
            if filled != quantity:
                raise ValueError('FOK matched quantity is not the requested quantity')
            internal_status = OrderStatus.FILLED
        elif status in {'LIVE', 'DELAYED', 'UNMATCHED'}:
            internal_status = (
                OrderStatus.PARTIALLY_FILLED if filled else OrderStatus.PLACED
            )
        elif status in {'CANCELED', 'CANCELLED'}:
            internal_status = OrderStatus.CANCELLED
        elif status == 'REJECTED' and filled == 0:
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

        price = Decimal('0')
        trades: list[Trade] = []
        uncertainty: list[str] = []
        if filled:
            ids = data.get('associate_trades')
            if (
                not isinstance(ids, list)
                or not ids
                or any(not isinstance(i, str) or not i for i in ids)
                or len(ids) != len(set(ids))
            ):
                raise ValueError('filled order has no unique associated trade IDs')
            reports = await self._get_order_trades(ids)
            if len(reports) != len(ids) or {item.get('id') for item in reports} != set(
                ids
            ):
                raise ValueError('incomplete associated trade reports')
            for report in reports:
                if (
                    report.get('taker_order_id') != venue_id
                    or report.get('asset_id') != ticker.token_id
                    or str(report.get('side', '')).upper() != side.value.upper()
                ):
                    raise ValueError('trade identity does not match FOK order')
                trade_price = self._decimal(report.get('price'), 'trade price')
                trade_size = self._decimal(report.get('size'), 'trade size')
                if not 0 < trade_price <= 1 or trade_size <= 0:
                    raise ValueError('invalid trade price or size')
                if (side == TradeSide.BUY and trade_price > limit_price) or (
                    side == TradeSide.SELL and trade_price < limit_price
                ):
                    raise ValueError('trade price violates limit')
                if str(report.get('status', '')).upper() != 'CONFIRMED':
                    uncertainty.append('associated trade settlement is not confirmed')
                fee_rate = self._decimal(report.get('fee_rate_bps'), 'trade fee rate')
                if fee_rate != 0:
                    uncertainty.append(
                        'fee-bearing settlement requires verified net amounts'
                    )
                trades.append(
                    Trade(
                        side,
                        ticker,
                        trade_price,
                        trade_size,
                        Decimal('0'),
                        venue_order_id=venue_id,
                        execution_id=report['id'],
                        fee_source='venue explicitly reports zero fee rate'
                        if fee_rate == 0
                        else 'unverified',
                    )
                )
            actual_quantity = sum((trade.quantity for trade in trades), Decimal('0'))
            if actual_quantity != filled:
                raise ValueError('associated trades do not equal matched shares')
            price = (
                sum((trade.quantity * trade.price for trade in trades), Decimal('0'))
                / filled
            )
        return Order(
            internal_status,
            side,
            ticker,
            limit_price,
            filled,
            price,
            trades if not uncertainty else [],
            quantity - filled,
            Decimal('0'),
            venue_order_id=venue_id,
            execution_uncertain=bool(uncertainty),
            reconciliation_reason='; '.join(dict.fromkeys(uncertainty)),
            accounting_confirmed=not uncertainty,
        )

    async def place_order(
        self,
        side: TradeSide,
        ticker: Ticker,
        limit_price: Decimal,
        quantity: Decimal,
        client_order_id: str | None = None,
    ) -> PlaceOrderResult:
        if not isinstance(ticker, PolyMarketTicker) or not ticker.token_id:
            return PlaceOrderResult(failure_reason=OrderFailureReason.INVALID_ORDER)
        if not self._valid_live_inputs(side, limit_price, quantity):
            return PlaceOrderResult(failure_reason=OrderFailureReason.INVALID_ORDER)
        # The SDK uses floats. Reject precision loss before signing a different
        # intent; non-finite float conversion must never reach the venue client.
        for value in (limit_price, quantity):
            converted = float(value)
            if not math.isfinite(converted) or Decimal(str(converted)) != value:
                return PlaceOrderResult(failure_reason=OrderFailureReason.INVALID_ORDER)
        return await self._place_live_order(
            side, ticker, limit_price, quantity, client_order_id
        )
