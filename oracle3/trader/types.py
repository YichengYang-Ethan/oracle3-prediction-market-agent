from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from oracle3.ticker.ticker import Ticker


class TradeSide(Enum):
    BUY = 'buy'
    SELL = 'sell'


@dataclass
class Trade:
    side: TradeSide
    ticker: Ticker
    price: Decimal
    quantity: Decimal
    commission: Decimal
    venue_order_id: str = ''
    client_order_id: str = ''
    execution_id: str = ''
    fee_source: str = ''

    def to_payload(self) -> dict[str, str | dict[str, str]]:
        return {
            'side': self.side.value,
            'ticker': _ticker_to_payload(self.ticker),
            'price': str(self.price),
            'quantity': str(self.quantity),
            'commission': str(self.commission),
            'venue_order_id': self.venue_order_id,
            'client_order_id': self.client_order_id,
            'execution_id': self.execution_id,
            'fee_source': self.fee_source,
        }


class OrderStatus(Enum):
    PLACED = 'placed'
    REJECTED = 'rejected'
    PARTIALLY_FILLED = 'partially_filled'
    FILLED = 'filled'
    CANCELLED = 'cancelled'
    UNKNOWN = 'unknown'


class OrderFailureReason(Enum):
    RISK_CHECK_FAILED = 'risk_check_failed'
    INVALID_ORDER = 'invalid_order'
    INSUFFICIENT_CASH = 'insufficient_cash'
    DUPLICATE_ORDER = 'duplicate_order'
    MARKET_NOT_ALLOWED = 'market_not_allowed'
    TRADING_DISABLED = 'trading_disabled'
    UNKNOWN = 'unknown'
    RECONCILIATION_REQUIRED = 'reconciliation_required'


@dataclass
class Order:
    status: OrderStatus
    side: TradeSide
    ticker: Ticker
    limit_price: Decimal
    filled_quantity: Decimal
    average_price: Decimal
    trades: list[Trade]
    remaining: Decimal
    commission: Decimal
    venue_order_id: str = ''
    client_order_id: str = ''
    execution_uncertain: bool = False
    reconciliation_reason: str = ''
    accounting_confirmed: bool = True

    @property
    def requires_reconciliation(self) -> bool:
        return (
            self.execution_uncertain
            or not self.accounting_confirmed
            or self.status
            in {OrderStatus.PLACED, OrderStatus.PARTIALLY_FILLED, OrderStatus.UNKNOWN}
        )

    def to_payload(self) -> dict[str, object]:
        return {
            'status': self.status.value,
            'side': self.side.value,
            'ticker': _ticker_to_payload(self.ticker),
            'limit_price': str(self.limit_price),
            'filled_quantity': str(self.filled_quantity),
            'average_price': str(self.average_price),
            'remaining': str(self.remaining),
            'commission': str(self.commission),
            'trades': [trade.to_payload() for trade in self.trades],
            'venue_order_id': self.venue_order_id,
            'client_order_id': self.client_order_id,
            'execution_uncertain': self.execution_uncertain,
            'reconciliation_reason': self.reconciliation_reason,
            'accounting_confirmed': self.accounting_confirmed,
            'requires_reconciliation': self.requires_reconciliation,
        }


@dataclass
class PlaceOrderResult:
    order: Order | None = None
    failure_reason: OrderFailureReason | None = None
    client_order_id: str = ''
    venue_order_id: str = ''
    reconciliation_required: bool = False

    @property
    def requires_reconciliation(self) -> bool:
        return (
            self.reconciliation_required
            or self.failure_reason
            in {OrderFailureReason.UNKNOWN, OrderFailureReason.RECONCILIATION_REQUIRED}
            or (self.order is not None and self.order.requires_reconciliation)
        )

    @property
    def status(self) -> str:
        if self.order is None:
            return 'failed'
        return self.order.status.value

    @property
    def accepted(self) -> bool:
        return self.order is not None and self.order.status not in {
            OrderStatus.REJECTED,
            OrderStatus.UNKNOWN,
        }

    @property
    def executed(self) -> bool:
        return (
            self.order is not None
            and self.order.accounting_confirmed
            and self.order.filled_quantity > 0
        )

    def to_payload(self) -> dict[str, object]:
        return {
            'ok': self.accepted and self.failure_reason is None,
            'accepted': self.accepted,
            'executed': self.executed,
            'status': self.status,
            'failure_reason': (
                self.failure_reason.value if self.failure_reason is not None else None
            ),
            'order': self.order.to_payload() if self.order is not None else None,
            'client_order_id': self.client_order_id
            or (self.order.client_order_id if self.order else ''),
            'venue_order_id': self.venue_order_id
            or (self.order.venue_order_id if self.order else ''),
            'requires_reconciliation': self.requires_reconciliation,
        }


def _ticker_to_payload(ticker: Ticker) -> dict[str, str]:
    payload = {
        'symbol': ticker.symbol,
        'name': getattr(ticker, 'name', '') or '',
    }
    for key in ('token_id', 'market_id', 'event_id', 'market_ticker', 'event_ticker'):
        value = getattr(ticker, key, '')
        if value:
            payload[key] = str(value)
    return payload
