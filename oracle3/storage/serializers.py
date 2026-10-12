"""JSON ↔ domain-object serialization helpers.

All Decimal values are serialized as strings to avoid floating-point loss.
Ticker subclass is stored in a ``ticker_type`` field; CashTicker singletons
are reconstructed by symbol.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from oracle3.position.position_manager import Position
from oracle3.ticker.ticker import (
    CashTicker,
    KalshiTicker,
    PolyMarketTicker,
    SolanaTicker,
    Ticker,
)
from oracle3.trader.types import Order, OrderStatus, Trade, TradeSide

# ---------------------------------------------------------------------------
# Ticker
# ---------------------------------------------------------------------------


def serialize_ticker(ticker: Ticker) -> dict:
    """Serialize a Ticker to a JSON-safe dict (fields inlined, no nesting)."""
    if isinstance(ticker, SolanaTicker):
        from dataclasses import asdict

        return {'ticker_type': 'SolanaTicker', **asdict(ticker)}
    if isinstance(ticker, PolyMarketTicker):
        return {
            'ticker_type': 'PolyMarketTicker',
            'symbol': ticker.symbol,
            'name': ticker.name,
            'token_id': ticker.token_id,
            'market_id': ticker.market_id,
            'event_id': ticker.event_id,
            'no_token_id': ticker.no_token_id,
        }
    if isinstance(ticker, KalshiTicker):
        return {
            'ticker_type': 'KalshiTicker',
            'symbol': ticker.symbol,
            'name': ticker.name,
            'market_ticker': ticker.market_ticker,
            'event_ticker': ticker.event_ticker,
            'series_ticker': ticker.series_ticker,
            'is_no_side': ticker.is_no_side,
        }
    if isinstance(ticker, CashTicker):
        return {
            'ticker_type': 'CashTicker',
            'symbol': ticker.symbol,
            'name': ticker.name,
        }
    raise ValueError(f'Unknown ticker type: {type(ticker).__name__}')


def deserialize_ticker(d: dict) -> Ticker:
    """Reconstruct a Ticker from a flattened dict (fields inlined)."""
    ticker_type = d.get('ticker_type', '')
    if ticker_type == 'SolanaTicker':
        return SolanaTicker(
            **{
                key: d[key]
                for key in (
                    'symbol',
                    'name',
                    'yes_mint',
                    'no_mint',
                    'market_ticker',
                    'event_ticker',
                    'series_ticker',
                    'is_no_side',
                )
                if key in d
            }
        )
    if ticker_type == 'PolyMarketTicker':
        return PolyMarketTicker(
            symbol=d['symbol'],
            name=d.get('name', ''),
            token_id=d.get('token_id', ''),
            market_id=d.get('market_id', ''),
            event_id=d.get('event_id', ''),
            no_token_id=d.get('no_token_id', ''),
        )
    if ticker_type == 'KalshiTicker':
        return KalshiTicker(
            symbol=d['symbol'],
            name=d.get('name', ''),
            market_ticker=d.get('market_ticker', ''),
            event_ticker=d.get('event_ticker', ''),
            series_ticker=d.get('series_ticker', ''),
            is_no_side=d.get('is_no_side', False),
        )
    if ticker_type == 'CashTicker':
        symbol = d['symbol']
        if symbol == CashTicker.POLYMARKET_USDC.symbol:
            return CashTicker.POLYMARKET_USDC
        if symbol == CashTicker.KALSHI_USD.symbol:
            return CashTicker.KALSHI_USD
        if symbol == CashTicker.DFLOW_USDC.symbol:
            return CashTicker.DFLOW_USDC
        return CashTicker(symbol=symbol, name=d.get('name', ''))
    raise ValueError(f'Unknown ticker_type: {ticker_type!r}')


# ---------------------------------------------------------------------------
# Trade
# ---------------------------------------------------------------------------


def serialize_trade(trade: Trade, timestamp: datetime | None = None) -> dict:
    """Serialize a Trade. If *timestamp* is provided it is included."""
    d: dict = {
        'side': trade.side.value,
        **serialize_ticker(trade.ticker),
        'price': str(trade.price),
        'quantity': str(trade.quantity),
        'commission': str(trade.commission),
        'venue_order_id': trade.venue_order_id,
        'client_order_id': trade.client_order_id,
        'execution_id': trade.execution_id,
        'fee_source': trade.fee_source,
    }
    if timestamp is not None:
        d['timestamp'] = timestamp.isoformat()
    return d


def deserialize_trade(d: dict) -> Trade:
    """Reconstruct a Trade from a dict (timestamp key, if present, is ignored)."""
    return Trade(
        side=TradeSide(d['side']),
        ticker=deserialize_ticker(d),
        price=Decimal(d['price']),
        quantity=Decimal(d['quantity']),
        commission=Decimal(d['commission']),
        venue_order_id=d.get('venue_order_id', ''),
        client_order_id=d.get('client_order_id', ''),
        execution_id=d.get('execution_id', ''),
        fee_source=d.get('fee_source', ''),
    )


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------


def serialize_position(pos: Position) -> dict:
    """Serialize a Position (ticker fields inlined)."""
    return {
        **serialize_ticker(pos.ticker),
        'quantity': str(pos.quantity),
        'average_cost': str(pos.average_cost),
        'realized_pnl': str(pos.realized_pnl),
    }


def deserialize_position(d: dict) -> Position:
    """Reconstruct a Position from a dict."""
    return Position(
        ticker=deserialize_ticker(d),
        quantity=Decimal(d['quantity']),
        average_cost=Decimal(d['average_cost']),
        realized_pnl=Decimal(d['realized_pnl']),
    )


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------


def serialize_order(order: Order) -> dict:
    """Serialize an Order (ticker fields inlined, nested trades list)."""
    return {
        'status': order.status.value,
        'side': order.side.value,
        **serialize_ticker(order.ticker),
        'limit_price': str(order.limit_price),
        'filled_quantity': str(order.filled_quantity),
        'average_price': str(order.average_price),
        'remaining': str(order.remaining),
        'commission': str(order.commission),
        'trades': [serialize_trade(t) for t in order.trades],
        'venue_order_id': order.venue_order_id,
        'client_order_id': order.client_order_id,
        'execution_uncertain': order.execution_uncertain,
        'accounting_confirmed': order.accounting_confirmed,
        'reconciliation_reason': order.reconciliation_reason,
    }


def deserialize_order(d: dict) -> Order:
    """Reconstruct a full Order (including nested trades) from a dict."""
    return Order(
        status=OrderStatus(d['status']),
        side=TradeSide(d['side']),
        ticker=deserialize_ticker(d),
        limit_price=Decimal(d['limit_price']),
        filled_quantity=Decimal(d['filled_quantity']),
        average_price=Decimal(d['average_price']),
        remaining=Decimal(d['remaining']),
        commission=Decimal(d['commission']),
        trades=[deserialize_trade(t) for t in d.get('trades', [])],
        venue_order_id=d.get('venue_order_id', ''),
        client_order_id=d.get('client_order_id', ''),
        execution_uncertain=d.get('execution_uncertain', False),
        accounting_confirmed=d.get('accounting_confirmed', True),
        reconciliation_reason=d.get('reconciliation_reason', ''),
    )


# ---------------------------------------------------------------------------
# EquityPoint  (imported lazily to avoid circular imports with
#               performance_analyzer → serializers → performance_analyzer)
# ---------------------------------------------------------------------------


def serialize_equity_point(pt: object) -> dict:
    """Serialize an EquityPoint to a dict."""
    return {
        'timestamp': pt.timestamp,  # type: ignore[attr-defined]
        'equity': str(pt.equity),  # type: ignore[attr-defined]
        'trade_index': pt.trade_index,  # type: ignore[attr-defined]
    }


def deserialize_equity_point(d: dict) -> object:
    """Reconstruct an EquityPoint from a dict."""
    from oracle3.analytics.performance_analyzer import EquityPoint

    return EquityPoint(
        timestamp=d['timestamp'],
        equity=Decimal(d['equity']),
        trade_index=d['trade_index'],
    )
