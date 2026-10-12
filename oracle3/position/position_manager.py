from dataclasses import dataclass
from decimal import Decimal

from oracle3.data.market_data_manager import MarketDataManager
from oracle3.ticker.ticker import CashTicker, Ticker
from oracle3.trader.types import Trade, TradeSide


@dataclass
class Position:
    ticker: Ticker
    quantity: Decimal
    average_cost: Decimal
    realized_pnl: Decimal


class PositionManager:
    def __init__(self) -> None:
        self.positions: dict[str, Position] = {}
        self._last_marks: dict[str, Decimal] = {}

    def update_position(self, position: Position) -> None:
        """Update a position"""
        self.positions[position.ticker.symbol] = position

    @staticmethod
    def _require_finite(value: Decimal, name: str, *, nonnegative: bool = True) -> None:
        if not isinstance(value, Decimal) or not value.is_finite():
            raise ValueError(f'{name} must be a finite Decimal')
        if nonnegative and value < 0:
            raise ValueError(f'{name} must be nonnegative')

    def _validate_trade(self, trade: Trade) -> None:
        if not isinstance(trade.side, TradeSide):
            raise ValueError('Invalid trade side')
        if isinstance(trade.ticker, CashTicker):
            raise ValueError('Cash tickers cannot be traded')
        for name in ('quantity', 'price', 'commission'):
            self._require_finite(getattr(trade, name), name)
        if trade.quantity == 0:
            raise ValueError('Trade quantity must be positive')

        if trade.ticker.symbol == trade.ticker.collateral.symbol:
            raise ValueError('Trade ticker cannot share its collateral symbol')

    def apply_trade(self, trade: Trade) -> Position:
        """Apply a fill atomically after validating inventory and collateral.

        Invalid trades never create a position or partially change the ledger.
        Accounting permits zero-price fills, such as a worthless settlement.
        """
        self._validate_trade(trade)
        ticker = trade.ticker
        cash = self.positions.get(ticker.collateral.symbol)
        if cash is None or not isinstance(cash.ticker, CashTicker):
            raise ValueError(f'Missing collateral cash position for {ticker.symbol}')
        self._require_finite(cash.quantity, 'Cash balance')

        pos = self.positions.get(ticker.symbol)
        if pos is not None and isinstance(pos.ticker, CashTicker):
            raise ValueError('Trade ticker conflicts with a cash position')
        if pos is None:
            pos = Position(ticker, Decimal('0'), Decimal('0'), Decimal('0'))
        self._require_finite(pos.quantity, 'Position quantity')
        self._require_finite(pos.average_cost, 'Average cost')
        self._require_finite(pos.realized_pnl, 'Realized PnL', nonnegative=False)

        buying = trade.side == TradeSide.BUY
        if not buying and trade.quantity > pos.quantity:
            raise ValueError(
                f'Cannot sell {trade.quantity} {ticker}, '
                f'exceeding the current position {pos.quantity}'
            )
        signed_quantity = trade.quantity if buying else -trade.quantity
        new_quantity = pos.quantity + signed_quantity
        new_cash = cash.quantity - trade.price * signed_quantity - trade.commission
        self._require_finite(new_cash, 'Resulting cash balance')

        new_average_cost = pos.average_cost
        new_realized_pnl = pos.realized_pnl
        if buying:
            new_average_cost = (
                pos.quantity * pos.average_cost
                + trade.quantity * trade.price
                + trade.commission
            ) / new_quantity
        else:
            new_realized_pnl += (
                trade.price - pos.average_cost
            ) * trade.quantity - trade.commission
        if new_quantity == 0:
            new_average_cost = Decimal('0')
        self._require_finite(new_quantity, 'Resulting position quantity')
        self._require_finite(new_average_cost, 'Resulting average cost')
        self._require_finite(
            new_realized_pnl, 'Resulting realized PnL', nonnegative=False
        )

        # All arithmetic and preconditions succeeded; commit both ledger entries.
        pos.quantity = new_quantity
        pos.average_cost = new_average_cost
        pos.realized_pnl = new_realized_pnl
        cash.quantity = new_cash
        self.positions[ticker.symbol] = pos
        return pos

    def _current_mark(
        self, ticker: Ticker, market_data: MarketDataManager
    ) -> Decimal | None:
        """Prefer a usable bid, including zero, then a usable ask."""
        for level in (
            market_data.get_best_bid(ticker),
            market_data.get_best_ask(ticker),
        ):
            if (
                level is not None
                and isinstance(level.price, Decimal)
                and level.price.is_finite()
                and 0 <= level.price <= 1
            ):
                self._last_marks[ticker.symbol] = level.price
                return level.price
        return None

    def _position_mark(
        self, position: Position, market_data: MarketDataManager
    ) -> Decimal:
        current = self._current_mark(position.ticker, market_data)
        if current is not None:
            return current
        return self._last_marks.get(position.ticker.symbol, position.average_cost)

    def get_missing_marks(self, market_data: MarketDataManager) -> list[str]:
        """Held assets without a usable current quote, even when a fallback exists.

        Valuations retain the last observed mark or cost basis for these assets;
        callers must not interpret those fallback values as current market prices.
        """
        return sorted(
            position.ticker.symbol
            for position in self.get_non_cash_positions()
            if position.quantity > 0
            and self._current_mark(position.ticker, market_data) is None
        )

    def get_position(self, ticker: Ticker) -> Position | None:
        """Get current position for a ticker"""
        return self.positions.get(ticker.symbol)

    def get_unrealized_pnl(
        self, ticker: Ticker, market_data: MarketDataManager
    ) -> Decimal:
        """Calculate unrealized PnL for a ticker"""
        if ticker.symbol not in self.positions:
            return Decimal('0')

        if isinstance(ticker, CashTicker):
            return Decimal('0')

        position = self.positions[ticker.symbol]
        if position.quantity <= 0:
            return Decimal('0')

        current_price = self._position_mark(position, market_data)
        return (current_price - position.average_cost) * position.quantity

    def get_realized_pnl(self, ticker: Ticker) -> Decimal:
        """Get realized PnL for a ticker"""
        if ticker.symbol not in self.positions:
            return Decimal('0')
        return self.positions[ticker.symbol].realized_pnl

    def get_pnl(self, ticker: Ticker, market_data: MarketDataManager) -> Decimal:
        """Get PnL for a ticker"""
        return self.get_realized_pnl(ticker) + self.get_unrealized_pnl(
            ticker, market_data
        )

    def get_total_realized_pnl(self) -> Decimal:
        """Get total realized PnL across all positions"""
        return sum(
            (pos.realized_pnl for pos in self.positions.values()), Decimal('0.0')
        )

    def get_total_unrealized_pnl(self, market_data: MarketDataManager) -> Decimal:
        """Get total unrealized PnL across all positions"""
        return sum(
            (
                self.get_unrealized_pnl(pos.ticker, market_data)
                for pos in self.positions.values()
            ),
            Decimal('0.0'),
        )

    def get_total_pnl(self, market_data: MarketDataManager) -> Decimal:
        """Get total PnL across all positions"""
        return self.get_total_realized_pnl() + self.get_total_unrealized_pnl(
            market_data
        )

    def get_cash_positions(self) -> list[Position]:
        """Get all cash positions"""
        return [
            pos for pos in self.positions.values() if isinstance(pos.ticker, CashTicker)
        ]

    def get_non_cash_positions(self) -> list[Position]:
        """Get all non-cash positions"""
        return [
            pos
            for pos in self.positions.values()
            if not isinstance(pos.ticker, CashTicker)
        ]

    def get_portfolio_value(self, market_data: MarketDataManager) -> dict[str, Decimal]:
        """Get portfolio value by collateral currencies.

        Missing quotes retain a prior observed mark or cost basis. Inspect
        ``get_missing_marks`` to distinguish these estimates from current marks.

        Returns:
            A dictionary where the keys are CashTicker symbols and
            the values are the total portfolio value in that collateral currency
        """
        portfolio_value: dict[str, Decimal] = {}

        for cash_pos in self.get_cash_positions():
            portfolio_value[cash_pos.ticker.symbol] = cash_pos.quantity

        for pos in self.get_non_cash_positions():
            if pos.quantity <= 0:
                continue
            collateral = pos.ticker.collateral

            current_price = self._position_mark(pos, market_data)
            position_value = pos.quantity * current_price

            portfolio_value[collateral.symbol] = (
                portfolio_value.get(collateral.symbol, Decimal('0')) + position_value
            )

        return portfolio_value
