"""Long-only binary-contract accounting and explicitly marked performance.

Equity is cash plus inventory at the last observed price (initially its fill
price). A missing mark never values a holding at zero. Win/loss statistics refer
to FIFO matched closing lots, not cash flows. Per-fill observations do not define
a return sampling period, so annualized Sharpe is deliberately unavailable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from oracle3.ticker.ticker import Ticker
from oracle3.trader.types import Trade, TradeSide

ZERO = Decimal('0')


@dataclass
class TradeStats:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal
    average_profit: Decimal
    average_loss: Decimal
    max_drawdown: Decimal
    sharpe_ratio: Decimal
    profit_factor: Decimal = ZERO
    total_pnl: Decimal = ZERO
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    total_fills: int = 0
    realized_pnl: Decimal = ZERO
    unrealized_pnl: Decimal = ZERO
    total_fees: Decimal = ZERO
    cash: Decimal = ZERO
    inventory_value: Decimal = ZERO
    valuation_complete: bool = True
    missing_marks: tuple[str, ...] = ()
    sharpe_available: bool = False
    sharpe_reason: str = 'No regularly sampled, timestamped portfolio returns supplied'


@dataclass
class EquityPoint:
    """Marked NAV observation; timestamp=0 means unknown, never a synthetic day."""

    timestamp: int
    equity: Decimal
    trade_index: int


@dataclass
class InventoryLot:
    quantity: Decimal
    price: Decimal
    entry_fee: Decimal


@dataclass
class ClosedLot:
    """One FIFO-matched quantity, with both sides' fees allocated to that quantity."""

    ticker: Ticker
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal
    entry_fee: Decimal
    exit_fee: Decimal
    pnl: Decimal
    timestamp: int
    reason: str


def _nonnegative(value: Decimal, name: str) -> None:
    if not value.is_finite() or value < ZERO:
        raise ValueError(f'{name} must be finite and nonnegative')


def _price(value: Decimal) -> None:
    _nonnegative(value, 'price')
    if value > 1:
        raise ValueError('binary contract price must be in [0, 1]')


class PerformanceAnalyzer:
    """Cash/inventory ledger for a single account with a common dollar numeraire.

    Opening fills establish FIFO lots. An unbacked sale is rejected; opening
    holdings must be loaded explicitly rather than inventing zero cost basis.
    ``total_trades`` now counts matched closing lots; ``total_fills`` counts fills.
    ``sharpe_ratio=0`` is a compatibility placeholder, not a computed estimate;
    consumers must inspect ``sharpe_available``.
    """

    def __init__(self, initial_capital: Decimal = Decimal('10000')) -> None:
        _nonnegative(initial_capital, 'initial_capital')
        self.initial_capital = initial_capital
        self.reset()

    def reset(self) -> None:
        self.trades: list[Trade] = []
        self.closed_lots: list[ClosedLot] = []
        self.cash = self.initial_capital
        self.total_fees = ZERO
        self.realized_pnl = ZERO
        self._lots: dict[Ticker, list[InventoryLot]] = {}
        self._marks: dict[Ticker, Decimal] = {}
        self._missing_marks: set[Ticker] = set()
        self._settled: dict[Ticker, int] = {}
        self._events: list[dict[str, Any]] = []
        self.equity_curve = [EquityPoint(0, self.initial_capital, -1)]

    def position_quantity(self, ticker: Ticker) -> Decimal:
        return sum((lot.quantity for lot in self._lots.get(ticker, [])), ZERO)

    def inventory_tickers(self) -> set[Ticker]:
        """Return a copy of the tickers with positive tracked inventory."""
        return set(self._lots)

    def seed_position(
        self,
        ticker: Ticker,
        quantity: Decimal,
        cost_per_share: Decimal,
        mark_price: Decimal | None = None,
    ) -> None:
        """Load opening inventory against an initial cash-plus-cost-basis baseline.

        Call before fills/marks/settlements. Cost basis may include past fees
        and exceed one; it is never silently treated as a current market quote.
        An absent quote falls back to basis and is flagged as an incomplete
        valuation. After loading all holdings, mark the complete portfolio to
        record an opening valuation, without order-dependent intermediate NAVs.
        No fill, new fee, or closed-lot statistic is created.
        """
        if any(event['kind'] != 'seed' for event in self._events):
            raise ValueError('opening positions must be seeded before other events')
        _nonnegative(quantity, 'quantity')
        _nonnegative(cost_per_share, 'cost_per_share')
        if quantity == 0:
            raise ValueError('opening quantity must be positive')
        if mark_price is not None:
            _price(mark_price)
        self.cash -= quantity * cost_per_share
        self._lots.setdefault(ticker, []).append(
            InventoryLot(quantity, cost_per_share, ZERO)
        )
        self._marks[ticker] = cost_per_share if mark_price is None else mark_price
        if mark_price is None:
            self._missing_marks.add(ticker)
        else:
            self._missing_marks.discard(ticker)
        self._events.append(
            {
                'kind': 'seed',
                'ticker': ticker,
                'quantity': quantity,
                'cost_per_share': cost_per_share,
                'mark_price': mark_price,
            }
        )

    def _record_equity(self, timestamp: int | None) -> None:
        self.equity_curve.append(
            EquityPoint(timestamp or 0, self.get_current_equity(), len(self.trades) - 1)
        )

    def add_trade(self, trade: Trade, *, timestamp: int | None = None) -> None:
        """Apply an actual fill once. Invalid fills leave the ledger unchanged."""
        _price(trade.price)
        _nonnegative(trade.quantity, 'quantity')
        _nonnegative(trade.commission, 'commission')
        if trade.quantity == 0:
            raise ValueError('filled quantity must be positive')
        if trade.ticker in self._settled:
            raise ValueError('cannot trade a settled contract')
        if trade.side not in (TradeSide.BUY, TradeSide.SELL):
            raise ValueError('unsupported trade side')
        if trade.side == TradeSide.SELL and trade.quantity > self.position_quantity(
            trade.ticker
        ):
            raise ValueError(
                'sale exceeds tracked inventory; load opening holdings first'
            )

        if trade.side == TradeSide.BUY:
            self.cash -= trade.price * trade.quantity + trade.commission
            self._lots.setdefault(trade.ticker, []).append(
                InventoryLot(trade.quantity, trade.price, trade.commission)
            )
        else:
            self._close(
                trade.ticker,
                trade.quantity,
                trade.price,
                trade.commission,
                timestamp,
                'sale',
            )
        self.total_fees += trade.commission
        self._marks[trade.ticker] = trade.price
        self._missing_marks.discard(trade.ticker)
        self.trades.append(trade)
        self._events.append({'kind': 'trade', 'trade': trade, 'timestamp': timestamp})
        self._record_equity(timestamp)

    def _close(
        self,
        ticker: Ticker,
        quantity: Decimal,
        price: Decimal,
        fee: Decimal,
        timestamp: int | None,
        reason: str,
    ) -> None:
        remaining, remaining_fee = quantity, fee
        lots = self._lots[ticker]
        while remaining > 0:
            lot = lots[0]
            matched = min(remaining, lot.quantity)
            entry_fee = (
                lot.entry_fee
                if matched == lot.quantity
                else lot.entry_fee * matched / lot.quantity
            )
            exit_fee = (
                remaining_fee if matched == remaining else fee * matched / quantity
            )
            pnl = matched * (price - lot.price) - entry_fee - exit_fee
            self.closed_lots.append(
                ClosedLot(
                    ticker,
                    matched,
                    lot.price,
                    price,
                    entry_fee,
                    exit_fee,
                    pnl,
                    timestamp or 0,
                    reason,
                )
            )
            self.realized_pnl += pnl
            remaining -= matched
            remaining_fee -= exit_fee
            lot.quantity -= matched
            lot.entry_fee -= entry_fee
            if lot.quantity == 0:
                lots.pop(0)
        if not lots:
            self._lots.pop(ticker)
            self._missing_marks.discard(ticker)
        self.cash += quantity * price - fee

    def mark_to_market(
        self,
        prices: Mapping[Ticker, Decimal],
        *,
        timestamp: int | None = None,
    ) -> Decimal:
        """Record an observed valuation snapshot, carrying omitted holdings' marks.

        Missing entries are reported through ``missing_marks`` and
        ``valuation_complete=False``. Carried values are the last observed quote
        or fill price, not zero and not an assertion that a quote is current.
        The caller chooses a consistent mark convention (e.g. executable bid).
        """
        for price in prices.values():
            _price(price)
        self._marks.update(prices)
        self._missing_marks = set(self._lots) - set(prices)
        self._events.append(
            {'kind': 'mark', 'prices': dict(prices), 'timestamp': timestamp}
        )
        self._record_equity(timestamp)
        return self.get_current_equity()

    def settle(
        self,
        ticker: Ticker,
        outcome: int,
        *,
        commission: Decimal = ZERO,
        timestamp: int | None = None,
    ) -> Decimal:
        """Redeem all held shares at a known 0/1 outcome; return realized increment.

        This is an accounting API, not resolution discovery. Repeating the same
        settlement with no new fee is idempotent; conflicting outcomes fail.
        """
        if outcome not in (0, 1):
            raise ValueError('outcome must be 0 or 1')
        _nonnegative(commission, 'commission')
        if ticker in self._settled:
            if self._settled[ticker] == outcome and commission == 0:
                return ZERO
            raise ValueError('contract has already settled')
        quantity = self.position_quantity(ticker)
        if quantity == 0 and commission != 0:
            raise ValueError('cannot allocate settlement commission without inventory')
        before = self.realized_pnl
        if quantity > 0:
            self._close(
                ticker, quantity, Decimal(outcome), commission, timestamp, 'settlement'
            )
        self.total_fees += commission
        self._settled[ticker] = outcome
        self._marks[ticker] = Decimal(outcome)
        self._events.append(
            {
                'kind': 'settlement',
                'ticker': ticker,
                'outcome': outcome,
                'commission': commission,
                'timestamp': timestamp,
            }
        )
        self._record_equity(timestamp)
        return self.realized_pnl - before

    @property
    def inventory_value(self) -> Decimal:
        return sum(
            (self.position_quantity(t) * self._marks[t] for t in self._lots), ZERO
        )

    @property
    def unrealized_pnl(self) -> Decimal:
        return sum(
            (
                lot.quantity * (self._marks[ticker] - lot.price) - lot.entry_fee
                for ticker, lots in self._lots.items()
                for lot in lots
            ),
            ZERO,
        )

    def get_current_equity(self) -> Decimal:
        return self.cash + self.inventory_value

    def get_equity_curve(self) -> list[EquityPoint]:
        return self.equity_curve.copy()

    def get_return_pct(self) -> Decimal:
        if self.initial_capital == 0:
            return ZERO
        return (self.get_current_equity() / self.initial_capital - 1) * 100

    def _calculate_max_drawdown(self) -> Decimal:
        peak, maximum = self.initial_capital, ZERO
        for point in self.equity_curve:
            peak = max(peak, point.equity)
            if peak > 0:
                maximum = max(maximum, (peak - point.equity) / peak)
        return maximum

    def _calculate_consecutive_streaks(self) -> tuple[int, int]:
        wins = losses = max_wins = max_losses = 0
        for lot in self.closed_lots:
            wins = wins + 1 if lot.pnl > 0 else 0
            losses = losses + 1 if lot.pnl < 0 else 0
            max_wins, max_losses = max(max_wins, wins), max(max_losses, losses)
        return max_wins, max_losses

    def get_stats(self) -> TradeStats:
        """Statistics on FIFO closing lots and marked portfolio NAV, net of fees."""
        wins = [lot.pnl for lot in self.closed_lots if lot.pnl > 0]
        losses = [lot.pnl for lot in self.closed_lots if lot.pnl < 0]
        count = len(self.closed_lots)
        gross_profit, gross_loss = sum(wins, ZERO), -sum(losses, ZERO)
        profit_factor = (
            gross_profit / gross_loss
            if gross_loss
            else Decimal('Infinity')
            if gross_profit
            else ZERO
        )
        max_wins, max_losses = self._calculate_consecutive_streaks()
        return TradeStats(
            total_trades=count,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=Decimal(len(wins)) / count if count else ZERO,
            average_profit=gross_profit / len(wins) if wins else ZERO,
            average_loss=-gross_loss / len(losses) if losses else ZERO,
            max_drawdown=self._calculate_max_drawdown(),
            sharpe_ratio=ZERO,
            profit_factor=profit_factor,
            total_pnl=self.get_current_equity() - self.initial_capital,
            max_consecutive_wins=max_wins,
            max_consecutive_losses=max_losses,
            total_fills=len(self.trades),
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
            total_fees=self.total_fees,
            cash=self.cash,
            inventory_value=self.inventory_value,
            valuation_complete=not self._missing_marks,
            missing_marks=tuple(sorted(t.symbol for t in self._missing_marks)),
        )

    def print_summary(self) -> None:
        stats = self.get_stats()
        print('\nPERFORMANCE SUMMARY')
        print(f'Initial Capital: ${self.initial_capital:,.2f}')
        print(f'Current Equity: ${self.get_current_equity():,.2f}')
        print(f'Total Trades (closed FIFO lots): {stats.total_trades}')
        print(f'Total Fills: {stats.total_fills}')
        print(f'Total PnL: ${stats.total_pnl:,.2f}')
        print(
            f'Realized / Unrealized: ${stats.realized_pnl:,.2f} / ${stats.unrealized_pnl:,.2f}'
        )
        print(f'Win Rate: {stats.win_rate:.2%}')
        print(f'Max Drawdown: {stats.max_drawdown:.2%}')
        print(f'Sharpe Ratio: unavailable ({stats.sharpe_reason})')
        if not stats.valuation_complete:
            print(f'Carried marks: {", ".join(stats.missing_marks)}')

    def to_dict(self) -> dict[str, Any]:
        """Versioned event replay preserves marks, fee allocation and settlement."""
        from oracle3.storage.serializers import (
            serialize_equity_point,
            serialize_ticker,
            serialize_trade,
        )

        events = []
        for event in self._events:
            saved = dict(event)
            if event['kind'] == 'trade':
                saved['trade'] = serialize_trade(event['trade'])
            elif event['kind'] == 'mark':
                saved['prices'] = [
                    {'ticker': serialize_ticker(t), 'price': str(p)}
                    for t, p in event['prices'].items()
                ]
            elif event['kind'] == 'seed':
                saved['ticker'] = serialize_ticker(event['ticker'])
                saved['quantity'] = str(event['quantity'])
                saved['cost_per_share'] = str(event['cost_per_share'])
                saved['mark_price'] = (
                    str(event['mark_price'])
                    if event['mark_price'] is not None
                    else None
                )
            else:
                saved['ticker'] = serialize_ticker(event['ticker'])
                saved['commission'] = str(event['commission'])
            events.append(saved)
        return {
            'accounting_version': 2,
            'initial_capital': str(self.initial_capital),
            'events': events,
            'trades': [serialize_trade(t) for t in self.trades],
            'equity_curve': [serialize_equity_point(p) for p in self.equity_curve],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerformanceAnalyzer:
        """Replay v2 events, or migrate old fills without trusting old cash curves.

        A legacy sale without preceding purchases raises rather than inventing
        cost basis. Legacy snapshots cannot recover marks or missing settlements.
        """
        from oracle3.storage.serializers import deserialize_ticker, deserialize_trade

        analyzer = cls(Decimal(data['initial_capital']))
        if 'accounting_version' not in data:
            for trade in data.get('trades', []):
                analyzer.add_trade(deserialize_trade(trade))
            return analyzer
        if data['accounting_version'] != 2:
            raise ValueError('unsupported accounting snapshot version')
        for event in data['events']:
            timestamp = event.get('timestamp')
            if event['kind'] == 'trade':
                analyzer.add_trade(
                    deserialize_trade(event['trade']), timestamp=timestamp
                )
            elif event['kind'] == 'mark':
                analyzer.mark_to_market(
                    {
                        deserialize_ticker(row['ticker']): Decimal(row['price'])
                        for row in event['prices']
                    },
                    timestamp=timestamp,
                )
            elif event['kind'] == 'settlement':
                analyzer.settle(
                    deserialize_ticker(event['ticker']),
                    event['outcome'],
                    commission=Decimal(event['commission']),
                    timestamp=timestamp,
                )
            elif event['kind'] == 'seed':
                mark = event.get('mark_price')
                analyzer.seed_position(
                    deserialize_ticker(event['ticker']),
                    Decimal(event['quantity']),
                    Decimal(event['cost_per_share']),
                    mark_price=Decimal(mark) if mark is not None else None,
                )
            else:
                raise ValueError(f'unknown accounting event: {event["kind"]}')
        return analyzer
