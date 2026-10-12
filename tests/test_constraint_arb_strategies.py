"""Tests for constraint-based arbitrage strategies.

Covers: ExclusivityArbStrategy, ImplicationArbStrategy,
ConditionalArbStrategy, EventSumArbStrategy, StructuralArbStrategy.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from oracle3.data.market_data_manager import MarketDataManager
from oracle3.events.events import NewsEvent, PriceChangeEvent
from oracle3.order.order_book import Level, OrderBook
from oracle3.position.position_manager import Position, PositionManager
from oracle3.risk.risk_manager import NoRiskManager
from oracle3.strategy.contrib.conditional_arb_strategy import (
    ConditionalArbStrategy,
)
from oracle3.strategy.contrib.event_sum_arb_strategy import EventSumArbStrategy
from oracle3.strategy.contrib.exclusivity_arb_strategy import (
    ExclusivityArbStrategy,
)
from oracle3.strategy.contrib.implication_arb_strategy import (
    ImplicationArbStrategy,
)
from oracle3.strategy.contrib.structural_arb_strategy import (
    StructuralArbStrategy,
)
from oracle3.ticker.ticker import CashTicker, PolyMarketTicker
from oracle3.trader.paper_trader import PaperTrader
from oracle3.trader.types import Order, OrderStatus, PlaceOrderResult, TradeSide

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ticker(
    symbol: str,
    name: str = '',
    market_id: str = '',
    event_id: str = '',
    no_token_id: str = '',
) -> PolyMarketTicker:
    return PolyMarketTicker(
        symbol=symbol,
        name=name or f'YES {symbol}',
        token_id=symbol,
        market_id=market_id or symbol,
        event_id=event_id,
        no_token_id=no_token_id,
    )


def _make_trader(
    tickers: list[PolyMarketTicker] | None = None,
    cash: Decimal = Decimal('10000'),
) -> PaperTrader:
    """Create a PaperTrader with 100% fill rate and no commissions."""
    md = MarketDataManager()
    pm = PositionManager()
    pm.update_position(
        Position(
            ticker=CashTicker.POLYMARKET_USDC,
            quantity=cash,
            average_cost=Decimal('1'),
            realized_pnl=Decimal('0'),
        )
    )
    if tickers:
        for t in tickers:
            ob = OrderBook()
            # Provide some liquidity around mid-price
            ob.update(
                asks=[Level(price=Decimal('0.55'), size=Decimal('1000'))],
                bids=[Level(price=Decimal('0.45'), size=Decimal('1000'))],
            )
            md.update_order_book(t, ob)
    trader = PaperTrader(
        market_data=md,
        risk_manager=NoRiskManager(),
        position_manager=pm,
        min_fill_rate=Decimal('1'),
        max_fill_rate=Decimal('1'),
        commission_rate=Decimal('0'),
    )
    return trader


def _seed_order_book(
    trader: PaperTrader,
    ticker: PolyMarketTicker,
    bid: Decimal,
    ask: Decimal,
    size: Decimal = Decimal('1000'),
) -> None:
    ob = OrderBook()
    ob.update(
        asks=[Level(price=ask, size=size)],
        bids=[Level(price=bid, size=size)],
    )
    trader.market_data.update_order_book(ticker, ob)


def _price_event(
    ticker: PolyMarketTicker, price: Decimal
) -> PriceChangeEvent:
    return PriceChangeEvent(ticker=ticker, price=price)


# ===================================================================
# ExclusivityArbStrategy Tests
# ===================================================================


class TestExclusivityArbDefaults:
    """Test constructor defaults and basic attribute setup."""

    def test_defaults(self):
        s = ExclusivityArbStrategy()
        assert s._id_a == ''
        assert s._id_b == ''
        assert s.trade_size == Decimal('10')
        assert s.min_edge == Decimal('0.02')
        assert s.cooldown_seconds == 120.0
        assert s.fee_rate is None
        assert s._position_state == 'flat'
        assert s._price_a is None
        assert s._price_b is None

    def test_custom_params(self):
        s = ExclusivityArbStrategy(
            market_id_a='mkt_a',
            market_id_b='mkt_b',
            trade_size=50.0,
            min_edge=0.05,
            cooldown_seconds=60.0,
            fee_rate=0.01,
        )
        assert s._id_a == 'mkt_a'
        assert s._id_b == 'mkt_b'
        assert s.trade_size == Decimal('50')
        assert s.min_edge == Decimal('0.05')

    def test_name_and_version(self):
        assert ExclusivityArbStrategy.name == 'exclusivity_arb'
        assert ExclusivityArbStrategy.version == '1.0.0'
        assert ExclusivityArbStrategy.supports_auto_tune()


class TestExclusivityArbPriceRegistration:
    """Test that price events are correctly registered."""

    async def test_registers_price_a(self):
        ticker_a = _make_ticker('A', market_id='mkt_a')
        s = ExclusivityArbStrategy(market_id_a='mkt_a', market_id_b='mkt_b')
        trader = _make_trader([ticker_a])
        event = _price_event(ticker_a, Decimal('0.40'))
        await s.process_event(event, trader)
        assert s._price_a == Decimal('0.40')
        assert s._price_b is None

    async def test_registers_price_b(self):
        ticker_b = _make_ticker('B', market_id='mkt_b')
        s = ExclusivityArbStrategy(market_id_a='mkt_a', market_id_b='mkt_b')
        trader = _make_trader([ticker_b])
        event = _price_event(ticker_b, Decimal('0.30'))
        await s.process_event(event, trader)
        assert s._price_b == Decimal('0.30')

    async def test_ignores_no_side_events(self):
        ticker_no = _make_ticker('A_NO', market_id='mkt_a')
        s = ExclusivityArbStrategy(market_id_a='mkt_a', market_id_b='mkt_b')
        trader = _make_trader([ticker_no])
        event = _price_event(ticker_no, Decimal('0.60'))
        await s.process_event(event, trader)
        assert s._price_a is None

    async def test_ignores_unrelated_events(self):
        ticker_x = _make_ticker('X', market_id='mkt_x')
        s = ExclusivityArbStrategy(market_id_a='mkt_a', market_id_b='mkt_b')
        trader = _make_trader([ticker_x])
        event = _price_event(ticker_x, Decimal('0.50'))
        await s.process_event(event, trader)
        assert s._price_a is None
        assert s._price_b is None

    async def test_ignores_non_price_events(self):
        s = ExclusivityArbStrategy(market_id_a='mkt_a', market_id_b='mkt_b')
        trader = _make_trader()
        news = NewsEvent(news='Breaking news')
        await s.process_event(news, trader)
        assert s._price_a is None


class TestExclusivityArbViolation:
    """Test violation detection (positive and negative cases)."""

    async def test_no_violation_when_sum_below_one(self):
        """A + B < 1 => no entry."""
        ticker_a = _make_ticker('A', market_id='mkt_a')
        ticker_b = _make_ticker('B', market_id='mkt_b')
        s = ExclusivityArbStrategy(
            market_id_a='mkt_a', market_id_b='mkt_b', min_edge=0.02
        )
        trader = _make_trader([ticker_a, ticker_b])

        await s.process_event(_price_event(ticker_a, Decimal('0.40')), trader)
        await s.process_event(_price_event(ticker_b, Decimal('0.30')), trader)

        assert s._position_state == 'flat'
        decisions = s.get_decisions()
        assert len(decisions) >= 1
        assert decisions[-1].action == 'HOLD'

    async def test_violation_detected_when_sum_exceeds_one(self):
        """A + B > 1 by more than min_edge => ENTER_ARB."""
        ticker_a = _make_ticker(
            'A', market_id='mkt_a', no_token_id='A_NO'
        )
        ticker_a_no = ticker_a.get_no_ticker()
        ticker_b = _make_ticker(
            'B', market_id='mkt_b', no_token_id='B_NO'
        )
        ticker_b_no = ticker_b.get_no_ticker()
        s = ExclusivityArbStrategy(
            market_id_a='mkt_a',
            market_id_b='mkt_b',
            min_edge=0.01,
            fee_rate=0.0,
            cooldown_seconds=0.0,
        )
        trader = _make_trader()

        # Seed order books for YES and NO tickers
        for t in [ticker_a, ticker_b]:
            _seed_order_book(trader, t, Decimal('0.55'), Decimal('0.65'))
        for t in [ticker_a_no, ticker_b_no]:
            _seed_order_book(trader, t, Decimal('0.35'), Decimal('0.45'))

        # Send price events that create a violation: 0.60 + 0.55 = 1.15 > 1
        await s.process_event(_price_event(ticker_a, Decimal('0.60')), trader)
        await s.process_event(_price_event(ticker_b, Decimal('0.55')), trader)

        assert s._position_state == 'short_both'
        decisions = s.get_decisions()
        enter_decisions = [d for d in decisions if d.action == 'ENTER_ARB']
        assert len(enter_decisions) >= 1

    async def test_no_entry_when_edge_below_minimum(self):
        """Violation exists but below min_edge => hold."""
        ticker_a = _make_ticker('A', market_id='mkt_a')
        ticker_b = _make_ticker('B', market_id='mkt_b')
        s = ExclusivityArbStrategy(
            market_id_a='mkt_a',
            market_id_b='mkt_b',
            min_edge=0.20,
            fee_rate=0.0,
        )
        trader = _make_trader([ticker_a, ticker_b])

        # Sum = 1.05, violation = 0.05, min_edge = 0.20
        await s.process_event(_price_event(ticker_a, Decimal('0.55')), trader)
        await s.process_event(_price_event(ticker_b, Decimal('0.50')), trader)

        assert s._position_state == 'flat'


class TestExclusivityArbPositionStates:
    """Test position state transitions: flat -> short_both -> flat."""

    async def test_full_lifecycle(self):
        """Enter on violation, exit when constraint restored."""
        ticker_a = _make_ticker('A', market_id='mkt_a', no_token_id='A_NO')
        ticker_a_no = ticker_a.get_no_ticker()
        ticker_b = _make_ticker('B', market_id='mkt_b', no_token_id='B_NO')
        ticker_b_no = ticker_b.get_no_ticker()
        s = ExclusivityArbStrategy(
            market_id_a='mkt_a',
            market_id_b='mkt_b',
            min_edge=0.01,
            fee_rate=0.0,
            cooldown_seconds=0.0,
            trade_size=5.0,
        )
        trader = _make_trader()

        # Seed all order books
        for t in [ticker_a, ticker_b]:
            _seed_order_book(trader, t, Decimal('0.55'), Decimal('0.65'))
        for t in [ticker_a_no, ticker_b_no]:
            _seed_order_book(trader, t, Decimal('0.35'), Decimal('0.45'))

        # Enter: sum = 1.15 > 1
        await s.process_event(_price_event(ticker_a, Decimal('0.60')), trader)
        await s.process_event(_price_event(ticker_b, Decimal('0.55')), trader)
        assert s._position_state == 'short_both'

        # Exit: sum = 0.80 <= 1
        await s.process_event(_price_event(ticker_a, Decimal('0.40')), trader)
        await s.process_event(_price_event(ticker_b, Decimal('0.40')), trader)
        assert s._position_state == 'flat'


class TestExclusivityArbCooldown:
    """Test cooldown enforcement between entries."""

    async def test_cooldown_prevents_reentry(self):
        ticker_a = _make_ticker('A', market_id='mkt_a', no_token_id='A_NO')
        ticker_a_no = ticker_a.get_no_ticker()
        ticker_b = _make_ticker('B', market_id='mkt_b', no_token_id='B_NO')
        ticker_b_no = ticker_b.get_no_ticker()
        s = ExclusivityArbStrategy(
            market_id_a='mkt_a',
            market_id_b='mkt_b',
            min_edge=0.01,
            fee_rate=0.0,
            cooldown_seconds=9999.0,  # very long cooldown
        )
        trader = _make_trader()
        for t in [ticker_a, ticker_b]:
            _seed_order_book(trader, t, Decimal('0.55'), Decimal('0.65'))
        for t in [ticker_a_no, ticker_b_no]:
            _seed_order_book(trader, t, Decimal('0.35'), Decimal('0.45'))

        # First entry should succeed
        await s.process_event(_price_event(ticker_a, Decimal('0.60')), trader)
        await s.process_event(_price_event(ticker_b, Decimal('0.55')), trader)
        assert s._position_state == 'short_both'

        # Manually reset to flat to test re-entry
        s._position_state = 'flat'

        # Second entry should be blocked by cooldown
        await s.process_event(_price_event(ticker_a, Decimal('0.60')), trader)
        await s.process_event(_price_event(ticker_b, Decimal('0.55')), trader)
        assert s._position_state == 'flat'


class TestExclusivityArbFeeAware:
    """Test that fee buffer is applied to edge filtering."""

    async def test_fee_rate_filters_marginal_edges(self):
        """When the violation minus fees is not positive, no entry."""
        ticker_a = _make_ticker('A', market_id='mkt_a', no_token_id='A_NO')
        ticker_b = _make_ticker('B', market_id='mkt_b', no_token_id='B_NO')
        s = ExclusivityArbStrategy(
            market_id_a='mkt_a',
            market_id_b='mkt_b',
            min_edge=0.01,
            fee_rate=0.10,  # 10% fees -- eliminates marginal edge
            cooldown_seconds=0.0,
        )
        trader = _make_trader([ticker_a, ticker_b])

        # Sum = 1.03, violation = 0.03 but after fees should not be profitable
        await s.process_event(_price_event(ticker_a, Decimal('0.52')), trader)
        await s.process_event(_price_event(ticker_b, Decimal('0.51')), trader)
        assert s._position_state == 'flat'


class TestExclusivityArbPaused:
    """Test that paused strategy ignores events."""

    async def test_paused_strategy_does_nothing(self):
        ticker_a = _make_ticker('A', market_id='mkt_a')
        s = ExclusivityArbStrategy(market_id_a='mkt_a', market_id_b='mkt_b')
        s.set_paused(True)
        trader = _make_trader([ticker_a])
        await s.process_event(_price_event(ticker_a, Decimal('0.60')), trader)
        assert s._price_a is None


# ===================================================================
# ImplicationArbStrategy Tests
# ===================================================================


class TestImplicationArbDefaults:
    def test_defaults(self):
        s = ImplicationArbStrategy()
        assert s._id_a == ''
        assert s._id_b == ''
        assert s._position_state == 'flat'
        assert s.min_edge == Decimal('0.01')

    def test_name(self):
        assert ImplicationArbStrategy.name == 'implication_arb'


class TestImplicationArbViolation:
    """Implication: A <= B. Violation when A > B."""

    async def test_no_violation_when_a_leq_b(self):
        ticker_a = _make_ticker('A', market_id='mkt_a')
        ticker_b = _make_ticker('B', market_id='mkt_b')
        s = ImplicationArbStrategy(
            market_id_a='mkt_a', market_id_b='mkt_b',
            min_edge=0.01, fee_rate=0.0,
        )
        trader = _make_trader([ticker_a, ticker_b])
        await s.process_event(_price_event(ticker_a, Decimal('0.30')), trader)
        await s.process_event(_price_event(ticker_b, Decimal('0.50')), trader)
        assert s._position_state == 'flat'

    async def test_violation_when_a_gt_b(self):
        ticker_a = _make_ticker('A', market_id='mkt_a', no_token_id='A_NO')
        ticker_a_no = ticker_a.get_no_ticker()
        ticker_b = _make_ticker('B', market_id='mkt_b')
        s = ImplicationArbStrategy(
            market_id_a='mkt_a', market_id_b='mkt_b',
            min_edge=0.01, fee_rate=0.0, cooldown_seconds=0.0,
        )
        trader = _make_trader()
        _seed_order_book(trader, ticker_a, Decimal('0.55'), Decimal('0.65'))
        _seed_order_book(trader, ticker_a_no, Decimal('0.35'), Decimal('0.45'))
        _seed_order_book(trader, ticker_b, Decimal('0.25'), Decimal('0.35'))

        # A=0.60 > B=0.30 => violation
        await s.process_event(_price_event(ticker_a, Decimal('0.60')), trader)
        await s.process_event(_price_event(ticker_b, Decimal('0.30')), trader)
        assert s._position_state == 'short_a_long_b'

    async def test_exit_when_constraint_restored(self):
        ticker_a = _make_ticker('A', market_id='mkt_a', no_token_id='A_NO')
        ticker_a_no = ticker_a.get_no_ticker()
        ticker_b = _make_ticker('B', market_id='mkt_b')
        s = ImplicationArbStrategy(
            market_id_a='mkt_a', market_id_b='mkt_b',
            min_edge=0.01, fee_rate=0.0, cooldown_seconds=0.0,
            trade_size=5.0,
        )
        trader = _make_trader()
        _seed_order_book(trader, ticker_a, Decimal('0.55'), Decimal('0.65'))
        _seed_order_book(trader, ticker_a_no, Decimal('0.35'), Decimal('0.45'))
        _seed_order_book(trader, ticker_b, Decimal('0.25'), Decimal('0.35'))

        # Enter
        await s.process_event(_price_event(ticker_a, Decimal('0.60')), trader)
        await s.process_event(_price_event(ticker_b, Decimal('0.30')), trader)
        assert s._position_state == 'short_a_long_b'

        # Exit: A=0.30 <= B=0.50
        await s.process_event(_price_event(ticker_a, Decimal('0.30')), trader)
        await s.process_event(_price_event(ticker_b, Decimal('0.50')), trader)
        assert s._position_state == 'flat'


# ===================================================================
# ConditionalArbStrategy Tests
# ===================================================================


class TestConditionalArbDefaults:
    def test_defaults(self):
        s = ConditionalArbStrategy()
        assert s.cond_lower == 0.0
        assert s.cond_upper == 1.0
        assert s._position_state == 'flat'

    def test_name(self):
        assert ConditionalArbStrategy.name == 'conditional_arb'


class TestConditionalArbBounds:
    def test_compute_bounds(self):
        s = ConditionalArbStrategy(cond_lower=0.4, cond_upper=0.9)
        # p(B) = 0.6: lower = 0.4 * 0.6 = 0.24, upper = 0.9 * 0.6 + 0.4 = 0.94
        lower, upper = s._compute_bounds(0.6)
        assert abs(lower - 0.24) < 1e-9
        assert abs(upper - 0.94) < 1e-9

    def test_bounds_at_b_zero(self):
        s = ConditionalArbStrategy(cond_lower=0.5, cond_upper=0.8)
        lower, upper = s._compute_bounds(0.0)
        assert lower == 0.0
        assert upper == 1.0

    def test_bounds_at_b_one(self):
        s = ConditionalArbStrategy(cond_lower=0.3, cond_upper=0.7)
        lower, upper = s._compute_bounds(1.0)
        assert abs(lower - 0.3) < 1e-9
        assert abs(upper - 0.7) < 1e-9


class TestConditionalArbViolation:

    async def test_a_within_band_holds(self):
        ticker_a = _make_ticker('A', market_id='mkt_a')
        ticker_b = _make_ticker('B', market_id='mkt_b', no_token_id='B_NO')
        s = ConditionalArbStrategy(
            market_id_a='mkt_a', market_id_b='mkt_b',
            cond_lower=0.4, cond_upper=0.9,
            min_edge=0.01, fee_rate=0.0,
        )
        trader = _make_trader([ticker_a, ticker_b])

        # B=0.6 => band=[0.24, 0.94], A=0.50 is inside
        await s.process_event(_price_event(ticker_b, Decimal('0.60')), trader)
        await s.process_event(_price_event(ticker_a, Decimal('0.50')), trader)
        assert s._position_state == 'flat'

    async def test_a_too_high_enters_short_a(self):
        ticker_a = _make_ticker('A', market_id='mkt_a', no_token_id='A_NO')
        ticker_a_no = ticker_a.get_no_ticker()
        ticker_b = _make_ticker('B', market_id='mkt_b', no_token_id='B_NO')
        s = ConditionalArbStrategy(
            market_id_a='mkt_a', market_id_b='mkt_b',
            cond_lower=0.4, cond_upper=0.6,
            min_edge=0.01, fee_rate=0.0, cooldown_seconds=0.0,
        )
        trader = _make_trader()
        _seed_order_book(trader, ticker_a, Decimal('0.55'), Decimal('0.65'))
        _seed_order_book(trader, ticker_a_no, Decimal('0.04'), Decimal('0.06'))
        _seed_order_book(trader, ticker_b, Decimal('0.45'), Decimal('0.55'))
        _seed_order_book(
            trader, ticker_b.get_no_ticker(), Decimal('0.45'), Decimal('0.55')
        )

        # B=0.5 => band=[0.20, 0.80], A=0.95 > upper+edge
        await s.process_event(_price_event(ticker_b, Decimal('0.50')), trader)
        await s.process_event(_price_event(ticker_a, Decimal('0.95')), trader)
        assert s._position_state == 'short_both'

    async def test_a_too_low_enters_long_a(self):
        ticker_a = _make_ticker('A', market_id='mkt_a')
        ticker_b = _make_ticker('B', market_id='mkt_b', no_token_id='B_NO')
        ticker_b_no = ticker_b.get_no_ticker()
        s = ConditionalArbStrategy(
            market_id_a='mkt_a', market_id_b='mkt_b',
            cond_lower=0.5, cond_upper=0.9,
            min_edge=0.01, fee_rate=0.0, cooldown_seconds=0.0,
        )
        trader = _make_trader()
        _seed_order_book(trader, ticker_a, Decimal('0.04'), Decimal('0.06'))
        _seed_order_book(trader, ticker_b, Decimal('0.55'), Decimal('0.65'))
        _seed_order_book(trader, ticker_b_no, Decimal('0.35'), Decimal('0.45'))

        # B=0.6 => lower=0.30, A=0.05 < lower-edge
        await s.process_event(_price_event(ticker_b, Decimal('0.60')), trader)
        await s.process_event(_price_event(ticker_a, Decimal('0.05')), trader)
        assert s._position_state == 'long_a_short_b'

    async def test_exit_when_a_returns_to_band(self):
        ticker_a = _make_ticker('A', market_id='mkt_a', no_token_id='A_NO')
        ticker_a_no = ticker_a.get_no_ticker()
        ticker_b = _make_ticker('B', market_id='mkt_b', no_token_id='B_NO')
        s = ConditionalArbStrategy(
            market_id_a='mkt_a', market_id_b='mkt_b',
            cond_lower=0.4, cond_upper=0.6,
            min_edge=0.01, fee_rate=0.0, cooldown_seconds=0.0,
            trade_size=5.0,
        )
        trader = _make_trader()
        _seed_order_book(trader, ticker_a, Decimal('0.55'), Decimal('0.65'))
        _seed_order_book(trader, ticker_a_no, Decimal('0.04'), Decimal('0.06'))
        _seed_order_book(trader, ticker_b, Decimal('0.45'), Decimal('0.55'))
        _seed_order_book(
            trader, ticker_b.get_no_ticker(), Decimal('0.45'), Decimal('0.55')
        )

        # Enter: B=0.5 band=[0.20, 0.80], A=0.95 > upper
        await s.process_event(_price_event(ticker_b, Decimal('0.50')), trader)
        await s.process_event(_price_event(ticker_a, Decimal('0.95')), trader)
        assert s._position_state == 'short_both'

        # Exit: A=0.50 in [0.20, 0.80]
        await s.process_event(_price_event(ticker_a, Decimal('0.50')), trader)
        assert s._position_state == 'flat'


# ===================================================================
# EventSumArbStrategy Tests
# ===================================================================


class TestEventSumArbDefaults:
    def test_defaults(self):
        s = EventSumArbStrategy()
        assert s.event_id == ''
        assert s.min_edge == Decimal('0.02')
        assert s.min_markets == 2

    def test_name(self):
        assert EventSumArbStrategy.name == 'event_sum_arb'


class TestEventSumArbRegistration:
    async def test_registers_tickers_by_event_id(self):
        t1 = PolyMarketTicker(
            symbol='T1', name='Outcome 1', token_id='t1',
            market_id='m1', event_id='evt_abc',
        )
        t2 = PolyMarketTicker(
            symbol='T2', name='Outcome 2', token_id='t2',
            market_id='m2', event_id='evt_abc',
        )
        s = EventSumArbStrategy(event_id='evt_abc')
        trader = _make_trader([t1, t2])

        await s.process_event(_price_event(t1, Decimal('0.30')), trader)
        await s.process_event(_price_event(t2, Decimal('0.40')), trader)

        assert 'm1' in s._tickers
        assert 'm2' in s._tickers

    async def test_ignores_wrong_event_id(self):
        t = PolyMarketTicker(
            symbol='T', name='X', token_id='t',
            market_id='m', event_id='evt_other',
        )
        s = EventSumArbStrategy(event_id='evt_abc')
        trader = _make_trader([t])
        await s.process_event(_price_event(t, Decimal('0.50')), trader)
        assert len(s._tickers) == 0


def _event_sum_strategy(tickers, **kwargs):
    return EventSumArbStrategy(
        event_id='evt1',
        outcome_market_ids=[ticker.market_id for ticker in tickers],
        yes_token_ids={ticker.market_id: ticker.token_id for ticker in tickers},
        outcomes_are_exhaustive=True,
        outcomes_are_mutually_exclusive=True,
        cooldown_seconds=0.0,
        trade_size=5.0,
        **kwargs,
    )


def _event_outcomes(count=2):
    return [
        _make_ticker(f'T{i}', market_id=f'm{i}', event_id='evt1', no_token_id=f'N{i}')
        for i in range(1, count + 1)
    ]


class TestEventSumArbDetection:
    async def test_default_never_infers_complete_partition(self):
        tickers = _event_outcomes()
        strategy = EventSumArbStrategy(event_id='evt1', cooldown_seconds=0)
        trader = _make_trader()
        for ticker in tickers:
            _seed_order_book(trader, ticker, Decimal('0.19'), Decimal('0.20'))
            await strategy.process_event(_price_event(ticker, Decimal('0.20')), trader)
        assert trader.orders == []
        assert strategy.get_decisions() == []

    async def test_incomplete_two_of_three_does_not_trade(self):
        tickers = _event_outcomes(3)
        strategy = _event_sum_strategy(tickers, fee_rate=0)
        trader = _make_trader()
        for ticker, price in zip(tickers[:2], [Decimal('0.20'), Decimal('0.30')]):
            _seed_order_book(trader, ticker, price - Decimal('0.01'), price)
            await strategy.process_event(_price_event(ticker, price), trader)
        assert trader.orders == []
        # The final outcome makes the full basket fairly priced, not an arb.
        _seed_order_book(trader, tickers[2], Decimal('0.49'), Decimal('0.50'))
        await strategy.process_event(_price_event(tickers[2], Decimal('0.50')), trader)
        assert trader.orders == []
        assert strategy.get_decisions()[-1].action == 'HOLD'

    async def test_complete_underpriced_partition_buys_all_yes_at_asks(self):
        tickers = _event_outcomes(3)
        strategy = _event_sum_strategy(tickers, fee_rate=0)
        trader = _make_trader()
        for ticker, ask in zip(
            tickers, [Decimal('0.20'), Decimal('0.30'), Decimal('0.40')]
        ):
            _seed_order_book(trader, ticker, ask - Decimal('0.10'), ask)
            await strategy.process_event(
                _price_event(ticker, ask - Decimal('0.05')), trader
            )
        assert len(trader.orders) == 3
        assert [order.limit_price for order in trader.orders] == [
            Decimal('0.20'),
            Decimal('0.30'),
            Decimal('0.40'),
        ]
        assert all(order.filled_quantity == Decimal('5') for order in trader.orders)
        assert strategy.get_decisions()[-1].action == 'BUY_YES'

    async def test_actual_no_asks_determine_buy_no_prices(self):
        tickers = _event_outcomes()
        strategy = _event_sum_strategy(tickers, fee_rate=0)
        trader = _make_trader()
        for ticker in tickers:
            _seed_order_book(trader, ticker, Decimal('0.55'), Decimal('0.75'))
            _seed_order_book(
                trader, ticker.get_no_ticker(), Decimal('0.25'), Decimal('0.44')
            )
            await strategy.process_event(_price_event(ticker, Decimal('0.70')), trader)
        assert len(trader.orders) == 2
        assert all(order.ticker.token_id.startswith('N') for order in trader.orders)
        assert all(order.limit_price == Decimal('0.44') for order in trader.orders)
        assert strategy.get_decisions()[-1].action == 'BUY_NO'

    async def test_wide_yes_spread_does_not_create_phantom_no_edge(self):
        tickers = _event_outcomes()
        strategy = _event_sum_strategy(tickers, fee_rate=0)
        trader = _make_trader()
        for ticker in tickers:
            # 1 - ask would fabricate a .20 NO buy; 1 - bid is .60.
            _seed_order_book(trader, ticker, Decimal('0.40'), Decimal('0.80'))
            await strategy.process_event(_price_event(ticker, Decimal('0.80')), trader)
        assert trader.orders == []
        assert strategy.get_decisions()[-1].signal_values[
            'edge_buy_no'
        ] == pytest.approx(-0.20)

    async def test_known_no_book_overrides_attractive_yes_complement(self):
        tickers = _event_outcomes()
        strategy = _event_sum_strategy(tickers, fee_rate=0)
        trader = _make_trader()
        for ticker in tickers:
            _seed_order_book(trader, ticker, Decimal('0.70'), Decimal('0.80'))
            _seed_order_book(
                trader, ticker.get_no_ticker(), Decimal('0.20'), Decimal('0.60')
            )
            await strategy.process_event(_price_event(ticker, Decimal('0.75')), trader)
        assert trader.orders == []
        assert strategy.get_decisions()[-1].signal_values[
            'edge_buy_no'
        ] == pytest.approx(-0.20)

    async def test_empty_no_book_blocks_complement_fallback(self):
        tickers = _event_outcomes()
        strategy = _event_sum_strategy(tickers, fee_rate=0)
        trader = _make_trader()
        for ticker in tickers:
            _seed_order_book(trader, ticker, Decimal('0.70'), Decimal('0.80'))
            trader.market_data.update_order_book(ticker.get_no_ticker(), OrderBook())
            await strategy.process_event(_price_event(ticker, Decimal('0.75')), trader)
        assert trader.orders == []
        assert strategy.get_decisions()[-1].signal_values.get('edge_buy_no') is None

    async def test_missing_no_token_prevents_partial_no_basket(self):
        tickers = _event_outcomes()
        tickers[1] = _make_ticker('T2', market_id='m2', event_id='evt1')
        strategy = _event_sum_strategy(tickers, fee_rate=0)
        trader = _make_trader()
        for ticker in tickers:
            _seed_order_book(trader, ticker, Decimal('0.70'), Decimal('0.80'))
            await strategy.process_event(_price_event(ticker, Decimal('0.75')), trader)
        assert trader.orders == []
        assert strategy.get_decisions()[-1].signal_values.get('edge_buy_no') is None

    async def test_no_first_update_cannot_register_as_yes(self):
        tickers = _event_outcomes()
        strategy = _event_sum_strategy(tickers, fee_rate=0)
        trader = _make_trader()
        for ticker in tickers:
            no_ticker = ticker.get_no_ticker()
            _seed_order_book(trader, no_ticker, Decimal('0.10'), Decimal('0.20'))
            await strategy.process_event(
                _price_event(no_ticker, Decimal('0.20')), trader
            )
        assert strategy._tickers == {}
        assert trader.orders == []

    async def test_missing_book_cannot_be_replaced_by_last_trade_price(self):
        tickers = _event_outcomes()
        strategy = _event_sum_strategy(tickers, fee_rate=0)
        trader = _make_trader()
        for ticker in tickers:
            await strategy.process_event(_price_event(ticker, Decimal('0.20')), trader)
        assert trader.orders == []

    async def test_unexpected_outcome_invalidates_declared_partition(self):
        tickers = _event_outcomes(3)
        strategy = _event_sum_strategy(tickers[:2], fee_rate=0)
        trader = _make_trader()
        await strategy.process_event(_price_event(tickers[2], Decimal('0.20')), trader)
        for ticker in tickers[:2]:
            _seed_order_book(trader, ticker, Decimal('0.10'), Decimal('0.20'))
            await strategy.process_event(_price_event(ticker, Decimal('0.20')), trader)
        assert trader.orders == []

    async def test_published_default_fees_remove_thin_gross_edge(self):
        tickers = _event_outcomes()
        strategy = _event_sum_strategy(tickers, min_edge=0.01)
        trader = _make_trader()
        for ticker in tickers:
            _seed_order_book(trader, ticker, Decimal('0.48'), Decimal('0.49'))
            await strategy.process_event(_price_event(ticker, Decimal('0.48')), trader)
        assert trader.orders == []
        assert strategy.get_decisions()[-1].signal_values['edge_buy_yes'] < 0

    async def test_explicit_market_fee_schedule_is_used(self):
        tickers = _event_outcomes()
        strategy = _event_sum_strategy(
            tickers, min_edge=0.01, market_fee_rates={'m1': 0, 'm2': 0}
        )
        trader = _make_trader()
        for ticker in tickers:
            _seed_order_book(trader, ticker, Decimal('0.48'), Decimal('0.49'))
            await strategy.process_event(_price_event(ticker, Decimal('0.48')), trader)
        assert len(trader.orders) == 2

    async def test_shallow_leg_prevents_entire_basket(self):
        tickers = _event_outcomes()
        strategy = _event_sum_strategy(tickers, fee_rate=0)
        trader = _make_trader()
        for ticker in tickers:
            _seed_order_book(
                trader, ticker, Decimal('0.10'), Decimal('0.20'), size=Decimal('1')
            )
            await strategy.process_event(_price_event(ticker, Decimal('0.20')), trader)
        assert trader.orders == []

    @pytest.mark.parametrize('invalid_fill', [
        'pending_full', 'overfill', 'remaining', 'wrong_side', 'wrong_ticker',
    ])
    async def test_invalid_first_leg_result_stops_basket(self, invalid_fill):
        tickers = _event_outcomes()
        strategy = _event_sum_strategy(tickers, fee_rate=0)
        trader = _make_trader()
        first_order = Order(
            status=OrderStatus.PLACED if invalid_fill == 'pending_full' else OrderStatus.FILLED,
            ticker=tickers[1] if invalid_fill == 'wrong_ticker' else tickers[0],
            side=TradeSide.SELL if invalid_fill == 'wrong_side' else TradeSide.BUY,
            limit_price=Decimal('0.20'),
            filled_quantity=Decimal('6') if invalid_fill == 'overfill' else Decimal('5'),
            average_price=Decimal('0.20'),
            trades=[],
            remaining=Decimal('1') if invalid_fill == 'remaining' else Decimal('0'),
            commission=Decimal('0'),
        )
        place_order = AsyncMock(return_value=PlaceOrderResult(order=first_order))
        trader.place_order = place_order
        for ticker in tickers:
            _seed_order_book(trader, ticker, Decimal('0.10'), Decimal('0.20'))
            await strategy.process_event(_price_event(ticker, Decimal('0.20')), trader)
        place_order.assert_awaited_once()
        assert strategy.get_decisions()[-1].signal_values['failed_legs'] == 1

    @pytest.mark.parametrize('cooldown', [-1, float('nan'), float('inf')])
    def test_invalid_cooldown_is_rejected(self, cooldown):
        with pytest.raises(ValueError, match='cooldown_seconds'):
            EventSumArbStrategy(cooldown_seconds=cooldown)

    @pytest.mark.parametrize('config', [
        {'event_id': ' evt1 '},
        {'outcome_market_ids': ['m1 ', 'm2']},
        {'outcome_market_ids': ['m1', 'm2'], 'yes_token_ids': {'m1': 'T1 ', 'm2': 'T2'}},
    ])
    def test_surrounding_identifier_whitespace_is_rejected(self, config):
        with pytest.raises(ValueError):
            EventSumArbStrategy(**config)

    @pytest.mark.parametrize('ids', [['m1', 'm1'], ['m1', ''], 'm1'])
    def test_invalid_outcome_list_is_rejected(self, ids):
        with pytest.raises(ValueError):
            EventSumArbStrategy(outcome_market_ids=ids)

    def test_duplicate_yes_token_is_rejected(self):
        with pytest.raises(ValueError, match='distinct YES token'):
            EventSumArbStrategy(
                outcome_market_ids=['m1', 'm2'],
                yes_token_ids={'m1': 'same', 'm2': 'same'},
            )

    @pytest.mark.parametrize('exhaustive,exclusive', [(False, True), (True, False)])
    async def test_both_partition_assertions_are_required(self, exhaustive, exclusive):
        tickers = _event_outcomes()
        strategy = EventSumArbStrategy(
            event_id='evt1',
            outcome_market_ids=['m1', 'm2'],
            yes_token_ids={'m1': 'T1', 'm2': 'T2'},
            outcomes_are_exhaustive=exhaustive,
            outcomes_are_mutually_exclusive=exclusive,
        )
        trader = _make_trader()
        for ticker in tickers:
            _seed_order_book(trader, ticker, Decimal('0.10'), Decimal('0.20'))
            await strategy.process_event(_price_event(ticker, Decimal('0.20')), trader)
        assert trader.orders == []


# ===================================================================
# StructuralArbStrategy Tests
# ===================================================================


class TestStructuralArbDefaults:
    def test_defaults(self):
        s = StructuralArbStrategy()
        assert s.slope == 1.0
        assert s.intercept == 0.0
        assert s._position_state == 'flat'

    def test_name(self):
        assert StructuralArbStrategy.name == 'structural_arb'


class TestStructuralArbExpected:
    def test_expected_a(self):
        s = StructuralArbStrategy(slope=2.0, intercept=0.1)
        assert abs(s._expected_a(0.3) - 0.7) < 1e-9  # 2*0.3 + 0.1

    def test_identity(self):
        s = StructuralArbStrategy(slope=1.0, intercept=0.0)
        assert s._expected_a(0.5) == 0.5


class TestStructuralArbViolation:

    async def test_no_violation_when_within_band(self):
        ticker_a = _make_ticker('A', market_id='mkt_a')
        ticker_b = _make_ticker('B', market_id='mkt_b')
        s = StructuralArbStrategy(
            market_id_a='mkt_a', market_id_b='mkt_b',
            slope=1.0, intercept=0.0, min_edge=0.05,
        )
        trader = _make_trader([ticker_a, ticker_b])

        # Expected A = 1.0 * 0.50 + 0.0 = 0.50, actual A = 0.52
        # residual = 0.02 < 0.05 => hold
        await s.process_event(_price_event(ticker_b, Decimal('0.50')), trader)
        await s.process_event(_price_event(ticker_a, Decimal('0.52')), trader)
        assert s._position_state == 'flat'

    async def test_a_overpriced_enters_short_a(self):
        ticker_a = _make_ticker('A', market_id='mkt_a', no_token_id='A_NO')
        ticker_a_no = ticker_a.get_no_ticker()
        ticker_b = _make_ticker('B', market_id='mkt_b')
        s = StructuralArbStrategy(
            market_id_a='mkt_a', market_id_b='mkt_b',
            slope=1.0, intercept=0.0, min_edge=0.02,
            cooldown_seconds=0.0,
        )
        trader = _make_trader()
        _seed_order_book(trader, ticker_a, Decimal('0.65'), Decimal('0.75'))
        _seed_order_book(trader, ticker_a_no, Decimal('0.25'), Decimal('0.35'))
        _seed_order_book(trader, ticker_b, Decimal('0.45'), Decimal('0.55'))

        # Expected A = 0.50, actual A = 0.70, residual = 0.20 > 0.02
        await s.process_event(_price_event(ticker_b, Decimal('0.50')), trader)
        await s.process_event(_price_event(ticker_a, Decimal('0.70')), trader)
        assert s._position_state == 'short_a_long_b'

    async def test_a_underpriced_enters_long_a(self):
        ticker_a = _make_ticker('A', market_id='mkt_a')
        ticker_b = _make_ticker('B', market_id='mkt_b', no_token_id='B_NO')
        ticker_b_no = ticker_b.get_no_ticker()
        s = StructuralArbStrategy(
            market_id_a='mkt_a', market_id_b='mkt_b',
            slope=1.0, intercept=0.0, min_edge=0.02,
            cooldown_seconds=0.0,
        )
        trader = _make_trader()
        _seed_order_book(trader, ticker_a, Decimal('0.25'), Decimal('0.35'))
        _seed_order_book(trader, ticker_b, Decimal('0.55'), Decimal('0.65'))
        _seed_order_book(trader, ticker_b_no, Decimal('0.35'), Decimal('0.45'))

        # Expected A = 0.60, actual A = 0.30, residual = -0.30 < -0.02
        await s.process_event(_price_event(ticker_b, Decimal('0.60')), trader)
        await s.process_event(_price_event(ticker_a, Decimal('0.30')), trader)
        assert s._position_state == 'long_a_short_b'

    async def test_exit_on_convergence(self):
        ticker_a = _make_ticker('A', market_id='mkt_a', no_token_id='A_NO')
        ticker_a_no = ticker_a.get_no_ticker()
        ticker_b = _make_ticker('B', market_id='mkt_b')
        s = StructuralArbStrategy(
            market_id_a='mkt_a', market_id_b='mkt_b',
            slope=1.0, intercept=0.0, min_edge=0.05,
            exit_fraction=0.5, cooldown_seconds=0.0,
            trade_size=5.0,
        )
        trader = _make_trader()
        _seed_order_book(trader, ticker_a, Decimal('0.55'), Decimal('0.75'))
        _seed_order_book(trader, ticker_a_no, Decimal('0.25'), Decimal('0.30'))
        _seed_order_book(trader, ticker_b, Decimal('0.45'), Decimal('0.55'))

        # Enter: residual = 0.70 - 0.50 = 0.20 > 0.05
        await s.process_event(_price_event(ticker_b, Decimal('0.50')), trader)
        await s.process_event(_price_event(ticker_a, Decimal('0.70')), trader)
        assert s._position_state == 'short_a_long_b'

        # Exit: residual = 0.51 - 0.50 = 0.01 < 0.05*0.5 = 0.025
        await s.process_event(_price_event(ticker_a, Decimal('0.51')), trader)
        assert s._position_state == 'flat'


# ===================================================================
# Integration: process_event with PaperTrader
# ===================================================================


class TestConstraintArbIntegration:
    """Integration tests that verify orders are placed via PaperTrader."""

    async def test_exclusivity_arb_places_orders(self):
        ticker_a = _make_ticker('A', market_id='mkt_a', no_token_id='A_NO')
        ticker_a_no = ticker_a.get_no_ticker()
        ticker_b = _make_ticker('B', market_id='mkt_b', no_token_id='B_NO')
        ticker_b_no = ticker_b.get_no_ticker()
        s = ExclusivityArbStrategy(
            market_id_a='mkt_a', market_id_b='mkt_b',
            min_edge=0.01, fee_rate=0.0, cooldown_seconds=0.0,
            trade_size=5.0,
        )
        trader = _make_trader()
        for t in [ticker_a, ticker_b]:
            _seed_order_book(trader, t, Decimal('0.55'), Decimal('0.65'))
        for t in [ticker_a_no, ticker_b_no]:
            _seed_order_book(trader, t, Decimal('0.35'), Decimal('0.45'))

        await s.process_event(_price_event(ticker_a, Decimal('0.60')), trader)
        await s.process_event(_price_event(ticker_b, Decimal('0.55')), trader)

        # Should have placed orders
        assert len(trader.orders) >= 1
        decisions = s.get_decisions()
        enter_d = [d for d in decisions if d.action == 'ENTER_ARB']
        assert len(enter_d) == 1
        assert enter_d[0].executed is True

    async def test_implication_arb_places_orders(self):
        ticker_a = _make_ticker('A', market_id='mkt_a', no_token_id='A_NO')
        ticker_a_no = ticker_a.get_no_ticker()
        ticker_b = _make_ticker('B', market_id='mkt_b')
        s = ImplicationArbStrategy(
            market_id_a='mkt_a', market_id_b='mkt_b',
            min_edge=0.01, fee_rate=0.0, cooldown_seconds=0.0,
            trade_size=5.0,
        )
        trader = _make_trader()
        _seed_order_book(trader, ticker_a, Decimal('0.55'), Decimal('0.65'))
        _seed_order_book(trader, ticker_a_no, Decimal('0.35'), Decimal('0.45'))
        _seed_order_book(trader, ticker_b, Decimal('0.25'), Decimal('0.35'))

        await s.process_event(_price_event(ticker_a, Decimal('0.60')), trader)
        await s.process_event(_price_event(ticker_b, Decimal('0.30')), trader)

        assert len(trader.orders) >= 1
