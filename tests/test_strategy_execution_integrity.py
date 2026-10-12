"""Regression cases for basket fills, instance ownership, quotes and assumptions."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
from test_constraint_arb_strategies import (
    _make_ticker,
    _make_trader,
    _seed_order_book,
)

from oracle3.events.events import PriceChangeEvent
from oracle3.position.position_manager import Position
from oracle3.strategy.contrib.conditional_arb_strategy import ConditionalArbStrategy
from oracle3.strategy.contrib.exclusivity_arb_strategy import ExclusivityArbStrategy
from oracle3.strategy.contrib.implication_arb_strategy import ImplicationArbStrategy
from oracle3.strategy.contrib.structural_arb_strategy import StructuralArbStrategy
from oracle3.ticker.ticker import KalshiTicker, PolyMarketTicker
from oracle3.trader.types import (
    Order,
    OrderFailureReason,
    OrderStatus,
    PlaceOrderResult,
    TradeSide,
)

STRATEGIES = [
    ImplicationArbStrategy,
    ExclusivityArbStrategy,
    ConditionalArbStrategy,
    StructuralArbStrategy,
]


def _setup(cls, **kwargs):
    a = _make_ticker('A', market_id='a', no_token_id='A_NO')
    b = _make_ticker('B', market_id='b', no_token_id='B_NO')
    trader = _make_trader()
    for ticker in (a, b, a.get_no_ticker(), b.get_no_ticker()):
        _seed_order_book(trader, ticker, Decimal('0.01'), Decimal('0.02'))
    params = {
        'market_id_a': 'a',
        'market_id_b': 'b',
        'trade_size': 10,
        'cooldown_seconds': 0,
        'fee_rate': 0,
    }
    if cls is ConditionalArbStrategy:
        params.update(cond_lower=0.5, cond_upper=0.5)
    params.update(kwargs)
    strategy = cls(**params)
    strategy._price_a, strategy._price_b = Decimal('0.95'), Decimal('0.1')
    return strategy, trader, a, b


async def _enter(strategy, trader):
    if isinstance(strategy, (ImplicationArbStrategy, ExclusivityArbStrategy)):
        await strategy._enter(trader, Decimal('0.2'))
    else:
        await strategy._enter_short_a(trader, 0.95, 0.0, 0.75)


def _result(request, *, filled=None, status=OrderStatus.FILLED):
    quantity = request['quantity']
    filled = quantity if filled is None else Decimal(str(filled))
    return PlaceOrderResult(
        order=Order(
            status=status,
            side=request['side'],
            ticker=request['ticker'],
            limit_price=request['limit_price'],
            filled_quantity=filled,
            average_price=request['limit_price'] if filled else Decimal('0'),
            remaining=quantity - filled,
            commission=Decimal('0'),
            trades=[],
        )
    )


@pytest.mark.parametrize('cls', STRATEGIES)
async def test_pending_zero_fill_blocks_next_leg_and_reentry(cls):
    strategy, trader, a, _ = _setup(cls)
    trader.place_order = AsyncMock(
        side_effect=lambda **kw: _result(kw, filled=0, status=OrderStatus.PLACED)
    )
    await _enter(strategy, trader)
    assert trader.place_order.await_count == 1
    assert strategy._owned_positions == {}
    assert strategy.requires_reconciliation
    assert strategy._position_state == 'reconciliation_required'
    assert not strategy.get_decisions()[-1].executed
    await strategy.process_event(
        PriceChangeEvent(ticker=a, price=Decimal('0.99')), trader
    )
    await strategy._close_owned(trader)
    assert trader.place_order.await_count == 1


@pytest.mark.parametrize('cls', STRATEGIES)
async def test_first_rejection_stops_basket_without_inventing_exposure(cls):
    strategy, trader, _, _ = _setup(cls)
    trader.place_order = AsyncMock(
        return_value=PlaceOrderResult(failure_reason=OrderFailureReason.INVALID_ORDER)
    )
    await _enter(strategy, trader)
    assert trader.place_order.await_count == 1
    assert strategy._position_state == 'flat'
    assert not strategy.requires_reconciliation
    assert not strategy._owned_positions
    assert not strategy.get_decisions()[-1].executed


@pytest.mark.parametrize('cls', STRATEGIES)
@pytest.mark.parametrize('terminal', [False, True])
async def test_partial_first_leg_waits_for_terminal_state_before_recovery(
    cls, terminal
):
    strategy, trader, _, _ = _setup(cls)
    calls = []

    async def place(**kw):
        calls.append(kw)
        if len(calls) == 1:
            return _result(
                kw,
                filled=3,
                status=OrderStatus.CANCELLED
                if terminal
                else OrderStatus.PARTIALLY_FILLED,
            )
        return _result(kw)

    trader.place_order = AsyncMock(side_effect=place)
    await _enter(strategy, trader)
    if not terminal:
        assert len(calls) == 1
        assert strategy._owned_positions == {calls[0]['ticker']: Decimal('3')}
        assert strategy.requires_reconciliation
        assert strategy._position_state == 'reconciliation_required'
        return
    assert len(calls) == 2
    assert calls[1]['side'] == TradeSide.SELL
    assert calls[1]['quantity'] == Decimal('3')
    assert calls[1]['ticker'] == calls[0]['ticker']
    assert not strategy._owned_positions
    assert strategy.requires_reconciliation is (not terminal)
    assert strategy._position_state == (
        'flat' if terminal else 'reconciliation_required'
    )


@pytest.mark.parametrize('cls', STRATEGIES)
async def test_failed_second_leg_unwinds_confirmed_first(cls):
    strategy, trader, _, _ = _setup(cls)
    calls = []

    async def place(**kw):
        calls.append(kw)
        if len(calls) == 2:
            return PlaceOrderResult(failure_reason=OrderFailureReason.INSUFFICIENT_CASH)
        return _result(kw)

    trader.place_order = AsyncMock(side_effect=place)
    await _enter(strategy, trader)
    assert len(calls) == 3
    assert calls[2]['side'] == TradeSide.SELL
    assert calls[2]['ticker'] == calls[0]['ticker']
    assert calls[2]['quantity'] == calls[0]['quantity']
    assert strategy._position_state == 'flat'
    assert strategy.last_execution_result.hedged


@pytest.mark.parametrize('cls', STRATEGIES)
async def test_failed_recovery_preserves_known_owned_residual(cls):
    strategy, trader, _, _ = _setup(cls)
    calls = []

    async def place(**kw):
        calls.append(kw)
        if len(calls) > 1:
            return PlaceOrderResult(failure_reason=OrderFailureReason.INSUFFICIENT_CASH)
        return _result(kw)

    trader.place_order = AsyncMock(side_effect=place)
    await _enter(strategy, trader)
    assert strategy._owned_positions == {calls[0]['ticker']: calls[0]['quantity']}
    assert strategy._position_state == 'recovery_required'
    assert not strategy.requires_reconciliation
    assert strategy.execution_state()['owned_positions'] == {
        calls[0]['ticker'].symbol: str(calls[0]['quantity'])
    }


@pytest.mark.parametrize('cls', STRATEGIES)
async def test_exit_sells_only_instance_fills_and_preserves_same_market_prior_holdings(
    cls,
):
    strategy, trader, a, _ = _setup(cls)
    owned_ticker = a.get_no_ticker()
    unrelated = _make_ticker('OTHER', market_id='other')
    for ticker, quantity in [(owned_ticker, Decimal('7')), (unrelated, Decimal('11'))]:
        trader.position_manager.update_position(
            Position(ticker, quantity, Decimal('0.02'), Decimal('0'))
        )
        _seed_order_book(trader, ticker, Decimal('0.01'), Decimal('0.02'))
    await _enter(strategy, trader)
    assert strategy._owned_positions[owned_ticker] == 10
    await strategy._close_owned(trader)
    assert trader.position_manager.positions[owned_ticker.symbol].quantity == 7
    assert trader.position_manager.positions[unrelated.symbol].quantity == 11
    assert strategy._position_state == 'flat'
    assert not strategy._owned_positions


@pytest.mark.parametrize('cls', STRATEGIES)
async def test_failed_exit_never_marks_owned_positions_flat(cls):
    strategy, trader, _, _ = _setup(cls)
    await _enter(strategy, trader)
    initial = strategy._owned_positions.copy()
    trader.place_order = AsyncMock(
        return_value=PlaceOrderResult(failure_reason=OrderFailureReason.INVALID_ORDER)
    )
    await strategy._close_owned(trader)
    assert strategy._owned_positions == initial
    assert strategy._position_state == 'recovery_required'
    assert trader.place_order.await_count == 1
    assert not strategy.get_decisions()[-1].executed


@pytest.mark.parametrize('cls', STRATEGIES)
async def test_partial_exit_keeps_unsold_quantity_and_stops_next_leg(cls):
    strategy, trader, _, _ = _setup(cls)
    await _enter(strategy, trader)
    first, second = list(strategy._owned_positions)
    remaining_second = strategy._owned_positions[second]
    trader.place_order = AsyncMock(
        side_effect=lambda **kw: _result(kw, filled=3, status=OrderStatus.CANCELLED)
    )
    await strategy._close_owned(trader)
    assert strategy._owned_positions[first] == 7
    assert strategy._owned_positions[second] == remaining_second
    assert strategy._position_state == 'recovery_required'
    assert trader.place_order.await_count == 1


@pytest.mark.parametrize('cls', STRATEGIES)
async def test_all_legs_preflight_before_first_order(cls):
    strategy, trader, _, b = _setup(cls)
    del trader.market_data.order_books[b]
    trader.place_order = AsyncMock()
    await _enter(strategy, trader)
    trader.place_order.assert_not_awaited()
    assert strategy._position_state == 'flat'


@pytest.mark.parametrize('cls', STRATEGIES)
async def test_explicit_high_fees_block_entry_for_all_strategies(cls):
    strategy, trader, _, _ = _setup(cls, fee_rate=1)
    trader.place_order = AsyncMock()
    await _enter(strategy, trader)
    trader.place_order.assert_not_awaited()
    assert strategy.get_decisions()[-1].signal_values['net_edge'] < 0


async def test_shared_kalshi_fees_reject_positive_gross_edge():
    strategy = ImplicationArbStrategy(market_id_a='a', market_id_b='b', min_edge=0.001)
    a, b = KalshiTicker('A', market_ticker='a'), KalshiTicker('B', market_ticker='b')
    trader = _make_trader()
    for ticker, ask in [(a, '0.51'), (a.get_no_ticker(), '0.499'), (b, '0.49')]:
        _seed_order_book(trader, ticker, Decimal(ask) - Decimal('0.01'), Decimal(ask))
    trader.place_order = AsyncMock()
    await strategy._enter(trader, Decimal('0.011'))
    trader.place_order.assert_not_awaited()
    signals = strategy.get_decisions()[-1].signal_values
    assert signals['gross_edge'] > 0
    assert signals['net_edge'] < 0


async def test_min_edge_applies_after_fees_not_to_raw_signal():
    strategy, trader, a, b = _setup(
        ImplicationArbStrategy, min_edge=0.05, fee_rate=0.02
    )
    for ticker in [a.get_no_ticker(), b]:
        _seed_order_book(trader, ticker, Decimal('0.45'), Decimal('0.46'))
    trader.place_order = AsyncMock()
    await strategy._enter(trader, Decimal('0.30'))
    trader.place_order.assert_not_awaited()
    assert strategy.get_decisions()[-1].signal_values['gross_edge'] == pytest.approx(
        0.08
    )
    assert strategy.get_decisions()[-1].signal_values['net_edge'] == pytest.approx(0.04)


@pytest.mark.parametrize('cls', STRATEGIES)
async def test_cancellation_leaves_reconciliation_barrier(cls):
    import asyncio

    strategy, trader, _, _ = _setup(cls)
    trader.place_order = AsyncMock(side_effect=asyncio.CancelledError)
    with pytest.raises(asyncio.CancelledError):
        await _enter(strategy, trader)
    assert strategy.requires_reconciliation
    assert strategy._position_state == 'reconciliation_required'
    assert not strategy._basket_busy


async def test_opaque_polymarket_outcomes_require_explicit_yes_mapping():
    a = PolyMarketTicker(
        '123', name='Question A', token_id='123', market_id='a', no_token_id='456'
    )
    b = PolyMarketTicker(
        '789', name='Question B', token_id='789', market_id='b', no_token_id='987'
    )
    trader = _make_trader()
    for ticker in [a, b, a.get_no_ticker(), b.get_no_ticker()]:
        _seed_order_book(trader, ticker, Decimal('0.01'), Decimal('0.02'))
    unknown = ImplicationArbStrategy(market_id_a='a', market_id_b='b', fee_rate=0)
    await unknown._enter(trader, Decimal('0.2'))
    assert trader.orders == []
    known = ImplicationArbStrategy(
        market_id_a='a',
        market_id_b='b',
        fee_rate=0,
        yes_token_ids={'a': '123', 'b': '789'},
    )
    await known.process_event(
        PriceChangeEvent(ticker=a.get_no_ticker(), price=Decimal('0.99')), trader
    )
    assert known._price_a is None
    await known._enter(trader, Decimal('0.2'))
    assert known._position_state == 'short_a_long_b'
    assert [order.ticker.token_id for order in trader.orders] == ['456', '789']


async def test_exact_identifiers_prevent_substring_market_match():
    strategy, trader, _, _ = _setup(ImplicationArbStrategy)
    unrelated = _make_ticker('X', market_id='abc')
    before = strategy._price_a
    await strategy.process_event(
        PriceChangeEvent(ticker=unrelated, price=Decimal('0.01')), trader
    )
    assert strategy._price_a == before


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
@pytest.mark.parametrize('cls', STRATEGIES)
def test_invalid_contract_quantity_rejected(cls, value):
    with pytest.raises(ValueError, match='trade_size'):
        cls(trade_size=value)


@pytest.mark.parametrize('cls', STRATEGIES)
async def test_nonfinite_price_events_ignored(cls):
    strategy, trader, a, _ = _setup(cls)
    before = strategy._price_a
    await strategy.process_event(
        PriceChangeEvent(ticker=a, price=Decimal('NaN')), trader
    )
    assert strategy._price_a == before
    assert trader.orders == []


async def test_conditional_upper_has_correct_no_b_hedge_weight():
    strategy, trader, a, b = _setup(ConditionalArbStrategy, cond_upper=0.6)
    await _enter(strategy, trader)
    assert strategy._owned_positions == {
        a.get_no_ticker(): Decimal('10'),
        b.get_no_ticker(): Decimal('4'),
    }
    assert strategy._position_state == 'short_both'
    assert strategy.get_decisions()[-1].signal_values['model_dependent'] == 1


async def test_conditional_lower_has_correct_no_b_hedge_weight():
    strategy, trader, a, b = _setup(ConditionalArbStrategy, cond_lower=0.3)
    await strategy._enter_long_a(trader, 0.02, 0.1, 0.9)
    assert strategy._owned_positions == {
        a: Decimal('10'),
        b.get_no_ticker(): Decimal('3'),
    }


@pytest.mark.parametrize(
    'slope,intercept,long_a',
    [(2.0, 0.0, False), (2.0, 0.0, True), (-2.0, 1.0, False), (-2.0, 1.0, True)],
)
async def test_structural_basket_respects_signed_slope(slope, intercept, long_a):
    strategy, trader, a, b = _setup(
        StructuralArbStrategy, slope=slope, intercept=intercept
    )
    if long_a:
        await strategy._enter_long_a(trader, 0.01, 0.5, -0.49)
    else:
        await _enter(strategy, trader)
    expected_a = a if long_a else a.get_no_ticker()
    b_yes = (slope < 0) if long_a else (slope >= 0)
    expected_b = b if b_yes else b.get_no_ticker()
    assert strategy._owned_positions == {
        expected_a: Decimal('10'),
        expected_b: Decimal('20'),
    }
    assert 'not a guaranteed payout' in strategy.get_decisions()[-1].reasoning
