"""Adapter uncertainty takes precedence over superficially complete fill fields."""

from dataclasses import replace
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
from test_spread_executor import (
    _confirmed_order,
    _make_ticker,
    _make_trader,
    _residuals,
)

from oracle3.trader.spread_executor import SpreadExecutor, SpreadLeg
from oracle3.trader.types import (
    OrderFailureReason,
    OrderStatus,
    PlaceOrderResult,
    TradeSide,
)


def _fixture():
    tickers = [_make_ticker(symbol) for symbol in ('A', 'B', 'C')]
    trader = _make_trader({t: (Decimal('0.4'), Decimal('0.5')) for t in tickers})
    legs = [SpreadLeg(TradeSide.BUY, t, Decimal('0.5'), Decimal('10')) for t in tickers]
    return trader, legs


@pytest.mark.parametrize(
    'flag',
    ['accounting_unconfirmed', 'execution_uncertain', 'result_requires_reconciliation'],
)
async def test_full_fill_flags_block_success_next_leg_and_unwind(flag):
    trader, legs = _fixture()
    reply = _confirmed_order(legs[0].ticker)
    if flag == 'accounting_unconfirmed':
        reply.order.accounting_confirmed = False
    elif flag == 'execution_uncertain':
        reply.order.execution_uncertain = True
    else:
        reply.reconciliation_required = True
    trader.place_order = AsyncMock(return_value=reply)
    result = await SpreadExecutor(trader).execute_spread(legs)
    assert trader.place_order.await_count == 1
    assert not result.success and not result.all_filled and not result.hedged
    assert result.execution_uncertain and result.requires_reconciliation
    assert not result.unwind_results
    assert _residuals(result) == (
        {'A': Decimal('10')} if flag == 'result_requires_reconciliation' else {}
    )


@pytest.mark.parametrize(
    'reply',
    [
        PlaceOrderResult(failure_reason=OrderFailureReason.RECONCILIATION_REQUIRED),
        PlaceOrderResult(
            failure_reason=OrderFailureReason.INVALID_ORDER,
            reconciliation_required=True,
        ),
    ],
)
async def test_reconciliation_flag_without_order_is_not_known_rejection(reply):
    trader, legs = _fixture()
    trader.place_order = AsyncMock(return_value=reply)
    result = await SpreadExecutor(trader).execute_spread(legs)
    assert result.execution_uncertain and result.requires_reconciliation
    assert not result.residual_exposures
    assert trader.place_order.await_count == 1


async def test_unconfirmed_second_leg_keeps_only_prior_known_exposure():
    trader, legs = _fixture()
    pending = _confirmed_order(legs[1].ticker)
    pending.order = replace(
        pending.order, accounting_confirmed=False, execution_uncertain=True
    )
    trader.place_order = AsyncMock(
        side_effect=[_confirmed_order(legs[0].ticker), pending]
    )
    result = await SpreadExecutor(trader).execute_spread(legs)
    assert trader.place_order.await_count == 2
    assert _residuals(result) == {'A': Decimal('10')}
    assert not result.unwind_results
    assert result.uncertain_leg_indices == [2]


@pytest.mark.parametrize('confirmed', [False, True])
async def test_uncertain_unwind_stops_before_reversing_other_legs(confirmed):
    trader, legs = _fixture()
    unwind = _confirmed_order(
        legs[1].ticker,
        '3',
        side=TradeSide.SELL,
        status=OrderStatus.PARTIALLY_FILLED,
        remaining='7',
    )
    if not confirmed:
        unwind.order.accounting_confirmed = False
        unwind.order.execution_uncertain = True
    trader.place_order = AsyncMock(
        side_effect=[
            _confirmed_order(legs[0].ticker),
            _confirmed_order(legs[1].ticker),
            PlaceOrderResult(failure_reason=OrderFailureReason.INVALID_ORDER),
            unwind,
        ]
    )
    result = await SpreadExecutor(trader).execute_spread(legs)
    assert trader.place_order.await_count == 4
    assert len(result.unwind_results) == 1
    assert _residuals(result) == {
        'A': Decimal('10'),
        'B': Decimal('7' if confirmed else '10'),
    }
    assert result.requires_reconciliation and not result.hedged
    assert 'Recovery stopped' in result.unwind_details
