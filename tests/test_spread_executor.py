"""Tests for sequential spread execution and truthful recovery state.

Covers: successful 2-leg spread, auto-unwind on leg 2 failure,
all legs succeed, execute_pair_spread convenience method.
"""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from oracle3.data.market_data_manager import MarketDataManager
from oracle3.order.order_book import Level, OrderBook
from oracle3.position.position_manager import Position, PositionManager
from oracle3.risk.risk_manager import NoRiskManager
from oracle3.ticker.ticker import CashTicker, PolyMarketTicker
from oracle3.trader.paper_trader import PaperTrader
from oracle3.trader.spread_executor import (
    SpreadExecutor,
    SpreadLeg,
    SpreadOrderResult,
)
from oracle3.trader.types import (
    Order,
    OrderFailureReason,
    OrderStatus,
    PlaceOrderResult,
    TradeSide,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ticker(
    symbol: str, name: str = '', no_token_id: str = ''
) -> PolyMarketTicker:
    return PolyMarketTicker(
        symbol=symbol,
        name=name or symbol,
        token_id=symbol,
        market_id=symbol,
        event_id='',
        no_token_id=no_token_id,
    )


def _make_trader(
    tickers_with_books: dict[PolyMarketTicker, tuple[Decimal, Decimal]] | None = None,
    cash: Decimal = Decimal('10000'),
) -> PaperTrader:
    """Create a PaperTrader with 100% fill, no commission, no risk checks."""
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
    if tickers_with_books:
        for ticker, (bid, ask) in tickers_with_books.items():
            ob = OrderBook()
            ob.update(
                asks=[Level(price=ask, size=Decimal('1000'))],
                bids=[Level(price=bid, size=Decimal('1000'))],
            )
            md.update_order_book(ticker, ob)

    return PaperTrader(
        market_data=md,
        risk_manager=NoRiskManager(),
        position_manager=pm,
        min_fill_rate=Decimal('1'),
        max_fill_rate=Decimal('1'),
        commission_rate=Decimal('0'),
    )


def _seed_position(
    trader: PaperTrader,
    ticker: PolyMarketTicker,
    quantity: Decimal,
    avg_cost: Decimal,
) -> None:
    """Seed a position for sell-side testing."""
    trader.position_manager.update_position(
        Position(
            ticker=ticker,
            quantity=quantity,
            average_cost=avg_cost,
            realized_pnl=Decimal('0'),
        )
    )


# ===================================================================
# SpreadLeg / SpreadOrderResult dataclass tests
# ===================================================================


class TestSpreadDataClasses:
    def test_spread_leg_creation(self):
        t = _make_ticker('A')
        leg = SpreadLeg(
            side=TradeSide.BUY,
            ticker=t,
            price=Decimal('0.50'),
            quantity=Decimal('10'),
        )
        assert leg.side == TradeSide.BUY
        assert leg.ticker == t
        assert leg.price == Decimal('0.50')
        assert leg.quantity == Decimal('10')

    def test_spread_order_result_success(self):
        r = SpreadOrderResult(success=True, leg_results=[])
        assert r.success is True
        assert r.hedged is False
        assert r.failure_reason == ''

    def test_spread_order_result_all_filled_empty(self):
        r = SpreadOrderResult(success=True, leg_results=[])
        assert r.all_filled is False

    def test_spread_order_result_failure(self):
        r = SpreadOrderResult(
            success=False,
            failure_reason='leg 2 failed: insufficient_cash',
        )
        assert r.success is False
        assert 'leg 2' in r.failure_reason


# ===================================================================
# SpreadExecutor: empty legs
# ===================================================================


class TestSpreadExecutorEmpty:
    async def test_empty_legs_returns_failure(self):
        trader = _make_trader()
        executor = SpreadExecutor(trader)
        result = await executor.execute_spread([])
        assert result.success is False
        assert 'no legs' in result.failure_reason


# ===================================================================
# SpreadExecutor: successful 2-leg spread
# ===================================================================


class TestSpreadExecutorSuccess:
    async def test_two_leg_buy_buy(self):
        """Both legs are BUY orders -- both should fill."""
        ticker_a = _make_ticker('A')
        ticker_b = _make_ticker('B')

        trader = _make_trader(
            {
                ticker_a: (Decimal('0.45'), Decimal('0.55')),
                ticker_b: (Decimal('0.35'), Decimal('0.45')),
            }
        )
        executor = SpreadExecutor(trader)
        legs = [
            SpreadLeg(TradeSide.BUY, ticker_a, Decimal('0.55'), Decimal('10')),
            SpreadLeg(TradeSide.BUY, ticker_b, Decimal('0.45'), Decimal('10')),
        ]

        result = await executor.execute_spread(legs)
        assert result.success is True
        assert len(result.leg_results) == 2
        assert result.all_filled is True
        assert result.hedged is False

    async def test_two_leg_buy_sell(self):
        """Buy A, sell B (requires position in B)."""
        ticker_a = _make_ticker('A')
        ticker_b = _make_ticker('B')

        trader = _make_trader(
            {
                ticker_a: (Decimal('0.45'), Decimal('0.55')),
                ticker_b: (Decimal('0.45'), Decimal('0.55')),
            }
        )
        # Seed position in B so we can sell it
        _seed_position(trader, ticker_b, Decimal('20'), Decimal('0.40'))

        executor = SpreadExecutor(trader)
        legs = [
            SpreadLeg(TradeSide.BUY, ticker_a, Decimal('0.55'), Decimal('10')),
            SpreadLeg(TradeSide.SELL, ticker_b, Decimal('0.45'), Decimal('10')),
        ]

        result = await executor.execute_spread(legs)
        assert result.success is True
        assert len(result.leg_results) == 2
        assert result.all_filled is True

    async def test_three_leg_spread(self):
        """All 3 legs succeed."""
        t1 = _make_ticker('T1')
        t2 = _make_ticker('T2')
        t3 = _make_ticker('T3')

        trader = _make_trader(
            {
                t1: (Decimal('0.25'), Decimal('0.35')),
                t2: (Decimal('0.25'), Decimal('0.35')),
                t3: (Decimal('0.25'), Decimal('0.35')),
            }
        )
        executor = SpreadExecutor(trader)
        legs = [
            SpreadLeg(TradeSide.BUY, t1, Decimal('0.35'), Decimal('5')),
            SpreadLeg(TradeSide.BUY, t2, Decimal('0.35'), Decimal('5')),
            SpreadLeg(TradeSide.BUY, t3, Decimal('0.35'), Decimal('5')),
        ]

        result = await executor.execute_spread(legs)
        assert result.success is True
        assert len(result.leg_results) == 3
        assert result.all_filled is True


# ===================================================================
# SpreadExecutor: auto-unwind on leg 2 failure
# ===================================================================


class TestSpreadExecutorUnwind:
    async def test_unwind_on_second_leg_failure(self):
        """Leg 1 fills, leg 2 fails (insufficient cash) => unwind leg 1."""
        ticker_a = _make_ticker('A')
        ticker_b = _make_ticker('B')

        # Very little cash -- enough for leg 1, not leg 2
        trader = _make_trader(
            {
                ticker_a: (Decimal('0.45'), Decimal('0.55')),
                ticker_b: (Decimal('0.85'), Decimal('0.95')),
            },
            cash=Decimal('6'),
        )
        executor = SpreadExecutor(trader)
        legs = [
            SpreadLeg(TradeSide.BUY, ticker_a, Decimal('0.55'), Decimal('10')),
            SpreadLeg(TradeSide.BUY, ticker_b, Decimal('0.95'), Decimal('10')),
        ]

        result = await executor.execute_spread(legs)
        assert result.success is False
        assert result.hedged is True
        assert 'leg 2' in result.failure_reason
        assert result.unwind_details != ''

    async def test_no_unwind_when_disabled(self):
        """When unwind_on_partial=False, no unwind occurs."""
        ticker_a = _make_ticker('A')
        ticker_b = _make_ticker('B')

        trader = _make_trader(
            {
                ticker_a: (Decimal('0.45'), Decimal('0.55')),
                ticker_b: (Decimal('0.85'), Decimal('0.95')),
            },
            cash=Decimal('6'),
        )
        executor = SpreadExecutor(trader)
        legs = [
            SpreadLeg(TradeSide.BUY, ticker_a, Decimal('0.55'), Decimal('10')),
            SpreadLeg(TradeSide.BUY, ticker_b, Decimal('0.95'), Decimal('10')),
        ]

        result = await executor.execute_spread(legs, unwind_on_partial=False)
        assert result.success is False
        assert result.hedged is False
        assert result.unwind_details == ''

    async def test_first_leg_failure_no_unwind_needed(self):
        """When the very first leg fails, no unwind is needed."""
        ticker_a = _make_ticker('A')

        trader = _make_trader(
            {ticker_a: (Decimal('0.45'), Decimal('0.55'))},
            cash=Decimal('0'),  # no cash at all
        )
        executor = SpreadExecutor(trader)
        legs = [
            SpreadLeg(TradeSide.BUY, ticker_a, Decimal('0.55'), Decimal('10')),
        ]

        result = await executor.execute_spread(legs)
        assert result.success is False
        assert result.hedged is False
        assert len(result.leg_results) == 1

    async def test_unwind_on_third_leg_failure(self):
        """3 legs: legs 1&2 fill, leg 3 fails => unwind legs 1&2."""
        t1 = _make_ticker('T1')
        t2 = _make_ticker('T2')
        t3 = _make_ticker('T3')

        # Enough cash for 2 legs but not 3
        trader = _make_trader(
            {
                t1: (Decimal('0.25'), Decimal('0.35')),
                t2: (Decimal('0.25'), Decimal('0.35')),
                t3: (Decimal('0.85'), Decimal('0.95')),
            },
            cash=Decimal('8'),
        )
        executor = SpreadExecutor(trader)
        legs = [
            SpreadLeg(TradeSide.BUY, t1, Decimal('0.35'), Decimal('10')),
            SpreadLeg(TradeSide.BUY, t2, Decimal('0.35'), Decimal('10')),
            SpreadLeg(TradeSide.BUY, t3, Decimal('0.95'), Decimal('10')),
        ]

        result = await executor.execute_spread(legs)
        assert result.success is False
        assert result.hedged is True
        assert 'leg 3' in result.failure_reason


# ===================================================================
# SpreadExecutor: execute_pair_spread convenience
# ===================================================================


class TestSpreadExecutorPairSpread:
    async def test_pair_spread_success(self):
        """Buy A, sell B using best ask/bid automatically."""
        ticker_a = _make_ticker('A')
        ticker_b = _make_ticker('B')

        trader = _make_trader(
            {
                ticker_a: (Decimal('0.45'), Decimal('0.55')),
                ticker_b: (Decimal('0.45'), Decimal('0.55')),
            }
        )
        # Need a position in B to sell
        _seed_position(trader, ticker_b, Decimal('20'), Decimal('0.40'))

        executor = SpreadExecutor(trader)
        result = await executor.execute_pair_spread(
            buy_ticker=ticker_a,
            sell_ticker=ticker_b,
            quantity=Decimal('10'),
        )
        assert result.success is True
        assert len(result.leg_results) == 2

    async def test_pair_spread_no_ask(self):
        """When no ask is available for buy_ticker, fail gracefully."""
        ticker_a = _make_ticker('A')
        ticker_b = _make_ticker('B')

        trader = _make_trader(
            {
                # No order book for ticker_a
                ticker_b: (Decimal('0.45'), Decimal('0.55')),
            }
        )

        executor = SpreadExecutor(trader)
        result = await executor.execute_pair_spread(
            buy_ticker=ticker_a,
            sell_ticker=ticker_b,
            quantity=Decimal('10'),
        )
        assert result.success is False
        assert 'no ask' in result.failure_reason

    async def test_pair_spread_no_bid(self):
        """When no bid is available for sell_ticker, fail gracefully."""
        ticker_a = _make_ticker('A')
        ticker_b = _make_ticker('B')

        trader = _make_trader({ticker_a: (Decimal('0.45'), Decimal('0.55'))})

        executor = SpreadExecutor(trader)
        result = await executor.execute_pair_spread(
            buy_ticker=ticker_a,
            sell_ticker=ticker_b,
            quantity=Decimal('10'),
        )
        assert result.success is False
        assert 'no bid' in result.failure_reason


# ===================================================================
# SpreadExecutor: all_filled property
# ===================================================================


class TestSpreadOrderResultAllFilled:
    async def test_all_filled_true_on_full_fills(self):
        ticker_a = _make_ticker('A')
        trader = _make_trader({ticker_a: (Decimal('0.45'), Decimal('0.55'))})
        executor = SpreadExecutor(trader)
        legs = [
            SpreadLeg(TradeSide.BUY, ticker_a, Decimal('0.55'), Decimal('10')),
        ]
        result = await executor.execute_spread(legs)
        assert result.all_filled is True

    def test_all_filled_false_on_no_fill(self):
        """Manually construct a result with a None order."""
        result = SpreadOrderResult(
            success=False,
            leg_results=[
                PlaceOrderResult(
                    order=None,
                    failure_reason=OrderFailureReason.INSUFFICIENT_CASH,
                )
            ],
        )
        assert result.all_filled is False


# ===================================================================
# Edge cases
# ===================================================================


class TestSpreadExecutorEdgeCases:
    async def test_single_leg_spread(self):
        """A single-leg 'spread' should work fine."""
        ticker = _make_ticker('X')
        trader = _make_trader({ticker: (Decimal('0.45'), Decimal('0.55'))})
        executor = SpreadExecutor(trader)
        legs = [
            SpreadLeg(TradeSide.BUY, ticker, Decimal('0.55'), Decimal('5')),
        ]
        result = await executor.execute_spread(legs)
        assert result.success is True
        assert len(result.leg_results) == 1

    async def test_position_updated_after_spread(self):
        """Verify that positions are correctly updated after a spread."""
        ticker_a = _make_ticker('A')
        ticker_b = _make_ticker('B')

        trader = _make_trader(
            {
                ticker_a: (Decimal('0.45'), Decimal('0.55')),
                ticker_b: (Decimal('0.25'), Decimal('0.35')),
            }
        )
        executor = SpreadExecutor(trader)
        legs = [
            SpreadLeg(TradeSide.BUY, ticker_a, Decimal('0.55'), Decimal('10')),
            SpreadLeg(TradeSide.BUY, ticker_b, Decimal('0.35'), Decimal('10')),
        ]

        result = await executor.execute_spread(legs)
        assert result.success is True

        # Check positions were created
        pos_a = trader.position_manager.get_position(ticker_a)
        pos_b = trader.position_manager.get_position(ticker_b)
        assert pos_a is not None
        assert pos_a.quantity == Decimal('10')
        assert pos_b is not None
        assert pos_b.quantity == Decimal('10')


def _confirmed_order(
    ticker: PolyMarketTicker,
    filled: str = '10',
    *,
    side: TradeSide = TradeSide.BUY,
    status: OrderStatus = OrderStatus.FILLED,
    remaining: str = '0',
) -> PlaceOrderResult:
    return PlaceOrderResult(
        order=Order(
            status=status,
            side=side,
            ticker=ticker,
            limit_price=Decimal('0.50'),
            filled_quantity=Decimal(filled),
            average_price=Decimal('0.50'),
            trades=[],
            remaining=Decimal(remaining),
            commission=Decimal('0'),
        )
    )


def _rejected_order() -> PlaceOrderResult:
    return PlaceOrderResult(failure_reason=OrderFailureReason.RISK_CHECK_FAILED)


def _residuals(result: SpreadOrderResult) -> dict[str, Decimal]:
    return {item.ticker.symbol: item.quantity for item in result.residual_exposures}


class TestSpreadRecoveryContract:
    @pytest.fixture
    def setup_spread(self):
        tickers = [_make_ticker(symbol) for symbol in ('A', 'B', 'C')]
        trader = _make_trader({t: (Decimal('.49'), Decimal('.50')) for t in tickers})
        legs = [
            SpreadLeg(TradeSide.BUY, t, Decimal('.50'), Decimal('10')) for t in tickers
        ]
        return trader, legs

    async def test_partial_fill_stops_later_legs_and_reports_actual_exposure(
        self, setup_spread
    ):
        trader, legs = setup_spread
        replies = [
            _confirmed_order(legs[0].ticker),
            _confirmed_order(
                legs[1].ticker, '1', status=OrderStatus.PARTIALLY_FILLED, remaining='9'
            ),
        ]
        with patch.object(
            trader, 'place_order', AsyncMock(side_effect=replies)
        ) as place:
            result = await SpreadExecutor(trader).execute_spread(
                legs, unwind_on_partial=False
            )
        assert place.await_count == 2
        assert not result.success and not result.all_filled and not result.hedged
        assert _residuals(result) == {'A': Decimal('10'), 'B': Decimal('1')}
        assert result.execution_uncertain and result.requires_reconciliation
        assert result.uncertain_leg_indices == [2]
        payload = json.loads(json.dumps(result.to_payload()))
        assert payload['requested_legs'][1]['quantity'] == '10'
        assert payload['leg_results'][1]['order']['filled_quantity'] == '1'

    @pytest.mark.parametrize(
        'status,uncertain',
        [
            (OrderStatus.CANCELLED, False),
            (OrderStatus.PARTIALLY_FILLED, True),
        ],
    )
    async def test_unwind_includes_current_partial_leg_but_pending_order_prevents_hedged(
        self, setup_spread, status, uncertain
    ):
        trader, legs = setup_spread
        replies = [
            _confirmed_order(legs[0].ticker),
            _confirmed_order(legs[1].ticker, '1', status=status, remaining='9'),
            _confirmed_order(legs[1].ticker, '1', side=TradeSide.SELL),
            _confirmed_order(legs[0].ticker, side=TradeSide.SELL),
        ]
        with patch.object(
            trader, 'place_order', AsyncMock(side_effect=replies)
        ) as place:
            result = await SpreadExecutor(trader).execute_spread(legs)
        assert not result.success and not result.all_filled
        assert result.hedged is (not uncertain)
        assert result.execution_uncertain is uncertain
        assert result.requires_reconciliation is uncertain
        assert not result.residual_exposures
        assert [c.kwargs['quantity'] for c in place.await_args_list] == [
            Decimal('10'),
            Decimal('10'),
            Decimal('1'),
            Decimal('10'),
        ]
        assert [item.leg_index for item in result.unwind_results] == [2, 1]
        assert all(item.result is not None for item in result.unwind_results)

    async def test_failed_unwind_retains_exposure(self, setup_spread):
        trader, legs = setup_spread
        replies = [
            _confirmed_order(legs[0].ticker),
            _rejected_order(),
            _rejected_order(),
        ]
        with patch.object(trader, 'place_order', AsyncMock(side_effect=replies)):
            result = await SpreadExecutor(trader).execute_spread(legs)
        assert not result.hedged and not result.execution_uncertain
        assert result.requires_reconciliation
        assert _residuals(result) == {'A': Decimal('10')}
        assert result.unwind_results[0].failure_reason == 'risk_check_failed'
        assert 'FAILED' in result.unwind_details

    @pytest.mark.parametrize(
        'status,uncertain',
        [
            (OrderStatus.CANCELLED, False),
            (OrderStatus.PARTIALLY_FILLED, True),
        ],
    )
    async def test_partial_unwind_reports_only_confirmed_reduction(
        self, setup_spread, status, uncertain
    ):
        trader, legs = setup_spread
        replies = [
            _confirmed_order(legs[0].ticker),
            _rejected_order(),
            _confirmed_order(
                legs[0].ticker, '4', side=TradeSide.SELL, status=status, remaining='6'
            ),
        ]
        with patch.object(trader, 'place_order', AsyncMock(side_effect=replies)):
            result = await SpreadExecutor(trader).execute_spread(legs)
        assert not result.hedged
        assert _residuals(result) == {'A': Decimal('6')}
        assert result.execution_uncertain is uncertain
        assert result.unwind_results[0].execution_uncertain is uncertain
        assert result.requires_reconciliation

    async def test_leg_exception_recovers_prior_fills_without_claiming_flat(
        self, setup_spread
    ):
        trader, legs = setup_spread
        replies = [
            _confirmed_order(legs[0].ticker),
            TimeoutError('unknown venue state'),
            _confirmed_order(legs[0].ticker, side=TradeSide.SELL),
        ]
        with patch.object(
            trader, 'place_order', AsyncMock(side_effect=replies)
        ) as place:
            result = await SpreadExecutor(trader).execute_spread(legs)
        assert place.await_count == 3
        assert not result.success and not result.hedged
        assert not result.residual_exposures
        assert result.execution_uncertain and result.requires_reconciliation
        assert result.uncertain_leg_indices == [2]
        assert 'TimeoutError' in result.failure_reason
        assert result.leg_results[1].failure_reason == OrderFailureReason.UNKNOWN

    async def test_unwind_exception_reports_uncertainty_and_known_residual(
        self, setup_spread
    ):
        trader, legs = setup_spread
        replies = [_confirmed_order(legs[0].ticker), _rejected_order(), TimeoutError()]
        with patch.object(trader, 'place_order', AsyncMock(side_effect=replies)):
            result = await SpreadExecutor(trader).execute_spread(legs)
        assert not result.hedged and result.execution_uncertain
        assert _residuals(result) == {'A': Decimal('10')}
        assert result.unwind_results[0].execution_uncertain
        assert 'TimeoutError' in result.unwind_results[0].failure_reason

    async def test_overfill_is_not_success_and_unwinds_actual_quantity(
        self, setup_spread
    ):
        trader, legs = setup_spread
        replies = [
            _confirmed_order(legs[0].ticker, '11'),
            _confirmed_order(legs[0].ticker, '11', side=TradeSide.SELL),
        ]
        with patch.object(
            trader, 'place_order', AsyncMock(side_effect=replies)
        ) as place:
            result = await SpreadExecutor(trader).execute_spread(legs)
        assert not result.success and not result.all_filled
        assert result.hedged and not result.execution_uncertain
        assert place.await_args_list[1].kwargs['quantity'] == Decimal('11')

    async def test_pending_zero_fill_is_not_a_known_rejection(self, setup_spread):
        trader, legs = setup_spread
        replies = [
            _confirmed_order(
                legs[0].ticker, '0', status=OrderStatus.PLACED, remaining='10'
            )
        ]
        with patch.object(
            trader, 'place_order', AsyncMock(side_effect=replies)
        ) as place:
            result = await SpreadExecutor(trader).execute_spread(legs)
        assert place.await_count == 1
        assert not result.success and not result.hedged
        assert result.execution_uncertain and result.requires_reconciliation
        assert not result.residual_exposures

    async def test_mismatched_order_is_not_reversed_as_the_requested_ticker(
        self, setup_spread
    ):
        trader, legs = setup_spread
        with patch.object(
            trader,
            'place_order',
            AsyncMock(return_value=_confirmed_order(legs[1].ticker)),
        ) as place:
            result = await SpreadExecutor(trader).execute_spread(legs)
        assert place.await_count == 1
        assert not result.success and not result.hedged and not result.all_filled
        assert result.execution_uncertain and result.requires_reconciliation

    @pytest.mark.parametrize('bad_quantity', ['0', '-1', 'NaN', 'Infinity'])
    async def test_invalid_later_leg_is_rejected_before_any_order(
        self, setup_spread, bad_quantity
    ):
        trader, legs = setup_spread
        legs[1].quantity = Decimal(bad_quantity)
        with patch.object(trader, 'place_order', AsyncMock()) as place:
            result = await SpreadExecutor(trader).execute_spread(legs)
        place.assert_not_awaited()
        assert not result.success and not result.all_filled
        assert not result.execution_uncertain and not result.requires_reconciliation

    async def test_missing_unwind_quote_preserves_residual_without_submitting_fallback(
        self, setup_spread
    ):
        trader, legs = setup_spread
        replies = [_confirmed_order(legs[0].ticker), _rejected_order()]
        with patch.object(
            trader, 'place_order', AsyncMock(side_effect=replies)
        ) as place:
            with patch.object(trader.market_data, 'get_best_bid', return_value=None):
                result = await SpreadExecutor(trader).execute_spread(legs)
        assert place.await_count == 2
        assert not result.hedged
        assert _residuals(result) == {'A': Decimal('10')}
        assert result.unwind_results[0].result is None
        assert 'no valid executable quote' in result.unwind_results[0].failure_reason

    async def test_malformed_order_status_returns_json_safe_uncertain_state(
        self, setup_spread
    ):
        trader, legs = setup_spread
        reply = _confirmed_order(legs[0].ticker)
        reply.order.status = 'unexpected adapter value'
        with patch.object(
            trader, 'place_order', AsyncMock(return_value=reply)
        ) as place:
            result = await SpreadExecutor(trader).execute_spread(legs)
        assert place.await_count == 1
        assert not result.success and result.execution_uncertain
        assert result.requires_reconciliation and not result.hedged
        payload = json.loads(json.dumps(result.to_payload()))
        assert payload['leg_results'][0]['failure_reason'] == 'unknown'
