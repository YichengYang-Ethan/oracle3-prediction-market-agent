"""Ledger atomicity and paper numeric contracts; no network or live orders."""

from copy import deepcopy
from decimal import Decimal
from typing import Any

import pytest

from oracle3.data.market_data_manager import MarketDataManager
from oracle3.order.order_book import Level, OrderBook
from oracle3.position.position_manager import Position, PositionManager
from oracle3.risk.risk_manager import NoRiskManager
from oracle3.ticker.ticker import PolyMarketTicker
from oracle3.trader.paper_trader import PaperTrader
from oracle3.trader.types import OrderFailureReason, Trade, TradeSide

D = Decimal
TICKER = PolyMarketTicker('POSITION_INTEGRITY', token_id='integrity-token')


def funded_positions(*, cash: str = '100', held: str = '10') -> PositionManager:
    positions = PositionManager()
    positions.update_position(Position(TICKER.collateral, D(cash), D('0'), D('0')))
    positions.update_position(Position(TICKER, D(held), D('0.4'), D('2')))
    return positions


def book_with(*, bid: str | None = None, ask: str | None = None) -> OrderBook:
    book = OrderBook()
    book.update(
        bids=[] if bid is None else [Level(D(bid), D('100'))],
        asks=[] if ask is None else [Level(D(ask), D('100'))],
    )
    return book


@pytest.mark.parametrize('field', ['price', 'quantity', 'commission'])
@pytest.mark.parametrize('value', ['NaN', 'sNaN', 'Infinity', '-Infinity', '-1'])
def test_invalid_trade_numeric_never_mutates_ledger(field: str, value: str):
    positions = funded_positions()
    before = deepcopy(positions.positions)
    trade = Trade(TradeSide.BUY, TICKER, D('0.5'), D('1'), D('0'))
    setattr(trade, field, D(value))
    with pytest.raises(ValueError):
        positions.apply_trade(trade)
    assert positions.positions == before


@pytest.mark.parametrize('side', [None, 'buy', 'sell', 'invalid'])
def test_invalid_trade_side_never_mutates_ledger(side: Any):
    positions = funded_positions()
    before = deepcopy(positions.positions)
    with pytest.raises(ValueError, match='side'):
        positions.apply_trade(Trade(side, TICKER, D('0.5'), D('1'), D('0')))
    assert positions.positions == before


@pytest.mark.parametrize('side', [TradeSide.BUY, TradeSide.SELL])
def test_missing_cash_never_changes_existing_position(side: TradeSide):
    positions = funded_positions()
    del positions.positions[TICKER.collateral.symbol]
    before = deepcopy(positions.positions)
    with pytest.raises(ValueError, match='Missing collateral'):
        positions.apply_trade(Trade(side, TICKER, D('0.5'), D('1'), D('0.01')))
    assert positions.positions == before


def test_missing_cash_never_creates_a_position():
    positions = PositionManager()
    with pytest.raises(ValueError, match='Missing collateral'):
        positions.apply_trade(Trade(TradeSide.BUY, TICKER, D('0.5'), D('1'), D('0')))
    assert positions.positions == {}


@pytest.mark.parametrize(
    ('side', 'quantity', 'price', 'commission'),
    [
        (TradeSide.BUY, '300', '0.5', '0'),
        (TradeSide.BUY, '200', '0.5', '0.01'),
        (TradeSide.SELL, '11', '0.5', '0'),
        (TradeSide.SELL, '1', '0.5', '101'),
        (TradeSide.BUY, '0', '0.5', '0'),
    ],
)
def test_balance_and_inventory_rejections_are_atomic(
    side: TradeSide, quantity: str, price: str, commission: str
):
    positions = funded_positions()
    before = deepcopy(positions.positions)
    with pytest.raises(ValueError):
        positions.apply_trade(Trade(side, TICKER, D(price), D(quantity), D(commission)))
    assert positions.positions == before


def test_zero_price_sell_records_actual_full_loss():
    positions = funded_positions()
    result = positions.apply_trade(
        Trade(TradeSide.SELL, TICKER, D('0'), D('10'), D('0'))
    )
    assert result.quantity == 0
    assert result.average_cost == 0
    assert result.realized_pnl == D('-2')  # Prior realized 2 less cost 4.
    assert positions.positions[TICKER.collateral.symbol].quantity == 100


def test_missing_quote_uses_cost_basis_and_is_reported():
    positions = funded_positions()
    market_data = MarketDataManager()
    assert positions.get_portfolio_value(market_data)[TICKER.collateral.symbol] == 104
    assert positions.get_total_unrealized_pnl(market_data) == 0
    assert positions.get_missing_marks(market_data) == [TICKER.symbol]


@pytest.mark.parametrize('observed', ['0', '0.7'])
def test_removed_quote_retains_last_observed_mark(observed: str):
    positions = funded_positions()
    market_data = MarketDataManager()
    market_data.order_books[TICKER] = book_with(bid=observed, ask='0.8')
    expected_value = D('100') + D('10') * D(observed)
    assert positions.get_portfolio_value(market_data)[TICKER.collateral.symbol] == (
        expected_value
    )
    assert positions.get_missing_marks(market_data) == []
    market_data.order_books.clear()
    assert positions.get_portfolio_value(market_data)[TICKER.collateral.symbol] == (
        expected_value
    )
    assert (
        positions.get_unrealized_pnl(TICKER, market_data)
        == (D(observed) - D('0.4')) * 10
    )
    assert positions.get_missing_marks(market_data) == [TICKER.symbol]


def test_real_zero_bid_is_not_replaced_by_ask_or_cost():
    positions = funded_positions()
    market_data = MarketDataManager()
    market_data.order_books[TICKER] = book_with(bid='0', ask='0.5')
    assert positions.get_portfolio_value(market_data)[TICKER.collateral.symbol] == 100
    assert positions.get_unrealized_pnl(TICKER, market_data) == D('-4')
    assert positions.get_missing_marks(market_data) == []


@pytest.mark.parametrize('bad_mark', ['NaN', 'Infinity', '-0.1', '1.1'])
def test_unusable_quote_is_missing_and_cannot_poison_value(bad_mark: str):
    positions = funded_positions()
    market_data = MarketDataManager()
    market_data.order_books[TICKER] = book_with(bid=bad_mark)
    assert positions.get_portfolio_value(market_data)[TICKER.collateral.symbol] == 104
    assert positions.get_missing_marks(market_data) == [TICKER.symbol]
    market_data.order_books[TICKER] = book_with(bid=bad_mark, ask='0.6')
    assert positions.get_portfolio_value(market_data)[TICKER.collateral.symbol] == 106
    assert positions.get_missing_marks(market_data) == []


def test_missing_marks_only_include_held_assets_and_are_sorted():
    positions = funded_positions()
    other = PolyMarketTicker('AAA')
    closed = PolyMarketTicker('CLOSED')
    positions.update_position(Position(other, D('1'), D('0.5'), D('0')))
    positions.update_position(Position(closed, D('0'), D('0'), D('0')))
    assert positions.get_missing_marks(MarketDataManager()) == ['AAA', TICKER.symbol]


def paper_trader(**rate_overrides: Decimal) -> PaperTrader:
    market_data = MarketDataManager()
    market_data.order_books[TICKER] = book_with(bid='0.4', ask='0.5')
    rates = {
        'min_fill_rate': D('1'),
        'max_fill_rate': D('1'),
        'commission_rate': D('0'),
    }
    rates.update(rate_overrides)
    return PaperTrader(
        market_data,
        NoRiskManager(),
        funded_positions(),
        min_fill_rate=rates['min_fill_rate'],
        max_fill_rate=rates['max_fill_rate'],
        commission_rate=rates['commission_rate'],
    )


@pytest.mark.parametrize('field', ['min_fill_rate', 'max_fill_rate', 'commission_rate'])
@pytest.mark.parametrize('value', ['NaN', 'sNaN', 'Infinity', '-Infinity', '-0.1'])
def test_paper_rejects_invalid_constructor_rates(field: str, value: str):
    with pytest.raises(ValueError):
        paper_trader(**{field: D(value)})


@pytest.mark.parametrize(
    ('minimum', 'maximum'), [('0.8', '0.7'), ('0', '1.1'), ('1.1', '1.2')]
)
def test_paper_rejects_invalid_fill_range(minimum: str, maximum: str):
    with pytest.raises(ValueError, match='Fill rates'):
        paper_trader(min_fill_rate=D(minimum), max_fill_rate=D(maximum))


@pytest.mark.asyncio
@pytest.mark.parametrize('field', ['limit_price', 'quantity'])
@pytest.mark.parametrize('value', ['NaN', 'sNaN', 'Infinity', '-Infinity', '-1', '0'])
async def test_paper_invalid_numeric_cannot_mutate_or_reserve_key(
    field: str, value: str
):
    trader = paper_trader()
    before = deepcopy(trader.position_manager.positions)
    inputs = {'limit_price': D('0.5'), 'quantity': D('1')}
    inputs[field] = D(value)
    result = await trader.place_order(
        TradeSide.BUY, TICKER, **inputs, client_order_id='same-id'
    )
    assert result.failure_reason == OrderFailureReason.INVALID_ORDER
    assert trader.position_manager.positions == before
    assert trader.orders == []
    valid = await trader.place_order(
        TradeSide.BUY, TICKER, D('0.5'), D('1'), client_order_id='same-id'
    )
    assert valid.executed


@pytest.mark.asyncio
async def test_paper_binary_price_boundary_and_invalid_side():
    trader = paper_trader()
    before = deepcopy(trader.position_manager.positions)
    for side, price in [('buy', D('0.5')), (TradeSide.BUY, D('1.01'))]:
        result = await trader.place_order(side, TICKER, price, D('1'))  # type: ignore[arg-type]
        assert result.failure_reason == OrderFailureReason.INVALID_ORDER
        assert trader.position_manager.positions == before
    valid = await trader.place_order(TradeSide.BUY, TICKER, D('1'), D('1'))
    assert valid.executed


@pytest.mark.asyncio
async def test_paper_sell_without_cash_ledger_does_not_mutate_position():
    trader = paper_trader()
    del trader.position_manager.positions[TICKER.collateral.symbol]
    before = deepcopy(trader.position_manager.positions)
    result = await trader.place_order(TradeSide.SELL, TICKER, D('0.4'), D('1'))
    assert result.failure_reason == OrderFailureReason.INSUFFICIENT_CASH
    assert trader.position_manager.positions == before
    assert trader.orders == []


@pytest.mark.asyncio
@pytest.mark.parametrize('field', ['price', 'size'])
@pytest.mark.parametrize('value', ['NaN', 'Infinity', '-1'])
async def test_bad_paper_book_cannot_modify_accounting(field: str, value: str):
    trader = paper_trader()
    level = trader.market_data.order_books[TICKER].asks[0]
    setattr(level, field, D(value))
    before = deepcopy(trader.position_manager.positions)
    result = await trader.place_order(TradeSide.BUY, TICKER, D('0.5'), D('1'))
    assert result.failure_reason == OrderFailureReason.INVALID_ORDER
    assert trader.position_manager.positions == before
    assert trader.orders == []


@pytest.mark.parametrize('use_cash_ticker', [True, False])
def test_cash_or_collateral_symbol_cannot_be_traded(use_cash_ticker: bool):
    positions = funded_positions()
    before = deepcopy(positions.positions)
    ticker = (
        TICKER.collateral
        if use_cash_ticker
        else PolyMarketTicker(TICKER.collateral.symbol)
    )
    with pytest.raises(ValueError):
        positions.apply_trade(Trade(TradeSide.BUY, ticker, D('0.5'), D('1'), D('0')))
    assert positions.positions == before


@pytest.mark.asyncio
async def test_paper_rechecks_balance_after_async_risk_check():
    trader = paper_trader()
    asset_before = deepcopy(trader.position_manager.positions[TICKER.symbol])

    class BalanceChangingRisk(NoRiskManager):
        async def check_trade(self, ticker, side, quantity, price):
            trader.position_manager.positions[TICKER.collateral.symbol].quantity = D(
                '0'
            )
            return True

    trader.risk_manager = BalanceChangingRisk()
    result = await trader.place_order(TradeSide.BUY, TICKER, D('0.5'), D('1'))
    assert result.failure_reason == OrderFailureReason.INVALID_ORDER
    assert trader.position_manager.positions[TICKER.symbol] == asset_before
    assert trader.position_manager.positions[TICKER.collateral.symbol].quantity == 0
    assert trader.orders == []
