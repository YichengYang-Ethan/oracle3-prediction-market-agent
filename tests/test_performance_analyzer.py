"""Accounting oracles: NAV, FIFO closures, costs, marks and binary settlement."""

import json
from decimal import Decimal as D

import pytest

from oracle3.analytics.performance_analyzer import (
    EquityPoint,
    PerformanceAnalyzer,
    TradeStats,
)
from oracle3.storage.serializers import serialize_trade
from oracle3.ticker.ticker import PolyMarketTicker
from oracle3.trader.types import Trade, TradeSide


@pytest.fixture
def ticker():
    return PolyMarketTicker(symbol='A', token_id='a', market_id='market-a')


@pytest.fixture
def other():
    return PolyMarketTicker(symbol='B', token_id='b', market_id='market-b')


@pytest.fixture
def analyzer():
    return PerformanceAnalyzer(D('1000'))


def fill(ticker, side, quantity, price, fee='0'):
    return Trade(
        side=side,
        ticker=ticker,
        quantity=D(quantity),
        price=D(price),
        commission=D(fee),
    )


def assert_reconciles(analyzer):
    stats = analyzer.get_stats()
    assert stats.cash + stats.inventory_value == analyzer.get_current_equity()
    assert stats.realized_pnl + stats.unrealized_pnl == stats.total_pnl
    assert analyzer.initial_capital + stats.total_pnl == analyzer.get_current_equity()


def test_public_result_types_remain_constructible():
    stats = TradeStats(0, 0, 0, D(0), D(0), D(0), D(0), D(0))
    assert stats.total_fills == 0
    assert not stats.sharpe_available
    assert EquityPoint(100, D('1050'), 5).timestamp == 100


def test_empty_portfolio(analyzer):
    stats = analyzer.get_stats()
    assert analyzer.get_current_equity() == D('1000')
    assert analyzer.get_return_pct() == 0
    assert stats.total_trades == stats.total_fills == 0
    assert stats.max_drawdown == stats.total_pnl == 0
    assert_reconciles(analyzer)


def test_opening_position_seed_and_sale_use_existing_basis(analyzer, ticker, other):
    analyzer.seed_position(ticker, D('10'), D('.4'), mark_price=D('.5'))
    analyzer.seed_position(other, D('20'), D('.2'))
    assert analyzer.cash == 992
    assert analyzer.get_current_equity() == 1001
    assert analyzer.inventory_tickers() == {ticker, other}
    stats = analyzer.get_stats()
    assert stats.total_fills == stats.total_trades == stats.total_fees == 0
    assert stats.missing_marks == ('B',)
    analyzer.add_trade(fill(ticker, TradeSide.SELL, '10', '.6', '.1'))
    assert analyzer.get_stats().realized_pnl == D('1.9')
    assert analyzer.inventory_tickers() == {other}
    assert_reconciles(analyzer)
    with pytest.raises(ValueError, match='before other events'):
        analyzer.seed_position(ticker, D('1'), D('.3'))


def test_opening_fee_inclusive_basis_above_one_and_snapshot(analyzer, ticker):
    analyzer.seed_position(ticker, D('10'), D('1.01'))
    assert analyzer.cash == D('989.9')
    assert analyzer.get_current_equity() == 1000
    assert not analyzer.get_stats().valuation_complete
    analyzer.mark_to_market({ticker: D('1')}, timestamp=100)
    assert analyzer.get_stats().unrealized_pnl == D('-.1')
    restored = PerformanceAnalyzer.from_dict(json.loads(json.dumps(analyzer.to_dict())))
    assert restored.get_stats() == analyzer.get_stats()
    assert restored.inventory_tickers() == {ticker}
    assert restored.settle(ticker, 1) == D('-.1')
    assert_reconciles(restored)


def test_invalid_opening_position_is_atomic(analyzer, ticker):
    for quantity, basis, mark in [
        ('0', '.5', '.5'),
        ('1', '-.1', '.5'),
        ('1', '.5', '2'),
    ]:
        before = analyzer.to_dict()
        with pytest.raises(ValueError):
            analyzer.seed_position(ticker, D(quantity), D(basis), D(mark))
        assert analyzer.to_dict() == before


def test_buy_at_unchanged_price_is_inventory_not_loss(analyzer, ticker):
    analyzer.add_trade(fill(ticker, TradeSide.BUY, '10', '.5'))
    stats = analyzer.get_stats()
    assert stats.cash == D('995')
    assert stats.inventory_value == 5
    assert stats.total_pnl == stats.max_drawdown == 0
    assert stats.total_fills == 1
    assert stats.total_trades == stats.losing_trades == 0
    assert analyzer.get_current_equity() == 1000
    assert_reconciles(analyzer)


def test_entry_fee_reduces_nav_without_creating_closed_loss(analyzer, ticker):
    analyzer.add_trade(fill(ticker, TradeSide.BUY, '100', '.5', '.5'))
    stats = analyzer.get_stats()
    assert analyzer.get_current_equity() == D('999.5')
    assert stats.realized_pnl == 0
    assert stats.unrealized_pnl == stats.total_pnl == D('-.5')
    assert stats.total_fees == D('.5')
    assert stats.total_trades == 0
    assert_reconciles(analyzer)


def test_losing_roundtrip_is_one_loss_not_buy_loss_sell_win(analyzer, ticker):
    analyzer.add_trade(fill(ticker, TradeSide.BUY, '10', '.5'))
    analyzer.add_trade(fill(ticker, TradeSide.SELL, '10', '.4'))
    stats = analyzer.get_stats()
    assert stats.total_fills == 2
    assert stats.total_trades == stats.losing_trades == 1
    assert stats.winning_trades == stats.win_rate == 0
    assert stats.average_loss == stats.realized_pnl == stats.total_pnl == -1
    assert stats.unrealized_pnl == stats.inventory_value == 0
    assert_reconciles(analyzer)


def test_partial_exit_allocates_entry_and_exit_fees(analyzer, ticker):
    analyzer.add_trade(fill(ticker, TradeSide.BUY, '100', '.4', '1'))
    analyzer.add_trade(fill(ticker, TradeSide.SELL, '40', '.6', '.4'))
    stats = analyzer.get_stats()
    assert analyzer.position_quantity(ticker) == 60
    assert stats.realized_pnl == D('7.2')
    assert stats.unrealized_pnl == D('11.4')
    assert stats.total_pnl == D('18.6')
    assert stats.total_fees == D('1.4')
    assert analyzer.closed_lots[0].entry_fee == D('.4')
    assert_reconciles(analyzer)


def test_fifo_multiple_entries_and_partial_second_lot(analyzer, ticker):
    analyzer.add_trade(fill(ticker, TradeSide.BUY, '10', '.2', '.1'))
    analyzer.add_trade(fill(ticker, TradeSide.BUY, '20', '.4', '.4'))
    analyzer.add_trade(fill(ticker, TradeSide.SELL, '15', '.6', '.3'))
    assert [lot.quantity for lot in analyzer.closed_lots] == [10, 5]
    assert [lot.pnl for lot in analyzer.closed_lots] == [D('3.7'), D('.8')]
    assert analyzer.get_stats().realized_pnl == D('4.5')
    assert analyzer.get_stats().unrealized_pnl == D('2.7')
    assert analyzer.get_stats().total_trades == 2
    assert analyzer.position_quantity(ticker) == 15
    assert_reconciles(analyzer)


def test_fractional_fee_remainders_reconcile_exactly(analyzer, ticker):
    analyzer.add_trade(fill(ticker, TradeSide.BUY, '3', '.4', '.1'))
    for fee in ('.01', '.02', '.03'):
        analyzer.add_trade(fill(ticker, TradeSide.SELL, '1', '.5', fee))
    assert sum(lot.entry_fee for lot in analyzer.closed_lots) == D('.1')
    assert sum(lot.exit_fee for lot in analyzer.closed_lots) == D('.06')
    assert analyzer.get_stats().realized_pnl == D('.14')
    assert_reconciles(analyzer)


def test_multiple_markets_have_separate_basis(analyzer, ticker, other):
    analyzer.add_trade(fill(ticker, TradeSide.BUY, '10', '.5'))
    analyzer.add_trade(fill(other, TradeSide.BUY, '20', '.2'))
    assert analyzer.mark_to_market({ticker: D('.7'), other: D('.3')}) == 1004
    analyzer.add_trade(fill(ticker, TradeSide.SELL, '10', '.6'))
    assert analyzer.get_stats().realized_pnl == 1
    assert analyzer.get_stats().unrealized_pnl == 2
    assert analyzer.get_current_equity() == 1003
    assert_reconciles(analyzer)


def test_missing_mark_carries_last_observation_and_reports_incomplete(
    analyzer, ticker, other
):
    analyzer.add_trade(fill(ticker, TradeSide.BUY, '10', '.5'))
    analyzer.add_trade(fill(other, TradeSide.BUY, '20', '.2'))
    analyzer.mark_to_market({ticker: D('.7'), other: D('.3')}, timestamp=100)
    assert analyzer.mark_to_market({ticker: D('.8')}, timestamp=200) == 1005
    stats = analyzer.get_stats()
    assert not stats.valuation_complete
    assert stats.missing_marks == ('B',)
    assert analyzer.mark_to_market({}, timestamp=300) == 1005
    assert analyzer.get_stats().missing_marks == ('A', 'B')
    analyzer.mark_to_market({ticker: D('.8'), other: D('.3')}, timestamp=400)
    assert analyzer.get_stats().valuation_complete
    assert_reconciles(analyzer)


def test_missing_first_quote_uses_fill_price_not_zero(analyzer, ticker):
    analyzer.add_trade(fill(ticker, TradeSide.BUY, '10', '.5', '.1'))
    assert analyzer.mark_to_market({}) == D('999.9')
    assert analyzer.get_stats().missing_marks == ('A',)


def test_mark_validation_is_atomic(analyzer, ticker, other):
    analyzer.add_trade(fill(ticker, TradeSide.BUY, '10', '.5'))
    before = analyzer.to_dict()
    with pytest.raises(ValueError):
        analyzer.mark_to_market({ticker: D('.7'), other: D('NaN')})
    assert analyzer.to_dict() == before


@pytest.mark.parametrize('outcome,realized', [(0, D('-5.2')), (1, D('4.8'))])
def test_binary_settlement_realizes_payout_and_fees(
    analyzer, ticker, outcome, realized
):
    analyzer.add_trade(fill(ticker, TradeSide.BUY, '10', '.5', '.1'))
    assert (
        analyzer.settle(ticker, outcome, commission=D('.1'), timestamp=200) == realized
    )
    stats = analyzer.get_stats()
    assert stats.realized_pnl == stats.total_pnl == realized
    assert stats.unrealized_pnl == stats.inventory_value == 0
    assert stats.total_fees == D('.2')
    assert stats.total_trades == stats.total_fills == 1
    assert analyzer.closed_lots[0].reason == 'settlement'
    assert analyzer.closed_lots[0].timestamp == 200
    assert_reconciles(analyzer)


def test_settlement_after_partial_sale(analyzer, ticker):
    analyzer.add_trade(fill(ticker, TradeSide.BUY, '100', '.4', '1'))
    analyzer.add_trade(fill(ticker, TradeSide.SELL, '40', '.6', '.4'))
    assert analyzer.settle(ticker, 1, commission=D('.6')) == D('34.8')
    assert analyzer.get_stats().total_pnl == 42
    assert analyzer.get_stats().total_fees == 2
    assert_reconciles(analyzer)


def test_settlement_repeat_is_idempotent_and_conflict_fails(analyzer, ticker):
    analyzer.add_trade(fill(ticker, TradeSide.BUY, '10', '.5'))
    analyzer.settle(ticker, 1)
    before = analyzer.to_dict()
    assert analyzer.settle(ticker, 1) == 0
    assert analyzer.to_dict() == before
    with pytest.raises(ValueError, match='already settled'):
        analyzer.settle(ticker, 0)
    with pytest.raises(ValueError, match='settled contract'):
        analyzer.add_trade(fill(ticker, TradeSide.BUY, '1', '.2'))
    assert analyzer.to_dict() == before


def test_unbacked_or_oversized_sale_fails_without_mutation(analyzer, ticker):
    with pytest.raises(ValueError, match='inventory'):
        analyzer.add_trade(fill(ticker, TradeSide.SELL, '1', '.5'))
    analyzer.add_trade(fill(ticker, TradeSide.BUY, '2', '.5'))
    before = analyzer.to_dict()
    with pytest.raises(ValueError, match='inventory'):
        analyzer.add_trade(fill(ticker, TradeSide.SELL, '3', '.5'))
    assert analyzer.to_dict() == before


@pytest.mark.parametrize(
    'quantity,price,fee',
    [
        ('0', '.5', '0'),
        ('-1', '.5', '0'),
        ('1', '-.1', '0'),
        ('1', '1.1', '0'),
        ('1', 'NaN', '0'),
        ('Infinity', '.5', '0'),
        ('1', '.5', '-.1'),
        ('1', '.5', 'NaN'),
    ],
)
def test_invalid_fill_is_atomic(analyzer, ticker, quantity, price, fee):
    before = analyzer.to_dict()
    with pytest.raises(ValueError):
        analyzer.add_trade(fill(ticker, TradeSide.BUY, quantity, price, fee))
    assert analyzer.to_dict() == before


def test_drawdown_is_marked_nav_not_purchase_outflow(analyzer, ticker):
    analyzer.add_trade(fill(ticker, TradeSide.BUY, '100', '.5'))
    analyzer.mark_to_market({ticker: D('.8')}, timestamp=100)
    analyzer.mark_to_market({ticker: D('.3')}, timestamp=200)
    assert analyzer.get_stats().max_drawdown == D('50') / D('1030')


def test_win_rate_profit_factor_and_streaks_use_closed_lots(analyzer, ticker):
    for exit_price in ('.7', '.7', '.4', '.4', '.6'):
        analyzer.add_trade(fill(ticker, TradeSide.BUY, '10', '.5'))
        analyzer.add_trade(fill(ticker, TradeSide.SELL, '10', exit_price))
    stats = analyzer.get_stats()
    assert stats.total_trades == 5
    assert stats.total_fills == 10
    assert stats.winning_trades == 3
    assert stats.losing_trades == 2
    assert stats.win_rate == D('.6')
    assert stats.profit_factor == D('2.5')
    assert stats.max_consecutive_wins == stats.max_consecutive_losses == 2
    assert_reconciles(analyzer)


def test_no_fake_sharpe_or_synthetic_daily_timestamps(analyzer, ticker):
    analyzer.add_trade(fill(ticker, TradeSide.BUY, '10', '.5'))
    analyzer.add_trade(fill(ticker, TradeSide.SELL, '10', '.7'))
    assert all(point.timestamp == 0 for point in analyzer.get_equity_curve())
    stats = analyzer.get_stats()
    assert stats.sharpe_ratio == 0
    assert not stats.sharpe_available
    assert 'regularly sampled' in stats.sharpe_reason
    analyzer.mark_to_market({}, timestamp=1000)
    assert analyzer.get_equity_curve()[-1].timestamp == 1000
    assert not analyzer.get_stats().sharpe_available


def test_versioned_snapshot_replays_marks_settlement_and_costs(analyzer, ticker, other):
    analyzer.add_trade(fill(ticker, TradeSide.BUY, '10', '.5', '.1'), timestamp=100)
    analyzer.add_trade(fill(other, TradeSide.BUY, '20', '.2', '.2'), timestamp=101)
    analyzer.mark_to_market({ticker: D('.7'), other: D('.3')}, timestamp=200)
    analyzer.add_trade(fill(ticker, TradeSide.SELL, '4', '.8', '.04'), timestamp=300)
    analyzer.settle(ticker, 1, commission=D('.06'), timestamp=400)
    analyzer.mark_to_market({}, timestamp=500)
    restored = PerformanceAnalyzer.from_dict(json.loads(json.dumps(analyzer.to_dict())))
    assert restored.get_stats() == analyzer.get_stats()
    assert restored.get_equity_curve() == analyzer.get_equity_curve()
    assert restored.closed_lots == analyzer.closed_lots
    assert restored.to_dict() == analyzer.to_dict()
    assert_reconciles(restored)


def test_legacy_snapshot_replays_fills_not_broken_cash_curve(ticker):
    old = {
        'initial_capital': '1000',
        'trades': [serialize_trade(fill(ticker, TradeSide.BUY, '10', '.5'))],
        'equity_curve': [{'timestamp': 1, 'equity': '995', 'trade_index': 0}],
    }
    restored = PerformanceAnalyzer.from_dict(old)
    assert restored.get_current_equity() == 1000
    assert restored.get_stats().total_pnl == 0


def test_legacy_unbacked_sale_cannot_invent_cost_basis(ticker):
    old = {
        'initial_capital': '1000',
        'trades': [serialize_trade(fill(ticker, TradeSide.SELL, '10', '.5'))],
    }
    with pytest.raises(ValueError, match='inventory'):
        PerformanceAnalyzer.from_dict(old)


def test_reset_and_summary(analyzer, ticker, capsys):
    analyzer.add_trade(fill(ticker, TradeSide.BUY, '10', '.5'))
    analyzer.mark_to_market({})
    analyzer.print_summary()
    printed = capsys.readouterr().out
    assert 'PERFORMANCE SUMMARY' in printed
    assert 'closed FIFO lots' in printed
    assert 'Sharpe Ratio: unavailable' in printed
    assert 'Carried marks: A' in printed
    analyzer.reset()
    assert analyzer.get_current_equity() == 1000
    assert analyzer.trades == analyzer.closed_lots == []
    assert len(analyzer.get_equity_curve()) == 1
    assert_reconciles(analyzer)
