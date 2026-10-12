"""All authenticated SDK entry points are mocked; these tests place no live orders."""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from oracle3.data.market_data_manager import MarketDataManager
from oracle3.position.position_manager import Position, PositionManager
from oracle3.risk.risk_manager import NoRiskManager
from oracle3.ticker.ticker import KalshiTicker, PolyMarketTicker
from oracle3.trader.kalshi_trader import KalshiTrader
from oracle3.trader.order_journal import OrderJournal
from oracle3.trader.polymarket_trader import PolymarketTrader
from oracle3.trader.trader import Trader
from oracle3.trader.types import OrderFailureReason, OrderStatus, TradeSide

D = Decimal
K = KalshiTicker(symbol='K', market_ticker='K')
P = PolyMarketTicker(symbol='P', token_id='token')


def trader(kind=KalshiTrader, path=None):
    """Bypass credential constructors; every potential network call is mocked."""
    result = object.__new__(kind)
    positions = PositionManager()
    ticker = K if kind is KalshiTrader else P
    positions.update_position(Position(ticker.collateral, D('100'), D('1'), D('0')))
    Trader.__init__(result, MarketDataManager(), NoRiskManager(), positions)
    result._init_order_journal(path, 'test-account')
    result._kill_switch_active = lambda: False
    result._submit_live_order = AsyncMock(
        return_value=(
            {'order': {'order_id': 'venue'}}
            if kind is KalshiTrader
            else {'success': True, 'orderID': 'venue'}
        )
    )
    result._get_order_status = AsyncMock()
    if kind is PolymarketTrader:
        result._get_order_trades = AsyncMock()
    return result


def kalshi(
    status='executed', filled='10', remaining='0', cost='4', fee='.04', **changes
):
    data = {
        'order_id': 'venue',
        'ticker': 'K',
        'action': 'buy',
        'side': 'yes',
        'status': status,
        'initial_count_fp': '10',
        'fill_count_fp': filled,
        'remaining_count_fp': remaining,
        'maker_fill_cost_dollars': '0',
        'taker_fill_cost_dollars': cost,
        'maker_fees_dollars': '0',
        'taker_fees_dollars': fee,
    }
    data.update(changes)
    return {'order': data}


def poly(status='MATCHED', filled='10', **changes):
    data = {
        'id': 'venue',
        'asset_id': 'token',
        'side': 'BUY',
        'original_size': '10',
        'size_matched': filled,
        'status': status,
        'associate_trades': ['fill'],
    }
    data.update(changes)
    return data


def fill(status='CONFIRMED', fee='0', size='10', price='.4', **changes):
    data = {
        'id': 'fill',
        'taker_order_id': 'venue',
        'asset_id': 'token',
        'side': 'BUY',
        'size': size,
        'price': price,
        'status': status,
        'fee_rate_bps': fee,
    }
    data.update(changes)
    return data


async def buy(t, ticker=K, key='intent'):
    return await t.place_order(TradeSide.BUY, ticker, D('.5'), D('10'), key)


def cash(t, ticker=K):
    return t.position_manager.get_position(ticker.collateral).quantity


@pytest.mark.asyncio
async def test_pending_is_acknowledgement_and_recovery_accounts_only_fill_deltas():
    t = trader()
    t._get_order_status.side_effect = [
        kalshi('resting', '0', '10', '0', '0'),
        kalshi('resting', '4', '6', '1.6', '.04'),
        kalshi('executed', '10', '0', '4.6', '.1'),
    ]
    result = await buy(t)
    assert result.order.status is OrderStatus.PLACED
    assert result.failure_reason is None and result.requires_reconciliation
    assert result.venue_order_id == 'venue' and result.client_order_id == 'intent'
    assert cash(t) == 100 and not t.orders
    blocked = await buy(t, key='another')
    assert blocked.failure_reason is OrderFailureReason.RECONCILIATION_REQUIRED
    (partial,) = await t.reconcile_pending_orders()
    assert partial.order.status is OrderStatus.PARTIALLY_FILLED
    assert cash(t) == D('98.36')
    (full,) = await t.reconcile_pending_orders()
    assert full.order.status is OrderStatus.FILLED and not full.requires_reconciliation
    assert cash(t) == D('95.3')
    assert t.position_manager.get_position(K).quantity == 10
    assert [order.filled_quantity for order in t.orders] == [D('4'), D('6')]
    assert [order.average_price for order in t.orders] == [D('.4'), D('.5')]
    assert [order.commission for order in t.orders] == [D('.04'), D('.06')]
    assert await t.reconcile_pending_orders() == []
    t._submit_live_order.assert_awaited_once()
    json.dumps(full.to_payload(), allow_nan=False)


@pytest.mark.asyncio
async def test_timeout_retry_and_restart_never_resubmit(tmp_path):
    path = tmp_path / 'orders.sqlite3'
    t = trader(path=path)
    t._submit_live_order.side_effect = TimeoutError('ambiguous transport failure')
    first = await buy(t)
    assert first.requires_reconciliation and first.order.status is OrderStatus.UNKNOWN
    assert cash(t) == 100
    duplicate = await buy(t)
    assert duplicate.failure_reason is OrderFailureReason.DUPLICATE_ORDER
    t._submit_live_order.assert_awaited_once()
    t.order_journal.close()
    recovered = trader(path=path)
    duplicate = await buy(recovered)
    assert duplicate.failure_reason is OrderFailureReason.DUPLICATE_ORDER
    results = await recovered.reconcile_pending_orders()
    assert results[0].requires_reconciliation and recovered.reconciliation_required
    recovered._submit_live_order.assert_not_awaited()
    recovered._get_order_status.assert_not_awaited()
    assert not recovered.orders and cash(recovered) == 100


@pytest.mark.asyncio
async def test_restart_does_not_replay_into_unverified_inventory(tmp_path):
    path = tmp_path / 'orders.sqlite3'
    t = trader(path=path)
    t._get_order_status.return_value = kalshi('resting', '4', '6', '1.6', '.04')
    await buy(t)
    t.order_journal.close()
    recovered = trader(path=path)
    recovered._get_order_status.return_value = kalshi()
    (result,) = await recovered.reconcile_pending_orders()
    assert result.requires_reconciliation and not result.order.accounting_confirmed
    assert 'restart inventory' in recovered.reconciliation_reason
    assert cash(recovered) == 100 and not recovered.orders
    assert (
        await buy(recovered, key='new')
    ).failure_reason is OrderFailureReason.RECONCILIATION_REQUIRED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'change',
    [
        {'fill_count_fp': '11'},
        {'fill_count_fp': None},
        {'remaining_count_fp': 'NaN'},
        {'taker_fill_cost_dollars': '15'},
        {'maker_fees_dollars': None},
        {'ticker': 'different'},
        {'side': 'no'},
        {'initial_count_fp': '12'},
        {'taker_fees_dollars': '-.1'},
        {'taker_fill_cost_dollars': 'NaN'},
    ],
)
async def test_kalshi_inconsistent_execution_never_changes_inventory(change):
    t = trader()
    t._get_order_status.return_value = kalshi(**change)
    result = await buy(t)
    assert result.requires_reconciliation and not result.order.accounting_confirmed
    assert cash(t) == 100 and not t.orders
    assert t.position_manager.get_position(K) is None


@pytest.mark.asyncio
async def test_cancelled_kalshi_requires_explicit_fill_count():
    t = trader()
    data = kalshi('canceled', '0', '0', '0', '0')
    del data['order']['fill_count_fp']
    t._get_order_status.return_value = data
    result = await buy(t)
    assert result.requires_reconciliation and cash(t) == 100


@pytest.mark.asyncio
async def test_terminal_zero_fill_rejection_unblocks_but_retains_client_key():
    t = trader()
    t._get_order_status.return_value = kalshi('rejected', '0', '10', '0', '0')
    result = await buy(t)
    assert result.order.status is OrderStatus.REJECTED
    assert not result.requires_reconciliation and not t.reconciliation_required
    assert (await buy(t)).failure_reason is OrderFailureReason.DUPLICATE_ORDER


@pytest.mark.asyncio
async def test_polymarket_partial_fok_does_not_manufacture_price_or_completion():
    t = trader(PolymarketTrader)
    t._get_order_status.return_value = poly(filled='4')
    result = await buy(t, P)
    assert result.order.status is OrderStatus.UNKNOWN
    assert result.requires_reconciliation and cash(t, P) == 100 and not t.orders
    t._get_order_trades.assert_not_awaited()


@pytest.mark.asyncio
async def test_polymarket_awaits_settlement_then_books_actual_weighted_price_once():
    t = trader(PolymarketTrader)
    t._get_order_status.return_value = poly(associate_trades=['fill', 'fill2'])
    t._get_order_trades.side_effect = [
        [fill(status='MINED', size='4'), fill(id='fill2', size='6', price='.45')],
        [fill(size='4'), fill(id='fill2', size='6', price='.45')],
    ]
    result = await buy(t, P)
    assert result.order.filled_quantity == 10 and result.requires_reconciliation
    assert not result.order.accounting_confirmed and cash(t, P) == 100
    (result,) = await t.reconcile_pending_orders()
    assert not result.requires_reconciliation and result.order.average_price == D('.43')
    assert cash(t, P) == D('95.7') and t.position_manager.get_position(P).quantity == 10
    assert len(t.orders) == 1
    assert await t.reconcile_pending_orders() == []


@pytest.mark.asyncio
async def test_polymarket_pending_is_not_rejected():
    t = trader(PolymarketTrader)
    t._get_order_status.return_value = poly('DELAYED', '0', associate_trades=[])
    result = await buy(t, P)
    assert result.order.status is OrderStatus.PLACED
    assert result.failure_reason is None and result.requires_reconciliation
    assert cash(t, P) == 100 and not t.orders


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'change',
    [
        {'fee_rate_bps': '200'},
        {'fee_rate_bps': None},
        {'status': 'FAILED'},
        {'size': '4'},
        {'asset_id': 'different'},
        {'taker_order_id': 'wrong'},
        {'price': '1.25'},
        {'price': 'NaN'},
        {'size': 'Infinity'},
    ],
)
async def test_polymarket_unverified_or_inconsistent_execution_is_not_booked(change):
    t = trader(PolymarketTrader)
    t._get_order_status.return_value = poly()
    t._get_order_trades.return_value = [fill(**change)]
    result = await buy(t, P)
    assert result.requires_reconciliation and not result.order.accounting_confirmed
    assert cash(t, P) == 100 and not t.orders


@pytest.mark.asyncio
@pytest.mark.parametrize('kind,ticker', [(KalshiTrader, K), (PolymarketTrader, P)])
async def test_live_allowlist_and_invalid_numbers_prevent_submission(kind, ticker):
    t = trader(kind)
    t.set_allowed_tickers([])
    assert (
        await buy(t, ticker)
    ).failure_reason is OrderFailureReason.MARKET_NOT_ALLOWED
    t.set_allowed_tickers([ticker])
    for price, size in [
        (D('NaN'), D('10')),
        (D('.5'), D('Infinity')),
        (D('1.1'), D('10')),
        (D('.5'), D('0')),
    ]:
        result = await t.place_order(TradeSide.BUY, ticker, price, size, 'invalid')
        assert result.failure_reason is OrderFailureReason.INVALID_ORDER
    result = await t.place_order('buy', ticker, D('.5'), D('10'), 'invalid')
    assert result.failure_reason is OrderFailureReason.INVALID_ORDER
    t._submit_live_order.assert_not_awaited()
    assert t.order_journal.records() == []


@pytest.mark.asyncio
async def test_kill_switch_rechecked_after_awaited_risk_check():
    t = trader()

    async def risk(*args):
        t.read_only = True
        return True

    t.risk_manager.check_trade = risk
    result = await buy(t)
    assert result.failure_reason is OrderFailureReason.TRADING_DISABLED
    t._submit_live_order.assert_not_awaited()
    assert not t.order_journal.records()


@pytest.mark.asyncio
async def test_journal_write_failure_does_not_commit_staged_positions(monkeypatch):
    t = trader()
    t._get_order_status.return_value = kalshi()
    update = t.order_journal.update

    def broken_update(*args, **kwargs):
        if 'applied_quantity' in kwargs:
            raise OSError('disk full')
        return update(*args, **kwargs)

    monkeypatch.setattr(t.order_journal, 'update', broken_update)
    result = await buy(t)
    assert result.requires_reconciliation and cash(t) == 100 and not t.orders


@pytest.mark.asyncio
async def test_kalshi_submission_forwards_client_key_to_sdk():
    t = trader()
    del t._submit_live_order
    t._submit_order = AsyncMock(return_value={'order': {'order_id': 'venue'}})
    t._get_order_status.return_value = kalshi()
    await buy(t)
    assert t._submit_order.await_args.kwargs['client_order_id'] == 'intent'


@pytest.mark.asyncio
async def test_kalshi_raw_sdk_response_preserves_cost_and_fee_fields():
    t = trader()
    del t._get_order_status
    response = MagicMock()
    response.read.return_value = json.dumps(kalshi()).encode()
    t._portfolio_api = MagicMock()
    t._portfolio_api.get_order_without_preload_content.return_value = response
    result = await buy(t)
    assert result.order.average_price == D('.4') and result.order.commission == D('.04')
    response.release_conn.assert_called_once()


def test_journal_reservation_is_durable_and_namespaced(tmp_path):
    path = tmp_path / 'orders.sqlite3'
    one = OrderJournal(path, 'one')
    two = OrderJournal(path, 'one')
    other = OrderJournal(path, 'other')
    assert one.reserve('same', {'quantity': '10'})
    assert not two.reserve('same', {'quantity': '10'})
    assert other.reserve('same', {'quantity': '10'})
    one.close()
    assert two.get('same')['needs_reconciliation']
    two.close()
    other.close()


@pytest.mark.asyncio
async def test_kalshi_partial_cancel_books_only_confirmed_contracts_and_sell_costs():
    t = trader()
    t._get_order_status.return_value = kalshi('canceled', '4', '0', '1.6', '.04')
    result = await buy(t)
    assert result.order.status is OrderStatus.CANCELLED
    assert not result.requires_reconciliation and cash(t) == D('98.36')
    t._get_order_status.return_value = kalshi(
        'executed', '4', '0', '2.4', '.02', action='sell', initial_count_fp='4'
    )
    result = await t.place_order(TradeSide.SELL, K, D('.5'), D('4'), 'exit')
    assert result.order.average_price == D('.6')
    assert cash(t) == D('100.74')
    assert t.position_manager.get_position(K).quantity == 0


@pytest.mark.asyncio
async def test_polymarket_confirmed_execution_and_sdk_share_units():
    t = trader(PolymarketTrader)
    del t._submit_live_order
    t.clob_client = MagicMock()
    t.clob_client.post_order.return_value = {'success': True, 'orderID': 'venue'}
    t._get_order_status.return_value = poly()
    t._get_order_trades.return_value = [fill()]
    result = await buy(t, P)
    assert result.executed and not result.requires_reconciliation
    args = t.clob_client.create_order.call_args.args[0]
    assert args.size == 10 and args.price == 0.5
    assert cash(t, P) == 96


@pytest.mark.asyncio
async def test_polymarket_unsettled_match_is_not_an_executed_receipt():
    t = trader(PolymarketTrader)
    t._get_order_status.return_value = poly()
    t._get_order_trades.return_value = [fill(status='MATCHED')]
    result = await buy(t, P)
    assert result.order.filled_quantity == 10
    assert not result.executed and not result.to_payload()['ok']
    assert result.requires_reconciliation


@pytest.mark.asyncio
async def test_polymarket_float_precision_loss_is_rejected_before_submission():
    t = trader(PolymarketTrader)
    result = await t.place_order(
        TradeSide.BUY, P, D('.500000000000000000001'), D('10'), 'precision'
    )
    assert result.failure_reason is OrderFailureReason.INVALID_ORDER
    t._submit_live_order.assert_not_awaited()
    assert not t.order_journal.records()
