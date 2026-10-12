"""Fake model decisions reach real paper execution through one explicit path."""

import asyncio
from decimal import Decimal
from types import SimpleNamespace

import pytest

from examples.strategies.openai_related_market_agent_strategy import (
    OpenAIRelatedMarketAgentStrategy,
)
from oracle3.data.market_data_manager import MarketDataManager
from oracle3.events.events import PriceChangeEvent
from oracle3.order.order_book import Level, OrderBook
from oracle3.position.position_manager import Position, PositionManager
from oracle3.risk.risk_manager import NoRiskManager
from oracle3.ticker.ticker import CashTicker, PolyMarketTicker
from oracle3.trader.paper_trader import PaperTrader


@pytest.fixture
def workflow(monkeypatch):
    strategy = OpenAIRelatedMarketAgentStrategy(decision_cooldown_events=1)
    md = MarketDataManager()
    tickers = [PolyMarketTicker(symbol=s, token_id=s) for s in ('A', 'B')]
    for ticker in tickers:
        md.process_price_change_event(PriceChangeEvent(ticker, Decimal('0.52')))
        book = OrderBook()
        book.update(
            bids=[Level(Decimal('0.51'), Decimal('100'))],
            asks=[Level(Decimal('0.52'), Decimal('100'))],
        )
        md.update_order_book(ticker, book)
    pm = PositionManager()
    pm.update_position(
        Position(
            CashTicker.POLYMARKET_USDC, Decimal('10000'), Decimal('1'), Decimal('0')
        )
    )
    trader = PaperTrader(
        market_data=md,
        position_manager=pm,
        risk_manager=NoRiskManager(),
        min_fill_rate=Decimal('1'),
        max_fill_rate=Decimal('1'),
        commission_rate=Decimal('0'),
    )
    event = PriceChangeEvent(tickers[0], Decimal('0.52'))
    context = strategy.bind_context(event, trader)
    state = SimpleNamespace(
        requests=[('A', 'buy', 10.0, 0.52)],
        receipts=[],
        error=False,
        final_output='{"action":"buy","side":"yes","target_symbol":"A"}',
    )

    class FakeAgent:
        def __init__(self, **kwargs):
            self.tools = kwargs['tools']

    class FakeRunner:
        @staticmethod
        async def run(agent, input_text, max_turns):
            place = next(tool for tool in agent.tools if tool.__name__ == 'place_trade')
            state.receipts = [place(*request) for request in state.requests]
            if state.error:
                raise RuntimeError('injected model failure')
            return SimpleNamespace(final_output=state.final_output)

    monkeypatch.setattr(
        'oracle3.strategy.agent_strategy._import_agents_sdk',
        lambda: (FakeAgent, FakeRunner, lambda fn: fn),
    )
    return strategy, trader, event, context, state


async def test_tool_trade_executes_once_despite_duplicate_final_json(workflow):
    strategy, trader, event, context, state = workflow
    await strategy.process_event(event, trader)
    assert 'not submitted or filled' in state.receipts[0]
    assert len(trader.orders) == 1
    assert trader.orders[0].filled_quantity == Decimal('10')
    assert strategy.get_decisions()[-1].executed
    assert strategy.last_trade_outcomes[0].execution.success
    assert await strategy.execute_pending_trades(context=context) == []
    assert len(trader.orders) == 1


@pytest.mark.parametrize('final_output', ['{"action":"hold"}', 'invalid json'])
async def test_queued_trade_does_not_depend_on_final_json(workflow, final_output):
    strategy, trader, event, _, state = workflow
    state.final_output = final_output
    await strategy.process_event(event, trader)
    assert len(trader.orders) == 1
    assert strategy.get_decisions()[-1].executed


async def test_json_only_action_uses_same_execution_path(workflow):
    strategy, trader, event, _, state = workflow
    state.requests = []
    await strategy.process_event(event, trader)
    assert len(trader.orders) == 1
    assert strategy.last_trade_outcomes[0].execution.success


async def test_rejection_is_not_recorded_as_execution_and_stops_queue(workflow):
    strategy, trader, event, _, state = workflow
    trader.set_read_only(True)
    state.requests.append(('B', 'buy', 10.0, 0.52))
    await strategy.process_event(event, trader)
    assert not any(outcome.executed for outcome in strategy.last_trade_outcomes)
    assert 'not submitted' in strategy.last_trade_outcomes[1].execution.failure_reason
    assert not any(decision.executed for decision in strategy.get_decisions())


async def test_actual_partial_fill_remains_incomplete(workflow):
    strategy, trader, event, _, _ = workflow
    book = OrderBook()
    book.update(asks=[Level(Decimal('0.52'), Decimal('5'))], bids=[])
    trader.market_data.update_order_book(event.ticker, book)
    await strategy.process_event(event, trader)
    outcome = strategy.last_trade_outcomes[0]
    assert trader.orders[0].filled_quantity == Decimal('5.0')
    assert outcome.executed and not outcome.execution.success
    assert outcome.execution.requires_reconciliation
    assert strategy.get_decisions()[-1].signal_values['fully_filled'] == 0


async def test_failed_analysis_explicitly_discards_unsubmitted_requests(workflow):
    strategy, trader, event, _, state = workflow
    state.error = True
    await strategy.process_event(event, trader)
    assert trader.orders == []
    assert strategy.take_pending_trades() == []
    assert '1 unsubmitted requests discarded' in strategy.get_decisions()[-1].reasoning


def test_tool_rebuild_does_not_silently_erase_pending_requests(workflow):
    strategy, _, _, context, _ = workflow
    tools = strategy.build_openai_tools(context)
    place = next(tool for tool in tools if tool.__name__ == 'place_trade')
    place('A', 'buy', 10, 0.52)
    with pytest.raises(RuntimeError, match='Unconsumed'):
        strategy.build_openai_tools(context)
    assert len(strategy.take_pending_trades()) == 1
    strategy.build_openai_tools(context)


async def test_cancellation_retains_uncertain_and_unsubmitted_outcomes(
    workflow, monkeypatch
):
    strategy, trader, _, context, _ = workflow
    tools = strategy.build_openai_tools(context)
    place = next(tool for tool in tools if tool.__name__ == 'place_trade')
    place('A', 'buy', 10, 0.52)
    place('B', 'buy', 10, 0.52)

    async def interrupted(**kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(trader, 'place_order', interrupted)
    with pytest.raises(asyncio.CancelledError):
        await strategy.execute_pending_trades(context=context)
    assert strategy.last_trade_outcomes[0].execution.execution_uncertain
    assert strategy.last_trade_outcomes[1].execution.failure_reason == 'not submitted'
    assert strategy.take_pending_trades() == []


@pytest.mark.parametrize(
    'quantity,price',
    [
        (float('nan'), 0.5),
        (float('inf'), 0.5),
        (0, 0.5),
        (1, float('nan')),
        (1, float('inf')),
        (1, 0),
        (1, 1),
        (1, 1.2),
    ],
)
def test_invalid_tool_request_never_enters_queue(workflow, quantity, price):
    strategy, _, _, context, _ = workflow
    tools = strategy.build_openai_tools(context)
    place = next(tool for tool in tools if tool.__name__ == 'place_trade')
    assert 'rejected' in place('A', 'buy', quantity, price)
    assert strategy.take_pending_trades() == []
