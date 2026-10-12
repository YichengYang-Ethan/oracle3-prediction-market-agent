"""Identical offline observations for two unmodified Oracle3 Git snapshots.

This worker replaces only the external Agents SDK/model driver with a scripted
tool call. Strategy, PaperTrader, spread execution and ledger code come from the
selected snapshot. It does not evaluate model reasoning or contact a venue.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import socket
import sys
import tempfile
from decimal import Decimal
from importlib.metadata import distributions
from pathlib import Path
from types import ModuleType, SimpleNamespace


def block_network() -> None:
    def deny(*args, **kwargs):
        raise RuntimeError('Network disabled by runtime regression probe')

    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.socket.sendto = deny
    socket.create_connection = deny
    socket.getaddrinfo = deny


def verify_snapshot_imports(snapshot: Path) -> list[str]:
    """Reject source-tree/editable-install leakage; report relative paths only."""
    paths = []
    for name, module in tuple(sys.modules.items()):
        if name != 'oracle3' and not name.startswith(('oracle3.', 'examples.')):
            continue
        location = getattr(module, '__file__', None)
        if location:
            path = Path(location).resolve()
            if snapshot not in path.parents:
                raise RuntimeError(f'{name} was imported from outside the snapshot')
            paths.append(path.relative_to(snapshot).as_posix())
    if 'oracle3/__init__.py' not in paths:
        raise RuntimeError('The snapshot oracle3 package was not imported')
    return sorted(set(paths))


async def observe_agent_queue() -> dict:
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

    strategy = OpenAIRelatedMarketAgentStrategy(decision_cooldown_events=1)
    market = MarketDataManager()
    tickers = [PolyMarketTicker(symbol=s, token_id=s) for s in ('A', 'B')]
    for ticker in tickers:
        market.process_price_change_event(PriceChangeEvent(ticker, Decimal('0.52')))
        book = OrderBook()
        book.update(
            bids=[Level(Decimal('0.51'), Decimal('100'))],
            asks=[Level(Decimal('0.52'), Decimal('100'))],
        )
        market.update_order_book(ticker, book)
    positions = PositionManager()
    positions.update_position(
        Position(
            CashTicker.POLYMARKET_USDC, Decimal('10000'), Decimal('1'), Decimal('0')
        )
    )
    trader = PaperTrader(
        market_data=market,
        position_manager=positions,
        risk_manager=NoRiskManager(),
        min_fill_rate=Decimal('1'),
        max_fill_rate=Decimal('1'),
        commission_rate=Decimal('0'),
    )
    event = PriceChangeEvent(tickers[0], Decimal('0.52'))
    strategy.bind_context(event, trader)
    receipts = []

    class ScriptedAgent:
        def __init__(self, **kwargs):
            self.tools = kwargs['tools']

    class ScriptedRunner:
        @staticmethod
        async def run(agent, input_text, max_turns):
            place = next(tool for tool in agent.tools if tool.__name__ == 'place_trade')
            receipts.append(place('A', 'buy', 10.0, 0.52))
            return SimpleNamespace(final_output='{"action":"hold"}')

    # Replace the external SDK boundary, never the snapshot's strategy methods.
    sdk = ModuleType('agents')
    sdk.Agent = ScriptedAgent
    sdk.Runner = ScriptedRunner
    sdk.function_tool = lambda function: function
    sys.modules['agents'] = sdk
    await strategy.process_event(event, trader)
    decision = strategy.get_decisions()[-1]
    return {
        'fixture': {
            'tool_request': ['A', 'buy', 10.0, 0.52],
            'final_output': '{"action":"hold"}',
            'paper_cash': 10000,
            'paper_fill_rate': 1,
            'paper_commission_rate': 0,
        },
        'tool_receipts': receipts,
        'orders': [order.to_payload() for order in trader.orders],
        'observed': {
            'order_count': len(trader.orders),
            'filled_quantities': [
                str(order.filled_quantity) for order in trader.orders
            ],
            'decision_action': decision.action,
            'decision_executed': decision.executed,
            'pending_count': len(getattr(strategy, '_pending_trades', [])),
        },
    }


async def observe_spread(responses: list[tuple[str, str]]) -> dict:
    from oracle3.data.market_data_manager import MarketDataManager
    from oracle3.order.order_book import Level, OrderBook
    from oracle3.ticker.ticker import PolyMarketTicker
    from oracle3.trader.spread_executor import SpreadExecutor, SpreadLeg
    from oracle3.trader.types import Order, OrderStatus, PlaceOrderResult, TradeSide

    tickers = {
        s: PolyMarketTicker(symbol=s, token_id=s, market_id=s) for s in ('A', 'B')
    }

    class ScriptedTrader:
        def __init__(self):
            self.responses = iter(responses)
            self.calls = []
            self.market_data = MarketDataManager()
            for ticker in tickers.values():
                book = OrderBook()
                book.update(
                    bids=[Level(Decimal('0.49'), Decimal('100'))],
                    asks=[Level(Decimal('0.50'), Decimal('100'))],
                )
                self.market_data.update_order_book(ticker, book)

        async def place_order(self, *, side, ticker, limit_price, quantity):
            self.calls.append(
                {
                    'side': side.value,
                    'ticker': ticker.symbol,
                    'quantity': str(quantity),
                    'limit_price': str(limit_price),
                }
            )
            status, amount = next(self.responses)
            filled = Decimal(amount)
            return PlaceOrderResult(
                order=Order(
                    status=OrderStatus(status),
                    side=side,
                    ticker=ticker,
                    limit_price=limit_price,
                    filled_quantity=filled,
                    average_price=limit_price,
                    trades=[],
                    remaining=quantity - filled,
                    commission=Decimal('0'),
                )
            )

    trader = ScriptedTrader()
    legs = [
        SpreadLeg(TradeSide.BUY, ticker, Decimal('0.50'), Decimal('10'))
        for ticker in tickers.values()
    ]
    result = await SpreadExecutor(trader).execute_spread(legs, unwind_on_partial=True)
    payload = result.to_payload()
    return {
        'fixture': {'scripted_responses': responses, 'unwind_on_partial': True},
        'calls': trader.calls,
        'result': payload,
        'observed': {
            'call_count': len(trader.calls),
            'unwind_count': len(payload['unwind_results']),
            'success': payload['success'],
            'hedged': payload['hedged'],
            'execution_uncertain': payload['execution_uncertain'],
            'requires_reconciliation': payload['requires_reconciliation'],
            'residual_exposures': {
                item['ticker']['symbol']: item['quantity']
                for item in payload['residual_exposures']
            },
        },
    }


def observe_ledger_writers() -> dict:
    from oracle3.fees import PolymarketSchedule
    from oracle3.mcp_server.paper import PaperLedger

    with tempfile.TemporaryDirectory(prefix='oracle3-replay-ledger-') as directory:
        path = Path(directory) / 'paper.json'
        first, second = PaperLedger(path), PaperLedger(path)
        arguments = {
            'venue': 'polymarket',
            'market_id': 'market-a',
            'side': 'yes',
            'contracts': 1.0,
            'limit_price': 0.6,
            'asks': [(0.5, 100)],
        }
        responses = [
            writer.buy(**arguments, schedule=PolymarketSchedule(rate=Decimal('0')))
            for writer in (first, second)
        ]
        portfolio = PaperLedger(path).portfolio()
        # Exclude only wall-clock timestamps and temporary filesystem locations.
        for response in responses:
            response.pop('time', None)
        portfolio.pop('ledger')
        return {
            'fixture': {
                **arguments,
                'schedule_rate': 0,
                'initial_cash': 10000,
                'sequence': 'construct two instances; buy once each; reload third',
            },
            'responses_without_timestamps': responses,
            'portfolio_without_path': portfolio,
            'observed': {
                'response_statuses': [response['status'] for response in responses],
                'fills': portfolio['fills'],
                'cash': portfolio['cash'],
                'contracts': portfolio['positions'][0]['contracts'],
            },
        }


async def collect(snapshot: Path) -> dict:
    cases = {
        'agent.queued_tool_request': await observe_agent_queue(),
        'execution.open_partial': await observe_spread(
            [
                ('filled', '10'),
                ('partially_filled', '1'),
                ('filled', '1'),
                ('filled', '10'),
            ]
        ),
        'paper.stale_writer': observe_ledger_writers(),
        'control.full_spread': await observe_spread(
            [('filled', '10'), ('filled', '10')]
        ),
    }
    return {
        'cases': cases,
        'provenance': {
            'snapshot_imports_verified': True,
            'snapshot_import_paths': verify_snapshot_imports(snapshot),
            'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'python_version': platform.python_version(),
            'dependencies': dict(
                sorted(
                    (dist.metadata['Name'].lower().replace('_', '-'), dist.version)
                    for dist in distributions()
                    if dist.metadata['Name']
                )
            ),
            'network': 'socket connections and DNS blocked before project imports',
            'model_driver': 'scripted external Agents SDK substitute; no model calls',
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, type=Path)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    if Path.cwd().resolve() != snapshot:
        raise RuntimeError('Worker cwd must be the exported snapshot')
    block_network()
    sys.path.insert(0, str(snapshot))
    print(json.dumps(asyncio.run(collect(snapshot)), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
