"""Offline regression cases for the tool behavior visible to an agent.

This runs the real MCP dispatch and spread executor against frozen mocks. It
does not call an LLM, contact an exchange, or place live orders.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import socket
import subprocess
from collections.abc import Awaitable, Callable
from decimal import Decimal
from importlib.metadata import distributions, version
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import httpx

import oracle3
from oracle3 import __version__
from oracle3.data.market_data_manager import MarketDataManager
from oracle3.mcp_server import venues
from oracle3.mcp_server.server import mcp
from oracle3.order.order_book import Level, OrderBook
from oracle3.ticker.ticker import PolyMarketTicker
from oracle3.trader.spread_executor import SpreadExecutor, SpreadLeg
from oracle3.trader.trader import Trader
from oracle3.trader.types import (
    Order,
    OrderFailureReason,
    OrderStatus,
    PlaceOrderResult,
    TradeSide,
)

Case = Callable[[], Awaitable[dict[str, Any]]]


def _require(condition: bool, message: str) -> None:
    # Do not use assert: these checks must also run under python -O.
    if not condition:
        raise AssertionError(message)


async def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Normalize both supported MCP SDK dispatch result shapes."""
    try:
        result = await mcp.call_tool(name, arguments)
    except Exception as exc:
        return {'is_error': True, 'message': str(exc)}
    if isinstance(result, tuple):
        content, structured = result
        is_error = False
    else:
        content = getattr(result, 'content', result)
        structured = getattr(result, 'structured_content', None)
        if structured is None:
            structured = getattr(result, 'structuredContent', None)
        is_error = bool(
            getattr(result, 'is_error', False) or getattr(result, 'isError', False)
        )
    message = '\n'.join(getattr(block, 'text', '') for block in content)
    if is_error:
        return {'is_error': True, 'message': message}
    data = structured if structured is not None else json.loads(message)
    if isinstance(data, dict) and set(data) == {'result'}:
        data = data['result']
    return {'is_error': False, 'data': data}


async def _invalid_quantity() -> dict[str, Any]:
    outcome = await _call_tool(
        'trading_fee', {'venue': 'kalshi', 'price': 0.5, 'contracts': -1}
    )
    _require(outcome['is_error'], 'Negative contract count was accepted')
    _require(
        any(word in outcome['message'].lower() for word in ('contracts', 'quantity')),
        'Input error did not identify the invalid quantity',
    )
    return {'input': {'contracts': -1}, 'outcome': outcome}


async def _invalid_relation() -> dict[str, Any]:
    outcome = await _call_tool(
        'check_constraint', {'relation': 'unknown', 'quotes': []}
    )
    _require(outcome['is_error'], 'Unknown relation was accepted')
    _require('relation' in outcome['message'].lower(), 'Error omitted relation field')
    return {'input': {'relation': 'unknown'}, 'outcome': outcome}


async def _missing_market() -> dict[str, Any]:
    outcome = await _call_tool(
        'get_market', {'venue': 'kalshi', 'market_id': 'FROZEN-MISSING'}
    )
    _require(outcome['is_error'], 'Missing market appeared successful')
    _require('HTTP 404' in outcome['message'], 'Venue HTTP status was lost')
    return {'fixture_http_status': 404, 'outcome': outcome}


async def _fee_eliminates_edge() -> dict[str, Any]:
    outcome = await _call_tool(
        'check_constraint',
        {
            'relation': 'implication',
            'quotes': [
                {'market_id': 'A', 'yes_bid': 0.51},
                {'market_id': 'B', 'yes_ask': 0.49},
            ],
            'contracts': 1,
        },
    )
    _require(not outcome['is_error'], 'Valid fee calculation failed')
    data = outcome['data']
    _require(data['violated'], 'Frozen two-cent gross edge disappeared')
    _require(not data['profitable_after_fees'], 'Fees did not remove thin edge')
    _require(data['best']['net_edge'] < 0, 'Net edge should be negative')
    return {'outcome': outcome}


async def _relation_assumption() -> dict[str, Any]:
    # A third possible outcome is deliberately omitted. The tool cannot infer
    # that from prices: its reported payoff is conditional on the caller's label.
    outcome = await _call_tool(
        'check_constraint',
        {
            'relation': 'event_sum',
            'quotes': [
                {'market_id': 'A', 'yes_ask': 0.2},
                {'market_id': 'B', 'yes_ask': 0.3},
            ],
        },
    )
    _require(not outcome['is_error'], 'Conditional static calculation failed')
    data = outcome['data']
    _require(data['profitable_after_fees'], 'Frozen conditional edge disappeared')
    _require(
        data.get('assessment')
        == {
            'relation_verified': False,
            'execution_verified': False,
            'conditional_on_relation': True,
        },
        'Static result did not expose its unverified scope',
    )
    _require(
        any('relation is specified correctly' in item for item in data['assumptions']),
        'Output omitted the unverified relation assumption',
    )
    return {
        'outcome': outcome,
        'fixture_omitted_outcome': 'C',
        'basket_payoff_if_C_occurs': 0,
        'interpretation': (
            'A positive check is conditional on the supplied relation. This case '
            'passes when that limitation is visible; it does not certify the relation.'
        ),
    }


def _ticker(symbol: str) -> PolyMarketTicker:
    return PolyMarketTicker(symbol=symbol, token_id=symbol, market_id=symbol)


class _ScriptedTrader:
    """A fixed sequence of exchange responses; no authenticated trader exists."""

    def __init__(self, responses: list[tuple[str, str]]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict[str, str]] = []
        self.market_data = MarketDataManager()
        for symbol in ('A', 'B'):
            book = OrderBook()
            book.update(
                bids=[Level(price=Decimal('0.49'), size=Decimal('100'))],
                asks=[Level(price=Decimal('0.50'), size=Decimal('100'))],
            )
            self.market_data.update_order_book(_ticker(symbol), book)

    async def place_order(
        self,
        *,
        side: TradeSide,
        ticker: PolyMarketTicker,
        limit_price: Decimal,
        quantity: Decimal,
    ) -> PlaceOrderResult:
        self.calls.append(
            {
                'side': side.value,
                'ticker': ticker.symbol,
                'quantity': str(quantity),
                'limit_price': str(limit_price),
            }
        )
        status, amount = next(self.responses)
        if status == 'rejected':
            return PlaceOrderResult(failure_reason=OrderFailureReason.INVALID_ORDER)
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


async def _execute(
    responses: list[tuple[str, str]],
    *,
    unwind: bool,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    trader = _ScriptedTrader(responses)
    legs = [
        SpreadLeg(TradeSide.BUY, _ticker(symbol), Decimal('0.50'), Decimal('10'))
        for symbol in ('A', 'B')
    ]
    result = await SpreadExecutor(cast(Trader, trader)).execute_spread(
        legs,
        unwind_on_partial=unwind,
    )
    return result.to_payload(), trader.calls


async def _full_fill() -> dict[str, Any]:
    payload, calls = await _execute([('filled', '10'), ('filled', '10')], unwind=False)
    _require(payload['success'] and payload['all_filled'], 'Exact fills not recognized')
    _require(not payload['execution_uncertain'], 'Terminal full fills marked uncertain')
    _require(len(calls) == 2, 'Unexpected execution calls')
    return {'result': payload, 'simulated_calls': calls}


async def _partial_fill() -> dict[str, Any]:
    payload, calls = await _execute(
        [('filled', '10'), ('cancelled', '1')],
        unwind=False,
    )
    _require(not payload['success'], 'Partial basket reported success')
    _require(not payload['all_filled'], 'One of ten contracts counted as full fill')
    _require(not payload['hedged'], 'Unwound exposure incorrectly marked hedged')
    _require(
        {
            item['ticker']['symbol']: item['quantity']
            for item in payload['residual_exposures']
        }
        == {'A': '10', 'B': '1'},
        'Partial fill did not report exact known residual quantities',
    )
    _require(not payload['execution_uncertain'], 'Terminal partial marked unresolved')
    _require(len(calls) == 2, 'Unrequested unwind was attempted')
    return {'result': payload, 'simulated_calls': calls}


async def _failed_unwind() -> dict[str, Any]:
    payload, calls = await _execute(
        [('filled', '10'), ('rejected', '0'), ('rejected', '0')],
        unwind=True,
    )
    _require(not payload['success'], 'Rejected second leg reported success')
    _require(not payload['hedged'], 'Rejected unwind incorrectly marked hedged')
    _require(
        {
            item['ticker']['symbol']: item['quantity']
            for item in payload['residual_exposures']
        }
        == {'A': '10'},
        'Failed unwind lost known exposure',
    )
    _require(len(calls) == 3 and calls[-1]['side'] == 'sell', 'Expected reverse order')
    return {'result': payload, 'simulated_calls': calls}


async def _unresolved_partial() -> dict[str, Any]:
    payload, calls = await _execute(
        [
            ('filled', '10'),
            ('partially_filled', '1'),
        ],
        unwind=True,
    )
    _require(not payload['success'], 'Open partial order reported success')
    _require(not payload['hedged'], 'Unknown remaining execution reported hedged')
    _require(payload['execution_uncertain'], 'Open remainder omitted uncertainty')
    _require(payload['requires_reconciliation'], 'Open remainder needs reconciliation')
    _require(
        {
            item['ticker']['symbol']: item['quantity']
            for item in payload['residual_exposures']
        }
        == {'A': '10', 'B': '1'},
        'Open remainder must retain all known exposure for reconciliation',
    )
    _require(not payload['unwind_results'], 'Uncertain execution must not auto-unwind')
    _require(len(calls) == 2, 'No further orders should follow uncertain execution')
    return {'result': payload, 'simulated_calls': calls}


CASES: tuple[tuple[str, Case], ...] = (
    ('mcp.reject_negative_quantity', _invalid_quantity),
    ('mcp.reject_unknown_relation', _invalid_relation),
    ('mcp.preserve_venue_error', _missing_market),
    ('mcp.fees_remove_thin_edge', _fee_eliminates_edge),
    ('relation.conditional_not_verified', _relation_assumption),
    ('execution.exact_full_fill', _full_fill),
    ('execution.partial_fill_is_incomplete', _partial_fill),
    ('execution.failed_unwind_keeps_exposure', _failed_unwind),
    ('execution.open_remainder_requires_reconciliation', _unresolved_partial),
)


def _deny_network(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError('Network is disabled by the deterministic contract runner')


def _source_fingerprint(package_root: Path, runner: Path) -> str:
    """Hash executed package sources plus fixtures, independent of checkout path."""
    digest = hashlib.sha256()
    files = [
        (f'oracle3/{path.relative_to(package_root).as_posix()}', path)
        for path in package_root.rglob('*.py')
    ]
    files.append(('scripts/check_agent_contracts.py', runner))
    for label, path in sorted(files):
        digest.update(label.encode('utf-8') + b'\0')
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _provenance() -> dict[str, Any]:
    runner = Path(__file__).resolve()
    package_root = Path(oracle3.__file__).resolve().parent
    revision = None
    dirty = None
    # A repository revision only applies if this runner imported its own source.
    if package_root == runner.parent.parent / 'oracle3':
        try:
            revision = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=runner.parent.parent,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout.strip()
            dirty = bool(
                subprocess.run(
                    ['git', 'status', '--porcelain', '--untracked-files=normal'],
                    cwd=runner.parent.parent,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                ).stdout.strip()
            )
        except (OSError, subprocess.SubprocessError):
            revision = None
    dependencies = {
        dist.metadata['Name'].lower().replace('_', '-'): dist.version
        for dist in distributions()
        if dist.metadata['Name']
    }
    return {
        'source_git_revision': revision,
        'source_git_dirty': dirty,
        'source_sha256': _source_fingerprint(package_root, runner),
        'source_hash_scope': 'oracle3 Python sources and this runner with its fixtures',
        'python_version': platform.python_version(),
        'dependencies': dict(sorted(dependencies.items())),
    }


async def run_suite() -> dict[str, Any]:
    """Return stable JSON evidence; any attempted socket connection fails closed."""
    results = []
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            404,
            json={'error': 'frozen missing market'},
            request=request,
        )
    )
    previous_transport = venues._transport
    venues.set_transport(transport)
    try:
        with (
            patch.object(socket.socket, 'connect', _deny_network),
            patch.object(socket.socket, 'connect_ex', _deny_network),
            patch.object(socket.socket, 'sendto', _deny_network),
            patch.object(socket, 'create_connection', _deny_network),
            patch.object(socket, 'getaddrinfo', _deny_network),
        ):
            for case_id, case in CASES:
                try:
                    evidence = await case()
                    results.append(
                        {'case_id': case_id, 'status': 'passed', 'evidence': evidence}
                    )
                except Exception as exc:
                    results.append(
                        {
                            'case_id': case_id,
                            'status': 'failed',
                            'error': {'type': type(exc).__name__, 'message': str(exc)},
                        }
                    )
    finally:
        venues.set_transport(previous_transport)
    failed = sum(item['status'] == 'failed' for item in results)
    return {
        'schema_version': 2,
        'evaluation': 'deterministic_tool_contract_regression',
        'fixture_version': '2',
        'oracle3_version': __version__,
        'mcp_version': version('mcp'),
        'deterministic': True,
        'determinism_scope': 'Fixed fixtures with identical source and dependencies',
        'provenance': _provenance(),
        'scope': 'Tool and execution behavior only; no LLM or skill efficacy evaluation.',
        'network': 'disabled',
        'summary': {
            'total': len(results),
            'passed': len(results) - failed,
            'failed': failed,
        },
        'cases': results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Write JSON here instead of stdout')
    args = parser.parse_args(argv)
    report = asyncio.run(run_suite())
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(encoded, encoding='utf-8')
    else:
        print(encoded, end='')
    return 1 if report['summary']['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
