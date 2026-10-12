"""Local, transactionally updated JSON paper ledger for the MCP server.

Orders walk a displayed ask snapshot; nothing is sent to a venue. Positions
are carried at cost, with no resolution or mark-to-market accounting. A
sidecar lock serializes cooperating processes on a local filesystem. Atomic
replacement prevents partially written ledgers; valid legacy JSON ledgers
gain request history on their next write. Corrupt ledgers fail closed.

The ledger lives at ``~/.oracle3/mcp_paper_ledger.json`` unless
``ORACLE3_MCP_LEDGER`` points elsewhere. Optional request IDs deduplicate
completed results until the ledger is reset, including rejected/unfilled buys.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from oracle3.fees import FeeSchedule

DEFAULT_INITIAL_CASH = 10_000.0


def ledger_path() -> Path:
    override = os.environ.get('ORACLE3_MCP_LEDGER')
    if override:
        return Path(override).expanduser()
    return Path.home() / '.oracle3' / 'mcp_paper_ledger.json'


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f'{name} must be a finite number')
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not math.isfinite(result):
        raise ValueError(f'{name} must be a finite number')
    if result < 0 or (positive and result == 0):
        requirement = 'positive' if positive else 'nonnegative'
        raise ValueError(f'{name} must be {requirement}')
    return result


def _identity(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonblank string')
    return value


def _price(value: Any, name: str, *, executable: bool = False) -> float:
    result = _number(value, name)
    if result > 1 or (executable and not 0 < result < 1):
        domain = 'strictly between 0 and 1' if executable else 'between 0 and 1'
        raise ValueError(f'{name} must be {domain}')
    return result


def _strict_constant(value: str) -> Any:
    raise ValueError(f'nonfinite JSON number: {value}')


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _finite_tree(value: Any) -> None:
    """Also reject overflow such as JSON 1e999 in unknown/additive fields."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('ledger contains a nonfinite number')
    if isinstance(value, dict):
        for child in value.values():
            _finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            _finite_tree(child)


def _validate_position(key: str, position: Any) -> None:
    if not isinstance(position, dict):
        raise ValueError('position must be an object')
    venue = _identity(position['venue'], 'position venue')
    market_id = _identity(position['market_id'], 'position market_id')
    side = position['side']
    if side not in ('yes', 'no') or key != f'{venue}:{market_id}:{side}':
        raise ValueError('position identity does not match its key')
    for field in ('contracts', 'cost', 'fees'):
        _number(position[field], f'position {field}')


def _validate_fill(fill: Any) -> None:
    if not isinstance(fill, dict):
        raise ValueError('fill must be an object')
    PaperLedger.order_intent(
        venue=fill['venue'],
        market_id=fill['market_id'],
        side=fill['side'],
        contracts=fill['requested'],
        limit_price=fill['limit_price'],
    )
    _identity(fill['time'], 'fill time')
    filled = _number(fill['filled'], 'filled contracts', positive=True)
    if filled > fill['requested']:
        raise ValueError('filled contracts exceed requested contracts')
    _price(fill['average_price'], 'average_price')
    for field in ('cost', 'fees'):
        _number(fill[field], f'fill {field}')
    if not isinstance(fill['levels'], list) or not fill['levels']:
        raise ValueError('fill levels must be a nonempty list')
    for level in fill['levels']:
        _price(level['price'], 'level price', executable=True)
        _number(level['contracts'], 'level contracts', positive=True)
        _number(level['fee'], 'level fee')
        if level['price'] > fill['limit_price']:
            raise ValueError('fill price exceeds its limit')
    totals = {
        'filled': sum(level['contracts'] for level in fill['levels']),
        'cost': round(
            sum(level['price'] * level['contracts'] for level in fill['levels']), 6
        ),
        'fees': round(sum(level['fee'] for level in fill['levels']), 6),
        'average_price': round(fill['cost'] / filled, 6),
    }
    for field, expected in totals.items():
        _assert_close(fill[field], expected, f'fill {field}')


def _assert_close(
    actual: float, expected: float, name: str, tolerance: float = 1e-6
) -> None:
    if not math.isfinite(expected) or not math.isclose(
        actual, expected, rel_tol=1e-12, abs_tol=tolerance
    ):
        raise ValueError(f'{name} does not agree with recorded fills')


def _validate_accounting(data: dict[str, Any]) -> None:
    totals: dict[str, dict[str, float]] = {}
    for fill in data['fills']:
        key = f'{fill["venue"]}:{fill["market_id"]}:{fill["side"]}'
        total = totals.setdefault(key, {'contracts': 0.0, 'cost': 0.0, 'fees': 0.0})
        for field, source in (
            ('contracts', 'filled'),
            ('cost', 'cost'),
            ('fees', 'fees'),
        ):
            total[field] += fill[source]
    if totals.keys() != data['positions'].keys():
        raise ValueError('positions do not agree with recorded fills')
    # Legacy ledgers round each position/cash update to six decimal places.
    tolerance = 1e-6 * (len(data['fills']) + 1)
    for key, total in totals.items():
        for field, expected in total.items():
            _assert_close(
                data['positions'][key][field], expected, f'position {field}', tolerance
            )
    spent = sum(total['cost'] + total['fees'] for total in totals.values())
    _assert_close(data['cash'], data['initial_cash'] - spent, 'cash', tolerance)


def _validate_requests(requests: Any) -> None:
    if not isinstance(requests, dict):
        raise ValueError('requests must be an object')
    for request_id, record in requests.items():
        _identity(request_id, 'request_id')
        if not isinstance(record, dict) or not isinstance(record['intent'], dict):
            raise ValueError('request record and intent must be objects')
        intent = PaperLedger.order_intent(**record['intent'])
        result = record['result']
        if not isinstance(result, dict):
            raise ValueError('request result must be an object')
        if result['status'] in ('filled', 'partially_filled'):
            _validate_fill(result)
            for field in ('venue', 'market_id', 'side', 'limit_price'):
                if result[field] != intent[field]:
                    raise ValueError('request result does not match its intent')
            if result['requested'] != intent['contracts']:
                raise ValueError('request quantity does not match its intent')
        elif result['status'] in ('unfilled', 'rejected'):
            _identity(result['reason'], 'request result reason')
        else:
            raise ValueError('invalid request result status')


def _validate_state(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError('ledger must be an object')
    _finite_tree(data)
    version = data.get('schema_version', 1)
    if type(version) is not int or version != 1:
        raise ValueError('unsupported ledger schema_version')
    _number(data['initial_cash'], 'initial_cash')
    _number(data['cash'], 'cash')
    if not isinstance(data['positions'], dict) or not isinstance(data['fills'], list):
        raise ValueError('positions must be an object and fills must be a list')
    for key, position in data['positions'].items():
        _validate_position(key, position)
    for fill in data['fills']:
        _validate_fill(fill)
    _validate_accounting(data)
    _validate_requests(data.get('requests', {}))
    data.setdefault('schema_version', 1)
    data.setdefault('requests', {})
    return data


class PaperLedger:
    def __init__(
        self, path: Path | None = None, initial_cash: float = DEFAULT_INITIAL_CASH
    ) -> None:
        # Resolve aliases so cooperating callers lock the same ledger sidecar.
        self.path = (path or ledger_path()).expanduser().resolve()
        self.initial_cash = _number(initial_cash, 'initial_cash')
        with self._locked():
            self.state = self._load()

    @staticmethod
    def order_intent(
        *, venue: str, market_id: str, side: str, contracts: float, limit_price: float
    ) -> dict[str, Any]:
        """Validate before HTTP or disk mutation; also define retry identity."""
        if side not in ('yes', 'no'):
            raise ValueError("side must be 'yes' or 'no'")
        return {
            'venue': _identity(venue, 'venue'),
            'market_id': _identity(market_id, 'market_id'),
            'side': side,
            'contracts': _number(contracts, 'contracts', positive=True),
            'limit_price': _price(limit_price, 'limit_price', executable=True),
        }

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_name(self.path.name + '.lock'), 'a+b') as lock:
            if sys.platform == 'win32':  # pragma: no cover - Windows primitive
                import msvcrt

                lock.seek(0, os.SEEK_END)
                if lock.tell() == 0:
                    lock.write(b'\0')
                    lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if sys.platform == 'win32':  # pragma: no cover
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _empty(self) -> dict[str, Any]:
        return {
            'schema_version': 1,
            'initial_cash': self.initial_cash,
            'cash': self.initial_cash,
            'positions': {},
            'fills': [],
            'requests': {},
        }

    def _load(self) -> dict[str, Any]:
        try:
            with open(self.path) as stream:
                data = json.load(
                    stream,
                    parse_constant=_strict_constant,
                    object_pairs_hook=_unique_object,
                )
            return _validate_state(data)
        except FileNotFoundError:
            return self._empty()
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            raise ValueError(
                f'Invalid paper ledger at {self.path}: {exc}. '
                'The file was preserved; restore a valid backup before retrying.'
            ) from exc

    def _save(self, state: dict[str, Any]) -> None:
        _validate_state(state)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w',
                dir=self.path.parent,
                prefix=f'.{self.path.name}.',
                suffix='.tmp',
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(state, stream, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self.state = state

    def _cached(
        self, state: dict[str, Any], request_id: str | None, intent: dict[str, Any]
    ) -> dict[str, Any] | None:
        if request_id is None:
            return None
        _identity(request_id, 'request_id')
        record = state['requests'].get(request_id)
        if record is None:
            return None
        if record['intent'] != intent:
            raise ValueError('request_id already used for a different paper order')
        return copy.deepcopy(record['result'])

    def prior_result(
        self, request_id: str | None, intent: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Avoid repeat HTTP for completed requests; buy rechecks under its lock."""
        with self._locked():
            self.state = self._load()
            return self._cached(self.state, request_id, intent)

    def reset(self) -> dict[str, Any]:
        with self._locked():
            self._load()  # Do not overwrite a corrupt ledger, even during reset.
            self._save(self._empty())
            return self._portfolio()

    def buy(
        self,
        *,
        venue: str,
        market_id: str,
        side: str,
        contracts: float,
        limit_price: float,
        asks: list[tuple[float, float]],
        schedule: FeeSchedule,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Fill a buy atomically; the same request ID and intent returns its result."""
        intent = self.order_intent(
            venue=venue,
            market_id=market_id,
            side=side,
            contracts=contracts,
            limit_price=limit_price,
        )
        if request_id is not None:
            _identity(request_id, 'request_id')
        with self._locked():
            state = self._load()
            cached = self._cached(state, request_id, intent)
            if cached is not None:
                self.state = state
                return cached
            result = self._fill(state, intent, asks, schedule)
            if request_id is not None:
                state['requests'][request_id] = {'intent': intent, 'result': result}
            if request_id is not None or result['status'] in (
                'filled',
                'partially_filled',
            ):
                self._save(state)
            else:
                self.state = state
            return copy.deepcopy(result)

    @staticmethod
    def _fill(
        state: dict[str, Any],
        intent: dict[str, Any],
        asks: list[tuple[float, float]],
        schedule: FeeSchedule,
    ) -> dict[str, Any]:
        levels = [
            (_price(price, 'ask price'), _number(size, 'ask size'))
            for price, size in asks
        ]
        remaining = intent['contracts']
        fills: list[dict[str, float]] = []
        for price, size in sorted(levels):
            if remaining <= 0 or price > intent['limit_price']:
                break
            if size == 0:
                continue
            _price(price, 'executable ask price', executable=True)
            qty = min(size, remaining)
            fee = _number(schedule.fee(price, Decimal(str(qty))), 'fill fee')
            fills.append({'price': price, 'contracts': qty, 'fee': fee})
            remaining -= qty
        # Subtracting a small fill from a very large request can round back to
        # the request itself. Sum actual fills instead of inferring them from it.
        filled = sum(level['contracts'] for level in fills)
        if filled <= 0:
            return {
                'status': 'unfilled',
                'reason': f'no displayed ask at or below {intent["limit_price"]}',
                'best_ask': min((p for p, _ in levels), default=None),
            }
        cost = _number(
            round(sum(f['price'] * f['contracts'] for f in fills), 6), 'cost'
        )
        fees = _number(round(sum(f['fee'] for f in fills), 6), 'fees')
        if cost + fees > state['cash'] + 1e-9:
            return {
                'status': 'rejected',
                'reason': f'insufficient paper cash: need {cost + fees:.4f}, have {state["cash"]:.4f}',
            }
        return PaperLedger._record_fill(state, intent, filled, cost, fees, fills)

    @staticmethod
    def _record_fill(
        state: dict[str, Any],
        intent: dict[str, Any],
        filled: float,
        cost: float,
        fees: float,
        fills: list[dict[str, float]],
    ) -> dict[str, Any]:
        identity = {field: intent[field] for field in ('venue', 'market_id', 'side')}
        key = f'{intent["venue"]}:{intent["market_id"]}:{intent["side"]}'
        position = state['positions'].setdefault(
            key,
            {**identity, 'contracts': 0.0, 'cost': 0.0, 'fees': 0.0},
        )
        for field, added in (('contracts', filled), ('cost', cost), ('fees', fees)):
            position[field] = _number(round(position[field] + added, 6), field)
        state['cash'] = round(max(0, state['cash'] - cost - fees), 6)
        record = {
            'time': datetime.now(timezone.utc).isoformat(),
            **identity,
            'requested': intent['contracts'],
            'filled': filled,
            'limit_price': intent['limit_price'],
            'average_price': round(cost / filled, 6),
            'cost': cost,
            'fees': fees,
            'levels': fills,
        }
        state['fills'].append(record)
        status = 'filled' if filled == intent['contracts'] else 'partially_filled'
        return {'status': status, **record}

    def _portfolio(self) -> dict[str, Any]:
        return {
            'ledger': str(self.path),
            'initial_cash': self.state['initial_cash'],
            'cash': self.state['cash'],
            'positions': copy.deepcopy(list(self.state['positions'].values())),
            'fills': len(self.state['fills']),
            'note': 'Paper ledger. Positions are carried at cost; resolution and P&L are not tracked.',
        }

    def portfolio(self) -> dict[str, Any]:
        with self._locked():
            self.state = self._load()
            return self._portfolio()
