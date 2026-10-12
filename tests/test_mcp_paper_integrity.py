"""Offline persistence, retry and concurrent-writer regressions for MCP paper buys."""

from __future__ import annotations

import json
import multiprocessing
from decimal import Decimal
from pathlib import Path

import pytest

from oracle3.fees import PolymarketSchedule
from oracle3.mcp_server.paper import PaperLedger

FREE = PolymarketSchedule(rate=Decimal('0'))
ORDER = {
    'venue': 'polymarket',
    'market_id': 'market-a',
    'side': 'yes',
    'contracts': 1.0,
    'limit_price': 0.6,
}


def buy(ledger, **overrides):
    arguments = {**ORDER, 'asks': [(0.5, 100)], 'schedule': FREE, **overrides}
    return ledger.buy(**arguments)


def _writer(path, barrier, results, request_id):
    ledger = PaperLedger(Path(path))
    barrier.wait(timeout=20)
    results.put(buy(ledger, request_id=request_id))


def test_two_preloaded_writers_and_reads_reload_inside_transaction(tmp_path):
    path = tmp_path / 'ledger.json'
    first, second = PaperLedger(path), PaperLedger(path)
    buy(first)
    buy(second)
    for ledger in (first, second, PaperLedger(path)):
        result = ledger.portfolio()
        assert result['fills'] == 2
        assert result['cash'] == 9999
        assert result['positions'][0]['contracts'] == 2
    first.reset()
    assert second.portfolio()['fills'] == 0
    buy(second)
    assert first.portfolio()['fills'] == 1


@pytest.mark.parametrize('deduplicate', [False, True])
def test_independent_processes_serialize_buys(tmp_path, deduplicate):
    path = tmp_path / 'ledger.json'
    context = multiprocessing.get_context('spawn')
    count = 4
    barrier = context.Barrier(count)
    results = context.Queue()
    processes = [
        context.Process(
            target=_writer,
            args=(
                str(path),
                barrier,
                results,
                'same-request' if deduplicate else str(i),
            ),
        )
        for i in range(count)
    ]
    try:
        for process in processes:
            process.start()
        responses = [results.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        results.close()
        results.join_thread()
    expected = 1 if deduplicate else count
    result = PaperLedger(path).portfolio()
    assert result['fills'] == expected
    assert result['positions'][0]['contracts'] == expected
    assert result['cash'] == 10000 - 0.5 * expected
    if deduplicate:
        assert all(response == responses[0] for response in responses)


def test_durable_retry_is_original_result_and_conflict_preserves_file(tmp_path):
    path = tmp_path / 'ledger.json'
    original = buy(PaperLedger(path), request_id='first')
    saved = path.read_bytes()
    assert buy(PaperLedger(path), request_id='first', asks=[]) == original
    assert path.read_bytes() == saved
    with pytest.raises(ValueError, match='different paper order'):
        buy(PaperLedger(path), request_id='first', contracts=2)
    assert path.read_bytes() == saved
    assert PaperLedger(path).portfolio()['fills'] == 1
    PaperLedger(path).reset()
    assert buy(PaperLedger(path), request_id='first', contracts=2)['filled'] == 2


@pytest.mark.parametrize('status', ['unfilled', 'rejected'])
def test_nonfills_are_deduplicated_too(tmp_path, status):
    path = tmp_path / 'ledger.json'
    ledger = PaperLedger(path, initial_cash=0.1)
    asks = [] if status == 'unfilled' else [(0.5, 100)]
    original = buy(ledger, asks=asks, request_id='attempt')
    assert original['status'] == status
    assert buy(PaperLedger(path), request_id='attempt', asks=[(0.05, 100)]) == original
    assert PaperLedger(path).portfolio()['fills'] == 0


@pytest.mark.parametrize(
    'field,value',
    [
        ('contracts', float('nan')),
        ('contracts', float('inf')),
        ('contracts', -float('inf')),
        ('contracts', 0),
        ('contracts', -1),
        ('limit_price', float('nan')),
        ('limit_price', float('inf')),
        ('limit_price', 0),
        ('limit_price', 1),
        ('market_id', '   '),
        ('request_id', ''),
    ],
)
def test_invalid_orders_cannot_mutate_ledger(tmp_path, field, value):
    path = tmp_path / 'ledger.json'
    ledger = PaperLedger(path)
    buy(ledger)
    saved = path.read_bytes()
    with pytest.raises(ValueError):
        buy(ledger, **{field: value})
    assert path.read_bytes() == saved


@pytest.mark.parametrize(
    'level',
    [
        (float('nan'), 1),
        (float('inf'), 1),
        (-0.01, 1),
        (1.01, 1),
        (0.7, float('nan')),
        (0.7, float('inf')),
        (0.7, -1),
    ],
)
def test_even_unused_bad_ask_levels_fail_before_mutation(tmp_path, level):
    path = tmp_path / 'ledger.json'
    ledger = PaperLedger(path)
    ledger.reset()
    saved = path.read_bytes()
    with pytest.raises(ValueError):
        buy(ledger, asks=[(0.5, 10), level])
    assert path.read_bytes() == saved


@pytest.mark.parametrize(
    'raw',
    [
        '{broken',
        '[]',
        '{}',
        '{"initial_cash":10000,"cash":NaN,"positions":{},"fills":[]}',
        '{"initial_cash":10000,"cash":Infinity,"positions":{},"fills":[]}',
        '{"initial_cash":10000,"cash":1e999,"positions":{},"fills":[]}',
        '{"initial_cash":10000,"cash":10000,"cash":1,"positions":{},"fills":[]}',
        '{"initial_cash":10000,"cash":-1,"positions":{},"fills":[]}',
        '{"initial_cash":10000,"cash":"10000","positions":{},"fills":[]}',
        '{"initial_cash":10000,"cash":10000,"positions":[],"fills":[]}',
        '{"initial_cash":10000,"cash":10000,"positions":{},"fills":[{}]}',
        '{"schema_version":2,"initial_cash":10000,"cash":10000,"positions":{},"fills":[]}',
        '{"initial_cash":10000,"cash":10000,"positions":{},"fills":[],"requests":{"a":{}}}',
    ],
)
def test_corrupt_ledger_fails_closed_and_preserves_bytes(tmp_path, raw):
    path = tmp_path / 'ledger.json'
    ledger = PaperLedger(path)
    path.write_text(raw)
    for action in (
        lambda: PaperLedger(path),
        ledger.portfolio,
        ledger.reset,
        lambda: buy(ledger),
    ):
        with pytest.raises(ValueError, match='Invalid paper ledger.*preserved'):
            action()
        assert path.read_text() == raw


def test_legacy_ledger_upgrades_additively_on_write(tmp_path):
    path = tmp_path / 'ledger.json'
    legacy = {'initial_cash': 10000, 'cash': 10000, 'positions': {}, 'fills': []}
    path.write_text(json.dumps(legacy))
    before = path.read_bytes()
    assert PaperLedger(path).portfolio()['cash'] == 10000
    assert path.read_bytes() == before
    buy(PaperLedger(path), request_id='upgrade')
    saved = json.loads(path.read_text())
    assert saved['schema_version'] == 1
    assert saved['requests']['upgrade']['result']['status'] == 'filled'


def test_legacy_ledger_with_fill_history_remains_readable(tmp_path):
    path = tmp_path / 'ledger.json'
    buy(PaperLedger(path), contracts=12.34567, asks=[(0.53123, 100)])
    data = json.loads(path.read_text())
    del data['schema_version']
    del data['requests']
    path.write_text(json.dumps(data))
    assert PaperLedger(path).portfolio()['fills'] == 1
    buy(PaperLedger(path), request_id='new')
    assert PaperLedger(path).portfolio()['fills'] == 2


def test_large_finite_request_records_actual_available_fill(tmp_path):
    result = buy(PaperLedger(tmp_path / 'ledger.json'), contracts=1e100)
    assert result['status'] == 'partially_filled'
    assert result['filled'] == 100


@pytest.mark.parametrize(
    'part', ['cash', 'position', 'fill_cost', 'fill_price', 'requests']
)
def test_inconsistent_accounting_is_not_silently_loaded(tmp_path, part):
    path = tmp_path / 'ledger.json'
    buy(PaperLedger(path), request_id='first')
    data = json.loads(path.read_text())
    if part == 'cash':
        data['cash'] += 1
    elif part == 'position':
        data['positions']['polymarket:market-a:yes']['contracts'] += 1
    elif part == 'fill_cost':
        data['fills'][0]['cost'] += 1
    elif part == 'fill_price':
        data['fills'][0]['levels'][0]['price'] = 0.9
    else:
        data['requests']['first']['intent']['contracts'] = 2
    path.write_text(json.dumps(data))
    saved = path.read_bytes()
    with pytest.raises(ValueError, match='Invalid paper ledger'):
        PaperLedger(path)
    assert path.read_bytes() == saved


def test_atomic_replace_failure_preserves_valid_previous_ledger(tmp_path, monkeypatch):
    path = tmp_path / 'ledger.json'
    ledger = PaperLedger(path)
    buy(ledger)
    saved = path.read_bytes()

    def fail_replace(*_args):
        raise OSError('simulated interrupted write')

    monkeypatch.setattr('oracle3.mcp_server.paper.os.replace', fail_replace)
    with pytest.raises(OSError, match='interrupted write'):
        buy(ledger, request_id='failed')
    assert path.read_bytes() == saved
    assert list(tmp_path.glob('*.tmp')) == []
    assert PaperLedger(path).portfolio()['fills'] == 1


def test_returned_objects_cannot_mutate_stored_history(tmp_path):
    ledger = PaperLedger(tmp_path / 'ledger.json')
    response = buy(ledger, request_id='first')
    response['levels'][0]['price'] = 0.01
    ledger.portfolio()['positions'][0]['cost'] = 0.01
    assert buy(ledger, request_id='first')['levels'][0]['price'] == 0.5
    assert ledger.portfolio()['positions'][0]['cost'] == 0.5
