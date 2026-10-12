from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from click.testing import CliRunner

from oracle3.cli.cli import cli


def _write_history(path: Path) -> None:
    rows = [
        {
            'event_id': 'E1',
            'market_id': 'M1',
            'time_series': {'Yes': [{'t': 1, 'p': 0.40}, {'t': 2, 'p': 0.45}]},
        },
        {
            'event_id': 'E1',
            'market_id': 'M1',
            'time_series': {'Yes': [{'t': 3, 'p': 0.55}, {'t': 4, 'p': 0.50}]},
        },
        {
            'event_id': 'E2',
            'market_id': 'M2',
            'time_series': {'Yes': [{'t': 1, 'p': 0.90}]},
        },
    ]
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8')


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_iso_history(path: Path) -> None:
    rows = [
        {
            'event_id': 'E1',
            'market_id': 'M1',
            'question': 'Market One',
            'volume': 1000,
            'time_series': {
                'Yes': [
                    {'t': f'2026-02-20T00:{i:02d}:00+00:00', 'p': 0.40 + i * 0.01}
                    for i in range(12)
                ]
            },
        },
        {
            'event_id': 'E2',
            'market_id': 'M2',
            'question': 'Market Two',
            'volume': 200,
            'time_series': {
                'Yes': [
                    {'t': f'2026-02-20T01:{i:02d}:00+00:00', 'p': 0.50 + i * 0.005}
                    for i in range(6)
                ]
            },
        },
    ]
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8')


def test_research_group_help_lists_tools() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ['research', '--help'])
    assert result.exit_code == 0
    assert 'strategy-discovery tooling' in result.output
    assert 'slice' in result.output
    assert 'walk-forward' in result.output
    assert 'stress-test' in result.output
    assert 'compare-runs' in result.output
    assert 'strategy-gate' in result.output
    assert 'markets' in result.output
    assert 'alpha-pipeline' in result.output
    assert 'auto-tune' in result.output
    assert 'memory' in result.output
    assert 'grid' in result.output
    # Removed commands should NOT appear
    assert 'features' not in result.output
    assert 'labels' not in result.output
    assert 'backtest-batch' not in result.output
    assert 'walk-forward-auto' not in result.output
    assert 'scan-markets' not in result.output


def test_research_slice(tmp_path: Path) -> None:
    history = tmp_path / 'history.jsonl'
    _write_history(history)
    sliced = tmp_path / 'slice.jsonl'

    runner = CliRunner()
    slice_result = runner.invoke(
        cli,
        [
            'research',
            'slice',
            '--history-file',
            str(history),
            '--market-id',
            'M1',
            '--event-id',
            'E1',
            '--start-ts',
            '2',
            '--end-ts',
            '4',
            '--output',
            str(sliced),
            '--json',
        ],
    )
    assert slice_result.exit_code == 0
    assert '"points": 3' in slice_result.output


def test_research_slice_supports_iso_timestamps(tmp_path: Path) -> None:
    history = tmp_path / 'history_iso.jsonl'
    rows = [
        {
            'event_id': 'E1',
            'market_id': 'M1',
            'time_series': {
                'Yes': [
                    {'t': '1970-01-01T00:00:01+00:00', 'p': 0.40},
                    {'t': '1970-01-01T00:00:02Z', 'p': 0.45},
                    {'t': '1970-01-01T00:00:03+00:00', 'p': 0.55},
                ]
            },
        }
    ]
    history.write_text(
        '\n'.join(json.dumps(row) for row in rows) + '\n',
        encoding='utf-8',
    )
    sliced = tmp_path / 'slice_iso.jsonl'

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            'research',
            'slice',
            '--history-file',
            str(history),
            '--market-id',
            'M1',
            '--event-id',
            'E1',
            '--start-ts',
            '2',
            '--end-ts',
            '3',
            '--output',
            str(sliced),
            '--json',
        ],
    )
    assert result.exit_code == 0
    assert '"points": 2' in result.output


def test_research_grid_with_params_jsonl(monkeypatch, tmp_path: Path) -> None:
    """grid --params-jsonl replaces the old backtest-batch command."""
    history = tmp_path / 'history.jsonl'
    _write_history(history)
    params = tmp_path / 'params.jsonl'
    params_rows = [
        {'trade_size': 10},
        {'entry_z': 1.2},
    ]
    params.write_text(
        '\n'.join(json.dumps(row) for row in params_rows) + '\n',
        encoding='utf-8',
    )
    output = tmp_path / 'grid_out.jsonl'

    def fake_run_backtest_once(**kwargs):
        return {
            'total_trades': 2,
            'valuation_complete': True,
            'total_pnl': str(Decimal('12.5')),
            'sharpe_ratio': '1.0',
            'win_rate': '0.5',
            'max_drawdown': '0.1',
            'received_kwargs': kwargs['strategy_kwargs'],
        }

    monkeypatch.setattr(
        'oracle3.cli.research_commands._run_backtest_once', fake_run_backtest_once
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            'research',
            'grid',
            '--history-file',
            str(history),
            '--market-id',
            'M1',
            '--event-id',
            'E1',
            '--strategy-ref',
            'oracle3.strategy.test_strategy:TestStrategy',
            '--params-jsonl',
            str(params),
            '--output',
            str(output),
            '--json',
        ],
    )
    assert result.exit_code == 0
    out_rows = _load_jsonl(output)
    assert len(out_rows) == 2
    assert out_rows[0]['ok'] is True
    assert out_rows[1]['metrics']['received_kwargs'] == {'entry_z': 1.2}


def test_research_slice_supports_iso_timestamps_in_json_array(tmp_path: Path) -> None:
    history = tmp_path / 'history.json'
    rows = [
        {
            'event_id': 'E1',
            'market_id': 'M1',
            'time_series': {
                'Yes': [
                    {'t': '2025-01-01T00:00:00+00:00', 'p': 0.40},
                    {'t': '2025-01-01T00:05:00+00:00', 'p': 0.45},
                    {'t': '2025-01-01T00:10:00+00:00', 'p': 0.50},
                ]
            },
        }
    ]
    history.write_text(json.dumps(rows), encoding='utf-8')
    out_file = tmp_path / 'slice_out.jsonl'
    start_ts = int(datetime(2025, 1, 1, 0, 5, tzinfo=timezone.utc).timestamp())

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            'research',
            'slice',
            '--history-file',
            str(history),
            '--market-id',
            'M1',
            '--event-id',
            'E1',
            '--start-ts',
            str(start_ts),
            '--output',
            str(out_file),
            '--json',
        ],
    )
    assert result.exit_code == 0
    rows_out = _load_jsonl(out_file)
    assert len(rows_out) == 2
    assert rows_out[0]['time_series']['Yes'][0]['t'] == start_ts


def test_research_scan_markets_removed() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ['research', 'scan-markets'])
    assert result.exit_code != 0
    assert 'No such command' in result.output


def test_research_auto_tune_runs_end_to_end(monkeypatch, tmp_path: Path) -> None:
    history = tmp_path / 'history_iso.jsonl'
    _write_iso_history(history)
    artifacts = tmp_path / 'auto_tune_run'
    strategy_file = tmp_path / 'discover_strategy.py'
    strategy_file.write_text(
        '\n'.join(
            [
                'from oracle3.events.events import Event',
                'from oracle3.strategy.strategy import Strategy',
                'from oracle3.trader.trader import Trader',
                '',
                'class DiscoverStrategy(Strategy):',
                '    def __init__(self, momentum_entry: float = 0.01):',
                '        self.momentum_entry = momentum_entry',
                '',
                '    async def process_event(self, event: Event, trader: Trader) -> None:',
                '        return',
            ]
        ),
        encoding='utf-8',
    )

    def fake_run_backtest_once(**kwargs):
        params = kwargs['strategy_kwargs']
        momentum = params.get('momentum_entry', 0.02)
        if momentum <= 0.01:
            pnl = Decimal('5.0')
            dd = Decimal('0.02')
        else:
            pnl = Decimal('-1.0')
            dd = Decimal('0.08')
        return {
            'total_trades': 4,
            'valuation_complete': True,
            'total_pnl': str(pnl),
            'sharpe_ratio': '0.8',
            'win_rate': '0.5',
            'max_drawdown': str(dd),
        }

    async def fake_paper_run(**kwargs):
        return None

    monkeypatch.setattr(
        'oracle3.cli.research_commands._run_backtest_once', fake_run_backtest_once
    )
    monkeypatch.setattr(
        'oracle3.cli.research_commands.run_live_paper_trading', fake_paper_run
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            'research',
            'auto-tune',
            '--history-file',
            str(history),
            '--strategy-ref',
            f'{strategy_file}:DiscoverStrategy',
            '--param-grid-json',
            '{"momentum_entry":[0.01,0.02]}',
            '--market-rank',
            '1',
            '--run-paper',
            '--paper-duration',
            '1',
            '--artifacts-dir',
            str(artifacts),
            '--json',
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload['passed'] is True
    assert payload['ok_runs'] == 2
    assert payload['best_run']['strategy_kwargs']['momentum_entry'] == 0.01
    assert (artifacts / 'discover_runs.jsonl').exists()
    assert (artifacts / 'best_run.json').exists()
    assert (artifacts / 'gate.json').exists()
    assert (artifacts / 'paper.json').exists()


def test_research_auto_tune_supports_single_strategy_gate(
    monkeypatch, tmp_path: Path
) -> None:
    history = tmp_path / 'history_iso.jsonl'
    _write_iso_history(history)
    artifacts = tmp_path / 'auto_tune_single'

    def fake_run_backtest_once(**kwargs):
        strategy_kwargs = kwargs['strategy_kwargs']
        entry = strategy_kwargs.get('entry', 0)
        pnl = Decimal('1.0') if entry == 2 else Decimal('-0.5')
        return {
            'total_trades': 3,
            'valuation_complete': True,
            'total_pnl': str(pnl),
            'sharpe_ratio': '0.6',
            'win_rate': '0.5',
            'max_drawdown': '0.05',
        }

    monkeypatch.setattr(
        'oracle3.cli.research_commands._run_backtest_once', fake_run_backtest_once
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            'research',
            'auto-tune',
            '--history-file',
            str(history),
            '--strategy-ref',
            'pkg.alpha:StrategyA',
            '--param-grid-json',
            '{"entry":[1,2]}',
            '--market-id',
            'M1',
            '--event-id',
            'E1',
            '--selection-key',
            'total_pnl',
            '--enforce-gate',
            '--min-trades',
            '1',
            '--min-total-pnl',
            '0',
            '--max-drawdown-pct',
            '0.30',
            '--artifacts-dir',
            str(artifacts),
            '--json',
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload['passed'] is True
    assert payload['ok_runs'] == 2
    assert payload['best_run']['strategy_ref'] == 'pkg.alpha:StrategyA'
    assert payload['best_run']['strategy_kwargs']['entry'] == 2
    assert (artifacts / 'discover_runs.jsonl').exists()
    assert (artifacts / 'best_run.json').exists()
    assert (artifacts / 'gate.json').exists()


def test_research_auto_tune_requires_strategy_ref(tmp_path: Path) -> None:
    history = tmp_path / 'history_iso.jsonl'
    _write_iso_history(history)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            'research',
            'auto-tune',
            '--history-file',
            str(history),
        ],
    )
    assert result.exit_code != 0
    assert '--strategy-ref' in result.output


def test_research_compare_runs_and_memory(tmp_path: Path) -> None:
    runs_file = tmp_path / 'runs.jsonl'
    runs_rows = [
        {
            'name': 'low',
            'ok': True,
            'metrics': {
                'total_pnl': '1.0',
                'win_rate': '0.5',
                'valuation_complete': True,
            },
        },
        {
            'name': 'high',
            'ok': True,
            'metrics': {
                'total_pnl': '8.0',
                'win_rate': '0.6',
                'valuation_complete': True,
            },
        },
    ]
    runs_file.write_text(
        '\n'.join(json.dumps(row) for row in runs_rows) + '\n',
        encoding='utf-8',
    )
    ranked_file = tmp_path / 'ranked.jsonl'
    memory_file = tmp_path / 'run_memory.jsonl'

    runner = CliRunner()
    compare = runner.invoke(
        cli,
        [
            'research',
            'compare-runs',
            '--input-file',
            str(runs_file),
            '--sort-key',
            'total_pnl',
            '--top',
            '1',
            '--output',
            str(ranked_file),
            '--json',
        ],
    )
    assert compare.exit_code == 0
    ranked_rows = _load_jsonl(ranked_file)
    assert len(ranked_rows) == 1
    assert ranked_rows[0]['name'] == 'high'

    add = runner.invoke(
        cli,
        [
            'research',
            'memory',
            'add',
            '--input-file',
            str(ranked_file),
            '--memory-file',
            str(memory_file),
            '--tag',
            'exp-a',
            '--json',
        ],
    )
    assert add.exit_code == 0

    list_res = runner.invoke(
        cli,
        [
            'research',
            'memory',
            'list',
            '--memory-file',
            str(memory_file),
            '--tag',
            'exp-a',
            '--json',
        ],
    )
    assert list_res.exit_code == 0
    payload = json.loads(list_res.output)
    assert payload['count'] == 1


def test_research_markets_and_walk_forward_target_runs(
    monkeypatch, tmp_path: Path
) -> None:
    history = tmp_path / 'history_iso.jsonl'
    _write_iso_history(history)
    wf_output = tmp_path / 'wf_auto.jsonl'

    def fake_run_backtest_once(**kwargs):
        return {
            'total_trades': 5,
            'valuation_complete': True,
            'total_pnl': '10.0',
            'sharpe_ratio': '0.8',
            'win_rate': '0.6',
            'max_drawdown': '0.1',
            'kwargs': kwargs['strategy_kwargs'],
        }

    monkeypatch.setattr(
        'oracle3.cli.research_commands._run_backtest_once', fake_run_backtest_once
    )

    runner = CliRunner()
    markets = runner.invoke(
        cli,
        [
            'research',
            'markets',
            '--history-file',
            str(history),
            '--sort-by',
            'points',
            '--limit',
            '1',
            '--json',
        ],
    )
    assert markets.exit_code == 0
    payload = json.loads(markets.output)
    assert payload['count'] == 1
    assert payload['markets'][0]['market_id'] == 'M1'
    assert payload['markets'][0]['points'] == 12

    wf = runner.invoke(
        cli,
        [
            'research',
            'walk-forward',
            '--history-file',
            str(history),
            '--market-id',
            'M1',
            '--event-id',
            'E1',
            '--strategy-ref',
            'oracle3.strategy.test_strategy:TestStrategy',
            '--train-size',
            '20',
            '--test-size',
            '10',
            '--target-runs',
            '2',
            '--output',
            str(wf_output),
            '--json',
        ],
    )
    assert wf.exit_code == 0
    wf_payload = json.loads(wf.output)
    assert wf_payload['n_points'] == 12
    # Auto sizing should shrink requested windows to fit the available points.
    assert wf_payload['train_size'] < 20
    assert wf_payload['test_size'] < 10
    assert wf_payload['auto_resized'] is True
    assert wf_payload['runs'] >= 1
    rows = _load_jsonl(wf_output)
    assert len(rows) == wf_payload['runs']


def test_research_walk_forward_auto_resizes(monkeypatch, tmp_path: Path) -> None:
    history = tmp_path / 'history_iso.jsonl'
    _write_iso_history(history)
    output = tmp_path / 'wf_manual.jsonl'

    def fake_run_backtest_once(**kwargs):
        return {
            'total_trades': 2,
            'valuation_complete': True,
            'total_pnl': '1.0',
            'sharpe_ratio': '0.2',
            'win_rate': '0.5',
            'max_drawdown': '0.1',
        }

    monkeypatch.setattr(
        'oracle3.cli.research_commands._run_backtest_once', fake_run_backtest_once
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            'research',
            'walk-forward',
            '--history-file',
            str(history),
            '--market-id',
            'M1',
            '--event-id',
            'E1',
            '--strategy-ref',
            'oracle3.strategy.test_strategy:TestStrategy',
            '--train-size',
            '999',
            '--test-size',
            '500',
            '--step-size',
            '120',
            '--output',
            str(output),
            '--json',
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload['auto_resized'] is True
    assert payload['runs'] >= 1
    assert output.exists()


def test_research_markets_filters_and_alpha_score(tmp_path: Path) -> None:
    history = tmp_path / 'history_iso.jsonl'
    _write_iso_history(history)
    runner = CliRunner()

    filtered = runner.invoke(
        cli,
        [
            'research',
            'markets',
            '--history-file',
            str(history),
            '--min-points',
            '10',
            '--min-volume',
            '500',
            '--min-span-seconds',
            '1',
            '--json',
        ],
    )
    assert filtered.exit_code == 0
    payload = json.loads(filtered.output)
    assert payload['count'] == 1
    assert payload['markets'][0]['market_id'] == 'M1'

    runs = tmp_path / 'runs.jsonl'
    runs.write_text(
        '\n'.join(
            [
                json.dumps(
                    {
                        'name': 'hi-dd',
                        'ok': True,
                        'metrics': {
                            'total_pnl': '50',
                            'max_drawdown': '0.20',
                            'total_trades': 50,
                            'valuation_complete': True,
                        },
                    }
                ),
                json.dumps(
                    {
                        'name': 'balanced',
                        'ok': True,
                        'metrics': {
                            'total_pnl': '40',
                            'max_drawdown': '0.02',
                            'total_trades': 10,
                            'valuation_complete': True,
                        },
                    }
                ),
            ]
        )
        + '\n',
        encoding='utf-8',
    )

    ranked = runner.invoke(
        cli,
        [
            'research',
            'compare-runs',
            '--input-file',
            str(runs),
            '--sort-key',
            'alpha_score',
            '--top',
            '1',
            '--json',
        ],
    )
    assert ranked.exit_code == 0
    ranked_payload = json.loads(ranked.output)
    assert ranked_payload['top'][0]['name'] == 'balanced'


def test_research_grid_supports_alpha_score_sort(monkeypatch, tmp_path: Path) -> None:
    history = tmp_path / 'history.jsonl'
    _write_history(history)
    output = tmp_path / 'grid.jsonl'

    def fake_run_backtest_once(**kwargs):
        threshold = kwargs['strategy_kwargs'].get('threshold', 0.2)
        if threshold <= 0.1:
            return {
                'total_trades': 5,
                'valuation_complete': True,
                'total_pnl': '10.0',
                'sharpe_ratio': '0.6',
                'win_rate': '0.5',
                'max_drawdown': '0.10',
            }
        return {
            'total_trades': 5,
            'valuation_complete': True,
            'total_pnl': '9.0',
            'sharpe_ratio': '0.7',
            'win_rate': '0.6',
            'max_drawdown': '0.01',
        }

    monkeypatch.setattr(
        'oracle3.cli.research_commands._run_backtest_once', fake_run_backtest_once
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            'research',
            'grid',
            '--history-file',
            str(history),
            '--market-id',
            'M1',
            '--event-id',
            'E1',
            '--strategy-ref',
            'oracle3.strategy.test_strategy:TestStrategy',
            '--param-grid-json',
            '{"threshold":[0.05,0.2]}',
            '--sort-key',
            'alpha_score',
            '--output',
            str(output),
            '--json',
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload['best']['threshold'] == 0.2


def test_research_alpha_pipeline(monkeypatch, tmp_path: Path) -> None:
    history = tmp_path / 'history_iso.jsonl'
    _write_iso_history(history)
    artifacts_dir = tmp_path / 'alpha_pipeline'

    def fake_dry_run(**kwargs):
        return {
            'ok': True,
            'events_requested': kwargs['dry_run_events'],
            'events_processed': kwargs['dry_run_events'],
            'orders_created': 1,
            'decision_stats': {},
            'error': None,
        }

    def fake_run_backtest_once(**kwargs):
        return {
            'total_trades': 8,
            'valuation_complete': True,
            'total_pnl': '12.0',
            'sharpe_ratio': '0.7',
            'win_rate': '0.55',
            'max_drawdown': '0.12',
        }

    monkeypatch.setattr(
        'oracle3.cli.research_commands._run_strategy_dry_run', fake_dry_run
    )
    monkeypatch.setattr(
        'oracle3.cli.research_commands._run_backtest_once', fake_run_backtest_once
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            'research',
            'alpha-pipeline',
            '--history-file',
            str(history),
            '--strategy-ref',
            'dummy.module:Dummy',
            '--market-sort-by',
            'points',
            '--market-rank',
            '1',
            '--min-trades',
            '1',
            '--min-total-pnl',
            '0',
            '--max-drawdown-pct',
            '0.30',
            '--no-run-batch-markets',
            '--artifacts-dir',
            str(artifacts_dir),
            '--json',
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload['passed'] is True
    assert payload['selected_market']['market_id'] == 'M1'
    assert (artifacts_dir / 'preflight.json').exists()
    assert (artifacts_dir / 'backtest_single.json').exists()
    assert (artifacts_dir / 'stress.jsonl').exists()
    assert (artifacts_dir / 'gate.json').exists()
