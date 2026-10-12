"""Research selection must not convert unavailable estimates into performance."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from click.testing import CliRunner

from oracle3.cli.cli import cli
from oracle3.cli.research_commands import (
    _alpha_score_from_metrics,
    _build_gate_checks,
    _metric_score,
    _to_float_metric,
)


def metrics(**overrides: object) -> dict[str, object]:
    return {
        'total_trades': 2,
        'total_pnl': '5',
        'max_drawdown': '0.01',
        'win_rate': '0.5',
        'valuation_complete': True,
        'sharpe_ratio': '0',
        'sharpe_available': False,
        **overrides,
    }


def history_file(tmp_path: Path) -> Path:
    path = tmp_path / 'history.jsonl'
    path.write_text(
        '\n'.join(
            json.dumps(
                {
                    'market_id': market_id,
                    'event_id': 'E1',
                    'time_series': {'Yes': [{'t': 1, 'p': 0.4}, {'t': 2, 'p': 0.5}]},
                }
            )
            for market_id in ['M1', 'M2']
        )
        + '\n'
    )
    return path


@pytest.mark.parametrize('value', ['NaN', 'Infinity', '-Infinity', True, None])
def test_nonfinite_and_boolean_metrics_are_not_estimates(value: object) -> None:
    assert _to_float_metric(value) is None
    assert _metric_score(metrics(total_pnl=value), 'total_pnl') is None


def test_sharpe_zero_only_valid_with_explicit_availability() -> None:
    assert _metric_score(metrics(), 'sharpe_ratio') is None
    assert _metric_score(metrics(sharpe_ratio='100'), 'sharpe_ratio') is None
    assert _metric_score(metrics(sharpe_available=True), 'sharpe_ratio') == 0
    legacy = metrics(sharpe_available=True)
    del legacy['sharpe_available']
    assert _metric_score(legacy, 'sharpe_ratio') is None


@pytest.mark.parametrize(
    'key', ['total_pnl', 'max_drawdown', 'sharpe_ratio', 'alpha_score']
)
@pytest.mark.parametrize('availability', [False, None, 'true'])
def test_unknown_or_missing_marks_exclude_nav_rankings(
    key: str, availability: object
) -> None:
    assert (
        _metric_score(
            metrics(valuation_complete=availability, sharpe_available=True), key
        )
        is None
    )
    legacy = metrics()
    del legacy['valuation_complete']
    assert _metric_score(legacy, key) is None


def test_alpha_heuristic_requires_complete_finite_inputs() -> None:
    assert _alpha_score_from_metrics(metrics()) == pytest.approx(2.46)
    assert _alpha_score_from_metrics(metrics(valuation_complete=False)) is None
    assert _alpha_score_from_metrics(metrics(total_pnl='NaN')) is None
    assert _alpha_score_from_metrics(metrics(total_trades=None)) is None
    assert _alpha_score_from_metrics(metrics(max_drawdown=-1)) is None
    assert _metric_score(metrics(total_trades=0), 'win_rate') is None


def test_gate_rejects_incomplete_valuation_and_nonfinite_inputs() -> None:
    for values in [
        metrics(valuation_complete=False),
        metrics(total_pnl='NaN'),
        metrics(max_drawdown='Infinity'),
        metrics(total_trades='NaN'),
    ]:
        passed, checks = _build_gate_checks(
            metrics=values,
            min_trades=1,
            min_total_pnl=Decimal('0'),
            max_drawdown_pct=Decimal('0.3'),
        )
        assert not passed
        assert not all(checks.values())


def test_compare_excludes_unavailable_sharpe_and_failed_runs(tmp_path: Path) -> None:
    path = tmp_path / 'runs.jsonl'
    rows = [
        {'name': 'missing', 'metrics': metrics(sharpe_ratio='100')},
        {'name': 'valid-zero', 'metrics': metrics(sharpe_available=True)},
        {
            'name': 'failed',
            'ok': False,
            'metrics': metrics(sharpe_available=True, sharpe_ratio='200'),
        },
    ]
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    result = CliRunner().invoke(
        cli,
        [
            'research',
            'compare-runs',
            '--input-file',
            str(path),
            '--sort-key',
            'sharpe_ratio',
            '--json',
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload['ranking_available'] is True
    assert payload['excluded_runs'] == 2
    assert [row['name'] for row in payload['top']] == ['valid-zero']
    assert 'Exploratory' in payload['selection_note']


def test_compare_default_pnl_excludes_unmarked_and_legacy_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'runs.jsonl'
    rows = [
        {
            'name': 'unmarked',
            'metrics': metrics(total_pnl='100', valuation_complete=False),
        },
        {'name': 'legacy', 'metrics': {'total_pnl': '200'}},
        {'name': 'valid', 'metrics': metrics()},
    ]
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    result = CliRunner().invoke(
        cli, ['research', 'compare-runs', '--input-file', str(path), '--json']
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload['sort_key'] == 'total_pnl'
    assert [row['name'] for row in payload['top']] == ['valid']


def test_grid_has_no_winner_when_sharpe_unavailable(
    monkeypatch, tmp_path: Path
) -> None:
    path = history_file(tmp_path)
    output = tmp_path / 'grid.jsonl'
    monkeypatch.setattr(
        'oracle3.cli.research_commands._run_backtest_once', lambda **_: metrics()
    )
    result = CliRunner().invoke(
        cli,
        [
            'research',
            'grid',
            '--history-file',
            str(path),
            '--market-id',
            'M1',
            '--event-id',
            'E1',
            '--strategy-ref',
            'unused:Mock',
            '--param-grid-json',
            '{"entry":[1,2]}',
            '--sort-key',
            'sharpe_ratio',
            '--output',
            str(output),
            '--json',
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload['best'] is None
    assert payload['ranking_available'] is False
    assert payload['unavailable_runs'] == 2
    assert len(output.read_text().splitlines()) == 2


def test_grid_default_selects_available_pnl(monkeypatch, tmp_path: Path) -> None:
    path = history_file(tmp_path)
    monkeypatch.setattr(
        'oracle3.cli.research_commands._run_backtest_once',
        lambda **kwargs: metrics(
            total_pnl='100' if kwargs['strategy_kwargs']['entry'] == 1 else '5',
            valuation_complete=kwargs['strategy_kwargs']['entry'] == 2,
        ),
    )
    result = CliRunner().invoke(
        cli,
        [
            'research',
            'grid',
            '--history-file',
            str(path),
            '--market-id',
            'M1',
            '--event-id',
            'E1',
            '--strategy-ref',
            'unused:Mock',
            '--param-grid-json',
            '{"entry":[1,2]}',
            '--output',
            str(tmp_path / 'grid.jsonl'),
            '--json',
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload['sort_key'] == 'total_pnl'
    assert payload['best']['entry'] == 2
    assert payload['unavailable_runs'] == 1


def test_auto_tune_unavailable_metric_does_not_select_or_start_paper(
    monkeypatch, tmp_path: Path
) -> None:
    path = history_file(tmp_path)
    artifacts = tmp_path / 'tune'
    monkeypatch.setattr(
        'oracle3.cli.research_commands._run_backtest_once', lambda **_: metrics()
    )

    async def forbidden_paper(**_):
        pytest.fail('No paper session should start without an eligible selected run')

    monkeypatch.setattr(
        'oracle3.cli.research_commands.run_live_paper_trading', forbidden_paper
    )
    result = CliRunner().invoke(
        cli,
        [
            'research',
            'auto-tune',
            '--history-file',
            str(path),
            '--market-id',
            'M1',
            '--event-id',
            'E1',
            '--strategy-ref',
            'unused:Mock',
            '--selection-key',
            'sharpe_ratio',
            '--run-paper',
            '--artifacts-dir',
            str(artifacts),
            '--json',
        ],
    )
    assert result.exit_code != 0
    payload, _ = json.JSONDecoder().raw_decode(result.output)
    assert payload['best_run'] is None
    assert payload['ranking_available'] is False
    assert (artifacts / 'discover_runs.jsonl').exists()
    assert not (artifacts / 'best_run.json').exists()
    assert not (artifacts / 'paper.json').exists()


@pytest.mark.parametrize('some_available', [False, True])
def test_batch_means_exclude_unavailable_values(
    monkeypatch, tmp_path: Path, some_available: bool
) -> None:
    path = history_file(tmp_path)
    monkeypatch.setattr(
        'oracle3.cli.research_commands._run_backtest_once',
        lambda **kwargs: metrics(
            valuation_complete=some_available and kwargs['market_id'] == 'M1',
            sharpe_available=some_available and kwargs['market_id'] == 'M1',
            sharpe_ratio='2' if kwargs['market_id'] == 'M1' else '0',
        ),
    )
    result = CliRunner().invoke(
        cli,
        [
            'research',
            'batch-markets',
            '--history-file',
            str(path),
            '--strategy-ref',
            'unused:Mock',
            '--output',
            str(tmp_path / 'batch.jsonl'),
            '--json',
        ],
    )
    assert result.exit_code == 0, result.output
    aggregate = json.loads(result.output)['aggregate']
    assert aggregate['sharpe_available_count'] == int(some_available)
    assert aggregate['sharpe_unavailable_count'] == 2 - int(some_available)
    assert aggregate['mean_sharpe'] == ('2.0' if some_available else None)
    assert aggregate['stddev_sharpe'] is None
    assert aggregate['mean_pnl'] == ('5.0' if some_available else None)
    assert aggregate['pct_profitable'] == ('100.0' if some_available else None)
