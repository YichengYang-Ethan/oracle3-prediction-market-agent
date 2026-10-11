"""Contract-runner reproducibility, evidence, and failure exit behavior."""

from __future__ import annotations

import json
import socket

from scripts import check_agent_contracts as harness


async def test_frozen_suite_passes_and_is_reproducible():
    first = await harness.run_suite()
    second = await harness.run_suite()
    assert first == second
    assert first['summary'] == {'total': 9, 'passed': 9, 'failed': 0}
    assert len({case['case_id'] for case in first['cases']}) == 9
    assert all(case['evidence'] for case in first['cases'])
    json.dumps(first, allow_nan=False)


async def test_runner_blocks_accidental_network_and_reports_failure(monkeypatch):
    async def attempts_network():
        socket.create_connection(('example.invalid', 443))
        return {'unexpected': 'network allowed'}

    monkeypatch.setattr(
        harness, 'CASES', (('fixture.accidental_network', attempts_network),)
    )
    report = await harness.run_suite()
    assert report['summary'] == {'total': 1, 'passed': 0, 'failed': 1}
    failure = report['cases'][0]
    assert failure['case_id'] == 'fixture.accidental_network'
    assert failure['error']['message'].startswith('Network is disabled')


def test_cli_exit_and_json_file_on_failure(monkeypatch, tmp_path):
    async def failing_case():
        raise AssertionError('injected behavioral regression')

    monkeypatch.setattr(harness, 'CASES', (('fixture.regression', failing_case),))
    target = tmp_path / 'contracts.json'
    assert harness.main(['--output', str(target)]) == 1
    report = json.loads(target.read_text())
    assert report['cases'][0]['status'] == 'failed'
    assert report['cases'][0]['error']['message'] == 'injected behavioral regression'


def test_cli_prints_json_and_returns_zero_on_success(monkeypatch, capsys):
    async def passing_case():
        return {'result': 'expected'}

    monkeypatch.setattr(harness, 'CASES', (('fixture.success', passing_case),))
    assert harness.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['summary'] == {'total': 1, 'passed': 1, 'failed': 0}
