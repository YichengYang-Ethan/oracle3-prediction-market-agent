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
    assert first['deterministic'] is True
    provenance = first['provenance']
    assert len(provenance['source_sha256']) == 64
    assert provenance['dependencies']['mcp'] == first['mcp_version']
    assert provenance['python_version']
    json.dumps(first, allow_nan=False)


def test_source_fingerprint_is_path_independent_and_detects_changes(tmp_path):
    hashes = []
    for folder in ('first', 'second'):
        package = tmp_path / folder / 'oracle3'
        package.mkdir(parents=True)
        (package / 'module.py').write_text('VALUE = 1\n')
        runner = package.parent / 'runner.py'
        runner.write_text('FIXTURE = 1\n')
        hashes.append(harness._source_fingerprint(package, runner))
    assert hashes[0] == hashes[1]
    (package / 'module.py').write_text('VALUE = 2\n')
    changed = harness._source_fingerprint(package, runner)
    assert changed != hashes[1]
    runner.write_text('FIXTURE = 2\n')
    assert harness._source_fingerprint(package, runner) != changed


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
