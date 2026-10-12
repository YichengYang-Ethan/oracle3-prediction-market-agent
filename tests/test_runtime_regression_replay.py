"""Verify snapshot isolation, offline execution and failure-sensitive replay gates."""

from __future__ import annotations

import copy
import io
import json
import os
import subprocess
import sys
import tarfile
from types import ModuleType

import pytest

from scripts import replay_runtime_regressions as replay
from scripts import runtime_regression_probe as probe


def test_active_kill_switch_blocks_before_snapshots_and_is_preserved(
    monkeypatch, capsys
):
    monkeypatch.setenv('PRED_MARKET_CLI_KILL_SWITCH', '1')

    def unexpected_snapshot(*args):
        pytest.fail('An active user control must block before evaluating snapshots')

    monkeypatch.setattr(replay, 'inspect_revision', unexpected_snapshot)
    assert replay.main([]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'blocked'
    assert report['error']['code'] == 'global_kill_switch_active'
    assert 'comparisons' not in report and 'snapshots' not in report
    assert os.environ['PRED_MARKET_CLI_KILL_SWITCH'] == '1'


def test_active_kill_switch_file_is_preserved(monkeypatch, tmp_path):
    sentinel = tmp_path / 'kill.switch'
    sentinel.write_text('user control')
    monkeypatch.setenv('PRED_MARKET_CLI_KILL_SWITCH_FILE', str(sentinel))
    with pytest.raises(replay.BlockedEnvironment):
        replay.check_environment()
    assert sentinel.read_text() == 'user control'


@pytest.fixture(scope='module')
def observed_report():
    for revision in (replay.BASELINE, replay.FIXED):
        available = subprocess.run(
            ['git', 'cat-file', '-e', f'{revision}^{{commit}}'],
            cwd=replay.REPO,
            capture_output=True,
            check=False,
        )
        if available.returncode:
            pytest.skip('Pinned history unavailable; use a checkout with full history')
    return replay.replay()


def test_real_snapshots_reproduce_failures_and_preserve_normal_control(observed_report):
    assert observed_report['status'] == 'passed'
    comparisons = {case['case_id']: case for case in observed_report['comparisons']}
    assert (
        comparisons['agent.queued_tool_request']['observed']['baseline']['order_count']
        == 0
    )
    assert (
        comparisons['agent.queued_tool_request']['observed']['fixed']['order_count']
        == 1
    )
    assert comparisons['paper.stale_writer']['observed']['fixed']['fills'] == 2
    control = comparisons['control.full_spread']['observed']
    assert control['baseline'] == control['fixed']
    assert control['fixed']['success']
    for label, revision in (('baseline', replay.BASELINE), ('fixed', replay.FIXED)):
        provenance = observed_report['snapshots'][label]['provenance']
        assert provenance['source_git_revision'] == revision
        assert provenance['snapshot_imports_verified']
        assert 'oracle3/__init__.py' in provenance['snapshot_import_paths']
        assert len(provenance['source_archive_sha256']) == 64
        assert provenance['python_version']
        assert provenance['dependencies']


@pytest.mark.parametrize('label', ['baseline', 'fixed'])
def test_changed_observation_fails_gate_and_cli(
    observed_report, monkeypatch, tmp_path, label
):
    snapshots = copy.deepcopy(observed_report['snapshots'])
    snapshots[label]['cases']['paper.stale_writer']['observed']['fills'] = 123
    by_revision = {
        replay.BASELINE: snapshots['baseline'],
        replay.FIXED: snapshots['fixed'],
    }
    monkeypatch.setattr(
        replay, 'inspect_revision', lambda repo, revision: by_revision[revision]
    )
    target = tmp_path / 'failed.json'
    assert replay.main(['--output', str(target)]) == 1
    report = json.loads(target.read_text())
    comparison = next(
        case
        for case in report['comparisons']
        if case['case_id'] == 'paper.stale_writer'
    )
    field = (
        'historical_behavior_reproduced'
        if label == 'baseline'
        else 'fixed_invariant_satisfied'
    )
    assert comparison[field] is False


def test_worker_failure_is_not_a_successful_empty_report(tmp_path, monkeypatch, capsys):
    worker = tmp_path / 'worker.py'
    worker.write_text("raise RuntimeError('injected worker failure')\n")
    with pytest.raises(RuntimeError, match='injected worker failure'):
        replay.run_worker(tmp_path, worker)
    monkeypatch.setattr(replay, 'replay', lambda: replay.run_worker(tmp_path, worker))
    assert replay.main([]) == 1
    assert json.loads(capsys.readouterr().out)['status'] == 'failed'


def test_imported_checkout_cannot_be_labeled_as_snapshot(tmp_path, monkeypatch):
    foreign = ModuleType('oracle3')
    foreign.__file__ = str(tmp_path / 'outside' / 'oracle3' / '__init__.py')
    monkeypatch.setitem(sys.modules, 'oracle3', foreign)
    with pytest.raises(RuntimeError, match='outside the snapshot'):
        probe.verify_snapshot_imports(tmp_path / 'snapshot')


@pytest.mark.parametrize(
    'network_call',
    [
        "socket.create_connection(('example.invalid', 443))",
        "socket.getaddrinfo('example.invalid', 443)",
    ],
)
def test_worker_network_block_rejects_connect_and_dns(network_call):
    code = (
        'import importlib.util, socket; '
        f'spec=importlib.util.spec_from_file_location("probe",{str(replay.PROBE)!r}); '
        'probe=importlib.util.module_from_spec(spec); spec.loader.exec_module(probe); '
        f'probe.block_network(); {network_call}'
    )
    result = subprocess.run(
        [sys.executable, '-I', '-c', code], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert 'Network disabled by runtime regression probe' in result.stderr


@pytest.mark.parametrize(
    'name,kind',
    [
        ('../escape.py', tarfile.REGTYPE),
        ('/absolute.py', tarfile.REGTYPE),
        ('C:/escape.py', tarfile.REGTYPE),
        ('link.py', tarfile.SYMTYPE),
        ('hardlink.py', tarfile.LNKTYPE),
    ],
)
def test_unsafe_archive_rejected_before_any_write(tmp_path, name, kind):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w') as archive:
        normal = tarfile.TarInfo('normal.py')
        normal.size = 1
        archive.addfile(normal, io.BytesIO(b'1'))
        unsafe = tarfile.TarInfo(name)
        unsafe.type = kind
        unsafe.linkname = '../outside'
        archive.addfile(unsafe)
    with pytest.raises(ValueError, match='Unsafe Git archive member'):
        replay.extract_snapshot(output.getvalue(), tmp_path)
    assert list(tmp_path.iterdir()) == []
