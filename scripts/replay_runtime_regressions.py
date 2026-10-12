"""Replay four fixed cases against two local, unmodified Oracle3 Git commits.

Run: python scripts/replay_runtime_regressions.py --output evidence/runtime-regressions.json
Dependencies must already be installed and both commits must exist locally.
This script never fetches, installs dependencies, calls a model, or trades live.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

BASELINE = 'ec0f1a5d6bde7e9fe5b92663ec90aea496f74082'
FIXED = '586bd89b1ee5039488183d10744d389264e2bed1'
REPO = Path(__file__).resolve().parents[1]
PROBE = Path(__file__).with_name('runtime_regression_probe.py')


class BlockedEnvironment(RuntimeError):
    """An existing user control prevents this paper-execution fixture."""


def check_environment() -> None:
    """Respect the pinned traders' kill-switch precedence without changing it."""
    if os.environ.get('PRED_MARKET_CLI_KILL_SWITCH', '').strip() == '1':
        source = 'PRED_MARKET_CLI_KILL_SWITCH=1'
    else:
        explicit = os.environ.get('PRED_MARKET_CLI_KILL_SWITCH_FILE', '').strip()
        sentinel = (
            Path(explicit) if explicit else Path.home() / '.oracle3' / 'kill.switch'
        )
        if not sentinel.exists():
            return
        source = (
            'PRED_MARKET_CLI_KILL_SWITCH_FILE' if explicit else '~/.oracle3/kill.switch'
        )
    raise BlockedEnvironment(
        f'Replay blocked by active global kill switch ({source}); '
        'no snapshot was evaluated and the control was left unchanged.'
    )


EXPECTED = {
    'agent.queued_tool_request': {
        'invariant': 'An accepted queued request reaches paper execution once, even when final text is HOLD.',
        'baseline': {
            'order_count': 0,
            'filled_quantities': [],
            'decision_action': 'HOLD',
            'decision_executed': False,
            'pending_count': 1,
        },
        'fixed': {
            'order_count': 1,
            'filled_quantities': ['10.0'],
            'decision_action': 'BUY_YES',
            'decision_executed': True,
            'pending_count': 0,
        },
    },
    'execution.open_partial': {
        'invariant': 'Uncertain open orders stop further submissions, retaining known exposure for reconciliation.',
        'baseline': {
            'call_count': 4,
            'unwind_count': 2,
            'success': False,
            'hedged': False,
            'execution_uncertain': True,
            'requires_reconciliation': True,
            'residual_exposures': {},
        },
        'fixed': {
            'call_count': 2,
            'unwind_count': 0,
            'success': False,
            'hedged': False,
            'execution_uncertain': True,
            'requires_reconciliation': True,
            'residual_exposures': {'A': '10', 'B': '1'},
        },
    },
    'paper.stale_writer': {
        'invariant': 'Two acknowledged independent paper buys survive a fresh ledger reload.',
        'baseline': {
            'response_statuses': ['filled', 'filled'],
            'fills': 1,
            'cash': 9999.5,
            'contracts': 1.0,
        },
        'fixed': {
            'response_statuses': ['filled', 'filled'],
            'fills': 2,
            'cash': 9999.0,
            'contracts': 2.0,
        },
    },
    'control.full_spread': {
        'invariant': 'Normal exact terminal fills remain successful without unwind or reconciliation.',
        **{
            label: {
                'call_count': 2,
                'unwind_count': 0,
                'success': True,
                'hedged': False,
                'execution_uncertain': False,
                'requires_reconciliation': False,
                'residual_exposures': {},
            }
            for label in ('baseline', 'fixed')
        },
    },
}


def extract_snapshot(archive: bytes, target: Path) -> None:
    """Extract only regular files/directories with safe relative paths (Python 3.10+)."""
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        members = source.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or '..' in path.parts
                or '\\' in member.name
                or any(':' in part for part in path.parts)
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f'Unsafe Git archive member: {member.name}')
        for member in members:
            destination = target.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                content = source.extractfile(member)
                if content is None:
                    raise ValueError(f'Unreadable Git archive member: {member.name}')
                destination.write_bytes(content.read())


def run_worker(snapshot: Path, probe: Path = PROBE) -> dict:
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    environment.pop('PYTHONHOME', None)
    result = subprocess.run(
        [sys.executable, '-I', '-B', str(probe), '--snapshot', str(snapshot)],
        cwd=snapshot,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        raise RuntimeError(f'Probe exited {result.returncode}: {result.stderr.strip()}')
    return json.loads(result.stdout)


def inspect_revision(repo: Path, revision: str) -> dict:
    def git(*arguments: str) -> bytes:
        return subprocess.check_output(['git', *arguments], cwd=repo, timeout=30)

    resolved = git('rev-parse', '--verify', f'{revision}^{{commit}}').decode().strip()
    if resolved != revision:
        raise ValueError('Revision did not resolve to the pinned commit')
    archive = git('archive', '--format=tar', revision)
    with tempfile.TemporaryDirectory(prefix='oracle3-runtime-replay-') as folder:
        snapshot = Path(folder).resolve()
        extract_snapshot(archive, snapshot)
        output = run_worker(snapshot)
    provenance = output['provenance']
    if not provenance['snapshot_imports_verified']:
        raise ValueError('Worker did not verify snapshot imports')
    expected_hash = hashlib.sha256(PROBE.read_bytes()).hexdigest()
    if provenance['probe_sha256'] != expected_hash:
        raise ValueError('Worker probe fingerprint does not match the replay driver')
    provenance.update(
        source_git_revision=resolved,
        source_git_tree=git('rev-parse', f'{revision}^{{tree}}').decode().strip(),
        source_archive_sha256=hashlib.sha256(archive).hexdigest(),
    )
    return output


def evaluate(snapshots: dict) -> list[dict]:
    comparisons = []
    for case_id, expectation in EXPECTED.items():
        observed = {
            label: snapshots[label]['cases'][case_id]['observed']
            for label in ('baseline', 'fixed')
        }
        comparisons.append(
            {
                'case_id': case_id,
                'invariant': expectation['invariant'],
                'expected_observations': {
                    label: expectation[label] for label in ('baseline', 'fixed')
                },
                'observed': observed,
                'historical_behavior_reproduced': observed['baseline']
                == expectation['baseline'],
                'fixed_invariant_satisfied': observed['fixed'] == expectation['fixed'],
            }
        )
    return comparisons


def replay(repo: Path = REPO) -> dict:
    check_environment()
    snapshots = {
        label: inspect_revision(repo, revision)
        for label, revision in (('baseline', BASELINE), ('fixed', FIXED))
    }
    for field in ('python_version', 'dependencies', 'probe_sha256'):
        if (
            snapshots['baseline']['provenance'][field]
            != snapshots['fixed']['provenance'][field]
        ):
            raise ValueError(f'Snapshots were evaluated with different {field}')
    comparisons = evaluate(snapshots)
    passed = all(
        case['historical_behavior_reproduced'] and case['fixed_invariant_satisfied']
        for case in comparisons
    )
    return {
        'schema_version': 1,
        'evaluation': 'pinned_runtime_regression_replay',
        'status': 'passed' if passed else 'failed',
        'baseline_scope': 'Earlier Oracle3 before PR #67; not the Coinjure baseline.',
        'scope': 'Three selected software regressions plus one normal control; no LLM, Skill, profitability, or comparative research evaluation.',
        'normalization': 'Selected observable fields plus raw order/execution/ledger payloads; paper-fill timestamps and temporary ledger paths are omitted.',
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'comparisons': comparisons,
        'snapshots': snapshots,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        report = replay()
    except BlockedEnvironment as exc:
        report = {
            'status': 'blocked',
            'error': {
                'type': type(exc).__name__,
                'code': 'global_kill_switch_active',
                'message': str(exc),
            },
        }
    except Exception as exc:
        report = {
            'status': 'failed',
            'error': {'type': type(exc).__name__, 'message': str(exc)},
        }
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end='')
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
