"""Smoke-test an installed wheel through a real, offline MCP stdio session.

Run with the Python environment into which the wheel was installed:
    /tmp/wheel-venv/bin/python scripts/check_installed_mcp.py --wheel dist/oracle3-*.whl
The server and client run outside the checkout. No venue or model is contacted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _unpack(result) -> dict:
    _require(
        not (getattr(result, 'isError', False) or getattr(result, 'is_error', False)),
        'MCP tool returned an error',
    )
    data = getattr(result, 'structuredContent', None)
    if data is None:
        data = getattr(result, 'structured_content', None)
    if data is None:
        data = json.loads(
            '\n'.join(getattr(block, 'text', '') for block in result.content)
        )
    return data.get('result', data) if set(data) == {'result'} else data


async def _check(wheel: Path) -> dict:
    from importlib.metadata import distribution, version

    import anyio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    import oracle3

    dist = distribution('oracle3')
    direct = json.loads(dist.read_text('direct_url.json') or '{}')
    _require(
        not direct.get('dir_info', {}).get('editable'),
        'Editable installs are not wheels',
    )
    wheel_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
    archive = direct.get('archive_info', {})
    installed_hash = archive.get('hashes', {}).get('sha256')
    if installed_hash is None and archive.get('hash', '').startswith('sha256='):
        installed_hash = archive['hash'].split('=', 1)[1]
    _require(
        installed_hash == wheel_hash, 'Installed distribution must match --wheel SHA256'
    )
    package_file = Path(oracle3.__file__).resolve()
    expected_file = Path(dist.locate_file('oracle3/__init__.py')).resolve()
    _require(
        package_file == expected_file, 'Imported oracle3 differs from installed wheel'
    )
    _require(
        Path(sys.prefix).resolve() in package_file.parents,
        'Use the isolated environment where the wheel was installed',
    )
    entry = Path(sys.executable).parent / (
        'oracle3-mcp.exe' if os.name == 'nt' else 'oracle3-mcp'
    )
    _require(entry.is_file(), 'Installed oracle3-mcp console entry point is missing')

    with tempfile.TemporaryDirectory(prefix='oracle3-mcp-wheel-') as folder:
        cwd = Path(folder)
        marker = cwd / 'shutdown.json'
        # Launch the installed console entry point after blocking network access.
        # Isolated Python ignores PYTHONPATH and the checkout's current directory.
        bootstrap = """
import json, pathlib, socket, sys
from importlib.metadata import distribution
def deny(*args, **kwargs):
    raise RuntimeError('Network disabled by installed-wheel smoke test')
socket.socket.connect = deny
socket.socket.connect_ex = deny
socket.socket.sendto = deny
socket.create_connection = deny
socket.getaddrinfo = deny
import oracle3
assert pathlib.Path(oracle3.__file__).resolve() == pathlib.Path(sys.argv[2])
entry, marker = sys.argv[1], sys.argv[3]
sys.argv = [entry]
try:
    entrypoint = next(item for item in distribution('oracle3').entry_points
                      if item.name == 'oracle3-mcp' and item.group == 'console_scripts')
    entrypoint.load()()
except SystemExit as exc:
    if exc.code not in (None, 0):
        raise
pathlib.Path(marker).write_text(json.dumps({'exit': 0}))
"""
        parameters = StdioServerParameters(
            command=sys.executable,
            args=['-I', '-c', bootstrap, str(entry), str(package_file), str(marker)],
            cwd=cwd,
            env={'ORACLE3_MCP_LEDGER': str(cwd / 'unused-ledger.json')},
        )
        with anyio.fail_after(45):
            async with stdio_client(parameters) as (read, write):
                async with ClientSession(read, write) as session:
                    initialized = await session.initialize()
                    tools = await session.list_tools()
                    names = {tool.name for tool in tools.tools}
                    _require(
                        {'list_relation_types', 'check_constraint', 'paper_order'}
                        <= names,
                        'Expected tool discovery entries are missing',
                    )
                    relations = _unpack(
                        await session.call_tool('list_relation_types', {})
                    )
                    _require(
                        relations['relations']['implication'] == 'P(A) <= P(B)',
                        'Relation tool returned unexpected data',
                    )
                    check = _unpack(
                        await session.call_tool(
                            'check_constraint',
                            {
                                'relation': 'implication',
                                'contracts': 100,
                                'quotes': [
                                    {'market_id': 'A', 'yes_bid': 0.60},
                                    {'market_id': 'B', 'yes_ask': 0.55},
                                ],
                            },
                        )
                    )
                    _require(
                        check['best']['fees'] == 3.4125
                        and check['best']['net_edge'] == 1.5875,
                        'Installed calculation differs from the frozen example',
                    )
                    _require(
                        check['assessment']
                        == {
                            'relation_verified': False,
                            'execution_verified': False,
                            'conditional_on_relation': True,
                        },
                        'Installed package lacks the current assessment contract',
                    )
            _require(
                marker.is_file(), 'Server did not exit gracefully after stdio shutdown'
            )
            _require(
                not (cwd / 'unused-ledger.json').exists(), 'Pure tools wrote a ledger'
            )
        return {
            'status': 'passed',
            'oracle3_version': dist.version,
            'mcp_version': version('mcp'),
            'wheel_sha256': wheel_hash,
            'transport': 'stdio',
            'network': 'blocked in server',
            'tool_count': len(names),
            'fees': check['best']['fees'],
            'net_edge': check['best']['net_edge'],
            'graceful_shutdown': True,
            'protocol_version': (
                getattr(initialized, 'protocol_version', None)
                or getattr(initialized, 'protocolVersion', None)
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    # Avoid source-tree imports even when this script was invoked from a checkout.
    checkout = Path(__file__).resolve().parents[1]
    sys.path[:] = [
        p
        for p in sys.path
        if p and Path(p).resolve() not in (checkout, checkout / 'scripts')
    ]
    import anyio

    previous_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix='oracle3-mcp-client-') as folder:
        try:
            os.chdir(folder)
            report = anyio.run(_check, wheel)
        finally:
            os.chdir(previous_cwd)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
