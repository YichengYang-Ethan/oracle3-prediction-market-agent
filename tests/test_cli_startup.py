"""MCP discovery must work without importing unrelated execution providers."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize('arguments', [['--help'], ['--version'], ['mcp', '--help']])
def test_cli_discovery_does_not_import_execution_modules(arguments):
    # A fresh interpreter prevents other tests' imports from hiding a regression.
    script = """
import importlib.abc
import sys
from click.testing import CliRunner

class BlockExecutionImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith((
            'oracle3.trader', 'oracle3.strategy', 'oracle3.cli.agent_commands',
            'oracle3.cli.monitor', 'oracle3.cli.research_commands',
            'solana', 'solders', 'py_clob_client', 'openai',
        )):
            raise AssertionError('Unexpected startup import: ' + fullname)

sys.meta_path.insert(0, BlockExecutionImports())
from oracle3.cli.cli import cli
result = CliRunner().invoke(cli, sys.argv[1:])
print(result.output)
if result.exception:
    raise result.exception
assert result.exit_code == 0
"""
    result = subprocess.run(
        [sys.executable, '-c', script, *arguments],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if arguments == ['--help']:
        for command in ('market', 'paper', 'live', 'research', 'mcp'):
            assert command in result.stdout


def test_lazy_command_resolves_and_runs(monkeypatch):
    import click
    from click.testing import CliRunner

    from oracle3.cli import cli as cli_module

    imports = []

    @click.command()
    def market():
        click.echo('selected market command')

    class Commands:
        pass

    module = Commands()
    module.market = market

    def load(name):
        imports.append(name)
        return module

    monkeypatch.setattr(cli_module, 'import_module', load)
    group = cli_module._LazyGroup()
    result = CliRunner().invoke(group, ['market'])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == 'selected market command'
    assert imports == ['oracle3.cli.market_commands']
    assert group.commands['market'] is market
