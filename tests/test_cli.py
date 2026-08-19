"""Tests for the unified :mod:`scripts.cli` entry point.

Verifies that a single ``claude-washer`` dispatcher routes subcommands to the
right module, rejects unknown commands, and surfaces per-subcommand help.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts import cli

FIXTURES = Path(__file__).resolve().parents[1] / "_fixtures"

EXPECTED_COMMANDS = {"scan", "wash", "pipeline", "file", "chat", "edit", "stat", "prompt"}


def test_subcommands_registered():
    assert EXPECTED_COMMANDS <= set(cli.SUBCOMMANDS)
    for cmd, (mod, desc) in cli.SUBCOMMANDS.items():
        assert mod
        assert desc


def test_no_args_shows_help(capsys):
    assert cli.main([]) == 0
    assert "claude-text-washer" in capsys.readouterr().out


def test_help_and_version_flags():
    assert cli.main(["--help"]) == 0
    assert cli.main(["--version"]) == 0


def test_unknown_command_returns_error(capsys):
    assert cli.main(["bogus"]) == 2
    err = capsys.readouterr().err
    assert "unknown command" in err.lower()


def test_subcommand_module_loads():
    # Every registered subcommand module must be importable and expose ``main``
    # (the only contract the unified CLI dispatch relies on).
    for cmd, (modname, _desc) in cli.SUBCOMMANDS.items():
        mod = cli._load_module(modname)
        assert callable(getattr(mod, "main", None)), f"{modname} missing main()"


@pytest.mark.parametrize("cmd", ["file", "wash", "scan", "pipeline"])
def test_subcommand_help_exits_zero(cmd):
    with pytest.raises(SystemExit) as e:
        cli.main([cmd, "--help"])
    assert e.value.code == 0


def test_dispatch_file_dry_run(capsys):
    # End-to-end: CLI -> file_washer -> format auto-detection -> analysis.
    rc = cli.main(["file", "--dry-run", str(FIXTURES / "sample.md")])
    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "md" in out
