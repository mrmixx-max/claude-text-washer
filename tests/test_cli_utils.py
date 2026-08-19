#!/usr/bin/env python3
"""Tests for shared CLI utilities (scripts/cli_utils.py)."""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.cli_utils import (  # noqa: E402
    Colors,
    ProgressBar,
    ProgressIndicator,
    add_common_args,
    add_io_args,
    cli_entry,
    colorize,
    format_duration,
    init_terminal,
    read_input_text,
    section_header,
    write_output_text,
)


class FakeTTY:
    """A minimal file-like object that reports as a TTY for rendering tests."""

    def __init__(self) -> None:
        self.parts: list[str] = []

    def write(self, s: str) -> int:
        self.parts.append(s)
        return len(s)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return True

    @property
    def text(self) -> str:
        return "".join(self.parts)


# --------------------------------------------------------------------------- #
# format_duration
# --------------------------------------------------------------------------- #
def test_format_duration_milliseconds():
    assert format_duration(0.05) == "50ms"
    assert format_duration(0.999) == "999ms"


def test_format_duration_seconds():
    assert format_duration(5.2) == "5.2s"


def test_format_duration_minutes():
    assert format_duration(125.0) == "2m 5.0s"


# --------------------------------------------------------------------------- #
# colorize
# --------------------------------------------------------------------------- #
def test_colorize_disabled(monkeypatch):
    monkeypatch.setattr(Colors, "enabled", False)
    assert colorize("hi", Colors.RED) == "hi"


def test_colorize_enabled(monkeypatch):
    monkeypatch.setattr(Colors, "enabled", True)
    out = colorize("hi", Colors.RED)
    assert out.startswith(Colors.RED)
    assert out.endswith(Colors.RESET)
    assert "hi" in out


def test_colorize_no_color(monkeypatch):
    # Empty color string -> returned as-is even when colour is enabled.
    monkeypatch.setattr(Colors, "enabled", True)
    assert colorize("hi", "") == "hi"


# --------------------------------------------------------------------------- #
# section_header
# --------------------------------------------------------------------------- #
def test_section_header_non_tty(capsys):
    section_header("My Section")
    captured = capsys.readouterr()
    assert "My Section" in captured.out


def test_section_header_tty_renders_banner():
    fake = FakeTTY()
    section_header("Banner", file=fake)
    text = fake.text
    assert "Banner" in text
    assert "=" in text


# --------------------------------------------------------------------------- #
# read / write helpers
# --------------------------------------------------------------------------- #
def test_read_input_text_from_file(tmp_path):
    p = tmp_path / "in.txt"
    p.write_text("hello world", encoding="utf-8")
    assert read_input_text(p) == "hello world"


def test_read_input_text_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_input_text(tmp_path / "nope.txt")


def test_read_input_text_directory_raises(tmp_path):
    with pytest.raises(IsADirectoryError):
        read_input_text(tmp_path)


def test_read_input_text_stdin(monkeypatch):
    monkeypatch.setattr("scripts.cli_utils.sys.stdin", io.StringIO("from stdin"))
    assert read_input_text("-") == "from stdin"
    monkeypatch.setattr("scripts.cli_utils.sys.stdin", io.StringIO("from stdin"))
    assert read_input_text(None) == "from stdin"


def test_write_output_text_creates_dirs(tmp_path):
    p = tmp_path / "sub" / "deep" / "out.txt"
    write_output_text(p, "content here")
    assert p.read_text(encoding="utf-8") == "content here"


# --------------------------------------------------------------------------- #
# ProgressIndicator (spinner)
# --------------------------------------------------------------------------- #
def test_progress_indicator_no_crash_when_disabled():
    with ProgressIndicator("working"):
        pass


def test_progress_indicator_runs_when_enabled(monkeypatch):
    import time

    monkeypatch.setattr(Colors, "enabled", True)
    fake = FakeTTY()
    with ProgressIndicator("spinning", file=fake):
        time.sleep(0.05)
    assert "spinning" in fake.text


# --------------------------------------------------------------------------- #
# ProgressBar
# --------------------------------------------------------------------------- #
def test_progress_bar_state_advances():
    bar = ProgressBar(total=5, label="t", file=FakeTTY())
    bar.start()
    bar.advance()
    assert bar._done == 1
    bar.advance(2)
    assert bar._done == 3
    bar.finish()
    assert bar._done == 5


def test_progress_bar_context_manager():
    with ProgressBar(total=3, label="ctx") as bar:
        bar.advance()
        bar.advance(2)
    assert bar._done == 3


def test_progress_bar_ascii_render(monkeypatch):
    # Force the stdlib fallback path (no rich) with a TTY-like sink.
    monkeypatch.setattr("scripts.cli_utils._HAS_RICH", False)
    monkeypatch.setattr("scripts.cli_utils._RICH_CONSOLE", None)
    fake = FakeTTY()
    with ProgressBar(total=4, label="wash", file=fake) as bar:
        bar.advance()
        bar.advance()
        bar.finish()
    out = fake.text
    assert "wash" in out
    assert "100%" in out
    assert "done" in out


def test_progress_bar_zero_total_does_not_crash():
    bar = ProgressBar(total=0, label="zero")
    with bar:
        bar.advance()


# --------------------------------------------------------------------------- #
# argparse helpers
# --------------------------------------------------------------------------- #
def test_add_common_args_parses():
    p = argparse.ArgumentParser()
    add_common_args(p, include_temperature=True)
    args = p.parse_args(["--model", "qwen-coder", "--temperature", "0.5", "--list-models"])
    assert args.model == "qwen-coder"
    assert args.temperature == 0.5
    assert args.list_models is True


def test_add_common_args_temperature_optional():
    p = argparse.ArgumentParser()
    add_common_args(p, include_temperature=False)
    args = p.parse_args(["--model", "llama3.2"])
    assert not hasattr(args, "temperature")
    assert args.model == "llama3.2"


def test_add_io_args_parses():
    p = argparse.ArgumentParser()
    add_io_args(p)
    args = p.parse_args(["input.txt", "-o", "out.txt"])
    assert args.input == "input.txt"
    assert args.output == "out.txt"


def test_add_io_args_default_none():
    p = argparse.ArgumentParser()
    add_io_args(p)
    args = p.parse_args([])
    assert args.input is None
    assert args.output is None


# --------------------------------------------------------------------------- #
# cli_entry
# --------------------------------------------------------------------------- #
def test_cli_entry_passes_through():
    ran = []
    with cli_entry():
        ran.append(True)
    assert ran == [True]


def test_cli_entry_handles_file_not_found(capsys):
    with pytest.raises(SystemExit):
        with cli_entry():
            raise FileNotFoundError("missing file")
    err = capsys.readouterr().err
    assert "missing file" in err


def test_cli_entry_handles_directory_error(capsys):
    with pytest.raises(SystemExit):
        with cli_entry():
            raise IsADirectoryError("is a dir")
    err = capsys.readouterr().err
    assert "is a dir" in err


def test_cli_entry_handles_value_error(capsys):
    with pytest.raises(SystemExit):
        with cli_entry():
            raise ValueError("bad value")
    err = capsys.readouterr().err
    assert "bad value" in err


def test_cli_entry_keyboard_interrupt(capsys):
    with pytest.raises(SystemExit) as exc_info:
        with cli_entry():
            raise KeyboardInterrupt()
    assert exc_info.value.code == 130


# --------------------------------------------------------------------------- #
# init_terminal
# --------------------------------------------------------------------------- #
def test_init_terminal_returns_bool():
    assert isinstance(init_terminal(), bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
