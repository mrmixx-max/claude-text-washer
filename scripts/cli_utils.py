#!/usr/bin/env python3
"""Shared CLI utilities for claude-text-washer.

Provides:
  - Centralized ANSI color constants and colored-print helpers
  - ``read_input_text`` — safe file/stdin reading with error handling
  - ``add_common_args`` — shared argparse argument group (--model, --list-models, --temperature)
  - ``ProgressIndicator`` — a spinner / progress bar for long-running LLM calls
  - ``format_duration`` — human-readable duration formatting

Design goals
------------
* No third-party dependencies (stdlib only).
* Every public function is unit-testable without a terminal.
* Scripts import from here to avoid the duplicated argparse / file-read /
  color boilerplate that previously appeared in every CLI entry point.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path


# --------------------------------------------------------------------------- #
# ANSI colours
# --------------------------------------------------------------------------- #

class Colors:
    """ANSI escape-code constants (centralized for the whole codebase)."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"

    # Background
    BG_BLUE = "\033[44m"
    BG_GREY = "\033[48;5;236m"
    BG_RED = "\033[41m"

    # Whether colour is enabled (auto-disables when stdout is not a TTY).
    enabled: bool = sys.stdout.isatty()


def colorize(text: str, color: str) -> str:
    """Wrap *text* in an ANSI colour code if colour output is enabled."""
    if not Colors.enabled or not color:
        return text
    return f"{color}{text}{Colors.RESET}"


# --------------------------------------------------------------------------- #
# Colored output helpers
# --------------------------------------------------------------------------- #

def print_error(msg: str, file=sys.stderr) -> None:
    """Print an error message in red."""
    print(f"{Colors.RED}{Colors.BOLD}✗{Colors.RESET} {Colors.RED}{msg}{Colors.RESET}", file=file)


def print_success(msg: str, file=sys.stdout) -> None:
    """Print a success message in green."""
    print(f"{Colors.GREEN}{Colors.BOLD}✓{Colors.RESET} {Colors.GREEN}{msg}{Colors.RESET}", file=file)


def print_info(msg: str, file=sys.stdout) -> None:
    """Print an informational message in cyan."""
    print(f"{Colors.CYAN}ℹ{Colors.RESET} {msg}", file=file)


def print_warning(msg: str, file=sys.stderr) -> None:
    """Print a warning message in yellow."""
    print(f"{Colors.YELLOW}{Colors.BOLD}⚠{Colors.RESET} {Colors.YELLOW}{msg}{Colors.RESET}", file=file)


def print_step(msg: str, file=sys.stdout) -> None:
    """Print a progress step in blue."""
    print(f"{Colors.BLUE}{Colors.BOLD}→{Colors.RESET} {Colors.BLUE}{msg}{Colors.RESET}", file=file, flush=True)


# --------------------------------------------------------------------------- #
# Safe input reading
# --------------------------------------------------------------------------- #

def read_input_text(path: str | Path | None, *, allow_stdin: str = "-") -> str:
    """Read text from *path* or stdin with user-friendly error handling.

    Parameters
    ----------
    path:
        File path to read, or ``None`` / ``allow_stdin`` to read from stdin.
    allow_stdin:
        The sentinel value (default ``"-"``) that requests stdin.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    IsADirectoryError
        If *path* points to a directory.
    UnicodeDecodeError
        If the file cannot be decoded as UTF-8.
    """
    if path is None or path == allow_stdin:
        return sys.stdin.read()

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if p.is_dir():
        raise IsADirectoryError(f"Input path is a directory: {path}")
    return p.read_text(encoding="utf-8")


def write_output_text(path: str | Path, text: str) -> None:
    """Write *text* to *path*, creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Duration formatting
# --------------------------------------------------------------------------- #

def format_duration(seconds: float) -> str:
    """Format a duration in seconds into a human-readable string."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remainder = seconds % 60
    return f"{minutes}m {remainder:.1f}s"


# --------------------------------------------------------------------------- #
# Progress indicator
# --------------------------------------------------------------------------- #

class ProgressIndicator:
    """A lightweight spinner shown on stderr during long-running operations.

    Usage::

        with ProgressIndicator("Washing text (pass 1/3)..."):
            result = wash_pass(text)
    """

    _SPINNERS = ("⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇", "🕐 🕑 🕒 🕓 🕔 🕕 🕖 🕗 🕘 🕙 🕚 🕛").split()

    def __init__(self, message: str = "", spinner: str = "dots", file=sys.stderr) -> None:
        self.message = message
        self.spinner_chars = self._SPINNERS[0 if spinner == "dots" else 1].split()
        self._file = file
        self._running = False
        self._thread: threading.Thread | None = None

    def _spin(self) -> None:
        for frame in itertools.cycle(self.spinner_chars):
            if not self._running:
                break
            self._file.write(f"\r{Colors.DIM}{frame}{Colors.RESET} {self.message}")
            self._file.flush()
            threading.Event().wait(0.1)

    def __enter__(self) -> "ProgressIndicator":
        if not Colors.enabled:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        # Clear the spinner line
        self._file.write("\r" + " " * (len(self.message) + 4) + "\r")
        self._file.flush()

    def update(self, message: str) -> None:
        """Update the spinner message while running."""
        self.message = message


@contextmanager
def progress(message: str, *, spinner: str = "dots", file=sys.stderr):
    """Context manager that shows a spinner while a block executes."""
    indicator = ProgressIndicator(message, spinner, file)
    with indicator:
        yield


# --------------------------------------------------------------------------- #
# Shared argparse helpers
# --------------------------------------------------------------------------- #

def _safe_default_model() -> str:
    """Return the default model name without crashing if models.yaml is absent."""
    try:
        from ollama_utils import get_default_model
        return get_default_model()
    except Exception:
        return "llama3.2"


def add_common_args(parser: argparse.ArgumentParser, *, include_temperature: bool = True) -> None:
    """Attach the standard ``--model`` / ``--list-models`` / ``--temperature`` args.

    Parameters
    ----------
    parser:
        The ArgumentParser to extend.
    include_temperature:
        Whether to add the ``--temperature`` argument.
    """
    from ollama_utils import get_model_names, get_default_model

    model_help = f"Ollama model to use (default: {_safe_default_model()})"
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help=model_help,
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available Ollama models and exit",
    )
    if include_temperature:
        parser.add_argument(
            "--temperature",
            type=float,
            default=None,
            metavar="N",
            help="Sampling temperature (default: model-specific)",
        )


def add_io_args(parser: argparse.ArgumentParser) -> None:
    """Attach standard ``input`` / ``-o --output`` arguments."""
    parser.add_argument(
        "input",
        nargs="?",
        help="Input text file (or '-' for stdin)",
    )
    parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Output file (default: stdout)",
    )


# --------------------------------------------------------------------------- #
# CLI entry helpers
# --------------------------------------------------------------------------- #

@contextmanager
def cli_entry():
    """Context manager for CLI entry points.

    Catches common errors and prints them in red, exiting with code 1.
    Usage::

        def main():
            with cli_entry():
                ...  # may raise FileNotFoundError, ValueError, etc.
    """
    try:
        yield
    except FileNotFoundError as exc:
        print_error(str(exc))
        sys.exit(1)
    except IsADirectoryError as exc:
        print_error(str(exc))
        sys.exit(1)
    except (ValueError, KeyError) as exc:
        print_error(str(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        print_warning("Interrupted.", file=sys.stderr)
        sys.exit(130)
    except BrokenPipeError:
        # Allow `wash ... | head` without a traceback.
        sys.exit(0)


def is_windows() -> bool:
    """Return True if running on Windows."""
    return os.name == "nt"
