#!/usr/bin/env python3
"""Shared CLI utilities for claude-text-washer.

Provides:
  - Centralized ANSI color constants and colored-print helpers
  - ``read_input_text`` / ``write_output_text`` — safe file/stdin I/O
  - ``add_common_args`` / ``add_io_args`` — shared argparse argument groups
  - ``ProgressIndicator`` — a spinner for long-running LLM calls
  - ``ProgressBar`` — a single-bar progress indicator (rich if available)
  - ``format_duration`` — human-readable duration formatting
  - ``section_header`` — colored section banner helper
  - ``init_terminal`` — colour initialisation (colorama on Windows)

Design goals
------------
* Optional third-party deps (``rich``/``colorama``); stdlib fallback so the
  module imports and works everywhere Python 3.11 runs.
* Every public function is unit-testable without a terminal.
* Scripts import from here to avoid the duplicated argparse / file-read /
  color boilerplate that previously appeared in every CLI entry point.
"""
from __future__ import annotations

import argparse
import itertools
import os
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path


# --------------------------------------------------------------------------- #
# Optional dependency resolution — colorama on Windows, rich for bars
# --------------------------------------------------------------------------- #

_IS_WINDOWS = os.name == "nt"

# colorama wraps stdout/stderr so ANSI escape codes render on legacy Windows
# consoles.  If it isn't installed we simply skip (modern Win10+ consoles
# understand ANSI natively, and every non-Windows terminal already does).
try:
    import colorama as _colorama  # type: ignore[import-not-found]
    _HAS_COLORAMA = True
except Exception:  # pragma: no cover — optional dependency
    _HAS_COLORAMA = False
    _colorama = None


def init_terminal() -> bool:
    """Initialise colour support (colorama on Windows).

    Safe to call multiple times; returns ``True`` when colour is enabled.
    """
    if _HAS_COLORAMA and _IS_WINDOWS:
        try:
            _colorama.init()
            Colors.enabled = True
            return True
        except Exception:  # pragma: no cover
            pass
    return Colors.enabled


# When ``rich`` is installed we delegate progress-bar painting to it for a
# nicer live experience; otherwise we fall back to a tiny stdlib bar/spinner.
try:
    from rich.console import Console as _RichConsole  # type: ignore[import-not-found]
    _RICH_CONSOLE = _RichConsole(stderr=True, soft_wrap=True)
    _HAS_RICH = True
except Exception:  # pragma: no cover — optional dependency
    _HAS_RICH = False
    _RICH_CONSOLE = None


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
def print_error(msg: str, file=None) -> None:
    """Print an error message in red."""
    if file is None:
        file = sys.stderr
    print(f"{Colors.RED}{Colors.BOLD}x{Colors.RESET} {Colors.RED}{msg}{Colors.RESET}", file=file)


def print_success(msg: str, file=None) -> None:
    """Print a success message in green."""
    if file is None:
        file = sys.stdout
    print(f"{Colors.GREEN}{Colors.BOLD}v{Colors.RESET} {Colors.GREEN}{msg}{Colors.RESET}", file=file)


def print_info(msg: str, file=None) -> None:
    """Print an informational message in cyan."""
    if file is None:
        file = sys.stdout
    print(f"{Colors.CYAN}i{Colors.RESET} {msg}", file=file)


def print_warning(msg: str, file=None) -> None:
    """Print a warning message in yellow."""
    if file is None:
        file = sys.stderr
    print(f"{Colors.YELLOW}{Colors.BOLD}!{Colors.RESET} {Colors.YELLOW}{msg}{Colors.RESET}", file=file)


def print_step(msg: str, file=None) -> None:
    """Print a progress step in blue."""
    if file is None:
        file = sys.stdout
    print(f"{Colors.BLUE}{Colors.BOLD}>{Colors.RESET} {Colors.BLUE}{msg}{Colors.RESET}", file=file, flush=True)


def section_header(title: str, char: str = "=", file=None) -> None:
    """Print a coloured section banner to *file*.

    The banner width adapts to the terminal; on a non-tty it degrades to a
    plain ``title`` line so test output stays clean.
    """
    if file is None:
        file = sys.stdout
    width = shutil.get_terminal_size((80, 24)).columns if file.isatty() else 0
    if width == 0 or width < 20:
        print(title, file=file)
        return
    bar = char * max(0, width - len(title) - 2)
    print(
        f"{Colors.BOLD}{Colors.CYAN}{title} {bar}{Colors.RESET}",
        file=file,
        flush=True,
    )


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
# Progress indicator (spinner)
# --------------------------------------------------------------------------- #
class ProgressIndicator:
    """A lightweight spinner shown on stderr during long-running operations.

    Usage::

        with ProgressIndicator("Washing text (pass 1/3)..."):
            result = wash_pass(text)
    """

    _SPINNERS = ("|/-\\", "⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏")

    def __init__(self, message: str = "", spinner: str = "dots", file=sys.stderr) -> None:
        self.message = message
        self.spinner_chars = self._SPINNERS[1].split() if spinner == "dots" else self._SPINNERS[0].split()
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
# Progress bar (rich-aware)
# --------------------------------------------------------------------------- #
class ProgressBar:
    """A simple progress bar for multi-step operations.

    Renders with ``rich`` when available and the output is a TTY; otherwise
    falls back to a minimal ASCII bar written to stderr (or suppresses
    entirely when not a TTY).

    Usage::

        bar = ProgressBar(total=3, label="washing")
        for i, item in enumerate(items):
            do_work(item)
            bar.advance()
    """

    def __init__(
        self,
        total: int,
        *,
        label: str = "working",
        file=sys.stderr,
        width: int | None = None,
    ) -> None:
        if total <= 0:
            total = 1
        self.total = total
        self.label = label
        self._file = file
        self._width = width
        self._done = 0
        self._start: float | None = None
        self._rich_task = None
        self._rich_live = None

    def _terminal_width(self) -> int:
        if self._width is not None:
            return self._width
        return shutil.get_terminal_size((80, 24)).columns

    def _enabled(self) -> bool:
        return bool(self._file.isatty())

    def start(self) -> "ProgressBar":
        """Begin the bar (context-manager friendly)."""
        self._start = time.time()
        if _HAS_RICH and self._enabled():
            from rich.progress import (  # type: ignore[import-not-found]
                BarColumn,
                Progress,
                TextColumn,
                TimeElapsedColumn,
            )
            self._rich_live = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                console=_RICH_CONSOLE,
            )
            self._rich_task = self._rich_live.add_task(self.label, total=self.total)
            self._rich_live.start()
        elif self._enabled():
            self._file.write(f"\r{self.label}: [----------] {self._render_pct()}\n")
            self._file.flush()
        return self

    def _render_pct(self) -> str:
        pct = min(100, int(self._done / self.total * 100)) if self.total else 0
        return f"{pct:3d}%"

    def advance(self, n: int = 1) -> None:
        """Mark *n* steps as complete and re-render."""
        self._done = min(self.total, self._done + n)
        if self._rich_task is not None and self._rich_live is not None:
            self._rich_live.update(self._rich_task, completed=self._done)
        elif self._enabled():
            w = self._terminal_width()
            filled = int((self._done / self.total) * max(0, w - 20)) if self.total else 0
            bar_w = max(1, w - 20)
            bar = "#" * filled + "-" * (bar_w - filled)
            eta = ""
            if self._start is not None and self._done > 0 and self._done < self.total:
                per = (time.time() - self._start) / self._done
                eta = f" ETA {per * (self.total - self._done):.0f}s"
            self._file.write(
                f"\r{Colors.DIM}{self.label}: [{bar}] {self._render_pct()}{eta}{Colors.RESET}"
            )
            self._file.flush()

    def finish(self) -> None:
        """Stop the bar and print a completion line."""
        self._done = self.total
        if self._rich_live is not None:
            self._rich_live.update(self._rich_task, completed=self.total)
            self._rich_live.stop()
            self._rich_live = None
            self._rich_task = None
        elif self._enabled():
            elapsed = format_duration((time.time() - self._start) if self._start else 0)
            self._file.write(
                f"\r{Colors.GREEN}{self.label}: [done] {self._render_pct()} "
                f"({elapsed}){Colors.RESET}\n"
            )
            self._file.flush()

    # -- context manager protocol ------------------------------------------- #
    def __enter__(self) -> "ProgressBar":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.finish()


# --------------------------------------------------------------------------- #
# Shared argparse helpers
# --------------------------------------------------------------------------- #
def _import_model_helpers():
    """Import model-pool helpers, tolerant of package vs direct execution."""
    try:
        from ollama_utils import get_model_names, get_default_model  # type: ignore
        return get_model_names, get_default_model
    except ImportError:
        from scripts.ollama_utils import get_model_names, get_default_model  # type: ignore
        return get_model_names, get_default_model


def _safe_default_model() -> str:
    """Return the default model name without crashing if models.yaml is absent."""
    try:
        _, get_default_model = _import_model_helpers()
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
