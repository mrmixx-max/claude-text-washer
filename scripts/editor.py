#!/usr/bin/env python3
"""Interactive terminal text editor for claude-text-washer.

Features:
- Real-time cursor navigation (arrow keys, Home/End, PgUp/PgDn)
- Text insertion / deletion at cursor position
- Line wrapping at configurable width
- Status bar: cursor position, character count, live AI score
- Ctrl+S save, Ctrl+Q quick-exit (prompts on unsaved changes)
- Load via ``python scripts/editor.py input.txt``
- Save via ``python scripts/editor.py input.txt --output clean.txt``

Uses ``msvcrt`` on Windows with a ``tty`` + ``termios`` fallback on Unix.
The core editing logic lives in :class:`TextBuffer` (fully unit-testable
without a terminal).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Import the statistical engine for live AI scoring.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stat_engine import analyze_text  # noqa: E402


# --------------------------------------------------------------------------- #
# ANSI colour helpers (works everywhere except very old Windows consoles)
# --------------------------------------------------------------------------- #

class ANSI:
    """Minimal ANSI escape-code constants."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    BG_BLUE = "\033[44m"
    BG_GREY = "\033[48;5;236m"


# --------------------------------------------------------------------------- #
# Platform key helpers
# --------------------------------------------------------------------------- #

_IS_WINDOWS = os.name == "nt"

try:
    import msvcrt as _msvcrt  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — exercised on Windows only
    _msvcrt = None


def _unix_getch() -> str:
    """Read a single keypress on Unix using ``tty`` + ``termios``."""
    import tty
    import termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        char = sys.stdin.read(1)
        if char == "\x1b":  # escape sequence
            char += sys.stdin.read(2)  # e.g. "[A"
        return char
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def getch() -> str:
    """Read one keypress, returning a normalised token.

    Returns either a single character, an escape-sequence string like
    ``\\x1b[A`` (up), or a control-letter string like ``\\x13`` (Ctrl+S).
    """
    if _msvcrt is not None:
        while True:
            ch = _msvcrt.getwch()
            if ch == "\x1b":  # possible escape / arrow
                if _msvcrt.kbhit():
                    nxt = _msvcrt.getwch()
                    if nxt == "[":  # ANSI arrow
                        ch += nxt + _msvcrt.getwch()
                    else:
                        # ESC alone — treat as escape
                        ch = "\x1b"
                        _msvcrt.putwch(nxt)
                return ch
            return ch
    return _unix_getch()


# --------------------------------------------------------------------------- #
# Core editing model — fully unit-testable
# --------------------------------------------------------------------------- #

@dataclass
class CursorPos:
    """Zero-indexed (row, col) position inside the buffer."""
    row: int
    col: int

    def clone(self) -> CursorPos:
        return CursorPos(self.row, self.col)


@dataclass
class TextBuffer:
    """A list-of-lines text buffer with a movable cursor.

    Lines are stored *without* trailing newlines.  Column positions refer to
    the visual/character column of the line (which may be longer than the
    wrap width when wrapped).
    """

    lines: list[str] = field(default_factory=lambda: [""])
    cursor: CursorPos = field(default_factory=lambda: CursorPos(0, 0))
    dirty: bool = False
    wrap_width: int = field(default=80)

    # -- construction ------------------------------------------------------- #

    @classmethod
    def from_text(cls, text: str, wrap_width: int = 80) -> "TextBuffer":
        """Build a buffer from raw text (lines split on ``\\n``)."""
        if text.endswith("\n"):
            text = text[:-1]
        lines = text.split("\n") if text else [""]
        return cls(lines=lines, wrap_width=wrap_width)

    @classmethod
    def from_file(cls, path: str | Path, wrap_width: int = 80) -> "TextBuffer":
        """Build a buffer by reading *path* as UTF-8."""
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_text(text, wrap_width=wrap_width)

    # -- queries ----------------------------------------------------------- #

    @property
    def char_count(self) -> int:
        return sum(len(line) for line in self.lines)

    @property
    def line_count(self) -> int:
        return len(self.lines)

    def _clamped_col(self, row: int, col: int) -> int:
        """Clamp *col* to valid range for *row*."""
        if 0 <= row < len(self.lines):
            return max(0, min(col, len(self.lines[row])))
        return 0

    def clamped_cursor(self) -> CursorPos:
        """Return a copy of the cursor clamped to valid bounds."""
        row = max(0, min(self.cursor.row, len(self.lines) - 1))
        col = self._clamped_col(row, self.cursor.col)
        return CursorPos(row, col)

    # -- mutation ---------------------------------------------------------- #

    def insert(self, text: str) -> None:
        """Insert *text* at the cursor (handles newlines internally)."""
        if not text:
            return
        row, col = self.cursor.row, self.cursor.col
        if row < 0 or row >= len(self.lines):
            return
        current = self.lines[row]
        before = current[:col]
        after = current[col:]

        # Split on newlines so multi-line paste works.
        parts = text.split("\n")
        new_lines = [before + parts[0]]
        for i in range(1, len(parts)):
            new_lines.append(parts[i])
        new_lines[-1] += after

        self.lines[row:row + 1] = new_lines
        self.cursor.row = row + len(parts) - 1
        # Cursor sits at the end of the inserted text on the last new line:
        #  - single-line insert: len(before) + len(parts[-1])  (before is on this line)
        #  - multi-line insert:   len(parts[-1])               (before is not on this line)
        self.cursor.col = len(before) + len(parts[-1]) if len(parts) == 1 else len(parts[-1])
        self.dirty = True

    def backspace(self) -> None:
        """Delete the character before the cursor (backspace)."""
        row, col = self.cursor.row, self.cursor.col
        if row < 0 or row >= len(self.lines):
            return
        if col > 0:
            line = self.lines[row]
            self.lines[row] = line[:col - 1] + line[col:]
            self.cursor.col = col - 1
            self.dirty = True
        elif row > 0:
            # Join with previous line
            prev = self.lines[row - 1]
            self.cursor.col = len(prev)
            self.lines[row - 1] = prev + self.lines[row]
            del self.lines[row]
            self.cursor.row = row - 1
            self.dirty = True

    def delete_char(self) -> None:
        """Delete the character *at* the cursor (forward delete)."""
        row, col = self.cursor.row, self.cursor.col
        if row < 0 or row >= len(self.lines):
            return
        line = self.lines[row]
        if col < len(line):
            self.lines[row] = line[:col] + line[col + 1:]
            self.dirty = True
        elif row + 1 < len(self.lines):
            # Join next line into current
            self.lines[row] = line + self.lines[row + 1]
            del self.lines[row + 1]
            self.dirty = True

    def move_left(self) -> None:
        c = self.clamped_cursor()
        if c.col > 0:
            self.cursor.col = c.col - 1
        elif c.row > 0:
            self.cursor.row = c.row - 1
            self.cursor.col = len(self.lines[self.cursor.row])

    def move_right(self) -> None:
        c = self.clamped_cursor()
        if c.col < len(self.lines[c.row]):
            self.cursor.col = c.col + 1
        elif c.row + 1 < len(self.lines):
            self.cursor.row = c.row + 1
            self.cursor.col = 0

    def move_up(self) -> None:
        c = self.clamped_cursor()
        if c.row > 0:
            self.cursor.row = c.row - 1
            self.cursor.col = self._clamped_col(self.cursor.row, c.col)

    def move_down(self) -> None:
        c = self.clamped_cursor()
        if c.row + 1 < len(self.lines):
            self.cursor.row = c.row + 1
            self.cursor.col = self._clamped_col(self.cursor.row, c.col)

    def move_to_line_start(self) -> None:
        c = self.clamped_cursor()
        self.cursor.col = 0

    def move_to_line_end(self) -> None:
        c = self.clamped_cursor()
        self.cursor.col = len(self.lines[c.row])

    def move_to_top(self) -> None:
        self.cursor.row = 0
        c = self.clamped_cursor()
        self.cursor.col = c.col

    def move_to_bottom(self) -> None:
        self.cursor.row = len(self.lines) - 1
        c = self.clamped_cursor()
        self.cursor.col = c.col

    def page_up(self, screen_rows: int) -> None:
        self.cursor.row = max(0, self.cursor.row - screen_rows + 1)
        c = self.clamped_cursor()
        self.cursor.col = c.col

    def page_down(self, screen_rows: int) -> None:
        self.cursor.row = min(len(self.lines) - 1, self.cursor.row + screen_rows - 1)
        c = self.clamped_cursor()
        self.cursor.col = c.col

    # -- rendering --------------------------------------------------------- #

    def visible_lines(self, width: int) -> list[str]:
        """Return the buffer content wrapped to *width* columns.

        Each source line is hard-wrapped at *width* (splitting on word
        boundaries where possible, falling back to character break).
        """
        result: list[str] = []
        for line in self.lines:
            result.extend(_wrap_text(line, width))
        return result if result else [""]

    def ai_score(self) -> float:
        """Compute the current AI-score (0-100) of the buffer text."""
        text = self.to_text()
        if not text.strip():
            return 0.0
        try:
            return analyze_text(text).ai_score
        except Exception:
            return 0.0

    def to_text(self) -> str:
        return "\n".join(self.lines)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_text(), encoding="utf-8")
        self.dirty = False


# --------------------------------------------------------------------------- #
# Text wrapping helper
# --------------------------------------------------------------------------- #

def _wrap_text(text: str, width: int) -> list[str]:
    """Greedy word-wrap *text* into lines no longer than *width* chars."""
    if width <= 0:
        return [text]
    if len(text) <= width:
        return [text]
    result: list[str] = []
    line = ""
    for word in text.split(" "):
        if not line:
            line = word
        elif len(line) + 1 + len(word) <= width:
            line = line + " " + word
        else:
            result.append(line)
            if len(word) > width:
                # Hard-break very long word
                while len(word) > width:
                    result.append(word[:width])
                    word = word[width:]
                line = word
            else:
                line = word
    if line:
        result.append(line)
    return result


def wrap_text(text: str, width: int) -> list[str]:
    """Public wrapper for :func:`_wrap_text`."""
    return _wrap_text(text, width)


# --------------------------------------------------------------------------- #
# Status bar rendering
# --------------------------------------------------------------------------- #

@dataclass
class StatusInfo:
    """Snapshot of editor state for the status bar."""
    row: int
    col: int
    char_count: int
    line_count: int
    dirty: bool
    ai_score: float
    width: int

    def render(self) -> str:
        pos = f" Ln {self.row + 1}, Col {self.col + 1} "
        counts = f"Chars {self.char_count}  Lines {self.line_count} "
        flag = "*" if self.dirty else " "
        score = f"AI {self.ai_score:.0f}/100 "
        left = f"{flag} {pos}{counts}{score}"
        # Pad to width so the bar clears leftover characters
        right = " Ctrl+S Save  Ctrl+Q Quit  Ctrl+F Find/Fmt "
        bar = left + " " * max(1, self.width - len(left) - len(right)) + right
        return bar[: self.width]


def build_status_info(buf: TextBuffer, width: int) -> StatusInfo:
    return StatusInfo(
        row=buf.cursor.row,
        col=buf.cursor.col,
        char_count=buf.char_count,
        line_count=buf.line_count,
        dirty=buf.dirty,
        ai_score=buf.ai_score(),
        width=width,
    )


# --------------------------------------------------------------------------- #
# Key mapping — translate raw getch() tokens into semantic actions
# --------------------------------------------------------------------------- #

# Control-key tokens we map to named actions.
# Format: (token) -> action_name
CTRL_KEYS: dict[str, str] = {
    "\x13": "save",       # Ctrl+S
    "\x11": "quit",       # Ctrl+Q
    "\x06": "find",       # Ctrl+F
    "\x01": "line_start",  # Ctrl+A
    "\x05": "line_end",   # Ctrl+E
    "\x09": "tab",
    "\n": "newline",
    "\r": "newline",
    "\x7f": "backspace",
    "\x08": "backspace",
}

# Escape / arrow tokens (already normalised by getch())
ARROW_KEYS: dict[str, str] = {
    "\x1b[A": "up",
    "\x1b[B": "down",
    "\x1b[C": "right",
    "\x1b[D": "left",
    "\x1b[H": "line_start",
    "\x1b[F": "line_end",
    "\x1b[5~": "page_up",
    "\x1b[6~": "page_down",
    "\x1b[2~": "insert",
    "\x7f": "backspace",
    "\x08": "backspace",
    "\x1b[3~": "delete",
    "\x1b[4~": "line_end",
}


def classify_key(token: str) -> tuple[str, str | None]:
    """Classify a raw key *token* into ``(kind, payload)``.

    *kind* is one of: ``"char"`` (payload = the char), ``"action"``
    (payload = action name), or ``"unknown"``.
    """
    if token in CTRL_KEYS:
        return "action", CTRL_KEYS[token]
    if token in ARROW_KEYS:
        return "action", ARROW_KEYS[token]
    # Arrow keys on some Windows consoles arrive as "\x00<letter>"
    if len(token) >= 2 and token[0] in ("\x00", "\xe0"):
        mapping = {
            "H": "up", "P": "down", "M": "right", "K": "left",
            "G": "line_start", "O": "line_end",
        }
        if token[1] in mapping:
            return "action", mapping[token[1]]
    if len(token) == 1 and token.isprintable():
        return "char", token
    if token in ("\n", "\r"):
        return "action", "newline"
    return "unknown", token


# --------------------------------------------------------------------------- #
# Screen drawing helpers (also unit-testable)
# --------------------------------------------------------------------------- #

def render_viewport(
    buf: TextBuffer,
    width: int,
    height: int,
    top_row: int = 0,
) -> tuple[list[str], CursorPos, int]:
    """Render *buf* into a viewport of *width*×*height* starting at *top_row*.

    Returns ``(lines, wrapped_cursor, adjusted_top_row)`` where
    *wrapped_cursor* is the screen-space position of the cursor for drawing
    the cursor glyph, and *adjusted_top_row* is the (possibly updated) top
    row needed to keep the cursor visible.

    The status bar occupies the last line, so *height* - 1 lines are used
    for text.
    """
    text_lines = buf.visible_lines(width)
    avail = max(1, height - 1)

    # Compute wrapped cursor position
    wrapped_cursor = _wrapped_cursor_position(buf, width, top_row)

    if wrapped_cursor.vrow < top_row:
        top_row = wrapped_cursor.vrow
    elif wrapped_cursor.vrow >= top_row + avail:
        top_row = wrapped_cursor.vrow - avail + 1

    top_row = max(0, top_row)
    end_row = min(len(text_lines), top_row + avail)
    view = list(text_lines[top_row:end_row])
    while len(view) < avail:
        view.append("")

    screen_cursor = CursorPos(wrapped_cursor.vrow - top_row, wrapped_cursor.col)
    return view, screen_cursor, top_row


@dataclass
class WrappedCursor:
    """Cursor position in *wrapped* (screen) line coordinates."""
    vrow: int   # visual row across all wrapped lines
    col: int    # column on that wrapped line


def _wrapped_cursor_position(buf: TextBuffer, width: int, top_row: int) -> WrappedCursor:
    """Compute the visual (wrapped) row/col of the buffer cursor."""
    vrow = 0
    for i, line in enumerate(buf.lines):
        if i < buf.cursor.row:
            wrapped = _wrap_text(line, width)
            vrow += len(wrapped)
        elif i == buf.cursor.row:
            # Count characters up to cursor col on this source line
            prefix = line[:buf.cursor.col]
            wrapped = _wrap_text(prefix, width)
            vrow += len(wrapped) - 1  # last wrapped line
            if len(wrapped) == 1:
                col = len(prefix)
            else:
                col = len(wrapped[-1])
            return WrappedCursor(vrow, col)
        else:
            break
    # Fallback
    col = buf.cursor.col
    return WrappedCursor(vrow, col)


# --------------------------------------------------------------------------- #
# Interactive run loop
# --------------------------------------------------------------------------- #

def run_editor(
    buf: TextBuffer,
    output_path: str | Path | None,
    width: int | None = None,
    height: int | None = None,
) -> int:
    """Run the interactive editor loop.

    Parameters
    ----------
    buf:
        The :class:`TextBuffer` to edit (mutated in place).
    output_path:
        Where to write on save.  If ``None``, stdout on save is used.
    width, height:
        Terminal dimensions.  Auto-detected if not provided.
    """
    if width is None or height is None:
        w, h = shutil.get_terminal_size((80, 24))
        width, height = (width or w), (height or h)

    buf.wrap_width = width

    # Hide cursor / enter raw mode
    if _IS_WINDOWS:
        os.system("")  # enable ANSI on recent Win10+
    print("\x1b[?25l", end="", flush=True)
    _old_termios = None
    if not _IS_WINDOWS:
        import tty
        import termios
        _fd = sys.stdin.fileno()
        _old_termios = termios.tcgetattr(_fd)
        try:
            tty.setraw(_fd)
        except Exception:
            pass

    state = {"quit": False}

    try:
        top_row = 0
        while not state["quit"]:
            top_row = _draw_and_step(buf, width, height, top_row, output_path, state)
    finally:
        print("\x1b[?25h", end="", flush=True)
        if _old_termios is not None:
            try:
                import termios
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _old_termios)
            except Exception:
                pass
        os.system("")
        print("\n", end="", flush=True)

    return 0


def _draw_and_step(
    buf: TextBuffer,
    width: int,
    height: int,
    top_row: int,
    output_path: str | Path | None,
    state: dict,
) -> int:
    """Draw the screen, process one keypress, return updated *top_row*."""
    view, screen_cursor, top_row = render_viewport(buf, width, height, top_row)
    _render_screen(buf, view, screen_cursor, width, height, output_path)

    token = getch()
    kind, payload = classify_key(token)
    avail = max(1, height - 1)

    if kind == "char":
        buf.insert(payload)
    elif kind == "action":
        action = payload
        if action == "save":
            path = output_path or "output.txt"
            if path:
                buf.save(path)
                _flash(f"Saved → {path}", color=ANSI.GREEN, row=height - 1)
            else:
                _flash("No output path set!", color=ANSI.RED, row=height - 1)
        elif action == "find":
            _simple_confirm("Find/Wrap-toggle: press any key to dismiss.", height)
        elif action == "quit":
            if buf.dirty and _confirm("Unsaved changes! Quit anyway? (y/n) "):
                state["quit"] = True
            else:
                state["quit"] = True
        elif action == "newline":
            buf.insert("\n")
        elif action == "tab":
            buf.insert("    ")
        elif action == "backspace":
            buf.backspace()
        elif action == "delete":
            buf.delete_char()
        elif action == "left":
            buf.move_left()
        elif action == "right":
            buf.move_right()
        elif action == "up":
            buf.move_up()
        elif action == "down":
            buf.move_down()
        elif action == "line_start":
            buf.move_to_line_start()
        elif action == "line_end":
            buf.move_to_line_end()
        elif action == "page_up":
            buf.page_up(avail)
        elif action == "page_down":
            buf.page_down(avail)

    # Recompute scroll so cursor stays visible
    view, screen_cursor, top_row = render_viewport(buf, width, height, top_row)
    return top_row


def _render_screen(buf: TextBuffer, view: list[str], screen_cursor: CursorPos,
                   width: int, height: int, output_path: str | Path | None) -> None:
    """Clear screen and draw the buffer + status bar."""
    print("\x1b[2J\x1b[H", end="", flush=True)
    for i, line in enumerate(view):
        if i == screen_cursor.row and screen_cursor.col <= width:
            before = line[:screen_cursor.col]
            after = line[screen_cursor.col:]
            print(f"{ANSI.BG_BLUE}{before}{ANSI.RESET}{after}")
        else:
            print(line)
    info = build_status_info(buf, width)
    print(f"{ANSI.BG_GREY}{info.render()}{ANSI.RESET}")


def _flash(msg: str, color: str = ANSI.YELLOW, row: int = 0) -> None:
    """Print a transient status-line message at *row*."""
    print(f"\x1b[{row + 1};1H{color}{msg}{ANSI.RESET}", end="", flush=True)


def _confirm(prompt: str) -> bool:
    print(prompt, end="", flush=True)
    if _msvcrt is not None:
        ch = _msvcrt.getwch()
    else:
        import tty
        import termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    print(ch, end="", flush=True)
    return ch.lower() == "y"


def _simple_confirm(msg: str, height: int = 24) -> None:
    """Print an info message and wait for any single keypress."""
    print(f"\x1b[{height};1H{ANSI.YELLOW}{msg}{ANSI.RESET}", end="", flush=True)
    getch()


# --------------------------------------------------------------------------- #
# ANSI screen control (for direct terminal writes)
# --------------------------------------------------------------------------- #

def clear_screen() -> None:
    print("\033[2J\033[H", end="", flush=True)


def save_cursor() -> None:
    print("\033[s", end="", flush=True)


def restore_cursor() -> None:
    print("\033[u", end="", flush=True)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Interactive text editor for AI-text washing")
    p.add_argument("input", nargs="?", help="Input text file to load")
    p.add_argument("-o", "--output", help="Output file (Ctrl+S saves here)")
    p.add_argument("--width", type=int, default=None, help="Force editor width")
    p.add_argument("--height", type=int, default=None, help="Force editor height")
    return p


def main() -> int:
    args = build_parser().parse_args()
    output_path = args.output
    if args.input:
        buf = TextBuffer.from_file(args.input)
    else:
        buf = TextBuffer()
    return run_editor(buf, output_path, args.width, args.height)


if __name__ == "__main__":
    raise SystemExit(main())
