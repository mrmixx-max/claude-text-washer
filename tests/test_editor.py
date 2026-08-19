#!/usr/bin/env python3
"""Tests for the interactive editor's pure, terminal-independent logic.

The editor's TextBuffer / wrapping / key-classification / rendering helpers
are fully unit-testable without a TTY, so we cover them here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.editor import (  # noqa: E402
    ANSI,
    ARROW_KEYS,
    CTRL_KEYS,
    CursorPos,
    StatusInfo,
    TextBuffer,
    build_status_info,
    classify_key,
    render_fullscreen_preview,
    render_viewport,
    wrap_text,
)


# --------------------------------------------------------------------------- #
# CursorPos
# --------------------------------------------------------------------------- #
def test_cursor_clone_independent():
    c = CursorPos(2, 3)
    c2 = c.clone()
    c2.row = 9
    assert c.row == 2


# --------------------------------------------------------------------------- #
# TextBuffer basics
# --------------------------------------------------------------------------- #
def test_from_text_single_line():
    buf = TextBuffer.from_text("hello")
    assert buf.lines == ["hello"]
    assert buf.char_count == 5
    assert buf.line_count == 1

def test_from_text_multi_line():
    buf = TextBuffer.from_text("a\nb\nc")
    assert buf.lines == ["a", "b", "c"]
    assert buf.line_count == 3

def test_from_text_trailing_newline():
    buf = TextBuffer.from_text("a\n")
    assert buf.lines == ["a"]

def test_from_text_empty():
    buf = TextBuffer.from_text("")
    assert buf.lines == [""]
    assert buf.char_count == 0

def test_word_count():
    buf = TextBuffer.from_text("hello world\nfoo bar baz")
    assert buf.word_count == 5

def test_to_text_roundtrip():
    text = "line one\nline two"
    assert TextBuffer.from_text(text).to_text() == text

def test_stats_returns_expected_keys():
    buf = TextBuffer.from_text("alpha beta gamma")
    s = buf.stats()
    assert s["chars"] == len("alpha beta gamma")
    assert s["words"] == 3
    assert s["lines"] == 1
    assert s["dirty"] is False
    assert 0.0 <= s["ai_score"] <= 100.0


def test_stats_empty_text_zero_score():
    buf = TextBuffer.from_text("")
    assert buf.stats()["ai_score"] == 0.0


# --------------------------------------------------------------------------- #
# TextBuffer mutation
# --------------------------------------------------------------------------- #
def test_insert_single_char_at_start():
    buf = TextBuffer.from_text("hello")
    buf.insert("X")
    assert buf.to_text() == "Xhello"
    assert buf.cursor.col == 1
    assert buf.dirty is True

def test_insert_multiline_paste():
    buf = TextBuffer.from_text("ab")
    buf.cursor = CursorPos(0, 1)  # a|b
    buf.insert("X\nY")
    assert buf.lines == ["aX", "Yb"]
    assert buf.cursor == CursorPos(1, 1)

def test_insert_at_end_appends():
    buf = TextBuffer.from_text("hi")
    buf.cursor = CursorPos(0, 2)
    buf.insert(" there")
    assert buf.to_text() == "hi there"

def test_backspace_deletes_prev_char():
    buf = TextBuffer.from_text("hello")
    buf.cursor = CursorPos(0, 3)
    buf.backspace()
    assert buf.to_text() == "helo"
    assert buf.cursor.col == 2

def test_backspace_joins_previous_line():
    buf = TextBuffer.from_text("ab\ncd")
    buf.cursor = CursorPos(1, 0)
    buf.backspace()
    assert buf.lines == ["abcd"]
    assert buf.cursor == CursorPos(0, 2)

def test_delete_char_forward():
    buf = TextBuffer.from_text("hello")
    buf.cursor = CursorPos(0, 1)
    buf.delete_char()
    assert buf.to_text() == "hllo"

def test_delete_char_joins_next_line():
    buf = TextBuffer.from_text("ab\ncd")
    buf.cursor = CursorPos(0, 2)
    buf.delete_char()
    assert buf.lines == ["abcd"]

def test_cursor_movement_clamped():
    buf = TextBuffer.from_text("hello\nworld")
    buf.cursor = CursorPos(0, 0)
    buf.move_left()  # stays
    assert buf.cursor == CursorPos(0, 0)
    buf.move_up()
    assert buf.cursor == CursorPos(0, 0)
    buf.move_down()
    assert buf.cursor == CursorPos(1, 0)  # column preserved (clamped from 0)

def test_move_to_top_and_bottom():
    buf = TextBuffer.from_text("a\nb\nc\nd")
    buf.cursor = CursorPos(1, 0)
    buf.move_to_bottom()
    assert buf.cursor.row == 3
    buf.move_to_top()
    assert buf.cursor.row == 0

def test_page_up_down():
    buf = TextBuffer.from_text("\n".join(f"line {i}" for i in range(20)))
    buf.cursor = CursorPos(15, 0)
    buf.page_up(5)
    assert buf.cursor.row == 11
    buf.page_down(5)
    assert buf.cursor.row == 15

def test_page_up_clamped_to_zero():
    buf = TextBuffer.from_text("a\nb")
    buf.cursor = CursorPos(0, 0)
    buf.page_up(5)
    assert buf.cursor.row == 0

def test_save_writes_text(tmp_path):
    buf = TextBuffer.from_text("content")
    buf.dirty = True
    out = tmp_path / "o.txt"
    buf.save(out)
    assert out.read_text(encoding="utf-8") == "content"
    assert buf.dirty is False


# --------------------------------------------------------------------------- #
# Wrapping
# --------------------------------------------------------------------------- #
def test_wrap_short_text_unchanged():
    assert wrap_text("short", 80) == ["short"]

def test_wrap_text_word_boundary():
    out = wrap_text("aa bb cc", 5)
    assert out == ["aa bb", "cc"]

def test_wrap_text_breaks_long_word():
    out = wrap_text("abcdefgh", 3)
    assert out == ["abc", "def", "gh"]

def test_wrap_text_width_zero_returns_text():
    assert wrap_text("hello", 0) == ["hello"]

def test_visible_lines_wraps():
    buf = TextBuffer.from_text("aa bb cc dd")
    assert buf.visible_lines(5) == ["aa bb", "cc dd"]


# --------------------------------------------------------------------------- #
# Key classification
# --------------------------------------------------------------------------- #
def test_classify_char():
    kind, payload = classify_key("x")
    assert kind == "char"
    assert payload == "x"

def test_classify_ctrl_save():
    kind, payload = classify_key(CTRL_KEYS and "\x13")
    # Ctrl+S token
    assert classify_key("\x13") == ("action", "save")

def test_classify_ctrl_preview():
    assert classify_key("\x10") == ("action", "preview")

def test_classify_arrow_up():
    assert classify_key("\x1b[A") == ("action", "up")

def test_classify_arrow_esc_sequences():
    assert classify_key("\x1b[B") == ("action", "down")
    assert classify_key("\x1b[C") == ("action", "right")
    assert classify_key("\x1b[D") == ("action", "left")
    assert classify_key("\x1b[H") == ("action", "line_start")

def test_classify_unknown():
    kind, payload = classify_key("\x00\xFF")
    assert kind == "unknown"

def test_classify_newline():
    assert classify_key("\n") == ("action", "newline")

def test_ctrl_keys_contains_expected():
    assert CTRL_KEYS["\x13"] == "save"
    assert CTRL_KEYS["\x11"] == "quit"
    assert CTRL_KEYS["\x10"] == "preview"


# --------------------------------------------------------------------------- #
# Status bar
# --------------------------------------------------------------------------- #
def test_status_info_render_contains_fields():
    buf = TextBuffer.from_text("hello world foo")
    info = build_status_info(buf, 100)
    rendered = info.render()
    assert "Words 3" in rendered
    assert "Chars 15" in rendered  # "hello world foo" is 15 characters
    assert "Lines 1" in rendered
    assert "AI" in rendered
    assert "Ctrl+P Preview" in rendered
    assert "Ctrl+S Save" in rendered
    assert len(rendered) <= 100  # width respected

def test_status_info_dirty_flag():
    buf = TextBuffer.from_text("abc")
    buf.dirty = True
    info = build_status_info(buf, 80)
    assert "*" in info.render()
    buf.dirty = False
    assert "*" not in build_status_info(buf, 80).render()


def test_status_info_truncated_to_width():
    buf = TextBuffer.from_text("x")
    info = build_status_info(buf, 40)
    assert len(info.render()) == 40


# --------------------------------------------------------------------------- #
# Viewport rendering
# --------------------------------------------------------------------------- #
def test_render_viewport_pads_short_content():
    buf = TextBuffer.from_text("hello")
    view, cursor, top = render_viewport(buf, 40, 5, 0)
    assert len(view) == 4  # height-1 for status bar
    assert view[0] == "hello"
    assert cursor == CursorPos(0, 0)

def test_render_viewport_keeps_cursor_visible():
    buf = TextBuffer.from_text("\n".join(f"line{i}" for i in range(10)))
    buf.cursor = CursorPos(8, 0)
    view, cursor, top = render_viewport(buf, 40, 5, 0)
    assert cursor.row <= 3  # cursor within viewport rows
    assert 0 <= top

def test_render_viewport_width_truncation():
    buf = TextBuffer.from_text("hello")
    view, cursor, top = render_viewport(buf, 40, 3, 0)
    # height=3 -> 2 text lines
    assert len(view) == 2


# --------------------------------------------------------------------------- #
# WYSIWYG preview
# --------------------------------------------------------------------------- #
def test_render_fullscreen_preview_structure():
    buf = TextBuffer.from_text("Hello world. This is a test.\nSecond line here.")
    lines = render_fullscreen_preview(buf, 40, 10)
    assert len(lines) == 10  # padded to height
    assert "WYSIWYG Preview" in lines[0]
    assert "Hello world. This is a test." in lines[1]
    # footer line contains stats
    footer = lines[-1]
    assert "Chars" in footer
    assert "Words" in footer
    assert "AI" in footer
    assert "clean" in footer  # not dirty

def test_render_fullscreen_preview_truncates_long_text():
    buf = TextBuffer.from_text("word " * 200)
    lines = render_fullscreen_preview(buf, 40, 6)
    assert len(lines) == 6  # height respected

def test_render_fullscreen_preview_shows_dirty_flag():
    buf = TextBuffer.from_text("dirty content here")
    buf.dirty = True
    lines = render_fullscreen_preview(buf, 40, 6)
    assert "DIRTY" in lines[-1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
