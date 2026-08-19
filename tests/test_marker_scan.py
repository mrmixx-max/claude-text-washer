#!/usr/bin/env python3
"""Tests for claude-text-washer scripts."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.marker_scan import PATTERNS, scan


def test_patterns_non_empty():
    assert len(PATTERNS) > 20


def test_scan_empty():
    assert scan("") == []


def test_scan_claude_closer():
    text = "Zusammenfassend lässt sich sagen, dass dies der Fall ist."
    hits = scan(text)
    assert any("Zusammenfassend" in h[2] for h in hits)


def test_scan_english_opener():
    text = "In today's digital world, AI is transforming everything."
    hits = scan(text)
    assert any("opener" in h[1].lower() for h in hits)


def test_scan_no_false_positives_on_real_text():
    text = "Die Sonne ging unter. Menschen gingen nach Hause. Ein Hund bellte."
    hits = scan(text)
    assert len(hits) == 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
