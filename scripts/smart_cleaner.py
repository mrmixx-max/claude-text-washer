#!/usr/bin/env python3
"""Smart Cleaner — regex-based pre/post processing for AI marker removal.

Runs BEFORE the LLM call (cheap, removes obvious markers)
and AFTER the LLM call (catches what the model missed).

Reuses the marker patterns from marker_scan.py so the detection
and removal logic never drift apart.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from marker_scan import PATTERNS

# Pre-compile all regex patterns once at module level (not per-call)
# Each entry: (severity, compiled_pattern, note)
_COMPILED_PATTERNS: list[tuple[int, re.Pattern, str]] = [
    (sev, re.compile(pat, re.IGNORECASE | re.DOTALL), note)
    for sev, pat, note in PATTERNS
]

# Invisible Unicode characters commonly found in AI-generated text.
# U+200A (Hair Space), U+202F (Narrow No-Break Space) — identified in
# forum research as the 28 most frequent invisible markers.
# Also: U+200B (Zero Width Space), U+200C (ZWNJ), U+200D (ZWJ), U+FEFF (BOM).
_INVISIBLE_UNICODE = re.compile(
    "[\u200A\u202F\u200B\u200C\u200D\uFEFF]+"
)


def clean_text(text: str, aggressive: bool = False) -> str:
    """Remove obvious AI markers via regex substitutions.

    Parameters
    ----------
    text:
        Input text (pre or post LLM).
    aggressive:
        If True, also remove mid/low markers (not just high).
        Use with care — may over-correct.
    """
    min_sev = 2 if not aggressive else 1
    result = text

    for sev, pat, note in _COMPILED_PATTERNS:
        if sev < min_sev:
            continue
        # Replace with empty string for stock phrases
        # For patterns that are structural (e.g. "not only X but also Y"),
        # we keep the content but remove the template wrapper.
        if "not only" in note.lower():
            # "not only X but also Y" → "X. Also Y"
            result = pat.sub(lambda m: _flatten_not_only(m), result)
        elif "template" in note.lower() or "opener" in note.lower() or "closer" in note.lower():
            # Remove template openers/closers entirely
            result = pat.sub("", result)
        else:
            # For buzzwords, remove the word but keep the sentence structure
            result = pat.sub("", result)

    # Remove invisible Unicode characters (AI artifacts)
    result = _INVISIBLE_UNICODE.sub(" ", result)

    # Clean up double spaces and trailing whitespace
    result = re.sub(r"  +", " ", result)
    result = re.sub(r"\n\n\n+", "\n\n", result)
    return result.strip()


def _flatten_not_only(m: re.Match) -> str:
    """Flatten 'not only X but also Y' → 'X. Also Y'."""
    full = m.group(0)
    # Simple heuristic: split on "but also"
    parts = re.split(r"\b(?:but also)\b", full, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        first = re.sub(r"^\s*[Nn]ot only\s*", "", parts[0]).strip()
        second = parts[1].strip()
        # Capitalize first letter of second part
        if second:
            second = second[0].upper() + second[1:]
        return f"{first}. Also, {second}"
    return full


def get_marker_count(text: str) -> dict[str, int]:
    """Count markers by severity."""
    high = mid = low = 0
    for sev, pat, _ in _COMPILED_PATTERNS:
        count = len(pat.findall(text))
        if sev == 3:
            high += count
        elif sev == 2:
            mid += count
        else:
            low += count
    return {"high": high, "mid": mid, "low": low}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python smart_cleaner.py <file> [--aggressive]")
        sys.exit(1)

    path = Path(sys.argv[1])
    aggressive = "--aggressive" in sys.argv
    text = path.read_text(encoding="utf-8")

    print(f"Before: {get_marker_count(text)}")
    cleaned = clean_text(text, aggressive=aggressive)
    print(f"After:  {get_marker_count(cleaned)}")
    print("---")
    print(cleaned)
