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

    for sev, pat, note in PATTERNS:
        if sev < min_sev:
            continue
        # Replace with empty string for stock phrases
        # For patterns that are structural (e.g. "not only X but also Y"),
        # we keep the content but remove the template wrapper.
        if "not only" in note.lower():
            # "not only X but also Y" → "X. Also Y"
            result = re.sub(
                pat,
                lambda m: _flatten_not_only(m),
                result,
                flags=re.IGNORECASE | re.DOTALL,
            )
        elif "template" in note.lower() or "opener" in note.lower() or "closer" in note.lower():
            # Remove template openers/closers entirely
            result = re.sub(pat, "", result, flags=re.IGNORECASE | re.DOTALL)
        else:
            # For buzzwords, remove the word but keep the sentence structure
            result = re.sub(pat, "", result, flags=re.IGNORECASE)

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
    for sev, pat, _ in PATTERNS:
        count = len(re.findall(pat, text, flags=re.IGNORECASE | re.DOTALL))
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
