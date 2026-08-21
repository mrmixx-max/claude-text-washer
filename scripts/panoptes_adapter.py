#!/usr/bin/env python3
"""Panoptes Integration for Claude Text Washer.

Integrates Panoptes detection methodology:
1. Seven attribution features from Panoptes bench/features.py
2. Heuristic scoring for AI text likelihood
3. Pre/post wash quality comparison
4. CLI: --panoptes to enable Panoptes scoring
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Panoptes canonical feature order
ATTRIBUTION_FEATURES = (
    "long_words",
    "connectors",
    "unique_ratio",
    "short_sentences",
    "structured",
    "digits",
    "balanced_lines",
)

# Connectors that AI overuses
_CONNECTORS = {"however", "therefore", "moreover", "additionally", "overall", "furthermore"}
# Structural markers common in AI text
_STRUCTURE_MARKERS = ("\n-", "\n*", ":", ";", "(", ")", "[", "]")
# Regex patterns
_WORD_RE = re.compile(r"\b[\w']+\b")
_SENTENCE_RE = re.compile(r"[.!?]+")


def word_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def extract_attribution_features(text: str) -> dict[str, float]:
    """Extract the seven Panoptes attribution features."""
    words = word_tokens(text)
    if not words:
        return {name: 0.0 for name in ATTRIBUTION_FEATURES}

    counts = Counter(words)
    total = len(words)

    # Long words: proportion > 6 chars
    long_words = sum(1 for w in words if len(w) > 6) / total

    # Connectors: proportion of AI-typical transition words
    connectors = sum(1 for w in words if w in _CONNECTORS) / total

    # Unique ratio: vocabulary richness
    unique_ratio = len(counts) / total

    # Short sentences: proportion of sentences < 10 words
    sentences = [s for s in _SENTENCE_RE.split(text) if s.strip()]
    short_sentences = sum(1 for s in sentences if len(_WORD_RE.findall(s)) < 10) / max(len(sentences), 1)

    # Structural: presence of list markers
    structured = sum(1 for m in _STRUCTURE_MARKERS if m in text) / max(len(_STRUCTURE_MARKERS), 1)

    # Digits: proportion of digit tokens
    digits = sum(1 for w in words if w.isdigit()) / total

    # Balanced lines: std dev of line lengths (low = more balanced = more AI)
    lines = [line for line in text.splitlines() if line.strip()]
    line_lengths = [len(line) for line in lines]
    if line_lengths:
        mean_len = sum(line_lengths) / len(line_lengths)
        variance = sum((l - mean_len) ** 2 for l in line_lengths) / len(line_lengths)
        # Normalize: high variance = more human (1.0), low = more AI (0.0)
        balanced_lines = min(math.sqrt(variance) / 50.0, 1.0)
    else:
        balanced_lines = 0.5

    return {
        "long_words": round(long_words, 4),
        "connectors": round(connectors, 4),
        "unique_ratio": round(unique_ratio, 4),
        "short_sentences": round(short_sentences, 4),
        "structured": round(structured, 4),
        "digits": round(digits, 4),
        "balanced_lines": round(balanced_lines, 4),
    }


def heuristic_ai_score(text: str) -> float:
    """Compute heuristic AI-likelihood score (0-100, higher = more AI-like).

    Based on Panoptes heuristic_raw_score approach.
    """
    features = extract_attribution_features(text)

    # Weighted combination — tuned for typical AI patterns
    score = 0.0
    score += features["long_words"] * 15      # AI uses longer words
    score += features["connectors"] * 25      # AI overuses transition words
    score += (1 - features["unique_ratio"]) * 20  # AI has lower vocab richness
    score += features["short_sentences"] * 15  # AI has uniform sentence length
    score += features["structured"] * 10      # AI loves lists
    score += features["balanced_lines"] * 15  # AI has balanced line lengths

    return min(max(score, 0.0), 100.0)


@dataclass
class PanoptesReport:
    """Panoptes-style analysis report."""
    features: dict[str, float]
    ai_score: float
    verdict: str  # "human", "uncertain", "ai"

    def to_dict(self) -> dict:
        return {
            "features": self.features,
            "ai_score": round(self.ai_score, 2),
            "verdict": self.verdict,
        }


def analyze_text(text: str) -> PanoptesReport:
    """Analyze text using Panoptes methodology."""
    features = extract_attribution_features(text)
    ai_score = heuristic_ai_score(text)

    # Verdict thresholds
    if ai_score < 30:
        verdict = "human"
    elif ai_score < 60:
        verdict = "uncertain"
    else:
        verdict = "ai"

    return PanoptesReport(features=features, ai_score=ai_score, verdict=verdict)


def compare_wash_quality(original: str, washed: str) -> dict:
    """Compare pre/post wash quality using Panoptes scoring."""
    before = analyze_text(original)
    after = analyze_text(washed)

    return {
        "before": before.to_dict(),
        "after": after.to_dict(),
        "improvement": round(before.ai_score - after.ai_score, 2),
        "success": after.ai_score < before.ai_score,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Panoptes Integration for Claude Text Washer")
    parser.add_argument("input", nargs="?", help="Input text file")
    parser.add_argument("--compare", help="Compare original vs washed file (original:washed)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.compare:
        orig_path, washed_path = args.compare.split(":")
        original = Path(orig_path).read_text(encoding="utf-8")
        washed = Path(washed_path).read_text(encoding="utf-8")
        result = compare_wash_quality(original, washed)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Before: {result['before']['ai_score']:.1f} ({result['before']['verdict']})")
            print(f"After:  {result['after']['ai_score']:.1f} ({result['after']['verdict']})")
            print(f"Improvement: {result['improvement']:.1f}")

    elif args.input:
        text = Path(args.input).read_text(encoding="utf-8")
        report = analyze_text(text)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"AI Score: {report.ai_score:.1f}/100 ({report.verdict})")
            print("Features:")
            for k, v in report.features.items():
                print(f"  {k}: {v}")

    else:
        # Demo
        demo_ai = "However, it is important to note that the aforementioned considerations must be taken into account. Therefore, we can conclude that further research is necessary."
        demo_hum = "I dunno, maybe. It just feels wrong, you know? Not sure what else to say."
        print("=== Panoptes Demo ===")
        print(f"AI text:   {analyze_text(demo_ai).ai_score:.1f}/100")
        print(f"Human text: {analyze_text(demo_hum).ai_score:.1f}/100")
