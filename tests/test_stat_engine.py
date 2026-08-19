#!/usr/bin/env python3
"""Tests for statistical watermark detection."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.stat_engine import (
    analyze_text,
    calculate_burstiness,
    calculate_entropy,
    calculate_hapax_ratio,
    calculate_sentence_entropy,
    calculate_type_token_ratio,
    calculate_zipf_coefficient,
    detect_green_list_bias,
    generate_anti_watermark_prompt,
    sentence_split,
    tokenize,
)


def test_tokenize():
    tokens = tokenize("Hello world, this is a test.")
    assert tokens == ["hello", "world", "this", "is", "a", "test"]


def test_sentence_split():
    sentences = sentence_split("Hello world. How are you? I'm fine!")
    assert len(sentences) == 3


def test_entropy_empty():
    assert calculate_entropy([]) == 0.0


def test_entropy_uniform():
    # Uniform distribution = high entropy
    tokens = ["a", "b", "c", "d", "e", "f", "g", "h"]
    entropy = calculate_entropy(tokens)
    assert entropy > 2.5


def test_entropy_repetitive():
    # Repetitive = low entropy
    tokens = ["the", "the", "the", "the", "the"]
    entropy = calculate_entropy(tokens)
    assert entropy < 1.0


def test_burstiness_uniform():
    # Uniform sentence lengths = low burstiness
    sentences = ["Hello world foo bar", "Baz qux corge grault", "Garply waldo fred plugh"]
    burst = calculate_burstiness(sentences)
    assert burst < 0.3


def test_burstiness_varied():
    # Varied sentence lengths = high burstiness
    sentences = ["Hi.", "This is a much longer sentence with many words.", "OK.", "Another fairly long sentence here."]
    burst = calculate_burstiness(sentences)
    assert burst > 0.3


def test_green_list_ratio():
    text = "The cat sat on the mat and the dog ran in the park"
    tokens = tokenize(text)
    ratio = detect_green_list_bias(tokens)
    assert 0.3 < ratio < 0.7


def test_analyze_text_basic():
    text = "In today's digital world, AI is transforming everything. It is important to note that this is significant. Furthermore, we must consider the implications."
    report = analyze_text(text)
    assert report.perplexity > 0
    assert report.burstiness >= 0
    assert report.ai_score >= 0


def test_analyze_text_human():
    # Human-like text should have lower AI score
    text = "I woke up late. Coffee was cold. Damn. Went back to bed. Later, tried again. Got stuff done."
    report = analyze_text(text)
    assert report.burstiness > 0.3


def test_generate_prompt():
    text = "This is a test sentence. Another one here."
    report = analyze_text(text)
    prompt = generate_anti_watermark_prompt(text, report)
    assert "Rewrite" in prompt
    assert text in prompt


def test_type_token_ratio():
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    ttr = calculate_type_token_ratio(tokens)
    assert 0.0 < ttr < 1.0


def test_type_token_ratio_repetitive():
    tokens = ["the", "the", "the", "the"]
    ttr = calculate_type_token_ratio(tokens)
    assert ttr == 0.25


def test_zipf_coefficient():
    # Natural-like distribution
    tokens = ["the"] * 50 + ["cat"] * 25 + ["sat"] * 15 + ["on"] * 10 + ["mat"] * 5
    zipf = calculate_zipf_coefficient(tokens)
    assert zipf > 0


def test_zipf_coefficient_short():
    tokens = ["hello", "world"]
    zipf = calculate_zipf_coefficient(tokens)
    assert zipf == 0.0


def test_hapax_ratio():
    tokens = ["the", "cat", "sat", "on", "mat"]
    hapax = calculate_hapax_ratio(tokens)
    assert hapax == 1.0  # All unique


def test_hapax_ratio_repetitive():
    tokens = ["the", "the", "the", "cat", "cat", "sat"]
    hapax = calculate_hapax_ratio(tokens)
    assert 0.0 < hapax < 1.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
