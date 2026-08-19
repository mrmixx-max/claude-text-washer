"""Tests for smart_cleaner module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from smart_cleaner import clean_text, get_marker_count


class TestCleanText:
    def test_remove_template_opener(self):
        """Template openers are removed."""
        text = "In today's digital world, AI is changing everything."
        cleaned = clean_text(text, aggressive=False)
        assert "In today's digital world" not in cleaned

    def test_remove_stock_words(self):
        """Stock AI words are removed."""
        text = "This is a comprehensive guide that will delve into the topic."
        cleaned = clean_text(text, aggressive=False)
        assert "comprehensive" not in cleaned.lower()
        assert "delve into" not in cleaned.lower()

    def test_remove_buzzwords(self):
        """German buzzwords are removed."""
        text = "Wir müssen die Synergien nutzen und optimieren."
        cleaned = clean_text(text, aggressive=True)
        assert "Synergien" not in cleaned

    def test_clean_preserves_content(self):
        """Cleaning preserves meaningful content."""
        text = "Der Algorithmus sortiert die Daten. Er nutzt Quicksort."
        cleaned = clean_text(text, aggressive=True)
        assert "Algorithmus" in cleaned or "sortiert" in cleaned

    def test_remove_double_spaces(self):
        """Double spaces are collapsed."""
        text = "This  has   too    many spaces."
        cleaned = clean_text(text)
        assert "  " not in cleaned

    def test_empty_string(self):
        """Empty string returns empty."""
        assert clean_text("") == ""

    def test_no_markers_unchanged(self):
        """Text without markers is unchanged."""
        text = "Hallo Welt."
        assert clean_text(text) == text


class TestGetMarkerCount:
    def test_count_high_markers(self):
        """Counts high-severity markers."""
        text = "In today's digital world, we must delve into this comprehensive solution. It is important to note that this is crucial."
        counts = get_marker_count(text)
        assert counts["high"] >= 2

    def test_count_empty(self):
        """Empty text has zero counts."""
        counts = get_marker_count("")
        assert counts == {"high": 0, "mid": 0, "low": 0}

    def test_count_clean_text(self):
        """Clean text has fewer markers than AI text."""
        ai_text = "In today's digital world, we must leverage this robust and comprehensive solution. Furthermore, it is important to note that this game-changer will revolutionize the landscape."
        clean_text_result = "Software entwickeln ist komplex."
        assert get_marker_count(ai_text)["high"] > get_marker_count(clean_text_result)["high"]
