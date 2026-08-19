#!/usr/bin/env python3
"""Tests for chat module."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.chat import SYSTEM_PROMPT, ai_score_color, ollama_chat

# ANSI codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def test_system_prompt_exists():
    assert "Ghostwriter" in SYSTEM_PROMPT
    assert "Burstiness" in SYSTEM_PROMPT


def test_ai_score_color_low():
    assert ai_score_color(10) == GREEN


def test_ai_score_color_medium():
    assert ai_score_color(40) == YELLOW


def test_ai_score_color_high():
    assert ai_score_color(80) == RED


def test_ollama_chat_success():
    mock_response = {
        "message": {"content": "Test response"}
    }
    with patch("scripts.chat.urllib.request.urlopen") as mock_urlopen:
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=MagicMock(
            read=MagicMock(return_value=b'{"message": {"content": "Test response"}}')
        ))
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_ctx
        result = ollama_chat("llama3.2", [{"role": "user", "content": "Hi"}])
        assert result == "Test response"


def test_ollama_chat_error():
    import urllib.error
    with patch("scripts.chat.urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        try:
            ollama_chat("llama3.2", [{"role": "user", "content": "Hi"}])
            assert False, "Should have raised"
        except RuntimeError as e:
            assert "Ollama error" in str(e)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
