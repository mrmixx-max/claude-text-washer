"""Tests for streaming, connection pooling, and --dry-run."""

import io
import json
import sys
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "scripts")

from ollama_utils import call_ollama_stream, _get_opener, CircuitBreaker
from multi_agent_washer import _wash_with_model_stream, multi_agent_wash, WashCandidate


class TestCallOllamaStream:
    def test_stream_yields_chunks(self):
        """Stream yields text chunks from line-delimited JSON."""
        lines = [
            json.dumps({"response": "Hello ", "done": False}),
            json.dumps({"response": "world", "done": True}),
        ]
        resp = io.BytesIO("\n".join(lines).encode("utf-8"))

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read = resp.read

        opener = MagicMock()
        opener.open.return_value = mock_resp

        with patch("ollama_utils._get_opener", return_value=opener):
            chunks = list(call_ollama_stream("hi", "llama3.2"))

        assert "".join(chunks).strip() == "Hello world"

    def test_stream_handles_empty_lines(self):
        """Empty lines in stream are skipped."""
        lines = [
            "",
            json.dumps({"response": "A", "done": False}),
            "",
            json.dumps({"response": "B", "done": True}),
        ]
        resp = io.BytesIO("\n".join(lines).encode("utf-8"))

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read = resp.read

        opener = MagicMock()
        opener.open.return_value = mock_resp

        with patch("ollama_utils._get_opener", return_value=opener):
            chunks = list(call_ollama_stream("hi", "llama3.2"))

        assert "".join(chunks) == "AB"

    def test_stream_raises_on_error(self):
        """Stream with error chunk raises RuntimeError."""
        lines = [json.dumps({"error": "model not found"})]
        resp = io.BytesIO("\n".join(lines).encode("utf-8"))

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read = resp.read

        opener = MagicMock()
        opener.open.return_value = mock_resp

        with patch("ollama_utils._get_opener", return_value=opener):
            with pytest.raises(RuntimeError, match="model not found"):
                list(call_ollama_stream("hi", "bad-model"))


class TestConnectionPooling:
    def test_get_opener_returns_opener(self):
        """_get_opener returns an OpenerDirector."""
        opener = _get_opener()
        assert hasattr(opener, "open")

    def test_get_opener_reuses_instance(self):
        """_get_opener returns the same instance on repeated calls."""
        opener1 = _get_opener()
        opener2 = _get_opener()
        assert opener1 is opener2


class TestWashWithModelStream:
    def test_stream_wash(self):
        """_wash_with_model_stream collects chunks and scores."""
        mock_chunks = ["Hello ", "world"]
        with patch("multi_agent_washer.call_ollama_stream", return_value=iter(mock_chunks)):
            out = io.StringIO()
            result = _wash_with_model_stream("test", {"name": "llama3.2", "label": "x", "temperature": 0.7, "max_tokens": 100}, out=out)
        assert isinstance(result, WashCandidate)
        assert result.error is None
        assert result.model == "llama3.2"


class TestMultiAgentDryRun:
    def test_dry_run_returns_score(self):
        """dry_run mode returns candidate with pre-wash score."""
        result = multi_agent_wash("test text for scoring", dry_run=True)
        assert result.model == "dry_run"
        assert result.label == "dry_run"
        assert result.ai_score >= 0
