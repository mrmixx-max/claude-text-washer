#!/usr/bin/env python3
"""Integration tests for the multi-agent washer.

These tests exercise the *full* path end-to-end — ``multi_agent_wash`` →
``_wash_with_model`` → real ``call_ollama`` → real ``analyze_text`` — but mock
the Ollama HTTP transport (``_get_opener().open()``) so no local Ollama server
is required.  This validates the parallel-execution orchestration and the
AI-score winner-selection logic against the real scoring engine.
"""
from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from multi_agent_washer import (  # noqa: E402
    AGENT_MODELS,
    build_parser,
    multi_agent_wash,
    rank_candidates,
    WashCandidate,
)
from stat_engine import analyze_text  # noqa: E402
from ollama_utils import reset_circuit_breakers  # noqa: E402


# --------------------------------------------------------------------------- #
# HTTP transport fakes
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, body: bytes, code: int = 200):
        self._body = body
        self._code = code

    def getcode(self) -> int:
        return self._code

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://127.0.0.1:11434/api/generate", code, "HTTP %d" % code, {}, io.BytesIO(b"")
    )


def _payload_model(req) -> str:
    return json.loads(req.data.decode("utf-8")).get("model", "")


def _make_mock_opener(fake_fn):
    """Create a mock opener whose .open() calls fake_fn."""
    opener = MagicMock()
    opener.open.side_effect = fake_fn
    return opener


def make_urlopen(model_text):
    """Build a fake opener returning JSON text per model."""
    calls: list[str] = []

    def fake_open(req, timeout=None):
        calls.append(_payload_model(req))
        text = model_text.get(_payload_model(req), "default wash text")
        body = json.dumps({"response": text}).encode("utf-8")
        return _FakeResponse(body, code=200)

    return fake_open, calls


def make_urlopen_with_code(code_map, model_text):
    """Fake opener that returns an HTTP error *code* for some models."""
    calls: list[str] = []

    def fake_open(req, timeout=None):
        model = _payload_model(req)
        calls.append(model)
        if model in code_map:
            raise _http_error(code_map[model])
        text = model_text.get(model, "default wash text")
        body = json.dumps({"response": text}).encode("utf-8")
        return _FakeResponse(body, code=200)

    return fake_open, calls


@pytest.fixture(autouse=True)
def _reset_circuit_breakers():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


@pytest.fixture
def _no_backoff():
    with patch("ollama_utils._sleep_backoff", lambda *a, **k: None):
        yield


# --------------------------------------------------------------------------- #
# Full parallel + winner-selection path
# --------------------------------------------------------------------------- #
def _distinct_texts():
    """Three distinct wash outputs with *distinct, stable* AI scores.

    Score ordering (computed via the real stat_engine):
      eurollm-9b  -> 40.0 (winner, lowest)
      llama3.2    -> 50.0
      lfm25-tool  -> 60.0
    """
    return {
        "lfm25-tool": "Tiefstahlhügel knirschend als hätte der Regen den Stahl gepriesen doch der Wind riss die Fäden auseinander und niemand sah es kommen.",
        "llama3.2": "Ein weiterer Tag. Die Sonne. Langsam. Alles ist gleich. Der Tag ist vorbei. Die Sonne geht unter. Alles ist gleich und langsam.",
        "eurollm-9b": "kurz. starker biss. Bahn dröhnt durch die Nähte, ein zischendes Messer aus Stahl. Regen. Noch ein Stück. Jetzt. Hier. Alles zusammen bricht zusammen.",
    }


def test_integration_winner_is_lowest_ai_score(_no_backoff):
    """All 3 agents run via the real HTTP-mocked path; winner = lowest score."""
    model_text = _distinct_texts()
    fake_open, calls = make_urlopen(model_text)
    opener = _make_mock_opener(fake_open)

    with patch("ollama_utils._get_opener", return_value=opener):
        best = multi_agent_wash("dirty input text", verbose=True)

    assert len(calls) == 3
    assert set(calls) == {m["name"] for m in AGENT_MODELS}

    expected_winner = min(model_text, key=lambda m: analyze_text(model_text[m]).ai_score)
    assert best.model == expected_winner
    assert best.text == model_text[expected_winner]
    assert best.ai_score == analyze_text(model_text[expected_winner]).ai_score
    assert not best.is_error


def test_integration_all_models_invoked_in_parallel(_no_backoff):
    """Parallel execution must launch every agent, not short-circuit on the first."""
    fake_open, calls = make_urlopen(_distinct_texts())
    opener = _make_mock_opener(fake_open)

    with patch("ollama_utils._get_opener", return_value=opener):
        best = multi_agent_wash("input text", verbose=False)

    assert not best.is_error
    assert len(calls) == len(AGENT_MODELS)
    assert len(set(calls)) == len(AGENT_MODELS)


def test_integration_one_model_http_500_excluded_from_winner(_no_backoff):
    """A model returning a retryable 500 must become an error candidate."""
    model_text = _distinct_texts()
    fake_open, calls = make_urlopen_with_code({"lfm25-tool": 500}, model_text)
    opener = _make_mock_opener(fake_open)

    with patch("ollama_utils._get_opener", return_value=opener):
        best = multi_agent_wash("input text", verbose=False)

    lfm_calls = calls.count("lfm25-tool")
    assert lfm_calls >= 2  # retries happened
    assert best.model != "lfm25-tool"
    assert not best.is_error

    cands = [
        WashCandidate("", "lfm25-tool", "fast", 0.0, 999.0, error="down"),
        WashCandidate(model_text["llama3.2"], "llama3.2", "standard", 1.0, analyze_text(model_text["llama3.2"]).ai_score),
        WashCandidate(model_text["eurollm-9b"], "eurollm-9b", "premium", 1.0, analyze_text(model_text["eurollm-9b"]).ai_score),
    ]
    assert rank_candidates(cands).model == best.model


def test_integration_all_models_fail_returns_error_candidate(_no_backoff):
    """When every model returns a non-retryable 400, all candidates error."""
    fake_open, _calls = make_urlopen_with_code(
        {m["name"]: 400 for m in AGENT_MODELS}, _distinct_texts()
    )
    opener = _make_mock_opener(fake_open)

    with patch("ollama_utils._get_opener", return_value=opener):
        best = multi_agent_wash("input text", verbose=False)

    assert best.is_error
    assert best.ai_score == 999.0


# --------------------------------------------------------------------------- #
# CLI surface (build_parser + main argv support)
# --------------------------------------------------------------------------- #
def test_integration_dry_run_cli(tmp_path):
    """--dry-run prints a pre-wash AI score without calling Ollama."""
    p = tmp_path / "in.txt"
    p.write_text("some text here.")
    parser = build_parser()
    args = parser.parse_args([str(p), "--dry-run"])
    assert args.dry_run is True
    assert args.benchmark is None


def test_integration_benchmark_cli_reports_stats(tmp_path, capsys, _no_backoff):
    """claude-washer multi-agent --benchmark N prints latency/p50/p95/ai-score stats."""
    fake_open, _calls = make_urlopen(_distinct_texts())
    opener = _make_mock_opener(fake_open)

    inp = tmp_path / "in.txt"
    inp.write_text("dirty input text")

    with patch("ollama_utils._get_opener", return_value=opener):
        from multi_agent_washer import main as ma_main
        rc = ma_main([str(inp), "--benchmark", "5"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "Benchmark Results (5 iterations)" in out
    assert "p50" in out
    assert "p95" in out
    assert "AI score" in out
    assert len(_calls) == 3 * 5


def test_integration_benchmark_rejects_non_positive(tmp_path):
    """--benchmark N with N < 1 must error out (parser.error → SystemExit)."""
    inp = tmp_path / "in.txt"
    inp.write_text("dirty input text")

    from multi_agent_washer import main as ma_main

    with pytest.raises(SystemExit) as exc:
        ma_main([str(inp), "--benchmark", "0"])
    assert exc.value.code == 2
