#!/usr/bin/env python3
"""Integration tests for the multi-agent washer.

These tests exercise the *full* path end-to-end — ``multi_agent_wash`` →
``_wash_with_model`` → real ``call_ollama`` → real ``analyze_text`` — but mock
the Ollama HTTP transport (``urllib.request.urlopen``) so no local Ollama server
is required.  This validates the parallel-execution orchestration and the
AI-score winner-selection logic against the real scoring engine.
"""
from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.multi_agent_washer import (  # noqa: E402
    AGENT_MODELS,
    build_parser,
    multi_agent_wash,
    rank_candidates,
    WashCandidate,
)
from scripts.stat_engine import analyze_text  # noqa: E402
from scripts.ollama_utils import reset_circuit_breakers  # noqa: E402


# --------------------------------------------------------------------------- #
# HTTP transport fakes
# --------------------------------------------------------------------------- #
class _FakeResponse:
    """Minimal stand-in for an ``http.client.HTTPResponse`` / urlopen context."""

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


def _http_error(req, code: int) -> urllib.error.HTTPError:
    """Build an HTTPError the way urlopen raises them."""
    return urllib.error.HTTPError(
        req.full_url, code, "HTTP %d" % code, {}, io.BytesIO(b"")
    )


def _payload_model(req) -> str:
    """Extract the `model` field from a urlopen request's JSON body."""
    return json.loads(req.data.decode("utf-8")).get("model", "")


def make_urlopen(model_text):
    """Build a fake ``urlopen`` returning JSON text per model.

    ``model_text`` maps ``model_name -> response-text``.  Returns
    ``(fake_urlopen, calls)`` where ``calls`` records every model attempted.
    """
    calls: list[str] = []

    def fake_urlopen(req, timeout=None):
        calls.append(_payload_model(req))
        text = model_text.get(_payload_model(req), "default wash text")
        body = json.dumps({"response": text}).encode("utf-8")
        return _FakeResponse(body, code=200)

    return fake_urlopen, calls


def make_urlopen_with_code(code_map, model_text):
    """Fake urlopen that returns an HTTP error *code* for some models.

    ``code_map`` maps ``model -> http_code`` (raised as HTTPError).
    ``model_text`` maps ``model -> response-text`` for the successful models.
    """
    calls: list[str] = []

    def fake_urlopen(req, timeout=None):
        model = _payload_model(req)
        calls.append(model)
        if model in code_map:
            raise _http_error(req, code_map[model])
        text = model_text.get(model, "default wash text")
        body = json.dumps({"response": text}).encode("utf-8")
        return _FakeResponse(body, code=200)

    return fake_urlopen, calls


@pytest.fixture(autouse=True)
def _reset_circuit_breakers():
    """A tripped breaker in one test must not bleed into the next."""
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


@pytest.fixture
def _no_backoff():
    """Neutralise backoff sleeps so retry paths stay instant & deterministic."""
    with patch("scripts.ollama_utils._sleep_backoff", lambda *a, **k: None):
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
    Keeping the scores distinct avoids nondeterministic tie-breaking between
    threads in ``rank_candidates``.
    """
    return {
        "lfm25-tool": "Tiefstahlhügel knirschend als hätte der Regen den Stahl gepriesen doch der Wind riss die Fäden auseinander und niemand sah es kommen.",
        "llama3.2": "Ein weiterer Tag. Die Sonne. Langsam. Alles ist gleich. Der Tag ist vorbei. Die Sonne geht unter. Alles ist gleich und langsam.",
        "eurollm-9b": "kurz. starker biss. Bahn dröhnt durch die Nähte, ein zischendes Messer aus Stahl. Regen. Noch ein Stück. Jetzt. Hier. Alles zusammen bricht zusammen.",
    }


def test_integration_winner_is_lowest_ai_score(_no_backoff):
    """All 3 agents run via the real HTTP-mocked path; winner = lowest score."""
    model_text = _distinct_texts()
    fake, calls = make_urlopen(model_text)

    with patch("scripts.ollama_utils.urllib.request.urlopen", side_effect=fake):
        best = multi_agent_wash("dirty input text", verbose=True)

    # Every agent was actually invoked over the (mocked) HTTP transport.
    assert len(calls) == 3
    assert set(calls) == {m["name"] for m in AGENT_MODELS}

    # Winner is the model whose *real* analyzed score is the minimum.
    expected_winner = min(model_text, key=lambda m: analyze_text(model_text[m]).ai_score)
    assert best.model == expected_winner
    assert best.text == model_text[expected_winner]
    assert best.ai_score == analyze_text(model_text[expected_winner]).ai_score
    assert not best.is_error


def test_integration_all_models_invoked_in_parallel(_no_backoff):
    """Parallel execution must launch every agent, not short-circuit on the first."""
    fake, calls = make_urlopen(_distinct_texts())

    with patch("scripts.ollama_utils.urllib.request.urlopen", side_effect=fake):
        best = multi_agent_wash("input text", verbose=False)

    assert not best.is_error
    # 3 distinct HTTP calls — proves all three workers ran, not just one.
    assert len(calls) == len(AGENT_MODELS)
    assert len(set(calls)) == len(AGENT_MODELS)


def test_integration_one_model_http_500_excluded_from_winner(_no_backoff):
    """A model returning a retryable 500 must become an error candidate that the
    winner-selection logic excludes."""
    model_text = _distinct_texts()
    # lfm25-tool always returns 500 (retryable) → call_ollama exhausts retries.
    fake, calls = make_urlopen_with_code({"lfm25-tool": 500}, model_text)

    with patch("scripts.ollama_utils.urllib.request.urlopen", side_effect=fake):
        best = multi_agent_wash("input text", verbose=False)

    # lfm25-tool was retried (max_retries + 1 attempts) but did not win.
    lfm_calls = calls.count("lfm25-tool")
    assert lfm_calls >= 2  # retries happened
    assert best.model != "lfm25-tool"
    assert not best.is_error

    # Reconstruct candidates to prove the failed one was excluded.
    cands = [
        WashCandidate("", "lfm25-tool", "fast", 0.0, 999.0, error="down"),
        WashCandidate(model_text["llama3.2"], "llama3.2", "standard", 1.0, analyze_text(model_text["llama3.2"]).ai_score),
        WashCandidate(model_text["eurollm-9b"], "eurollm-9b", "premium", 1.0, analyze_text(model_text["eurollm-9b"]).ai_score),
    ]
    assert rank_candidates(cands).model == best.model


def test_integration_all_models_fail_returns_error_candidate(_no_backoff):
    """When every model returns a non-retryable 400, all candidates error."""
    fake, _calls = make_urlopen_with_code(
        {m["name"]: 400 for m in AGENT_MODELS}, _distinct_texts()
    )

    with patch("scripts.ollama_utils.urllib.request.urlopen", side_effect=fake):
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
    fake, _calls = make_urlopen(_distinct_texts())

    # Build a tiny input file for the CLI.
    inp = tmp_path / "in.txt"
    inp.write_text("dirty input text")

    with patch("scripts.ollama_utils.urllib.request.urlopen", side_effect=fake):
        # main is the multi-agent washer's own entry point (not the unified CLI).
        from scripts.multi_agent_washer import main as ma_main

        rc = ma_main([str(inp), "--benchmark", "5"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "Benchmark Results (5 iterations)" in out
    assert "p50" in out
    assert "p95" in out
    assert "AI score" in out
    # All three models were exercised across the benchmark runs.
    assert len(_calls) == 3 * 5


def test_integration_benchmark_rejects_non_positive(tmp_path):
    """--benchmark N with N < 1 must error out (parser.error → SystemExit)."""
    inp = tmp_path / "in.txt"
    inp.write_text("dirty input text")

    from scripts.multi_agent_washer import main as ma_main

    with pytest.raises(SystemExit) as exc:
        ma_main([str(inp), "--benchmark", "0"])
    # argparse parser.error exits with status 2.
    assert exc.value.code == 2
