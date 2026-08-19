#!/usr/bin/env python3
"""Tests for the circuit-breaker pattern and HTTP retry layer in ollama_utils."""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Use the same module instance as multi_agent_washer
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import ollama_utils as ou

from ollama_utils import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    DEFAULT_CB_COOLDOWN,
    call_ollama,
    get_circuit_breaker,
    get_circuit_breaker_state,
    reset_circuit_breakers,
)


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


def _ok_response(text: str) -> _FakeResponse:
    return _FakeResponse(json.dumps({"response": text}).encode("utf-8"), code=200)


def _payload_model(req) -> str:
    return json.loads(req.data.decode("utf-8")).get("model", "")


def _make_mock_opener(fake_fn):
    """Create a mock opener whose .open() calls fake_fn."""
    opener = MagicMock()
    opener.open.side_effect = fake_fn
    return opener


@pytest.fixture(autouse=True)
def _reset_breakers():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


@pytest.fixture
def _no_backoff():
    with patch("ollama_utils._sleep_backoff", lambda *a, **k: None):
        yield


# --------------------------------------------------------------------------- #
# CircuitBreaker unit tests (direct, deterministic)
# --------------------------------------------------------------------------- #
def test_circuit_breaker_starts_closed():
    cb = CircuitBreaker("m", failure_threshold=3, cooldown=30.0)
    assert cb.state == CircuitBreaker.CLOSED
    assert not cb.is_open()
    assert cb.failure_count == 0


def test_circuit_breaker_trips_after_threshold():
    cb = CircuitBreaker("m", failure_threshold=3, cooldown=60.0)
    cb.record_failure()
    assert cb.state == CircuitBreaker.CLOSED
    cb.record_failure()
    assert cb.state == CircuitBreaker.CLOSED
    cb.record_failure()  # 3rd → trips
    assert cb.state == CircuitBreaker.OPEN
    assert cb.is_open() is True


def test_circuit_breaker_success_resets():
    cb = CircuitBreaker("m", failure_threshold=3, cooldown=60.0)
    cb.record_failure(); cb.record_failure()
    cb.record_success()
    assert cb.state == CircuitBreaker.CLOSED
    assert cb.failure_count == 0


def test_circuit_breaker_half_open_recovery_on_success():
    cb = CircuitBreaker("m", failure_threshold=1, cooldown=0.0)
    cb.record_failure()
    assert cb.state == CircuitBreaker.OPEN
    cb._opened_at = time.time() - 100
    assert cb.is_open() is False
    assert cb.state == CircuitBreaker.HALF_OPEN
    cb.record_success()
    assert cb.state == CircuitBreaker.CLOSED


def test_circuit_breaker_half_open_failure_reopens():
    cb = CircuitBreaker("m", failure_threshold=1, cooldown=60.0)
    cb.record_failure()
    cb._opened_at = time.time() - 100
    assert not cb.is_open()
    cb.record_failure()
    assert cb.state == CircuitBreaker.OPEN
    assert cb.is_open() is True


def test_circuit_breaker_allows_only_half_open_max_probes():
    cb = CircuitBreaker("m", failure_threshold=1, cooldown=0.0)
    cb.record_failure()
    cb._opened_at = time.time() - 100
    assert cb.is_open() is False
    assert cb.allow_probe() is True
    assert cb.allow_probe() is False


# --------------------------------------------------------------------------- #
# call_ollama + breaker integration (mocked HTTP transport)
# --------------------------------------------------------------------------- #
def test_call_ollama_trips_breaker_after_repeated_429(_no_backoff):
    counter = {"n": 0}

    def fake_open(req, timeout=None):
        counter["n"] += 1
        raise _http_error(429)

    model = "rate-limited-model-x1"
    opener = _make_mock_opener(fake_open)
    with patch("ollama_utils._get_opener", return_value=opener):
        for _ in range(3):
            with pytest.raises(RuntimeError):
                call_ollama("prompt", model, max_retries=3)
        assert get_circuit_breaker_state(model) == CircuitBreaker.OPEN
        with pytest.raises(CircuitBreakerOpenError):
            call_ollama("prompt", model, max_retries=3)

    # 3 calls × 4 attempts = 12; the 4th failed-fast call added 0.
    assert counter["n"] == 12


def test_call_ollama_non_retryable_400_does_not_trip_breaker(_no_backoff):
    counter = {"n": 0}
    model = "bad-model-x2"

    def fake_open(req, timeout=None):
        counter["n"] += 1
        raise _http_error(400)

    opener = _make_mock_opener(fake_open)
    with patch("ollama_utils._get_opener", return_value=opener):
        for _ in range(5):
            with pytest.raises(RuntimeError):
                call_ollama("prompt", model, max_retries=3)

    # 400 is not retryable → 1 attempt each, breaker never trips.
    assert counter["n"] == 5
    assert get_circuit_breaker_state(model) == CircuitBreaker.CLOSED


def test_call_ollama_recovers_once_cooldown_elapses(_no_backoff):
    model = "flaky-model-x3"
    calls = {"n": 0}

    def failing_open(req, timeout=None):
        calls["n"] += 1
        raise _http_error(429)

    def success_open(req, timeout=None):
        return _ok_response("recovered text")

    # Phase 1: trip the breaker with repeated 429s (threshold=2).
    opener = _make_mock_opener(failing_open)
    with patch("ollama_utils._get_opener", return_value=opener):
        for _ in range(2):
            with pytest.raises(RuntimeError):
                call_ollama("p", model, max_retries=2, cb_failure_threshold=2)
    assert get_circuit_breaker_state(model) == CircuitBreaker.OPEN

    # Phase 2: breaker open → fail fast (no HTTP).
    with patch("ollama_utils._get_opener", return_value=opener):
        with pytest.raises(CircuitBreakerOpenError):
            call_ollama("p", model, max_retries=2, cb_failure_threshold=2)
    first_phase_calls = calls["n"]

    # Phase 3: simulate cooldown elapsed, then a half-open probe succeeds.
    breaker = get_circuit_breaker(model)
    breaker._opened_at = time.time() - 100
    breaker.cooldown = DEFAULT_CB_COOLDOWN
    success_opener = _make_mock_opener(success_open)
    with patch("ollama_utils._get_opener", return_value=success_opener):
        result = call_ollama("p", model, max_retries=2, cb_failure_threshold=2)

    assert result == "recovered text"
    assert get_circuit_breaker_state(model) == CircuitBreaker.CLOSED
    assert calls["n"] == first_phase_calls


# --------------------------------------------------------------------------- #
# Multi-agent cascade prevention
# --------------------------------------------------------------------------- #
def test_multi_agent_washer_breaker_prevents_cascade(_no_backoff):
    """A persistently 429-ing model trips its own breaker and then fails FAST."""
    import multi_agent_washer as maw
    maw.reset_circuit_breakers()

    per_model = {"lfm25-tool": 0, "llama3.2": 0, "eurollm-9b": 0}

    def fake_open(req, timeout=None):
        model = _payload_model(req)
        per_model[model] += 1
        if model == "lfm25-tool":
            raise _http_error(429)
        return _ok_response(f"wash from {model}")

    opener = _make_mock_opener(fake_open)
    with patch("ollama_utils._get_opener", return_value=opener):
        # 3 runs → lfm25-tool's breaker trips.
        for _ in range(3):
            best = maw.multi_agent_wash("dirty input text", verbose=False)
        assert best.model in ("llama3.2", "eurollm-9b")
        assert not best.is_error

        # 4th run: lfm25-tool must fail fast — no new 429 hits.
        lfm_before = per_model["lfm25-tool"]
        best = maw.multi_agent_wash("dirty input text", verbose=False)

    assert per_model["lfm25-tool"] == lfm_before
    assert per_model["llama3.2"] >= 4
    assert per_model["eurollm-9b"] >= 4
    assert best.model in ("llama3.2", "eurollm-9b")
    assert not best.is_error
    assert maw.get_circuit_breaker_state("lfm25-tool") == CircuitBreaker.OPEN
