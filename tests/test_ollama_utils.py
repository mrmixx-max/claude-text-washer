#!/usr/bin/env python3
"""Tests for the circuit-breaker pattern and HTTP retry layer in ollama_utils."""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ollama_utils import (  # noqa: E402
    CircuitBreaker,
    CircuitBreakerOpenError,
    DEFAULT_CB_COOLDOWN,
    DEFAULT_CB_FAILURE_THRESHOLD,
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


@pytest.fixture(autouse=True)
def _reset_breakers():
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


@pytest.fixture
def _no_backoff():
    with patch("scripts.ollama_utils._sleep_backoff", lambda *a, **k: None):
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
    cb.record_failure()  # trips immediately
    assert cb.state == CircuitBreaker.OPEN
    # Simulate cooldown having elapsed.
    cb._opened_at = time.time() - 100
    assert cb.is_open() is False  # transitions OPEN → HALF-OPEN, allows probe
    assert cb.state == CircuitBreaker.HALF_OPEN
    cb.record_success()
    assert cb.state == CircuitBreaker.CLOSED


def test_circuit_breaker_half_open_failure_reopens():
    cb = CircuitBreaker("m", failure_threshold=1, cooldown=60.0)
    cb.record_failure()  # trips
    cb._opened_at = time.time() - 100
    assert not cb.is_open()  # → HALF_OPEN, probe allowed
    cb.record_failure()  # probe failed → back to OPEN
    assert cb.state == CircuitBreaker.OPEN
    assert cb.is_open() is True


def test_circuit_breaker_allows_only_half_open_max_probes():
    cb = CircuitBreaker("m", failure_threshold=1, cooldown=0.0)
    cb.record_failure()
    cb._opened_at = time.time() - 100
    assert cb.is_open() is False  # HALF_OPEN
    assert cb.allow_probe() is True
    assert cb.allow_probe() is False  # only 1 probe allowed


# --------------------------------------------------------------------------- #
# call_ollama + breaker integration (mocked HTTP transport)
# --------------------------------------------------------------------------- #
def test_call_ollama_trips_breaker_after_repeated_429(_no_backoff):
    counter = {"n": 0}

    def fake_urlopen(req, timeout=None):
        counter["n"] += 1
        raise _http_error(429)

    model = "rate-limited-model"
    with patch("scripts.ollama_utils.urllib.request.urlopen", side_effect=fake_urlopen):
        # max_retries=3 → 4 attempts each. 3 consecutive *calls* trip the breaker.
        errs = []
        for _ in range(3):
            with pytest.raises(RuntimeError):
                call_ollama("prompt", model, max_retries=3)
            errs.append(get_circuit_breaker_state(model))
        # After 3 failures the breaker is OPEN.
        assert errs[-1] == CircuitBreaker.OPEN
        assert get_circuit_breaker_state(model) == CircuitBreaker.OPEN

        # 4th call must fail FAST — no further HTTP hits.
        with pytest.raises(CircuitBreakerOpenError):
            call_ollama("prompt", model, max_retries=3)

    # 3 calls × 4 attempts = 12 urlopen hits; the 4th failed-fast call added 0.
    assert counter["n"] == 12


def test_call_ollama_non_retryable_400_does_not_trip_breaker(_no_backoff):
    counter = {"n": 0}
    model = "bad-model"

    def fake_urlopen(req, timeout=None):
        counter["n"] += 1
        raise _http_error(400)

    with patch("scripts.ollama_utils.urllib.request.urlopen", side_effect=fake_urlopen):
        for _ in range(5):
            with pytest.raises(RuntimeError):
                call_ollama("prompt", model, max_retries=3)

    # 400 is not retryable → 1 attempt each, no sleep, breaker never trips.
    assert counter["n"] == 5
    assert get_circuit_breaker_state(model) == CircuitBreaker.CLOSED


def test_call_ollama_recovers_once_cooldown_elapses(_no_backoff):
    model = "flaky-model"
    calls = {"n": 0}

    def failing_urlopen(req, timeout=None):
        calls["n"] += 1
        raise _http_error(429)

    def success_urlopen(req, timeout=None):
        return _ok_response("recovered text")

    # Phase 1: trip the breaker with repeated 429s (threshold=2 for speed).
    with patch("scripts.ollama_utils.urllib.request.urlopen", side_effect=failing_urlopen):
        for _ in range(2):
            with pytest.raises(RuntimeError):
                call_ollama("p", model, max_retries=2, cb_failure_threshold=2)
    assert get_circuit_breaker_state(model) == CircuitBreaker.OPEN

    # Phase 2: breaker open → fail fast (no HTTP).
    with patch("scripts.ollama_utils.urllib.request.urlopen", side_effect=failing_urlopen):
        with pytest.raises(CircuitBreakerOpenError):
            call_ollama("p", model, max_retries=2, cb_failure_threshold=2)
    first_phase_calls = calls["n"]

    # Phase 3: simulate cooldown elapsed, then a half-open probe succeeds.
    breaker = get_circuit_breaker(model)
    breaker._opened_at = time.time() - 100
    breaker.cooldown = DEFAULT_CB_COOLDOWN
    with patch("scripts.ollama_utils.urllib.request.urlopen", side_effect=success_urlopen):
        result = call_ollama("p", model, max_retries=2, cb_failure_threshold=2)

    assert result == "recovered text"
    assert get_circuit_breaker_state(model) == CircuitBreaker.CLOSED
    # The fail-fast call in phase 2 did NOT hit the transport.
    assert calls["n"] == first_phase_calls


# --------------------------------------------------------------------------- #
# Multi-agent cascade prevention
# --------------------------------------------------------------------------- #
def test_multi_agent_washer_breaker_prevents_cascade(_no_backoff):
    """A persistently 429-ing model trips its own breaker and then fails FAST,
    so its retry storms never cascade into the sibling models."""
    import scripts.multi_agent_washer as maw

    # ``multi_agent_washer`` does a bare ``from ollama_utils import ...`` which
    # may load a *second* module instance under ``scripts.ollama_utils``.  The
    # breaker registry therefore lives in that instance's globals — read it
    # through the bound function so we observe exactly what call_ollama sees.
    wglobals = maw.call_ollama.__globals__
    wglobals["reset_circuit_breakers"]()
    wstate = wglobals["get_circuit_breaker_state"]

    per_model = {"lfm25-tool": 0, "llama3.2": 0, "eurollm-9b": 0}

    def fake_urlopen(req, timeout=None):
        model = _payload_model(req)
        per_model[model] += 1
        if model == "lfm25-tool":
            raise _http_error(429)
        return _ok_response(f"wash from {model}")

    with patch("scripts.ollama_utils.urllib.request.urlopen", side_effect=fake_urlopen):
        # 3 runs → lfm25-tool's breaker trips (3 consecutive failed call_ollama).
        for _ in range(3):
            best = maw.multi_agent_wash("dirty input text", verbose=False)
        assert best.model in ("llama3.2", "eurolm-9b")
        assert not best.is_error

        # 4th run: lfm25-tool must fail fast (circuit OPEN) — no new 429 hits —
        # while the other two models still get fresh HTTP calls and succeed.
        lfm_before = per_model["lfm25-tool"]
        best = maw.multi_agent_wash("dirty input text", verbose=False)

    # lfm25-tool was NOT retried during the 4th run (fail-fast).
    assert per_model["lfm25-tool"] == lfm_before
    # The healthy models were still exercised on the 4th run.
    assert per_model["llama3.2"] >= 4
    assert per_model["eurollm-9b"] >= 4
    assert best.model in ("llama3.2", "eurollm-9b")
    assert not best.is_error
    assert wstate("lfm25-tool") == CircuitBreaker.OPEN
