#!/usr/bin/env python3
"""Shared Ollama utilities for claude-text-washer.

Provides:
  - Model-pool loading from models.yaml
  - Model validation against the configured pool
  - ``--list-models`` formatting and early-exit helper
  - A single ``call_ollama`` HTTP helper used by all CLI scripts
"""
from __future__ import annotations

import json
import random
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_STREAM_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_TOKENS = 1024

# --- connection pooling ------------------------------------------------------
# A module-level opener with keep-alive so multi-agent runs reuse HTTP
# connections instead of opening a new TCP socket per call.
_opener_lock = threading.Lock()
_opener: urllib.request.OpenerDirector | None = None


def _get_opener() -> urllib.request.OpenerDirector:
    """Return the shared opener (lazy-initialized, thread-safe)."""
    global _opener
    if _opener is not None:
        return _opener
    with _opener_lock:
        if _opener is None:
            handler = urllib.request.HTTPHandler()
            _opener = urllib.request.build_opener(handler)
    return _opener

# --- retry / robustness configuration ---------------------------------------
DEFAULT_TIMEOUT = 300          # seconds — per HTTP request
DEFAULT_MAX_RETRIES = 3        # attempts after the initial call that fail transitively
DEFAULT_BACKOFF_BASE = 1.0     # seconds — initial backoff
DEFAULT_BACKOFF_MAX = 10.0     # seconds — cap on a single backoff sleep

# --- circuit breaker configuration ------------------------------------------
# A circuit breaker per model prevents a persistently-failing (e.g. repeatedly
# rate-limited 429) model from consuming retry/headoff budget and cascading
# failures across the parallel multi-agent washer.  After
# ``DEFAULT_CB_FAILURE_THRESHOLD`` *consecutive* failed ``call_ollama`` calls
# the breaker trips OPEN; subsequent calls fail fast for
# ``DEFAULT_CB_COOLDOWN`` seconds before a single half-open probe is allowed.
DEFAULT_CB_FAILURE_THRESHOLD = 3   # consecutive failures before tripping
DEFAULT_CB_COOLDOWN = 30.0         # seconds the breaker stays OPEN
DEFAULT_CB_HALF_OPEN_MAX_CALLS = 1  # probes allowed per half-open cycle

# HTTP status codes worth retrying (transient server-side failures).
# 429 = rate-limited, 502/503/504 = gateway / service unavailable.
_RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})

# Exception types that indicate a *transient* failure worth retrying.
# socket.timeout is a subclass of OSError on Python ≥ 3.10, but we list it
# explicitly for clarity and older interpreters.
_RETRYABLE_EXC_TYPES = (urllib.error.URLError, OSError, socket.timeout)


# --------------------------------------------------------------------------- #
# Circuit breaker — per-model, thread-safe
# --------------------------------------------------------------------------- #
class CircuitBreakerOpenError(RuntimeError):
    """Raised when a model's circuit breaker is OPEN.

    Failing fast avoids exhausting retry/backoff budget (and triggering a
    cascade of retries across the parallel multi-agent washer) when a model
    is persistently failing — e.g. returning repeated HTTP 429 responses.
    """


class CircuitBreaker:
    """Per-model circuit breaker (Closed → Open → Half-Open → Closed).

    Only *consecutive* failures trip the breaker: a single ``record_success``
    resets the failure counter and closes the circuit.

    Thread-safe via an internal lock.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        model: str,
        failure_threshold: int = DEFAULT_CB_FAILURE_THRESHOLD,
        cooldown: float = DEFAULT_CB_COOLDOWN,
        half_open_max_calls: int = DEFAULT_CB_HALF_OPEN_MAX_CALLS,
    ) -> None:
        self.model = model
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self.half_open_max_calls = half_open_max_calls
        self._lock = threading.Lock()
        self._state = self.CLOSED
        self._failure_count = 0
        self._opened_at: float = 0.0
        self._half_open_probes = 0

    @property
    def state(self) -> str:
        """Current state (read-only snapshot)."""
        with self._lock:
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    def is_open(self) -> bool:
        """Return True if the breaker is currently OPEN (fail-fast).

        If the breaker is OPEN but the cooldown has elapsed, it transitions
        to HALF-OPEN and returns False so a single probe request is allowed
        through.
        """
        with self._lock:
            if self._state == self.OPEN:
                if (time.time() - self._opened_at) >= self.cooldown:
                    self._state = self.HALF_OPEN
                    self._half_open_probes = 0
                else:
                    return True
            return False

    def record_success(self) -> None:
        """Record a successful call — resets the breaker to CLOSED."""
        with self._lock:
            self._failure_count = 0
            self._state = self.CLOSED
            self._half_open_probes = 0

    def record_failure(self) -> None:
        """Record a failed call; trip OPEN once the threshold is reached."""
        with self._lock:
            self._failure_count += 1
            if self._state == self.HALF_OPEN:
                # A probe failed — re-open immediately.
                self._state = self.OPEN
                self._opened_at = time.time()
                return
            if self._failure_count >= self.failure_threshold:
                self._state = self.OPEN
                self._opened_at = time.time()

    def allow_probe(self) -> bool:
        """In HALF-OPEN, allow up to ``half_open_max_calls`` probe(s)."""
        with self._lock:
            if self._state != self.HALF_OPEN:
                return False
            if self._half_open_probes < self.half_open_max_calls:
                self._half_open_probes += 1
                return True
            return False


# --- module-level breaker registry ------------------------------------------
# One CircuitBreaker per model name, shared across threads so the parallel
# multi-agent workers all observe the same trip state.
_circuit_breakers: dict[str, CircuitBreaker] = {}
_circuit_breakers_lock = threading.Lock()


def get_circuit_breaker(
    model: str,
    failure_threshold: int = DEFAULT_CB_FAILURE_THRESHOLD,
    cooldown: float = DEFAULT_CB_COOLDOWN,
    half_open_max_calls: int = DEFAULT_CB_HALF_OPEN_MAX_CALLS,
) -> CircuitBreaker:
    """Return the shared :class:`CircuitBreaker` for *model* (creating it once)."""
    with _circuit_breakers_lock:
        cb = _circuit_breakers.get(model)
        if cb is None:
            cb = CircuitBreaker(
                model,
                failure_threshold=failure_threshold,
                cooldown=cooldown,
                half_open_max_calls=half_open_max_calls,
            )
            _circuit_breakers[model] = cb
        return cb


def reset_circuit_breakers() -> None:
    """Clear the entire breaker registry (test helper / admin escape hatch)."""
    with _circuit_breakers_lock:
        _circuit_breakers.clear()


def get_circuit_breaker_state(model: str) -> str:
    """Return the state name of the breaker for *model*, or ``'absent'``."""
    with _circuit_breakers_lock:
        cb = _circuit_breakers.get(model)
    return cb.state if cb is not None else "absent"

# Default system prompt — shared across all washer scripts.
SYSTEM_PROMPT = """Du bist ein knallharter, menschlicher Lektor und Ghostwriter. Deine Aufgabe ist es, den übergebenen Text komplett neu zu verfassen und jegliche Muster von maschinell generierter Sprache restlos zu vernichten.

Halte dich an folgende absolute Restriktionen:
1. Burstiness maximieren: Wechsle radikal zwischen sehr kurzen, prägnanten Sätzen (1-4 Wörtern) und längeren, asymmetrischen Satzgefüden.
2. Perplexität erzwingen: Nutze unkonventionelle, treffende Verben. Vermeide vorhersehbare Adjektiv-Substantiv-Kombinationen.
3. Blacklist: Verwende NIEMALS Phrasen wie "Zusammenfassend lässt sich sagen", "Es ist wichtig zu beachten", "Ein weiteres Element" oder Wörter wie "facettenreich", "Geflecht", "Tapestry", "essenziell", "dynamisch".
4. Tonalität: Organisch, direkt und menschlich. Lass es leicht kantig klingen, als käme es aus der Feder eines erfahrenen Thriller-Autors. Keine weichgespülte Objektivität.
5. Output: Gib AUSSCHLIESSLICH den umgeschriebenen Text zurück. Keine Einleitungen, keine Erklärungen, keine Höflichkeitsfloskeln."""

# --- module-level cache for loaded model pool ---------------------------------
# Guarded by ``_models_lock`` so parallel workers (ThreadPoolExecutor) that
# race on first access don't each re-read & re-parse models.yaml.
_models_cache: dict[str, Any] | None = None
_models_file_cache: Path | None = None
_models_lock = threading.Lock()


def _models_file() -> Path:
    """Locate ``scripts/models.yaml`` relative to this module."""
    here = Path(__file__).resolve().parent
    return here / "models.yaml"


def load_models() -> dict[str, Any]:
    """Load and cache the model pool from ``models.yaml``.

    Returns a dict with an optional ``default`` key plus a ``models`` mapping
    of ``model_name -> {size, description, default}``.

    Thread-safe: the YAML is parsed at most once thanks to a module-level
    lock and double-checked locking, so parallel ``wash_batch`` workers
    never contend on file I/O.
    """
    global _models_cache
    global _models_file_cache
    path = _models_file()
    # Fast path — cache already warm for this file, no lock needed.
    if _models_cache is not None and _models_file_cache == path:
        return _models_cache
    with _models_lock:
        # Re-check inside the lock (another thread may have loaded it).
        if _models_file_cache != path or _models_cache is None:
            try:
                import yaml  # PyYAML
            except ImportError:
                raise RuntimeError(
                    "PyYAML is required to read models.yaml. "
                    "Install with:  uv pip install pyyaml"
                )
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            _models_cache = data
            _models_file_cache = path
        return _models_cache


def get_model_names() -> list[str]:
    """Return the list of model names defined in the pool."""
    return list(load_models().get("models", {}).keys())


def get_default_model() -> str:
    """Return the name of the default model.

    Falls back to the ``default`` top-level key, then to any model
    marked ``default: true``, then to the first entry.
    """
    data = load_models()
    # 1) top-level `default` key
    top_default = data.get("default")
    if top_default and top_default in data.get("models", {}):
        return str(top_default)
    # 2) per-model `default: true`
    for name, cfg in data.get("models", {}).items():
        if isinstance(cfg, dict) and cfg.get("default"):
            return name
    # 3) first entry
    names = get_model_names()
    if names:
        return names[0]
    raise RuntimeError("No models defined in models.yaml")


def validate_model(model_name: str) -> bool:
    """Return ``True`` if *model_name* exists in the pool."""
    return model_name in load_models().get("models", {})


def get_model_config(model_name: str) -> dict[str, Any]:
    """Return the config dict for *model_name*."""
    models = load_models().get("models", {})
    return models.get(model_name, {})


def format_model_list() -> str:
    """Return a human-readable multi-line listing of all models."""
    data = load_models()
    models = data.get("models", {})
    default = get_default_model()
    width = max((len(n) for n in models), default=0)
    lines: list[str] = []
    for name, cfg in models.items():
        size = cfg.get("size", "") if isinstance(cfg, dict) else ""
        desc = cfg.get("description", "") if isinstance(cfg, dict) else ""
        marker = " *" if name == default else ""
        lines.append(f"  {name:<{width}}{marker}  ({size})  {desc}")
    return "\n".join(lines)


def handle_list_models() -> None:
    """Print the model list and exit the process (for CLI ``--list-models``)."""
    print("Available Ollama models:")
    print(format_model_list())
    print()
    print(f"Default model: {get_default_model()}")
    print(f"(* marks the default)")
    sys.exit(0)


def _sleep_backoff(attempt: int, base: float, maximum: float) -> None:
    """Sleep before a retry using exponential backoff with full jitter.

    ``attempt`` is the zero-based retry index (0 = first retry).
    The sleep is ``random.uniform(0, cap)`` where
    ``cap = min(base * 2**attempt, maximum)`` — "full jitter" is the
    recommended backoff strategy for retrying transient failures.
    """
    cap = min(base * (2 ** attempt), maximum)
    time.sleep(random.uniform(0, cap))


def _is_retryable_exception(exc: BaseException) -> bool:
    """Return True if *exc* represents a transient Ollama failure to retry."""
    # HTTPError carries a .code we check separately, but is also a URLError.
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in _RETRYABLE_HTTP_STATUS
    return isinstance(exc, _RETRYABLE_EXC_TYPES)


def call_ollama(
    prompt: str,
    model: str,
    system_prompt: str = "",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    backoff_max: float = DEFAULT_BACKOFF_MAX,
    cb_failure_threshold: int = DEFAULT_CB_FAILURE_THRESHOLD,
    cb_cooldown: float = DEFAULT_CB_COOLDOWN,
) -> str:
    """Call a local Ollama model via HTTP and return the response text.

    Retries with exponential backoff + jitter on *transient* failures
    (``URLError`` / ``OSError`` / ``socket.timeout`` and HTTP 429 & 5xx).
    Non-retryable failures (e.g. HTTP 400 Bad Request, a 404 for a missing
    model) raise immediately.

    A per-model **circuit breaker** wraps the retries: after
    ``cb_failure_threshold`` *consecutive* failed calls the breaker trips
    OPEN and further calls to that model fail fast with
    :class:`CircuitBreakerOpenError` — preventing a persistently
    rate-limited (429) model from cascading retries across the parallel
    multi-agent washer.  The breaker closes again on the first success
    (or after ``cb_cooldown`` seconds when in HALF-OPEN).

    Parameters
    ----------
    prompt:
        The user prompt / text to send.
    model:
        Ollama model name (e.g. ``llama3.2``).
    system_prompt:
        Optional system prompt to steer the model.
    temperature:
        Sampling temperature.
    max_tokens:
        Maximum tokens to generate (``num_predict`` in Ollama).
    timeout:
        Per-request HTTP timeout in seconds.
    max_retries:
        Number of *additional* attempts after the initial call fails
        transiently (default 3).
    backoff_base:
        Base seconds for exponential backoff (default 1.0).
    backoff_max:
        Cap on a single backoff sleep in seconds (default 10.0).
    cb_failure_threshold:
        Consecutive failures required to trip the circuit breaker (default 3).
    cb_cooldown:
        Seconds the breaker stays OPEN before a half-open probe is allowed
        (default 30.0).
    """
    breaker = get_circuit_breaker(
        model,
        failure_threshold=cb_failure_threshold,
        cooldown=cb_cooldown,
    )
    if breaker.is_open():
        # Fail fast: the model has been persistently unhealthy (e.g. repeated
        # 429s).  Raising here skips the HTTP call and its retry/backoff loop,
        # so rate-limited models don't cascade retries into the other workers.
        raise CircuitBreakerOpenError(
            f"Circuit breaker OPEN for model '{model}' after "
            f"{breaker.failure_count} consecutive failures — failing fast "
            f"(cooldown {cb_cooldown}s). This prevents cascading retries "
            f"across parallel agents."
        ) from None

    payload: dict[str, Any] = {
        "model": model,
        "system": system_prompt,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    data = json.dumps(payload).encode("utf-8")

    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(
            OLLAMA_URL,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            opener = _get_opener()
            with opener.open(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                breaker.record_success()
                return result.get("response", "").strip()
        except urllib.error.HTTPError as exc:
            # HTTPError is a subclass of URLError but carries a status code.
            # Only retry on server-side / rate-limit codes.
            if exc.code in _RETRYABLE_HTTP_STATUS:
                if attempt < max_retries:
                    last_exc = exc
                    _sleep_backoff(attempt, backoff_base, backoff_max)
                    continue
                # Retries exhausted on a *retryable* status (e.g. 429) — the
                # model is persistently unhealthy, so trip the circuit breaker.
                breaker.record_failure()
            raise RuntimeError(f"Ollama error ({model}): {exc}") from exc
        except _RETRYABLE_EXC_TYPES as exc:
            last_exc = exc
            if attempt < max_retries:
                _sleep_backoff(attempt, backoff_base, backoff_max)
                continue
            breaker.record_failure()
            raise RuntimeError(f"Ollama error ({model}): {exc}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as exc:
            # Malformed response — not retryable (retry won't fix a parse error)
            # and not a transient/429 condition, so we do NOT trip the breaker.
            raise RuntimeError(f"Ollama malformed response ({model}): {exc}") from exc

    # Exhausted retries. Should be unreachable, but guard anyway.
    breaker.record_failure()
    raise RuntimeError(f"Ollama error ({model}): {last_exc}") from last_exc


def call_ollama_stream(
    prompt: str,
    model: str,
    system_prompt: str = "",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
):
    """Call Ollama with stream=True, yield text chunks as they arrive.

    Yields stripped text chunks. Raises RuntimeError on failure.
    No retry/backoff — streaming calls are not retried (would replay
    the whole stream). Use non-streaming call_ollama for retry logic.
    """
    payload: dict[str, Any] = {
        "model": model,
        "system": system_prompt,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_STREAM_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    opener = _get_opener()
    try:
        with opener.open(req, timeout=timeout) as resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    raise RuntimeError(f"Ollama error: {chunk['error']}")
                yield chunk.get("response", "")
                if chunk.get("done"):
                    return
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"Ollama streaming error ({model}): {exc}") from exc


def resolve_model(
    requested: str | None,
    script_default: str = "llama3.2",
) -> str:
    """Resolve the effective model name.

    Priority:
      1. Explicit ``--model`` value (validated against pool; error if unknown)
      2. Pool default from ``models.yaml``
      3. *script_default* fallback
    """
    if requested:
        if not validate_model(requested):
            available = ", ".join(get_model_names())
            raise ValueError(
                f"Model '{requested}' is not in the configured pool.\n"
                f"Available models: {available}\n"
                f"Use --list-models to see the full list."
            )
        return requested
    try:
        return get_default_model()
    except (RuntimeError, FileNotFoundError):
        return script_default


def add_model_args(parser) -> None:
    """Attach standard ``--model`` and ``--list-models`` arguments to *parser*."""
    parser.add_argument(
        "--model",
        default=None,
        help=f"Ollama model to use (default: {get_default_model()})",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available Ollama models and exit",
    )
