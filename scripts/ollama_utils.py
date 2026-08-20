#!/usr/bin/env python3
"""Shared LLM utilities for claude-text-washer.

Provides:
  - Model-pool loading from models.yaml
  - Model validation against the configured pool
  - ``--list-models`` formatting and early-exit helper
  - A generic ``call_llm`` HTTP helper used by all CLI scripts
  - Backend abstraction: Ollama + OpenAI-compatible APIs (vLLM, LM Studio, OpenRouter, ...)
"""
from __future__ import annotations

import json
import os
import random
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- backend configuration ---------------------------------------------------
# Default: Ollama local. Override via set_backend() or CLI args.

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_STREAM_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_TOKENS = 1024


@dataclass
class SamplingConfig:
    """LLM sampling parameters for generation control."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repeat_penalty: float = 1.1
    min_p: float = 0.0
    seed: int | None = None
    max_tokens: int = 4096
    stop: list[str] = field(default_factory=list)
    response_format: str | None = None


# Preset configs for common use cases
SAMPLING_PRESETS: dict[str, SamplingConfig] = {
    "conservative": SamplingConfig(temperature=0.3, top_p=0.8, top_k=20),
    "balanced": SamplingConfig(temperature=0.7, top_p=0.9, top_k=40),
    "creative": SamplingConfig(temperature=0.9, top_p=0.95, top_k=60),
    "chaotic": SamplingConfig(temperature=1.2, top_p=1.0, top_k=100),
}

DEFAULT_CFG = SamplingConfig()


@dataclass
class BackendConfig:
    """Configuration for an LLM backend.

    Attributes
    ----------
    name:
        Human-readable name (e.g. "ollama-local", "vllm", "lm-studio").
    base_url:
        API endpoint URL for generation.
    stream_url:
        API endpoint URL for streaming (defaults to base_url if unset).
    backend_type:
        ``"ollama"`` or ``"openai"``. Use ``"auto"`` to detect from URL.
    api_key:
        Optional API key (sent as Bearer token).
    extra_headers:
        Optional extra HTTP headers.
    """
    name: str = "ollama-local"
    base_url: str = OLLAMA_URL
    stream_url: str | None = None
    backend_type: str = "auto"
    api_key: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stream_url is None:
            self.stream_url = self.base_url
        if self.backend_type == "auto":
            self.backend_type = _detect_backend_type(self.base_url)


def _detect_backend_type(url: str) -> str:
    """Detect backend type from URL pattern.

    Returns ``"ollama"`` if the URL looks like an Ollama endpoint,
    ``"openai"`` otherwise (vLLM, LM Studio, OpenRouter, Together, etc.).
    """
    url_lower = url.lower()
    if "11434" in url or "/api/generate" in url_lower or "/api/chat" in url_lower:
        return "ollama"
    return "openai"


# --- global backend state -----------------------------------------------------
_backend_config: BackendConfig | None = None
_backend_lock = threading.Lock()


def set_backend(config: BackendConfig) -> None:
    """Set the global backend configuration (thread-safe)."""
    global _backend_config
    with _backend_lock:
        _backend_config = config


def get_backend() -> BackendConfig:
    """Return the global backend configuration (default: Ollama local)."""
    global _backend_config
    if _backend_config is not None:
        return _backend_config
    with _backend_lock:
        if _backend_config is None:
            _backend_config = BackendConfig()
        return _backend_config


def reset_backend() -> None:
    """Reset backend to default (Ollama local)."""
    global _backend_config
    with _backend_lock:
        _backend_config = None

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
SYSTEM_PROMPT = """You are a hard-nosed, human editor and ghostwriter. Your job is to completely rewrite the given text and destroy every trace of machine-generated language.

Follow these absolute rules:
1. Maximize burstiness: alternate radically between very short, punchy sentences (1-4 words) and longer, asymmetrical sentence structures.
2. Force perplexity: use unconventional, precise verbs. Avoid predictable adjective-noun combinations.
3. Blacklist: NEVER use phrases like "In conclusion, it is clear", "It is important to note", "Another element", or words like "multifaceted", "tapestry", "essential", "dynamic".
4. Tone: organic, direct, and human. Make it sound slightly edgy, as if written by a seasoned thriller author. No washed-out objectivity.
5. Output: return ONLY the rewritten text. No introductions, no explanations, no pleasantries."""

# --- module-level cache for loaded model pool ---------------------------------
# Guarded by ``_models_lock`` so parallel workers (ThreadPoolExecutor) that
# race on first access don't each re-read & re-parse models.yaml.
_models_cache: dict[str, Any] | None = None
_models_file_cache: Path | None = None
_models_lock = threading.Lock()


def _models_file() -> Path:
    """Locate ``scripts/models.yaml`` relative to this module or PyInstaller bundle."""
    if getattr(sys, "_MEIPASS", None):
        base = Path(sys._MEIPASS) / "scripts"
    else:
        base = Path(__file__).resolve().parent
    return base / "models.yaml"


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
    """Print the model list and backend info, then exit."""
    print("Available models (Ollama pool):")
    print(format_model_list())
    print()
    print(f"Default model: {get_default_model()}")
    print(f"(* marks the default)")
    print()
    backend = get_backend()
    print(f"Active backend: {backend.name} ({backend.backend_type})")
    print(f"  URL: {backend.base_url}")
    if backend.api_key:
        print(f"  API key: ***{backend.api_key[-4:]}")
    print()
    print("Use --base-url to target any OpenAI-compatible API (vLLM, LM Studio, OpenRouter, etc.)")
    print("Use --backend-type ollama|openai|auto to override auto-detection")
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


def call_llm(
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
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    backend_type: str | None = None,
) -> str:
    """Call an LLM via HTTP and return the response text.

    Supports both Ollama and OpenAI-compatible APIs (vLLM, LM Studio,
    OpenRouter, Together, etc.). The backend type is auto-detected from
    the URL or can be specified explicitly.

    Retries with exponential backoff + jitter on *transient* failures
    (``URLError`` / ``OSError`` / ``socket.timeout`` and HTTP 429 & 5xx).
    Non-retryable failures raise immediately.

    A per-model **circuit breaker** wraps the retries.
    """
    backend = get_backend()
    url = base_url or backend.base_url
    key = api_key if api_key is not None else backend.api_key
    btype = backend_type or backend.backend_type
    if btype == "auto":
        btype = _detect_backend_type(url)

    breaker = get_circuit_breaker(
        f"{btype}:{url}:{model}",
        failure_threshold=cb_failure_threshold,
        cooldown=cb_cooldown,
    )
    if breaker.is_open():
        raise CircuitBreakerOpenError(
            f"Circuit breaker OPEN for model '{model}' via {btype} at {url} "
            f"after {breaker.failure_count} consecutive failures — failing fast "
            f"(cooldown {cb_cooldown}s)."
        ) from None

    payload = _build_payload(prompt, model, system_prompt, temperature, max_tokens, btype)
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    headers.update(backend.extra_headers)

    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            opener = _get_opener()
            with opener.open(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                breaker.record_success()
                return _extract_response(result, btype).strip()
        except urllib.error.HTTPError as exc:
            if exc.code in _RETRYABLE_HTTP_STATUS:
                if attempt < max_retries:
                    last_exc = exc
                    _sleep_backoff(attempt, backoff_base, backoff_max)
                    continue
                breaker.record_failure()
            raise RuntimeError(f"LLM error ({model}@{url}): {exc}") from exc
        except _RETRYABLE_EXC_TYPES as exc:
            last_exc = exc
            if attempt < max_retries:
                _sleep_backoff(attempt, backoff_base, backoff_max)
                continue
            breaker.record_failure()
            raise RuntimeError(f"LLM error ({model}@{url}): {exc}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as exc:
            raise RuntimeError(f"LLM malformed response ({model}@{url}): {exc}") from exc

    breaker.record_failure()
    raise RuntimeError(f"LLM error ({model}@{url}): {last_exc}") from last_exc


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

    Backward-compatible alias for :func:`call_llm` with Ollama defaults.
    """
    return call_llm(
        prompt=prompt,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        backoff_base=backoff_base,
        backoff_max=backoff_max,
        cb_failure_threshold=cb_failure_threshold,
        cb_cooldown=cb_cooldown,
    )


def _build_payload(
    prompt: str,
    model: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    backend_type: str,
) -> dict[str, Any]:
    """Build request payload for the given backend type."""
    if backend_type == "ollama":
        return {
            "model": model,
            "system": system_prompt,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
    # OpenAI-compatible format
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def _extract_response(result: dict[str, Any], backend_type: str) -> str:
    """Extract text from response based on backend type."""
    if backend_type == "ollama":
        return result.get("response", "")
    # OpenAI-compatible format
    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        # Fallback: try common response formats
        if "response" in result:
            return result["response"]
        if "content" in result:
            return result["content"]
        if "text" in result:
            return result["text"]
        raise RuntimeError(f"Unexpected response format: {list(result.keys())}")


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
    *,
    allow_remote: bool = True,
) -> str:
    """Resolve the effective model name.

    Priority:
      1. Explicit ``--model`` value (validated against pool unless ``allow_remote=True``)
      2. Pool default from ``models.yaml``
      3. *script_default* fallback

    When ``allow_remote`` is True and the model is not in the pool,
    it is accepted as-is (for use with remote backends like OpenRouter, vLLM, etc.).
    """
    if requested:
        if not validate_model(requested):
            if not allow_remote:
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


def add_backend_args(parser) -> None:
    """Attach backend-related CLI arguments to *parser*.

    Adds ``--base-url``, ``--api-key``, ``--backend-type`` and
    ``--temperature`` for full control over the LLM backend.
    """
    parser.add_argument(
        "--base-url",
        default=None,
        help="LLM API endpoint URL (default: http://127.0.0.1:11434/api/generate for Ollama)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for remote backends (sent as Bearer token)",
    )
    parser.add_argument(
        "--backend-type",
        choices=["ollama", "openai", "auto"],
        default=None,
        help="Backend type: 'ollama' for local Ollama, 'openai' for OpenAI-compatible APIs (vLLM, LM Studio, OpenRouter), 'auto' to detect from URL",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (0.0=deterministic, 1.0=creative). Default: 0.8",
    )


def add_model_args(parser) -> None:
    """Attach standard ``--model`` and ``--list-models`` arguments to *parser*."""
    parser.add_argument(
        "--model",
        default=None,
        help=f"Model to use (default: {get_default_model()})",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit",
    )


def resolve_temperature(requested: float | None, *, interactive: bool = False) -> float:
    """Resolve the effective temperature.

    Priority:
      1. Explicit ``requested`` value (from CLI ``--temperature``)
      2. Interactive prompt (if ``interactive=True`` and no value given)
      3. Default (0.8)
    """
    if requested is not None:
        return clamp_temperature(requested)

    if interactive and sys.stdin.isatty():
        return _interactive_temperature()

    return DEFAULT_TEMPERATURE


def _interactive_temperature() -> float:
    """Prompt the user to choose a temperature preset."""
    presets = {
        "1": ("Conservative (0.3) — precise, repetitive, less creative", 0.3),
        "2": ("Balanced (0.7) — natural, slightly varied", 0.7),
        "3": ("Creative (0.9) — diverse, unpredictable (default)", 0.9),
        "4": ("Chaotic (1.2) — highly random, experimental", 1.2),
    }
    print("\n🌡️  Temperature selection:", file=sys.stderr)
    for key, (desc, _) in presets.items():
        marker = " ← default" if key == "3" else ""
        print(f"   [{key}] {desc}{marker}", file=sys.stderr)
    print(file=sys.stderr)
    while True:
        try:
            choice = input("Choose [1-4] (Enter=default): ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return DEFAULT_TEMPERATURE
        if not choice:
            return DEFAULT_TEMPERATURE
        if choice in presets:
            return presets[choice][1]
        # Allow direct numeric input
        try:
            return clamp_temperature(float(choice))
        except ValueError:
            print("   Invalid choice. Enter 1-4 or a number like 0.5", file=sys.stderr)


def clamp_temperature(value: float) -> float:
    """Clamp temperature to a safe range [0.0, 2.0]."""
    return max(0.0, min(2.0, value))
