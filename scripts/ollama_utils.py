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
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_TOKENS = 1024

# Default system prompt — shared across all washer scripts.
SYSTEM_PROMPT = """Du bist ein knallharter, menschlicher Lektor und Ghostwriter. Deine Aufgabe ist es, den übergebenen Text komplett neu zu verfassen und jegliche Muster von maschinell generierter Sprache restlos zu vernichten.

Halte dich an folgende absolute Restriktionen:
1. Burstiness maximieren: Wechsle radikal zwischen sehr kurzen, prägnanten Sätzen (1-4 Wörtern) und längeren, asymmetrischen Satzgefüden.
2. Perplexität erzwingen: Nutze unkonventionelle, treffende Verben. Vermeide vorhersehbare Adjektiv-Substantiv-Kombinationen.
3. Blacklist: Verwende NIEMALS Phrasen wie "Zusammenfassend lässt sich sagen", "Es ist wichtig zu beachten", "Ein weiteres Element" oder Wörter wie "facettenreich", "Geflecht", "Tapestry", "essenziell", "dynamisch".
4. Tonalität: Organisch, direkt und menschlich. Lass es leicht kantig klingen, als käme es aus der Feder eines erfahrenen Thriller-Autors. Keine weichgespülte Objektivität.
5. Output: Gib AUSSCHLIESSLICH den umgeschriebenen Text zurück. Keine Einleitungen, keine Erklärungen, keine Höflichkeitsfloskeln."""

# --- module-level cache for loaded model pool ---------------------------------

_models_cache: dict[str, Any] | None = None
_models_file_cache: Path | None = None


def _models_file() -> Path:
    """Locate ``scripts/models.yaml`` relative to this module."""
    here = Path(__file__).resolve().parent
    return here / "models.yaml"


def load_models() -> dict[str, Any]:
    """Load and cache the model pool from ``models.yaml``.

    Returns a dict with an optional ``default`` key plus a ``models`` mapping
    of ``model_name -> {size, description, default}``.
    """
    global _models_cache
    global _models_file_cache
    path = _models_file()
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


def call_ollama(
    prompt: str,
    model: str,
    system_prompt: str = "",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = 300,
) -> str:
    """Call a local Ollama model via HTTP and return the response text.

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
        Request timeout in seconds.
    """
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
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "").strip()
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"Ollama error ({model}): {exc}")


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
