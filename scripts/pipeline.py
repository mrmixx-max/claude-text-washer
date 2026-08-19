#!/usr/bin/env python3
"""Multi-pass text washing pipeline with swappable local LLMs.

Supports any Ollama model from the pool defined in scripts/models.yaml.
  --model MODEL       Select model (overrides --preset; default: llama3.2)
  --list-models       Show all available models and exit
  --preset NAME       Use a speed/quality preset (fast|standard|premium)
  --passes N          Number of rewrite passes (1-3)
  --temperature N     Override sampling temperature
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Ensure sibling modules (ollama_utils.py) are importable when run directly
# or when imported as part of the `scripts` package by pytest.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ollama_utils import (
    SYSTEM_PROMPT,
    call_ollama,
    get_default_model,
    handle_list_models,
    resolve_model,
)

# Model presets — speed vs quality tradeoffs.
# Each preset can be overridden by an explicit --model flag.
MODELS: dict[str, dict] = {
    "fast": {"model": "lfm25-tool", "temperature": 0.7, "max_tokens": 512},
    "standard": {"model": "llama3.2", "temperature": 0.8, "max_tokens": 1024},
    "premium": {"model": "llama3.2", "temperature": 0.9, "max_tokens": 2048},
}


@dataclass
class WashResult:
    text: str
    model: str
    duration: float
    passes: int


def wash_pass(text: str, preset: str = "standard") -> WashResult:
    """Single-pass wash with specified model preset.

    Parameters
    ----------
    text:
        Text to rewrite.
    preset:
        One of ``fast``, ``standard``, ``premium``. If a custom model was
        injected via :func:`set_override_model`, it is used instead.
    """
    cfg = MODELS[preset]
    start = time.time()
    cleaned = call_ollama(
        prompt=text,
        model=cfg["model"],
        system_prompt=SYSTEM_PROMPT,
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
    )
    duration = time.time() - start
    return WashResult(cleaned, cfg["model"], duration, 1)


def wash_multi_pass(text: str, preset: str = "standard", passes: int = 2) -> WashResult:
    """Multi-pass wash for deeper cleaning."""
    cfg = MODELS[preset]
    current = text
    total_start = time.time()

    for i in range(passes):
        # Vary temperature slightly per pass for diversity
        temp = cfg["temperature"] + (i * 0.05)
        current = call_ollama(
            prompt=current,
            model=cfg["model"],
            system_prompt=SYSTEM_PROMPT,
            temperature=temp,
            max_tokens=cfg["max_tokens"],
        )

    total_duration = time.time() - total_start
    return WashResult(current, cfg["model"], total_duration, passes)


def resolve_preset(
    preset: str = "standard",
    model: str | None = None,
    temperature: float | None = None,
) -> dict:
    """Resolve a model config dict for *preset* with optional overrides.

    Unlike :func:`_set_override_model`, this returns a **fresh copy** and
    never mutates the shared ``MODELS`` mapping, so it is safe to call
    concurrently from parallel wash workers (no shared-global race).
    """
    base = MODELS.get(preset, MODELS["standard"])
    cfg = dict(base)  # shallow copy — enough, values are immutable scalars
    if model:
        cfg["model"] = model
    if temperature is not None:
        cfg["temperature"] = temperature
    return cfg


def wash_pass_cfg(
    text: str,
    cfg: dict,
    *,
    max_retries: int = 3,
    timeout: int = 300,
) -> WashResult:
    """Single-pass wash using an explicit config dict (thread-safe).

    *cfg* must contain ``model``, ``temperature`` and ``max_tokens`` keys,
    e.g. as produced by :func:`resolve_preset`.
    """
    start = time.time()
    cleaned = call_ollama(
        prompt=text,
        model=cfg["model"],
        system_prompt=SYSTEM_PROMPT,
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        timeout=timeout,
        max_retries=max_retries,
    )
    return WashResult(cleaned, cfg["model"], time.time() - start, 1)


def wash_multi_pass_cfg(
    text: str,
    cfg: dict,
    passes: int = 2,
    *,
    max_retries: int = 3,
    timeout: int = 300,
) -> WashResult:
    """Multi-pass wash using an explicit config dict (thread-safe).

    Temperature is varied slightly per pass for diversity, mirroring
    :func:`wash_multi_pass` but without touching global state.
    """
    current = text
    total_start = time.time()
    for i in range(passes):
        temp = cfg["temperature"] + (i * 0.05)
        current = call_ollama(
            prompt=current,
            model=cfg["model"],
            system_prompt=SYSTEM_PROMPT,
            temperature=temp,
            max_tokens=cfg["max_tokens"],
            timeout=timeout,
            max_retries=max_retries,
        )
    return WashResult(current, cfg["model"], time.time() - total_start, passes)


def _set_override_model(model: str, temperature: float | None = None) -> None:
    """Override the 'standard' preset with a custom model.

    This allows ``--model`` to override the preset-driven model while
    keeping the preset interface intact for ``wash_pass`` / ``wash_multi_pass``.
    """
    cfg = MODELS["standard"]
    cfg["model"] = model
    if temperature is not None:
        cfg["temperature"] = temperature


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Claude Text Washer — multi-pass AI marker removal"
    )
    parser.add_argument("input", nargs="?", help="Input text file or - for stdin")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument(
        "--preset",
        choices=["fast", "standard", "premium"],
        default="standard",
        help="Model preset (fast=lfm25-tool, standard/premium=llama3.2)",
    )
    parser.add_argument("--passes", type=int, default=1, help="Number of rewrite passes (1-3)")
    parser.add_argument(
        "--model",
        default=None,
        help=f"Override Ollama model (ignores --preset; default: {get_default_model()})",
    )
    parser.add_argument("--temperature", type=float, help="Override sampling temperature")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available Ollama models and exit",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list_models:
        handle_list_models()

    if not args.input:
        parser.error("input file is required (or use --list-models)")

    # Resolve model: explicit --model wins (validated), else use pool default.
    try:
        model = resolve_model(args.model, script_default="llama3.2")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.input == "-":
        text = sys.stdin.read()
    else:
        text = Path(args.input).read_text(encoding="utf-8")

    # If --model was given, inject it into the "standard" preset so it
    # overrides the speed/quality presets while keeping the preset interface
    # intact for wash_pass / wash_multi_pass.
    if args.model:
        _set_override_model(model, args.temperature)
        preset = "standard"
    else:
        preset = args.preset
        if args.temperature is not None:
            MODELS[preset]["temperature"] = args.temperature

    if args.passes == 1:
        result = wash_pass(text, preset)
    else:
        result = wash_multi_pass(text, preset, args.passes)

    if args.output:
        Path(args.output).write_text(result.text, encoding="utf-8")
        print(
            f"Wrote {args.output} ({result.duration:.1f}s, {result.passes} pass(es), model={result.model})",
            file=sys.stderr,
        )
    else:
        print(result.text)


if __name__ == "__main__":
    main()
