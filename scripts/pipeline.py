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
from cli_utils import (  # noqa: E402
    ProgressBar,
    add_common_args,
    print_error,
    print_info,
    print_success,
    read_input_text,
    write_output_text,
)
from smart_cleaner import clean_text, get_marker_count  # noqa: E402
from stat_engine import analyze_text  # noqa: E402

# Model presets — speed vs quality tradeoffs.
# Each preset can be overridden by an explicit --model flag.
MODELS: dict[str, dict] = {
    "fast": {"model": "lfm25-tool", "temperature": 0.7, "max_tokens": 512},
    "standard": {"model": "llama3.2", "temperature": 0.8, "max_tokens": 1024},
    "premium": {"model": "llama3.2", "temperature": 0.9, "max_tokens": 2048},
}

# Progressive pass prompts — each pass has a different focus.
# Pass 1: General rewrite (destroy AI patterns)
# Pass 2: Pattern fixer (target remaining markers, vary structure)
# Pass 3: Naturalizer (break rhythm, add imperfections)
PASS_PROMPTS: list[str] = [
    # Pass 1: General rewrite — destroy AI patterns (uses default SYSTEM_PROMPT)
    SYSTEM_PROMPT,
    # Pass 2: Pattern fixer
    """Du bist ein Spezialist für das Entfernen von KI-Textmustern. Der folgende Text wurde bereits einmal umgeschrieben, aber enthält noch versteckte AI-Muster.

Deine Aufgabe:
1. Finde und entferne alle verbleibenden KI-typischen Phrasen (Zusammenfassungen, Füllwörter, Template-Strukturen)
2. Variiere die Satzlänge radikal: wechsle zwischen kurzen (3-5 Wörter) und langen (20+ Wörtern) Sätzen
3. Brich rhythmische Muster — wenn zwei Sätze gleich lang sind, ändere einen
4. Ersetze generische Adjektiv-Substantiv-Kombinationen durch treffendere Verben
5. Gib NUR den überarbeiteten Text zurück""",
    # Pass 3: Naturalizer
    """Du bist ein erfahrener Lektor, der Texte natürlich und menschlich klingen lässt. Der Text wurde bereits zweimal bearbeitet.

Deine Aufgabe:
1. Brich alle verbleibenden rhythmischen Strukturen
2. Füge kontrollierte "Unvollkommenheiten" ein: ein Fragment hier, ein Doppelpunkt dort
3. Stelle sicher, dass keine zwei aufeinanderfolgenden Sätze ähnliche Länge haben
4. Ersetze formelle Wendungen durch umgangssprachlichere Alternativen
5. Achte auf Tonalität: kantig, direkt, wie von einem Thriller-Autor
6. Gib NUR den finalen Text zurück""",
]

# Early termination threshold — if ai_score drops below this, stop early
EARLY_TERMINATION_THRESHOLD = 25.0


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
    """Multi-pass wash for deeper cleaning.

    Uses progressive prompts (pass 1: general, pass 2: pattern fixer,
    pass 3: naturalizer) and early termination if ai_score drops below
    threshold.
    """
    cfg = MODELS[preset]
    current = text
    total_start = time.time()

    for i in range(passes):
        temp = cfg["temperature"] + (i * 0.05)
        system_prompt = PASS_PROMPTS[i % 3]
        current = call_ollama(
            prompt=current,
            model=cfg["model"],
            system_prompt=system_prompt,
            temperature=temp,
            max_tokens=cfg["max_tokens"],
        )
        # Early termination: if ai_score is low enough, stop wasting passes
        if i >= 1:
            score = analyze_text(current).ai_score
            if score < EARLY_TERMINATION_THRESHOLD:
                break

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

    Uses progressive prompts and early termination. Temperature is varied
    slightly per pass for diversity.
    """
    current = text
    total_start = time.time()
    for i in range(passes):
        temp = cfg["temperature"] + (i * 0.05)
        system_prompt = PASS_PROMPTS[i % 3]
        current = call_ollama(
            prompt=current,
            model=cfg["model"],
            system_prompt=system_prompt,
            temperature=temp,
            max_tokens=cfg["max_tokens"],
            timeout=timeout,
            max_retries=max_retries,
        )
        # Early termination
        if i >= 1:
            score = analyze_text(current).ai_score
            if score < EARLY_TERMINATION_THRESHOLD:
                break
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
    parser.add_argument(
        "--smart-clean",
        action="store_true",
        help="Pre/post clean obvious markers via regex (cheap, no LLM cost)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        handle_list_models()

    if not args.input:
        parser.error("input file is required (or use --list-models)")

    try:
        model = resolve_model(args.model, script_default="llama3.2")
    except ValueError as exc:
        print_error(str(exc))
        sys.exit(1)

    text = read_input_text(args.input, allow_stdin="-")

    # Smart Clean: pre-clean obvious markers
    if args.smart_clean:
        before = get_marker_count(text)
        text = clean_text(text, aggressive=False)
        after = get_marker_count(text)
        print_info(f"Smart clean: {before} → {after} markers")

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

    # Clamp passes to the documented 1-3 range.
    passes = max(1, min(3, args.passes))

    with ProgressBar(total=passes, label=f"wash ({preset})") as bar:
        if passes == 1:
            result = wash_pass(text, preset)
            bar.advance()
        else:
            # Re-implement multi-pass inline so the progress bar advances per pass.
            cfg = MODELS[preset]
            current = text
            total_start = time.time()
            actual_passes = 0
            for i in range(passes):
                bar.advance()
                actual_passes += 1
                system_prompt = PASS_PROMPTS[i % 3]
                current = call_ollama(
                    prompt=current,
                    model=cfg["model"],
                    system_prompt=system_prompt,
                    temperature=cfg["temperature"],
                    max_tokens=cfg["max_tokens"],
                )
                # Early termination
                if i >= 1:
                    score = analyze_text(current).ai_score
                    if score < EARLY_TERMINATION_THRESHOLD:
                        break
            result = WashResult(current, cfg["model"], time.time() - total_start, actual_passes)

    # Smart Clean: post-clean (catch what the model missed)
    if args.smart_clean:
        result.text = clean_text(result.text, aggressive=False)

    if args.output:
        write_output_text(args.output, result.text)
        print_success(
            f"Wrote {args.output} ({result.duration:.1f}s, "
            f"{result.passes} pass(es), model={result.model})"
        )
    else:
        print(result.text)


if __name__ == "__main__":
    main()
