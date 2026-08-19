#!/usr/bin/env python3
"""Multi-pass text washing pipeline with swappable local LLMs."""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Model presets — speed vs quality tradeoffs
MODELS = {
    "fast": {"model": "lfm25-tool", "temperature": 0.7, "max_tokens": 512},
    "standard": {"model": "llama3.2", "temperature": 0.8, "max_tokens": 1024},
    "premium": {"model": "llama3.2", "temperature": 0.9, "max_tokens": 2048},
}

SYSTEM_PROMPT = """Du bist ein knallharter, menschlicher Lektor und Ghostwriter. Deine Aufgabe ist es, den übergebenen Text komplett neu zu verfassen und jegliche Muster von maschinell generierter Sprache restlos zu vernichten.

Halte dich an folgende absolute Restriktionen:
1. Burstiness maximieren: Wechsle radikal zwischen sehr kurzen, prägnanten Sätzen (1-4 Wörtern) und längeren, asymmetrischen Satzgefügen.
2. Perplexität erzwingen: Nutze unkonventionelle, treffende Verben. Vermeide vorhersehbare Adjektiv-Substantiv-Kombinationen.
3. Blacklist: Verwende NIEMALS Phrasen wie "Zusammenfassend lässt sich sagen", "Es ist wichtig zu beachten", "Ein weiteres Element" oder Wörter wie "facettenreich", "Geflecht", "Tapestry", "essenziell", "dynamisch".
4. Tonalität: Organisch, direkt und menschlich. Lass es leicht kantig klingen, als käme es aus der Feder eines erfahrenen Thriller-Autors. Keine weichgespülte Objektivität.
5. Output: Gib AUSSCHLIESSLICH den umgeschriebenen Text zurück. Keine Einleitungen, keine Erklärungen, keine Höflichkeitsfloskeln."""


@dataclass
class WashResult:
    text: str
    model: str
    duration: float
    passes: int


def call_ollama(text: str, model: str, temperature: float, max_tokens: int) -> str:
    """Call local Ollama model via HTTP API (compatible with Ollama 0.32+)."""

    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": text,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "").strip()
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"Ollama error ({model}): {e}")


def wash_pass(text: str, preset: str = "standard") -> WashResult:
    """Single-pass wash with specified model preset."""
    cfg = MODELS[preset]
    start = time.time()
    cleaned = call_ollama(text, cfg["model"], cfg["temperature"], cfg["max_tokens"])
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
        current = call_ollama(current, cfg["model"], temp, cfg["max_tokens"])

    total_duration = time.time() - total_start
    return WashResult(current, cfg["model"], total_duration, passes)


def main():
    parser = argparse.ArgumentParser(description="Claude Text Washer — multi-pass AI marker removal")
    parser.add_argument("input", help="Input text file or - for stdin")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument("--preset", choices=["fast", "standard", "premium"], default="standard",
                        help="Model preset (fast=lfm25-tool, standard/premium=llama3.2)")
    parser.add_argument("--passes", type=int, default=1, help="Number of rewrite passes (1-3)")
    parser.add_argument("--model", help="Override Ollama model (ignores --preset)")
    parser.add_argument("--temperature", type=float, help="Override temperature")
    args = parser.parse_args()

    if args.input == "-":
        text = sys.stdin.read()
    else:
        text = Path(args.input).read_text(encoding="utf-8")

    # Override model if specified
    if args.model:
        MODELS["standard"] = {
            "model": args.model,
            "temperature": args.temperature or 0.8,
            "max_tokens": 1024,
        }
        args.preset = "standard"

    if args.passes == 1:
        result = wash_pass(text, args.preset)
    else:
        result = wash_multi_pass(text, args.preset, args.passes)

    if args.output:
        Path(args.output).write_text(result.text, encoding="utf-8")
        print(f"Wrote {args.output} ({result.duration:.1f}s, {result.passes} pass(es), model={result.model})",
              file=sys.stderr)
    else:
        print(result.text)


if __name__ == "__main__":
    main()
