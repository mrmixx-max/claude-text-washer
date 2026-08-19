#!/usr/bin/env python3
"""Claude Text Washer — CLI: scan for AI markers, rewrite via local Ollama."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SYSTEM_PROMPT = """Du bist ein knallharter, menschlicher Lektor und Ghostwriter. Deine Aufgabe ist es, den übergebenen Text komplett neu zu verfassen und jegliche Muster von maschinell generierter Sprache restlos zu vernichten.

Halte dich an folgende absolute Restriktionen:
1. Burstiness maximieren: Wechsle radikal zwischen sehr kurzen, prägnanten Sätzen (1-4 Wörtern) und längeren, asymmetrischen Satzgefügen.
2. Perplexität erzwingen: Nutze unkonventionelle, treffende Verben. Vermeide vorhersehbare Adjektiv-Substantiv-Kombinationen.
3. Blacklist: Verwende NIEMALS Phrasen wie "Zusammenfassend lässt sich sagen", "Es ist wichtig zu beachten", "Ein weiteres Element" oder Wörter wie "facettenreich", "Geflecht", "Tapestry", "essenziell", "dynamisch".
4. Tonalität: Organisch, direkt und menschlich. Lass es leicht kantig klingen, als käme es aus der Feder eines erfahrenen Thriller-Autors. Keine weichgespülte Objektivität.
5. Output: Gib AUSSCHLIESSLICH den umgeschriebenen Text zurück. Keine Einleitungen, keine Erklärungen, keine Höflichkeitsfloskeln."""


def wash(text: str, model: str = "llama3.2", temperature: float = 0.8) -> str:
    """Rewrite text via local Ollama model using HTTP API."""
    import json
    import urllib.request

    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": text,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 1024,
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


def main():
    parser = argparse.ArgumentParser(description="Claude Text Washer — strip AI markers")
    parser.add_argument("input", help="Input text file or - for stdin")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument("--model", default="llama3.2", help="Ollama model")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    args = parser.parse_args()

    if args.input == "-":
        text = sys.stdin.read()
    else:
        text = Path(args.input).read_text(encoding="utf-8")

    cleaned = wash(text, model=args.model, temperature=args.temperature)

    if args.output:
        Path(args.output).write_text(cleaned, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(cleaned)


if __name__ == "__main__":
    main()
