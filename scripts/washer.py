#!/usr/bin/env python3
"""Claude Text Washer — CLI: scan for AI markers, rewrite via local Ollama.

Supports any Ollama model from the pool defined in scripts/models.yaml.
  --model MODEL       Select model (default: llama3.2)
  --list-models       Show all available models and exit
  --temperature N     Sampling temperature
"""
from __future__ import annotations

import argparse
import sys
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

# Re-export system prompt for scripts that import it
SYSTEM_PROMPT = SYSTEM_PROMPT


def wash(text: str, model: str = "llama3.2", temperature: float = 0.8) -> str:
    """Rewrite text via local Ollama model using HTTP API.

    Any model from the models.yaml pool is accepted.
    """
    return call_ollama(
        prompt=text,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=1024,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Claude Text Washer — strip AI markers")
    parser.add_argument("input", nargs="?", help="Input text file or - for stdin")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument(
        "--model",
        default=None,
        help=f"Ollama model to use (default: {get_default_model()})",
    )
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
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

    try:
        model = resolve_model(args.model, script_default="llama3.2")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.input == "-":
        text = sys.stdin.read()
    else:
        text = Path(args.input).read_text(encoding="utf-8")

    cleaned = wash(text, model=model, temperature=args.temperature)

    if args.output:
        Path(args.output).write_text(cleaned, encoding="utf-8")
        print(f"Wrote {args.output} (model={model})", file=sys.stderr)
    else:
        print(cleaned)


if __name__ == "__main__":
    main()
