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

from cli_utils import (  # noqa: E402
    ProgressBar,
    add_common_args,
    add_io_args,
    cli_entry,
    print_success,
    read_input_text,
    write_output_text,
)
from ollama_utils import (  # noqa: E402
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
    parser = argparse.ArgumentParser(
        description="Claude Text Washer - strip AI markers and rewrite organically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_io_args(parser)
    add_common_args(parser, include_temperature=True)
    # washer.py historically defaults temperature to 0.8 (not None) so that
    # wash() keeps its documented signature; expose it as an explicit default.
    parser.set_defaults(temperature=0.8)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        handle_list_models()

    if not args.input:
        parser.error("input file is required (or use --list-models)")

    with cli_entry():
        model = resolve_model(args.model, script_default="llama3.2")
        text = read_input_text(args.input, allow_stdin="-")

        with ProgressBar(total=1, label="washing") as bar:
            cleaned = wash(
                text,
                model=model,
                temperature=args.temperature if args.temperature is not None else 0.8,
            )
            bar.advance()

        if args.output:
            write_output_text(args.output, cleaned)
            print_success(f"Wrote {args.output} (model={model})")
        else:
            print(cleaned)
    return 0


if __name__ == "__main__":
    main()
