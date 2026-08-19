#!/usr/bin/env python3
"""Statistical anti-watermark prompt engineer.

Reads input text, runs statistical watermark analysis, and generates an
engineered prompt designed to break AI statistical markers.  The prompt can
be output to stdout (for piping to another tool) or sent to an Ollama model
for a live preview using any model from the pool in scripts/models.yaml.

Usage:
  python scripts/stat_prompt.py input.txt
  python scripts/stat_prompt.py input.txt --preview --model qwen-coder-7b
  python scripts/stat_prompt.py input.txt --output prompt.txt
  python scripts/stat_prompt.py --list-models
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure sibling modules are importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ollama_utils import (
    call_ollama,
    get_default_model,
    handle_list_models,
    resolve_model,
)
from stat_engine import analyze_text, generate_anti_watermark_prompt, format_report  # noqa: E402
from cli_utils import (  # noqa: E402
    ProgressBar,
    print_info,
    print_success,
    read_input_text,
    write_output_text,
)


STAT_SYSTEM_PROMPT = (
    "Du bist ein Experten-Textredaktor. Schreibe Texte um, die wie menschlich "
    "geschrieben wirken. Gib NUR den umgeschriebenen Text zurück."
)


def build_prompt(text: str, model: str = "llama3.2", temperature: float = 0.85) -> tuple[str, object]:
    """Generate an anti-watermark prompt from *text*.

    Returns a tuple of ``(prompt, report)``.
    """
    report = analyze_text(text)
    prompt = generate_anti_watermark_prompt(text, report)
    return prompt, report


def preview_wash(text: str, model: str = "llama3.2", temperature: float = 0.85) -> tuple[str, object]:
    """Generate the prompt AND send it to the LLM for a live preview.

    Returns ``(cleaned_text, report)``.
    """
    report = analyze_text(text)
    prompt = generate_anti_watermark_prompt(text, report)
    cleaned = call_ollama(
        prompt=prompt,
        model=model,
        system_prompt=STAT_SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=2048,
    )
    return cleaned, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Statistical anti-watermark prompt engineer"
    )
    parser.add_argument("input", nargs="?", help="Input text file or - for stdin")
    parser.add_argument(
        "-o", "--output",
        help="Output file for the generated prompt (default: stdout)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Ollama model for preview (default: {get_default_model()})",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.85,
        help="Sampling temperature for preview (default: 0.85)",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Send the prompt to the LLM and show the washed result",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available Ollama models and exit",
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
        from cli_utils import print_error
        print_error(str(exc))
        sys.exit(1)

    text = read_input_text(args.input, allow_stdin="-")

    if args.preview:
        with ProgressBar(total=1, label="preview wash") as bar:
            cleaned, report = preview_wash(text, model=model, temperature=args.temperature)
            bar.advance()
        print_info(f"AI Score: {report.ai_score:.1f}/100  Model: {model}", file=sys.stderr)
        if args.output:
            write_output_text(args.output, cleaned)
            print_success(f"wrote {args.output} (model={model})")
        else:
            print(cleaned)
    else:
        prompt, report = build_prompt(text)
        print_info(f"Statistical watermark report (model={model})", file=sys.stderr)
        print_info(format_report(report), file=sys.stderr)
        if args.output:
            write_output_text(args.output, prompt)
            print_success(f"wrote {args.output}")
        else:
            print(prompt)


if __name__ == "__main__":
    main()
