#!/usr/bin/env python3
"""File washer — batch-wash entire text files through the wash pipeline.

Usage:
    python scripts/file_washer.py input.txt -o clean.txt [--preset standard]

Currently a thin wrapper around the multi-pass statistical washer
(:mod:`scripts.pipeline`).  Future: add file-glob expansion, directory
walk, and format auto-detection (docx, md, etc.).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import wash_multi_pass, wash_pass, MODELS  # noqa: E402
from stat_engine import analyze_text  # noqa: E402


def wash_file(
    path: str | Path,
    preset: str = "standard",
    passes: int = 1,
    model: str | None = None,
    temperature: float | None = None,
) -> str:
    """Wash a single file and return the cleaned text."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if model or temperature:
        MODELS["standard"] = {
            "model": model or "llama3.2",
            "temperature": temperature or 0.8,
            "max_tokens": 1024,
        }
    if passes == 1:
        result = wash_pass(text, preset)
    else:
        result = wash_multi_pass(text, preset, passes)
    return result.text


def main() -> int:
    p = argparse.ArgumentParser(description="Wash a text file through the AI marker-removal pipeline")
    p.add_argument("input", help="Input file path (UTF-8)")
    p.add_argument("-o", "--output", help="Output file (default: stdout)")
    p.add_argument("--preset", choices=list(MODELS.keys()), default="standard")
    p.add_argument("--passes", type=int, default=1, help="Number of wash passes (1-3)")
    p.add_argument("--model", help="Override Ollama model")
    p.add_argument("--temperature", type=float, help="Override temperature")
    p.add_argument("--dry-run", action="store_true", help="Show AI score without washing")
    args = p.parse_args()

    if args.dry_run:
        text = Path(args.input).read_text(encoding="utf-8")
        report = analyze_text(text)
        print(f"AI Score: {report.ai_score:.1f}/100")
        print(f"Burstiness: {report.burstiness:.3f}")
        print(f"Perplexity: {report.perplexity:.1f}")
        return 0

    cleaned = wash_file(
        args.input,
        preset=args.preset,
        passes=args.passes,
        model=args.model,
        temperature=args.temperature,
    )
    if args.output:
        Path(args.output).write_text(cleaned, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(cleaned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
