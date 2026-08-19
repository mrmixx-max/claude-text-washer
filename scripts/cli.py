#!/usr/bin/env python3
"""Unified CLI entry point for claude-text-washer.

Provides a single ``claude-washer`` command whose subcommands dispatch to the
individual scripts:

    claude-washer scan     <file>            # scan for AI surface markers
    claude-washer wash     <file|-> -o out   # single-pass LLM rewrite
    claude-washer pipeline  <file|-> -o out  # multi-pass wash w/ presets
    claude-washer file     <files-or-globs>  # batch wash (docx/md/pdf/html/txt)
    claude-washer chat                       # interactive chat w/ local Ollama
    claude-washer edit     [file]            # interactive text editor
    claude-washer stat     <file|->          # statistical watermark analysis
    claude-washer prompt   <file|->          # anti-watermark prompt engineer

Dispatch is lazy: each subcommand module is imported on demand, so a broken
optional dependency in one tool never takes down the whole CLI.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Make sibling scripts importable both when run directly (``python
# scripts/cli.py``) and when imported as a package (``python claude-washer``).
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# command -> (module name, short description)
SUBCOMMANDS: dict[str, tuple[str, str]] = {
    "scan": ("marker_scan", "Scan text for AI surface markers"),
    "wash": ("washer", "Single-pass AI-marker rewrite via local Ollama"),
    "pipeline": ("pipeline", "Multi-pass wash with presets (fast/standard/premium)"),
    "file": ("file_washer", "Batch wash files (docx/md/pdf/html/txt), globs & dirs"),
    "chat": ("chat", "Interactive chat with local Ollama models"),
    "edit": ("editor", "Interactive text editor with live AI score"),
    "stat": ("stat_engine", "Statistical watermark analysis (no Ollama needed)"),
    "prompt": ("stat_prompt", "Generate an anti-watermark prompt from statistics"),
}


def _load_module(modname: str):
    """Import a sibling module, tolerant of package vs direct execution."""
    try:
        return importlib.import_module(f"scripts.{modname}")
    except ImportError:
        # Fallback for direct execution (``python scripts/cli.py``) where
        # ``scripts`` is not on sys.path as a package but its directory is.
        return importlib.import_module(modname)


def print_help() -> None:
    print("claude-text-washer — unified CLI")
    print("Usage: claude-washer <command> [args...]  (e.g. claude-washer wash input.txt -o out.txt)")
    print()
    print("Commands:")
    width = max(len(cmd) for cmd in SUBCOMMANDS)
    for cmd, (_, desc) in SUBCOMMANDS.items():
        print(f"  {cmd:<{width}}  {desc}")
    print()
    print("Each command accepts --help for its own options, e.g.:")
    print("  claude-washer file --help")
    print()
    print("Available model flags (--model, --list-models, --temperature) are")
    print("shared by: scan, wash, pipeline, file, stat, prompt.")


def main(argv: list[str] | None = None) -> int:
    """Entry point.  *argv* defaults to ``sys.argv[1:]``.

    The first token selects a subcommand; everything after it is forwarded
    verbatim to that subcommand's own ``main(argv)`` so each tool keeps its
    full, independent argparse interface.
    """
    args = sys.argv[1:] if argv is None else list(argv)

    # No args / help → show top-level usage.
    if not args or args[0] in ("-h", "--help"):
        print_help()
        return 0
    if args[0] in ("-V", "--version"):
        print("claude-text-washer unified CLI")
        return 0

    command = args[0]
    rest = args[1:]

    entry = SUBCOMMANDS.get(command)
    if entry is None:
        print(f"error: unknown command '{command}'", file=sys.stderr)
        print(file=sys.stderr)
        print_help()
        return 2

    modname, _desc = entry
    try:
        mod = _load_module(modname)
    except ImportError as exc:
        print(f"error: could not load '{command}' module ({modname}): {exc}", file=sys.stderr)
        return 1

    result = mod.main(rest)
    # Each module's main may return an exit code (int) or None.
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
