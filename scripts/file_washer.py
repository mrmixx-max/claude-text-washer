#!/usr/bin/env python3
"""File washer — batch-wash entire text files through the wash pipeline.

Format auto-detection: reads DOCX, PDF, Markdown, HTML and plain-text files
by sniffing magic bytes (not just the extension) via
:mod:`scripts.formats.documents`.  Heavy libs (``python-docx``, ``pymupdf``,
``beautifulsoup4``) are imported lazily, so this module loads with no optional
deps installed.

Batch / glob: accepts multiple files, shell-style globs (``*.md``), and
directories (walked recursively with ``-r``).  Each input is washed and written
to ``<stem>.washed.txt`` (or ``--outdir`` / ``-o``).

Usage::

    python scripts/file_washer.py input.txt -o clean.txt [--preset standard]
    python scripts/file_washer.py *.md --outdir cleaned/
    python scripts/file_washer.py inputs/ --recursive --outdir cleaned/
    python scripts/file_washer.py input.docx --dry-run
"""
from __future__ import annotations

import argparse
import glob as _glob
import sys
from pathlib import Path

# Ensure sibling modules are importable both when run directly and via the
# unified ``claude-washer`` entry point.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cli_utils import (  # noqa: E402
    ProgressBar,
    print_info,
    print_success,
    print_error,
)
from formats.documents import (  # noqa: E402  -- magic-byte sniffing + extractors
    detect_format as _detect_format_bytes,
    extract_text as _extract_text,
)
from pipeline import MODELS, resolve_preset, wash_multi_pass_cfg, wash_pass_cfg  # noqa: E402
from stat_engine import analyze_text  # noqa: E402
from ollama_utils import resolve_model  # noqa: E402


# --------------------------------------------------------------------------- #
# Format detection & text extraction (format auto-detection)
# --------------------------------------------------------------------------- #

def detect_format(path: str | Path) -> str:
    """Detect the document format of *path* from its magic bytes.

    Sniffs the first 8 KiB so huge binaries are not loaded fully just to tell a
    PDF from a DOCX.  Falls back to the file extension when detection fails.

    Returns one of ``txt``, ``md``, ``docx``, ``pdf``, ``html``, ``htm``,
    ``epub``, ``odt``, ``xlsx``, ``pptx`` or ``unknown``.
    """
    p = Path(path)
    with p.open("rb") as fh:
        head = fh.read(8192)
    return _detect_format_bytes(head, p.name)


def extract_file_text(path: str | Path, fmt: str | None = None) -> str:
    """Extract readable text from *path*.

    Auto-detects the format (docx, md, pdf, html, txt, ...) unless *fmt* is
    given to force a specific handler.
    """
    p = Path(path)
    data = p.read_bytes()
    doc = _extract_text(data, p.name, force_format=fmt)
    return doc.text


def read_text(source: str | Path) -> str:
    """Read *source* as text, auto-detecting format for files.

    The sentinel ``"-"`` reads UTF-8 text from stdin (no format detection).
    """
    if str(source) == "-":
        return sys.stdin.read()
    return extract_file_text(source)


# --------------------------------------------------------------------------- #
# Batch / glob helpers
# --------------------------------------------------------------------------- #

_GLOB_CHARS = ("*", "?", "[")


def expand_inputs(inputs: list[str], recursive: bool = False) -> list[Path]:
    """Expand a list of file paths and/or glob patterns into concrete files.

    - Glob metacharacters (``*`` / ``?`` / ``[``) are expanded with
      :func:`glob.glob` (``recursive`` enables ``**`` semantics).
    - Bare directories are walked (``**/*`` when *recursive*, ``*`` otherwise).
    - Plain files are kept as-is.
    - Results are sorted and de-duplicated (by resolved path).
    """
    if not inputs:
        return []
    expanded: list[Path] = []
    seen: set[Path] = set()

    for raw in inputs:
        if any(ch in raw for ch in _GLOB_CHARS):
            for match in sorted(_glob.glob(raw, recursive=recursive)):
                mp = Path(match)
                if mp.is_file() and mp not in seen:
                    seen.add(mp)
                    expanded.append(mp)
            continue

        p = Path(raw)
        if p.is_dir():
            pattern = "**/*" if recursive else "*"
            for child in sorted(p.glob(pattern)):
                if child.is_file() and child not in seen:
                    seen.add(child)
                    expanded.append(child)
        elif p.is_file():
            if p not in seen:
                seen.add(p)
                expanded.append(p)
        else:
            # Bare path that didn't resolve but may still be a glob on this OS.
            for match in sorted(_glob.glob(raw, recursive=recursive)):
                mp = Path(match)
                if mp.is_file() and mp not in seen:
                    seen.add(mp)
                    expanded.append(mp)

    return expanded


def output_path_for(input_path: Path, output: str | None, outdir: str | None) -> Path:
    """Determine the output path for a washed version of *input_path*."""
    if outdir:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        return out / (input_path.stem + ".washed.txt")
    if output:
        return Path(output)
    return input_path.with_name(input_path.stem + ".washed.txt")


# --------------------------------------------------------------------------- #
# Washing
# --------------------------------------------------------------------------- #

def wash_file(
    path: str | Path,
    preset: str = "standard",
    passes: int = 1,
    model: str | None = None,
    temperature: float | None = None,
    fmt: str | None = None,
) -> str:
    """Wash a single file and return the cleaned text.

    *fmt* overrides format auto-detection (e.g. ``"pdf"`` or ``"docx"``).
    Uses a fresh, thread-safe config dict per call (no global state mutation).
    """
    text = extract_file_text(path, fmt=fmt)
    cfg = resolve_preset(preset, model=model, temperature=temperature)
    if passes <= 1:
        result = wash_pass_cfg(text, cfg)
    else:
        result = wash_multi_pass_cfg(text, cfg, passes)
    return result.text


# --------------------------------------------------------------------------- #
# Analysis (dry-run)
# --------------------------------------------------------------------------- #

def analyze_file(path: str | Path, fmt: str | None = None) -> dict:
    """Return a statistical + marker analysis report for *path* (no LLM call)."""
    from marker_scan import scan  # local import to avoid module-load cost

    text = extract_file_text(path, fmt=fmt)
    report = analyze_text(text)
    hits = scan(text)
    high = sum(1 for h in hits if h[0] >= 3)
    mid = sum(1 for h in hits if h[0] == 2)
    low = sum(1 for h in hits if h[0] == 1)
    return {
        "path": str(path),
        "format": detect_format(path) if fmt is None else (fmt or "unknown"),
        "chars": len(text),
        "ai_score": report.ai_score,
        "burstiness": report.burstiness,
        "perplexity": report.perplexity,
        "markers": {"high": high, "mid": mid, "low": low, "total": len(hits)},
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch-wash text/document files (docx, md, pdf, html, txt) with AI marker removal",
    )
    p.add_argument(
        "inputs", nargs="*", help="Input file(s), glob(s) (*.md), or directory"
    )
    p.add_argument("-o", "--output", help="Output file (single input only; use --outdir for batch)")
    p.add_argument("--outdir", help="Directory to write washed files (batch mode)")
    p.add_argument(
        "--format",
        choices=["auto", "txt", "md", "docx", "pdf", "html", "htm", "epub", "odt"],
        default="auto",
        help="Force input format (default: auto-detect from magic bytes)",
    )
    p.add_argument(
        "--preset", choices=list(MODELS.keys()), default="standard", help="Model preset"
    )
    p.add_argument("--passes", type=int, default=1, help="Number of wash passes (1-3)")
    p.add_argument("--model", help="Override Ollama model")
    p.add_argument("--temperature", type=float, help="Override sampling temperature")
    p.add_argument("-r", "--recursive", action="store_true", help="Recurse into directories")
    p.add_argument("--dry-run", action="store_true", help="Analyze only, no washing (no Ollama call)")
    p.add_argument("--list-models", action="store_true", help="List Ollama models and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        from ollama_utils import handle_list_models
        handle_list_models()

    if not args.inputs and not args.list_models:
        parser.print_help()
        return 0

    # Validate model early (before doing any work).
    model = None
    if args.model:
        try:
            model = resolve_model(args.model, script_default="llama3.2")
        except ValueError as exc:
            print_error(str(exc))
            return 1

    fmt = None if args.format == "auto" else args.format
    files = expand_inputs(args.inputs, recursive=args.recursive)
    if not files:
        print_error("no input files found")
        return 1

    # dry-run: analysis only (no LLM)
    if args.dry_run:
        if len(files) == 1:
            rep = analyze_file(files[0], fmt=fmt)
            print_info(f"File: {rep['path']}  (format={rep['format']}, chars={rep['chars']})")
            print(f"  AI Score:   {rep['ai_score']:.1f}/100")
            print(f"  Burstiness: {rep['burstiness']:.3f}")
            print(f"  Perplexity: {rep['perplexity']:.1f}")
            m = rep["markers"]
            print(f"  Markers:    high={m['high']} mid={m['mid']} low={m['low']} total={m['total']}")
        else:
            print(f"{'file':40s} {'fmt':8s} {'chars':>7s} {'AI':>6s} {'hi':>3s} {'mid':>3s}")
            for f in files:
                rep = analyze_file(f, fmt=fmt)
                m = rep["markers"]
                print(f"{str(f):40s} {rep['format']:8s} {rep['chars']:7d} "
                      f"{rep['ai_score']:6.1f} {m['high']:3d} {m['mid']:3d}")
        return 0

    # Single output file requires a single input.
    if args.output and len(files) > 1:
        print_error("-o/--output can only be used with a single input (use --outdir for batch)")
        return 1

    passes = max(1, min(3, args.passes))
    ok, fail = 0, 0
    with ProgressBar(total=len(files), label=f"file wash ({args.preset})") as bar:
        for f in files:
            try:
                cleaned = wash_file(
                    f,
                    preset=args.preset,
                    passes=passes,
                    model=model,
                    temperature=args.temperature,
                    fmt=fmt,
                )
                out = output_path_for(f, args.output, args.outdir)
                out.write_text(cleaned, encoding="utf-8")
                det = detect_format(f) if fmt is None else (fmt or "auto")
                print_info(f"Wrote {out}  (format={det}, passes={passes}, model={model or 'preset-default'})")
                ok += 1
            except FileNotFoundError as exc:
                print_error(f"{f}: not found")
                fail += 1
            except Exception as exc:  # noqa: BLE001 - report per-file, keep batch going
                print_error(f"{f}: {exc}")
                fail += 1
            finally:
                bar.advance()

    print_success(f"Done: {ok} washed, {fail} failed.")
    return 1 if (fail and not ok) else 0


if __name__ == "__main__":
    raise SystemExit(main())
