#!/usr/bin/env python3
"""Multi-Agent Washer — parallel 3 models, AI-score driven selection.

Architecture:
  Agent 1: lfm25-tool   (fast, 2.7B)  — rough rewrite
  Agent 2: llama3.2     (balanced)    — medium rewrite
  Agent 3: eurollm-9b   (quality, 9B) — premium rewrite

All 3 run in parallel via ``concurrent.futures``.  The best result wins
(lowest post-wash AI score from :mod:`stat_engine`).

Usage:
    python scripts/multi_agent_washer.py input.txt -o clean.txt
    python scripts/multi_agent_washer.py input.txt --verbose
    python scripts/multi_agent_washer.py input.txt --dry-run
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# Ensure sibling modules (ollama_utils.py) are importable when run directly
# or when imported as part of the `scripts` package by pytest.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cli_utils import (  # noqa: E402
    ProgressBar,
    print_info,
    print_success,
    print_warning,
    read_input_text,
    write_output_text,
)
from ollama_utils import SYSTEM_PROMPT, call_ollama  # noqa: E402
from stat_engine import analyze_text  # noqa: E402

# The 3-agent model pool.  Each entry is a self-contained config dict so the
# candidate selection logic is fully testable without touching Ollama.
AGENT_MODELS = [
    {"name": "lfm25-tool", "temperature": 0.7, "max_tokens": 1024, "label": "fast"},
    {"name": "llama3.2", "temperature": 0.8, "max_tokens": 1536, "label": "standard"},
    {"name": "eurollm-9b", "temperature": 0.85, "max_tokens": 2048, "label": "premium"},
]


@dataclass
class WashCandidate:
    """One model's wash output together with scoring metadata."""

    text: str
    model: str
    label: str
    duration: float
    ai_score: float = 0.0
    error: str | None = None

    @property
    def is_error(self) -> bool:
        """Whether this candidate failed to produce output."""
        return self.error is not None


def _wash_with_model(text: str, model_cfg: dict) -> WashCandidate:
    """Single model wash — runs in a worker thread.

    Catches per-model failures so one bad agent doesn't sink the whole run;
    a failed agent is returned as an error candidate so the caller can report
    which model failed.
    """
    start = time.time()
    try:
        result = call_ollama(
            prompt=text,
            model=model_cfg["name"],
            system_prompt=SYSTEM_PROMPT,
            temperature=model_cfg["temperature"],
            max_tokens=model_cfg["max_tokens"],
        )
    except Exception as exc:  # noqa: BLE001 — report, don't crash the pool
        duration = time.time() - start
        return WashCandidate(
            text="",
            model=model_cfg["name"],
            label=model_cfg["label"],
            duration=duration,
            ai_score=999.0,
            error=str(exc),
        )
    duration = time.time() - start
    # AI-score the result
    report = analyze_text(result)
    return WashCandidate(
        text=result,
        model=model_cfg["name"],
        label=model_cfg["label"],
        duration=duration,
        ai_score=report.ai_score,
    )


def rank_candidates(candidates: list[WashCandidate]) -> WashCandidate:
    """Pick the best candidate — lowest AI score among successful ones.

    Error candidates are excluded from selection but *only* when at least one
    successful candidate exists.  If every candidate failed, the first error
    candidate is returned so the caller can surface a meaningful message.
    """
    successful = [c for c in candidates if not c.is_error]
    if successful:
        return min(successful, key=lambda c: c.ai_score)
    # All failed (or empty input) — return a sentinel so the error is surfaced.
    if not candidates:
        return WashCandidate("", "", "", 0.0, 999.0, error="no candidates")
    # All failed — return the first so the error is surfaced.
    return candidates[0]


def format_candidate_summary(candidate: WashCandidate) -> str:
    """Render a single candidate as a one-line summary (color-ready)."""
    if candidate.is_error:
        return (
            f"[{candidate.label}] {candidate.model}: "
            f"FAILED t={candidate.duration:.1f}s - {candidate.error}"
        )
    return (
        f"[{candidate.label}] {candidate.model}: "
        f"AI={candidate.ai_score:.1f} t={candidate.duration:.1f}s "
        f"chars={len(candidate.text)}"
    )


def format_results_table(candidates: list[WashCandidate], winner: WashCandidate) -> str:
    """Render the full results table as a list of human-readable lines.

    Pure function — no I/O, no Ollama — suitable for unit testing.
    """
    lines: list[str] = []
    lines.append("Multi-Agent Wash Results")
    lines.append("=" * 48)
    for c in sorted(candidates, key=lambda x: x.ai_score):
        marker = "WINNER" if c.model == winner.model and not c.is_error else "      "
        lines.append(f"  {marker} {format_candidate_summary(c)}")
    lines.append("-" * 48)
    if winner.is_error:
        lines.append(f"Winner: {winner.error or 'no successful agents'}")
    else:
        lines.append(
            f"Winner: [{winner.label}] {winner.model} "
            f"(AI={winner.ai_score:.1f}, {winner.duration:.1f}s, "
            f"{len(winner.text)} chars)"
        )
    return "\n".join(lines)


def multi_agent_wash(text: str, verbose: bool = False) -> WashCandidate:
    """Run 3 models in parallel, return the best candidate.

    Parameters
    ----------
    text:
        The text to rewrite.
    verbose:
        When True, per-model stats are printed to stderr as they complete.
    """
    results: list[WashCandidate] = []
    print_info(f"Launching {len(AGENT_MODELS)} parallel agents...", file=sys.stderr)

    with ProgressBar(total=len(AGENT_MODELS), label="agents") as bar:
        with ThreadPoolExecutor(max_workers=len(AGENT_MODELS)) as executor:
            futures = {
                executor.submit(_wash_with_model, text, cfg): cfg
                for cfg in AGENT_MODELS
            }
            for future in as_completed(futures):
                try:
                    candidate = future.result()
                    results.append(candidate)
                    bar.advance()
                    if verbose:
                        print(f"  {format_candidate_summary(candidate)}", file=sys.stderr)
                except Exception as e:  # noqa: BLE001
                    cfg = futures[future]
                    failed = WashCandidate(
                        text="",
                        model=cfg["name"],
                        label=cfg["label"],
                        duration=0.0,
                        ai_score=999.0,
                        error=str(e),
                    )
                    results.append(failed)
                    bar.advance()
                    print_warning(f"Agent failed: {cfg['name']}: {e}", file=sys.stderr)

    if not results:
        raise RuntimeError("All agents failed")

    best = rank_candidates(results)

    if verbose:
        print_info("\n" + format_results_table(results, best), file=sys.stderr)
    else:
        print_info(
            f"  Winner: [{best.label}] {best.model} (AI={best.ai_score:.1f})",
            file=sys.stderr,
        )

    return best


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the multi-agent washer."""
    parser = argparse.ArgumentParser(
        description=(
            "Multi-Agent Washer - run 3 local Ollama models in parallel and "
            "pick the best (lowest AI-score) rewrite."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The three agents run concurrently via a thread pool. Each result "
            "is scored with the statistical watermark engine (stat_engine) "
            "and the rewrite with the lowest AI score wins.\n\n"
            "Available agents:\n"
            + "\n".join(
                f"  {c['label']:<10} {c['name']:<14} t={c['temperature']} max={c['max_tokens']}"
                for c in AGENT_MODELS
            )
        ),
    )
    parser.add_argument("input", help="Input text file (or '-' for stdin)")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show per-model stats and results table"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only score the input (no wash); prints the pre-wash AI score",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    text = read_input_text(args.input, allow_stdin="-")

    if args.dry_run:
        report = analyze_text(text)
        print(f"Pre-wash AI Score: {report.ai_score:.1f}/100")
        print(f"Burstiness: {report.burstiness:.3f}")
        print(f"Perplexity: {report.perplexity:.1f}")
        return 0

    print_info("Running 3-agent parallel wash...", file=sys.stderr)
    best = multi_agent_wash(text, verbose=args.verbose)

    if args.output:
        write_output_text(args.output, best.text)
        print_success(
            f"Wrote {args.output} (winner={best.model}, AI={best.ai_score:.1f})"
        )
    else:
        print(best.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
