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
import math
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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


# --------------------------------------------------------------------------- #
# Benchmarking
# --------------------------------------------------------------------------- #
def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (0–100) of a sequence.

    Accepts unsorted input; sorts internally for safety.  Use
    :func:`statistics.quantiles` semantics approximated here with linear
    interpolation, matching the p50/p95 reporting in ``--benchmark``.
    """
    if not sorted_values:
        return 0.0
    s = sorted(sorted_values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    weight = k - lo
    return s[lo] + (s[hi] - s[lo]) * weight


@dataclass
class BenchmarkResult:
    """Aggregated stats from running the washer N times."""

    iterations: int
    durations: list[float] = field(default_factory=list)
    ai_scores: list[float] = field(default_factory=list)
    winners: list[str] = field(default_factory=list)
    errors: int = 0

    # -- lazy-derived stats (percentiles sort internally) -------------------
    @property
    def p50_latency(self) -> float:
        return _percentile(self.durations, 50)

    @property
    def p95_latency(self) -> float:
        return _percentile(self.durations, 95)

    @property
    def mean_latency(self) -> float:
        return statistics.mean(self.durations) if self.durations else 0.0

    @property
    def min_latency(self) -> float:
        return min(self.durations) if self.durations else 0.0

    @property
    def max_latency(self) -> float:
        return max(self.durations) if self.durations else 0.0

    @property
    def mean_ai_score(self) -> float:
        return statistics.mean(self.ai_scores) if self.ai_scores else 0.0

    @property
    def min_ai_score(self) -> float:
        return min(self.ai_scores) if self.ai_scores else 0.0

    @property
    def max_ai_score(self) -> float:
        return max(self.ai_scores) if self.ai_scores else 0.0

    def format_report(self) -> str:
        """Render a human-readable benchmark report."""
        lines: list[str] = []
        lines.append(f"Benchmark Results ({self.iterations} iterations)")
        lines.append("=" * 56)
        lines.append(
            f"  Latency (s):  mean={self.mean_latency:.2f}  "
            f"p50={self.p50_latency:.2f}  p95={self.p95_latency:.2f}  "
            f"min={self.min_latency:.2f}  max={self.max_latency:.2f}"
        )
        lines.append(
            f"  AI score:     mean={self.mean_ai_score:.1f}  "
            f"min={self.min_ai_score:.1f}  max={self.max_ai_score:.1f}"
        )
        lines.append(f"  Winners:      {dict((m, c) for m, c in statistics.Counter(self.winners).items())}")
        lines.append(f"  Failed runs:  {self.errors}")
        return "\n".join(lines)


def run_benchmark(
    text: str,
    iterations: int = 10,
    verbose: bool = False,
    wash_fn=multi_agent_wash,
) -> BenchmarkResult:
    """Run the multi-agent wash *iterations* times and aggregate stats.

    ``wash_fn`` is injectable (defaults to :func:`multi_agent_wash`) so tests can
    pass a fake/slow implementation without touching Ollama.
    """
    result = BenchmarkResult(iterations=iterations)
    for i in range(iterations):
        candidate = wash_fn(text, verbose=False)
        result.durations.append(candidate.duration)
        result.winners.append(candidate.model if not candidate.is_error else "ERROR")
        if candidate.is_error:
            result.errors += 1
            result.ai_scores.append(999.0)
        else:
            result.ai_scores.append(candidate.ai_score)
        if verbose:
            print_info(
                f"  [{i + 1}/{iterations}] winner={candidate.model} "
                f"AI={candidate.ai_score:.1f} t={candidate.duration:.2f}s",
                file=sys.stderr,
            )
    return result


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
    parser.add_argument(
        "--benchmark",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Run the multi-agent wash N times and report latency/p50/p95 "
            "and AI-score stats. No output file is written unless --output "
            "is also given, in which case the last run's winner text is written."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.input and not args.dry_run and args.benchmark is None:
        parser.error("input file is required (or use --help)")

    text = read_input_text(args.input, allow_stdin="-") if args.input else ""

    if args.dry_run:
        report = analyze_text(text)
        print(f"Pre-wash AI Score: {report.ai_score:.1f}/100")
        print(f"Burstiness: {report.burstiness:.3f}")
        print(f"Perplexity: {report.perplexity:.1f}")
        return 0

    if args.benchmark is not None:
        if args.benchmark < 1:
            parser.error("--benchmark N requires N >= 1")
        print_info(
            f"Running {args.benchmark}-iteration multi-agent benchmark...",
            file=sys.stderr,
        )
        result = run_benchmark(
            text, iterations=args.benchmark, verbose=args.verbose
        )
        print(result.format_report())
        # If an output file was also requested, write the last run's winner text.
        if args.output and result.winners:
            last_text = _last_winner_text(text, args.benchmark)
            if last_text is not None:
                write_output_text(args.output, last_text)
                print_success(f"Wrote {args.output} (benchmark winner text)")
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


def _last_winner_text(text: str, iterations: int) -> str | None:
    """Re-run the wash once to obtain output text for ``--benchmark -o``."""
    try:
        best = multi_agent_wash(text, verbose=False)
        return best.text
    except Exception:  # noqa: BLE001 — best-effort, don't mask benchmark output
        return None


if __name__ == "__main__":
    raise SystemExit(main())
