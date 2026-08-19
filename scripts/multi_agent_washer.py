#!/usr/bin/env python3
"""Multi-Agent Washer — parallel 3 models, AI-score driven selection.

Architecture:
  Agent 1: lfm25-tool  (fast, 2.7B) — rough rewrite
  Agent 2: llama3.2    (balanced)   — medium rewrite  
  Agent 3: eurollm-9b  (quality, 9B) — premium rewrite

All 3 run in parallel via concurrent.futures. The best result wins
(lowest post-wash AI score from stat_engine).

Usage:
    python scripts/multi_agent_washer.py input.txt -o clean.txt
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ollama_utils import SYSTEM_PROMPT, call_ollama
from stat_engine import analyze_text

# The 3-agent model pool
AGENT_MODELS = [
    {"name": "lfm25-tool", "temperature": 0.7, "max_tokens": 1024, "label": "fast"},
    {"name": "llama3.2", "temperature": 0.8, "max_tokens": 1536, "label": "standard"},
    {"name": "eurollm-9b", "temperature": 0.85, "max_tokens": 2048, "label": "premium"},
]


@dataclass
class WashCandidate:
    text: str
    model: str
    label: str
    duration: float
    ai_score: float = 0.0


def _wash_with_model(text: str, model_cfg: dict) -> WashCandidate:
    """Single model wash — runs in thread."""
    start = time.time()
    result = call_ollama(
        prompt=text,
        model=model_cfg["name"],
        system_prompt=SYSTEM_PROMPT,
        temperature=model_cfg["temperature"],
        max_tokens=model_cfg["max_tokens"],
    )
    duration = time.time() - start
    # AI score the result
    report = analyze_text(result)
    return WashCandidate(
        text=result,
        model=model_cfg["name"],
        label=model_cfg["label"],
        duration=duration,
        ai_score=report.ai_score,
    )


def multi_agent_wash(text: str, verbose: bool = False) -> WashCandidate:
    """Run 3 models in parallel, return the best candidate."""
    results: list[WashCandidate] = []
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_wash_with_model, text, cfg): cfg 
            for cfg in AGENT_MODELS
        }
        for future in as_completed(futures):
            try:
                candidate = future.result()
                results.append(candidate)
                if verbose:
                    print(
                        f"  [{candidate.label}] {candidate.model}: "
                        f"AI={candidate.ai_score:.1f} t={candidate.duration:.1f}s",
                        file=sys.stderr,
                    )
            except Exception as e:
                cfg = futures[future]
                print(f"  [FAIL] {cfg['name']}: {e}", file=sys.stderr)
    
    if not results:
        raise RuntimeError("All agents failed")
    
    # Pick the lowest AI score (cleanest)
    best = min(results, key=lambda c: c.ai_score)
    
    if verbose:
        print(
            f"\n  Winner: [{best.label}] {best.model} "
            f"(AI={best.ai_score:.1f})",
            file=sys.stderr,
        )
    
    return best


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Washer — 3 models in parallel")
    parser.add_argument("input", help="Input text file")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show per-model stats")
    parser.add_argument("--dry-run", action="store_true", help="Score only, no wash")
    args = parser.parse_args()
    
    text = Path(args.input).read_text(encoding="utf-8")
    
    if args.dry_run:
        report = analyze_text(text)
        print(f"Pre-wash AI Score: {report.ai_score:.1f}/100")
        print(f"Burstiness: {report.burstiness:.3f}")
        print(f"Perplexity: {report.perplexity:.1f}")
        return 0
    
    print("Running 3-agent wash...", file=sys.stderr)
    best = multi_agent_wash(text, verbose=args.verbose)
    
    if args.output:
        Path(args.output).write_text(best.text, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(best.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
