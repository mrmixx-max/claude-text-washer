#!/usr/bin/env python3
"""Statistical watermark detection and prompt engineering for AI text.

Detects and counters statistical watermarks (green-list bias, n-gram patterns)
used by LLMs like Claude, GPT, and others.

Supports any Ollama model from the pool defined in scripts/models.yaml.
  --model MODEL       Select model (default: llama3.2)
  --list-models       Show all available models and exit
  --analyze-only      Only run statistical analysis, skip LLM washing
  --json              Output analysis as JSON
"""
from __future__ import annotations

import json
import math
import re
import sys
import urllib.error
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Ensure sibling modules (ollama_utils.py) are importable when run directly
# or when imported as part of the `scripts` package by pytest.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ollama_utils import (
    call_ollama,
    get_default_model,
    handle_list_models,
    resolve_model,
)
from cli_utils import (  # noqa: E402
    ProgressBar,
    print_info,
    print_success,
    read_input_text,
    write_output_text,
)

STAT_SYSTEM_PROMPT = (
    "You are an expert text editor. Rewrite text to sound human-written. "
    "Return ONLY the rewritten text."
)


@dataclass
class WatermarkReport:
    """Statistical analysis of text for AI watermark patterns."""
    perplexity: float
    burstiness: float
    ngram_bias: float
    green_list_ratio: float
    sentence_entropy: float
    word_entropy: float
    type_token_ratio: float
    zipf_coefficient: float
    hapax_ratio: float
    ai_score: float  # 0-100, higher = more likely AI
    details: dict


def tokenize(text: str) -> list[str]:
    """Simple word tokenizer."""
    return re.findall(r'\b\w+\b', text.lower())


def sentence_split(text: str) -> list[str]:
    """Split text into sentences."""
    return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]


def calculate_entropy(tokens: list[str]) -> float:
    """Calculate Shannon entropy of token distribution."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def calculate_type_token_ratio(tokens: list[str], counts: Counter | None = None) -> float:
    """Calculate Type-Token Ratio (TTR) — vocabulary richness.

    Higher TTR = more diverse vocabulary (more human-like).
    Lower TTR = repetitive vocabulary (more AI-like).
    """
    if not tokens:
        return 0.0
    if counts is not None:
        return len(counts) / len(tokens)
    unique = len(set(tokens))
    return unique / len(tokens)


def calculate_zipf_coefficient(tokens: list[str], counts: Counter | None = None) -> float:
    """Calculate Zipf coefficient — deviation from ideal Zipf distribution.

    Human text follows Zipf's law. AI text often deviates.
    Returns slope of log(rank) vs log(freq) regression.
    """
    if len(tokens) < 10:
        return 0.0
    if counts is None:
        counts = Counter(tokens)
    sorted_freq = sorted(counts.values(), reverse=True)
    # Take top 50 tokens to avoid noise from rare words
    top_freq = sorted_freq[:50]
    if len(top_freq) < 5:
        return 0.0
    log_ranks = [math.log(i + 1) for i in range(len(top_freq))]
    log_freqs = [math.log(f) for f in top_freq]
    n = len(log_ranks)
    sum_x = sum(log_ranks)
    sum_y = sum(log_freqs)
    sum_xy = sum(x * y for x, y in zip(log_ranks, log_freqs))
    sum_x2 = sum(x * x for x in log_ranks)
    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        return 0.0
    slope = (n * sum_xy - sum_x * sum_y) / denom
    return -slope  # Return positive value (ideal Zipf ≈ 1.0)


def calculate_hapax_ratio(tokens: list[str], counts: Counter | None = None) -> float:
    """Calculate hapax legomena ratio — words appearing only once.

    Higher ratio = more unique words = more human-like.
    AI text tends to reuse vocabulary more.
    """
    if not tokens:
        return 0.0
    if counts is None:
        counts = Counter(tokens)
    hapax = sum(1 for c in counts.values() if c == 1)
    return hapax / len(counts) if counts else 0.0


def calculate_perplexity(tokens: list[str]) -> float:
    """Estimate perplexity from token entropy."""
    entropy = calculate_entropy(tokens)
    return 2 ** entropy


def calculate_burstiness(sentences: list[str]) -> float:
    """Calculate burstiness (variance in sentence length)."""
    if len(sentences) < 2:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    mean_len = sum(lengths) / len(lengths)
    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
    return math.sqrt(variance) / mean_len if mean_len > 0 else 0.0


def analyze_ngrams(tokens: list[str], n: int = 2) -> dict:
    """Analyze n-gram distribution for bias patterns."""
    if len(tokens) < n:
        return {}
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    total = len(ngrams)
    return {str(ng): count / total for ng, count in counts.most_common(20)}


def detect_green_list_bias(tokens: list[str], counts: Counter | None = None) -> float:
    """Detect potential green-list watermark bias.

    Green-list watermarks bias token selection toward a subset of vocabulary.
    This measures the ratio of "expected" vs "unexpected" tokens.
    """
    if not tokens:
        return 0.0

    # Common English/German stopwords (expected in natural text)
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
        'used', 'der', 'die', 'das', 'ein', 'eine', 'und', 'oder', 'aber', 'auf', 'zu', 'für', 'von', 'mit', 'bei', 'aus', 'als', 'ist', 'war',
        'sind', 'waren', 'sein', 'haben', 'hat', 'hatte', 'wird', 'würde', 'kann',
    }

    if counts is not None:
        stop_count = sum(counts[t] for t in counts if t in stopwords)
    else:
        stop_count = sum(1 for t in tokens if t in stopwords)
    return stop_count / len(tokens) if tokens else 0.0


def calculate_sentence_entropy(sentences: list[str]) -> float:
    """Calculate entropy across sentence structures."""
    if len(sentences) < 2:
        return 0.0

    # Classify sentences by length category
    categories = []
    for s in sentences:
        wc = len(s.split())
        if wc < 8:
            categories.append('short')
        elif wc < 20:
            categories.append('medium')
        else:
            categories.append('long')

    counts = Counter(categories)
    total = len(categories)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def analyze_text(text: str) -> WatermarkReport:
    """Full statistical analysis of text for AI watermark patterns.

    Performance: tokenizes once, computes Counter once, reuses for all metrics.
    Short texts (< 20 tokens) skip expensive ngram/zipf calculations.
    """
    tokens = tokenize(text)
    sentences = sentence_split(text)
    token_count = len(tokens)

    if not tokens or not sentences:
        return WatermarkReport(
            perplexity=0.0, burstiness=0.0, ngram_bias=0.0,
            green_list_ratio=0.0, sentence_entropy=0.0, word_entropy=0.0,
            type_token_ratio=0.0, zipf_coefficient=0.0, hapax_ratio=0.0,
            ai_score=0.0, details={"error": "empty text"}
        )

    # Pre-compute Counter ONCE — reused by entropy, TTR, zipf, hapax, green-list
    counts = Counter(tokens)

    # Fast-path for short texts: skip expensive zipf/ngram calculations
    if token_count < 20:
        word_entropy = calculate_entropy(tokens)
        sentence_entropy = calculate_sentence_entropy(sentences)
        burstiness = calculate_burstiness(sentences)
        return WatermarkReport(
            perplexity=2 ** word_entropy,
            burstiness=burstiness, ngram_bias=0.0,
            green_list_ratio=detect_green_list_bias(tokens, counts),
            sentence_entropy=sentence_entropy,
            word_entropy=word_entropy,
            type_token_ratio=calculate_type_token_ratio(tokens, counts),
            zipf_coefficient=0.0, hapax_ratio=calculate_hapax_ratio(tokens, counts),
            ai_score=_calc_ai_score(
                burstiness=burstiness,
                word_entropy=word_entropy,
                sentence_entropy=sentence_entropy,
                green_list_ratio=detect_green_list_bias(tokens, counts),
                ngram_bias=0.0,
                type_token_ratio=calculate_type_token_ratio(tokens, counts),
                hapax_ratio=calculate_hapax_ratio(tokens, counts),
                zipf_coefficient=0.0,
            ),
            details={
                "token_count": token_count,
                "sentence_count": len(sentences),
                "note": "short text — zipf/ngram skipped",
            }
        )

    # Full analysis for longer texts
    perplexity = calculate_perplexity(tokens)
    burstiness = calculate_burstiness(sentences)
    word_entropy = calculate_entropy(tokens)
    sentence_entropy = calculate_sentence_entropy(sentences)
    green_list_ratio = detect_green_list_bias(tokens, counts)
    type_token_ratio = calculate_type_token_ratio(tokens, counts)
    zipf_coefficient = calculate_zipf_coefficient(tokens, counts)
    hapax_ratio = calculate_hapax_ratio(tokens, counts)

    # N-gram bias (measure of repetitive patterns)
    bigrams = analyze_ngrams(tokens, 2)
    trigrams = analyze_ngrams(tokens, 3)
    ngram_bias = max(bigrams.values()) if bigrams else 0.0

    ai_score = _calc_ai_score(
        burstiness=burstiness,
        word_entropy=word_entropy,
        sentence_entropy=sentence_entropy,
        green_list_ratio=green_list_ratio,
        ngram_bias=ngram_bias,
        type_token_ratio=type_token_ratio,
        hapax_ratio=hapax_ratio,
        zipf_coefficient=zipf_coefficient,
    )

    return WatermarkReport(
        perplexity=perplexity,
        burstiness=burstiness,
        ngram_bias=ngram_bias,
        green_list_ratio=green_list_ratio,
        sentence_entropy=sentence_entropy,
        word_entropy=word_entropy,
        type_token_ratio=type_token_ratio,
        zipf_coefficient=zipf_coefficient,
        hapax_ratio=hapax_ratio,
        ai_score=min(ai_score, 100.0),
        details={
            "token_count": token_count,
            "sentence_count": len(sentences),
            "avg_sentence_length": sum(len(s.split()) for s in sentences) / len(sentences),
            "top_bigrams": dict(list(bigrams.items())[:5]) if bigrams else {},
            "top_trigrams": dict(list(trigrams.items())[:5]) if trigrams else {},
        }
    )


def _calc_ai_score(
    *,
    burstiness: float,
    word_entropy: float,
    sentence_entropy: float,
    green_list_ratio: float,
    ngram_bias: float,
    type_token_ratio: float,
    hapax_ratio: float,
    zipf_coefficient: float,
) -> float:
    """Calculate AI score (0-100) from pre-computed metrics."""
    ai_score = 0.0
    if burstiness < 0.3:
        ai_score += 20
    if word_entropy < 6.0:
        ai_score += 20
    if sentence_entropy < 0.8:
        ai_score += 15
    if green_list_ratio > 0.4:
        ai_score += 10
    if ngram_bias > 0.05:
        ai_score += 10
    if type_token_ratio < 0.4:
        ai_score += 10
    if hapax_ratio < 0.3:
        ai_score += 10
    if zipf_coefficient < 0.5:
        ai_score += 5
    return ai_score


def generate_anti_watermark_prompt(text: str, report: WatermarkReport) -> str:
    """Generate a prompt engineered to break statistical watermarks."""

    strategies = []

    if report.burstiness < 0.3:
        strategies.append("Vary sentence length dramatically: mix 3-word punches with 25-word flowing sentences.")

    if report.word_entropy < 6.0:
        strategies.append("Use rare, specific vocabulary. Avoid predictable word pairs.")

    if report.sentence_entropy < 0.8:
        strategies.append("Break structural patterns: alternate between statements, questions, and fragments.")

    if report.green_list_ratio > 0.4:
        strategies.append("Reduce common function words. Use more nouns and verbs.")

    if report.ngram_bias > 0.05:
        strategies.append("Avoid repetitive word sequences. Introduce unexpected transitions.")

    if not strategies:
        strategies.append("Text appears natural. Maintain current style with minor variations.")

    prompt = f"""Rewrite the following text to eliminate statistical AI markers while preserving meaning.

Anti-watermark strategies to apply:
{chr(10).join(f"- {s}" for s in strategies)}

Additional rules:
- Break any rhythmic patterns in sentence structure
- Replace common word pairs with unexpected alternatives
- Introduce controlled "noise" through varied punctuation
- Maintain factual accuracy and original claims
- Output ONLY the rewritten text, no explanations

Original text:
{text}"""

    return prompt


def format_report(report: WatermarkReport, *, as_json: bool = False) -> str:
    """Format a :class:`WatermarkReport` for human (or JSON) consumption.

    Pure function — no I/O — suitable for unit testing.  When *as_json* is
    ``True`` a deterministic JSON string is returned instead of the aligned
    human-readable block.
    """
    if as_json:
        import json

        output = {
            "perplexity": round(report.perplexity, 2),
            "burstiness": round(report.burstiness, 3),
            "ngram_bias": round(report.ngram_bias, 4),
            "green_list_ratio": round(report.green_list_ratio, 3),
            "sentence_entropy": round(report.sentence_entropy, 3),
            "word_entropy": round(report.word_entropy, 3),
            "type_token_ratio": round(report.type_token_ratio, 3),
            "zipf_coefficient": round(report.zipf_coefficient, 3),
            "hapax_ratio": round(report.hapax_ratio, 3),
            "ai_score": round(report.ai_score, 1),
            "details": report.details,
        }
        return json.dumps(output, indent=2)

    lines = [
        f"AI Score:      {report.ai_score:.1f}/100",
        f"Perplexity:    {report.perplexity:.2f}",
        f"Burstiness:    {report.burstiness:.3f}",
        f"Word Entropy:  {report.word_entropy:.3f}",
        f"Sent. Entropy: {report.sentence_entropy:.3f}",
        f"Green-list:    {report.green_list_ratio:.3f}",
        f"N-gram bias:   {report.ngram_bias:.4f}",
        f"TTR:           {report.type_token_ratio:.3f}",
        f"Zipf coeff:    {report.zipf_coefficient:.3f}",
        f"Hapax ratio:   {report.hapax_ratio:.3f}",
    ]
    return "\n".join(lines)


def wash_statistical(text: str, model: str = "llama3.2", temperature: float = 0.85) -> tuple[str, WatermarkReport]:
    """Wash text using statistical analysis + engineered prompt.

    Any model from the models.yaml pool is accepted.
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
    """CLI for statistical watermark analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Statistical AI watermark detection")
    parser.add_argument("input", nargs="?", help="Input text file or - for stdin")
    parser.add_argument("-o", "--output", help="Output file for washed text")
    parser.add_argument(
        "--model",
        default=None,
        help=f"Ollama model to use (default: {get_default_model()})",
    )
    parser.add_argument("--temperature", type=float, default=0.85, help="Sampling temperature")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze, don't wash")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
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

    if args.analyze_only:
        report = analyze_text(text)
        if args.json:
            print(format_report(report, as_json=True))
        else:
            print_info("Statistical watermark analysis:")
            print(format_report(report))
        return

    report = analyze_text(text)
    if not args.json:
        print_info(f"Pre-wash AI score: {report.ai_score:.1f}/100")
    with ProgressBar(total=1, label="statistical wash") as bar:
        cleaned, report = wash_statistical(text, model, args.temperature)
        bar.advance()
    post = analyze_text(cleaned)
    print_info(
        f"Post-wash AI score: {post.ai_score:.1f}/100 "
        f"(was {report.ai_score:.1f})"
    )
    if args.output:
        write_output_text(args.output, cleaned)
        print_success(f"Wrote {args.output} (model={model})")
    else:
        print(cleaned)


if __name__ == "__main__":
    main()
