#!/usr/bin/env python3
"""Statistical watermark detection and prompt engineering for AI text.

Detects and counters statistical watermarks (green-list bias, n-gram patterns)
used by LLMs like Claude, GPT, and others.
"""
from __future__ import annotations

import json
import math
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WatermarkReport:
    """Statistical analysis of text for AI watermark patterns."""
    perplexity: float
    burstiness: float
    ngram_bias: float
    green_list_ratio: float
    sentence_entropy: float
    word_entropy: float
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


def detect_green_list_bias(tokens: list[str]) -> float:
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
        'used', 'der', 'die', 'das', 'ein', 'eine', 'und', 'oder', 'aber', 'in',
        'auf', 'an', 'zu', 'für', 'von', 'mit', 'bei', 'aus', 'als', 'ist', 'war',
        'sind', 'waren', 'sein', 'haben', 'hat', 'hatte', 'wird', 'würde', 'kann',
    }
    
    stop_count = sum(1 for t in tokens if t.lower() in stopwords)
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
    """Full statistical analysis of text for AI watermark patterns."""
    tokens = tokenize(text)
    sentences = sentence_split(text)
    
    if not tokens or not sentences:
        return WatermarkReport(
            perplexity=0.0, burstiness=0.0, ngram_bias=0.0,
            green_list_ratio=0.0, sentence_entropy=0.0, word_entropy=0.0,
            ai_score=0.0, details={"error": "empty text"}
        )
    
    perplexity = calculate_perplexity(tokens)
    burstiness = calculate_burstiness(sentences)
    word_entropy = calculate_entropy(tokens)
    sentence_entropy = calculate_sentence_entropy(sentences)
    green_list_ratio = detect_green_list_bias(tokens)
    
    # N-gram bias (measure of repetitive patterns)
    bigrams = analyze_ngrams(tokens, 2)
    trigrams = analyze_ngrams(tokens, 3)
    ngram_bias = max(bigrams.values()) if bigrams else 0.0
    
    # AI Score calculation (0-100)
    # Low burstiness + low entropy + high green-list ratio = likely AI
    ai_score = 0.0
    if burstiness < 0.3:
        ai_score += 25
    if word_entropy < 6.0:
        ai_score += 25
    if sentence_entropy < 0.8:
        ai_score += 20
    if green_list_ratio > 0.4:
        ai_score += 15
    if ngram_bias > 0.05:
        ai_score += 15
    
    return WatermarkReport(
        perplexity=perplexity,
        burstiness=burstiness,
        ngram_bias=ngram_bias,
        green_list_ratio=green_list_ratio,
        sentence_entropy=sentence_entropy,
        word_entropy=word_entropy,
        ai_score=min(ai_score, 100.0),
        details={
            "token_count": len(tokens),
            "sentence_count": len(sentences),
            "avg_sentence_length": sum(len(s.split()) for s in sentences) / len(sentences),
            "top_bigrams": dict(list(bigrams.items())[:5]) if bigrams else {},
            "top_trigrams": dict(list(trigrams.items())[:5]) if trigrams else {},
        }
    )


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
{chr(10).join(f'- {s}' for s in strategies)}

Additional rules:
- Break any rhythmic patterns in sentence structure
- Replace common word pairs with unexpected alternatives
- Introduce controlled "noise" through varied punctuation
- Maintain factual accuracy and original claims
- Output ONLY the rewritten text, no explanations

Original text:
{text}"""
    
    return prompt


def wash_statistical(text: str, model: str = "llama3.2", temperature: float = 0.85) -> tuple[str, WatermarkReport]:
    """Wash text using statistical analysis + engineered prompt."""
    report = analyze_text(text)
    prompt = generate_anti_watermark_prompt(text, report)
    
    # Call Ollama with engineered prompt
    payload = {
        "model": model,
        "system": "Du bist ein Experten-Textredaktor. Schreibe Texte um, die wie menschlich geschrieben wirken. Gib NUR den umgeschriebenen Text zurück.",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 2048,
        },
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            cleaned = result.get("response", "").strip()
            return cleaned, report
    except Exception as e:
        raise RuntimeError(f"Ollama error ({model}): {e}")


def main():
    """CLI for statistical watermark analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Statistical AI watermark detection")
    parser.add_argument("input", help="Input text file or - for stdin")
    parser.add_argument("-o", "--output", help="Output file for washed text")
    parser.add_argument("--model", default="llama3.2", help="Ollama model")
    parser.add_argument("--temperature", type=float, default=0.85, help="Sampling temperature")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze, don't wash")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    if args.input == "-":
        text = __import__("sys").stdin.read()
    else:
        text = Path(args.input).read_text(encoding="utf-8")
    
    report = analyze_text(text)
    
    if args.analyze_only:
        output = {
            "perplexity": round(report.perplexity, 2),
            "burstiness": round(report.burstiness, 3),
            "ngram_bias": round(report.ngram_bias, 4),
            "green_list_ratio": round(report.green_list_ratio, 3),
            "sentence_entropy": round(report.sentence_entropy, 3),
            "word_entropy": round(report.word_entropy, 3),
            "ai_score": round(report.ai_score, 1),
            "details": report.details,
        }
        if args.json:
            print(json.dumps(output, indent=2))
        else:
            print(f"AI Score: {output['ai_score']}/100")
            print(f"Perplexity: {output['perplexity']}")
            print(f"Burstiness: {output['burstiness']}")
            print(f"Word Entropy: {output['word_entropy']}")
            print(f"Sentence Entropy: {output['sentence_entropy']}")
            print(f"Green-list Ratio: {output['green_list_ratio']}")
            print(f"N-gram Bias: {output['ngram_bias']}")
    else:
        cleaned, report = wash_statistical(text, args.model, args.temperature)
        if args.output:
            Path(args.output).write_text(cleaned, encoding="utf-8")
            print(f"Wrote {args.output}", file=__import__("sys").stderr)
        else:
            print(cleaned)


if __name__ == "__main__":
    main()
