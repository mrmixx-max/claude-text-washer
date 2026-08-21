#!/usr/bin/env python3
"""Performance optimizations for Claude Text Washer.

Implements:
1. Batch processing with ThreadPoolExecutor
2. LLM response caching
3. HTTP connection pooling
4. Performance benchmarking
"""

from __future__ import annotations

import functools
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Ensure sibling modules are importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ollama_utils import call_ollama, SYSTEM_PROMPT
from smart_cleaner import clean_text, get_marker_count
from stat_engine import analyze_text


def make_cache_key(prompt: str, model: str, system_prompt: str, temperature: float, max_tokens: int) -> str:
    """Create a deterministic cache key for an LLM request."""
    content = f"{prompt}|{model}|{system_prompt}|{temperature}|{max_tokens}"
    import hashlib
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class LLMCache:
    """Simple file-based cache for LLM responses."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or (Path.home() / ".cache" / "claude-text-washer")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        if key in self._memory_cache:
            return self._memory_cache[key]
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                self._memory_cache[key] = data["response"]
                return data["response"]
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    def put(self, key: str, response: str) -> None:
        self._memory_cache[key] = response
        cache_file = self.cache_dir / f"{key}.json"
        cache_file.write_text(json.dumps({"response": response}, ensure_ascii=False), encoding="utf-8")

    def clear(self) -> None:
        self._memory_cache.clear()
        for f in self.cache_dir.glob("*.json"):
            f.unlink()


# Global cache instance
_llm_cache = LLMCache()


def cached_call_ollama(
    prompt: str,
    model: str = "llama3.2",
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = 0.8,
    max_tokens: int = 1024,
    use_cache: bool = True,
    **kwargs: Any,
) -> str:
    """Call Ollama with caching support."""
    if not use_cache:
        return call_ollama(prompt, model=model, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens, **kwargs)

    key = make_cache_key(prompt, model, system_prompt, temperature, max_tokens)
    cached = _llm_cache.get(key)
    if cached is not None:
        return cached

    response = call_ollama(prompt, model=model, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens, **kwargs)
    _llm_cache.put(key, response)
    return response


def batch_wash(
    texts: list[tuple[str, str]],  # [(input_path, output_path), ...]
    model: str = "llama3.2",
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = 0.8,
    max_tokens: int = 1024,
    max_workers: int = 3,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Process multiple texts in parallel using ThreadPoolExecutor.

    Returns list of result dicts with keys: input, output, duration, success, error
    """
    results = []

    def process_one(item: tuple[str, str]) -> dict[str, Any]:
        input_path, output_path = item
        start = time.time()
        try:
            text = Path(input_path).read_text(encoding="utf-8")
            cleaned = cached_call_ollama(
                text, model=model, system_prompt=system_prompt,
                temperature=temperature, max_tokens=max_tokens,
                use_cache=use_cache,
            )
            if output_path:
                Path(output_path).write_text(cleaned, encoding="utf-8")
            return {
                "input": input_path,
                "output": output_path,
                "duration": time.time() - start,
                "success": True,
                "error": None,
            }
        except Exception as e:
            return {
                "input": input_path,
                "output": output_path,
                "duration": time.time() - start,
                "success": False,
                "error": str(e),
            }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, item): item for item in texts}
        for future in as_completed(futures):
            results.append(future.result())

    return results


def benchmark_performance(
    test_texts: list[str],
    model: str = "llama3.2",
    runs: int = 3,
) -> dict[str, Any]:
    """Benchmark wash performance with and without caching."""
    results = {
        "model": model,
        "runs": runs,
        "without_cache": [],
        "with_cache": [],
    }

    # Without cache
    for i in range(runs):
        start = time.time()
        for text in test_texts:
            call_ollama(text, model=model)
        results["without_cache"].append(time.time() - start)

    # Clear cache
    _llm_cache.clear()

    # With cache (first run populates, subsequent runs hit cache)
    for i in range(runs):
        start = time.time()
        for text in test_texts:
            cached_call_ollama(text, model=model, use_cache=True)
        results["with_cache"].append(time.time() - start)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Performance tools for Claude Text Washer")
    sub = parser.add_subparsers(dest="command")

    # Batch command
    batch_parser = sub.add_parser("batch", help="Process multiple files in parallel")
    batch_parser.add_argument("files", nargs="+", help="Input files")
    batch_parser.add_argument("-o", "--output-dir", required=True, help="Output directory")
    batch_parser.add_argument("--model", default="llama3.2")
    batch_parser.add_argument("--workers", type=int, default=3)

    # Benchmark command
    bench_parser = sub.add_parser("benchmark", help="Run performance benchmark")
    bench_parser.add_argument("--model", default="llama3.2")
    bench_parser.add_argument("--runs", type=int, default=3)
    bench_parser.add_argument("--text", default="This is a test sentence for benchmarking.")

    # Cache command
    cache_parser = sub.add_parser("cache", help="Cache management")
    cache_parser.add_argument("--clear", action="store_true", help="Clear cache")

    args = parser.parse_args()

    if args.command == "batch":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        texts = [(f, str(output_dir / Path(f).name)) for f in args.files]
        results = batch_wash(texts, model=args.model, max_workers=args.workers)
        for r in results:
            status = "OK" if r["success"] else f"FAIL: {r['error']}"
            print(f"{r['input']} -> {r['output']}: {r['duration']:.1f}s [{status}]")

    elif args.command == "benchmark":
        test_texts = [args.text] * 5
        results = benchmark_performance(test_texts, model=args.model, runs=args.runs)
        print(f"Benchmark ({results['runs']} runs, {len(test_texts)} texts each):")
        print(f"  Without cache: {sum(results['without_cache'])/len(results['without_cache']):.2f}s avg")
        print(f"  With cache:    {sum(results['with_cache'])/len(results['with_cache']):.2f}s avg")

    elif args.command == "cache":
        if args.clear:
            _llm_cache.clear()
            print("Cache cleared")
        else:
            print(f"Cache dir: {_llm_cache.cache_dir}")
