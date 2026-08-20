# claude-text-washer

**Strip AI watermarks. Rewrite text in organic, asymmetrical prose.**

Detects and removes statistical AI markers (green-list bias, burstiness, entropy patterns) and rewrites text via local or remote LLMs. Supports **any LLM backend** — Ollama, vLLM, LM Studio, OpenRouter, Together, Groq, OpenAI.

## Features

- **Universal LLM support** — Ollama (local) + any OpenAI-compatible API
- **Statistical watermark detection** — perplexity, burstiness, entropy, Zipf, hapax, TTR, n-gram bias
- **Multi-pass rewriting** — progressive prompts (rewrite → pattern fix → naturalize) with early termination
- **Smart cleaning** — regex-based pre/post processing + invisible Unicode marker removal
- **Temperature control** — CLI flag or interactive preset selection
- **Batch processing** — DOCX, PDF, XLSX, PPTX, EPUB, ODT, HTML, Markdown, TXT
- **Circuit breaker + retry** — resilient against rate limits and failures
- **Zero LLM cost mode** — statistical analysis works without any model

## Quickstart

```bash
# Wash text with local Ollama (default)
python scripts/washer.py input.txt --output clean.txt

# Use any OpenAI-compatible API
python scripts/washer.py input.txt \
  --base-url https://openrouter.ai/api/v1/chat/completions \
  --api-key $OPENROUTER_API_KEY \
  --model google/gemini-2.0-flash-001

# Multi-pass with temperature
python scripts/pipeline.py input.txt --passes 3 --temperature 0.9

# Statistical analysis only (no LLM needed)
python scripts/stat_engine.py input.txt --analyze-only --json
```

## Backends

Supports **any LLM backend** — auto-detects Ollama vs OpenAI-compatible APIs:

| Backend        | Type    | Example URL                                      |
|----------------|---------|--------------------------------------------------|
| Ollama (local) | ollama  | `http://127.0.0.1:11434/api/generate` (default)  |
| vLLM           | openai  | `http://localhost:8000/v1/chat/completions`      |
| LM Studio      | openai  | `http://localhost:1234/v1/chat/completions`      |
| OpenRouter     | openai  | `https://openrouter.ai/api/v1/chat/completions`  |
| Together       | openai  | `https://api.together.xyz/v1/chat/completions`   |
| Groq           | openai  | `https://api.groq.com/openai/v1/chat/completions`|
| OpenAI         | openai  | `https://api.openai.com/v1/chat/completions`     |

Backend profiles are defined in `scripts/backends.yaml`. Add your own:

```yaml
my-server:
  type: openai
  base_url: http://my-server:8080/v1/chat/completions
  api_key_env: MY_API_KEY
```

## Temperature

Control randomness with `--temperature` or interactive preset selection:

| Value   | Preset      | Style                |
|---------|-------------|----------------------|
| 0.0–0.3 | Conservative | Precise, factual     |
| 0.5–0.7 | Balanced    | Natural prose        |
| 0.8–1.0 | Creative    | Varied, unpredictable|
| 1.0–2.0 | Chaotic     | Experimental         |

```bash
python scripts/washer.py input.txt --temperature 0.3   # conservative
python scripts/washer.py input.txt --temperature 0.9   # creative (default)
python scripts/washer.py input.txt                     # interactive prompt
```

## Structure

```
claude-text-washer/
├── scripts/
│   ├── backends.yaml       # LLM backend profiles (Ollama, vLLM, OpenRouter, ...)
│   ├── models.yaml         # Ollama model pool
│   ├── ollama_utils.py     # Generic LLM client (Ollama + OpenAI-compatible)
│   ├── marker_scan.py      # AI surface-marker scanner
│   ├── smart_cleaner.py    # Regex pre/post processing + Unicode cleanup
│   ├── washer.py           # Single-pass rewriter
│   ├── pipeline.py         # Multi-pass rewriter with progressive prompts
│   ├── stat_engine.py      # Statistical watermark detection
│   ├── stat_prompt.py      # Anti-watermark prompt engineering
│   ├── file_washer.py      # File-level batch orchestrator
│   ├── chat.py             # Interactive chat
│   ├── editor.py           # Interactive text editor
│   ├── menu.py             # TUI menu
│   └── multi_agent_washer.py  # Parallel multi-model washing
├── tests/                  # 235+ tests
├── references/             # Marker lists, blacklists
├── docs/                   # Research findings
└── skills/content/text-washer/  # Hermes Skill
```

## Unified CLI

```bash
claude-washer wash input.txt           # Single-pass
claude-washer pipeline input.txt -p 3  # Multi-pass
claude-washer file *.md -r --outdir cleaned/  # Batch
claude-washer scan input.txt           # Marker scan only
claude-washer stat input.txt           # Statistical analysis
claude-washer chat                     # Interactive chat
claude-washer edit                     # Interactive editor
```

## Detection Engine

Statistical watermark detection (no LLM required):

| Metric           | What it measures                          |
|------------------|-------------------------------------------|
| Perplexity       | Predictability of token distribution      |
| Burstiness       | Variance in sentence length               |
| Word entropy     | Shannon entropy of token distribution     |
| Sentence entropy | Structural variety across sentences       |
| Green-list bias  | Overuse of common function words          |
| N-gram bias      | Repetitive word pair frequency            |
| TTR              | Type-token ratio (vocabulary richness)    |
| Zipf coefficient | Deviation from natural Zipf distribution  |
| Hapax ratio      | Words appearing only once                 |
| **AI Score**     | Aggregate 0–100 (higher = more likely AI) |

## Presets (multi-pass pipeline)

| Preset    | Model       | Temperature | Max Tokens | Use case          |
|-----------|-------------|-------------|------------|-------------------|
| fast      | lfm25-tool  | 0.7         | 512        | Quick draft       |
| standard  | llama3.2    | 0.8         | 1024       | Balanced quality  |
| premium   | llama3.2    | 0.9         | 2048       | Maximum quality   |

Progressive pass prompts:
1. **Rewrite** — destroy AI patterns, maximize burstiness
2. **Pattern fixer** — target remaining markers, vary structure
3. **Naturalizer** — break rhythm, add controlled imperfections

Early termination when AI score drops below 25.

## Tests

```bash
python -m pytest tests/ -v
```

235+ tests covering detection, cleaning, pipeline, streaming, circuit breakers, and integration.

## License

MIT
