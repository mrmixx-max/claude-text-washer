# claude-text-washer

Strips AI watermarks and rewrites text in organic, asymmetrical prose.

## Structure

```
claude-text-washer/
├── skills/content/text-washer/   # Hermes Skill
├── references/                   # Marker lists, blacklists
├── scripts/                      # CLI tools
│   ├── models.yaml               # Ollama model pool (config)
│   ├── ollama_utils.py           # Shared: model loading, validation, Ollama API
│   ├── marker_scan.py            # AI surface-marker scanner
│   ├── washer.py                 # Single-pass rewriter
│   ├── pipeline.py               # Multi-pass rewriter with presets
│   ├── stat_engine.py            # Statistical watermark detection
│   ├── file_washer.py            # File-level scan + wash orchestrator
│   └── stat_prompt.py            # Statistical anti-watermark prompt engineer
├── tests/                        # Tests
└── docs/                         # Research, findings
```

## Quickstart

```bash
# List models
python scripts/washer.py --list-models

# Scan markers
python scripts/marker_scan.py input.txt

# Wash text (CLI) — default model is llama3.2
python scripts/washer.py input.txt --output clean.txt

# Multi-pass with specific model
python scripts/pipeline.py input.txt --model qwen-coder --passes 2

# Statistical analysis (no Ollama needed)
python scripts/stat_engine.py input.txt --analyze-only --json

# File wash: scan + rewrite in one step
python scripts/file_washer.py input.txt --output clean.txt

# Generate statistical prompt + preview
python scripts/stat_prompt.py input.txt --preview --model darkest
```

## Unified CLI (claude-washer)

One command dispatches all tools — each subcommand module is lazy-imported,
so a broken optional dependency never breaks the entire CLI.

| Command                       | Module        | Task                                  |
|-------------------------------|---------------|---------------------------------------|
| `claude-washer scan <f>`      | marker_scan   | Scan for AI surface markers           |
| `claude-washer wash <f>`      | washer        | Single LLM rewrite (local via Ollama) |
| `claude-washer pipeline <f>`  | pipeline      | Multi-pass wash with presets          |
| `claude-washer file <files>`  | file_washer   | Batch wash: DOCX/MD/PDF/HTML/TXT      |
| `claude-washer chat`          | chat          | Interactive chat with Ollama          |
| `claude-washer edit [f]`      | editor        | Interactive text editor               |
| `claude-washer stat <f>`      | stat_engine   | Statistical analysis (no Ollama)      |
| `claude-washer prompt <f>`    | stat_prompt   | Anti-watermark prompt from statistics |

```bash
# Batch + glob + directory (recursive with -r)
claude-washer file *.md --outdir cleaned/
claude-washer file inputs/ -r --outdir cleaned/

# Override format detection
claude-washer file lie.txt --format pdf

# Analyze only (no Ollama call)
claude-washer file doc.pdf --dry-run

# Single file with explicit output
claude-washer file input.docx -o clean.txt
```

## Backends

claude-text-washer supports **any LLM backend** — local or remote:

| Backend        | Type    | Example URL                                      |
|----------------|---------|--------------------------------------------------|
| Ollama (local) | ollama  | `http://127.0.0.1:11434/api/generate` (default)  |
| vLLM           | openai  | `http://localhost:8000/v1/chat/completions`      |
| LM Studio      | openai  | `http://localhost:1234/v1/chat/completions`      |
| OpenRouter     | openai  | `https://openrouter.ai/api/v1/chat/completions`  |
| Together       | openai  | `https://api.together.xyz/v1/chat/completions`   |
| Groq           | openai  | `https://api.groq.com/openai/v1/chat/completions`|
| OpenAI         | openai  | `https://api.openai.com/v1/chat/completions`     |

### CLI Examples

```bash
# Local Ollama (default)
python scripts/washer.py input.txt --model llama3.2

# Remote OpenAI-compatible API
python scripts/washer.py input.txt \
  --base-url https://openrouter.ai/api/v1/chat/completions \
  --api-key $OPENROUTER_API_KEY \
  --model google/gemini-2.0-flash-001

# vLLM local
python scripts/washer.py input.txt \
  --base-url http://localhost:8000/v1/chat/completions \
  --model mistralai/Mistral-7B-Instruct-v0.3

# With temperature control
python scripts/washer.py input.txt --temperature 0.3  # conservative
python scripts/washer.py input.txt --temperature 1.2  # creative
```

### Temperature

Control randomness with `--temperature`:

| Value | Style       | Use case              |
|-------|-------------|-----------------------|
| 0.0-0.3 | Precise    | Factual, repetitive   |
| 0.5-0.7 | Balanced   | Natural prose         |
| 0.8-1.0 | Creative   | Varied, unpredictable |
| 1.0-2.0 | Chaotic    | Experimental          |

Run without `--temperature` for interactive preset selection.

### Configuration

Backend profiles are defined in `scripts/backends.yaml`. Add your own:

```yaml
my-custom:
  type: openai
  base_url: http://my-server:8080/v1/chat/completions
  api_key_env: MY_API_KEY
```

## Format Detection & Batch Processing

`file_washer` detects file format by **magic bytes** (not extension), so
e.g. `lie.txt` (actually a PDF) is correctly processed as PDF.
Supported formats: DOCX, PDF, XLSX, PPTX, EPUB, ODT, HTML, Markdown, TXT.
Heavyweights (`python-docx`, `pymupdf`, `beautifulsoup4`, `openpyxl`,
`python-pptx`) are lazy-imported — the tool runs without them (UTF-8
fallback for PDF/text).

Inputs can be multiple files, shell globs (`*.md`), and directories.
Results are written as `<stem>.washed.txt` — either next to the source,
in `--outdir` (batch) or explicitly via `-o`/`--output`.

## Model Selection

All CLI scripts accept `--model MODEL` and `--list-models`.

```bash
# List available models
python scripts/washer.py --list-models

# Use a specific model
python scripts/washer.py input.txt --model qwen-coder-7b --output clean.txt
python scripts/pipeline.py input.txt --model darkest --passes 3
python scripts/stat_engine.py input.txt --model gutenberg-26b
python scripts/file_washer.py input.txt --model qwen3-30b-a3b
python scripts/stat_prompt.py input.txt --model nemo-heretic --preview
```

### Available Models

| Model            | Size   | Description                         | Default |
|------------------|--------|-------------------------------------|---------|
| llama3.2         | 2GB    | Meta Llama 3.2 — general purpose    | ✅      |
| qwen-coder-7b    | 4.7GB  | Qwen Coder 7B — coding assistant    |         |
| qwen-coder       | 9GB    | Qwen Coder 9B — coding, larger      |         |
| eurollm-9b       | 5.6GB  | EuroLLM 9B — multilingual EU        |         |
| nemo-heretic     | 7.5GB  | NeMo Heretic — uncensored reasoning |         |
| darkest          | 9.6GB  | Darkest — creative / storytelling   |         |
| gutenberg-26b    | 15GB   | Gutenberg 26B — long-form writing   |         |
| lfm2-24b-a2b     | 14GB   | LFM2 24B A2B — large language model |         |
| qwen3-30b-a3b    | 17GB   | Qwen3 30B A3B — MoE, highest quality|         |
| lfm25-tool       | 2.2GB  | LFM25 Tool — lightweight, fast      |         |
| gemma-4-e4b      | 6.8GB  | Gemma 4 E4B — vision + language     |         |

### Presets (pipeline.py)

| Preset    | Model       | Temperature | Max Tokens |
|-----------|-------------|-------------|------------|
| fast      | lfm25-tool  | 0.7         | 512        |
| standard  | llama3.2    | 0.8         | 1024       |
| premium   | llama3.2    | 0.9         | 2048       |

`--model` overrides the preset.

### Model Configuration

The model list is defined in `scripts/models.yaml`. New models can be
added there — they appear automatically in `--list-models` and are
validated. An unknown model is rejected with an error message.

```yaml
# scripts/models.yaml
default: llama3.2
models:
  llama3.2:
    size: "2GB"
    description: "Meta Llama 3.2 — general purpose, default"
    default: true
  # ... more models
```

## Model Recommendation

- **llama3.2** (Ollama) — local, no data leaks
- **Temperature 0.8** — more creative, less predictable

## Tests

```bash
python -m pytest tests/ -v
```
