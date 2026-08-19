# claude-text-washer

Entfernt KI-Wasserzeichen und schreibt Texte in organischer, asymmetrischer Prosa um.

## Struktur

```
claude-text-washer/
├── skills/content/text-washer/   # Hermes Skill
├── references/                   # Marker-Listen, Blacklists
├── scripts/                      # CLI-Tools
│   ├── models.yaml               # Ollama model pool (config)
│   ├── ollama_utils.py           # Shared: model loading, validation, Ollama API
│   ├── marker_scan.py            # AI surface-marker scanner
│   ├── washer.py                 # Single-pass rewriter
│   ├── pipeline.py               # Multi-pass rewriter with presets
│   ├── stat_engine.py            # Statistical watermark detection
│   ├── file_washer.py            # File-level scan + wash orchestrator
│   └── stat_prompt.py            # Statistical anti-watermark prompt engineer
├── tests/                        # Tests
└── docs/                         # Erkenntnisse, Forschung
```

## Schnellstart

```bash
# Modelle anzeigen
python scripts/washer.py --list-models

# Marker scannen
python scripts/marker_scan.py input.txt

# Text waschen (CLI) — Standardmodell ist llama3.2
python scripts/washer.py input.txt --output clean.txt

# Multi-pass mit anderem Modell
python scripts/pipeline.py input.txt --model qwen-coder --passes 2

# Statistische Analyse (kein Ollama nötig)
python scripts/stat_engine.py input.txt --analyze-only --json

# Datei-waschen: Scan + Rewrite in einem Schritt
python scripts/file_washer.py input.txt --output clean.txt

# Statistischen Prompt generieren + Vorschau
python scripts/stat_prompt.py input.txt --preview --model darkest
```

## Einheitliche CLI (claude-washer)

Ein einziger Befehl leitet alle Werkzeuge weiter — jedes Subcommand-Modul
wird lazy importiert, sodass ein defekter optionaler Abhängigkeit nie die
ganze CLI bricht.

| Befehl                       | Modul         | Aufgabe                                  |
|------------------------------|---------------|------------------------------------------|
| `claude-washer scan <f>`     | marker_scan   | AI-Oberflächen-Marker scannen            |
| `claude-washer wash <f>`     | washer        | Einzelner LLM-Rewrite (lokal via Ollama) |
| `claude-washer pipeline <f>` | pipeline      | Multi-Pass-Wash mit Presets              |
| `claude-washer file <files>` | file_washer   | Batch-Wash: DOCX/MD/PDF/HTML/TXT         |
| `claude-washer chat`         | chat          | Interaktiver Chat mit Ollama             |
| `claude-washer edit [f]`     | editor        | Interaktiver Texteditor                  |
| `claude-washer stat <f>`     | stat_engine   | Statistische Analyse (kein Ollama nötig) |
| `claude-washer prompt <f>`   | stat_prompt   | Anti-Watermark-Prompt aus Statistik      |

```bash
# Batch + Glob + Verzeichnis (rekursiv mit -r)
claude-washer file *.md --outdir cleaned/
claude-washer file inputs/ -r --outdir cleaned/

# Format-Erkennung überschreiben
claude-washer file lie.txt --format pdf

# Nur analysieren (kein Ollama-Aufruf)
claude-washer file doc.pdf --dry-run

# Einzelne Datei mit ausgegebener Ausgabedatei
claude-washer file input.docx -o clean.txt
```

## Format-Erkennung & Batch-Verarbeitung

`file_washer` erkennt das Dateiformat anhand **Magischer Bytes** (nicht der
Endung), sodass z. B. `lie.txt` (das eigentlich eine PDF ist) korrekt als PDF
verarbeitet wird. Unterstützte Formate: DOCX, PDF, XLSX, PPTX, EPUB, ODT, HTML
und Markdown sowie TXT. Schwerpunkte (`python-docx`, `pymupdf`,
`beautifulsoup4`, `openpyxl`, `python-pptx`) werden lazy importiert — das Tool
läuft auch ohne sie (UTF-8-Fallback für PDF/Text).

Eingaben dürfen mehrere Dateien, Shell-Globs (`*.md`) und Verzeichnisse sein.
Ergebnisse werden jeweils als `<stem>.washed.txt` geschrieben — entweder neben
der Quelldatei, in `--outdir` (Batch) oder explizit via `-o`/`--output`.

## Modellauswahl

Alle CLI-Skripte akzeptieren `--model MODEL` und `--list-models`.

```bash
# Verfügbare Modelle auflisten
python scripts/washer.py --list-models

# Ein bestimmtes Modell verwenden
python scripts/washer.py input.txt --model qwen-coder-7b --output clean.txt
python scripts/pipeline.py input.txt --model darkest --passes 3
python scripts/stat_engine.py input.txt --model gutenberg-26b
python scripts/file_washer.py input.txt --model qwen3-30b-a3b
python scripts/stat_prompt.py input.txt --model nemo-heretic --preview
```

### Verfügbare Modelle

| Modell            | Größe  | Beschreibung                        | Standard |
|--------------------|--------|-------------------------------------|----------|
| llama3.2           | 2GB    | Meta Llama 3.2 — general purpose    | ✅       |
| qwen-coder-7b      | 4.7GB  | Qwen Coder 7B — coding assistant    |          |
| qwen-coder         | 9GB    | Qwen Coder 9B — coding, larger      |          |
| eurollm-9b         | 5.6GB  | EuroLLM 9B — multilingual EU        |          |
| nemo-heretic       | 7.5GB  | NeMo Heretic — uncensored reasoning |          |
| darkest            | 9.6GB  | Darkest — creative / storytelling   |          |
| gutenberg-26b      | 15GB   | Gutenberg 26B — long-form writing   |          |
| lfm2-24b-a2b       | 14GB   | LFM2 24B A2B — large language model |          |
| qwen3-30b-a3b      | 17GB   | Qwen3 30B A3B — MoE, highest quality|          |
| lfm25-tool         | 2.2GB  | LFM25 Tool — lightweight, fast      |          |
| gemma-4-e4b        | 6.8GB  | Gemma 4 E4B — vision + language     |          |

### Presets (pipeline.py)

| Preset    | Modell      | Temperatur | Max. Tokens |
|-----------|-------------|------------|-------------|
| fast      | lfm25-tool  | 0.7        | 512         |
| standard  | llama3.2    | 0.8        | 1024        |
| premium   | llama3.2    | 0.9        | 2048        |

`--model` überschreibt das Preset.

### Modell-Konfiguration

Die Modellliste ist in `scripts/models.yaml` definiert. Neue Modelle können dort
hinzugefügt werden — sie werden automatisch in `--list-models` angezeigt und
validiert. Ein unbekanntes Modell wird mit einer Fehlermeldung abgelehnt.

```yaml
# scripts/models.yaml
default: llama3.2
models:
  llama3.2:
    size: "2GB"
    description: "Meta Llama 3.2 — general purpose, default"
    default: true
  # ... weitere Modelle
```

## Modell-Empfehlung

- **llama3.2** (Ollama) — lokal, keine Datenabflüsse
- **Temperature 0.8** — kreativer, weniger vorhersehbar

## Tests

```bash
python -m pytest tests/ -v
```
