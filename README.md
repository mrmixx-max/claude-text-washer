# claude-text-washer

Entfernt KI-Wasserzeichen und schreibt Texte in organischer, asymmetrischer Prosa um.

## Struktur

```
claude-text-washer/
├── skills/content/text-washer/   # Hermes Skill
├── references/                   # Marker-Listen, Blacklists
├── scripts/                      # CLI-Tools (marker_scan, washer)
├── tests/                        # Tests
└── docs/                         # Erkenntnisse, Forschung
```

## Schnellstart

```bash
# Marker scannen
python scripts/marker_scan.py input.txt

# Text waschen (CLI)
python scripts/washer.py input.txt --output clean.txt
```

## Modell-Empfehlung

- **llama3.2** (Ollama) — lokal, keine Datenabflüsse
- **Temperature 0.8** — kreativer, weniger vorhersehbar
