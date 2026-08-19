# Claude Text Washer — Erkenntnisse & Forschung

## Claude-spezifische Marker (2026)

### Lexikalische Muster

Claude (Anthropic) verwendet charakteristische Phrasen, die sich von GPT-Mustern unterscheiden:

| Kategorie | Claude-Marker | Häufigkeit |
|-----------|---------------|------------|
| Einleitung | "Lassen Sie mich...", "Ich möchte betonen..." | Hoch |
| Abschluss | "Zusammenfassend...", "Abschließend..." | Sehr hoch |
| Übergang | "Darüber hinaus", "Des Weiteren" | Hoch |
| Abschwächung | "Es könnte argumentiert werden", "Einige würden sagen" | Mittel |
| Struktur | Drei-Gliederung (erstens, zweitens, drittens) | Sehr hoch |

### Claude vs. GPT — Unterschiede

| Merkmal | Claude | GPT |
|---------|--------|-----|
| Satzlänge | Länger, verschachtelter | Mittellang, gleichmäßig |
| Abschwächung | Stark ("möglicherweise", "in gewisser Weise") | Gering |
| Struktur | Dreigliedrig, ausgewogen | Viergliedrig, variabler |
| Ton | Höflich, akademisch | Direkt, pragmatisch |
| Em-Dash | Selten | Häufig |

### Statistische Wasserzeichen

Claude verwendet kein bekanntes statistisches Textwasserzeichen (wie Googles SynthID). Die Erkennung erfolgt über:

1. **Stilometrie** — Satzlänge, Vokabular, Syntax
2. **N-Gramm-Analyse** — charakteristische Wortpaare
3. **Perplexität** — Claude-Text hat niedrigere Perplexität (vorhersehbarer)

### Rewrite-Strategien

1. **Burstigkeit erhöhen** — kurze Sätze zwischen lange mischen
2. **Abschwächungen entfernen** — direkte Aussagen statt "man könnte"
3. **Drei-Gliederung brechen** — ungerade Anzahlen, asymmetrische Abschnitte
4. **Vokabular variieren** — seltenere Verben, konkrete Substantive
5. **Übergänge reduzieren** — nicht jeder Absatz braucht Einleitung

## Quellen

- [Anthropic — Constititional AI](https://www.anthropic.com/research/claude-model-spec)
- [Pangram Labs — AI Detection](https://pangram.com/)
- [GPTZero — Perplexity Scoring](https://gptzero.me/)
