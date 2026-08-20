# Claude Text Washer — Research & Findings

## AI Watermark Detection (2026)

### Detection Methods

#### 1. Lexical Surface Markers

High-frequency phrases characteristic of AI-generated text:

| Category   | Claude Markers                            | Frequency |
|------------|-------------------------------------------|-----------|
| Opening    | "Let me...", "I'd like to emphasize..."   | High      |
| Closing    | "In summary...", "To conclude..."         | Very high |
| Transition | "Furthermore", "In addition"              | High      |
| Hedging    | "It could be argued", "Some would say"    | Medium    |
| Structure  | Tripartite (first, second, third)         | Very high |

#### 2. Statistical Watermarks

Claude uses no known statistical text watermark (like Google's SynthID). Detection relies on:

1. **Stylometry** — sentence length, vocabulary, syntax
2. **N-gram analysis** — characteristic word pairs
3. **Perplexity** — Claude text has lower perplexity (more predictable)
4. **Green-list bias** — overuse of common function words
5. **Zipf deviation** — AI text deviates from natural Zipf distribution

#### 3. Invisible Unicode Markers

Forum research identified 28 invisible Unicode characters commonly found in AI text:

| Character | Name                   | Frequency |
|-----------|------------------------|-----------|
| U+200A    | Hair Space             | ×10       |
| U+202F    | Narrow No-Break Space  | ×18       |

Additional markers: U+200B (Zero Width Space), U+200C (ZWNJ), U+200D (ZWJ), U+FEFF (BOM).

### Claude vs. GPT — Key Differences

| Feature       | Claude                          | GPT                     |
|---------------|---------------------------------|-------------------------|
| Sentence len  | Longer, more nested             | Medium, uniform         |
| Hedging       | Strong ("possibly", "somewhat") | Weak                    |
| Structure     | Tripartite, balanced            | Quadripartite, variable |
| Tone          | Polite, academic                | Direct, pragmatic       |
| Em-dash       | Rare                            | Frequent                |
| Perplexity    | Lower (more predictable)        | Higher                  |

### Rewrite Strategies

1. **Maximize burstiness** — alternate radically between short (1-4 words) and long (20+ words) sentences
2. **Remove hedging** — direct statements instead of "one could argue"
3. **Break tripartite structure** — odd numbers, asymmetrical sections
4. **Force perplexity** — use unconventional, precise verbs
5. **Reduce transitions** — not every paragraph needs an introduction
6. **Add imperfections** — controlled "noise" through varied punctuation and fragments

### Temperature Impact

| Temperature | Effect on AI Detection                          |
|-------------|-------------------------------------------------|
| 0.0–0.3     | More predictable, easier to detect              |
| 0.5–0.7     | Natural balance, moderate detection resistance  |
| 0.8–1.0     | Diverse output, harder to detect                |
| 1.0–2.0     | Highly random, most resistant but less coherent |

## Multi-Pass Architecture

### Progressive Prompts

Each pass has a different focus for deeper cleaning:

1. **Pass 1 — Rewrite**: Destroy AI patterns, maximize burstiness, force perplexity
2. **Pass 2 — Pattern Fixer**: Target remaining markers, vary sentence structure
3. **Pass 3 — Naturalizer**: Break rhythmic patterns, add controlled imperfections

### Early Termination

The pipeline terminates early when the AI score drops below 25, avoiding unnecessary LLM calls.

### Circuit Breaker

Per-model circuit breaker prevents cascading failures:
- After 3 consecutive failures → OPEN (fail fast for 30s)
- Half-open probe after cooldown → CLOSE on success

## Sources

- [Anthropic — Constitutional AI](https://www.anthropic.com/research/claude-model-spec)
- [Pangram Labs — AI Detection](https://pangram.com/)
- [GPTZero — Perplexity Scoring](https://gptzero.me/)
- [OpenAI — AI Text Detection](https://openai.com/blog/new-ai-tools-for-educators/)
