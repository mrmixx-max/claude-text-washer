# Claude Text Washer — Research & Findings

## Claude-Specific Markers (2026)

### Lexical Patterns

Claude (Anthropic) uses characteristic phrases that differ from GPT patterns:

| Category   | Claude Marker                              | Frequency |
|------------|--------------------------------------------|-----------|
| Opening    | "Let me...", "I'd like to emphasize..."    | High      |
| Closing    | "In summary...", "To conclude..."          | Very high |
| Transition | "Furthermore", "In addition"               | High      |
| Hedging    | "It could be argued", "Some would say"     | Medium    |
| Structure  | Tripartite (first, second, third)          | Very high |

### Claude vs. GPT — Differences

| Feature      | Claude                          | GPT                     |
|--------------|---------------------------------|-------------------------|
| Sentence len | Longer, more nested             | Medium, uniform         |
| Hedging      | Strong ("possibly", "somewhat") | Weak                    |
| Structure    | Tripartite, balanced            | Quadripartite, variable |
| Tone         | Polite, academic                | Direct, pragmatic       |
| Em-dash      | Rare                            | Frequent                |

### Statistical Watermarks

Claude uses no known statistical text watermark (like Google's SynthID). Detection relies on:

1. **Stylometry** — sentence length, vocabulary, syntax
2. **N-gram analysis** — characteristic word pairs
3. **Perplexity** — Claude text has lower perplexity (more predictable)

### Rewrite Strategies

1. **Increase burstiness** — mix short and long sentences
2. **Remove hedging** — direct statements instead of "one could"
3. **Break tripartite structure** — odd numbers, asymmetrical sections
4. **Vary vocabulary** — rarer verbs, concrete nouns
5. **Reduce transitions** — not every paragraph needs an introduction

## Sources

- [Anthropic — Constitutional AI](https://www.anthropic.com/research/claude-model-spec)
- [Pangram Labs — AI Detection](https://pangram.com/)
- [GPTZero — Perplexity Scoring](https://gptzero.me/)
