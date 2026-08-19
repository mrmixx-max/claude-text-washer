# Claude / LLM Text Markers — Deutsch & Englisch

## High-Severity (sev 1 — always a marker)

| Phrase (DE) | Phrase (EN) | Notes |
|--------------|-------------|-------|
| Zusammenfassend lässt sich sagen | In conclusion, it is clear that | Template closer |
| Es ist wichtig zu beachten | It is important to note | Filler opener |
| Ein weiteres Element | Another element | Generic transition |
| In der heutigen Zeit | In today's digital landscape | Cliché opener |
| Darüber hinaus | Furthermore / Moreover | Twin-adverb |
| Im Folgenden | In the following | Template transition |
| Es sei denn, | Unless otherwise stated | Legalistic padding |
| Vor diesem Hintergrund | Against this context | German biz cliché |
| Nicht zuletzt | Last but not least | Empty stacking |

## Medium-Severity (sev 2 — suspicious in clusters)

| Phrase (DE) | Phrase (EN) | Notes |
|--------------|-------------|-------|
| facettenreich | multifaceted | Buzzword |
| Geflecht / Netzwerk | tapestry / landscape | Abstract noun |
| essenziell | essential | Overused adj |
| dynamisch | dynamic | Empty adj |
| nahtlos | seamlessly | Tech buzz |
| ganzheitlich | holistic | Mgmt buzz |
| im Kern | at its core | Filler |
| letztlich | ultimately | Hedging |
| schließlich | finally / after all | Transition crutch |
| zugleich | at the same time | Parallelism crutch |

## Structural Markers (not lexical)

- Three identical sentence openings in a row
- Perfectly balanced paragraph lengths
- Every paragraph starts with a transition phrase
- No sentence shorter than 10 words
- No sentence longer than 35 words
- All bullets follow identical grammatical pattern
- Intro → 3 pillars → Conclusion arc

## Claude-Specific Patterns

- "I'd be happy to help with that!"
- "Let me break this down for you..."
- "Here's what you need to know:"
- "It's worth noting that..."
- "That said,..."
- Excessive hedging: "It appears that...", "One might consider..."

## Detection Heuristic

A text with >3 sev-1 hits OR >6 sev-2 hits is very likely AI-generated.
