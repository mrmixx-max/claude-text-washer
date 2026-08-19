#!/usr/bin/env python3
"""Scan text for high-frequency AI surface markers (DE/EN). Stdlib only."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# (severity, pattern, note) — severity: 1=low 2=mid 3=high
PATTERNS: list[tuple[int, str, str]] = [
    # English
    (3, r"\bdelve(?:s|d)? into\b", "EN stock: delve into"),
    (3, r"\bleverage(?:s|d)?\b", "EN stock: leverage"),
    (3, r"\bunderscore(?:s|d)?\b", "EN stock: underscore"),
    (3, r"\btapestry\b", "EN stock: tapestry"),
    (3, r"\bin today'?s (?:digital |fast-paced )?world\b", "EN stock opener"),
    (3, r"\bit is important to (?:note|remember|understand)\b", "EN filler"),
    (3, r"\bin conclusion,?\s+it is clear\b", "EN template closer"),
    (2, r"\bcomprehensive (?:guide|overview|solution)\b", "EN marketing AI"),
    (2, r"\brobust(?:ness)?\b", "EN overused robust"),
    (2, r"\bseamlessly\b", "EN seamlessly"),
    (2, r"\bcutting-edge\b", "EN cutting-edge"),
    (2, r"\bgame-?changer\b", "EN game-changer"),
    (2, r"\bnavigat(?:e|ing) the (?:complex )?landscape\b", "EN landscape"),
    (2, r"\ba testament to\b", "EN testament"),
    (2, r"\bin the realm of\b", "EN realm"),
    (2, r"\bnot only\b.+\bbut also\b", "EN not only/but also"),
    (2, r"\bfurthermore,?\b", "EN furthermore"),
    (2, r"\bmoreover,?\b", "EN moreover"),
    (1, r"\butilize(?:s|d)?\b", "EN utilize→use?"),
    (1, r"\benhance(?:s|d)?\b", "EN enhance overuse"),
    # German
    (3, r"\b[Ii]n der heutigen (?:digitalen )?Zeit\b", "DE stock opener"),
    (3, r"\b[Ii]n der heutigen digitalen Welt\b", "DE stock opener"),
    (3, r"\b[Ee]s ist wichtig (?:zu betonen|festzuhalten|zu beachten)\b", "DE filler"),
    (3, r"\b[Zz]usammenfassend l[aä]sst sich sagen\b", "DE template closer"),
    (3, r"\b[Dd]ar[uü]ber hinaus\b", "DE darüber hinaus overuse"),
    (3, r"\bganzheitlich(?:e|en|er|es)?\b", "DE ganzheitlich buzz"),
    (3, r"\bnahtlos(?:e|en|er|es)?\b", "DE nahtlos buzz"),
    (2, r"\bSynergien\b", "DE Synergien buzz"),
    (2, r"\boptimieren\b", "DE optimieren overuse"),
    (2, r"\bim digitalen Zeitalter\b", "DE cliché"),
    (2, r"\b Immersive\b", "DE anglicism padding"),
    (2, r"\binnovativ(?:e|en|er|es)?\b", "DE innovativ overuse"),
    (2, r"\bwegweisend(?:e|en|er|es)?\b", "DE wegweisend"),
    (2, r"\bNicht nur\b.+\bsondern auch\b", "DE nicht nur/sondern auch"),
    (2, r"\bAbschließend\b", "DE abschließend template"),
    (1, r"\bdes Weiteren\b", "DE des Weiteren"),
    (1, r"\bferner\b", "DE ferner formal AI"),
]


def scan(text: str) -> list[tuple[int, str, str, str]]:
    hits: list[tuple[int, str, str, str]] = []
    for sev, pat, note in PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE | re.DOTALL):
            snippet = m.group(0).replace("\n", " ")
            if len(snippet) > 80:
                snippet = snippet[:77] + "..."
            hits.append((sev, note, snippet, str(m.start())))
    hits.sort(key=lambda h: (-h[0], int(h[3])))
    return hits


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AI surface-marker scanner")
    p.add_argument("path", nargs="?", help="Input file (UTF-8). Omit with --stdin")
    p.add_argument("--stdin", action="store_true", help="Read text from stdin")
    p.add_argument("--json", action="store_true", help="Machine-readable lines")
    args = p.parse_args(argv)

    if args.stdin or not args.path:
        text = sys.stdin.read()
    else:
        text = Path(args.path).read_text(encoding="utf-8")

    hits = scan(text)
    high = sum(1 for h in hits if h[0] >= 3)
    mid = sum(1 for h in hits if h[0] == 2)
    low = sum(1 for h in hits if h[0] == 1)

    if args.json:
        for sev, note, snippet, pos in hits:
            print(f"{sev}\t{pos}\t{note}\t{snippet}")
    else:
        print(f"markers: high={high} mid={mid} low={low} total={len(hits)}")
        for sev, note, snippet, pos in hits:
            label = {1: "low", 2: "mid", 3: "high"}[sev]
            print(f"[{label}] @{pos} {note} :: {snippet}")
        if high == 0 and mid == 0:
            print("OK: no mid/high surface markers found")
        else:
            print("FAIL: rewrite remaining mid/high hits")

    return 1 if high else 0


if __name__ == "__main__":
    raise SystemExit(main())
