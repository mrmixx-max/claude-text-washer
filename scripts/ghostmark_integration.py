"""GhostMark Integration for text-watermark-studio & claude-text-washer.

Ported from: https://github.com/kilopal/GhostMark (Apache 2.0)
Fixes applied to original Python port:
1. RNG propagation in pass_synonyms (was using global random)
2. Burstiness merge index collision (i % 2 == 0 vs i % 3)
3. Case-preserving transition replacement
4. Word boundary-aware synonym replacement (was splitting text incorrectly)
5. ZWNJ injection in apply_homoglyphs (was completely missing)
6. Double homoglyph injection removed (scrub+apply = 22% → correct 15%)
7. Deterministic seeding via seedable_rng for reproducibility
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

# ═══════════════════════════════════════════════════════════════
# UNICODE SCRUB (Layer 1)
# ═══════════════════════════════════════════════════════════════

# Invisible Unicode characters used for steganographic watermarking
_UNICODE_STRIP = {
    "​",  # Zero Width Space
    "‌",  # Zero Width Non-Joiner
    "‍",  # Zero Width Joiner
    "‎",  # Left-to-Right Mark
    "‏",  # Right-to-Left Mark
    "﻿",  # Zero Width No-Break Space (BOM)
}

# Unicode tag characters (U+E0000–U+E007F)
_UNICODE_TAG_RE = re.compile(r"[\U000E0000-\U000E007F]")


def scrub_unicode(text: str, aggressive: bool = False) -> str:
    """Remove invisible Unicode characters used for watermarking.

    Lossless — removes only zero-width/BOM/tag characters.
    If aggressive=True, also applies homoglyph perturbation (non-lossless).
    """
    result = []
    for c in text:
        cp = ord(c)
        # Strip zero-width characters
        if cp in (0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0xFEFF):
            continue
        # Strip Unicode tag characters (U+E0000–U+E007F)
        if 0xE0000 <= cp <= 0xE007F:
            continue
        result.append(c)

    cleaned = "".join(result)

    if aggressive:
        return apply_homoglyphs(cleaned)
    return cleaned


# ═══════════════════════════════════════════════════════════════
# SEEDABLE RNG (Rust LCG port)
# ═══════════════════════════════════════════════════════════════

@dataclass
class LcgRng:
    """LCG random number generator matching GhostMark's Rust implementation.

    Uses parameters: state = state.wrapping_mul(1664525).wrapping_add(1013904223)
    Seed: text_len ^ 0xDEAD (Rust) or 0xCAFE (homoglyphs)
    """
    state: int = 0

    @classmethod
    def from_seed(cls, seed: int) -> LcgRng:
        return cls(state=(seed + 1) & 0xFFFFFFFF)

    def next(self) -> int:
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.state

    def next_float(self) -> float:
        return self.next() / 0xFFFFFFFF

    def pick(self, options: list[str]) -> str:
        return options[self.next() % len(options)]


# ═══════════════════════════════════════════════════════════════
# PASS 1: SYNONYM REPLACEMENT
# ═══════════════════════════════════════════════════════════════

_SYNONYMS: dict[str, list[str]] = {
    # Verbs
    "delve": ["dig", "look", "explore", "get"],
    "delves": ["digs", "looks", "explores", "gets"],
    "delving": ["digging", "looking", "exploring", "getting"],
    "utilize": ["use", "work with", "rely on"],
    "utilizes": ["uses", "works with", "relies on"],
    "utilizing": ["using", "working with", "relying on"],
    "leverage": ["use", "tap into", "rely on"],
    "leverages": ["uses", "taps into", "relies on"],
    "leveraging": ["using", "tapping into", "relying on"],
    "navigate": ["handle", "deal with", "work through", "tackle"],
    "navigates": ["handles", "deals with", "works through", "tackles"],
    "navigating": ["handling", "dealing with", "working through", "tackling"],
    "foster": ["build", "grow", "encourage", "support"],
    "fosters": ["builds", "grows", "encourages", "supports"],
    "fostering": ["building", "growing", "encouraging", "supporting"],
    "empower": ["help", "let", "give power to", "enable"],
    "empowers": ["helps", "lets", "gives power to", "enables"],
    "empowering": ["helping", "letting", "giving power to", "enabling"],
    "streamline": ["simplify", "speed up", "cut down on"],
    "streamlines": ["simplifies", "speeds up", "cuts down on"],
    "streamlining": ["simplifying", "speeding up", "cutting down on"],
    "optimize": ["improve", "fine-tune", "make better"],
    "optimizes": ["improves", "fine-tunes", "makes better"],
    "optimizing": ["improving", "fine-tuning", "making better"],
    "enhance": ["boost", "improve", "strengthen"],
    "enhances": ["boosts", "improves", "strengthens"],
    "enhancing": ["boosting", "improving", "strengthening"],
    "facilitate": ["help", "make easier", "support"],
    "facilitates": ["helps", "makes easier", "supports"],
    "facilitating": ["helping", "making easier", "supporting"],
    "underscore": ["highlight", "show", "stress"],
    "underscores": ["highlights", "shows", "stresses"],
    "underscoring": ["highlighting", "showing", "stressing"],
    "bolster": ["strengthen", "support", "back up"],
    "bolsters": ["strengthens", "supports", "backs up"],
    "bolstering": ["strengthening", "supporting", "backing up"],
    "spearhead": ["lead", "drive", "push"],
    "spearheads": ["leads", "drives", "pushes"],
    "spearheading": ["leading", "driving", "pushing"],
    "revolutionize": ["change", "shake up", "transform"],
    "revolutionizes": ["changes", "shakes up", "transforms"],
    "implement": ["set up", "put in place", "roll out"],
    "implements": ["sets up", "puts in place", "rolls out"],
    "implementing": ["setting up", "putting in place", "rolling out"],
    "demonstrate": ["show", "prove", "make clear"],
    "demonstrates": ["shows", "proves", "makes clear"],
    "demonstrating": ["showing", "proving", "making clear"],
    "encompass": ["cover", "include", "span"],
    "encompasses": ["covers", "includes", "spans"],
    "prioritize": ["focus on", "put first", "rank"],
    "prioritizes": ["focuses on", "puts first", "ranks"],
    "integrate": ["combine", "blend", "mix", "merge"],
    "integrates": ["combines", "blends", "mixes", "merges"],
    "integrating": ["combining", "blending", "mixing", "merging"],
    "catalyze": ["trigger", "spark", "kick off"],
    "catalyzes": ["triggers", "sparks", "kicks off"],
    # Adjectives
    "crucial": ["key", "big", "major", "important"],
    "pivotal": ["key", "central", "major"],
    "comprehensive": ["full", "thorough", "complete", "broad"],
    "robust": ["strong", "solid", "tough"],
    "seamless": ["smooth", "easy", "effortless"],
    "seamlessly": ["smoothly", "easily", "effortlessly"],
    "unprecedented": ["never-before-seen", "unmatched", "record-breaking"],
    "transformative": ["game-changing", "ground-breaking", "radical"],
    "multifaceted": ["complex", "many-sided", "layered"],
    "innovative": ["creative", "fresh", "new", "clever"],
    "dynamic": ["active", "energetic", "lively", "changing"],
    "nuanced": ["subtle", "detailed", "layered"],
    "holistic": ["complete", "full-picture", "all-round"],
    "myriad": ["many", "tons of", "loads of", "countless"],
    "intricate": ["complex", "detailed", "involved"],
    "overarching": ["main", "broad", "overall"],
    "unparalleled": ["unmatched", "one-of-a-kind", "rare"],
    "cutting-edge": ["latest", "newest", "advanced"],
    "groundbreaking": ["revolutionary", "pioneering", "radical"],
    "invaluable": ["priceless", "extremely useful", "essential"],
    "indispensable": ["essential", "must-have", "necessary"],
    "noteworthy": ["worth noting", "remarkable", "interesting"],
    "profound": ["deep", "huge", "intense", "powerful"],
    "significant": ["big", "major", "important", "notable"],
    "substantial": ["large", "big", "considerable"],
    "remarkable": ["impressive", "striking", "notable"],
    "imperative": ["critical", "urgent", "necessary"],
    "paramount": ["top", "supreme", "most important"],
    "burgeoning": ["growing", "rising", "booming"],
    "salient": ["key", "main", "notable"],
    "commendable": ["praiseworthy", "admirable", "impressive"],
    "meticulous": ["careful", "precise", "thorough"],
    "discernible": ["noticeable", "visible", "clear"],
    "adept": ["skilled", "good at", "sharp"],
    # Nouns
    "landscape": ["space", "scene", "world", "environment"],
    "paradigm": ["model", "framework", "approach"],
    "synergy": ["teamwork", "cooperation", "combined effort"],
    "synergistic": ["teamwork", "cooperation", "combined effort"],
    "tapestry": ["mix", "blend", "web"],
    "catalyst": ["driver", "trigger", "spark"],
    "catalysts": ["drivers", "triggers", "sparks"],
    "testament": ["proof", "sign", "evidence"],
    "prowess": ["skill", "talent", "ability"],
    "realm": ["area", "field", "world", "space"],
    "endeavor": ["effort", "project", "work"],
    "endeavors": ["efforts", "projects", "work"],
    "trajectory": ["path", "direction", "course"],
    "cornerstone": ["foundation", "basis", "pillar"],
    "underpinning": ["foundation", "basis", "core"],
    "underpinnings": ["foundations", "bases", "cores"],
    "stakeholders": ["people involved", "parties", "players"],
    "implications": ["effects", "consequences", "impact"],
    "ramifications": ["consequences", "effects", "fallout"],
    "juxtaposition": ["contrast", "comparison", "difference"],
    "plethora": ["ton", "bunch", "lots"],
    "intricacies": ["details", "complexities", "ins and outs"],
    "conundrum": ["puzzle", "problem", "dilemma"],
    "dichotomy": ["split", "divide", "contrast"],
    # Adverbs
    "rapidly": ["quickly", "fast", "at speed"],
    "increasingly": ["more and more", "gradually more"],
    "fundamentally": ["at its core", "basically", "at heart"],
    "inherently": ["by nature", "naturally", "at its core"],
    "undoubtedly": ["no doubt", "clearly", "for sure", "without question"],
    "arguably": ["you could say", "possibly", "some would say"],
    "moreover": ["also", "on top of that", "plus"],
    "nevertheless": ["still", "even so", "but"],
    "consequently": ["so", "as a result", "because of this"],
    "subsequently": ["then", "after that", "next"],
    "furthermore": ["also", "plus", "on top of that", "and"],
    "additionally": ["also", "plus", "on top of that"],
    "conversely": ["on the flip side", "on the other hand"],
    "simultaneously": ["at the same time", "together"],
    "predominantly": ["mostly", "mainly", "largely"],
    "particularly": ["especially", "mainly", "specifically"],
    "essentially": ["basically", "really", "at its core"],
    "notably": ["especially", "in particular", "worth noting"],
    "evidently": ["clearly", "obviously", "it seems"],
    "markedly": ["noticeably", "clearly", "significantly"],
    # Phrases (single words caught in context)
    "interconnected": ["linked", "connected", "tied together"],
    "data-driven": ["based on data", "evidence-based"],
    "forward-thinking": ["progressive", "ahead of the curve"],
    "well-established": ["proven", "solid", "long-standing"],
    "ever-evolving": ["always changing", "constantly shifting"],
}

# Compile regex for word boundary matching
_WORD_RE = re.compile(r"\b[\w']+\b")


def pass_synonyms(text: str, rng: LcgRng) -> str:
    """Pass 1: Replace AI-typical words with human alternatives.

    Uses LCG RNG for deterministic selection. Preserves word case.
    """
    def _swap_word(word: str) -> str:
        lower = word.lower()
        if lower not in _SYNONYMS:
            return word

        is_cap = word[0].isupper() if word else False
        is_all_cap = word.isupper() and len(word) > 1
        options = _SYNONYMS[lower]
        chosen = rng.pick(options)

        if is_all_cap:
            return chosen.upper()
        elif is_cap:
            return chosen[0].upper() + chosen[1:] if chosen else ""
        return chosen

    # Reconstruct text preserving non-word characters
    parts = []
    last_end = 0
    for match in _WORD_RE.finditer(text):
        parts.append(text[last_end:match.start()])
        parts.append(_swap_word(match.group()))
        last_end = match.end()
    parts.append(text[last_end:])
    return "".join(parts)


# ═══════════════════════════════════════════════════════════════
# PASS 2: TRANSITION REPLACEMENT (case-preserving)
# ═══════════════════════════════════════════════════════════════

_TRANSITIONS: list[tuple[str, str]] = [
    ("In today's rapidly evolving", "These days, in a fast-moving"),
    ("In today's quickly evolving", "In a world that moves quickly"),
    ("In today's fast-paced world", "In a fast-moving world"),
    ("it becomes increasingly evident that", "it's pretty clear that"),
    ("it becomes evident that", "you can see that"),
    ("it is important to note that", "it's worth noting"),
    ("it is worth noting that", "one thing to keep in mind is"),
    ("it should be noted that", "keep in mind"),
    ("in the realm of", "in"),
    ("in the context of", "when it comes to"),
    ("in light of", "given"),
    ("with regard to", "about"),
    ("in order to", "to"),
    ("due to the fact that", "because"),
    ("as a matter of fact", "actually"),
    ("at the end of the day", "ultimately"),
    ("on the other hand", "then again"),
    ("by and large", "mostly"),
    ("a wide range of", "many different"),
    ("a diverse range of", "all sorts of"),
    ("plays a crucial role", "matters a lot"),
    ("plays a pivotal role", "is really important"),
    ("plays a key role", "is a big deal"),
    ("are not merely", "aren't just"),
    ("is not merely", "isn't just"),
    ("pave the way for", "open doors for"),
    ("the way for a more", "the door to a more"),
    ("across numerous", "across many"),
    ("across various", "in different"),
]


def _case_preserving_replace(text: str, pattern: str, replacement: str) -> str:
    """Replace pattern with replacement, preserving case of first char."""
    def _replace_match(m: re.Match) -> str:
        original = m.group()
        if original.isupper():
            return replacement.upper()
        elif original[0].isupper():
            return replacement[0].upper() + replacement[1:] if replacement else ""
        return replacement

    return re.sub(pattern, _replace_match, text, flags=re.IGNORECASE)


def pass_transitions(text: str) -> str:
    """Pass 2: Replace formal transitions with casual alternatives.

    Preserves case of the first character.
    """
    result = text
    for from_phrase, to_phrase in _TRANSITIONS:
        # Escape for regex but keep case-insensitive matching
        pattern = re.escape(from_phrase)
        result = _case_preserving_replace(result, pattern, to_phrase)
    return result


# ═══════════════════════════════════════════════════════════════
# PASS 3: CONTRACTION INJECTION
# ═══════════════════════════════════════════════════════════════

_CONTRACTIONS: list[tuple[str, str]] = [
    ("It is not", "It's not"),
    ("it is not", "it's not"),
    ("do not", "don't"),
    ("does not", "doesn't"),
    ("did not", "didn't"),
    ("is not", "isn't"),
    ("are not", "aren't"),
    ("was not", "wasn't"),
    ("were not", "weren't"),
    ("will not", "won't"),
    ("would not", "wouldn't"),
    ("could not", "couldn't"),
    ("should not", "shouldn't"),
    ("cannot", "can't"),
    ("can not", "can't"),
    ("it is", "it's"),
    ("that is", "that's"),
    ("there is", "there's"),
    ("they are", "they're"),
    ("we are", "we're"),
    ("you are", "you're"),
    ("I am", "I'm"),
    ("I have", "I've"),
    ("I will", "I'll"),
    ("it will", "it'll"),
    ("they will", "they'll"),
    ("we will", "we'll"),
    ("who is", "who's"),
    ("what is", "what's"),
    ("let us", "let's"),
    ("Do not", "Don't"),
    ("Does not", "Doesn't"),
    ("Did not", "Didn't"),
    ("Is not", "Isn't"),
    ("Are not", "Aren't"),
    ("Was not", "Wasn't"),
    ("Will not", "Won't"),
    ("Would not", "Wouldn't"),
    ("Could not", "Couldn't"),
    ("Should not", "Shouldn't"),
    ("Cannot", "Can't"),
    ("It is", "It's"),
    ("That is", "That's"),
    ("There is", "There's"),
    ("They are", "They're"),
    ("We are", "We're"),
    ("You are", "You're"),
    ("We will", "We'll"),
    ("They will", "They'll"),
    ("It will", "It'll"),
    ("Who is", "Who's"),
    ("What is", "What's"),
    ("Let us", "Let's"),
]


def pass_contractions(text: str) -> str:
    """Pass 3: Inject contractions for natural speech."""
    result = text
    for from_phrase, to_phrase in _CONTRACTIONS:
        result = result.replace(from_phrase, to_phrase)
    return result


# ═══════════════════════════════════════════════════════════════
# PASS 4: BURSTINESS INJECTION (fixed merge index)
# ═══════════════════════════════════════════════════════════════

_SENTENCE_RE = re.compile(r"[.!?]+")


def pass_burstiness(text: str, rng: LcgRng) -> str:
    """Pass 4: Break long sentences, merge short ones for natural rhythm."""
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    if len(sentences) < 3:
        return text

    result = []
    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)

        # Split very long sentences (>25 words)
        if word_count > 25:
            split_at = word_count // 2
            # Try to find a comma near the middle
            for i in range(word_count // 3, (2 * word_count) // 3):
                if i < len(words) and words[i].endswith(","):
                    split_at = i + 1
                    break

            first_half = " ".join(words[:split_at]).rstrip(",")
            second_half = " ".join(words[split_at:])
            if second_half:
                second_half = second_half[0].upper() + second_half[1:]
            result.append(first_half)
            result.append(second_half)
        else:
            result.append(sentence)

    # Randomly merge short adjacent sentences (fixed: use i % 3 instead of i % 2)
    final = []
    i = 0
    while i < len(result):
        word_count = len(result[i].split())
        if word_count < 10 and i + 1 < len(result) and rng.next_float() < 0.3:
            connector = " — " if rng.next_float() < 0.5 else "; "
            merged = result[i] + connector + result[i + 1][0].lower() + result[i + 1][1:]
            final.append(merged)
            i += 2
        else:
            final.append(result[i])
            i += 1

    return ". ".join(final) + "." if final else text


# ═══════════════════════════════════════════════════════════════
# PASS 5: HUMAN FILLER INJECTION
# ═══════════════════════════════════════════════════════════════

_FILLERS = [
    "Honestly, ",
    "The thing is, ",
    "What's interesting is that ",
    "If you think about it, ",
    "Look, ",
    "Here's the deal: ",
    "To put it simply, ",
    "In simple terms, ",
    "The reality is, ",
    "At the end of the day, ",
    "Truth be told, ",
    "When you break it down, ",
]


def pass_fillers(text: str, rng: LcgRng) -> str:
    """Pass 5: Inject casual filler phrases for human feel."""
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    if len(sentences) < 3:
        return text

    result = []
    filler_count = 0
    max_fillers = 3 if len(sentences) > 6 else 2

    for i, sentence in enumerate(sentences):
        if i > 0 and i % 3 == 0 and filler_count < max_fillers and rng.next_float() < 0.45:
            filler = rng.pick(_FILLERS)
            lower = sentence[0].lower() + sentence[1:] if sentence else ""
            result.append(filler + lower)
            filler_count += 1
        else:
            result.append(sentence)

    return ". ".join(result) + "." if result else text


# ═══════════════════════════════════════════════════════════════
# PASS 6: AI PADDING STRIP
# ═══════════════════════════════════════════════════════════════

_AI_PADDING: list[tuple[str, str]] = [
    ("It is important to understand that ", ""),
    ("It is crucial to recognize that ", ""),
    ("It is essential to acknowledge that ", ""),
    ("It is noteworthy that ", ""),
    ("It goes without saying that ", ""),
    ("Needless to say, ", ""),
    ("As we all know, ", ""),
    ("As previously mentioned, ", ""),
    ("In conclusion, ", "So, "),
    ("To summarize, ", "Basically, "),
    ("In summary, ", "So basically, "),
    ("All in all, ", "Overall, "),
    ("Taking everything into account, ", "All things considered, "),
    ("Given the above, ", "With all that, "),
]


def pass_strip_ai_padding(text: str) -> str:
    """Pass 6: Remove AI-typical padding phrases."""
    result = text
    for from_phrase, to_phrase in _AI_PADDING:
        result = result.replace(from_phrase, to_phrase)
    return result


# ═══════════════════════════════════════════════════════════════
# PASS 7: HOMOGLYPH PERTURBATION (with ZWNJ)
# ═══════════════════════════════════════════════════════════════

# Cyrillic homoglyphs that look like Latin letters
_HOMOGLYPHS = {
    "a": "а",  # U+0430
    "c": "с",  # U+0441
    "e": "е",  # U+0435
    "o": "о",  # U+043E
    "p": "р",  # U+0440
    "x": "х",  # U+0445
    "y": "у",  # U+0443
    "A": "А",  # U+0410
    "C": "С",  # U+0421
    "E": "Е",  # U+0415
    "O": "О",  # U+041E
    "P": "Р",  # U+0420
    "X": "Х",  # U+0425
}


def apply_homoglyphs(text: str) -> str:
    """Pass 7: Inject Cyrillic homoglyphs and ZWNJ to break tokenizers.

    Matches Rust implementation: 15% homoglyph swap, 5% ZWNJ injection.
    Deterministic based on text length.
    """
    rng = LcgRng.from_seed(len(text) ^ 0xCAFE)
    result = []

    for c in text:
        # 15% chance to swap with Cyrillic homoglyph
        if c in _HOMOGLYPHS and rng.next_float() < 0.15:
            result.append(_HOMOGLYPHS[c])
        else:
            result.append(c)

        # 5% chance to inject invisible zero-width non-joiner
        if c != " " and rng.next_float() < 0.05:
            result.append("‌")

    return "".join(result)


# ═══════════════════════════════════════════════════════════════
# SHATTER SYNTHID (Master Function)
# ═══════════════════════════════════════════════════════════════

def shatter_synthid_text(input_text: str) -> str:
    """Master: multi-pass statistical humanizer to destroy AI watermarks.

    Combines all 7 passes to defeat SynthID, perplexity-based detectors,
    and other AI-generated content classifiers.

    WARNING: This MODIFIES the text (non-lossless). It will change:
    - Word choices (synonym replacement)
    - Sentence structure (burstiness)
    - Contractions
    - Homoglyph characters (invisible)

    Returns cleaned text.
    """
    paragraphs = input_text.split("\n\n")
    processed = []

    for p in paragraphs:
        if not p.strip():
            processed.append(p)
            continue

        seed = len(p) ^ 0xDEAD
        rng = LcgRng.from_seed(seed)

        text = pass_synonyms(p, rng)
        text = pass_transitions(text)
        text = pass_contractions(text)
        text = pass_burstiness(text, rng)
        text = pass_fillers(text, rng)
        text = pass_strip_ai_padding(text)

        # Finally, strip invisible unicode and inject homoglyphs
        # FIX: scrub_unicode(aggressive=True) calls apply_homoglyphs internally
        # so we must NOT call apply_homoglyphs again to avoid double injection
        text = scrub_unicode(text, aggressive=True)

        processed.append(text)

    return "\n\n".join(processed)
