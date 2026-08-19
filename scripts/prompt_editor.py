#!/usr/bin/env python3
"""Interactive prompt editor for claude-text-washer.

Lets the user:
- Select a built-in anti-watermark template
- Edit the prompt text in real-time (line editor)
- Preview the generated full prompt
- Send the current text directly to a local Ollama model
- Save a prompt template to a file

Pure functions (:func:`render_prompt`, :class:`PromptTemplate`) are
unit-testable without a terminal.  The interactive loop lives in
:func:`run_prompt_editor`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Reuse editor's platform key helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))
from editor import ANSI, getch, classify_key  # noqa: E402
from stat_engine import generate_anti_watermark_prompt  # noqa: E402

# Reuse Ollama helper
try:
    from ollama_utils import call_ollama, get_default_model, get_model_names
except Exception:  # pragma: no cover
    call_ollama = None
    get_default_model = lambda: "llama3.2"  # noqa: E731
    get_model_names = lambda: ["llama3.2"]  # noqa: E731

_IS_WINDOWS = os.name == "nt"

if _IS_WINDOWS:
    try:
        import msvcrt as _msvcrt  # type: ignore[import-not-found]
    except ImportError:
        _msvcrt = None
else:
    _msvcrt = None


# --------------------------------------------------------------------------- #
# Template definitions
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PromptTemplate:
    """A reusable prompt template with a strategy header and body."""
    name: str
    description: str
    system: str
    body: str

    def render(self, user_text: str) -> str:
        """Substitute *user_text* into the body and return the full prompt."""
        return self.body.format(text=user_text) if "{text}" in self.body else self.body + "\n\n" + user_text


# The six built-in templates the task asked for.
TEMPLATES: list[PromptTemplate] = [
    PromptTemplate(
        name="Green-list Bias Breaker",
        description="Counter vocabulary bias from green-list watermarking",
        system="Du bist ein Textredaktor. Entferne alle KI-typischen Wortwahl-Muster.",
        body="""Entferne den Text von Green-List-Wasserzeichen-Vorhersagen.
Vermeide vorhersgbare Stopwörter und Funktionswörter.
Ersetze häufige Wortpaare durch unerwartete Alternativen.
Gib NUR den umgeschriebenen Text zurück, nichts andere.

Originaltext:
{text}""",
    ),
    PromptTemplate(
        name="N-gram Repetition Breaker",
        description="Break repetitive n-gram patterns from AI output",
        system="Du bist ein stilistischer Redakteur. Zerschlage Wiederholungsmuster.",
        body="""Zerstöre wiederkehrende N-Gramm-Muster im folgenden Text.
Jedes zweite und dritte Wort muss anders stehen als vorher.
Vermeide Sequenzen von 2+ wiederkehrenden Wörtern.
Behalte die Bedeutung bei.

Originaltext:
{text}""",
    ),
    PromptTemplate(
        name="SynthID-class Counter",
        description="Counteract semantic-style watermark signatures",
        system="Du bist ein Ghostwriter für natürliche Prosa.",
        body="""Dieser Text trägt möglicherweise SynthID- oder statistische Wasserzeichen.
Schreibe ihn um mit unregelmäßiger Satzstruktur, unterschiedlicher Satzlänge
und persönlichem Ton. Keine Standardformulierungen.

Originaltext:
{text}""",
    ),
    PromptTemplate(
        name="Perplexity Booster",
        description="Increase unpredictability and vocabulary richness",
        system="Du bist ein kreativer Autor mit unkonventioneller Sprache.",
        body="""Erhöhe die Perplexität des folgenden Textes.
Nutze selteneres Vokabular, unkonventionelle Metaphern,
und unerwartete Satzanfänge. Die Aussage bleibt gleich.

Originaltext:
{text}""",
    ),
    PromptTemplate(
        name="Burstiness Injector",
        description="Mix short punches with long flowing sentences",
        system="Du schreibst gehackte, unregelmäßige Prosa.",
        body="""Maximiere Burstiness: mische 1-3 Wort-Sätze mit langen,
asymmetrischen Sätzen. Unterbreche den Fluss absichtlich.
Keine gleichmäßigen Satzlängen.

Originaltext:
{text}""",
    ),
    PromptTemplate(
        name="Custom",
        description="Editable blank template",
        system="Du bist ein erfahrener Textredaktor.",
        body="""Bitte überarbeite den folgenden Text nach diesen Regeln:
{regeln}

Originaltext:
{text}""",
    ),
]


def template_names() -> list[str]:
    """Return the list of template names."""
    return [t.name for t in TEMPLATES]


def get_template(name: str) -> PromptTemplate | None:
    """Look up a template by name (case-insensitive)."""
    for t in TEMPLATES:
        if t.name.lower() == name.lower():
            return t
    return None


def all_templates() -> list[PromptTemplate]:
    """Return all built-in templates."""
    return list(TEMPLATES)


# --------------------------------------------------------------------------- #
# Pure rendering — testable without a terminal
# --------------------------------------------------------------------------- #

def render_prompt(
    template: PromptTemplate,
    user_text: str,
    rules: str = "",
) -> str:
    """Render the full prompt string for *template* + *user_text*.

    For the ``Custom`` template, *rules* replaces the ``{regeln}`` placeholder.
    """
    body = template.body
    if "{regeln}" in body and "{text}" in body:
        return body.format(regeln=rules, text=user_text)
    if "{text}" in body:
        return body.format(text=user_text)
    if "{regeln}" in body:
        return body.format(regeln=rules)
    return body + "\n\n" + user_text


def validate_template_dict(data: dict[str, Any]) -> list[str]:
    """Validate a user-supplied template dict. Returns list of errors."""
    errors: list[str] = []
    for key in ("name", "description", "system", "body"):
        if key not in data or not isinstance(data[key], str) or not data[key]:
            errors.append(f"Missing or empty '{key}'")
    return errors


def template_to_dict(t: PromptTemplate) -> dict[str, Any]:
    return {
        "name": t.name,
        "description": t.description,
        "system": t.system,
        "body": t.body,
    }


def template_from_dict(d: dict[str, Any]) -> PromptTemplate:
    return PromptTemplate(
        name=str(d["name"]),
        description=str(d["description"]),
        system=str(d["system"]),
        body=str(d["body"]),
    )


# --------------------------------------------------------------------------- #
# Editor state model — testable core
# --------------------------------------------------------------------------- #

@dataclass
class PromptEditorState:
    """Mutable state for the prompt editor (the "model" in MVC)."""
    template: PromptTemplate = field(default_factory=lambda: get_template("Custom") or TEMPLATES[0])
    user_text: str = ""
    rules: str = ""
    model: str = "llama3.2"
    temperature: float = 0.8

    @property
    def rendered(self) -> str:
        return render_prompt(self.template, self.user_text, self.rules)

    @property
    def template_index(self) -> int:
        try:
            return TEMPLATES.index(self.template)
        except ValueError:
            return 0

    def cycle_template(self, direction: int = 1) -> None:
        idx = (self.template_index + direction) % len(TEMPLATES)
        self.template = TEMPLATES[idx]

    def select_template(self, index: int) -> bool:
        if 0 <= index < len(TEMPLATES):
            self.template = TEMPLATES[index]
            return True
        return False

    def to_json() -> str:
        pass

    def export_template(self) -> str:
        """Return the current template + rules as a JSON string."""
        return json.dumps({
            "name": self.template.name,
            "description": self.template.description,
            "system": self.template.system,
            "body": self.template.body,
            "rules": self.rules,
        }, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Ollama dispatch (kept thin so it can be mocked in tests)
# --------------------------------------------------------------------------- #

def send_to_ollama(state: PromptEditorState) -> str:
    """Send the rendered prompt to Ollama and return the response.

    Raises ``RuntimeError`` if no Ollama connection or if ``call_ollama``
    is unavailable.
    """
    if call_ollama is None:
        raise RuntimeError("Ollama utilities not available")
    payload = state.rendered
    response = call_ollama(
        prompt=payload,
        model=state.model,
        system_prompt=state.template.system,
        temperature=state.temperature,
        max_tokens=2048,
    )
    return response


def save_template(state: PromptEditorState, path: str | Path) -> None:
    """Persist the current template + rules to *path* as JSON."""
    data = {
        "name": state.template.name,
        "description": state.template.description,
        "system": state.template.system,
        "body": state.template.body,
        "rules": state.rules,
    }
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Simple line-editor buffer (single-line text input for prompt editing)
# --------------------------------------------------------------------------- #

def edit_line(label: str, default: str = "") -> str:
    """Interactive single-line editor. Returns the entered string.

    Uses readline-style editing with backspace (Ctrl+H / DEL),
    Enter to confirm, Ctrl+C to cancel (returns *default*).
    """
    line = default
    print(f"{ANSI.CYAN}{label}{ANSI.RESET} ", end="", flush=True)
    _print_inline_value(line)
    while True:
        token = getch()
        kind, payload = classify_key(token)
        if kind == "char":
            line += payload
            _print_inline_value(line)
        elif kind == "action" and payload == "backspace":
            line = line[:-1]
            _print_inline_value(line)
        elif kind == "action" and payload == "newline":
            print()
            return line
        elif token.lower() == "c" and _IS_WINDOWS:
            # Ctrl+C handling
            pass
        # Ctrl+C / Ctrl+G aborts
        if token == "\x03" or token == "\x07":
            print()
            return default
    return line


def _print_inline_value(line: str) -> None:
    """Redraw the current line value."""
    print(f"\r{ANSI.CYAN}Edit: {ANSI.RESET}{line}", end="", flush=True)


# --------------------------------------------------------------------------- #
# Terminal menu helpers
# --------------------------------------------------------------------------- #

def selection_menu(title: str, options: list[str], selected: int = 0) -> int:
    """Render a selectable vertical menu in the terminal.

    Returns the index of the selected option.  Uses arrow keys / Enter.
    """
    import shutil

    width, height = shutil.get_terminal_size((80, 24))
    # Move cursor to bottom area
    start_line = 4
    print(f"\x1b[{start_line};1H", end="", flush=True)
    print(f"{ANSI.BOLD}{ANSI.CYAN}=== {title} ==={ANSI.RESET}", flush=True)
    print(f"{ANSI.DIM}↑/↓ navigate · Enter select · Ctrl+C cancel{ANSI.RESET}", flush=True)
    print(flush=True)
    while True:
        for i, opt in enumerate(options):
            if i == selected:
                print(f"\r{ANSI.BG_BLUE}{ANSI.BOLD}→ {opt:<50}{ANSI.RESET}", flush=True)
            else:
                print(f"\r  {opt:<50}", flush=True)
        token = getch()
        kind, payload = classify_key(token)
        if kind == "action":
            if payload == "up":
                selected = (selected - 1) % len(options)
            elif payload == "down":
                selected = (selected + 1) % len(options)
            elif payload == "quit" and token == "\x03":
                return -1
        if token == "\r" or token == "\n":
            return selected
        # redraw
        print(f"\x1b[{start_line + 4};1H", end="", flush=True)


# --------------------------------------------------------------------------- #
# Main interactive run loop
# --------------------------------------------------------------------------- #

def run_prompt_editor(
    initial_text: str = "",
    initial_template: int = 5,  # "Custom"
    output_path: str | Path | None = None,
) -> PromptEditorState:
    """Run the interactive prompt editor loop.

    Press:
      - Tab            : select a template from the menu
      - Ctrl+E         : edit the user text
      - Ctrl+R         : edit the rules (Custom template)
      - Ctrl+P         : preview the full rendered prompt
      - Ctrl+O         : send to Ollama
      - Ctrl+S         : save template to file
      - Ctrl+Q         : quit
    """
    state = PromptEditorState(
        template=TEMPLATES[initial_template],
        user_text=initial_text,
    )
    # Resolve default model
    try:
        state.model = get_default_model()
    except Exception:
        pass

    print("\x1b[?25l", end="", flush=True)
    if _IS_WINDOWS:
        os.system("")
    try:
        while True:
            _render_prompt_screen(state)
            token = getch()
            kind, payload = classify_key(token)
            action: str | None = None
            if kind == "action":
                action = payload
            elif token == "\x05":  # Ctrl+E
                action = "edit_text"
            elif token == "\x12":  # Ctrl+R
                action = "edit_rules"
            elif token == "\x10":  # Ctrl+P
                action = "preview"
            elif token == "\x0f":  # Ctrl+O
                action = "ollama"
            elif token == "\x13":  # Ctrl+S
                action = "save"
            elif token == "\x11":  # Ctrl+Q
                action = "quit"
            elif token == "\t":
                action = "templates"

            if action == "templates":
                _template_picker(state)
            elif action == "edit_text":
                state.user_text = _multiline_edit("Prompt text", state.user_text)
            elif action == "edit_rules":
                state.rules = _multiline_edit("Rules", state.rules)
            elif action == "preview":
                _preview(state.rendered)
            elif action == "ollama":
                _ollama_output(state)
            elif action == "save":
                path = output_path or "prompt_template.json"
                save_template(state, path)
                _flash_msg(f"Saved template → {path}")
            elif action == "quit":
                break
    finally:
        print("\x1b[?25h", end="", flush=True)
        os.system("")
        print("\n", end="", flush=True)

    return state


def _render_prompt_screen(state: PromptEditorState) -> None:
    print("\x1b[2J\x1b[H", end="", flush=True)
    print(f"{ANSI.BOLD}{ANSI.MAGENTA}"
          r"""
  ____            _    _               _                    
 |  _ \ _ __ __ _| | _(_)_ __   __ _  ___|___   _ __ ___   __ _ 
 | |_) | '__/ _` | |/ / | '_ \ / _` |/ _ \___| | '_ ` _ \ / _` |
 |  __/| | | (_| |   <| | | | | | (_| |  __/___|_| | | | | | (_| |
 |_|   |_|  \__,_|_|\_\_|_| |_|\__,_|\___|         |_|_| |_|_|__, |
                                                             |___/ 
"""
          f"{ANSI.RESET}")
    print(f"{ANSI.CYAN}{ANSI.BOLD}── Prompt Editor ──{ANSI.RESET}", flush=True)
    print(f"Template: {ANSI.YELLOW}{state.template.name}{ANSI.RESET}  "
          f"({state.template_index + 1}/{len(TEMPLATES)})", flush=True)
    print(f"Model:    {ANSI.GREEN}{state.model}{ANSI.RESET}  "
          f"Temp: {state.temperature}", flush=True)
    print(f"{ANSI.DIM}─" * 70 + ANSI.RESET, flush=True)
    print(f"{ANSI.BOLD}User text:{ANSI.RESET}", flush=True)
    for line in state.user_text.split("\n") or [""]:
        print(f"  {line}", flush=True)
    print(f"{ANSI.DIM}─" * 70 + ANSI.RESET, flush=True)
    print(f"{ANSI.BOLD}Rules:{ANSI.RESET} {state.rules or '(none)'}", flush=True)
    print(f"{ANSI.DIM}─" * 70 + ANSI.RESET, flush=True)
    # Show first 300 chars of rendered
    rendered = state.rendered
    preview = rendered[:300] + ("…" if len(rendered) > 300 else "")
    print(f"{ANSI.BOLD}Preview (first 300 chars):{ANSI.RESET}", flush=True)
    print(f"{ANSI.YELLOW}{preview}{ANSI.RESET}", flush=True)
    print(f"\n{ANSI.DIM}"
          "Keys: Tab=template  Ctrl+E=edit text  Ctrl+R=rules  "
          "Ctrl+P=preview  Ctrl+O=ollama  Ctrl+S=save  Ctrl+Q=quit"
          f"{ANSI.RESET}", flush=True)


def _template_picker(state: PromptEditorState) -> None:
    """Show a menu to pick a template (simple inline rendering)."""
    print("\x1b[2J\x1b[H", end="", flush=True)
    print(f"{ANSI.BOLD}{ANSI.CYAN}=== Template Selection ==={ANSI.RESET}", flush=True)
    for i, t in enumerate(TEMPLATES):
        marker = f"{ANSI.GREEN}→{ANSI.RESET}" if i == state.template_index else " "
        print(f"  {marker} {i + 1}. {ANSI.BOLD}{t.name}{ANSI.RESET} — {t.description}", flush=True)
    print(f"\n{ANSI.DIM}↑/↓ or 1-{len(TEMPLATES)} select · Enter confirm · Ctrl+C cancel{ANSI.RESET}", flush=True)
    while True:
        token = getch()
        kind, payload = classify_key(token)
        if kind == "action":
            if payload == "up":
                state.select_template((state.template_index - 1) % len(TEMPLATES))
                _redraw_template_picker(state)
            elif payload == "down":
                state.select_template((state.template_index + 1) % len(TEMPLATES))
                _redraw_template_picker(state)
        if token == "\r" or token == "\n":
            return
        if token == "\x03":
            return


def _redraw_template_picker(state: PromptEditorState) -> None:
    print("\x1b[2J\x1b[H", end="", flush=True)
    print(f"{ANSI.BOLD}{ANSI.CYAN}=== Template Selection ==={ANSI.RESET}", flush=True)
    for i, t in enumerate(TEMPLATES):
        marker = f"{ANSI.GREEN}→{ANSI.RESET}" if i == state.template_index else " "
        print(f"  {marker} {i + 1}. {ANSI.BOLD}{t.name}{ANSI.RESET} — {t.description}", flush=True)


def _multiline_edit(label: str, initial: str) -> str:
    """Simple full-screen multiline editor for prompt/rules text.

    Ctrl+S saves and returns.  Ctrl+Q cancels (returns unchanged).
    Arrow keys navigate.  This reuses TextBuffer from editor.py.
    """
    from editor import TextBuffer, run_editor
    buf = TextBuffer.from_text(initial)
    # We run a minimal local loop instead of the full run_editor so we
    # can return the text without saving to disk.
    import shutil
    width, height = shutil.get_terminal_size((80, 24))
    buf.wrap_width = width
    finished = {"done": False, "saved": False}

    print("\x1b[2J\x1b[H", end="", flush=True)
    print(f"{ANSI.BOLD}{ANSI.CYAN}── Editing: {label} ──{ANSI.RESET}", flush=True)
    print(f"{ANSI.DIM}Ctrl+S save  Ctrl+Q cancel{ANSI.RESET}", flush=True)
    print(f"{ANSI.DIM}─" * 70 + ANSI.RESET, flush=True)

    while not finished["done"]:
        _render_edit_buffer(buf, width, label)
        token = getch()
        kind, payload = classify_key(token)
        if kind == "char":
            buf.insert(payload)
        elif kind == "action":
            avail = max(1, height - 5)
            if payload == "save":
                finished["done"] = True
            elif payload == "quit":
                finished["done"] = True
                finished["saved"] = False
            elif payload == "newline":
                buf.insert("\n")
            elif payload == "tab":
                buf.insert("    ")
            elif payload == "backspace":
                buf.backspace()
            elif payload == "delete":
                buf.delete_char()
            elif payload == "left":
                buf.move_left()
            elif payload == "right":
                buf.move_right()
            elif payload == "up":
                buf.move_up()
            elif payload == "down":
                buf.move_down()
    return buf.to_text()


def _render_edit_buffer(buf, width: int, label: str) -> None:
    """Draw a TextBuffer in the multiline editor (no status bar needed)."""
    print("\x1b[2J\x1b[H", end="", flush=True)
    print(f"{ANSI.BOLD}{ANSI.CYAN}── Editing: {label} ──{ANSI.RESET}", flush=True)
    print(f"{ANSI.DIM}Ctrl+S save  Ctrl+Q cancel{ANSI.RESET}", flush=True)
    print(f"{ANSI.DIM}─" * 70 + ANSI.RESET, flush=True)
    lines = buf.visible_lines(width)
    for i, line in enumerate(lines):
        print(f"  {line}", flush=True)
    # Simple cursor indicator
    print(f"\n{ANSI.YELLOW}Ln {buf.cursor.row + 1}, Col {buf.cursor.col + 1}  "
          f"Chars {buf.char_count}{ANSI.RESET}", flush=True)


def _preview(text: str) -> None:
    """Display the full rendered prompt, press any key to return."""
    import shutil
    width, height = shutil.get_terminal_size((80, 24))
    print("\x1b[2J\x1b[H", end="", flush=True)
    print(f"{ANSI.BOLD}{ANSI.CYAN}── Full Prompt Preview ──{ANSI.RESET}", flush=True)
    print(f"{ANSI.DIM}─" * 70 + ANSI.RESET, flush=True)
    for line in text.split("\n"):
        # Soft-wrap long lines
        if len(line) > width:
            parts = [line[i:i + width] for i in range(0, len(line), width)]
            for p in parts:
                print(p, flush=True)
        else:
            print(line, flush=True)
    print(f"\n{ANSI.DIM}─" * 70 + ANSI.RESET, flush=True)
    print(f"{ANSI.DIM}Press any key to return...{ANSI.RESET}", flush=True)
    getch()


def _ollama_output(state: PromptEditorState) -> None:
    """Send to Ollama and display the result."""
    print("\x1b[2J\x1b[H", end="", flush=True)
    print(f"{ANSI.BOLD}{ANSI.GREEN}Sending to Ollama ({state.model}, temp={state.temperature})...{ANSI.RESET}", flush=True)
    sys.stdout.flush()
    try:
        result = send_to_ollama(state)
        print(f"\r{ANSI.BOLD}── Ollama Response ──{ANSI.RESET}", flush=True)
        for line in result.split("\n"):
            print(f"  {line}", flush=True)
    except RuntimeError as e:
        print(f"{ANSI.RED}Error: {e}{ANSI.RESET}", flush=True)
    print(f"\n{ANSI.DIM}Press any key to return...{ANSI.RESET}", flush=True)
    getch()


def _flash_msg(msg: str) -> None:
    print(f"{ANSI.GREEN}{msg}{ANSI.RESET}", flush=True)
    import time
    time.sleep(1)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Interactive prompt editor for AI-text washing")
    p.add_argument("--text", "-t", default="", help="Initial user text")
    p.add_argument("--template", type=int, default=5,
                   help=f"Initial template index (1-{len(TEMPLATES)}, default 6=Custom)")
    p.add_argument("--output", "-o", help="Save template to this file")
    p.add_argument("--model", default=None, help="Ollama model to use")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.model:
        # Validate / override
        try:
            state = run_prompt_editor(
                initial_text=args.text,
                initial_template=max(0, min(args.template - 1, len(TEMPLATES) - 1)),
                output_path=args.output,
            )
        except KeyboardInterrupt:
            return 0
    else:
        state = run_prompt_editor(
            initial_text=args.text,
            initial_template=max(0, min(args.template - 1, len(TEMPLATES) - 1)),
            output_path=args.output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
