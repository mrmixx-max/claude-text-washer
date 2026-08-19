#!/usr/bin/env python3
"""Claude Text Washer — Direct Chat with local Ollama models."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from scripts.stat_engine import analyze_text
from scripts.ollama_utils import get_model_names, get_model_config, handle_list_models, validate_model

# Wrapper for list_models compatibility
def list_models():
    """Return all models from the pool."""
    from scripts.ollama_utils import load_models
    return load_models().get("models", {})

# ANSI Colors
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

SYSTEM_PROMPT = """Du bist ein knallharter, menschlicher Lektor und Ghostwriter. Deine Aufgabe ist es, den übergebenen Text komplett neu zu verfassen und jegliche Muster von maschinell generierter Sprache restlos zu vernichten.

Halte dich an folgende absolute Restriktionen:
1. Burstiness maximieren: Wechsle radikal zwischen sehr kurzen, prägnanten Sätzen (1-4 Wörtern) und längeren, asymmetrischen Satzgefügen.
2. Perplexität erzwingen: Nutze unkonventionelle, treffende Verben. Vermeide vorhersehbare Adjektiv-Substantiv-Kombinationen.
3. Blacklist: Verwende NIEMALS Phrasen wie "Zusammenfassend lässt sich sagen", "Es ist wichtig zu beachten", "Ein weiteres Element" oder Wörter wie "facettenreich", "Geflecht", "Tapestry", "essenziell", "dynamisch".
4. Tonalität: Organisch, direkt und menschlich. Lass es leicht kantig klingen, als käme es aus der Feder eines erfahrenen Thriller-Autors. Keine weichgespülte Objektivität.
5. Output: Gib AUSSCHLIESSLICH den umgeschriebenen Text zurück. Keine Einleitungen, keine Erklärungen, keine Höflichkeitsfloskeln."""


def ollama_chat(model: str, messages: list[dict], system: str = SYSTEM_PROMPT) -> str:
    """Send chat messages to Ollama and return response."""
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": False,
        "options": {"temperature": 0.8, "num_predict": 2048},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("message", {}).get("content", "").strip()
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"Ollama error: {e}")


def ai_score_color(score: float) -> str:
    """Return ANSI color based on AI score."""
    if score <= 25:
        return GREEN
    elif score <= 50:
        return YELLOW
    else:
        return RED


def print_header(model: str) -> None:
    """Print ASCII art header."""
    print(f"""{CYAN}{BOLD}
 ╔═╗┬  ┌─┐┬ ┬┌┬┐┌─┐╔╦╗┌─┐┌─┐┬ ┬┌─┐┬─┐
 ║  │  ├─┤│ │ ││├╣  ║║║├─┤│  ├─┤├┤ ├┬┘
 ╚═╝┴─┘┴ ┴┴─┘─┴┘└─┘╩ ╩┴ ┴└─┘┴ ┴└─┘┴└─
{RESET}{DIM}  AI Watermark Detection & Removal  •  v2.0{RESET}
""")


def print_status(model: str, messages_count: int, last_score: float | None) -> None:
    """Print status bar."""
    score_str = f"{ai_score_color(last_score)}{last_score:.1f}{RESET}" if last_score is not None else "N/A"
    print(f"\n{DIM}[{RESET}{BOLD}{model}{RESET}{DIM}]{RESET}  "
          f"{DIM}msgs:{RESET}{messages_count}  "
          f"{DIM}last AI score:{RESET}{score_str}  "
          f"{DIM}/help for commands{RESET}")


def print_help() -> None:
    """Print available commands."""
    print(f"""
{BOLD}Commands:{RESET}
  {CYAN}/help{RESET}      Show this help
  {CYAN}/clear{RESET}     Clear chat history
  {CYAN}/model{RESET}     Switch model (shows available)
  {CYAN}/score{RESET}     Show AI score of last response
  {CYAN}/save{RESET}      Save chat to file
  {CYAN}/load{RESET}      Load chat from file
  {CYAN}/quit{RESET}      Exit chat
""")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Claude Text Washer — Chat with local Ollama")
    parser.add_argument("--model", default="llama3.2", help="Ollama model (default: llama3.2)")
    parser.add_argument("--system", default=SYSTEM_PROMPT, help="Custom system prompt")
    args = parser.parse_args(argv)

    model = args.model
    system = args.system
    history: list[dict] = []
    last_score: float | None = None

    print_header(model)
    print(f"{DIM}Model: {BOLD}{model}{RESET}{DIM}  •  Type /help for commands{RESET}\n")

    while True:
        try:
            user_input = input(f"{BLUE}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{YELLOW}Bye!{RESET}")
            break

        if not user_input:
            continue

        # Commands
        if user_input.startswith("/"):
            cmd = user_input.lower().strip()

            if cmd == "/quit" or cmd == "/q":
                print(f"{YELLOW}Bye!{RESET}")
                break

            elif cmd == "/help":
                print_help()
                continue

            elif cmd == "/clear":
                history = []
                last_score = None
                print(f"{YELLOW}Chat history cleared.{RESET}")
                continue

            elif cmd == "/model":
                print(f"\n{BOLD}Available models:{RESET}")
                for m, cfg in list_models().items():
                    marker = f"{GREEN}*{RESET}" if m == model else " "
                    print(f"  {marker} {m:15s} ({cfg.get('size', '?')}) {cfg.get('description', '')}")
                new_model = input(f"\n{DIM}Model name (Enter to keep {model}):{RESET} ").strip()
                if new_model:
                    try:
                        validate_model(new_model)
                        model = new_model
                        print(f"{GREEN}Switched to {model}{RESET}")
                    except ValueError:
                        print(f"{RED}Model not found.{RESET}")
                continue

            elif cmd == "/score":
                if last_score is not None:
                    color = ai_score_color(last_score)
                    print(f"{BOLD}Last AI Score:{RESET} {color}{last_score:.1f}/100{RESET}")
                else:
                    print(f"{DIM}No score yet.{RESET}")
                continue

            elif cmd == "/save":
                path = input(f"{DIM}Save to file:{RESET} ").strip()
                if path:
                    data = [{"role": m["role"], "content": m["content"]} for m in history]
                    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                    print(f"{GREEN}Saved {len(history)} messages to {path}{RESET}")
                continue

            elif cmd == "/load":
                path = input(f"{DIM}Load from file:{RESET} ").strip()
                if path and Path(path).exists():
                    data = json.loads(Path(path).read_text(encoding="utf-8"))
                    history = [m for m in data if m["role"] in ("user", "assistant")]
                    print(f"{GREEN}Loaded {len(history)} messages{RESET}")
                continue

            else:
                print(f"{RED}Unknown command: {user_input}{RESET}")
                continue

        # Normal message
        history.append({"role": "user", "content": user_input})

        try:
            print(f"\n{GREEN}Assistant:{RESET} ", end="", flush=True)
            response = ollama_chat(model, history, system)
            print(response)

            history.append({"role": "assistant", "content": response})

            # Live AI Score
            report = analyze_text(response)
            last_score = report.ai_score
            color = ai_score_color(last_score)
            print(f"\n{DIM}[AI Score: {color}{last_score:.1f}/100{RESET}{DIM}]{RESET}")

        except RuntimeError as e:
            print(f"{RED}Error: {e}{RESET}")
        except KeyboardInterrupt:
            print()

        print_status(model, len(history) // 2, last_score)


if __name__ == "__main__":
    main()
