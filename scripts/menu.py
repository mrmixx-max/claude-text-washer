#!/usr/bin/env python3
"""Claude Text Washer — Interactive TUI Menu (Textual-based)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
    from textual.app import App, ComposeResult
    from textual.containers import Container, Header, Footer
    from textual.widgets import Button, Static
    from textual.css.query import NoMatches
    from textual import events
    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MENU_ITEMS = [
    ("editor", "Text Editor", "Edit text with live AI score", "primary"),
    ("analyze", "Analyze Text", "Run statistical watermark analysis", "default"),
    ("wash", "Wash Text", "Full text wash pipeline", "success"),
    ("wash_file", "Wash File", "Strip metadata from any file format", "success"),
    ("prompt", "Prompt Editor", "Build anti-watermark prompts", "warning"),
    ("benchmark", "Benchmark", "Compare AI vs human text metrics", "default"),
    ("settings", "Settings", "Model selection & configuration", "primary"),
]


def run_script(name: str) -> None:
    """Run a claude-text-washer script."""
    script_map = {
        "editor": "editor.py",
        "analyze": "stat_engine.py",
        "wash": "pipeline.py",
        "wash_file": "file_washer.py",
        "prompt": "prompt_editor.py",
        "benchmark": "bench.py",
    }
    script = script_map.get(name)
    if not script:
        return
    path = SCRIPT_DIR / script
    if not path.exists():
        print(f"[NOT FOUND] {path}")
        input("Press Enter to return...")
        return
    try:
        subprocess.run([sys.executable, str(path)], cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        pass


if HAS_TEXTUAL:
    class WasherApp(App):
        """Claude Text Washer TUI."""

        TITLE = "claude-text-washer"

        CSS = """
        Screen {
            layout: grid;
            grid-size: 1;
            grid-rows: auto 1fr auto;
        }
        #header {
            height: 5;
            background: $primary;
            text-align: center;
            content-align: center middle;
        }
        #title {
            text-style: bold;
            color: $text;
        }
        #subtitle {
            color: $text-muted;
        }
        #menu {
            layout: vertical;
            align: center middle;
            padding: 2;
        }
        Button {
            width: 50;
            margin: 1;
            text-align: center;
        }
        #footer {
            height: 3;
            background: $surface;
            text-align: center;
            content-align: center middle;
            color: $text-muted;
        }
        """

        def compose(self) -> ComposeResult:
            yield Header(show_clock=False)
            with Container(id="header"):
                yield Static("CLAUDE TEXT WASHER", id="title")
                yield Static("AI Watermark Detection & Removal  •  [dim]v2.0[/dim]", id="subtitle")
            with Container(id="menu"):
                for key, label, desc, variant in MENU_ITEMS:
                    yield Button(f"{label}\n[dim]{desc}[/dim]", id=key, variant=variant)
                yield Button("Exit", id="exit", variant="error")
            yield Footer("↑↓ navigate  •  Enter select  •  q quit")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            button_id = event.button.id
            if button_id == "exit":
                self.exit()
            else:
                run_script(button_id)

        def on_key(self, event: events.Key) -> None:
            if event.key in ("q", "escape"):
                self.exit()

else:
    # Fallback: simple ANSI menu
    class WasherApp:
        def __init__(self) -> None:
            self.running = True

        def run(self) -> None:
            while self.running:
                self._draw()
                try:
                    key = input("Select (1-8): ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                try:
                    idx = int(key) - 1
                except ValueError:
                    continue
                if idx == 7:
                    break
                if 0 <= idx < len(MENU_ITEMS):
                    run_script(MENU_ITEMS[idx][0])

        def _draw(self) -> None:
            print("\033[2J\033[H")  # Clear screen
            print("=" * 60)
            print("  ██╗    ██╗ █████╗ ███████╗██╗  ██╗███████╗██████╗ ")
            print("  ██║    ██║██╔══██╗██╔════╝██║  ██║██╔════╝██╔══██╗")
            print("  ██║ █╗ ██║███████║███████╗███████║█████╗  ██████╔╝")
            print("  ██║███╗██║██╔══██║╚════██║██╔══██║██╔══╝  ██╔══██╗")
            print("  ╚███╔███╔╝██║  ██║███████║██║  ██║███████╗██║  ██║")
            print("   ╚══╝╚══╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝")
            print("=" * 60)
            print()
            for i, (key, label, desc, _) in enumerate(MENU_ITEMS, 1):
                print(f"  [{i}] {label:20s}  {desc}")
            print(f"  [8] Exit")
            print()
            print("-" * 60)


def main() -> None:
    """Entry point."""
    if HAS_TEXTUAL:
        app = WasherApp()
        app.run()
    else:
        print("[INFO] textual not installed — using fallback menu")
        app = WasherApp()
        app.run()


if __name__ == "__main__":
    main()
