#!/usr/bin/env python3
"""Claude Text Washer — Windows GUI

A desktop app that strips AI watermarks and rewrites text in organic,
asymmetrical prose. Supports any LLM backend (Ollama, vLLM, OpenRouter, etc.).
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

# ---- resource_path for PyInstaller ----
def resource_path(relative: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / relative

# ---- stdlib ----
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk

# ---- project imports (all stdlib, no external deps) ----
from stat_engine import analyze_text, format_report, generate_anti_watermark_prompt
from smart_cleaner import clean_text, get_marker_count
from ollama_utils import (
    call_llm, get_model_names, SYSTEM_PROMPT,
    SamplingConfig, DEFAULT_CFG,
    BackendConfig, set_backend, get_backend,
)
from pipeline import wash_multi_pass, wash_pass


# ============================================================
# Main Application (tkinter — always available)
# ============================================================

class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Claude Text Washer")
        self.geometry("1100x750")
        self.minsize(900, 600)
        self.configure(bg="#1e1e1e")

        self._setup_styles()
        self._build_ui()
        self._set_status("Ready")

    def _setup_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#1e1e1e", borderwidth=0)
        style.configure("TNotebook.Tab", background="#2d2d2d", foreground="white", padding=[12, 4])
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabel", background="#1e1e1e", foreground="white")
        style.configure("TButton", background="#3d3d3d", foreground="white")
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Sub.TLabel", font=("Segoe UI", 10), foreground="#aaaaaa")
        style.configure("Status.TLabel", font=("Segoe UI", 9), foreground="#888888")

    def _build_ui(self) -> None:
        # Header
        header = tk.Frame(self, bg="#1a1a2e", height=50)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="Claude Text Washer", font=("Segoe UI", 16, "bold"),
                 bg="#1a1a2e", fg="white").pack(side="left", padx=20, pady=10)
        tk.Label(header, text="Strip AI watermarks — Rewrite in organic prose",
                 font=("Segoe UI", 10), bg="#1a1a2e", fg="gray").pack(side="left", padx=10)

        # Notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(0, 5))

        self.tab_wash = ttk.Frame(self.notebook)
        self.tab_analyze = ttk.Frame(self.notebook)
        self.tab_settings = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_wash, text="  Wash  ")
        self.notebook.add(self.tab_analyze, text="  Analyze  ")
        self.notebook.add(self.tab_settings, text="  Settings  ")

        self._build_tab_wash()
        self._build_tab_analyze()
        self._build_tab_settings()

        # Status bar
        self.statusbar = tk.Label(self, text="", font=("Segoe UI", 9),
                                   fg="#888888", bg="#1a1a2e", anchor="w")
        self.statusbar.pack(fill="x", padx=15, pady=(0, 8))

    # ---- Tab: Wash ----
    def _build_tab_wash(self) -> None:
        self.tab_wash.columnconfigure(0, weight=1)
        self.tab_wash.columnconfigure(1, weight=1)
        self.tab_wash.rowconfigure(2, weight=1)

        # Controls row
        ctrl = ttk.Frame(self.tab_wash)
        ctrl.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

        ttk.Label(ctrl, text="Model:").pack(side="left", padx=(0, 5))
        self.model_var = tk.StringVar(value="llama3.2")
        models = get_model_names()
        self.model_combo = ttk.Combobox(ctrl, textvariable=self.model_var,
                                        values=models if models else ["llama3.2"],
                                        width=18, state="readonly")
        self.model_combo.pack(side="left", padx=(0, 15))

        ttk.Label(ctrl, text="Preset:").pack(side="left", padx=(0, 5))
        self.preset_var = tk.StringVar(value="standard")
        ttk.Combobox(ctrl, textvariable=self.preset_var,
                     values=["fast", "standard", "premium"],
                     width=12, state="readonly").pack(side="left", padx=(0, 15))

        ttk.Label(ctrl, text="Temp:").pack(side="left", padx=(0, 5))
        self.temp_var = tk.StringVar(value="0.7")
        ttk.Combobox(ctrl, textvariable=self.temp_var,
                     values=["0.3", "0.5", "0.7", "0.9", "1.0", "1.2"],
                     width=6, state="readonly").pack(side="left")

        # Labels
        ttk.Label(self.tab_wash, text="Input Text:", style="Sub.TLabel").grid(
            row=1, column=0, sticky="nw", padx=10)
        ttk.Label(self.tab_wash, text="Output:", style="Sub.TLabel").grid(
            row=1, column=1, sticky="nw", padx=(5, 10))

        # Text areas
        self.input_text = scrolledtext.ScrolledText(
            self.tab_wash, wrap="word", font=("Consolas", 11),
            bg="#0d1117", fg="white", insertbackground="white")
        self.input_text.grid(row=2, column=0, sticky="nsew", padx=(10, 5), pady=5)

        self.output_text = scrolledtext.ScrolledText(
            self.tab_wash, wrap="word", font=("Consolas", 11),
            bg="#1a1a2e", fg="white", insertbackground="white")
        self.output_text.grid(row=2, column=1, sticky="nsew", padx=(5, 10), pady=5)

        # Buttons
        btns = ttk.Frame(self.tab_wash)
        btns.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        ttk.Button(btns, text="Wash (Multi-Pass)", command=self._on_wash).pack(side="left", padx=5)
        ttk.Button(btns, text="Single Pass", command=self._on_single_pass).pack(side="left", padx=5)
        ttk.Button(btns, text="Analyze", command=self._on_analyze_only).pack(side="left", padx=5)
        ttk.Button(btns, text="Clear", command=self._on_clear).pack(side="left", padx=5)
        ttk.Button(btns, text="Load File...", command=self._on_load_file).pack(side="left", padx=5)
        ttk.Button(btns, text="Save Output...", command=self._on_save_output).pack(side="left", padx=5)

    # ---- Tab: Analyze ----
    def _build_tab_analyze(self) -> None:
        self.tab_analyze.columnconfigure(0, weight=1)
        self.tab_analyze.rowconfigure(1, weight=1)
        self.tab_analyze.rowconfigure(3, weight=1)

        ttk.Label(self.tab_analyze, text="Paste text for statistical AI-analysis:",
                  style="Sub.TLabel").grid(row=0, column=0, sticky="nw", padx=10, pady=10)

        self.analysis_input = scrolledtext.ScrolledText(
            self.tab_analyze, wrap="word", font=("Consolas", 11),
            bg="#0d1117", fg="white", insertbackground="white")
        self.analysis_input.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 5))

        btns = ttk.Frame(self.tab_analyze)
        btns.grid(row=2, column=0, sticky="w", padx=10, pady=5)
        ttk.Button(btns, text="Run Analysis", command=self._on_analyze).pack(side="left", padx=5)
        ttk.Button(btns, text="Generate Anti-WM Prompt", command=self._on_gen_prompt).pack(side="left", padx=5)

        self.analysis_output = scrolledtext.ScrolledText(
            self.tab_analyze, wrap="word", font=("Consolas", 10),
            bg="#0d1117", fg="#58a6ff", insertbackground="white")
        self.analysis_output.grid(row=3, column=0, sticky="nsew", padx=10, pady=(5, 10))

    # ---- Tab: Settings ----
    def _build_tab_settings(self) -> None:
        f = self.tab_settings
        f.columnconfigure(1, weight=1)

        # Backend section
        ttk.Label(f, text="LLM Backend", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=15, pady=(15, 5))

        ttk.Label(f, text="Base URL:").grid(row=1, column=0, sticky="w", padx=15, pady=5)
        self.base_url_var = tk.StringVar(value="http://127.0.0.1:11434/api/generate")
        ttk.Entry(f, textvariable=self.base_url_var, width=50).grid(
            row=1, column=1, sticky="w", padx=15, pady=5)

        ttk.Label(f, text="API Key:").grid(row=2, column=0, sticky="w", padx=15, pady=5)
        self.api_key_var = tk.StringVar(value="")
        ttk.Entry(f, textvariable=self.api_key_var, width=50, show="*").grid(
            row=2, column=1, sticky="w", padx=15, pady=5)

        ttk.Label(f, text="Backend Type:").grid(row=3, column=0, sticky="w", padx=15, pady=5)
        self.backend_type_var = tk.StringVar(value="auto")
        ttk.Combobox(f, textvariable=self.backend_type_var,
                     values=["auto", "ollama", "openai"],
                     width=15, state="readonly").grid(row=3, column=1, sticky="w", padx=15, pady=5)

        ttk.Button(f, text="Apply Backend", command=self._on_apply_backend).grid(
            row=4, column=1, sticky="w", padx=15, pady=10)

        # Sampling section
        ttk.Label(f, text="Sampling Parameters", font=("Segoe UI", 12, "bold")).grid(
            row=5, column=0, columnspan=2, sticky="w", padx=15, pady=(20, 5))

        ttk.Label(f, text="Temperature:").grid(row=6, column=0, sticky="w", padx=15, pady=5)
        self.set_temp_var = tk.StringVar(value="0.7")
        ttk.Entry(f, textvariable=self.set_temp_var, width=10).grid(
            row=6, column=1, sticky="w", padx=15, pady=5)

        ttk.Label(f, text="Top-P:").grid(row=7, column=0, sticky="w", padx=15, pady=5)
        self.top_p_var = tk.StringVar(value="0.9")
        ttk.Entry(f, textvariable=self.top_p_var, width=10).grid(
            row=7, column=1, sticky="w", padx=15, pady=5)

        ttk.Label(f, text="Max Tokens:").grid(row=8, column=0, sticky="w", padx=15, pady=5)
        self.max_tokens_var = tk.StringVar(value="4096")
        ttk.Entry(f, textvariable=self.max_tokens_var, width=10).grid(
            row=8, column=1, sticky="w", padx=15, pady=5)

        btns2 = ttk.Frame(f)
        btns2.grid(row=9, column=1, sticky="w", padx=15, pady=10)
        ttk.Button(btns2, text="Apply Sampling", command=self._on_apply_sampling).pack(side="left", padx=(0, 10))
        ttk.Button(btns2, text="Test Connection", command=self._on_test_connection).pack(side="left")

    # ---- Status ----
    def _set_status(self, msg: str) -> None:
        self.statusbar.configure(text=msg)
        self.update_idletasks()

    # ---- Actions ----
    def _get_temperature(self) -> float:
        """Get temperature from settings tab (or wash tab fallback)."""
        try:
            return float(self.set_temp_var.get())
        except ValueError:
            pass
        try:
            return float(self.temp_var.get())
        except (ValueError, AttributeError):
            return 0.7

    def _get_max_tokens(self) -> int:
        try:
            return int(self.max_tokens_var.get())
        except (ValueError, AttributeError):
            return 4096

    def _on_wash(self) -> None:
        text = self.input_text.get("1.0", "end-1c")
        if not text.strip():
            self._set_status("No input text")
            return
        self._set_status("Washing (multi-pass)...")
        threading.Thread(target=self._do_wash, args=(text,), daemon=True).start()

    def _do_wash(self, text: str) -> None:
        try:
            model = self.model_var.get()
            temp = self._get_temperature()
            max_tokens = self._get_max_tokens()
            result = wash_multi_pass(text, passes=2, model=model,
                                     temperature=temp, max_tokens=max_tokens)
            self.output_text.delete("1.0", "end")
            self.output_text.insert("1.0", result.output)
            self._set_status(f"Done: {result.passes} passes, {result.time_s:.1f}s, "
                             f"AI score {result.ai_score_before}→{result.ai_score_after}")
        except Exception as e:
            self._set_status(f"Error: {e}")

    def _on_single_pass(self) -> None:
        text = self.input_text.get("1.0", "end-1c")
        if not text.strip():
            return
        self._set_status("Single pass...")
        threading.Thread(target=self._do_single_pass, args=(text,), daemon=True).start()

    def _do_single_pass(self, text: str) -> None:
        try:
            model = self.model_var.get()
            temp = self._get_temperature()
            max_tokens = self._get_max_tokens()
            result = wash_pass(text, model=model, temperature=temp, max_tokens=max_tokens)
            self.output_text.delete("1.0", "end")
            self.output_text.insert("1.0", result.output)
            self._set_status(f"Done: 1 pass, {result.time_s:.1f}s")
        except Exception as e:
            self._set_status(f"Error: {e}")

    def _on_analyze_only(self) -> None:
        self.notebook.select(self.tab_analyze)
        # Copy text from wash tab to analyze tab
        text = self.input_text.get("1.0", "end-1c")
        if text.strip():
            self.analysis_input.delete("1.0", "end")
            self.analysis_input.insert("1.0", text)
        self._on_analyze()

    def _on_analyze(self) -> None:
        text = self.analysis_input.get("1.0", "end-1c")
        if not text.strip():
            return
        self._set_status("Analyzing...")
        threading.Thread(target=self._do_analyze, args=(text,), daemon=True).start()

    def _do_analyze(self, text: str) -> None:
        try:
            report = analyze_text(text)
            formatted = format_report(report)
            self.analysis_output.delete("1.0", "end")
            self.analysis_output.insert("1.0", formatted)
            self._set_status(f"AI Score: {report.get('ai_score', '?')}/100")
        except Exception as e:
            self._set_status(f"Error: {e}")

    def _on_gen_prompt(self) -> None:
        text = self.analysis_input.get("1.0", "end-1c")
        if not text.strip():
            return
        try:
            prompt = generate_anti_watermark_prompt(text)
            self.analysis_output.delete("1.0", "end")
            self.analysis_output.insert("1.0", prompt)
            self._set_status("Anti-WM prompt generated")
        except Exception as e:
            self._set_status(f"Error: {e}")

    def _on_clear(self) -> None:
        self.input_text.delete("1.0", "end")
        self.output_text.delete("1.0", "end")

    def _on_load_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if path:
            text = Path(path).read_text(encoding="utf-8")
            self.input_text.delete("1.0", "end")
            self.input_text.insert("1.0", text)
            self._set_status(f"Loaded: {Path(path).name}")

    def _on_save_output(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".txt",
                                             filetypes=[("Text", "*.txt")])
        if path:
            output = self.output_text.get("1.0", "end-1c")
            Path(path).write_text(output, encoding="utf-8")
            self._set_status(f"Saved: {Path(path).name}")

    def _on_apply_backend(self) -> None:
        base_url = self.base_url_var.get().strip()
        api_key = self.api_key_var.get().strip() or None
        btype = self.backend_type_var.get()
        cfg = BackendConfig(base_url=base_url, api_key=api_key, backend_type=btype)
        set_backend(cfg)
        self._set_status(f"Backend: {btype} @ {base_url[:50]}...")

    def _on_apply_sampling(self) -> None:
        try:
            temp = float(self.set_temp_var.get())
            top_p = float(self.top_p_var.get())
            max_tokens = int(self.max_tokens_var.get())
            self._set_status(f"Sampling: temp={temp}, top_p={top_p}, max_tokens={max_tokens}")
        except ValueError as e:
            self._set_status(f"Invalid value: {e}")

    def _on_test_connection(self) -> None:
        self._set_status("Testing connection...")
        threading.Thread(target=self._do_test, daemon=True).start()

    def _do_test(self) -> None:
        try:
            result = call_llm("Reply with: OK", model="llama3.2",
                              max_tokens=10, timeout=15)
            self._set_status(f"OK: {result[:80]}")
        except Exception as e:
            self._set_status(f"Failed: {e}")


# ============================================================
# Entry Point
# ============================================================

def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
