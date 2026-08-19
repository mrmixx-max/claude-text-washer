"""Tests for :mod:`scripts.file_washer`: format auto-detection on disk,
batch/glob expansion, output-path planning, stdin reading, and the dry-run
CLI path.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from scripts.file_washer import (
    detect_format,
    expand_inputs,
    extract_file_text,
    output_path_for,
    read_text,
)

FIXTURES = Path(__file__).resolve().parents[1] / "_fixtures"


# --------------------------------------------------------------------------- #
# Format auto-detection on real files (magic bytes, not extension)
# --------------------------------------------------------------------------- #
def test_detect_format_by_magic_bytes(tmp_path):
    f = tmp_path / "rep.pdf"
    f.write_bytes(b"%PDF-1.4 content")
    assert detect_format(f) == "pdf"


@pytest.mark.parametrize(
    "name,expected",
    [
        ("sample.docx", "docx"),
        ("sample.pdf", "pdf"),
        ("sample.html", "html"),
        ("sample.md", "md"),
        ("sample.txt", "txt"),
    ],
)
def test_detect_format_fixtures(name, expected):
    assert detect_format(FIXTURES / name) == expected


def test_detect_format_ignores_extension(tmp_path):
    # A .txt file whose bytes are really a PDF is detected as PDF.
    f = tmp_path / "lie.txt"
    f.write_bytes(b"%PDF-1.4 hello")
    assert detect_format(f) == "pdf"


# --------------------------------------------------------------------------- #
# extract_file_text end-to-end
# --------------------------------------------------------------------------- #
def test_extract_pdf_text():
    assert "Hello PDF text" in extract_file_text(FIXTURES / "sample.pdf")


def test_extract_docx_text():
    text = extract_file_text(FIXTURES / "sample.docx")
    assert "Hello docx world" in text
    assert "Second paragraph here" in text


def test_extract_html_strips_tags():
    text = extract_file_text(FIXTURES / "sample.html")
    assert "<" not in text
    assert "Hello HTML paragraph." in text


def test_extract_markdown_text():
    text = extract_file_text(FIXTURES / "sample.md")
    assert "# Title" in text
    assert "markdown" in text


def test_extract_force_format_overrides_detection(tmp_path):
    # PDF-magic bytes forced through the txt handler (utf-8 decode).
    f = tmp_path / "x.txt"
    f.write_bytes(b"%PDF-1.4 forced")
    assert "forced" in extract_file_text(f, fmt="txt")


# --------------------------------------------------------------------------- #
# read_text — stdin sentinel
# --------------------------------------------------------------------------- #
def test_read_text_stdin(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("from stdin"))
    monkeypatch.setattr("scripts.file_washer.sys.stdin", io.StringIO("from stdin"))
    assert read_text("-") == "from stdin"


def test_read_text_file(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello file", encoding="utf-8")
    assert read_text(f) == "hello file"


# --------------------------------------------------------------------------- #
# Batch / glob expansion
# --------------------------------------------------------------------------- #
def test_expand_inputs_plain_files(tmp_path):
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text("a")
    b.write_text("b")
    assert expand_inputs([str(a), str(b)]) == [a, b]


def test_expand_inputs_glob(tmp_path):
    (tmp_path / "a.md").write_text("a")
    (tmp_path / "b.md").write_text("b")
    (tmp_path / "c.txt").write_text("c")
    files = expand_inputs([str(tmp_path / "*.md")])
    assert [p.name for p in files] == ["a.md", "b.md"]


def test_expand_inputs_directory_recursive(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "a.md").write_text("a")
    (sub / "b.md").write_text("b")
    names = sorted(p.name for p in expand_inputs([str(tmp_path)], recursive=True))
    assert names == ["a.md", "b.md"]


def test_expand_inputs_directory_non_recursive_skips_subdirs(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "a.md").write_text("a")
    (sub / "b.md").write_text("b")
    assert [p.name for p in expand_inputs([str(tmp_path)])] == ["a.md"]


def test_expand_inputs_dedup(tmp_path):
    (tmp_path / "a.md").write_text("a")
    files = expand_inputs(
        [str(tmp_path / "a.md"), str(tmp_path / "a.md")]
    )
    assert len(files) == 1


def test_expand_inputs_missing_returns_empty():
    assert expand_inputs(["/no/such/path/xyz"]) == []


def test_expand_inputs_empty_returns_empty():
    assert expand_inputs([]) == []


# --------------------------------------------------------------------------- #
# Output path planning
# --------------------------------------------------------------------------- #
def test_output_path_for_default(tmp_path):
    out = output_path_for(tmp_path / "input.md", None, None)
    assert out == tmp_path / "input.washed.txt"


def test_output_path_for_outdir_creates_dir(tmp_path):
    out = output_path_for(tmp_path / "input.md", None, str(tmp_path / "out"))
    assert out == tmp_path / "out" / "input.washed.txt"
    assert out.parent.exists()


def test_output_path_for_single_output(tmp_path):
    out = output_path_for(
        tmp_path / "input.md", str(tmp_path / "clean.txt"), None
    )
    assert out == tmp_path / "clean.txt"


# --------------------------------------------------------------------------- #
# Dry-run CLI (format detection + analysis, no Ollama needed)
# --------------------------------------------------------------------------- #
def test_dry_run_main_on_fixture(capsys):
    from scripts.file_washer import main

    rc = main(["--dry-run", str(FIXTURES / "sample.pdf")])
    assert rc == 0
    cap = capsys.readouterr().out.lower()
    assert "pdf" in cap
    assert "ai score" in cap or "ai:" in cap
