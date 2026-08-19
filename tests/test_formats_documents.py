"""Tests for format auto-detection + document text extraction.

Covers :mod:`scripts.formats.documents`: magic-byte sniffing across formats
and the lazy extractors for PDF / DOCX / HTML / Markdown / TXT, plus the
``force_format`` override that underpins ``file_washer`` format selection.
"""
from __future__ import annotations

import io
import sys
import zipfile

import pytest

from scripts.formats.documents import (
    SUPPORTED,
    ExtractedDocument,
    detect_format,
    extract_text,
)


# --------------------------------------------------------------------------- #
# Magic-byte format detection
# --------------------------------------------------------------------------- #
def test_supported_includes_key_formats():
    for fmt in ("pdf", "docx", "html", "htm", "md", "markdown", "txt"):
        assert fmt in SUPPORTED


def test_detect_pdf_by_magic_bytes():
    assert detect_format(b"%PDF-1.7\nbody", "no-extension") == "pdf"


def test_detect_pdf_ignores_mismatched_extension():
    # Extension says .txt but magic bytes are PDF -> magic bytes win.
    assert detect_format(b"%PDF-1.4 body", "doc.txt") == "pdf"


def test_detect_docx_by_magic_bytes():
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", "<document/>")
    data = bio.getvalue()
    # Extension is irrelevant when the ZIP contains word/document.xml
    assert detect_format(data, "whatever.bin") == "docx"


def test_detect_xlsx_by_magic_bytes():
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("xl/workbook.xml", "<workbook/>")
    assert detect_format(bio.getvalue(), "sheet") == "xlsx"


def test_detect_html_by_magic_bytes():
    html = b"<html><head></head><body><p>hi</p></body></html>"
    assert detect_format(html, "page.xyz") == "html"
    assert detect_format(b"<!doctype html><html></html>", "a.htm") == "html"


def test_detect_html_alias_doctype():
    assert detect_format(b"<!doctype html>", "page.html") == "html"


def test_detect_md_falls_back_to_extension():
    assert detect_format(b"# Title\n\nbody text", "notes.md") == "md"
    assert detect_format(b"# Title\n\nbody text", "notes.markdown") == "markdown"


def test_detect_txt_default():
    assert detect_format(b"just plain text here", "notes.txt") == "txt"


def test_detect_empty_defaults_to_txt():
    assert detect_format(b"", "noext") == "txt"


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def test_extract_pdf_text_real():
    fitz = pytest.importorskip("pymupdf")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello PDF text world")
    data = doc.tobytes()
    doc.close()
    result = extract_text(data, "doc.pdf")
    assert isinstance(result, ExtractedDocument)
    assert result.format == "pdf"
    assert "Hello PDF text world" in result.text
    assert result.metadata is not None  # pymupdf docinfo present


def test_extract_pdf_falls_back_without_lib(monkeypatch):
    # Block both import names so the graceful UTF-8 fallback engages.
    monkeypatch.setitem(sys.modules, "pymupdf", None)
    monkeypatch.setitem(sys.modules, "fitz", None)
    result = extract_text(b"%PDF-1.4 hi there", "doc.pdf")
    assert result.format == "pdf"
    assert "hi there" in result.text
    assert result.metadata.get("error", "").startswith("pymupdf")


def test_extract_docx_text_real():
    Document = pytest.importorskip("docx").Document
    bio = io.BytesIO()
    d = Document()
    d.add_paragraph("First paragraph")
    d.add_paragraph("Second paragraph")
    d.save(bio)
    result = extract_text(bio.getvalue(), "doc.docx")
    assert result.format == "docx"
    assert "First paragraph" in result.text
    assert "Second paragraph" in result.text


def test_extract_html_strips_tags_and_scripts():
    html = b"<html><body><script>var x=1;</script><p>Hello <b>world</b></p></body></html>"
    result = extract_text(html, "page.html")
    assert result.format == "html"
    assert "Hello" in result.text
    assert "world" in result.text
    assert "<" not in result.text
    assert "var x" not in result.text  # script content stripped


def test_extract_markdown_strips_front_matter():
    md = b"---\ntitle: Hello\ntags: [a, b]\n---\n\nBody of the doc.\n"
    result = extract_text(md, "post.md")
    assert result.format == "md"
    assert "Body of the doc." in result.text
    assert "tags" in result.metadata


def test_extract_txt_roundtrip():
    result = extract_text(b"Hello UTF-8 text", "note.txt")
    assert result.format == "txt"
    assert result.text == "Hello UTF-8 text"


# --------------------------------------------------------------------------- #
# force_format override
# --------------------------------------------------------------------------- #
def test_force_format_overrides_magic_bytes():
    # PDF-magic data forced through the markdown (utf-8) path.
    result = extract_text(b"%PDF-1.4 something", "file.bin", force_format="md")
    assert result.format == "md"
    assert "something" in result.text


def test_force_format_pdf_on_garbage_bytes():
    result = extract_text(b"not really a pdf", "x.txt", force_format="pdf")
    assert result.format == "pdf"
    # No real PDF -> pymupdf open fails -> graceful fallback decode.
    assert "not really a pdf" in result.text
    assert "error" in result.metadata
