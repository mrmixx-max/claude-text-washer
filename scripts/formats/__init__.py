"""Document-format handlers for claude-text-washer.

Each submodule is self-contained and imports heavy libraries lazily so the
package can be imported (e.g. for unit tests) without the optional deps being
installed.

Modules:
    image_metadata  — PNG/JPEG/WebP/AVIF/HEIC/BMP/GIF/TIFF/SVG metadata + C2PA strip
    pdf_watermark   — PDF metadata strip + text-layer watermark detect/embed
    docx_repair     — DOCX metadata strip + structural repair
    documents       — text extraction + metadata strip for DOCX/XLSX/PPTX/EPUB/ODT/HTML/Markdown
    encoding_detect — encoding detection (BOM, UTF-8/16, Latin-1, mixed, conversion attacks)
"""
from __future__ import annotations
