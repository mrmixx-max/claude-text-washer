"""GhostMark Document Metadata Stripper — Python port.

Strips metadata from PDF, DOCX, SVG, EPUB, ODT without modifying content.
Ported from GhostMark's Rust implementation.

Based on: https://github.com/kilopal/GhostMark (Apache 2.0)
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path


def strip_pdf_metadata(input_path: str | Path, output_path: str | Path | None = None) -> bytes:
    """Strip metadata from PDF. Returns cleaned bytes."""
    input_path = Path(input_path)
    raw = input_path.read_bytes()

    cleaned = _strip_pdf(raw)

    output_path = Path(output_path) if output_path else input_path
    output_path.write_bytes(cleaned)
    return cleaned


def _strip_pdf(raw: bytes) -> bytes:
    """Strip PDF metadata. Removes /Info dictionary and XMP /Metadata streams."""
    # Simple but effective: remove all /Info references and metadata streams
    # This is a lossy operation for metadata only, content is preserved
    text = raw.decode("latin-1", errors="replace")

    # Remove /Info dictionary references
    text = re.sub(r"/Info\s+\d+\s+\d+\s+R", "", text)

    # Remove XMP metadata streams (between "stream" and "endstream" with /Type /Metadata)
    # This is a heuristic but works for most PDFs
    text = re.sub(r"stream\n<<[^>]*?/Type\s*/Metadata[^>]*?>>\nendstream", "stream\nendstream", text)

    # Remove XMP metadata in objects
    text = re.sub(r"/Metadata\s+\d+\s+\d+\s+R", "", text)

    return text.encode("latin-1", errors="replace")


def strip_docx_metadata(input_path: str | Path, output_path: str | Path | None = None) -> bytes:
    """Strip metadata from DOCX. Returns cleaned bytes."""
    input_path = Path(input_path)
    raw = input_path.read_bytes()

    cleaned = _strip_docx(raw)

    output_path = Path(output_path) if output_path else input_path
    output_path.write_bytes(cleaned)
    return cleaned


def _strip_docx(raw: bytes) -> bytes:
    """Strip DOCX metadata. Removes docProps/ and customXml/ directories."""
    out = io.BytesIO()

    with zipfile.ZipFile(io.BytesIO(raw), "r") as zin:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.namelist():
                # Skip metadata directories
                if item.startswith("docProps/") or item.startswith("customXml/"):
                    continue
                zout.writestr(item, zin.read(item))

    return out.getvalue()


def strip_svg_metadata(input_path: str | Path, output_path: str | Path | None = None) -> bytes:
    """Strip metadata from SVG. Returns cleaned bytes."""
    input_path = Path(input_path)
    raw = input_path.read_bytes()

    cleaned = _strip_svg(raw)

    output_path = Path(output_path) if output_path else input_path
    output_path.write_bytes(cleaned)
    return cleaned


def _strip_svg(raw: bytes) -> bytes:
    """Strip SVG metadata. Removes <metadata>, <!-- comments -->, data-c2pa-* attributes."""
    text = raw.decode("utf-8", errors="replace")

    # Remove <metadata>...</metadata> blocks
    text = re.sub(r"<metadata.*?</metadata>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove XML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Remove data-c2pa-* attributes
    text = re.sub(r'\s+data-c2pa-[a-z-]+="[^"]*"', "", text, flags=re.IGNORECASE)

    return text.encode("utf-8", errors="replace")


def strip_epub_metadata(input_path: str | Path, output_path: str | Path | None = None) -> bytes:
    """Strip metadata from EPUB. Returns cleaned bytes."""
    input_path = Path(input_path)
    raw = input_path.read_bytes()

    cleaned = _strip_epub(raw)

    output_path = Path(output_path) if output_path else input_path
    output_path.write_bytes(cleaned)
    return cleaned


def _strip_epub(raw: bytes) -> bytes:
    """Strip EPUB metadata. Removes META-INF/signatures, encryption, rights, OPF metadata."""
    out = io.BytesIO()

    with zipfile.ZipFile(io.BytesIO(raw), "r") as zin:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.namelist():
                # Skip digital signatures and encryption metadata
                if item.startswith("META-INF/signatures") or item.startswith("META-INF/encryption") or item.startswith("META-INF/rights"):
                    continue

                data = zin.read(item)

                # Strip metadata from OPF files
                if item.endswith(".opf"):
                    data = _strip_opf_metadata(data)

                zout.writestr(item, data)

    return out.getvalue()


def _strip_opf_metadata(data: bytes) -> bytes:
    """Strip <metadata> block from OPF content."""
    text = data.decode("utf-8", errors="replace")
    text = re.sub(r"<metadata.*?</metadata>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.encode("utf-8", errors="replace")


def strip_odt_metadata(input_path: str | Path, output_path: str | Path | None = None) -> bytes:
    """Strip metadata from ODT. Returns cleaned bytes."""
    input_path = Path(input_path)
    raw = input_path.read_bytes()

    cleaned = _strip_odt(raw)

    output_path = Path(output_path) if output_path else input_path
    output_path.write_bytes(cleaned)
    return cleaned


def _strip_odt(raw: bytes) -> bytes:
    """Strip ODT metadata. Removes meta.xml and digital signatures."""
    out = io.BytesIO()

    with zipfile.ZipFile(io.BytesIO(raw), "r") as zin:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.namelist():
                # Skip meta.xml (author, date, stats)
                if item == "meta.xml":
                    continue
                # Skip digital signature files
                if item.startswith("META-INF/documentsignatures"):
                    continue
                zout.writestr(item, zin.read(item))

    return out.getvalue()


def batch_strip(input_dir: str | Path, output_dir: str | Path | None = None, suffix: str = "") -> list[Path]:
    """Recursively strip metadata from all supported files in a directory.

    Args:
        input_dir: Directory to scan
        output_dir: Output directory (None = in-place)
        suffix: Only process files ending with this suffix (e.g. ".pdf")

    Returns:
        List of processed file paths
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir

    processed = []
    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        if suffix and not path.suffix.lower() == suffix.lower():
            continue

        ext = path.suffix.lower()
        try:
            if ext in (".txt", ".md", ".json"):
                # Text scrubbing is handled separately
                pass
            elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".gif"):
                strip_image_metadata(path)
                processed.append(path)
            elif ext == ".pdf":
                strip_pdf_metadata(path)
                processed.append(path)
            elif ext == ".docx":
                strip_docx_metadata(path)
                processed.append(path)
            elif ext == ".svg":
                strip_svg_metadata(path)
                processed.append(path)
            elif ext == ".epub":
                strip_epub_metadata(path)
                processed.append(path)
            elif ext == ".odt":
                strip_odt_metadata(path)
                processed.append(path)
        except Exception as e:
            print(f"Failed to strip {path}: {e}")

    return processed
