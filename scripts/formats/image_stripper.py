"""GhostMark Image Metadata Stripper — Python port.

Strips C2PA, EXIF, XMP, ICC, IPTC metadata from JPEG, PNG, BMP, GIF
without modifying pixel data. Ported from GhostMark's Rust implementation.

Based on: https://github.com/kilopal/GhostMark (Apache 2.0)
"""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import BinaryIO


def strip_image_metadata(input_path: str | Path, output_path: str | Path | None = None) -> bytes:
    """Strip metadata from an image file. Returns cleaned bytes.

    Supports: JPEG, PNG, BMP, GIF.
    If output_path is None, overwrites input file in-place.
    """
    input_path = Path(input_path)
    raw = input_path.read_bytes()

    cleaned = strip_image_bytes(raw)

    output_path = Path(output_path) if output_path else input_path
    output_path.write_bytes(cleaned)
    return cleaned


def strip_image_bytes(raw: bytes) -> bytes:
    """Strip metadata from raw image bytes in memory.

    Supports: JPEG, PNG, BMP, GIF.
    """
    # Try JPEG (magic bytes: FF D8)
    if raw[:2] == bytes([0xFF, 0xD8]):
        return _strip_jpeg(raw)

    # Try PNG (magic bytes: 89 50 4E 47 0D 0A 1A 0A)
    if raw[:8] == bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]):
        return _strip_png(raw)

    # Try BMP (magic bytes: 42 4D = "BM")
    if raw[:2] == b"BM":
        return _strip_bmp_trailing_bytes(raw)

    # Try GIF (magic bytes: 47 49 46 = "GIF")
    if raw[:3] == b"GIF":
        return _strip_gif_trailing_bytes(raw)

    raise ValueError("Unsupported format. Supported: JPEG, PNG, BMP, GIF.")


def _strip_jpeg(raw: bytes) -> bytes:
    """Strip metadata segments from JPEG.

    Drops: APP1 (EXIF/XMP), APP2 (ICC), APP11 (JPEG XT/C2PA/JUMBF), APP13 (Photoshop IRB)
    Keeps: SOI, APP0 (JFIF), DQT, DHT, SOF, SOS, EOI, and all scan data.
    """
    out = io.BytesIO()
    i = 0

    while i < len(raw):
        # Find next marker
        if raw[i] != 0xFF:
            i += 1
            continue

        marker = raw[i + 1]

        # Skip padding
        if marker == 0x00 or marker == 0xFF:
            i += 1
            continue

        # SOI — always keep
        if marker == 0xD8:
            out.write(raw[i:i + 2])
            i += 2
            continue

        # EOI — always keep
        if marker == 0xD9:
            out.write(raw[i:i + 2])
            i += 2
            continue

        # SOS — start of scan, copy everything until next marker
        if marker == 0xDA:
            # Copy SOS header first
            if i + 4 <= len(raw):
                length = struct.unpack(">H", raw[i + 2:i + 4])[0]
                out.write(raw[i:i + 2 + length])
                i += 2 + length
            # Copy scan data until next non-0xFF byte or EOI
            while i < len(raw):
                if raw[i] == 0xFF and i + 1 < len(raw) and raw[i + 1] not in (0x00, 0xFF):
                    break
                out.write(bytes([raw[i]]))
                i += 1
            continue

        # Get segment length
        if i + 4 > len(raw):
            break
        length = struct.unpack(">H", raw[i + 2:i + 4])[0]

        # Drop tracking segments
        if marker in (0xE1, 0xE2, 0xEB, 0xED):
            i += 2 + length
            continue

        # Keep all other segments
        end = i + 2 + length
        out.write(raw[i:end])
        i = end

    return out.getvalue()


def _strip_png(raw: bytes) -> bytes:
    """Strip metadata chunks from PNG.

    Keeps only: IHDR, PLTE, IDAT, IEND, tRNS
    Drops: c2pa, iTXt, tEXt, eXIf, and all other ancillary chunks.
    """
    out = io.BytesIO()
    # Write PNG signature
    out.write(raw[:8])

    i = 8
    while i + 12 <= len(raw):
        length = struct.unpack(">I", raw[i:i + 4])[0]
        chunk_type = raw[i + 4:i + 8]

        # Keep only critical rendering chunks
        if chunk_type in (b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS"):
            out.write(raw[i:i + 12 + length])

        i += 12 + length

        if chunk_type == b"IEND":
            break

    return out.getvalue()


def _strip_bmp_trailing_bytes(raw: bytes) -> bytes:
    """Strip trailing bytes from BMP files.

    BMP declares total file size at bytes 2-5 (little-endian u32).
    Anything after that is extraneous.
    """
    if len(raw) < 14:
        return raw
    declared_size = struct.unpack("<I", raw[2:6])[0]
    return raw[:declared_size]


def _strip_gif_trailing_bytes(raw: bytes) -> bytes:
    """Strip trailing bytes from GIF files.

    GIF ends with 0x3B trailer. Anything after is extraneous.
    """
    trailer_pos = raw.rfind(b"\x3b")
    if trailer_pos == -1:
        return raw
    return raw[:trailer_pos + 1]
