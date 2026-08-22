"""Tests for GhostMark integration.

Tests cover:
- Unicode scrubbing (lossless)
- Homoglyph perturbation
- Shatter SynthID pipeline
- Image metadata stripping
- Document metadata stripping
"""

import io
import struct
import zipfile

import pytest
from pathlib import Path


class TestUnicodeScrub:
    """Test Unicode sanitization (Layer 1)."""

    def test_zero_width_space_removed(self):
        from ghostmark_integration import scrub_unicode
        text = "Hello​World"  # Zero-width space
        result = scrub_unicode(text)
        assert result == "HelloWorld"

    def test_bom_removed(self):
        from ghostmark_integration import scrub_unicode
        text = "﻿Hello"  # BOM
        result = scrub_unicode(text)
        assert result == "Hello"

    def test_tag_characters_removed(self):
        from ghostmark_integration import scrub_unicode
        text = "Hello\U000E0001World"  # Tag character
        result = scrub_unicode(text)
        assert result == "HelloWorld"

    def test_normal_text_preserved(self):
        from ghostmark_integration import scrub_unicode
        text = "Hello, World! This is a test."
        result = scrub_unicode(text)
        assert result == text

    def test_aggressive_applies_homoglyphs(self):
        from ghostmark_integration import scrub_unicode
        text = "Hello World"
        result = scrub_unicode(text, aggressive=True)
        # Result should have same visual length (homoglyphs are 1:1)
        # But ZWNJ injection adds characters, so length may be >= original
        assert len(result) >= len(text)
        # Should not contain zero-width chars (except ZWNJ from homoglyphs)
        assert "​" not in result
        # Should have Cyrillic homoglyphs (probabilistic, but with 15% chance on 11 chars, ~1-2 expected)
        cyrillic_chars = set("аоехАСРХ")
        has_cyrillic = any(c in cyrillic_chars for c in result)
        # With 15% probability per char over 11 chars, very likely to have at least 1
        assert has_cyrillic or len(text) < 5  # Allow failure for very short texts


class TestLcgRng:
    """Test deterministic RNG."""

    def test_same_seed_same_output(self):
        from ghostmark_integration import LcgRng
        rng1 = LcgRng.from_seed(12345)
        rng2 = LcgRng.from_seed(12345)
        assert rng1.state == rng2.state
        for _ in range(100):
            assert rng1.next() == rng2.next()

    def test_different_seeds_differ(self):
        from ghostmark_integration import LcgRng
        rng1 = LcgRng.from_seed(12345)
        rng2 = LcgRng.from_seed(54321)
        assert rng1.state != rng2.state
        # Very unlikely to produce same sequence
        seq1 = [rng1.next() for _ in range(10)]
        seq2 = [rng2.next() for _ in range(10)]
        assert seq1 != seq2


class TestSynonymPass:
    """Test synonym replacement."""

    def test_delve_replaced(self):
        from ghostmark_integration import pass_synonyms, LcgRng
        text = "We need to delve deeper."
        rng = LcgRng.from_seed(42)
        result = pass_synonyms(text, rng)
        assert "delve" not in result.lower()

    def test_case_preserved(self):
        from ghostmark_integration import pass_synonyms, LcgRng
        text = "Delve into this."
        rng = LcgRng.from_seed(42)
        result = pass_synonyms(text, rng)
        # First letter should be capitalized
        assert result[0].isupper()


class TestBurstiness:
    """Test burstiness injection."""

    def test_short_paragraph_unchanged(self):
        from ghostmark_integration import pass_burstiness, LcgRng
        text = "Short. Text."
        rng = LcgRng.from_seed(42)
        result = pass_burstiness(text, rng)
        assert result == text

    def test_long_sentence_split(self):
        from ghostmark_integration import pass_burstiness, LcgRng
        text = "This is a very long sentence that has many words and should be split because it exceeds the twenty-five word limit and continues still. Another sentence here."
        rng = LcgRng.from_seed(42)
        result = pass_burstiness(text, rng)
        # Should have more sentences than original
        original_sentences = len([s for s in text.split('.') if s.strip()])
        result_sentences = len([s for s in result.split('.') if s.strip()])
        assert result_sentences >= original_sentences


class TestHomoglyphs:
    def test_cyrillic_injection(self):
        from ghostmark_integration import apply_homoglyphs
        text = "The quick brown fox jumps over the lazy dog and the cat sat on the mat"
        result = apply_homoglyphs(text)
        cyrillic_chars = set("аоехАСРХ")
        has_cyrillic = any(c in cyrillic_chars for c in result)
        assert has_cyrillic

    def test_zwnj_injection(self):
        from ghostmark_integration import apply_homoglyphs
        text = "The quick brown fox jumps over the lazy dog and the cat sat on the mat"
        result = apply_homoglyphs(text)
        assert "‌" in result


class TestShatterSynthID:
    def test_full_pipeline(self):
        from ghostmark_integration import shatter_synthid_text
        text = "In today's rapidly evolving landscape, it is crucial to recognize the importance of leveraging comprehensive solutions. We must delve deeper into the realm of innovative technologies. It is noteworthy that stakeholders across various sectors are increasingly seeking robust, seamless solutions that foster growth and empower individuals to achieve their full potential."
        result = shatter_synthid_text(text)
        assert result != text
        assert len(result) > 0
        # Should not contain original AI-heavy phrases
        assert "rapidly evolving" not in result
        assert "it is crucial to recognize" not in result

    def test_deterministic_with_same_input(self):
        from ghostmark_integration import shatter_synthid_text
        text = "The delve process is very crucial. It is important to note that we must leverage synergies. In conclusion, the landscape is rapidly evolving."
        result1 = shatter_synthid_text(text)
        result2 = shatter_synthid_text(text)
        assert result1 == result2


class TestImageStripper:
    """Test image metadata stripping.

    These tests use generated images (no external files needed).
    """

    def test_jpeg_strip(self):
        from image_stripper import strip_image_bytes
        # Minimal JPEG with APP1 (EXIF) segment
        # SOI + APP1 with "Exif\0\0" + minimal content
        import io
        jpeg = io.BytesIO()
        jpeg.write(b"\xff\xd8")  # SOI
        jpeg.write(b"\xff\xe1")  # APP1 marker
        jpeg.write(b"\x00\x10")  # Length: 16
        jpeg.write(b"Exif\x00\x00")  # Exif header
        jpeg.write(b"\x00" * 10)  # Padding
        jpeg.write(b"\xff\xd9")  # EOI
        raw = jpeg.getvalue()

        result = strip_image_bytes(raw)
        # APP1 should be removed
        assert b"\xff\xe1" not in result
        # SOI and EOI should be preserved
        assert result[:2] == b"\xff\xd8"
        assert result[-2:] == b"\xff\xd9"

    def test_png_strip(self):
        from image_stripper import strip_image_bytes
        import io
        png = io.BytesIO()
        png.write(b"\x89PNG\r\n\x1a\n")  # PNG signature
        # IHDR chunk (13 bytes data)
        ihdr_data = b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"  # 1x1 RGB
        ihdr = io.BytesIO()
        ihdr.write(struct.pack(">I", 13))  # Length
        ihdr.write(b"IHDR")
        ihdr.write(ihdr_data)
        ihdr.write(b"\x00\x00\x00\x00")  # CRC (fake)
        png.write(ihdr.getvalue())
        # tEXt chunk (should be removed)
        text_chunk = io.BytesIO()
        text_chunk.write(struct.pack(">I", 4))  # Length
        text_chunk.write(b"tEXt")
        text_chunk.write(b"Test")
        text_chunk.write(b"\x00\x00\x00\x00")  # CRC (fake)
        png.write(text_chunk.getvalue())
        # IEND chunk
        iend = io.BytesIO()
        iend.write(struct.pack(">I", 0))  # Length
        iend.write(b"IEND")
        iend.write(b"\x00\x00\x00\x00")  # CRC (fake)
        png.write(iend.getvalue())
        raw = png.getvalue()

        result = strip_image_bytes(raw)
        # tEXt should be removed
        assert b"tEXt" not in result
        # PNG signature and IHDR should be preserved
        assert result[:8] == b"\x89PNG\r\n\x1a\n"
        assert b"IHDR" in result

    def test_bmp_strip(self):
        from image_stripper import strip_image_bytes
        # BMP with declared size smaller than actual
        data = b"BM"  # Magic
        data += struct.pack("<I", 100)  # Declared size: 100 bytes
        data += b"\x00" * 92  # Padding to reach 100 bytes
        data += b"\xFF" * 50  # Trailing bytes (should be stripped)

        result = strip_image_bytes(data)
        assert len(result) == 100
        assert not result[100:]  # No trailing bytes

    def test_gif_strip(self):
        from image_stripper import strip_image_bytes
        # GIF with trailing bytes after trailer
        data = b"GIF89a"  # Header
        data += b"\x00" * 10  # Some data
        data += b"\x3b"  # Trailer
        data += b"\xFF" * 20  # Trailing bytes

        result = strip_image_bytes(data)
        assert result[-1:] == b"\x3b"
        assert len(result) == len(data) - 20


import struct


class TestDocumentStripper:
    """Test document metadata stripping."""

    def test_docx_strip(self):
        from document_stripper import _strip_docx
        import io
        # Create minimal DOCX with docProps
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
            zout.writestr("[Content_Types].xml", "<?xml version='1.0'?>")
            zout.writestr("word/document.xml", "<w:document/>")
            zout.writestr("docProps/core.xml", "<?xml version='1.0'?>")
            zout.writestr("docProps/app.xml", "<?xml version='1.0'?>")
        raw = buf.getvalue()

        result = _strip_docx(raw)
        with zipfile.ZipFile(io.BytesIO(result), "r") as zin:
            names = zin.namelist()
            assert "docProps/core.xml" not in names
            assert "docProps/app.xml" not in names
            assert "word/document.xml" in names

    def test_svg_strip(self):
        from document_stripper import _strip_svg
        svg = '''<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" data-c2pa-signer="test">
  <metadata>Test metadata</metadata>
  <!-- A comment -->
  <rect width="100" height="100"/>
</svg>'''.encode("utf-8")

        result = _strip_svg(svg)
        text = result.decode("utf-8")
        assert "<metadata>" not in text
        assert "<!--" not in text
        assert "data-c2pa-" not in text
        assert "<rect" in text  # Content preserved

    def test_odt_strip(self):
        from document_stripper import _strip_odt
        import io
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
            zout.writestr("content.xml", "<?xml version='1.0'?>")
            zout.writestr("meta.xml", "<?xml version='1.0'?>")
            zout.writestr("META-INF/documentsignatures.xml", "<?xml version='1.0'?>")
        raw = buf.getvalue()

        result = _strip_odt(raw)
        with zipfile.ZipFile(io.BytesIO(result), "r") as zin:
            names = zin.namelist()
            assert "meta.xml" not in names
            assert "META-INF/documentsignatures.xml" not in names
            assert "content.xml" in names


import io
import zipfile
