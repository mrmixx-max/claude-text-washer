"""PDF text-layer watermarking + metadata stripping.

Adapted from TWS ``metadata/pdf_watermark.py`` and the PDF metadata-stripping
logic from ``metadata/service.py``.

Two capabilities:

1. **Metadata strip** — byte-level removal of XMP /Metadata streams and
   /Producer and /Creator Info entries where unambiguously located.
   ``pdf_clean_metadata(data) -> (cleaned, report)`` is the entry point used by
   ``file_washer``.

2. **Text-layer watermark** — embed/detect invisible watermarks in the PDF *text*
   layer using subtle text-positioning tricks:

   - Inter-word spacing watermark: encodes bits as small variations in space
     width between words (imperceptible to the eye).
   - TJS (Trivial JavaScript Stamp) watermark: embeds a JSON payload with
     HMAC-SHA256 signature as a metadata stream object (provenance marker).
   - Text color watermark: encodes bits as near-zero RGB shifts
     (0,0,0 vs 0,0,1 — black vs near-black).

Honest boundaries:
- PDF cleaning is byte-level best-effort. ``exiftool`` remains the stronger
  tool for hard cases; this layer degrades gracefully without it.
- C2PA *soft binding* (in-content marks that re-link a remote manifest) and
  pixel-domain marks are OUT OF SCOPE.
- Spacing watermark survives copy-paste only if the reader preserves kerning.
  TJS metadata watermark is removed by any tool that re-generates the PDF.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass, field
from typing import Literal

# Watermark types
WM_SPACING = "spacing"
WM_METADATA = "metadata"
WM_COLOR = "color"

# Marker prefix for metadata watermark
WM_MARKER = "TWS-PDF-WM"

# ---------------------------------------------------------------- metadata strip
_META_STREAM_RE = re.compile(rb"<<\s*/Type\s*/Metadata[\s\S]{0,2000}?>>\s*stream\s*[\s\S]{0,200000}?endstream")


@dataclass
class PdfCleanReport:
    """Report from stripping PDF metadata / watermarks."""

    format: str = "pdf"
    actions: list[str] = field(default_factory=list)
    removed_bytes: int = 0
    hard_bound_c2pa_present: bool = False
    metadata_marker_found: bool = False
    spacing_marker_found: bool = False
    color_marker_found: bool = False
    cleaned: bytes | None = None

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "actions": self.actions,
            "removed_bytes": self.removed_bytes,
            "hard_bound_c2pa_present": self.hard_bound_c2pa_present,
            "metadata_marker_found": self.metadata_marker_found,
            "spacing_marker_found": self.spacing_marker_found,
            "color_marker_found": self.color_marker_found,
        }


def _sign(secret: str, payload: bytes) -> str:
    """HMAC-SHA256 signature."""
    return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


def _make_packet(key_id: str, secret: str, payload: bytes) -> bytes:
    """Create a signed watermark packet."""
    sig = _sign(secret, payload)
    return json.dumps(
        {
            "marker": WM_MARKER,
            "key_id": key_id,
            "sig": sig,
            "payload_b64": payload.hex(),
            "v": 1,
        },
        separators=(",", ":"),
    ).encode()


def _parse_packet(raw: bytes) -> dict | None:
    """Parse and validate a watermark packet."""
    try:
        d = json.loads(raw.decode("utf-8"))
        if d.get("marker") != WM_MARKER:
            return None
        return d
    except (ValueError, TypeError, AttributeError):
        return None


# ---------------------------------------------------------------- Spacing watermark
_TD_RE = re.compile(rb"\s*(-?[\d.]+)\s+(-?[\d.]+)\s+Td")
_TJ_RE = re.compile(rb"\[(.*?)\\]TJ")
_SPACE_UNIT = 1000


def _text_to_bits(text: str) -> list[int]:
    """Convert a short string message to a bit list."""
    bits = []
    for ch in text.encode("utf-8"):
        for i in range(7, -1, -1):
            bits.append((ch >> i) & 1)
    return bits


def _bits_to_bytes(bits: list[int]) -> bytes:
    """Convert a bit list back to bytes."""
    out = bytearray()
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        out.append(byte)
    return bytes(out)


def embed_spacing_watermark(pdf_data: bytes, message: str, secret: str) -> bytes:
    """Embed a spacing watermark in the PDF content stream."""
    msg_bytes = message.encode("utf-8")
    length_bits = []
    length = len(msg_bytes) * 8
    for i in range(15, -1, -1):
        length_bits.append((length >> i) & 1)
    all_bits = length_bits + _text_to_bits(message)

    bit_idx = 0
    out = bytearray()
    pos = 0
    for m in _TD_RE.finditer(pdf_data):
        out.extend(pdf_data[pos : m.end()])
        pos = m.end()
        if bit_idx < len(all_bits):
            shift = 75 if all_bits[bit_idx] == 1 else 25
            out.extend(f" {shift} Tc ".encode())
            bit_idx += 1
    out.extend(pdf_data[pos:])
    return bytes(out)


def detect_spacing_watermark(pdf_data: bytes, secret: str) -> dict:
    """Detect and decode a spacing watermark from the PDF content stream."""
    bits = []
    for m in _TD_RE.finditer(pdf_data):
        after = pdf_data[m.end() : m.end() + 30]
        tc_match = re.match(rb"\s+(\d+)\s+Tc", after)
        if tc_match:
            val = int(tc_match.group(1))
            bits.append(1 if val > 50 else 0)
    if len(bits) < 16:
        return {"found": False, "reason": "no_spacing_markers", "message": None}
    length = 0
    for b in bits[:16]:
        length = (length << 1) | b
    if length == 0 or length > len(bits) - 16:
        return {"found": False, "reason": "invalid_length", "message": None}
    msg_bits = bits[16 : 16 + length]
    if len(msg_bits) < length:
        return {"found": False, "reason": "truncated", "message": None}
    while len(msg_bits) % 8 != 0:
        msg_bits.append(0)
    msg_bytes = _bits_to_bytes(msg_bits)
    try:
        message = msg_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return {"found": False, "reason": "decode_error", "message": None}
    return {"found": True, "message": message, "bits_read": 16 + length, "secret": secret}


# ---------------------------------------------------------------- Metadata watermark (TJS-style)
def embed_metadata_watermark(pdf_data: bytes, key_id: str, secret: str) -> bytes:
    """Embed a signed metadata watermark as a new PDF stream object."""
    payload = _make_packet(key_id, secret, pdf_data)
    stream_obj = (
        b"<< /Type /Metadata /Subtype /XML /Length "
        + str(len(payload)).encode()
        + b" >>\nstream\n"
        + payload
        + b"\nendstream"
    )
    if not pdf_data.endswith(b"%%EOF\n") and b"%%EOF" not in pdf_data[-20:]:
        pdf_data = pdf_data.rstrip(b"\n") + b"\n%%EOF\n"
    return pdf_data + stream_obj


def detect_metadata_watermark(pdf_data: bytes, secrets: dict[str, str]) -> dict:
    """Detect a signed metadata watermark in the PDF."""
    result = {"found": False, "valid": False, "key_id": None, "marks": []}
    pattern = re.compile(
        rb"<<\s*/Type\s*/Metadata\s*/Subtype\s*/XML\s*/Length\s+\d+\s*>>"
        rb"\s*stream\s*([\s\S]*?)endstream",
        re.IGNORECASE,
    )
    for m in pattern.finditer(pdf_data):
        raw = m.group(1).strip()
        packet = _parse_packet(raw)
        if packet is None:
            continue
        result["found"] = True
        result["marks"].append(packet)
        key_id = packet.get("key_id", "")
        stored_sig = packet.get("sig", "")
        restored = pdf_data[: m.start()] + pdf_data[m.end() :]
        expected_sig = _sign(secrets.get(key_id, ""), restored)
        if hmac.compare_digest(stored_sig, expected_sig):
            result["valid"] = True
            result["key_id"] = key_id
            result["reason"] = "hmac_valid"
            break
        result["reason"] = "hmac_invalid_or_unknown_key"
    return result


# ---------------------------------------------------------------- Text color watermark
def embed_color_watermark(pdf_data: bytes, message: str, secret: str) -> bytes:
    """Embed a watermark via near-zero text color shifts."""
    msg_bytes = message.encode("utf-8")
    length_bits = []
    length = len(msg_bytes) * 8
    for i in range(15, -1, -1):
        length_bits.append((length >> i) & 1)
    all_bits = length_bits + _text_to_bits(message)

    color_re = re.compile(rb"(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+rg")
    bit_idx = 0
    out = bytearray()
    pos = 0
    for m in color_re.finditer(pdf_data):
        out.extend(pdf_data[pos : m.end()])
        pos = m.end()
        if bit_idx < len(all_bits):
            if all_bits[bit_idx] == 1:
                out.extend(b" 0 0 0.004 rg")
            else:
                out.extend(b" 0 0 0.001 rg")
            bit_idx += 1
    out.extend(pdf_data[pos:])
    return bytes(out)


def detect_color_watermark(pdf_data: bytes, secret: str) -> dict:
    """Detect a text color watermark."""
    bits = []
    color_re = re.compile(rb"(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+rg")
    for m in color_re.finditer(pdf_data):
        after = pdf_data[m.end() : m.end() + 30]
        if re.match(rb"\s*0\s+0\s+0\.004\s+rg", after):
            bits.append(1)
        elif re.match(rb"\s*0\s+0\s+0\.001\s+rg", after):
            bits.append(0)
    if len(bits) < 16:
        return {"found": False, "reason": "no_color_markers", "message": None}
    length = 0
    for b in bits[:16]:
        length = (length << 1) | b
    if length == 0 or length > len(bits) - 16:
        return {"found": False, "reason": "invalid_length", "message": None}
    msg_bits = bits[16 : 16 + length]
    while len(msg_bits) % 8 != 0:
        msg_bits.append(0)
    msg_bytes = _bits_to_bytes(msg_bits)
    try:
        message = msg_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return {"found": False, "reason": "decode_error", "message": None}
    return {"found": True, "message": message, "bits_read": 16 + length}


# ---------------------------------------------------------------- Unified metadata strip
def pdf_clean_metadata(data: bytes) -> tuple[bytes, dict]:
    """Strip PDF metadata streams and Info entries (best-effort, byte-level).

    Returns (cleaned_data, report_dict).
    """
    rep = PdfCleanReport(format="pdf")
    if data[:5] != b"%PDF-":
        rep.actions.append("not_a_pdf")
        rep.cleaned = data
        return data, rep.to_dict()

    pdf_data = data
    # Detect watermark markers (so a subsequent wash knows what was there)
    if detect_metadata_watermark(pdf_data, {}).get("found"):
        rep.metadata_marker_found = True
    spacing = detect_spacing_watermark(pdf_data, "")
    if spacing.get("found"):
        rep.spacing_marker_found = True
    if detect_color_watermark(pdf_data, "").get("found"):
        rep.color_marker_found = True

    if b"/Metadata" in pdf_data or b"/Producer" in pdf_data or b"/Creator" in pdf_data:
        rep.actions.append("metadata_reference_found")
    if b"jumb" in pdf_data.lower() or b"c2pa" in pdf_data.lower():
        rep.hard_bound_c2pa_present = True

    # remove XMP metadata streams: << /Type /Metadata ... stream ... endstream
    new = _META_STREAM_RE.sub(b"", pdf_data, count=8)
    new = re.sub(rb"/Producer\s*\([^)]*\)", b"/Producer ()", new)
    new = re.sub(rb"/Creator\s*\([^)]*\)", b"/Creator ()", new)
    new = re.sub(rb"/Title\s*\([^)]*\)", b"/Title ()", new)
    new = re.sub(rb"/Author\s*\([^)]*\)", b"/Author ()", new)
    new = re.sub(rb"/Subject\s*\([^)]*\)", b"/Subject ()", new)
    new = re.sub(rb"/Keywords\s*\([^)]*\)", b"/Keywords ()", new)

    if new != data:
        rep.actions.append("removed_pdf_xmp_streams_and_info")
        rep.removed_bytes = len(data) - len(new)
    else:
        rep.actions.append("no_pdf_metadata_removed")
    rep.cleaned = new
    return new, rep.to_dict()


# ---------------------------------------------------------------- Unified API (embed / detect)
def embed_pdf_watermark(
    pdf_data: bytes,
    message: str,
    secret: str,
    method: Literal["spacing", "metadata", "color"] = "metadata",
    key_id: str = "default",
) -> bytes:
    """Embed a text-layer watermark in a PDF.

    method="spacing": inter-word spacing perturbation (survives rendering).
    method="metadata": signed XMP-style stream (provenance, removed on regen).
    method="color": near-zero RGB shift (invisible, survives rendering).
    """
    if method == WM_SPACING:
        return embed_spacing_watermark(pdf_data, message, secret)
    if method == WM_METADATA:
        return embed_metadata_watermark(pdf_data, key_id, secret)
    if method == WM_COLOR:
        return embed_color_watermark(pdf_data, message, secret)
    raise ValueError(f"unknown watermark method: {method}")


def detect_pdf_watermark(
    pdf_data: bytes,
    secret: str,
    secrets: dict[str, str] | None = None,
    method: Literal["spacing", "metadata", "color", "auto"] = "auto",
) -> dict:
    """Detect a text-layer watermark in a PDF.

    method="auto" tries all methods in order: metadata, spacing, color.
    """
    if method == "auto":
        for m in (WM_METADATA, WM_SPACING, WM_COLOR):
            r = detect_pdf_watermark(pdf_data, secret, secrets, m)
            if r.get("found"):
                return r
        return {"found": False, "reason": "no_watermark_detected"}

    if method == WM_SPACING:
        return detect_spacing_watermark(pdf_data, secret)
    if method == WM_METADATA:
        return detect_metadata_watermark(pdf_data, secrets or {secret: secret})
    if method == WM_COLOR:
        return detect_color_watermark(pdf_data, secret)
    raise ValueError(f"unknown watermark method: {method}")
