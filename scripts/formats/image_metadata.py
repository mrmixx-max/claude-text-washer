"""File metadata layer: strip C2PA / EXIF / XMP / AI-provenance from images.

Adapted from TWS ``metadata/service.py``, scoped to image formats + SVG.
PDF lives in ``pdf_watermark``; DOCX lives in ``docx_repair``; HTML/Markdown
live in ``documents``.

Stdlib-only by design (matching the rest of the studio's core). Format
dispatch by extension. Every cleaner returns a MetaReport with verifiable
actions (what was removed) so results separate verifiable from best-effort.

Honest limits (documented, not hidden):
- Byte-level C2PA markers (jumbf/c2pa signatures) that we can strip are
  detected. C2PA *soft binding* (in-content links to remote manifests) and
  pixel-domain marks are OUT OF SCOPE.
- JPEG extended XMP (multi-segment with MD5 splice) is only partially
  removed: the main packet is dropped; spliced continuation segments are
  detected and removed when they carry the extended marker.
- BMP / GIF / TIFF handlers are best-effort metadata strippers; they drop
  EXIF/XMP/IPTC/ICC segments where they can be located unambiguously.
"""
from __future__ import annotations

import io
import re
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED = ["png", "jpg", "jpeg", "webp", "avif", "heic", "bmp", "gif", "tiff", "tif", "svg"]

# metadata keys that smell like AI provenance / generator attribution
_AI_KEY_HINTS = re.compile(
    r"(generated|generator|created[-_ ]?by|ai[-_ ]?model|ai[-_ ]?assistant|"
    r"model[-_ ]?name|prompt|content[-_ ]?credentials|c2pa|provenance|synthid|"
    r"produced[-_ ]?by|authoring[-_ ]?tool)",
    re.IGNORECASE,
)

_AI_META_NAMES = re.compile(
    r"(generator|provenance|synthid|c2pa|content[-_ ]?credentials|"
    r"ai[-_ ]?(generated|assistant|model|tool|content)|created[-_ ]?by|"
    r"authoring[-_ ]?tool)",
    re.IGNORECASE,
)

# TIFF type -> byte size
_TIFF_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 9: 2, 10: 4, 11: 8, 12: 8}

# TIFF metadata / provenance tags to scrub (best-effort, in-place zeroing of
# inline values; overflow values left untouched — see honest limits).
_TIFF_DROP_TAGS = {
    0x010E,  # ImageDescription
    0x010F,  # Make
    0x0110,  # Model
    0x0131,  # Software
    0x0132,  # DateTime
    0x013B,  # Artist
    0x013D,  # HostComputer
    0x014A,  # Copyright
    0x0190,  # XPosition
    0x0191,  # YPosition
    0x02BC,  # XMLPacket (XMP)
    0x8649,  # IPTC
    0x8769,  # ExifIFDPointer
    0x8825,  # GPSInfo
    0xA005,  # InteroperabilityIFD
    0xA432,  # LensInfo
    0xA433,  # LensSpecification  (note: real tag is 0xA433? LensModel is 0xA434)
    0xA434,  # LensModel
    0xA437,  # LensSerialNumber
    0x0129,  # DocumentName
}


@dataclass
class MetaReport:
    format: str
    actions: list[str] = field(default_factory=list)
    removed_keys: list[str] = field(default_factory=list)
    removed_bytes: int = 0
    hard_bound_c2pa_present: bool = False
    cleaned: bytes | None = None

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "actions": self.actions,
            "removed_keys": self.removed_keys,
            "removed_bytes": self.removed_bytes,
            "hard_bound_c2pa_present": self.hard_bound_c2pa_present,
        }


def inspect(data: bytes, filename: str) -> dict:
    ext = Path(filename).suffix.lower().lstrip(".")
    report = _dispatch(data, ext, clean=False)
    return report.to_dict()


def clean(data: bytes, filename: str) -> tuple[bytes, dict]:
    ext = Path(filename).suffix.lower().lstrip(".")
    report = _dispatch(data, ext, clean=True)
    return report.cleaned if report.cleaned is not None else data, report.to_dict()


def verify_clean(data: bytes, filename: str) -> dict:
    """Clean a file, then VERIFY the result by re-inspecting it."""
    ext = Path(filename).suffix.lower().lstrip(".")
    before = _dispatch(data, ext, clean=False)
    cleaned, clean_actions = clean(data, filename)
    after = _dispatch(cleaned, ext, clean=False)

    c2pa_before = bool(before.hard_bound_c2pa_present)
    c2pa_after = bool(after.hard_bound_c2pa_present)
    if c2pa_before and not c2pa_after:
        verification = "verified_clear"
    elif c2pa_before and c2pa_after:
        verification = "residual_hard_bound"
    elif "unsupported_format" in clean_actions.get("actions", []):
        verification = "unsupported_format"
    else:
        verification = "no_c2pa_present"

    return {
        "format": ext or "unknown",
        "clean_actions": clean_actions.get("actions", []),
        "c2pa_before": c2pa_before,
        "c2pa_after": c2pa_after,
        "c2pa_cleared": c2pa_before and not c2pa_after,
        "c2pa_residual": c2pa_before and c2pa_after,
        "verification": verification,
    }


def _dispatch(data: bytes, ext: str, clean: bool) -> MetaReport:
    if ext == "png":
        return _png(data, clean)
    if ext in ("jpg", "jpeg"):
        return _jpeg(data, clean)
    if ext == "webp":
        return _webp(data, clean)
    if ext in ("avif", "heic"):
        return _isobmff(data, clean, ext)
    if ext == "svg":
        return _svg(data, clean)
    if ext == "bmp":
        return _bmp(data, clean)
    if ext == "gif":
        return _gif(data, clean)
    if ext in ("tiff", "tif"):
        return _tiff(data, clean)
    return MetaReport(format=ext or "unknown", actions=["unsupported_format"])


# ---------------------------------------------------------------- PNG
def _png(data: bytes, clean: bool) -> MetaReport:
    rep = MetaReport(format="png")
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        rep.actions.append("not_a_png")
        return rep
    out = io.BytesIO()
    out.write(data[:8])
    i = 8
    removed = 0
    while i + 8 <= len(data):
        length = int.from_bytes(data[i : i + 4], "big")
        ctype = data[i + 4 : i + 8]
        if i + 12 + length > len(data):
            break
        chunk = data[i : i + 12 + length]
        kind = ctype.decode("latin1", "replace")
        if kind == "eXIf" or (
            kind in ("iTXt", "zTXt", "tEXt")
            and (
                b"XML:com.adobe.xmp" in chunk
                or b"provenance" in chunk.lower()
                or b"c2pa" in chunk.lower()
                or b"ai" in chunk.lower()[:200]
            )
        ):
            removed += 12 + length
            if kind == "eXIf":
                rep.actions.append("removed_eXIf_EXIF_chunk")
            else:
                rep.actions.append(f"removed_{kind}_metadata_chunk")
            rep.removed_bytes = removed
        elif clean:
            out.write(chunk)
        i += 12 + length
    if clean:
        rep.cleaned = out.getvalue()
    if b"jumbf" in data.lower() or b"c2pa" in data.lower():
        rep.hard_bound_c2pa_present = True
        rep.actions.append("c2pa_jumbf_markers_detected_not_removed")
    return rep


# ---------------------------------------------------------------- JPEG
def _jpeg(data: bytes, clean: bool) -> MetaReport:
    rep = MetaReport(format="jpeg")
    if data[:2] != b"\xff\xd8":
        rep.actions.append("not_a_jpeg")
        return rep
    out = io.BytesIO()
    out.write(b"\xff\xd8")
    i = 2
    removed = 0
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            out.write(data[i : i + 2])
            i += 2
            continue
        seg_len = int.from_bytes(data[i + 2 : i + 4], "big")
        if seg_len < 2 or i + 2 + seg_len > len(data):
            break
        seg = data[i : i + 2 + seg_len]
        drop = False
        if marker == 0xE1:  # APP1: EXIF ("Exif\0\0") or XMP
            if seg[4:10] == b"Exif\x00\x00":
                drop = True
                rep.actions.append("removed_APP1_EXIF")
            elif seg[4:33] == b"http://ns.adobe.com/xap/1.0/\x00":
                drop = True
                rep.actions.append("removed_APP1_XMP")
        elif marker == 0xEB:  # APP11: Adobe XMP / AI hints
            if b"http://ns.adobe.com/xap" in seg or b"XML:com.adobe.xmp" in seg:
                drop = True
                rep.actions.append("removed_APP11_XMP")
            elif _AI_KEY_HINTS.search(seg[:512]):
                drop = True
                rep.actions.append("removed_APP11_ai_metadata")
        if drop:
            removed += 2 + seg_len
            rep.removed_bytes = removed
        elif clean:
            out.write(seg)
        i += 2 + seg_len
    if clean:
        if removed == 0:
            rep.cleaned = data
        else:
            out.write(data[i:] if i < len(data) else b"")
            rep.cleaned = out.getvalue()
    if b"c2pa" in data.lower() or b"jumbf" in data.lower():
        rep.hard_bound_c2pa_present = True
        rep.actions.append("c2pa_jumbf_markers_detected_not_removed")
    return rep


# ---------------------------------------------------------------- SVG
def _svg(data: bytes, clean: bool) -> MetaReport:
    rep = MetaReport(format="svg")
    text = data.decode("utf-8", "replace")
    new_text = re.sub(r"<metadata[\s\S]*?</metadata>", "", text, flags=re.IGNORECASE)
    new_text = re.sub(r"<rdf:RDF[\s\S]*?</rdf:RDF>", "", new_text, flags=re.IGNORECASE)

    def _strip_attrs(m):
        tag = m.group(0)
        for attr in ("data-ai-origin", "data-provenance", "data-ai-model", "content-credentials"):
            tag = re.sub(rf"\s{attr}='[^']*'", "", tag, flags=re.IGNORECASE)
            tag = re.sub(rf'\s{attr}="[^"]*"', "", tag, flags=re.IGNORECASE)
        return tag

    new_text = re.sub(r"<[a-zA-Z][^>]*>", _strip_attrs, new_text)
    if new_text != text:
        rep.actions.append("removed_svg_metadata_and_ai_attrs")
        rep.removed_bytes = len(text.encode()) - len(new_text.encode())
    if clean:
        rep.cleaned = new_text.encode("utf-8")
    return rep


# ---------------------------------------------------------------- AVIF / HEIC (ISOBMFF)
XMP_UUID = b"\xbe\x7a\xcf\xcb\x97\xa9\x42\xe8\x9c\x71\x99\x94\x91\xe3\xaf\xac"
_C2PA_BOXES = (b"jumb", b"c2pa")


def _iter_isobmff_boxes(data: bytes, start: int = 0):
    """Yield (fourcc, payload, header_size) for top-level ISOBMFF boxes."""
    i = start
    n = len(data)
    while i + 8 <= n:
        size = int.from_bytes(data[i : i + 4], "big")
        fourcc = data[i + 4 : i + 8]
        header = 8
        if size == 1:
            if i + 16 > n:
                break
            size = int.from_bytes(data[i + 8 : i + 16], "big")
            header = 16
        elif size == 0:
            size = n - i
        if size < header or i + size > n:
            break
        yield fourcc, data[i + header : i + size], header
        i += size


def _isobmff(data: bytes, clean: bool, fmt: str) -> MetaReport:
    rep = MetaReport(format=fmt)
    boxes = list(_iter_isobmff_boxes(data))
    if not boxes or boxes[0][0] != b"ftyp":
        rep.actions.append(f"not_an_{fmt}")
        return rep
    out = bytearray() if clean else None
    removed = 0

    def _box_bytes(fourcc: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload) + 8) + fourcc + payload

    for fourcc, payload, header in boxes:
        name = fourcc.decode("latin1", "replace")
        if fourcc in _C2PA_BOXES or name.lower().startswith("c2"):
            rep.actions.append(f"removed_top_level_{name}_c2pa_box")
            removed += header + len(payload)
            continue
        if fourcc == b"uuid":
            if payload.startswith(XMP_UUID):
                rep.actions.append(f"removed_top_level_{name}_xmp_uuid_box")
                removed += header + len(payload)
                continue
            if _AI_KEY_HINTS.search(payload[:512]):
                rep.actions.append(f"removed_top_level_{name}_ai_uuid_box")
                removed += header + len(payload)
                continue
        if fourcc == b"meta":
            verflags = payload[:4] if len(payload) >= 4 else b"\x00\x00\x00\x00"
            sub_clean = bytearray()
            sub_removed = 0
            for s_fourcc, s_payload, s_header in _iter_isobmff_boxes(payload, start=4):
                s_name = s_fourcc.decode("latin1", "replace")
                if s_fourcc in _C2PA_BOXES or s_name.lower().startswith("c2"):
                    rep.actions.append(f"removed_meta_subbox_{s_name}_c2pa")
                    sub_removed += s_header + len(s_payload)
                    continue
                if s_fourcc == b"uuid":
                    if s_payload.startswith(XMP_UUID):
                        rep.actions.append(f"removed_meta_subbox_{s_name}_xmp")
                        sub_removed += s_header + len(s_payload)
                        continue
                    if _AI_KEY_HINTS.search(s_payload[:512]):
                        rep.actions.append(f"removed_meta_subbox_{s_name}_ai_uuid")
                        sub_removed += s_header + len(s_payload)
                        continue
                if s_fourcc in (b"xml ", b"bxml") and _AI_KEY_HINTS.search(s_payload[:512]):
                    rep.actions.append(f"removed_meta_subbox_{s_name}_xml_metadata")
                    sub_removed += s_header + len(s_payload)
                    continue
                sub_clean.extend(_box_bytes(s_fourcc, s_payload))
            new_meta = verflags + bytes(sub_clean)
            if out is not None:
                out.extend(_box_bytes(b"meta", new_meta))
            removed += sub_removed
            continue
        if out is not None:
            out.extend(_box_bytes(fourcc, payload))
    if removed:
        rep.removed_bytes = removed
        if clean:
            rep.cleaned = bytes(out)
    else:
        if clean:
            rep.cleaned = data
        rep.actions.append(f"no_{fmt}_metadata_boxes_removed")
    if b"jumb" in data or b"c2pa" in data.lower():
        rep.hard_bound_c2pa_present = True
    return rep


# ---------------------------------------------------------------- WebP (RIFF)
def _webp(data: bytes, clean: bool) -> MetaReport:
    rep = MetaReport(format="webp")
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        rep.actions.append("not_a_webp")
        return rep
    riff_size = int.from_bytes(data[4:8], "little")
    if riff_size + 8 > len(data):
        rep.actions.append("truncated_webp")
        return rep
    out = bytearray(data[:12])
    i = 12
    removed = 0
    while i + 8 <= len(data):
        fourcc = data[i : i + 4]
        size = int.from_bytes(data[i + 4 : i + 8], "little")
        if i + 8 + size > len(data):
            break
        chunk = data[i : i + 8 + size]
        name = fourcc.decode("latin1", "replace")
        drop = False
        if fourcc in (b"EXIF", b"XMP ", b"C2PA"):
            drop = True
            rep.actions.append(f"removed_{name}_chunk")
        elif fourcc == b"ICCP" and _AI_KEY_HINTS.search(chunk[:512]):
            drop = True
            rep.actions.append("removed_ICCP_ai_profile")
        if drop:
            removed += 8 + size
        else:
            out.extend(chunk)
        i += 8 + size
    if removed:
        rep.removed_bytes = removed
        if clean:
            rep.cleaned = bytes(out)
    else:
        if clean:
            rep.cleaned = data
        rep.actions.append("no_webp_metadata_chunks_removed")
    if b"c2pa" in data.lower() or b"jumbf" in data.lower():
        rep.hard_bound_c2pa_present = True
    return rep


# ---------------------------------------------------------------- BMP
# Layout: BITMAPFILEHEADER(14) + DIB header + pixel data + [profile/extra].
# BMP carries metadata only in optional trailing sections (ICC profiles /
# EXIF / XMP). We strip those while preserving the pixel array.
def _bmp(data: bytes, clean: bool) -> MetaReport:
    rep = MetaReport(format="bmp")
    if len(data) < 18 or data[:2] != b"BM":
        rep.actions.append("not_a_bmp")
        return rep
    file_size = int.from_bytes(data[2:6], "little")
    pixel_offset = int.from_bytes(data[10:14], "little")
    if pixel_offset < 14 or pixel_offset > len(data):
        rep.actions.append("invalid_pixel_offset")
        return rep
    dib_size = int.from_bytes(data[14:18], "little") if len(data) >= 18 else 0
    header_size = 14 + dib_size
    if pixel_offset < header_size:
        rep.actions.append("invalid_bmp_layout")
        return rep

    trailing = data[pixel_offset:]
    has_trailing_meta = False
    for needle in (b"XML:com.adobe.xmp", b"Exif", b"c2pa", b"ICC_PROFILE"):
        if needle in trailing:
            has_trailing_meta = True
            rep.actions.append(f"detected_trailing_{needle!r}_in_bmp")

    if clean and has_trailing_meta:
        usable_len = file_size if 0 < file_size <= len(data) else len(data)
        rep.cleaned = data[:usable_len]
        rep.removed_bytes = len(data) - usable_len
    elif clean:
        rep.cleaned = data
        if not has_trailing_meta:
            rep.actions.append("no_bmp_metadata_removed")
    return rep


# ---------------------------------------------------------------- GIF
# GIF87a / GIF89a: logical screen descriptor + image data blocks, followed by
# extension blocks. We strip COMMENT (0xFF), PLAIN TEXT (0x01) and
# APPLICATION (0xFE, when it carries XMP) extensions while preserving image
# descriptors and raster data. Benign app extensions (e.g. Netscape looping)
# are kept.
#   block codes: 0x21=extension(introducer), 0x2C=image desc, 0x3B=trailer
_GIF_DROP_LABELS = {0xFF, 0x01, 0xFE}  # COMMENT, PLAIN TEXT, APPLICATION


def _gif(data: bytes, clean: bool) -> MetaReport:
    rep = MetaReport(format="gif")
    if data[:6] not in (b"GIF87a", b"GIF89a"):
        rep.actions.append("not_a_gif")
        return rep
    out = bytearray(data[:6])
    removed = 0
    i = 6
    n = len(data)
    while i < n:
        block = data[i]
        if block == 0x21:  # extension introducer
            label = data[i + 1] if i + 1 < n else 0
            seg_start = i
            i += 2
            sub = bytearray()
            while i < n and data[i] != 0:
                sz = data[i]
                sub.extend(data[i : i + 1 + sz])
                i += 1 + sz
            if i < n:
                i += 1  # block terminator
            if label in _GIF_DROP_LABELS:
                if label == 0xFE and b"XMP " not in sub:
                    # benign app extension (Netscape looping) — keep
                    if clean:
                        out.extend(data[seg_start:i])
                    continue
                removed += i - seg_start
                rep.actions.append(f"removed_gif_ext_label_{label:#x}")
            else:
                if clean:
                    out.extend(data[seg_start:i])
        elif block == 0x2C:  # image descriptor
            end = i + 10
            if end > n:
                if clean:
                    out.extend(data[i:])
                break
            packed = data[end - 1]
            if packed & 0x80:
                size = 2 ** ((packed & 0x07) + 1) * 3
                end += size
            if end < n:
                end += 1  # LZW min code size
            else:
                if clean:
                    out.extend(data[i:])
                break
            j = end
            while j < n and data[j] != 0:
                sz = data[j]
                j += 1 + sz
            if j < n:
                j += 1
            if clean:
                out.extend(data[i:j])
            i = j
        elif block == 0x3B:  # trailer
            if clean:
                out.extend(data[i:])
            break
        else:
            if clean:
                out.append(block)
            i += 1
    if removed:
        rep.removed_bytes = removed
        if clean:
            rep.cleaned = bytes(out)
    else:
        if clean:
            rep.cleaned = data
        rep.actions.append("no_gif_metadata_removed")
    if b"c2pa" in data.lower() or b"jumbf" in data.lower():
        rep.hard_bound_c2pa_present = True
    return rep


# ---------------------------------------------------------------- TIFF
def _tiff(data: bytes, clean: bool) -> MetaReport:
    rep = MetaReport(format="tiff")
    if len(data) < 8:
        rep.actions.append("not_a_tiff")
        return rep
    bo = data[:2]
    if bo == b"II":
        endian = "<"
    elif bo == b"MM":
        endian = ">"
    else:
        rep.actions.append("not_a_tiff")
        return rep
    magic = struct.unpack(endian + "H", data[2:4])[0]
    if magic != 42:
        rep.actions.append("not_a_tiff")
        return rep
    ifd0 = struct.unpack(endian + "I", data[4:8])[0]
    if ifd0 == 0 or ifd0 > len(data) - 2:
        rep.actions.append("invalid_tiff_ifd_offset")
        return rep

    # Collect every IFD and its entries so we can scrub metadata tags.
    ifds = []  # (ifd_offset, [(tag, typ, count, value), ...])
    cur = ifd0
    while cur and cur + 2 <= len(data) and len(ifds) < 64:
        n = struct.unpack(endian + "H", data[cur : cur + 2])[0]
        entries_start = cur + 2
        entries = []
        ok = True
        for e in range(n):
            off = entries_start + e * 12
            if off + 12 > len(data):
                ok = False
                break
            tag, typ, count = struct.unpack(endian + "HHI", data[off : off + 8])
            value = data[off + 8 : off + 12]
            entries.append((tag, typ, count, value))
        if not ok:
            break
        ifds.append((cur, entries))
        next_ptr_off = entries_start + n * 12
        if next_ptr_off + 4 > len(data):
            break
        nxt = struct.unpack(endian + "I", data[next_ptr_off : next_ptr_off + 4])[0]
        cur = nxt
        if cur == 0 or cur < ifd0:
            break

    drop_count = 0
    for (_ifd_off, entries) in ifds:
        for (tag, _typ, _count, _value) in entries:
            if tag in _TIFF_DROP_TAGS:
                drop_count += 1
                rep.removed_keys.append(f"tiff_tag_{tag:#06x}")
                rep.actions.append(f"removed_tiff_tag_{tag:#06x}")

    if not clean:
        return rep

    # Best-effort in-place scrub of inline (<=4 byte) metadata tag values.
    out = bytearray(data)
    total_removed = 0
    for (ifd_off, entries) in ifds:
        n = len(entries)
        entries_start = ifd_off + 2
        for idx, (tag, typ, count, _value) in enumerate(entries):
            if tag not in _TIFF_DROP_TAGS:
                continue
            eoff = entries_start + idx * 12
            sz = _TIFF_TYPE_SIZES.get(typ, 1) * count
            if sz <= 4:
                out[eoff + 8 : eoff + 12] = b"\x00" * 4
                total_removed += sz
    if drop_count:
        rep.removed_bytes = total_removed
        rep.cleaned = bytes(out)
    else:
        rep.cleaned = data
        rep.actions.append("no_tiff_metadata_removed")
    if b"c2pa" in data.lower() or b"jumbf" in data.lower():
        rep.hard_bound_c2pa_present = True
    return rep
