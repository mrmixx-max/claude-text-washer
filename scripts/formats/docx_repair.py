"""DOCX repair — rebuild corrupted DOCX files after cleaning/metadata stripping.

Adapted from TWS ``metadata/docx_repair.py``.

DOCX files are ZIP archives containing XML parts. After watermark cleaning or
metadata stripping, the file can become corrupted:

- Missing or broken relationships (_rels/.rels)
- Missing [Content_Types].xml entries
- Orphaned parts referenced in relationships but missing from archive
- Invalid XML in document parts
- Missing required parts (document.xml, styles.xml, etc.)

This module repairs these issues by:
1. Validating the ZIP structure
2. Rebuilding [Content_Types].xml from actual archive contents
3. Rebuilding _rels/.rels with correct relationships
4. Removing orphaned relationship entries
5. Validating XML well-formedness of all parts
6. Re-zipping with correct structure

Honest boundaries:
- Repair can fix STRUCTURAL issues but cannot recover CONTENT that was deleted.
- The repair is lossy for custom XML parts not in the standard schema.
- A repaired DOCX may differ byte-for-byte — functional reconstruction, not
  byte-exact restoration.
"""
from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field

# Required parts for a minimal valid DOCX
REQUIRED_PARTS = [
    "word/document.xml",
]

# Standard content types for DOCX parts
CONTENT_TYPES = {
    ".xml": "application/xml",
    ".rels": "application/vnd.openxmlformats-package.relationships+xml",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".emf": "image/x-emf",
    ".wmf": "image/x-wmf",
    ".tiff": "image/tiff",
    ".bmp": "image/bmp",
    ".ico": "image/x-icon",
    ".svg": "image/svg+xml",
}

OVERRIDE_TYPES = {
    "word/document.xml": "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml",
    "word/styles.xml": "application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml",
    "word/settings.xml": "application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml",
    "word/fontTable.xml": "application/vnd.openxmlformats-officedocument.wordprocessingml.fontTable+xml",
    "word/webSettings.xml": "application/vnd.openxmlformats-officedocument.wordprocessingml.webSettings+xml",
    "word/theme/theme1.xml": "application/vnd.openxmlformats-officedocument.theme+xml",
    "word/numbering.xml": "application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml",
    "word/footnotes.xml": "application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml",
    "word/endnotes.xml": "application/vnd.openxmlformats-officedocument.wordprocessingml.endnotes+xml",
    "docProps/core.xml": "application/vnd.openxmlformats-package.core-properties+xml",
    "docProps/app.xml": "application/vnd.openxmlformats-officedocument.extended-properties+xml",
    "docProps/custom.xml": "application/vnd.openxmlformats-officedocument.custom-properties+xml",
    "[Content_Types].xml": "application/vnd.openxmlformats-package.core-properties+xml",
}


@dataclass
class RepairReport:
    """Report of what was repaired."""

    was_valid: bool = False
    repairs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    parts_before: int = 0
    parts_after: int = 0
    data: bytes | None = None

    def to_dict(self) -> dict:
        return {
            "was_valid": self.was_valid,
            "repairs": self.repairs,
            "warnings": self.warnings,
            "errors": self.errors,
            "parts_before": self.parts_before,
            "parts_after": self.parts_after,
            "repaired": self.data is not None,
        }


def _is_valid_xml(data: bytes) -> bool:
    """Check if bytes are valid XML."""
    try:
        ET.fromstring(data)
        return True
    except ET.ParseError:
        return False


def _guess_content_type(part_name: str) -> str:
    """Guess the content type for a DOCX part."""
    if part_name in OVERRIDE_TYPES:
        return OVERRIDE_TYPES[part_name]
    ext = "." + part_name.rsplit(".", 1)[-1] if "." in part_name else ""
    return CONTENT_TYPES.get(ext, "application/octet-stream")


def _build_content_types_xml(parts: list[str]) -> bytes:
    """Build a [Content_Types].xml from the list of parts in the archive."""
    root = ET.Element("Types")
    root.set("xmlns", "http://schemas.openxmlformats.org/package/2006/content-types")
    defaults = [
        ("rels", "application/vnd.openxmlformats-package.relationships+xml"),
        ("xml", "application/xml"),
    ]
    for ext, ctype in defaults:
        d = ET.SubElement(root, "Default")
        d.set("Extension", ext)
        d.set("ContentType", ctype)
    for part in parts:
        if part in OVERRIDE_TYPES:
            o = ET.SubElement(root, "Override")
            o.set("PartName", "/" + part)
            o.set("ContentType", OVERRIDE_TYPES[part])
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _build_rels_xml(relationships: list[dict]) -> bytes:
    """Build a _rels file XML from relationship entries."""
    root = ET.Element("Relationships")
    root.set("xmlns", "http://schemas.openxmlformats.org/package/2006/relationships")
    for rel in relationships:
        r = ET.SubElement(root, "Relationship")
        r.set("Id", rel.get("Id", "rId1"))
        r.set("Type", rel.get("Type", ""))
        r.set("Target", rel.get("Target", ""))
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _parse_rels_xml(data: bytes) -> list[dict]:
    """Parse a .rels file into a list of relationship dicts."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return []
    rels = []
    for elem in root:
        if elem.tag.endswith("}Relationship") or elem.tag == "Relationship":
            rels.append(
                {
                    "Id": elem.get("Id", ""),
                    "Type": elem.get("Type", ""),
                    "Target": elem.get("Target", ""),
                    "TargetMode": elem.get("TargetMode", "Internal"),
                },
            )
    return rels


def validate_docx(data: bytes) -> RepairReport:
    """Validate a DOCX file and return a report of issues found."""
    report = RepairReport()
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        report.errors.append(f"not_a_valid_zip: {e}")
        return report
    parts = zf.namelist()
    report.parts_before = len(parts)
    for req in REQUIRED_PARTS:
        if req not in parts:
            report.errors.append(f"missing_required_part: {req}")
    if "[Content_Types].xml" not in parts:
        report.errors.append("missing_content_types")
    if "_rels/.rels" not in parts:
        report.warnings.append("missing_root_rels")
    for part in parts:
        if part.endswith((".xml", ".rels")):
            try:
                content = zf.read(part)
                if not _is_valid_xml(content):
                    report.errors.append(f"invalid_xml: {part}")
            except (OSError, KeyError, ValueError) as e:
                report.errors.append(f"read_error: {part}: {e}")
    if "_rels/.rels" in parts:
        try:
            rels_data = zf.read("_rels/.rels")
            rels = _parse_rels_xml(rels_data)
            for rel in rels:
                target = rel.get("Target", "").lstrip("/")
                if rel.get("TargetMode", "Internal") == "Internal" and target:
                    if target and target not in parts:
                        report.warnings.append(f"orphaned_rel: {target}")
        except (OSError, KeyError, ValueError) as e:
            report.warnings.append(f"rels_parse_error: {e}")
    report.was_valid = len(report.errors) == 0
    return report


def repair_docx(data: bytes) -> RepairReport:
    """Repair a corrupted DOCX file.

    Rebuilds the ZIP structure, content types, and relationships.
    Returns a RepairReport with the repaired data (if successful).
    """
    report = RepairReport()
    validation = validate_docx(data)
    if validation.was_valid:
        report.was_valid = True
        report.data = data
        report.parts_before = validation.parts_before
        report.parts_after = validation.parts_before
        return report
    report.errors = validation.errors
    report.warnings = validation.warnings
    report.parts_before = validation.parts_before

    try:
        zin = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        report.errors.append(f"cannot_open_zip: {e}")
        return report

    parts = zin.namelist()
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        valid_parts = []
        for part in parts:
            try:
                content = zin.read(part)
                if (part.endswith((".xml", ".rels"))) and not _is_valid_xml(content):
                    report.repairs.append(f"skipped_invalid_xml: {part}")
                    continue
                zout.writestr(part, content)
                valid_parts.append(part)
            except (OSError, KeyError, ValueError) as e:
                report.repairs.append(f"skipped_corrupt_part: {part}: {e}")

        if "[Content_Types].xml" not in valid_parts:
            ct_xml = _build_content_types_xml(valid_parts)
            zout.writestr("[Content_Types].xml", ct_xml)
            valid_parts.append("[Content_Types].xml")
            report.repairs.append("rebuilt_content_types")

        if "_rels/.rels" not in valid_parts:
            rels = []
            if "word/document.xml" in valid_parts:
                rels.append(
                    {
                        "Id": "rId1",
                        "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument",
                        "Target": "word/document.xml",
                    },
                )
            if "docProps/core.xml" in valid_parts:
                rels.append(
                    {
                        "Id": "rId2",
                        "Type": "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties",
                        "Target": "docProps/core.xml",
                    },
                )
            rels_xml = _build_rels_xml(rels)
            zout.writestr("_rels/.rels", rels_xml)
            valid_parts.append("_rels/.rels")
            report.repairs.append("rebuilt_root_rels")

        if "word/document.xml" in valid_parts and "word/_rels/document.xml.rels" not in valid_parts:
            rels_xml = _build_rels_xml([])
            zout.writestr("word/_rels/document.xml.rels", rels_xml)
            valid_parts.append("word/_rels/document.xml.rels")
            report.repairs.append("created_document_rels")

    report.parts_after = len(valid_parts)
    report.data = out.getvalue()
    return report


def rebuild_document_rels(data: bytes) -> bytes:
    """Rebuild only the document relationships file.

    Useful when relationships are broken but the rest of the DOCX is intact.
    """
    try:
        zin = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return data
    parts = zin.namelist()
    if "word/document.xml" not in parts:
        return data

    existing_rels = []
    rels_path = "word/_rels/document.xml.rels"
    if rels_path in parts:
        try:
            existing_rels = _parse_rels_xml(zin.read(rels_path))
        except (OSError, KeyError, ValueError):
            existing_rels = []

    valid_rels = []
    for rel in existing_rels:
        target = rel.get("Target", "").lstrip("/")
        if target in parts:
            valid_rels.append(rel)

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for part in parts:
            if part == rels_path:
                continue
            zout.writestr(part, zin.read(part))
        zout.writestr(rels_path, _build_rels_xml(valid_rels))
    return out.getvalue()
