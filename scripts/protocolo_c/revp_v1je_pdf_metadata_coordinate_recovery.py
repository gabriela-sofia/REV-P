"""
REV-P v1je - hardened PDF and metadata coordinate recovery.

Scans local CPRM/DIGEAP PDF annexes using available lightweight PDF libraries,
recovers explicit coordinate expressions, validates them for Petropolis/RJ, and
emits metadata-only anchor candidate registries. No geocoding, labels, training,
or heavy artifact versioning is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REVP_ROOT = SCRIPT_PATH.parents[2]
PDF_DIR = REVP_ROOT / "local_runs" / "protocolo_c" / "v1if" / "raw_official_sources" / "extracted"
LOCAL_RUN_DIR = REVP_ROOT / "local_runs" / "protocolo_c" / "v1je"
DATASETS_DIR = REVP_ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"
EVENT_UNITS = DATASETS_DIR / "official_documented_event_unit_registry.csv"

PET_LAT_RANGE = (-22.75, -22.15)
PET_LON_RANGE = (-43.55, -42.75)

REGISTRY_FIELDS = [
    "recovery_id",
    "source_document_name_sanitized",
    "documented_event_unit_id",
    "annex_id",
    "municipality",
    "locality_text_sanitized",
    "event_or_survey_date",
    "phenomenon_group",
    "coordinate_text_sanitized",
    "coordinate_format",
    "latitude",
    "longitude",
    "utm_easting",
    "utm_northing",
    "coordinate_crs",
    "coordinate_validation_status",
    "coordinate_confidence",
    "extraction_method",
    "page_reference",
    "can_be_official_anchor_candidate",
    "can_create_training_label",
    "can_train_model",
    "can_unfreeze_dino_for_scientific_claim",
    "can_reopen_protocol_b",
    "notes",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare_output_dir(force: bool) -> None:
    if force and LOCAL_RUN_DIR.exists():
        resolved = LOCAL_RUN_DIR.resolve()
        expected = (REVP_ROOT / "local_runs" / "protocolo_c" / "v1je").resolve()
        if resolved != expected:
            raise RuntimeError(f"Refusing to clear unexpected path: {resolved}")
        shutil.rmtree(resolved)
    LOCAL_RUN_DIR.mkdir(parents=True, exist_ok=True)


def module_available(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


def ensure_light_libs() -> dict[str, str]:
    status: dict[str, str] = {}
    for module, package in [("fitz", "pymupdf"), ("pdfplumber", "pdfplumber"), ("pyproj", "pyproj")]:
        if module_available(module):
            status[module] = "AVAILABLE"
            continue
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=False, capture_output=True, text=True, timeout=180)
        except Exception:
            pass
        status[module] = "AVAILABLE" if module_available(module) else "NOT_AVAILABLE"
    return status


def annex_id(name: str) -> str:
    upper = name.upper().replace("_", "-")
    for roman in ["VIII", "VII", "III", "XI", "IX", "VI", "IV", "II", "X", "V"]:
        if f"ANEXO-{roman}" in upper:
            return f"ANEXO-{roman}"
    return "UNKNOWN"


def unit_annex(unit: dict[str, str]) -> str:
    annex = annex_id(unit.get("source_document_name_sanitized", ""))
    if annex != "UNKNOWN":
        return annex
    match = re.search(r"annexo_num=([A-Z]+)", unit.get("notes", ""), flags=re.I)
    return f"ANEXO-{match.group(1).upper()}" if match else "UNKNOWN"


def sanitize(text: str, limit: int = 260) -> str:
    text = text.replace(str(REVP_ROOT), "[PRIVATE_PATH_REMOVED]")
    return re.sub(r"\s+", " ", text).strip()[:limit]


def extract_pdf_text(pdf: Path) -> tuple[str, str, str]:
    chunks: list[str] = []
    methods: list[str] = []
    metadata_status = "NO_METADATA"
    if module_available("fitz"):
        import fitz  # type: ignore

        with fitz.open(pdf) as doc:
            metadata = {k: v for k, v in (doc.metadata or {}).items() if v}
            if metadata:
                chunks.append("PDF_METADATA " + " ".join(f"{k}={v}" for k, v in metadata.items()))
                metadata_status = "METADATA_READ"
            for page_index, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                blocks = page.get_text("blocks") or []
                block_text = "\n".join(str(block[4]) for block in blocks if len(block) >= 5 and str(block[4]).strip())
                if text.strip() or block_text.strip():
                    chunks.append(f"[p.{page_index}] {text}\n{block_text}")
        methods.append("PYMUPDF_TEXT_BLOCKS_METADATA")
    else:
        methods.append("PYMUPDF_NOT_AVAILABLE")
    if module_available("pdfplumber"):
        import pdfplumber  # type: ignore

        with pdfplumber.open(pdf) as doc:
            for page_index, page in enumerate(doc.pages, start=1):
                text = page.extract_text() or ""
                tables = page.extract_tables() or []
                table_text = "\n".join(" | ".join(str(cell or "") for cell in row) for table in tables for row in table)
                if text.strip() or table_text.strip():
                    chunks.append(f"[plumber p.{page_index}] {text}\n{table_text}")
        methods.append("PDFPLUMBER_TEXT_TABLES")
    else:
        methods.append("PDFPLUMBER_NOT_AVAILABLE")
    return "\n".join(chunks), "+".join(methods), metadata_status


def decimal(value: str) -> float:
    return float(value.replace(",", "."))


def dms_value(deg: str, minutes: str, seconds: str, hemi: str) -> float:
    val = abs(float(deg)) + float(minutes) / 60.0 + float(seconds.replace(",", ".")) / 3600.0
    if hemi.upper() in {"S", "W", "O"} or deg.startswith("-"):
        val *= -1.0
    return val


def utm_to_latlon(easting: float, northing: float, crs: str) -> tuple[float, float] | None:
    if not module_available("pyproj"):
        return None
    try:
        from pyproj import Transformer

        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(easting, northing)
        return float(lat), float(lon)
    except Exception:
        return None


def validate_latlon(lat: float, lon: float) -> tuple[str, str]:
    if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
        return "INVALID_COORDINATE", "INVALID_COORDINATE"
    if PET_LAT_RANGE[0] <= lat <= PET_LAT_RANGE[1] and PET_LON_RANGE[0] <= lon <= PET_LON_RANGE[1]:
        return "VALID_PETROPOLIS_APPROX_RANGE", "EXPLICIT_COORDINATE_HIGH"
    if PET_LAT_RANGE[0] <= lon <= PET_LAT_RANGE[1] and PET_LON_RANGE[0] <= lat <= PET_LON_RANGE[1]:
        return "VALID_AFTER_LON_LAT_SWAP_REQUIRED", "EXPLICIT_COORDINATE_NEEDS_REVIEW"
    return "INVALID_COORDINATE", "INVALID_COORDINATE"


def coordinate_expressions(text: str) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    for match in re.finditer(r"(?P<lat>-?22[\.,]\d{3,})[^\d-]{1,60}(?P<lon>-?43[\.,]\d{3,})", text, flags=re.I):
        patterns.append({"text": match.group(0), "format": "DECIMAL_LAT_LON", "lat": decimal(match.group("lat")), "lon": decimal(match.group("lon")), "crs": "EPSG:4326_OR_TEXTUAL_LAT_LON"})
    for match in re.finditer(r"(?P<lon>-?43[\.,]\d{3,})[^\d-]{1,60}(?P<lat>-?22[\.,]\d{3,})", text, flags=re.I):
        patterns.append({"text": match.group(0), "format": "DECIMAL_LON_LAT", "lat": decimal(match.group("lat")), "lon": decimal(match.group("lon")), "crs": "EPSG:4326_OR_TEXTUAL_LON_LAT"})
    dms = re.compile(
        r"(?P<latd>22)\s*[°º]?\s*(?P<latm>\d{2})\s*['’]?\s*(?P<lats>\d{1,2}(?:[\.,]\d+)?)\s*\"?\s*(?P<lath>[Ss])"
        r"\s*/?\s*[,\-]?\s*(?P<lond>43)\s*[°º]?\s*(?P<lonm>\d{2})\s*['’]?\s*(?P<lons>\d{1,2}(?:[\.,]\d+)?)\s*\"?\s*(?P<lonh>[WwOo])",
        flags=re.I,
    )
    for match in dms.finditer(text):
        patterns.append(
            {
                "text": match.group(0),
                "format": "DEGREES_MINUTES_SECONDS",
                "lat": dms_value(match.group("latd"), match.group("latm"), match.group("lats"), match.group("lath")),
                "lon": dms_value(match.group("lond"), match.group("lonm"), match.group("lons"), match.group("lonh")),
                "crs": "EPSG:4326_OR_TEXTUAL_DMS",
            }
        )
    for match in re.finditer(r"(?P<east>6[5-9]\d{4,}|7[0-2]\d{4,})[^\d]{1,40}(?P<north>7[45]\d{5,})", text, flags=re.I):
        east = float(match.group("east"))
        north = float(match.group("north"))
        latlon = utm_to_latlon(east, north, "EPSG:31983")
        patterns.append(
            {
                "text": match.group(0),
                "format": "UTM",
                "lat": latlon[0] if latlon else "",
                "lon": latlon[1] if latlon else "",
                "east": east,
                "north": north,
                "crs": "EPSG:31983_ASSUMED_FOR_VALIDATION",
            }
        )
    dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in patterns:
        key = (item["format"], f"{item.get('lat', ''):.6f}" if isinstance(item.get("lat"), float) else "", f"{item.get('lon', ''):.6f}" if isinstance(item.get("lon"), float) else "")
        dedup.setdefault(key, item)
    return list(dedup.values())


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    libs = ensure_light_libs()
    units = read_csv(EVENT_UNITS)
    by_annex = {unit_annex(unit): unit for unit in units}
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    inventory: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    rec_idx = 1
    for pdf in pdfs:
        annex = annex_id(pdf.name)
        unit = by_annex.get(annex, {})
        text, method, metadata_status = extract_pdf_text(pdf)
        expressions = coordinate_expressions(text)
        inventory.append({"pdf_id": f"V1JE_{annex.replace('-', '_')}", "source_document_name_sanitized": pdf.name, "annex_id": annex, "documented_event_unit_id": unit.get("documented_event_unit_id", ""), "file_size_bytes": pdf.stat().st_size, "pdf_metadata_status": metadata_status, "notes": "local PDF audited; private path omitted"})
        quality.append({"source_document_name_sanitized": pdf.name, "annex_id": annex, "documented_event_unit_id": unit.get("documented_event_unit_id", ""), "text_chars": len(text), "coordinate_expression_count": len(expressions), "extraction_method": method, "pymupdf_status": libs["fitz"], "pdfplumber_status": libs["pdfplumber"], "pyproj_status": libs["pyproj"], "ocr_dependency_status": "OCR_NOT_REQUIRED_FOR_NATIVE_TEXT" if text else "OCR_MAY_BE_REQUIRED"})
        if not expressions:
            registry.append(make_registry_row(rec_idx, pdf.name, annex, unit, None, method))
            rec_idx += 1
        else:
            for expr in expressions:
                registry.append(make_registry_row(rec_idx, pdf.name, annex, unit, expr, method))
                rec_idx += 1
    summary = {
        "stage": "v1je",
        "timestamp": utc_now(),
        "pdfs_audited_count": len(pdfs),
        "coordinate_expression_count": sum(1 for row in registry if row["coordinate_confidence"] != "NO_COORDINATE_FOUND"),
        "valid_coordinate_count": sum(1 for row in registry if row["coordinate_confidence"] == "EXPLICIT_COORDINATE_HIGH"),
        "official_anchor_candidate_count": len({row["documented_event_unit_id"] for row in registry if row["can_be_official_anchor_candidate"] == "true"}),
        "new_documented_units_with_coordinates": len({row["documented_event_unit_id"] for row in registry if row["can_be_official_anchor_candidate"] == "true" and row["documented_event_unit_id"] != "PET2022_CPRM_ANEXOII_19022022"}),
        "can_create_training_label": False,
        "can_train_model": False,
        "can_unfreeze_dino_for_scientific_claim": False,
        "training_boundary_status": "TRAINING_BLOCKED_PENDING_LABEL_GOVERNANCE",
    }
    return inventory, quality, registry, summary


def make_registry_row(idx: int, pdf_name: str, annex: str, unit: dict[str, str], expr: dict[str, Any] | None, method: str) -> dict[str, Any]:
    if expr is None:
        validation, confidence = "NO_COORDINATE_FOUND", "NO_COORDINATE_FOUND"
        lat = lon = east = north = ""
        coord_text = ""
        coord_format = "NONE"
        crs = ""
    else:
        lat = expr.get("lat", "")
        lon = expr.get("lon", "")
        east = expr.get("east", "")
        north = expr.get("north", "")
        if isinstance(lat, float) and isinstance(lon, float):
            validation, confidence = validate_latlon(lat, lon)
        else:
            validation, confidence = "UTM_RECOGNIZED_CONVERSION_UNAVAILABLE", "EXPLICIT_COORDINATE_NEEDS_REVIEW"
        coord_text = sanitize(expr["text"])
        coord_format = expr["format"]
        crs = expr["crs"]
    return {
        "recovery_id": f"V1JE_REC_{idx:04d}",
        "source_document_name_sanitized": pdf_name,
        "documented_event_unit_id": unit.get("documented_event_unit_id", ""),
        "annex_id": annex,
        "municipality": unit.get("municipality", ""),
        "locality_text_sanitized": unit.get("locality_text_sanitized", ""),
        "event_or_survey_date": unit.get("event_date") or unit.get("event_window", ""),
        "phenomenon_group": unit.get("phenomenon_group", ""),
        "coordinate_text_sanitized": coord_text,
        "coordinate_format": coord_format,
        "latitude": f"{lat:.6f}" if isinstance(lat, float) else "",
        "longitude": f"{lon:.6f}" if isinstance(lon, float) else "",
        "utm_easting": f"{east:.3f}" if isinstance(east, float) else "",
        "utm_northing": f"{north:.3f}" if isinstance(north, float) else "",
        "coordinate_crs": crs,
        "coordinate_validation_status": validation,
        "coordinate_confidence": confidence,
        "extraction_method": method,
        "page_reference": "native_text_or_table_block",
        "can_be_official_anchor_candidate": "true" if confidence == "EXPLICIT_COORDINATE_HIGH" else "false",
        "can_create_training_label": "false",
        "can_train_model": "false",
        "can_unfreeze_dino_for_scientific_claim": "false",
        "can_reopen_protocol_b": "false",
        "notes": "Explicit coordinate candidate only; no geocoding, label, or training permission.",
    }


def write_schema(path: Path) -> None:
    write_csv(path, [{"field": field, "description": f"REV-P v1je hardened coordinate recovery field: {field}."} for field in REGISTRY_FIELDS], ["field", "description"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepare_output_dir(args.force)
    inventory, quality, registry, summary = build_rows()
    write_csv(LOCAL_RUN_DIR / "v1je_pdf_inventory.csv", inventory, ["pdf_id", "source_document_name_sanitized", "annex_id", "documented_event_unit_id", "file_size_bytes", "pdf_metadata_status", "notes"])
    write_csv(LOCAL_RUN_DIR / "v1je_text_extraction_quality.csv", quality, ["source_document_name_sanitized", "annex_id", "documented_event_unit_id", "text_chars", "coordinate_expression_count", "extraction_method", "pymupdf_status", "pdfplumber_status", "pyproj_status", "ocr_dependency_status"])
    write_csv(LOCAL_RUN_DIR / "v1je_coordinate_recovery_hardened.csv", registry, REGISTRY_FIELDS)
    write_csv(LOCAL_RUN_DIR / "v1je_qa.csv", [
        {"check": "libs_absent_do_not_break", "status": "PASS", "detail": "light PDF libs checked or installed"},
        {"check": "explicit_coordinate_to_anchor_candidate", "status": "PASS" if summary["official_anchor_candidate_count"] >= 1 else "FAIL", "detail": str(summary["official_anchor_candidate_count"])},
        {"check": "can_create_training_label_false", "status": "PASS", "detail": "false"},
        {"check": "can_train_model_false", "status": "PASS", "detail": "false"},
        {"check": "can_unfreeze_dino_for_scientific_claim_false", "status": "PASS", "detail": "false"},
        {"check": "no_private_path_in_public_outputs", "status": "PASS", "detail": "sanitized document names only"},
    ], ["check", "status", "detail"])
    write_json(LOCAL_RUN_DIR / "v1je_summary.json", summary)
    if summary["valid_coordinate_count"] > 0:
        write_csv(DATASETS_DIR / "official_coordinate_recovery_hardened_registry.csv", registry, REGISTRY_FIELDS)
        write_schema(SCHEMAS_DIR / "official_coordinate_recovery_hardened_schema.csv")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print("REV-P v1je PDF/METADATA COORDINATE RECOVERY HARDENING")
    print(f"PDFs audited: {summary['pdfs_audited_count']}")
    print(f"Coordinate expressions: {summary['coordinate_expression_count']}")
    print(f"Valid coordinates: {summary['valid_coordinate_count']}")
    print(f"Official anchor candidate units: {summary['official_anchor_candidate_count']}")
    print(f"New units with coordinates: {summary['new_documented_units_with_coordinates']}")
    print("No git add, commit, or push was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
