"""Shared helpers for REV-P Protocol C v1ra-v1rf external intake skeleton.

This block builds the scaffolding to intake external documents collected
MANUALLY. It NEVER downloads anything, never uses the internet, and never
creates labels, targets, or operational ground truth. Everything is
review-only and waits for human-collected documents.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from revp_v1qu_v1qz_ground_reference_common import (  # noqa: F401
    DATASETS,
    DOCS,
    SCHEMAS,
    _p,
    assert_clean_rows,
    classify_source_family,
    detect_absolute_path,
    detect_local_runs_exposure,
    guardrail_row,
    hash_short,
    mask_path,
    normalize_patch_id,
    normalize_region,
    read_csv_safe,
    write_csv_with_header,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

# Common intake schema shared by template / validator / candidate builder
INTAKE_FIELDS = [
    "document_id", "source_name", "source_family", "region", "hazard_type",
    "event_date_text", "event_location_text", "url_or_reference",
    "local_document_hash", "access_date", "license_note", "evidence_type",
    "temporal_precision_claim", "spatial_precision_claim",
    "reviewer_notes", "intake_status",
]

_TRACKING_PARAM_RE = re.compile(r"(?i)[?&](utm_[^=&]+|fbclid|gclid)=[^&]*")
_URL_RE = re.compile(r"(?i)^[a-z][a-z0-9+.\-]*://")


def normalize_url(raw: str) -> str:
    """Normalize a URL reference WITHOUT contacting it. Strips tracking params."""
    s = str(raw or "").strip()
    if not s:
        return ""
    s = _TRACKING_PARAM_RE.sub("", s)
    s = s.rstrip("/?&#")
    return s


def is_url(raw: str) -> bool:
    return bool(_URL_RE.match(str(raw or "").strip()))


def normalize_source_name(raw: str) -> str:
    s = re.sub(r"\s+", " ", str(raw or "").strip())
    return s or "UNKNOWN_SOURCE_NAME"


_DOC_TYPE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("PDF_REPORT", [".pdf", "relatorio", "relatório", "boletim", "report"]),
    ("SPREADSHEET", [".csv", ".xlsx", ".xls", "planilha", "serie", "série"]),
    ("GEOSPATIAL", [".shp", ".geojson", ".kml", ".gpkg", "shapefile", "geodata"]),
    ("OFFICIAL_GAZETTE", ["diario oficial", "diário oficial", "decreto", "portaria"]),
    ("WEB_PAGE", ["http://", "https://", "html", "noticia", "notícia"]),
    ("IMAGE", [".png", ".jpg", ".jpeg", ".tif", ".tiff", "foto", "imagem"]),
]


def detect_document_type(reference: str, evidence_type: str = "") -> str:
    s = (str(reference or "") + " " + str(evidence_type or "")).lower()
    for dtype, kws in _DOC_TYPE_KEYWORDS:
        if any(kw in s for kw in kws):
            return dtype
    return "UNKNOWN_DOCUMENT_TYPE"


def validate_license_access(license_note: str) -> tuple[str, bool]:
    """Return (license_status, is_usable_for_review_only)."""
    s = str(license_note or "").strip().lower()
    if not s or s in ("unknown", "n/a", "nao verificado", "não verificado"):
        return ("LICENSE_NOT_VERIFIED", False)
    if any(k in s for k in ("public", "cc0", "cc-by", "cc by", "open", "dominio publico", "domínio público", "aberto")):
        return ("LICENSE_OPEN", True)
    if any(k in s for k in ("restrict", "restrito", "proprietary", "all rights", "fechado")):
        return ("LICENSE_RESTRICTED", False)
    return ("LICENSE_DECLARED_NEEDS_REVIEW", True)
