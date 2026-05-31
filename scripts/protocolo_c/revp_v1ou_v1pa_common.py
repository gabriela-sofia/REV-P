"""Shared helpers for REV-P Protocol C v1ou-v1pa.

This block builds an auditable observational evidence layer for events and
external sources. It NEVER creates ground truth labels, training targets,
or operational classifications.

All outputs are review-only, candidate-review-only, contextual, or blocked.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from revp_v1lj_v1lq_common import DATASETS, DOCS, SCHEMAS, read_csv, write_csv, write_schema

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Allowed use values — NEVER includes operational label or training target
# ---------------------------------------------------------------------------

ALLOWED_USE_VALUES = frozenset([
    "CONTEXTUAL_ONLY",
    "REVIEW_ONLY",
    "CANDIDATE_REFERENCE_REVIEW_ONLY",
    "BLOCKED_INSUFFICIENT_EVIDENCE",
    "BLOCKED_FIXTURE_OR_SYNTHETIC",
    "BLOCKED_NO_EVENT_DATE",
    "BLOCKED_NO_LOCATION",
    "BLOCKED_NO_SOURCE",
    "BLOCKED_TEMPORAL_SENTINEL_MISSING",
    "BLOCKED_NO_FORMAL_NEGATIVE",
])

FORBIDDEN_ALLOWED_USE = frozenset([
    "GROUND_TRUTH",
    "OPERATIONAL_LABEL",
    "TRAINING_TARGET",
    "CAN_TRAIN",
    "LABEL",
])

# ---------------------------------------------------------------------------
# Evidence terms for scanning
# ---------------------------------------------------------------------------

EVIDENCE_TERMS = [
    "evento", "enchente", "inundacao", "inundação", "alagamento",
    "recife", "petropolis", "petrópolis", "curitiba",
    "fonte", "noticia", "notícia", "relatorio", "relatório",
    "defesa civil", "defesacivil", "compdec",
    "data_evento", "event_id", "source_url",
    "reliability", "spatial_precision", "temporal_precision",
    "deslizamento", "flood", "landslide", "cicatriz",
    "candidate_date", "candidate_source", "candidate_location",
    "observed_event", "ground_reference", "external_evidence",
]

# Fixture/synthetic detection patterns
FIXTURE_PATCH_RE = re.compile(r"^REC_\d{5}$")
FIXTURE_SOURCE_RE = re.compile(r"^[A-Z]\d{1,4}$")
FIXTURE_CANDIDATE_RE = re.compile(r"^C\d{1,3}$")

# Windows absolute path detection (not allowed in outputs)
ABS_PATH_RE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")

# Date patterns for event date normalization
ISO_DATE_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
BR_DATE_RE = re.compile(r"\b(\d{2})[/\-](\d{2})[/\-](20\d{2})\b")
YEAR_MONTH_RE = re.compile(r"\b(20\d{2})[/\-](\d{2})\b")
YEAR_ONLY_RE = re.compile(r"\b(20\d{2})\b")


# ---------------------------------------------------------------------------
# Path hash
# ---------------------------------------------------------------------------

def path_hash_sanitized(rel_path: str) -> str:
    """Return 16-char hex hash of a relative path string."""
    return hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Region normalization
# ---------------------------------------------------------------------------

REGION_ALIASES: dict[str, str] = {
    "recife": "RECIFE",
    "rec": "RECIFE",
    "petropolis": "PET",
    "petrópolis": "PET",
    "pet": "PET",
    "curitiba": "CURITIBA",
    "cwb": "CURITIBA",
}


def normalize_region(raw: str) -> str:
    key = raw.strip().lower()
    return REGION_ALIASES.get(key, raw.strip().upper() or "UNKNOWN")


# ---------------------------------------------------------------------------
# Patch/alias normalization
# ---------------------------------------------------------------------------

def normalize_patch_id(raw: str) -> str:
    s = raw.strip().upper()
    return s if s else "UNKNOWN_PATCH"


# ---------------------------------------------------------------------------
# Date normalization
# ---------------------------------------------------------------------------

def normalize_event_date(raw: str) -> tuple[str, str]:
    """Return (iso_date_or_period, date_status).

    date_status values:
      DAY_EXPLICIT, MONTH_PERIOD, YEAR_PERIOD, UNKNOWN
    """
    if not raw or not raw.strip():
        return ("", "UNKNOWN")
    t = raw.strip()
    m = ISO_DATE_RE.search(t)
    if m:
        return (f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "DAY_EXPLICIT")
    m = BR_DATE_RE.search(t)
    if m:
        return (f"{m.group(3)}-{m.group(2)}-{m.group(1)}", "DAY_EXPLICIT")
    m = YEAR_MONTH_RE.search(t)
    if m:
        return (f"{m.group(1)}-{m.group(2)}", "MONTH_PERIOD")
    m = YEAR_ONLY_RE.search(t)
    if m:
        return (m.group(1), "YEAR_PERIOD")
    return (t, "UNKNOWN")


def classify_temporal_precision(date_status: str) -> str:
    """Return temporal precision level from date_status."""
    mapping = {
        "DAY_EXPLICIT": "HIGH",
        "MONTH_PERIOD": "MODERATE",
        "YEAR_PERIOD": "LOW",
        "UNKNOWN": "NONE",
    }
    return mapping.get(date_status, "NONE")


# ---------------------------------------------------------------------------
# Spatial precision
# ---------------------------------------------------------------------------

def classify_spatial_precision(location_raw: str, lat: str, lon: str) -> str:
    """Classify spatial precision given available location data."""
    has_coords = bool(lat.strip() and lon.strip())
    if has_coords:
        try:
            float(lat)
            float(lon)
            return "POINT_EXPLICIT"
        except ValueError:
            pass
    loc = location_raw.strip().lower()
    if any(kw in loc for kw in ["rua", "bairro", "logradouro", "endereço", "address"]):
        return "ADDRESS_LEVEL"
    if loc and loc not in ("", "unknown", "n/a"):
        return "ADMINISTRATIVE"
    return "NONE"


# ---------------------------------------------------------------------------
# Source reliability
# ---------------------------------------------------------------------------

OFFICIAL_SOURCE_KEYWORDS = [
    "cprm", "sgb", "cemaden", "ibge", "inpe", "sedec", "emlurb",
    "defesa civil", "compdec", "governo", "prefeitura", "municipal",
    "federal", "estadual", "decreto", "diário oficial", "diario oficial",
    "boletim oficial", "s2id", "drm",
]

NEWS_SOURCE_KEYWORDS = ["globo", "folha", "agência", "agencia", "jornalismo", "notícia", "blog"]

CONTEXTUAL_KEYWORDS = ["pe3d", "esig", "drainage", "drenagem", "mde", "mdt"]


def classify_source_reliability(source_name: str, source_type: str) -> str:
    s = (source_name + " " + source_type).lower()
    if any(kw in s for kw in OFFICIAL_SOURCE_KEYWORDS):
        return "OFFICIAL_HIGH"
    if any(kw in s for kw in NEWS_SOURCE_KEYWORDS):
        return "NEWS_LIMITED"
    if any(kw in s for kw in CONTEXTUAL_KEYWORDS):
        return "CONTEXTUAL_LIMITED"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Fixture detection
# ---------------------------------------------------------------------------

def is_fixture_or_synthetic(row: dict[str, str]) -> str:
    """Return non-empty reason if row looks like test fixture."""
    src = row.get("selected_source_id") or row.get("resolution_id") or row.get("source_candidate_id") or ""
    if FIXTURE_SOURCE_RE.match(src):
        return f"FIXTURE_SOURCE_ID:{src}"
    cand = row.get("candidate_id") or row.get("event_id") or ""
    if FIXTURE_CANDIDATE_RE.match(cand):
        return f"FIXTURE_CANDIDATE_ID:{cand}"
    # Check for obviously synthetic data markers
    notes = row.get("notes", "").lower()
    if "synthetic" in notes or "fixture" in notes or "test_only" in notes:
        return "FIXTURE_NOTES_MARKER"
    return ""


# ---------------------------------------------------------------------------
# Central evidence use classifier — GUARDRAIL
# ---------------------------------------------------------------------------

def classify_evidence_use(evidence_row: dict[str, str]) -> str:
    """Classify the allowed use of an evidence row.

    Returns one of ALLOWED_USE_VALUES. NEVER returns any label or
    operational ground truth classification.
    """
    # 1. Fixture check
    fixture_reason = is_fixture_or_synthetic(evidence_row)
    if fixture_reason:
        return "BLOCKED_FIXTURE_OR_SYNTHETIC"

    # 2. Source check
    source = (
        evidence_row.get("candidate_source_name")
        or evidence_row.get("source_name")
        or evidence_row.get("source_institution")
        or ""
    ).strip()
    if not source:
        return "BLOCKED_NO_SOURCE"

    # 3. Date check
    date_raw = (
        evidence_row.get("candidate_date_raw")
        or evidence_row.get("event_date_iso")
        or evidence_row.get("event_or_survey_date")
        or evidence_row.get("candidate_event_period")
        or ""
    ).strip()
    if not date_raw:
        return "BLOCKED_NO_EVENT_DATE"

    # 4. Location check
    location = (
        evidence_row.get("candidate_location_raw")
        or evidence_row.get("location_text")
        or evidence_row.get("municipality")
        or evidence_row.get("region")
        or ""
    ).strip()
    if not location:
        return "BLOCKED_NO_LOCATION"

    # 5. Temporal Sentinel check — fail-closed
    sentinel_status = (
        evidence_row.get("sentinel_scene_date_status")
        or evidence_row.get("scene_date_status")
        or ""
    ).upper()
    if sentinel_status and sentinel_status not in (
        "PRODUCT_DATE_CONFIRMED", "NOT_ASSESSED", "", "UNKNOWN"
    ):
        if "BLOCKED" in sentinel_status or "MISSING" in sentinel_status or "FAIL" in sentinel_status:
            return "BLOCKED_TEMPORAL_SENTINEL_MISSING"

    # 6. Assess confidence/tier
    confidence = (evidence_row.get("confidence_preliminary") or "").upper()
    dossier_status = (evidence_row.get("dossier_status") or "").upper()
    current_blocker = (evidence_row.get("current_blocker") or "").upper()

    if current_blocker and current_blocker not in ("", "NONE", "UNKNOWN"):
        # LICENSE_UNKNOWN means source exists but license not yet verified.
        # The event candidate IS documented (contextual); only the source use is blocked.
        if current_blocker.upper() == "LICENSE_UNKNOWN":
            return "CONTEXTUAL_ONLY"
        return "BLOCKED_INSUFFICIENT_EVIDENCE"

    if confidence == "HIGH":
        return "CANDIDATE_REFERENCE_REVIEW_ONLY"
    if confidence in ("MODERATE",):
        return "REVIEW_ONLY"
    if dossier_status in ("DOSSIER_OPEN",):
        return "CONTEXTUAL_ONLY"

    return "CONTEXTUAL_ONLY"


# ---------------------------------------------------------------------------
# Absolute path guardrail
# ---------------------------------------------------------------------------

def require_no_abs_paths_in_rows(rows: list[dict[str, str]], label: str) -> None:
    for i, row in enumerate(rows):
        for k, v in row.items():
            if ABS_PATH_RE.search(str(v)):
                raise ValueError(
                    f"Absolute Windows path in {label} row {i} field '{k}': {v!r}"
                )


# ---------------------------------------------------------------------------
# CSV / schema write wrappers (pass-through to common)
# ---------------------------------------------------------------------------

def write_csv_safe(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write CSV, always writing header even when rows is empty."""
    write_csv(path, rows, fields)


def write_schema_safe(path: Path, fields: list[str], prefix: str) -> None:
    write_schema(path, fields, prefix)


# ---------------------------------------------------------------------------
# Safe CSV reader (header + first N rows only, for scanning)
# ---------------------------------------------------------------------------

def read_csv_metadata(path: Path, max_rows: int = 3) -> tuple[list[str], list[dict[str, str]]]:
    """Return (fieldnames, first_N_rows) from a CSV without reading all rows."""
    if not path.exists():
        return ([], [])
    try:
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            fields = list(reader.fieldnames or [])
            rows = []
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                rows.append(dict(row))
        return (fields, rows)
    except Exception:
        return ([], [])


# ---------------------------------------------------------------------------
# Doc writer
# ---------------------------------------------------------------------------

def write_doc(path: Path, title: str, paragraphs: list[str]) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# " + title + "\n\n" + "\n\n".join(paragraphs).strip() + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Validation: no operational guardrail fields set to true
# ---------------------------------------------------------------------------

FORBIDDEN_TRUE_FIELDS = [
    "ground_truth",
    "can_train_model",
    "can_create_operational_label",
    "can_promote_to_label",
    "dino_can_create_label",
    "dino_can_train_model",
    "dino_target_field_created",
    "can_be_used_for_training",
    "can_create_label",
]


def assert_no_forbidden_true(rows: list[dict[str, Any]], label: str) -> None:
    for i, row in enumerate(rows):
        for field in FORBIDDEN_TRUE_FIELDS:
            val = str(row.get(field, "false")).strip().lower()
            if val == "true":
                raise ValueError(
                    f"GUARDRAIL VIOLATION in {label} row {i}: {field}=true is forbidden"
                )


# ---------------------------------------------------------------------------
# Env-overridable path helper
# ---------------------------------------------------------------------------

def _p(env: str, default: Path) -> Path:
    return Path(os.environ[env]) if env in os.environ else default
