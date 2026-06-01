"""Shared helpers for REV-P Protocol C v1qu-v1qz.

Ground reference partial validation workbench. This block organizes the
search for partial ground reference into auditable, review-only packages.

It NEVER creates operational labels, training targets, operational ground
truth, or formal negatives. DINO embeddings may prioritize review but never
validate an event. Absence of evidence is never treated as a formal negative.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from revp_v1lj_v1lq_common import DATASETS, DOCS, SCHEMAS  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Source families
# ---------------------------------------------------------------------------

OFFICIAL_HYDROMETEOROLOGICAL = "OFFICIAL_HYDROMETEOROLOGICAL"
OFFICIAL_GEOLOGICAL = "OFFICIAL_GEOLOGICAL"
OFFICIAL_CIVIL_DEFENSE = "OFFICIAL_CIVIL_DEFENSE"
OFFICIAL_GOVERNMENT_PUBLICATION = "OFFICIAL_GOVERNMENT_PUBLICATION"
SCIENTIFIC_DATASET = "SCIENTIFIC_DATASET"
TECHNICAL_REPORT = "TECHNICAL_REPORT"
NEWS_MEDIA_SECONDARY = "NEWS_MEDIA_SECONDARY"
SOCIAL_MEDIA_SECONDARY = "SOCIAL_MEDIA_SECONDARY"
UNKNOWN_SOURCE = "UNKNOWN_SOURCE"

SOURCE_FAMILIES = frozenset([
    OFFICIAL_HYDROMETEOROLOGICAL,
    OFFICIAL_GEOLOGICAL,
    OFFICIAL_CIVIL_DEFENSE,
    OFFICIAL_GOVERNMENT_PUBLICATION,
    SCIENTIFIC_DATASET,
    TECHNICAL_REPORT,
    NEWS_MEDIA_SECONDARY,
    SOCIAL_MEDIA_SECONDARY,
    UNKNOWN_SOURCE,
])

OFFICIAL_FAMILIES = frozenset([
    OFFICIAL_HYDROMETEOROLOGICAL,
    OFFICIAL_GEOLOGICAL,
    OFFICIAL_CIVIL_DEFENSE,
    OFFICIAL_GOVERNMENT_PUBLICATION,
])

SECONDARY_FAMILIES = frozenset([NEWS_MEDIA_SECONDARY, SOCIAL_MEDIA_SECONDARY])

# ---------------------------------------------------------------------------
# Protocol C decisions
# ---------------------------------------------------------------------------

C1_CONTEXTUAL_ONLY = "C1_CONTEXTUAL_ONLY"
C2_REVIEW_ONLY_CANDIDATE = "C2_REVIEW_ONLY_CANDIDATE"
C3_REFERENCE_CANDIDATE_NEEDS_ADJUDICATION = "C3_REFERENCE_CANDIDATE_NEEDS_ADJUDICATION"
C3_REFERENCE_CANDIDATE_READY_FOR_SUPERVISOR_REVIEW = "C3_REFERENCE_CANDIDATE_READY_FOR_SUPERVISOR_REVIEW"
C4_NEGATIVE_BLOCKED = "C4_NEGATIVE_BLOCKED"
BLOCKED_INSUFFICIENT_EVIDENCE = "BLOCKED_INSUFFICIENT_EVIDENCE"

PROTOCOL_DECISIONS = frozenset([
    C1_CONTEXTUAL_ONLY,
    C2_REVIEW_ONLY_CANDIDATE,
    C3_REFERENCE_CANDIDATE_NEEDS_ADJUDICATION,
    C3_REFERENCE_CANDIDATE_READY_FOR_SUPERVISOR_REVIEW,
    C4_NEGATIVE_BLOCKED,
    BLOCKED_INSUFFICIENT_EVIDENCE,
])

# ---------------------------------------------------------------------------
# Guardrail fields (every output row carries these)
# ---------------------------------------------------------------------------

GUARDRAIL_FIELDS = [
    "review_only",
    "can_create_operational_label",
    "can_train_model",
    "target_created",
    "ground_truth_operational",
    "dino_validates_event",
    "absence_as_negative",
    "formal_negative",
]

# Any of these set to "true" is a guardrail violation (formal_negative excepted
# only when an explicit formal negative source is documented; expected false).
FORBIDDEN_TRUE_FIELDS = [
    "can_create_operational_label",
    "can_train_model",
    "target_created",
    "ground_truth_operational",
    "dino_validates_event",
    "absence_as_negative",
]


def guardrail_row() -> dict[str, str]:
    """Return the standard review-only guardrail field block."""
    return {
        "review_only": "true",
        "can_create_operational_label": "false",
        "can_train_model": "false",
        "target_created": "false",
        "ground_truth_operational": "false",
        "dino_validates_event": "false",
        "absence_as_negative": "false",
        "formal_negative": "false",
    }


# ---------------------------------------------------------------------------
# Path / safety helpers
# ---------------------------------------------------------------------------

ABS_PATH_RE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")
LOCAL_RUNS_RE = re.compile(r"local_runs", re.IGNORECASE)


def detect_absolute_path(text: str) -> bool:
    return bool(ABS_PATH_RE.search(str(text)))


def detect_local_runs_exposure(text: str) -> bool:
    return bool(LOCAL_RUNS_RE.search(str(text)))


def mask_path(text: str) -> str:
    """Redact absolute Windows/Unix paths and local_runs references."""
    s = str(text)
    s = re.sub(r"[A-Za-z]:[\\/][^\s,;\"']*", "[PATH_REDACTED]", s)
    s = re.sub(r"(?i)local_runs[\\/][^\s,;\"']*", "[LOCAL_RUNS_REDACTED]", s)
    s = re.sub(r"(?i)\blocal_runs\b", "[LOCAL_RUNS_REDACTED]", s)
    s = s.replace("\\", "/")
    s = re.sub(r"(?i)gabriela", "[USER_REDACTED]", s)
    return s


def safe_relpath(path: Path | str, root: Path | None = None) -> str:
    """Return a repo-relative posix path; never an absolute path."""
    root = root or ROOT
    p = Path(path)
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return p.name


def hash_short(value: str, n: int = 16) -> str:
    return hashlib.sha256(str(value).encode("utf-8", errors="ignore")).hexdigest()[:n]


# ---------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------

REGION_ALIASES: dict[str, str] = {
    "recife": "RECIFE",
    "rec": "RECIFE",
    "petropolis": "PET",
    "petrópolis": "PET",
    "pet": "PET",
    "curitiba": "CURITIBA",
    "cwb": "CURITIBA",
    "cur": "CURITIBA",
}


def normalize_region(raw: str) -> str:
    key = str(raw or "").strip().lower()
    if not key:
        return "UNKNOWN"
    if key in REGION_ALIASES:
        return REGION_ALIASES[key]
    for alias, canon in REGION_ALIASES.items():
        if alias in key:
            return canon
    return str(raw).strip().upper() or "UNKNOWN"


def normalize_event_id(raw: str) -> str:
    s = str(raw or "").strip().upper()
    s = re.sub(r"\s+", "_", s)
    return s or "UNKNOWN_EVENT"


def normalize_patch_id(raw: str) -> str:
    s = str(raw or "").strip().upper()
    s = re.sub(r"\s+", "_", s)
    return s or "UNKNOWN_PATCH"


def normalize_alias(raw: str) -> str:
    s = str(raw or "").strip()
    return s or "UNKNOWN_ALIAS"


def normalize_source_id(raw: str) -> str:
    s = str(raw or "").strip().upper()
    s = re.sub(r"[^A-Z0-9_]+", "_", s).strip("_")
    return s or "UNKNOWN_SOURCE_ID"


# ---------------------------------------------------------------------------
# Source family classification
# ---------------------------------------------------------------------------

_FAMILY_KEYWORDS: list[tuple[str, list[str]]] = [
    (OFFICIAL_HYDROMETEOROLOGICAL, [
        "cemaden", "inmet", "bdmep", "ana ", "hidroweb", "hidro web",
        "pluviom", "precipita", "chuva horaria", "chuva diaria",
        "estacao hidrolog", "estação hidrológ", "nivel", "nível", "vazao", "vazão",
        "hidrometeor", "meteorolog",
    ]),
    (OFFICIAL_GEOLOGICAL, [
        "sgb", "cprm", "geolog", "geotecn", "suscetibilidade",
        "movimento de massa", "deslizament", "carta de risco",
        "drm", "servico geologico", "serviço geológico",
    ]),
    (OFFICIAL_CIVIL_DEFENSE, [
        "defesa civil", "defesacivil", "compdec", "sedec", "codecir",
        "coordenadoria de defesa", "comando de defesa",
    ]),
    (OFFICIAL_GOVERNMENT_PUBLICATION, [
        "diario oficial", "diário oficial", "decreto", "portaria",
        "situacao de emergencia", "situação de emergência",
        "reconhecimento", "estado de calamidade", "prefeitura",
        "ibge", "gov.br", "ministerio", "ministério", "secretaria",
    ]),
    (SCIENTIFIC_DATASET, [
        "mapbiomas", "scientific dataset", "dataset cientifico",
        "uso e cobertura", "land cover", "embrapa dataset",
    ]),
    (TECHNICAL_REPORT, [
        "relatorio tecnico", "relatório técnico", "technical report",
        "artigo", "paper", "scientific article", "nota tecnica", "nota técnica",
    ]),
    (NEWS_MEDIA_SECONDARY, [
        "globo", "g1", "folha", "jornal", "noticia", "notícia", "uol",
        "estadao", "estadão", "midia", "mídia", "reportagem",
    ]),
    (SOCIAL_MEDIA_SECONDARY, [
        "twitter", "facebook", "instagram", "tiktok", "youtube",
        "rede social", "social media", "whatsapp", "telegram",
    ]),
]


def classify_source_family(source_name: str, source_type: str = "") -> str:
    s = (str(source_name or "") + " " + str(source_type or "")).lower()
    for family, keywords in _FAMILY_KEYWORDS:
        if any(kw in s for kw in keywords):
            return family
    return UNKNOWN_SOURCE


# ---------------------------------------------------------------------------
# Scoring (all in [0.0, 1.0]) — never a supervised target
# ---------------------------------------------------------------------------

_RELIABILITY_BY_FAMILY = {
    OFFICIAL_HYDROMETEOROLOGICAL: 0.95,
    OFFICIAL_GEOLOGICAL: 0.95,
    OFFICIAL_CIVIL_DEFENSE: 0.90,
    OFFICIAL_GOVERNMENT_PUBLICATION: 0.90,
    SCIENTIFIC_DATASET: 0.80,
    TECHNICAL_REPORT: 0.75,
    NEWS_MEDIA_SECONDARY: 0.40,
    SOCIAL_MEDIA_SECONDARY: 0.20,
    UNKNOWN_SOURCE: 0.10,
}


def score_source_reliability(source_family: str) -> float:
    return _RELIABILITY_BY_FAMILY.get(source_family, 0.10)


def score_temporal_precision(status: str) -> float:
    s = str(status or "").strip().upper()
    if s in ("DAY_EXPLICIT", "DAY", "HIGH", "EXACT"):
        return 1.0
    if s in ("MONTH_PERIOD", "MONTH", "MODERATE"):
        return 0.6
    if s in ("YEAR_PERIOD", "YEAR", "LOW"):
        return 0.3
    return 0.0


def score_spatial_precision(level: str) -> float:
    s = str(level or "").strip().upper()
    if s in ("POINT_EXPLICIT", "POINT", "HIGH", "COORDINATE"):
        return 1.0
    if s in ("ADDRESS_LEVEL", "ADDRESS"):
        return 0.8
    if s in ("ADMINISTRATIVE", "MODERATE", "NEIGHBORHOOD"):
        return 0.5
    return 0.0


def score_provenance(source_family: str, has_documented_reference: bool = False) -> float:
    if source_family in OFFICIAL_FAMILIES:
        return 0.9 if has_documented_reference else 0.6
    if source_family in (SCIENTIFIC_DATASET, TECHNICAL_REPORT):
        return 0.7 if has_documented_reference else 0.5
    if source_family in SECONDARY_FAMILIES:
        return 0.3 if has_documented_reference else 0.2
    return 0.1


def score_independence(n_independent_sources: int) -> float:
    n = max(0, int(n_independent_sources or 0))
    if n >= 3:
        return 1.0
    if n == 2:
        return 0.8
    if n == 1:
        return 0.5
    return 0.0


def score_review_agreement(decision_a: str, decision_b: str) -> float:
    a = str(decision_a or "").strip().upper()
    b = str(decision_b or "").strip().upper()
    if a and b:
        return 1.0 if a == b else 0.3
    if a or b:
        return 0.5
    return 0.0


_COMPOSITE_WEIGHTS = {
    "source_reliability_score": 0.25,
    "temporal_precision_score": 0.20,
    "spatial_precision_score": 0.20,
    "provenance_score": 0.15,
    "independence_score": 0.10,
    "review_agreement_score": 0.10,
}


def composite_score(scores: dict[str, float]) -> float:
    total = 0.0
    for key, weight in _COMPOSITE_WEIGHTS.items():
        total += weight * float(scores.get(key, 0.0) or 0.0)
    return round(total, 4)


def decision_from_scores(
    scores: dict[str, float],
    source_family: str,
    has_formal_negative_source: bool = False,
) -> str:
    """Map scores + source family to a Protocol C decision.

    Never returns C4 unless an explicit formal negative source is documented;
    secondary sources never close the C3 gate.
    """
    rel = float(scores.get("source_reliability_score", 0.0) or 0.0)
    temp = float(scores.get("temporal_precision_score", 0.0) or 0.0)
    spat = float(scores.get("spatial_precision_score", 0.0) or 0.0)
    agree = float(scores.get("review_agreement_score", 0.0) or 0.0)
    indep = float(scores.get("independence_score", 0.0) or 0.0)
    comp = composite_score(scores) if "composite" not in scores else float(scores["composite"])

    if source_family in ("", UNKNOWN_SOURCE):
        return BLOCKED_INSUFFICIENT_EVIDENCE

    if source_family in SECONDARY_FAMILIES:
        # Secondary evidence never closes the C3 gate on its own.
        return C2_REVIEW_ONLY_CANDIDATE if comp >= 0.35 else C1_CONTEXTUAL_ONLY

    # Official / scientific / technical sources
    if temp < 0.6 or spat < 0.5:
        return C2_REVIEW_ONLY_CANDIDATE

    if source_family in OFFICIAL_FAMILIES and rel >= 0.75:
        if agree >= 0.8 and indep >= 0.5 and comp >= 0.75:
            return C3_REFERENCE_CANDIDATE_READY_FOR_SUPERVISOR_REVIEW
        return C3_REFERENCE_CANDIDATE_NEEDS_ADJUDICATION

    return C2_REVIEW_ONLY_CANDIDATE


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def read_csv_safe(path: Path | str) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        with p.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


def write_csv_with_header(path: Path | str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write CSV always emitting the header, even with zero rows.

    Guardrails: refuses to write absolute paths or local_runs references.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        for k, v in row.items():
            sv = str(v)
            if detect_absolute_path(sv):
                raise ValueError(f"Absolute path in {p.name} row {i} field {k!r}: {sv!r}")
            if detect_local_runs_exposure(sv):
                raise ValueError(f"local_runs exposure in {p.name} row {i} field {k!r}: {sv!r}")
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in fields})


def write_json_safe(path: Path | str, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def write_schema_safe(path: Path | str, fields: list[str], prefix: str) -> None:
    rows = [{"field": f, "description": f"{prefix}: {f}."} for f in fields]
    write_csv_with_header(path, rows, ["field", "description"])


def write_doc(path: Path | str, title: str, paragraphs: list[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# " + title + "\n\n" + "\n\n".join(paragraphs).strip() + "\n", encoding="utf-8")


def assert_no_forbidden_true(rows: list[dict[str, Any]], label: str) -> None:
    for i, row in enumerate(rows):
        for field in FORBIDDEN_TRUE_FIELDS:
            if str(row.get(field, "false")).strip().lower() == "true":
                raise ValueError(f"GUARDRAIL VIOLATION in {label} row {i}: {field}=true forbidden")


def assert_clean_rows(rows: list[dict[str, Any]], label: str) -> None:
    assert_no_forbidden_true(rows, label)
    for i, row in enumerate(rows):
        for k, v in row.items():
            if detect_absolute_path(str(v)):
                raise ValueError(f"Absolute path in {label} row {i} field {k!r}")
            if detect_local_runs_exposure(str(v)):
                raise ValueError(f"local_runs exposure in {label} row {i} field {k!r}")


# ---------------------------------------------------------------------------
# Env-overridable path helper
# ---------------------------------------------------------------------------

def _p(env: str, default: Path) -> Path:
    return Path(os.environ[env]) if env in os.environ else default


# ---------------------------------------------------------------------------
# Existing Protocol C context loader
# ---------------------------------------------------------------------------

def load_existing_protocol_c_context(datasets: Path | None = None) -> dict[str, list[dict[str, str]]]:
    """Load existing Protocol C registries (best-effort, fail-soft)."""
    ds = datasets or DATASETS
    candidates = {
        "observed_events": "recife_ground_reference_observed_event_registry_v1ov.csv",
        "event_patch_linkages": "recife_event_patch_linkage_registry_v1ox.csv",
        "dino_review_queue": "recife_dino_review_only_representation_queue_v1oz.csv",
        "candidate_decisions": "recife_ground_truth_candidate_decision_audit_v1oy.csv",
        "source_inventory": "recife_external_evidence_source_inventory_v1ou.csv",
        "evidence_scoring": "recife_ground_reference_evidence_scoring_v1ow.csv",
    }
    out: dict[str, list[dict[str, str]]] = {}
    for key, fname in candidates.items():
        out[key] = read_csv_safe(ds / fname)
    return out
