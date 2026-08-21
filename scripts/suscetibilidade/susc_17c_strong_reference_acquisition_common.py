"""SUSC-17C Strong Reference Acquisition Canary.

Builds a review-only, fail-closed acquisition queue for strong observational
reference candidates. The sprint is metadata-only: it does not download heavy
rasters, does not change score v6, does not create score v7, does not create
ground truth, and does not create trainable rows.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import ensure_dir, read_csv, rel, write_csv, write_json, write_markdown  # noqa: E402

DATASETS = ROOT / "datasets"
OUT_TABLES = ROOT / "outputs_public" / "tables"
OUT_SUSC = ROOT / "outputs_public" / "suscetibilidade"
OUT_DATA = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_17c_strong_reference_acquisition_canary"
OUT_REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

OBSERVED_EVENTS = OUT_TABLES / "revp_observed_event_registry_v2dz.csv"
SPATIAL_BINDING = OUT_TABLES / "revp_patch_event_spatial_binding_v2ec.csv"
TEMPORAL_ALIGNMENT = OUT_TABLES / "revp_patch_event_temporal_alignment_v2eb.csv"
OBSERVED_REFERENCE_CANDIDATES = DATASETS / "observed_event_reference_candidate_registry.csv"
OFFICIAL_VECTOR_REGISTRY = DATASETS / "official_observed_event_vector_registry.csv"
CONSOLIDATED_VECTOR_REGISTRY = DATASETS / "consolidated_observed_event_vector_candidate_registry.csv"
EVENT_PATCH_LINKAGE = DATASETS / "event_patch_linkage_registry.csv"
C4_S1_QUEUE = DATASETS / "c4_s1_completion_queue.csv"
EVIDENCE_SOURCE_REGISTRY = DATASETS / "ground_reference_evidence_source_registry.csv"
SOURCE_TARGETS_17C3 = OUT_SUSC / "susc_17c3_official_source_acquisition_targets.csv"
SAR_METADATA_17C31 = OUT_SUSC / "susc_17c31_sar_metadata_feasibility.csv"
TECH_FOOTPRINT_17C31 = OUT_SUSC / "susc_17c31_technical_footprint_candidate_registry.csv"
GRR_17C31 = OUT_SUSC / "susc_17c31_ground_reference_readiness_evaluation.csv"
CANARY_17C33 = OUT_SUSC / "susc_17c33_event_anchored_canary_patch_registry.csv"
SCORE_V6 = DATASETS / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DATASETS / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv"

PREFLIGHT = OUT_DATA / "susc_17c_preflight_registry_inventory.csv"
TARGET_PACK = OUT_DATA / "susc_17c_source_target_pack.csv"
DATE_RESOLVER = OUT_DATA / "susc_17c_event_date_resolver.csv"
SAR_FEASIBILITY = OUT_DATA / "susc_17c_sar_feasibility_pack.csv"
CANARY_PRIORITY = OUT_DATA / "susc_17c_canary_priority_events.csv"
STRONG_REGISTRY = OUT_DATA / "reference_strong_acquisition_registry.csv"
SAR_REGISTRY = OUT_DATA / "reference_sar_feasibility_registry.csv"
QA_QUEUE = OUT_DATA / "reference_canary_qa_queue.csv"
SUMMARY = OUT_DATA / "susc_17c_strong_reference_acquisition_summary.json"
REPORT = OUT_REPORTS / "SUSC_17C_STRONG_REFERENCE_ACQUISITION_CANARY_REPORT.md"

SCHEMA_TARGET = SCHEMAS / "susc_17c_source_target_pack_schema_v1.json"
SCHEMA_DATE = SCHEMAS / "susc_17c_event_date_resolver_schema_v1.json"
SCHEMA_SAR = SCHEMAS / "susc_17c_sar_feasibility_pack_schema_v1.json"
SCHEMA_PRIORITY = SCHEMAS / "susc_17c_canary_priority_schema_v1.json"
SCHEMA_REGISTRY = SCHEMAS / "susc_17c_reference_registry_schema_v1.json"

REQUIRED_INPUTS = [
    OBSERVED_EVENTS,
    SPATIAL_BINDING,
    TEMPORAL_ALIGNMENT,
    OBSERVED_REFERENCE_CANDIDATES,
    OFFICIAL_VECTOR_REGISTRY,
    CONSOLIDATED_VECTOR_REGISTRY,
    EVENT_PATCH_LINKAGE,
    C4_S1_QUEUE,
    EVIDENCE_SOURCE_REGISTRY,
    SOURCE_TARGETS_17C3,
    SAR_METADATA_17C31,
    TECH_FOOTPRINT_17C31,
    GRR_17C31,
    CANARY_17C33,
    SCORE_V6,
]

REQUIRED_OUTPUTS = [
    PREFLIGHT,
    TARGET_PACK,
    DATE_RESOLVER,
    SAR_FEASIBILITY,
    CANARY_PRIORITY,
    STRONG_REGISTRY,
    SAR_REGISTRY,
    QA_QUEUE,
    SUMMARY,
    REPORT,
    SCHEMA_TARGET,
    SCHEMA_DATE,
    SCHEMA_SAR,
    SCHEMA_PRIORITY,
    SCHEMA_REGISTRY,
]

ALLOWED_SOURCE_TYPES = {
    "official_observed_event_polygon",
    "official_observed_event_point",
    "technical_remote_sensing_flood_footprint",
    "official_address_resolved",
    "administrative_disaster_record",
    "documentary_context",
    "alert_only",
    "risk_area_not_event",
    "insufficient",
}
STRONG_SOURCE_TYPES = {
    "official_observed_event_polygon",
    "official_observed_event_point",
    "technical_remote_sensing_flood_footprint",
}
BLOCKED_STRONG_TYPES = {
    "alert_only",
    "risk_area_not_event",
    "documentary_context",
    "administrative_disaster_record",
    "official_address_resolved",
    "insufficient",
}
NO_GT_REASON = "review-only acquisition candidate; not ground truth, not trainable, no score v7"
SCORE_V7_FORBIDDEN = "false"

TARGET_FIELDS = [
    "candidate_event_id",
    "event_id",
    "city",
    "region",
    "source_id",
    "source_name",
    "source_authority",
    "source_type",
    "artifact_ref",
    "event_date_candidate",
    "event_date_precision",
    "phenomenon_type",
    "location_text",
    "has_official_geometry",
    "has_point",
    "has_bbox",
    "has_address",
    "has_patch_overlap_possible",
    "has_sentinel_window_candidate",
    "authority_tier",
    "documentary_strength",
    "priority",
    "strong_candidate",
    "geometry_status",
    "blocking_reason",
    "review_only",
    "trainable",
    "ground_truth",
    "score_v7_allowed",
    "not_ground_truth_reason",
]

DATE_FIELDS = [
    "candidate_event_id",
    "event_id",
    "event_date_resolved",
    "date_precision",
    "pre_start",
    "pre_end",
    "post_start",
    "post_end",
    "temporal_eligible",
    "temporal_blocking_reason",
    "review_only",
    "trainable",
    "ground_truth",
    "score_v7_allowed",
]

SAR_FIELDS = [
    "candidate_event_id",
    "event_id",
    "source_id",
    "city",
    "region",
    "event_date",
    "pre_event_window",
    "post_event_window",
    "has_sentinel1_pre",
    "has_sentinel1_post",
    "has_sentinel2_context",
    "intersects_patch_region",
    "has_water_mask_source",
    "has_hand_or_slope_context",
    "sar_feasible",
    "sar_blocking_reason",
    "future_query_required",
    "metadata_sources",
    "review_only",
    "trainable",
    "ground_truth",
    "score_v7_allowed",
]

PRIORITY_FIELDS = [
    "priority_rank",
    "candidate_event_id",
    "event_id",
    "city",
    "region",
    "source_id",
    "source_type",
    "event_date",
    "phenomenon_type",
    "geometry_status",
    "sar_feasible",
    "patch_link_potential",
    "priority_score",
    "selection_reason",
    "qa_status",
    "review_only",
    "ground_truth",
    "eligible_for_training",
    "score_v7_allowed",
]

REGISTRY_FIELDS = [
    "candidate_event_id",
    "event_id",
    "source_id",
    "geometry_status",
    "event_date",
    "pre_event_window",
    "post_event_window",
    "phenomenon_type",
    "uncertainty_m",
    "eligible_for_evaluation",
    "eligible_for_calibration",
    "eligible_for_training",
    "ground_truth",
    "review_only",
    "qa_status",
    "not_ground_truth_reason",
    "score_v7_allowed",
]

PREFLIGHT_FIELDS = [
    "artifact_role",
    "path",
    "exists",
    "row_count",
    "columns_found",
    "notes",
]

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ISO_MONTH = re.compile(r"^\d{4}-\d{2}$")


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _gov() -> dict:
    return {
        "review_only": "true",
        "trainable": "false",
        "ground_truth": "false",
        "score_v7_allowed": SCORE_V7_FORBIDDEN,
    }


def _run_git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _require_inputs() -> None:
    missing = [path for path in REQUIRED_INPUTS if not path.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(path) for path in missing))
    if SCORE_V7.exists():
        raise AssertionError("score_v7 exists and is not allowed for SUSC-17C")


def _read(path: Path) -> list[dict]:
    return read_csv(path) if path.exists() else []


def _row_count(path: Path) -> str:
    if not path.exists():
        return "0"
    if path.suffix.lower() == ".csv":
        return str(len(read_csv(path)))
    return "not_csv"


def _columns(path: Path) -> str:
    rows = _read(path)
    if rows:
        return ";".join(rows[0].keys())
    if path.exists() and path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return handle.readline().strip().replace(",", ";")
    return ""


def preflight_rows() -> list[dict]:
    sources = [
        ("observed_event_registry", OBSERVED_EVENTS, "eventos observados/documentais"),
        ("patch_event_spatial_binding", SPATIAL_BINDING, "vinculo espacial e status CRS/intersecao"),
        ("patch_event_temporal_alignment", TEMPORAL_ALIGNMENT, "alinhamento temporal patch-evento"),
        ("observed_event_reference_candidates", OBSERVED_REFERENCE_CANDIDATES, "candidatos multi-regiao"),
        ("official_observed_event_vector_registry", OFFICIAL_VECTOR_REGISTRY, "registro oficial vetorial"),
        ("consolidated_vector_candidate_registry", CONSOLIDATED_VECTOR_REGISTRY, "consolidado de candidatos vetoriais"),
        ("event_patch_linkage_registry", EVENT_PATCH_LINKAGE, "sensor stack e linkage strength"),
        ("c4_s1_completion_queue", C4_S1_QUEUE, "fila S1 pre/post"),
        ("evidence_source_registry", EVIDENCE_SOURCE_REGISTRY, "fontes de mascara/contexto"),
        ("susc_17c3_source_targets", SOURCE_TARGETS_17C3, "alvos de aquisicao 17C3"),
        ("susc_17c31_sar_metadata", SAR_METADATA_17C31, "metadado Sentinel-1/SAR"),
        ("susc_17c31_technical_footprint", TECH_FOOTPRINT_17C31, "candidato tecnico de footprint"),
        ("susc_17c31_ground_reference_readiness", GRR_17C31, "gates G4/G5/G7 herdados"),
        ("susc_17c33_canary_patches", CANARY_17C33, "patches canario ancorados"),
        ("score_v6_official", SCORE_V6, "somente leitura; nao alterado"),
        ("score_v7_forbidden", SCORE_V7, "deve permanecer ausente"),
    ]
    return [
        {
            "artifact_role": role,
            "path": rel(path),
            "exists": _bool(path.exists()),
            "row_count": _row_count(path),
            "columns_found": _columns(path),
            "notes": notes,
        }
        for role, path, notes in sources
    ]


def _city_from_region(region: str, fallback: str = "") -> str:
    value = (region or "").strip()
    if value in {"PET", "Petropolis", "Petrópolis"}:
        return "Petropolis"
    if value in {"REC", "Recife", "Olinda/PE"}:
        return "Recife"
    if value in {"CUR", "CTB", "Curitiba"}:
        return "Curitiba"
    return fallback or value or "unknown"


def _region_code(region: str) -> str:
    value = (region or "").strip()
    if value in {"PET", "Petropolis", "Petrópolis"}:
        return "PET"
    if value in {"REC", "Recife", "Olinda/PE"}:
        return "REC"
    if value in {"CUR", "CTB", "Curitiba"}:
        return "CUR"
    return value or "UNKNOWN"


def _canonical_event_id(event_id: str) -> str:
    event_id = (event_id or "").strip()
    if event_id == "REC_2022_05":
        return "REC_2022_05_24_30"
    return event_id or "unknown_event"


def _aliases(event_id: str) -> set[str]:
    eid = _canonical_event_id(event_id)
    aliases = {eid}
    if eid.startswith("EVENT_"):
        aliases.add(eid[len("EVENT_"):])
    else:
        aliases.add(f"EVENT_{eid}")
    if eid == "REC_2022_05_24_30":
        aliases.add("REC_2022_05")
    return aliases


def _first_by_event(rows: list[dict], event_field: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        for alias in _aliases(row.get(event_field, "")):
            out.setdefault(alias, row)
    return out


def _parse_iso(value: str) -> date | None:
    value = (value or "").strip()
    if not ISO_DATE.match(value):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _precision_and_dates(raw_precision: str, start: str, end: str) -> tuple[str, str, date | None, date | None]:
    s = _parse_iso(start)
    e = _parse_iso(end) or s
    raw = (raw_precision or "").upper()
    if s and e:
        if s == e and raw in {"DAY_EXPLICIT", "EXACT_DATE", "SHORT_WINDOW", ""}:
            return "exact_day", s.isoformat(), s, e
        return "range", f"{s.isoformat()}..{e.isoformat()}", s, e
    for value in [start, end]:
        if ISO_MONTH.match((value or "").strip()):
            return "month_only", value.strip(), None, None
    if "MONTH" in raw:
        return "month_only", "not_available", None, None
    return "unknown", "not_available", None, None


def _phenomenon(text: str) -> str:
    t = (text or "").lower()
    if any(term in t for term in ["alag", "inunda", "flood", "cheia", "enxurr", "chuva"]):
        if any(term in t for term in ["desliz", "massa", "landslide", "escorreg"]):
            return "flood_or_mass_movement"
        return "flood_inundation_alagamento"
    if any(term in t for term in ["desliz", "massa", "landslide", "escorreg"]):
        return "mass_movement"
    return "hydrometeorological_or_unknown"


def _source_authority(source_family: str, source_name: str = "") -> str:
    family = (source_family or "").lower()
    name = source_name or source_family or "not_available"
    if "ground_reference_event" in family or "official_documented_event_unit" in family:
        return "SGB/CPRM or official documented registry"
    if "official_observed_event_vector" in family:
        return "official observed event vector registry"
    if "tp2_candidate" in family:
        return "REV-P TP2 candidate priority registry"
    return name


def _authority_tier(authority: str, source_type: str) -> str:
    text = f"{authority} {source_type}".lower()
    if "sgb" in text or "cprm" in text or "defesa civil" in text or "prefeitura" in text:
        return "official"
    if "sentinel" in text or "asf" in text or "copernicus" in text or "charter" in text:
        return "technical"
    if "rev-p" in text:
        return "internal_registry"
    return "documentary"


def _source_type_from_observed(row: dict) -> str:
    family = (row.get("source_family") or "").lower()
    event_name = (row.get("event_name") or "").lower()
    blocking = (row.get("blocking_reason") or "").lower()
    has_geom = row.get("observed_geometry_available") == "true"
    has_date = row.get("event_start_date") or row.get("event_end_date")
    if "risk" in event_name or "suscept" in event_name or "area de risco" in event_name:
        return "risk_area_not_event"
    if "alert" in event_name or "alerta" in event_name:
        return "alert_only"
    if has_geom and ("ground_reference_event" in family or "official_documented_event_unit" in family):
        return "official_observed_event_point"
    if has_geom and "official_observed_event_vector" in family:
        return "official_observed_event_polygon"
    if not has_date or "missing date" in blocking:
        return "insufficient"
    if "official" in family:
        return "administrative_disaster_record"
    return "documentary_context"


def _source_type_from_reference(row: dict) -> str:
    spatial = (row.get("spatial_precision_level") or "").upper()
    primary = (row.get("primary_source_type") or "").upper()
    if "TECHNICAL_MAP" in spatial and "TECHNICAL" in primary:
        return "documentary_context"
    if "STREET_OR_POINT" in spatial and "OFFICIAL" in primary:
        return "official_address_resolved"
    if "NEIGHBORHOOD" in spatial and "OFFICIAL" in primary:
        return "administrative_disaster_record"
    if "PARTIAL" in spatial:
        return "documentary_context"
    return "documentary_context"


def _documentary_strength(source_type: str, has_date: bool, has_geometry: bool, sar_feasible: bool = False) -> str:
    if source_type in STRONG_SOURCE_TYPES and has_date and (has_geometry or sar_feasible):
        return "strong_candidate_pending_qa"
    if source_type in {"administrative_disaster_record", "official_address_resolved"} and has_date:
        return "official_context_pending_geometry"
    if source_type == "documentary_context":
        return "documentary_only"
    if source_type in {"alert_only", "risk_area_not_event"}:
        return "not_observed_event"
    return "insufficient"


def _spatial_maps() -> tuple[dict[str, dict], dict[str, dict]]:
    return (
        _first_by_event(_read(SPATIAL_BINDING), "observed_event_id"),
        _first_by_event(_read(TEMPORAL_ALIGNMENT), "observed_event_id"),
    )


def _linkage_maps() -> tuple[dict[str, dict], dict[str, dict]]:
    return (
        _first_by_event(_read(EVENT_PATCH_LINKAGE), "event_id"),
        _first_by_event(_read(C4_S1_QUEUE), "event_id"),
    )


def _has_water_mask(region: str) -> bool:
    code = _region_code(region).lower()
    for row in _read(EVIDENCE_SOURCE_REGISTRY):
        text = " ".join(row.get(k, "") for k in row).lower()
        if code in text and any(term in text for term in ["water", "hidro", "drain", "drenagem"]):
            return True
        if code == "rec" and any(term in text for term in ["drain", "drenagem", "hidrografia"]):
            return True
    return False


def _has_hand_or_slope(region: str, linkage: dict | None = None) -> bool:
    if linkage and linkage.get("dem_status") == "QA_PASS":
        return True
    code = _region_code(region).lower()
    for row in _read(EVIDENCE_SOURCE_REGISTRY):
        text = " ".join(row.get(k, "") for k in row).lower()
        if code in text and any(term in text for term in ["hand", "slope", "mde", "dem", "terrain", "decliv"]):
            return True
        if code == "rec" and any(term in text for term in ["mde", "pe3d", "terrain"]):
            return True
    return False


def target_pack_rows() -> list[dict]:
    spatial_by_event, temporal_by_event = _spatial_maps()
    rows: list[dict] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for idx, src in enumerate(_read(OBSERVED_EVENTS), start=1):
        event_id = _canonical_event_id(src.get("observed_event_id", ""))
        region = _region_code(src.get("region", ""))
        city = _city_from_region(region)
        precision, date_candidate, _, _ = _precision_and_dates(
            src.get("event_date_precision", ""), src.get("event_start_date", ""), src.get("event_end_date", "")
        )
        source_type = _source_type_from_observed(src)
        spatial = next((spatial_by_event.get(alias) for alias in _aliases(event_id) if spatial_by_event.get(alias)), {})
        temporal = next((temporal_by_event.get(alias) for alias in _aliases(event_id) if temporal_by_event.get(alias)), {})
        has_official_geometry = src.get("observed_geometry_available") == "true" and source_type in STRONG_SOURCE_TYPES
        has_patch_possible = (
            spatial.get("observed_geometry_available") == "true"
            and spatial.get("patch_boundary_available") == "true"
        )
        has_sentinel_window = temporal.get("temporal_alignment_status") == "TEMPORAL_ALIGNMENT_CANDIDATE"
        authority = _source_authority(src.get("source_family", ""))
        has_date = precision in {"exact_day", "range"}
        strong = source_type in STRONG_SOURCE_TYPES and has_date and has_official_geometry
        reason = "not_applicable" if strong else _blocking_reason_for_target(
            source_type, has_date, has_official_geometry, has_patch_possible
        )
        key = (event_id, src.get("source_id", ""), source_type)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rows.append({
            "candidate_event_id": f"S17C_REF_{len(rows) + 1:04d}",
            "event_id": event_id,
            "city": city,
            "region": region,
            "source_id": src.get("source_id") or event_id,
            "source_name": src.get("event_name") or src.get("source_family") or "not_available",
            "source_authority": authority,
            "source_type": source_type,
            "artifact_ref": src.get("source_url_or_reference") or rel(OBSERVED_EVENTS),
            "event_date_candidate": date_candidate,
            "event_date_precision": precision,
            "phenomenon_type": _phenomenon(src.get("event_name", "")),
            "location_text": city if city != "unknown" else region,
            "has_official_geometry": _bool(has_official_geometry),
            "has_point": _bool(source_type == "official_observed_event_point" and has_official_geometry),
            "has_bbox": "false",
            "has_address": "false",
            "has_patch_overlap_possible": _bool(has_patch_possible),
            "has_sentinel_window_candidate": _bool(has_sentinel_window),
            "authority_tier": _authority_tier(authority, source_type),
            "documentary_strength": _documentary_strength(source_type, has_date, has_official_geometry),
            "priority": _priority_label(strong, has_patch_possible, has_sentinel_window, precision),
            "strong_candidate": _bool(strong),
            "geometry_status": _geometry_status(source_type, has_official_geometry, has_patch_possible),
            "blocking_reason": reason,
            **_gov(),
            "not_ground_truth_reason": NO_GT_REASON,
        })

    for src in _read(OBSERVED_REFERENCE_CANDIDATES):
        event_id = _canonical_event_id(src.get("observed_event_id", ""))
        region = _region_code(src.get("region", ""))
        city = _city_from_region(region)
        precision, date_candidate, _, _ = _precision_and_dates(
            src.get("temporal_precision_level", ""), src.get("date_start", ""), src.get("date_end", "")
        )
        source_type = _source_type_from_reference(src)
        has_date = precision in {"exact_day", "range"}
        has_address = source_type == "official_address_resolved"
        strong = False
        authority = src.get("primary_source_name") or src.get("primary_source_type") or "not_available"
        key = (event_id, src.get("primary_source_name", ""), source_type)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rows.append({
            "candidate_event_id": f"S17C_REF_{len(rows) + 1:04d}",
            "event_id": event_id,
            "city": city,
            "region": region,
            "source_id": src.get("primary_source_type") or event_id,
            "source_name": src.get("event_name") or "not_available",
            "source_authority": authority,
            "source_type": source_type,
            "artifact_ref": src.get("primary_source_url") or rel(OBSERVED_REFERENCE_CANDIDATES),
            "event_date_candidate": date_candidate,
            "event_date_precision": precision,
            "phenomenon_type": _phenomenon(src.get("event_type", "") + " " + src.get("event_name", "")),
            "location_text": city,
            "has_official_geometry": "false",
            "has_point": "false",
            "has_bbox": "false",
            "has_address": _bool(has_address),
            "has_patch_overlap_possible": "false",
            "has_sentinel_window_candidate": "false",
            "authority_tier": _authority_tier(authority, source_type),
            "documentary_strength": _documentary_strength(source_type, has_date, False),
            "priority": _priority_label(strong, False, False, precision),
            "strong_candidate": "false",
            "geometry_status": "address_text_only" if has_address else "documentary_or_administrative_only",
            "blocking_reason": _blocking_reason_for_target(source_type, has_date, False, False),
            **_gov(),
            "not_ground_truth_reason": NO_GT_REASON,
        })

    for sar in _read(TECH_FOOTPRINT_17C31):
        event_id = _canonical_event_id(sar.get("event_id", ""))
        sar_meta = next((r for r in _read(SAR_METADATA_17C31) if _canonical_event_id(r.get("event_id")) == event_id), {})
        period = sar.get("post_event_window", "")
        start = sar_meta.get("event_period_start", "")
        end = sar_meta.get("event_period_end", "")
        precision, date_candidate, _, _ = _precision_and_dates("DATE_RANGE_EXPLICIT", start, end)
        feasible = sar_meta.get("sar_footprint_generation_feasible") == "true"
        strong = feasible and precision in {"exact_day", "range"}
        rows.append({
            "candidate_event_id": f"S17C_REF_{len(rows) + 1:04d}",
            "event_id": event_id,
            "city": "Recife",
            "region": "REC",
            "source_id": sar.get("technical_footprint_candidate_id") or "S17C31_TFC_0001",
            "source_name": "Sentinel-1 SAR metadata feasibility from SUSC-17C31",
            "source_authority": "ASF Sentinel-1 metadata / REV-P 17C31",
            "source_type": "technical_remote_sensing_flood_footprint",
            "artifact_ref": rel(SAR_METADATA_17C31),
            "event_date_candidate": date_candidate,
            "event_date_precision": precision,
            "phenomenon_type": "flood_inundation_alagamento",
            "location_text": "Recife AOI bbox",
            "has_official_geometry": "false",
            "has_point": "false",
            "has_bbox": "true",
            "has_address": "false",
            "has_patch_overlap_possible": "true",
            "has_sentinel_window_candidate": _bool(feasible),
            "authority_tier": "technical",
            "documentary_strength": _documentary_strength(
                "technical_remote_sensing_flood_footprint", precision in {"exact_day", "range"}, True, feasible
            ),
            "priority": "P1" if strong else "P2",
            "strong_candidate": _bool(strong),
            "geometry_status": "technical_bbox_metadata_not_footprint",
            "blocking_reason": "SAR metadata feasible; footprint raster not generated and requires future 17D QA",
            **_gov(),
            "not_ground_truth_reason": NO_GT_REASON,
        })
    return rows


def _blocking_reason_for_target(source_type: str, has_date: bool, has_geometry: bool, patch_possible: bool) -> str:
    if source_type in {"alert_only", "risk_area_not_event"}:
        return "alert_or_risk_area_is_not_observed_event"
    if not has_date:
        return "event_date_missing_or_month_only"
    if source_type in {"documentary_context", "administrative_disaster_record"}:
        return "documentary_or_administrative_context_without_strong_spatial_evidence"
    if source_type == "official_address_resolved":
        return "address_text_requires_spatial_QA_before_patch_link"
    if not has_geometry:
        return "strong_source_class_missing_official_or_technical_geometry"
    if not patch_possible:
        return "geometry_available_but_patch_overlap_not_confirmed"
    return "pending_human_QA"


def _priority_label(strong: bool, patch_possible: bool, sentinel: bool, precision: str) -> str:
    if strong and (patch_possible or sentinel):
        return "P1"
    if precision in {"exact_day", "range"}:
        return "P2"
    if precision == "month_only":
        return "P3"
    return "P4"


def _geometry_status(source_type: str, has_geometry: bool, patch_possible: bool) -> str:
    if source_type == "technical_remote_sensing_flood_footprint":
        return "technical_metadata_only"
    if source_type == "official_observed_event_polygon" and has_geometry:
        return "official_polygon_unvalidated"
    if source_type == "official_observed_event_point" and has_geometry:
        return "official_point_or_coordinate_unvalidated"
    if source_type == "official_address_resolved":
        return "address_text_only"
    if source_type == "risk_area_not_event":
        return "risk_area_not_event"
    if patch_possible:
        return "patch_overlap_possible_unvalidated"
    return "no_strong_geometry"


def date_resolver_rows(targets: list[dict] | None = None) -> list[dict]:
    targets = targets or target_pack_rows()
    rows = []
    for target in targets:
        precision = target["event_date_precision"]
        candidate = target["event_date_candidate"]
        start: date | None = None
        end: date | None = None
        if precision == "exact_day":
            start = end = _parse_iso(candidate)
        elif precision == "range" and ".." in candidate:
            left, right = candidate.split("..", 1)
            start, end = _parse_iso(left), _parse_iso(right)
        eligible = precision in {"exact_day", "range"} and start is not None and end is not None
        if eligible:
            pre_start = (start - timedelta(days=30)).isoformat()
            pre_end = (start - timedelta(days=7)).isoformat()
            post_start = start.isoformat()
            post_end = (end + timedelta(days=7)).isoformat()
            resolved = candidate
            reason = "not_applicable"
        else:
            pre_start = pre_end = post_start = post_end = "not_available"
            resolved = "not_available"
            reason = "month_only_blocks_sar" if precision == "month_only" else "unknown_date_blocks_sar"
        rows.append({
            "candidate_event_id": target["candidate_event_id"],
            "event_id": target["event_id"],
            "event_date_resolved": resolved,
            "date_precision": precision,
            "pre_start": pre_start,
            "pre_end": pre_end,
            "post_start": post_start,
            "post_end": post_end,
            "temporal_eligible": _bool(eligible),
            "temporal_blocking_reason": reason,
            **_gov(),
        })
    return rows


def sar_feasibility_rows(targets: list[dict] | None = None, dates: list[dict] | None = None) -> list[dict]:
    targets = targets or target_pack_rows()
    dates = dates or date_resolver_rows(targets)
    target_by_id = {row["candidate_event_id"]: row for row in targets}
    date_by_id = {row["candidate_event_id"]: row for row in dates}
    linkage_by_event, s1_by_event = _linkage_maps()
    sar_meta_by_event = _first_by_event(_read(SAR_METADATA_17C31), "event_id")
    rows = []
    for date_row in dates:
        if date_row["temporal_eligible"] != "true":
            continue
        target = target_by_id[date_row["candidate_event_id"]]
        event_id = target["event_id"]
        linkage = next((linkage_by_event.get(alias) for alias in _aliases(event_id) if linkage_by_event.get(alias)), {})
        s1 = next((s1_by_event.get(alias) for alias in _aliases(event_id) if s1_by_event.get(alias)), {})
        sar_meta = next((sar_meta_by_event.get(alias) for alias in _aliases(event_id) if sar_meta_by_event.get(alias)), {})
        s1_status = " ".join([linkage.get("sensor_stack", ""), linkage.get("s1_pair_status", ""), s1.get("current_s1_status", "")])
        has_s1_pre = "S1_PRE_POST" in s1_status or "pre=QA_PASS" in s1_status or sar_meta.get("pre_event_s1_scene_count", "0") not in {"", "0"}
        has_s1_post = "S1_PRE_POST" in s1_status or "post=QA_PASS" in s1_status or sar_meta.get("during_or_post_event_s1_scene_count", "0") not in {"", "0"}
        has_s2 = "S2_PRE_POST" in linkage.get("sensor_stack", "") or linkage.get("s2_pair_status") == "QA_PASS"
        intersects = target["has_patch_overlap_possible"] == "true" or bool(linkage)
        water = _has_water_mask(target["region"])
        hand = _has_hand_or_slope(target["region"], linkage)
        feasible = has_s1_pre and has_s1_post and intersects and water and hand
        metadata_sources = []
        if linkage:
            metadata_sources.append(rel(EVENT_PATCH_LINKAGE))
        if s1:
            metadata_sources.append(rel(C4_S1_QUEUE))
        if sar_meta:
            metadata_sources.append(rel(SAR_METADATA_17C31))
        if not metadata_sources:
            metadata_sources.append("future_query_queue_no_sentinel_metadata_in_repo")
        reason = "not_applicable" if feasible else _sar_blocking_reason(has_s1_pre, has_s1_post, intersects, water, hand)
        rows.append({
            "candidate_event_id": target["candidate_event_id"],
            "event_id": event_id,
            "source_id": target["source_id"],
            "city": target["city"],
            "region": target["region"],
            "event_date": date_row["event_date_resolved"],
            "pre_event_window": f"{date_row['pre_start']}..{date_row['pre_end']}",
            "post_event_window": f"{date_row['post_start']}..{date_row['post_end']}",
            "has_sentinel1_pre": _bool(has_s1_pre),
            "has_sentinel1_post": _bool(has_s1_post),
            "has_sentinel2_context": _bool(has_s2),
            "intersects_patch_region": _bool(intersects),
            "has_water_mask_source": _bool(water),
            "has_hand_or_slope_context": _bool(hand),
            "sar_feasible": _bool(feasible),
            "sar_blocking_reason": reason,
            "future_query_required": _bool(not feasible),
            "metadata_sources": ";".join(metadata_sources),
            **_gov(),
        })
    return rows


def _sar_blocking_reason(pre: bool, post: bool, intersects: bool, water: bool, hand: bool) -> str:
    missing = []
    if not pre:
        missing.append("sentinel1_pre_metadata")
    if not post:
        missing.append("sentinel1_post_metadata")
    if not intersects:
        missing.append("patch_region_intersection")
    if not water:
        missing.append("water_mask_source")
    if not hand:
        missing.append("hand_or_slope_context")
    return "missing_" + "_and_".join(missing) + "_future_query_required"


def _priority_score(target: dict, date_row: dict, sar_row: dict | None) -> int:
    score = 0
    linkage_by_event, _ = _linkage_maps()
    linkage = next((linkage_by_event.get(alias) for alias in _aliases(target["event_id"]) if linkage_by_event.get(alias)), {})
    if date_row and date_row["temporal_eligible"] == "true":
        score += 30
    if target["source_type"] in STRONG_SOURCE_TYPES:
        score += 30
    if target["authority_tier"] in {"official", "technical"}:
        score += 10
    if target["has_official_geometry"] == "true" or target["has_point"] == "true" or target["has_bbox"] == "true":
        score += 15
    if target["has_patch_overlap_possible"] == "true":
        score += 10
    if sar_row and sar_row["sar_feasible"] == "true":
        score += 15
    if linkage.get("linkage_strength") == "STRONG_PATCH_LINKAGE":
        score += 12
    elif linkage.get("linkage_strength") == "MODERATE_MULTIMODAL_LINKAGE":
        score += 4
    if linkage.get("s1_pair_status") == "QA_PASS":
        score += 5
    if "flood" in target["phenomenon_type"] or "alagamento" in target["phenomenon_type"]:
        score += 5
    if target["documentary_strength"] == "strong_candidate_pending_qa":
        score += 5
    return score


def canary_priority_rows(
    targets: list[dict] | None = None,
    dates: list[dict] | None = None,
    sar_rows: list[dict] | None = None,
) -> list[dict]:
    targets = targets or target_pack_rows()
    dates = dates or date_resolver_rows(targets)
    sar_rows = sar_rows or sar_feasibility_rows(targets, dates)
    date_by_id = {row["candidate_event_id"]: row for row in dates}
    sar_by_id = {row["candidate_event_id"]: row for row in sar_rows}
    best_by_event: dict[str, tuple[int, dict, dict, dict | None]] = {}
    for target in targets:
        date_row = date_by_id.get(target["candidate_event_id"], {})
        sar_row = sar_by_id.get(target["candidate_event_id"])
        score = _priority_score(target, date_row, sar_row)
        event_id = target["event_id"]
        current = best_by_event.get(event_id)
        if current is None or score > current[0] or (score == current[0] and target["candidate_event_id"] < current[1]["candidate_event_id"]):
            best_by_event[event_id] = (score, target, date_row, sar_row)
    ranked = sorted(best_by_event.values(), key=lambda item: (-item[0], item[1]["region"], item[1]["event_id"]))
    selected: list[tuple[int, dict, dict, dict | None]] = []
    regions: set[str] = set()
    for item in ranked:
        if len(selected) >= 5:
            break
        score, target, date_row, sar_row = item
        if score < 40:
            continue
        if len(selected) < 3 or target["region"] not in regions or len(selected) < 5:
            selected.append(item)
            regions.add(target["region"])
    # If all top rows came from one region, force the best other region if it exists.
    if len({item[1]["region"] for item in selected}) < 2:
        for item in ranked:
            if item[1]["region"] not in {sel[1]["region"] for sel in selected} and item[0] >= 35:
                selected.append(item)
                break
    selected = sorted(selected[:5], key=lambda item: (-item[0], item[1]["region"], item[1]["event_id"]))
    rows = []
    for rank, (score, target, date_row, sar_row) in enumerate(selected, start=1):
        reasons = []
        if date_row.get("temporal_eligible") == "true":
            reasons.append("data resolvida")
        if target["source_type"] in STRONG_SOURCE_TYPES:
            reasons.append("classe fonte forte")
        if target["has_official_geometry"] == "true" or target["has_bbox"] == "true":
            reasons.append("geometria ou bbox candidata")
        if sar_row and sar_row["sar_feasible"] == "true":
            reasons.append("SAR metadata factivel")
        if target["has_patch_overlap_possible"] == "true":
            reasons.append("patch-link possivel mas nao aceito")
        rows.append({
            "priority_rank": str(rank),
            "candidate_event_id": target["candidate_event_id"],
            "event_id": target["event_id"],
            "city": target["city"],
            "region": target["region"],
            "source_id": target["source_id"],
            "source_type": target["source_type"],
            "event_date": date_row.get("event_date_resolved", "not_available"),
            "phenomenon_type": target["phenomenon_type"],
            "geometry_status": target["geometry_status"],
            "sar_feasible": (sar_row or {}).get("sar_feasible", "false"),
            "patch_link_potential": target["has_patch_overlap_possible"],
            "priority_score": str(score),
            "selection_reason": "; ".join(reasons) or "fila futura",
            "qa_status": "pending_human_qa",
            "review_only": "true",
            "ground_truth": "false",
            "eligible_for_training": "false",
            "score_v7_allowed": SCORE_V7_FORBIDDEN,
        })
    return rows


def registry_rows(
    targets: list[dict] | None = None,
    dates: list[dict] | None = None,
    sar_rows: list[dict] | None = None,
) -> list[dict]:
    targets = targets or target_pack_rows()
    dates = dates or date_resolver_rows(targets)
    sar_rows = sar_rows or sar_feasibility_rows(targets, dates)
    date_by_id = {row["candidate_event_id"]: row for row in dates}
    sar_by_id = {row["candidate_event_id"]: row for row in sar_rows}
    rows = []
    for target in targets:
        d = date_by_id.get(target["candidate_event_id"], {})
        s = sar_by_id.get(target["candidate_event_id"], {})
        strong_spatial = target["has_official_geometry"] == "true" or target["has_bbox"] == "true" or s.get("sar_feasible") == "true"
        eligible_eval = target["strong_candidate"] == "true" and d.get("temporal_eligible") == "true" and strong_spatial
        rows.append({
            "candidate_event_id": target["candidate_event_id"],
            "event_id": target["event_id"],
            "source_id": target["source_id"],
            "geometry_status": target["geometry_status"],
            "event_date": d.get("event_date_resolved", "not_available"),
            "pre_event_window": f"{d.get('pre_start', 'not_available')}..{d.get('pre_end', 'not_available')}",
            "post_event_window": f"{d.get('post_start', 'not_available')}..{d.get('post_end', 'not_available')}",
            "phenomenon_type": target["phenomenon_type"],
            "uncertainty_m": _uncertainty(target),
            "eligible_for_evaluation": _bool(eligible_eval),
            "eligible_for_calibration": "false",
            "eligible_for_training": "false",
            "ground_truth": "false",
            "review_only": "true",
            "qa_status": "pending_human_qa" if eligible_eval else "blocked_pending_evidence",
            "not_ground_truth_reason": NO_GT_REASON,
            "score_v7_allowed": SCORE_V7_FORBIDDEN,
        })
    return rows


def _uncertainty(target: dict) -> str:
    if target["source_type"] == "technical_remote_sensing_flood_footprint":
        return "not_computed_metadata_only"
    if target["has_point"] == "true":
        return "unknown_crs_missing"
    if target["has_bbox"] == "true":
        return "bbox_aoi_not_precision"
    if target["has_address"] == "true":
        return "address_text_unresolved"
    return "not_available"


def sar_registry_rows(sar_rows: list[dict] | None = None, targets: list[dict] | None = None) -> list[dict]:
    targets = targets or target_pack_rows()
    target_by_id = {row["candidate_event_id"]: row for row in targets}
    sar_rows = sar_rows or sar_feasibility_rows(targets)
    out = []
    for row in sar_rows:
        target = target_by_id[row["candidate_event_id"]]
        out.append({
            "candidate_event_id": row["candidate_event_id"],
            "event_id": row["event_id"],
            "source_id": row["source_id"],
            "geometry_status": target["geometry_status"],
            "event_date": row["event_date"],
            "pre_event_window": row["pre_event_window"],
            "post_event_window": row["post_event_window"],
            "phenomenon_type": target["phenomenon_type"],
            "uncertainty_m": _uncertainty(target),
            "eligible_for_evaluation": row["sar_feasible"],
            "eligible_for_calibration": "false",
            "eligible_for_training": "false",
            "ground_truth": "false",
            "review_only": "true",
            "qa_status": "pending_human_qa" if row["sar_feasible"] == "true" else "queued_future_metadata_probe",
            "not_ground_truth_reason": NO_GT_REASON,
            "score_v7_allowed": SCORE_V7_FORBIDDEN,
        })
    return out


def qa_queue_rows(priority_rows: list[dict] | None = None, registry: list[dict] | None = None) -> list[dict]:
    priority_rows = priority_rows or canary_priority_rows()
    registry = registry or registry_rows()
    reg_by_id = {row["candidate_event_id"]: row for row in registry}
    rows = []
    for row in priority_rows:
        reg = reg_by_id[row["candidate_event_id"]]
        rows.append({
            **reg,
            "qa_status": "pending_human_qa",
            "eligible_for_evaluation": reg["eligible_for_evaluation"],
        })
    return rows


def _schema(title: str, fields: list[str], consts: dict[str, str] | None = None, enums: dict[str, list[str]] | None = None) -> dict:
    consts = consts or {}
    enums = enums or {}
    properties = {}
    for field in fields:
        if field in consts:
            properties[field] = {"const": consts[field]}
        elif field in enums:
            properties[field] = {"enum": enums[field]}
        else:
            properties[field] = {"type": "string", "minLength": 1}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        "type": "object",
        "description": "SUSC-17C review-only fail-closed schema. No ground truth, no training, no score v7.",
        "required": fields,
        "properties": properties,
        "additionalProperties": True,
    }


def write_schemas() -> None:
    bool_enum = ["true", "false"]
    write_json(SCHEMA_TARGET, _schema(
        "SUSC-17C source target pack",
        TARGET_FIELDS,
        consts={"review_only": "true", "trainable": "false", "ground_truth": "false", "score_v7_allowed": "false"},
        enums={
            "source_type": sorted(ALLOWED_SOURCE_TYPES),
            "event_date_precision": ["exact_day", "range", "month_only", "unknown"],
            "strong_candidate": bool_enum,
            "has_official_geometry": bool_enum,
            "has_point": bool_enum,
            "has_bbox": bool_enum,
            "has_address": bool_enum,
            "has_patch_overlap_possible": bool_enum,
            "has_sentinel_window_candidate": bool_enum,
        },
    ))
    write_json(SCHEMA_DATE, _schema(
        "SUSC-17C event date resolver",
        DATE_FIELDS,
        consts={"review_only": "true", "trainable": "false", "ground_truth": "false", "score_v7_allowed": "false"},
        enums={"date_precision": ["exact_day", "range", "month_only", "unknown"], "temporal_eligible": bool_enum},
    ))
    write_json(SCHEMA_SAR, _schema(
        "SUSC-17C SAR feasibility pack",
        SAR_FIELDS,
        consts={"review_only": "true", "trainable": "false", "ground_truth": "false", "score_v7_allowed": "false"},
        enums={
            "has_sentinel1_pre": bool_enum,
            "has_sentinel1_post": bool_enum,
            "has_sentinel2_context": bool_enum,
            "intersects_patch_region": bool_enum,
            "has_water_mask_source": bool_enum,
            "has_hand_or_slope_context": bool_enum,
            "sar_feasible": bool_enum,
            "future_query_required": bool_enum,
        },
    ))
    write_json(SCHEMA_PRIORITY, _schema(
        "SUSC-17C canary priority events",
        PRIORITY_FIELDS,
        consts={"review_only": "true", "ground_truth": "false", "eligible_for_training": "false", "score_v7_allowed": "false"},
        enums={"sar_feasible": bool_enum, "patch_link_potential": bool_enum},
    ))
    write_json(SCHEMA_REGISTRY, _schema(
        "SUSC-17C reference registry",
        REGISTRY_FIELDS,
        consts={"review_only": "true", "ground_truth": "false", "eligible_for_training": "false", "score_v7_allowed": "false"},
        enums={"eligible_for_evaluation": bool_enum, "eligible_for_calibration": bool_enum},
    ))


def _status_counts() -> dict:
    status = _run_git(["status", "--short"]).splitlines()
    staged = _run_git(["diff", "--cached", "--name-only"]).splitlines()
    own_prefixes = [
        "outputs_public/data/linhagem_anterior/susc_17c_strong_reference_acquisition_canary/",
        "outputs_public/reports/SUSC_17C_STRONG_REFERENCE_ACQUISITION_CANARY_REPORT.md",
    ]
    scoped = [line for line in status if not any(prefix in line.replace("\\", "/") for prefix in own_prefixes)]
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len([line for line in staged if line.strip()]),
        "dirty_paths_outside_17c_outputs_count": len([line for line in scoped if line.strip()]),
        "score_v6_changed": bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)])),
        "score_v7_created": SCORE_V7.exists(),
    }


def summary_obj(
    targets: list[dict],
    dates: list[dict],
    sar_rows: list[dict],
    priority_rows: list[dict],
    registry: list[dict],
) -> dict:
    temporal_eligible_ids = {row["event_id"] for row in dates if row["temporal_eligible"] == "true"}
    strong_geometry = [
        row for row in targets
        if row["source_type"] in STRONG_SOURCE_TYPES and (row["has_official_geometry"] == "true" or row["has_bbox"] == "true")
    ]
    sar_feasible = [row for row in sar_rows if row["sar_feasible"] == "true"]
    link_rows = _read(EVENT_PATCH_LINKAGE)
    strong_linkage_count = sum(1 for row in link_rows if row.get("linkage_strength") == "STRONG_PATCH_LINKAGE")
    spatial_rows = _read(SPATIAL_BINDING)
    confirmed_intersections = sum(1 for row in spatial_rows if row.get("intersection_computed") == "true" and row.get("candidate_overlap_ratio") not in {"", "0", "0.0"})
    priority_regions = sorted({row["region"] for row in priority_rows})
    footprint_regions = sorted({
        row["region"] for row in targets
        if row["source_type"] in {"official_observed_event_polygon", "technical_remote_sensing_flood_footprint"}
        and (row["has_official_geometry"] == "true" or row["has_bbox"] == "true")
    })
    preflight = _status_counts()
    minimum_3_dated = len(temporal_eligible_ids) >= 3
    minimum_2_regions = len(priority_regions) >= 2
    minimum_footprint_per_priority_region = set(priority_regions).issubset(set(footprint_regions)) and bool(priority_regions)
    minimum_20_links = strong_linkage_count >= 20 and confirmed_intersections >= 20
    return {
        **preflight,
        "target_candidates_count": len(targets),
        "unique_event_count": len({row["event_id"] for row in targets}),
        "temporal_resolved_unique_event_count": len(temporal_eligible_ids),
        "strong_candidate_rows_count": sum(1 for row in targets if row["strong_candidate"] == "true"),
        "strong_geometry_rows_count": len(strong_geometry),
        "sar_feasible_rows_count": len(sar_feasible),
        "canary_priority_count": len(priority_rows),
        "canary_priority_regions": priority_regions,
        "reference_registry_rows_count": len(registry),
        "eligible_for_evaluation_count": sum(1 for row in registry if row["eligible_for_evaluation"] == "true"),
        "eligible_for_calibration_count": sum(1 for row in registry if row["eligible_for_calibration"] == "true"),
        "eligible_for_training_count": sum(1 for row in registry if row["eligible_for_training"] == "true"),
        "ground_truth_true_count": sum(1 for row in registry if row["ground_truth"] == "true"),
        "score_v7_allowed_true_count": sum(1 for row in registry if row["score_v7_allowed"] == "true"),
        "strong_patch_linkage_rows_count": strong_linkage_count,
        "confirmed_spatial_intersections_count": confirmed_intersections,
        "minimum_3_dated_events": minimum_3_dated,
        "minimum_2_regions": minimum_2_regions,
        "minimum_1_official_or_technical_footprint_per_priority_region": minimum_footprint_per_priority_region,
        "minimum_20_strong_patch_links": minimum_20_links,
        "eligible_for_17b_now": minimum_3_dated and minimum_2_regions and minimum_footprint_per_priority_region and minimum_20_links,
        "review_only": True,
        "ground_truth": False,
        "trainable": False,
        "score_v7_created": preflight["score_v7_created"],
        "score_v6_changed": preflight["score_v6_changed"],
        "decision": "17B_BLOQUEADO_FAIL_CLOSED",
    }


def _counts_table(counter: Counter) -> str:
    if not counter:
        return "| item | count |\n|---|---|\n"
    lines = ["| item | count |", "|---|---|"]
    for key, value in sorted(counter.items(), key=lambda item: (str(item[0]), item[1])):
        lines.append(f"| {key or 'not_available'} | {value} |")
    return "\n".join(lines) + "\n"


def report_markdown(targets: list[dict], dates: list[dict], sar_rows: list[dict], priority_rows: list[dict], summary: dict) -> str:
    by_city = Counter(row["city"] for row in targets)
    by_region = Counter(row["region"] for row in targets)
    by_source = Counter(row["source_type"] for row in targets)
    by_class = Counter(row["documentary_strength"] for row in targets)
    sources_found = [row for row in preflight_rows() if row["exists"] == "true"]
    source_lines = "\n".join(f"- `{row['path']}`: {row['row_count']} linhas ({row['artifact_role']})" for row in sources_found)
    priority_lines = "\n".join(
        f"- {row['priority_rank']}. `{row['event_id']}` / {row['region']} / {row['source_type']} / score {row['priority_score']} / {row['selection_reason']}"
        for row in priority_rows
    ) or "- nenhum evento priorizado"
    blockers = [
        "patch-links fortes insuficientes: "
        f"{summary['strong_patch_linkage_rows_count']} linkage interno forte e "
        f"{summary['confirmed_spatial_intersections_count']} intersecoes espaciais confirmadas",
        "footprint tecnico/oficial por regiao prioritaria ainda incompleto",
        "QA humana 17D pendente; nenhuma linha accepted",
        "score_v7, treino supervisionado e ground truth continuam proibidos",
    ]
    blocker_lines = "\n".join(f"- {item}" for item in blockers)
    criteria_lines = "\n".join([
        f"- minimo 3 eventos datados: {summary['minimum_3_dated_events']} ({summary['temporal_resolved_unique_event_count']})",
        f"- minimo 2 regioes priorizadas: {summary['minimum_2_regions']} ({', '.join(summary['canary_priority_regions'])})",
        f"- minimo 1 footprint tecnico/oficial por regiao prioritaria: {summary['minimum_1_official_or_technical_footprint_per_priority_region']}",
        f"- minimo 20 patch-links fortes: {summary['minimum_20_strong_patch_links']} (linkage forte={summary['strong_patch_linkage_rows_count']}; intersecoes confirmadas={summary['confirmed_spatial_intersections_count']})",
    ])
    return f"""# SUSC-17C Strong Reference Acquisition Canary

## Estado inicial

- Branch: `{summary['branch']}`
- HEAD: `{summary['head']}`
- Area staged: {summary['staged_count']} arquivo(s)
- Worktree fora dos outputs 17C desta sprint: {summary['dirty_paths_outside_17c_outputs_count']} caminho(s)
- `score_v6` alterado: {summary['score_v6_changed']}
- `score_v7` criado: {summary['score_v7_created']}

## Fontes e registries encontrados

{source_lines}

## Contagens de candidatos

Total de linhas candidatas: {summary['target_candidates_count']}
Eventos unicos: {summary['unique_event_count']}

### Por cidade

{_counts_table(by_city)}
### Por regiao

{_counts_table(by_region)}
### Por fonte/classe

{_counts_table(by_source)}
### Por forca documental

{_counts_table(by_class)}
## Resolucao temporal, geometria e SAR

- Eventos unicos com data resolvida e janelas pre/post: {summary['temporal_resolved_unique_event_count']}
- Linhas com geometria forte candidata: {summary['strong_geometry_rows_count']}
- Linhas com SAR factivel por metadado/manifest: {summary['sar_feasible_rows_count']}
- Linhas elegiveis para avaliacao review-only: {summary['eligible_for_evaluation_count']}
- Linhas elegiveis para calibracao: {summary['eligible_for_calibration_count']}
- Linhas elegiveis para treino: {summary['eligible_for_training_count']}
- Linhas ground truth: {summary['ground_truth_true_count']}
- Linhas com score_v7_allowed=true: {summary['score_v7_allowed_true_count']}

## Eventos priorizados para canario

{priority_lines}

## Criterios minimos

{criteria_lines}

## Blockers reais

{blocker_lines}

## Decisao 17B

17B permanece **bloqueado**. A sprint destrava uma fila auditavel de aquisicao forte e alguns candidatos review-only, mas ainda nao ha 20 patch-links fortes confirmados, nao ha footprint tecnico/oficial completo por regiao prioritaria, e nenhuma linha foi aceita por QA humana.

## Proximos passos 17D Human QA

1. Revisar manualmente os 3-5 eventos de `reference_canary_qa_queue.csv`.
2. Confirmar CRS, geometria observada e intersecao patch-evento antes de qualquer aceite.
3. Para Recife, transformar a viabilidade SAR em artefato tecnico sob politica explicita, sem publicar raster pesado.
4. Para Petropolis/Curitiba, completar metadado Sentinel-1 pre/post ou registrar bloqueio por ausencia.
5. Manter `ground_truth=false`, `eligible_for_training=false` e `score_v7_allowed=false` ate nova decisao metodologica.
"""


def build_all() -> dict:
    _require_inputs()
    ensure_dir(OUT_DATA)
    ensure_dir(OUT_REPORTS)
    write_schemas()
    targets = target_pack_rows()
    dates = date_resolver_rows(targets)
    sar_rows = sar_feasibility_rows(targets, dates)
    priority = canary_priority_rows(targets, dates, sar_rows)
    registry = registry_rows(targets, dates, sar_rows)
    sar_registry = sar_registry_rows(sar_rows, targets)
    qa_rows = qa_queue_rows(priority, registry)
    summary = summary_obj(targets, dates, sar_rows, priority, registry)
    write_csv(PREFLIGHT, preflight_rows(), PREFLIGHT_FIELDS)
    write_csv(TARGET_PACK, targets, TARGET_FIELDS)
    write_csv(DATE_RESOLVER, dates, DATE_FIELDS)
    write_csv(SAR_FEASIBILITY, sar_rows, SAR_FIELDS)
    write_csv(CANARY_PRIORITY, priority, PRIORITY_FIELDS)
    write_csv(STRONG_REGISTRY, registry, REGISTRY_FIELDS)
    write_csv(SAR_REGISTRY, sar_registry, REGISTRY_FIELDS)
    write_csv(QA_QUEUE, qa_rows, REGISTRY_FIELDS)
    write_json(SUMMARY, summary)
    write_markdown(REPORT, report_markdown(targets, dates, sar_rows, priority, summary))
    return summary


def _schema_violations(row: dict, schema: dict) -> list[str]:
    errors = []
    for field in schema.get("required", []):
        if field not in row or str(row.get(field, "")) == "":
            errors.append(f"missing_required:{field}")
    for field, rule in schema.get("properties", {}).items():
        if field not in row:
            continue
        value = row.get(field)
        if "const" in rule and value != rule["const"]:
            errors.append(f"{field}:expected_const:{rule['const']}:got:{value}")
        if "enum" in rule and value not in rule["enum"]:
            errors.append(f"{field}:not_in_enum:{value}")
    return errors


def validate_output_rows(
    targets: list[dict],
    dates: list[dict],
    sar_rows: list[dict],
    registry: list[dict],
    sar_registry: list[dict],
    qa_rows: list[dict],
) -> list[str]:
    errors: list[str] = []
    date_by_id = {row["candidate_event_id"]: row for row in dates}
    sar_by_id = {row["candidate_event_id"]: row for row in sar_rows}
    reg_by_id = {row["candidate_event_id"]: row for row in registry}
    schemas = {
        "target": (targets, SCHEMA_TARGET),
        "date": (dates, SCHEMA_DATE),
        "sar": (sar_rows, SCHEMA_SAR),
        "registry": (registry + sar_registry + qa_rows, SCHEMA_REGISTRY),
    }
    for name, (rows, schema_path) in schemas.items():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        for idx, row in enumerate(rows, start=1):
            for err in _schema_violations(row, schema):
                errors.append(f"{name}[{idx}]:{err}")

    for rowset_name, rows in [
        ("target", targets),
        ("date", dates),
        ("sar", sar_rows),
        ("registry", registry),
        ("sar_registry", sar_registry),
        ("qa", qa_rows),
    ]:
        for row in rows:
            row_id = row.get("candidate_event_id") or row.get("event_id") or "unknown"
            for field in ["ground_truth", "trainable", "eligible_for_training"]:
                if row.get(field) == "true":
                    errors.append(f"{rowset_name}:{row_id}:{field}_true_forbidden")
            if row.get("score_v7_allowed") == "true":
                errors.append(f"{rowset_name}:{row_id}:score_v7_allowed_true_forbidden")

    for target in targets:
        cid = target["candidate_event_id"]
        if target["source_type"] not in ALLOWED_SOURCE_TYPES:
            errors.append(f"target:{cid}:source_type_not_allowed")
        if target["source_type"] in BLOCKED_STRONG_TYPES and target["strong_candidate"] == "true":
            errors.append(f"target:{cid}:blocked_source_type_marked_strong")
        if target["strong_candidate"] == "true" and target["source_type"] not in STRONG_SOURCE_TYPES:
            errors.append(f"target:{cid}:strong_candidate_source_type_invalid")
        if target["has_address"] == "true" and target["has_official_geometry"] == "false" and target["strong_candidate"] == "true":
            errors.append(f"target:{cid}:address_text_promoted_to_strong_patch_link")
        if target["strong_candidate"] == "true":
            date_row = date_by_id.get(cid)
            reg = reg_by_id.get(cid)
            if not date_row or date_row.get("temporal_eligible") != "true":
                errors.append(f"target:{cid}:strong_candidate_without_temporal_window")
            if not reg or "not_available" in (reg.get("pre_event_window", "") + reg.get("post_event_window", "") + reg.get("event_date", "")):
                errors.append(f"target:{cid}:strong_candidate_missing_event_date_or_windows")

    for date_row in dates:
        if date_row["date_precision"] in {"month_only", "unknown"} and date_row["temporal_eligible"] == "true":
            errors.append(f"date:{date_row['candidate_event_id']}:month_or_unknown_passed_to_sar")

    for row in qa_rows:
        cid = row["candidate_event_id"]
        sar = sar_by_id.get(cid, {})
        target = next((t for t in targets if t["candidate_event_id"] == cid), {})
        spatial_strong = (
            target.get("has_official_geometry") == "true"
            or target.get("has_bbox") == "true"
            or sar.get("sar_feasible") == "true"
        )
        if row.get("qa_status") == "accepted" and not spatial_strong:
            errors.append(f"qa:{cid}:accepted_without_strong_spatial_evidence")
    return errors


def validate() -> int:
    first = {path: path.read_bytes() for path in REQUIRED_OUTPUTS if path.exists()}
    summary = build_all()
    second = {path: path.read_bytes() for path in REQUIRED_OUTPUTS}
    if first and first != second:
        changed = [rel(path) for path in REQUIRED_OUTPUTS if first.get(path) != second.get(path)]
        print("17C validation rebuilt outputs; changed before compare: " + "; ".join(changed), file=sys.stderr)
    targets = _read(TARGET_PACK)
    dates = _read(DATE_RESOLVER)
    sar_rows = _read(SAR_FEASIBILITY)
    registry = _read(STRONG_REGISTRY)
    sar_registry = _read(SAR_REGISTRY)
    qa_rows = _read(QA_QUEUE)
    errors = validate_output_rows(targets, dates, sar_rows, registry, sar_registry, qa_rows)
    if summary["score_v6_changed"]:
        errors.append("score_v6_changed_forbidden")
    if summary["score_v7_created"]:
        errors.append("score_v7_created_forbidden")
    if summary["eligible_for_training_count"] != 0:
        errors.append("eligible_for_training_count_nonzero")
    if summary["ground_truth_true_count"] != 0:
        errors.append("ground_truth_true_count_nonzero")
    if summary["score_v7_allowed_true_count"] != 0:
        errors.append("score_v7_allowed_true_count_nonzero")
    if not targets:
        errors.append("empty_target_pack")
    if not qa_rows:
        errors.append("empty_qa_queue")
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(
        "17C strong reference acquisition validated: "
        f"targets={summary['target_candidates_count']} "
        f"dated_events={summary['temporal_resolved_unique_event_count']} "
        f"sar_feasible={summary['sar_feasible_rows_count']} "
        f"priority={summary['canary_priority_count']} "
        f"eligible_training={summary['eligible_for_training_count']} "
        f"ground_truth={summary['ground_truth_true_count']} "
        f"decision={summary['decision']}"
    )
    return 0


def run_all() -> int:
    summary = build_all()
    print(
        "17C strong reference acquisition built: "
        f"targets={summary['target_candidates_count']} "
        f"priority={summary['canary_priority_count']} "
        f"17B={summary['decision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_all())
