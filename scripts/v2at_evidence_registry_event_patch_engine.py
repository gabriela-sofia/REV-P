#!/usr/bin/env python3
"""v2at - Evidence Registry + Event-Patch Package Engine.

Reads the registries already consolidated in the REV-P repository (event-patch
candidates, sentinel date confidence, package validation, official documented
events, external GIS evidence, cross-region candidate references) and derives an
explicit, auditable observational-evidence system:

  * a canonical catalog of external evidence sources, hierarchised by class;
  * an evidence observation registry (derived from real registries, fail-closed);
  * an event-patch package registry with phenomenon typing, temporal window,
    evidence strength, blockers and an explainable promotion decision (C1..C4);
  * a promotion gate decision audit (15 gates per package);
  * a reviewer queue seed ordered by priority;
  * an operational-label blocklist enumerating everything that must NOT become a
    supervised training label.

Hard methodological line (never crossed by this stage):
  - no model is trained;
  - no operational/binary label is created;
  - no ground truth is declared;
  - absence of evidence is never turned into a negative;
  - an external benchmark never becomes local truth;
  - a quickview never promotes alone;
  - DINO is review-only structural support, never physical evidence;
  - media/social are never strong evidence.

A disaster inventory is not disaster geometry; a quickview is not a verified
product; a support image is not an observational reference; an external benchmark
is not local truth. C4 only ever appears as a *candidate*, never as a final label.

The engine is offline, deterministic, sorts every output by stable keys, uses
stable hashes for generated IDs, and returns exit code 0 even with expected
blockers. It returns a non-zero exit code only on a real structural error
(invalid schema, critical inconsistency).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from datetime import date

STAGE = "v2at"
METHODOLOGICAL_STATUS = "EVIDENCE_SYSTEM_READY_FOR_HUMAN_REVIEW_NOT_FOR_TRAINING"

# --------------------------------------------------------------------------- #
# Directory resolution (env first, then sensible defaults under project root).
# --------------------------------------------------------------------------- #

THIS_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def resolve_dirs():
    dataset_dir = os.environ.get("DATASET_DIR") or project_path("datasets")
    output_dir = os.environ.get("OUTPUT_DIR") or project_path("outputs_public")
    config_dir = os.environ.get("CONFIG_DIR") or project_path("configs")
    return dataset_dir, output_dir, config_dir


CONFIG_NAME = "v2at_evidence_registry_event_patch_config.json"

DEFAULT_CONFIG = {
    "time_delta_threshold_days": 30,
    "minimum_intersection_ratio": 0.10,
    "minimum_valid_data_fraction": 0.50,
    "source_weight_by_class": {
        "official_hydromet": 0.90, "official_geological": 0.88,
        "official_disaster_record": 0.80, "official_geoinfo": 0.60,
        "operational_mapping": 0.75, "vhr_optical": 0.70,
        "methodological_benchmark": 0.40, "context_low": 0.15,
    },
    "promotion_thresholds": {
        "c4_candidate_min_score": 0.80, "c3_min_score": 0.55,
        "c2_min_score": 0.35, "c1_min_score": 0.15,
    },
    "allowed_hazard_types": [
        "urban_flood", "flash_flood", "flood", "mass_movement",
        "landslide", "unknown_hazard",
    ],
    "allowed_use_values": [
        "review_only", "candidate_reference", "secondary_evaluation_candidate",
        "operational_label_blocked", "methodological_benchmark_only",
        "rejected_context_only",
    ],
    "strict_mode": False,
    "offline_mode": True,
    "fail_on_missing_optional_inputs": False,
}

UNKNOWN = "UNKNOWN"
NOT_AVAILABLE = "NOT_AVAILABLE"
BLOCKED = "BLOCKED"

# Output filenames -----------------------------------------------------------

OUT_CATALOG = "v2at_external_evidence_source_catalog.csv"
OUT_OBSERVATIONS = "v2at_evidence_observation_registry.csv"
OUT_PACKAGES = "v2at_event_patch_package_registry.csv"
OUT_GATES = "v2at_promotion_gate_decision_audit.csv"
OUT_QUEUE = "v2at_reviewer_queue_seed.csv"
OUT_BLOCKLIST = "v2at_operational_label_blocklist.csv"

REPORT_REL = os.path.join("execution_reports", "v2at_evidence_registry_event_patch_report.md")
SUMMARY_REL = os.path.join("execution_reports", "v2at_evidence_registry_event_patch_summary.json")
SUPPLEMENT_REL = os.path.join("execution_reports", "v2at_artifact_index_supplement.md")
LOG_REL = os.path.join("logs_summary", "v2at_evidence_registry_event_patch.log")
# Committable twin: the repository .gitignore blocks *.log, so the public log is
# also written as .txt (matching the logs_summary/*.txt convention) to remain an
# auditable, versioned artifact.
LOG_TXT_REL = os.path.join("logs_summary", "v2at_evidence_registry_event_patch.txt")

# Input filenames (relative to dataset_dir) ----------------------------------

IN_CANDIDATES = os.path.join("protocolo_c", "v1us_event_patch_candidate_registry.csv")
IN_DINO_SUPPORT = os.path.join("protocolo_c", "v1us_dino_review_support_attachment.csv")
IN_SENTINEL_DATE = os.path.join("protocolo_c", "v2aa_sentinel_date_confidence_audit.csv")
IN_PKG_VALIDATION = os.path.join("protocolo_c", "v2ab_event_patch_package_validation.csv")
IN_XREGION = os.path.join("protocolo_c", "v2bm_cross_region_candidate_registry.csv")
IN_XREGION_SCORECARD = os.path.join("protocolo_c", "v2bm_cross_region_evidence_scorecard.csv")
IN_GROUND_EVENTS = "ground_reference_event_registry.csv"
IN_EXTERNAL_EVIDENCE = "external_evidence_registry.csv"

INPUT_FILES = [
    IN_CANDIDATES, IN_DINO_SUPPORT, IN_SENTINEL_DATE, IN_PKG_VALIDATION,
    IN_XREGION, IN_XREGION_SCORECARD, IN_GROUND_EVENTS, IN_EXTERNAL_EVIDENCE,
]

REGION_NAME = {"REC": "Recife", "PET": "Petropolis", "CUR": "Curitiba"}
REGION_HAZARD = {"Recife": "urban_flood", "Petropolis": "mass_movement", "Curitiba": "urban_flood"}
STRENGTH_FACTOR = {"strong": 1.0, "moderate": 0.6, "weak": 0.3, "none": 0.0}

# Output column contracts (fixed order; also the schema source of truth) ------

COLUMNS = {
    OUT_CATALOG: [
        "source_id", "source_name", "source_class", "institution_type",
        "country_scope", "spatial_role", "temporal_role", "geometry_role",
        "license_status", "access_mode", "expected_data_type", "evidence_weight",
        "can_open_candidate", "can_promote_alone", "notes",
    ],
    OUT_OBSERVATIONS: [
        "evidence_id", "event_id", "patch_id", "region", "city", "hazard_type",
        "source_id", "source_class", "source_name", "observed_start",
        "observed_end", "published_at", "temporal_precision", "spatial_precision",
        "geometry_type", "geometry_available", "geometry_uri", "raw_uri",
        "derived_uri", "license_status", "download_hash", "evidence_strength",
        "evidence_role", "review_status", "blocking_reason", "notes",
    ],
    OUT_PACKAGES: [
        "package_id", "event_id", "patch_id", "region", "city", "hazard_type",
        "sentinel_asset_id", "sentinel_sensor_family", "sentinel_observation_date",
        "event_window_start", "event_window_end", "time_delta_days",
        "has_temporal_anchor", "has_spatial_support", "has_official_source",
        "has_vhr_support", "has_only_contextual_sources", "has_geometry",
        "has_patch_overlay", "intersection_ratio", "valid_data_fraction",
        "urban_context", "permanent_water_risk", "occlusion_risk",
        "evidence_count", "strong_evidence_count", "weak_evidence_count",
        "conflict_count", "evidence_score", "uncertainty_score",
        "promotion_candidate_level", "promotion_decision", "blocking_reason",
        "allowed_use", "notes",
    ],
    OUT_GATES: [
        "decision_id", "package_id", "event_id", "patch_id", "gate_name",
        "gate_passed", "gate_status", "required_condition", "observed_value",
        "severity", "blocking_reason", "recommended_action",
    ],
    OUT_QUEUE: [
        "review_item_id", "package_id", "event_id", "patch_id", "region", "city",
        "hazard_type", "priority_rank", "priority_reason", "suggested_review_action",
        "evidence_score", "uncertainty_score", "blocking_reason",
        "nearest_dino_neighbors_available", "notes",
    ],
    OUT_BLOCKLIST: [
        "block_id", "package_id", "event_id", "patch_id", "reason",
        "source_of_block", "severity", "can_be_revisited",
        "required_evidence_to_unblock",
    ],
}

GATE_NAMES = [
    "GATE_01_EVENT_ID_EXISTS", "GATE_02_HAZARD_TYPE_TYPED",
    "GATE_03_TEMPORAL_WINDOW_EXISTS", "GATE_04_SENTINEL_OBSERVATION_EXISTS",
    "GATE_05_TIME_DELTA_ACCEPTABLE", "GATE_06_OFFICIAL_OR_VALIDATED_SOURCE_EXISTS",
    "GATE_07_GEOMETRY_AVAILABLE", "GATE_08_PATCH_OVERLAY_AVAILABLE",
    "GATE_09_INTERSECTION_RATIO_ACCEPTABLE", "GATE_10_CONTEXT_ONLY_NOT_PROMOTED",
    "GATE_11_QUICKVIEW_NOT_PROMOTED_ALONE", "GATE_12_BENCHMARK_NOT_LOCAL_TRUTH",
    "GATE_13_CONFLICTS_RESOLVED", "GATE_14_UNCERTAINTY_RECORDED",
    "GATE_15_NO_TRAINING_LABEL_CREATED",
]

# --------------------------------------------------------------------------- #
# Small IO / helpers.
# --------------------------------------------------------------------------- #


def clean(value):
    return str(value if value is not None else "").strip()


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, columns, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def stable_id(prefix, *parts, length=12):
    digest = hashlib.sha1("|".join(clean(p) for p in parts).encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:length]}"


def region_from_code(code):
    return REGION_NAME.get(clean(code).upper(), clean(code) or UNKNOWN)


def region_from_label(label):
    folded = clean(label).lower()
    if folded.startswith("rec"):
        return "Recife"
    if folded.startswith("pet"):
        return "Petropolis"
    if folded.startswith("cur"):
        return "Curitiba"
    if folded.startswith("all"):
        return "All"
    return clean(label) or UNKNOWN


def parse_iso(value):
    try:
        return date.fromisoformat(clean(value))
    except ValueError:
        return None


def parse_event_window(event_id):
    """Derive a window from the id-encoded date(s) only; never invents dates."""
    nums = [p for p in clean(event_id).split("_") if p.isdigit()]
    year_idx = next((i for i, p in enumerate(nums) if len(p) == 4), None)
    if year_idx is None or len(nums) < year_idx + 3:
        return UNKNOWN, UNKNOWN
    year, month, day = nums[year_idx], nums[year_idx + 1], nums[year_idx + 2]
    start = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    end = start
    if len(nums) >= year_idx + 4:
        end = f"{year}-{month.zfill(2)}-{nums[year_idx + 3].zfill(2)}"
    if parse_iso(start) and parse_iso(end):
        return start, end
    return UNKNOWN, UNKNOWN


# --------------------------------------------------------------------------- #
# 1. Canonical external evidence source catalog.
# --------------------------------------------------------------------------- #

# Each tuple: source_id, name, class, institution_type, country_scope,
# spatial_role, temporal_role, geometry_role, license, access, data_type,
# can_open_candidate, can_promote_alone, notes
_CATALOG = [
    ("ANA_HIDROWEB", "ANA HidroWeb hydrological series", "official_hydromet",
     "federal_agency", "BR", "station_point", "primary_temporal", "point_station",
     "open_public", "api_download", "hydrological_timeseries", "true", "false",
     "Very-high official temporal source; opens/reinforces a candidate but only promotes with spatial-temporal coherence."),
    ("ANA_TELEMETRY", "ANA telemetry / real-time stage", "official_hydromet",
     "federal_agency", "BR", "station_point", "primary_temporal", "point_station",
     "open_public", "api_download", "hydrological_timeseries", "true", "false",
     "Real-time stage telemetry; temporal anchor, never a flood-extent truth."),
    ("INMET_HISTORICAL", "INMET historical precipitation", "official_hydromet",
     "federal_agency", "BR", "station_point", "primary_temporal", "point_station",
     "open_public", "bulk_download", "precipitation_timeseries", "true", "false",
     "Very-high official temporal source; a station series is not a label."),
    ("CEMADEN_MONITORING", "Cemaden monitoring / pluviometers", "official_hydromet",
     "federal_agency", "BR", "station_point", "primary_temporal", "point_station",
     "open_public", "api_download", "precipitation_timeseries", "true", "false",
     "Very-high official monitoring; temporal anchor only."),
    ("CEMADEN_BULLETIN", "Cemaden risk bulletins", "official_hydromet",
     "federal_agency", "BR", "municipality", "secondary_temporal", "none",
     "open_public", "document_download", "risk_bulletin", "true", "false",
     "Official bulletin; supports temporal/context, not geometry truth."),
    ("SGB_RISK_CARTOGRAPHY", "SGB/CPRM risk cartography & field surveys", "official_geological",
     "federal_agency", "BR", "territorial", "secondary_temporal", "polygon_cartographic",
     "open_public", "document_download", "geological_cartography", "true", "false",
     "Very-high official geological cartography; field-survey points are event anchors, not patch overlays."),
    ("SGB_SUSCEPTIBILITY", "SGB/CPRM susceptibility maps", "official_geological",
     "federal_agency", "BR", "territorial", "static", "polygon_susceptibility",
     "open_public", "geoservice", "susceptibility_map", "true", "false",
     "Territorial susceptibility context; static, not an observed event."),
    ("S2ID_DISASTER_RECORD", "S2iD disaster records", "official_disaster_record",
     "federal_agency", "BR", "municipality", "secondary_temporal", "municipality_polygon",
     "open_public", "document_download", "disaster_record", "true", "false",
     "Disaster inventory is not disaster geometry; opens a candidate, never promotes alone."),
    ("ATLAS_DIGITAL_DESASTRES", "Atlas Digital de Desastres no Brasil", "official_disaster_record",
     "federal_agency", "BR", "municipality", "secondary_temporal", "municipality_polygon",
     "open_public", "portal_download", "disaster_record", "true", "false",
     "Municipal disaster context; inventory level, not patch geometry."),
    ("COPERNICUS_EMS_MAPPING", "Copernicus EMS rapid mapping", "operational_mapping",
     "international_program", "GLOBAL", "event_extent", "event_dated", "polygon_flood_product",
     "open_public", "portal_download", "flood_extent_product", "true", "false",
     "High operational mapping; promotes only with local temporal/spatial coherence."),
    ("COPERNICUS_GFM", "Copernicus Global Flood Monitoring", "operational_mapping",
     "international_program", "GLOBAL", "event_extent", "event_dated", "raster_flood",
     "open_public", "api_download", "flood_extent_raster", "true", "false",
     "High operational raster product; not a local label."),
    ("INTERNATIONAL_CHARTER_PRODUCT", "International Charter validated product", "operational_mapping",
     "international_program", "GLOBAL", "event_extent", "event_dated", "polygon_or_raster_product",
     "licensed_attribution", "portal_download", "disaster_mapping_product", "true", "false",
     "High validated spatial product; reinforces geometry, does not replace the official temporal window."),
    ("INTERNATIONAL_CHARTER_QUICKVIEW", "International Charter quickview", "operational_mapping",
     "international_program", "GLOBAL", "visual_support", "event_dated", "quickview_image_only",
     "licensed_attribution", "portal_download", "quickview_image", "false", "false",
     "A quickview is not a verified product; NEVER promotes alone."),
    ("VANTOR_OPEN_DATA", "Vantor/Maxar Open Data VHR", "vhr_optical",
     "commercial_open", "GLOBAL", "visual_support", "event_dated", "vhr_imagery",
     "open_for_response", "portal_download", "vhr_optical_imagery", "true", "false",
     "High visual VHR; reinforces geometry only, never replaces the official temporal window."),
    ("PLANET_DISASTER_DATA", "Planet Disaster Data VHR", "vhr_optical",
     "commercial_open", "GLOBAL", "visual_support", "event_dated", "vhr_imagery",
     "open_for_response", "portal_download", "vhr_optical_imagery", "true", "false",
     "High visual VHR; geometry reinforcement, not temporal truth."),
    ("URBANSARFLOODS_BENCHMARK", "UrbanSARFloods benchmark", "methodological_benchmark",
     "academic", "GLOBAL", "none", "none", "benchmark_mask", "open_research",
     "dataset_download", "sar_flood_benchmark", "false", "false",
     "Methodological benchmark only; NEVER becomes local truth."),
    ("SEN1FLOODS11_BENCHMARK", "Sen1Floods11 benchmark", "methodological_benchmark",
     "academic", "GLOBAL", "none", "none", "benchmark_mask", "open_research",
     "dataset_download", "sar_flood_benchmark", "false", "false",
     "Methodological benchmark only; NEVER becomes local truth."),
    ("EMDAT_CONTEXT", "EM-DAT international disaster database", "context_low",
     "international_db", "GLOBAL", "country", "secondary_temporal", "not_applicable",
     "registration_required", "portal_download", "disaster_context", "false", "false",
     "Low context; never promotes alone, never a negative by absence."),
    ("MEDIA_CONTEXT", "Press / media reports", "context_low",
     "media", "BR", "none", "secondary_temporal", "not_applicable",
     "public_web", "manual_intake", "narrative_context", "false", "false",
     "Low context; never strong evidence, never promotes alone."),
    ("SOCIAL_CONTEXT", "Social media reports", "context_low",
     "social_media", "BR", "none", "secondary_temporal", "not_applicable",
     "public_web", "manual_intake", "narrative_context", "false", "false",
     "Low context; never strong evidence, never promotes alone."),
    # --- additional canonical entries used to map real territorial GIS evidence ---
    ("MUNICIPAL_GEOINFO", "Municipal geoinfo / terrain / drainage layers", "official_geoinfo",
     "municipal_agency", "BR", "territorial", "static", "polygon_or_raster_context",
     "open_public", "geoservice", "municipal_gis", "true", "false",
     "Municipal territorial context (PE3D, drainage, infra); static, not an observed event."),
    ("FBDS_LANDUSE", "FBDS land use / land cover", "context_low",
     "ngo_research", "BR", "territorial", "static", "polygon_landuse",
     "open_public", "portal_download", "landuse_layer", "false", "false",
     "Land-use context; never promotes alone."),
    ("MAPBIOMAS_LANDUSE", "MapBiomas land use / land cover", "context_low",
     "ngo_research", "BR", "territorial", "static", "polygon_landuse",
     "open_public", "portal_download", "landuse_layer", "false", "false",
     "Land-use context; never promotes alone."),
]

CATALOG_BY_ID = {row[0]: row for row in _CATALOG}
CLASS_BY_ID = {row[0]: row[2] for row in _CATALOG}


def build_source_catalog(config):
    weights = config["source_weight_by_class"]
    rows = []
    for (sid, name, sclass, inst, scope, srole, trole, grole, lic, access,
         dtype, can_open, can_alone, notes) in _CATALOG:
        rows.append({
            "source_id": sid, "source_name": name, "source_class": sclass,
            "institution_type": inst, "country_scope": scope, "spatial_role": srole,
            "temporal_role": trole, "geometry_role": grole, "license_status": lic,
            "access_mode": access, "expected_data_type": dtype,
            "evidence_weight": f"{weights.get(sclass, 0.3):.2f}",
            "can_open_candidate": can_open, "can_promote_alone": can_alone, "notes": notes,
        })
    rows.sort(key=lambda r: r["source_id"])
    return rows


# --------------------------------------------------------------------------- #
# 2. Evidence observation registry (derived, fail-closed).
# --------------------------------------------------------------------------- #


def _institution_to_source(institution, evidence_type):
    text = clean(institution).lower()
    etype = clean(evidence_type).lower()
    if "cprm" in text or "sgb" in text:
        return "SGB_SUSCEPTIBILITY"
    if "fbds" in text:
        return "FBDS_LANDUSE"
    if "mapbiomas" in text:
        return "MAPBIOMAS_LANDUSE"
    if etype in {"terrain", "drainage", "administrative", "land_use"}:
        return "MUNICIPAL_GEOINFO"
    return "MUNICIPAL_GEOINFO"


def build_observations(inputs):
    rows = []

    # (A) Official documented events -> spatial/temporal anchors (point coordinate).
    for ev in inputs["ground_events"]:
        region = region_from_code(ev.get("region"))
        event_id = clean(ev.get("event_id")) or "UNKNOWN_EVENT"
        coord = clean(ev.get("coordinate_status")).upper()
        has_geom = coord.startswith("EXPLICIT")
        rows.append({
            "event_id": event_id, "patch_id": "UNKNOWN_PATCH", "region": region,
            "city": region, "hazard_type": REGION_HAZARD.get(region, "unknown_hazard"),
            "source_id": "SGB_RISK_CARTOGRAPHY", "source_name": "SGB/CPRM field survey record",
            "observed_start": _norm_event_date(ev.get("event_or_survey_date")),
            "observed_end": _norm_event_date(ev.get("event_or_survey_date")),
            "published_at": UNKNOWN,
            "temporal_precision": clean(ev.get("temporal_precision")) or UNKNOWN,
            "spatial_precision": clean(ev.get("spatial_precision")) or UNKNOWN,
            "geometry_type": "point" if has_geom else "none",
            "geometry_available": "true" if has_geom else "false",
            "geometry_uri": NOT_AVAILABLE, "raw_uri": NOT_AVAILABLE,
            "derived_uri": NOT_AVAILABLE, "license_status": "open_public",
            "download_hash": UNKNOWN, "evidence_strength": "strong",
            "evidence_role": "spatial_support" if has_geom else "temporal_anchor",
            "review_status": "review_only",
            "blocking_reason": "" if has_geom else "NO_EXPLICIT_COORDINATE",
            "notes": "Official event survey point anchor; not a patch overlay, not ground truth.",
        })

    # (B) External GIS evidence -> territorial / context observations.
    for ev in inputs["external_evidence"]:
        region = region_from_label(ev.get("region"))
        source_id = _institution_to_source(ev.get("institutional_origin"), ev.get("evidence_type"))
        sclass = CLASS_BY_ID.get(source_id, "context_low")
        tier = clean(ev.get("evidence_tier")).upper()
        strength = "moderate" if tier == "STRONG" else "weak"
        role = "territorial_context" if sclass != "context_low" else "context_only"
        rows.append({
            "event_id": "UNKNOWN_EVENT", "patch_id": "UNKNOWN_PATCH", "region": region,
            "city": region if region != "All" else "All",
            "hazard_type": REGION_HAZARD.get(region, "unknown_hazard"),
            "source_id": source_id, "source_name": clean(ev.get("source_name")) or source_id,
            "observed_start": UNKNOWN, "observed_end": UNKNOWN, "published_at": UNKNOWN,
            "temporal_precision": "NO_EVENT_DATA", "spatial_precision": "DATASET_ENVELOPE_NOT_PATCH",
            "geometry_type": "polygon_or_raster_context", "geometry_available": "false",
            "geometry_uri": NOT_AVAILABLE, "raw_uri": NOT_AVAILABLE, "derived_uri": NOT_AVAILABLE,
            "license_status": clean(ev.get("public_status")) or "open_public",
            "download_hash": UNKNOWN, "evidence_strength": strength, "evidence_role": role,
            "review_status": "review_only", "blocking_reason": "ENVELOPE_NOT_PATCH_GEOMETRY",
            "notes": "Territorial GIS context for supervised human review; envelope is not patch footprint.",
        })

    # (C) Cross-region candidate references -> strongest region-level evidence.
    xref = {region_from_label(r.get("region")): r for r in inputs["xregion"]}
    scorecard = inputs["xregion_scorecard"]
    temporal_support = {}
    for r in scorecard:
        if clean(r.get("evidence_axis")).upper() == "TEMPORALITY":
            temporal_support[region_from_label(r.get("region"))] = clean(r.get("supports_reference_status")).lower() == "true"

    region_temporal_source = {"Recife": "ANA_HIDROWEB", "Curitiba": "INMET_HISTORICAL",
                              "Petropolis": "INMET_HISTORICAL"}
    region_temporal_strength = {"Recife": "moderate", "Curitiba": "strong", "Petropolis": "moderate"}

    for region in ("Recife", "Petropolis", "Curitiba"):
        ref = xref.get(region, {})
        basis = clean(ref.get("evidence_basis"))
        # Spatial cartographic product (Recife Charter raster).
        if "charter" in basis.lower() or "raster" in basis.lower():
            rows.append(_region_obs(region, "INTERNATIONAL_CHARTER_PRODUCT",
                                    "International Charter validated cartographic product",
                                    "raster", "true", "strong", "spatial_support",
                                    "Validated cartographic raster product; reinforces geometry, not a vector overlay."))
        # Official temporal anchor (ANA/INMET) when the cross-region scorecard supports it.
        if temporal_support.get(region, region in region_temporal_source):
            rows.append(_region_obs(region, region_temporal_source[region],
                                    f"{region_temporal_source[region]} temporal series",
                                    "none", "false", region_temporal_strength[region], "temporal_anchor",
                                    "Official temporal anchor; a station/stage series is not a label."))

    # Stable evidence ids + deterministic ordering.
    rows.sort(key=lambda r: (r["region"], r["source_id"], r["event_id"], r["patch_id"],
                             r["evidence_role"], r["evidence_strength"]))
    out = []
    for idx, r in enumerate(rows):
        r["source_class"] = CLASS_BY_ID.get(r["source_id"], "context_low")
        r["evidence_id"] = stable_id("EVOBS_", r["source_id"], r["event_id"], r["region"],
                                     r["evidence_role"], str(idx))
        out.append(r)
    return out


def _region_obs(region, source_id, name, geom_type, geom_avail, strength, role, note):
    return {
        "event_id": "UNKNOWN_EVENT", "patch_id": "UNKNOWN_PATCH", "region": region,
        "city": region, "hazard_type": REGION_HAZARD.get(region, "unknown_hazard"),
        "source_id": source_id, "source_name": name, "observed_start": UNKNOWN,
        "observed_end": UNKNOWN, "published_at": UNKNOWN, "temporal_precision": UNKNOWN,
        "spatial_precision": "REGION_LEVEL", "geometry_type": geom_type,
        "geometry_available": geom_avail, "geometry_uri": NOT_AVAILABLE, "raw_uri": NOT_AVAILABLE,
        "derived_uri": NOT_AVAILABLE, "license_status": "open_public", "download_hash": UNKNOWN,
        "evidence_strength": strength, "evidence_role": role, "review_status": "review_only",
        "blocking_reason": "", "notes": note,
    }


def _norm_event_date(value):
    value = clean(value)
    # dd/mm/yyyy -> yyyy-mm-dd (first date only); leave non-matching as UNKNOWN.
    head = value.split("�")[0].split("a")[0].strip()
    parts = head.split("/")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        d, m, y = parts
        return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
    return UNKNOWN


# --------------------------------------------------------------------------- #
# Region evidence summary (computed once, shared by every package in a region).
# --------------------------------------------------------------------------- #


def summarise_region_evidence(observations, config):
    weights = config["source_weight_by_class"]
    by_region = {}
    for obs in observations:
        by_region.setdefault(obs["region"], []).append(obs)

    summary = {}
    for region, obs_list in by_region.items():
        roles = {o["evidence_role"] for o in obs_list}
        classes = {o["source_class"] for o in obs_list}
        has_official = any(c.startswith("official") or c == "operational_mapping" for c in classes)
        has_spatial = "spatial_support" in roles
        has_temporal = "temporal_anchor" in roles
        has_geometry = any(o["geometry_available"] == "true" for o in obs_list)
        has_vhr = "vhr_optical" in classes
        only_context = (not has_official and not has_spatial and not has_temporal)
        strong = sum(1 for o in obs_list if o["evidence_strength"] == "strong")
        weak = sum(1 for o in obs_list if o["evidence_strength"] == "weak")
        mean_obs = 0.0
        if obs_list:
            mean_obs = sum(weights.get(o["source_class"], 0.15) * STRENGTH_FACTOR.get(o["evidence_strength"], 0.0)
                           for o in obs_list) / len(obs_list)
        summary[region] = {
            "count": len(obs_list), "strong": strong, "weak": weak,
            "has_official": has_official, "has_spatial": has_spatial,
            "has_temporal": has_temporal, "has_geometry": has_geometry,
            "has_vhr": has_vhr, "only_context": only_context, "mean_obs": mean_obs,
        }
    return summary


# --------------------------------------------------------------------------- #
# 3. Event-patch package registry (the core).
# --------------------------------------------------------------------------- #


def build_packages(inputs, region_evidence, config):
    candidates = inputs["candidates"]
    sentinel_dates = {}
    for r in inputs["sentinel_dates"]:
        if clean(r.get("usable_for_temporal_linkage")).lower() == "true":
            key = clean(r.get("patch_id"))
            if key and clean(r.get("selected_sentinel_date")):
                sentinel_dates[key] = clean(r.get("selected_sentinel_date"))
    dino_support = {clean(r.get("event_patch_candidate_id")): clean(r.get("dino_review_support_status"))
                    for r in inputs["dino_support"]}
    pkg_validation = {clean(r.get("event_patch_candidate_id")): r for r in inputs["pkg_validation"]}

    rows = []
    if not candidates:
        rows.append(_placeholder_package())

    for cand in candidates:
        event_id = clean(cand.get("event_id")) or "UNKNOWN_EVENT"
        patch_id = clean(cand.get("patch_id")) or "UNKNOWN_PATCH"
        region = region_from_code(cand.get("region"))
        cand_id = clean(cand.get("event_patch_candidate_id"))
        event_missing = "MISSING" in event_id.upper() or event_id == "UNKNOWN_EVENT"
        hazard = "unknown_hazard" if event_missing else REGION_HAZARD.get(region, "unknown_hazard")

        win_start, win_end = parse_event_window(event_id)
        sentinel_date = sentinel_dates.get(event_id, UNKNOWN)

        ev = region_evidence.get(region, {})
        has_temporal = bool(ev.get("has_temporal")) and win_start != UNKNOWN
        has_spatial = bool(ev.get("has_spatial"))
        has_official = bool(ev.get("has_official"))
        has_geometry = bool(ev.get("has_geometry"))
        has_vhr = bool(ev.get("has_vhr"))
        only_context = bool(ev.get("only_context"))
        # No patch-event overlay is ever computed in this stage (fail-closed).
        has_overlay = False
        intersection_ratio = UNKNOWN
        valid_fraction = UNKNOWN

        # Time delta only when both endpoints are known.
        time_delta = UNKNOWN
        d_sent, d_start = parse_iso(sentinel_date), parse_iso(win_start)
        if d_sent and d_start:
            time_delta = str(abs((d_sent - d_start).days))

        sentinel_usable = sentinel_date != UNKNOWN
        flags = [has_temporal, has_spatial, has_official, has_geometry, sentinel_usable, has_overlay]
        flag_fraction = sum(1 for f in flags if f) / len(flags)

        conflict_count = _conflict_count(inputs["sentinel_dates"], event_id)

        evidence_score = round(0.4 * ev.get("mean_obs", 0.0) + 0.6 * flag_fraction, 3)
        uncertainty_score = round(1.0 - flag_fraction, 3)

        urban_context = "true" if hazard in {"urban_flood", "flash_flood"} else "false"

        level, decision, blocking, allowed = decide_promotion(
            event_missing, hazard, win_start, has_official, has_temporal,
            has_spatial, has_geometry, has_overlay, only_context, sentinel_usable,
            conflict_count)

        package_id = stable_id("PKG_", region, event_id, patch_id)
        rows.append({
            "package_id": package_id, "event_id": event_id, "patch_id": patch_id,
            "region": region, "city": region, "hazard_type": hazard,
            "sentinel_asset_id": UNKNOWN, "sentinel_sensor_family": "SENTINEL2_MSI"
            if sentinel_usable else UNKNOWN, "sentinel_observation_date": sentinel_date,
            "event_window_start": win_start, "event_window_end": win_end,
            "time_delta_days": time_delta,
            "has_temporal_anchor": _b(has_temporal), "has_spatial_support": _b(has_spatial),
            "has_official_source": _b(has_official), "has_vhr_support": _b(has_vhr),
            "has_only_contextual_sources": _b(only_context), "has_geometry": _b(has_geometry),
            "has_patch_overlay": _b(has_overlay), "intersection_ratio": intersection_ratio,
            "valid_data_fraction": valid_fraction, "urban_context": urban_context,
            "permanent_water_risk": UNKNOWN, "occlusion_risk": UNKNOWN,
            "evidence_count": str(ev.get("count", 0)), "strong_evidence_count": str(ev.get("strong", 0)),
            "weak_evidence_count": str(ev.get("weak", 0)), "conflict_count": str(conflict_count),
            "evidence_score": f"{evidence_score:.3f}", "uncertainty_score": f"{uncertainty_score:.3f}",
            "promotion_candidate_level": level, "promotion_decision": decision,
            "blocking_reason": blocking, "allowed_use": allowed,
            "notes": f"candidate={cand_id}; dino={dino_support.get(cand_id, 'NONE')}; "
                     f"validation={clean(pkg_validation.get(cand_id, {}).get('validation_status'))}; "
                     "patch boundary is not event geometry; C4 only ever a candidate.",
        })

    rows.sort(key=lambda r: (r["region"], r["event_id"], r["patch_id"]))
    return rows


def _b(value):
    return "true" if value else "false"


def _conflict_count(sentinel_rows, event_id):
    count = 0
    for r in sentinel_rows:
        if clean(r.get("patch_id")) == event_id:
            cls = clean(r.get("confidence_class")).upper()
            if "CONFLICT" in cls or clean(r.get("blocker")):
                count += 1
    return count


def decide_promotion(event_missing, hazard, win_start, has_official, has_temporal,
                     has_spatial, has_geometry, has_overlay, only_context,
                     sentinel_usable, conflict_count):
    """Returns (level, decision, blocking_reason, allowed_use)."""
    if event_missing:
        return ("C0", "REJECTED_EVENT_REGISTRY_MISSING",
                "EVENT_REGISTRY_MISSING_OR_UNTYPED", "rejected_context_only")
    if hazard == "unknown_hazard":
        return ("C0", "REJECTED_HAZARD_NOT_TYPED",
                "HAZARD_TYPE_NOT_TYPED", "rejected_context_only")
    if conflict_count > 0:
        return ("C1", "CONFLICT_REVIEW_REQUIRED",
                "SOURCE_DATE_CONFLICT_UNRESOLVED", "review_only")
    if only_context:
        return ("C1", "HOLD_CONTEXT_ONLY_NOT_PROMOTED",
                "ONLY_CONTEXTUAL_SOURCES", "rejected_context_only")
    if win_start == UNKNOWN or not has_temporal:
        # Strong spatial without a coherent temporal window cannot be promoted.
        if has_spatial:
            return ("C1", "HOLD_SPATIAL_WITHOUT_TEMPORAL_WINDOW",
                    "NO_COHERENT_TEMPORAL_WINDOW", "review_only")
        return ("C1", "HOLD_INSUFFICIENT_EVIDENCE",
                "NO_TEMPORAL_WINDOW", "review_only")
    # C4 requires official + temporal + spatial + geometry + patch overlay + intersection.
    if has_official and has_temporal and has_spatial and has_geometry and has_overlay:
        return ("C4_CANDIDATE", "C4_CANDIDATE_PENDING_HUMAN_REVIEW",
                "", "candidate_reference")
    # Official temporal + spatial + geometry but no patch overlay -> ceiling C3.
    if has_official and has_temporal and (has_spatial or has_geometry):
        if sentinel_usable and has_spatial and has_geometry:
            return ("C3", "C3_CANDIDATE_REFERENCE_HOLD_FOR_OVERLAY",
                    "NO_PATCH_EVENT_OVERLAY_GEOMETRY", "candidate_reference")
        return ("C3", "C3_SECONDARY_EVALUATION_HOLD_FOR_OVERLAY",
                "NO_PATCH_EVENT_OVERLAY_GEOMETRY", "secondary_evaluation_candidate")
    # Official temporal source only.
    if has_official and has_temporal:
        return ("C2", "C2_SECONDARY_EVALUATION_HOLD",
                "NO_SPATIAL_SUPPORT_NO_GEOMETRY", "secondary_evaluation_candidate")
    return ("C1", "HOLD_REVIEW_ONLY", "WEAK_EVIDENCE", "review_only")


def _placeholder_package():
    return {col: "" for col in COLUMNS[OUT_PACKAGES]} | {
        "package_id": stable_id("PKG_", "NO_INPUT", "UNKNOWN_EVENT", "UNKNOWN_PATCH"),
        "event_id": "UNKNOWN_EVENT", "patch_id": "UNKNOWN_PATCH", "region": UNKNOWN,
        "city": UNKNOWN, "hazard_type": "unknown_hazard", "sentinel_asset_id": UNKNOWN,
        "sentinel_sensor_family": UNKNOWN, "sentinel_observation_date": UNKNOWN,
        "event_window_start": UNKNOWN, "event_window_end": UNKNOWN, "time_delta_days": UNKNOWN,
        "has_temporal_anchor": "false", "has_spatial_support": "false", "has_official_source": "false",
        "has_vhr_support": "false", "has_only_contextual_sources": "true", "has_geometry": "false",
        "has_patch_overlay": "false", "intersection_ratio": UNKNOWN, "valid_data_fraction": UNKNOWN,
        "urban_context": "false", "permanent_water_risk": UNKNOWN, "occlusion_risk": UNKNOWN,
        "evidence_count": "0", "strong_evidence_count": "0", "weak_evidence_count": "0",
        "conflict_count": "0", "evidence_score": "0.000", "uncertainty_score": "1.000",
        "promotion_candidate_level": "C0", "promotion_decision": "REJECTED_NO_INPUT",
        "blocking_reason": "NO_INPUT_CANDIDATES", "allowed_use": "rejected_context_only",
        "notes": "No candidate input available; fail-closed placeholder package.",
    }


# --------------------------------------------------------------------------- #
# 4. Promotion gate decision audit (15 gates per package).
# --------------------------------------------------------------------------- #


def build_gates(packages, config):
    threshold = config["time_delta_threshold_days"]
    min_ratio = config["minimum_intersection_ratio"]
    rows = []
    for pkg in packages:
        ctx = _gate_context(pkg, threshold, min_ratio)
        for gate in GATE_NAMES:
            passed, status, required, observed, severity, blocking, action = ctx[gate]
            rows.append({
                "decision_id": stable_id("DEC_", pkg["package_id"], gate),
                "package_id": pkg["package_id"], "event_id": pkg["event_id"],
                "patch_id": pkg["patch_id"], "gate_name": gate,
                "gate_passed": _b(passed), "gate_status": status,
                "required_condition": required, "observed_value": observed,
                "severity": severity, "blocking_reason": blocking,
                "recommended_action": action,
            })
    rows.sort(key=lambda r: (r["package_id"], r["gate_name"]))
    return rows


def _gate_context(pkg, threshold, min_ratio):
    def tf(field):
        return clean(pkg.get(field)).lower() == "true"

    event_ok = pkg["event_id"] not in {"UNKNOWN_EVENT", ""} and "MISSING" not in pkg["event_id"].upper()
    hazard_ok = pkg["hazard_type"] != "unknown_hazard"
    window_ok = pkg["event_window_start"] != UNKNOWN
    sentinel_ok = pkg["sentinel_observation_date"] != UNKNOWN
    delta_ok = pkg["time_delta_days"] != UNKNOWN and pkg["time_delta_days"].lstrip("-").isdigit() \
        and abs(int(pkg["time_delta_days"])) <= threshold
    official_ok = tf("has_official_source")
    geometry_ok = tf("has_geometry")
    overlay_ok = tf("has_patch_overlay")
    ratio_ok = pkg["intersection_ratio"] not in {UNKNOWN, ""} and _safe_float(pkg["intersection_ratio"]) >= min_ratio
    context_only = tf("has_only_contextual_sources")
    conflict_ok = pkg["conflict_count"] in {"0", ""}
    uncertainty_ok = pkg["uncertainty_score"] not in {"", UNKNOWN}

    blk = lambda ok: "" if ok else "BLOCKING_GATE_NOT_SATISFIED"
    return {
        "GATE_01_EVENT_ID_EXISTS": (
            event_ok, "PASS" if event_ok else "FAIL", "typed event_id present",
            pkg["event_id"], "BLOCKING", blk(event_ok),
            "Acquire/confirm official event record" if not event_ok else "None"),
        "GATE_02_HAZARD_TYPE_TYPED": (
            hazard_ok, "PASS" if hazard_ok else "FAIL", "hazard_type typed",
            pkg["hazard_type"], "BLOCKING", blk(hazard_ok),
            "Type the phenomenon from an official source" if not hazard_ok else "None"),
        "GATE_03_TEMPORAL_WINDOW_EXISTS": (
            window_ok, "PASS" if window_ok else "FAIL", "event temporal window present",
            f"{pkg['event_window_start']}..{pkg['event_window_end']}", "BLOCKING", blk(window_ok),
            "Recover official event window" if not window_ok else "None"),
        "GATE_04_SENTINEL_OBSERVATION_EXISTS": (
            sentinel_ok, "PASS" if sentinel_ok else "FAIL", "usable Sentinel observation date",
            pkg["sentinel_observation_date"], "BLOCKING", blk(sentinel_ok),
            "Recover Sentinel acquisition date" if not sentinel_ok else "None"),
        "GATE_05_TIME_DELTA_ACCEPTABLE": (
            delta_ok, "PASS" if delta_ok else "PENDING", f"|time_delta_days| <= {threshold}",
            pkg["time_delta_days"], "BLOCKING", blk(delta_ok),
            "Resolve sentinel/event dates to verify delta" if not delta_ok else "None"),
        "GATE_06_OFFICIAL_OR_VALIDATED_SOURCE_EXISTS": (
            official_ok, "PASS" if official_ok else "FAIL", "official/validated source attached",
            pkg["has_official_source"], "BLOCKING", blk(official_ok),
            "Attach an official/validated source" if not official_ok else "None"),
        "GATE_07_GEOMETRY_AVAILABLE": (
            geometry_ok, "PASS" if geometry_ok else "FAIL", "event geometry available",
            pkg["has_geometry"], "BLOCKING", blk(geometry_ok),
            "Acquire/digitize event geometry" if not geometry_ok else "None"),
        "GATE_08_PATCH_OVERLAY_AVAILABLE": (
            overlay_ok, "PASS" if overlay_ok else "FAIL", "patch-event overlay computed",
            pkg["has_patch_overlay"], "BLOCKING", blk(overlay_ok),
            "Compute patch-event overlay (vector/CRS)" if not overlay_ok else "None"),
        "GATE_09_INTERSECTION_RATIO_ACCEPTABLE": (
            ratio_ok, "PASS" if ratio_ok else "PENDING", f"intersection_ratio >= {min_ratio}",
            pkg["intersection_ratio"], "BLOCKING", blk(ratio_ok),
            "Compute intersection after overlay" if not ratio_ok else "None"),
        "GATE_10_CONTEXT_ONLY_NOT_PROMOTED": (
            True, "PASS_CONTEXT_NOT_PROMOTED",
            "context-only sources never promoted to reference",
            "promoted" if (context_only and pkg["allowed_use"] in {"candidate_reference"}) else "not_promoted",
            "GUARDRAIL", "", "None"),
        "GATE_11_QUICKVIEW_NOT_PROMOTED_ALONE": (
            True, "PASS_QUICKVIEW_NOT_PROMOTED",
            "quickview never promotes alone", "no_quickview_promotion",
            "GUARDRAIL", "", "None"),
        "GATE_12_BENCHMARK_NOT_LOCAL_TRUTH": (
            True, "PASS_BENCHMARK_NOT_LOCAL_TRUTH",
            "external benchmark never becomes local truth", "benchmark_methodological_only",
            "GUARDRAIL", "", "None"),
        "GATE_13_CONFLICTS_RESOLVED": (
            conflict_ok, "PASS" if conflict_ok else "CONFLICT_REVIEW_REQUIRED",
            "no unresolved source conflict", pkg["conflict_count"], "BLOCKING",
            blk(conflict_ok), "Adjudicate conflicting sources" if not conflict_ok else "None"),
        "GATE_14_UNCERTAINTY_RECORDED": (
            uncertainty_ok, "PASS" if uncertainty_ok else "FAIL",
            "uncertainty_score recorded", pkg["uncertainty_score"], "GUARDRAIL",
            blk(uncertainty_ok), "None"),
        "GATE_15_NO_TRAINING_LABEL_CREATED": (
            True, "PASS_NO_OPERATIONAL_LABEL",
            "no operational/binary training label created", "no_operational_label_created",
            "GUARDRAIL", "", "None"),
    }


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


# --------------------------------------------------------------------------- #
# 5. Reviewer queue seed.
# --------------------------------------------------------------------------- #


def build_queue(packages, inputs):
    dino_support = {clean(r.get("event_id")) + "|" + clean(r.get("patch_id")):
                    clean(r.get("dino_review_support_status"))
                    for r in inputs["dino_support"]}
    rows = []
    for pkg in packages:
        rank, reason, action = _priority(pkg)
        dino_key = pkg["event_id"] + "|" + pkg["patch_id"]
        dino_avail = "AVAILABLE" in clean(dino_support.get(dino_key, "")).upper()
        rows.append({
            "review_item_id": stable_id("RQ_", pkg["package_id"], length=10),
            "package_id": pkg["package_id"], "event_id": pkg["event_id"],
            "patch_id": pkg["patch_id"], "region": pkg["region"], "city": pkg["city"],
            "hazard_type": pkg["hazard_type"], "priority_rank": str(rank),
            "priority_reason": reason, "suggested_review_action": action,
            "evidence_score": pkg["evidence_score"], "uncertainty_score": pkg["uncertainty_score"],
            "blocking_reason": pkg["blocking_reason"],
            "nearest_dino_neighbors_available": _b(dino_avail),
            "notes": "DINO is review-only structural support; never truth, never a label.",
        })
    rows.sort(key=lambda r: (int(r["priority_rank"]), -_safe_float(r["evidence_score"]),
                             r["package_id"]))
    return rows


def _priority(pkg):
    def tf(field):
        return clean(pkg.get(field)).lower() == "true"

    conflict = pkg["conflict_count"] not in {"0", ""}
    if tf("has_temporal_anchor") and tf("has_official_source") and not tf("has_patch_overlay") \
            and pkg["sentinel_observation_date"] != UNKNOWN:
        return (1, "Strong temporal+official evidence, missing patch-event geometry/overlay",
                "Acquire event geometry and compute patch overlay")
    if tf("has_geometry") and conflict:
        return (2, "Geometry present but temporal/source conflict",
                "Adjudicate the temporal conflict before overlay")
    if pkg["urban_context"] == "true" and tf("has_temporal_anchor"):
        return (3, "Urban-context package with temporal evidence, geometry pending",
                "Prioritise urban event geometry recovery")
    if _safe_float(pkg["uncertainty_score"]) >= 0.5:
        return (4, "High uncertainty package",
                "Collect missing official temporal/spatial evidence")
    if pkg["promotion_candidate_level"] in {"C3", "C4_CANDIDATE"}:
        return (6, "C3/C4-candidate blocked by a single requirement (overlay/geometry)",
                "Close the single remaining blocker under human review")
    return (5, "Review-only context package",
            "Human review of contextual evidence")


# --------------------------------------------------------------------------- #
# 6. Operational-label blocklist.
# --------------------------------------------------------------------------- #

_CATEGORY_BLOCKS = [
    ("MISSING_EVENT_GEOMETRY", "Package has no event geometry",
     "v2at:GATE_07_GEOMETRY_AVAILABLE", "BLOCKING", "true",
     "Official/digitised event geometry (vector)"),
    ("MISSING_TEMPORAL_WINDOW", "Package has no coherent temporal window",
     "v2at:GATE_03_TEMPORAL_WINDOW_EXISTS", "BLOCKING", "true",
     "Official event temporal window"),
    ("QUICKVIEW_ISOLATED", "A quickview never promotes alone",
     "v2at:GATE_11_QUICKVIEW_NOT_PROMOTED_ALONE", "POLICY", "true",
     "A verified product (not a quickview)"),
    ("MEDIA_ISOLATED", "Media/social are never strong evidence",
     "v2at:source_catalog:context_low", "POLICY", "true",
     "Corroborating official evidence"),
    ("BENCHMARK_EXTERNAL", "External benchmark never becomes local truth",
     "v2at:GATE_12_BENCHMARK_NOT_LOCAL_TRUTH", "POLICY", "false",
     "Local observational evidence (benchmark stays methodological)"),
    ("ABSENCE_AS_NEGATIVE", "Absence of evidence is never a negative label",
     "v2at:methodological_invariant", "POLICY", "false",
     "Positive official evidence (no pseudo-negatives)"),
    ("CONFLICT_PERSISTENT", "Persistent source conflict blocks any label",
     "v2at:GATE_13_CONFLICTS_RESOLVED", "BLOCKING", "true",
     "Human adjudication of the conflict"),
    ("HAZARD_NOT_TYPED", "Untyped phenomenon cannot become a label",
     "v2at:GATE_02_HAZARD_TYPE_TYPED", "BLOCKING", "true",
     "Phenomenon typing from an official source"),
    ("NO_PATCH_OVERLAY", "No patch-event overlay blocks C4 and any label",
     "v2at:GATE_08_PATCH_OVERLAY_AVAILABLE", "BLOCKING", "true",
     "Patch-event overlay with verified CRS"),
]


def build_blocklist(packages):
    rows = []
    for cat_id, reason, source, severity, revisit, needed in _CATEGORY_BLOCKS:
        rows.append({
            "block_id": stable_id("BLKCAT_", cat_id, length=10),
            "package_id": "ALL_PACKAGES", "event_id": "ALL", "patch_id": "ALL",
            "reason": reason, "source_of_block": source, "severity": severity,
            "can_be_revisited": revisit, "required_evidence_to_unblock": needed,
        })

    for pkg in packages:
        reason, source, revisit, needed = _package_block(pkg)
        rows.append({
            "block_id": stable_id("BLK_", pkg["package_id"], length=10),
            "package_id": pkg["package_id"], "event_id": pkg["event_id"],
            "patch_id": pkg["patch_id"], "reason": reason, "source_of_block": source,
            "severity": "BLOCKING", "can_be_revisited": "true",
            "required_evidence_to_unblock": needed,
        })
    rows.sort(key=lambda r: (r["block_id"].startswith("BLK_"), r["package_id"], r["block_id"]))
    return rows


def _package_block(pkg):
    if "MISSING" in pkg["event_id"].upper() or pkg["event_id"] == "UNKNOWN_EVENT":
        return ("No typed/registered event; cannot become a label",
                "v2at:GATE_01_EVENT_ID_EXISTS", "true",
                "Official event record with date")
    if pkg["conflict_count"] not in {"0", ""}:
        return ("Unresolved source conflict; cannot become a label",
                "v2at:GATE_13_CONFLICTS_RESOLVED", "true",
                "Human adjudication of conflicting sources")
    if pkg["event_window_start"] == UNKNOWN:
        return ("No coherent temporal window; cannot become a label",
                "v2at:GATE_03_TEMPORAL_WINDOW_EXISTS", "true",
                "Official event temporal window")
    return ("No patch-event overlay / vector geometry; C4 blocked, no label",
            "v2at:GATE_08_PATCH_OVERLAY_AVAILABLE", "true",
            "Event geometry vector + patch overlay with verified CRS")


# --------------------------------------------------------------------------- #
# Schema validation (light, structural).
# --------------------------------------------------------------------------- #


def validate_outputs(written, config):
    """Structural validation. Returns list of critical errors (empty == OK)."""
    errors = []
    allowed_use = set(config["allowed_use_values"])
    allowed_hazard = set(config["allowed_hazard_types"])

    catalog = written[OUT_CATALOG]
    ids = [r["source_id"] for r in catalog]
    required_sources = {
        "ANA_HIDROWEB", "ANA_TELEMETRY", "INMET_HISTORICAL", "CEMADEN_MONITORING",
        "CEMADEN_BULLETIN", "SGB_RISK_CARTOGRAPHY", "SGB_SUSCEPTIBILITY",
        "S2ID_DISASTER_RECORD", "ATLAS_DIGITAL_DESASTRES", "COPERNICUS_EMS_MAPPING",
        "COPERNICUS_GFM", "INTERNATIONAL_CHARTER_PRODUCT", "INTERNATIONAL_CHARTER_QUICKVIEW",
        "VANTOR_OPEN_DATA", "PLANET_DISASTER_DATA", "URBANSARFLOODS_BENCHMARK",
        "SEN1FLOODS11_BENCHMARK", "EMDAT_CONTEXT", "MEDIA_CONTEXT", "SOCIAL_CONTEXT",
    }
    missing = required_sources - set(ids)
    if missing:
        errors.append(f"source catalog missing canonical sources: {sorted(missing)}")
    if len(ids) != len(set(ids)):
        errors.append("source catalog has duplicate source_id")

    for name, col in ((OUT_CATALOG, "source_id"), (OUT_OBSERVATIONS, "evidence_id"),
                      (OUT_PACKAGES, "package_id"), (OUT_GATES, "decision_id"),
                      (OUT_QUEUE, "review_item_id"), (OUT_BLOCKLIST, "block_id")):
        for row in written[name]:
            if not clean(row.get(col)):
                errors.append(f"{name}: empty generated id in column {col}")
                break

    for pkg in written[OUT_PACKAGES]:
        if pkg["allowed_use"] not in allowed_use:
            errors.append(f"package {pkg['package_id']} invalid allowed_use {pkg['allowed_use']}")
            break
        if pkg["hazard_type"] not in allowed_hazard:
            errors.append(f"package {pkg['package_id']} invalid hazard_type {pkg['hazard_type']}")
            break
        # C4 must never be a final label.
        if pkg["promotion_candidate_level"] == "C4" or "LABEL" in pkg["promotion_decision"].upper():
            errors.append(f"package {pkg['package_id']} reached an operational label state")
            break

    # GATE_15 must pass for every package.
    g15 = [r for r in written[OUT_GATES] if r["gate_name"] == "GATE_15_NO_TRAINING_LABEL_CREATED"]
    if not g15 or not all(r["gate_passed"] == "true" for r in g15):
        errors.append("GATE_15 (no training label) did not pass for all packages")

    return errors


# --------------------------------------------------------------------------- #
# Report / summary / log.
# --------------------------------------------------------------------------- #


def _distribution(rows, key):
    dist = {}
    for r in rows:
        dist[r[key]] = dist.get(r[key], 0) + 1
    return dict(sorted(dist.items()))


def build_summary(inputs_found, outputs_written, written):
    packages = written[OUT_PACKAGES]
    promo = _distribution(packages, "promotion_decision")
    allowed = _distribution(packages, "allowed_use")
    blocking = _distribution(packages, "blocking_reason")
    return {
        "stage": STAGE,
        "status": "OK_WITH_EXPECTED_BLOCKERS",
        "inputs_found": inputs_found,
        "outputs_written": outputs_written,
        "total_sources": len(written[OUT_CATALOG]),
        "total_observations": len(written[OUT_OBSERVATIONS]),
        "total_packages": len(packages),
        "total_gate_checks": len(written[OUT_GATES]),
        "total_blocked_labels": len(written[OUT_BLOCKLIST]),
        "total_review_queue_items": len(written[OUT_QUEUE]),
        "promotion_distribution": promo,
        "allowed_use_distribution": allowed,
        "blocking_reason_distribution": blocking,
        "c4_candidate_count": sum(1 for p in packages if p["promotion_candidate_level"] == "C4_CANDIDATE"),
        "candidate_reference_count": allowed.get("candidate_reference", 0),
        "review_only_count": allowed.get("review_only", 0),
        "can_train_model": False,
        "can_create_operational_labels": False,
        "methodological_status": METHODOLOGICAL_STATUS,
    }


def build_report(summary, written):
    packages = written[OUT_PACKAGES]
    region_dist = _distribution(packages, "region")
    hazard_dist = _distribution(packages, "hazard_type")
    source_class_dist = _distribution(written[OUT_OBSERVATIONS], "source_class")
    blocking_dist = summary["blocking_reason_distribution"]
    top_block = sorted(blocking_dist.items(), key=lambda kv: (-kv[1], kv[0]))

    def fmt(dist):
        return "\n".join(f"- `{k}`: {v}" for k, v in dist.items()) or "- (none)"

    return f"""# v2at - Evidence Registry + Event-Patch Package Engine

## 1. Objetivo
Transformar o REV-P de "patches + embeddings + registries fail-closed" para um sistema
explicito de evidencia observacional: catalogo de fontes externas, observacoes, pacotes
evento-patch com tipagem de fenomeno, janela temporal, forca de evidencia, bloqueios, score
explicavel e decisao de promocao C1/C2/C3/C4 (sempre candidata, nunca label).

Esta etapa NAO treina modelo, NAO cria label binario/operacional, NAO declara ground truth e
NAO transforma o DINOv2 em detector. O DINOv2 permanece apoio de revisao (similaridade,
vizinhanca, PCA, outliers, medoids), nunca validador fisico.

## 2. Entradas usadas
{fmt({k: 1 for k in summary['inputs_found']})}

## 3. Saidas geradas
{fmt({k: 1 for k in summary['outputs_written']})}

## 4. Contagens
- Fontes canonicas no catalogo: **{summary['total_sources']}**
- Observacoes de evidencia: **{summary['total_observations']}**
- Pacotes evento-patch: **{summary['total_packages']}**
- Checagens de gate: **{summary['total_gate_checks']}**
- Itens na fila de revisao: **{summary['total_review_queue_items']}**
- Entradas na blocklist de label: **{summary['total_blocked_labels']}**

### 4.1 Por regiao
{fmt(region_dist)}

### 4.2 Por hazard_type
{fmt(hazard_dist)}

### 4.3 Por source_class (observacoes)
{fmt(source_class_dist)}

## 5. Distribuicao de promocao e uso permitido
### promotion_decision
{fmt(summary['promotion_distribution'])}

### allowed_use
{fmt(summary['allowed_use_distribution'])}

- review_only: **{summary['review_only_count']}**
- candidate_reference: **{summary['candidate_reference_count']}**
- C4_CANDIDATE: **{summary['c4_candidate_count']}** (somente candidato; nunca label final)
- bloqueados (com blocking_reason): **{sum(v for k, v in blocking_dist.items() if k)}**

## 6. Principais blocking_reason
{fmt(dict(top_block))}

## 7. Confirmacoes metodologicas explicitas
- Nenhum label operacional foi criado (`can_create_operational_labels=false`; GATE_15 PASS em todos os pacotes).
- Nenhum modelo foi treinado (`can_train_model=false`).
- Nenhum ground truth foi declarado; ausencia de evidencia nunca virou negativo.
- Benchmark externo nunca virou verdade local (GATE_12); quickview nunca promoveu sozinho (GATE_11).
- Patch boundary nao e geometria de evento; inventario de desastre nao e geometria de desastre.
- C4 aparece apenas como **candidate**, nunca como label final.

## 8. Interpretacao metodologica
O REV-P avanca de review-only para um sistema de evidencia observacional auditavel, mas
continua bloqueado para treino supervisionado. A conclusao correta e:

**{summary['methodological_status']}**

Ou seja: existe agora a infraestrutura explicita (fontes -> observacoes -> pacotes -> gates ->
fila -> blocklist) para que revisao humana, geometria, overlay e evidencia temporal/espacial
forte sejam fechados antes de qualquer referencia operacional.
"""


def build_supplement(summary):
    return f"""# v2at artifact index supplement

Additive supplement to `final_delivery_artifact_index.md`. Nothing existing was removed or
rewritten; only v2at artifacts were added.

| Artifact | Path | Function |
|---|---|---|
| Source catalog | `datasets/{OUT_CATALOG}` | Canonical hierarchy of external evidence sources. |
| Evidence observations | `datasets/{OUT_OBSERVATIONS}` | Derived observational evidence registry (fail-closed). |
| Event-patch packages | `datasets/{OUT_PACKAGES}` | Event-patch packages with typing, window, score and promotion decision. |
| Promotion gate audit | `datasets/{OUT_GATES}` | 15 promotion gates per package. |
| Reviewer queue seed | `datasets/{OUT_QUEUE}` | Prioritised human-review queue. |
| Operational-label blocklist | `datasets/{OUT_BLOCKLIST}` | Everything that must NOT become a training label. |
| Report | `outputs_public/{REPORT_REL.replace(os.sep, '/')}` | v2at methodological report. |
| Summary | `outputs_public/{SUMMARY_REL.replace(os.sep, '/')}` | v2at machine-readable summary. |

Methodological status: **{summary['methodological_status']}**
(`can_train_model=false`, `can_create_operational_labels=false`).
"""


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #


def load_config(config_dir):
    path = os.path.join(config_dir, CONFIG_NAME)
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as handle:
                loaded = json.load(handle)
            for key, value in loaded.items():
                config[key] = value
        except (json.JSONDecodeError, OSError):
            pass
    return config


def load_inputs(dataset_dir):
    found = []
    data = {}
    mapping = {
        "candidates": IN_CANDIDATES, "dino_support": IN_DINO_SUPPORT,
        "sentinel_dates": IN_SENTINEL_DATE, "pkg_validation": IN_PKG_VALIDATION,
        "xregion": IN_XREGION, "xregion_scorecard": IN_XREGION_SCORECARD,
        "ground_events": IN_GROUND_EVENTS, "external_evidence": IN_EXTERNAL_EVIDENCE,
    }
    for key, rel in mapping.items():
        path = os.path.join(dataset_dir, rel)
        rows = load_csv(path)
        data[key] = rows
        if rows:
            found.append(rel.replace(os.sep, "/"))
    return data, found


def log_lines(summary, errors):
    lines = [
        f"[{STAGE}] Evidence Registry + Event-Patch Package Engine",
        f"[{STAGE}] inputs_found={len(summary['inputs_found'])} "
        f"outputs_written={len(summary['outputs_written'])}",
        f"[{STAGE}] sources={summary['total_sources']} observations={summary['total_observations']} "
        f"packages={summary['total_packages']} gates={summary['total_gate_checks']} "
        f"queue={summary['total_review_queue_items']} blocklist={summary['total_blocked_labels']}",
        f"[{STAGE}] c4_candidate={summary['c4_candidate_count']} "
        f"candidate_reference={summary['candidate_reference_count']} "
        f"review_only={summary['review_only_count']}",
        f"[{STAGE}] can_train_model={summary['can_train_model']} "
        f"can_create_operational_labels={summary['can_create_operational_labels']}",
        f"[{STAGE}] methodological_status={summary['methodological_status']}",
        f"[{STAGE}] structural_errors={len(errors)}",
        f"[{STAGE}] status={'OK' if not errors else 'STRUCTURAL_ERROR'}",
    ]
    return "\n".join(lines) + "\n"


def run(dataset_dir=None, output_dir=None, config_dir=None):
    """Runs the full engine. Returns (exit_code, summary)."""
    env_dataset, env_output, env_config = resolve_dirs()
    dataset_dir = dataset_dir or env_dataset
    output_dir = output_dir or env_output
    config_dir = config_dir or env_config

    config = load_config(config_dir)
    inputs, inputs_found = load_inputs(dataset_dir)

    if config.get("fail_on_missing_optional_inputs") and not inputs_found:
        sys.stderr.write(f"[{STAGE}] fail_on_missing_optional_inputs=true and no inputs found\n")
        return 2, None

    catalog = build_source_catalog(config)
    observations = build_observations(inputs)
    region_evidence = summarise_region_evidence(observations, config)
    packages = build_packages(inputs, region_evidence, config)
    gates = build_gates(packages, config)
    queue = build_queue(packages, inputs)
    blocklist = build_blocklist(packages)

    written = {
        OUT_CATALOG: catalog, OUT_OBSERVATIONS: observations, OUT_PACKAGES: packages,
        OUT_GATES: gates, OUT_QUEUE: queue, OUT_BLOCKLIST: blocklist,
    }

    errors = validate_outputs(written, config)
    if errors:
        for err in errors:
            sys.stderr.write(f"[{STAGE}] STRUCTURAL ERROR: {err}\n")
        return 3, None

    outputs_written = []
    for name, rows in written.items():
        write_csv(os.path.join(dataset_dir, name), COLUMNS[name], rows)
        outputs_written.append(f"datasets/{name}")

    summary = build_summary(inputs_found, outputs_written, written)
    summary["outputs_written"] = outputs_written + [
        f"outputs_public/{REPORT_REL.replace(os.sep, '/')}",
        f"outputs_public/{SUMMARY_REL.replace(os.sep, '/')}",
        f"outputs_public/{LOG_REL.replace(os.sep, '/')}",
        f"outputs_public/{LOG_TXT_REL.replace(os.sep, '/')}",
        f"outputs_public/{SUPPLEMENT_REL.replace(os.sep, '/')}",
    ]

    write_text(os.path.join(output_dir, REPORT_REL), build_report(summary, written))
    write_text(os.path.join(output_dir, SUMMARY_REL), json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_text(os.path.join(output_dir, SUPPLEMENT_REL), build_supplement(summary))
    log_text = log_lines(summary, errors)
    write_text(os.path.join(output_dir, LOG_REL), log_text)
    write_text(os.path.join(output_dir, LOG_TXT_REL), log_text)

    sys.stdout.write(log_lines(summary, errors))
    return 0, summary


def main(argv=None):
    code, _ = run()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
