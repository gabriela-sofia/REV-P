"""SUSC-17C3 Official Source Acquisition & Patch Coverage Audit (review-only).

Turns the NotebookLM / Protocol C findings into a programmatic, fail-closed
pipeline that (1) registers missing official/cartographic sources, (2) prioritizes
real observational acquisition, (3) audits whether candidate event geometry falls
inside existing patch grids, and (4) decides the rational next path (official
acquisition / grid expansion / SAR runtime / canary swap).

It NEVER changes score v6, creates score v7, training data, labels, models or
ground truth. Risk/susceptibility maps are never observed events; PE3D/MDE is a
contextual physical layer, not event evidence; neighborhood centroids are never
strong geometry; PDF without vector/coordinate stays blocked. No package,
coordinate, polygon, date or source is invented: everything is read from real
committed registries; absent things are recorded as missing/blocked.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import read_csv, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
DATASETS = ROOT / "datasets"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

AUDIT = OUT / "SUSC_WORKTREE_AUDIT_BEFORE_17C3.md"
SOURCE_SCHEMA = SCHEMAS / "susc_17c3_source_acquisition_target_schema_v1.json"
COVERAGE_SCHEMA = SCHEMAS / "susc_17c3_patch_coverage_audit_schema_v1.json"

# --- inputs (read-only) ---------------------------------------------------- #
OFFICIAL_REGISTRY = DATASETS / "official_observed_event_vector_registry.csv"
TEMPORAL_WINDOWS = DATASETS / "event_sentinel_temporal_window_registry.csv"
EXTERNAL_REGISTRY = DATASETS / "external_evidence_registry.csv"
EVENTS_13A = DAT / "susc_13a_strong_observed_events_parsed_v1.csv"
FEATURES = DAT / "susc_features_by_patch_v1.csv"
CHARTER_GEOJSON_17C2 = OUT / "susc_17c2_candidate_footprints.geojson"
SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"

# --- outputs --------------------------------------------------------------- #
SOURCE_TARGETS = OUT / "susc_17c3_official_source_acquisition_targets.csv"
EVIDENCE_LADDER = OUT / "susc_17c3_candidate_event_evidence_ladder.csv"
COVERAGE_AUDIT = OUT / "susc_17c3_patch_coverage_audit.csv"
MISSING_MANIFEST = OUT / "susc_17c3_missing_official_artifacts_manifest.csv"
DECISION_TABLE = OUT / "susc_17c3_next_action_decision_table.csv"
SUMMARY = OUT / "susc_17c3_summary.json"
REPORT = OUT / "SUSC_17C3_OFFICIAL_SOURCE_ACQUISITION_AND_PATCH_COVERAGE_REPORT.md"

REQUIRED_INPUTS = [AUDIT, SOURCE_SCHEMA, COVERAGE_SCHEMA, OFFICIAL_REGISTRY,
                   TEMPORAL_WINDOWS, EXTERNAL_REGISTRY, EVENTS_13A, FEATURES, SCORE_V6]

MANDATORY = (
    "O SUSC-17C3 audita aquisicao de fonte oficial e cobertura de patches "
    "review-only. Mapa de risco/suscetibilidade nunca e evento observado; "
    "PE3D/MDE e camada fisica contextual, nao evento; bairro/centroide nunca e "
    "geometria forte; PDF sem vetor/coordenada fica bloqueado. O 17C3 nao altera "
    "o score v6 oficial, nao cria score v7, nao cria treino, modelo, label ou "
    "ground truth, nao inventa pacote, coordenada, poligono, data ou fonte, e nao "
    "executa o benchmark 17B."
)

REGION_CITY = {"PET": "Petropolis", "REC": "Recife", "CUR": "Curitiba"}
REGION_KEY = {"PET": "petropolis", "REC": "recife", "CUR": "curitiba"}
# canonical event ids used across the SUSC-17 chain
EVENT_CANON = {"REC_2022_05": "REC_2022_05_24_30"}
CHARTER_EVENT = "REC_2022_05_24_30"

NOT_GT = "official_or_technical_candidate_evidence_review_only_not_ground_truth"


def _require_inputs() -> None:
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(str(p.relative_to(ROOT)) for p in missing))


def _na(value) -> str:
    value = ("" if value is None else str(value)).strip()
    return value if value else "not_available"


def _slug(text: str) -> str:
    out = "".join(c if c.isalnum() else "_" for c in (text or "").lower())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "x"


def _parse_bbox(bbox: str):
    parts = [p for p in (bbox or "").replace(",", ";").split(";") if p.strip()]
    try:
        vals = [float(p) for p in parts[:4]]
    except (ValueError, TypeError):
        return None
    if len(vals) != 4 or vals[2] < vals[0] or vals[3] < vals[1]:
        return None
    return tuple(vals)


def _charter_bbox() -> str:
    row = next((r for r in read_csv(EVENTS_13A) if r.get("event_id") == "SUSC13A_00001"), {})
    return (row.get("bbox") or "").strip()


# --------------------------------------------------------------------------- #
# Part 1 - official source acquisition targets
# --------------------------------------------------------------------------- #

SOURCE_FIELDS = [
    "source_target_id", "event_id", "city", "region", "event_date_or_period",
    "phenomenon_type", "source_name", "source_authority", "source_artifact_status",
    "expected_artifact_type", "expected_geometry_type", "expected_revp_class",
    "current_blocking_reason", "can_be_patch_level_if_acquired", "requires_formal_request",
    "requires_pdf_extraction", "requires_vector_download", "requires_human_review",
    "review_only", "trainable", "ground_truth",
]


def _gov() -> dict:
    return {"review_only": "true", "trainable": "false", "ground_truth": "false"}


def _status_from_official(row: dict) -> tuple[str, str]:
    access = (row.get("source_access_status") or "").upper()
    ingest = (row.get("local_ingestion_status") or "").upper()
    asset = (row.get("source_asset_name") or "").lower()
    if access == "PENDING_FORMAL_REQUEST" or ingest == "NOT_INGESTED":
        return "missing_source_artifact", "official_data_not_public_requires_formal_request"
    if access == "DOWNLOAD_OK" and ingest == "INGESTED":
        if asset.endswith((".pdf", ".zip")) or "pdf" in asset:
            return "pdf_only", "pdf_or_zip_without_vector_needs_coordinate_or_vector_extraction"
        return "available_local", "ingested_local_artifact_needs_human_review"
    if access in ("", "UNKNOWN"):
        return "unknown", "source_access_unknown"
    return "blocked_access", "source_access_blocked"


def _expected_class_for_authority(authority: str) -> str:
    a = (authority or "").lower()
    if "charter" in a:
        return "technical_remote_sensing_flood_footprint_candidate"
    if "pe3d" in a or "mde" in a:
        return "contextual_physical_layer"
    if "drm" in a or "compdec" in a or "defesa civil" in a or "ippuc" in a or "prefeitura" in a:
        return "official_observed_event_polygon"
    if "sgb" in a or "cprm" in a:
        return "official_observed_event_point"
    return "documentary_context"


def build_source_targets() -> list[dict]:
    rows: list[dict] = []
    # 1. official observed event vector registry (real Protocol C rows).
    for src in read_csv(OFFICIAL_REGISTRY):
        region = (src.get("region") or "").upper()
        event_id = EVENT_CANON.get(src.get("event_id", ""), src.get("event_id", ""))
        authority = src.get("source_institution", "")
        status, blocking = _status_from_official(src)
        asset = src.get("source_asset_name", "")
        expected_class = _expected_class_for_authority(authority)
        rows.append({
            "source_target_id": f"S17C3_ST_{event_id}_{_slug(authority)}_{_slug(asset)}",
            "event_id": event_id,
            "city": REGION_CITY.get(region, _na(region)),
            "region": _na(region),
            "event_date_or_period": _na(src.get("event_date")),
            "phenomenon_type": "flood_or_mass_movement",
            "source_name": _na(asset),
            "source_authority": _na(authority),
            "source_artifact_status": status,
            "expected_artifact_type": _na(src.get("source_asset_type")),
            "expected_geometry_type": "official_event_polygon_or_point",
            "expected_revp_class": expected_class,
            "current_blocking_reason": blocking,
            "can_be_patch_level_if_acquired": "true",
            "requires_formal_request": "true" if status == "missing_source_artifact" else "false",
            "requires_pdf_extraction": "true" if status == "pdf_only" else "false",
            "requires_vector_download": "true" if status == "available_remote_not_downloaded" else "false",
            "requires_human_review": "true",
            **_gov(),
        })
    # 2. International Charter Recife 2022-05-24/30 (only event geometry available).
    charter_bbox = _charter_bbox()
    rows.append({
        "source_target_id": f"S17C3_ST_{CHARTER_EVENT}_international_charter",
        "event_id": CHARTER_EVENT, "city": "Recife", "region": "REC",
        "event_date_or_period": "2022-05-24/2022-05-30",
        "phenomenon_type": "urban_flooding",
        "source_name": "International Charter flood extent (Recife 2022-05)",
        "source_authority": "International Charter Space and Major Disasters",
        "source_artifact_status": "available_local" if _parse_bbox(charter_bbox) else "missing_source_artifact",
        "expected_artifact_type": "remote_sensing_flood_extent_bbox",
        "expected_geometry_type": "Polygon_bbox",
        "expected_revp_class": "technical_remote_sensing_flood_footprint_candidate",
        "current_blocking_reason": "geometry_available_but_outside_patch_grid_and_needs_qa",
        "can_be_patch_level_if_acquired": "true",
        "requires_formal_request": "false", "requires_pdf_extraction": "false",
        "requires_vector_download": "false", "requires_human_review": "true", **_gov(),
    })
    # 3. PET_2024 Valparaiso-Floresta (dated, windows ready, geometry NOT acquired).
    pet24 = next((r for r in read_csv(TEMPORAL_WINDOWS)
                  if r.get("observed_event_id") == "PET_2024_03_21_28"), {})
    if pet24:
        rows.append({
            "source_target_id": "S17C3_ST_PET_2024_03_21_28_valparaiso_floresta",
            "event_id": "PET_2024_03_21_28", "city": "Petropolis", "region": "PET",
            "event_date_or_period": f"{_na(pet24.get('event_date_start'))}/{_na(pet24.get('event_date_end'))}",
            "phenomenon_type": "flash_flood_landslide",
            "source_name": _na(pet24.get("event_name")),
            "source_authority": "Defesa Civil Petropolis / DRM-RJ (a confirmar)",
            "source_artifact_status": "missing_source_artifact",
            "expected_artifact_type": "official_event_geometry",
            "expected_geometry_type": "official_event_polygon_or_point",
            "expected_revp_class": "official_observed_event_polygon",
            "current_blocking_reason": "temporal_windows_ready_but_acquisition_status_not_acquired",
            "can_be_patch_level_if_acquired": "true",
            "requires_formal_request": "true", "requires_pdf_extraction": "false",
            "requires_vector_download": "false", "requires_human_review": "true", **_gov(),
        })
    # 4. PE3D/MDE Recife - contextual physical layer, never event evidence.
    pe3d = next((r for r in read_csv(EXTERNAL_REGISTRY)
                 if "pe3d" in str(r.get("evidence_id", "")).lower()), {})
    if pe3d:
        rows.append({
            "source_target_id": "S17C3_ST_REC_pe3d_mde_context",
            "event_id": "REC_CONTEXT_PE3D", "city": "Recife", "region": "REC",
            "event_date_or_period": "not_applicable_context_layer",
            "phenomenon_type": "topographic_context",
            "source_name": _na(pe3d.get("source_name")),
            "source_authority": "Programa PE3D Pernambuco 3D",
            "source_artifact_status": "not_applicable_context_only",
            "expected_artifact_type": "terrain_raster_mdt_mds",
            "expected_geometry_type": "raster_terrain",
            "expected_revp_class": "contextual_physical_layer",
            "current_blocking_reason": "terrain_layer_is_context_not_observed_event",
            "can_be_patch_level_if_acquired": "false",
            "requires_formal_request": "false", "requires_pdf_extraction": "false",
            "requires_vector_download": "false", "requires_human_review": "false", **_gov(),
        })
    rows = [{f: _na(r.get(f)) if f not in _gov() else r[f] for f in SOURCE_FIELDS} for r in rows]
    rows.sort(key=lambda r: r["source_target_id"])
    return rows


# --------------------------------------------------------------------------- #
# Part 2 - evidence ladder
# --------------------------------------------------------------------------- #

LADDER_FIELDS = [
    "event_id", "evidence_level_current", "evidence_level_if_artifact_acquired",
    "current_revp_class", "possible_revp_class_after_acquisition", "has_event_date",
    "has_official_source", "has_geometry", "has_patch_intersection",
    "has_pre_post_window", "has_qa", "eligible_for_17b_now", "blocking_reason",
    "review_only", "trainable", "ground_truth",
]


def build_evidence_ladder(source_targets: list[dict], coverage: list[dict]) -> list[dict]:
    cov_by_event = {c["event_id"]: c for c in coverage}
    by_event: dict[str, list[dict]] = defaultdict(list)
    for st in source_targets:
        by_event[st["event_id"]].append(st)
    rows = []
    for event_id, targets in by_event.items():
        if event_id == "REC_CONTEXT_PE3D":
            continue  # PE3D is contextual, not an event ladder entry
        has_geom = any(t["source_artifact_status"] == "available_local"
                       and "technical_remote_sensing" in t["expected_revp_class"] for t in targets)
        has_date = any(t["event_date_or_period"] not in ("not_available", "not_applicable_context_layer")
                       for t in targets)
        cov = cov_by_event.get(event_id, {})
        intersects = cov.get("coverage_status") == "intersects_patch_grid"
        has_window = event_id in (CHARTER_EVENT, "PET_2024_03_21_28")
        pdf_only = all(t["source_artifact_status"] in ("pdf_only", "missing_source_artifact")
                       for t in targets)
        if has_geom and not intersects:
            current_class = "technical_remote_sensing_flood_footprint_candidate"
            level = "official_geometry_coarse_outside_grid_needs_qa"
        elif pdf_only:
            current_class = "documentary_context"
            level = "documentary_or_pdf_or_pending_formal_request_only"
        else:
            current_class = "insufficient_reference"
            level = "date_or_source_only_no_usable_geometry"
        rows.append({
            "event_id": event_id,
            "evidence_level_current": level,
            "evidence_level_if_artifact_acquired": (
                "official_observed_event_geometry_patch_linkable_after_qa"),
            "current_revp_class": current_class,
            "possible_revp_class_after_acquisition": (
                "official_observed_event_polygon"
                if event_id != CHARTER_EVENT else
                "technical_remote_sensing_flood_footprint"),
            "has_event_date": "true" if has_date else "false",
            "has_official_source": "true",
            "has_geometry": "true" if has_geom else "false",
            "has_patch_intersection": "true" if intersects else "false",
            "has_pre_post_window": "true" if has_window else "false",
            "has_qa": "false",
            "eligible_for_17b_now": "false",
            "blocking_reason": (
                "geometry_outside_patch_grid_and_no_qa" if has_geom and not intersects
                else "no_usable_official_geometry_yet"),
            **_gov(),
        })
    rows.sort(key=lambda r: r["event_id"])
    return rows


# --------------------------------------------------------------------------- #
# Part 3 - patch coverage audit
# --------------------------------------------------------------------------- #

COVERAGE_FIELDS = [
    "coverage_audit_id", "event_id", "city", "candidate_geometry_source",
    "geometry_available", "geometry_type", "geometry_bbox", "patch_grid_available",
    "patch_count_region", "intersecting_patch_count", "nearest_patch_distance_m",
    "coverage_status", "recommended_action", "review_only", "trainable", "ground_truth",
]


def _patch_grid_by_region() -> dict[str, list[tuple]]:
    grids: dict[str, list[tuple]] = defaultdict(list)
    for p in read_csv(FEATURES):
        try:
            box = (float(p["xmin"]), float(p["ymin"]), float(p["xmax"]), float(p["ymax"]))
        except (KeyError, ValueError, TypeError):
            continue
        grids[p.get("regiao", "")].append(box)
    return grids


def _bbox_gap_meters(fb, pb) -> float:
    dx = max(0.0, fb[0] - pb[2], pb[0] - fb[2])
    dy = max(0.0, fb[1] - pb[3], pb[1] - fb[3])
    lat = (fb[1] + fb[3]) / 2.0
    mx = dx * 111320.0 * math.cos(math.radians(lat))
    my = dy * 110540.0
    return math.hypot(mx, my)


def build_coverage_audit(source_targets: list[dict]) -> list[dict]:
    grids = _patch_grid_by_region()
    charter_bbox = _charter_bbox()
    # one audit per event that has (or could be tested for) geometry
    events = {}
    for st in source_targets:
        if st["event_id"] in ("REC_CONTEXT_PE3D",):
            continue
        events.setdefault(st["event_id"], st)
    rows = []
    for event_id, st in events.items():
        region_key = REGION_KEY.get(st["region"], "")
        grid = grids.get(region_key, [])
        patch_grid_available = "true" if grid else "false"
        fb = _parse_bbox(charter_bbox) if event_id == CHARTER_EVENT else None
        if fb is not None:
            inter = sum(1 for pb in grid
                        if min(fb[2], pb[2]) > max(fb[0], pb[0]) and min(fb[3], pb[3]) > max(fb[1], pb[1]))
            nearest = min((_bbox_gap_meters(fb, pb) for pb in grid), default=None)
            if not grid:
                status, action = "no_patch_grid", "needs_grid_expansion_review"
            elif inter > 0:
                status, action = "intersects_patch_grid", "human_qa_existing_geometry"
            else:
                status, action = "outside_patch_grid", "needs_grid_expansion_review"
            rows.append({
                "coverage_audit_id": f"S17C3_COV_{event_id}",
                "event_id": event_id, "city": st["city"],
                "candidate_geometry_source": "international_charter_bbox_susc_13a",
                "geometry_available": "true", "geometry_type": "Polygon_bbox",
                "geometry_bbox": charter_bbox,
                "patch_grid_available": patch_grid_available,
                "patch_count_region": len(grid),
                "intersecting_patch_count": inter,
                "nearest_patch_distance_m": f"{nearest:.1f}" if nearest is not None else "not_available",
                "coverage_status": status, "recommended_action": action, **_gov(),
            })
        else:
            rows.append({
                "coverage_audit_id": f"S17C3_COV_{event_id}",
                "event_id": event_id, "city": st["city"],
                "candidate_geometry_source": "not_available_no_official_geometry_yet",
                "geometry_available": "false", "geometry_type": "not_available",
                "geometry_bbox": "not_available",
                "patch_grid_available": patch_grid_available,
                "patch_count_region": len(grid),
                "intersecting_patch_count": 0,
                "nearest_patch_distance_m": "not_available",
                "coverage_status": "no_geometry_to_test",
                "recommended_action": "needs_geometry_first", **_gov(),
            })
    rows.sort(key=lambda r: r["coverage_audit_id"])
    return rows


# --------------------------------------------------------------------------- #
# Part 4 - missing official artifacts manifest
# --------------------------------------------------------------------------- #

MANIFEST_FIELDS = [
    "artifact_request_id", "source_target_id", "event_id", "artifact_name",
    "expected_provider", "expected_format", "why_needed", "expected_revp_use",
    "request_method", "priority", "blocking_17b", "notes",
]


def _priority_for(target: dict) -> str | None:
    """Map a missing source target to one of the six named priority buckets, or
    None if it is not in the named-priority scope (e.g. Defesa Civil Petropolis,
    CUR_HISTORICAL) - those stay tracked in source_targets and decisions only."""
    auth = target["source_authority"].lower()
    name = target["source_name"].lower()
    status = target["source_artifact_status"]
    if target["event_id"] == "PET_2024_03_21_28":
        return "P1_PET_2024_VALPARAISO_FLORESTA_GEOMETRY"
    if "drm" in auth or "pkg_fr_pet_001" in name:
        return "P0_DRM_RJ_PKG_FR_PET_001"
    if "compdec" in auth or "pkg_fr_rec" in name:
        return "P0_COMPDEC_RECIFE_2022_POINTS_OR_ADDRESSES"
    if ("sgb" in auth or "cprm" in auth) and status == "pdf_only":
        return "P1_SGB_CPRM_FIELD_COORDINATES_OR_VECTOR"
    if "pe3d" in auth or "pe3d" in name:
        return "P2_PE3D_MDE_CONTEXT_LAYER"
    return None


def build_missing_manifest(source_targets: list[dict]) -> list[dict]:
    rows = []
    for st in source_targets:
        if st["source_artifact_status"] in ("available_local",):
            continue  # already have it locally
        priority = _priority_for(st)
        if priority is None:
            continue  # not in the named-priority scope; tracked in targets/decisions only
        is_context = st["expected_revp_class"] == "contextual_physical_layer"
        rows.append({
            "artifact_request_id": f"S17C3_REQ_{st['source_target_id'].replace('S17C3_ST_', '')}",
            "source_target_id": st["source_target_id"],
            "event_id": st["event_id"],
            "artifact_name": st["source_name"],
            "expected_provider": st["source_authority"],
            "expected_format": st["expected_artifact_type"],
            "why_needed": ("camada fisica de contexto para revisao" if is_context
                           else "geometria oficial de evento ocorrido para link patch-level e avaliacao"),
            "expected_revp_use": st["expected_revp_class"],
            "request_method": ("formal_institutional_request" if st["requires_formal_request"] == "true"
                               else "pdf_coordinate_extraction" if st["requires_pdf_extraction"] == "true"
                               else "context_layer_download"),
            "priority": priority,
            "blocking_17b": "false" if is_context else "true",
            "notes": st["current_blocking_reason"],
            **_gov(),
        })
    # add SAR runtime credentials request (P2) - real gap from 17C2
    rows.append({
        "artifact_request_id": "S17C3_REQ_SAR_RUNTIME_CREDENTIALS",
        "source_target_id": "not_applicable_runtime",
        "event_id": "RECIFE_SAR_WINDOWS_2014",
        "artifact_name": "GEE/STAC runtime credentials for Sentinel-1 execution",
        "expected_provider": "Google Earth Engine / Copernicus Data Space",
        "expected_format": "api_credential",
        "why_needed": "executar os 4 canaries SAR Recife bloqueados no SUSC-17C2",
        "expected_revp_use": "technical_remote_sensing_flood_footprint_candidate",
        "request_method": "credential_provisioning",
        "priority": "P2_SAR_RUNTIME_CREDENTIALS",
        "blocking_17b": "true",
        "notes": "sem runtime no ambiente atual (SUSC-17C2 blocked_no_runtime_access)",
        **_gov(),
    })
    rows = [{f: _na(r.get(f)) if f not in _gov() else r[f] for f in MANIFEST_FIELDS + list(_gov())} for r in rows]
    rows.sort(key=lambda r: r["artifact_request_id"])
    return rows


# --------------------------------------------------------------------------- #
# Part 5 - next action decision table
# --------------------------------------------------------------------------- #

DECISION_FIELDS = [
    "decision_id", "event_id", "source_target_id", "city",
    "current_state", "next_action", "rationale", "blocking_17b",
    "review_only", "trainable", "ground_truth",
]

NEXT_ACTIONS = {
    "acquire_official_polygon", "acquire_official_point", "request_formal_data",
    "extract_pdf_coordinates", "run_sar_runtime", "expand_patch_grid_review",
    "drop_for_17b_context_only", "human_qa_existing_geometry", "needs_more_source",
}


def _decide(st: dict, coverage_by_event: dict) -> tuple[str, str]:
    status = st["source_artifact_status"]
    cls = st["expected_revp_class"]
    if cls == "contextual_physical_layer":
        return "drop_for_17b_context_only", "camada topografica de contexto, nao evento observado"
    if st["event_id"] == CHARTER_EVENT and status == "available_local":
        cov = coverage_by_event.get(CHARTER_EVENT, {})
        if cov.get("coverage_status") == "intersects_patch_grid":
            return "human_qa_existing_geometry", "geometria oficial intersecta a grade; revisar"
        return "expand_patch_grid_review", "unica geometria disponivel cai fora da grade de patches"
    if status == "missing_source_artifact" and st["requires_formal_request"] == "true":
        return "request_formal_data", "pacote oficial nao publico; exige solicitacao formal"
    if status == "pdf_only":
        return "extract_pdf_coordinates", "PDF/relatorio sem vetor; extrair coordenada/vetor antes de qualquer uso forte"
    if status == "available_remote_not_downloaded":
        return "acquire_official_point", "fonte remota disponivel; baixar e auditar"
    return "needs_more_source", "fonte insuficiente para evidencia patch-level"


def build_decision_table(source_targets: list[dict], coverage: list[dict]) -> list[dict]:
    cov_by_event = {c["event_id"]: c for c in coverage}
    rows = []
    for st in source_targets:
        action, rationale = _decide(st, cov_by_event)
        rows.append({
            "decision_id": f"S17C3_DEC_{st['source_target_id'].replace('S17C3_ST_', '')}",
            "event_id": st["event_id"], "source_target_id": st["source_target_id"],
            "city": st["city"], "current_state": st["source_artifact_status"],
            "next_action": action, "rationale": rationale,
            "blocking_17b": "false" if st["expected_revp_class"] == "contextual_physical_layer" else "true",
            **_gov(),
        })
    rows.sort(key=lambda r: r["decision_id"])
    return rows


# --------------------------------------------------------------------------- #
# summary + report
# --------------------------------------------------------------------------- #

def build_summary(source_targets, coverage, manifest, ladder) -> dict:
    status_counts = Counter(s["source_artifact_status"] for s in source_targets)
    cov_counts = Counter(c["coverage_status"] for c in coverage)
    p0 = sum(1 for m in manifest if m["priority"].startswith("P0"))
    blocks_17b = [
        "no_qa_accepted_yet",
        "p0_official_packages_missing_drm_rj_and_compdec",
        "only_event_geometry_charter_falls_outside_patch_grid",
        "sar_runtime_unavailable",
    ]
    next_milestone = (
        "SUSC-17C4 Official Artifact Ingestion (ingerir/solicitar PKG_FR_PET_001 DRM-RJ e "
        "PKG_FR_REC_002 COMPDEC, P0); em paralelo SUSC-17C4 Patch Grid Expansion Review para a "
        "geometria do International Charter que cai fora da grade; SAR runtime e QA dependem dessas etapas")
    summary = {
        "total_source_targets": len(source_targets),
        "p0_targets": p0,
        "missing_official_artifacts_count": status_counts.get("missing_source_artifact", 0),
        "available_local_artifacts_count": status_counts.get("available_local", 0),
        "pdf_only_count": status_counts.get("pdf_only", 0),
        "context_only_count": status_counts.get("not_applicable_context_only", 0),
        "coverage_audits_count": len(coverage),
        "intersects_patch_grid_count": cov_counts.get("intersects_patch_grid", 0),
        "outside_patch_grid_count": cov_counts.get("outside_patch_grid", 0),
        "needs_geometry_first_count": cov_counts.get("no_geometry_to_test", 0),
        "missing_artifacts_manifest_count": len(manifest),
        "events_with_geometry": sorted({l["event_id"] for l in ladder if l["has_geometry"] == "true"}),
        "events_intersecting_grid": sorted({c["event_id"] for c in coverage
                                            if c["coverage_status"] == "intersects_patch_grid"}),
        "events_outside_grid": sorted({c["event_id"] for c in coverage
                                       if c["coverage_status"] == "outside_patch_grid"}),
        "eligible_for_17b_now": False,
        "blocks_17b": blocks_17b,
        "recommended_next_milestone": next_milestone,
        "review_only": True, "trainable": False, "ground_truth": False,
        "score_v6_changed": False, "score_v7_created": False,
    }
    write_json(SUMMARY, summary)
    return summary


def build_report(summary, coverage, manifest) -> int:
    p0_list = [m for m in manifest if m["priority"].startswith("P0")]
    p0_lines = "\n".join(f"- `{m['priority']}` -> {m['artifact_name']} ({m['expected_provider']})" for m in p0_list)
    cov_lines = "\n".join(
        f"- `{c['event_id']}` ({c['city']}): {c['coverage_status']} "
        f"(intersecoes={c['intersecting_patch_count']}, nearest={c['nearest_patch_distance_m']})"
        for c in coverage)
    report = f"""# SUSC-17C3 - Official Source Acquisition & Patch Coverage Audit (review-only)

Status: review-only. `trainable=false`; `ground_truth=false`; `score_v6_changed=false`; `score_v7_created=false`; `eligible_for_17b_now=false`.

{MANDATORY}

## 1. O que o NotebookLM encontrou

Fontes oficiais reais ja referenciadas no Protocolo C: relatorios e anexos SGB/CPRM (Petropolis 2022), pacote formal DRM-RJ `PKG_FR_PET_001`, pacote formal COMPDEC/Defesa Civil PE `PKG_FR_REC_002`, evento Petropolis 2024 (Valparaiso-Floresta), o mapa de extensao de inundacao do International Charter (Recife 2022-05) e a camada topografica PE3D/MDE.

## 2. O que e realmente utilizavel

Apenas o footprint do International Charter (Recife 2022-05) traz geometria de evento utilizavel hoje - e ainda assim coarse, sem QA e fora da grade de patches. Os demais sao PDFs sem vetor ou pacotes nao baixados.

## 3. O que e so contexto

PE3D/MDE e camada fisica contextual (terreno), nunca evidencia de evento observado. Entra como `contextual_physical_layer`, fora do caminho de evidencia forte.

## 4. Por que `ground reference` foi rebaixado para `reference_evidence`

O Protocolo C usava o termo `ground_reference`, mas no SUSC-17A o protocolo foi formalizado como `reference_evidence` review-only: footprints e geometrias oficiais sao evidencia candidata, nunca ground truth nem label. O 17C3 mantem essa disciplina (`ground_truth=false` em todas as linhas).

## 5. Por que DRM-RJ / PKG_FR_PET_001 e P0

E o pacote formal com maior chance de trazer setores/poligonos oficiais do evento Petropolis 2022-02-15 dentro da malha urbana (potencial intersecao com a grade de patches), hoje `PENDING_FORMAL_REQUEST`.

## 6. Por que COMPDEC Recife e P0

O `PKG_FR_REC_002` (COMPDEC/Defesa Civil PE) pode trazer pontos/enderecos oficiais do evento Recife 2022-05 dentro da grade, complementando o footprint coarse do International Charter; hoje `PENDING_FORMAL_REQUEST`.

## 7. Por que PE3D/MDE e contextual, nao evento

PE3D/MDE e modelo de terreno (MDT/MDS): descreve a fisiografia, nao registra onde/quando houve inundacao. Usar como evento confundiria suscetibilidade fisica com ocorrencia.

## 8. Por que SGB/CPRM PDF esta bloqueado ate vetor/coordenada

Os artefatos SGB/CPRM existem localmente (`DOWNLOAD_OK`, `INGESTED`) mas sao PDF/ZIP sem vetor. Sem extracao de coordenada/vetor auditavel ficam `pdf_only` e nunca viram avaliacao forte.

## 9. Qual candidato intersecta ou nao a grade de patches

{cov_lines}

A unica geometria (International Charter) cai FORA da grade Recife (adjacente a borda NE). Os demais nao tem geometria para testar.

## 10. Proximo passo mais racional

P0 - ingestao/solicitacao dos pacotes oficiais:
{p0_lines}

{summary['recommended_next_milestone']}

`blocks_17b`: {json.dumps(summary['blocks_17b'], ensure_ascii=False)}.
"""
    write_markdown(REPORT, report)
    return 0


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

def run_all() -> int:
    _require_inputs()
    source_targets = build_source_targets()
    write_csv(SOURCE_TARGETS, source_targets, SOURCE_FIELDS)
    coverage = build_coverage_audit(source_targets)
    write_csv(COVERAGE_AUDIT, coverage, COVERAGE_FIELDS)
    ladder = build_evidence_ladder(source_targets, coverage)
    write_csv(EVIDENCE_LADDER, ladder, LADDER_FIELDS)
    manifest = build_missing_manifest(source_targets)
    write_csv(MISSING_MANIFEST, manifest, MANIFEST_FIELDS + list(_gov()))
    decisions = build_decision_table(source_targets, coverage)
    write_csv(DECISION_TABLE, decisions, DECISION_FIELDS)
    summary = build_summary(source_targets, coverage, manifest, ladder)
    build_report(summary, coverage, manifest)
    print(f"17C3 -> targets={len(source_targets)} coverage={len(coverage)} "
          f"ladder={len(ladder)} manifest={len(manifest)} decisions={len(decisions)}")
    return 0


# --------------------------------------------------------------------------- #
# validation (fail-closed)
# --------------------------------------------------------------------------- #

def schema_violations(row: dict, schema: dict) -> list[str]:
    out = []
    props = schema.get("properties", {})
    for field in schema.get("required", []):
        if not str(row.get(field, "")).strip():
            out.append(f"required empty: {field}")
    for field, spec in props.items():
        if field not in row:
            continue
        value = row.get(field, "")
        if "const" in spec and value != spec["const"]:
            out.append(f"{field}={value} must be {spec['const']}")
        if "enum" in spec and value not in spec["enum"]:
            out.append(f"{field}={value} not in enum")
    return out


def _gov_violations(row: dict) -> list[str]:
    out = []
    if row.get("review_only") != "true":
        out.append("review_only not true")
    if row.get("trainable") != "false":
        out.append("trainable true")
    if row.get("ground_truth") != "false":
        out.append("ground_truth true")
    if row.get("score_v7_created", "false") == "true":
        out.append("score_v7_created true")
    if row.get("official_score_changed", "false") == "true":
        out.append("official_score_changed true")
    return out


def validate() -> int:
    errors: list[str] = []
    required_files = [AUDIT, SOURCE_SCHEMA, COVERAGE_SCHEMA, SOURCE_TARGETS,
                      EVIDENCE_LADDER, COVERAGE_AUDIT, MISSING_MANIFEST,
                      DECISION_TABLE, SUMMARY, REPORT]
    for path in required_files:
        if not path.exists():
            errors.append(f"missing: {path.relative_to(ROOT)}")
    if errors:
        print("FAILED: SUSC-17C3 validation")
        for e in errors:
            print("  ERROR:", e)
        return 1

    source_schema = json.loads(SOURCE_SCHEMA.read_text(encoding="utf-8"))
    coverage_schema = json.loads(COVERAGE_SCHEMA.read_text(encoding="utf-8"))
    targets = read_csv(SOURCE_TARGETS)
    coverage = read_csv(COVERAGE_AUDIT)
    ladder = read_csv(EVIDENCE_LADDER)
    manifest = read_csv(MISSING_MANIFEST)
    decisions = read_csv(DECISION_TABLE)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    # unique + deterministic IDs
    for label, rows, key in (
        ("targets", targets, "source_target_id"),
        ("coverage", coverage, "coverage_audit_id"),
        ("manifest", manifest, "artifact_request_id"),
        ("decisions", decisions, "decision_id"),
    ):
        ids = [r[key] for r in rows]
        if len(ids) != len(set(ids)):
            errors.append(f"{label}: ids not unique")
        if ids != sorted(ids):
            errors.append(f"{label}: ids not sorted (non-deterministic)")

    # schema + governance on source targets
    for line_no, row in enumerate(targets, start=2):
        for v in schema_violations(row, source_schema):
            errors.append(f"{SOURCE_TARGETS.name}:{line_no}: {v}")
        for v in _gov_violations(row):
            errors.append(f"{SOURCE_TARGETS.name}:{line_no}: {v}")
        # PE3D/MDE never an observed event (rule 10)
        if "pe3d" in row["source_authority"].lower() and row["expected_revp_class"] != "contextual_physical_layer":
            errors.append(f"{SOURCE_TARGETS.name}:{line_no}: PE3D not contextual")
        # PDF without vector never strong (rule 12)
        if row["source_artifact_status"] == "pdf_only" and row["requires_pdf_extraction"] != "true":
            errors.append(f"{SOURCE_TARGETS.name}:{line_no}: pdf_only without extraction flag")
        # neighborhood/centroid never strong geometry (rule 13)
        if "centroid" in row["expected_geometry_type"].lower() or "neighborhood" in row["expected_geometry_type"].lower():
            if "polygon" in row["expected_revp_class"].lower():
                errors.append(f"{SOURCE_TARGETS.name}:{line_no}: neighborhood treated as strong")

    # schema + governance + geometry rules on coverage
    for line_no, row in enumerate(coverage, start=2):
        for v in schema_violations(row, coverage_schema):
            errors.append(f"{COVERAGE_AUDIT.name}:{line_no}: {v}")
        for v in _gov_violations(row):
            errors.append(f"{COVERAGE_AUDIT.name}:{line_no}: {v}")
        # no geometry -> no invented bbox (rule 14/16)
        if row["geometry_available"] == "false" and row["geometry_bbox"] != "not_available":
            errors.append(f"{COVERAGE_AUDIT.name}:{line_no}: invented bbox without geometry")
        # outside grid or no geometry -> 0 intersecting patches (rules 14/15)
        if row["coverage_status"] in ("outside_patch_grid", "no_geometry_to_test", "needs_geometry_first"):
            if str(row["intersecting_patch_count"]) not in ("0", ""):
                errors.append(f"{COVERAGE_AUDIT.name}:{line_no}: patch link outside grid / no geometry")

    # ladder + manifest + decisions governance
    for rows, name in ((ladder, "ladder"), (manifest, "manifest"), (decisions, "decisions")):
        for row in rows:
            for v in _gov_violations(row):
                errors.append(f"{name}: {v}")

    # ladder: eligible_for_17b_now false (rules 16/17)
    for row in ladder:
        if row["eligible_for_17b_now"] != "false":
            errors.append("ladder: eligible_for_17b_now not false before QA")

    # decisions: next_action within allowed set
    for row in decisions:
        if row["next_action"] not in NEXT_ACTIONS:
            errors.append(f"decisions: bad next_action {row['next_action']}")

    # P0 includes DRM-RJ PKG_FR_PET_001 and COMPDEC Recife (rules 16/17 of spec)
    priorities = {m["priority"] for m in manifest}
    if "P0_DRM_RJ_PKG_FR_PET_001" not in priorities:
        errors.append("manifest missing P0 DRM-RJ PKG_FR_PET_001")
    if "P0_COMPDEC_RECIFE_2022_POINTS_OR_ADDRESSES" not in priorities:
        errors.append("manifest missing P0 COMPDEC Recife")

    # summary honesty
    if summary.get("eligible_for_17b_now") is not False:
        errors.append("summary eligible_for_17b_now true")
    if summary.get("score_v7_created") is not False or summary.get("score_v6_changed") is not False:
        errors.append("summary score flags wrong")
    if summary.get("trainable") is not False or summary.get("ground_truth") is not False:
        errors.append("summary trainable/ground_truth wrong")
    if summary.get("total_source_targets") != len(targets):
        errors.append("summary total_source_targets mismatch")
    if summary.get("coverage_audits_count") != len(coverage):
        errors.append("summary coverage count mismatch")
    cov_counts = Counter(c["coverage_status"] for c in coverage)
    if summary.get("intersects_patch_grid_count") != cov_counts.get("intersects_patch_grid", 0):
        errors.append("summary intersects count mismatch")
    if summary.get("outside_patch_grid_count") != cov_counts.get("outside_patch_grid", 0):
        errors.append("summary outside count mismatch")

    # cross guards
    if SCORE_V7.exists():
        errors.append("score v7 artifact exists")
    if not SCORE_V6.exists():
        errors.append("score v6 official source missing")
    if MANDATORY not in REPORT.read_text(encoding="utf-8"):
        errors.append("mandatory report phrase missing")

    if errors:
        print("FAILED: SUSC-17C3 validation")
        for e in errors:
            print("  ERROR:", e)
        return 1
    print("PASSED: SUSC-17C3 validations passed")
    return 0
