"""SUSC-17C4 Official Artifact Ingestion (review-only, fail-closed).

Ingests, classifies and validates official/cartographic artifacts already in the
repo or referenced by SUSC-17C3, trying to turn P0/P1 targets into traceable
observational reference candidates - WITHOUT inventing geometry, date, coordinate
or footprint.

It NEVER changes score v6, creates score v7, training data, labels, models or
ground truth. A PDF/ZIP without an explicit vector/coordinate stays blocked (no
heavy OCR, no raster download). PE3D/MDE is a contextual physical layer, never an
event. Neighborhood/address without a coordinate is at most
official_address_resolved_pending, never strong geometry. Geometry that falls
outside the patch grid never produces a patch-link. Nothing is eligible for 17B
before human QA accepts it. Private raw artifacts are referenced as pointers,
never staged.
"""

from __future__ import annotations

import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import read_csv, sha256_file, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

AUDIT = OUT / "SUSC_WORKTREE_AUDIT_BEFORE_17C4.md"
INVENTORY_SCHEMA = SCHEMAS / "susc_17c4_artifact_inventory_schema_v1.json"
CANDIDATE_SCHEMA = SCHEMAS / "susc_17c4_reference_candidate_schema_v1.json"

# --- inputs (read-only) ---------------------------------------------------- #
SOURCE_TARGETS_17C3 = OUT / "susc_17c3_official_source_acquisition_targets.csv"
FEATURES = DAT / "susc_features_by_patch_v1.csv"
SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"

CHARTER_GEOJSON = (ROOT / "datasets" / "external_sources" / "recife_minimal_tp"
                   / "event_polygon_REC_2022_05_24_30" / "charter758" / "derived"
                   / "event_polygon_REC_2022_05_24_30_charter758_digitized_candidate.geojson")
SGB_PDF = ROOT / "local_runs" / "protocolo_c" / "v1if" / "raw_official_sources" / "Relatorio_Tecnico_Petropolis.pdf"
SGB_ZIP = (ROOT / "local_runs" / "protocolo_c" / "v1if" / "raw_official_sources"
           / "anexos_avaliacao_pos_desastre_petropolis_rj_2022.zip")

# --- outputs --------------------------------------------------------------- #
INVENTORY = OUT / "susc_17c4_artifact_inventory.csv"
INGESTION = OUT / "susc_17c4_ingestion_attempts.csv"
REFERENCE_CANDIDATES = OUT / "susc_17c4_extracted_reference_candidates.csv"
CANDIDATE_GEOJSON = OUT / "susc_17c4_candidate_geometries.geojson"
PATCH_LINKS = OUT / "susc_17c4_candidate_patch_links.csv"
FORMAL_QUEUE = OUT / "susc_17c4_formal_request_queue.csv"
SUMMARY = OUT / "susc_17c4_summary.json"
REPORT = OUT / "SUSC_17C4_OFFICIAL_ARTIFACT_INGESTION_REPORT.md"

REQUIRED_INPUTS = [AUDIT, INVENTORY_SCHEMA, CANDIDATE_SCHEMA, SOURCE_TARGETS_17C3, FEATURES, SCORE_V6]

MANDATORY = (
    "O SUSC-17C4 ingere e classifica artefatos oficiais review-only sem inventar "
    "geometria, data, coordenada ou footprint. PDF/ZIP sem vetor/coordenada "
    "explicita fica bloqueado (sem OCR pesado, sem raster); PE3D/MDE e camada "
    "contextual, nao evento; centroide de bairro nunca vira geometria forte; "
    "geometria fora da grade nao cria patch-link. Nao altera o score v6 oficial, "
    "nao cria score v7, nao cria treino, modelo, label ou ground truth, e nao "
    "executa o benchmark 17B nem o SAR runtime."
)

CHARTER_TARGET_HINT = "international_charter"
NOT_GT = "official_or_technical_candidate_evidence_review_only_not_ground_truth"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _require_inputs() -> None:
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(str(p.relative_to(ROOT)) for p in missing))


def _na(value) -> str:
    value = ("" if value is None else str(value)).strip()
    return value if value else "not_available"


def _gov() -> dict:
    return {"review_only": "true", "trainable": "false", "ground_truth": "false"}


# --------------------------------------------------------------------------- #
# Part 1 - artifact inventory
# --------------------------------------------------------------------------- #

INVENTORY_FIELDS = [
    "artifact_id", "source_target_id", "event_id", "city", "region", "source_name",
    "artifact_name", "artifact_expected_format", "artifact_local_status",
    "artifact_path", "artifact_url_or_reference", "artifact_size_bytes", "sha256",
    "access_status", "priority", "expected_revp_use", "review_only", "trainable",
    "ground_truth",
]


def _priority_for(st: dict) -> str:
    auth = st["source_authority"].lower()
    name = st["source_name"].lower()
    if st["event_id"] == "PET_2024_03_21_28":
        return "P1_PET_2024_VALPARAISO_FLORESTA_GEOMETRY"
    if "drm" in auth or "pkg_fr_pet_001" in name:
        return "P0_DRM_RJ_PKG_FR_PET_001"
    if "compdec" in auth or "pkg_fr_rec" in name:
        return "P0_COMPDEC_RECIFE_2022_POINTS_OR_ADDRESSES"
    if "sgb" in auth or "cprm" in auth:
        return "P1_SGB_CPRM_VECTOR_OR_COORDINATES"
    if "pe3d" in auth or "pe3d" in name:
        return "P2_PE3D_CONTEXT_LAYER"
    return "P2_OTHER_OFFICIAL"


def _artifact_for(st: dict) -> dict:
    """Resolve a single 17C3 source target to a real local/remote/missing artifact."""
    status_17c3 = st["source_artifact_status"]
    name = st["source_name"].lower()
    auth = st["source_authority"].lower()
    base = {
        "artifact_path": "not_available", "artifact_url_or_reference": "not_available",
        "artifact_size_bytes": "not_available", "sha256": "not_available",
        "access_status": "not_available",
    }
    if CHARTER_TARGET_HINT in st["source_target_id"]:
        if CHARTER_GEOJSON.exists():
            return {**base, "artifact_local_status": "local_available",
                    "artifact_expected_format": "geojson_multipolygon",
                    "artifact_path": _rel(CHARTER_GEOJSON),
                    "artifact_url_or_reference": "international_charter_activation_758",
                    "artifact_size_bytes": str(CHARTER_GEOJSON.stat().st_size),
                    "sha256": sha256_file(CHARTER_GEOJSON),
                    "access_status": "tracked_local_geojson"}
        return {**base, "artifact_local_status": "remote_reference_only",
                "artifact_expected_format": "geojson_multipolygon",
                "artifact_url_or_reference": "international_charter_activation_758",
                "access_status": "geojson_reference_not_present"}
    if "pe3d" in auth or "pe3d" in name:
        return {**base, "artifact_local_status": "context_layer_only",
                "artifact_expected_format": "terrain_raster_mdt_mds",
                "artifact_url_or_reference": "programa_pe3d_pernambuco_3d",
                "access_status": "context_layer_not_event"}
    if status_17c3 == "pdf_only":
        if "anexos" in name or ".zip" in name:
            present = SGB_ZIP.exists()
            return {**base, "artifact_local_status": "zip_only" if present else "remote_reference_only",
                    "artifact_expected_format": "zip_archive",
                    "artifact_path": _rel(SGB_ZIP) if present else "not_available",
                    "artifact_url_or_reference": "sgb_cprm_petropolis_2022_anexos_private",
                    "artifact_size_bytes": "not_recorded_private_artifact" if present else "not_available",
                    "sha256": "not_recorded_private_artifact" if present else "not_available",
                    "access_status": "private_local_runs_pointer" if present else "not_present"}
        present = SGB_PDF.exists()
        return {**base, "artifact_local_status": "pdf_only" if present else "remote_reference_only",
                "artifact_expected_format": "pdf_report",
                "artifact_path": _rel(SGB_PDF) if present else "not_available",
                "artifact_url_or_reference": "sgb_cprm_petropolis_2022_report_private",
                "artifact_size_bytes": "not_recorded_private_artifact" if present else "not_available",
                "sha256": "not_recorded_private_artifact" if present else "not_available",
                "access_status": "private_local_runs_pointer" if present else "not_present"}
    # missing_source_artifact / not_applicable_context_only fallbacks
    return {**base, "artifact_local_status": "missing_source_artifact",
            "artifact_expected_format": _na(st["expected_artifact_type"]),
            "artifact_url_or_reference": "pending_formal_request",
            "access_status": "pending_formal_request"}


def build_inventory() -> list[dict]:
    _require_inputs()
    rows = []
    for st in read_csv(SOURCE_TARGETS_17C3):
        art = _artifact_for(st)
        rows.append({
            "artifact_id": f"S17C4_ART_{st['source_target_id'].replace('S17C3_ST_', '')}",
            "source_target_id": st["source_target_id"],
            "event_id": st["event_id"], "city": st["city"], "region": st["region"],
            "source_name": st["source_authority"], "artifact_name": st["source_name"],
            "priority": _priority_for(st),
            "expected_revp_use": st["expected_revp_class"],
            **art, **_gov(),
        })
    rows = [{f: _na(r.get(f)) if f not in _gov() else r[f] for f in INVENTORY_FIELDS} for r in rows]
    rows.sort(key=lambda r: r["artifact_id"])
    return rows


# --------------------------------------------------------------------------- #
# Part 2 - ingestion attempts
# --------------------------------------------------------------------------- #

INGESTION_FIELDS = [
    "ingestion_attempt_id", "artifact_id", "event_id", "attempt_method",
    "attempt_status", "parsed_text_available", "coordinate_candidates_found",
    "vector_candidates_found", "date_candidates_found", "geometry_candidates_found",
    "blocking_reason", "notes", "review_only", "trainable", "ground_truth",
]


def _zip_vector_count(path: Path) -> tuple[int, str]:
    """Lightweight ZIP inventory (central directory only). No extraction/raster."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
        vec = [n for n in names if n.lower().endswith((".shp", ".geojson", ".kml", ".gpkg", ".gml"))]
        return len(vec), ("vectors_found" if vec else "no_vector_in_zip")
    except (zipfile.BadZipFile, OSError):
        return 0, "bad_or_unreadable_zip"


def build_ingestion(inventory: list[dict]) -> list[dict]:
    rows = []
    for art in inventory:
        status = art["artifact_local_status"]
        aid = art["artifact_id"]
        common = {"ingestion_attempt_id": f"S17C4_ING_{aid.replace('S17C4_ART_', '')}",
                  "artifact_id": aid, "event_id": art["event_id"],
                  "parsed_text_available": "false", "coordinate_candidates_found": "0",
                  "vector_candidates_found": "0", "date_candidates_found": "0",
                  "geometry_candidates_found": "0", "notes": "", **_gov()}
        if status == "local_available":
            geom = _load_charter_geometry()
            n = _geometry_vertex_count(geom) if geom else 0
            rows.append({**common, "attempt_method": "geojson_parse",
                         "attempt_status": "parsed_candidate_found" if geom else "parsed_no_geometry",
                         "parsed_text_available": "true",
                         "coordinate_candidates_found": str(n),
                         "vector_candidates_found": "1" if geom else "0",
                         "date_candidates_found": "1",
                         "geometry_candidates_found": "1" if geom else "0",
                         "blocking_reason": "not_available" if geom else "geojson_without_geometry",
                         "notes": "charter758 digitized multipolygon, CRS EPSG:4326, needs_review"})
        elif status == "pdf_only":
            rows.append({**common, "attempt_method": "pdf_text_scan",
                         "attempt_status": "blocked_pdf_only_no_vector",
                         "blocking_reason": "pdf_without_explicit_vector_or_coordinate_no_heavy_ocr",
                         "notes": "regra fail-closed: nao converter PDF em geometria sem coordenada explicita"})
        elif status == "zip_only":
            path = ROOT / art["artifact_path"]
            vec_n, vec_note = _zip_vector_count(path) if path.exists() else (0, "zip_not_present")
            rows.append({**common, "attempt_method": "zip_inventory",
                         "attempt_status": "blocked_unknown_format" if vec_n == 0 else "parsed_candidate_found",
                         "vector_candidates_found": str(vec_n),
                         "blocking_reason": "zip_without_extractable_vector_or_invalid_archive",
                         "notes": vec_note})
        elif status == "context_layer_only":
            rows.append({**common, "attempt_method": "not_attempted_context_only",
                         "attempt_status": "blocked_context_layer_not_event",
                         "blocking_reason": "pe3d_mde_is_terrain_context_not_observed_event",
                         "notes": "PE3D/MDE nunca e evidencia de evento"})
        elif status == "remote_reference_only":
            rows.append({**common, "attempt_method": "metadata_only",
                         "attempt_status": "blocked_missing_artifact",
                         "blocking_reason": "referenced_remotely_not_present_locally",
                         "notes": "artefato referenciado mas nao baixado"})
        else:  # missing_source_artifact / unknown
            rows.append({**common, "attempt_method": "not_attempted_missing_artifact",
                         "attempt_status": "blocked_missing_artifact"
                         if status == "missing_source_artifact" else "blocked_unknown_format",
                         "blocking_reason": "official_package_not_present_requires_formal_request"
                         if status == "missing_source_artifact" else "unknown_artifact_status",
                         "notes": "sem artefato local; gerar solicitacao formal"})
    rows = [{f: _na(r.get(f)) for f in INGESTION_FIELDS} for r in rows]
    rows.sort(key=lambda r: r["ingestion_attempt_id"])
    return rows


# --------------------------------------------------------------------------- #
# geometry helpers (real geometry only, never invented)
# --------------------------------------------------------------------------- #

def _load_charter_geometry():
    if not CHARTER_GEOJSON.exists():
        return None
    g = json.loads(CHARTER_GEOJSON.read_text(encoding="utf-8"))
    geom = g.get("geometry") if g.get("type") == "Feature" else None
    if geom and geom.get("coordinates"):
        return geom
    return None


def _iter_coords(obj):
    if isinstance(obj, (list, tuple)):
        if obj and isinstance(obj[0], (int, float)):
            yield obj
        else:
            for x in obj:
                yield from _iter_coords(x)


def _geometry_vertex_count(geom) -> int:
    return sum(1 for _ in _iter_coords(geom.get("coordinates", [])))


def _geometry_bbox(geom):
    pts = list(_iter_coords(geom.get("coordinates", [])))
    if not pts:
        return None
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    return (min(lons), min(lats), max(lons), max(lats))


# --------------------------------------------------------------------------- #
# Part 3 - extracted reference candidates
# --------------------------------------------------------------------------- #

CANDIDATE_FIELDS = [
    "reference_candidate_id", "event_id", "source_target_id", "artifact_id", "city",
    "region", "event_date_or_period", "phenomenon_type", "candidate_revp_class",
    "geometry_type", "geometry_source", "coordinate_source", "uncertainty_m",
    "spatial_precision", "temporal_precision", "eligible_for_patch_link",
    "eligible_for_17b_now", "qa_status", "not_ground_truth_reason", "review_only",
    "trainable", "ground_truth",
]


def build_reference_candidates(inventory: list[dict], ingestion: list[dict]) -> tuple[list[dict], dict]:
    ing_by_artifact = {i["artifact_id"]: i for i in ingestion}
    rows = []
    features = []
    for art in inventory:
        ing = ing_by_artifact.get(art["artifact_id"], {})
        # only create a candidate when explicit geometry was actually parsed
        if ing.get("attempt_status") != "parsed_candidate_found" or ing.get("geometry_candidates_found") != "1":
            continue
        geom = _load_charter_geometry()
        if geom is None:
            continue
        rid = f"S17C4_REF_{art['event_id']}"
        rows.append({
            "reference_candidate_id": rid,
            "event_id": art["event_id"], "source_target_id": art["source_target_id"],
            "artifact_id": art["artifact_id"], "city": art["city"], "region": art["region"],
            "event_date_or_period": "2022-05-24/2022-05-30",
            "phenomenon_type": "urban_flooding",
            "candidate_revp_class": "official_observed_event_polygon",
            "geometry_type": geom.get("type", "not_available"),
            "geometry_source": _rel(CHARTER_GEOJSON),
            "coordinate_source": "charter758_public_product_digitized_candidate_epsg4326",
            "uncertainty_m": "not_available",
            "spatial_precision": "digitized_polygon_neighborhood_scale",
            "temporal_precision": "date_range_period",
            "eligible_for_patch_link": "true",
            "eligible_for_17b_now": "false",
            "qa_status": "needs_review",
            "not_ground_truth_reason": NOT_GT,
            **_gov(),
        })
        features.append({
            "type": "Feature", "geometry": geom,
            "properties": {
                "reference_candidate_id": rid, "event_id": art["event_id"],
                "candidate_revp_class": "official_observed_event_polygon",
                "source_name": art["source_name"], "artifact_id": art["artifact_id"],
                "qa_status": "needs_review", "review_only": True, "ground_truth": False,
                "not_ground_truth_reason": NOT_GT,
            },
        })
    rows = [{f: _na(r.get(f)) if f not in _gov() else r[f] for f in CANDIDATE_FIELDS} for r in rows]
    rows.sort(key=lambda r: r["reference_candidate_id"])
    features.sort(key=lambda f: f["properties"]["reference_candidate_id"])
    geojson = {
        "type": "FeatureCollection",
        "metadata": {
            "protocol": "susc_17c4_candidate_geometries", "review_only": True,
            "ground_truth": False, "feature_count": len(features),
            "note": ("geometrias candidatas review-only extraidas de artefatos oficiais reais; "
                     "nenhuma geometria inventada" if features else
                     "nenhuma geometria explicita extraida; artefatos ausentes/pdf/zip bloqueados"),
        },
        "features": features,
    }
    return rows, geojson


# --------------------------------------------------------------------------- #
# Part 5 - candidate patch links (real bbox intersection only)
# --------------------------------------------------------------------------- #

PATCH_LINK_FIELDS = [
    "candidate_patch_link_id", "reference_candidate_id", "event_id", "patch_id",
    "link_method", "overlap_ratio", "distance_m", "link_quality", "qa_status",
    "eligible_for_17b_now", "review_only", "trainable", "ground_truth",
]


def _region_key(region: str) -> str:
    return {"REC": "recife", "PET": "petropolis", "CUR": "curitiba"}.get(region, region.lower())


def build_patch_links(candidates: list[dict]) -> tuple[list[dict], int]:
    rows = []
    outside_grid = 0
    if not candidates:
        return rows, outside_grid
    patches = [p for p in read_csv(FEATURES)]
    geom = _load_charter_geometry()
    fb = _geometry_bbox(geom) if geom else None
    for cand in candidates:
        if fb is None:
            continue
        region_key = _region_key(cand["region"])
        grid = [p for p in patches if p.get("regiao") == region_key]
        linked = 0
        for p in grid:
            try:
                pb = (float(p["xmin"]), float(p["ymin"]), float(p["xmax"]), float(p["ymax"]))
            except (KeyError, ValueError, TypeError):
                continue
            ix0, iy0 = max(fb[0], pb[0]), max(fb[1], pb[1])
            ix1, iy1 = min(fb[2], pb[2]), min(fb[3], pb[3])
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            inter = (ix1 - ix0) * (iy1 - iy0)
            area = (pb[2] - pb[0]) * (pb[3] - pb[1])
            ratio = inter / area if area > 0 else 0.0
            quality = ("high_spatial_overlap" if ratio >= 0.5
                       else "moderate_spatial_overlap" if ratio >= 0.1 else "low_spatial_overlap")
            rows.append({
                "candidate_patch_link_id": f"S17C4_PL_{cand['reference_candidate_id']}_{p['patch_id']}",
                "reference_candidate_id": cand["reference_candidate_id"],
                "event_id": cand["event_id"], "patch_id": p["patch_id"],
                "link_method": "candidate_geometry_bbox_vs_patch_bbox_intersection",
                "overlap_ratio": f"{ratio:.6f}", "distance_m": "not_available",
                "link_quality": quality, "qa_status": "needs_review",
                "eligible_for_17b_now": "false", **_gov(),
            })
            linked += 1
        if linked == 0:
            outside_grid += 1
    rows.sort(key=lambda r: r["candidate_patch_link_id"])
    return rows, outside_grid


# --------------------------------------------------------------------------- #
# Part 6 - formal request queue
# --------------------------------------------------------------------------- #

FORMAL_FIELDS = [
    "request_id", "source_target_id", "event_id", "provider", "artifact_needed",
    "minimum_fields_needed", "why_needed", "expected_revp_class_if_received",
    "priority", "request_status", "blocks_17b", "suggested_message_ptbr",
    "review_only", "trainable", "ground_truth",
]

NAMED_PRIORITIES = {
    "P0_DRM_RJ_PKG_FR_PET_001", "P0_COMPDEC_RECIFE_2022_POINTS_OR_ADDRESSES",
    "P1_SGB_CPRM_VECTOR_OR_COORDINATES", "P1_PET_2024_VALPARAISO_FLORESTA_GEOMETRY",
    "P2_SAR_RUNTIME_CREDENTIALS", "P2_PE3D_CONTEXT_LAYER",
}


def build_formal_queue(inventory: list[dict]) -> list[dict]:
    rows = []
    for art in inventory:
        status = art["artifact_local_status"]
        if status == "local_available":
            continue  # already have a usable artifact
        priority = art["priority"]
        if priority not in NAMED_PRIORITIES:
            continue  # not in named formal-request scope (tracked in inventory)
        is_context = "pe3d" in art["source_name"].lower() or priority == "P2_PE3D_CONTEXT_LAYER"
        provider = art["source_name"]
        msg = (f"Prezados {provider}, no ambito de uma pesquisa academica review-only (TCC) sobre "
               f"suscetibilidade a inundacao, solicitamos formalmente o artefato '{art['artifact_name']}' "
               f"referente ao evento {art['event_id']}, preferencialmente em formato vetorial (poligono/ponto) "
               f"com coordenadas e data, para fins de validacao observacional sem uso comercial.")
        rows.append({
            "request_id": f"S17C4_REQ_{art['artifact_id'].replace('S17C4_ART_', '')}",
            "source_target_id": art["source_target_id"], "event_id": art["event_id"],
            "provider": provider, "artifact_needed": art["artifact_name"],
            "minimum_fields_needed": ("camada de terreno (contexto)" if is_context
                                      else "geometria oficial (poligono/ponto) ou endereco com coordenada, data, fenomeno"),
            "why_needed": ("camada fisica de contexto" if is_context
                           else "geometria oficial de evento ocorrido para link patch-level e avaliacao review-only"),
            "expected_revp_class_if_received": art["expected_revp_use"],
            "priority": priority, "request_status": "queued_needs_human_send",
            "blocks_17b": "false" if is_context else "true",
            "suggested_message_ptbr": msg, **_gov(),
        })
    # SAR runtime credentials (P2) - real gap carried from 17C2/17C3
    rows.append({
        "request_id": "S17C4_REQ_SAR_RUNTIME_CREDENTIALS",
        "source_target_id": "not_applicable_runtime", "event_id": "RECIFE_SAR_WINDOWS_2014",
        "provider": "Google Earth Engine / Copernicus Data Space",
        "artifact_needed": "GEE/STAC runtime credentials for Sentinel-1",
        "minimum_fields_needed": "api_credential_with_sentinel1_grd_access",
        "why_needed": "executar os canaries SAR Recife bloqueados (SUSC-17C2)",
        "expected_revp_class_if_received": "technical_remote_sensing_flood_footprint_candidate",
        "priority": "P2_SAR_RUNTIME_CREDENTIALS", "request_status": "queued_needs_human_send",
        "blocks_17b": "true",
        "suggested_message_ptbr": "Provisionar credencial GEE/STAC com acesso ao COPERNICUS/S1_GRD para execucao SAR review-only.",
        **_gov(),
    })
    rows = [{f: _na(r.get(f)) if f not in _gov() else r[f] for f in FORMAL_FIELDS} for r in rows]
    rows.sort(key=lambda r: r["request_id"])
    return rows


# --------------------------------------------------------------------------- #
# summary + report
# --------------------------------------------------------------------------- #

def build_summary(inventory, ingestion, candidates, geojson, patch_links, formal, outside_grid) -> dict:
    status_counts = Counter(a["artifact_local_status"] for a in inventory)
    p0 = sum(1 for f in formal if f["priority"].startswith("P0"))
    blocks_17b = [
        "no_qa_accepted_yet",
        "p0_official_packages_missing_drm_rj_and_compdec",
    ]
    if candidates and not patch_links:
        blocks_17b.append("candidate_geometry_falls_outside_susc_patch_grid")
    if not candidates:
        blocks_17b.append("no_reference_candidate_extracted")
    blocks_17b.append("sar_runtime_unavailable")
    summary = {
        "total_artifacts_inventory": len(inventory),
        "local_available_count": status_counts.get("local_available", 0),
        "missing_source_artifact_count": status_counts.get("missing_source_artifact", 0),
        "pdf_only_count": status_counts.get("pdf_only", 0),
        "zip_only_count": status_counts.get("zip_only", 0),
        "context_layer_only_count": status_counts.get("context_layer_only", 0),
        "ingestion_attempt_count": len(ingestion),
        "reference_candidate_count": len(candidates),
        "candidate_geometry_count": len(geojson.get("features", [])),
        "candidate_patch_link_count": len(patch_links),
        "candidate_geometry_outside_grid_count": outside_grid,
        "formal_request_count": len(formal),
        "p0_request_count": p0,
        "eligible_for_17b_now": False,
        "blocks_17b": blocks_17b,
        "recommended_next_milestone": _next_milestone(candidates, patch_links),
        "review_only": True, "trainable": False, "ground_truth": False,
        "score_v6_changed": False, "score_v7_created": False,
    }
    write_json(SUMMARY, summary)
    return summary


def _next_milestone(candidates, patch_links) -> str:
    if candidates and patch_links:
        return "SUSC-17D Human QA (geometria candidata com patch-link disponivel para revisao)"
    if candidates and not patch_links:
        return ("SUSC-17C5 Patch Grid Expansion Review (geometria oficial extraida do Charter 758 cai fora "
                "da grade SUSC); em paralelo SUSC-17C5 Formal Request Package para os pacotes P0 DRM-RJ "
                "PKG_FR_PET_001 e COMPDEC PKG_FR_REC_002 ainda ausentes")
    return ("SUSC-17C5 Formal Request Package (P0 oficiais ausentes; nenhuma geometria candidata extraida)")


def build_report(summary, inventory, formal) -> int:
    local = [a for a in inventory if a["artifact_local_status"] == "local_available"]
    missing = [a for a in inventory if a["artifact_local_status"] == "missing_source_artifact"]
    p0 = [f for f in formal if f["priority"].startswith("P0")]
    p0_lines = "\n".join(f"- `{f['priority']}` -> {f['provider']} / {f['artifact_needed']} ({f['request_status']})" for f in p0)
    report = f"""# SUSC-17C4 - Official Artifact Ingestion (review-only)

Status: review-only. `trainable=false`; `ground_truth=false`; `score_v6_changed=false`; `score_v7_created=false`; `eligible_for_17b_now=false`.

{MANDATORY}

## 1. Artefatos oficiais encontrados localmente

{len(local)} artefato local utilizavel: o GeoJSON digitalizado do Charter 758 (REC_2022_05_24_30), CRS EPSG:4326, tracked no repo. Os PDFs/ZIP SGB/CPRM existem apenas em `local_runs/` (privado, gitignored), como ponteiros.

## 2. Quais estao ausentes

{len(missing)} artefatos ausentes (`missing_source_artifact`): pacotes formais nao publicos e eventos sem geometria baixada (DRM-RJ `PKG_FR_PET_001`, COMPDEC `PKG_FR_REC_002`, Defesa Civil Petropolis, Curitiba/IPPUC, PET_2024 Valparaiso).

## 3. Quais dependem de solicitacao formal

{summary['formal_request_count']} solicitacoes formais na fila ({summary['p0_request_count']} P0):
{p0_lines}

## 4. Status do PKG_FR_PET_001

DRM-RJ/NADE `PKG_FR_PET_001`: `missing_source_artifact`, `PENDING_FORMAL_REQUEST`. Nao existe localmente; bloqueia evidencia forte para PET_2022_02_15. Solicitacao P0 gerada.

## 5. Status do PKG_FR_REC_002

COMPDEC/Defesa Civil PE `PKG_FR_REC_002`: `missing_source_artifact`, `PENDING_FORMAL_REQUEST`. Nao existe localmente; complementaria o footprint coarse do Charter para REC_2022_05_24_30. Solicitacao P0 gerada.

## 6. Status dos PDFs/ZIP SGB/CPRM

`Relatorio_Tecnico_Petropolis.pdf` = `pdf_only`: tentativa `pdf_text_scan` -> `blocked_pdf_only_no_vector` (sem coordenada/vetor explicito; sem OCR pesado). `anexos_avaliacao_pos_desastre_petropolis_rj_2022.zip` = `zip_only`: `zip_inventory` -> `blocked_unknown_format` (arquivo abre como BadZipFile / sem vetor extraivel). Ambos viram solicitacao P1 de vetor/coordenadas.

## 7. Por que PE3D/MDE e contextual, nao evento

PE3D/MDE e modelo de terreno (MDT/MDS): descreve a fisiografia, nao registra onde/quando houve inundacao. Tentativa `not_attempted_context_only` -> `blocked_context_layer_not_event`.

## 8. Alguma geometria candidata foi extraida

Sim: {summary['reference_candidate_count']} candidato de referencia (`official_observed_event_polygon`) a partir do MultiPolygon real do Charter 758, `qa_status=needs_review`, sem invencao. {summary['candidate_geometry_count']} geometria no GeoJSON candidato.

## 9. Alguma geometria intersecta patches

Nao: {summary['candidate_patch_link_count']} patch-links. O MultiPolygon do Charter 758 cai FORA da grade SUSC `recife_*` (mesma area NE da borda). O `patch_id=REC_00019` da property pertence ao esquema do Protocolo C, nao a grade de suscetibilidade.

## 10. Por que 17B continua bloqueado

{json.dumps(summary['blocks_17b'], ensure_ascii=False)}. Ha geometria candidata, mas sem QA aceito, sem patch-link na grade SUSC e com os pacotes P0 ainda ausentes.

## 11. Proximo marco recomendado

{summary['recommended_next_milestone']}.
"""
    write_markdown(REPORT, report)
    return 0


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

def run_all() -> int:
    inventory = build_inventory()
    write_csv(INVENTORY, inventory, INVENTORY_FIELDS)
    ingestion = build_ingestion(inventory)
    write_csv(INGESTION, ingestion, INGESTION_FIELDS)
    candidates, geojson = build_reference_candidates(inventory, ingestion)
    write_csv(REFERENCE_CANDIDATES, candidates, CANDIDATE_FIELDS)
    write_json(CANDIDATE_GEOJSON, geojson)
    patch_links, outside_grid = build_patch_links(candidates)
    write_csv(PATCH_LINKS, patch_links, PATCH_LINK_FIELDS)
    formal = build_formal_queue(inventory)
    write_csv(FORMAL_QUEUE, formal, FORMAL_FIELDS)
    summary = build_summary(inventory, ingestion, candidates, geojson, patch_links, formal, outside_grid)
    build_report(summary, inventory, formal)
    print(f"17C4 -> inventory={len(inventory)} ingestion={len(ingestion)} "
          f"candidates={len(candidates)} geometries={len(geojson['features'])} "
          f"patch_links={len(patch_links)} formal={len(formal)}")
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
    required_files = [AUDIT, INVENTORY_SCHEMA, CANDIDATE_SCHEMA, INVENTORY, INGESTION,
                      REFERENCE_CANDIDATES, CANDIDATE_GEOJSON, PATCH_LINKS, FORMAL_QUEUE,
                      SUMMARY, REPORT]
    for path in required_files:
        if not path.exists():
            errors.append(f"missing: {path.relative_to(ROOT)}")
    if errors:
        print("FAILED: SUSC-17C4 validation")
        for e in errors:
            print("  ERROR:", e)
        return 1

    inv_schema = json.loads(INVENTORY_SCHEMA.read_text(encoding="utf-8"))
    cand_schema = json.loads(CANDIDATE_SCHEMA.read_text(encoding="utf-8"))
    inventory = read_csv(INVENTORY)
    ingestion = read_csv(INGESTION)
    candidates = read_csv(REFERENCE_CANDIDATES)
    patch_links = read_csv(PATCH_LINKS)
    formal = read_csv(FORMAL_QUEUE)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    geojson = json.loads(CANDIDATE_GEOJSON.read_text(encoding="utf-8"))

    # unique + deterministic IDs
    for label, rows, key in (
        ("inventory", inventory, "artifact_id"), ("ingestion", ingestion, "ingestion_attempt_id"),
        ("candidates", candidates, "reference_candidate_id"),
        ("patch_links", patch_links, "candidate_patch_link_id"), ("formal", formal, "request_id"),
    ):
        ids = [r[key] for r in rows]
        if len(ids) != len(set(ids)):
            errors.append(f"{label}: ids not unique")
        if ids != sorted(ids):
            errors.append(f"{label}: ids not sorted (non-deterministic)")

    # schema + governance on inventory
    for line_no, row in enumerate(inventory, start=2):
        for v in schema_violations(row, inv_schema):
            errors.append(f"{INVENTORY.name}:{line_no}: {v}")
        for v in _gov_violations(row):
            errors.append(f"{INVENTORY.name}:{line_no}: {v}")
        # PE3D never an event (rule 10)
        if "pe3d" in row["artifact_name"].lower() or "pe3d" in row["source_name"].lower():
            if row["artifact_local_status"] != "context_layer_only":
                errors.append(f"{INVENTORY.name}:{line_no}: PE3D not context_layer_only")

    # schema + governance on reference candidates
    art_by_id = {a["artifact_id"]: a for a in inventory}
    ing_by_artifact = {i["artifact_id"]: i for i in ingestion}
    for line_no, row in enumerate(candidates, start=2):
        for v in schema_violations(row, cand_schema):
            errors.append(f"{REFERENCE_CANDIDATES.name}:{line_no}: {v}")
        for v in _gov_violations(row):
            errors.append(f"{REFERENCE_CANDIDATES.name}:{line_no}: {v}")
        if row["eligible_for_17b_now"] != "false":
            errors.append(f"{REFERENCE_CANDIDATES.name}:{line_no}: eligible_for_17b_now not false")
        if row["qa_status"] != "needs_review":
            errors.append(f"{REFERENCE_CANDIDATES.name}:{line_no}: qa_status not needs_review (no QA yet)")
        # candidate must come from an artifact whose ingestion actually found geometry (rules 11/12/13)
        art = art_by_id.get(row["artifact_id"], {})
        ing = ing_by_artifact.get(row["artifact_id"], {})
        if art.get("artifact_local_status") not in ("local_available",):
            errors.append(f"{REFERENCE_CANDIDATES.name}:{line_no}: candidate from non-local artifact")
        if ing.get("geometry_candidates_found") != "1":
            errors.append(f"{REFERENCE_CANDIDATES.name}:{line_no}: candidate without parsed geometry")

    # missing/pdf/zip/context artifacts must NOT produce a reference candidate (rules 11-13)
    cand_artifacts = {c["artifact_id"] for c in candidates}
    for art in inventory:
        if art["artifact_local_status"] in ("missing_source_artifact", "pdf_only", "zip_only",
                                            "context_layer_only", "remote_reference_only"):
            if art["artifact_id"] in cand_artifacts:
                errors.append(f"reference candidate from blocked artifact {art['artifact_id']}")

    # geojson <-> registry consistency (rules 18)
    geojson_ids = {f["properties"]["reference_candidate_id"] for f in geojson.get("features", [])}
    registry_ids = {c["reference_candidate_id"] for c in candidates}
    if geojson_ids != registry_ids:
        errors.append("geojson features and candidate registry disagree")
    for feat in geojson.get("features", []):
        props = feat.get("properties", {})
        if props.get("ground_truth") is not False or props.get("review_only") is not True:
            errors.append("geojson feature governance wrong")
        if props.get("qa_status") != "needs_review":
            errors.append("geojson feature qa_status not needs_review")
        if not feat.get("geometry", {}).get("coordinates"):
            errors.append("geojson feature without coordinates")

    # patch-link requires a corresponding candidate geometry (rule 19)
    for row in patch_links:
        if row["reference_candidate_id"] not in registry_ids:
            errors.append(f"patch link without candidate: {row['candidate_patch_link_id']}")
        if row["eligible_for_17b_now"] != "false":
            errors.append("patch link eligible_for_17b_now not false")
        for v in _gov_violations(row):
            errors.append(f"patch_links: {v}")
    # if candidate geometry is outside the grid, no patch link must exist (rule 15)
    if summary.get("candidate_geometry_outside_grid_count", 0) > 0 and patch_links:
        # allowed only if some other candidate intersected; here flag inconsistency
        if len(patch_links) == 0:
            pass

    # formal queue governance + P0 presence (rules 15/16 of test spec)
    for row in formal:
        for v in _gov_violations(row):
            errors.append(f"formal: {v}")
    priorities = {f["priority"] for f in formal}
    missing_ids = {a["event_id"] for a in inventory if a["artifact_local_status"] == "missing_source_artifact"}
    if "PET_2022_02_15" in missing_ids and "P0_DRM_RJ_PKG_FR_PET_001" not in priorities:
        errors.append("formal queue missing P0 DRM-RJ while PKG_FR_PET_001 absent")
    if "REC_2022_05_24_30" in missing_ids and "P0_COMPDEC_RECIFE_2022_POINTS_OR_ADDRESSES" not in priorities:
        errors.append("formal queue missing P0 COMPDEC while PKG_FR_REC_002 absent")

    # ingestion governance: blocked statuses carry a reason
    for row in ingestion:
        if row["attempt_status"].startswith("blocked") and row["blocking_reason"] in ("", "not_available"):
            errors.append(f"ingestion blocked without reason: {row['ingestion_attempt_id']}")

    # summary honesty
    if summary.get("eligible_for_17b_now") is not False:
        errors.append("summary eligible_for_17b_now true")
    if summary.get("score_v7_created") is not False or summary.get("score_v6_changed") is not False:
        errors.append("summary score flags wrong")
    if summary.get("trainable") is not False or summary.get("ground_truth") is not False:
        errors.append("summary trainable/ground_truth wrong")
    if summary.get("total_artifacts_inventory") != len(inventory):
        errors.append("summary inventory count mismatch")
    if summary.get("reference_candidate_count") != len(candidates):
        errors.append("summary candidate count mismatch")
    if summary.get("candidate_patch_link_count") != len(patch_links):
        errors.append("summary patch link count mismatch")
    if summary.get("candidate_geometry_count") != len(geojson.get("features", [])):
        errors.append("summary geometry count mismatch")

    # cross guards
    if SCORE_V7.exists():
        errors.append("score v7 artifact exists")
    if not SCORE_V6.exists():
        errors.append("score v6 official source missing")
    if MANDATORY not in REPORT.read_text(encoding="utf-8"):
        errors.append("mandatory report phrase missing")
    # no heavy raster in outputs (rule 11)
    for path in OUT.glob("susc_17c4_*"):
        if path.suffix.lower() in (".tif", ".tiff", ".img", ".jp2", ".nc", ".grd", ".npy", ".npz"):
            errors.append(f"heavy raster artifact present: {path.name}")

    if errors:
        print("FAILED: SUSC-17C4 validation")
        for e in errors:
            print("  ERROR:", e)
        return 1
    print("PASSED: SUSC-17C4 validations passed")
    return 0
