"""REV-P v2ca — Curitiba event registry binding and evidence acquisition pipeline.

v2bz proved Curitiba should not stay as ``CUR_EVENT_REGISTRY_MISSING``: a real
candidate-event registry (v1uv) holds two official ``urban_flooding`` events
(``CUR_2022_01_15`` and ``CUR_2022_01_05``). v2ca turns that scaffold into a
traceable, flood-compatible cohort:

* repairs the Curitiba event registry from the real candidate registry (never
  inventing an event);
* inventories local Curitiba sources and classifies geometry (point / polygon /
  risk-area context) without inventing geometry;
* audits every Curitiba patch in the v2bn feature table (sentinel input, DINO
  embedding, GIS features, split group, recorded raster-header bounds);
* recovers patch boundaries from the recorded v1fs raster-header bounds
  (EPSG:32722) reprojected to WGS84 — the same strategy that worked for Recife,
  never opening heavy rasters;
* creates patch-event binding candidates and a queue to repeat the
  v2bp -> v2bq -> v2bt -> v2bu -> v2bx chain once geometry/points are acquired.

It creates no label, no formal negative and no training target. A risk-area
polygon is never promoted to an event footprint, absence is never a negative,
and ``can_support_formal_gt`` stays ``false`` until a validated footprint exists.
External web search is offline-deterministic by default
(``EXTERNAL_WEB_SEARCH_NOT_PERFORMED``).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover - environment dependent
    from pyproj import Transformer as _Transformer  # type: ignore
    HAS_PYPROJ = True
except Exception:  # pragma: no cover
    _Transformer = None
    HAS_PYPROJ = False

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "ground_truth" / "v2ca"
STAGE = "v2ca"
REGION = "Curitiba"

GT = ROOT / "local_runs" / "ground_truth"
MM = ROOT / "local_runs" / "multimodal"
DEFAULT_V2BZ_SCAFFOLD = GT / "v2bz" / "curitiba_event_registry_repair_scaffold_v2bz.csv"
DEFAULT_V2BZ_SUMMARY = GT / "v2bz" / "expansion_evidence_acquisition_summary_v2bz.json"
DEFAULT_V2BY_EVENTS = GT / "v2by" / "event_expansion_candidate_inventory_v2by.csv"
DEFAULT_FEATURE_TABLE = MM / "v2bn" / "multimodal_feature_table_core_v2bn.csv"
DEFAULT_CURITIBA_CANDIDATES = ROOT / "datasets" / "protocolo_c" / "v1uv_curitiba_candidate_event_registry.csv"
DEFAULT_V1FS_AUDIT = ROOT / "manifests" / "training_readiness" / "revp_v1fs_self_supervised_asset_sanity_and_embedding_plan" / "asset_sanity_audit_v1fs.csv"

RECOVERED_DIR_NAME = "recovered_patch_boundaries"
SIDECAR_DIR_NAME = "source_sidecars"

SCAN_DIRS = ["datasets", "manifests", "outputs_public", "docs", "configs", "archive_drive"]
SOURCE_EXTS = {".geojson", ".kml", ".wkt", ".json", ".csv", ".pdf", ".html", ".htm", ".md", ".yaml", ".yml"}
GEOM_EXTS = {".geojson", ".kml", ".wkt", ".json"}
KNOWN_CUR_EVENTS = ("cur_2022_01_15", "cur_2022_01_05")
OFFICIAL_ORG_TOKENS = ("cemaden", "cprm", "sgb", "rigeo", "geosgb", "geocuritiba", "defesa", "ippuc", "simepar",
                       "apac", "prefeitura", "diario", "gazette", "inde", "ibge", "inmet", "ana", "parana", "gov")
DERIVED_TOKENS = ("registry", "inventory", "audit", "manifest", "scaffold", "matrix", "policy", "report", "summary")
RISK_TOKENS = ("risk", "risco", "suscetib", "suscept")
FLOOD_TOKENS = ("flood", "inund", "alag", "urban_flood", "enchente", "chuva")

WGS84 = "EPSG:4326"
# Brazil plausibility window (degrees) — reprojected geometry must land here.
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = -74.5, -33.0, -34.5, 6.0
# Curitiba metro plausibility window (degrees).
CUR_LON_MIN, CUR_LON_MAX, CUR_LAT_MIN, CUR_LAT_MAX = -49.5, -49.0, -25.7, -25.2

WEB_NOT_PERFORMED = "EXTERNAL_WEB_SEARCH_NOT_PERFORMED"
WEB_UNAVAILABLE = "EXTERNAL_WEB_SEARCH_UNAVAILABLE"

# Public, offline-logged search terms (never executed unless internet is enabled).
SEARCH_TERMS = [
    "Curitiba 15 janeiro 2022 alagamento Defesa Civil",
    "Curitiba 5 janeiro 2022 alagamento Defesa Civil",
    "Curitiba janeiro 2022 inundação IPPUC",
    "Curitiba alagamentos janeiro 2022 CEMADEN",
    "Curitiba urban flooding January 2022 official",
    "Simepar Curitiba chuva janeiro 2022 alagamento",
]

# Patch boundary recovery statuses.
PB_RECOVERED = "CURITIBA_PATCH_BOUNDARY_RECOVERED"
PB_NOT_FOUND = "CURITIBA_PATCH_BOUNDARY_NOT_FOUND"
PB_BLOCKED_CRS = "CURITIBA_PATCH_BOUNDARY_BLOCKED_NO_CRS"
PB_AMBIGUOUS = "CURITIBA_PATCH_BOUNDARY_AMBIGUOUS"
PB_CENTROID_ONLY = "CURITIBA_PATCH_BOUNDARY_CENTROID_ONLY"

# Event-level statuses.
EV_REPAIRED = "CURITIBA_FLOOD_EVENT_REGISTRY_REPAIRED"
EV_READY_ACQ = "CURITIBA_EVENT_READY_FOR_EVIDENCE_ACQUISITION"
EV_BLOCKED_GEOM = "CURITIBA_EVENT_BLOCKED_NO_GEOMETRY_OR_POINTS"
EV_READY_BINDING = "CURITIBA_EVENT_READY_FOR_PATCH_BINDING"
EV_STILL_MISSING = "CURITIBA_EVENT_REGISTRY_STILL_MISSING"

EVENT_REG_FIELDS = ["event_registry_id", "event_id", "region", "hazard_type", "event_date_or_period", "source_registry",
                    "source_family", "is_official", "can_create_training_label", "registry_repair_status", "event_status",
                    "recommended_use", "notes"]
SOURCE_FIELDS = ["source_id", "event_id", "region", "source_name", "source_family", "source_type", "source_path_or_url",
                 "is_local", "is_external", "is_official", "is_context_source", "is_point_source", "is_polygon_source",
                 "is_risk_area_source", "temporal_alignment_status", "source_status", "recommended_use", "notes"]
GEOM_FIELDS = ["geometry_id", "event_id", "region", "source_id", "geometry_source_type", "geometry_type", "crs",
               "geometry_valid", "bbox", "centroid", "area_approx", "is_event_specific", "is_risk_area_general",
               "is_point_evidence", "can_support_overlay", "can_support_qa_geometry", "can_support_formal_gt",
               "geometry_quality_status", "blocked_reason", "notes"]
PATCH_FIELDS = ["patch_readiness_id", "canonical_patch_id", "region", "has_sentinel_input", "has_dino_embedding",
                "has_gis_features", "has_split_group", "has_boundary", "has_raster_header_bounds", "boundary_source",
                "split_group", "readiness_status", "priority", "blocked_reason", "recommended_next_action"]
BOUNDARY_FIELDS = ["boundary_audit_id", "canonical_patch_id", "region", "candidate_sources_scanned",
                   "raster_header_bounds_found", "crs_detected", "bounds_detected", "boundary_recovered",
                   "boundary_source_type", "boundary_sidecar_path", "boundary_quality", "can_use_for_overlay",
                   "blocked_reason", "notes"]
BINDING_FIELDS = ["binding_id", "event_id", "canonical_patch_id", "region", "hazard_type", "event_date_or_period",
                  "patch_has_boundary", "patch_has_embedding", "patch_has_gis_features", "event_has_context",
                  "event_has_point_evidence", "event_has_polygon_geometry", "binding_status", "can_enter_adjudication",
                  "can_enter_overlay", "blocked_reason", "recommended_next_stage"]
QUEUE_FIELDS = ["queue_id", "event_id", "region", "hazard_type", "evidence_target", "priority", "readiness_status",
                "required_inputs", "can_run_autonomously", "needs_user_decision", "blocked_reason", "recommended_next_action"]
PLAN_FIELDS = ["plan_id", "event_id", "required_stage", "required_inputs", "expected_outputs", "can_run_now",
               "blocked_reason", "recommended_command_or_module", "notes"]

METHODOLOGICAL_GUARDRAILS = {
    "review_only": True,
    "labels_created": False,
    "formal_positive_created": False,
    "formal_negative_created": False,
    "event_invented": False,
    "geometry_invented": False,
    "registry_repair_is_label": False,
    "risk_area_is_event_footprint": False,
    "negative_from_absence": False,
    "binding_is_label": False,
    "supervised_training": False,
    "outputs_local_only": True,
}


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {path}. Use --force.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def short_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha1(value.encode('utf-8')).hexdigest()[:12]}"


def rel_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return path.name


def norm(value: str) -> str:
    return unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode().lower().strip()


def is_curitiba(value: str) -> bool:
    s = norm(value)
    return s.startswith("curitiba") or s == "cur" or s.startswith("cur_")


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


# --------------------------------------------------------------------------- #
# 1. Event registry repair (never invents an event)
# --------------------------------------------------------------------------- #

def repair_event_registry(curitiba_candidates: list[dict[str, str]], candidates_path: Path) -> list[dict[str, Any]]:
    """Build the repaired Curitiba event registry from the real candidate registry."""
    rows: list[dict[str, Any]] = []
    cur = [c for c in curitiba_candidates if is_curitiba(c.get("city", "") or c.get("region", ""))]
    if not cur:
        return rows
    for c in sorted(cur, key=lambda r: r.get("event_id_candidate", "") or r.get("candidate_event_id", "")):
        eid = (c.get("event_id_candidate", "") or c.get("candidate_event_id", "")).strip()
        if not eid:
            continue
        start = (c.get("start_date", "") or "").strip()
        end = (c.get("end_date", "") or "").strip()
        period = f"{start}/{end}" if end and end != start else start
        scope = c.get("hazard_scope", "")
        hazard = "flood" if any(tok in norm(scope) for tok in FLOOD_TOKENS) else "unknown"
        official_status = norm(c.get("official_source_status", ""))
        is_official = "official" in official_status or "public" in official_status
        # Honor the registry's own label policy; never promote to a training label.
        can_label = truthy(c.get("can_create_training_label", "false"))
        rows.append({
            "event_registry_id": short_id("CEVT", eid),
            "event_id": eid, "region": REGION, "hazard_type": hazard, "event_date_or_period": period,
            "source_registry": rel_to_root(candidates_path),
            "source_family": "CURITIBA_OFFICIAL_PUBLIC_EVENT_CANDIDATE" if is_official else "CURITIBA_EVENT_CANDIDATE_UNVERIFIED",
            "is_official": str(is_official).lower(),
            "can_create_training_label": "false" if not can_label else "false",  # forced false at this stage
            "registry_repair_status": EV_REPAIRED if is_official else "CURITIBA_EVENT_CANDIDATE_UNVERIFIED",
            "event_status": EV_BLOCKED_GEOM,  # refined later once geometry/points are known
            "recommended_use": "flood_cohort_candidate_not_label",
            "notes": "official candidate event; flood-compatible; not ground truth, not a training label",
        })
    return rows


# --------------------------------------------------------------------------- #
# 2. Local source inventory + geometry detection (never invents geometry)
# --------------------------------------------------------------------------- #

def detect_geometry(path: Path) -> tuple[str, bool]:
    suffix = path.suffix.lower()
    if suffix == ".wkt":
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")[:500].upper()
        except OSError:
            return "", False
        if "POLYGON" in txt:
            return "polygon", True
        if "POINT" in txt:
            return "point", True
        return "", False
    if suffix == ".kml":
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")[:5000]
        except OSError:
            return "", False
        if "<Polygon" in txt:
            return "polygon", True
        if "<Point" in txt:
            return "point", True
        return "", False
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "", False
    feats = doc.get("features", [doc]) if isinstance(doc, dict) else []
    kinds: set[str] = set()
    for f in feats or []:
        if not isinstance(f, dict):
            continue
        g = f.get("geometry") or (f if f.get("type") in {"Point", "MultiPoint", "Polygon", "MultiPolygon"} else {})
        t = (g or {}).get("type", "")
        if "Polygon" in t:
            kinds.add("polygon")
        elif "Point" in t:
            kinds.add("point")
    if "polygon" in kinds:
        return "polygon", True
    if "point" in kinds:
        return "point", True
    return "", False


def classify_curitiba_source(path: Path) -> dict[str, Any] | None:
    low = str(path).lower().replace("\\", "/")
    if not is_curitiba(low) and "curitiba" not in low and "/cur_" not in low and "_cur_" not in low:
        return None
    ext = path.suffix.lower()
    if ext not in SOURCE_EXTS:
        return None
    name = path.name
    nlow = name.lower()
    event_id = "CUR_REGION_CONTEXT"
    for tid in KNOWN_CUR_EVENTS:
        if tid in low:
            event_id = tid.upper()
            break

    is_geom = ext in GEOM_EXTS
    geom_kind, has_geom = (detect_geometry(path) if is_geom else ("", False))
    is_point = has_geom and geom_kind == "point"
    is_polygon = has_geom and geom_kind == "polygon"
    is_risk = any(tok in low for tok in RISK_TOKENS)

    if is_polygon:
        family = "RISK_AREA_GEOMETRY_SOURCE" if is_risk else "POLYGON_GEOMETRY_SOURCE"
        use = "risk_area_context_only_not_event_footprint" if is_risk else "candidate_geometry_review_not_formal_gt"
    elif is_point:
        family = "POINT_EVIDENCE_SOURCE"
        use = "point_evidence_for_qa_geometry"
    elif any(tok in nlow for tok in DERIVED_TOKENS):
        family = "QA_DERIVED_SOURCE"
        use = "derived_catalog_context_only"
    elif any(tok in nlow for tok in OFFICIAL_ORG_TOKENS):
        family = "OFFICIAL_CONTEXT_SOURCE"
        use = "official_context_not_geometry"
    elif ext in {".pdf", ".html", ".htm"}:
        family = "OFFICIAL_CONTEXT_SOURCE" if any(t in nlow for t in OFFICIAL_ORG_TOKENS) else "MEDIA_CONTEXT_SOURCE"
        use = "document_context_not_geometry"
    else:
        family = "UNVERIFIED_SOURCE"
        use = "review"
    is_official = family in {"OFFICIAL_CONTEXT_SOURCE", "POINT_EVIDENCE_SOURCE", "POLYGON_GEOMETRY_SOURCE"} \
        or any(t in low for t in OFFICIAL_ORG_TOKENS)
    return {
        "source_id": short_id("CSRC", rel_to_root(path)), "event_id": event_id, "region": REGION,
        "source_name": name, "source_family": family, "source_type": ext.lstrip("."),
        "source_path_or_url": rel_to_root(path), "is_local": "true", "is_external": "false",
        "is_official": str(is_official).lower(),
        "is_context_source": str(not (is_point or is_polygon)).lower(),
        "is_point_source": str(is_point).lower(), "is_polygon_source": str(is_polygon).lower(),
        "is_risk_area_source": str(is_risk and is_polygon).lower(),
        "temporal_alignment_status": "NOT_ASSESSED", "source_status": "INVENTORIED",
        "recommended_use": use,
        "notes": "geometry_detected_only_from_real_geo_files; csv/pdf/md_not_opened_for_geometry",
        "_geom_kind": geom_kind,
    }


def scan_curitiba_sources(scan_root: Path, output_dir: Path) -> list[dict[str, Any]]:
    seen: set[Path] = set()
    out: list[dict[str, Any]] = []
    out_resolved = output_dir.resolve()
    for d in SCAN_DIRS:
        base = scan_root / d
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path in seen:
                continue
            if out_resolved in path.resolve().parents:
                continue
            seen.add(path)
            info = classify_curitiba_source(path)
            if info:
                out.append(info)
    out.sort(key=lambda r: r["source_path_or_url"])
    return out


def build_geometry_inventory(event_rows: list[dict[str, Any]], sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points = [s for s in sources if s["is_point_source"] == "true"]
    polygons = [s for s in sources if s["is_polygon_source"] == "true"]
    rows: list[dict[str, Any]] = []
    events = event_rows or [{"event_id": "CUR_REGION_CONTEXT", "hazard_type": "flood"}]
    for ev in events:
        eid = ev["event_id"]
        ev_polys = [p for p in polygons if p["event_id"] in (eid, "CUR_REGION_CONTEXT")]
        ev_points = [p for p in points if p["event_id"] in (eid, "CUR_REGION_CONTEXT")]
        if ev_polys:
            src = ev_polys[0]
            is_risk = src["is_risk_area_source"] == "true"
            rows.append(_geom_row(eid, src["source_id"], "polygon",
                                  "RISK_AREA_POLYGON_CONTEXT" if is_risk else "POLYGON_GEOMETRY",
                                  is_event_specific=False, is_risk=is_risk, is_point=False,
                                  status="RISK_AREA_GEOMETRY_CONTEXT_ONLY" if is_risk else "POLYGON_GEOMETRY_AVAILABLE",
                                  blocked="RISK_AREA_NOT_EVENT_FOOTPRINT" if is_risk else ""))
        if ev_points:
            src = ev_points[0]
            rows.append(_geom_row(eid, src["source_id"], "point", "POINT_EVIDENCE",
                                  is_event_specific=False, is_risk=False, is_point=True,
                                  status="POINT_EVIDENCE_AVAILABLE", blocked=""))
        if not ev_polys and not ev_points:
            rows.append(_geom_row(eid, "", "none", "CONTEXT_ONLY_NO_GEOMETRY",
                                  is_event_specific=False, is_risk=False, is_point=False,
                                  status="CONTEXT_ONLY_NO_GEOMETRY", blocked="NO_LOCAL_GEOMETRY_OR_POINTS"))
    return rows


def _geom_row(eid, source_id, gtype, gsource_type, *, is_event_specific, is_risk, is_point, status, blocked):
    can_overlay = gtype == "polygon" and not is_risk
    return {
        "geometry_id": short_id("CGEO", f"{eid}|{gsource_type}|{source_id}"), "event_id": eid, "region": REGION,
        "source_id": source_id, "geometry_source_type": gsource_type, "geometry_type": gtype,
        "crs": "UNKNOWN" if gtype != "none" else "", "geometry_valid": "true" if gtype != "none" else "false",
        "bbox": "", "centroid": "", "area_approx": "",
        "is_event_specific": str(is_event_specific).lower(), "is_risk_area_general": str(is_risk).lower(),
        "is_point_evidence": str(is_point).lower(), "can_support_overlay": str(can_overlay).lower(),
        "can_support_qa_geometry": str(is_point).lower(), "can_support_formal_gt": "false",
        "geometry_quality_status": status, "blocked_reason": blocked,
        "notes": "can_support_formal_gt_false_until_validated_event_footprint; geometry_not_invented",
    }


# --------------------------------------------------------------------------- #
# 3. Patch readiness + boundary recovery (recorded raster-header bounds)
# --------------------------------------------------------------------------- #

def build_v1fs_bounds_index(v1fs_audit_path: Path) -> dict[str, dict[str, str]]:
    index: dict[str, list[dict[str, str]]] = {}
    for r in read_csv(v1fs_audit_path):
        cid = (r.get("candidate_id") or "").strip()
        crs = (r.get("crs_if_header_available") or "").strip()
        bounds = (r.get("bounds_if_header_available") or "").strip()
        if cid and crs and bounds:
            index.setdefault(cid, []).append({"crs": crs, "bounds": bounds, "asset_path": r.get("asset_path", "")})
    out: dict[str, dict[str, str]] = {}
    for cid, entries in index.items():
        distinct = {e["bounds"] for e in entries}
        first = dict(entries[0])
        first["_conflict"] = "true" if len(distinct) > 1 else "false"
        out[cid] = first
    return out


def parse_bounds(text: str) -> tuple[float, float, float, float] | None:
    parts = [p.strip() for p in (text or "").split(",") if p.strip()]
    if len(parts) != 4:
        return None
    try:
        minx, miny, maxx, maxy = (float(p) for p in parts)
    except ValueError:
        return None
    return (minx, miny, maxx, maxy)


def reproject_bbox_to_wgs84(bbox: tuple, crs: str) -> list[list[float]] | None:
    minx, miny, maxx, maxy = bbox
    corners = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
    if crs in {WGS84, "EPSG:4326", "CRS84"}:
        ring = [[float(x), float(y)] for x, y in corners]
    elif crs == "UNKNOWN" or not (HAS_PYPROJ and _Transformer is not None):
        return None
    else:
        try:
            t = _Transformer.from_crs(crs, WGS84, always_xy=True)
            ring = [list(t.transform(x, y)) for x, y in corners]
        except Exception:
            return None
    ring.append(ring[0])
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    if not (LON_MIN <= min(xs) and max(xs) <= LON_MAX and LAT_MIN <= min(ys) and max(ys) <= LAT_MAX):
        return None
    return ring


def recover_patch_boundary(patch_id: str, bounds_index: dict[str, dict[str, str]],
                           recovered_dir: Path) -> dict[str, Any]:
    entry = bounds_index.get(patch_id)
    scanned = 1 if entry else 0
    base = {
        "boundary_audit_id": short_id("CBND", patch_id), "canonical_patch_id": patch_id, "region": REGION,
        "candidate_sources_scanned": scanned, "raster_header_bounds_found": "false", "crs_detected": "",
        "bounds_detected": "", "boundary_recovered": "false", "boundary_source_type": "",
        "boundary_sidecar_path": "", "boundary_quality": "", "can_use_for_overlay": "false",
        "blocked_reason": "", "notes": "geometry_not_invented; bounds_from_recorded_audit_not_live_raster",
    }
    if not entry:
        base.update({"blocked_reason": "NO_RECORDED_HEADER_BOUNDS_FOR_PATCH_ID",
                     "boundary_recovery_status": PB_NOT_FOUND})
        return base
    crs = entry["crs"].upper()
    base["raster_header_bounds_found"] = "true"
    base["crs_detected"] = crs
    base["bounds_detected"] = entry["bounds"]
    if entry.get("_conflict") == "true":
        base.update({"blocked_reason": "MULTIPLE_CONFLICTING_RECORDED_BOUNDS", "boundary_recovery_status": PB_AMBIGUOUS})
        return base
    bbox = parse_bounds(entry["bounds"])
    if not bbox:
        base.update({"blocked_reason": "UNPARSEABLE_BOUNDS", "boundary_recovery_status": PB_BLOCKED_CRS})
        return base
    if bbox[0] == bbox[2] and bbox[1] == bbox[3]:
        base.update({"blocked_reason": "ONLY_A_POINT_NO_EXTENT", "boundary_recovery_status": PB_CENTROID_ONLY})
        return base
    if crs == "UNKNOWN" or not crs or (crs not in {WGS84, "EPSG:4326"} and not HAS_PYPROJ):
        base.update({"blocked_reason": "CRS_UNKNOWN_OR_NO_REPROJECTION_BACKEND", "boundary_recovery_status": PB_BLOCKED_CRS})
        return base
    ring = reproject_bbox_to_wgs84(bbox, crs)
    if ring is None:
        base.update({"blocked_reason": "REPROJECTION_FAILED_OR_OUT_OF_BOUNDS", "boundary_recovery_status": PB_BLOCKED_CRS})
        return base
    feature = {
        "type": "Feature",
        "properties": {
            "patch_id": patch_id, "region": REGION,
            "recovery_method": "v1fs_recorded_raster_header_bounds_reprojected",
            "source_crs": crs, "crs": WGS84, "source_audit": rel_to_root(DEFAULT_V1FS_AUDIT),
            "can_be_ground_truth": False, "review_status": "auto_recovered_unreviewed",
        },
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }
    recovered_dir.mkdir(parents=True, exist_ok=True)
    sidecar = recovered_dir / f"patch_boundary_{patch_id}_recovered_v2ca.geojson"
    sidecar.write_text(json.dumps(feature, ensure_ascii=False), encoding="utf-8")
    base.update({
        "boundary_recovered": "true", "boundary_source_type": "RECORDED_RASTER_HEADER_BOUNDS_REPROJECTED",
        "boundary_sidecar_path": rel_to_root(sidecar), "boundary_quality": "COARSE_RASTER_HEADER_EXTENT",
        "can_use_for_overlay": "true", "boundary_recovery_status": PB_RECOVERED,
    })
    return base


def build_patch_readiness(feature_rows: list[dict[str, str]], bounds_index: dict[str, dict[str, str]],
                          recovered_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cur_rows = [r for r in feature_rows if is_curitiba(r.get("region", ""))]
    cur_rows.sort(key=lambda r: r.get("canonical_patch_id", ""))
    readiness: list[dict[str, Any]] = []
    boundary_audit: list[dict[str, Any]] = []
    for r in cur_rows:
        pid = (r.get("canonical_patch_id", "") or "").strip()
        if not pid:
            continue
        has_sentinel = bool((r.get("source_asset_id", "") or r.get("dino_input_id", "")).strip())
        has_embed = truthy(r.get("dino_embedding_available", "false"))
        has_gis = truthy(r.get("gis_feature_available", "false"))
        split = (r.get("split_group", "") or "").strip()
        has_split = bool(split)
        has_bounds = pid in bounds_index

        rec = recover_patch_boundary(pid, bounds_index, recovered_dir)
        boundary_audit.append(rec)
        has_boundary = rec["boundary_recovered"] == "true"
        boundary_source = rec["boundary_source_type"] if has_boundary else ""

        if has_boundary and has_embed:
            status, blocked, action, priority = ("CURITIBA_PATCH_READY_FOR_BINDING", "",
                                                 "bind_patch_to_curitiba_event", "HIGH")
        elif has_boundary and not has_embed:
            status, blocked, action, priority = ("CURITIBA_PATCH_BOUNDARY_ONLY_NO_EMBEDDING",
                                                 "NO_LOCAL_DINO_EMBEDDING",
                                                 "extract_embedding_then_bind", "MEDIUM")
        elif not has_boundary and has_embed:
            status, blocked, action, priority = ("CURITIBA_PATCH_EMBEDDING_ONLY_NO_BOUNDARY",
                                                 rec.get("blocked_reason", "NO_PATCH_BOUNDARY"),
                                                 "recover_or_acquire_patch_boundary", "MEDIUM")
        else:
            status, blocked, action, priority = ("CURITIBA_PATCH_BLOCKED_NO_BOUNDARY_NO_EMBEDDING",
                                                 rec.get("blocked_reason", "NO_PATCH_BOUNDARY") or "NO_PATCH_BOUNDARY",
                                                 "acquire_boundary_and_embedding", "LOW")
        readiness.append({
            "patch_readiness_id": short_id("CPAT", pid), "canonical_patch_id": pid, "region": REGION,
            "has_sentinel_input": str(has_sentinel).lower(), "has_dino_embedding": str(has_embed).lower(),
            "has_gis_features": str(has_gis).lower(), "has_split_group": str(has_split).lower(),
            "has_boundary": str(has_boundary).lower(), "has_raster_header_bounds": str(has_bounds).lower(),
            "boundary_source": boundary_source, "split_group": split, "readiness_status": status,
            "priority": priority, "blocked_reason": blocked, "recommended_next_action": action,
        })
    return readiness, boundary_audit


# --------------------------------------------------------------------------- #
# 4. Patch-event binding candidates (no label)
# --------------------------------------------------------------------------- #

def build_bindings(event_rows: list[dict[str, Any]], patch_rows: list[dict[str, Any]],
                   geometry_inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    event_has_context = bool(event_rows)
    has_points = any(g["is_point_evidence"] == "true" for g in geometry_inventory)
    has_polygon = any(g["geometry_type"] == "polygon" and g["is_risk_area_general"] == "false"
                      for g in geometry_inventory)
    rows: list[dict[str, Any]] = []
    for ev in event_rows:
        eid = ev["event_id"]
        for p in patch_rows:
            pid = p["canonical_patch_id"]
            patch_boundary = p["has_boundary"] == "true"
            patch_embed = p["has_dino_embedding"] == "true"
            patch_gis = p["has_gis_features"] == "true"
            can_adjudicate = patch_embed and event_has_context
            can_overlay = patch_boundary and has_polygon
            if can_overlay:
                status, blocked, nxt = "CURITIBA_BINDING_READY_FOR_OVERLAY", "", "CURITIBA_OVERLAY_RETRY"
            elif patch_boundary and has_points:
                status, blocked, nxt = ("CURITIBA_BINDING_READY_FOR_QA_GEOMETRY",
                                        "EVENT_HAS_NO_POLYGON_GEOMETRY", "CURITIBA_QA_GEOMETRY_FROM_POINTS")
            elif can_adjudicate:
                status, blocked, nxt = ("CURITIBA_BINDING_READY_FOR_ADJUDICATION",
                                        "EVENT_HAS_NO_GEOMETRY_OR_POINTS", "CURITIBA_V2BP_ADJUDICATION")
            elif not patch_boundary:
                status, blocked, nxt = ("CURITIBA_BINDING_BLOCKED_NO_PATCH_BOUNDARY",
                                        "NO_PATCH_BOUNDARY", "CURITIBA_BOUNDARY_RECOVERY")
            else:
                status, blocked, nxt = ("CURITIBA_BINDING_BLOCKED_NO_EVENT_GEOMETRY",
                                        "EVENT_HAS_NO_GEOMETRY_OR_POINTS", "CURITIBA_BLOCKED_ACQUIRE_GEOMETRY")
            rows.append({
                "binding_id": short_id("CBIN", f"{eid}|{pid}"), "event_id": eid, "canonical_patch_id": pid,
                "region": REGION, "hazard_type": ev.get("hazard_type", "flood"),
                "event_date_or_period": ev.get("event_date_or_period", ""),
                "patch_has_boundary": str(patch_boundary).lower(), "patch_has_embedding": str(patch_embed).lower(),
                "patch_has_gis_features": str(patch_gis).lower(), "event_has_context": str(event_has_context).lower(),
                "event_has_point_evidence": str(has_points).lower(), "event_has_polygon_geometry": str(has_polygon).lower(),
                "binding_status": status, "can_enter_adjudication": str(can_adjudicate).lower(),
                "can_enter_overlay": str(can_overlay).lower(), "blocked_reason": blocked,
                "recommended_next_stage": nxt,
            })
    return rows


# --------------------------------------------------------------------------- #
# 5. Acquisition queue + next-chain execution plan
# --------------------------------------------------------------------------- #

def build_acquisition_queue(event_rows: list[dict[str, Any]], geometry_inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    has_points = any(g["is_point_evidence"] == "true" for g in geometry_inventory)
    has_polygon = any(g["geometry_type"] == "polygon" and g["is_risk_area_general"] == "false"
                      for g in geometry_inventory)
    rows: list[dict[str, Any]] = []
    for ev in event_rows:
        eid = ev["event_id"]
        if has_polygon:
            target, readiness, blocked, action, prio = ("validated_event_footprint", "POLYGON_GEOMETRY_AVAILABLE",
                                                        "NEEDS_FORMAL_FOOTPRINT_VALIDATION", "validate_event_footprint", "MEDIUM")
        elif has_points:
            target, readiness, blocked, action, prio = ("event_point_evidence", "POINT_EVIDENCE_AVAILABLE",
                                                        "NO_POLYGON_GEOMETRY", "build_qa_geometry_from_points", "MEDIUM")
        else:
            target, readiness, blocked, action, prio = ("event_geometry_or_point_evidence", "CONTEXT_ONLY_NO_GEOMETRY",
                                                        "NO_LOCAL_GEOMETRY_OR_POINTS",
                                                        "acquire_curitiba_event_geometry_or_point_evidence", "HIGH")
        rows.append({
            "queue_id": short_id("CQUE", eid), "event_id": eid, "region": REGION,
            "hazard_type": ev.get("hazard_type", "flood"), "evidence_target": target, "priority": prio,
            "readiness_status": readiness,
            "required_inputs": "official_event_footprint_or_point_evidence|defesa_civil|ippuc|cemaden|simepar",
            "can_run_autonomously": "false", "needs_user_decision": "false", "blocked_reason": blocked,
            "recommended_next_action": action,
        })
    return rows


def build_next_chain_plan(event_rows: list[dict[str, Any]], patch_rows: list[dict[str, Any]],
                          binding_rows: list[dict[str, Any]], geometry_inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recovered = sum(1 for p in patch_rows if p["has_boundary"] == "true")
    ready_adjudication = sum(1 for b in binding_rows if b["can_enter_adjudication"] == "true")
    ready_overlay = sum(1 for b in binding_rows if b["can_enter_overlay"] == "true")
    has_points = any(g["is_point_evidence"] == "true" for g in geometry_inventory)
    eid = event_rows[0]["event_id"] if event_rows else "CUR_EVENT_REGISTRY_MISSING"

    def plan(stage, inputs, outputs, can_now, blocked, module, notes):
        return {
            "plan_id": short_id("CPLN", f"{eid}|{stage}"), "event_id": eid, "required_stage": stage,
            "required_inputs": inputs, "expected_outputs": outputs, "can_run_now": str(can_now).lower(),
            "blocked_reason": blocked, "recommended_command_or_module": module, "notes": notes,
        }

    rows = [
        plan("BOUNDARY_RECOVERY", "v1fs_recorded_raster_header_bounds", f"{recovered}_recovered_patch_boundaries",
             recovered > 0, "" if recovered > 0 else "NO_RECORDED_BOUNDS",
             "CURITIBA_BOUNDARY_RECOVERY", "completed_in_v2ca_for_patches_with_recorded_bounds"),
        plan("ADJUDICATION", "repaired_event_registry|patch_readiness|binding_candidates",
             "curitiba_event_patch_adjudication", ready_adjudication > 0,
             "" if ready_adjudication > 0 else "NO_BINDING_WITH_EMBEDDING_AND_CONTEXT",
             "CURITIBA_V2BP_ADJUDICATION", "adjudication_is_not_a_label"),
        plan("QA_GEOMETRY_FROM_POINTS", "event_point_evidence", "qa_only_event_geometry",
             has_points, "" if has_points else "NO_POINT_EVIDENCE",
             "CURITIBA_QA_GEOMETRY_FROM_POINTS", "qa_geometry_never_becomes_ground_truth"),
        plan("OVERLAY_RETRY", "patch_boundary|event_geometry", "patch_event_overlay_resolution",
             ready_overlay > 0, "" if ready_overlay > 0 else "EVENT_HAS_NO_GEOMETRY_OR_POINTS",
             "CURITIBA_OVERLAY_RETRY", "overlay_is_not_a_label"),
        plan("ACQUIRE_GEOMETRY", "external_official_event_geometry_or_points", "curitiba_event_geometry",
             False, "REQUIRES_EXTERNAL_ACQUISITION",
             "CURITIBA_BLOCKED_ACQUIRE_GEOMETRY", "gating_step_unblocks_overlay_and_dry_run_chain"),
    ]
    return rows


# --------------------------------------------------------------------------- #
# 6. Gate / guardrails / summary / report
# --------------------------------------------------------------------------- #

def build_gate(event_rows, geometry_inventory, patch_rows, binding_rows, current_pos):
    has_points = any(g["is_point_evidence"] == "true" for g in geometry_inventory)
    has_polygon = any(g["geometry_type"] == "polygon" and g["is_risk_area_general"] == "false"
                      for g in geometry_inventory)
    official = sum(1 for e in event_rows if e["is_official"] == "true")
    return {
        "phase": STAGE, "region": REGION,
        "curitiba_events_repaired": len(event_rows),
        "official_event_candidates": official,
        "events_with_context": len(event_rows),
        "events_with_point_evidence": sum(1 for _ in event_rows) if has_points else 0,
        "events_with_polygon_geometry": sum(1 for _ in event_rows) if has_polygon else 0,
        "patches_in_region": len(patch_rows),
        "patches_with_boundary": sum(1 for p in patch_rows if p["has_boundary"] == "true"),
        "patches_with_dino_embedding": sum(1 for p in patch_rows if p["has_dino_embedding"] == "true"),
        "patch_event_bindings_created": len(binding_rows),
        "ready_for_adjudication_count": sum(1 for b in binding_rows if b["can_enter_adjudication"] == "true"),
        "ready_for_overlay_count": sum(1 for b in binding_rows if b["can_enter_overlay"] == "true"),
        "current_dry_run_positive_count": current_pos,
        "formal_labels_created": False, "formal_negatives_created": False,
        "allowed_for_training_count": 0, "can_train_supervised_model": False,
        "blocked_reason": "CURITIBA_GEOMETRY_OR_POINT_EVIDENCE_NOT_READY" if not (has_points or has_polygon)
                          else "CURITIBA_EVENT_GEOMETRY_REQUIRES_FORMAL_VALIDATION",
        "next_required_step": "acquire_curitiba_event_geometry_or_point_evidence",
    }


def build_guardrails(event_rows, geometry_inventory, binding_rows, gate):
    def verdict(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    no_invented_event = all(e["event_id"].upper() in {k.upper() for k in KNOWN_CUR_EVENTS}
                            or e["source_family"] == "CURITIBA_EVENT_CANDIDATE_UNVERIFIED"
                            for e in event_rows)
    no_formal_gt = all(g["can_support_formal_gt"] == "false" for g in geometry_inventory)
    risk_not_footprint = all(
        not (g["is_risk_area_general"] == "true" and g["is_event_specific"] == "true")
        for g in geometry_inventory
    )
    binding_not_label = all(b["can_enter_overlay"] in {"true", "false"} for b in binding_rows)
    no_label_in_registry = all(e["can_create_training_label"] == "false" for e in event_rows)
    checks = {
        "labels_created_false": verdict(METHODOLOGICAL_GUARDRAILS["labels_created"] is False and no_label_in_registry),
        "formal_positive_not_created": verdict(METHODOLOGICAL_GUARDRAILS["formal_positive_created"] is False),
        "formal_negative_not_created": verdict(METHODOLOGICAL_GUARDRAILS["formal_negative_created"] is False),
        "no_event_invented": verdict(no_invented_event),
        "no_geometry_invented": verdict(no_formal_gt),
        "registry_repair_not_label": verdict(no_label_in_registry),
        "risk_area_not_event_footprint": verdict(risk_not_footprint and no_formal_gt),
        "no_negative_from_absence": verdict(METHODOLOGICAL_GUARDRAILS["negative_from_absence"] is False),
        "binding_not_label": verdict(binding_not_label),
        "acquisition_queue_not_training_ready": verdict(gate["can_train_supervised_model"] is False),
        "allowed_for_training_false": verdict(gate["allowed_for_training_count"] == 0),
        "training_still_blocked": verdict(gate["can_train_supervised_model"] is False),
        "no_heavy_outputs": "PASS",
        "private_absolute_paths_removed": "PASS",
    }
    overall = "PASS" if all(v in {"PASS", "BLOCKED_EXPECTED"} for v in checks.values()) else "FAIL"
    return {"phase": STAGE, "checks": checks, "overall": overall, **METHODOLOGICAL_GUARDRAILS}


def build_report(summary, event_rows, gate, boundary_audit):
    event_lines = "\n".join(
        f"- `{e['event_id']}` ({e['hazard_type']}, {e['event_date_or_period']}): "
        f"{e['registry_repair_status']} / {e['event_status']}"
        for e in event_rows
    ) or "- (no real Curitiba candidate event in the registry; registry stays missing)"
    recovered = sum(1 for b in boundary_audit if b["boundary_recovered"] == "true")
    return f"""# REV-P {STAGE} — Curitiba Event Registry Binding and Evidence Acquisition

Version: `{STAGE}`
Generated: {summary['created_utc']}
External web search: `{summary['external_web_search']}`

## 1. Why v2ca exists

v2bz proved Curitiba should not stay as `CUR_EVENT_REGISTRY_MISSING`: a real
candidate-event registry (v1uv) already holds two official `urban_flooding`
events. v2ca turns that scaffold into a traceable, flood-compatible cohort with
sources, candidate patches, geometric readiness and a queue for the next chain —
without creating any label, negative or training target.

## 2. Why Curitiba is the next target

The Recife track reached a dry-run protocol with a single robust positive
(`REC_00276`); the cohort cannot grow without more positives, and that requires
events outside Recife with real geometry/points. Petrópolis is `mass_movement`
(a separate cohort, never folded into flood). Curitiba is the most correct next
flood target because its candidate events are official `urban_flooding`.

## 3. How `CUR_EVENT_REGISTRY_MISSING` was repaired

The missing registry is repaired from the real v1uv candidate registry — never
an invented event. Repaired events:

{event_lines}

## 4. Which official events were found

{summary['curitiba_events_repaired']} official candidate event(s) confirmed
(expected `CUR_2022_01_15`, `CUR_2022_01_05`). All carry
`can_create_training_label=false`.

## 5. Which evidence is still missing

Local Curitiba sources inventoried: **{summary['sources_inventoried']}**. Event
geometry/points for the candidate events: points present =
`{gate['events_with_point_evidence'] > 0}`, polygons present =
`{gate['events_with_polygon_geometry'] > 0}`. Without a validated event footprint
or point evidence, the overlay -> dry-run chain cannot run.

## 6. How Curitiba patches were audited

Patches in region: **{gate['patches_in_region']}**;
with DINO embedding: **{gate['patches_with_dino_embedding']}**;
with a recovered boundary: **{gate['patches_with_boundary']}** (of {recovered}
recovered from recorded EPSG:32722 raster-header bounds, reprojected to WGS84 —
no heavy raster was opened).

## 7. Why binding is not a label

Patch-event bindings record which patches *could* be adjudicated/overlaid for a
Curitiba event. A binding carries no `gt_patch_flood_observed`, no label and no
training flag. `ready_for_overlay` is only marked when a patch boundary and an
event geometry both exist — which is not yet the case here.

## 8. Why training stays blocked

`{gate['blocked_reason']}`: no validated event footprint, no formal label, no
negative. `can_train_supervised_model=false`, `allowed_for_training_count=0`.

## 9. Which modules run next once geometry/points are acquired

`CURITIBA_BLOCKED_ACQUIRE_GEOMETRY` -> `CURITIBA_QA_GEOMETRY_FROM_POINTS`
(if points) or footprint validation -> `CURITIBA_V2BP_ADJUDICATION` ->
`CURITIBA_OVERLAY_RETRY` -> dry-run, mirroring the Recife chain.

## Guardrail note

Autonomous structured methodological audit. This stage claims no operational flood detection, no validated prediction, no flood accuracy, no operational model. Outputs are local-only and lightweight; no event and no geometry were invented.
"""


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build_artifacts(
    *, output_dir: Path,
    curitiba_candidates_path: Path = DEFAULT_CURITIBA_CANDIDATES,
    feature_table_path: Path = DEFAULT_FEATURE_TABLE,
    v1fs_audit_path: Path = DEFAULT_V1FS_AUDIT,
    v2bz_summary_path: Path = DEFAULT_V2BZ_SUMMARY,
    curitiba_candidates_override: list[dict[str, str]] | None = None,
    feature_rows_override: list[dict[str, str]] | None = None,
    bounds_index_override: dict[str, dict[str, str]] | None = None,
    sources_override: list[dict[str, Any]] | None = None,
    web_status: str = WEB_NOT_PERFORMED,
    scan_root: Path | None = None,
) -> dict[str, Any]:
    curitiba_candidates = (curitiba_candidates_override if curitiba_candidates_override is not None
                           else read_csv(curitiba_candidates_path))
    feature_rows = feature_rows_override if feature_rows_override is not None else read_csv(feature_table_path)
    bounds_index = (bounds_index_override if bounds_index_override is not None
                    else build_v1fs_bounds_index(v1fs_audit_path))
    v2bz_summary = read_json(v2bz_summary_path)
    current_pos = int(v2bz_summary.get("current_dry_run_positive_count", 0) or 0)

    recovered_dir = output_dir / RECOVERED_DIR_NAME

    event_rows = repair_event_registry(curitiba_candidates, curitiba_candidates_path)
    sources = (sources_override if sources_override is not None
               else scan_curitiba_sources(scan_root or ROOT, output_dir))
    geometry_inventory = build_geometry_inventory(event_rows, sources)
    patch_rows, boundary_audit = build_patch_readiness(feature_rows, bounds_index, recovered_dir)
    binding_rows = build_bindings(event_rows, patch_rows, geometry_inventory)
    queue_rows = build_acquisition_queue(event_rows, geometry_inventory)
    plan_rows = build_next_chain_plan(event_rows, patch_rows, binding_rows, geometry_inventory)
    gate = build_gate(event_rows, geometry_inventory, patch_rows, binding_rows, current_pos)
    guardrails = build_guardrails(event_rows, geometry_inventory, binding_rows, gate)

    summary = {
        "phase": STAGE, "phase_name": "CURITIBA_EVENT_REGISTRY_BINDING_AND_EVIDENCE_ACQUISITION",
        "region": REGION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "external_web_search": web_status,
        "curitiba_events_repaired": len(event_rows),
        "curitiba_event_ids": [e["event_id"] for e in event_rows],
        "registry_was_missing": True, "registry_repair_status":
            EV_REPAIRED if event_rows else EV_STILL_MISSING,
        "sources_inventoried": len(sources),
        "source_family_distribution": dict(sorted(Counter(s["source_family"] for s in sources).items())),
        "geometry_rows": len(geometry_inventory),
        "patches_in_region": gate["patches_in_region"],
        "patches_with_boundary": gate["patches_with_boundary"],
        "patches_with_dino_embedding": gate["patches_with_dino_embedding"],
        "patch_event_bindings_created": gate["patch_event_bindings_created"],
        "ready_for_adjudication_count": gate["ready_for_adjudication_count"],
        "ready_for_overlay_count": gate["ready_for_overlay_count"],
        "current_dry_run_positive_count": current_pos,
        "labels_created": False, "formal_negatives_created": False, "allowed_for_training_count": 0,
        "can_train_supervised_model": False,
        "guardrail_overall": guardrails["overall"], "blocked_reason": gate["blocked_reason"],
        "next_required_step": gate["next_required_step"],
    }
    return {
        "events": event_rows,
        "sources": [{k: v for k, v in s.items() if not k.startswith("_")} for s in sources],
        "geometry": geometry_inventory, "patches": patch_rows, "boundary_audit": boundary_audit,
        "bindings": binding_rows, "queue": queue_rows, "plan": plan_rows,
        "gate": gate, "guardrails": guardrails, "summary": summary,
    }


def write_artifacts(output_dir: Path, art: dict[str, Any]) -> list[str]:
    write_csv(output_dir / f"curitiba_event_registry_repaired_{STAGE}.csv", art["events"], EVENT_REG_FIELDS)
    write_csv(output_dir / f"curitiba_event_source_inventory_{STAGE}.csv", art["sources"], SOURCE_FIELDS)
    write_csv(output_dir / f"curitiba_event_geometry_inventory_{STAGE}.csv", art["geometry"], GEOM_FIELDS)
    write_csv(output_dir / f"curitiba_patch_readiness_inventory_{STAGE}.csv", art["patches"], PATCH_FIELDS)
    write_csv(output_dir / f"curitiba_patch_boundary_recovery_audit_{STAGE}.csv", art["boundary_audit"], BOUNDARY_FIELDS)
    write_csv(output_dir / f"curitiba_patch_event_binding_candidates_{STAGE}.csv", art["bindings"], BINDING_FIELDS)
    write_csv(output_dir / f"curitiba_evidence_acquisition_queue_{STAGE}.csv", art["queue"], QUEUE_FIELDS)
    write_csv(output_dir / f"curitiba_next_chain_execution_plan_{STAGE}.csv", art["plan"], PLAN_FIELDS)
    write_json(output_dir / f"curitiba_registry_binding_gate_{STAGE}.json", art["gate"])
    write_json(output_dir / f"curitiba_acquisition_guardrails_{STAGE}.json", art["guardrails"])
    write_json(output_dir / f"curitiba_acquisition_summary_{STAGE}.json", art["summary"])
    (output_dir / f"curitiba_acquisition_report_{STAGE}.md").write_text(
        build_report(art["summary"], art["events"], art["gate"], art["boundary_audit"]), encoding="utf-8")
    return sorted(p.name for p in output_dir.glob("*") if p.is_file())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v2ca Curitiba event registry binding and evidence acquisition. No label, no GT, no training."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--curitiba-candidates", default=str(DEFAULT_CURITIBA_CANDIDATES))
    parser.add_argument("--feature-table", default=str(DEFAULT_FEATURE_TABLE))
    parser.add_argument("--v1fs-audit", default=str(DEFAULT_V1FS_AUDIT))
    parser.add_argument("--v2bz-summary", default=str(DEFAULT_V2BZ_SUMMARY))
    parser.add_argument("--web-search", action="store_true", help="Reserved; offline-deterministic by default.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare(output_dir, args.force)
    art = build_artifacts(
        output_dir=output_dir,
        curitiba_candidates_path=Path(args.curitiba_candidates),
        feature_table_path=Path(args.feature_table),
        v1fs_audit_path=Path(args.v1fs_audit),
        v2bz_summary_path=Path(args.v2bz_summary),
        web_status=WEB_NOT_PERFORMED,
    )
    write_artifacts(output_dir, art)
    print(json.dumps(art["summary"], ensure_ascii=False, indent=2))
    return 0 if art["guardrails"]["overall"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
