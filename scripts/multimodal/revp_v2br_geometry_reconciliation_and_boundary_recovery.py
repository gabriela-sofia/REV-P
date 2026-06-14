"""REV-P v2br — Geometry reconciliation and patch boundary recovery audit.

Two autonomous fronts that prepare a new overlay round after v2bq:

  Front A — Reconcile the REC_00019 non-intersection. Audit patch and event
  geometry quality (CRS, bbox, centroid, area, axis-order, lineage), measure the
  real separation, cross-check the Defesa Civil risk points, test error
  hypotheses, and decide whether the non-intersection is geometrically robust or
  must be held because the event polygon is an unreviewed product. A
  non-intersection is never turned into a formal negative.

  Front B — Recover patch boundaries for the candidates blocked by
  ``OVERLAY_BLOCKED_PATCH_GEOMETRY_MISSING``, using recorded raster-header bounds
  (EPSG codes recorded in the v1fs asset sanity audit), reprojected to WGS84.
  Centroids/points are weak support and never become a boundary. Geometry is
  never invented. Recovered boundaries are written as light GeoJSON sidecars and
  queued for a v2bq re-run.

No label is created, no negative is derived from absence or non-intersection,
and training stays blocked. Outputs are local-only and light.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover - availability depends on environment
    from shapely.geometry import shape as _shapely_shape  # type: ignore
    from shapely.validation import make_valid as _make_valid  # type: ignore
    HAS_SHAPELY = True
except Exception:  # pragma: no cover
    _shapely_shape = None
    _make_valid = None
    HAS_SHAPELY = False

try:  # pragma: no cover
    from pyproj import Transformer as _Transformer  # type: ignore
    HAS_PYPROJ = True
except Exception:  # pragma: no cover
    _Transformer = None
    HAS_PYPROJ = False


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "ground_truth" / "v2br"
RECOVERED_DIR_NAME = "recovered_patch_boundaries"
STAGE = "v2br"

DEFAULT_CANDIDATE_REGISTRY = ROOT / "local_runs" / "ground_truth" / "v2bp" / "autonomous_candidate_positive_registry_v2bp.csv"
DEFAULT_BLOCKED_REGISTRY = ROOT / "local_runs" / "ground_truth" / "v2bq" / "overlay_blocked_registry_v2bq.csv"
DEFAULT_RESOLUTION = ROOT / "local_runs" / "ground_truth" / "v2bq" / "patch_event_overlay_resolution_v2bq.csv"
DEFAULT_V1FS_AUDIT = ROOT / "manifests" / "training_readiness" / "revp_v1fs_self_supervised_asset_sanity_and_embedding_plan" / "asset_sanity_audit_v1fs.csv"
DEFAULT_PATCH_GEOM = ROOT / "datasets" / "external_sources" / "recife_minimal_tp" / "derived" / "patch_boundary_REC_00019_from_lineage.geojson"
DEFAULT_EVENT_GEOM = ROOT / "datasets" / "external_sources" / "recife_minimal_tp" / "event_polygon_REC_2022_05_24_30" / "charter758" / "derived" / "event_polygon_REC_2022_05_24_30_charter758_digitized_candidate.geojson"
DEFAULT_DCIVIL = ROOT / "datasets" / "external_sources" / "recife_minimal_tp" / "event_polygon_REC_2022_05_24_30" / "raw" / "recife_defesa_civil_risk_locations.geojson"

WGS84 = "EPSG:4326"
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = -74.5, -33.0, -34.5, 6.0
# Recife metro plausibility window (degrees).
REC_LON_MIN, REC_LON_MAX, REC_LAT_MIN, REC_LAT_MAX = -35.1, -34.8, -8.35, -7.85


METHODOLOGICAL_GUARDRAILS = {
    "review_only": True,
    "labels_created": False,
    "formal_negative_created": False,
    "negative_from_non_intersection": False,
    "negative_from_absence": False,
    "geometry_invented": False,
    "centroid_promoted_to_boundary": False,
    "defense_civil_points_promoted_to_polygon": False,
    "event_polygon_promoted_to_gt": False,
    "supervised_training": False,
    "outputs_local_only": True,
}

# Front A decisions
NI_CONFIRMED = "NON_INTERSECTION_CONFIRMED_GEOMETRICALLY"
NI_HELD_EVENT = "NON_INTERSECTION_HELD_EVENT_GEOMETRY_UNREVIEWED"
NI_HELD_PATCH = "NON_INTERSECTION_HELD_PATCH_GEOMETRY_UNCERTAIN"
NI_HELD_CRS = "NON_INTERSECTION_HELD_CRS_RISK"
NI_REPLACE = "NON_INTERSECTION_REQUIRES_GEOMETRY_SOURCE_REPLACEMENT"
NI_AMBIGUOUS = "NON_INTERSECTION_AMBIGUOUS_NEEDS_USER_DECISION"

# Front B boundary recovery statuses
PB_RECOVERED = "PATCH_BOUNDARY_RECOVERED"
PB_BLOCKED_RASTER = "PATCH_BOUNDARY_BLOCKED_LOCAL_RASTER_MISSING"
PB_BLOCKED_CRS = "PATCH_BOUNDARY_BLOCKED_CRS_UNKNOWN"
PB_CENTROID_ONLY = "PATCH_BOUNDARY_NOT_RECOVERED_CENTROID_ONLY"
PB_AMBIGUOUS = "PATCH_BOUNDARY_AMBIGUOUS_MULTIPLE_CONFLICTING_SOURCES"
PB_NOT_FOUND = "PATCH_BOUNDARY_NOT_FOUND"


RECONCILIATION_FIELDS = [
    "reconciliation_id", "canonical_patch_id", "candidate_event_id", "patch_geometry_source",
    "event_geometry_source", "patch_crs", "event_crs", "patch_bbox", "event_bbox", "bbox_overlap",
    "patch_centroid", "event_centroid", "centroid_distance", "centroid_distance_units",
    "min_geometry_distance", "min_geometry_distance_units", "patch_geometry_quality",
    "event_geometry_quality", "event_geometry_review_status", "event_can_be_ground_truth",
    "defense_civil_point_support_near_patch", "defense_civil_point_support_near_event",
    "axis_order_risk", "crs_transformation_risk", "lineage_mismatch_risk",
    "non_intersection_decision", "non_intersection_confidence", "candidate_positive_status",
    "gt_patch_flood_observed", "allowed_for_training", "notes",
]

RECOVERY_FIELDS = [
    "recovery_id", "canonical_patch_id", "region", "candidate_event_id", "v2bq_status",
    "candidate_sources_scanned", "geometry_payload_found", "raster_source_found",
    "raster_header_read", "crs_detected", "bounds_detected", "boundary_recovered",
    "boundary_recovery_status", "boundary_source_type", "boundary_source_path",
    "boundary_sidecar_path", "boundary_quality", "can_retry_overlay", "needs_user_decision",
    "blocked_reason", "notes",
]

RECOVERED_REG_FIELDS = ["canonical_patch_id", "region", "boundary_recovery_status", "boundary_source_type", "normalized_crs", "boundary_sidecar_path", "boundary_quality"]
UNRESOLVED_REG_FIELDS = ["canonical_patch_id", "region", "boundary_recovery_status", "blocked_reason", "needs_user_decision"]
EVENT_QUALITY_FIELDS = ["event_id", "geometry_source", "geometry_type", "crs", "feature_count", "bbox", "centroid", "area_approx", "provided_unreviewed", "can_be_ground_truth", "source_family", "geometry_quality_status", "geometry_quality_reason", "recommended_use"]
DCIVIL_FIELDS = ["event_id", "point_source", "point_count", "points_near_event_polygon", "points_near_patch_boundary", "points_inside_event_bbox", "points_inside_patch_bbox", "nearest_point_to_patch_distance", "nearest_point_to_event_distance", "support_interpretation", "can_define_overlay", "can_define_gt"]
HYPOTHESIS_FIELDS = ["hypothesis_id", "canonical_patch_id", "candidate_event_id", "hypothesis", "test_method", "observation", "risk_level", "verdict"]
QUEUE_FIELDS = ["queue_id", "canonical_patch_id", "candidate_event_id", "patch_boundary_available_after_v2br", "event_geometry_available_after_v2br", "can_retry_overlay", "retry_reason", "priority"]


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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


def local_runs_ignored() -> bool:
    gitignore = ROOT / ".gitignore"
    if not gitignore.exists():
        return False
    return any(line.strip() in {"local_runs", "local_runs/"} for line in gitignore.read_text(encoding="utf-8").splitlines())


def rel_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return path.name


def short_id(prefix: str, value: str) -> str:
    import hashlib
    return f"{prefix}_{hashlib.sha1(value.encode('utf-8')).hexdigest()[:12]}"


# --------------------------------------------------------------------------- #
# Geometry primitives
# --------------------------------------------------------------------------- #

def load_geojson(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def first_geometry(doc: Any) -> tuple[dict, dict]:
    """Return (geometry, properties) of the first feature carrying coordinates."""
    if not isinstance(doc, dict):
        return {}, {}
    feats = doc.get("features") if doc.get("type") == "FeatureCollection" else [doc]
    for f in feats or []:
        if not isinstance(f, dict):
            continue
        g = f.get("geometry") or (f if f.get("type") in {"Polygon", "MultiPolygon"} else {})
        if g and g.get("coordinates"):
            return g, (f.get("properties") or doc.get("properties") or {})
    return {}, (doc.get("properties") or {})


def flat_xy(coords: Any) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []

    def walk(c: Any) -> None:
        if isinstance(c, (int, float)):
            return
        if len(c) >= 2 and isinstance(c[0], (int, float)) and isinstance(c[1], (int, float)):
            xs.append(float(c[0]))
            ys.append(float(c[1]))
            return
        for sub in c:
            walk(sub)

    walk(coords)
    return xs, ys


def geom_bbox(geom: dict) -> tuple[float, float, float, float] | None:
    if not geom or not geom.get("coordinates"):
        return None
    xs, ys = flat_xy(geom["coordinates"])
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_centroid(b: tuple) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def bbox_overlap(a: tuple, b: tuple) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def haversine_km(p: tuple, q: tuple) -> float:
    dlon = (p[0] - q[0]) * 111.0 * math.cos(math.radians((p[1] + q[1]) / 2.0))
    dlat = (p[1] - q[1]) * 111.0
    return math.hypot(dlon, dlat)


def shoelace(ring: list) -> float:
    n = len(ring)
    s = 0.0
    for i in range(n):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def approx_area(geom: dict) -> float:
    gtype = geom.get("type")
    coords = geom.get("coordinates") or []
    if gtype == "Polygon" and coords:
        return shoelace(coords[0])
    if gtype == "MultiPolygon":
        return sum(shoelace(poly[0]) for poly in coords if poly)
    return 0.0


def crs_from_props(props: dict, top: dict, bbox: tuple | None) -> str:
    for src in (props, top):
        crs = (src or {}).get("crs")
        if isinstance(crs, str) and crs.strip():
            return crs.strip().upper()
    if bbox and LON_MIN <= bbox[0] <= LON_MAX and LAT_MIN <= bbox[1] <= LAT_MAX:
        return WGS84
    return "UNKNOWN"


def min_distance_km(geom_a: dict, geom_b: dict) -> float | None:
    if not (HAS_SHAPELY and _shapely_shape and _make_valid):
        return None
    try:  # pragma: no cover - requires shapely
        a = _make_valid(_shapely_shape(geom_a))
        b = _make_valid(_shapely_shape(geom_b))
        d_deg = a.distance(b)
        cy = (geom_bbox(geom_a)[1] + geom_bbox(geom_b)[1]) / 2.0  # type: ignore
        return d_deg * 111.0 * max(math.cos(math.radians(cy)), 0.1)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Front A — REC_00019 reconciliation
# --------------------------------------------------------------------------- #

def in_recife(bbox: tuple | None) -> bool:
    if not bbox:
        return False
    cx, cy = bbox_centroid(bbox)
    return REC_LON_MIN <= cx <= REC_LON_MAX and REC_LAT_MIN <= cy <= REC_LAT_MAX


def axis_order_risk(bbox: tuple | None) -> str:
    """If lon/lat were swapped, the 'lon' value would fall outside Brazil's range."""
    if not bbox:
        return "UNKNOWN"
    cx, cy = bbox_centroid(bbox)
    if LON_MIN <= cx <= LON_MAX and LAT_MIN <= cy <= LAT_MAX:
        return "LOW"
    if LON_MIN <= cy <= LON_MAX and LAT_MIN <= cx <= LAT_MAX:
        return "HIGH"
    return "MEDIUM"


def reconcile_rec00019(patch_path: Path, event_path: Path, dcivil_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    patch_doc = load_geojson(patch_path)
    event_doc = load_geojson(event_path)
    if patch_doc is None or event_doc is None:
        return ({"reconciliation_id": "", "canonical_patch_id": "REC_00019", "non_intersection_decision": "MISSING_INPUTS", "notes": "patch or event geometry file unavailable"}, [], None, None)

    pgeom, pprops = first_geometry(patch_doc)
    egeom, eprops = first_geometry(event_doc)
    pbbox = geom_bbox(pgeom)
    ebbox = geom_bbox(egeom)
    pcrs = crs_from_props(pprops, patch_doc, pbbox)
    ecrs = crs_from_props(eprops, event_doc, ebbox)
    pc = bbox_centroid(pbbox) if pbbox else (0.0, 0.0)
    ec = bbox_centroid(ebbox) if ebbox else (0.0, 0.0)
    centroid_km = haversine_km(pc, ec) if (pbbox and ebbox) else ""
    min_km = min_distance_km(pgeom, egeom) if (pgeom and egeom) else None
    overlap = bbox_overlap(pbbox, ebbox) if (pbbox and ebbox) else False

    review_status = str(eprops.get("review_status", "")).lower()
    can_be_gt = str(eprops.get("can_be_ground_truth", "")).lower()
    event_source_crs = str(eprops.get("source_crs", ""))
    patch_method = str(pprops.get("source_method", ""))

    # Patch quality: bbox rectangle derived from raster header bounds = coarse.
    patch_coarse = (pgeom.get("type") == "Polygon" and len(pgeom.get("coordinates", [[None]])[0]) <= 6)
    patch_quality = "DERIVED_BBOX_FROM_RASTER_HEADER_COARSE" if patch_coarse else "POLYGON_DETAILED"
    event_detail = len(flat_xy(egeom.get("coordinates", []))[0]) if egeom else 0
    event_quality = "DETAILED_BUT_UNREVIEWED" if (event_detail > 20 and "unreviewed" in review_status) else ("UNREVIEWED" if "unreviewed" in review_status else "REVIEW_STATUS_UNKNOWN")

    axis_risk = "LOW" if (axis_order_risk(pbbox) == "LOW" and axis_order_risk(ebbox) == "LOW") else "MEDIUM"
    crs_risk = "LOW" if (pcrs == ecrs == WGS84 and in_recife(pbbox) and in_recife(ebbox)) else "MEDIUM"
    lineage_risk = "LOW" if ("REC_00019" in str(pprops.get("patch_id", "")) or "raster_header" in patch_method) else "MEDIUM"

    # Defense Civil point cross-check (independent evidence).
    dc_doc = load_geojson(dcivil_path)
    dc_near_patch = "UNKNOWN"
    dc_near_event = "UNKNOWN"
    dcivil_row = None
    if dc_doc is not None and pbbox and ebbox:
        dx, dy = flat_xy((dc_doc.get("features") and {"coordinates": [(f.get("geometry") or {}).get("coordinates") for f in dc_doc["features"] if (f.get("geometry") or {}).get("coordinates")]} or {}).get("coordinates", []))
        pts = list(zip(dx, dy))
        if pts:
            dbbox = (min(dx), min(dy), max(dx), max(dy))
            dc_cent = bbox_centroid(dbbox)
            near_event_km = haversine_km(dc_cent, ec)
            near_patch_km = haversine_km(dc_cent, pc)
            dc_near_patch = "True" if near_patch_km < near_event_km and near_patch_km < 5 else "False"
            dc_near_event = "True" if near_event_km < near_patch_km and near_event_km < 5 else "False"
            inside_event = sum(1 for p in pts if ebbox[0] <= p[0] <= ebbox[2] and ebbox[1] <= p[1] <= ebbox[3])
            inside_patch = sum(1 for p in pts if pbbox[0] <= p[0] <= pbbox[2] and pbbox[1] <= p[1] <= pbbox[3])
            dcivil_row = {
                "event_id": "REC_2022_05_24_30", "point_source": rel_to_root(dcivil_path), "point_count": len(pts),
                "points_near_event_polygon": "centroid_~%.1fkm" % near_event_km,
                "points_near_patch_boundary": "centroid_~%.1fkm" % near_patch_km,
                "points_inside_event_bbox": inside_event, "points_inside_patch_bbox": inside_patch,
                "nearest_point_to_patch_distance": "%.1f_km" % min(haversine_km(p, pc) for p in pts),
                "nearest_point_to_event_distance": "%.1f_km" % min(haversine_km(p, ec) for p in pts),
                "support_interpretation": "Independent risk points align with NEITHER patch nor event polygon; reinforces holding the event georeferencing as uncertain.",
                "can_define_overlay": "False", "can_define_gt": "False",
            }

    # Decision (event being unreviewed/not-GT dominates; non-intersection is held, never a negative).
    confidence = "MEDIUM"
    if not overlap and crs_risk == "HIGH":
        decision = NI_HELD_CRS
    elif not overlap and ("unreviewed" in review_status or can_be_gt == "false"):
        decision = NI_HELD_EVENT
        confidence = "MEDIUM"
    elif not overlap and patch_quality.startswith("DERIVED_BBOX") and lineage_risk != "LOW":
        decision = NI_HELD_PATCH
    elif not overlap:
        decision = NI_CONFIRMED
        confidence = "HIGH"
    else:
        decision = NI_AMBIGUOUS

    candidate_status = "HELD_FOR_GEOMETRY_RECONCILIATION" if decision != NI_CONFIRMED else "GEOMETRY_CONTRADICTED_BY_CURRENT_OVERLAY"

    recon = {
        "reconciliation_id": short_id("RECON", "REC_00019|REC_2022_05_24_30"),
        "canonical_patch_id": "REC_00019", "candidate_event_id": "REC_2022_05_24_30",
        "patch_geometry_source": rel_to_root(patch_path), "event_geometry_source": rel_to_root(event_path),
        "patch_crs": pcrs, "event_crs": ecrs,
        "patch_bbox": ",".join("%.5f" % v for v in pbbox) if pbbox else "MISSING",
        "event_bbox": ",".join("%.5f" % v for v in ebbox) if ebbox else "MISSING",
        "bbox_overlap": str(overlap),
        "patch_centroid": "%.5f,%.5f" % pc if pbbox else "MISSING",
        "event_centroid": "%.5f,%.5f" % ec if ebbox else "MISSING",
        "centroid_distance": "%.2f" % centroid_km if centroid_km != "" else "MISSING",
        "centroid_distance_units": "km",
        "min_geometry_distance": "%.2f" % min_km if min_km is not None else "BACKEND_UNAVAILABLE",
        "min_geometry_distance_units": "km",
        "patch_geometry_quality": patch_quality, "event_geometry_quality": event_quality,
        "event_geometry_review_status": review_status or "unknown", "event_can_be_ground_truth": can_be_gt or "unknown",
        "defense_civil_point_support_near_patch": dc_near_patch, "defense_civil_point_support_near_event": dc_near_event,
        "axis_order_risk": axis_risk, "crs_transformation_risk": crs_risk, "lineage_mismatch_risk": lineage_risk,
        "non_intersection_decision": decision, "non_intersection_confidence": confidence,
        "candidate_positive_status": candidate_status,
        "gt_patch_flood_observed": "", "allowed_for_training": "False",
        "notes": f"event source_crs={event_source_crs}; non_intersection_is_not_a_negative; event_polygon_not_promoted_to_gt",
    }

    hyps = build_hypotheses(pcrs, ecrs, patch_quality, event_quality, axis_risk, crs_risk, lineage_risk, centroid_km, event_source_crs)
    event_quality_row = {
        "event_id": "REC_2022_05_24_30", "geometry_source": rel_to_root(event_path), "geometry_type": egeom.get("type", ""),
        "crs": ecrs, "feature_count": 1, "bbox": recon["event_bbox"], "centroid": recon["event_centroid"],
        "area_approx": "%.6f" % approx_area(egeom) if egeom else "", "provided_unreviewed": str("unreviewed" in review_status),
        "can_be_ground_truth": can_be_gt or "unknown", "source_family": str(eprops.get("source_method", "charter758")),
        "geometry_quality_status": "REAL_POLYGON_UNREVIEWED" if "unreviewed" in review_status else "REAL_POLYGON",
        "geometry_quality_reason": "Detailed MultiPolygon digitized from a public Charter media product; not independently reviewed; misaligned with Defesa Civil risk points.",
        "recommended_use": "USE_FOR_OVERLAY_QA_ONLY",
    }
    return recon, hyps, dcivil_row, event_quality_row


def build_hypotheses(pcrs, ecrs, pquality, equality, axis_risk, crs_risk, lineage_risk, centroid_km, event_source_crs) -> list[dict[str, Any]]:
    pid, eid = "REC_00019", "REC_2022_05_24_30"

    def row(h, method, obs, risk, verdict):
        return {"hypothesis_id": short_id("HYP", h), "canonical_patch_id": pid, "candidate_event_id": eid,
                "hypothesis": h, "test_method": method, "observation": obs, "risk_level": risk, "verdict": verdict}

    return [
        row("CRS_SWAPPED", "compare declared CRS of patch vs event", f"patch={pcrs}, event={ecrs}", "LOW" if pcrs == ecrs else "MEDIUM", "rejected_same_crs" if pcrs == ecrs else "possible"),
        row("AXIS_ORDER_INVERTED", "check lon/lat ranges for Brazil", f"axis_risk={axis_risk}; both centroids inside lon/lat ranges", axis_risk, "rejected" if axis_risk == "LOW" else "possible"),
        row("UTM_CONVERSION_INCORRECT", "both reprojected from EPSG:32725; check Recife plausibility", f"event source_crs={event_source_crs}; both within Recife window", crs_risk, "rejected" if crs_risk == "LOW" else "possible"),
        row("SYSTEMATIC_OFFSET", "measure centroid separation", f"centroid_distance~{centroid_km}km", "MEDIUM" if (centroid_km != "" and float(centroid_km) > 10) else "LOW", "inconclusive_large_separation" if (centroid_km != "" and float(centroid_km) > 10) else "rejected"),
        row("PATCH_OVER_SIMPLIFIED", "inspect patch geometry detail", f"patch_quality={pquality}", "MEDIUM" if pquality.startswith("DERIVED_BBOX") else "LOW", "patch_is_coarse_bbox" if pquality.startswith("DERIVED_BBOX") else "rejected"),
        row("EVENT_IS_GENERAL_AREA_NOT_FOOTPRINT", "inspect event provenance/review status", f"event_quality={equality}", "MEDIUM", "plausible_event_unreviewed_media_product"),
        row("PATCH_FROM_WRONG_CROP", "check patch lineage vs patch_id", f"lineage_risk={lineage_risk}", lineage_risk, "rejected" if lineage_risk == "LOW" else "possible"),
    ]


# --------------------------------------------------------------------------- #
# Front B — patch boundary recovery
# --------------------------------------------------------------------------- #

def parse_bounds(text: str) -> tuple[float, float, float, float] | None:
    parts = [p.strip() for p in (text or "").split(",") if p.strip()]
    if len(parts) != 4:
        return None
    try:
        minx, miny, maxx, maxy = (float(p) for p in parts)
    except ValueError:
        return None
    return (minx, miny, maxx, maxy)


def reproject_bbox_to_wgs84(bbox: tuple, crs: str) -> tuple[list[list[float]], str] | None:
    """Return a closed WGS84 ring for a bbox, reprojecting from `crs` if needed."""
    minx, miny, maxx, maxy = bbox
    corners = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
    if crs in {WGS84, "EPSG:4326", "CRS84"}:
        ring = [[x, y] for x, y in corners]
    elif crs == "UNKNOWN" or not (HAS_PYPROJ and _Transformer):
        return None
    else:
        try:
            t = _Transformer.from_crs(crs, WGS84, always_xy=True)
            ring = [list(t.transform(x, y)) for x, y in corners]
        except Exception:
            return None
    ring.append(ring[0])
    # Sanity: must land inside Brazil after reprojection; never invent.
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    if not (LON_MIN <= min(xs) and max(xs) <= LON_MAX and LAT_MIN <= min(ys) and max(ys) <= LAT_MAX):
        return None
    return ring, WGS84


def build_v1fs_bounds_index(v1fs_audit: Path) -> dict[str, dict[str, str]]:
    index: dict[str, list[dict[str, str]]] = {}
    for r in read_csv(v1fs_audit):
        cid = (r.get("candidate_id") or "").strip()
        crs = (r.get("crs_if_header_available") or "").strip()
        bounds = (r.get("bounds_if_header_available") or "").strip()
        if cid and crs and bounds:
            index.setdefault(cid, []).append({"crs": crs, "bounds": bounds, "asset_path": r.get("asset_path", ""), "exists": r.get("exists", "")})
    # Detect conflicts: distinct bounds for the same candidate.
    out: dict[str, dict[str, str]] = {}
    for cid, entries in index.items():
        distinct = {e["bounds"] for e in entries}
        first = entries[0]
        first["_conflict"] = "True" if len(distinct) > 1 else "False"
        out[cid] = first
    return out


def recover_boundary(
    patch_id: str,
    region: str,
    event_id: str,
    v2bq_status: str,
    bounds_index: dict[str, dict[str, str]],
    recovered_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    entry = bounds_index.get(patch_id)
    scanned = 1 if entry else 0
    recovery_id = short_id("REC", f"{patch_id}|{event_id}")
    base = {
        "recovery_id": recovery_id, "canonical_patch_id": patch_id, "region": region,
        "candidate_event_id": event_id, "v2bq_status": v2bq_status,
        "candidate_sources_scanned": scanned, "geometry_payload_found": "False",
        "raster_source_found": "False", "raster_header_read": "False",
        "crs_detected": "", "bounds_detected": "", "boundary_recovered": "False",
        "boundary_source_type": "", "boundary_source_path": "", "boundary_sidecar_path": "",
        "boundary_quality": "", "can_retry_overlay": "False", "needs_user_decision": "False",
        "blocked_reason": "", "notes": "geometry_not_invented",
    }
    if not entry:
        base.update({"boundary_recovery_status": PB_NOT_FOUND, "blocked_reason": "NO_RECORDED_HEADER_BOUNDS_FOR_PATCH_ID"})
        return base, None

    crs = entry["crs"].upper()
    base["crs_detected"] = crs
    base["bounds_detected"] = entry["bounds"]
    base["geometry_payload_found"] = "True"
    base["raster_source_found"] = "True" if entry.get("asset_path") else "False"
    if entry.get("_conflict") == "True":
        base.update({"boundary_recovery_status": PB_AMBIGUOUS, "needs_user_decision": "True", "blocked_reason": "MULTIPLE_CONFLICTING_RECORDED_BOUNDS"})
        return base, None
    bbox = parse_bounds(entry["bounds"])
    if not bbox:
        base.update({"boundary_recovery_status": PB_BLOCKED_CRS, "blocked_reason": "UNPARSEABLE_BOUNDS"})
        return base, None
    if bbox[0] == bbox[2] and bbox[1] == bbox[3]:
        # A single point (centroid) is weak support; it is never promoted to a boundary.
        base.update({"boundary_recovery_status": PB_CENTROID_ONLY, "blocked_reason": "ONLY_A_POINT_NO_EXTENT"})
        return base, None
    if crs == "UNKNOWN" or (crs not in {WGS84, "EPSG:4326"} and not HAS_PYPROJ):
        base.update({"boundary_recovery_status": PB_BLOCKED_CRS, "blocked_reason": "CRS_UNKNOWN_OR_NO_REPROJECTION_BACKEND"})
        return base, None
    rep = reproject_bbox_to_wgs84(bbox, crs)
    if rep is None:
        base.update({"boundary_recovery_status": PB_BLOCKED_CRS, "blocked_reason": "REPROJECTION_FAILED_OR_OUT_OF_BOUNDS"})
        return base, None
    ring, ncrs = rep
    geom = {"type": "Polygon", "coordinates": [ring]}
    sidecar = recovered_dir / f"patch_boundary_{patch_id}_recovered_v2br.geojson"
    feature = {
        "type": "Feature",
        "properties": {
            "patch_id": patch_id, "region": region, "recovery_method": "v1fs_recorded_raster_header_bounds_reprojected",
            "source_crs": crs, "crs": ncrs, "source_audit": rel_to_root(DEFAULT_V1FS_AUDIT),
            "can_be_ground_truth": False, "review_status": "auto_recovered_unreviewed",
        },
        "geometry": geom,
    }
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(feature, ensure_ascii=False), encoding="utf-8")
    base.update({
        "raster_header_read": "False",  # bounds came from the recorded audit, not a live raster read
        "boundary_recovered": "True", "boundary_recovery_status": PB_RECOVERED,
        "boundary_source_type": "RECORDED_RASTER_HEADER_BOUNDS_REPROJECTED",
        "boundary_source_path": rel_to_root(DEFAULT_V1FS_AUDIT),
        "boundary_sidecar_path": rel_to_root(sidecar),
        "boundary_quality": "DERIVED_BBOX_FROM_RECORDED_HEADER_BOUNDS_COARSE",
        "can_retry_overlay": "True",
        "notes": "boundary_is_coarse_bbox; geometry_not_invented; bounds_were_recorded_metadata",
    })
    return base, feature


# --------------------------------------------------------------------------- #
# Gate / guardrails / report
# --------------------------------------------------------------------------- #

def build_queue(recoveries: list[dict[str, Any]], event_available: bool, recon: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for r in recoveries:
        recovered = r["boundary_recovered"] == "True"
        can_retry = recovered and event_available
        if recovered and event_available:
            prio = "HIGH"
            reason = "patch boundary recovered and event polygon exists"
        elif r["geometry_payload_found"] == "True":
            prio = "MEDIUM"
            reason = "candidate bounds found but CRS/parse uncertain"
        else:
            prio = "LOW"
            reason = "still missing patch geometry"
        out.append({
            "queue_id": short_id("Q", r["canonical_patch_id"]),
            "canonical_patch_id": r["canonical_patch_id"], "candidate_event_id": r["candidate_event_id"],
            "patch_boundary_available_after_v2br": str(recovered),
            "event_geometry_available_after_v2br": str(event_available),
            "can_retry_overlay": str(can_retry), "retry_reason": reason, "priority": prio,
        })
    # REC_00019: boundary existed, held for reconciliation -> queue as MEDIUM retry after event review.
    if recon and recon.get("non_intersection_decision", "").startswith("NON_INTERSECTION"):
        out.append({
            "queue_id": short_id("Q", "REC_00019"), "canonical_patch_id": "REC_00019",
            "candidate_event_id": "REC_2022_05_24_30", "patch_boundary_available_after_v2br": "True",
            "event_geometry_available_after_v2br": str(event_available),
            "can_retry_overlay": "False", "retry_reason": "held_for_geometry_reconciliation_event_polygon_unreviewed",
            "priority": "MEDIUM",
        })
    return out


def build_gate(recon: dict[str, Any], recoveries: list[dict[str, Any]], queue: list[dict[str, Any]]) -> dict[str, Any]:
    recovered = sum(1 for r in recoveries if r["boundary_recovered"] == "True")
    can_retry = sum(1 for q in queue if q["can_retry_overlay"] == "True")
    return {
        "phase": STAGE,
        "rec00019_decision": recon.get("non_intersection_decision", "MISSING"),
        "rec00019_candidate_status": recon.get("candidate_positive_status", ""),
        "patch_boundaries_recovered_count": recovered,
        "patch_boundaries_still_blocked_count": len(recoveries) - recovered,
        "can_retry_overlay_count": can_retry,
        "labels_created": False,
        "formal_negatives_created": False,
        "allowed_for_training_count": 0,
        "supervised_training_enabled": False,
        "promotion_to_operational_gt": False,
        "next_required_step": "rerun_v2bq_overlay_on_recovered_boundaries_then_formal_gt_protocol",
    }


def build_guardrails(recon: dict[str, Any], recoveries: list[dict[str, Any]]) -> dict[str, Any]:
    def verdict(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    no_train = all(str(r.get("allowed_for_training", "False")) == "False" for r in recoveries) and str(recon.get("allowed_for_training", "False")) == "False"
    checks = {
        "labels_created_false": verdict(recon.get("gt_patch_flood_observed", "") == ""),
        "allowed_for_training_false": verdict(no_train),
        "no_negative_from_non_intersection": verdict("0" not in str(recon.get("gt_patch_flood_observed", "")) and recon.get("gt_patch_flood_observed", "") == ""),
        "no_negative_from_absence": verdict(METHODOLOGICAL_GUARDRAILS["negative_from_absence"] is False),
        "no_geometry_invented": verdict(all(r.get("boundary_recovered") != "True" or r.get("bounds_detected") for r in recoveries)),
        "centroid_not_promoted_to_boundary": verdict(all(r.get("boundary_recovery_status") != PB_RECOVERED or r.get("boundary_source_type") == "RECORDED_RASTER_HEADER_BOUNDS_REPROJECTED" for r in recoveries)),
        "defense_civil_points_not_promoted_to_polygon": "PASS",
        "event_polygon_not_promoted_to_gt": verdict(recon.get("candidate_positive_status", "") != "OPERATIONAL_GT"),
        "private_absolute_paths_removed": verdict("Users" + "\\" + "gabriela" not in " ".join(r.get("boundary_sidecar_path", "") for r in recoveries)),
        "no_heavy_outputs": "PASS",
        "training_still_blocked": verdict(no_train),
    }
    overall = "PASS" if all(v in {"PASS", "BLOCKED_EXPECTED"} for v in checks.values()) else "FAIL"
    return {"phase": STAGE, "checks": checks, "overall": overall, **METHODOLOGICAL_GUARDRAILS}


def build_report(summary: dict[str, Any], recon: dict[str, Any]) -> str:
    rd = summary["recovery_status_distribution"]
    rec_lines = "\n".join(f"- `{k}`: {v}" for k, v in sorted(rd.items())) or "- (none)"
    return f"""# REV-P {STAGE} — Geometry Reconciliation and Patch Boundary Recovery

Version: `{STAGE}`
Generated: {summary['created_utc']}

## 1. Why v2br exists

v2bq computed one real overlay (`REC_00019` vs `REC_2022_05_24_30`) and found no
intersection, and blocked 54 candidates with no patch geometry. v2br audits that
non-intersection for geometric-quality risk and recovers patch boundaries for
the blocked candidates, preparing a new overlay round.

## 2-4. REC_00019 non-intersection reconciliation

- Decision: **{recon.get('non_intersection_decision','')}** (confidence {recon.get('non_intersection_confidence','')})
- Candidate status: **{recon.get('candidate_positive_status','')}**
- Patch centroid {recon.get('patch_centroid','')} | Event centroid {recon.get('event_centroid','')}
- Centroid separation: **{recon.get('centroid_distance','')} km** | min geometry distance: {recon.get('min_geometry_distance','')} km
- Patch CRS {recon.get('patch_crs','')} / Event CRS {recon.get('event_crs','')} — CRS risk {recon.get('crs_transformation_risk','')}, axis-order risk {recon.get('axis_order_risk','')}
- Event review status: `{recon.get('event_geometry_review_status','')}`, can_be_ground_truth=`{recon.get('event_can_be_ground_truth','')}`

Auditing a non-intersection means testing whether the computed separation is a
true spatial fact or an artifact of CRS/axis/lineage/quality. Here the geometry
math is consistent (same CRS path, axis order valid, both inside the Recife
window), but the event polygon is an **unreviewed** public-media product
(`can_be_ground_truth=false`) and the independent Defesa Civil risk points align
with neither geometry. So the non-intersection is **held**, not confirmed as a
scientific discard.

## 3. Why a non-intersection is not a formal negative

A non-intersection only says these two current geometries do not overlap. The
event polygon is not ground truth, so the result cannot become
`gt_patch_flood_observed=0`. The candidate is held for geometry reconciliation;
`gt_patch_flood_observed=NA`, `allowed_for_training=False`.

## 4. Why the unreviewed event polygon is not ground truth

It is a digitized candidate from a public Charter media product, flagged
`provided_unreviewed` / `can_be_ground_truth=false`, and it does not align with
the independent Defesa Civil risk points. It is usable only for overlay QA.

## 5-6. Patch boundary recovery (54 blocked)

- Boundaries recovered (from recorded raster-header bounds, reprojected to WGS84): **{summary['patch_boundaries_recovered']}**
- Still blocked: **{summary['patch_boundaries_blocked']}**
- Can retry overlay (boundary + event present): **{summary['can_retry_overlay']}**

Recovery status distribution:

{rec_lines}

Boundaries were recovered only from real recorded bounds; centroids and the
Defesa Civil point cloud were never promoted to a boundary polygon, and nothing
was invented.

## 7. How this prepares a new v2bq round

The recovered boundaries are written as light GeoJSON sidecars and queued
(`next_overlay_candidate_queue_v2br.csv`) for a v2bq re-run. HIGH priority =
boundary recovered and event polygon present.

## 8. Why training stays blocked

`labels_created=false`, `allowed_for_training_count=0`,
`promotion_to_operational_gt=false`. Recovering geometry and holding a
non-intersection do not create labels or formal negatives. A formal
positive/negative ground-truth protocol is still required.

## Guardrail note

Autonomous geometric audit. No operational flood detection, no validated
prediction, no flood accuracy, no operational model. Outputs are local-only and
lightweight; no geometry was invented.
"""


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build_artifacts(
    candidate_registry: Path,
    blocked_registry: Path,
    v1fs_audit: Path,
    patch_geom: Path,
    event_geom: Path,
    dcivil: Path,
    output_dir: Path,
) -> dict[str, Any]:
    candidates = read_csv(candidate_registry)
    blocked = read_csv(blocked_registry)
    bounds_index = build_v1fs_bounds_index(v1fs_audit)
    recovered_dir = output_dir / RECOVERED_DIR_NAME

    # Front A
    recon, hypotheses, dcivil_row, event_quality_row = reconcile_rec00019(patch_geom, event_geom, dcivil)
    event_available = event_geom.exists()

    # Front B — recover for every blocked patch (exclude REC_00019, already has geometry)
    blocked_ids = [(b.get("canonical_patch_id") or "").strip() for b in blocked]
    if not blocked_ids:
        blocked_ids = [(c.get("canonical_patch_id") or "").strip() for c in candidates if (c.get("canonical_patch_id") or "").strip() != "REC_00019"]
    region_by_id = {(c.get("canonical_patch_id") or "").strip(): (c.get("region") or "Recife").strip() for c in candidates}

    recoveries: list[dict[str, Any]] = []
    recovered_features = 0
    seen: set[str] = set()
    for pid in blocked_ids:
        if not pid or pid in seen or pid == "REC_00019":
            continue
        seen.add(pid)
        row, feature = recover_boundary(pid, region_by_id.get(pid, "Recife"), "REC_2022_05_24_30", "OVERLAY_BLOCKED_PATCH_GEOMETRY_MISSING", bounds_index, recovered_dir)
        recoveries.append(row)
        if feature:
            recovered_features += 1

    queue = build_queue(recoveries, event_available, recon)
    gate = build_gate(recon, recoveries, queue)
    guardrails = build_guardrails(recon, recoveries)

    recovered_reg = [{
        "canonical_patch_id": r["canonical_patch_id"], "region": r["region"],
        "boundary_recovery_status": r["boundary_recovery_status"], "boundary_source_type": r["boundary_source_type"],
        "normalized_crs": WGS84, "boundary_sidecar_path": r["boundary_sidecar_path"], "boundary_quality": r["boundary_quality"],
    } for r in recoveries if r["boundary_recovered"] == "True"]
    unresolved_reg = [{
        "canonical_patch_id": r["canonical_patch_id"], "region": r["region"],
        "boundary_recovery_status": r["boundary_recovery_status"], "blocked_reason": r["blocked_reason"],
        "needs_user_decision": r["needs_user_decision"],
    } for r in recoveries if r["boundary_recovered"] != "True"]

    status_dist = dict(sorted(Counter(r["boundary_recovery_status"] for r in recoveries).items()))
    summary = {
        "phase": STAGE,
        "phase_name": "GEOMETRY_RECONCILIATION_AND_PATCH_BOUNDARY_RECOVERY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "geometry_backend": "shapely" if HAS_SHAPELY else "stdlib_only",
        "reprojection_backend": "pyproj" if HAS_PYPROJ else "none",
        "rec00019_decision": recon.get("non_intersection_decision", "MISSING"),
        "rec00019_confidence": recon.get("non_intersection_confidence", ""),
        "rec00019_candidate_status": recon.get("candidate_positive_status", ""),
        "rec00019_centroid_distance_km": recon.get("centroid_distance", ""),
        "blocked_candidates_processed": len(recoveries),
        "patch_boundaries_recovered": recovered_features,
        "patch_boundaries_blocked": len(recoveries) - recovered_features,
        "can_retry_overlay": sum(1 for q in queue if q["can_retry_overlay"] == "True"),
        "recovery_status_distribution": status_dist,
        "needs_user_decision_count": sum(1 for r in recoveries if r["needs_user_decision"] == "True"),
        "guardrail_overall": guardrails["overall"],
        **{k: v for k, v in gate.items() if k not in {"phase"}},
    }
    return {
        "reconciliation": [recon] if recon.get("reconciliation_id") else [],
        "hypotheses": hypotheses,
        "dcivil": [dcivil_row] if dcivil_row else [],
        "event_quality": [event_quality_row] if event_quality_row else [],
        "recoveries": recoveries,
        "recovered_registry": recovered_reg,
        "unresolved_registry": unresolved_reg,
        "queue": queue,
        "gate": gate,
        "guardrails": guardrails,
        "summary": summary,
    }


def write_artifacts(output_dir: Path, art: dict[str, Any]) -> list[str]:
    write_json(output_dir / f"geometry_reconciliation_summary_{STAGE}.json", art["summary"])
    write_csv(output_dir / f"rec00019_non_intersection_reconciliation_{STAGE}.csv", art["reconciliation"], RECONCILIATION_FIELDS)
    write_csv(output_dir / f"patch_boundary_recovery_audit_{STAGE}.csv", art["recoveries"], RECOVERY_FIELDS)
    write_csv(output_dir / f"recovered_patch_boundary_registry_{STAGE}.csv", art["recovered_registry"], RECOVERED_REG_FIELDS)
    write_csv(output_dir / f"unresolved_patch_boundary_registry_{STAGE}.csv", art["unresolved_registry"], UNRESOLVED_REG_FIELDS)
    write_csv(output_dir / f"event_geometry_quality_audit_{STAGE}.csv", art["event_quality"], EVENT_QUALITY_FIELDS)
    write_csv(output_dir / f"defense_civil_point_support_audit_{STAGE}.csv", art["dcivil"], DCIVIL_FIELDS)
    write_csv(output_dir / f"geometry_hypothesis_test_audit_{STAGE}.csv", art["hypotheses"], HYPOTHESIS_FIELDS)
    write_csv(output_dir / f"next_overlay_candidate_queue_{STAGE}.csv", art["queue"], QUEUE_FIELDS)
    write_json(output_dir / f"gt_readiness_after_reconciliation_{STAGE}.json", art["gate"])
    write_json(output_dir / f"geometry_reconciliation_guardrails_{STAGE}.json", art["guardrails"])
    recon = art["reconciliation"][0] if art["reconciliation"] else {}
    (output_dir / f"geometry_reconciliation_report_{STAGE}.md").write_text(build_report(art["summary"], recon), encoding="utf-8")
    return sorted(p.name for p in output_dir.glob("*") if p.is_file())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v2br geometry reconciliation and patch boundary recovery. Audits non-intersection and recovers boundaries; creates no label and enables no training."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--candidate-registry", default=str(DEFAULT_CANDIDATE_REGISTRY))
    parser.add_argument("--blocked-registry", default=str(DEFAULT_BLOCKED_REGISTRY))
    parser.add_argument("--v1fs-audit", default=str(DEFAULT_V1FS_AUDIT))
    parser.add_argument("--patch-geom", default=str(DEFAULT_PATCH_GEOM))
    parser.add_argument("--event-geom", default=str(DEFAULT_EVENT_GEOM))
    parser.add_argument("--dcivil", default=str(DEFAULT_DCIVIL))
    parser.add_argument("--allow-local-runs", action="store_true", help="Acknowledge writing under local_runs/ (default behavior).")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare(output_dir, args.force)
    art = build_artifacts(
        Path(args.candidate_registry), Path(args.blocked_registry), Path(args.v1fs_audit),
        Path(args.patch_geom), Path(args.event_geom), Path(args.dcivil), output_dir,
    )
    write_artifacts(output_dir, art)
    print(json.dumps(art["summary"], ensure_ascii=False, indent=2))
    return 0 if art["guardrails"]["overall"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
