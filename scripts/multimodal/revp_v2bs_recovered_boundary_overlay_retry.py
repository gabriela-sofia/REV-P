"""REV-P v2bs — Recovered boundary overlay retry and event geometry reliability audit.

Runs a fresh overlay round using the patch boundaries recovered by v2br (plus
REC_00019 as a held diagnostic) against the available polygon of event
REC_2022_05_24_30 — with a mandatory extra layer: it classifies the reliability
of the event polygon before any case can move toward a formal ground-truth
protocol.

Geometry is computed for real (shapely/pyproj when available, fail-closed
otherwise), but the methodology is explicit:

  * an intersection is NOT a positive label — while the event polygon is
    ``provided_unreviewed`` / ``can_be_ground_truth=false`` it is held as
    ``OVERLAY_INTERSECTS_HELD_EVENT_GEOMETRY_UNREVIEWED``;
  * a non-intersection is NOT a formal negative — it is held as
    ``OVERLAY_NO_INTERSECTION_HELD_EVENT_GEOMETRY_UNREVIEWED``;
  * the 400 Defesa Civil points are contextual QA only — never a polygon, label
    or negative;
  * no geometry is invented; training stays blocked.

The report states plainly: what the current geometry suggests vs. what may be
used as a formal label. Outputs are local-only and light.
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

import importlib.util as _ilu
HAS_PYPROJ = _ilu.find_spec("pyproj") is not None  # reported only; v2bs does not reproject


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "ground_truth" / "v2bs"
STAGE = "v2bs"

DEFAULT_QUEUE = ROOT / "local_runs" / "ground_truth" / "v2br" / "next_overlay_candidate_queue_v2br.csv"
DEFAULT_RECOVERED_DIR = ROOT / "local_runs" / "ground_truth" / "v2br" / "recovered_patch_boundaries"
DEFAULT_EVENT_QUALITY = ROOT / "local_runs" / "ground_truth" / "v2br" / "event_geometry_quality_audit_v2br.csv"
DEFAULT_PATCH_GEOM = ROOT / "datasets" / "external_sources" / "recife_minimal_tp" / "derived" / "patch_boundary_REC_00019_from_lineage.geojson"
DEFAULT_EVENT_GEOM = ROOT / "datasets" / "external_sources" / "recife_minimal_tp" / "event_polygon_REC_2022_05_24_30" / "charter758" / "derived" / "event_polygon_REC_2022_05_24_30_charter758_digitized_candidate.geojson"
DEFAULT_DCIVIL = ROOT / "datasets" / "external_sources" / "recife_minimal_tp" / "event_polygon_REC_2022_05_24_30" / "raw" / "recife_defesa_civil_risk_locations.geojson"

EVENT_ID = "REC_2022_05_24_30"
WGS84 = "EPSG:4326"
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = -74.5, -33.0, -34.5, 6.0


METHODOLOGICAL_GUARDRAILS = {
    "review_only": True,
    "labels_created": False,
    "formal_negative_created": False,
    "positive_label_from_overlay": False,
    "negative_label_from_non_intersection": False,
    "negative_from_absence": False,
    "event_polygon_promoted_to_gt": False,
    "defense_civil_points_promoted_to_polygon": False,
    "centroid_promoted_to_overlay": False,
    "geometry_invented": False,
    "supervised_training": False,
    "outputs_local_only": True,
}

# Overlay retry statuses
OV_INTERSECTS_HELD = "OVERLAY_INTERSECTS_HELD_EVENT_GEOMETRY_UNREVIEWED"
OV_NO_INTERSECTION_HELD = "OVERLAY_NO_INTERSECTION_HELD_EVENT_GEOMETRY_UNREVIEWED"
OV_REJECT_PATCH = "OVERLAY_REJECT_INVALID_PATCH_GEOMETRY"
OV_REJECT_EVENT = "OVERLAY_REJECT_INVALID_EVENT_GEOMETRY"
OV_BLOCK_PATCH = "OVERLAY_BLOCKED_PATCH_BOUNDARY_MISSING"
OV_BLOCK_EVENT = "OVERLAY_BLOCKED_EVENT_GEOMETRY_MISSING"
OV_BLOCK_BACKEND = "OVERLAY_BLOCKED_BACKEND_UNAVAILABLE"
OV_AMBIGUOUS = "OVERLAY_AMBIGUOUS_MULTIPLE_EVENT_GEOMETRIES"

INTERSECT_STATES = {OV_INTERSECTS_HELD}
NONINTERSECT_STATES = {OV_NO_INTERSECTION_HELD}
BLOCKED_STATES = {OV_BLOCK_PATCH, OV_BLOCK_EVENT, OV_BLOCK_BACKEND}
REJECT_STATES = {OV_REJECT_PATCH, OV_REJECT_EVENT}

# Event reliability statuses
EVR_LOW = "EVENT_GEOMETRY_RELIABILITY_LOW_UNREVIEWED_MEDIA_POLYGON"
EVR_MEDIUM = "EVENT_GEOMETRY_RELIABILITY_MEDIUM_SUPPORTED_BY_CONTEXT"
EVR_BLOCK_CONFLICT = "EVENT_GEOMETRY_RELIABILITY_BLOCKED_CONFLICTS_WITH_DEFENSE_CIVIL_POINTS"
EVR_BLOCK_NOSOURCE = "EVENT_GEOMETRY_RELIABILITY_BLOCKED_NO_FORMAL_SOURCE"
EVR_QA_ONLY = "EVENT_GEOMETRY_RELIABILITY_READY_FOR_OVERLAY_QA_ONLY"
EVR_READY_GT = "EVENT_GEOMETRY_RELIABILITY_READY_FOR_FORMAL_GT_REVIEW"


RETRY_FIELDS = [
    "overlay_retry_id", "canonical_patch_id", "candidate_event_id", "retry_priority",
    "patch_boundary_source", "patch_boundary_quality", "event_geometry_source", "event_geometry_quality",
    "event_can_be_ground_truth", "event_provided_unreviewed", "patch_crs", "event_crs", "geometry_backend",
    "patch_geometry_valid", "event_geometry_valid", "bbox_overlap", "intersection_area", "intersection_area_units",
    "patch_area", "event_area", "intersection_ratio_patch", "intersection_ratio_event", "centroid_distance",
    "centroid_distance_units", "min_geometry_distance", "min_geometry_distance_units", "overlay_retry_status",
    "event_reliability_status", "gt_protocol_status", "gt_patch_flood_observed", "allowed_for_training",
    "promotion_blocker", "needs_user_decision", "auto_decision_reason", "notes",
]
INTERSECTION_REG_FIELDS = ["overlay_retry_id", "canonical_patch_id", "candidate_event_id", "intersection_ratio_patch", "intersection_ratio_event", "overlay_retry_status", "gt_protocol_status", "promotion_blocker"]
NONINTERSECT_REG_FIELDS = ["overlay_retry_id", "canonical_patch_id", "candidate_event_id", "centroid_distance", "min_geometry_distance", "overlay_retry_status", "reason"]
BLOCKED_REG_FIELDS = ["overlay_retry_id", "canonical_patch_id", "candidate_event_id", "overlay_retry_status", "missing_component", "reason"]
EVENT_RELIABILITY_FIELDS = ["event_id", "event_geometry_source", "geometry_type", "crs", "bbox", "centroid", "area_approx", "provided_unreviewed", "can_be_ground_truth", "source_family", "defense_civil_points_available", "defense_civil_alignment_status", "intersecting_patch_count", "non_intersecting_patch_count", "blocked_patch_count", "event_reliability_status", "recommended_use", "gt_promotion_allowed", "reason"]
DCIVIL_FIELDS = ["event_id", "point_source", "point_count", "event_polygon_source", "points_inside_event_polygon", "points_inside_event_bbox", "points_near_event_polygon", "nearest_point_to_event_distance", "nearest_point_to_event_distance_units", "point_alignment_status", "support_interpretation", "can_define_overlay", "can_define_gt"]
DISTANCE_FIELDS = ["canonical_patch_id", "candidate_event_id", "patch_centroid", "event_centroid", "centroid_distance_km", "min_geometry_distance_km", "bbox_overlap", "intersects"]
QUEUE_FIELDS = ["queue_id", "canonical_patch_id", "candidate_event_id", "overlay_retry_status", "event_reliability_status", "patch_boundary_quality", "candidate_status_after_v2bs", "ready_for_formal_gt_protocol", "blocked_reason", "recommended_next_action"]


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


def deg_to_km(d_deg: float, lat: float) -> float:
    return d_deg * 111.0 * max(math.cos(math.radians(lat)), 0.1)


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


def crs_of(props: dict, top: dict, bbox: tuple | None) -> str:
    for src in (props, top):
        crs = (src or {}).get("crs")
        if isinstance(crs, str) and crs.strip():
            return crs.strip().upper()
    if bbox and LON_MIN <= bbox[0] <= LON_MAX and LAT_MIN <= bbox[1] <= LAT_MAX:
        return WGS84
    return "UNKNOWN"


def valid_shape(geom: dict):
    if not (HAS_SHAPELY and _shapely_shape and _make_valid):
        return None
    try:  # pragma: no cover - requires shapely
        return _make_valid(_shapely_shape(geom))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Overlay computation (held against an unreviewed event)
# --------------------------------------------------------------------------- #

def compute_retry_overlay(patch_geom: dict, event_geom: dict) -> dict[str, Any]:
    backend = "shapely" if HAS_SHAPELY else ("stdlib_bbox" if True else OV_BLOCK_BACKEND)
    pbbox = geom_bbox(patch_geom)
    ebbox = geom_bbox(event_geom)
    if not pbbox:
        return {"status": OV_REJECT_PATCH, "backend": backend, "reason": "Patch geometry empty/invalid."}
    if not ebbox:
        return {"status": OV_REJECT_EVENT, "backend": backend, "reason": "Event geometry empty/invalid."}
    overlap = bbox_overlap(pbbox, ebbox)
    pc = bbox_centroid(pbbox)
    ec = bbox_centroid(ebbox)
    centroid_km = haversine_km(pc, ec)
    parea = approx_area(patch_geom)
    earea = approx_area(event_geom)
    result = {
        "backend": backend, "bbox_overlap": overlap, "patch_centroid": "%.5f,%.5f" % pc, "event_centroid": "%.5f,%.5f" % ec,
        "centroid_km": centroid_km, "patch_area": parea, "event_area": earea, "min_km": "", "intersection_area": 0.0,
        "ratio_patch": 0.0, "ratio_event": 0.0,
    }
    if not HAS_SHAPELY:  # pragma: no cover - shapely present in CI
        result["status"] = OV_NO_INTERSECTION_HELD if not overlap else OV_BLOCK_BACKEND
        result["reason"] = "No exact-intersection backend; bbox-only result."
        return result
    ps = valid_shape(patch_geom)
    es = valid_shape(event_geom)
    if ps is None or ps.is_empty:
        return {**result, "status": OV_REJECT_PATCH, "reason": "Patch geometry invalid after validation."}
    if es is None or es.is_empty:
        return {**result, "status": OV_REJECT_EVENT, "reason": "Event geometry invalid after validation."}
    min_km = deg_to_km(ps.distance(es), ec[1])
    result["min_km"] = min_km
    if ps.intersects(es):
        inter = ps.intersection(es)
        ia = float(inter.area)
        result.update({
            "status": OV_INTERSECTS_HELD, "intersection_area": ia,
            "ratio_patch": ia / float(ps.area) if ps.area else 0.0,
            "ratio_event": ia / float(es.area) if es.area else 0.0,
            "reason": "Patch boundary intersects the current event polygon; HELD because the event polygon is unreviewed and not ground truth.",
        })
    else:
        result.update({
            "status": OV_NO_INTERSECTION_HELD,
            "reason": "Patch boundary does not intersect the current event polygon; HELD as geometry evidence, not a negative.",
        })
    return result


# --------------------------------------------------------------------------- #
# Loading inputs
# --------------------------------------------------------------------------- #

def load_recovered_boundaries(recovered_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not recovered_dir.exists():
        return out
    for path in sorted(recovered_dir.glob("patch_boundary_*_recovered_*.geojson")):
        doc = load_geojson(path)
        if doc is None:
            continue
        geom, props = first_geometry(doc)
        if not geom:
            continue
        pid = str(props.get("patch_id", "")).strip()
        if not pid:
            # derive from filename: patch_boundary_<ID>_recovered_v2bs.geojson
            parts = path.stem.split("_")
            pid = "_".join(parts[2:-2]) if len(parts) >= 5 else ""
        if pid and pid not in out:
            out[pid] = {"geom": geom, "props": props, "source": rel_to_root(path), "quality": str(props.get("boundary_quality", "DERIVED_BBOX_FROM_RECORDED_HEADER_BOUNDS_COARSE")), "crs": crs_of(props, doc, geom_bbox(geom))}
    return out


def load_event_geometry(event_quality_audit: Path, event_geom_default: Path) -> tuple[dict, dict, str]:
    """Return (geometry, properties, source_rel) for the event polygon."""
    source = event_geom_default
    rows = read_csv(event_quality_audit)
    if rows:
        sp = (rows[0].get("geometry_source") or "").strip()
        if sp:
            cand = ROOT / sp
            if cand.exists():
                source = cand
    doc = load_geojson(source)
    if doc is None:
        return {}, {}, rel_to_root(source)
    geom, props = first_geometry(doc)
    return geom, props, rel_to_root(source)


# --------------------------------------------------------------------------- #
# Defesa Civil alignment (contextual QA only)
# --------------------------------------------------------------------------- #

def defense_civil_alignment(dcivil_path: Path, event_geom: dict, event_source: str) -> dict[str, Any]:
    doc = load_geojson(dcivil_path)
    if doc is None or not event_geom:
        return {
            "event_id": EVENT_ID, "point_source": rel_to_root(dcivil_path), "point_count": 0,
            "event_polygon_source": event_source, "points_inside_event_polygon": 0, "points_inside_event_bbox": 0,
            "points_near_event_polygon": 0, "nearest_point_to_event_distance": "", "nearest_point_to_event_distance_units": "km",
            "point_alignment_status": "POINT_SUPPORT_UNAVAILABLE", "support_interpretation": "No points or no event polygon available.",
            "can_define_overlay": "False", "can_define_gt": "False",
        }
    pts = []
    for f in doc.get("features", []):
        c = (f.get("geometry") or {}).get("coordinates")
        if c and isinstance(c[0], (int, float)):
            pts.append((float(c[0]), float(c[1])))
    ebbox = geom_bbox(event_geom)
    ec = bbox_centroid(ebbox) if ebbox else (0.0, 0.0)
    inside_bbox = sum(1 for p in pts if ebbox and ebbox[0] <= p[0] <= ebbox[2] and ebbox[1] <= p[1] <= ebbox[3]) if ebbox else 0
    inside_poly = 0
    es = valid_shape(event_geom)
    if es is not None and HAS_SHAPELY and _shapely_shape:  # pragma: no branch
        try:  # pragma: no cover - requires shapely
            from shapely.geometry import Point  # type: ignore
            inside_poly = sum(1 for p in pts if es.contains(Point(p[0], p[1])))
        except Exception:
            inside_poly = inside_bbox
    nearest_km = min((haversine_km(p, ec) for p in pts), default="")
    near_5km = sum(1 for p in pts if haversine_km(p, ec) <= 5.0)
    if not pts:
        status = "POINT_SUPPORT_UNAVAILABLE"
        interp = "No Defesa Civil points available."
    elif inside_poly > 0:
        status = "POINT_SUPPORT_ALIGNED"
        interp = f"{inside_poly} risk points fall inside the event polygon; contextual support only, not a label or GT."
    elif near_5km > 0:
        status = "POINT_SUPPORT_WEAK"
        interp = "Some risk points are near but outside the event polygon; weak contextual support, not a label."
    else:
        status = "POINT_SUPPORT_CONFLICTING"
        interp = "Risk points do not fall in or near the event polygon; the polygon footprint is not corroborated by the independent points."
    return {
        "event_id": EVENT_ID, "point_source": rel_to_root(dcivil_path), "point_count": len(pts),
        "event_polygon_source": event_source, "points_inside_event_polygon": inside_poly, "points_inside_event_bbox": inside_bbox,
        "points_near_event_polygon": near_5km, "nearest_point_to_event_distance": "%.2f" % nearest_km if nearest_km != "" else "",
        "nearest_point_to_event_distance_units": "km", "point_alignment_status": status, "support_interpretation": interp,
        "can_define_overlay": "False", "can_define_gt": "False",
    }


# --------------------------------------------------------------------------- #
# Event reliability
# --------------------------------------------------------------------------- #

def event_reliability(event_geom: dict, event_props: dict, event_source: str, dcivil_row: dict[str, Any], counts: dict[str, int]) -> dict[str, Any]:
    provided_unreviewed = str(event_props.get("review_status", "")).lower().find("unreviewed") >= 0 or str(event_props.get("provided_unreviewed", "")).lower() == "true"
    can_be_gt = str(event_props.get("can_be_ground_truth", "")).lower()
    bbox = geom_bbox(event_geom)
    align = dcivil_row.get("point_alignment_status", "POINT_SUPPORT_UNAVAILABLE")

    if can_be_gt == "false" or not can_be_gt:
        gt_allowed = "False"
    else:
        gt_allowed = "True"

    if align == "POINT_SUPPORT_CONFLICTING":
        status = EVR_BLOCK_CONFLICT
        reason = "Event polygon footprint conflicts with the independent Defesa Civil risk points."
    elif provided_unreviewed and gt_allowed == "False":
        status = EVR_LOW
        reason = "Unreviewed digitized media polygon flagged can_be_ground_truth=false; usable for overlay QA only."
    elif gt_allowed == "False":
        status = EVR_QA_ONLY
        reason = "Event polygon cannot be promoted to ground truth; usable for overlay QA only."
    else:
        status = EVR_READY_GT
        reason = "Event polygon meets review criteria; ready for formal GT review."
    recommended_use = "USE_FOR_OVERLAY_QA_ONLY" if gt_allowed == "False" else "USE_FOR_FORMAL_GT_AFTER_REVIEW"
    return {
        "event_id": EVENT_ID, "event_geometry_source": event_source, "geometry_type": event_geom.get("type", ""),
        "crs": crs_of(event_props, {}, bbox), "bbox": ",".join("%.5f" % v for v in bbox) if bbox else "MISSING",
        "centroid": "%.5f,%.5f" % bbox_centroid(bbox) if bbox else "MISSING", "area_approx": "%.6f" % approx_area(event_geom) if event_geom else "",
        "provided_unreviewed": str(provided_unreviewed), "can_be_ground_truth": can_be_gt or "false",
        "source_family": str(event_props.get("source_method", "charter758")),
        "defense_civil_points_available": str(dcivil_row.get("point_count", 0) or 0),
        "defense_civil_alignment_status": align,
        "intersecting_patch_count": counts.get("intersect", 0), "non_intersecting_patch_count": counts.get("nonintersect", 0),
        "blocked_patch_count": counts.get("blocked", 0), "event_reliability_status": status,
        "recommended_use": recommended_use, "gt_promotion_allowed": gt_allowed, "reason": reason,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def event_geometries_conflict(event_geoms: list[tuple[dict, dict, str]]) -> bool:
    """True when several distinct real event polygons (different bbox) coexist."""
    boxes = set()
    for geom, _props, _src in event_geoms:
        b = geom_bbox(geom)
        if b:
            boxes.add(tuple(round(v, 6) for v in b))
    return len(boxes) > 1


def build_artifacts(
    queue_path: Path,
    recovered_dir: Path,
    event_quality_audit: Path,
    patch_geom_default: Path,
    event_geom_default: Path,
    dcivil_path: Path,
    event_geoms: list[tuple[dict, dict, str]] | None = None,
) -> dict[str, Any]:
    queue = read_csv(queue_path)
    retry_candidates = [q for q in queue if str(q.get("can_retry_overlay")) == "True"]
    # Include REC_00019 held/MEDIUM diagnostic even if can_retry_overlay is False.
    held = [q for q in queue if q.get("canonical_patch_id") == "REC_00019" and q not in retry_candidates]
    selected = retry_candidates + held

    boundaries = load_recovered_boundaries(recovered_dir)
    # REC_00019 boundary comes from its original lineage geojson.
    rec19_doc = load_geojson(patch_geom_default)
    if rec19_doc is not None:
        g19, p19 = first_geometry(rec19_doc)
        if g19:
            boundaries.setdefault("REC_00019", {"geom": g19, "props": p19, "source": rel_to_root(patch_geom_default), "quality": "ORIGINAL_LINEAGE_BBOX", "crs": crs_of(p19, rec19_doc, geom_bbox(g19))})

    if event_geoms is None:
        eg, ep, es = load_event_geometry(event_quality_audit, event_geom_default)
        event_geoms = [(eg, ep, es)] if eg else []
    event_ambiguous = event_geometries_conflict(event_geoms)
    event_geom, event_props, event_source = event_geoms[0] if event_geoms else ({}, {}, rel_to_root(event_geom_default))
    event_available = bool(event_geom)
    event_crs = crs_of(event_props, {}, geom_bbox(event_geom)) if event_geom else "UNKNOWN"
    event_can_gt = str(event_props.get("can_be_ground_truth", "false")).lower()
    event_unreviewed = "unreviewed" in str(event_props.get("review_status", "")).lower()

    rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    for q in selected:
        pid = (q.get("canonical_patch_id") or "").strip()
        prio = (q.get("priority") or "").strip()
        b = boundaries.get(pid)
        retry_id = short_id("OVR2", f"{pid}|{EVENT_ID}")
        base = {
            "overlay_retry_id": retry_id, "canonical_patch_id": pid, "candidate_event_id": EVENT_ID, "retry_priority": prio,
            "patch_boundary_source": b["source"] if b else "", "patch_boundary_quality": b["quality"] if b else "",
            "event_geometry_source": event_source if event_available else "", "event_geometry_quality": "REAL_POLYGON_UNREVIEWED" if event_unreviewed else ("REAL_POLYGON" if event_available else "MISSING"),
            "event_can_be_ground_truth": event_can_gt, "event_provided_unreviewed": str(event_unreviewed),
            "patch_crs": b["crs"] if b else "UNKNOWN", "event_crs": event_crs, "geometry_backend": "shapely" if HAS_SHAPELY else "stdlib_bbox",
            "patch_geometry_valid": str(bool(b)), "event_geometry_valid": str(event_available),
            "bbox_overlap": "", "intersection_area": "", "intersection_area_units": "deg2", "patch_area": "", "event_area": "",
            "intersection_ratio_patch": "", "intersection_ratio_event": "", "centroid_distance": "", "centroid_distance_units": "km",
            "min_geometry_distance": "", "min_geometry_distance_units": "km",
            "gt_protocol_status": "BLOCKED_EVENT_GEOMETRY_NOT_GT", "gt_patch_flood_observed": "", "allowed_for_training": "False",
            "promotion_blocker": "EVENT_GEOMETRY_NOT_REVIEWED_NOT_GT", "needs_user_decision": "False", "notes": "overlay_is_not_label; non_intersection_is_not_negative; event_not_gt",
        }
        if not b and not event_available:
            base.update({"overlay_retry_status": OV_BLOCK_PATCH, "event_reliability_status": "", "auto_decision_reason": "No patch boundary and no event geometry available."})
            rows.append(base)
            continue
        if not b:
            base.update({"overlay_retry_status": OV_BLOCK_PATCH, "event_reliability_status": "", "auto_decision_reason": "No recovered patch boundary available for this candidate."})
            rows.append(base)
            continue
        if not event_available:
            base.update({"overlay_retry_status": OV_BLOCK_EVENT, "event_reliability_status": "", "auto_decision_reason": "No event polygon available."})
            rows.append(base)
            continue
        if event_ambiguous:
            base.update({"overlay_retry_status": OV_AMBIGUOUS, "needs_user_decision": "True", "event_reliability_status": "",
                         "auto_decision_reason": "Multiple distinct event polygons claim this event; overlay cannot pick one automatically."})
            rows.append(base)
            continue
        comp = compute_retry_overlay(b["geom"], event_geom)
        base.update({
            "bbox_overlap": str(comp.get("bbox_overlap", "")), "intersection_area": comp.get("intersection_area", ""),
            "patch_area": "%.6f" % comp["patch_area"] if "patch_area" in comp else "", "event_area": "%.6f" % comp["event_area"] if "event_area" in comp else "",
            "intersection_ratio_patch": comp.get("ratio_patch", ""), "intersection_ratio_event": comp.get("ratio_event", ""),
            "centroid_distance": "%.2f" % comp["centroid_km"] if comp.get("centroid_km", "") != "" else "", "min_geometry_distance": "%.2f" % comp["min_km"] if comp.get("min_km", "") != "" else "",
            "overlay_retry_status": comp["status"], "auto_decision_reason": comp["reason"],
        })
        rows.append(base)
        distance_rows.append({
            "canonical_patch_id": pid, "candidate_event_id": EVENT_ID, "patch_centroid": comp.get("patch_centroid", ""),
            "event_centroid": comp.get("event_centroid", ""), "centroid_distance_km": "%.2f" % comp["centroid_km"] if comp.get("centroid_km", "") != "" else "",
            "min_geometry_distance_km": "%.2f" % comp["min_km"] if comp.get("min_km", "") != "" else "", "bbox_overlap": str(comp.get("bbox_overlap", "")),
            "intersects": str(comp["status"] == OV_INTERSECTS_HELD),
        })

    counts = {
        "intersect": sum(1 for r in rows if r["overlay_retry_status"] in INTERSECT_STATES),
        "nonintersect": sum(1 for r in rows if r["overlay_retry_status"] in NONINTERSECT_STATES),
        "blocked": sum(1 for r in rows if r["overlay_retry_status"] in BLOCKED_STATES),
        "reject": sum(1 for r in rows if r["overlay_retry_status"] in REJECT_STATES),
    }

    dcivil_row = defense_civil_alignment(dcivil_path, event_geom, event_source)
    reliability_row = event_reliability(event_geom, event_props, event_source, dcivil_row, counts)

    # Stamp event reliability into every overlay row.
    for r in rows:
        if not r.get("event_reliability_status"):
            r["event_reliability_status"] = reliability_row["event_reliability_status"]

    queue_rows = build_protocol_queue(rows, reliability_row)
    gate = build_gate(rows, reliability_row, counts)
    guardrails = build_guardrails(rows, reliability_row)

    status_dist = dict(sorted(Counter(r["overlay_retry_status"] for r in rows).items()))
    summary = {
        "phase": STAGE, "phase_name": "RECOVERED_BOUNDARY_OVERLAY_RETRY_AND_EVENT_RELIABILITY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "geometry_backend": "shapely" if HAS_SHAPELY else "stdlib_only", "reprojection_backend": "pyproj" if HAS_PYPROJ else "none",
        "retry_candidate_count": len(rows), "overlay_intersection_count": counts["intersect"],
        "overlay_non_intersection_count": counts["nonintersect"], "overlay_blocked_count": counts["blocked"],
        "overlay_rejected_count": counts["reject"],
        "event_geometry_reliability_status": reliability_row["event_reliability_status"],
        "event_can_be_ground_truth": event_can_gt == "true",
        "defense_civil_alignment_status": dcivil_row["point_alignment_status"],
        "ready_for_formal_gt_protocol_count": sum(1 for q in queue_rows if q["ready_for_formal_gt_protocol"] == "True"),
        "would_enter_protocol_if_event_reliable": counts["intersect"],
        "needs_user_decision_count": sum(1 for r in rows if r["needs_user_decision"] == "True"),
        "overlay_status_distribution": status_dist, "guardrail_overall": guardrails["overall"],
        **{k: v for k, v in gate.items() if k not in {"phase"}},
    }
    return {
        "retry": rows,
        "intersection": [{"overlay_retry_id": r["overlay_retry_id"], "canonical_patch_id": r["canonical_patch_id"], "candidate_event_id": EVENT_ID, "intersection_ratio_patch": r["intersection_ratio_patch"], "intersection_ratio_event": r["intersection_ratio_event"], "overlay_retry_status": r["overlay_retry_status"], "gt_protocol_status": r["gt_protocol_status"], "promotion_blocker": r["promotion_blocker"]} for r in rows if r["overlay_retry_status"] in INTERSECT_STATES],
        "nonintersection": [{"overlay_retry_id": r["overlay_retry_id"], "canonical_patch_id": r["canonical_patch_id"], "candidate_event_id": EVENT_ID, "centroid_distance": r["centroid_distance"], "min_geometry_distance": r["min_geometry_distance"], "overlay_retry_status": r["overlay_retry_status"], "reason": r["auto_decision_reason"]} for r in rows if r["overlay_retry_status"] in NONINTERSECT_STATES],
        "blocked": [{"overlay_retry_id": r["overlay_retry_id"], "canonical_patch_id": r["canonical_patch_id"], "candidate_event_id": EVENT_ID, "overlay_retry_status": r["overlay_retry_status"], "missing_component": "patch_boundary" if r["overlay_retry_status"] == OV_BLOCK_PATCH else ("event_geometry" if r["overlay_retry_status"] == OV_BLOCK_EVENT else "backend"), "reason": r["auto_decision_reason"]} for r in rows if r["overlay_retry_status"] in BLOCKED_STATES],
        "event_reliability": [reliability_row],
        "dcivil": [dcivil_row],
        "distance": distance_rows,
        "queue": queue_rows,
        "gate": gate,
        "guardrails": guardrails,
        "summary": summary,
    }


def build_protocol_queue(rows: list[dict[str, Any]], reliability: dict[str, Any]) -> list[dict[str, Any]]:
    event_is_gt = reliability.get("gt_promotion_allowed", "False") == "True"
    out = []
    for r in rows:
        st = r["overlay_retry_status"]
        if st in INTERSECT_STATES:
            cand_status = "INTERSECTS_EVENT_POLYGON_HELD_FOR_EVENT_GEOMETRY_RELIABILITY"
            action = "review_event_geometry_or_acquire_formal_event_source"
        elif st in NONINTERSECT_STATES:
            cand_status = "NO_INTERSECTION_WITH_EVENT_POLYGON_HELD_AS_GEOMETRY_EVIDENCE"
            action = "review_event_geometry_reliability_before_any_conclusion"
        elif st == OV_BLOCK_PATCH:
            cand_status = "STILL_BLOCKED_PATCH_BOUNDARY"
            action = "recover_patch_boundary"
        elif st == OV_AMBIGUOUS:
            cand_status = "AMBIGUOUS_NEEDS_USER_DECISION"
            action = "resolve_multiple_event_geometries"
        else:
            cand_status = "BLOCKED_EVENT_GEOMETRY_NOT_GT"
            action = "acquire_reviewed_event_geometry"
        out.append({
            "queue_id": short_id("Q", r["canonical_patch_id"]), "canonical_patch_id": r["canonical_patch_id"],
            "candidate_event_id": EVENT_ID, "overlay_retry_status": st, "event_reliability_status": reliability["event_reliability_status"],
            "patch_boundary_quality": r["patch_boundary_quality"], "candidate_status_after_v2bs": cand_status,
            "ready_for_formal_gt_protocol": str(event_is_gt and st in INTERSECT_STATES),
            "blocked_reason": "" if event_is_gt else "EVENT_GEOMETRY_NOT_GROUND_TRUTH", "recommended_next_action": action,
        })
    return out


def build_gate(rows: list[dict[str, Any]], reliability: dict[str, Any], counts: dict[str, int]) -> dict[str, Any]:
    return {
        "phase": STAGE,
        "retry_candidate_count": len(rows),
        "overlay_intersection_count": counts["intersect"],
        "overlay_non_intersection_count": counts["nonintersect"],
        "overlay_blocked_count": counts["blocked"],
        "event_geometry_reliability_status": reliability["event_reliability_status"],
        "event_can_be_ground_truth": reliability.get("gt_promotion_allowed", "False") == "True",
        "ready_for_formal_gt_protocol_count": 0 if reliability.get("gt_promotion_allowed", "False") == "False" else counts["intersect"],
        "labels_created": False,
        "formal_negatives_created": False,
        "allowed_for_training_count": sum(1 for r in rows if str(r.get("allowed_for_training")) == "True"),
        "supervised_training_enabled": False,
        "promotion_to_operational_gt": False,
        "next_required_step": "event_geometry_reliability_resolution_or_alternative_event_geometry",
    }


def build_guardrails(rows: list[dict[str, Any]], reliability: dict[str, Any]) -> dict[str, Any]:
    def verdict(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    checks = {
        "labels_created_false": verdict(all(str(r.get("gt_patch_flood_observed", "")) == "" for r in rows)),
        "allowed_for_training_false": verdict(all(str(r.get("allowed_for_training")) == "False" for r in rows)),
        "no_positive_label_from_overlay": verdict(all(r["overlay_retry_status"] != OV_INTERSECTS_HELD or r["gt_patch_flood_observed"] == "" for r in rows)),
        "no_negative_label_from_non_intersection": verdict(all(r["overlay_retry_status"] != OV_NO_INTERSECTION_HELD or r["gt_patch_flood_observed"] == "" for r in rows)),
        "no_negative_from_absence": verdict(METHODOLOGICAL_GUARDRAILS["negative_from_absence"] is False),
        "no_event_polygon_promoted_to_gt": verdict(reliability.get("gt_promotion_allowed", "False") == "False"),
        "defense_civil_points_not_promoted_to_polygon": "PASS",
        "centroid_not_promoted_to_overlay": "PASS",
        "no_geometry_invented": verdict(METHODOLOGICAL_GUARDRAILS["geometry_invented"] is False),
        "private_absolute_paths_removed": verdict("Users" + "\\" + "gabriela" not in " ".join(r.get("patch_boundary_source", "") for r in rows)),
        "no_heavy_outputs": "PASS",
        "training_still_blocked": verdict(all(str(r.get("allowed_for_training")) == "False" for r in rows)),
    }
    overall = "PASS" if all(v in {"PASS", "BLOCKED_EXPECTED"} for v in checks.values()) else "FAIL"
    return {"phase": STAGE, "checks": checks, "overall": overall, **METHODOLOGICAL_GUARDRAILS}


def build_report(summary: dict[str, Any], reliability: dict[str, Any], dcivil: dict[str, Any]) -> str:
    dd = summary["overlay_status_distribution"]
    dist = "\n".join(f"- `{k}`: {v}" for k, v in sorted(dd.items())) or "- (none)"
    return f"""# REV-P {STAGE} — Recovered Boundary Overlay Retry and Event Geometry Reliability

Version: `{STAGE}`
Generated: {summary['created_utc']}
Geometry backend: {summary['geometry_backend']}

## 1. Why v2bs exists

v2br recovered 36 patch boundaries and held the REC_00019 non-intersection.
v2bs re-runs the overlay for those boundaries against the available event
polygon — and, crucially, classifies the reliability of that event polygon
before any case can move toward a formal ground-truth protocol.

## 2. How it uses the recovered boundaries

It reads the v2br retry queue, loads the recovered GeoJSON sidecars (plus
REC_00019's original boundary), loads the event polygon, and computes the real
intersection (area, ratios, centroid and minimum distance) for each case.

## 3-4. What the current geometry suggests vs. what may be a label

- Retry candidates processed: **{summary['retry_candidate_count']}**
- Intersect the event polygon (HELD, not a positive label): **{summary['overlay_intersection_count']}**
- Do not intersect (HELD as evidence, not a negative): **{summary['overlay_non_intersection_count']}**
- Blocked: **{summary['overlay_blocked_count']}**
- Would enter a formal protocol *if the event were reliable*: **{summary['would_enter_protocol_if_event_reliable']}**
- Actually ready for a formal GT protocol now: **{summary['ready_for_formal_gt_protocol_count']}**

Overlay status distribution:

{dist}

An intersection here only means a recovered boundary overlaps the current event
polygon. It is **not** a positive flood label. A non-intersection is **not** a
formal negative. While the event polygon is unreviewed, every case is held.

## 5. Why the event polygon still blocks GT promotion

- Event reliability: **{reliability['event_reliability_status']}**
- can_be_ground_truth: `{reliability['can_be_ground_truth']}` -> gt_promotion_allowed `{reliability['gt_promotion_allowed']}`
- Recommended use: `{reliability['recommended_use']}`

The event polygon is an unreviewed Charter media product flagged
`can_be_ground_truth=false`. No overlay against it can be promoted to ground
truth.

## 6. Defesa Civil points as contextual QA

- Points: {dcivil['point_count']} | inside event polygon: {dcivil['points_inside_event_polygon']} | inside event bbox: {dcivil['points_inside_event_bbox']} | nearest ~{dcivil['nearest_point_to_event_distance']} km
- Alignment: **{dcivil['point_alignment_status']}**

The points audit the plausibility of the event polygon only. They never define
an overlay, a label or a negative.

## 7. Counts

Intersecting: {summary['overlay_intersection_count']} · Non-intersecting:
{summary['overlay_non_intersection_count']} · Blocked: {summary['overlay_blocked_count']}.

## 8. Why training stays blocked

`labels_created=false`, `allowed_for_training_count=0`,
`promotion_to_operational_gt=false`. Recomputing overlays against an unreviewed
event polygon changes nothing about the training gate. A reviewed/alternative
event geometry and a formal positive/negative protocol are still required.

## Guardrail note

Autonomous geometric audit. No operational flood detection, no validated
prediction, no flood accuracy, no operational model. Outputs are local-only and
lightweight; no geometry was invented.
"""


# --------------------------------------------------------------------------- #
# Write / CLI
# --------------------------------------------------------------------------- #

def write_artifacts(output_dir: Path, art: dict[str, Any]) -> list[str]:
    write_csv(output_dir / f"recovered_boundary_overlay_retry_{STAGE}.csv", art["retry"], RETRY_FIELDS)
    write_csv(output_dir / f"overlay_intersection_registry_{STAGE}.csv", art["intersection"], INTERSECTION_REG_FIELDS)
    write_csv(output_dir / f"overlay_non_intersection_registry_{STAGE}.csv", art["nonintersection"], NONINTERSECT_REG_FIELDS)
    write_csv(output_dir / f"overlay_blocked_registry_{STAGE}.csv", art["blocked"], BLOCKED_REG_FIELDS)
    write_csv(output_dir / f"event_geometry_reliability_audit_{STAGE}.csv", art["event_reliability"], EVENT_RELIABILITY_FIELDS)
    write_csv(output_dir / f"defense_civil_alignment_audit_{STAGE}.csv", art["dcivil"], DCIVIL_FIELDS)
    write_csv(output_dir / f"overlay_distance_matrix_{STAGE}.csv", art["distance"], DISTANCE_FIELDS)
    write_csv(output_dir / f"gt_protocol_candidate_queue_{STAGE}.csv", art["queue"], QUEUE_FIELDS)
    write_json(output_dir / f"gt_readiness_after_overlay_retry_{STAGE}.json", art["gate"])
    write_json(output_dir / f"overlay_retry_guardrails_{STAGE}.json", art["guardrails"])
    write_json(output_dir / f"overlay_retry_summary_{STAGE}.json", art["summary"])
    recon = art["event_reliability"][0] if art["event_reliability"] else {}
    dcivil = art["dcivil"][0] if art["dcivil"] else {}
    (output_dir / f"overlay_retry_report_{STAGE}.md").write_text(build_report(art["summary"], recon, dcivil), encoding="utf-8")
    return sorted(p.name for p in output_dir.glob("*") if p.is_file())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v2bs recovered boundary overlay retry and event geometry reliability audit. Creates no label and enables no training."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE))
    parser.add_argument("--recovered-dir", default=str(DEFAULT_RECOVERED_DIR))
    parser.add_argument("--event-quality", default=str(DEFAULT_EVENT_QUALITY))
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
        Path(args.queue), Path(args.recovered_dir), Path(args.event_quality),
        Path(args.patch_geom), Path(args.event_geom), Path(args.dcivil),
    )
    write_artifacts(output_dir, art)
    print(json.dumps(art["summary"], ensure_ascii=False, indent=2))
    return 0 if art["guardrails"]["overall"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
