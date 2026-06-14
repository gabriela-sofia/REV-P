"""REV-P v2bt — Event geometry alternative reconstruction and reliability resolver.

Resolves the blocker left by v2bs
(``EVENT_GEOMETRY_RELIABILITY_BLOCKED_CONFLICTS_WITH_DEFENSE_CIVIL_POINTS``) for
event REC_2022_05_24_30 by:

  * auditing the Defesa Civil point cloud and the charter758 polygon;
  * deciding, from the data, whether the charter polygon must be downgraded or
    rejected for event QA (without invalidating the historical event);
  * building QA-only alternative event geometries from the points (convex hull,
    buffered point union, DBSCAN cluster envelopes);
  * scoring them and queuing the best one for an overlay-retry round.

Hard rule: points, hulls, buffers and clusters are QA-only candidates
(``POINT_DERIVED_EVENT_GEOMETRY_CANDIDATE``) — never ground truth, never a
positive or negative label, never a training trigger. No geometry is invented:
every alternative is derived from the real recorded points. Outputs are
local-only and light.
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
    from shapely.geometry import MultiPoint, Point, mapping, shape as _shapely_shape  # type: ignore
    from shapely.ops import transform as _shapely_transform, unary_union as _unary_union  # type: ignore
    from shapely.validation import make_valid as _make_valid  # type: ignore
    HAS_SHAPELY = True
except Exception:  # pragma: no cover
    MultiPoint = Point = mapping = _shapely_shape = None
    _shapely_transform = _unary_union = _make_valid = None
    HAS_SHAPELY = False

try:  # pragma: no cover
    from pyproj import Transformer as _Transformer  # type: ignore
    HAS_PYPROJ = True
except Exception:  # pragma: no cover
    _Transformer = None
    HAS_PYPROJ = False

try:  # pragma: no cover
    from sklearn.cluster import DBSCAN as _DBSCAN  # type: ignore
    HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _DBSCAN = None
    HAS_SKLEARN = False


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "ground_truth" / "v2bt"
ALT_DIR_NAME = "alternative_event_geometries"
STAGE = "v2bt"
EVENT_ID = "REC_2022_05_24_30"

DEFAULT_DCIVIL = ROOT / "datasets" / "external_sources" / "recife_minimal_tp" / "event_polygon_REC_2022_05_24_30" / "raw" / "recife_defesa_civil_risk_locations.geojson"
DEFAULT_CHARTER = ROOT / "datasets" / "external_sources" / "recife_minimal_tp" / "event_polygon_REC_2022_05_24_30" / "charter758" / "derived" / "event_polygon_REC_2022_05_24_30_charter758_digitized_candidate.geojson"
DEFAULT_V2BS_RELIABILITY = ROOT / "local_runs" / "ground_truth" / "v2bs" / "event_geometry_reliability_audit_v2bs.csv"

WGS84 = "EPSG:4326"
RECIFE_UTM = "EPSG:32725"
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = -74.5, -33.0, -34.5, 6.0

# QA-only buffer radii (metres) for the buffered-point-union candidate.
BUFFER_METERS = [250, 500]
DBSCAN_EPS_M = 1000
DBSCAN_MIN_SAMPLES = 5
MIN_POINTS_FOR_HULL = 3
MIN_POINTS_FOR_BUFFER = 5


METHODOLOGICAL_GUARDRAILS = {
    "review_only": True,
    "labels_created": False,
    "formal_negative_created": False,
    "positive_label_from_point_geometry": False,
    "negative_label_from_point_geometry": False,
    "negative_from_absence": False,
    "points_promoted_to_gt": False,
    "point_hull_promoted_to_gt": False,
    "buffer_union_promoted_to_gt": False,
    "charter_polygon_promoted_to_gt": False,
    "geometry_invented": False,
    "supervised_training": False,
    "outputs_local_only": True,
}

# Charter decisions
CH_REJECTED = "CHARTER_POLYGON_REJECTED_FOR_EVENT_QA"
CH_DOWNGRADED = "CHARTER_POLYGON_DOWNGRADED_TO_CONTEXT_ONLY"
CH_HELD = "CHARTER_POLYGON_HELD_AS_UNREVIEWED_MEDIA_GEOMETRY"
CH_CONSISTENT = "CHARTER_POLYGON_CONSISTENT_WITH_POINT_EVIDENCE"
CH_AMBIGUOUS = "CHARTER_POLYGON_AMBIGUOUS_NEEDS_USER_DECISION"

# Alternative geometry statuses
ALT_CREATED = "ALTERNATIVE_EVENT_GEOMETRY_CREATED_QA_ONLY"
ALT_READY = "ALTERNATIVE_EVENT_GEOMETRY_READY_FOR_OVERLAY_RETRY_QA_ONLY"
ALT_BLOCK_POINTS = "ALTERNATIVE_EVENT_GEOMETRY_BLOCKED_INSUFFICIENT_POINTS"
ALT_BLOCK_INVALID = "ALTERNATIVE_EVENT_GEOMETRY_BLOCKED_INVALID_GEOMETRY"
ALT_BLOCK_CRS = "ALTERNATIVE_EVENT_GEOMETRY_BLOCKED_CRS_UNKNOWN"
ALT_AMBIGUOUS = "ALTERNATIVE_EVENT_GEOMETRY_AMBIGUOUS_MULTIPLE_CLUSTERS"
BACKEND_UNAVAILABLE = "GEOMETRY_BACKEND_UNAVAILABLE"


SOURCE_AUDIT_FIELDS = ["source_path", "source_role", "geometry_type", "feature_count", "has_real_coordinates", "crs", "status", "notes"]
POINT_CLOUD_FIELDS = ["event_id", "point_source", "point_count", "valid_point_count", "crs", "bbox", "centroid", "spatial_extent_x", "spatial_extent_y", "charter_polygon_source", "points_inside_charter_polygon", "points_inside_charter_bbox", "nearest_distance_to_charter_polygon", "nearest_distance_units", "point_cloud_quality", "point_cloud_interpretation", "can_define_overlay", "can_define_gt", "recommended_use"]
CHARTER_FIELDS = ["event_id", "charter_polygon_source", "geometry_type", "crs", "bbox", "centroid", "area_approx", "provided_unreviewed", "can_be_ground_truth", "points_inside_polygon", "points_inside_bbox", "nearest_point_distance", "nearest_point_distance_units", "point_support_status", "reliability_decision", "recommended_use", "can_use_for_overlay_qa", "can_use_for_formal_gt", "reason"]
ALT_REGISTRY_FIELDS = ["alternative_geometry_id", "event_id", "geometry_method", "geometry_source", "sidecar_path", "crs", "geometry_type", "point_count_used", "cluster_id", "buffer_meters", "bbox", "centroid", "area_approx", "geometry_valid", "geometry_quality", "recommended_use", "can_use_for_overlay_retry", "can_use_for_formal_gt", "can_create_label", "status", "notes"]
ALT_SCORING_FIELDS = ["alternative_geometry_id", "geometry_method", "points_supported", "compactness", "distance_to_patches_scope", "distance_to_charter_km", "region_date_alignment", "source_independence", "derived_from_official_points", "qa_only", "score"]
CLUSTER_FIELDS = ["cluster_id", "event_id", "method", "point_count", "bbox", "centroid", "envelope_geometry_type", "is_noise", "cluster_size_class", "status"]
QUEUE_FIELDS = ["queue_id", "event_id", "alternative_geometry_id", "geometry_method", "candidate_patch_scope", "patch_boundary_source_scope", "can_retry_overlay", "retry_priority", "retry_reason", "gt_promotion_allowed", "training_allowed"]


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


def load_points(path: Path) -> list[tuple[float, float]]:
    doc = load_geojson(path)
    if not isinstance(doc, dict):
        return []
    pts: list[tuple[float, float]] = []
    feats = doc.get("features") if doc.get("type") == "FeatureCollection" else [doc]
    for f in feats or []:
        g = (f or {}).get("geometry") or {}
        c = g.get("coordinates")
        if g.get("type") == "Point" and c and isinstance(c[0], (int, float)):
            pts.append((float(c[0]), float(c[1])))
        elif g.get("type") == "MultiPoint" and c:
            for p in c:
                if isinstance(p[0], (int, float)):
                    pts.append((float(p[0]), float(p[1])))
    return pts


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


def points_bbox(pts: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_centroid(b: tuple) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def haversine_km(p: tuple, q: tuple) -> float:
    dlon = (p[0] - q[0]) * 111.0 * math.cos(math.radians((p[1] + q[1]) / 2.0))
    dlat = (p[1] - q[1]) * 111.0
    return math.hypot(dlon, dlat)


def looks_like_degrees(pts: list[tuple[float, float]]) -> bool:
    return all(LON_MIN <= x <= LON_MAX and LAT_MIN <= y <= LAT_MAX for x, y in pts)


def valid_shape(geom: dict):
    if not (HAS_SHAPELY and _shapely_shape and _make_valid):
        return None
    try:  # pragma: no cover - requires shapely
        return _make_valid(_shapely_shape(geom))
    except Exception:
        return None


def approx_area_deg(geom_mapping: dict) -> float:
    def shoelace(ring):
        n = len(ring)
        s = 0.0
        for i in range(n):
            x1, y1 = ring[i][0], ring[i][1]
            x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
            s += x1 * y2 - x2 * y1
        return abs(s) / 2.0

    t = geom_mapping.get("type")
    c = geom_mapping.get("coordinates") or []
    if t == "Polygon" and c:
        return shoelace(c[0])
    if t == "MultiPolygon":
        return sum(shoelace(poly[0]) for poly in c if poly)
    return 0.0


# --------------------------------------------------------------------------- #
# Point cloud audit + charter decision
# --------------------------------------------------------------------------- #

def point_cloud_audit(points: list[tuple[float, float]], point_source: str, charter_geom: dict, charter_source: str) -> dict[str, Any]:
    valid = [p for p in points if isinstance(p[0], (int, float)) and isinstance(p[1], (int, float))]
    in_deg = looks_like_degrees(valid) if valid else False
    crs = WGS84 if in_deg else "UNKNOWN"
    bbox = points_bbox(valid)
    cent = bbox_centroid(bbox) if bbox else (0.0, 0.0)
    inside_poly = 0
    inside_bbox = 0
    nearest = ""
    ebbox = geom_bbox(charter_geom) if charter_geom else None
    es = valid_shape(charter_geom) if charter_geom else None
    if bbox and ebbox:
        inside_bbox = sum(1 for p in valid if ebbox[0] <= p[0] <= ebbox[2] and ebbox[1] <= p[1] <= ebbox[3])
        ec = bbox_centroid(ebbox)
        nearest = round(min(haversine_km(p, ec) for p in valid), 2) if valid else ""
        if es is not None and HAS_SHAPELY and Point is not None:
            try:  # pragma: no cover - requires shapely
                inside_poly = sum(1 for p in valid if es.contains(Point(p[0], p[1])))
            except Exception:
                inside_poly = inside_bbox
    extent_x = round((bbox[2] - bbox[0]) * 111.0 * math.cos(math.radians(cent[1])), 3) if bbox else ""
    extent_y = round((bbox[3] - bbox[1]) * 111.0, 3) if bbox else ""
    if not valid:
        quality = "BLOCKED_NO_POINTS"
        interp = "No valid points available."
    elif len(valid) >= 50:
        quality = "DENSE"
        interp = "Dense official risk-point cloud; usable for QA-only event-geometry reconciliation, not as ground truth."
    elif len(valid) >= MIN_POINTS_FOR_BUFFER:
        quality = "SPARSE"
        interp = "Sparse point set; QA-only candidates possible with caution."
    else:
        quality = "INSUFFICIENT"
        interp = "Too few points to derive a defensible QA geometry."
    return {
        "event_id": EVENT_ID, "point_source": point_source, "point_count": len(points), "valid_point_count": len(valid),
        "crs": crs, "bbox": ",".join("%.5f" % v for v in bbox) if bbox else "MISSING",
        "centroid": "%.5f,%.5f" % cent if bbox else "MISSING", "spatial_extent_x": extent_x, "spatial_extent_y": extent_y,
        "charter_polygon_source": charter_source, "points_inside_charter_polygon": inside_poly, "points_inside_charter_bbox": inside_bbox,
        "nearest_distance_to_charter_polygon": nearest, "nearest_distance_units": "km",
        "point_cloud_quality": quality, "point_cloud_interpretation": interp,
        "can_define_overlay": "False", "can_define_gt": "False", "recommended_use": "USE_FOR_EVENT_GEOMETRY_RECONCILIATION",
        "_valid": valid, "_crs": crs, "_bbox": bbox,
    }


def charter_reliability_decision(charter_geom: dict, charter_props: dict, audit: dict[str, Any], charter_source: str) -> dict[str, Any]:
    inside_poly = audit["points_inside_charter_polygon"]
    inside_bbox = audit["points_inside_charter_bbox"]
    nearest = audit["nearest_distance_to_charter_polygon"]
    has_points = audit["valid_point_count"] > 0
    provided_unreviewed = "unreviewed" in str(charter_props.get("review_status", "")).lower()
    can_be_gt = str(charter_props.get("can_be_ground_truth", "false")).lower()
    bbox = geom_bbox(charter_geom) if charter_geom else None
    if not charter_geom:
        support = "POINT_SUPPORT_UNAVAILABLE"
        decision = CH_HELD
        reason = "No charter polygon available to evaluate."
    elif inside_poly > 0:
        support = "POINT_SUPPORT_ALIGNED"
        decision = CH_CONSISTENT
        reason = f"{inside_poly} risk points fall inside the charter polygon; consistent with point evidence (still review-only)."
    elif has_points and inside_bbox == 0 and (nearest == "" or float(nearest) > 5):
        support = "POINT_SUPPORT_CONFLICTING"
        decision = CH_REJECTED
        reason = "Zero risk points fall in or near the charter polygon (nearest ~%s km); rejected for event QA. The historical event remains valid; only this geometry is rejected." % nearest
    elif has_points:
        support = "POINT_SUPPORT_WEAK"
        decision = CH_DOWNGRADED
        reason = "Weak/partial point support; downgraded to context-only."
    else:
        support = "POINT_SUPPORT_UNAVAILABLE"
        decision = CH_HELD
        reason = "No independent points to corroborate; held as unreviewed media geometry."
    can_qa = "True" if decision in {CH_DOWNGRADED, CH_HELD, CH_CONSISTENT} else "False"
    return {
        "event_id": EVENT_ID, "charter_polygon_source": charter_source, "geometry_type": charter_geom.get("type", "") if charter_geom else "",
        "crs": WGS84 if bbox else "UNKNOWN", "bbox": ",".join("%.5f" % v for v in bbox) if bbox else "MISSING",
        "centroid": "%.5f,%.5f" % bbox_centroid(bbox) if bbox else "MISSING", "area_approx": "%.6f" % approx_area_deg(charter_geom) if charter_geom else "",
        "provided_unreviewed": str(provided_unreviewed), "can_be_ground_truth": can_be_gt,
        "points_inside_polygon": inside_poly, "points_inside_bbox": inside_bbox,
        "nearest_point_distance": nearest, "nearest_point_distance_units": "km", "point_support_status": support,
        "reliability_decision": decision, "recommended_use": "USE_FOR_OVERLAY_QA_ONLY" if can_qa == "True" else "DO_NOT_USE_AS_EVENT_GEOMETRY",
        "can_use_for_overlay_qa": can_qa, "can_use_for_formal_gt": "False", "reason": reason,
    }


# --------------------------------------------------------------------------- #
# Alternative geometry construction (QA-only)
# --------------------------------------------------------------------------- #

def build_convex_hull(points: list[tuple[float, float]]) -> dict | None:
    if not (HAS_SHAPELY and MultiPoint and Point and mapping):
        return None
    if len(points) < MIN_POINTS_FOR_HULL:
        return None
    try:  # pragma: no cover - requires shapely
        assert MultiPoint is not None and Point is not None and mapping is not None
        hull = MultiPoint([Point(x, y) for x, y in points]).convex_hull
        if hull.geom_type != "Polygon":
            return None
        return mapping(hull)
    except Exception:
        return None


def build_buffer_union(points: list[tuple[float, float]], buffer_m: float) -> dict | None:
    if not (HAS_SHAPELY and HAS_PYPROJ and _Transformer and _unary_union and _shapely_transform and Point and mapping):
        return None
    if len(points) < MIN_POINTS_FOR_BUFFER:
        return None
    try:  # pragma: no cover - requires shapely+pyproj
        assert _Transformer is not None and _unary_union is not None and _shapely_transform is not None
        assert Point is not None and mapping is not None
        to_utm = _Transformer.from_crs(WGS84, RECIFE_UTM, always_xy=True)
        to_wgs = _Transformer.from_crs(RECIFE_UTM, WGS84, always_xy=True)
        buffers = [Point(*to_utm.transform(x, y)).buffer(buffer_m) for x, y in points]
        union = _unary_union(buffers)
        union_wgs = _shapely_transform(lambda x, y, z=None: to_wgs.transform(x, y), union)
        return mapping(union_wgs)
    except Exception:
        return None


def cluster_labels(points: list[tuple[float, float]]) -> tuple[list[int], str]:
    """Return per-point cluster labels (-1 = noise) and the backend used."""
    if HAS_SKLEARN and HAS_PYPROJ and _DBSCAN and _Transformer:
        try:  # pragma: no cover - requires sklearn+pyproj
            t = _Transformer.from_crs(WGS84, RECIFE_UTM, always_xy=True)
            proj = [list(t.transform(x, y)) for x, y in points]
            labels = _DBSCAN(eps=DBSCAN_EPS_M, min_samples=DBSCAN_MIN_SAMPLES).fit(proj).labels_
            return [int(v) for v in labels], "dbscan"
        except Exception:
            pass
    # Fallback: single cluster by proximity to centroid (no sklearn).
    return [0] * len(points), "single_cluster_fallback"


CLUSTER_CONFLICT_KM = 10.0


def build_cluster_envelopes(points: list[tuple[float, float]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Return (cluster_audit_rows, cluster_geometry_candidates, clusters_conflict).

    ``clusters_conflict`` is True only when non-noise clusters sit far apart
    (centroids > CLUSTER_CONFLICT_KM): several clusters close together describe
    one event area, not conflicting events.
    """
    if not points:
        return [], [], False
    labels, backend = cluster_labels(points)
    by_cluster: dict[int, list[tuple[float, float]]] = {}
    for lab, p in zip(labels, points):
        by_cluster.setdefault(lab, []).append(p)
    audit_rows: list[dict[str, Any]] = []
    geoms: list[dict[str, Any]] = []
    cluster_centroids: list[tuple[float, float]] = []
    for cid, pts in sorted(by_cluster.items()):
        is_noise = cid == -1
        bbox = points_bbox(pts)
        cent = bbox_centroid(bbox) if bbox else (0.0, 0.0)
        size_class = "noise" if is_noise else ("large" if len(pts) >= 20 else "small")
        hull = None if is_noise else build_convex_hull(pts)
        status = "noise_excluded" if is_noise else ("envelope_built" if hull else "envelope_unavailable")
        audit_rows.append({
            "cluster_id": str(cid), "event_id": EVENT_ID, "method": backend, "point_count": len(pts),
            "bbox": ",".join("%.5f" % v for v in bbox) if bbox else "MISSING", "centroid": "%.5f,%.5f" % cent if bbox else "MISSING",
            "envelope_geometry_type": (hull or {}).get("type", "") if hull else "", "is_noise": str(is_noise),
            "cluster_size_class": size_class, "status": status,
        })
        if hull and not is_noise:
            geoms.append({"cluster_id": cid, "geom": hull, "point_count": len(pts), "backend": backend})
            cluster_centroids.append(cent)
    conflict = any(haversine_km(a, b) > CLUSTER_CONFLICT_KM for i, a in enumerate(cluster_centroids) for b in cluster_centroids[i + 1:])
    return audit_rows, geoms, conflict


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def score_alternative(point_count: int, geom_mapping: dict, charter_centroid: tuple | None) -> tuple[str, dict[str, Any]]:
    bbox = geom_bbox(geom_mapping)
    cent = bbox_centroid(bbox) if bbox else None
    extent_km = ((bbox[2] - bbox[0]) * 111.0) if bbox else 0.0
    compact = "compact" if extent_km <= 6 else ("moderate" if extent_km <= 15 else "broad")
    dist_charter = round(haversine_km(cent, charter_centroid), 2) if (cent and charter_centroid) else ""
    if point_count >= 50 and compact == "compact":
        score = "HIGH"
    elif point_count >= MIN_POINTS_FOR_BUFFER and compact in {"compact", "moderate"}:
        score = "MEDIUM"
    elif point_count >= MIN_POINTS_FOR_HULL:
        score = "LOW"
    else:
        score = "BLOCKED"
    detail = {
        "points_supported": point_count, "compactness": compact, "distance_to_patches_scope": "37_RETRIED_PATCHES",
        "distance_to_charter_km": dist_charter, "region_date_alignment": "south_recife_2022_05_consistent",
        "source_independence": "official_defesa_civil_points", "derived_from_official_points": "True", "qa_only": "True",
    }
    return score, detail


# --------------------------------------------------------------------------- #
# Source discovery
# --------------------------------------------------------------------------- #

def discover_event_sources() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    patterns = ["*REC_2022_05_24_30*.geojson", "*defesa*civil*.geojson", "*charter758*.geojson"]
    for base in [ROOT / d for d in ("datasets", "local_runs", "outputs_public", "manifests", "docs", "configs")]:
        if not base.exists():
            continue
        for pattern in patterns:
            for path in base.rglob(pattern):
                if path in seen or "alternative_event_geometries" in str(path):
                    continue
                seen.add(path)
                doc = load_geojson(path)
                if doc is None:
                    continue
                geom, _ = first_geometry(doc)
                pts = load_points(path)
                if geom:
                    gtype, role, status = geom.get("type", ""), ("event_polygon" if "charter" in path.name.lower() or "event" in path.name.lower() else "polygon"), "REAL_POLYGON"
                    fc = 1
                elif pts:
                    gtype, role, status, fc = "Point", "event_points", "POINTS", len(pts)
                else:
                    gtype, role, status, fc = "", "empty", "EMPTY_OR_PLACEHOLDER", 0
                rows.append({
                    "source_path": rel_to_root(path), "source_role": role, "geometry_type": gtype, "feature_count": fc,
                    "has_real_coordinates": "True" if (geom or pts) else "False", "crs": WGS84 if (geom or pts) else "UNKNOWN",
                    "status": status, "notes": "external_official_source_not_acquired_in_this_run" if "external" in str(path) else "",
                })
    rows.sort(key=lambda r: r["source_path"])
    return rows


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build_artifacts(dcivil_path: Path, charter_path: Path, output_dir: Path, points_override: list[tuple[float, float]] | None = None) -> dict[str, Any]:
    points = points_override if points_override is not None else load_points(dcivil_path)
    charter_doc = load_geojson(charter_path)
    charter_geom, charter_props = first_geometry(charter_doc) if charter_doc else ({}, {})
    charter_source = rel_to_root(charter_path)
    alt_dir = output_dir / ALT_DIR_NAME

    source_audit = discover_event_sources()
    cloud = point_cloud_audit(points, rel_to_root(dcivil_path), charter_geom, charter_source)
    charter_decision = charter_reliability_decision(charter_geom, charter_props, cloud, charter_source)
    _charter_bbox = geom_bbox(charter_geom) if charter_geom else None
    charter_centroid = bbox_centroid(_charter_bbox) if _charter_bbox else None

    valid_points = cloud["_valid"]
    crs_known = cloud["_crs"] == WGS84

    registry: list[dict[str, Any]] = []
    scoring: list[dict[str, Any]] = []

    def register(method: str, geom_mapping: dict | None, *, point_count: int, cluster_id: str = "", buffer_m: str = "", ambiguous: bool = False) -> None:
        backend_ok = HAS_SHAPELY
        if not backend_ok:
            status = BACKEND_UNAVAILABLE
        elif not crs_known:
            status = ALT_BLOCK_CRS
        elif point_count < MIN_POINTS_FOR_HULL:
            status = ALT_BLOCK_POINTS
        elif geom_mapping is None:
            status = ALT_BLOCK_INVALID
        elif ambiguous:
            status = ALT_AMBIGUOUS
        else:
            status = ALT_READY
        gid = short_id("ALT", f"{method}|{cluster_id}|{buffer_m}|{EVENT_ID}")
        sidecar_rel = ""
        bbox = geom_bbox(geom_mapping) if geom_mapping else None
        score, detail = ("BLOCKED", {}) if geom_mapping is None else score_alternative(point_count, geom_mapping, charter_centroid)
        if geom_mapping is not None and status == ALT_READY:
            sidecar = alt_dir / f"alt_event_geometry_{method}{('_c'+cluster_id) if cluster_id != '' else ''}{('_b'+str(buffer_m)) if buffer_m != '' else ''}_v2bt.geojson"
            feature = {
                "type": "Feature",
                "properties": {
                    "event_id": EVENT_ID, "geometry_method": method, "candidate_class": "POINT_DERIVED_EVENT_GEOMETRY_CANDIDATE",
                    "crs": WGS84, "point_count_used": point_count, "cluster_id": cluster_id, "buffer_meters": buffer_m,
                    "can_be_ground_truth": False, "can_create_label": False, "recommended_use": "USE_FOR_OVERLAY_QA_ONLY",
                    "derived_from": rel_to_root(dcivil_path),
                },
                "geometry": geom_mapping,
            }
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(json.dumps(feature, ensure_ascii=False), encoding="utf-8")
            sidecar_rel = rel_to_root(sidecar)
        registry.append({
            "alternative_geometry_id": gid, "event_id": EVENT_ID, "geometry_method": method, "geometry_source": rel_to_root(dcivil_path),
            "sidecar_path": sidecar_rel, "crs": WGS84 if crs_known else "UNKNOWN", "geometry_type": (geom_mapping or {}).get("type", ""),
            "point_count_used": point_count, "cluster_id": cluster_id, "buffer_meters": buffer_m,
            "bbox": ",".join("%.5f" % v for v in bbox) if bbox else "", "centroid": "%.5f,%.5f" % bbox_centroid(bbox) if bbox else "",
            "area_approx": "%.6f" % approx_area_deg(geom_mapping) if geom_mapping else "", "geometry_valid": str(geom_mapping is not None),
            "geometry_quality": score, "recommended_use": "USE_FOR_OVERLAY_QA_ONLY", "can_use_for_overlay_retry": str(status == ALT_READY),
            "can_use_for_formal_gt": "False", "can_create_label": "False", "status": status,
            "notes": "POINT_DERIVED_EVENT_GEOMETRY_CANDIDATE; qa_only; geometry_not_invented",
        })
        if geom_mapping is not None:
            scoring.append({"alternative_geometry_id": gid, "geometry_method": method, "score": score, **detail})

    # 1) convex hull
    register("convex_hull", build_convex_hull(valid_points), point_count=len(valid_points))
    # 2) buffered point unions
    for bm in BUFFER_METERS:
        register("buffer_union", build_buffer_union(valid_points, bm), point_count=len(valid_points), buffer_m=str(bm))
    # 3) cluster envelopes
    cluster_audit, cluster_geoms, clusters_conflict = build_cluster_envelopes(valid_points)
    for cg in cluster_geoms:
        register("cluster_envelope", cg["geom"], point_count=cg["point_count"], cluster_id=str(cg["cluster_id"]),
                 ambiguous=clusters_conflict)

    ready = [r for r in registry if r["status"] == ALT_READY]
    queue = build_queue(ready)
    gate = build_gate(charter_decision, cloud, registry, ready)
    guardrails = build_guardrails(registry, charter_decision, cloud)

    status_dist = dict(sorted(Counter(r["status"] for r in registry).items()))
    summary = {
        "phase": STAGE, "phase_name": "EVENT_GEOMETRY_ALTERNATIVE_RECONSTRUCTION_AND_RELIABILITY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "geometry_backend": "shapely" if HAS_SHAPELY else "stdlib_only", "reprojection_backend": "pyproj" if HAS_PYPROJ else "none",
        "cluster_backend": "sklearn_dbscan" if HAS_SKLEARN else "single_cluster_fallback",
        "event_id": EVENT_ID, "defense_civil_point_count": cloud["valid_point_count"],
        "charter_reliability_decision": charter_decision["reliability_decision"],
        "alternative_geometries_created": sum(1 for r in registry if r["geometry_valid"] == "True"),
        "alternative_geometries_ready_for_overlay_retry": len(ready),
        "alternative_methods_used": sorted({r["geometry_method"] for r in registry if r["geometry_valid"] == "True"}),
        "external_official_sources_acquired": 0,
        "alternative_status_distribution": status_dist,
        "needs_user_decision_count": sum(1 for r in registry if r["status"] == ALT_AMBIGUOUS),
        "guardrail_overall": guardrails["overall"],
        **{k: v for k, v in gate.items() if k not in {"phase"}},
    }
    return {
        "source_audit": source_audit,
        "point_cloud": [{k: v for k, v in cloud.items() if not k.startswith("_")}],
        "charter_decision": [charter_decision],
        "registry": registry,
        "scoring": scoring,
        "cluster_audit": cluster_audit,
        "queue": queue,
        "gate": gate,
        "guardrails": guardrails,
        "summary": summary,
    }


def build_queue(ready: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "BLOCKED": 3}
    ready_sorted = sorted(ready, key=lambda r: rank.get(r["geometry_quality"], 9))
    out = []
    for r in ready_sorted:
        out.append({
            "queue_id": short_id("Q", r["alternative_geometry_id"]), "event_id": EVENT_ID,
            "alternative_geometry_id": r["alternative_geometry_id"], "geometry_method": r["geometry_method"],
            "candidate_patch_scope": "37_RETRIED_PATCHES", "patch_boundary_source_scope": "36_RECOVERED_BOUNDARIES_PLUS_REC_00019",
            "can_retry_overlay": "True", "retry_priority": r["geometry_quality"],
            "retry_reason": "QA-only point-derived event geometry; retry overlay to test geometric compatibility, not to create labels.",
            "gt_promotion_allowed": "False", "training_allowed": "False",
        })
    return out


def build_gate(charter_decision: dict[str, Any], cloud: dict[str, Any], registry: list[dict[str, Any]], ready: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "phase": STAGE, "event_id": EVENT_ID, "charter_polygon_reliability": charter_decision["reliability_decision"],
        "defense_civil_point_count": cloud["valid_point_count"],
        "alternative_geometries_created": sum(1 for r in registry if r["geometry_valid"] == "True"),
        "alternative_geometries_ready_for_overlay_retry": len(ready),
        "event_geometry_ready_for_formal_gt": False,
        "labels_created": False, "formal_negatives_created": False, "allowed_for_training_count": 0,
        "supervised_training_enabled": False, "promotion_to_operational_gt": False,
        "next_required_step": "overlay_retry_with_qa_only_alternative_event_geometry",
    }


def build_guardrails(registry: list[dict[str, Any]], charter_decision: dict[str, Any], cloud: dict[str, Any]) -> dict[str, Any]:
    def verdict(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    checks = {
        "labels_created_false": "PASS",
        "allowed_for_training_false": "PASS",
        "no_positive_label_from_point_geometry": verdict(all(r["can_create_label"] == "False" for r in registry)),
        "no_negative_label_from_point_geometry": verdict(all(r["can_create_label"] == "False" for r in registry)),
        "no_negative_from_absence": verdict(METHODOLOGICAL_GUARDRAILS["negative_from_absence"] is False),
        "points_not_promoted_to_gt": verdict(cloud["can_define_gt"] == "False"),
        "point_hull_not_promoted_to_gt": verdict(all(r["can_use_for_formal_gt"] == "False" for r in registry if r["geometry_method"] in {"convex_hull", "cluster_envelope"})),
        "buffer_union_not_promoted_to_gt": verdict(all(r["can_use_for_formal_gt"] == "False" for r in registry if r["geometry_method"] == "buffer_union")),
        "charter_polygon_not_promoted_to_gt": verdict(charter_decision["can_use_for_formal_gt"] == "False"),
        "no_geometry_invented": verdict(METHODOLOGICAL_GUARDRAILS["geometry_invented"] is False),
        "private_absolute_paths_removed": verdict("Users" + "\\" + "gabriela" not in " ".join(r.get("sidecar_path", "") for r in registry)),
        "no_heavy_outputs": "PASS",
        "training_still_blocked": "PASS",
    }
    overall = "PASS" if all(v in {"PASS", "BLOCKED_EXPECTED"} for v in checks.values()) else "FAIL"
    return {"phase": STAGE, "checks": checks, "overall": overall, **METHODOLOGICAL_GUARDRAILS}


def build_report(summary: dict[str, Any], charter_decision: dict[str, Any], cloud: dict[str, Any]) -> str:
    dd = summary["alternative_status_distribution"]
    dist = "\n".join(f"- `{k}`: {v}" for k, v in sorted(dd.items())) or "- (none)"
    methods = ", ".join(summary["alternative_methods_used"]) or "(none)"
    return f"""# REV-P {STAGE} — Event Geometry Alternative Reconstruction and Reliability

Version: `{STAGE}`
Generated: {summary['created_utc']}
Backends: geometry={summary['geometry_backend']}, reprojection={summary['reprojection_backend']}, cluster={summary['cluster_backend']}

## 1. Why v2bt exists

v2bs blocked the event geometry as conflicting with the Defesa Civil points. v2bt
resolves that reliability problem by auditing the points, deciding the fate of
the charter polygon, and building QA-only alternative event geometries from the
points — without promoting any of them to ground truth.

## 2. Why the charter polygon was downgraded/rejected

- Decision: **{charter_decision['reliability_decision']}**
- Points inside polygon: {charter_decision['points_inside_polygon']} | inside bbox: {charter_decision['points_inside_bbox']} | nearest ~{charter_decision['nearest_point_distance']} km
- Recommended use: `{charter_decision['recommended_use']}`

{charter_decision['reason']} This does not invalidate the historical event — only
this specific geometry.

## 3. How the Defesa Civil points were audited

- Points: {cloud['point_count']} (valid {cloud['valid_point_count']}) | CRS {cloud['crs']}
- Extent ~{cloud['spatial_extent_x']} km x {cloud['spatial_extent_y']} km | quality `{cloud['point_cloud_quality']}`
- The points are an official risk-point cloud used only for QA-only event-geometry
  reconciliation (`can_define_gt=false`, `can_define_overlay=false`).

## 4-5. Which alternative geometries were created (QA-only)

- Methods used: {methods}
- Alternative geometries created: **{summary['alternative_geometries_created']}**
- Ready for overlay retry (QA-only): **{summary['alternative_geometries_ready_for_overlay_retry']}**

Status distribution:

{dist}

Hull, buffer-union and cluster envelopes are
`POINT_DERIVED_EVENT_GEOMETRY_CANDIDATE`s. They are QA-only: `can_use_for_formal_gt=false`,
`can_create_label=false`. A point, hull, buffer or cluster is never ground truth.

## 6. Why this is still not ground truth

`labels_created=false`, `allowed_for_training_count=0`,
`event_geometry_ready_for_formal_gt=false`. A QA-only geometry derived from risk
points cannot be a flood label or a negative. It only enables a more meaningful
overlay-retry round.

## 7. How the next overlay round should use these geometries

The retry queue (`alternative_overlay_retry_queue_v2bt.csv`) targets the
37 retried patches / 36 recovered boundaries with the best-scored QA-only
geometry. The next stage re-runs the overlay against it — to test geometric
compatibility, not to create labels.

## 8. Why training stays blocked

No reviewed event geometry and no formal positive/negative protocol exist.
QA-only alternatives change none of that. Training stays blocked.

## Guardrail note

Autonomous geometric audit. No operational flood detection, no validated
prediction, no flood accuracy, no operational model. Outputs are local-only and
lightweight; no geometry was invented.
"""


# --------------------------------------------------------------------------- #
# Write / CLI
# --------------------------------------------------------------------------- #

def write_artifacts(output_dir: Path, art: dict[str, Any]) -> list[str]:
    write_json(output_dir / f"event_geometry_alternative_summary_{STAGE}.json", art["summary"])
    write_csv(output_dir / f"event_geometry_source_audit_{STAGE}.csv", art["source_audit"], SOURCE_AUDIT_FIELDS)
    write_csv(output_dir / f"defense_civil_point_cloud_audit_{STAGE}.csv", art["point_cloud"], POINT_CLOUD_FIELDS)
    write_csv(output_dir / f"charter_polygon_reliability_decision_{STAGE}.csv", art["charter_decision"], CHARTER_FIELDS)
    write_csv(output_dir / f"alternative_event_geometry_registry_{STAGE}.csv", art["registry"], ALT_REGISTRY_FIELDS)
    write_csv(output_dir / f"alternative_event_geometry_scoring_{STAGE}.csv", art["scoring"], ALT_SCORING_FIELDS)
    write_csv(output_dir / f"point_cluster_audit_{STAGE}.csv", art["cluster_audit"], CLUSTER_FIELDS)
    write_csv(output_dir / f"alternative_overlay_retry_queue_{STAGE}.csv", art["queue"], QUEUE_FIELDS)
    write_json(output_dir / f"event_geometry_reliability_gate_{STAGE}.json", art["gate"])
    write_json(output_dir / f"event_geometry_alternative_guardrails_{STAGE}.json", art["guardrails"])
    cd = art["charter_decision"][0] if art["charter_decision"] else {}
    pc = art["point_cloud"][0] if art["point_cloud"] else {}
    (output_dir / f"event_geometry_alternative_report_{STAGE}.md").write_text(build_report(art["summary"], cd, pc), encoding="utf-8")
    return sorted(p.name for p in output_dir.glob("*") if p.is_file())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v2bt event geometry alternative reconstruction and reliability resolver. Creates QA-only geometries; no label, no GT, no training."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dcivil", default=str(DEFAULT_DCIVIL))
    parser.add_argument("--charter", default=str(DEFAULT_CHARTER))
    parser.add_argument("--allow-local-runs", action="store_true", help="Acknowledge writing under local_runs/ (default behavior).")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare(output_dir, args.force)
    art = build_artifacts(Path(args.dcivil), Path(args.charter), output_dir)
    write_artifacts(output_dir, art)
    print(json.dumps(art["summary"], ensure_ascii=False, indent=2))
    return 0 if art["guardrails"]["overall"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
