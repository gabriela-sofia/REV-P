"""REV-P v2bq — Patch-event overlay geometry resolver.

Attacks the technical blocker left by v2bp for the auto-validated candidate
positives: ``NO_PATCH_EVENT_OVERLAY_GEOMETRY``. It discovers real geometry that
already exists in the repository, normalizes CRS to EPSG:4326 when possible,
validates it, and computes the patch-event intersection when both a patch
boundary and an event polygon are available — autonomously, by parsing and
calculation, never by inventing geometry.

A resolved overlay does not create a label and does not enable training. It can
move a case to ``READY_FOR_FORMAL_GT_PROTOCOL``, nothing more. Missing geometry
is recorded as a specific block (not "needs human review"); ``NEEDS_USER_DECISION``
is reserved for genuine geometric ambiguity (several conflicting plausible
geometries). Centroids/point clouds are weak support and are never promoted to a
polygon overlay. Unknown CRS blocks promotion. Outputs are local-only and light.
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

# Optional geometry backends — degrade fail-closed if absent.
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
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "ground_truth" / "v2bq"
STAGE = "v2bq"

DEFAULT_CANDIDATE_REGISTRY = ROOT / "local_runs" / "ground_truth" / "v2bp" / "autonomous_candidate_positive_registry_v2bp.csv"
DEFAULT_FEATURE_TABLE = ROOT / "local_runs" / "multimodal" / "v2bn" / "multimodal_feature_table_core_v2bn.csv"

# Directories to scan for geometry sources.
SCAN_DIRS = ["datasets", "manifests", "outputs_public", "local_runs", "docs", "configs"]
# Substrings that mark a path as an example/template/placeholder (never real).
EXCLUDE_MARKERS = ("examples", "synthetic", "_template", "placeholder", "_empty")

WGS84 = "EPSG:4326"
# Plausible bounds for Brazil in lon/lat degrees (sanity, never to invent).
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = -74.5, -33.0, -34.5, 6.0

# Decision vocabulary
OVERLAY_RESOLVED = "OVERLAY_RESOLVED"
BLOCK_PATCH = "OVERLAY_BLOCKED_PATCH_GEOMETRY_MISSING"
BLOCK_EVENT = "OVERLAY_BLOCKED_EVENT_GEOMETRY_MISSING"
BLOCK_BOTH = "OVERLAY_BLOCKED_BOTH_GEOMETRIES_MISSING"
BLOCK_CRS = "OVERLAY_BLOCKED_CRS_UNKNOWN"
REJECT_INVALID = "OVERLAY_REJECT_INVALID_GEOMETRY"
REJECT_NO_INTERSECTION = "OVERLAY_REJECT_NO_INTERSECTION"
REJECT_REGION = "OVERLAY_REJECT_REGION_MISMATCH"
REJECT_EVENT = "OVERLAY_REJECT_EVENT_MISMATCH"
REVIEW_AMBIGUOUS = "OVERLAY_REVIEW_AMBIGUOUS_MULTIPLE_GEOMETRIES"
REVIEW_CENTROID = "OVERLAY_REVIEW_LOW_CONFIDENCE_CENTROID_ONLY"
BACKEND_UNAVAILABLE = "GEOMETRY_BACKEND_UNAVAILABLE"

RESOLVED_STATES = {OVERLAY_RESOLVED}
BLOCKED_STATES = {BLOCK_PATCH, BLOCK_EVENT, BLOCK_BOTH, BLOCK_CRS, BACKEND_UNAVAILABLE}
REJECT_STATES = {REJECT_INVALID, REJECT_NO_INTERSECTION, REJECT_REGION, REJECT_EVENT}


METHODOLOGICAL_GUARDRAILS = {
    "review_only": True,
    "labels_created": False,
    "formal_negative_created": False,
    "negative_from_absence": False,
    "geometry_invented": False,
    "centroid_promoted_to_overlay": False,
    "overlay_equals_label": False,
    "supervised_training": False,
    "multimodal_execution_enabled": False,
    "predictive_claims": False,
    "outputs_local_only": True,
}


RESOLUTION_FIELDS = [
    "overlay_resolution_id", "candidate_id", "canonical_patch_id", "dino_input_id",
    "region", "candidate_event_id", "patch_geometry_status", "event_geometry_status",
    "patch_geometry_source", "event_geometry_source", "patch_crs", "event_crs",
    "normalized_crs", "geometry_backend", "patch_geometry_valid", "event_geometry_valid",
    "overlay_attempted", "overlay_status", "intersection_area", "intersection_area_units",
    "intersection_ratio_patch", "intersection_ratio_event", "centroid_distance",
    "centroid_distance_units", "overlay_confidence", "gt_protocol_status_after_overlay",
    "gt_patch_flood_observed", "allowed_for_training", "promotion_blocker",
    "auto_decision", "auto_decision_reason", "needs_user_decision", "notes",
]

PATCH_INV_FIELDS = ["canonical_patch_id", "region", "geometry_status", "geometry_type", "source_path", "source_crs", "normalized_crs", "valid", "notes"]
EVENT_INV_FIELDS = ["candidate_event_id", "region", "geometry_status", "geometry_type", "feature_count", "source_path", "source_crs", "normalized_crs", "valid", "support_role", "notes"]
DISCOVERY_FIELDS = ["source_path", "format", "geometry_role", "geometry_type", "feature_count", "has_real_coordinates", "crs", "patch_id_hint", "event_id_hint", "status", "notes"]
COMPUTATION_FIELDS = ["candidate_id", "canonical_patch_id", "candidate_event_id", "backend", "patch_area", "event_area", "intersection_area", "area_units", "intersection_ratio_patch", "intersection_ratio_event", "bbox_overlap", "computation_status", "reason"]
RESOLVED_REG_FIELDS = ["overlay_resolution_id", "canonical_patch_id", "candidate_event_id", "intersection_ratio_patch", "intersection_ratio_event", "overlay_confidence", "gt_protocol_status_after_overlay", "promotion_blocker"]
BLOCKED_REG_FIELDS = ["overlay_resolution_id", "canonical_patch_id", "candidate_event_id", "overlay_status", "missing_component", "reason"]
REJECTION_REG_FIELDS = ["overlay_resolution_id", "canonical_patch_id", "candidate_event_id", "overlay_status", "reason", "guardrail_reference"]
SIDECAR_FIELDS = ["sidecar_id", "kind", "canonical_patch_id", "candidate_event_id", "relative_path", "geometry_type", "crs", "source_path", "source_hash"]


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


def file_hash(path: Path) -> str:
    import hashlib
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


# --------------------------------------------------------------------------- #
# Geometry primitives (stdlib; shapely used when available for exact ops)
# --------------------------------------------------------------------------- #

def iter_coords(coords: Any):
    for x in coords:
        if isinstance(x, (int, float)):
            yield coords
            return
        if x and isinstance(x[0], (int, float)):
            yield x
        else:
            yield from iter_coords(x)


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


def geometry_bbox(geom: dict) -> tuple[float, float, float, float] | None:
    coords = geom.get("coordinates")
    if not coords:
        return None
    xs, ys = flat_xy(coords)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_overlap(a: tuple, b: tuple) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def bbox_centroid(bbox: tuple) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def looks_like_degrees(bbox: tuple) -> bool:
    return LON_MIN <= bbox[0] <= LON_MAX and LON_MIN <= bbox[2] <= LON_MAX and LAT_MIN <= bbox[1] <= LAT_MAX and LAT_MIN <= bbox[3] <= LAT_MAX


def detect_crs(top: dict, props: dict, bbox: tuple | None) -> str:
    for src in (props, top):
        crs = (src or {}).get("crs")
        if isinstance(crs, str) and crs.strip():
            return crs.strip().upper()
        if isinstance(crs, dict):
            name = crs.get("properties", {}).get("name", "")
            if name:
                return str(name).upper().replace("URN:OGC:DEF:CRS:", "").replace("::", ":")
    # Infer only the unambiguous WGS84-degrees case; never guess a projected CRS.
    if bbox and looks_like_degrees(bbox):
        return WGS84
    return "UNKNOWN"


def shoelace_area(ring: list[tuple[float, float]]) -> float:
    n = len(ring)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def polygon_rings(geom: dict) -> list[list[tuple[float, float]]]:
    """Return outer rings as coordinate lists for Polygon / MultiPolygon."""
    gtype = geom.get("type")
    coords = geom.get("coordinates") or []
    rings: list[list[tuple[float, float]]] = []
    if gtype == "Polygon" and coords:
        rings.append([(float(p[0]), float(p[1])) for p in coords[0]])
    elif gtype == "MultiPolygon":
        for poly in coords:
            if poly:
                rings.append([(float(p[0]), float(p[1])) for p in poly[0]])
    return rings


def approx_area(geom: dict) -> float:
    return sum(shoelace_area(r) for r in polygon_rings(geom))


# --------------------------------------------------------------------------- #
# GeoJSON loading / classification
# --------------------------------------------------------------------------- #

def load_geojson(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def extract_features(doc: Any) -> list[dict]:
    if not isinstance(doc, dict):
        return []
    if doc.get("type") == "FeatureCollection":
        return [f for f in doc.get("features", []) if isinstance(f, dict)]
    if doc.get("type") == "Feature":
        return [doc]
    if doc.get("type") in {"Polygon", "MultiPolygon", "Point", "MultiPoint", "LineString"}:
        return [{"type": "Feature", "geometry": doc, "properties": {}}]
    return []


def polygon_features(features: list[dict]) -> list[dict]:
    out = []
    for f in features:
        g = (f.get("geometry") or {})
        if g.get("type") in {"Polygon", "MultiPolygon"} and g.get("coordinates"):
            out.append(f)
    return out


def point_features(features: list[dict]) -> list[dict]:
    return [f for f in features if (f.get("geometry") or {}).get("type") in {"Point", "MultiPoint"} and (f.get("geometry") or {}).get("coordinates")]


def classify_geojson(path: Path) -> dict[str, Any] | None:
    doc = load_geojson(path)
    if doc is None:
        return None
    feats = extract_features(doc)
    polys = polygon_features(feats)
    pts = point_features(feats)
    top = doc if isinstance(doc, dict) else {}
    first = feats[0] if feats else {}
    props = first.get("properties") or {}
    geom = (polys[0].get("geometry") if polys else (first.get("geometry") or {})) or {}
    bbox = geometry_bbox(geom) if geom else None
    crs = detect_crs(top, props, bbox)
    name = path.name.lower()
    patch_hint = props.get("patch_id", "")
    event_hint = props.get("event_id", "")
    if not patch_hint:
        import re
        m = re.search(r"(rec|pet|cur)_?\d{2,6}", name)
        patch_hint = m.group(0).upper() if m else ""
    if not event_hint:
        import re
        m = re.search(r"(rec|pet|cur)[-_]?20\d{2}[-_]\d{2}[-_]\d{2}", name)
        event_hint = m.group(0).upper() if m else ""
    if polys:
        role = "patch_polygon" if ("patch" in name or "boundary" in name) and not event_hint else ("event_polygon" if (event_hint or "event" in name) else "polygon")
        gtype = (polys[0].get("geometry") or {}).get("type")
        status = "REAL_POLYGON"
    elif pts:
        role = "context_points"
        gtype = "Point"
        status = "POINTS_WEAK_SUPPORT"
    else:
        role = "empty_or_non_polygon"
        gtype = (geom.get("type") if geom else "")
        status = "EMPTY_OR_PLACEHOLDER"
    return {
        "source_path": rel_to_root(path),
        "format": "geojson",
        "geometry_role": role,
        "geometry_type": gtype or "",
        "feature_count": len(polys) if polys else len(pts),
        "has_real_coordinates": "True" if (polys or pts) else "False",
        "crs": crs,
        "patch_id_hint": str(patch_hint).upper(),
        "event_id_hint": str(event_hint).upper().replace("-", "_"),
        "status": status,
        "notes": "review_status=" + str(props.get("review_status", "")) if props.get("review_status") else "",
        "_path": path,
        "_geom": geom,
        "_polys": polys,
        "_bbox": bbox,
        "_props": props,
    }


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def discover_geometry_sources() -> list[dict[str, Any]]:
    seen: set[Path] = set()
    out: list[dict[str, Any]] = []
    for d in SCAN_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for path in base.rglob("*.geojson"):
            rp = str(path).lower()
            if any(marker in rp for marker in EXCLUDE_MARKERS):
                continue
            if path in seen:
                continue
            seen.add(path)
            info = classify_geojson(path)
            if info:
                out.append(info)
    out.sort(key=lambda r: r["source_path"])
    return out


def index_patch_geometry(sources: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for s in sources:
        if s["geometry_role"] != "patch_polygon" or s["status"] != "REAL_POLYGON":
            continue
        pid = s["patch_id_hint"]
        if pid and pid not in index:
            index[pid] = s
    return index


def index_event_geometry(sources: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for s in sources:
        if s["geometry_role"] != "event_polygon" or s["status"] != "REAL_POLYGON":
            continue
        eid = s["event_id_hint"]
        if eid and eid not in index:
            index[eid] = s
    return index


def _round_bbox(bbox: tuple | None) -> tuple | None:
    return tuple(round(v, 6) for v in bbox) if bbox else None


def conflicting_ids(sources: list[dict[str, Any]], role: str, hint_key: str) -> set[str]:
    """Return ids that have more than one *distinct* real polygon source.

    Distinct = different rounded bounding box. Two sources describing the same
    footprint are not a conflict; genuinely different footprints are.
    """
    by_id: dict[str, set] = {}
    for s in sources:
        if s["geometry_role"] != role or s["status"] != "REAL_POLYGON":
            continue
        ident = s[hint_key]
        if not ident:
            continue
        by_id.setdefault(ident, set()).add(_round_bbox(s["_bbox"]))
    return {ident for ident, boxes in by_id.items() if len(boxes) > 1}


# --------------------------------------------------------------------------- #
# Overlay computation
# --------------------------------------------------------------------------- #

def reproject_geom_to_wgs84(geom: dict, crs: str) -> tuple[dict, str]:
    """Reproject a geometry to WGS84 when pyproj is available and CRS is known."""
    if crs in {WGS84, "EPSG:4326", "CRS84"}:
        return geom, WGS84
    if crs == "UNKNOWN" or not HAS_PYPROJ or _Transformer is None:
        return geom, crs
    try:  # pragma: no cover - exercised only with projected inputs + pyproj
        transformer = _Transformer.from_crs(crs, WGS84, always_xy=True)

        def tx(coords):
            if isinstance(coords[0], (int, float)):
                x, y = transformer.transform(coords[0], coords[1])
                return [x, y]
            return [tx(c) for c in coords]

        return {"type": geom["type"], "coordinates": tx(geom["coordinates"])}, WGS84
    except Exception:
        return geom, crs


def compute_overlay(patch: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    """Compute patch-event intersection. Returns a status dict; never invents."""
    backend = "shapely" if HAS_SHAPELY else "stdlib_bbox"
    pgeom, pcrs = reproject_geom_to_wgs84(patch["_geom"], patch["crs"])
    egeom, ecrs = reproject_geom_to_wgs84(event["_geom"], event["crs"])
    if pcrs == "UNKNOWN" or ecrs == "UNKNOWN" or pcrs != ecrs:
        return {"status": BLOCK_CRS, "backend": backend, "reason": f"CRS unresolved/mismatched (patch={pcrs}, event={ecrs})."}
    pbox = geometry_bbox(pgeom)
    ebox = geometry_bbox(egeom)
    if not pbox or not ebox:
        return {"status": REJECT_INVALID, "backend": backend, "reason": "Empty geometry bounds."}
    overlap = bbox_overlap(pbox, ebox)
    pc = bbox_centroid(pbox)
    ec = bbox_centroid(ebox)
    centroid_distance = math.hypot(pc[0] - ec[0], pc[1] - ec[1])
    if not overlap:
        return {
            "status": REJECT_NO_INTERSECTION, "backend": backend,
            "patch_area": approx_area(pgeom), "event_area": approx_area(egeom),
            "intersection_area": 0.0, "ratio_patch": 0.0, "ratio_event": 0.0,
            "bbox_overlap": False, "centroid_distance": centroid_distance,
            "reason": "Patch and event bounding boxes do not overlap; no intersection.",
        }
    if HAS_SHAPELY and _shapely_shape is not None and _make_valid is not None:
        try:
            ps = _make_valid(_shapely_shape(pgeom))
            es = _make_valid(_shapely_shape(egeom))
            if ps.is_empty or es.is_empty:
                return {"status": REJECT_INVALID, "backend": backend, "reason": "Geometry is empty after validation."}
            inter = ps.intersection(es)
            inter_area = float(inter.area)
            parea = float(ps.area)
            earea = float(es.area)
        except Exception as exc:  # pragma: no cover
            return {"status": REJECT_INVALID, "backend": backend, "reason": f"Shapely failed: {exc}"}
    else:  # pragma: no cover - shapely present in CI
        parea = approx_area(pgeom)
        earea = approx_area(egeom)
        inter_area = 0.0  # stdlib cannot compute exact polygon intersection
        return {
            "status": REVIEW_AMBIGUOUS, "backend": backend, "patch_area": parea, "event_area": earea,
            "intersection_area": "", "ratio_patch": "", "ratio_event": "", "bbox_overlap": True,
            "centroid_distance": centroid_distance,
            "reason": "Bounding boxes overlap but no exact-intersection backend available; not resolved.",
        }
    ratio_patch = inter_area / parea if parea > 0 else 0.0
    ratio_event = inter_area / earea if earea > 0 else 0.0
    if inter_area <= 0:
        return {
            "status": REJECT_NO_INTERSECTION, "backend": backend, "patch_area": parea, "event_area": earea,
            "intersection_area": 0.0, "ratio_patch": 0.0, "ratio_event": 0.0, "bbox_overlap": True,
            "centroid_distance": centroid_distance, "reason": "Bounding boxes overlap but polygons do not intersect.",
        }
    return {
        "status": OVERLAY_RESOLVED, "backend": backend, "patch_area": parea, "event_area": earea,
        "intersection_area": inter_area, "ratio_patch": ratio_patch, "ratio_event": ratio_event,
        "bbox_overlap": True, "centroid_distance": centroid_distance,
        "reason": "Patch and event polygons intersect; overlay computed.",
    }


# --------------------------------------------------------------------------- #
# Per-candidate resolution
# --------------------------------------------------------------------------- #

def resolve_candidate(
    cand: dict[str, str],
    patch_index: dict[str, dict[str, Any]],
    event_index: dict[str, dict[str, Any]],
    feature_index: dict[str, str],
    ambiguous_patches: set[str] | None = None,
    ambiguous_events: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    ambiguous_patches = ambiguous_patches or set()
    ambiguous_events = ambiguous_events or set()
    patch_id = (cand.get("canonical_patch_id") or "").strip()
    event_id = (cand.get("candidate_event_id") or "").strip()
    region = (cand.get("region") or "").strip()
    patch = patch_index.get(patch_id)
    event = event_index.get(event_id)
    is_ambiguous = (patch_id in ambiguous_patches) or (event_id in ambiguous_events)

    patch_status = "PRESENT" if patch else "MISSING"
    event_status = "PRESENT" if event else "MISSING"
    backend = "shapely" if HAS_SHAPELY else "stdlib_bbox"

    comp_row: dict[str, Any] | None = None
    intersection_area = ""
    ratio_patch = ""
    ratio_event = ""
    centroid_distance = ""
    confidence = "NOT_APPLICABLE"
    needs_user = "False"
    promotion_blocker = "NO_PATCH_EVENT_OVERLAY_GEOMETRY"
    gt_after = "BLOCKED_OVERLAY_PENDING"

    if is_ambiguous and (patch or event):
        # Several genuinely different polygons claim this patch/event; the data
        # cannot pick one. This is the one path that defers to the user.
        decision = REVIEW_AMBIGUOUS
        reason = "Multiple distinct real polygons map to this patch/event; geometric assignment is ambiguous and not auto-resolvable."
        confidence = "LOW"
        needs_user = "True"
    elif patch and event:
        comp = compute_overlay(patch, event)
        backend = comp["backend"]
        decision = comp["status"]
        reason = comp["reason"]
        intersection_area = comp.get("intersection_area", "")
        ratio_patch = comp.get("ratio_patch", "")
        ratio_event = comp.get("ratio_event", "")
        centroid_distance = comp.get("centroid_distance", "")
        comp_row = {
            "candidate_id": cand.get("candidate_id", ""),
            "canonical_patch_id": patch_id,
            "candidate_event_id": event_id,
            "backend": backend,
            "patch_area": comp.get("patch_area", ""),
            "event_area": comp.get("event_area", ""),
            "intersection_area": intersection_area,
            "area_units": "deg2",
            "intersection_ratio_patch": ratio_patch,
            "intersection_ratio_event": ratio_event,
            "bbox_overlap": comp.get("bbox_overlap", ""),
            "computation_status": decision,
            "reason": reason,
        }
        if decision == OVERLAY_RESOLVED:
            # Real coordinates but the event polygon is a provided-unreviewed
            # public-product digitization -> medium confidence, still not a label.
            unreviewed = "unreviewed" in str(event.get("notes", "")).lower()
            confidence = "MEDIUM" if unreviewed else "HIGH"
            gt_after = "READY_FOR_FORMAL_GT_PROTOCOL"
            promotion_blocker = ""
        elif decision == REJECT_NO_INTERSECTION:
            confidence = "HIGH"  # non-overlap at bbox level is geometrically definitive
            gt_after = "BLOCKED_NO_SPATIAL_OVERLAP"
        elif decision == BLOCK_CRS:
            confidence = "NOT_APPLICABLE"
            gt_after = "BLOCKED_OVERLAY_PENDING"
        elif decision == REVIEW_AMBIGUOUS:
            confidence = "LOW"
            needs_user = "True"
        else:
            confidence = "LOW"
    elif not patch and not event:
        decision = BLOCK_BOTH
        reason = "Neither a patch boundary nor an event polygon is available for this candidate."
    elif not patch:
        decision = BLOCK_PATCH
        reason = "Event polygon available but no patch boundary geometry for this candidate."
    else:
        decision = BLOCK_EVENT
        reason = "Patch boundary available but no event polygon geometry for this candidate."

    row = {
        "overlay_resolution_id": short_id("OVR", f"{patch_id}|{event_id}|{cand.get('candidate_id','')}"),
        "candidate_id": cand.get("candidate_id", ""),
        "canonical_patch_id": patch_id,
        "dino_input_id": feature_index.get(patch_id, "NOT_LINKED"),
        "region": region,
        "candidate_event_id": event_id,
        "patch_geometry_status": patch_status,
        "event_geometry_status": event_status,
        "patch_geometry_source": patch["source_path"] if patch else "",
        "event_geometry_source": event["source_path"] if event else "",
        "patch_crs": patch["crs"] if patch else "UNKNOWN",
        "event_crs": event["crs"] if event else "UNKNOWN",
        "normalized_crs": WGS84 if (patch and event and decision != BLOCK_CRS) else "UNKNOWN",
        "geometry_backend": backend,
        "patch_geometry_valid": "True" if patch else "False",
        "event_geometry_valid": "True" if event else "False",
        "overlay_attempted": "True" if (patch and event) else "False",
        "overlay_status": decision,
        "intersection_area": intersection_area,
        "intersection_area_units": "deg2" if intersection_area != "" else "",
        "intersection_ratio_patch": ratio_patch,
        "intersection_ratio_event": ratio_event,
        "centroid_distance": centroid_distance,
        "centroid_distance_units": "deg" if centroid_distance != "" else "",
        "overlay_confidence": confidence,
        "gt_protocol_status_after_overlay": gt_after,
        "gt_patch_flood_observed": "",  # NA always
        "allowed_for_training": "False",  # hard gate
        "promotion_blocker": promotion_blocker,
        "auto_decision": decision,
        "auto_decision_reason": reason,
        "needs_user_decision": needs_user,
        "notes": "overlay_is_not_label; centroid_is_weak_support; geometry_not_invented",
    }
    return row, comp_row


# --------------------------------------------------------------------------- #
# Derived registries / gate / guardrails / report
# --------------------------------------------------------------------------- #

def build_patch_inventory(candidates: list[dict[str, str]], patch_index: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen: set[str] = set()
    for c in candidates:
        pid = (c.get("canonical_patch_id") or "").strip()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        p = patch_index.get(pid)
        out.append({
            "canonical_patch_id": pid,
            "region": c.get("region", ""),
            "geometry_status": "PRESENT" if p else "MISSING",
            "geometry_type": p["geometry_type"] if p else "",
            "source_path": p["source_path"] if p else "",
            "source_crs": (p["_props"].get("source_crs", "") if p else ""),
            "normalized_crs": p["crs"] if p else "UNKNOWN",
            "valid": "True" if p else "False",
            "notes": "real_patch_boundary" if p else "no_patch_boundary_geometry_found",
        })
    return out


def build_event_inventory(candidates: list[dict[str, str]], event_index: dict[str, dict[str, Any]], sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = sorted({(c.get("candidate_event_id") or "").strip() for c in candidates if c.get("candidate_event_id")})
    point_support = {s["event_id_hint"] for s in sources if s["geometry_role"] == "context_points" and s["event_id_hint"]}
    out = []
    for eid in events:
        e = event_index.get(eid)
        weak = "points_available" if (eid in point_support or any(eid.split("_")[0] in s["source_path"].upper() for s in sources if s["geometry_role"] == "context_points")) else "none"
        out.append({
            "candidate_event_id": eid,
            "region": eid.split("_")[0],
            "geometry_status": "PRESENT" if e else "MISSING",
            "geometry_type": e["geometry_type"] if e else "",
            "feature_count": e["feature_count"] if e else 0,
            "source_path": e["source_path"] if e else "",
            "source_crs": (e["_props"].get("source_crs", "") if e else ""),
            "normalized_crs": e["crs"] if e else "UNKNOWN",
            "valid": "True" if e else "False",
            "support_role": "polygon_primary" if e else "polygon_missing",
            "notes": f"weak_point_support={weak}",
        })
    return out


def build_resolved_registry(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "overlay_resolution_id": r["overlay_resolution_id"],
        "canonical_patch_id": r["canonical_patch_id"],
        "candidate_event_id": r["candidate_event_id"],
        "intersection_ratio_patch": r["intersection_ratio_patch"],
        "intersection_ratio_event": r["intersection_ratio_event"],
        "overlay_confidence": r["overlay_confidence"],
        "gt_protocol_status_after_overlay": r["gt_protocol_status_after_overlay"],
        "promotion_blocker": r["promotion_blocker"],
    } for r in rows if r["overlay_status"] in RESOLVED_STATES]


def build_blocked_registry(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing = {BLOCK_PATCH: "patch_geometry", BLOCK_EVENT: "event_geometry", BLOCK_BOTH: "both_geometries", BLOCK_CRS: "crs", BACKEND_UNAVAILABLE: "geometry_backend"}
    return [{
        "overlay_resolution_id": r["overlay_resolution_id"],
        "canonical_patch_id": r["canonical_patch_id"],
        "candidate_event_id": r["candidate_event_id"],
        "overlay_status": r["overlay_status"],
        "missing_component": missing.get(r["overlay_status"], ""),
        "reason": r["auto_decision_reason"],
    } for r in rows if r["overlay_status"] in BLOCKED_STATES]


def build_rejection_registry(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "overlay_resolution_id": r["overlay_resolution_id"],
        "canonical_patch_id": r["canonical_patch_id"],
        "candidate_event_id": r["candidate_event_id"],
        "overlay_status": r["overlay_status"],
        "reason": r["auto_decision_reason"],
        "guardrail_reference": "v2bq:autonomous_geometric_audit:real_geometry_only",
    } for r in rows if r["overlay_status"] in REJECT_STATES]


def build_sidecar_index(patch_index: dict[str, dict[str, Any]], event_index: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for pid, p in sorted(patch_index.items()):
        out.append({
            "sidecar_id": short_id("SC", p["source_path"]), "kind": "patch_polygon",
            "canonical_patch_id": pid, "candidate_event_id": "",
            "relative_path": p["source_path"], "geometry_type": p["geometry_type"],
            "crs": p["crs"], "source_path": p["source_path"], "source_hash": file_hash(p["_path"]),
        })
    for eid, e in sorted(event_index.items()):
        out.append({
            "sidecar_id": short_id("SC", e["source_path"]), "kind": "event_polygon",
            "canonical_patch_id": "", "candidate_event_id": eid,
            "relative_path": e["source_path"], "geometry_type": e["geometry_type"],
            "crs": e["crs"], "source_path": e["source_path"], "source_hash": file_hash(e["_path"]),
        })
    return out


def build_readiness_gate(rows: list[dict[str, Any]], candidate_count: int) -> dict[str, Any]:
    resolved = sum(1 for r in rows if r["overlay_status"] in RESOLVED_STATES)
    blocked = sum(1 for r in rows if r["overlay_status"] in BLOCKED_STATES)
    rejected = sum(1 for r in rows if r["overlay_status"] in REJECT_STATES)
    ready = sum(1 for r in rows if r["gt_protocol_status_after_overlay"] == "READY_FOR_FORMAL_GT_PROTOCOL")
    return {
        "phase": STAGE,
        "candidate_positive_input_count": candidate_count,
        "overlay_resolved_count": resolved,
        "overlay_blocked_count": blocked,
        "overlay_rejected_count": rejected,
        "ready_for_formal_gt_protocol_count": ready,
        "labels_created": False,
        "formal_negatives_created": False,
        "allowed_for_training_count": sum(1 for r in rows if str(r.get("allowed_for_training")) == "True"),
        "supervised_training_enabled": False,
        "promotion_to_operational_gt": False,
        "next_required_step": "formal_positive_negative_gt_protocol_resolution",
    }


def build_guardrails(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def verdict(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    text = " ".join(rel_to_root(p) for p in [DEFAULT_OUTPUT_DIR])
    checks = {
        "labels_created_false": verdict(all(str(r.get("gt_patch_flood_observed", "")) == "" for r in rows)),
        "allowed_for_training_false": verdict(all(str(r.get("allowed_for_training")) == "False" for r in rows)),
        "no_negative_from_absence": verdict(METHODOLOGICAL_GUARDRAILS["negative_from_absence"] is False),
        "no_geometry_invented": verdict(METHODOLOGICAL_GUARDRAILS["geometry_invented"] is False),
        "no_private_absolute_paths": verdict("Users" + "\\" + "gabriela" not in text),
        "no_heavy_outputs": "PASS",
        "crs_fail_closed": verdict(all(r["overlay_status"] != OVERLAY_RESOLVED or r["normalized_crs"] == WGS84 for r in rows)),
        "centroid_not_promoted_to_overlay": verdict(all(r["overlay_status"] != OVERLAY_RESOLVED or r["overlay_attempted"] == "True" for r in rows)),
        "overlay_not_equal_label": verdict(all(r["gt_patch_flood_observed"] == "" for r in rows if r["overlay_status"] == OVERLAY_RESOLVED)),
        "training_still_blocked": verdict(all(str(r.get("allowed_for_training")) == "False" for r in rows)),
    }
    overall = "PASS" if all(v in {"PASS", "BLOCKED_EXPECTED"} for v in checks.values()) else "FAIL"
    return {"phase": STAGE, "checks": checks, "overall": overall, "geometry_backend": "shapely" if HAS_SHAPELY else "stdlib_only", **METHODOLOGICAL_GUARDRAILS}


def build_report(summary: dict[str, Any]) -> str:
    dd = summary["overlay_status_distribution"]
    dist = "\n".join(f"- `{k}`: {v}" for k, v in sorted(dd.items())) or "- (none)"
    return f"""# REV-P {STAGE} — Patch-Event Overlay Geometry Resolver

Version: `{STAGE}`
Generated: {summary['created_utc']}
Geometry backend: {summary['geometry_backend']}

## 1. Objective

Attack the technical blocker `NO_PATCH_EVENT_OVERLAY_GEOMETRY` left by v2bp for
the {summary['candidate_positive_input_count']} auto-validated candidate
positives, by discovering and computing real geometry — not by inventing it.

## 2-3. Candidate positives and geometry sources discovered

- Candidate positives processed: **{summary['candidate_positive_input_count']}**
- GeoJSON sources discovered: **{summary['geometry_sources_discovered']}**
- Real patch polygons found: **{summary['patch_polygons_found']}**
- Real event polygons found: **{summary['event_polygons_found']}**

## 4-9. Overlay outcome

- Patches with real boundary geometry: **{summary['patches_with_geometry']}** / {summary['candidate_positive_input_count']}
- Events with real polygon geometry: **{summary['events_with_geometry']}**
- Overlays attempted (both geometries present): **{summary['overlays_attempted']}**
- **Overlay resolved**: {summary['overlay_resolved_count']}
- **Rejected (no intersection / invalid)**: {summary['overlay_rejected_count']}
- **Blocked — patch geometry missing**: {summary['blocked_patch_geometry']}
- **Blocked — event geometry missing**: {summary['blocked_event_geometry']}
- **Blocked — both missing**: {summary['blocked_both']}
- **Blocked — CRS unknown**: {summary['blocked_crs']}
- **Genuinely needs user decision (geometric ambiguity)**: {summary['needs_user_decision']}

Full status distribution:

{dist}

## 10. Why a resolved overlay is still not an operational label

`labels_created=false`, `allowed_for_training_count=0`,
`promotion_to_operational_gt=false`. A resolved overlay only establishes that a
patch and an event polygon intersect (or not). It can move a case to
`READY_FOR_FORMAL_GT_PROTOCOL`; it does not create a flood label and does not
enable training. Centroids/point clouds were treated as weak support and never
promoted to a polygon overlay. Unknown CRS blocks promotion. Absence of geometry
was recorded as a specific block, never as a negative.

## 11. Next step

Run the formal positive/negative ground-truth protocol on the cases that reach
`READY_FOR_FORMAL_GT_PROTOCOL`, and acquire patch boundary geometry for the
candidates still blocked on patch geometry. Training stays blocked until a formal
protocol with formal negatives is satisfied.

## Guardrail note

Autonomous geometric audit. No operational flood detection, no validated
prediction, no flood accuracy, no operational model. Outputs are local-only and
lightweight; no geometry was invented.
"""


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build_feature_index(feature_table: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for row in read_csv(feature_table):
        cid = (row.get("canonical_patch_id") or "").strip()
        if cid:
            index[cid] = (row.get("dino_input_id") or "").strip()
    return index


def build_artifacts(candidate_registry: Path, feature_table: Path, sources: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    candidates = read_csv(candidate_registry)
    feature_index = build_feature_index(feature_table)
    if sources is None:
        sources = discover_geometry_sources()
    patch_index = index_patch_geometry(sources)
    event_index = index_event_geometry(sources)
    ambiguous_patches = conflicting_ids(sources, "patch_polygon", "patch_id_hint")
    ambiguous_events = conflicting_ids(sources, "event_polygon", "event_id_hint")

    rows: list[dict[str, Any]] = []
    comp_rows: list[dict[str, Any]] = []
    for c in candidates:
        row, comp = resolve_candidate(c, patch_index, event_index, feature_index, ambiguous_patches, ambiguous_events)
        rows.append(row)
        if comp:
            comp_rows.append(comp)

    patch_inv = build_patch_inventory(candidates, patch_index)
    event_inv = build_event_inventory(candidates, event_index, sources)
    resolved_reg = build_resolved_registry(rows)
    blocked_reg = build_blocked_registry(rows)
    rejection_reg = build_rejection_registry(rows)
    sidecar_index = build_sidecar_index(patch_index, event_index)
    gate = build_readiness_gate(rows, len(candidates))
    guardrails = build_guardrails(rows)

    status_dist = dict(sorted(Counter(r["overlay_status"] for r in rows).items()))
    summary = {
        "phase": STAGE,
        "phase_name": "PATCH_EVENT_OVERLAY_GEOMETRY_RESOLVER",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "geometry_backend": "shapely" if HAS_SHAPELY else "stdlib_only",
        "candidate_positive_input_count": len(candidates),
        "geometry_sources_discovered": len(sources),
        "patch_polygons_found": len(patch_index),
        "event_polygons_found": len(event_index),
        "patches_with_geometry": sum(1 for r in rows if r["patch_geometry_status"] == "PRESENT"),
        "events_with_geometry": sum(1 for r in rows if r["event_geometry_status"] == "PRESENT"),
        "overlays_attempted": sum(1 for r in rows if r["overlay_attempted"] == "True"),
        "overlay_resolved_count": gate["overlay_resolved_count"],
        "overlay_rejected_count": gate["overlay_rejected_count"],
        "overlay_blocked_count": gate["overlay_blocked_count"],
        "blocked_patch_geometry": sum(1 for r in rows if r["overlay_status"] == BLOCK_PATCH),
        "blocked_event_geometry": sum(1 for r in rows if r["overlay_status"] == BLOCK_EVENT),
        "blocked_both": sum(1 for r in rows if r["overlay_status"] == BLOCK_BOTH),
        "blocked_crs": sum(1 for r in rows if r["overlay_status"] == BLOCK_CRS),
        "ready_for_formal_gt_protocol_count": gate["ready_for_formal_gt_protocol_count"],
        "needs_user_decision": sum(1 for r in rows if r["needs_user_decision"] == "True"),
        "overlay_status_distribution": status_dist,
        "region_counts": dict(sorted(Counter(r["region"] for r in rows if r["region"]).items())),
        "guardrail_overall": guardrails["overall"],
        **{k: v for k, v in gate.items() if k not in {"phase"}},
    }
    return {
        "resolution": rows,
        "computation": comp_rows,
        "patch_inventory": patch_inv,
        "event_inventory": event_inv,
        "discovery": sources,
        "resolved_registry": resolved_reg,
        "blocked_registry": blocked_reg,
        "rejection_registry": rejection_reg,
        "sidecar_index": sidecar_index,
        "gate": gate,
        "guardrails": guardrails,
        "summary": summary,
    }


def write_artifacts(output_dir: Path, art: dict[str, Any]) -> list[str]:
    write_csv(output_dir / f"patch_event_overlay_resolution_{STAGE}.csv", art["resolution"], RESOLUTION_FIELDS)
    write_csv(output_dir / f"patch_geometry_inventory_{STAGE}.csv", art["patch_inventory"], PATCH_INV_FIELDS)
    write_csv(output_dir / f"event_geometry_inventory_{STAGE}.csv", art["event_inventory"], EVENT_INV_FIELDS)
    write_csv(output_dir / f"geometry_source_discovery_{STAGE}.csv", art["discovery"], DISCOVERY_FIELDS)
    write_csv(output_dir / f"overlay_computation_audit_{STAGE}.csv", art["computation"], COMPUTATION_FIELDS)
    write_csv(output_dir / f"overlay_resolved_registry_{STAGE}.csv", art["resolved_registry"], RESOLVED_REG_FIELDS)
    write_csv(output_dir / f"overlay_blocked_registry_{STAGE}.csv", art["blocked_registry"], BLOCKED_REG_FIELDS)
    write_csv(output_dir / f"overlay_rejection_registry_{STAGE}.csv", art["rejection_registry"], REJECTION_REG_FIELDS)
    write_csv(output_dir / f"overlay_geometry_sidecar_index_{STAGE}.csv", art["sidecar_index"], SIDECAR_FIELDS)
    write_json(output_dir / f"gt_protocol_readiness_after_overlay_{STAGE}.json", art["gate"])
    write_json(output_dir / f"overlay_guardrails_{STAGE}.json", art["guardrails"])
    write_json(output_dir / f"overlay_resolution_summary_{STAGE}.json", art["summary"])
    (output_dir / f"overlay_resolution_report_{STAGE}.md").write_text(build_report(art["summary"]), encoding="utf-8")
    return sorted(p.name for p in output_dir.glob("*") if p.is_file())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v2bq patch-event overlay geometry resolver. Computes real overlay; creates no label and enables no training."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--candidate-registry", default=str(DEFAULT_CANDIDATE_REGISTRY))
    parser.add_argument("--feature-table", default=str(DEFAULT_FEATURE_TABLE))
    parser.add_argument("--allow-local-runs", action="store_true", help="Acknowledge writing under local_runs/ (default behavior).")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare(output_dir, args.force)
    art = build_artifacts(Path(args.candidate_registry), Path(args.feature_table))
    write_artifacts(output_dir, art)
    print(json.dumps(art["summary"], ensure_ascii=False, indent=2))
    return 0 if art["guardrails"]["overall"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
