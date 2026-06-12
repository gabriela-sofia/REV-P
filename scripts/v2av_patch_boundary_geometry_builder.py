#!/usr/bin/env python3
"""v2av - Patch Boundary Source Manifest + Patch Geometry Builder.

Attacks the dominant v2au blocker ``BLOCKED_MISSING_PATCH_GEOMETRY``: there are
172 event-patch packages but no patch boundary vector exists in the repository.

This engine:
  1. discovers every patch_id used across the REV-P registries;
  2. traces the spatial provenance of each patch (manifests/, datasets/, ...);
  3. looks for spatial metadata sufficient to reconstruct a boundary
     (bbox, WKT, GeoJSON, raster bounds, tile bounds, center+size);
  4. emits a patch boundary source manifest;
  5. builds GeoJSON/WKT/bbox boundaries WHEN (and only when) there is enough
     metadata AND a known CRS;
  6. marks everything else as BLOCKED (fail-closed), never inventing geometry;
  7. produces a prioritised recovery/digitization queue.

Hard methodological line (never crossed):
  - no final ground truth, no binary/operational label, no model training;
  - absence of metadata is never a negative; it is a blocker;
  - geometry is never invented; no CRS => no boundary;
  - a center point never becomes a boundary unless explicitly opted in;
  - an unverified default patch size is never used to fabricate a boundary;
  - the v2at and v2au registries are never overwritten (only v2av deltas).

A built patch boundary is a geometric artefact for v2au to overlay; it is never an
event label or ground truth. The maximum downstream meaning is
``READY_FOR_V2AU_OVERLAY`` — a candidate, pending the v2au overlay + human review.

Offline, deterministic; outputs sorted by stable keys; stable hashes for ids;
exit code 0 even with expected blockers, non-zero only on a structural error.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys

STAGE = "v2av"
METHODOLOGICAL_STATUS = "PATCH_BOUNDARY_RECOVERY_READY_FOR_OVERLAY_NOT_FOR_TRAINING"

THIS_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))

UNKNOWN = "UNKNOWN"
NOT_AVAILABLE = "NOT_AVAILABLE"
EARTH_RADIUS_M = 6378137.0


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def resolve_dirs():
    dataset_dir = os.environ.get("DATASET_DIR") or project_path("datasets")
    output_dir = os.environ.get("OUTPUT_DIR") or project_path("outputs_public")
    config_dir = os.environ.get("CONFIG_DIR") or project_path("configs")
    return dataset_dir, output_dir, config_dir


CONFIG_NAME = "v2av_patch_boundary_geometry_builder_config.json"

DEFAULT_CONFIG = {
    "accepted_crs": ["EPSG:4326", "EPSG:3857", "EPSG:31982", "EPSG:31983"],
    "target_crs": "EPSG:3857",
    "allow_center_point_buffer": False,
    "default_patch_size_meters": 640.0,
    "allow_default_patch_size": False,
    "strict_crs": True,
    "offline_mode": True,
    "fail_on_missing_optional_patch_geometry": False,
    "minimum_required_fields_for_boundary": ["crs", "geometry_payload"],
    "search_paths": ["manifests", "datasets", "outputs_public", "docs"],
}

# Output filenames -----------------------------------------------------------

OUT_MANIFEST = "v2av_patch_boundary_source_manifest.csv"
OUT_REGISTRY = "v2av_patch_boundary_geometry_registry.csv"
OUT_AUDIT = "v2av_patch_boundary_build_audit.csv"
OUT_QUEUE = "v2av_patch_boundary_recovery_queue.csv"
GEOJSON_SUBDIR = os.path.join("geometries", "patch_boundaries")

REPORT_REL = os.path.join("execution_reports", "v2av_patch_boundary_geometry_builder_report.md")
SUMMARY_REL = os.path.join("execution_reports", "v2av_patch_boundary_geometry_builder_summary.json")
SUPPLEMENT_REL = os.path.join("execution_reports", "v2av_artifact_index_supplement.md")
LOG_REL = os.path.join("logs_summary", "v2av_patch_boundary_geometry_builder.txt")

# Input filenames (relative to dataset_dir) ----------------------------------

IN_PACKAGES = "v2at_event_patch_package_registry.csv"
IN_PATCH_RESOLUTION = os.path.join("protocolo_c", "v1us_patch_registry_resolution.csv")
# Optional manifest of provided patch geometry sources (absent in the offline repo).
IN_GEOMETRY_SOURCES = "v2av_patch_geometry_sources.csv"

REGION_NAME = {"REC": "Recife", "PET": "Petropolis", "CUR": "Curitiba"}

BUILD_METHODS = ("from_bbox", "from_wkt", "from_geojson", "from_raster_bounds",
                 "from_tile_bounds", "from_center_buffer", "NONE")

GATE_NAMES = [
    "PATCH_BOUNDARY_GATE_01_PATCH_ID_EXISTS", "PATCH_BOUNDARY_GATE_02_SOURCE_METADATA_FOUND",
    "PATCH_BOUNDARY_GATE_03_CRS_EXISTS", "PATCH_BOUNDARY_GATE_04_CRS_ACCEPTED",
    "PATCH_BOUNDARY_GATE_05_BOUNDARY_METHOD_ALLOWED", "PATCH_BOUNDARY_GATE_06_GEOMETRY_VALID",
    "PATCH_BOUNDARY_GATE_07_AREA_COMPUTABLE", "PATCH_BOUNDARY_GATE_08_NOT_BUILT_FROM_UNVERIFIED_DEFAULT",
    "PATCH_BOUNDARY_GATE_09_NO_EVENT_LABEL_CREATED", "PATCH_BOUNDARY_GATE_10_READY_FOR_V2AU_OVERLAY",
]

COLUMNS = {
    OUT_MANIFEST: [
        "patch_id", "region", "city", "source_file", "source_field", "source_type",
        "has_bbox", "has_wkt", "has_geojson", "has_raster_transform", "has_center_point",
        "has_resolution", "has_crs", "crs", "can_build_boundary", "boundary_build_method",
        "blocking_reason", "notes",
    ],
    OUT_REGISTRY: [
        "patch_geometry_id", "patch_id", "region", "city", "geometry_type",
        "geometry_format", "crs", "crs_status", "geometry_wkt", "geometry_geojson_path",
        "bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy", "area_m2", "geometry_hash",
        "build_method", "source_file", "source_confidence", "is_valid_geometry",
        "blocking_reason", "notes",
    ],
    OUT_AUDIT: [
        "decision_id", "patch_id", "gate_name", "gate_passed", "gate_status",
        "required_condition", "observed_value", "severity", "blocking_reason",
        "recommended_action",
    ],
    OUT_QUEUE: [
        "review_item_id", "patch_id", "region", "city", "priority_rank", "priority_reason",
        "missing_fields", "suggested_recovery_action", "candidate_source_files",
        "is_needed_by_packages_count", "is_recife_priority", "blocking_reason",
    ],
}

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


def _b(value):
    return "true" if value else "false"


# --------------------------------------------------------------------------- #
# Geometry kernel (pure Python, offline; same conventions as v2au).
# --------------------------------------------------------------------------- #


def normalise_crs(raw):
    raw = clean(raw).upper().replace(" ", "")
    if not raw:
        return ""
    if raw.isdigit():
        return f"EPSG:{raw}"
    return raw


def _parse_floats(text):
    out = []
    for token in clean(text).replace(",", " ").split():
        try:
            out.append(float(token))
        except ValueError:
            return None
    return out


def _close_ring(points):
    ring = [(float(x), float(y)) for x, y in points]
    if len(ring) < 3:
        return None
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def _bbox_ring(minx, miny, maxx, maxy):
    return [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]


def parse_geometry_payload(source_type, value, *, center_lat=None, center_lon=None,
                           size_meters=None, allow_center_buffer=False):
    """Return (build_method, ring) or (None, None). Never invents coordinates."""
    source_type = clean(source_type).lower()
    value = clean(value)
    try:
        if source_type in ("bbox", "raster_bounds", "tile_bounds"):
            nums = _parse_floats(value)
            if not nums or len(nums) < 4:
                return None, None
            ring = _bbox_ring(nums[0], nums[1], nums[2], nums[3])
            method = {"bbox": "from_bbox", "raster_bounds": "from_raster_bounds",
                      "tile_bounds": "from_tile_bounds"}[source_type]
            return method, ring
        if source_type == "wkt":
            ring = _parse_wkt_polygon(value)
            return ("from_wkt", ring) if ring else (None, None)
        if source_type in ("geojson_inline", "geojson"):
            ring = _parse_geojson_polygon(json.loads(value))
            return ("from_geojson", ring) if ring else (None, None)
        if source_type == "center_point":
            if not allow_center_buffer:
                return None, None
            if center_lat is None or center_lon is None or not size_meters:
                return None, None
            half = float(size_meters) / 2.0
            cx, cy = float(center_lon), float(center_lat)
            return "from_center_buffer", _bbox_ring(cx - half, cy - half, cx + half, cy + half)
    except (ValueError, json.JSONDecodeError, TypeError):
        return None, None
    return None, None


def _parse_wkt_polygon(text):
    upper = clean(text).upper()
    if not upper.startswith("POLYGON"):
        return None
    try:
        inner = text[text.index("((") + 2: text.index("))")]
    except ValueError:
        return None
    pts = []
    for pair in inner.split(","):
        nums = _parse_floats(pair)
        if not nums or len(nums) < 2:
            return None
        pts.append((nums[0], nums[1]))
    return _close_ring(pts)


def _parse_geojson_polygon(obj):
    geom = obj.get("geometry", obj) if isinstance(obj, dict) else None
    if not isinstance(geom, dict):
        return None
    gtype = clean(geom.get("type")).lower()
    coords = geom.get("coordinates")
    if gtype == "polygon" and isinstance(coords, list) and coords:
        return _close_ring([(c[0], c[1]) for c in coords[0]])
    if gtype == "multipolygon" and isinstance(coords, list) and coords:
        return _close_ring([(c[0], c[1]) for c in coords[0][0]])
    return None


def _mercator(lon, lat):
    x = EARTH_RADIUS_M * math.radians(lon)
    lat = max(min(lat, 89.9), -89.9)
    y = EARTH_RADIUS_M * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return (x, y)


def ring_to_metric(ring, crs):
    if crs in ("EPSG:3857", "EPSG:31982", "EPSG:31983"):
        return ring
    if crs == "EPSG:4326":
        return [_mercator(x, y) for (x, y) in ring]
    return None


def shoelace_area(ring):
    area = 0.0
    for i in range(len(ring) - 1):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def bbox_of(ring):
    xs = [x for x, _ in ring]
    ys = [y for _, y in ring]
    return (min(xs), min(ys), max(xs), max(ys))


def ring_to_wkt(ring):
    return "POLYGON((" + ", ".join(f"{x:.6f} {y:.6f}" for x, y in ring) + "))"


def ring_to_geojson(ring, crs):
    return {
        "type": "Feature",
        "properties": {"crs": crs, "stage": "v2av",
                       "note": "Patch boundary geometry for v2au overlay; not a label/ground truth."},
        "geometry": {"type": "Polygon", "coordinates": [[[round(x, 6), round(y, 6)] for x, y in ring]]},
    }


# --------------------------------------------------------------------------- #
# Patch universe discovery + spatial-source tracing.
# --------------------------------------------------------------------------- #


def discover_patches(inputs):
    """Return ordered list of {patch_id, region, city, source_file, package_count}."""
    patches = {}

    # Authoritative universe: the patch registry resolution (region/city per patch).
    res_file = "protocolo_c/v1us_patch_registry_resolution.csv"
    for row in inputs["patch_resolution"]:
        pid = clean(row.get("patch_id"))
        if not pid:
            continue
        patches.setdefault(pid, {
            "patch_id": pid, "region": region_from_code(row.get("region")),
            "city": clean(row.get("city")) or region_from_code(row.get("region")),
            "source_file": res_file,
            "has_patch_geometry_flag": clean(row.get("has_patch_geometry")).lower() == "true",
            "has_patch_bounds_flag": clean(row.get("has_patch_bounds")).lower() == "true",
            "package_count": 0, "candidate_reference": False,
        })

    # Patches referenced by v2at packages (add any missing; count references).
    for pkg in inputs["packages"]:
        pid = clean(pkg.get("patch_id"))
        if not pid or pid == "UNKNOWN_PATCH":
            continue
        entry = patches.setdefault(pid, {
            "patch_id": pid, "region": clean(pkg.get("region")) or UNKNOWN,
            "city": clean(pkg.get("city")) or clean(pkg.get("region")) or UNKNOWN,
            "source_file": IN_PACKAGES, "has_patch_geometry_flag": False,
            "has_patch_bounds_flag": False, "package_count": 0, "candidate_reference": False,
        })
        entry["package_count"] += 1
        if clean(pkg.get("allowed_use")) == "candidate_reference":
            entry["candidate_reference"] = True

    return sorted(patches.values(), key=lambda p: (p["region"], p["patch_id"]))


def index_geometry_sources(inputs):
    """Map patch_id -> provided geometry source row (optional manifest)."""
    by_patch = {}
    for src in inputs["geometry_sources"]:
        pid = clean(src.get("patch_id"))
        if pid:
            by_patch.setdefault(pid, src)
    return by_patch


# --------------------------------------------------------------------------- #
# Per-patch build attempt.
# --------------------------------------------------------------------------- #


def _evaluate_patch(patch, source, config):
    """Returns a dict with manifest + registry + build facts for one patch."""
    accepted = set(config["accepted_crs"])
    allow_center = bool(config["allow_center_point_buffer"])
    allow_default_size = bool(config["allow_default_patch_size"])

    facts = {
        "has_bbox": False, "has_wkt": False, "has_geojson": False,
        "has_raster_transform": False, "has_center_point": False, "has_resolution": False,
        "has_crs": False, "crs": UNKNOWN, "source_file": patch["source_file"],
        "source_field": "", "source_type": "none", "build_method": "NONE",
        "can_build": False, "blocking_reason": "NO_SPATIAL_METADATA_FOUND",
        "ring": None, "crs_status": "UNKNOWN", "source_confidence": "NONE",
        "size_meters": "", "center": False,
    }

    if source is None:
        return facts

    source_type = clean(source.get("source_type")).lower() or clean(source.get("geometry_format")).lower()
    value = clean(source.get("geometry_value"))
    crs = normalise_crs(source.get("crs"))
    facts["source_file"] = clean(source.get("source_file")) or IN_GEOMETRY_SOURCES
    facts["source_field"] = clean(source.get("source_field")) or "geometry_value"
    facts["source_type"] = source_type or "none"
    facts["crs"] = crs or UNKNOWN
    facts["has_crs"] = bool(crs)
    facts["crs_status"] = "KNOWN" if crs in accepted else "UNKNOWN"
    facts["source_confidence"] = clean(source.get("source_confidence")) or "PROVIDED"
    facts["size_meters"] = clean(source.get("size_meters"))

    facts["has_bbox"] = source_type in ("bbox", "raster_bounds", "tile_bounds")
    facts["has_wkt"] = source_type == "wkt"
    facts["has_geojson"] = source_type in ("geojson_inline", "geojson", "geojson_file")
    facts["has_raster_transform"] = source_type in ("raster_bounds", "raster_transform")
    facts["has_center_point"] = source_type == "center_point"
    facts["has_resolution"] = bool(facts["size_meters"]) or facts["has_raster_transform"]
    facts["center"] = source_type == "center_point"

    # Geometry payload (read GeoJSON file when referenced).
    if source_type == "geojson_file":
        gpath = clean(source.get("geometry_path"))
        abs_path = gpath if os.path.isabs(gpath) else os.path.join(PROJECT_ROOT, gpath)
        ring = None
        if gpath and os.path.exists(abs_path):
            try:
                with open(abs_path, encoding="utf-8") as handle:
                    ring = _parse_geojson_polygon(json.load(handle))
            except (OSError, json.JSONDecodeError):
                ring = None
        method = "from_geojson" if ring else None
    else:
        method, ring = parse_geometry_payload(
            source_type, value, center_lat=source.get("center_lat"),
            center_lon=source.get("center_lon"), size_meters=facts["size_meters"] or None,
            allow_center_buffer=allow_center)

    # Fail-closed gates, in order.
    if not crs:
        facts["blocking_reason"] = "MISSING_CRS"
        return facts
    if crs not in accepted:
        facts["blocking_reason"] = "UNACCEPTED_OR_UNKNOWN_CRS"
        return facts
    if facts["center"] and not allow_center:
        facts["blocking_reason"] = "CENTER_POINT_BUFFER_NOT_ALLOWED"
        return facts
    if facts["center"] and not facts["size_meters"] and not allow_default_size:
        facts["blocking_reason"] = "DEFAULT_PATCH_SIZE_NOT_ALLOWED"
        return facts
    if method is None or ring is None:
        facts["blocking_reason"] = "UNPARSEABLE_OR_INSUFFICIENT_GEOMETRY"
        return facts

    metric = ring_to_metric(ring, crs)
    area = shoelace_area(metric) if metric is not None else 0.0
    if metric is None or area <= 0:
        facts["blocking_reason"] = "NON_COMPUTABLE_OR_DEGENERATE_AREA"
        facts["ring"] = ring
        facts["build_method"] = method
        return facts

    facts["ring"] = ring
    facts["build_method"] = method
    facts["area_m2"] = area
    facts["can_build"] = True
    facts["blocking_reason"] = ""
    return facts


# --------------------------------------------------------------------------- #
# Builders for each output.
# --------------------------------------------------------------------------- #


def build_all(patches, source_index, config, dataset_dir):
    manifest_rows, registry_rows, audit_rows, queue_rows = [], [], [], []
    geojson_written = []
    geojson_dir = os.path.join(dataset_dir, GEOJSON_SUBDIR)

    for patch in patches:
        source = source_index.get(patch["patch_id"])
        facts = _evaluate_patch(patch, source, config)
        pid = patch["patch_id"]

        manifest_rows.append({
            "patch_id": pid, "region": patch["region"], "city": patch["city"],
            "source_file": facts["source_file"], "source_field": facts["source_field"],
            "source_type": facts["source_type"], "has_bbox": _b(facts["has_bbox"]),
            "has_wkt": _b(facts["has_wkt"]), "has_geojson": _b(facts["has_geojson"]),
            "has_raster_transform": _b(facts["has_raster_transform"]),
            "has_center_point": _b(facts["has_center_point"]),
            "has_resolution": _b(facts["has_resolution"]), "has_crs": _b(facts["has_crs"]),
            "crs": facts["crs"], "can_build_boundary": _b(facts["can_build"]),
            "boundary_build_method": facts["build_method"],
            "blocking_reason": facts["blocking_reason"],
            "notes": "Spatial provenance traced; geometry never invented.",
        })

        registry_rows.append(_registry_row(patch, facts, geojson_dir, geojson_written))
        audit_rows.extend(_audit_rows(patch, facts, config))
        queue_rows.append(_queue_row(patch, facts))

    manifest_rows.sort(key=lambda r: (r["region"], r["patch_id"]))
    registry_rows.sort(key=lambda r: (r["region"], r["patch_id"]))
    audit_rows.sort(key=lambda r: (r["patch_id"], r["gate_name"]))
    queue_rows.sort(key=lambda r: (int(r["priority_rank"]),
                                   -int(r["is_needed_by_packages_count"] or 0), r["patch_id"]))
    return manifest_rows, registry_rows, audit_rows, queue_rows, geojson_written


def _registry_row(patch, facts, geojson_dir, geojson_written):
    pid = patch["patch_id"]
    geojson_path = ""
    wkt = ""
    bbox = ("", "", "", "")
    area = ""
    geom_hash = "INVALID"
    geom_type = UNKNOWN
    geom_format = UNKNOWN
    valid = False

    if facts["can_build"] and facts["ring"] is not None:
        ring = facts["ring"]
        crs = facts["crs"]
        bbox = bbox_of(ring)
        wkt = ring_to_wkt(ring)
        area = f"{facts['area_m2']:.2f}"
        geom_hash = hashlib.sha1(repr((ring, crs)).encode("utf-8")).hexdigest()[:16]
        geom_type = "polygon"
        geom_format = "polygon_wkt_and_geojson"
        valid = True
        # Write a committable GeoJSON next to the dataset (only when built).
        rel = os.path.join(GEOJSON_SUBDIR, f"patch_boundary_{pid}.geojson")
        write_text(os.path.join(geojson_dir, f"patch_boundary_{pid}.geojson"),
                   json.dumps(ring_to_geojson(ring, crs), indent=2, ensure_ascii=False) + "\n")
        geojson_path = f"datasets/{rel.replace(os.sep, '/')}"
        geojson_written.append(geojson_path)

    return {
        "patch_geometry_id": stable_id("PBG_", pid, facts["build_method"], geom_hash),
        "patch_id": pid, "region": patch["region"], "city": patch["city"],
        "geometry_type": geom_type, "geometry_format": geom_format,
        "crs": facts["crs"], "crs_status": facts["crs_status"], "geometry_wkt": wkt,
        "geometry_geojson_path": geojson_path or NOT_AVAILABLE,
        "bbox_minx": _fmt(bbox[0]), "bbox_miny": _fmt(bbox[1]),
        "bbox_maxx": _fmt(bbox[2]), "bbox_maxy": _fmt(bbox[3]),
        "area_m2": area or "0.00", "geometry_hash": geom_hash,
        "build_method": facts["build_method"], "source_file": facts["source_file"],
        "source_confidence": facts["source_confidence"], "is_valid_geometry": _b(valid),
        "blocking_reason": facts["blocking_reason"],
        "notes": "Patch boundary geometry for v2au overlay; never an event label or ground truth.",
    }


def _fmt(value):
    if value == "":
        return ""
    return f"{float(value):.6f}"


def _audit_rows(patch, facts, config):
    pid = patch["patch_id"]
    accepted = set(config["accepted_crs"])
    method_allowed = facts["build_method"] in BUILD_METHODS and facts["build_method"] != "NONE"
    is_default_size = facts["build_method"] == "from_center_buffer" and not facts["size_meters"] \
        and not config["allow_default_patch_size"]

    g = {
        "PATCH_BOUNDARY_GATE_01_PATCH_ID_EXISTS": (
            bool(pid), "typed patch_id present", pid, "BLOCKING"),
        "PATCH_BOUNDARY_GATE_02_SOURCE_METADATA_FOUND": (
            facts["source_type"] != "none", "spatial source metadata found",
            facts["source_type"], "BLOCKING"),
        "PATCH_BOUNDARY_GATE_03_CRS_EXISTS": (
            facts["has_crs"], "CRS present", facts["crs"], "BLOCKING"),
        "PATCH_BOUNDARY_GATE_04_CRS_ACCEPTED": (
            facts["crs"] in accepted, "CRS in accepted list", facts["crs"], "BLOCKING"),
        "PATCH_BOUNDARY_GATE_05_BOUNDARY_METHOD_ALLOWED": (
            method_allowed, "a build method is available/allowed",
            facts["build_method"], "BLOCKING"),
        "PATCH_BOUNDARY_GATE_06_GEOMETRY_VALID": (
            facts["can_build"], "geometry parses to a valid polygon",
            _b(facts["can_build"]), "BLOCKING"),
        "PATCH_BOUNDARY_GATE_07_AREA_COMPUTABLE": (
            facts["can_build"], "polygon area computable", _b(facts["can_build"]), "BLOCKING"),
        "PATCH_BOUNDARY_GATE_08_NOT_BUILT_FROM_UNVERIFIED_DEFAULT": (
            not is_default_size, "boundary not fabricated from an unverified default size",
            "not_default" if not is_default_size else "default_blocked", "GUARDRAIL"),
        "PATCH_BOUNDARY_GATE_09_NO_EVENT_LABEL_CREATED": (
            True, "no event label / ground truth created", "no_label_created", "GUARDRAIL"),
        "PATCH_BOUNDARY_GATE_10_READY_FOR_V2AU_OVERLAY": (
            facts["can_build"], "patch boundary ready for v2au overlay",
            _b(facts["can_build"]), "INFO"),
    }
    rows = []
    for gate, (passed, required, observed, severity) in g.items():
        rows.append({
            "decision_id": stable_id("PBD_", pid, gate), "patch_id": pid,
            "gate_name": gate, "gate_passed": _b(passed),
            "gate_status": "PASS" if passed else ("PENDING" if severity == "INFO" else "FAIL"),
            "required_condition": required, "observed_value": observed, "severity": severity,
            "blocking_reason": "" if passed or severity != "BLOCKING" else facts["blocking_reason"]
            or "GATE_NOT_SATISFIED",
            "recommended_action": "None" if passed else _recommend(gate),
        })
    return rows


def _recommend(gate):
    if gate in ("PATCH_BOUNDARY_GATE_02_SOURCE_METADATA_FOUND",
                "PATCH_BOUNDARY_GATE_05_BOUNDARY_METHOD_ALLOWED",
                "PATCH_BOUNDARY_GATE_06_GEOMETRY_VALID",
                "PATCH_BOUNDARY_GATE_07_AREA_COMPUTABLE",
                "PATCH_BOUNDARY_GATE_10_READY_FOR_V2AU_OVERLAY"):
        return "Acquire/digitize patch boundary geometry (bbox/WKT/GeoJSON) with CRS"
    if gate in ("PATCH_BOUNDARY_GATE_03_CRS_EXISTS", "PATCH_BOUNDARY_GATE_04_CRS_ACCEPTED"):
        return "Resolve the patch CRS to an accepted EPSG code"
    return "Review patch geometry provenance"


def _queue_row(patch, facts):
    pid = patch["patch_id"]
    region = patch["region"]
    count = patch["package_count"]
    recife_priority = region == "Recife" and patch["candidate_reference"]

    if facts["can_build"]:
        rank, reason = 6, "Boundary already built; ready for v2au overlay"
    elif recife_priority:
        rank, reason = 1, "Recife patch used by a v2at candidate_reference package"
    elif count >= 2:
        rank, reason = 2, "Patch referenced by multiple packages"
    elif facts["source_type"] != "none":
        rank, reason = 3, "Partial spatial metadata present; recoverable"
    elif region == "Petropolis":
        rank, reason = 4, "Petropolis patch pending geometry"
    else:
        rank, reason = 5, "Curitiba/context-only patch pending geometry"

    missing = []
    if not facts["has_crs"]:
        missing.append("crs")
    if facts["source_type"] == "none":
        missing.append("spatial_geometry_source")
    if not (facts["has_bbox"] or facts["has_wkt"] or facts["has_geojson"]):
        missing.append("bbox_or_wkt_or_geojson")

    return {
        "review_item_id": stable_id("PBRQ_", pid, length=10), "patch_id": pid,
        "region": region, "city": patch["city"], "priority_rank": str(rank),
        "priority_reason": reason, "missing_fields": "|".join(missing) or "none",
        "suggested_recovery_action": "Digitize/acquire patch boundary polygon (CRS-verified) "
                                     "from the patch Sentinel tile/footprint",
        "candidate_source_files": facts["source_file"],
        "is_needed_by_packages_count": str(count),
        "is_recife_priority": _b(recife_priority),
        "blocking_reason": facts["blocking_reason"],
    }


# --------------------------------------------------------------------------- #
# Validation.
# --------------------------------------------------------------------------- #


def validate_outputs(written):
    errors = []
    for name, col in ((OUT_MANIFEST, "patch_id"), (OUT_REGISTRY, "patch_geometry_id"),
                      (OUT_AUDIT, "decision_id"), (OUT_QUEUE, "review_item_id")):
        for row in written[name]:
            if not clean(row.get(col)):
                errors.append(f"{name}: empty generated id in {col}")
                break

    for row in written[OUT_REGISTRY]:
        if row["build_method"] not in BUILD_METHODS:
            errors.append(f"registry {row['patch_geometry_id']} invalid build_method {row['build_method']}")
            break
        # A built boundary must never be tagged as an event label / ground truth.
        if "LABEL" in row["notes"].upper() and "NEVER" not in row["notes"].upper():
            errors.append(f"registry {row['patch_geometry_id']} suspicious label note")
            break

    g09 = [r for r in written[OUT_AUDIT] if r["gate_name"] == "PATCH_BOUNDARY_GATE_09_NO_EVENT_LABEL_CREATED"]
    if not g09 or not all(r["gate_passed"] == "true" for r in g09):
        errors.append("GATE_09 (no event label) did not pass for all patches")
    return errors


# --------------------------------------------------------------------------- #
# Report / summary / log.
# --------------------------------------------------------------------------- #


def build_summary(inputs_found, outputs_written, written, geojson_written):
    manifest = written[OUT_MANIFEST]
    registry = written[OUT_REGISTRY]
    built = sum(1 for r in registry if r["is_valid_geometry"] == "true")
    blocked = len(registry) - built
    block_dist = {}
    for r in registry:
        if r["blocking_reason"]:
            block_dist[r["blocking_reason"]] = block_dist.get(r["blocking_reason"], 0) + 1
    return {
        "stage": STAGE, "status": "OK_WITH_EXPECTED_BLOCKERS",
        "inputs_found": inputs_found, "outputs_written": outputs_written,
        "total_unique_patches": len(manifest),
        "patches_with_spatial_metadata": sum(1 for m in manifest if m["source_type"] != "none"),
        "patches_with_crs": sum(1 for m in manifest if m["has_crs"] == "true"),
        "patch_boundaries_built": built, "patch_boundaries_blocked": blocked,
        "geojson_files_written": len(geojson_written),
        "ready_for_v2au_overlay": built,
        "blocking_reason_distribution": dict(sorted(block_dist.items())),
        "recife_priority_patches": sum(1 for q in written[OUT_QUEUE] if q["is_recife_priority"] == "true"),
        "total_gate_checks": len(written[OUT_AUDIT]),
        "total_recovery_queue_items": len(written[OUT_QUEUE]),
        "can_train_model": False, "can_create_operational_labels": False,
        "methodological_status": METHODOLOGICAL_STATUS,
    }


def build_report(summary):
    def fmt(dist):
        return "\n".join(f"- `{k}`: {v}" for k, v in dist.items()) or "- (none)"

    return f"""# v2av - Patch Boundary Source Manifest + Patch Geometry Builder

## 1. Objetivo
Atacar o blocker dominante da v2au (`BLOCKED_MISSING_PATCH_GEOMETRY`): descobrir todos os
patch_id do REV-P, rastrear a proveniencia espacial de cada um, construir geometria de
boundary de patch (GeoJSON/WKT/bbox) APENAS quando ha metadado suficiente e CRS conhecido,
e gerar uma fila de recuperacao/digitalizacao para o resto.

Nada de label, ground truth final ou modelo. Geometria nunca e inventada; sem CRS nao ha
boundary. O significado maximo de um boundary construido e `READY_FOR_V2AU_OVERLAY`.

## 2. Entradas usadas
{fmt({k: 1 for k in summary['inputs_found']})}

## 3. Saidas geradas
{fmt({k: 1 for k in summary['outputs_written']})}

## 4. Contagens
- Patches unicos descobertos: **{summary['total_unique_patches']}**
- Patches com fonte espacial encontrada: **{summary['patches_with_spatial_metadata']}**
- Patches com CRS: **{summary['patches_with_crs']}**
- Boundaries construidos: **{summary['patch_boundaries_built']}**
- Boundaries bloqueados: **{summary['patch_boundaries_blocked']}**
- Arquivos GeoJSON escritos: **{summary['geojson_files_written']}**
- Prontos para overlay v2au: **{summary['ready_for_v2au_overlay']}**
- Patches prioridade Recife: **{summary['recife_priority_patches']}**

## 5. Principais blocking_reason
{fmt(summary['blocking_reason_distribution'])}

## 6. Confirmacoes metodologicas explicitas
- Nenhum label operacional/binario foi criado (`can_create_operational_labels=false`; GATE_09 PASS).
- Nenhum modelo foi treinado (`can_train_model=false`).
- Nenhum ground truth final foi declarado; ausencia de metadado nunca virou negativo.
- Geometria nunca foi inventada; sem CRS nao ha boundary; tamanho default nao foi usado.
- Center point so vira boundary com opt-in explicito (desligado por padrao).

## 7. Interpretacao metodologica
{summary['methodological_status']}.

Quando o repositorio nao tem metadado espacial de patch suficiente, o resultado correto e
bloquear os boundaries e gerar uma fila clara de recuperacao (prioridade Recife). Quando
bbox/WKT/GeoJSON real com CRS existir, os boundaries sao construidos e ficam prontos para a
v2au processar em nova execucao, sempre como candidato a overlay, nunca como verdade final.
"""


def build_supplement(summary):
    return f"""# v2av artifact index supplement

Additive supplement to `final_delivery_artifact_index.md`. Nothing existing was removed or
rewritten; only v2av artifacts were added. The v2at and v2au registries were NOT overwritten.

| Artifact | Path | Function |
|---|---|---|
| Patch boundary source manifest | `datasets/{OUT_MANIFEST}` | Per-patch spatial provenance and build feasibility. |
| Patch boundary geometry registry | `datasets/{OUT_REGISTRY}` | Built patch boundaries (WKT/GeoJSON/bbox) or blockers. |
| Patch boundary build audit | `datasets/{OUT_AUDIT}` | 10 build gates per patch. |
| Patch boundary recovery queue | `datasets/{OUT_QUEUE}` | Prioritised geometry recovery/digitization queue. |
| Built GeoJSON | `datasets/{GEOJSON_SUBDIR.replace(os.sep, '/')}/patch_boundary_<id>.geojson` | One file per built boundary (none when no metadata). |
| Report | `outputs_public/{REPORT_REL.replace(os.sep, '/')}` | v2av methodological report. |
| Summary | `outputs_public/{SUMMARY_REL.replace(os.sep, '/')}` | v2av machine-readable summary. |

Methodological status: **{summary['methodological_status']}**
(`can_train_model=false`, `can_create_operational_labels=false`).
"""


def log_lines(summary, errors):
    lines = [
        f"[{STAGE}] Patch Boundary Source Manifest + Patch Geometry Builder",
        f"[{STAGE}] inputs_found={len(summary['inputs_found'])} outputs_written={len(summary['outputs_written'])}",
        f"[{STAGE}] unique_patches={summary['total_unique_patches']} "
        f"with_spatial_metadata={summary['patches_with_spatial_metadata']} "
        f"with_crs={summary['patches_with_crs']}",
        f"[{STAGE}] boundaries_built={summary['patch_boundaries_built']} "
        f"boundaries_blocked={summary['patch_boundaries_blocked']} "
        f"geojson_written={summary['geojson_files_written']} "
        f"ready_for_v2au_overlay={summary['ready_for_v2au_overlay']}",
        f"[{STAGE}] recife_priority_patches={summary['recife_priority_patches']}",
        f"[{STAGE}] can_train_model={summary['can_train_model']} "
        f"can_create_operational_labels={summary['can_create_operational_labels']}",
        f"[{STAGE}] methodological_status={summary['methodological_status']}",
        f"[{STAGE}] structural_errors={len(errors)}",
        f"[{STAGE}] status={'OK' if not errors else 'STRUCTURAL_ERROR'}",
    ]
    return "\n".join(lines) + "\n"


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
    config["accepted_crs"] = [normalise_crs(c) for c in config.get("accepted_crs", [])]
    config["target_crs"] = normalise_crs(config.get("target_crs", "EPSG:3857"))
    return config


def load_inputs(dataset_dir):
    found = []
    data = {}
    mapping = {"packages": IN_PACKAGES, "patch_resolution": IN_PATCH_RESOLUTION,
               "geometry_sources": IN_GEOMETRY_SOURCES}
    for key, rel in mapping.items():
        rows = load_csv(os.path.join(dataset_dir, rel))
        data[key] = rows
        if rows:
            found.append(rel.replace(os.sep, "/"))
    return data, found


def run(dataset_dir=None, output_dir=None, config_dir=None):
    """Runs the full engine. Returns (exit_code, summary)."""
    env_dataset, env_output, env_config = resolve_dirs()
    dataset_dir = dataset_dir or env_dataset
    output_dir = output_dir or env_output
    config_dir = config_dir or env_config

    config = load_config(config_dir)
    inputs, inputs_found = load_inputs(dataset_dir)

    if config.get("fail_on_missing_optional_patch_geometry") and IN_GEOMETRY_SOURCES not in [
            f.split("/")[-1] for f in inputs_found]:
        sys.stderr.write(f"[{STAGE}] fail_on_missing_optional_patch_geometry=true and no geometry sources\n")
        return 2, None

    patches = discover_patches(inputs)
    if not patches:
        patches = [{"patch_id": "UNKNOWN_PATCH", "region": UNKNOWN, "city": UNKNOWN,
                    "source_file": "NONE", "has_patch_geometry_flag": False,
                    "has_patch_bounds_flag": False, "package_count": 0, "candidate_reference": False}]
    source_index = index_geometry_sources(inputs)

    manifest, registry, audit, queue, geojson_written = build_all(
        patches, source_index, config, dataset_dir)

    written = {OUT_MANIFEST: manifest, OUT_REGISTRY: registry, OUT_AUDIT: audit, OUT_QUEUE: queue}

    errors = validate_outputs(written)
    if errors:
        for err in errors:
            sys.stderr.write(f"[{STAGE}] STRUCTURAL ERROR: {err}\n")
        return 3, None

    outputs_written = []
    for name in (OUT_MANIFEST, OUT_REGISTRY, OUT_AUDIT, OUT_QUEUE):
        write_csv(os.path.join(dataset_dir, name), COLUMNS[name], written[name])
        outputs_written.append(f"datasets/{name}")
    outputs_written.extend(geojson_written)

    summary = build_summary(inputs_found, outputs_written, written, geojson_written)
    summary["outputs_written"] = outputs_written + [
        f"outputs_public/{REPORT_REL.replace(os.sep, '/')}",
        f"outputs_public/{SUMMARY_REL.replace(os.sep, '/')}",
        f"outputs_public/{LOG_REL.replace(os.sep, '/')}",
        f"outputs_public/{SUPPLEMENT_REL.replace(os.sep, '/')}",
    ]

    write_text(os.path.join(output_dir, REPORT_REL), build_report(summary))
    write_text(os.path.join(output_dir, SUMMARY_REL), json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_text(os.path.join(output_dir, SUPPLEMENT_REL), build_supplement(summary))
    write_text(os.path.join(output_dir, LOG_REL), log_lines(summary, errors))

    sys.stdout.write(log_lines(summary, errors))
    return 0, summary


def main(_argv=None):
    code, _ = run()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
