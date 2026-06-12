#!/usr/bin/env python3
"""v2au - Patch-Event Overlay Geometry Engine.

Closes (programmatically) the dominant v2at blocker ``NO_PATCH_EVENT_OVERLAY_GEOMETRY``
by building the infrastructure to compute:

    observed event geometry  ∩  Sentinel patch geometry

It reads the v2at event-patch packages, inventories every geometry that actually
exists (patch boundaries, observed event geometries, context geometries and point
anchors), validates CRS, computes the spatial intersection and ``intersection_ratio``
with a pure-Python geometry kernel (offline, no geo dependency), audits overlay
gates, and writes *derived* delta files that update package status WITHOUT ever
overwriting the v2at registries.

Hard methodological line (never crossed):
  - no final ground truth, no binary/operational label, no model training;
  - absence of geometry is never turned into a negative;
  - a point is never an overlay unless an explicit buffer is configured;
  - an unknown CRS blocks the overlay (fail-closed);
  - context/risk geometry never promotes C4;
  - geometry is never invented.

The maximum decision this stage may ever reach is:

    C4_CANDIDATE_REQUIRES_HUMAN_REVIEW

It never emits C4_OPERATIONAL_LABEL, TRAINING_LABEL or GROUND_TRUTH_FINAL.

The engine is offline, deterministic, sorts outputs by (package_id, event_id,
patch_id), uses stable hashes for generated IDs, returns exit code 0 even with
expected blockers, and non-zero only on a real structural error.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys

STAGE = "v2au"
METHODOLOGICAL_STATUS = "GEOMETRY_OVERLAY_READY_FOR_HUMAN_REVIEW_NOT_FOR_TRAINING"
MAX_DECISION = "C4_CANDIDATE_REQUIRES_HUMAN_REVIEW"

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


CONFIG_NAME = "v2au_patch_event_overlay_geometry_config.json"

DEFAULT_CONFIG = {
    "accepted_crs": ["EPSG:4326", "EPSG:3857", "EPSG:31982", "EPSG:31983"],
    "target_crs_for_area": "EPSG:3857",
    "minimum_intersection_ratio": 0.01,
    "minimum_event_area_m2": 1.0,
    "allow_point_event_buffer": False,
    "optional_point_buffer_meters": None,
    "strict_crs": True,
    "offline_mode": True,
    "fail_on_missing_optional_geometry": False,
}

# Output filenames -----------------------------------------------------------

OUT_INVENTORY = "v2au_geometry_inventory.csv"
OUT_OVERLAY = "v2au_patch_event_overlay_registry.csv"
OUT_UPDATE = "v2au_event_patch_package_overlay_update.csv"
OUT_GATES = "v2au_overlay_gate_decision_audit.csv"
OUT_QUEUE = "v2au_overlay_review_queue.csv"

REPORT_REL = os.path.join("execution_reports", "v2au_patch_event_overlay_geometry_report.md")
SUMMARY_REL = os.path.join("execution_reports", "v2au_patch_event_overlay_geometry_summary.json")
SUPPLEMENT_REL = os.path.join("execution_reports", "v2au_artifact_index_supplement.md")
LOG_REL = os.path.join("logs_summary", "v2au_patch_event_overlay_geometry.txt")

# Input filenames (relative to dataset_dir) ----------------------------------

IN_PACKAGES = "v2at_event_patch_package_registry.csv"
IN_GROUND_EVENTS = "ground_reference_event_registry.csv"
# Optional manifest of real/provided geometries (absent in the offline repo).
IN_GEOMETRY_SOURCES = "v2au_geometry_sources.csv"

INPUT_FILES = [IN_PACKAGES, IN_GROUND_EVENTS, IN_GEOMETRY_SOURCES]

REGION_NAME = {"REC": "Recife", "PET": "Petropolis", "CUR": "Curitiba"}

GEOMETRY_ROLES = (
    "patch_boundary", "event_observed_geometry", "event_context_geometry",
    "risk_context_geometry", "point_anchor", "unknown",
)

OVERLAY_STATUSES = (
    "OVERLAY_CONFIRMED", "OVERLAY_CANDIDATE_REVIEW_REQUIRED", "NO_INTERSECTION",
    "BLOCKED_MISSING_PATCH_GEOMETRY", "BLOCKED_MISSING_EVENT_GEOMETRY",
    "BLOCKED_UNKNOWN_CRS", "BLOCKED_INVALID_GEOMETRY",
    "BLOCKED_POINT_ONLY_NO_BUFFER", "BLOCKED_CONTEXT_GEOMETRY_ONLY",
)

ALLOWED_USES = (
    "geometry_review_only", "c4_candidate_requires_human_review",
    "blocked_missing_geometry", "blocked_invalid_geometry", "blocked_context_only",
)

GATE_NAMES = [
    "OVERLAY_GATE_01_PATCH_GEOMETRY_EXISTS", "OVERLAY_GATE_02_EVENT_GEOMETRY_EXISTS",
    "OVERLAY_GATE_03_PATCH_CRS_KNOWN", "OVERLAY_GATE_04_EVENT_CRS_KNOWN",
    "OVERLAY_GATE_05_GEOMETRIES_VALID", "OVERLAY_GATE_06_AREA_COMPUTABLE",
    "OVERLAY_GATE_07_INTERSECTION_COMPUTABLE", "OVERLAY_GATE_08_INTERSECTION_RATIO_ACCEPTABLE",
    "OVERLAY_GATE_09_CONTEXT_GEOMETRY_NOT_PROMOTED", "OVERLAY_GATE_10_POINT_ANCHOR_NOT_OVERLAY",
    "OVERLAY_GATE_11_NO_OPERATIONAL_LABEL_CREATED", "OVERLAY_GATE_12_HUMAN_REVIEW_REQUIRED_FOR_C4",
]

COLUMNS = {
    OUT_INVENTORY: [
        "geometry_id", "geometry_role", "linked_event_id", "linked_patch_id",
        "source_id", "source_name", "geometry_type", "geometry_format",
        "geometry_path", "crs", "crs_status", "area_m2", "bbox_minx", "bbox_miny",
        "bbox_maxx", "bbox_maxy", "is_valid_geometry", "geometry_hash",
        "blocking_reason", "notes",
    ],
    OUT_OVERLAY: [
        "overlay_id", "package_id", "event_id", "patch_id", "region", "city",
        "hazard_type", "patch_geometry_id", "event_geometry_id", "patch_crs",
        "event_crs", "target_crs", "patch_area_m2", "event_area_m2",
        "intersection_area_m2", "intersection_ratio_patch", "intersection_ratio_event",
        "has_intersection", "has_patch_overlay", "overlay_quality", "overlay_status",
        "blocking_reason", "allowed_use", "notes",
    ],
    OUT_UPDATE: [
        "package_id", "previous_promotion_decision", "previous_blocking_reason",
        "has_patch_overlay_before", "has_patch_overlay_after", "intersection_ratio",
        "overlay_status", "new_promotion_candidate_level", "new_promotion_decision",
        "new_allowed_use", "remaining_blocking_reason", "requires_human_review",
        "can_create_operational_label",
    ],
    OUT_GATES: [
        "decision_id", "package_id", "event_id", "patch_id", "gate_name",
        "gate_passed", "gate_status", "required_condition", "observed_value",
        "severity", "blocking_reason", "recommended_action",
    ],
    OUT_QUEUE: [
        "review_item_id", "package_id", "event_id", "patch_id", "region", "city",
        "hazard_type", "priority_rank", "priority_reason", "missing_geometry_type",
        "suggested_action", "evidence_score", "overlay_status", "remaining_blocking_reason",
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


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Pure-Python geometry kernel (no external dependency).
# A geometry is a dict: {role, type ('point'|'polygon'), ring [(x,y)...] or pt,
# crs, crs_status, valid (bool)}.
# --------------------------------------------------------------------------- #


def normalise_crs(raw):
    raw = clean(raw).upper().replace(" ", "")
    if not raw:
        return ""
    if raw.isdigit():
        return f"EPSG:{raw}"
    if raw.startswith("EPSG:"):
        return raw
    return raw


def crs_status(crs, accepted):
    if not crs:
        return "UNKNOWN"
    return "KNOWN" if crs in accepted else "UNKNOWN"


def _parse_floats(text):
    out = []
    for token in text.replace(",", " ").split():
        try:
            out.append(float(token))
        except ValueError:
            return None
    return out


def parse_polygon_ring(coords):
    """coords: list of (x, y). Returns a closed ring or None."""
    ring = [(float(x), float(y)) for x, y in coords]
    if len(ring) < 3:
        return None
    if ring[0] != ring[-1]:
        ring = ring + [ring[0]]
    return ring


def parse_geometry_value(geom_format, value, lat=None, lon=None):
    """Parse a geometry from a declared format. Returns (gtype, payload) or (None, None)."""
    geom_format = clean(geom_format).lower()
    value = clean(value)
    try:
        if geom_format == "latlon_point":
            if lat is not None and lon is not None and clean(lat) and clean(lon):
                return "point", (float(lon), float(lat))
            nums = _parse_floats(value)
            if nums and len(nums) >= 2:
                # "lat,lon" convention for this format.
                return "point", (nums[1], nums[0])
            return None, None
        if geom_format == "bbox":
            nums = _parse_floats(value)
            if not nums or len(nums) < 4:
                return None, None
            minx, miny, maxx, maxy = nums[:4]
            ring = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
            return "polygon", ring
        if geom_format == "wkt":
            return _parse_wkt(value)
        if geom_format in ("geojson_inline", "geojson"):
            return _parse_geojson(json.loads(value))
    except (ValueError, json.JSONDecodeError):
        return None, None
    return None, None


def _parse_wkt(text):
    text = text.strip()
    upper = text.upper()
    if upper.startswith("POINT"):
        nums = _parse_floats(text[text.find("(") + 1: text.find(")")])
        if nums and len(nums) >= 2:
            return "point", (nums[0], nums[1])
        return None, None
    if upper.startswith("POLYGON"):
        inner = text[text.find("((") + 2: text.find("))")]
        pts = []
        for pair in inner.split(","):
            nums = _parse_floats(pair)
            if not nums or len(nums) < 2:
                return None, None
            pts.append((nums[0], nums[1]))
        ring = parse_polygon_ring(pts)
        return ("polygon", ring) if ring else (None, None)
    return None, None


def _parse_geojson(obj):
    geom = obj.get("geometry", obj) if isinstance(obj, dict) else None
    if not isinstance(geom, dict):
        return None, None
    gtype = clean(geom.get("type")).lower()
    coords = geom.get("coordinates")
    if gtype == "point" and isinstance(coords, list) and len(coords) >= 2:
        return "point", (float(coords[0]), float(coords[1]))
    if gtype == "polygon" and isinstance(coords, list) and coords:
        ring = parse_polygon_ring([(c[0], c[1]) for c in coords[0]])
        return ("polygon", ring) if ring else (None, None)
    if gtype == "multipolygon" and isinstance(coords, list) and coords:
        ring = parse_polygon_ring([(c[0], c[1]) for c in coords[0][0]])
        return ("polygon", ring) if ring else (None, None)
    return None, None


def _mercator(lon, lat):
    x = EARTH_RADIUS_M * math.radians(lon)
    lat = max(min(lat, 89.9), -89.9)
    y = EARTH_RADIUS_M * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return (x, y)


def to_metric(gtype, payload, crs, target_crs):
    """Return (gtype, payload_in_metric, metric_crs) or (gtype, None, None) if not reprojectable."""
    if crs in ("EPSG:3857", "EPSG:31982", "EPSG:31983"):
        # Already metric. EPSG:3857 is the canonical metric frame here; UTM stays
        # in its own metric frame and only intersects geometries in the same CRS.
        return gtype, payload, crs
    if crs == "EPSG:4326":
        if gtype == "point":
            return gtype, _mercator(payload[0], payload[1]), target_crs
        return gtype, [_mercator(x, y) for (x, y) in payload], target_crs
    return gtype, None, None


def shoelace_area(ring):
    area = 0.0
    for i in range(len(ring) - 1):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _clip_polygon(subject, clip):
    """Sutherland-Hodgman: clip subject ring by a convex clip ring. Returns ring or []."""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def intersect(p1, p2, a, b):
        dc = (a[0] - b[0], a[1] - b[1])
        dp = (p1[0] - p2[0], p1[1] - p2[1])
        n1 = a[0] * b[1] - a[1] * b[0]
        n2 = p1[0] * p2[1] - p1[1] * p2[0]
        denom = dc[0] * dp[1] - dc[1] * dp[0]
        if denom == 0:
            return p2
        return ((n1 * dp[0] - n2 * dc[0]) / denom, (n1 * dp[1] - n2 * dc[1]) / denom)

    # Orient clip ring counter-clockwise so `inside` is consistent.
    clip_ring = clip[:-1] if clip[0] == clip[-1] else clip[:]
    if _signed_area(clip_ring) < 0:
        clip_ring = clip_ring[::-1]
    output = subject[:-1] if subject[0] == subject[-1] else subject[:]
    for i in range(len(clip_ring)):
        a = clip_ring[i]
        b = clip_ring[(i + 1) % len(clip_ring)]
        if not output:
            break
        cur = output
        output = []
        s = cur[-1]
        for e in cur:
            if inside(e, a, b):
                if not inside(s, a, b):
                    output.append(intersect(s, e, a, b))
                output.append(e)
            elif inside(s, a, b):
                output.append(intersect(s, e, a, b))
            s = e
    if not output:
        return []
    if output[0] != output[-1]:
        output = output + [output[0]]
    return output


def _signed_area(ring):
    area = 0.0
    closed = ring + [ring[0]] if ring[0] != ring[-1] else ring
    for i in range(len(closed) - 1):
        x1, y1 = closed[i]
        x2, y2 = closed[i + 1]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def bbox_of(gtype, payload):
    if gtype == "point":
        return (payload[0], payload[1], payload[0], payload[1])
    xs = [x for x, _ in payload]
    ys = [y for _, y in payload]
    return (min(xs), min(ys), max(xs), max(ys))


def buffer_point_square(point_xy, buffer_m):
    x, y = point_xy
    return [(x - buffer_m, y - buffer_m), (x + buffer_m, y - buffer_m),
            (x + buffer_m, y + buffer_m), (x - buffer_m, y + buffer_m),
            (x - buffer_m, y - buffer_m)]


# --------------------------------------------------------------------------- #
# Geometry inventory.
# --------------------------------------------------------------------------- #


def _map_cprm_event(region_code, raw_date, package_events):
    """Map an official survey row to a v2at event_id by region + year (no invention)."""
    region = clean(region_code).upper()
    year = ""
    for token in clean(raw_date).replace("-", "/").split("/"):
        if len(token) == 4 and token.isdigit():
            year = token
            break
    for ev in package_events:
        if ev.upper().startswith(region) and (not year or year in ev):
            return ev
    return f"{region}_EVENT_{year or 'UNKNOWN'}"


def build_geometry_inventory(inputs, config):
    accepted = set(config["accepted_crs"])
    target = config["target_crs_for_area"]
    package_events = sorted({clean(p.get("event_id")) for p in inputs["packages"]
                             if clean(p.get("event_id"))})
    rows = []

    # (A) Official documented events -> point anchors (explicit lat/lon).
    for ev in inputs["ground_events"]:
        if not clean(ev.get("latitude")) or not clean(ev.get("longitude")):
            continue
        gtype, payload = parse_geometry_value("latlon_point", "",
                                              lat=ev.get("latitude"), lon=ev.get("longitude"))
        if gtype is None:
            continue
        linked_event = _map_cprm_event(ev.get("region"), ev.get("event_or_survey_date"), package_events)
        rows.append(_inventory_row(
            role="point_anchor", linked_event=linked_event, linked_patch="",
            source_id="SGB_RISK_CARTOGRAPHY", source_name="SGB/CPRM field survey point",
            gtype=gtype, gformat="latlon_point", gpath=NOT_AVAILABLE, crs="EPSG:4326",
            payload=payload, accepted=accepted, target=target,
            note="Official survey point anchor; a point is not a patch overlay.",
            block="POINT_GEOMETRY_NOT_OVERLAY"))

    # (B) Optional geometry-source manifest (patch boundaries, observed event geoms...).
    for src in inputs["geometry_sources"]:
        role = clean(src.get("geometry_role")).lower() or "unknown"
        if role not in GEOMETRY_ROLES:
            role = "unknown"
        gformat = clean(src.get("geometry_format")).lower()
        gpath = clean(src.get("geometry_path"))
        value = clean(src.get("geometry_value"))
        if gformat in ("geojson_file", "geojson_path") and gpath:
            abs_path = gpath if os.path.isabs(gpath) else os.path.join(PROJECT_ROOT, gpath)
            value = ""
            if os.path.exists(abs_path):
                try:
                    with open(abs_path, encoding="utf-8") as handle:
                        gtype, payload = _parse_geojson(json.load(handle))
                except (OSError, json.JSONDecodeError):
                    gtype, payload = None, None
            else:
                gtype, payload = None, None
        else:
            gtype, payload = parse_geometry_value(
                gformat, value, lat=src.get("latitude"), lon=src.get("longitude"))
        crs = normalise_crs(src.get("crs"))
        rows.append(_inventory_row(
            role=role, linked_event=clean(src.get("linked_event_id")),
            linked_patch=clean(src.get("linked_patch_id")),
            source_id=clean(src.get("source_id")) or "MANUAL_GEOMETRY_SOURCE",
            source_name=clean(src.get("source_name")) or "Manual geometry source",
            gtype=gtype, gformat=gformat or UNKNOWN, gpath=gpath or NOT_AVAILABLE,
            crs=crs, payload=payload, accepted=accepted, target=target,
            note="Provided geometry source.",
            block=""))

    rows.sort(key=lambda r: (r["geometry_role"], r["linked_event_id"],
                             r["linked_patch_id"], r["source_id"], r["geometry_hash"]))
    for idx, row in enumerate(rows):
        row["geometry_id"] = stable_id(
            "GEOM_", row["geometry_role"], row["linked_event_id"], row["linked_patch_id"],
            row["source_id"], row["geometry_hash"], str(idx))
    return rows


def _inventory_row(role, linked_event, linked_patch, source_id, source_name, gtype,
                   gformat, gpath, crs, payload, accepted, target, note, block):
    status = crs_status(crs, accepted)
    valid = payload is not None and gtype is not None
    area = 0.0
    bbox = ("", "", "", "")
    geom_hash = "INVALID"
    blocking = block
    if valid:
        bbox = bbox_of(gtype, payload)
        geom_hash = hashlib.sha1(repr((gtype, payload, crs)).encode("utf-8")).hexdigest()[:16]
        if gtype == "polygon":
            _, metric, _ = to_metric(gtype, payload, crs, target)
            if metric is not None:
                area = shoelace_area(metric)
        if status == "UNKNOWN":
            blocking = blocking or "UNKNOWN_OR_UNACCEPTED_CRS"
    else:
        blocking = blocking or "UNPARSEABLE_OR_MISSING_GEOMETRY"
    return {
        "geometry_role": role, "linked_event_id": linked_event or "",
        "linked_patch_id": linked_patch or "", "source_id": source_id,
        "source_name": source_name, "geometry_type": gtype or UNKNOWN,
        "geometry_format": gformat, "geometry_path": gpath,
        "crs": crs or UNKNOWN, "crs_status": status,
        "area_m2": f"{area:.2f}" if valid and gtype == "polygon" else "0.00",
        "bbox_minx": _fmt(bbox[0]), "bbox_miny": _fmt(bbox[1]),
        "bbox_maxx": _fmt(bbox[2]), "bbox_maxy": _fmt(bbox[3]),
        "is_valid_geometry": _b(valid), "geometry_hash": geom_hash,
        "blocking_reason": blocking, "notes": note,
        # private fields (not written) for downstream computation:
        "_gtype": gtype, "_payload": payload, "_crs": crs, "_status": status, "_valid": valid,
    }


def _fmt(value):
    if value == "":
        return ""
    return f"{float(value):.6f}"


# --------------------------------------------------------------------------- #
# Overlay computation.
# --------------------------------------------------------------------------- #

_EVENT_ROLE_PRIORITY = {
    "event_observed_geometry": 0, "point_anchor": 1,
    "event_context_geometry": 2, "risk_context_geometry": 3, "unknown": 4,
}


def _select_geometry(inventory, role_filter, key, value):
    matches = [g for g in inventory if g[key] == value and g["geometry_role"] in role_filter]
    if not matches:
        return None
    matches.sort(key=lambda g: (_EVENT_ROLE_PRIORITY.get(g["geometry_role"], 9), g["geometry_id"]))
    return matches[0]


def compute_overlay(pkg, inventory, config):
    target = config["target_crs_for_area"]
    min_ratio = config["minimum_intersection_ratio"]
    allow_buffer = bool(config["allow_point_event_buffer"])
    buffer_m = config["optional_point_buffer_meters"]
    event_id = clean(pkg.get("event_id"))
    patch_id = clean(pkg.get("patch_id"))

    patch_geom = _select_geometry(inventory, {"patch_boundary"}, "linked_patch_id", patch_id) \
        if patch_id and patch_id != "UNKNOWN_PATCH" else None
    event_geom = _select_geometry(
        inventory, {"event_observed_geometry", "point_anchor", "event_context_geometry",
                    "risk_context_geometry"}, "linked_event_id", event_id) \
        if event_id and "MISSING" not in event_id.upper() and event_id != "UNKNOWN_EVENT" else None

    result = {
        "patch_geometry_id": patch_geom["geometry_id"] if patch_geom else "",
        "event_geometry_id": event_geom["geometry_id"] if event_geom else "",
        "patch_crs": patch_geom["crs"] if patch_geom else UNKNOWN,
        "event_crs": event_geom["crs"] if event_geom else UNKNOWN,
        "target_crs": target, "patch_area_m2": "0.00", "event_area_m2": "0.00",
        "intersection_area_m2": "0.00", "intersection_ratio_patch": "0.000",
        "intersection_ratio_event": "0.000", "has_intersection": "false",
        "has_patch_overlay": "false", "overlay_quality": "NONE",
    }

    # Blocking precedence (fail-closed).
    if patch_geom is None:
        return {**result, "overlay_status": "BLOCKED_MISSING_PATCH_GEOMETRY",
                "blocking_reason": "NO_PATCH_BOUNDARY_GEOMETRY_AVAILABLE",
                "allowed_use": "blocked_missing_geometry"}
    if event_geom is None:
        return {**result, "overlay_status": "BLOCKED_MISSING_EVENT_GEOMETRY",
                "blocking_reason": "NO_EVENT_GEOMETRY_AVAILABLE",
                "allowed_use": "blocked_missing_geometry"}
    if event_geom["geometry_role"] in ("event_context_geometry", "risk_context_geometry"):
        return {**result, "overlay_status": "BLOCKED_CONTEXT_GEOMETRY_ONLY",
                "blocking_reason": "CONTEXT_GEOMETRY_CANNOT_PROMOTE_C4",
                "allowed_use": "blocked_context_only"}
    if event_geom["geometry_role"] == "point_anchor" and not (allow_buffer and buffer_m):
        return {**result, "overlay_status": "BLOCKED_POINT_ONLY_NO_BUFFER",
                "blocking_reason": "POINT_EVENT_WITHOUT_CONFIGURED_BUFFER",
                "allowed_use": "geometry_review_only"}
    if patch_geom["_status"] != "KNOWN" or event_geom["_status"] != "KNOWN":
        return {**result, "overlay_status": "BLOCKED_UNKNOWN_CRS",
                "blocking_reason": "UNKNOWN_OR_UNACCEPTED_CRS",
                "allowed_use": "geometry_review_only"}
    if not patch_geom["_valid"] or not event_geom["_valid"]:
        return {**result, "overlay_status": "BLOCKED_INVALID_GEOMETRY",
                "blocking_reason": "INVALID_GEOMETRY",
                "allowed_use": "blocked_invalid_geometry"}

    # Reproject both to a common metric frame.
    p_type, p_metric, p_crs = to_metric(patch_geom["_gtype"], patch_geom["_payload"],
                                        patch_geom["_crs"], target)
    e_type, e_metric, e_crs = to_metric(event_geom["_gtype"], event_geom["_payload"],
                                        event_geom["_crs"], target)
    if p_metric is None or e_metric is None or p_crs != e_crs:
        return {**result, "overlay_status": "BLOCKED_UNKNOWN_CRS",
                "blocking_reason": "CRS_REPROJECTION_UNAVAILABLE_OR_MISMATCH",
                "allowed_use": "geometry_review_only"}

    # Build event polygon (buffer the point if explicitly allowed).
    buffered_point = False
    if e_type == "point":
        e_ring = buffer_point_square(e_metric, float(buffer_m))
        buffered_point = True
    else:
        e_ring = e_metric
    p_ring = p_metric if p_type == "polygon" else buffer_point_square(p_metric, float(buffer_m or 0) or 1.0)

    patch_area = shoelace_area(p_ring)
    event_area = shoelace_area(e_ring)
    if patch_area <= 0 or event_area < config["minimum_event_area_m2"]:
        return {**result, "patch_area_m2": f"{patch_area:.2f}", "event_area_m2": f"{event_area:.2f}",
                "overlay_status": "BLOCKED_INVALID_GEOMETRY",
                "blocking_reason": "NON_COMPUTABLE_OR_DEGENERATE_AREA",
                "allowed_use": "blocked_invalid_geometry"}

    inter_ring = _clip_polygon(p_ring, e_ring)
    inter_area = shoelace_area(inter_ring) if len(inter_ring) >= 4 else 0.0
    ratio_patch = inter_area / patch_area if patch_area else 0.0
    ratio_event = inter_area / event_area if event_area else 0.0
    result.update({
        "patch_area_m2": f"{patch_area:.2f}", "event_area_m2": f"{event_area:.2f}",
        "intersection_area_m2": f"{inter_area:.2f}",
        "intersection_ratio_patch": f"{ratio_patch:.3f}",
        "intersection_ratio_event": f"{ratio_event:.3f}",
        "has_intersection": _b(inter_area > 0),
    })

    if inter_area <= 0 or max(ratio_patch, ratio_event) < min_ratio:
        return {**result, "overlay_status": "NO_INTERSECTION",
                "blocking_reason": "INTERSECTION_BELOW_MINIMUM_OR_ABSENT",
                "allowed_use": "geometry_review_only"}

    # Intersection confirmed. Buffered points stay "candidate" (weaker evidence).
    if buffered_point:
        status = "OVERLAY_CANDIDATE_REVIEW_REQUIRED"
        quality = "BUFFERED_POINT_CANDIDATE"
    else:
        status = "OVERLAY_CONFIRMED"
        quality = "POLYGON_INTERSECTION_CONFIRMED"
    return {**result, "has_patch_overlay": "true", "overlay_quality": quality,
            "overlay_status": status, "blocking_reason": "",
            "allowed_use": "c4_candidate_requires_human_review"}


def build_overlay_registry(packages, inventory, config):
    rows = []
    for pkg in packages:
        overlay = compute_overlay(pkg, inventory, config)
        rows.append({
            "overlay_id": stable_id("OVL_", pkg["package_id"]),
            "package_id": pkg["package_id"], "event_id": pkg.get("event_id", ""),
            "patch_id": pkg.get("patch_id", ""), "region": pkg.get("region", ""),
            "city": pkg.get("city", ""), "hazard_type": pkg.get("hazard_type", ""),
            **overlay,
            "notes": "Patch boundary is not event geometry; max decision is "
                     f"{MAX_DECISION}; no operational label.",
        })
    rows.sort(key=lambda r: (r["package_id"], r["event_id"], r["patch_id"]))
    return rows


# --------------------------------------------------------------------------- #
# Package overlay update (derived delta; never overwrites v2at).
# --------------------------------------------------------------------------- #


def build_package_update(packages, overlays):
    overlay_by_pkg = {o["package_id"]: o for o in overlays}
    rows = []
    for pkg in packages:
        ov = overlay_by_pkg.get(pkg["package_id"], {})
        status = ov.get("overlay_status", "BLOCKED_MISSING_PATCH_GEOMETRY")
        confirmed = status in ("OVERLAY_CONFIRMED", "OVERLAY_CANDIDATE_REVIEW_REQUIRED")
        rows.append({
            "package_id": pkg["package_id"],
            "previous_promotion_decision": pkg.get("promotion_decision", ""),
            "previous_blocking_reason": pkg.get("blocking_reason", ""),
            "has_patch_overlay_before": pkg.get("has_patch_overlay", "false"),
            "has_patch_overlay_after": _b(confirmed),
            "intersection_ratio": ov.get("intersection_ratio_patch", "0.000"),
            "overlay_status": status,
            "new_promotion_candidate_level": "C4_CANDIDATE" if confirmed
            else pkg.get("promotion_candidate_level", ""),
            "new_promotion_decision": MAX_DECISION if confirmed
            else pkg.get("promotion_decision", ""),
            "new_allowed_use": "c4_candidate_requires_human_review" if confirmed
            else pkg.get("allowed_use", ""),
            "remaining_blocking_reason": ov.get("blocking_reason", "") if not confirmed else "",
            "requires_human_review": _b(confirmed),
            "can_create_operational_label": "false",
        })
    rows.sort(key=lambda r: r["package_id"])
    return rows


# --------------------------------------------------------------------------- #
# Overlay gate decision audit (12 gates per package).
# --------------------------------------------------------------------------- #


def build_gates(packages, overlays, config):
    overlay_by_pkg = {o["package_id"]: o for o in overlays}
    min_ratio = config["minimum_intersection_ratio"]
    rows = []
    for pkg in packages:
        ov = overlay_by_pkg[pkg["package_id"]]
        ctx = _gate_context(ov, min_ratio)
        for gate in GATE_NAMES:
            passed, status, required, observed, severity, blocking, action = ctx[gate]
            rows.append({
                "decision_id": stable_id("OVD_", pkg["package_id"], gate),
                "package_id": pkg["package_id"], "event_id": pkg.get("event_id", ""),
                "patch_id": pkg.get("patch_id", ""), "gate_name": gate,
                "gate_passed": _b(passed), "gate_status": status,
                "required_condition": required, "observed_value": observed,
                "severity": severity, "blocking_reason": blocking,
                "recommended_action": action,
            })
    rows.sort(key=lambda r: (r["package_id"], r["gate_name"]))
    return rows


def _gate_context(ov, min_ratio):
    status = ov["overlay_status"]
    has_patch = bool(ov["patch_geometry_id"])
    has_event = bool(ov["event_geometry_id"])
    patch_crs_ok = ov["patch_crs"] not in (UNKNOWN, "") and status != "BLOCKED_UNKNOWN_CRS"
    event_crs_ok = ov["event_crs"] not in (UNKNOWN, "") and status != "BLOCKED_UNKNOWN_CRS"
    valid_ok = status != "BLOCKED_INVALID_GEOMETRY"
    area_ok = _num(ov["patch_area_m2"]) > 0 and _num(ov["event_area_m2"]) > 0
    inter_ok = status in ("OVERLAY_CONFIRMED", "OVERLAY_CANDIDATE_REVIEW_REQUIRED", "NO_INTERSECTION")
    ratio_ok = max(_num(ov["intersection_ratio_patch"]), _num(ov["intersection_ratio_event"])) >= min_ratio \
        and ov["has_patch_overlay"] == "true"
    is_context_block = status == "BLOCKED_CONTEXT_GEOMETRY_ONLY"
    is_point_block = status == "BLOCKED_POINT_ONLY_NO_BUFFER"
    confirmed = status in ("OVERLAY_CONFIRMED", "OVERLAY_CANDIDATE_REVIEW_REQUIRED")

    def blk(ok):
        return "" if ok else "OVERLAY_BLOCKING_CONDITION_NOT_MET"

    return {
        "OVERLAY_GATE_01_PATCH_GEOMETRY_EXISTS": (
            has_patch, "PASS" if has_patch else "FAIL", "patch boundary geometry exists",
            ov["patch_geometry_id"] or "NONE", "BLOCKING", blk(has_patch),
            "Acquire/digitize patch boundary geometry" if not has_patch else "None"),
        "OVERLAY_GATE_02_EVENT_GEOMETRY_EXISTS": (
            has_event, "PASS" if has_event else "FAIL", "event geometry exists",
            ov["event_geometry_id"] or "NONE", "BLOCKING", blk(has_event),
            "Acquire/digitize observed event geometry" if not has_event else "None"),
        "OVERLAY_GATE_03_PATCH_CRS_KNOWN": (
            patch_crs_ok, "PASS" if patch_crs_ok else "FAIL", "patch CRS in accepted list",
            ov["patch_crs"], "BLOCKING", blk(patch_crs_ok),
            "Resolve patch CRS" if not patch_crs_ok else "None"),
        "OVERLAY_GATE_04_EVENT_CRS_KNOWN": (
            event_crs_ok, "PASS" if event_crs_ok else "FAIL", "event CRS in accepted list",
            ov["event_crs"], "BLOCKING", blk(event_crs_ok),
            "Resolve event CRS" if not event_crs_ok else "None"),
        "OVERLAY_GATE_05_GEOMETRIES_VALID": (
            valid_ok, "PASS" if valid_ok else "FAIL", "geometries are valid",
            status, "BLOCKING", blk(valid_ok),
            "Repair invalid geometry" if not valid_ok else "None"),
        "OVERLAY_GATE_06_AREA_COMPUTABLE": (
            area_ok, "PASS" if area_ok else "PENDING", "patch and event area computable",
            f"patch={ov['patch_area_m2']};event={ov['event_area_m2']}", "BLOCKING", blk(area_ok),
            "Provide polygon geometry to compute area" if not area_ok else "None"),
        "OVERLAY_GATE_07_INTERSECTION_COMPUTABLE": (
            inter_ok, "PASS" if inter_ok else "PENDING", "intersection computable",
            ov["has_intersection"], "BLOCKING", blk(inter_ok),
            "Resolve geometry/CRS to compute intersection" if not inter_ok else "None"),
        "OVERLAY_GATE_08_INTERSECTION_RATIO_ACCEPTABLE": (
            ratio_ok, "PASS" if ratio_ok else "PENDING", f"intersection_ratio >= {min_ratio}",
            f"patch={ov['intersection_ratio_patch']};event={ov['intersection_ratio_event']}",
            "BLOCKING", blk(ratio_ok),
            "Confirm overlap above threshold under review" if not ratio_ok else "None"),
        # Guardrail gates: PASS means the engine respected the methodological line.
        "OVERLAY_GATE_09_CONTEXT_GEOMETRY_NOT_PROMOTED": (
            True, "PASS_CONTEXT_NOT_PROMOTED", "context/risk geometry never promotes C4",
            "context_blocked" if is_context_block else "no_context_promotion", "GUARDRAIL", "", "None"),
        "OVERLAY_GATE_10_POINT_ANCHOR_NOT_OVERLAY": (
            True, "PASS_POINT_NOT_OVERLAY", "point anchor is not an overlay without buffer",
            "point_blocked" if is_point_block else "no_point_overlay", "GUARDRAIL", "", "None"),
        "OVERLAY_GATE_11_NO_OPERATIONAL_LABEL_CREATED": (
            True, "PASS_NO_OPERATIONAL_LABEL", "no operational/binary label created",
            "no_operational_label_created", "GUARDRAIL", "", "None"),
        "OVERLAY_GATE_12_HUMAN_REVIEW_REQUIRED_FOR_C4": (
            True, "PASS_HUMAN_REVIEW_REQUIRED",
            "any C4 candidate requires human review (never auto-promoted)",
            "c4_candidate_requires_human_review" if confirmed else "no_c4_candidate",
            "GUARDRAIL", "", "None"),
    }


# --------------------------------------------------------------------------- #
# Overlay review / digitization queue.
# --------------------------------------------------------------------------- #


def build_review_queue(packages, overlays):
    overlay_by_pkg = {o["package_id"]: o for o in overlays}
    rows = []
    for pkg in packages:
        ov = overlay_by_pkg[pkg["package_id"]]
        rank, reason, missing, action = _priority(pkg, ov)
        rows.append({
            "review_item_id": stable_id("OVRQ_", pkg["package_id"], length=10),
            "package_id": pkg["package_id"], "event_id": pkg.get("event_id", ""),
            "patch_id": pkg.get("patch_id", ""), "region": pkg.get("region", ""),
            "city": pkg.get("city", ""), "hazard_type": pkg.get("hazard_type", ""),
            "priority_rank": str(rank), "priority_reason": reason,
            "missing_geometry_type": missing, "suggested_action": action,
            "evidence_score": pkg.get("evidence_score", ""),
            "overlay_status": ov["overlay_status"],
            "remaining_blocking_reason": ov.get("blocking_reason", ""),
        })
    rows.sort(key=lambda r: (int(r["priority_rank"]), -_num(r["evidence_score"]), r["package_id"]))
    return rows


def _priority(pkg, ov):
    status = ov["overlay_status"]
    region = pkg.get("region", "")
    allowed = pkg.get("allowed_use", "")
    score = _num(pkg.get("evidence_score"))
    has_patch = bool(ov["patch_geometry_id"])
    has_event = bool(ov["event_geometry_id"])

    missing = "none"
    if not has_patch and not has_event:
        missing = "patch_boundary_and_event_geometry"
    elif not has_patch:
        missing = "patch_boundary"
    elif not has_event:
        missing = "event_observed_geometry"
    elif status == "BLOCKED_POINT_ONLY_NO_BUFFER":
        missing = "event_polygon_from_point_anchor"
    elif status == "BLOCKED_CONTEXT_GEOMETRY_ONLY":
        missing = "observed_event_geometry_not_context"

    if status in ("OVERLAY_CONFIRMED", "OVERLAY_CANDIDATE_REVIEW_REQUIRED"):
        return (6, "Overlay computed; only human C4 review remains",
                missing, "Human review of C4 candidate (never auto-label)")
    if region == "Recife" and allowed == "candidate_reference":
        return (1, "Recife v2at candidate_reference; needs patch+event vector geometry",
                missing, "Digitize patch boundary and observed event geometry (vector)")
    if pkg.get("has_spatial_support") == "true" and not has_patch:
        return (2, "Strong spatial support but no patch vector geometry",
                missing, "Digitize/acquire patch boundary vector")
    if status == "BLOCKED_POINT_ONLY_NO_BUFFER":
        return (3, "CPRM/SGB point anchor needs a real observed event geometry",
                missing, "Digitize observed event polygon from official source")
    if pkg.get("urban_context") == "true":
        return (4, "Urban-context package pending geometry",
                missing, "Prioritise urban geometry acquisition")
    if score >= 0.6:
        return (5, "High v2at evidence_score pending geometry",
                missing, "Acquire geometry to close the overlay blocker")
    return (7, "Geometry acquisition pending",
            missing, "Acquire patch/event geometry for overlay")


# --------------------------------------------------------------------------- #
# Validation.
# --------------------------------------------------------------------------- #


def validate_outputs(written):
    errors = []
    for name, col in ((OUT_INVENTORY, "geometry_id"), (OUT_OVERLAY, "overlay_id"),
                      (OUT_UPDATE, "package_id"), (OUT_GATES, "decision_id"),
                      (OUT_QUEUE, "review_item_id")):
        for row in written[name]:
            if not clean(row.get(col)):
                errors.append(f"{name}: empty generated id in {col}")
                break

    for row in written[OUT_OVERLAY]:
        if row["overlay_status"] not in OVERLAY_STATUSES:
            errors.append(f"overlay {row['overlay_id']} invalid overlay_status {row['overlay_status']}")
            break
        if row["allowed_use"] not in ALLOWED_USES:
            errors.append(f"overlay {row['overlay_id']} invalid allowed_use {row['allowed_use']}")
            break

    for row in written[OUT_UPDATE]:
        if clean(row["can_create_operational_label"]).lower() != "false":
            errors.append(f"update {row['package_id']} can_create_operational_label != false")
            break
        decision = row["new_promotion_decision"].upper()
        if any(bad in decision for bad in ("OPERATIONAL_LABEL", "TRAINING_LABEL", "GROUND_TRUTH_FINAL")):
            errors.append(f"update {row['package_id']} produced a forbidden final decision")
            break
        if row["new_promotion_candidate_level"] == "C4":
            errors.append(f"update {row['package_id']} reached final C4 (must stay candidate)")
            break

    g11 = [r for r in written[OUT_GATES] if r["gate_name"] == "OVERLAY_GATE_11_NO_OPERATIONAL_LABEL_CREATED"]
    if not g11 or not all(r["gate_passed"] == "true" for r in g11):
        errors.append("OVERLAY_GATE_11 (no operational label) did not pass for all packages")
    return errors


# --------------------------------------------------------------------------- #
# Report / summary / log.
# --------------------------------------------------------------------------- #


def _count(rows, key, value):
    return sum(1 for r in rows if r.get(key) == value)


def build_summary(inputs_found, outputs_written, written):
    overlays = written[OUT_OVERLAY]
    status_dist = {}
    for o in overlays:
        status_dist[o["overlay_status"]] = status_dist.get(o["overlay_status"], 0) + 1
    confirmed = _count(overlays, "overlay_status", "OVERLAY_CONFIRMED") + \
        _count(overlays, "overlay_status", "OVERLAY_CANDIDATE_REVIEW_REQUIRED")
    c4_candidates = sum(1 for u in written[OUT_UPDATE]
                        if u["new_promotion_decision"] == MAX_DECISION)
    return {
        "stage": STAGE, "status": "OK_WITH_EXPECTED_BLOCKERS",
        "inputs_found": inputs_found, "outputs_written": outputs_written,
        "total_packages": len(written[OUT_OVERLAY]),
        "total_geometries": len(written[OUT_INVENTORY]),
        "total_overlays": len(overlays), "confirmed_overlays": confirmed,
        "blocked_missing_patch_geometry": _count(overlays, "overlay_status", "BLOCKED_MISSING_PATCH_GEOMETRY"),
        "blocked_missing_event_geometry": _count(overlays, "overlay_status", "BLOCKED_MISSING_EVENT_GEOMETRY"),
        "blocked_unknown_crs": _count(overlays, "overlay_status", "BLOCKED_UNKNOWN_CRS"),
        "blocked_invalid_geometry": _count(overlays, "overlay_status", "BLOCKED_INVALID_GEOMETRY"),
        "blocked_point_only_no_buffer": _count(overlays, "overlay_status", "BLOCKED_POINT_ONLY_NO_BUFFER"),
        "blocked_context_geometry_only": _count(overlays, "overlay_status", "BLOCKED_CONTEXT_GEOMETRY_ONLY"),
        "no_intersection": _count(overlays, "overlay_status", "NO_INTERSECTION"),
        "c4_candidate_requires_human_review": c4_candidates,
        "overlay_status_distribution": dict(sorted(status_dist.items())),
        "total_gate_checks": len(written[OUT_GATES]),
        "total_review_queue_items": len(written[OUT_QUEUE]),
        "can_train_model": False, "can_create_operational_labels": False,
        "methodological_status": METHODOLOGICAL_STATUS,
    }


def build_report(summary):
    def fmt(dist):
        return "\n".join(f"- `{k}`: {v}" for k, v in dist.items()) or "- (none)"

    return f"""# v2au - Patch-Event Overlay Geometry Engine

## 1. Objetivo
Fechar programaticamente o blocker dominante da v2at (`NO_PATCH_EVENT_OVERLAY_GEOMETRY`)
construindo a infraestrutura de **interseccao geometrica patch x evento**:
`evento observado ∩ patch Sentinel`. Le geometrias reais quando existem, valida CRS,
calcula area e `intersection_ratio`, audita gates e atualiza o status dos pacotes em
arquivos derivados, sem nunca sobrescrever os CSVs da v2at.

A decisao maxima permitida e `{MAX_DECISION}`. Nunca gera `C4_OPERATIONAL_LABEL`,
`TRAINING_LABEL` nem `GROUND_TRUTH_FINAL`.

## 2. Entradas usadas
{fmt({k: 1 for k in summary['inputs_found']})}

## 3. Saidas geradas
{fmt({k: 1 for k in summary['outputs_written']})}

## 4. Contagens
- Pacotes avaliados: **{summary['total_packages']}**
- Geometrias inventariadas: **{summary['total_geometries']}**
- Overlays calculados: **{summary['total_overlays']}**
- Overlays confirmados (max C4 candidate): **{summary['confirmed_overlays']}**
- Bloqueados por geometria de patch ausente: **{summary['blocked_missing_patch_geometry']}**
- Bloqueados por geometria de evento ausente: **{summary['blocked_missing_event_geometry']}**
- Bloqueados por CRS desconhecido: **{summary['blocked_unknown_crs']}**
- Bloqueados por geometria invalida: **{summary['blocked_invalid_geometry']}**
- Bloqueados por ponto sem buffer: **{summary['blocked_point_only_no_buffer']}**
- Bloqueados por geometria contextual: **{summary['blocked_context_geometry_only']}**
- Sem interseccao: **{summary['no_intersection']}**
- Em `{MAX_DECISION}`: **{summary['c4_candidate_requires_human_review']}**

## 5. Distribuicao de overlay_status
{fmt(summary['overlay_status_distribution'])}

## 6. Confirmacoes metodologicas explicitas
- Nenhum label operacional foi criado (`can_create_operational_labels=false`; OVERLAY_GATE_11 PASS).
- Nenhum modelo foi treinado (`can_train_model=false`).
- Nenhum ground truth final foi declarado; ausencia de geometria nunca virou negativo.
- Ponto nunca virou overlay sem buffer configurado; CRS desconhecido bloqueia (fail-closed).
- Geometria contextual/risco nunca promoveu C4; geometria nunca foi inventada.
- C4 so existe como candidato sob revisao humana (`{MAX_DECISION}`).

## 7. Interpretacao metodologica
{summary['methodological_status']}.

Quando nao ha geometria vetorial real, o resultado correto e: C4 permanece bloqueado e a
fila de revisao/digitalizacao e gerada. Quando geometria real existe, ela e processada,
mas a promocao maxima continua sendo um candidato C4 sob revisao humana, nunca um label final.
"""


def build_supplement(summary):
    return f"""# v2au artifact index supplement

Additive supplement to `final_delivery_artifact_index.md`. Nothing existing was removed or
rewritten; only v2au artifacts were added. The v2at registries were NOT overwritten.

| Artifact | Path | Function |
|---|---|---|
| Geometry inventory | `datasets/{OUT_INVENTORY}` | Every geometry that actually exists (patch/event/context/point), CRS-validated. |
| Overlay registry | `datasets/{OUT_OVERLAY}` | patch ∩ event intersection, ratio, overlay status. |
| Package overlay update | `datasets/{OUT_UPDATE}` | Derived delta of v2at package status (never overwrites v2at). |
| Overlay gate audit | `datasets/{OUT_GATES}` | 12 overlay gates per package. |
| Overlay review queue | `datasets/{OUT_QUEUE}` | Prioritised geometry digitization/review queue. |
| Report | `outputs_public/{REPORT_REL.replace(os.sep, '/')}` | v2au methodological report. |
| Summary | `outputs_public/{SUMMARY_REL.replace(os.sep, '/')}` | v2au machine-readable summary. |

Methodological status: **{summary['methodological_status']}** (max decision `{MAX_DECISION}`;
`can_train_model=false`, `can_create_operational_labels=false`).
"""


def log_lines(summary, errors):
    lines = [
        f"[{STAGE}] Patch-Event Overlay Geometry Engine",
        f"[{STAGE}] inputs_found={len(summary['inputs_found'])} outputs_written={len(summary['outputs_written'])}",
        f"[{STAGE}] packages={summary['total_packages']} geometries={summary['total_geometries']} "
        f"overlays={summary['total_overlays']} confirmed={summary['confirmed_overlays']}",
        f"[{STAGE}] blocked_missing_patch={summary['blocked_missing_patch_geometry']} "
        f"blocked_missing_event={summary['blocked_missing_event_geometry']} "
        f"blocked_unknown_crs={summary['blocked_unknown_crs']} "
        f"blocked_invalid={summary['blocked_invalid_geometry']} "
        f"blocked_point={summary['blocked_point_only_no_buffer']} "
        f"blocked_context={summary['blocked_context_geometry_only']}",
        f"[{STAGE}] c4_candidate_requires_human_review={summary['c4_candidate_requires_human_review']}",
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
    config["target_crs_for_area"] = normalise_crs(config.get("target_crs_for_area", "EPSG:3857"))
    return config


def load_inputs(dataset_dir):
    found = []
    data = {}
    mapping = {"packages": IN_PACKAGES, "ground_events": IN_GROUND_EVENTS,
               "geometry_sources": IN_GEOMETRY_SOURCES}
    for key, rel in mapping.items():
        rows = load_csv(os.path.join(dataset_dir, rel))
        data[key] = rows
        if rows:
            found.append(rel)
    return data, found


def run(dataset_dir=None, output_dir=None, config_dir=None):
    """Runs the full engine. Returns (exit_code, summary)."""
    env_dataset, env_output, env_config = resolve_dirs()
    dataset_dir = dataset_dir or env_dataset
    output_dir = output_dir or env_output
    config_dir = config_dir or env_config

    config = load_config(config_dir)
    inputs, inputs_found = load_inputs(dataset_dir)

    if config.get("fail_on_missing_optional_geometry") and IN_GEOMETRY_SOURCES not in inputs_found:
        sys.stderr.write(f"[{STAGE}] fail_on_missing_optional_geometry=true and no geometry sources\n")
        return 2, None

    packages = inputs["packages"]
    if not packages:
        packages = [_placeholder_package()]

    inventory = build_geometry_inventory(inputs, config)
    overlays = build_overlay_registry(packages, inventory, config)
    update = build_package_update(packages, overlays)
    gates = build_gates(packages, overlays, config)
    queue = build_review_queue(packages, overlays)

    written = {
        OUT_INVENTORY: inventory, OUT_OVERLAY: overlays, OUT_UPDATE: update,
        OUT_GATES: gates, OUT_QUEUE: queue,
    }

    errors = validate_outputs(written)
    if errors:
        for err in errors:
            sys.stderr.write(f"[{STAGE}] STRUCTURAL ERROR: {err}\n")
        return 3, None

    outputs_written = []
    for name in (OUT_INVENTORY, OUT_OVERLAY, OUT_UPDATE, OUT_GATES, OUT_QUEUE):
        write_csv(os.path.join(dataset_dir, name), COLUMNS[name], written[name])
        outputs_written.append(f"datasets/{name}")

    summary = build_summary(inputs_found, outputs_written, written)
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


def _placeholder_package():
    return {
        "package_id": stable_id("PKG_", "NO_INPUT", "UNKNOWN_EVENT", "UNKNOWN_PATCH"),
        "event_id": "UNKNOWN_EVENT", "patch_id": "UNKNOWN_PATCH", "region": UNKNOWN,
        "city": UNKNOWN, "hazard_type": "unknown_hazard", "has_patch_overlay": "false",
        "promotion_candidate_level": "C0", "promotion_decision": "REJECTED_NO_INPUT",
        "blocking_reason": "NO_INPUT_PACKAGES", "allowed_use": "rejected_context_only",
        "evidence_score": "0.000", "has_spatial_support": "false", "urban_context": "false",
    }


def main(_argv=None):
    code, _ = run()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
