#!/usr/bin/env python3
"""v2as Official Geometry Deep Probe and Digitization Attempt.

Continues directly from v2ar, whose 9 GeoJSON candidates all stayed ``geometry: null``
(4 ready for a digitization attempt; next action MANUAL_DIGITIZE...). v2as makes a deeper
*programmatic* attempt to recover explicit geometry from the already-registered official
sources/documents/URLs/anchors/metadata. When an explicit geometry payload exists
(coordinate, bbox, GeoJSON/KML/WKT/CSV-of-points, or a registry with versioned geometry),
a real-geometry GeoJSON candidate is exported. Otherwise ``geometry: null`` is preserved
and the geometric gap is documented (gap proven, not invented).

This stage never creates operational ground truth, never creates labels, never executes
overlay, never opens Protocol B, never infers Sentinel dates/crosswalks, and never
geocodes neighborhoods/streets as if they were observed geometry. Light HTTP is opt-in via
``V2AS_NETWORK=1``; small files may be cached to a git-ignored ``evidence_cache/`` directory
only — no raw payload is ever versioned. Only ``v2as_*`` artifacts are written; previous
outputs are preserved read-only. ``patch_truth_allowed`` stays false.
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import re

PROTOCOL_VERSION = "v2as"
DATASET_ROOT = os.environ.get("DATASET_ROOT", "datasets")
DATASET_DIR = os.environ.get("DATASET_DIR", os.path.join(DATASET_ROOT, "protocolo_c"))
DOCS_DIR = os.environ.get("DOCS_DIR", "docs/protocolo_c/v2as_official_geometry_deep_probe")
GEOJSON_DIR = os.environ.get("GEOJSON_DIR", os.path.join(DOCS_DIR, "geojson_candidates"))
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(DOCS_DIR, "evidence_cache"))
CONFIG_DIR = os.environ.get("CONFIG_DIR", "configs/protocolo_c")

NETWORK_ENV = "V2AS_NETWORK"
USER_AGENT = "REV-P-v2as-deep-probe/1.0 (audit-only; light fetch to ignored cache)"
HTTP_TIMEOUT = 8
MAX_CACHE_BYTES = 2 * 1024 * 1024  # only small files are cached temporarily

V2AR_INPUTS = {
    "priority": "v2ar_geometry_digitization_priority.csv",
    "source_registry": "v2ar_official_geometry_source_registry.csv",
    "probe": "v2ar_source_access_metadata_probe.csv",
    "readiness": "v2ar_digitization_readiness_matrix.csv",
    "extraction": "v2ar_geometry_extraction_attempts.csv",
    "geojson_index": "v2ar_geojson_candidate_index.csv",
    "geojson_validation": "v2ar_geojson_export_validation.csv",
    "external_packet": "v2ar_patch_link_external_validation_packet.csv",
    "license_crs": "v2ar_license_crs_checklist.csv",
    "task_refinement": "v2ar_digitization_task_refinement.csv",
    "boundary": "v2ar_patch_truth_boundary_audit.csv",
}
V2AQ_OPTIONAL = {
    "event_geometry": "v2aq_event_geometry_candidates.csv",
    "anchor": "v2aq_spatial_anchor_strength.csv",
    "crosswalk_join": "v2aq_crosswalk_geometry_join_candidates.csv",
    "patch_match": "v2aq_patch_geometry_match_candidates.csv",
}
V2AN_OPTIONAL = {
    "spatial": "v2an_spatial_anchor_registry.csv",
    "metadata": "v2an_document_metadata_registry.csv",
}
OBSERVED_REGISTRY = "observed_event_reference_candidate_registry.csv"

REGION_BY_PREFIX = {"REC": "Recife", "PET": "Petropolis", "CTB": "Curitiba"}
BRAZIL_BOUNDS = {"lat": (-34.0, 6.0), "lon": (-74.0, -34.0)}
REGION_BOUNDS = {
    "Recife": {"lat": (-8.6, -7.6), "lon": (-35.3, -34.6)},
    "Petropolis": {"lat": (-22.9, -22.1), "lon": (-43.6, -42.8)},
    "Curitiba": {"lat": (-25.9, -25.0), "lon": (-49.7, -48.8)},
}
GEOJSON_GEOMETRY_TYPES = {"Point", "MultiPoint", "LineString", "MultiLineString",
                          "Polygon", "MultiPolygon"}

# --- guardrail vocabulary (aligned with v2ar) ------------------------------
FORBIDDEN_TRUE_FIELDS = {
    "ground_truth_created", "ground_reference_created", "label_created",
    "operational_ground_truth", "training_ready", "training_use_allowed",
    "overlay_ready", "overlay_executed", "prediction_ready", "promotion_allowed",
    "can_create_ground_truth", "can_create_label", "can_use_for_ground_truth",
    "raw_data_versioned", "raw_data_downloaded", "sentinel_date_inferred",
    "crosswalk_inferred", "geometry_inferred", "coordinate_invented",
    "geometry_invented", "protocol_b_open", "protocol_b_reopened",
    "operational_use_allowed", "patch_truth_allowed", "flood_mask_created",
    "operational_validation", "geocoded_as_geometry",
}
FORBIDDEN_STATUS_VALUES = {
    "GROUND_TRUTH_VALIDATED", "GROUND_REFERENCE_CREATED", "GROUND_REFERENCE_TRUE",
    "LABEL_READY", "LABEL_POSITIVE", "LABEL_NEGATIVE", "TRAINING_READY",
    "PROTOCOL_B_OPEN", "PROTOCOL_B_REOPENED", "OPERATIONAL_VALIDATION",
    "PATCH_POSITIVE", "PATCH_NEGATIVE", "FLOOD_DETECTED", "PROMOTION_ALLOWED",
    "OVERLAY_EXECUTED",
}
FORBIDDEN_KV_MARKERS = [
    "ground_truth=true", "ground_reference=true", "label=true", "training=true",
    "overlay=true", "prediction=true", "protocol_b_open=true", "protocol_b_reopen=true",
    "sentinel_date_inferred=true", "crosswalk_inferred=true", "geometry_inferred=true",
    "operational_validation=true", "promotion_allowed=true",
    "can_create_ground_truth=true", "can_create_label=true", "raw_data_versioned=true",
    "patch_truth_allowed=true",
]
UNSAFE_LANGUAGE = [
    "ground truth validado", "classe positiva", "classe negativa", "label operacional",
    "deteccao de enchente", "deteccao de inundacao", "predicao de inundacao",
    "mascara de inundacao", "modelo preditivo", "validacao operacional",
    "treinamento supervisionado pronto", "similaridade visual confirma",
    "dino confirma evento", "overlay operacional executado", "coordenada inventada",
]
SAFE_UNSAFE_FIELDS = {
    "forbidden_use", "forbidden_decisions", "do_not_infer", "blocking_reason",
    "why_not_ground_truth", "why_still_blocked", "dominant_blocker", "notes",
    "deep_probe_reason", "geometry_null_reason", "geometry_status", "geometry_source",
    "geometry_detection_status", "geometry_payload_source", "payload_type",
    "coordinate_validation_status", "coordinate_source", "coordinate_precision_level",
    "source_status", "source_type", "license_status", "crs_status", "probe_status",
    "validation_status", "recommended_action", "geometry_confidence", "source_trace",
    "deep_probe_priority", "v2ar_priority_band", "region", "next_action",
    "recommended_artifact", "required_input", "metric", "status", "reason", "purpose",
    "safe_use", "required_properties_present", "geometry_source_explicit", "notes_extra",
}
SAFE_CONTEXT_MARKERS = [
    "nao pode dizer", "nao usar", "nao afirmar", "nao ha", "nao deve", "nao temos",
    "nao realiza", "nao detecta", "nao cria", "nao existe", "nao produz", "nao inferir",
    "nao inventar", "nao significa", "nao implica", "nao foi", "nao e ", "nao ", "do_not",
    "do not", "proibid", "forbidden", "limitation", "limitacao", "blocker", "blocked",
    "bloque", "does not", "not ", "no ", "sem ", "evitar", "ausencia", "pendente",
    "candidato", "review-only", "needs_", "missing", "not_established", "not_created",
    "insufficient", "digitization_required", "digitization", "manual", "null",
    "anchor_only", "textual_anchor", "external_validation", "explicit", "pending",
    "still_null", "deterministic", "offline", "no_explicit",
]
ABSOLUTE_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/Users/|/home/|/mnt/|\\\\)")
LOCAL_ONLY_MARKER = "local" + "_" + "only"
_WKT_RE = re.compile(r"^\s*(POINT|LINESTRING|POLYGON|MULTIPOINT|MULTIPOLYGON)\s*\(", re.IGNORECASE)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_args(argv=None):
    return argparse.ArgumentParser().parse_args(argv)


# --- path helpers ----------------------------------------------------------
def dataset_path(name):
    return os.path.join(DATASET_DIR, name)


def root_dataset_path(name):
    return os.path.join(DATASET_ROOT, name)


def doc_path(name):
    return os.path.join(DOCS_DIR, name)


def geojson_path(name):
    return os.path.join(GEOJSON_DIR, name)


def cache_path(name):
    return os.path.join(CACHE_DIR, name)


def rel_dataset(name):
    return f"datasets/protocolo_c/{name}"


def rel_doc(name):
    return f"docs/protocolo_c/v2as_official_geometry_deep_probe/{name}"


def rel_geojson(name):
    return f"docs/protocolo_c/v2as_official_geometry_deep_probe/geojson_candidates/{name}"


def rel_cache(name):
    return f"docs/protocolo_c/v2as_official_geometry_deep_probe/evidence_cache/{name}"


def repo_relative_path(path):
    raw = str(path)
    if ABSOLUTE_PATH_RE.search(raw):
        raise ValueError(f"Refusing absolute path: {path}")
    return raw.replace("\\", "/")


# --- value helpers ---------------------------------------------------------
def clean(value):
    return str(value or "").strip()


def is_true(value):
    return clean(value).lower() == "true"


def normalize_bool(value):
    return "true" if is_true(value) else "false"


def safe_slug(text):
    return re.sub(r"[^a-z0-9]+", "-", clean(text).lower()).strip("-") or "item"


def short_fragment(text, limit=160):
    return re.sub(r"\s+", " ", clean(text))[:limit].strip()


def normalize_region(value, candidate_id=""):
    prefix = clean(candidate_id)[:3].upper()
    if prefix in REGION_BY_PREFIX:
        return REGION_BY_PREFIX[prefix]
    low = clean(value).lower()
    if "recife" in low or low == "rec":
        return "Recife"
    if "petrop" in low or low == "pet":
        return "Petropolis"
    if "curitiba" in low or low == "ctb":
        return "Curitiba"
    return clean(value) or "Unspecified"


def normalize_candidate_id(value):
    return clean(value).upper()


# --- io helpers ------------------------------------------------------------
def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def assert_output_is_v2as(path):
    base = os.path.basename(str(path))
    if LOCAL_ONLY_MARKER in str(path).lower():
        raise ValueError(f"Refusing local_only output path: {path}")
    if base == ".gitignore":
        return True
    if not base.startswith("v2as_"):
        raise ValueError(f"Refusing to write non-v2as output: {path}")
    return True


def write_csv(path, columns, rows):
    assert_output_is_v2as(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_text(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_markdown(path, lines):
    assert_output_is_v2as(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_json(path, payload):
    assert_output_is_v2as(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_geojson(path, features):
    write_json(path, {"type": "FeatureCollection", "features": features})


def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)
    gi = cache_path(".gitignore")
    if not os.path.exists(gi):
        with open(gi, "w", encoding="utf-8") as f:
            f.write("*\n!.gitignore\n")
    return CACHE_DIR


def sha256_file(path):
    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def write_markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        cells = [clean(c).replace("|", "\\|").replace("\n", " ") for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


# --- network (opt-in, light only) ------------------------------------------
def is_network_enabled():
    return clean(os.environ.get(NETWORK_ENV)) == "1"


def probe_url_light(url, method="HEAD"):
    url = clean(url)
    base = {
        "url": url, "probe_method": method, "http_status": "", "content_type": "",
        "content_length": "", "accessed_at_utc": "",
        "probe_status": "NETWORK_DISABLED_DETERMINISTIC_RUN", "notes": "",
    }
    if not url or not is_network_enabled():
        return base
    try:
        import urllib.request
        req = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            base.update({
                "http_status": str(getattr(resp, "status", "") or resp.getcode()),
                "content_type": clean(resp.headers.get("Content-Type")),
                "content_length": clean(resp.headers.get("Content-Length")),
                "accessed_at_utc": datetime.datetime.now(datetime.timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"),
                "probe_status": "PROBED_METADATA_ONLY",
                "notes": "Light metadata-only probe; no raw versioning.",
            })
    except Exception as exc:  # noqa: BLE001 - fail-closed metadata record
        base.update({"probe_status": "PROBE_ERROR",
                     "notes": short_fragment(f"{type(exc).__name__}: {exc}", 120)})
    return base


def fetch_small_to_ignored_cache(url, candidate_id):
    """Fetch a small file to the git-ignored evidence cache (opt-in only).

    Returns a metadata dict. Never versions raw data: the cache directory carries a
    ``.gitignore`` excluding everything but itself. Offline -> no fetch.
    """
    url = clean(url)
    result = {"url": url, "cached_temporarily": "false", "cache_sha256": "",
              "content_type": "", "content_length": "", "cache_rel_path": "",
              "fetch_status": "NETWORK_DISABLED_DETERMINISTIC_RUN", "notes": ""}
    if not url or not is_network_enabled():
        return result
    try:
        import urllib.request
        ensure_cache_dir()
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            ctype = clean(resp.headers.get("Content-Type"))
            clen = resp.headers.get("Content-Length")
            if clen and int(clen) > MAX_CACHE_BYTES:
                result.update({"content_type": ctype, "content_length": clean(clen),
                               "fetch_status": "SKIPPED_TOO_LARGE",
                               "notes": "File exceeds small-cache limit; not fetched."})
                return result
            data = resp.read(MAX_CACHE_BYTES + 1)
            if len(data) > MAX_CACHE_BYTES:
                result.update({"content_type": ctype, "fetch_status": "SKIPPED_TOO_LARGE",
                               "notes": "Stream exceeded small-cache limit; discarded."})
                return result
            ext = _guess_ext(ctype, url)
            fname = f"v2as_payload_{safe_slug(candidate_id)}{ext}"
            with open(cache_path(fname), "wb") as f:
                f.write(data)
            result.update({
                "cached_temporarily": "true",
                "cache_sha256": hashlib.sha256(data).hexdigest(),
                "content_type": ctype, "content_length": str(len(data)),
                "cache_rel_path": rel_cache(fname),
                "fetch_status": "CACHED_SMALL_FILE_IGNORED",
                "notes": "Small file cached to git-ignored evidence_cache; not versioned.",
            })
    except Exception as exc:  # noqa: BLE001 - fail-closed
        result.update({"fetch_status": "FETCH_ERROR",
                       "notes": short_fragment(f"{type(exc).__name__}: {exc}", 120)})
    return result


def _guess_ext(content_type, url):
    ct = clean(content_type).lower()
    low = clean(url).lower()
    for token, ext in (("geojson", ".geojson"), ("kml", ".kml"), ("json", ".json"),
                       ("csv", ".csv"), ("xml", ".xml")):
        if token in ct or low.endswith(ext) or low.endswith("." + token):
            return ext
    return ".bin"


# --- geometry payload detection (explicit only) ----------------------------
def _finite_number(value):
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _polygon_from_bbox(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    nums = [_finite_number(v) for v in bbox[:4]]
    if any(n is None for n in nums):
        return None
    minx, miny, maxx, maxy = nums
    return {"type": "Polygon", "coordinates": [[
        [minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]]}


def _coords_have_finite_numbers(coords):
    if isinstance(coords, (int, float)):
        return math.isfinite(coords)
    if isinstance(coords, (list, tuple)):
        return bool(coords) and all(_coords_have_finite_numbers(c) for c in coords)
    return False


def validate_geojson_geometry_payload(geometry):
    """Validate a GeoJSON geometry dict has a known type and finite coordinates."""
    if not isinstance(geometry, dict):
        return False
    gtype = clean(geometry.get("type"))
    coords = geometry.get("coordinates")
    if gtype not in GEOJSON_GEOMETRY_TYPES or coords is None:
        return False
    return _coords_have_finite_numbers(coords)


def _parse_kml_coordinates(content):
    blocks = re.findall(r"<coordinates>(.*?)</coordinates>", content,
                        re.IGNORECASE | re.DOTALL)
    points = []
    for block in blocks:
        for token in re.split(r"\s+", block.strip()):
            parts = token.split(",")
            if len(parts) >= 2:
                lon, lat = _finite_number(parts[0]), _finite_number(parts[1])
                if lon is not None and lat is not None:
                    points.append([lon, lat])
    return points


def _parse_wkt(content):
    m = _WKT_RE.search(content)
    if not m:
        return None
    kind = m.group(1).upper()
    nums = [_finite_number(x) for x in _NUM_RE.findall(content)]
    nums = [n for n in nums if n is not None]
    if len(nums) < 2:
        return None
    pairs = [[nums[i], nums[i + 1]] for i in range(0, len(nums) - 1, 2)]
    if kind == "POINT":
        return {"type": "Point", "coordinates": pairs[0]}
    if kind == "LINESTRING":
        return {"type": "LineString", "coordinates": pairs}
    if kind in {"POLYGON", "MULTIPOLYGON"}:
        ring = pairs
        if ring[0] != ring[-1]:
            ring = ring + [ring[0]]
        return {"type": "Polygon", "coordinates": [ring]}
    if kind == "MULTIPOINT":
        return {"type": "MultiPoint", "coordinates": pairs}
    return None


def _parse_csv_points(content):
    try:
        rows = list(csv.DictReader(content.splitlines()))
    except Exception:  # noqa: BLE001
        return None
    if not rows:
        return None
    headers = {clean(h).lower(): h for h in rows[0].keys()}
    lat_key = next((headers[h] for h in headers if h in {"lat", "latitude"}), None)
    lon_key = next((headers[h] for h in headers if h in {"lon", "lng", "long", "longitude"}), None)
    if not lat_key or not lon_key:
        return None
    points = []
    for r in rows:
        lat, lon = _finite_number(r.get(lat_key)), _finite_number(r.get(lon_key))
        if lat is not None and lon is not None:
            points.append([lon, lat])
    if not points:
        return None
    return {"type": "Point", "coordinates": points[0]} if len(points) == 1 else \
        {"type": "MultiPoint", "coordinates": points}


def detect_geometry_payload(content, hint_type=""):
    """Detect an explicit geometry payload in content (JSON/GeoJSON/KML/WKT/CSV).

    Conservative: only explicit structured payloads count. Broad textual mentions and
    neighborhood/street names are NOT accepted, and nothing is geocoded.
    """
    result = {
        "payload_type": "none", "geojson_found": False, "kml_found": False,
        "wkt_found": False, "csv_point_found": False, "coordinate_found": False,
        "bbox_found": False, "explicit_geometry_found": False, "geometry": None,
        "geometry_payload_valid": False, "source_label": "",
    }
    text = clean(content)
    if not text:
        return result
    geometry = None
    # 1) JSON / GeoJSON
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        parsed = None
    if isinstance(parsed, dict):
        ptype = clean(parsed.get("type"))
        if ptype == "FeatureCollection" and isinstance(parsed.get("features"), list) and parsed["features"]:
            geometry = parsed["features"][0].get("geometry")
            result["geojson_found"] = True
        elif ptype == "Feature":
            geometry = parsed.get("geometry")
            result["geojson_found"] = True
        elif ptype in GEOJSON_GEOMETRY_TYPES:
            geometry = parsed
            result["geojson_found"] = True
        elif isinstance(parsed.get("bbox"), list):
            geometry = _polygon_from_bbox(parsed["bbox"])
            result["bbox_found"] = geometry is not None
        if geometry is not None:
            result["payload_type"] = "geojson"
            result["source_label"] = "explicit_geojson_payload"
    # 2) bbox as bare list of 4 numbers
    if geometry is None and isinstance(parsed, list) and len(parsed) == 4 \
            and all(_finite_number(v) is not None for v in parsed):
        geometry = _polygon_from_bbox(parsed)
        if geometry is not None:
            result["bbox_found"] = True
            result["payload_type"] = "bbox"
            result["source_label"] = "explicit_bbox_payload"
    # 3) WKT
    if geometry is None and _WKT_RE.search(text):
        geometry = _parse_wkt(text)
        if geometry is not None:
            result["wkt_found"] = True
            result["payload_type"] = "wkt"
            result["source_label"] = "explicit_wkt_payload"
    # 4) KML
    if geometry is None and "<coordinates>" in text.lower():
        pts = _parse_kml_coordinates(text)
        if pts:
            geometry = {"type": "Point", "coordinates": pts[0]} if len(pts) == 1 else \
                {"type": "LineString", "coordinates": pts}
            result["kml_found"] = True
            result["payload_type"] = "kml"
            result["source_label"] = "explicit_kml_payload"
    # 5) CSV of points
    if geometry is None and ("," in text and "\n" in text):
        geom = _parse_csv_points(text)
        if geom is not None:
            geometry = geom
            result["csv_point_found"] = True
            result["payload_type"] = "csv_points"
            result["source_label"] = "explicit_csv_point_payload"
    if geometry is not None and validate_geojson_geometry_payload(geometry):
        result["geometry"] = geometry
        result["geometry_payload_valid"] = True
        result["explicit_geometry_found"] = True
        gtype = clean(geometry.get("type"))
        result["coordinate_found"] = result["coordinate_found"] or gtype in {"Point", "MultiPoint"}
    elif geometry is not None:
        # geometry was structurally detected but failed validation
        result["payload_type"] = result["payload_type"] or "invalid"
    return result


def representative_lat_lon(geometry):
    """Return a representative (lat, lon) from an explicit geometry, or None."""
    if not isinstance(geometry, dict):
        return None
    gtype = clean(geometry.get("type"))
    coords = geometry.get("coordinates")

    def first_pair(obj):
        if (isinstance(obj, (list, tuple)) and len(obj) >= 2
                and _finite_number(obj[0]) is not None and _finite_number(obj[1]) is not None
                and not isinstance(obj[0], (list, tuple))):
            return [float(obj[0]), float(obj[1])]
        if isinstance(obj, (list, tuple)):
            for item in obj:
                pair = first_pair(item)
                if pair:
                    return pair
        return None

    if gtype not in GEOJSON_GEOMETRY_TYPES:
        return None
    pair = first_pair(coords)
    if not pair:
        return None
    lon, lat = pair[0], pair[1]
    return lat, lon


def candidate_payload_contents(candidate_id, event_geom_row):
    """Collect candidate geometry payload contents from explicit, versioned sources.

    Sources: the v2aq event-geometry explicit payload field (versioned) and any small
    file cached to the git-ignored evidence cache for this candidate. Never fabricates.
    """
    contents = []
    raw = clean(event_geom_row.get("explicit_geometry_geojson")) if event_geom_row else ""
    if raw:
        contents.append((rel_dataset(V2AQ_OPTIONAL["event_geometry"]) + "#explicit_geometry_geojson", raw))
    if os.path.isdir(CACHE_DIR):
        prefix = f"v2as_payload_{safe_slug(candidate_id)}"
        for name in sorted(os.listdir(CACHE_DIR)):
            if name.startswith(prefix) and name != ".gitignore":
                contents.append((rel_cache(name), read_text(cache_path(name))))
    return contents


def detect_candidate_geometry(candidate_id, event_geom_row):
    """Best explicit-geometry detection for a candidate across its payload sources."""
    for source, content in candidate_payload_contents(candidate_id, event_geom_row):
        det = detect_geometry_payload(content)
        if det["explicit_geometry_found"]:
            det = dict(det)
            det["source_label"] = det["source_label"] or "explicit_payload"
            det["source_trace"] = source
            return det
    return {"payload_type": "none", "geojson_found": False, "kml_found": False,
            "wkt_found": False, "csv_point_found": False, "coordinate_found": False,
            "bbox_found": False, "explicit_geometry_found": False, "geometry": None,
            "geometry_payload_valid": False, "source_label": "", "source_trace": ""}


# --- schema / loaders ------------------------------------------------------
def assert_min_schema(rows, required, artifact):
    if not rows:
        raise FileNotFoundError(f"Required artifact is missing or empty: {artifact}")
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"{artifact} missing required columns: {','.join(missing)}")
    return True


def load_v2ar_stack():
    missing = [name for name in V2AR_INPUTS.values()
               if not os.path.exists(dataset_path(name))]
    if missing:
        raise FileNotFoundError("v2as requires v2ar outputs; missing: " + ",".join(missing))
    stack = {key: load_csv(dataset_path(name)) for key, name in V2AR_INPUTS.items()}
    assert_min_schema(stack["priority"], ["candidate_id", "priority_band"],
                      V2AR_INPUTS["priority"])
    assert_min_schema(stack["readiness"], ["candidate_id", "can_digitize_now"],
                      V2AR_INPUTS["readiness"])
    return stack


def load_aux():
    return {
        "event_geometry": load_csv(dataset_path(V2AQ_OPTIONAL["event_geometry"])),
        "anchor": load_csv(dataset_path(V2AQ_OPTIONAL["anchor"])),
        "crosswalk_join": load_csv(dataset_path(V2AQ_OPTIONAL["crosswalk_join"])),
        "patch_match": load_csv(dataset_path(V2AQ_OPTIONAL["patch_match"])),
        "spatial": load_csv(dataset_path(V2AN_OPTIONAL["spatial"])),
        "metadata": load_csv(dataset_path(V2AN_OPTIONAL["metadata"])),
        "observed": {clean(r.get("observed_event_id")): r
                     for r in load_csv(root_dataset_path(OBSERVED_REGISTRY))},
    }


# --- guardrail assertions --------------------------------------------------
def _iter_values(rows_or_text):
    if isinstance(rows_or_text, list):
        for idx, row in enumerate(rows_or_text):
            values = row.values() if isinstance(row, dict) else [row]
            for v in values:
                yield idx, v
    else:
        yield 0, rows_or_text


def assert_no_absolute_paths_in_content(rows_or_text):
    for idx, value in _iter_values(rows_or_text):
        if ABSOLUTE_PATH_RE.search(clean(value)):
            raise ValueError(f"Absolute path in content at row {idx}: {value}")
    return True


def assert_no_local_only(rows_or_text):
    for idx, value in _iter_values(rows_or_text):
        if LOCAL_ONLY_MARKER in clean(value).lower():
            raise ValueError(f"local_only marker in content at row {idx}: {value}")
    return True


def _field_allows_unsafe(key, value):
    key_l = clean(key).lower()
    value_l = clean(value).lower()
    return key_l in SAFE_UNSAFE_FIELDS or any(m in value_l for m in SAFE_CONTEXT_MARKERS)


def assert_no_operational_promotion(rows):
    violations = []
    for idx, row in enumerate(rows):
        for key, value in row.items():
            key_l = clean(key).lower()
            value_s = clean(value)
            value_l = value_s.lower()
            if key_l in FORBIDDEN_TRUE_FIELDS and is_true(value_s):
                violations.append((idx, key, "forbidden_true"))
            if value_s in FORBIDDEN_STATUS_VALUES:
                violations.append((idx, key, "forbidden_status"))
            if ABSOLUTE_PATH_RE.search(value_s):
                violations.append((idx, key, "absolute_path"))
            if LOCAL_ONLY_MARKER in value_l:
                violations.append((idx, key, "local_only"))
            squashed = re.sub(r"\s*=\s*", "=", value_l)
            for marker in FORBIDDEN_KV_MARKERS:
                if marker in squashed:
                    violations.append((idx, key, f"forbidden_kv:{marker}"))
            for phrase in UNSAFE_LANGUAGE:
                if phrase in value_l and not _field_allows_unsafe(key, value_s):
                    violations.append((idx, key, f"unsafe_language:{phrase}"))
    if violations:
        sample = "; ".join(f"row={r[0]} field={r[1]} type={r[2]}" for r in violations[:5])
        raise ValueError(f"Operational promotion violation: {sample}")
    return True


def assert_no_label_creation(rows):
    for idx, row in enumerate(rows):
        for key, value in row.items():
            key_l = clean(key).lower()
            if key_l in {"can_create_label", "label_created", "training_use_allowed"} and is_true(value):
                raise ValueError(f"label/training creation at row {idx}: {key_l}")
            if clean(value) in {"LABEL_READY", "TRAINING_READY"}:
                raise ValueError(f"label/training status at row {idx}: {value}")
    return True


def assert_no_fake_ground_truth(rows):
    for idx, row in enumerate(rows):
        for key, value in row.items():
            key_l = clean(key).lower()
            value_s = clean(value)
            if key_l in {"operational_ground_truth_status", "ground_truth_status"}:
                if value_s.upper() not in {"NOT_ESTABLISHED", "", "NOT_CREATED"}:
                    raise ValueError(f"ground truth must stay NOT_ESTABLISHED, got {value_s}")
            if key_l in {"can_create_ground_truth", "can_use_for_ground_truth",
                         "ground_truth_created"} and is_true(value_s):
                raise ValueError(f"{key_l}=true forbidden at row {idx}.")
    return True


def assert_no_fake_geometry(rows):
    for idx, row in enumerate(rows):
        for key, value in row.items():
            key_l = clean(key).lower()
            if key_l in {"coordinate_invented", "geometry_invented", "geometry_inferred",
                         "has_invented_geometry_flag", "geocoded_as_geometry"} and is_true(value):
                raise ValueError(f"invented/inferred geometry at row {idx}: {key_l}")
    return True


def assert_no_fake_overlay(rows):
    for idx, row in enumerate(rows):
        for key, value in row.items():
            if clean(key).lower() in {"overlay_executed", "overlay_ready",
                                      "flood_mask_created"} and is_true(value):
                raise ValueError(f"overlay/mask flagged at row {idx}: {key}")
    return True


def assert_no_raw_data_versioned(rows):
    for idx, row in enumerate(rows):
        for key, value in row.items():
            if clean(key).lower() in {"raw_data_versioned", "raw_data_downloaded"} and is_true(value):
                raise ValueError(f"raw data versioned/downloaded at row {idx}: {key}")
    return True


def scan_text_violations(text):
    counts = {"absolute_path": 0, "local_only": 0, "forbidden_kv": 0,
              "unsafe_language": 0, "forbidden_true_flag": 0, "forbidden_status": 0}
    for line in text.splitlines():
        line_l = line.lower()
        safe_context = any(m in line_l for m in SAFE_CONTEXT_MARKERS)
        if ABSOLUTE_PATH_RE.search(line):
            counts["absolute_path"] += 1
        if LOCAL_ONLY_MARKER in line_l:
            counts["local_only"] += 1
        squashed = re.sub(r"\s*=\s*", "=", line_l)
        for marker in FORBIDDEN_KV_MARKERS:
            if marker in squashed:
                counts["forbidden_kv"] += 1
        if not safe_context and any(s in line for s in FORBIDDEN_STATUS_VALUES):
            counts["forbidden_status"] += 1
        for phrase in UNSAFE_LANGUAGE:
            if phrase in line_l and not safe_context:
                counts["unsafe_language"] += 1
    return counts


def assert_safe_text(text):
    counts = scan_text_violations(text)
    bad = {k: v for k, v in counts.items() if v}
    if bad:
        raise ValueError(f"Unsafe text detected: {bad}")
    return True


# --- record constructors ---------------------------------------------------
FORBIDDEN_USE = "ground_truth|label|training|overlay|prediction|protocol_b_reopen"


def build_deep_probe_row(candidate_id, source_url, probe, network_enabled):
    return {
        "probe_id": f"DP_v2as_{candidate_id}",
        "candidate_id": candidate_id,
        "source_url": clean(source_url),
        "network_enabled": normalize_bool(network_enabled),
        "http_status": clean(probe.get("http_status")),
        "content_type": clean(probe.get("content_type")),
        "content_length": clean(probe.get("content_length")),
        "cached_temporarily": normalize_bool(probe.get("cached_temporarily")),
        "cache_sha256": clean(probe.get("cache_sha256")),
        "probe_status": clean(probe.get("fetch_status")) or clean(probe.get("probe_status"))
        or "NETWORK_DISABLED_DETERMINISTIC_RUN",
        "raw_data_versioned": "false",
        "notes": short_fragment(probe.get("notes"), 140) or "Sondagem profunda; sem versionar bruto.",
    }


def build_geometry_payload_row(candidate_id, source, det):
    return {
        "payload_id": f"GP_v2as_{candidate_id}",
        "candidate_id": candidate_id,
        "source": source,
        "payload_type": clean(det.get("payload_type")) or "none",
        "explicit_geometry_found": normalize_bool(det.get("explicit_geometry_found")),
        "coordinate_found": normalize_bool(det.get("coordinate_found")),
        "bbox_found": normalize_bool(det.get("bbox_found")),
        "geojson_found": normalize_bool(det.get("geojson_found")),
        "kml_found": normalize_bool(det.get("kml_found")),
        "wkt_found": normalize_bool(det.get("wkt_found")),
        "csv_point_found": normalize_bool(det.get("csv_point_found")),
        "geometry_payload_valid": normalize_bool(det.get("geometry_payload_valid")),
        "geometry_payload_source": clean(det.get("source_trace")) or clean(det.get("source_label")),
        "geometry_detection_status": ("EXPLICIT_GEOMETRY_PAYLOAD_DETECTED"
                                      if det.get("explicit_geometry_found")
                                      else "NO_EXPLICIT_GEOMETRY_PAYLOAD"),
        "blocking_reason": ("" if det.get("explicit_geometry_found")
                            else "Sem payload geometrico explicito; geometria textual ampla nao "
                                 "aceita; sem geocoding automatico. Digitalizacao manual."),
    }


def build_geojson_export_row(candidate_id, rel_path, geometry_present, geometry_status,
                             geometry_source, null_reason, manual_required):
    return {
        "geojson_id": f"GJ_v2as_{candidate_id}",
        "candidate_id": candidate_id,
        "geojson_path": rel_path,
        "geometry_present": normalize_bool(geometry_present),
        "geometry_status": geometry_status,
        "geometry_source": geometry_source,
        "geometry_null_reason": null_reason,
        "manual_digitization_required": normalize_bool(manual_required),
        "safe_use": "Referencia observacional candidata para revisao/digitalizacao externa.",
        "forbidden_use": FORBIDDEN_USE,
    }


# --- candidate derivation --------------------------------------------------
def derive_candidates():
    stack = load_v2ar_stack()
    priority = {r["candidate_id"]: r for r in stack["priority"]}
    readiness = {r["candidate_id"]: r for r in stack["readiness"]}
    license_by_cid = {}
    for r in stack["license_crs"]:
        license_by_cid.setdefault(clean(r.get("candidate_id")), r)
    sources = {}
    for r in stack["source_registry"]:
        sources.setdefault(clean(r.get("candidate_id")), []).append(r)
    aux = load_aux()
    crosswalk = {r["candidate_id"]: r for r in aux["crosswalk_join"]}
    patch_match = {r["candidate_id"]: r for r in aux["patch_match"]}

    out = []
    for cid, prow in priority.items():
        rd = readiness.get(cid, {})
        lic = license_by_cid.get(cid, {})
        srcs = sources.get(cid, [])
        primary = next((s for s in srcs if clean(s.get("source_role")) == "primary"), srcs[0] if srcs else {})
        out.append({
            "candidate_id": cid,
            "region": normalize_region(prow.get("region"), cid),
            "v2ar_priority_band": clean(prow.get("priority_band")),
            "priority_rank": int(clean(prow.get("priority_rank")) or 0),
            "anchor_strength_band": clean(prow.get("anchor_strength_band")),
            "reference_level": clean(prow.get("reference_level")),
            "can_digitize_now": is_true(rd.get("can_digitize_now")),
            "source_status": clean(rd.get("digitization_status")),
            "geometry_candidate_status": clean(prow.get("geometry_candidate_status")),
            "license_status": clean(lic.get("license_status")) or "UNKNOWN_NEEDS_REVIEW",
            "crs_status": clean(lic.get("crs_status")) or "NOT_DOCUMENTED_NEEDS_ASSIGNMENT",
            "source_url": clean(primary.get("source_url_or_document")),
            "source_name": clean(primary.get("source_name")),
            "source_urls": [clean(s.get("source_url_or_document")) for s in srcs
                            if clean(s.get("source_url_or_document"))],
            "has_crosswalk_candidate": is_true(crosswalk.get(cid, {}).get("has_crosswalk_candidate")),
            "patch_id": clean(crosswalk.get(cid, {}).get("patch_id"))
            or clean(patch_match.get(cid, {}).get("patch_id")),
        })
    out.sort(key=lambda d: (d["priority_rank"], d["candidate_id"]))
    return out


# --- column schemas --------------------------------------------------------
PRIORITY_COLUMNS = [
    "priority_id", "candidate_id", "region", "v2ar_priority_band", "can_digitize_now",
    "source_status", "license_status", "crs_status", "deep_probe_priority", "deep_probe_reason",
]
PROBE_COLUMNS = [
    "probe_id", "candidate_id", "source_url", "network_enabled", "http_status",
    "content_type", "content_length", "cached_temporarily", "cache_sha256", "probe_status",
    "raw_data_versioned", "notes",
]
PAYLOAD_COLUMNS = [
    "payload_id", "candidate_id", "source", "payload_type", "explicit_geometry_found",
    "coordinate_found", "bbox_found", "geojson_found", "kml_found", "wkt_found",
    "csv_point_found", "geometry_payload_valid", "geometry_payload_source",
    "geometry_detection_status", "blocking_reason",
]
COORD_COLUMNS = [
    "validation_id", "candidate_id", "coordinate_source", "lat", "lon", "inside_brazil_bounds",
    "inside_region_plausible_bounds", "coordinate_precision_level",
    "coordinate_validation_status", "blocking_reason",
]
CLASSIFICATION_COLUMNS = [
    "classification_id", "candidate_id", "geometry_status", "geometry_confidence",
    "source_trace", "manual_digitization_required", "can_export_real_geometry",
    "can_use_for_patch_link_review", "can_use_for_ground_truth",
]
GEOJSON_INDEX_COLUMNS = [
    "geojson_id", "candidate_id", "geojson_path", "geometry_present", "geometry_status",
    "geometry_source", "geometry_null_reason", "manual_digitization_required", "safe_use",
    "forbidden_use",
]
GEOJSON_VALIDATION_COLUMNS = [
    "validation_id", "candidate_id", "geojson_path", "is_valid_json", "is_feature_collection",
    "geometry_present", "geometry_source_explicit", "geometry_null_allowed",
    "required_properties_present", "validation_status", "blocking_reason",
]
READINESS_UPDATE_COLUMNS = [
    "update_id", "candidate_id", "geometry_status", "geometry_present",
    "patch_geometry_available", "crosswalk_candidate_available", "external_validation_pending",
    "patch_link_review_ready", "patch_truth_allowed", "blocking_reason",
]
GAP_COLUMNS = [
    "gap_id", "candidate_id", "missing_geometry", "missing_license", "missing_crs",
    "missing_external_validation", "missing_patch_link_review", "recommended_action", "do_not_infer",
]
BOUNDARY_COLUMNS = [
    "boundary_id", "candidate_id", "geometry_present", "geometry_source_explicit",
    "patch_link_review_ready", "external_validation_pending", "patch_truth_allowed",
    "can_create_ground_truth", "can_create_label", "protocol_b_status", "why_still_blocked",
]
REGRESSION_COLUMNS = [
    "regression_id", "artifact_path", "check_type", "violation_count", "status", "severity", "notes",
]
NEXT_COLUMNS = [
    "rank", "next_action", "score", "allowed", "blocked_operational_use", "required_input",
    "recommended_artifact", "notes",
]
MANIFEST_COLUMNS = ["step_order", "step_name", "status", "outputs", "output_hashes", "notes"]
COMPLETION_COLUMNS = ["completion_id", "metric", "value", "status", "notes"]

GEOJSON_REQUIRED_PROPS = [
    "candidate_id", "region", "geometry_status", "source_trace", "not_ground_truth",
    "not_label", "patch_truth_allowed", "overlay_executed", "raw_data_versioned",
]
EXPLICIT_REAL_STATUSES = {"EXPLICIT_VALID_GEOMETRY_AVAILABLE", "EXPLICIT_COORDINATE_AVAILABLE",
                          "EXPLICIT_BBOX_AVAILABLE"}


# --- runners ---------------------------------------------------------------
def run_deep_probe_priority_builder(args=None):
    rows = []
    for d in derive_candidates():
        high = d["v2ar_priority_band"] == "HIGH_PRIORITY"
        if high and d["can_digitize_now"]:
            band, reason = "DEEP_PROBE_HIGH", "HIGH_PRIORITY|can_digitize_now"
        elif high:
            band, reason = "DEEP_PROBE_HIGH", "HIGH_PRIORITY"
        elif d["v2ar_priority_band"] == "MEDIUM_PRIORITY":
            band, reason = "DEEP_PROBE_MEDIUM", "MEDIUM_PRIORITY_textual_anchor"
        else:
            band, reason = "DEEP_PROBE_LOW", "LOW_PRIORITY_insufficient_spatial_evidence"
        if d["geometry_candidate_status"] == "OFFICIAL_MAP_DIGITIZATION_REQUIRED":
            reason += "|OFFICIAL_MAP_DIGITIZATION_REQUIRED"
        rows.append({
            "priority_id": f"DPP_v2as_{d['candidate_id']}",
            "candidate_id": d["candidate_id"],
            "region": d["region"],
            "v2ar_priority_band": d["v2ar_priority_band"],
            "can_digitize_now": normalize_bool(d["can_digitize_now"]),
            "source_status": d["source_status"],
            "license_status": d["license_status"],
            "crs_status": d["crs_status"],
            "deep_probe_priority": band,
            "deep_probe_reason": reason,
        })
    assert_no_operational_promotion(rows)
    write_csv(dataset_path("v2as_deep_probe_priority.csv"), PRIORITY_COLUMNS, rows)
    lines = [
        "# v2as - prioridade de deep probe",
        "",
        "Prioriza HIGH + can_digitize_now (mapa/laudo/ponto/rua, OFFICIAL_MAP_DIGITIZATION_REQUIRED).",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "v2ar_priority_band", "can_digitize_now", "deep_probe_priority", "deep_probe_reason"],
        [(r["candidate_id"], r["v2ar_priority_band"], r["can_digitize_now"],
          r["deep_probe_priority"], r["deep_probe_reason"]) for r in rows]))
    write_markdown(doc_path("v2as_deep_probe_priority.md"), lines)
    return rows


def run_source_deep_probe(args=None):
    network = is_network_enabled()
    if network:
        ensure_cache_dir()
    rows = []
    for d in derive_candidates():
        url = d["source_url"]
        if network and url:
            probe = fetch_small_to_ignored_cache(url, d["candidate_id"])
        else:
            probe = {"http_status": "", "content_type": "", "content_length": "",
                     "cached_temporarily": "false", "cache_sha256": "",
                     "fetch_status": "NETWORK_DISABLED_DETERMINISTIC_RUN",
                     "notes": "Rede desabilitada; rodar com V2AS_NETWORK=1 para sondagem profunda."}
            if not url:
                probe["fetch_status"] = "NO_URL_REGISTERED"
                probe["notes"] = "Fonte sem URL; abrir documento oficial manualmente."
        rows.append(build_deep_probe_row(d["candidate_id"], url, probe, network))
    assert_no_operational_promotion(rows)
    assert_no_raw_data_versioned(rows)
    assert_no_absolute_paths_in_content(rows)
    write_csv(dataset_path("v2as_source_deep_probe.csv"), PROBE_COLUMNS, rows)
    cached = sum(1 for r in rows if is_true(r["cached_temporarily"]))
    lines = [
        "# v2as - sondagem profunda de fontes",
        "",
        f"Modo de rede: {'V2AS_NETWORK=1' if network else 'offline deterministico'}.",
        f"Arquivos pequenos em cache ignorado: {cached}. raw_data_versioned=false sempre.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "network_enabled", "probe_status", "cached_temporarily", "raw_data_versioned"],
        [(r["candidate_id"], r["network_enabled"], r["probe_status"], r["cached_temporarily"],
          r["raw_data_versioned"]) for r in rows]))
    write_markdown(doc_path("v2as_source_deep_probe.md"), lines)
    return rows


def run_geometry_payload_detector(args=None):
    event_geom = {r["candidate_id"]: r for r in load_csv(dataset_path(V2AQ_OPTIONAL["event_geometry"]))}
    rows = []
    for d in derive_candidates():
        det = detect_candidate_geometry(d["candidate_id"], event_geom.get(d["candidate_id"], {}))
        source = clean(det.get("source_trace")) or rel_dataset(V2AQ_OPTIONAL["event_geometry"])
        rows.append(build_geometry_payload_row(d["candidate_id"], source, det))
    assert_no_operational_promotion(rows)
    assert_no_fake_geometry(rows)
    write_csv(dataset_path("v2as_geometry_payload_detection.csv"), PAYLOAD_COLUMNS, rows)
    found = sum(1 for r in rows if is_true(r["explicit_geometry_found"]))
    lines = [
        "# v2as - deteccao de payload geometrico",
        "",
        f"Payload geometrico explicito detectado: {found}/{len(rows)}. Geometria textual ampla",
        "nao e aceita e nao ha geocoding automatico de bairro/rua. Sem payload -> manual.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "payload_type", "explicit_geometry_found", "geometry_detection_status"],
        [(r["candidate_id"], r["payload_type"], r["explicit_geometry_found"],
          r["geometry_detection_status"]) for r in rows]))
    write_markdown(doc_path("v2as_geometry_payload_detection.md"), lines)
    return rows


def run_coordinate_sanity_validator(args=None):
    event_geom = {r["candidate_id"]: r for r in load_csv(dataset_path(V2AQ_OPTIONAL["event_geometry"]))}
    rows = []
    for d in derive_candidates():
        det = detect_candidate_geometry(d["candidate_id"], event_geom.get(d["candidate_id"], {}))
        latlon = representative_lat_lon(det.get("geometry")) if det.get("explicit_geometry_found") else None
        if not latlon:
            rows.append({
                "validation_id": f"CS_v2as_{d['candidate_id']}",
                "candidate_id": d["candidate_id"], "coordinate_source": "",
                "lat": "", "lon": "", "inside_brazil_bounds": "false",
                "inside_region_plausible_bounds": "false", "coordinate_precision_level": "none",
                "coordinate_validation_status": "NO_EXPLICIT_COORDINATE",
                "blocking_reason": "Sem coordenada explicita; nao inventar; digitalizacao manual.",
            })
            continue
        lat, lon = latlon
        bz = BRAZIL_BOUNDS
        inside_bz = bz["lat"][0] <= lat <= bz["lat"][1] and bz["lon"][0] <= lon <= bz["lon"][1]
        rb = REGION_BOUNDS.get(d["region"])
        inside_region = bool(rb and rb["lat"][0] <= lat <= rb["lat"][1]
                             and rb["lon"][0] <= lon <= rb["lon"][1])
        if not inside_bz:
            status = "OUTSIDE_BRAZIL_BOUNDS"
            blocking = "Coordenada fora dos limites do Brasil; nao usar."
        elif not inside_region:
            status = "INSIDE_BRAZIL_OUTSIDE_REGION"
            blocking = "Dentro do Brasil mas fora dos limites plausiveis da regiao; revisar manualmente."
        else:
            status = "EXPLICIT_COORDINATE_PLAUSIBLE"
            blocking = ""
        rows.append({
            "validation_id": f"CS_v2as_{d['candidate_id']}",
            "candidate_id": d["candidate_id"],
            "coordinate_source": clean(det.get("source_trace")) or clean(det.get("source_label")),
            "lat": f"{lat:.6f}", "lon": f"{lon:.6f}",
            "inside_brazil_bounds": normalize_bool(inside_bz),
            "inside_region_plausible_bounds": normalize_bool(inside_region),
            "coordinate_precision_level": "explicit_point",
            "coordinate_validation_status": status,
            "blocking_reason": blocking,
        })
    assert_no_operational_promotion(rows)
    assert_no_fake_geometry(rows)
    write_csv(dataset_path("v2as_coordinate_sanity_validation.csv"), COORD_COLUMNS, rows)
    lines = [
        "# v2as - validacao de sanidade de coordenadas",
        "",
        "Sem coordenada explicita -> NO_EXPLICIT_COORDINATE. Coordenadas explicitas sao checadas",
        "contra limites do Brasil e limites plausiveis da regiao. Nada e inventado.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "lat", "lon", "inside_brazil_bounds", "coordinate_validation_status"],
        [(r["candidate_id"], r["lat"], r["lon"], r["inside_brazil_bounds"],
          r["coordinate_validation_status"]) for r in rows]))
    write_markdown(doc_path("v2as_coordinate_sanity_validation.md"), lines)
    return rows


def run_geometry_candidate_classifier(args=None):
    event_geom = {r["candidate_id"]: r for r in load_csv(dataset_path(V2AQ_OPTIONAL["event_geometry"]))}
    payloads = {r["candidate_id"]: r for r in load_csv(dataset_path("v2as_geometry_payload_detection.csv"))}
    coords = {r["candidate_id"]: r for r in load_csv(dataset_path("v2as_coordinate_sanity_validation.csv"))}
    rows = []
    for d in derive_candidates():
        det = detect_candidate_geometry(d["candidate_id"], event_geom.get(d["candidate_id"], {}))
        pay = payloads.get(d["candidate_id"], {})
        coord = coords.get(d["candidate_id"], {})
        found = det.get("explicit_geometry_found")
        gtype = clean(det.get("geometry", {}).get("type")) if found else ""
        coord_status = clean(coord.get("coordinate_validation_status"))
        coord_bad = found and gtype in {"Point", "MultiPoint"} and coord_status in {
            "OUTSIDE_BRAZIL_BOUNDS"}
        if found and coord_bad:
            status, conf = "EXPLICIT_GEOMETRY_INVALID", "LOW"
            can_export, can_link = False, False
        elif found and gtype in {"Point", "MultiPoint"}:
            status, conf = "EXPLICIT_COORDINATE_AVAILABLE", "MEDIUM"
            can_export, can_link = True, True
        elif found and is_true(pay.get("bbox_found")):
            status, conf = "EXPLICIT_BBOX_AVAILABLE", "MEDIUM"
            can_export, can_link = True, True
        elif found:
            status, conf = "EXPLICIT_VALID_GEOMETRY_AVAILABLE", "MEDIUM"
            can_export, can_link = True, True
        elif d["can_digitize_now"]:
            status, conf = "MANUAL_DIGITIZATION_REQUIRED", "NONE"
            can_export, can_link = False, False
        else:
            status, conf = "NO_EXPLICIT_GEOMETRY_STILL_NULL", "NONE"
            can_export, can_link = False, False
        rows.append({
            "classification_id": f"GC_v2as_{d['candidate_id']}",
            "candidate_id": d["candidate_id"],
            "geometry_status": status,
            "geometry_confidence": conf,
            "source_trace": clean(det.get("source_trace")) or clean(pay.get("geometry_payload_source"))
            or rel_dataset(V2AQ_OPTIONAL["event_geometry"]),
            "manual_digitization_required": normalize_bool(not can_export),
            "can_export_real_geometry": normalize_bool(can_export),
            "can_use_for_patch_link_review": normalize_bool(can_link),
            "can_use_for_ground_truth": "false",
        })
    assert_no_operational_promotion(rows)
    assert_no_fake_geometry(rows)
    assert_no_fake_ground_truth(rows)
    for r in rows:
        if is_true(r["can_use_for_ground_truth"]):
            raise ValueError("can_use_for_ground_truth must stay false.")
        if is_true(r["can_export_real_geometry"]) and r["geometry_status"] not in EXPLICIT_REAL_STATUSES:
            raise ValueError("can_export_real_geometry only for explicit geometry statuses.")
    write_csv(dataset_path("v2as_geometry_candidate_classification.csv"), CLASSIFICATION_COLUMNS, rows)
    real = sum(1 for r in rows if is_true(r["can_export_real_geometry"]))
    lines = [
        "# v2as - classificacao de geometria candidata",
        "",
        f"can_export_real_geometry=true: {real}/{len(rows)} (somente payload explicito).",
        "can_use_for_ground_truth=false sempre.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "geometry_status", "can_export_real_geometry", "can_use_for_patch_link_review"],
        [(r["candidate_id"], r["geometry_status"], r["can_export_real_geometry"],
          r["can_use_for_patch_link_review"]) for r in rows]))
    write_markdown(doc_path("v2as_geometry_candidate_classification.md"), lines)
    return rows


def run_geojson_candidate_exporter(args=None):
    event_geom = {r["candidate_id"]: r for r in load_csv(dataset_path(V2AQ_OPTIONAL["event_geometry"]))}
    classification = {r["candidate_id"]: r for r in
                      load_csv(dataset_path("v2as_geometry_candidate_classification.csv"))}
    index = []
    for d in derive_candidates():
        cid = d["candidate_id"]
        cls = classification.get(cid, {})
        det = detect_candidate_geometry(cid, event_geom.get(cid, {}))
        geometry = None
        geometry_source = "none_manual_digitization"
        if is_true(cls.get("can_export_real_geometry")) and det.get("explicit_geometry_found"):
            geometry = det.get("geometry")
            geometry_source = clean(det.get("source_trace")) or clean(det.get("source_label")) \
                or "explicit_payload"
        status = clean(cls.get("geometry_status")) or "NO_EXPLICIT_GEOMETRY_STILL_NULL"
        if geometry is not None:
            null_reason = ""
        elif status == "MANUAL_DIGITIZATION_REQUIRED":
            null_reason = "manual_digitization_required"
        elif status == "EXPLICIT_GEOMETRY_INVALID":
            null_reason = "explicit_geometry_invalid_not_exported"
        else:
            null_reason = "no_explicit_geometry_still_null"
        props = {
            "candidate_id": cid,
            "region": d["region"],
            "geometry_status": status,
            "source_trace": clean(cls.get("source_trace")),
            "not_ground_truth": True,
            "not_label": True,
            "patch_truth_allowed": False,
            "overlay_executed": False,
            "raw_data_versioned": False,
        }
        feature = {"type": "Feature", "geometry": geometry, "properties": props}
        fname = f"v2as_event_geometry_{safe_slug(cid)}.geojson"
        write_geojson(geojson_path(fname), [feature])
        index.append(build_geojson_export_row(
            cid, rel_geojson(fname), geometry is not None, status, geometry_source,
            null_reason, cls.get("manual_digitization_required", "true")))
    assert_no_operational_promotion(index)
    assert_no_absolute_paths_in_content(index)
    write_csv(dataset_path("v2as_geojson_candidate_index.csv"), GEOJSON_INDEX_COLUMNS, index)
    present = sum(1 for r in index if r["geometry_present"] == "true")
    lines = [
        "# v2as - indice de GeoJSON candidatos",
        "",
        f"GeoJSONs: {len(index)}; com geometria real: {present}; geometry null: {len(index) - present}.",
        "Geometria real so quando ha payload explicito valido; caso contrario geometry: null.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "geometry_present", "geometry_status", "geometry_null_reason"],
        [(r["candidate_id"], r["geometry_present"], r["geometry_status"],
          r["geometry_null_reason"]) for r in index]))
    write_markdown(doc_path("v2as_geojson_candidate_index.md"), lines)
    return index


def run_geojson_validation(args=None):
    index = load_csv(dataset_path("v2as_geojson_candidate_index.csv"))
    event_geom = {r["candidate_id"]: r for r in load_csv(dataset_path(V2AQ_OPTIONAL["event_geometry"]))}
    rows = []
    for entry in index:
        cid = clean(entry.get("candidate_id"))
        rel = clean(entry.get("geojson_path"))
        fname = os.path.basename(rel)
        raw = read_text(geojson_path(fname))
        is_valid_json, payload = True, None
        try:
            payload = json.loads(raw) if raw else None
            is_valid_json = isinstance(payload, dict)
        except (ValueError, TypeError):
            is_valid_json = False
        is_fc = bool(payload) and payload.get("type") == "FeatureCollection"
        feats = payload.get("features", []) if is_fc else []
        feat = feats[0] if feats else {}
        geometry = feat.get("geometry") if feat else None
        geometry_present = geometry is not None
        props = feat.get("properties", {}) if feat else {}
        has_props = all(p in props for p in GEOJSON_REQUIRED_PROPS)
        det = detect_candidate_geometry(cid, event_geom.get(cid, {}))
        explicit = det.get("explicit_geometry_found")
        geometry_source_explicit = geometry_present and explicit
        if not is_valid_json:
            status, blocking = "INVALID_JSON", "GeoJSON nao e JSON valido."
        elif not is_fc:
            status, blocking = "NOT_FEATURE_COLLECTION", "GeoJSON nao e FeatureCollection."
        elif not has_props:
            status, blocking = "MISSING_REQUIRED_PROPERTIES", "Propriedades obrigatorias ausentes."
        elif geometry_present and not explicit:
            status, blocking = "INVALID_GEOMETRY_WITHOUT_EXPLICIT_SOURCE", \
                "Geometria real sem fonte explicita; nao inventar geometria."
        else:
            status, blocking = "VALID", ""
        rows.append({
            "validation_id": f"GV_v2as_{cid}",
            "candidate_id": cid,
            "geojson_path": rel,
            "is_valid_json": normalize_bool(is_valid_json),
            "is_feature_collection": normalize_bool(is_fc),
            "geometry_present": normalize_bool(geometry_present),
            "geometry_source_explicit": normalize_bool(geometry_source_explicit),
            "geometry_null_allowed": "true",
            "required_properties_present": normalize_bool(has_props),
            "validation_status": status,
            "blocking_reason": blocking,
        })
    assert_no_operational_promotion(rows)
    assert_no_fake_geometry(rows)
    failures = [r for r in rows if r["validation_status"] != "VALID"]
    write_csv(dataset_path("v2as_geojson_validation.csv"), GEOJSON_VALIDATION_COLUMNS, rows)
    if failures:
        raise ValueError("v2as GeoJSON validation failed: "
                         + ", ".join(f"{r['candidate_id']}:{r['validation_status']}"
                                     for r in failures[:5]))
    lines = [
        "# v2as - validacao de GeoJSON",
        "",
        "Falha-fechado: geometria real sem fonte explicita invalida o export. geometry: null permitido.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "geometry_present", "geometry_source_explicit", "validation_status"],
        [(r["candidate_id"], r["geometry_present"], r["geometry_source_explicit"],
          r["validation_status"]) for r in rows]))
    write_markdown(doc_path("v2as_geojson_validation.md"), lines)
    return rows


def run_patch_link_readiness_update(args=None):
    classification = {r["candidate_id"]: r for r in
                      load_csv(dataset_path("v2as_geometry_candidate_classification.csv"))}
    geojson = {r["candidate_id"]: r for r in load_csv(dataset_path("v2as_geojson_candidate_index.csv"))}
    rows = []
    for d in derive_candidates():
        cid = d["candidate_id"]
        cls = classification.get(cid, {})
        gj = geojson.get(cid, {})
        geometry_present = is_true(gj.get("geometry_present"))
        patch_avail = bool(d["patch_id"])
        xc_avail = d["has_crosswalk_candidate"]
        ready = geometry_present and patch_avail and xc_avail
        if ready:
            blocking = ""
        elif not geometry_present:
            blocking = "Sem geometria real explicita; digitalizacao manual ainda necessaria."
        elif not xc_avail:
            blocking = "Sem crosswalk Sentinel candidato; nao inferir por similaridade visual."
        else:
            blocking = "Sem geometria de patch disponivel."
        rows.append({
            "update_id": f"PLU_v2as_{cid}",
            "candidate_id": cid,
            "geometry_status": clean(cls.get("geometry_status")),
            "geometry_present": normalize_bool(geometry_present),
            "patch_geometry_available": normalize_bool(patch_avail),
            "crosswalk_candidate_available": normalize_bool(xc_avail),
            "external_validation_pending": "true",
            "patch_link_review_ready": normalize_bool(ready),
            "patch_truth_allowed": "false",
            "blocking_reason": blocking,
        })
    assert_no_operational_promotion(rows)
    for r in rows:
        if is_true(r["patch_truth_allowed"]):
            raise ValueError("patch_truth_allowed must stay false in v2as.")
    write_csv(dataset_path("v2as_patch_link_readiness_update.csv"), READINESS_UPDATE_COLUMNS, rows)
    ready_n = sum(1 for r in rows if is_true(r["patch_link_review_ready"]))
    lines = [
        "# v2as - atualizacao de readiness patch-link",
        "",
        f"patch_link_review_ready=true: {ready_n} (exige geometria real + patch geometry + crosswalk).",
        "patch_truth_allowed=false sempre.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "geometry_present", "patch_geometry_available", "crosswalk_candidate_available",
         "patch_link_review_ready"],
        [(r["candidate_id"], r["geometry_present"], r["patch_geometry_available"],
          r["crosswalk_candidate_available"], r["patch_link_review_ready"]) for r in rows]))
    write_markdown(doc_path("v2as_patch_link_readiness_update.md"), lines)
    return rows


def run_digitization_gap_report_builder(args=None):
    classification = {r["candidate_id"]: r for r in
                      load_csv(dataset_path("v2as_geometry_candidate_classification.csv"))}
    geojson = {r["candidate_id"]: r for r in load_csv(dataset_path("v2as_geojson_candidate_index.csv"))}
    readiness = {r["candidate_id"]: r for r in
                 load_csv(dataset_path("v2as_patch_link_readiness_update.csv"))}
    rows = []
    for d in derive_candidates():
        cid = d["candidate_id"]
        gj = geojson.get(cid, {})
        ru = readiness.get(cid, {})
        missing_geom = not is_true(gj.get("geometry_present"))
        missing_license = clean(d["license_status"]).upper() in {"UNKNOWN_NEEDS_REVIEW", "", "MISSING"}
        missing_crs = "DOCUMENTED" not in clean(d["crs_status"]).upper() or "NOT_DOCUMENTED" in clean(d["crs_status"]).upper()
        missing_review = not is_true(ru.get("patch_link_review_ready"))
        if missing_geom and d["can_digitize_now"]:
            action = "MANUAL_DIGITIZE_FROM_OFFICIAL_SOURCE"
        elif missing_geom:
            action = "COLLECT_OFFICIAL_SOURCE_THEN_DIGITIZE"
        elif missing_license or missing_crs:
            action = "VERIFY_LICENSE_AND_CRS"
        else:
            action = "PREPARE_EXTERNAL_VALIDATION"
        rows.append({
            "gap_id": f"GAP_v2as_{cid}",
            "candidate_id": cid,
            "missing_geometry": normalize_bool(missing_geom),
            "missing_license": normalize_bool(missing_license),
            "missing_crs": normalize_bool(missing_crs),
            "missing_external_validation": "true",
            "missing_patch_link_review": normalize_bool(missing_review),
            "recommended_action": action,
            "do_not_infer": "true",
        })
    assert_no_operational_promotion(rows)
    for r in rows:
        if not is_true(r["do_not_infer"]):
            raise ValueError("Every gap row must carry do_not_infer=true.")
    write_csv(dataset_path("v2as_digitization_gap_report.csv"), GAP_COLUMNS, rows)
    lines = [
        "# v2as - relatorio de lacunas de digitalizacao",
        "",
        "do_not_infer=true para todos. Lacuna geometrica comprovada quando nao ha payload explicito.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "missing_geometry", "missing_license", "missing_crs", "recommended_action"],
        [(r["candidate_id"], r["missing_geometry"], r["missing_license"], r["missing_crs"],
          r["recommended_action"]) for r in rows]))
    write_markdown(doc_path("v2as_digitization_gap_report.md"), lines)
    return rows


def run_patch_truth_boundary_audit(args=None):
    geojson = {r["candidate_id"]: r for r in load_csv(dataset_path("v2as_geojson_candidate_index.csv"))}
    validation = {r["candidate_id"]: r for r in load_csv(dataset_path("v2as_geojson_validation.csv"))}
    readiness = {r["candidate_id"]: r for r in
                 load_csv(dataset_path("v2as_patch_link_readiness_update.csv"))}
    rows = []
    for d in derive_candidates():
        cid = d["candidate_id"]
        gj = geojson.get(cid, {})
        val = validation.get(cid, {})
        ru = readiness.get(cid, {})
        rows.append({
            "boundary_id": f"PTB_v2as_{cid}",
            "candidate_id": cid,
            "geometry_present": normalize_bool(gj.get("geometry_present")),
            "geometry_source_explicit": normalize_bool(val.get("geometry_source_explicit")),
            "patch_link_review_ready": normalize_bool(ru.get("patch_link_review_ready")),
            "external_validation_pending": "true",
            "patch_truth_allowed": "false",
            "can_create_ground_truth": "false",
            "can_create_label": "false",
            "protocol_b_status": "BLOCKED",
            "why_still_blocked": ("Mesmo com geometria explicita, falta validacao externa e licenca/CRS; "
                                  "referencia de evento nao e ground truth, label ou patch-truth."),
        })
    assert_no_operational_promotion(rows)
    assert_no_label_creation(rows)
    assert_no_fake_ground_truth(rows)
    for r in rows:
        if (is_true(r["patch_truth_allowed"]) or is_true(r["can_create_ground_truth"])
                or is_true(r["can_create_label"]) or r["protocol_b_status"] != "BLOCKED"):
            raise ValueError("patch_truth/ground_truth/label must stay blocked; protocol_b BLOCKED.")
    write_csv(dataset_path("v2as_patch_truth_boundary_audit.csv"), BOUNDARY_COLUMNS, rows)
    lines = [
        "# v2as - patch truth boundary audit",
        "",
        "patch_truth_allowed=false, can_create_ground_truth=false, can_create_label=false,",
        "protocol_b_status=BLOCKED para todos os candidatos.",
        "",
    ]
    lines.extend(write_markdown_table(
        ["candidate_id", "geometry_present", "patch_link_review_ready", "patch_truth_allowed",
         "protocol_b_status"],
        [(r["candidate_id"], r["geometry_present"], r["patch_link_review_ready"],
          r["patch_truth_allowed"], r["protocol_b_status"]) for r in rows]))
    write_markdown(doc_path("v2as_patch_truth_boundary_audit.md"), lines)
    return rows


# --- guardrail regression --------------------------------------------------
def _regression_artifacts():
    artifacts = []
    if os.path.isdir(DATASET_DIR):
        for n in sorted(os.listdir(DATASET_DIR)):
            if n.endswith(".csv") and n.startswith("v2as_"):
                artifacts.append((rel_dataset(n), dataset_path(n), "csv"))
    if os.path.isdir(DOCS_DIR):
        for n in sorted(os.listdir(DOCS_DIR)):
            if n.endswith(".md"):
                artifacts.append((rel_doc(n), doc_path(n), "text"))
    if os.path.isdir(GEOJSON_DIR):
        for n in sorted(os.listdir(GEOJSON_DIR)):
            if n.endswith(".geojson"):
                artifacts.append((rel_geojson(n), geojson_path(n), "text"))
    return artifacts


def _scan_csv(path):
    counts = {"forbidden_true_flag": 0, "forbidden_status": 0, "absolute_path": 0,
              "local_only": 0, "forbidden_kv": 0, "unsafe_language": 0}
    for row in load_csv(path):
        for key, value in row.items():
            key_l = clean(key).lower()
            value_s = clean(value)
            value_l = value_s.lower()
            if key_l in FORBIDDEN_TRUE_FIELDS and is_true(value_s):
                counts["forbidden_true_flag"] += 1
            if value_s in FORBIDDEN_STATUS_VALUES:
                counts["forbidden_status"] += 1
            if ABSOLUTE_PATH_RE.search(value_s):
                counts["absolute_path"] += 1
            if LOCAL_ONLY_MARKER in value_l:
                counts["local_only"] += 1
            squashed = re.sub(r"\s*=\s*", "=", value_l)
            for marker in FORBIDDEN_KV_MARKERS:
                if marker in squashed:
                    counts["forbidden_kv"] += 1
            for phrase in UNSAFE_LANGUAGE:
                if phrase in value_l and not _field_allows_unsafe(key, value_s):
                    counts["unsafe_language"] += 1
    return counts


def run_guardrail_regression(args=None):
    check_types = ["forbidden_true_flag", "forbidden_status", "absolute_path",
                   "non_versionable_path_marker", "forbidden_kv", "unsafe_language"]
    rows = []
    total_fail = 0
    for rel, path, kind in _regression_artifacts():
        counts = _scan_csv(path) if kind == "csv" else scan_text_violations(read_text(path))
        for check_type in check_types:
            key = "local_only" if check_type == "non_versionable_path_marker" else check_type
            count = counts.get(key, 0)
            status = "PASS" if count == 0 else "FAIL"
            if status == "FAIL":
                total_fail += 1
            rows.append({
                "regression_id": f"GR_v2as_{len(rows):05d}",
                "artifact_path": rel,
                "check_type": check_type,
                "violation_count": str(count),
                "status": status,
                "severity": "none" if count == 0 else "blocking",
                "notes": "Fail-closed guardrail regression over v2as outputs (incl. geojson).",
            })
    write_csv(dataset_path("v2as_guardrail_regression.csv"), REGRESSION_COLUMNS, rows)
    if total_fail:
        fails = [(r["artifact_path"], r["check_type"]) for r in rows if r["status"] == "FAIL"]
        raise ValueError(f"v2as guardrail regression failed: {fails[:5]}")
    return rows


# --- next action -----------------------------------------------------------
def run_next_action_ranker(args=None):
    geojson = load_csv(dataset_path("v2as_geojson_candidate_index.csv"))
    readiness = load_csv(dataset_path("v2as_patch_link_readiness_update.csv"))
    has_real_geometry = any(r.get("geometry_present") == "true" for r in geojson)
    review_ready = any(is_true(r.get("patch_link_review_ready")) for r in readiness)
    if review_ready:
        top = "PREPARE_PATCH_LINK_REVIEW"
    elif has_real_geometry:
        top = "EXTERNAL_VALIDATE_EXPLICIT_EVENT_GEOMETRY"
    else:
        top = "MANUAL_DIGITIZE_EVENT_GEOMETRY_FROM_OFFICIAL_SOURCES"
    options = [
        (top, 100, "v2as geojson/patch-link readiness", "v2as_geojson_candidate_index.csv"),
        ("OPEN_OFFICIAL_MAPS_AND_DIGITIZE", 85, "v2as deep probe priority",
         "v2as_deep_probe_priority.csv"),
        ("VERIFY_LICENSE_AND_CRS", 70, "v2as digitization gap report",
         "v2as_digitization_gap_report.csv"),
        ("MAINTAIN_EVENT_REFERENCE_ONLY", 55, "patch truth boundary audit",
         "v2as_patch_truth_boundary_audit.csv"),
        ("TRAINING_PROTOCOL_B_OVERLAY_LABEL_GT_PROMOTION", 0, "blocked by guardrails", "none"),
    ]
    rows, seen = [], set()
    rank = 1
    for action, score, required, artifact in sorted(options, key=lambda x: (-x[1], x[0])):
        if action in seen:
            continue
        seen.add(action)
        rows.append({
            "rank": str(rank),
            "next_action": action,
            "score": str(score),
            "allowed": "false" if score == 0 else "true",
            "blocked_operational_use": "true",
            "required_input": required,
            "recommended_artifact": artifact,
            "notes": ("No next action may recommend training, Protocol B, automatic overlay, "
                      "labels, operational ground truth, automatic geometry inference, or promotion."),
        })
        rank += 1
    write_csv(dataset_path("v2as_next_actions_registry.csv"), NEXT_COLUMNS, rows)
    return rows


# --- completion report -----------------------------------------------------
def run_completion_report(args=None):
    priority = load_csv(dataset_path("v2as_deep_probe_priority.csv"))
    probes = load_csv(dataset_path("v2as_source_deep_probe.csv"))
    payloads = load_csv(dataset_path("v2as_geometry_payload_detection.csv"))
    coords = load_csv(dataset_path("v2as_coordinate_sanity_validation.csv"))
    classification = load_csv(dataset_path("v2as_geometry_candidate_classification.csv"))
    geojson = load_csv(dataset_path("v2as_geojson_candidate_index.csv"))
    validation = load_csv(dataset_path("v2as_geojson_validation.csv"))
    readiness = load_csv(dataset_path("v2as_patch_link_readiness_update.csv"))
    gaps = load_csv(dataset_path("v2as_digitization_gap_report.csv"))
    boundary = load_csv(dataset_path("v2as_patch_truth_boundary_audit.csv"))
    regression = load_csv(dataset_path("v2as_guardrail_regression.csv"))
    next_rows = load_csv(dataset_path("v2as_next_actions_registry.csv"))

    payload_found = sum(1 for r in payloads if is_true(r.get("explicit_geometry_found")))
    coord_ok = sum(1 for r in coords if r.get("coordinate_validation_status") == "EXPLICIT_COORDINATE_PLAUSIBLE")
    can_export = sum(1 for r in classification if is_true(r.get("can_export_real_geometry")))
    geom_real = sum(1 for r in geojson if r.get("geometry_present") == "true")
    geom_null = sum(1 for r in geojson if r.get("geometry_present") == "false")
    valid_ok = sum(1 for r in validation if r.get("validation_status") == "VALID")
    review_ready = sum(1 for r in readiness if is_true(r.get("patch_link_review_ready")))
    boundary_blocked = sum(1 for r in boundary if r.get("patch_truth_allowed") == "false"
                           and r.get("protocol_b_status") == "BLOCKED")
    regression_fail = sum(1 for r in regression if r.get("status") == "FAIL")
    rows = [
        {"completion_id": "CR_v2as_000", "metric": "deep_probe_candidates", "value": str(len(priority)),
         "status": "RECORDED", "notes": "Prioritized from v2ar."},
        {"completion_id": "CR_v2as_001", "metric": "source_probes", "value": str(len(probes)),
         "status": "RECORDED", "notes": "Offline deterministic by default; raw never versioned."},
        {"completion_id": "CR_v2as_002", "metric": "geometry_payloads_detected", "value": str(payload_found),
         "status": "RECORDED", "notes": "Explicit structured payloads only."},
        {"completion_id": "CR_v2as_003", "metric": "coordinates_plausible", "value": str(coord_ok),
         "status": "RECORDED", "notes": "Explicit coordinates within Brazil/region bounds."},
        {"completion_id": "CR_v2as_004", "metric": "can_export_real_geometry", "value": str(can_export),
         "status": "RECORDED", "notes": "Only explicit geometry statuses."},
        {"completion_id": "CR_v2as_005", "metric": "geojson_created", "value": str(len(geojson)),
         "status": "RECORDED", "notes": f"real geometry={geom_real}, null={geom_null}."},
        {"completion_id": "CR_v2as_006", "metric": "geojson_geometry_real", "value": str(geom_real),
         "status": "RECORDED", "notes": "Only from explicit payload."},
        {"completion_id": "CR_v2as_007", "metric": "geojson_geometry_null", "value": str(geom_null),
         "status": "RECORDED", "notes": "geometry: null; manual digitization or proven gap."},
        {"completion_id": "CR_v2as_008", "metric": "geojson_validation_valid", "value": str(valid_ok),
         "status": "PASS" if validation and valid_ok == len(validation) else "FAIL",
         "notes": "All exported GeoJSON valid."},
        {"completion_id": "CR_v2as_009", "metric": "patch_link_review_ready", "value": str(review_ready),
         "status": "RECORDED", "notes": "Requires explicit geometry + patch + crosswalk."},
        {"completion_id": "CR_v2as_010", "metric": "digitization_gap_rows", "value": str(len(gaps)),
         "status": "RECORDED", "notes": "All carry do_not_infer=true."},
        {"completion_id": "CR_v2as_011", "metric": "patch_truth_boundary_blocked_all", "value": str(boundary_blocked),
         "status": "PASS" if boundary and boundary_blocked == len(boundary) else "FAIL",
         "notes": "patch_truth/ground_truth/label blocked; protocol_b BLOCKED."},
        {"completion_id": "CR_v2as_012", "metric": "guardrail_regression_failures", "value": str(regression_fail),
         "status": "PASS" if regression_fail == 0 else "FAIL", "notes": "Fail-closed."},
        {"completion_id": "CR_v2as_013", "metric": "next_action_rank_1",
         "value": next_rows[0]["next_action"] if next_rows else "", "status": "SAFE_NEXT_ACTION",
         "notes": "Digitization / external validation / patch-link path."},
        {"completion_id": "CR_v2as_014", "metric": "final_decision",
         "value": "deep_probe_executed_geometry_candidate_extracted_or_gap_proven_no_operational_ground_truth",
         "status": "NO_OPERATIONAL_GROUND_TRUTH",
         "notes": "patch_truth_allowed=false; protocol_b blocked; no raw data versioned."},
    ]
    write_csv(dataset_path("v2as_completion_report.csv"), COMPLETION_COLUMNS, rows)
    lines = [
        "# v2as completion report",
        "",
        f"Deep probe candidates: {len(priority)}.",
        f"Source probes: {len(probes)} (offline deterministic by default).",
        f"Geometry payloads detected: {payload_found}.",
        f"Coordinates plausible: {coord_ok}.",
        f"Can export real geometry: {can_export}.",
        f"GeoJSON created: {len(geojson)} (real geometry: {geom_real}, null: {geom_null}).",
        f"GeoJSON validation valid: {valid_ok}/{len(validation)}.",
        f"Patch-link review ready: {review_ready}.",
        f"Digitization gap rows: {len(gaps)} (all do_not_infer=true).",
        f"Patch-truth boundary blocked: {boundary_blocked}/{len(boundary)}.",
        f"Guardrail regression failures: {regression_fail}.",
        f"Next action rank 1: {next_rows[0]['next_action'] if next_rows else ''}.",
        "Final decision: deep probe executed; explicit geometry extracted where available, gap proven otherwise; no operational ground truth.",
    ]
    write_markdown(doc_path("v2as_completion_report.md"), lines)
    return rows


# --- orchestrator ----------------------------------------------------------
_ORCHESTRATION = [
    ("deep_probe_priority", "run_deep_probe_priority_builder",
     ["v2as_deep_probe_priority.csv"], ["v2as_deep_probe_priority.md"]),
    ("source_deep_probe", "run_source_deep_probe",
     ["v2as_source_deep_probe.csv"], ["v2as_source_deep_probe.md"]),
    ("geometry_payload_detector", "run_geometry_payload_detector",
     ["v2as_geometry_payload_detection.csv"], ["v2as_geometry_payload_detection.md"]),
    ("coordinate_sanity_validator", "run_coordinate_sanity_validator",
     ["v2as_coordinate_sanity_validation.csv"], ["v2as_coordinate_sanity_validation.md"]),
    ("geometry_candidate_classifier", "run_geometry_candidate_classifier",
     ["v2as_geometry_candidate_classification.csv"], ["v2as_geometry_candidate_classification.md"]),
    ("geojson_candidate_exporter", "run_geojson_candidate_exporter",
     ["v2as_geojson_candidate_index.csv"], ["v2as_geojson_candidate_index.md"]),
    ("geojson_validation", "run_geojson_validation",
     ["v2as_geojson_validation.csv"], ["v2as_geojson_validation.md"]),
    ("patch_link_readiness_update", "run_patch_link_readiness_update",
     ["v2as_patch_link_readiness_update.csv"], ["v2as_patch_link_readiness_update.md"]),
    ("digitization_gap_report", "run_digitization_gap_report_builder",
     ["v2as_digitization_gap_report.csv"], ["v2as_digitization_gap_report.md"]),
    ("patch_truth_boundary_audit", "run_patch_truth_boundary_audit",
     ["v2as_patch_truth_boundary_audit.csv"], ["v2as_patch_truth_boundary_audit.md"]),
    ("guardrail_regression", "run_guardrail_regression",
     ["v2as_guardrail_regression.csv"], []),
    ("next_action_ranker", "run_next_action_ranker",
     ["v2as_next_actions_registry.csv"], []),
    ("completion_report", "run_completion_report",
     ["v2as_completion_report.csv"], ["v2as_completion_report.md"]),
]


def _manifest_row(order, name, status, ds_out, doc_out, notes):
    outputs = [rel_dataset(o) for o in ds_out] + [rel_doc(o) for o in doc_out]
    hashes = [sha256_file(dataset_path(o))[:16] for o in ds_out]
    hashes += [sha256_file(doc_path(o))[:16] for o in doc_out]
    return {
        "step_order": str(order), "step_name": name, "status": status,
        "outputs": "|".join(outputs), "output_hashes": "|".join(h for h in hashes if h),
        "notes": notes,
    }


def _write_manifest_md(rows):
    lines = ["# v2as - orchestrator run manifest", "",
             f"Etapas executadas: {len(rows)}. Nenhuma operacao git foi executada.", ""]
    lines.extend(write_markdown_table(
        ["ordem", "etapa", "status", "outputs"],
        [(r["step_order"], r["step_name"], r["status"], r["outputs"]) for r in rows]))
    write_markdown(doc_path("v2as_orchestrator_run_manifest.md"), lines)


def run_master_orchestrator(args=None):
    rows = []
    for order, (name, func_name, ds_out, doc_out) in enumerate(_ORCHESTRATION, 1):
        func = globals()[func_name]
        try:
            func(args)
        except Exception as exc:
            rows.append(_manifest_row(order, name, "FAIL", ds_out, doc_out,
                                      f"{type(exc).__name__}: {exc}"))
            write_csv(dataset_path("v2as_orchestrator_run_manifest.csv"), MANIFEST_COLUMNS, rows)
            _write_manifest_md(rows)
            raise
        rows.append(_manifest_row(order, name, "OK", ds_out, doc_out, "Completed."))
    write_csv(dataset_path("v2as_orchestrator_run_manifest.csv"), MANIFEST_COLUMNS, rows)
    _write_manifest_md(rows)
    return rows


def run_all(args=None):
    return run_master_orchestrator(args)
