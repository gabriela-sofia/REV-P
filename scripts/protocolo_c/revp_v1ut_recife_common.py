#!/usr/bin/env python3
"""v1ut Recife public CKAN coordinate recovery.

This module audits only coordinates already present in public Recife CKAN
assets downloaded in v1uj/v1uk. It writes versionable summaries with hashes,
counts, classifications and blockers; it never geocodes, infers centroids,
executes overlays, creates ground references, or creates labels.
"""

import csv
import hashlib
import json
import os
import re
import unicodedata
from datetime import date, datetime

PROTOCOL_VERSION = "v1ut"
EVENT_ID = "REC_2022_05_24_30"
DATASET_DIR = "datasets/protocolo_c"
DOCS_DIR = "docs/metodologia_cientifica"
CONFIG_DIR = "configs/protocolo_c"
RAW_DIR = "local_only/protocolo_c/focused_public_artifacts/raw/v1uj/ckan/REC_2022_05_24_30"
STAGING_DIR = "local_only/protocolo_c/recife_coordinate_recovery/staging/v1ut"
REPORTS_DIR = "local_only/protocolo_c/recife_coordinate_recovery/reports/v1ut"

CORE_START = date(2022, 5, 24)
CORE_END = date(2022, 5, 30)
RECIFE_LAT_MIN = -9.5
RECIFE_LAT_MAX = -7.0
RECIFE_LON_MIN = -36.0
RECIFE_LON_MAX = -33.0
MAX_STATUS = "RECIFE_PUBLIC_COORDINATE_CANDIDATE_FOR_REVIEW"

LAT_FIELD_TERMS = {"lat", "latitude", "y"}
LON_FIELD_TERMS = {"lon", "lng", "long", "longitude", "x"}
COORD_FIELD_TERMS = {"coordenada", "coordenadas", "coord", "geometry", "geometria", "wkt"}
DATE_FIELD_TERMS = {"data", "date", "dt"}
HAZARD_TERMS = {
    "alagamento", "inundacao", "enchente", "chuva", "precipitacao",
    "risco", "deslizamento", "barreira", "encosta", "emergencia",
    "defesa civil", "ocorrencia", "solicitacao", "atendimento",
}
FLOOD_TERMS = {"alagamento", "inundacao", "enchente"}
RAIN_TERMS = {"chuva", "precipitacao", "pluvial"}
LANDSLIDE_TERMS = {"deslizamento", "barreira", "encosta"}
OCCURRENCE_TERMS = {"atendimento", "ocorrencia", "solicitacao", "vistoria", "defesa civil"}
CONTEXT_TERMS = {
    "iluminacao", "parque", "praca", "ciclovia", "ciclavel", "rota",
    "entidade", "equipamento", "descartar", "coleta", "infra",
    "equipment", "facility", "context",
}
ADMIN_TERMS = {"bairro", "rpa", "regional", "limite", "setor"}
RISK_TERMS = {"risco", "suscept", "deslizamento", "alagamento", "inundacao"}

ASSET_LOCATOR_COLUMNS = [
    "coordinate_asset_id", "event_id", "asset_id", "artifact_id", "source_id",
    "resource_name_hash", "asset_type", "row_count",
    "rows_with_coordinates_reported", "has_coordinate_fields",
    "coordinate_field_candidates", "has_geometry", "geometry_type",
    "previous_classification", "should_reparse", "suspected_blocker", "notes",
]
SCHEMA_REPARSE_COLUMNS = [
    "schema_reparse_id", "coordinate_asset_id", "event_id", "asset_id",
    "artifact_id", "asset_type", "local_asset_hash_prefix", "row_count_checked",
    "coordinate_field_candidates", "coordinate_encoding",
    "decimal_comma_detected", "coordinate_inversion_suspected",
    "rows_with_parseable_coordinates", "rows_in_recife_plausible_range",
    "rows_outside_recife_plausible_range", "geometry_type",
    "coordinate_semantics", "reparse_status", "no_coordinates_invented",
    "geocoding_executed", "centroid_used", "notes",
]
GEOJSON_CLASS_COLUMNS = [
    "geojson_context_id", "event_id", "asset_id", "artifact_id",
    "coordinate_asset_id", "geometry_type", "feature_count",
    "property_fields_hash", "source_context_class", "coordinate_role",
    "can_promote_to_occurrence_candidate", "can_create_ground_reference",
    "can_create_training_label", "classification_basis", "notes",
]
JOIN_AUDIT_COLUMNS = [
    "join_audit_id", "event_id", "asset_id", "coordinate_asset_id",
    "candidate_rows_count", "coordinate_rows_count", "matched_row_hash_count",
    "event_window_rows_count", "join_status", "join_blocker",
    "can_create_ground_reference", "can_create_training_label", "notes",
]
WINDOW_FILTER_COLUMNS = [
    "window_coordinate_id", "event_id", "asset_id", "coordinate_asset_id",
    "row_hash", "parsed_date", "event_window_match", "has_coordinates",
    "coordinate_status", "coordinate_role", "raw_values_versioned",
    "no_coordinates_invented", "notes",
]
HAZARD_CROSSFILTER_COLUMNS = [
    "hazard_coordinate_id", "event_id", "asset_id", "coordinate_asset_id",
    "row_hash", "event_window_match", "coordinate_status", "coordinate_role",
    "has_hazard_term", "hazard_class", "hazard_coordinate_status",
    "can_promote_to_coordinate_candidate", "notes",
]
COORDINATE_CANDIDATE_COLUMNS = [
    "coordinate_candidate_id", "event_id", "asset_id", "coordinate_asset_id",
    "candidate_status", "coordinate_role", "event_window_match",
    "hazard_coordinate_status", "max_allowed_status",
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "can_reopen_protocol_b", "dino_usage",
    "no_overlay_executed", "no_coordinates_invented",
    "patch_bound_truth", "operational_validation",
    "coordinate_recovery_from_public_data_only", "geocoding_executed",
    "centroid_used", "blocker", "notes",
]
OVERLAY_BLOCKER_COLUMNS = [
    "overlay_blocker_id", "event_id", "coordinate_candidate_id",
    "asset_id", "candidate_status", "can_execute_overlay_now",
    "no_overlay_executed", "overlay_preflight_status", "blocking_reason",
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "notes",
]
READINESS_UPDATE_COLUMNS = [
    "readiness_update_id", "event_patch_candidate_id", "event_id", "patch_id",
    "region", "dimension", "previous_classification",
    "v1ut_classification", "v1ut_basis", "patch_bound_truth",
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "no_overlay_executed",
    "no_coordinates_invented", "notes",
]
BLOCKER_MATRIX_COLUMNS = [
    "blocker_id", "event_id", "gate", "gate_status", "blocking_reason",
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "can_reopen_protocol_b", "dino_usage",
    "no_overlay_executed", "no_coordinates_invented",
    "patch_bound_truth", "operational_validation", "notes",
]
NEXT_ACTION_COLUMNS = [
    "action_id", "event_id", "action_type", "priority", "description",
    "target", "status", "notes",
]
MANIFEST_COLUMNS = [
    "artifact_id", "artifact_path", "artifact_type", "protocol_version",
    "sha256_prefix", "file_size_bytes", "is_versionable", "reason",
]

V1UT_ARTIFACTS = [
    "configs/protocolo_c/v1ut_recife_coordinate_recovery_policy.yaml",
    "configs/protocolo_c/v1ut_recife_coordinate_field_patterns.yaml",
    "configs/protocolo_c/v1ut_recife_geojson_classification_policy.yaml",
    "configs/protocolo_c/v1ut_recife_event_window_policy.yaml",
    "configs/protocolo_c/v1ut_recife_candidate_scoring_policy.yaml",
    "configs/protocolo_c/v1ut_recife_overlay_blocker_policy.yaml",
    "datasets/protocolo_c/v1ut_recife_coordinate_asset_locator.csv",
    "datasets/protocolo_c/v1ut_recife_coordinate_schema_reparse.csv",
    "datasets/protocolo_c/v1ut_recife_geojson_context_classification.csv",
    "datasets/protocolo_c/v1ut_recife_coordinate_row_join_audit.csv",
    "datasets/protocolo_c/v1ut_recife_event_window_coordinate_filter.csv",
    "datasets/protocolo_c/v1ut_recife_hazard_coordinate_crossfilter.csv",
    "datasets/protocolo_c/v1ut_recife_coordinate_candidate_audit.csv",
    "datasets/protocolo_c/v1ut_recife_overlay_preflight_blocker.csv",
    "datasets/protocolo_c/v1ut_recife_event_patch_readiness_update.csv",
    "datasets/protocolo_c/v1ut_recife_ground_reference_blocker_matrix.csv",
    "datasets/protocolo_c/v1ut_next_actions_registry.csv",
    "datasets/protocolo_c/v1ut_versionable_artifacts_manifest.csv",
    "docs/metodologia_cientifica/protocolo_c_v1ut_recife_coordinate_recovery.md",
    "docs/metodologia_cientifica/protocolo_c_relatorio_v1ut_recife_coordinate_recovery.md",
    "docs/metodologia_cientifica/protocolo_c_status_atual_v1ut.md",
]


def norm(value):
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower().strip()


def token_key(value):
    return re.sub(r"[^a-z0-9]+", "_", norm(value)).strip("_")


def hash_text(value, n=16):
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:n]


def bool_text(value):
    return "true" if bool(value) else "false"


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, columns, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, lines):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def dataset_path(name):
    return os.path.join(DATASET_DIR, name)


def config_path(name):
    return os.path.join(CONFIG_DIR, name)


def doc_path(name):
    return os.path.join(DOCS_DIR, name)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_encoding(raw):
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            pass
    return "utf-8"


def sniff_dialect(sample):
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except Exception:
        class SimpleDialect(csv.excel):
            delimiter = ";"
        return SimpleDialect


def iter_csv_rows(path):
    with open(path, "rb") as f:
        raw = f.read(65536)
    enc = detect_encoding(raw)
    sample = raw.decode(enc, errors="replace")
    dialect = sniff_dialect(sample)
    with open(path, "r", encoding=enc, errors="replace", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        for row in reader:
            yield row, reader.fieldnames or [], enc, dialect.delimiter


def read_csv_rows(path):
    rows = []
    columns = []
    enc = "utf-8"
    delim = ";"
    for row, columns, enc, delim in iter_csv_rows(path):
        rows.append(row)
    return rows, columns, enc, delim


def read_geojson(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        doc = json.load(f)
    feats = doc.get("features", []) if isinstance(doc, dict) else []
    return doc, feats


def parse_safe_filename(fname):
    parts = fname.split("__")
    if len(parts) >= 5:
        return {
            "event_id": parts[0],
            "source_id": parts[1],
            "asset_id": parts[2],
            "url_sha1_12": parts[3],
            "title": "__".join(parts[4:]),
        }
    return {
        "event_id": EVENT_ID,
        "source_id": "ckan",
        "asset_id": hash_text(fname),
        "url_sha1_12": "",
        "title": fname,
    }


def raw_path_for_internal(internal_path, raw_dir=None):
    raw_dir = raw_dir or RAW_DIR
    return os.path.join(raw_dir, os.path.basename(internal_path or ""))


def parse_bool(value):
    return str(value or "").strip().lower() in {"true", "1", "yes", "sim"}


def parse_int(value):
    try:
        return int(float(str(value or "0").replace(",", ".")))
    except ValueError:
        return 0


def parse_float(value):
    text = str(value or "").strip()
    if not text:
        return None
    text = text.replace(" ", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def in_recife_range(lat, lon):
    return (
        lat is not None and lon is not None
        and RECIFE_LAT_MIN <= lat <= RECIFE_LAT_MAX
        and RECIFE_LON_MIN <= lon <= RECIFE_LON_MAX
    )


def coordinate_field_roles(columns):
    lat = []
    lon = []
    generic = []
    for col in columns:
        key = token_key(col)
        if key in LAT_FIELD_TERMS or key.endswith("_lat") or key.startswith("lat_"):
            lat.append(col)
        elif key in LON_FIELD_TERMS or key.endswith("_lon") or key.startswith("lon_") or key.startswith("long"):
            lon.append(col)
        elif any(term in key for term in COORD_FIELD_TERMS):
            generic.append(col)
    return lat, lon, generic


def parse_date(value):
    text = str(value or "").strip()
    if not text:
        return None
    text = text.split()[0]
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def event_window_match(d):
    if not d:
        return "NO_DATE_FIELD"
    if CORE_START <= d <= CORE_END:
        return "REC_2022_CORE_WINDOW"
    return "OUTSIDE_REC_2022_CORE_WINDOW"


def has_any(text, terms):
    nt = norm(text)
    return any(term in nt for term in terms)


def row_hash(asset_id, index, row):
    material = asset_id + "|" + str(index) + "|" + "|".join(
        f"{k}={row.get(k, '')}" for k in sorted(row)
    )
    return hash_text(material, 24)


def pick_first(row, fields):
    for field in fields:
        value = row.get(field, "")
        if str(value or "").strip():
            return value
    return ""


def first_date_from_row(row):
    for key, value in row.items():
        if any(term in token_key(key) for term in DATE_FIELD_TERMS):
            d = parse_date(value)
            if d:
                return d
    return None


def hazard_class_from_text(text):
    nt = norm(text)
    classes = []
    if any(term in nt for term in FLOOD_TERMS):
        classes.append("FLOOD_OR_INUNDATION_TERM")
    if any(term in nt for term in RAIN_TERMS):
        classes.append("RAIN_TERM")
    if any(term in nt for term in LANDSLIDE_TERMS):
        classes.append("LANDSLIDE_OR_SLOPE_TERM")
    if not classes and any(term in nt for term in HAZARD_TERMS):
        classes.append("GENERIC_HAZARD_OR_SERVICE_TERM")
    return "|".join(classes) if classes else "NO_HAZARD_TERM"


def geometry_type_from_geometry(geometry):
    if not isinstance(geometry, dict):
        return ""
    return str(geometry.get("type") or "")


def iter_geojson_coords(geometry):
    gtype = geometry_type_from_geometry(geometry)
    coords = geometry.get("coordinates") if isinstance(geometry, dict) else None
    if not coords:
        return
    if gtype == "Point" and isinstance(coords, list) and len(coords) >= 2:
        yield coords[1], coords[0]
    elif gtype in {"MultiPoint", "LineString"}:
        for pt in coords:
            if isinstance(pt, list) and len(pt) >= 2:
                yield pt[1], pt[0]
    elif gtype in {"MultiLineString", "Polygon"}:
        for part in coords:
            for pt in part:
                if isinstance(pt, list) and len(pt) >= 2:
                    yield pt[1], pt[0]
    elif gtype == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                for pt in ring:
                    if isinstance(pt, list) and len(pt) >= 2:
                        yield pt[1], pt[0]


def source_context_class(name, fields_text, previous_classification=""):
    text = norm(" ".join([name or "", fields_text or "", previous_classification or ""]))
    if any(term in text for term in ADMIN_TERMS):
        return "ADMIN_REGION"
    if any(term in text for term in CONTEXT_TERMS):
        if any(term in text for term in {"iluminacao", "rota", "cicl", "infra"}):
            return "INFRASTRUCTURE_CONTEXT"
        return "FACILITY_OR_EQUIPMENT_CONTEXT"
    if any(term in text for term in RISK_TERMS):
        return "RISK_CONTEXT"
    if any(term in text for term in OCCURRENCE_TERMS):
        return "OCCURRENCE_OR_SERVICE_CALL"
    return "UNKNOWN_CONTEXT"


def coordinate_role_from_context(context_class, previous_classification=""):
    previous = norm(previous_classification)
    if "occurrence_coordinates_candidate" in previous or context_class == "OCCURRENCE_OR_SERVICE_CALL":
        return "OCCURRENCE_OR_SERVICE_CALL_COORDINATE"
    if context_class == "ADMIN_REGION":
        return "ADMIN_REGION_GEOMETRY"
    if context_class == "RISK_CONTEXT":
        return "RISK_CONTEXT_GEOMETRY"
    if context_class in {"FACILITY_OR_EQUIPMENT_CONTEXT", "INFRASTRUCTURE_CONTEXT"}:
        return "CONTEXTUAL_EQUIPMENT_OR_INFRASTRUCTURE_COORDINATE"
    return "UNKNOWN_COORDINATE_ROLE"


def is_contextual_class(context_class, previous_classification=""):
    previous = norm(previous_classification)
    return (
        context_class in {"ADMIN_REGION", "RISK_CONTEXT", "FACILITY_OR_EQUIPMENT_CONTEXT", "INFRASTRUCTURE_CONTEXT"}
        or "context" in previous
        or "infra" in previous
    )


def load_inventory():
    return load_csv(dataset_path("v1uj_focused_artifact_inventory.csv"))


def load_schema():
    return load_csv(dataset_path("v1uk_recife_asset_schema_registry.csv"))


def load_coordinate_audit():
    return load_csv(dataset_path("v1uk_recife_coordinate_evidence_audit.csv"))


def asset_index():
    index = {}
    for inv in load_inventory():
        if inv.get("event_id") != EVENT_ID:
            continue
        fname = os.path.basename(inv.get("internal_path", ""))
        parsed = parse_safe_filename(fname)
        asset_id = parsed["asset_id"]
        row = dict(inv)
        row.update(parsed)
        row["safe_filename"] = fname
        row["raw_path"] = raw_path_for_internal(fname)
        index[asset_id] = row
    return index


def artifact_index():
    return {r.get("inventory_id", ""): r for r in load_inventory()}


def write_policy_configs():
    policies = {
        "v1ut_recife_coordinate_recovery_policy.yaml": [
            "protocol_version: v1ut",
            "coordinate_recovery_from_public_data_only: true",
            "geocoding_allowed: false",
            "centroid_allowed: false",
            "overlay_allowed: false",
            "max_status: RECIFE_PUBLIC_COORDINATE_CANDIDATE_FOR_REVIEW",
        ],
        "v1ut_recife_coordinate_field_patterns.yaml": [
            "latitude_terms: [lat, latitude, y]",
            "longitude_terms: [lon, lng, long, longitude, x]",
            "generic_coordinate_terms: [coordenada, coordenadas, coord, geometry, geometria, wkt]",
            "recife_plausible_latitude: [-9.5, -7.0]",
            "recife_plausible_longitude: [-36.0, -33.0]",
        ],
        "v1ut_recife_geojson_classification_policy.yaml": [
            "promotable_roles: [OCCURRENCE_OR_SERVICE_CALL_COORDINATE]",
            "contextual_roles_blocked: [ADMIN_REGION_GEOMETRY, RISK_CONTEXT_GEOMETRY, CONTEXTUAL_EQUIPMENT_OR_INFRASTRUCTURE_COORDINATE]",
        ],
        "v1ut_recife_event_window_policy.yaml": [
            "event_id: REC_2022_05_24_30",
            "core_start_date: '2022-05-24'",
            "core_end_date: '2022-05-30'",
            "date_required_for_candidate: true",
        ],
        "v1ut_recife_candidate_scoring_policy.yaml": [
            "candidate_requires: [public_source, explicit_coordinate, event_window_date, hazard_term, occurrence_or_service_call_role]",
            "forbidden_statuses: [GROUND_REFERENCE, GROUND_TRUTH, TRAINING_LABEL, PATCH_POSITIVE, PATCH_NEGATIVE, OPERATIONAL_VALIDATED, OBSERVED_FLOOD_LABEL, FLOOD_DETECTED]",
        ],
        "v1ut_recife_overlay_blocker_policy.yaml": [
            "overlay_execution_allowed: false",
            "preflight_only: true",
            "ground_truth_operational: false",
            "can_create_ground_reference: false",
            "can_create_training_label: false",
        ],
    }
    for name, lines in policies.items():
        write_text(config_path(name), lines)


def run_coordinate_asset_locator():
    write_policy_configs()
    inv_by_asset = asset_index()
    audit_by_asset = {r.get("asset_id", ""): r for r in load_coordinate_audit()}
    rows = []
    for schema in load_schema():
        if schema.get("event_id") != EVENT_ID:
            continue
        asset_id = schema.get("asset_id", "")
        audit = audit_by_asset.get(asset_id, {})
        inv = inv_by_asset.get(asset_id, {})
        rows_with_coords = parse_int(audit.get("rows_with_coordinates") or audit.get("rows_with_coordinates_reported"))
        has_coord_fields = parse_bool(schema.get("has_coordinate_fields"))
        asset_type = schema.get("asset_type") or inv.get("asset_type", "")
        has_geometry = asset_type == "geospatial_vector" or parse_int(audit.get("rows_with_coordinates")) > 0 and "GEOMETRY" in audit.get("geometry_status", "")
        previous = audit.get("coordinate_classification") or inv.get("classification", "")
        should_reparse = rows_with_coords > 0 or has_coord_fields or has_geometry
        name_text = inv.get("title") or schema.get("title") or inv.get("safe_filename", "")
        context = source_context_class(name_text, schema.get("coordinate_field_candidates", ""), previous)
        if rows_with_coords == 0 and not has_coord_fields and not has_geometry:
            blocker = "NO_PUBLIC_COORDINATE_FIELDS_OR_GEOMETRY"
        elif is_contextual_class(context, previous):
            blocker = "COORDINATES_CONTEXTUAL_LAYER"
        elif "atendimento" in norm(name_text) and not has_coord_fields:
            blocker = "OCCURRENCE_TABLE_NO_COORDINATE_FIELDS"
        else:
            blocker = "REPARSE_REQUIRED_BEFORE_PROMOTION"
        rows.append({
            "coordinate_asset_id": f"CA_v1ut_{len(rows):05d}",
            "event_id": EVENT_ID,
            "asset_id": asset_id,
            "artifact_id": schema.get("artifact_id", ""),
            "source_id": schema.get("source_id", inv.get("source_id", "ckan")),
            "resource_name_hash": hash_text(name_text, 24),
            "asset_type": asset_type,
            "row_count": schema.get("row_count", ""),
            "rows_with_coordinates_reported": str(rows_with_coords),
            "has_coordinate_fields": bool_text(has_coord_fields),
            "coordinate_field_candidates": schema.get("coordinate_field_candidates", ""),
            "has_geometry": bool_text(has_geometry),
            "geometry_type": "GeoJSON" if asset_type == "geospatial_vector" else "",
            "previous_classification": previous,
            "should_reparse": bool_text(should_reparse),
            "suspected_blocker": blocker,
            "notes": "Locator uses v1uj/v1uk registries only; no promotion or coordinate inference.",
        })
    out = dataset_path("v1ut_recife_coordinate_asset_locator.csv")
    write_csv(out, ASSET_LOCATOR_COLUMNS, rows)
    print(f"[v1ut coordinate asset locator] rows={len(rows)} -> {out}")
    return rows


def _summarize_csv_coordinates(path, asset_id):
    columns = []
    enc = "utf-8"
    delim = ";"
    lat_fields, lon_fields, generic_fields = coordinate_field_roles(columns)
    parseable = in_range = outside = inverted = row_count = 0
    decimal_comma = False
    for idx, (row, current_columns, enc, delim) in enumerate(iter_csv_rows(path)):
        if not columns:
            columns = current_columns
            lat_fields, lon_fields, generic_fields = coordinate_field_roles(columns)
        row_count += 1
        lat_raw = pick_first(row, lat_fields)
        lon_raw = pick_first(row, lon_fields)
        if "," in str(lat_raw) or "," in str(lon_raw):
            decimal_comma = True
        lat = parse_float(lat_raw)
        lon = parse_float(lon_raw)
        swapped = False
        if lat is not None and lon is not None and not in_recife_range(lat, lon) and in_recife_range(lon, lat):
            swapped = True
            inverted += 1
        if lat is None or lon is None:
            continue
        parseable += 1
        if in_recife_range(lat, lon):
            in_range += 1
        elif swapped:
            in_range += 1
        else:
            outside += 1
    return {
        "row_count": row_count,
        "columns": columns,
        "encoding": f"{enc};delimiter={delim}",
        "coordinate_fields": lat_fields + lon_fields + generic_fields,
        "decimal_comma": decimal_comma,
        "inverted": inverted,
        "parseable": parseable,
        "in_range": in_range,
        "outside": outside,
    }


def _summarize_geojson_coordinates(path):
    doc, feats = read_geojson(path)
    gtypes = set()
    in_range = outside = parseable = 0
    fields = set()
    for feat in feats:
        props = feat.get("properties") or {}
        fields.update(props.keys())
        geom = feat.get("geometry") or {}
        gtype = geometry_type_from_geometry(geom)
        if gtype:
            gtypes.add(gtype)
        feature_has_in_range = False
        feature_has_coord = False
        for lat, lon in iter_geojson_coords(geom):
            lat_f, lon_f = parse_float(lat), parse_float(lon)
            if lat_f is None or lon_f is None:
                continue
            feature_has_coord = True
            if in_recife_range(lat_f, lon_f):
                feature_has_in_range = True
        if feature_has_coord:
            parseable += 1
            if feature_has_in_range:
                in_range += 1
            else:
                outside += 1
    return {
        "row_count": len(feats),
        "columns": sorted(fields),
        "geometry_type": "|".join(sorted(gtypes)),
        "parseable": parseable,
        "in_range": in_range,
        "outside": outside,
        "features": feats,
    }


def run_coordinate_schema_reparser():
    locator = load_csv(dataset_path("v1ut_recife_coordinate_asset_locator.csv")) or run_coordinate_asset_locator()
    inv_by_asset = asset_index()
    rows = []
    for loc in locator:
        if not parse_bool(loc.get("should_reparse")):
            continue
        asset_id = loc.get("asset_id", "")
        inv = inv_by_asset.get(asset_id, {})
        path = inv.get("raw_path", "")
        if not path or not os.path.exists(path):
            rows.append({
                "schema_reparse_id": f"SR_v1ut_{len(rows):05d}",
                "coordinate_asset_id": loc.get("coordinate_asset_id", ""),
                "event_id": EVENT_ID,
                "asset_id": asset_id,
                "artifact_id": loc.get("artifact_id", ""),
                "asset_type": loc.get("asset_type", ""),
                "reparse_status": "RAW_LOCAL_ASSET_MISSING",
                "no_coordinates_invented": "true",
                "geocoding_executed": "false",
                "centroid_used": "false",
                "notes": "Raw local asset not found; no fallback inference attempted.",
            })
            continue
        local_hash = sha256_file(path)[:16]
        asset_type = loc.get("asset_type", "")
        if asset_type == "geospatial_vector" or path.lower().endswith(".geojson"):
            summary = _summarize_geojson_coordinates(path)
            context = source_context_class(inv.get("title", ""), "|".join(summary["columns"]), loc.get("previous_classification", ""))
            role = coordinate_role_from_context(context, loc.get("previous_classification", ""))
            rows.append({
                "schema_reparse_id": f"SR_v1ut_{len(rows):05d}",
                "coordinate_asset_id": loc.get("coordinate_asset_id", ""),
                "event_id": EVENT_ID,
                "asset_id": asset_id,
                "artifact_id": loc.get("artifact_id", ""),
                "asset_type": asset_type,
                "local_asset_hash_prefix": local_hash,
                "row_count_checked": str(summary["row_count"]),
                "coordinate_field_candidates": "geometry",
                "coordinate_encoding": "geojson_geometry",
                "decimal_comma_detected": "false",
                "coordinate_inversion_suspected": "false",
                "rows_with_parseable_coordinates": str(summary["parseable"]),
                "rows_in_recife_plausible_range": str(summary["in_range"]),
                "rows_outside_recife_plausible_range": str(summary["outside"]),
                "geometry_type": summary["geometry_type"],
                "coordinate_semantics": role,
                "reparse_status": "REPARSED_PUBLIC_GEOMETRY",
                "no_coordinates_invented": "true",
                "geocoding_executed": "false",
                "centroid_used": "false",
                "notes": "GeoJSON coordinates counted from public geometry only; no centroid generated.",
            })
        else:
            summary = _summarize_csv_coordinates(path, asset_id)
            context = source_context_class(inv.get("title", ""), "|".join(summary["columns"]), loc.get("previous_classification", ""))
            role = coordinate_role_from_context(context, loc.get("previous_classification", ""))
            rows.append({
                "schema_reparse_id": f"SR_v1ut_{len(rows):05d}",
                "coordinate_asset_id": loc.get("coordinate_asset_id", ""),
                "event_id": EVENT_ID,
                "asset_id": asset_id,
                "artifact_id": loc.get("artifact_id", ""),
                "asset_type": asset_type,
                "local_asset_hash_prefix": local_hash,
                "row_count_checked": str(summary["row_count"]),
                "coordinate_field_candidates": "|".join(summary["coordinate_fields"]),
                "coordinate_encoding": summary["encoding"],
                "decimal_comma_detected": bool_text(summary["decimal_comma"]),
                "coordinate_inversion_suspected": bool_text(summary["inverted"] > 0),
                "rows_with_parseable_coordinates": str(summary["parseable"]),
                "rows_in_recife_plausible_range": str(summary["in_range"]),
                "rows_outside_recife_plausible_range": str(summary["outside"]),
                "geometry_type": "",
                "coordinate_semantics": role,
                "reparse_status": "REPARSED_PUBLIC_COORDINATE_FIELDS",
                "no_coordinates_invented": "true",
                "geocoding_executed": "false",
                "centroid_used": "false",
                "notes": "CSV coordinates parsed from explicit fields only; ambiguous fields not fixed.",
            })
    out = dataset_path("v1ut_recife_coordinate_schema_reparse.csv")
    write_csv(out, SCHEMA_REPARSE_COLUMNS, rows)
    print(f"[v1ut coordinate schema reparse] rows={len(rows)} -> {out}")
    return rows


def run_geojson_context_classifier():
    locator = load_csv(dataset_path("v1ut_recife_coordinate_asset_locator.csv")) or run_coordinate_asset_locator()
    inv_by_asset = asset_index()
    rows = []
    for loc in locator:
        asset_id = loc.get("asset_id", "")
        inv = inv_by_asset.get(asset_id, {})
        path = inv.get("raw_path", "")
        if loc.get("asset_type") != "geospatial_vector" and not str(path).lower().endswith(".geojson"):
            continue
        if not os.path.exists(path):
            continue
        summary = _summarize_geojson_coordinates(path)
        fields_hash = hash_text("|".join(summary["columns"]), 24)
        context = source_context_class(inv.get("title", ""), "|".join(summary["columns"]), loc.get("previous_classification", ""))
        role = coordinate_role_from_context(context, loc.get("previous_classification", ""))
        promotable = role == "OCCURRENCE_OR_SERVICE_CALL_COORDINATE"
        rows.append({
            "geojson_context_id": f"GJ_v1ut_{len(rows):05d}",
            "event_id": EVENT_ID,
            "asset_id": asset_id,
            "artifact_id": loc.get("artifact_id", ""),
            "coordinate_asset_id": loc.get("coordinate_asset_id", ""),
            "geometry_type": summary["geometry_type"],
            "feature_count": str(summary["row_count"]),
            "property_fields_hash": fields_hash,
            "source_context_class": context,
            "coordinate_role": role,
            "can_promote_to_occurrence_candidate": bool_text(promotable),
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "classification_basis": "filename_and_property_schema_no_overlay",
            "notes": "GeoJSON context classification only; contextual geometry cannot be promoted.",
        })
    out = dataset_path("v1ut_recife_geojson_context_classification.csv")
    write_csv(out, GEOJSON_CLASS_COLUMNS, rows)
    print(f"[v1ut geojson classifier] rows={len(rows)} -> {out}")
    return rows


def run_coordinate_row_join_audit():
    reparse = load_csv(dataset_path("v1ut_recife_coordinate_schema_reparse.csv")) or run_coordinate_schema_reparser()
    candidate_rows = load_csv(dataset_path("v1uk_recife_candidate_row_registry.csv"))
    event_rows = load_csv(dataset_path("v1uk_recife_event_window_match_registry.csv"))
    cand_count = {}
    event_count = {}
    for r in candidate_rows:
        cand_count[r.get("asset_id", "")] = cand_count.get(r.get("asset_id", ""), 0) + 1
    for r in event_rows:
        event_count[r.get("asset_id", "")] = event_count.get(r.get("asset_id", ""), 0) + 1
    rows = []
    for rep in reparse:
        asset_id = rep.get("asset_id", "")
        coord_rows = parse_int(rep.get("rows_in_recife_plausible_range"))
        crows = cand_count.get(asset_id, 0)
        erows = event_count.get(asset_id, 0)
        if coord_rows and crows:
            status = "COORDINATE_AND_CANDIDATE_SAME_ASSET_NEEDS_ROW_HASH_CONFIRMATION"
            blocker = "ROW_LEVEL_JOIN_NOT_CONFIRMED"
        elif coord_rows and not crows:
            status = "COORDINATE_ROWS_IN_DIFFERENT_CONTEXT_ASSET"
            blocker = "COORDINATE_ASSET_NOT_EVENT_CANDIDATE_TABLE"
        elif crows and not coord_rows:
            status = "EVENT_CANDIDATE_ROWS_WITHOUT_COORDINATES"
            blocker = "OCCURRENCE_TABLE_NO_COORDINATE_ROWS"
        else:
            status = "NO_JOINABLE_COORDINATE_ROWS"
            blocker = "NO_COORDINATE_EVIDENCE"
        rows.append({
            "join_audit_id": f"JA_v1ut_{len(rows):05d}",
            "event_id": EVENT_ID,
            "asset_id": asset_id,
            "coordinate_asset_id": rep.get("coordinate_asset_id", ""),
            "candidate_rows_count": str(crows),
            "coordinate_rows_count": str(coord_rows),
            "matched_row_hash_count": "0",
            "event_window_rows_count": str(erows),
            "join_status": status,
            "join_blocker": blocker,
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": "No raw row values are versioned; join remains hash/count based.",
        })
    out = dataset_path("v1ut_recife_coordinate_row_join_audit.csv")
    write_csv(out, JOIN_AUDIT_COLUMNS, rows)
    print(f"[v1ut row join audit] rows={len(rows)} -> {out}")
    return rows


def _window_rows_for_asset(loc, limit_per_asset=250):
    inv = asset_index().get(loc.get("asset_id", ""), {})
    path = inv.get("raw_path", "")
    if not os.path.exists(path):
        return []
    out = []
    role = loc.get("coordinate_semantics") or loc.get("coordinate_role") or ""
    if path.lower().endswith(".geojson") or loc.get("asset_type") == "geospatial_vector":
        _doc, feats = read_geojson(path)
        for idx, feat in enumerate(feats):
            props = feat.get("properties") or {}
            d = first_date_from_row(props)
            geom = feat.get("geometry") or {}
            has_coord = any(in_recife_range(parse_float(lat), parse_float(lon)) for lat, lon in iter_geojson_coords(geom))
            if not has_coord and not d:
                continue
            out.append((row_hash(loc.get("asset_id", ""), idx, props), d, has_coord, props))
            if len(out) >= limit_per_asset:
                break
    else:
        lat_fields = []
        lon_fields = []
        for idx, (row, columns, _enc, _delim) in enumerate(iter_csv_rows(path)):
            if idx == 0:
                lat_fields, lon_fields, _generic = coordinate_field_roles(columns)
            lat = parse_float(pick_first(row, lat_fields))
            lon = parse_float(pick_first(row, lon_fields))
            has_coord = in_recife_range(lat, lon)
            d = first_date_from_row(row)
            if not has_coord and not d:
                continue
            out.append((row_hash(loc.get("asset_id", ""), idx, row), d, has_coord, row))
            if len(out) >= limit_per_asset:
                break
    return out


def _row_hazard_index_for_asset(rep, limit_per_asset=250):
    inv = asset_index().get(rep.get("asset_id", ""), {})
    path = inv.get("raw_path", "")
    if not os.path.exists(path):
        return {}
    out = {}
    if path.lower().endswith(".geojson") or rep.get("asset_type") == "geospatial_vector":
        _doc, feats = read_geojson(path)
        for idx, feat in enumerate(feats[:limit_per_asset]):
            props = feat.get("properties") or {}
            out[row_hash(rep.get("asset_id", ""), idx, props)] = hazard_class_from_text(" ".join(str(v or "") for v in props.values()))
    else:
        for idx, (row, _columns, _enc, _delim) in enumerate(iter_csv_rows(path)):
            if idx >= limit_per_asset:
                break
            out[row_hash(rep.get("asset_id", ""), idx, row)] = hazard_class_from_text(" ".join(str(v or "") for v in row.values()))
    return out


def run_event_window_coordinate_filter():
    reparse = load_csv(dataset_path("v1ut_recife_coordinate_schema_reparse.csv")) or run_coordinate_schema_reparser()
    rows = []
    for rep in reparse:
        if parse_int(rep.get("rows_in_recife_plausible_range")) == 0:
            continue
        sampled = _window_rows_for_asset(rep)
        if not sampled:
            rows.append({
                "window_coordinate_id": f"WC_v1ut_{len(rows):05d}",
                "event_id": EVENT_ID,
                "asset_id": rep.get("asset_id", ""),
                "coordinate_asset_id": rep.get("coordinate_asset_id", ""),
                "row_hash": "ASSET_SUMMARY_" + hash_text(rep.get("asset_id", ""), 12),
                "parsed_date": "",
                "event_window_match": "NO_DATE_FIELD",
                "has_coordinates": "true",
                "coordinate_status": "PUBLIC_COORDINATES_PRESENT_NO_DATE_JOIN",
                "coordinate_role": rep.get("coordinate_semantics", ""),
                "raw_values_versioned": "false",
                "no_coordinates_invented": "true",
                "notes": "Coordinate asset has no row-level event date field suitable for REC_2022 window.",
            })
            continue
        for rh, d, has_coord, _row in sampled:
            rows.append({
                "window_coordinate_id": f"WC_v1ut_{len(rows):05d}",
                "event_id": EVENT_ID,
                "asset_id": rep.get("asset_id", ""),
                "coordinate_asset_id": rep.get("coordinate_asset_id", ""),
                "row_hash": rh,
                "parsed_date": d.isoformat() if d else "",
                "event_window_match": event_window_match(d),
                "has_coordinates": bool_text(has_coord),
                "coordinate_status": "PUBLIC_COORDINATE_IN_RECIFE_RANGE" if has_coord else "NO_VALID_COORDINATE_IN_ROW",
                "coordinate_role": rep.get("coordinate_semantics", ""),
                "raw_values_versioned": "false",
                "no_coordinates_invented": "true",
                "notes": "Hashed row-level event-window coordinate filter; values remain local only.",
            })
    out = dataset_path("v1ut_recife_event_window_coordinate_filter.csv")
    write_csv(out, WINDOW_FILTER_COLUMNS, rows)
    print(f"[v1ut event-window coordinate filter] rows={len(rows)} -> {out}")
    return rows


def run_hazard_coordinate_crossfilter():
    window_rows = load_csv(dataset_path("v1ut_recife_event_window_coordinate_filter.csv")) or run_event_window_coordinate_filter()
    reparse_by_asset = {r.get("asset_id", ""): r for r in load_csv(dataset_path("v1ut_recife_coordinate_schema_reparse.csv"))}
    hazard_indexes = {}
    rows = []
    for wr in window_rows:
        asset_id = wr.get("asset_id", "")
        rep = reparse_by_asset.get(asset_id, {})
        inv = asset_index().get(asset_id, {})
        path = inv.get("raw_path", "")
        if asset_id not in hazard_indexes:
            hazard_indexes[asset_id] = _row_hazard_index_for_asset(rep)
        # Re-open only enough local row context to classify the hash; exact raw
        # values are intentionally not written to versionable outputs.
        hazard_class = hazard_indexes[asset_id].get(wr.get("row_hash", ""), "NO_HAZARD_TERM")
        if hazard_class == "NO_HAZARD_TERM":
            name_context = inv.get("title", "") + " " + rep.get("coordinate_semantics", "")
            hazard_class = hazard_class_from_text(name_context)
        has_hazard = hazard_class != "NO_HAZARD_TERM"
        promotable = (
            wr.get("event_window_match") == "REC_2022_CORE_WINDOW"
            and wr.get("coordinate_status") == "PUBLIC_COORDINATE_IN_RECIFE_RANGE"
            and has_hazard
            and rep.get("coordinate_semantics") == "OCCURRENCE_OR_SERVICE_CALL_COORDINATE"
        )
        if promotable:
            status = "COORDINATE_WINDOW_HAZARD_CANDIDATE_FOR_REVIEW"
        elif rep.get("coordinate_semantics") != "OCCURRENCE_OR_SERVICE_CALL_COORDINATE":
            status = "CONTEXTUAL_COORDINATE_NOT_PROMOTABLE"
        elif wr.get("event_window_match") != "REC_2022_CORE_WINDOW":
            status = "COORDINATE_NOT_IN_EVENT_WINDOW"
        elif not has_hazard:
            status = "COORDINATE_WITHOUT_HAZARD_TERM"
        else:
            status = "NO_COORDINATE_HAZARD_MATCH"
        rows.append({
            "hazard_coordinate_id": f"HC_v1ut_{len(rows):05d}",
            "event_id": EVENT_ID,
            "asset_id": asset_id,
            "coordinate_asset_id": wr.get("coordinate_asset_id", ""),
            "row_hash": wr.get("row_hash", ""),
            "event_window_match": wr.get("event_window_match", ""),
            "coordinate_status": wr.get("coordinate_status", ""),
            "coordinate_role": rep.get("coordinate_semantics", ""),
            "has_hazard_term": bool_text(has_hazard),
            "hazard_class": hazard_class,
            "hazard_coordinate_status": status,
            "can_promote_to_coordinate_candidate": bool_text(promotable),
            "notes": f"Public asset context only; local asset path present={bool(path)}; no raw values versioned.",
        })
    out = dataset_path("v1ut_recife_hazard_coordinate_crossfilter.csv")
    write_csv(out, HAZARD_CROSSFILTER_COLUMNS, rows)
    print(f"[v1ut hazard coordinate crossfilter] rows={len(rows)} -> {out}")
    return rows


def run_coordinate_candidate_audit():
    hazards = load_csv(dataset_path("v1ut_recife_hazard_coordinate_crossfilter.csv")) or run_hazard_coordinate_crossfilter()
    rows = []
    by_asset = {}
    for h in hazards:
        by_asset.setdefault(h.get("asset_id", ""), []).append(h)
    if not by_asset:
        by_asset[""] = []
    for asset_id, hs in by_asset.items():
        promotable = [h for h in hs if parse_bool(h.get("can_promote_to_coordinate_candidate"))]
        if promotable:
            status = MAX_STATUS
            blocker = "OVERLAY_PREFLIGHT_NOT_EXECUTED_GROUND_REFERENCE_BLOCKED"
            role = promotable[0].get("coordinate_role", "")
            window = promotable[0].get("event_window_match", "")
            hazard = promotable[0].get("hazard_coordinate_status", "")
            coord_asset = promotable[0].get("coordinate_asset_id", "")
        elif hs:
            status = "NO_RECIFE_COORDINATE_CANDIDATE_CONTEXT_ONLY"
            blocker = "NO_ROW_COMBINES_OCCURRENCE_ROLE_EVENT_WINDOW_HAZARD_AND_COORDINATE"
            role = hs[0].get("coordinate_role", "")
            window = "|".join(sorted({h.get("event_window_match", "") for h in hs if h.get("event_window_match")}))
            hazard = "|".join(sorted({h.get("hazard_coordinate_status", "") for h in hs if h.get("hazard_coordinate_status")}))
            coord_asset = hs[0].get("coordinate_asset_id", "")
        else:
            status = "NO_RECIFE_PUBLIC_COORDINATE_CANDIDATE"
            blocker = "NO_PUBLIC_COORDINATE_HAZARD_WINDOW_MATCH"
            role = ""
            window = ""
            hazard = ""
            coord_asset = ""
        rows.append({
            "coordinate_candidate_id": f"CC_v1ut_{len(rows):05d}",
            "event_id": EVENT_ID,
            "asset_id": asset_id,
            "coordinate_asset_id": coord_asset,
            "candidate_status": status,
            "coordinate_role": role,
            "event_window_match": window,
            "hazard_coordinate_status": hazard,
            "max_allowed_status": MAX_STATUS,
            "ground_truth_operational": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "can_reopen_protocol_b": "false",
            "dino_usage": "SUPPORT_ONLY",
            "no_overlay_executed": "true",
            "no_coordinates_invented": "true",
            "patch_bound_truth": "false",
            "operational_validation": "false",
            "coordinate_recovery_from_public_data_only": "true",
            "geocoding_executed": "false",
            "centroid_used": "false",
            "blocker": blocker,
            "notes": "Candidate audit capped at review-only status; no labels, no truth, no overlay.",
        })
    out = dataset_path("v1ut_recife_coordinate_candidate_audit.csv")
    write_csv(out, COORDINATE_CANDIDATE_COLUMNS, rows)
    print(f"[v1ut coordinate candidate audit] rows={len(rows)} -> {out}")
    return rows


def run_overlay_preflight_blocker():
    candidates = load_csv(dataset_path("v1ut_recife_coordinate_candidate_audit.csv")) or run_coordinate_candidate_audit()
    rows = []
    for cand in candidates:
        has_review_candidate = cand.get("candidate_status") == MAX_STATUS
        rows.append({
            "overlay_blocker_id": f"OB_v1ut_{len(rows):05d}",
            "event_id": cand.get("event_id", EVENT_ID),
            "coordinate_candidate_id": cand.get("coordinate_candidate_id", ""),
            "asset_id": cand.get("asset_id", ""),
            "candidate_status": cand.get("candidate_status", ""),
            "can_execute_overlay_now": "false",
            "no_overlay_executed": "true",
            "overlay_preflight_status": "FUTURE_PREFLIGHT_REQUIRED" if has_review_candidate else "BLOCKED_NO_COORDINATE_CANDIDATE",
            "blocking_reason": "review_candidate_requires_future_preflight_no_overlay_now" if has_review_candidate else "no_coordinate_candidate_for_overlay_preflight",
            "ground_truth_operational": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": "Overlay preflight blocker only; no spatial operation executed in v1ut.",
        })
    out = dataset_path("v1ut_recife_overlay_preflight_blocker.csv")
    write_csv(out, OVERLAY_BLOCKER_COLUMNS, rows)
    print(f"[v1ut overlay preflight blocker] rows={len(rows)} -> {out}")
    return rows


def run_event_patch_readiness_updater():
    candidates = load_csv(dataset_path("v1ut_recife_coordinate_candidate_audit.csv")) or run_coordinate_candidate_audit()
    has_review_candidate = any(r.get("candidate_status") == MAX_STATUS for r in candidates)
    v1us_cands = [r for r in load_csv(dataset_path("v1us_event_patch_candidate_registry.csv")) if r.get("event_id") == EVENT_ID or r.get("region") == "REC"]
    v1us_ready = load_csv(dataset_path("v1us_event_patch_readiness_matrix.csv"))
    prev_by_candidate_dim = {
        (r.get("event_patch_candidate_id", ""), r.get("dimension", "")): r.get("classification", "")
        for r in v1us_ready
    }
    classification = "REVIEW_CANDIDATE" if has_review_candidate else "BLOCKED_CONTEXT_ONLY_OR_NO_COORDINATE_JOIN"
    basis = "v1ut found a public coordinate hazard window review candidate" if has_review_candidate else "v1ut found contextual coordinates but no event-window hazard coordinate row"
    rows = []
    for cand in v1us_cands:
        for dim in ("coordinate_support", "overlay_readiness", "ground_reference_readiness", "training_readiness"):
            v1ut_cls = classification if dim == "coordinate_support" else "BLOCKED"
            rows.append({
                "readiness_update_id": f"RU_v1ut_{len(rows):05d}",
                "event_patch_candidate_id": cand.get("event_patch_candidate_id", ""),
                "event_id": cand.get("event_id", ""),
                "patch_id": cand.get("patch_id", ""),
                "region": cand.get("region", ""),
                "dimension": dim,
                "previous_classification": prev_by_candidate_dim.get((cand.get("event_patch_candidate_id", ""), dim), ""),
                "v1ut_classification": v1ut_cls,
                "v1ut_basis": basis,
                "patch_bound_truth": "false",
                "ground_truth_operational": "false",
                "can_create_ground_reference": "false",
                "can_create_training_label": "false",
                "no_overlay_executed": "true",
                "no_coordinates_invented": "true",
                "notes": "v1ut readiness update is additive; v1us registry is not modified.",
            })
    out = dataset_path("v1ut_recife_event_patch_readiness_update.csv")
    write_csv(out, READINESS_UPDATE_COLUMNS, rows)
    print(f"[v1ut readiness update] rows={len(rows)} -> {out}")
    return rows


def run_completion_report():
    locator = load_csv(dataset_path("v1ut_recife_coordinate_asset_locator.csv")) or run_coordinate_asset_locator()
    reparse = load_csv(dataset_path("v1ut_recife_coordinate_schema_reparse.csv")) or run_coordinate_schema_reparser()
    candidates = load_csv(dataset_path("v1ut_recife_coordinate_candidate_audit.csv")) or run_coordinate_candidate_audit()
    overlay = load_csv(dataset_path("v1ut_recife_overlay_preflight_blocker.csv")) or run_overlay_preflight_blocker()
    readiness = load_csv(dataset_path("v1ut_recife_event_patch_readiness_update.csv")) or run_event_patch_readiness_updater()
    coord_assets = sum(1 for r in locator if parse_int(r.get("rows_with_coordinates_reported")) > 0 or parse_bool(r.get("has_coordinate_fields")) or parse_bool(r.get("has_geometry")))
    rows_reported = sum(parse_int(r.get("rows_with_coordinates_reported")) for r in locator)
    rows_reparsed = sum(parse_int(r.get("rows_in_recife_plausible_range")) for r in reparse)
    review_candidates = sum(1 for r in candidates if r.get("candidate_status") == MAX_STATUS)
    if review_candidates:
        next_action = "v1uu - Recife Coordinate Candidate Overlay Preflight"
        next_reason = "review-only coordinate candidate exists but overlay is blocked until a future preflight"
    elif coord_assets:
        next_action = "v1uu - Recife Contextual Coordinate Layer Consolidation"
        next_reason = "public coordinates exist, but they are contextual or not joined to event-window hazard rows"
    else:
        next_action = "v1uu - Curitiba Event Registry and Public Source Discovery"
        next_reason = "no Recife public coordinate asset was recoverable"
    blockers = [
        {
            "blocker_id": "GB_v1ut_0000", "event_id": EVENT_ID,
            "gate": "ground_truth_operational", "gate_status": "BLOCKED",
            "blocking_reason": "v1ut is review-only coordinate recovery from public data",
            "ground_truth_operational": "false", "can_create_ground_reference": "false",
            "can_create_training_label": "false", "can_reopen_protocol_b": "false",
            "dino_usage": "SUPPORT_ONLY", "no_overlay_executed": "true",
            "no_coordinates_invented": "true", "patch_bound_truth": "false",
            "operational_validation": "false", "notes": "No operational validation or truth claim.",
        },
        {
            "blocker_id": "GB_v1ut_0001", "event_id": EVENT_ID,
            "gate": "coordinate_candidate", "gate_status": "REVIEW_ONLY" if review_candidates else "BLOCKED",
            "blocking_reason": "max status is RECIFE_PUBLIC_COORDINATE_CANDIDATE_FOR_REVIEW" if review_candidates else "no row combines event window, hazard, occurrence role and explicit coordinate",
            "ground_truth_operational": "false", "can_create_ground_reference": "false",
            "can_create_training_label": "false", "can_reopen_protocol_b": "false",
            "dino_usage": "SUPPORT_ONLY", "no_overlay_executed": "true",
            "no_coordinates_invented": "true", "patch_bound_truth": "false",
            "operational_validation": "false", "notes": "Coordinate review candidate cannot become label/truth.",
        },
    ]
    write_csv(dataset_path("v1ut_recife_ground_reference_blocker_matrix.csv"), BLOCKER_MATRIX_COLUMNS, blockers)
    next_rows = [{
        "action_id": "NA_v1ut_0000", "event_id": EVENT_ID,
        "action_type": next_action, "priority": "1",
        "description": next_reason, "target": "RECIFE_COORDINATE_RECOVERY",
        "status": "RECOMMENDED_NEXT_STEP", "notes": "Selected from v1ut observed blockers.",
    }]
    write_csv(dataset_path("v1ut_next_actions_registry.csv"), NEXT_ACTION_COLUMNS, next_rows)
    docs_common = [
        "# Protocolo C v1ut - Recife Coordinate Recovery from Public CKAN",
        "",
        f"- event_id: `{EVENT_ID}`",
        f"- coordinate assets located: `{coord_assets}`",
        f"- rows with coordinates reported by v1uk: `{rows_reported}`",
        f"- rows in Recife plausible range after reparse: `{rows_reparsed}`",
        f"- review-only coordinate candidates: `{review_candidates}`",
        f"- max status: `{MAX_STATUS}`",
        "- ground_truth_operational: `false`",
        "- can_create_ground_reference: `false`",
        "- can_create_training_label: `false`",
        "- can_reopen_protocol_b: `false`",
        "- dino_usage: `SUPPORT_ONLY`",
        "- no_overlay_executed: `true`",
        "- no_coordinates_invented: `true`",
        "- geocoding_executed: `false`",
        "- centroid_used: `false`",
        "",
        "v1ut recovers only coordinates explicitly present in public CKAN assets already downloaded locally. Raw coordinate values remain local-only; versionable outputs store counts, hashes, classifications and blockers.",
        "",
        f"Next recommended action: `{next_action}` because {next_reason}.",
    ]
    write_text(doc_path("protocolo_c_v1ut_recife_coordinate_recovery.md"), docs_common)
    write_text(doc_path("protocolo_c_relatorio_v1ut_recife_coordinate_recovery.md"), docs_common + [
        "",
        "## Result",
        "REC_2022 does not become ground reference, ground truth, patch positive, patch negative, observed flood label, flood detected or operationally validated in v1ut.",
    ])
    write_text(doc_path("protocolo_c_status_atual_v1ut.md"), [
        "# Status atual - Protocolo C v1ut",
        "",
        f"REC_2022 coordinate recovery status: `{MAX_STATUS if review_candidates else 'NO_RECIFE_COORDINATE_CANDIDATE_CONTEXT_ONLY'}`.",
        f"Next action: `{next_action}`.",
        "",
        "All ground-reference, training, Protocol B reopening, overlay and operational validation gates remain blocked.",
    ])
    manifest = []
    def resolve_artifact_path(artifact):
        base = os.path.basename(artifact)
        if artifact.startswith("datasets/protocolo_c/"):
            return dataset_path(base)
        if artifact.startswith("configs/protocolo_c/"):
            return config_path(base)
        if artifact.startswith("docs/metodologia_cientifica/"):
            return doc_path(base)
        return artifact

    for idx, artifact in enumerate(V1UT_ARTIFACTS):
        real_path = resolve_artifact_path(artifact)
        if not os.path.exists(real_path):
            continue
        manifest.append({
            "artifact_id": f"MAN_v1ut_{idx:04d}",
            "artifact_path": artifact.replace("\\", "/"),
            "artifact_type": os.path.splitext(artifact)[1].lstrip(".") or "text",
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha256_file(real_path)[:16],
            "file_size_bytes": str(os.path.getsize(real_path)),
            "is_versionable": "true",
            "reason": "v1ut public coordinate recovery audit artifact; no raw private path.",
        })
    write_csv(dataset_path("v1ut_versionable_artifacts_manifest.csv"), MANIFEST_COLUMNS, manifest)
    os.makedirs(STAGING_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    print(
        f"[v1ut completion] locator={len(locator)} reparse={len(reparse)} "
        f"candidates={len(candidates)} overlay={len(overlay)} readiness={len(readiness)}"
    )
    return {
        "coordinate_assets": coord_assets,
        "rows_with_coordinates_reported": rows_reported,
        "rows_in_recife_range": rows_reparsed,
        "review_candidates": review_candidates,
        "next_action": next_action,
    }


def run_all():
    run_coordinate_asset_locator()
    run_coordinate_schema_reparser()
    run_geojson_context_classifier()
    run_coordinate_row_join_audit()
    run_event_window_coordinate_filter()
    run_hazard_coordinate_crossfilter()
    run_coordinate_candidate_audit()
    run_overlay_preflight_blocker()
    run_event_patch_readiness_updater()
    return run_completion_report()
