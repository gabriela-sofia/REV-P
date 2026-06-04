#!/usr/bin/env python3
"""Shared utilities for v1uk Recife CKAN schema and event-window audit."""

import argparse
import csv
import hashlib
import json
import os
import re
import unicodedata
from datetime import date, datetime, timedelta

PROTOCOL_VERSION = "v1uk"
EVENT_ID = "REC_2022_05_24_30"
CORE_START = date(2022, 5, 24)
CORE_END = date(2022, 5, 30)
PRE_3_START = CORE_START - timedelta(days=3)
PRE_7_START = CORE_START - timedelta(days=7)
POST_3_END = CORE_END + timedelta(days=3)

RAW_DIR = "local_only/protocolo_c/focused_public_artifacts/raw/v1uj/ckan/REC_2022_05_24_30"
DATASET_DIR = "datasets/protocolo_c"
DOCS_DIR = "docs/metodologia_cientifica"

SENSITIVE_TERMS = {
    "endereco", "logradouro", "numero", "protocolo", "processo",
    "descricao", "solicitacao_descricao", "nome", "avaliador", "cpf",
    "telefone", "email", "documento", "observacao",
}
DATE_TERMS = {"data", "date", "dt", "ano", "mes"}
HAZARD_TERMS = {
    "ocorrencia", "ocorrencia", "risco", "alagamento", "inundacao",
    "chuva", "deslizamento", "barreira", "emergencia", "desastre",
    "vistoria", "solicitacao", "atendimento",
}
FLOOD_TERMS = {"alagamento", "inundacao", "enchente", "flood"}
RAIN_TERMS = {"chuva", "pluvial", "precipitacao", "rain"}
LANDSLIDE_TERMS = {"deslizamento", "barreira", "encosta", "landslide"}
LOCALITY_TERMS = {"bairro", "localidade", "regional", "rpa", "microrregiao", "setor"}
ADDRESS_TERMS = {"endereco", "logradouro", "rua", "avenida", "numero"}
LAT_TERMS = {"latitude", "lat"}
LON_TERMS = {"longitude", "lon", "lng"}

RELEVANT_CLASSIFICATIONS = {
    "TABLE_WITH_COORDINATES_CANDIDATE_FOR_REVIEW",
    "DOCUMENTED_OCCURRENCE_TABLE_NO_GEOMETRY",
    "CONTEXTUAL_OFFICIAL_LAYER",
    "CONTEXT_ONLY",
}
RELEVANT_NAME_TERMS = {
    "defesa", "civil", "atendimento", "ocorrencia", "ocorrencias",
    "alagamento", "inundacao", "risco", "desastre", "emergencia",
    "coordenadas", "sedec", "vistoria", "solicitacao",
}


def norm(value):
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower().strip()


def token_key(value):
    return re.sub(r"[^a-z0-9]+", "_", norm(value)).strip("_")


def hash_text(value, n=16):
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:n]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, columns, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


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
    return {"event_id": EVENT_ID, "source_id": "ckan", "asset_id": hash_text(fname),
            "url_sha1_12": "", "title": fname}


def local_path_for_internal(internal_path, raw_dir=RAW_DIR):
    return os.path.join(raw_dir, os.path.basename(internal_path))


def discover_assets(inventory_path=None, manifest_path=None, raw_dir=RAW_DIR):
    inventory_path = (inventory_path or os.environ.get("V1UK_INVENTORY_PATH")
                      or os.path.join(DATASET_DIR, "v1uj_focused_artifact_inventory.csv"))
    manifest_path = (manifest_path or os.environ.get("V1UK_MANIFEST_PATH")
                     or os.path.join(DATASET_DIR, "v1uj_focused_download_manifest.csv"))
    manifest = {r.get("safe_filename", ""): r for r in load_csv(manifest_path)}
    assets = []
    for inv in load_csv(inventory_path):
        if inv.get("event_id") != EVENT_ID:
            continue
        internal = inv.get("internal_path", "")
        fname = os.path.basename(internal)
        meta = parse_safe_filename(fname)
        name_text = norm(fname + " " + inv.get("columns_detected", "") + " " + inv.get("classification", ""))
        relevant = (inv.get("classification") in RELEVANT_CLASSIFICATIONS
                    or any(t in name_text for t in RELEVANT_NAME_TERMS))
        if not relevant:
            continue
        path = os.path.join(raw_dir, fname)
        if not os.path.exists(path):
            continue
        m = manifest.get(fname, {})
        assets.append({
            "event_id": EVENT_ID,
            "artifact_id": inv.get("inventory_id", ""),
            "asset_id": meta["asset_id"],
            "source_id": meta["source_id"],
            "title": meta["title"],
            "internal_path": fname,
            "path": path,
            "asset_type": inv.get("asset_type", ""),
            "extension": inv.get("extension", os.path.splitext(fname)[1].lower()),
            "v1uj_classification": inv.get("classification", ""),
            "sha256": m.get("sha256") or inv.get("sha256") or sha256_file(path),
            "file_size_bytes": m.get("file_size_bytes") or inv.get("file_size_bytes", ""),
        })
    return assets


def detect_encoding(raw):
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "utf-8"


def sniff_dialect(sample):
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except Exception:
        class SimpleDialect(csv.excel):
            delimiter = ";"
        return SimpleDialect


def read_csv_rows(path):
    with open(path, "rb") as f:
        raw = f.read(65536)
    enc = detect_encoding(raw)
    sample = raw.decode(enc, errors="replace")
    dialect = sniff_dialect(sample)
    with open(path, "r", encoding=enc, errors="replace", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        rows = list(reader)
    columns = reader.fieldnames or []
    return rows, columns, enc, dialect.delimiter


def read_geojson(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        doc = json.load(f)
    feats = doc.get("features", []) if isinstance(doc, dict) else []
    crs = ""
    if isinstance(doc, dict) and isinstance(doc.get("crs"), dict):
        crs = str(doc.get("crs", {}).get("properties", {}).get("name", ""))
    return doc, feats, crs


def apparent_type(values):
    vals = [v for v in values if str(v or "").strip()][:50]
    if not vals:
        return "empty"
    parsed_dates = sum(1 for v in vals if parse_date(v))
    if parsed_dates >= max(1, len(vals) // 2):
        return "date"
    nums = 0
    for v in vals:
        try:
            float(str(v).replace(",", "."))
            nums += 1
        except ValueError:
            pass
    if nums >= max(1, len(vals) // 2):
        return "number"
    return "text"


def field_role(column):
    c = token_key(column)
    if c in {"data", "data_demanda", "data_da_acao", "solicitacao_data", "vistoria_data"}:
        return "event_date"
    if "hora" in c:
        return "event_time"
    if any(t in c for t in ("protocolo", "processo", "numero")) and "ano" not in c:
        return "protocol_id"
    if c in {"ocorrencia", "tipo_ocorrencia", "natureza", "categoria"}:
        return "occurrence_type"
    if "risco" in c or any(t in c for t in ("alagamento", "inundacao", "deslizamento", "chuva")):
        return "hazard_type"
    if "servico" in c or "solicitacao" in c or "demanda" in c:
        return "service_type"
    if "bairro" in c:
        return "neighborhood"
    if "localidade" in c or "regional" in c or c in {"rpa", "rpa_nome"}:
        return "locality"
    if any(t in c for t in ADDRESS_TERMS):
        return "address"
    if c in LAT_TERMS:
        return "latitude"
    if c in LON_TERMS:
        return "longitude"
    if "geometry" in c or "geometria" in c:
        return "geometry"
    if "situacao" in c or "status" in c:
        return "status"
    if "origem" in c or "fonte" in c:
        return "source"
    return "unmapped"


def role_fields(columns):
    roles = {}
    for col in columns:
        role = field_role(col)
        roles.setdefault(role, []).append(col)
    return roles


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


def window_type(d):
    if not d:
        return ""
    if CORE_START <= d <= CORE_END:
        return "event_core_window"
    if PRE_3_START <= d < CORE_START:
        return "pre_event_3d"
    if PRE_7_START <= d < PRE_3_START:
        return "pre_event_7d"
    if CORE_END < d <= POST_3_END:
        return "post_event_3d"
    return "outside_event_window"


def row_text(row):
    return " ".join(str(v or "") for v in row.values())


def has_any(text, terms):
    t = norm(text)
    return any(term in t for term in terms)


def row_hash(asset_id, index, row):
    material = asset_id + "|" + str(index) + "|" + "|".join(
        f"{k}={row.get(k, '')}" for k in sorted(row))
    return hash_text(material, 24)


def pick_first(row, fields):
    for f in fields:
        if str(row.get(f, "")).strip():
            return row.get(f, "")
    return ""


def coordinate_status_for_row(row, roles):
    lat = pick_first(row, roles.get("latitude", []))
    lon = pick_first(row, roles.get("longitude", []))
    if not lat or not lon:
        return "NO_COORDINATES"
    try:
        lat_f = float(str(lat).replace(",", "."))
        lon_f = float(str(lon).replace(",", "."))
    except ValueError:
        return "INVALID_COORDINATES"
    if -8.4 <= lat_f <= -7.6 and -35.3 <= lon_f <= -34.6:
        return "OCCURRENCE_COORDINATES_CANDIDATE"
    return "INVALID_COORDINATES"


def locality_status_for_row(row, roles):
    if pick_first(row, roles.get("address", [])):
        return "ADDRESS_TEXT_AVAILABLE"
    if pick_first(row, roles.get("neighborhood", [])):
        return "NEIGHBORHOOD_LEVEL_LOCALITY"
    if pick_first(row, roles.get("locality", [])):
        return "LOCALITY_AMBIGUOUS"
    return "NO_LOCALITY"


def redact_hash(value):
    return hash_text(value, 16) if str(value or "").strip() else ""


ASSET_SCHEMA_COLUMNS = [
    "asset_schema_id", "event_id", "artifact_id", "asset_id", "source_id",
    "title", "local_asset_hash", "asset_type", "row_count", "column_count",
    "encoding_status", "delimiter", "columns_hash", "has_date_field",
    "date_field_candidates", "has_hazard_field", "hazard_field_candidates",
    "has_locality_field", "locality_field_candidates", "has_address_field",
    "address_field_candidates", "has_coordinate_fields",
    "coordinate_field_candidates", "has_sensitive_fields",
    "sensitive_field_candidates", "schema_status", "notes",
]

FIELD_SEMANTICS_COLUMNS = [
    "field_semantics_id", "event_id", "asset_id", "source_field",
    "canonical_field", "apparent_type", "is_sensitive", "is_ambiguous",
    "mapping_status", "notes",
]

OCCURRENCE_PROFILE_COLUMNS = [
    "table_profile_id", "event_id", "asset_id", "table_name", "total_rows",
    "parseable_date_rows", "min_date", "max_date", "rows_in_event_window",
    "rows_pre_3d", "rows_pre_7d", "rows_post_3d", "rows_with_flood_terms",
    "rows_with_rain_terms", "rows_with_landslide_terms", "rows_with_coordinates",
    "rows_with_neighborhood", "rows_with_address", "likely_occurrence_table",
    "limitations", "notes",
]

EVENT_MATCH_COLUMNS = [
    "match_id", "event_id", "asset_id", "row_hash", "window_type",
    "parsed_date", "has_flood_term", "has_rain_term", "has_landslide_term",
    "has_hazard_term", "has_neighborhood", "neighborhood_hash", "has_address",
    "address_hash", "has_coordinates", "coordinate_status", "candidate_status",
    "limitations",
]

COORDINATE_AUDIT_COLUMNS = [
    "coordinate_audit_id", "event_id", "asset_id", "asset_type",
    "coordinate_fields", "rows_checked", "rows_with_coordinates",
    "rows_in_recife_range", "crs_status", "geometry_status",
    "coordinate_classification", "can_create_ground_reference",
    "can_create_training_label", "notes",
]

LOCALITY_AUDIT_COLUMNS = [
    "locality_audit_id", "event_id", "asset_id", "locality_fields",
    "rows_checked", "rows_with_neighborhood", "rows_with_address",
    "rows_with_locality", "locality_classification",
    "sufficient_for_human_review", "sufficient_for_overlay", "notes",
]

CANDIDATE_ROW_COLUMNS = [
    "candidate_row_id", "event_id", "asset_id", "row_hash", "candidate_class",
    "event_window_match", "hazard_term_status", "date_status",
    "coordinate_status", "locality_status", "evidence_strength",
    "review_priority", "can_be_observed_occurrence_candidate",
    "can_be_ground_reference_candidate", "can_create_ground_reference",
    "can_create_training_label", "blocker", "required_next_action", "notes",
]

PREPACKAGE_COLUMNS = [
    "prepackage_id", "event_id", "package_status", "candidate_rows_count",
    "coordinate_candidates_count", "locality_only_candidates_count",
    "documented_occurrence_no_geometry_count", "review_task",
    "reviewer_decision_options", "can_advance_to_v1ul",
    "cannot_advance_reason", "notes",
]

BLOCKER_COLUMNS = [
    "blocker_id", "event_id", "blocker", "status", "evidence_count",
    "can_create_ground_reference", "can_create_training_label", "notes",
]

NEXT_ACTION_COLUMNS = [
    "action_id", "event_id", "action_type", "priority", "description",
    "target", "status", "notes",
]

MANIFEST_COLUMNS = [
    "artifact_id", "artifact_path", "artifact_type", "protocol_version",
    "sha256_prefix", "file_size_bytes", "is_versionable", "reason",
]


def run_schema_audit(out_path=None, inventory_path=None, manifest_path=None, raw_dir=RAW_DIR):
    rows = []
    for seq, asset in enumerate(discover_assets(inventory_path, manifest_path, raw_dir)):
        ext = asset["extension"]
        columns = []
        row_count = 0
        enc_status = "not_applicable"
        delimiter = ""
        notes = ""
        if ext == ".csv":
            try:
                data, columns, enc, delim = read_csv_rows(asset["path"])
                row_count, enc_status, delimiter = len(data), enc, delim
            except Exception as e:
                notes = str(e)[:120]
        elif ext == ".geojson":
            try:
                _doc, feats, crs = read_geojson(asset["path"])
                row_count = len(feats)
                columns = sorted({k for f in feats[:50] for k in (f.get("properties") or {}).keys()})
                enc_status = "utf-8"
                notes = f"crs={crs or 'missing'}"
            except Exception as e:
                notes = str(e)[:120]
        roles = role_fields(columns)
        hazard_cols = [c for c in columns if has_any(c, HAZARD_TERMS)]
        locality_cols = roles.get("neighborhood", []) + roles.get("locality", [])
        address_cols = roles.get("address", [])
        coord_cols = roles.get("latitude", []) + roles.get("longitude", []) + roles.get("geometry", [])
        sensitive_cols = [c for c in columns if has_any(c, SENSITIVE_TERMS)]
        rows.append({
            "asset_schema_id": f"SCHEMA_{PROTOCOL_VERSION}_{seq:04d}",
            "event_id": asset["event_id"],
            "artifact_id": asset["artifact_id"],
            "asset_id": asset["asset_id"],
            "source_id": asset["source_id"],
            "title": asset["title"],
            "local_asset_hash": asset["sha256"],
            "asset_type": asset["asset_type"],
            "row_count": str(row_count),
            "column_count": str(len(columns)),
            "encoding_status": enc_status,
            "delimiter": delimiter,
            "columns_hash": hash_text("|".join(columns), 24),
            "has_date_field": str(bool(roles.get("event_date") or any(has_any(c, DATE_TERMS) for c in columns))).lower(),
            "date_field_candidates": "|".join(roles.get("event_date", [])),
            "has_hazard_field": str(bool(hazard_cols)).lower(),
            "hazard_field_candidates": "|".join(hazard_cols),
            "has_locality_field": str(bool(locality_cols)).lower(),
            "locality_field_candidates": "|".join(locality_cols),
            "has_address_field": str(bool(address_cols)).lower(),
            "address_field_candidates": "|".join(address_cols),
            "has_coordinate_fields": str(bool(coord_cols)).lower(),
            "coordinate_field_candidates": "|".join(coord_cols),
            "has_sensitive_fields": str(bool(sensitive_cols)).lower(),
            "sensitive_field_candidates": "|".join(sensitive_cols),
            "schema_status": "SCHEMA_PROFILED" if not notes or notes.startswith("crs=") else "SCHEMA_ERROR",
            "notes": notes,
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1uk_recife_asset_schema_registry.csv")
    write_csv(out_path, ASSET_SCHEMA_COLUMNS, rows)
    print(f"[v1uk schema audit] assets={len(rows)} -> {out_path}")
    return rows


def run_field_semantics(out_path=None, schema_path=None, raw_dir=RAW_DIR):
    schema = load_csv(schema_path or os.path.join(DATASET_DIR, "v1uk_recife_asset_schema_registry.csv"))
    assets_by_id = {a["asset_id"]: a for a in discover_assets(raw_dir=raw_dir)}
    rows = []
    seq = 0
    for schema_row in schema:
        asset = assets_by_id.get(schema_row["asset_id"])
        if not asset:
            continue
        columns = []
        samples = {}
        if asset["extension"] == ".csv":
            data, columns, _enc, _delim = read_csv_rows(asset["path"])
            for col in columns:
                samples[col] = [r.get(col, "") for r in data[:100]]
        elif asset["extension"] == ".geojson":
            _doc, feats, _crs = read_geojson(asset["path"])
            columns = sorted({k for f in feats[:50] for k in (f.get("properties") or {}).keys()})
            for col in columns:
                samples[col] = [(f.get("properties") or {}).get(col, "") for f in feats[:100]]
            columns.append("geometry")
        role_counts = {}
        for col in columns:
            role = field_role(col)
            role_counts[role] = role_counts.get(role, 0) + 1
        for col in columns:
            role = field_role(col)
            sensitive = has_any(col, SENSITIVE_TERMS)
            ambiguous = role == "unmapped" or role_counts.get(role, 0) > 1 and role in {"event_date", "service_type", "locality"}
            rows.append({
                "field_semantics_id": f"FIELD_{PROTOCOL_VERSION}_{seq:05d}",
                "event_id": EVENT_ID,
                "asset_id": asset["asset_id"],
                "source_field": col,
                "canonical_field": role,
                "apparent_type": apparent_type(samples.get(col, [])),
                "is_sensitive": str(sensitive).lower(),
                "is_ambiguous": str(ambiguous).lower(),
                "mapping_status": "MAPPED" if role != "unmapped" else "UNMAPPED_NEEDS_REVIEW",
                "notes": "field_name_only_no_raw_values",
            })
            seq += 1
    out_path = out_path or os.path.join(DATASET_DIR, "v1uk_recife_field_semantics_registry.csv")
    write_csv(out_path, FIELD_SEMANTICS_COLUMNS, rows)
    print(f"[v1uk field semantics] fields={len(rows)} -> {out_path}")
    return rows


def parse_table_profile_for_asset(asset):
    if asset["extension"] != ".csv":
        return None
    data, columns, _enc, _delim = read_csv_rows(asset["path"])
    roles = role_fields(columns)
    date_fields = roles.get("event_date", [])
    dates = []
    counts = {
        "event": 0, "pre3": 0, "pre7": 0, "post3": 0, "flood": 0,
        "rain": 0, "landslide": 0, "coords": 0, "neighborhood": 0,
        "address": 0,
    }
    for row in data:
        d = parse_date(pick_first(row, date_fields))
        if d:
            dates.append(d)
            w = window_type(d)
            counts["event"] += int(w == "event_core_window")
            counts["pre3"] += int(w == "pre_event_3d")
            counts["pre7"] += int(w == "pre_event_7d")
            counts["post3"] += int(w == "post_event_3d")
        text = row_text(row)
        counts["flood"] += int(has_any(text, FLOOD_TERMS))
        counts["rain"] += int(has_any(text, RAIN_TERMS))
        counts["landslide"] += int(has_any(text, LANDSLIDE_TERMS))
        counts["coords"] += int(coordinate_status_for_row(row, roles) == "OCCURRENCE_COORDINATES_CANDIDATE")
        counts["neighborhood"] += int(bool(pick_first(row, roles.get("neighborhood", []))))
        counts["address"] += int(bool(pick_first(row, roles.get("address", []))))
    likely = bool(date_fields and (roles.get("neighborhood") or roles.get("address"))
                  and (has_any(asset["title"], {"atendimento", "sedec", "vistoria", "solicitacao"})
                       or counts["flood"] or counts["rain"] or counts["landslide"]))
    return {
        "event_id": EVENT_ID,
        "asset_id": asset["asset_id"],
        "table_name": asset["title"],
        "total_rows": str(len(data)),
        "parseable_date_rows": str(len(dates)),
        "min_date": min(dates).isoformat() if dates else "",
        "max_date": max(dates).isoformat() if dates else "",
        "rows_in_event_window": str(counts["event"]),
        "rows_pre_3d": str(counts["pre3"]),
        "rows_pre_7d": str(counts["pre7"]),
        "rows_post_3d": str(counts["post3"]),
        "rows_with_flood_terms": str(counts["flood"]),
        "rows_with_rain_terms": str(counts["rain"]),
        "rows_with_landslide_terms": str(counts["landslide"]),
        "rows_with_coordinates": str(counts["coords"]),
        "rows_with_neighborhood": str(counts["neighborhood"]),
        "rows_with_address": str(counts["address"]),
        "likely_occurrence_table": str(likely).lower(),
        "limitations": "values_redacted_no_geocoding_no_overlay",
        "notes": "",
    }


def run_occurrence_parser(out_path=None, raw_dir=RAW_DIR):
    rows = []
    for seq, asset in enumerate(discover_assets(raw_dir=raw_dir)):
        prof = parse_table_profile_for_asset(asset)
        if not prof:
            continue
        prof["table_profile_id"] = f"TABLE_{PROTOCOL_VERSION}_{seq:04d}"
        rows.append(prof)
    out_path = out_path or os.path.join(DATASET_DIR, "v1uk_recife_occurrence_table_profile.csv")
    write_csv(out_path, OCCURRENCE_PROFILE_COLUMNS, rows)
    print(f"[v1uk occurrence parser] tables={len(rows)} -> {out_path}")
    return rows


def iter_event_matches(raw_dir=RAW_DIR):
    for asset in discover_assets(raw_dir=raw_dir):
        if asset["extension"] != ".csv":
            continue
        data, columns, _enc, _delim = read_csv_rows(asset["path"])
        roles = role_fields(columns)
        date_fields = roles.get("event_date", [])
        for idx, row in enumerate(data):
            d = parse_date(pick_first(row, date_fields))
            w = window_type(d)
            if w == "outside_event_window" or not w:
                continue
            text = row_text(row)
            coord_status = coordinate_status_for_row(row, roles)
            loc_status = locality_status_for_row(row, roles)
            h_flood = has_any(text, FLOOD_TERMS)
            h_rain = has_any(text, RAIN_TERMS)
            h_land = has_any(text, LANDSLIDE_TERMS)
            h_any = h_flood or h_rain or h_land or has_any(text, HAZARD_TERMS)
            has_nei = bool(pick_first(row, roles.get("neighborhood", [])))
            has_addr = bool(pick_first(row, roles.get("address", [])))
            rh = row_hash(asset["asset_id"], idx, row)
            yield asset, row, roles, {
                "event_id": EVENT_ID,
                "asset_id": asset["asset_id"],
                "row_hash": rh,
                "window_type": w,
                "parsed_date": d.isoformat() if d else "",
                "has_flood_term": str(h_flood).lower(),
                "has_rain_term": str(h_rain).lower(),
                "has_landslide_term": str(h_land).lower(),
                "has_hazard_term": str(h_any).lower(),
                "has_neighborhood": str(has_nei).lower(),
                "neighborhood_hash": redact_hash(pick_first(row, roles.get("neighborhood", []))),
                "has_address": str(has_addr).lower(),
                "address_hash": redact_hash(pick_first(row, roles.get("address", []))),
                "has_coordinates": str(coord_status == "OCCURRENCE_COORDINATES_CANDIDATE").lower(),
                "coordinate_status": coord_status,
                "candidate_status": "EVENT_WINDOW_OCCURRENCE_CANDIDATE_FOR_REVIEW" if h_any else "REJECTED_NO_HAZARD_SIGNAL",
                "limitations": "row_values_redacted_no_geocoding_no_overlay",
            }, loc_status


def run_event_window_filter(out_path=None, raw_dir=RAW_DIR):
    rows = []
    for seq, (_asset, _row, _roles, reg, _loc_status) in enumerate(iter_event_matches(raw_dir)):
        reg["match_id"] = f"MATCH_{PROTOCOL_VERSION}_{seq:06d}"
        rows.append(reg)
    out_path = out_path or os.path.join(DATASET_DIR, "v1uk_recife_event_window_match_registry.csv")
    write_csv(out_path, EVENT_MATCH_COLUMNS, rows)
    print(f"[v1uk event window] matches={len(rows)} -> {out_path}")
    return rows


def run_coordinate_audit(out_path=None, raw_dir=RAW_DIR):
    rows = []
    for seq, asset in enumerate(discover_assets(raw_dir=raw_dir)):
        coord_fields = []
        rows_checked = rows_with = rows_range = 0
        crs_status = "not_applicable"
        geom_status = "not_applicable"
        classification = "NO_COORDINATES"
        notes = ""
        if asset["extension"] == ".csv":
            data, columns, _enc, _delim = read_csv_rows(asset["path"])
            roles = role_fields(columns)
            coord_fields = roles.get("latitude", []) + roles.get("longitude", [])
            rows_checked = len(data)
            for row in data:
                status = coordinate_status_for_row(row, roles)
                rows_with += int(status != "NO_COORDINATES")
                rows_range += int(status == "OCCURRENCE_COORDINATES_CANDIDATE")
            if rows_with and rows_range:
                if has_any(asset["title"], {"solicitacoes", "solicitacao", "atendimentos", "atendimento", "ocorrencia"}):
                    classification = "OCCURRENCE_COORDINATES_CANDIDATE"
                elif has_any(asset["title"], {"risco", "regional", "coordenadas"}):
                    classification = "REGIONAL_CONTEXT_POINTS"
                else:
                    classification = "INFRASTRUCTURE_CONTEXT"
            elif rows_with:
                classification = "INVALID_COORDINATES"
        elif asset["extension"] == ".geojson":
            try:
                _doc, feats, crs = read_geojson(asset["path"])
                rows_checked = len(feats)
                crs_status = "CRS_PRESENT" if crs else "CRS_MISSING"
                geom_status = "GEOMETRY_PRESENT" if feats else "GEOMETRY_MISSING"
                rows_with = len(feats)
                rows_range = len(feats)
                if has_any(asset["title"], {"coordenadas", "regional", "defesa", "equipamento"}):
                    classification = "REGIONAL_CONTEXT_POINTS"
                else:
                    classification = "INFRASTRUCTURE_CONTEXT"
            except Exception as e:
                classification = "INVALID_COORDINATES"
                notes = str(e)[:120]
        rows.append({
            "coordinate_audit_id": f"COORD_{PROTOCOL_VERSION}_{seq:04d}",
            "event_id": EVENT_ID,
            "asset_id": asset["asset_id"],
            "asset_type": asset["asset_type"],
            "coordinate_fields": "|".join(coord_fields),
            "rows_checked": str(rows_checked),
            "rows_with_coordinates": str(rows_with),
            "rows_in_recife_range": str(rows_range),
            "crs_status": crs_status,
            "geometry_status": geom_status,
            "coordinate_classification": classification,
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": notes,
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1uk_recife_coordinate_evidence_audit.csv")
    write_csv(out_path, COORDINATE_AUDIT_COLUMNS, rows)
    print(f"[v1uk coordinate audit] assets={len(rows)} -> {out_path}")
    return rows


def run_locality_audit(out_path=None, raw_dir=RAW_DIR):
    rows = []
    for seq, asset in enumerate(discover_assets(raw_dir=raw_dir)):
        fields = []
        checked = n_nei = n_addr = n_loc = 0
        classification = "NO_LOCALITY"
        if asset["extension"] == ".csv":
            data, columns, _enc, _delim = read_csv_rows(asset["path"])
            roles = role_fields(columns)
            fields = roles.get("neighborhood", []) + roles.get("locality", []) + roles.get("address", [])
            checked = len(data)
            for row in data:
                n_nei += int(bool(pick_first(row, roles.get("neighborhood", []))))
                n_addr += int(bool(pick_first(row, roles.get("address", []))))
                n_loc += int(bool(pick_first(row, roles.get("locality", []))))
            if n_addr:
                classification = "ADDRESS_TEXT_AVAILABLE"
            elif n_nei:
                classification = "NEIGHBORHOOD_LEVEL_LOCALITY"
            elif n_loc:
                classification = "LOCALITY_AMBIGUOUS"
        rows.append({
            "locality_audit_id": f"LOC_{PROTOCOL_VERSION}_{seq:04d}",
            "event_id": EVENT_ID,
            "asset_id": asset["asset_id"],
            "locality_fields": "|".join(fields),
            "rows_checked": str(checked),
            "rows_with_neighborhood": str(n_nei),
            "rows_with_address": str(n_addr),
            "rows_with_locality": str(n_loc),
            "locality_classification": classification,
            "sufficient_for_human_review": str(classification != "NO_LOCALITY").lower(),
            "sufficient_for_overlay": "false",
            "notes": "no_geocoding_no_centroid_no_overlay",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1uk_recife_locality_evidence_audit.csv")
    write_csv(out_path, LOCALITY_AUDIT_COLUMNS, rows)
    print(f"[v1uk locality audit] assets={len(rows)} -> {out_path}")
    return rows


def run_candidate_builder(out_path=None, matches_path=None, coord_path=None, loc_path=None):
    matches = load_csv(matches_path or os.path.join(DATASET_DIR, "v1uk_recife_event_window_match_registry.csv"))
    coords = {r["asset_id"]: r for r in load_csv(coord_path or os.path.join(DATASET_DIR, "v1uk_recife_coordinate_evidence_audit.csv"))}
    locs = {r["asset_id"]: r for r in load_csv(loc_path or os.path.join(DATASET_DIR, "v1uk_recife_locality_evidence_audit.csv"))}
    rows = []
    for seq, m in enumerate(matches):
        coord_status = m.get("coordinate_status", "NO_COORDINATES")
        loc_status = locs.get(m["asset_id"], {}).get("locality_classification", "NO_LOCALITY")
        has_hazard = m.get("has_hazard_term") == "true"
        in_core = m.get("window_type") == "event_core_window"
        has_coords = coord_status == "OCCURRENCE_COORDINATES_CANDIDATE"
        has_loc = loc_status != "NO_LOCALITY"
        if not in_core:
            cls = "REJECTED_OUTSIDE_EVENT_WINDOW"
            blocker = "outside_core_event_window"
        elif not has_hazard:
            cls = "REJECTED_NO_HAZARD_SIGNAL"
            blocker = "no_hazard_signal"
        elif has_coords:
            cls = "ROW_LEVEL_OCCURRENCE_WITH_COORDINATES_FOR_REVIEW"
            blocker = "no_supervisor_review_no_overlay_label_forbidden"
        elif has_loc:
            cls = "ROW_LEVEL_OCCURRENCE_WITH_LOCALITY_ONLY_FOR_REVIEW"
            blocker = "locality_only_no_coordinates_no_overlay"
        else:
            cls = "EVENT_WINDOW_DOCUMENTED_OCCURRENCE_NO_GEOMETRY"
            blocker = "no_coordinates_no_locality"
        strength = "high" if cls.endswith("COORDINATES_FOR_REVIEW") else "medium" if "LOCALITY" in cls else "low"
        rows.append({
            "candidate_row_id": f"CAND_{PROTOCOL_VERSION}_{seq:06d}",
            "event_id": EVENT_ID,
            "asset_id": m["asset_id"],
            "row_hash": m["row_hash"],
            "candidate_class": cls,
            "event_window_match": m["window_type"],
            "hazard_term_status": "HAS_HAZARD_SIGNAL" if has_hazard else "NO_HAZARD_SIGNAL",
            "date_status": "IN_CORE_EVENT_WINDOW" if in_core else "OUTSIDE_CORE_EVENT_WINDOW",
            "coordinate_status": coord_status,
            "locality_status": loc_status,
            "evidence_strength": strength,
            "review_priority": "1" if cls.endswith("COORDINATES_FOR_REVIEW") else "2" if "LOCALITY" in cls else "3",
            "can_be_observed_occurrence_candidate": str(cls.startswith("ROW_LEVEL_OCCURRENCE")).lower(),
            "can_be_ground_reference_candidate": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "blocker": blocker,
            "required_next_action": "HUMAN_REVIEW" if cls.startswith("ROW_LEVEL_OCCURRENCE") else "DO_NOT_PROMOTE",
            "notes": "row_values_redacted_no_ground_reference_no_label",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1uk_recife_candidate_row_registry.csv")
    write_csv(out_path, CANDIDATE_ROW_COLUMNS, rows)
    print(f"[v1uk candidate rows] rows={len(rows)} -> {out_path}")
    return rows


def run_supervisor_prepackage(out_path=None, candidates_path=None):
    candidates = load_csv(candidates_path or os.path.join(DATASET_DIR, "v1uk_recife_candidate_row_registry.csv"))
    coord = sum(1 for r in candidates if r.get("candidate_class") == "ROW_LEVEL_OCCURRENCE_WITH_COORDINATES_FOR_REVIEW")
    loc = sum(1 for r in candidates if r.get("candidate_class") == "ROW_LEVEL_OCCURRENCE_WITH_LOCALITY_ONLY_FOR_REVIEW")
    doc = sum(1 for r in candidates if r.get("candidate_class") == "EVENT_WINDOW_DOCUMENTED_OCCURRENCE_NO_GEOMETRY")
    count = coord + loc + doc
    if coord:
        status = "READY_FOR_HUMAN_REVIEW_WITH_COORDINATE_CANDIDATES"
        can_v1ul = "true"
        reason = ""
    elif loc:
        status = "READY_FOR_HUMAN_REVIEW_LOCALITY_ONLY"
        can_v1ul = "true"
        reason = ""
    else:
        status = "NOT_READY_NO_REVIEWABLE_OCCURRENCE_ROWS"
        can_v1ul = "false"
        reason = "no_core_event_window_hazard_candidate"
    rows = [{
        "prepackage_id": f"PREPKG_{PROTOCOL_VERSION}_0000",
        "event_id": EVENT_ID,
        "package_status": status,
        "candidate_rows_count": str(count),
        "coordinate_candidates_count": str(coord),
        "locality_only_candidates_count": str(loc),
        "documented_occurrence_no_geometry_count": str(doc),
        "review_task": "Review redacted Recife CKAN candidate rows; no overlay or labels in v1uk",
        "reviewer_decision_options": "|".join([
            "ACCEPT_OCCURRENCE_CANDIDATE_FOR_GEOMETRY_REVIEW",
            "ACCEPT_LOCALITY_ONLY_CONTEXT",
            "REJECT_CONTEXT_ONLY",
            "REJECT_OUTSIDE_EVENT_WINDOW",
            "REQUEST_SCHEMA_CLARIFICATION",
            "DO_NOT_PROMOTE",
        ]),
        "can_advance_to_v1ul": can_v1ul,
        "cannot_advance_reason": reason,
        "notes": "ground_reference=false training_label=false no_overlay=true",
    }]
    out_path = out_path or os.path.join(DATASET_DIR, "v1uk_recife_supervisor_review_prepackage_registry.csv")
    write_csv(out_path, PREPACKAGE_COLUMNS, rows)
    print(f"[v1uk supervisor prepackage] status={status} -> {out_path}")
    return rows


def write_blocker_matrix(path=None):
    candidates = load_csv(os.path.join(DATASET_DIR, "v1uk_recife_candidate_row_registry.csv"))
    pre = load_csv(os.path.join(DATASET_DIR, "v1uk_recife_supervisor_review_prepackage_registry.csv"))
    coord = sum(1 for r in candidates if r.get("candidate_class") == "ROW_LEVEL_OCCURRENCE_WITH_COORDINATES_FOR_REVIEW")
    loc = sum(1 for r in candidates if r.get("candidate_class") == "ROW_LEVEL_OCCURRENCE_WITH_LOCALITY_ONLY_FOR_REVIEW")
    blockers = [
        ("no_verified_observed_geometry", "ACTIVE", coord, "candidate coordinates are review candidates only"),
        ("no_overlay", "ACTIVE", 0, "overlay not executed"),
        ("no_supervisor_review", "ACTIVE", int(bool(pre)), "human review not completed"),
        ("coordinate_absent_or_contextual", "ACTIVE" if coord == 0 else "PARTIAL", loc, "some rows are locality-only or contextual"),
        ("locality_only", "ACTIVE" if loc else "INACTIVE", loc, "locality cannot be converted to coordinates"),
        ("event_window_uncertainty", "ACTIVE", len(candidates), "row dates parsed but not ground truth"),
        ("hazard_ambiguity", "ACTIVE", len(candidates), "hazard terms need human review"),
        ("sensitive_data_redaction_required", "ACTIVE", len(candidates), "row values are redacted"),
        ("label_forbidden", "ACTIVE", 0, "labels forbidden in v1uk"),
    ]
    rows = []
    for seq, (name, status, count, notes) in enumerate(blockers):
        rows.append({
            "blocker_id": f"BLOCK_{PROTOCOL_VERSION}_{seq:04d}",
            "event_id": EVENT_ID,
            "blocker": name,
            "status": status,
            "evidence_count": str(count),
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": notes,
        })
    path = path or os.path.join(DATASET_DIR, "v1uk_recife_ground_reference_blocker_matrix.csv")
    write_csv(path, BLOCKER_COLUMNS, rows)
    return rows


V1UK_ARTIFACTS = [
    "configs/protocolo_c/v1uk_recife_ckan_schema_policy.yaml",
    "configs/protocolo_c/v1uk_recife_event_window_policy.yaml",
    "configs/protocolo_c/v1uk_recife_field_mapping_policy.yaml",
    "configs/protocolo_c/v1uk_recife_sensitive_value_policy.yaml",
    "configs/protocolo_c/v1uk_recife_candidate_scoring_policy.yaml",
    "datasets/protocolo_c/v1uk_recife_asset_schema_registry.csv",
    "datasets/protocolo_c/v1uk_recife_field_semantics_registry.csv",
    "datasets/protocolo_c/v1uk_recife_occurrence_table_profile.csv",
    "datasets/protocolo_c/v1uk_recife_event_window_match_registry.csv",
    "datasets/protocolo_c/v1uk_recife_coordinate_evidence_audit.csv",
    "datasets/protocolo_c/v1uk_recife_locality_evidence_audit.csv",
    "datasets/protocolo_c/v1uk_recife_candidate_row_registry.csv",
    "datasets/protocolo_c/v1uk_recife_supervisor_review_prepackage_registry.csv",
    "datasets/protocolo_c/v1uk_recife_ground_reference_blocker_matrix.csv",
    "datasets/protocolo_c/v1uk_next_actions_registry.csv",
    "datasets/protocolo_c/v1uk_versionable_artifacts_manifest.csv",
    "docs/metodologia_cientifica/protocolo_c_v1uk_recife_ckan_schema_deep_audit.md",
    "docs/metodologia_cientifica/protocolo_c_relatorio_v1uk_recife_ckan_schema_deep_audit.md",
    "docs/metodologia_cientifica/protocolo_c_status_atual_v1uk.md",
]


def run_completion_report():
    write_blocker_matrix()
    schema = load_csv(os.path.join(DATASET_DIR, "v1uk_recife_asset_schema_registry.csv"))
    profile = load_csv(os.path.join(DATASET_DIR, "v1uk_recife_occurrence_table_profile.csv"))
    matches = load_csv(os.path.join(DATASET_DIR, "v1uk_recife_event_window_match_registry.csv"))
    coord = load_csv(os.path.join(DATASET_DIR, "v1uk_recife_coordinate_evidence_audit.csv"))
    loc = load_csv(os.path.join(DATASET_DIR, "v1uk_recife_locality_evidence_audit.csv"))
    candidates = load_csv(os.path.join(DATASET_DIR, "v1uk_recife_candidate_row_registry.csv"))
    pre = load_csv(os.path.join(DATASET_DIR, "v1uk_recife_supervisor_review_prepackage_registry.csv"))
    coord_candidates = sum(1 for r in candidates if r.get("candidate_class") == "ROW_LEVEL_OCCURRENCE_WITH_COORDINATES_FOR_REVIEW")
    locality_candidates = sum(1 for r in candidates if r.get("candidate_class") == "ROW_LEVEL_OCCURRENCE_WITH_LOCALITY_ONLY_FOR_REVIEW")
    total_rows = sum(int(r.get("total_rows") or 0) for r in profile)
    rows_event = sum(int(r.get("rows_in_event_window") or 0) for r in profile)
    rows_hazard = sum(int(r.get("rows_with_flood_terms") or 0) + int(r.get("rows_with_rain_terms") or 0) + int(r.get("rows_with_landslide_terms") or 0) for r in profile)
    rows_loc = sum(int(r.get("rows_with_neighborhood") or 0) + int(r.get("rows_with_address") or 0) for r in profile)
    rows_coord = sum(int(r.get("rows_with_coordinates") or 0) for r in profile)
    geojson_defesa = next((r for r in coord if r.get("asset_id") == "ec18759d-fac2-445e-ae72-af9d9210b831"), {})
    coord_classes = {}
    for r in coord:
        coord_classes[r.get("coordinate_classification", "")] = coord_classes.get(r.get("coordinate_classification", ""), 0) + 1
    if coord_candidates:
        next_action = "v1ul - Recife Occurrence Candidate Supervisor Review and Overlay Preflight"
    elif locality_candidates:
        next_action = "v1ul - Recife Locality-Only Human Review and Non-Overlay Evidence Package"
    else:
        next_action = "Deepen another public dataset"
    action_rows = [{
        "action_id": f"ACT_{PROTOCOL_VERSION}_0000",
        "event_id": EVENT_ID,
        "action_type": "HUMAN_REVIEW_PREPACKAGE" if coord_candidates or locality_candidates else "DEEPEN_PUBLIC_DATASET",
        "priority": "1",
        "description": next_action,
        "target": "REC_2022_05_24_30 CKAN Recife",
        "status": "PENDING",
        "notes": "no_ground_reference_no_label_no_overlay",
    }]
    write_csv(os.path.join(DATASET_DIR, "v1uk_next_actions_registry.csv"), NEXT_ACTION_COLUMNS, action_rows)
    manifest = []
    for seq, path in enumerate(V1UK_ARTIFACTS):
        exists = os.path.exists(path)
        manifest.append({
            "artifact_id": f"ART_{PROTOCOL_VERSION}_{seq:04d}",
            "artifact_path": path,
            "artifact_type": "config" if path.startswith("configs/") else "doc" if path.startswith("docs/") else "dataset",
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha256_file(path)[:16] if exists else "MISSING",
            "file_size_bytes": str(os.path.getsize(path) if exists else 0),
            "is_versionable": str(exists).lower(),
            "reason": "Safe metadata-only artifact" if exists else "File not found",
        })
    write_csv(os.path.join(DATASET_DIR, "v1uk_versionable_artifacts_manifest.csv"), MANIFEST_COLUMNS, manifest)
    os.makedirs(DOCS_DIR, exist_ok=True)
    report = [
        "# Protocolo C v1uk - Recife CKAN Schema Deep Audit",
        "",
        "## Scope",
        f"- Event: {EVENT_ID}",
        "- Event window: 2022-05-24 to 2022-05-30",
        "- Source: CKAN Recife assets downloaded in v1uj",
        "- No web search, no geocoding, no centroid, no overlay, no labels.",
        "",
        "## Findings",
        f"- Audited assets: {len(schema)}",
        f"- Occurrence-like tables profiled: {len(profile)}",
        f"- Total table rows profiled: {total_rows}",
        f"- Rows in event window: {rows_event}",
        f"- Rows with flood/rain/landslide terms: {rows_hazard}",
        f"- Rows with neighborhood or address evidence: {rows_loc}",
        f"- Rows with coordinates: {rows_coord}",
        f"- Event-window row matches: {len(matches)}",
        f"- Coordinate review candidates: {coord_candidates}",
        f"- Locality-only review candidates: {locality_candidates}",
        "",
        "## Registro de Atendimentos da Defesa Civil",
        "- Attendance CSVs were parsed as documented occurrence tables.",
        "- They expose date, occurrence/request, address, neighborhood, locality, and risk/action fields.",
        "- Public registries contain only hashes/flags/counts, not raw sensitive values.",
        "",
        "## Coordinate Evidence",
        f"- Coordinate audit classes: {coord_classes}",
        f"- Defesa Civil GeoJSON Coordenadas geograficas da Regiao Sul: {geojson_defesa.get('coordinate_classification', 'not_found')}.",
        "- Contextual layers are not promoted to occurrence geometry.",
        "",
        "## REC_2022 Status",
        f"- Can advance to human review: {str(bool(coord_candidates or locality_candidates)).lower()}",
        f"- Can advance to overlay preflight now: {str(bool(coord_candidates)).lower()}",
        "- can_create_ground_reference=false",
        "- can_create_training_label=false",
        "- no_overlay_executed=true",
        "",
        "## Next Action",
        f"- {next_action}",
    ]
    report_path = os.path.join(DOCS_DIR, "protocolo_c_relatorio_v1uk_recife_ckan_schema_deep_audit.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    methodology_path = os.path.join(DOCS_DIR, "protocolo_c_v1uk_recife_ckan_schema_deep_audit.md")
    with open(methodology_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report + ["", "## Method", "Schema, semantics, event-window, coordinate, locality, and row-candidate registries were built from local v1uj CKAN artifacts only."]))
    status = [
        "# Status Atual - Protocolo C v1uk",
        "",
        f"event_id={EVENT_ID}",
        f"audited_assets={len(schema)}",
        f"event_window_matches={len(matches)}",
        f"coordinate_review_candidates={coord_candidates}",
        f"locality_only_review_candidates={locality_candidates}",
        f"next_action={next_action}",
        "ground_truth_operational=false",
        "can_create_ground_reference=false",
        "can_create_training_label=false",
        "no_overlay_executed=true",
        "supervisor_review_completed=false",
    ]
    status_path = os.path.join(DOCS_DIR, "protocolo_c_status_atual_v1uk.md")
    with open(status_path, "w", encoding="utf-8") as f:
        f.write("\n".join(status))
    print(f"[v1uk completion] report={report_path}")
    return {
        "audited_assets": len(schema), "profile_tables": len(profile),
        "total_rows": total_rows, "rows_event": rows_event,
        "rows_hazard": rows_hazard, "rows_loc": rows_loc,
        "rows_coord": rows_coord, "coord_candidates": coord_candidates,
        "locality_candidates": locality_candidates, "next_action": next_action,
    }


def simple_main(fn):
    parser = argparse.ArgumentParser()
    parser.parse_args()
    fn()
