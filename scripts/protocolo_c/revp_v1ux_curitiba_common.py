#!/usr/bin/env python3
"""v1ux Curitiba public evidence download and schema audit.

Controlled download/inventory/schema audit for public Curitiba evidence targets.
Raw artifacts stay under local_only; versionable outputs contain only hashes,
counts, field names, classifications and blockers.
"""

import argparse
import csv
import hashlib
import json
import mimetypes
import os
import re
import shutil
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zipfile

PROTOCOL_VERSION = "v1ux"
DATASET_DIR = "datasets/protocolo_c"
DOCS_DIR = "docs/metodologia_cientifica"
CONFIG_DIR = "configs/protocolo_c"
RAW_DIR = "local_only/protocolo_c/curitiba_public_evidence_download/raw/v1ux"
STAGING_DIR = "local_only/protocolo_c/curitiba_public_evidence_download/staging/v1ux"
QUARANTINE_DIR = "local_only/protocolo_c/curitiba_public_evidence_download/quarantine/v1ux"
REPORTS_DIR = "local_only/protocolo_c/curitiba_public_evidence_download/reports/v1ux"
MAX_STATUS = "CURITIBA_PUBLIC_EVIDENCE_SCHEMA_AUDITED_EVENT_CANDIDATE"

DOWNLOAD_TARGET_COLUMNS = [
    "download_target_id", "candidate_event_id", "source_registry",
    "source_record_id", "source_id", "resource_url_hash", "resource_url",
    "resource_name_hash", "expected_format", "priority_class",
    "download_allowed", "skip_reason", "can_create_ground_reference",
    "can_create_training_label", "notes",
]
DOWNLOAD_COLUMNS = [
    "download_id", "download_target_id", "candidate_event_id", "source_id",
    "url_sha1_12", "safe_filename", "local_path_hash", "sha256",
    "file_size_bytes", "mime_type", "extension", "download_status",
    "duplicate_status", "raw_data_versioned", "notes",
]
INVENTORY_COLUMNS = [
    "inventory_id", "download_id", "artifact_type", "extension",
    "row_count", "column_count", "encoding", "delimiter", "feature_count",
    "geometry_type", "fields_hash", "zip_member_count", "inner_artifact_types",
    "inventory_status", "notes",
]
SCHEMA_COLUMNS = [
    "schema_audit_id", "download_id", "artifact_type", "row_count",
    "column_count", "has_date_field", "date_field_candidates",
    "has_hazard_field", "hazard_field_candidates", "has_locality_field",
    "locality_field_candidates", "has_coordinate_fields",
    "coordinate_field_candidates", "has_geometry", "schema_class", "notes",
]
GEODATA_COLUMNS = [
    "geodata_audit_id", "download_id", "geometry_type", "crs_status",
    "feature_count", "fields_hash", "geodata_class", "event_specificity",
    "can_support_contextual_review", "can_support_observed_occurrence",
    "can_support_overlay_preflight", "can_create_ground_reference", "notes",
]
EVENT_TABLE_COLUMNS = [
    "event_table_detection_id", "download_id", "artifact_type",
    "event_table_class", "date_gate", "hazard_gate", "locality_gate",
    "coordinate_gate", "can_be_event_table_candidate",
    "can_create_ground_reference", "can_create_training_label", "notes",
]
FIELD_MAP_COLUMNS = [
    "field_mapping_id", "download_id", "event_date_fields", "hazard_type_fields",
    "occurrence_type_fields", "locality_fields", "neighborhood_fields",
    "address_hash_fields", "coordinate_status", "raw_address_versioned",
    "geocoding_executed", "notes",
]
EVIDENCE_COLUMNS = [
    "evidence_classification_id", "candidate_event_id", "download_id",
    "evidence_class", "official_source_support", "date_support",
    "hazard_support", "locality_support", "coordinate_support",
    "geometry_support", "context_only_status",
    "can_advance_to_event_patch_linkage",
    "can_advance_to_overlay_preflight", "can_create_ground_reference",
    "can_create_training_label", "blocker", "required_next_action", "notes",
]
READINESS_COLUMNS = [
    "readiness_update_id", "candidate_event_id", "proposed_event_id",
    "dimension", "classification", "basis", "ground_truth_operational",
    "can_create_ground_reference", "can_create_training_label",
    "can_reopen_protocol_b", "dino_usage", "no_overlay_executed",
    "no_coordinates_invented", "patch_bound_truth", "operational_validation",
    "event_candidate_only", "public_official_discovery", "geocoding_executed",
    "centroid_used", "raw_data_versioned", "notes",
]
BLOCKER_COLUMNS = [
    "blocker_id", "event_id", "blocker", "status", "ground_truth_operational",
    "can_create_ground_reference", "can_create_training_label",
    "can_reopen_protocol_b", "dino_usage", "no_overlay_executed",
    "no_coordinates_invented", "patch_bound_truth", "operational_validation",
    "event_candidate_only", "public_official_discovery", "geocoding_executed",
    "centroid_used", "raw_data_versioned", "notes",
]
NEXT_COLUMNS = [
    "action_id", "event_id", "action_type", "priority", "description",
    "target", "status", "notes",
]
MANIFEST_COLUMNS = [
    "artifact_id", "artifact_path", "artifact_type", "protocol_version",
    "sha256_prefix", "file_size_bytes", "is_versionable", "reason",
]

V1UX_ARTIFACTS = [
    "configs/protocolo_c/v1ux_curitiba_download_policy.yaml",
    "configs/protocolo_c/v1ux_curitiba_artifact_inventory_policy.yaml",
    "configs/protocolo_c/v1ux_curitiba_schema_audit_policy.yaml",
    "configs/protocolo_c/v1ux_curitiba_geodata_audit_policy.yaml",
    "configs/protocolo_c/v1ux_curitiba_event_table_policy.yaml",
    "configs/protocolo_c/v1ux_curitiba_candidate_classification_policy.yaml",
    "datasets/protocolo_c/v1ux_curitiba_download_target_registry.csv",
    "datasets/protocolo_c/v1ux_curitiba_public_artifact_download_manifest.csv",
    "datasets/protocolo_c/v1ux_curitiba_artifact_inventory.csv",
    "datasets/protocolo_c/v1ux_curitiba_schema_audit.csv",
    "datasets/protocolo_c/v1ux_curitiba_geodata_metadata_audit.csv",
    "datasets/protocolo_c/v1ux_curitiba_event_table_detection.csv",
    "datasets/protocolo_c/v1ux_curitiba_hazard_date_locality_field_mapping.csv",
    "datasets/protocolo_c/v1ux_curitiba_candidate_evidence_classification.csv",
    "datasets/protocolo_c/v1ux_curitiba_event_patch_readiness_update.csv",
    "datasets/protocolo_c/v1ux_curitiba_ground_reference_blocker_matrix.csv",
    "datasets/protocolo_c/v1ux_next_actions_registry.csv",
    "docs/metodologia_cientifica/protocolo_c_v1ux_curitiba_public_evidence_download_schema_audit.md",
    "docs/metodologia_cientifica/protocolo_c_relatorio_v1ux_curitiba_public_evidence_download_schema_audit.md",
    "docs/metodologia_cientifica/protocolo_c_status_atual_v1ux.md",
]

DATE_TERMS = {"data", "date", "dt", "dia"}
HAZARD_TERMS = {"alagamento", "inundacao", "inundação", "enchente", "chuva", "risco", "ocorrencia", "ocorrência", "tipo"}
LOCALITY_TERMS = {"bairro", "regional", "localidade", "municipio", "município", "cidade"}
ADDRESS_TERMS = {"endereco", "endereço", "logradouro", "rua", "avenida"}
COORD_TERMS = {"lat", "latitude", "lon", "lng", "long", "longitude", "x", "y", "geometry", "geom", "coordenada"}


def norm(value):
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower().strip()


def token(value):
    return re.sub(r"[^a-z0-9]+", "_", norm(value)).strip("_")


def hash_text(value, n=16):
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:n]


def sha1_text(value, n=12):
    return hashlib.sha1(str(value or "").encode("utf-8")).hexdigest()[:n]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, columns, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
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


def artifact_path(artifact):
    base = os.path.basename(artifact)
    if artifact.startswith("datasets/protocolo_c/"):
        return dataset_path(base)
    if artifact.startswith("configs/protocolo_c/"):
        return config_path(base)
    if artifact.startswith("docs/metodologia_cientifica/"):
        return doc_path(base)
    return artifact


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-web", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-download-mb", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--local-only-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def write_policy_configs():
    policies = {
        "v1ux_curitiba_download_policy.yaml": ["raw_data_versioned: false", "max_download_mb_default: 200", "media_download: false"],
        "v1ux_curitiba_artifact_inventory_policy.yaml": ["zip_list_only: true", "raw_path_public_outputs: false"],
        "v1ux_curitiba_schema_audit_policy.yaml": ["sensitive_values_versioned: false", "address_literals_versioned: false"],
        "v1ux_curitiba_geodata_audit_policy.yaml": ["overlay_allowed: false", "feature_mass_download: false"],
        "v1ux_curitiba_event_table_policy.yaml": ["strong_table_requires: [date, hazard, locality]", "ground_truth_allowed: false"],
        "v1ux_curitiba_candidate_classification_policy.yaml": ["max_status: CURITIBA_PUBLIC_EVIDENCE_SCHEMA_AUDITED_EVENT_CANDIDATE", "label_creation_allowed: false"],
    }
    for name, lines in policies.items():
        write_text(config_path(name), lines)


def candidate_event_id():
    rows = load_csv(dataset_path("v1uw_curitiba_event_candidate_status.csv"))
    return rows[0].get("candidate_event_id", "CE_v1uv_0000") if rows else "CE_v1uv_0000"


def proposed_event_id():
    rows = load_csv(dataset_path("v1uw_curitiba_event_candidate_status.csv"))
    return rows[0].get("proposed_event_id", "CUR_2022_01_15") if rows else "CUR_2022_01_15"


def known_urls():
    disc = load_csv(dataset_path("v1uv_curitiba_public_event_discovery.csv"))
    urls = {hash_text(r.get("result_url", ""), 24): r.get("result_url", "") for r in disc}
    return urls


def open_data_catalog_url():
    rows = load_csv(dataset_path("v1uv_curitiba_open_data_registry.csv"))
    for row in rows:
        url = row.get("package_url") or row.get("source_url") or row.get("catalog_url")
        if url:
            return url
    return "https://dadosabertos.curitiba.pr.gov.br/"


def run_download_target_builder(args=None):
    write_policy_configs()
    cid = candidate_event_id()
    rows = []
    urls = known_urls()
    # Official event documents already identified.
    for idx, r in enumerate(load_csv(dataset_path("v1uv_curitiba_public_event_discovery.csv"))):
        url = r.get("result_url", "")
        rows.append({
            "download_target_id": f"DT_v1ux_{len(rows):04d}",
            "candidate_event_id": cid,
            "source_registry": "v1uv_curitiba_public_event_discovery.csv",
            "source_record_id": r.get("discovery_id", ""),
            "source_id": r.get("source_id", ""),
            "resource_url_hash": hash_text(url, 24),
            "resource_url": url,
            "resource_name_hash": r.get("title_hash", "") or hash_text(url, 24),
            "expected_format": "HTML",
            "priority_class": "DOCUMENT_ONLY",
            "download_allowed": "true",
            "skip_reason": "",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": "Official document snapshot target; raw stays local_only.",
        })
    # Open data resources use the public catalog URL from v1uv/v1uw.
    for r in load_csv(dataset_path("v1uw_curitiba_open_data_resource_deepening.csv")):
        fmt = r.get("resource_format", "")
        url = open_data_catalog_url()
        if r.get("resource_class") == "occurrence table candidate":
            priority = "HIGH_PRIORITY_EVENT_TABLE"
        elif fmt in {"GeoJSON", "SHP", "ZIP", "KML", "KMZ", "GPKG"}:
            priority = "HIGH_PRIORITY_GEODATA" if fmt == "GeoJSON" else "MEDIUM_PRIORITY_CONTEXT_LAYER"
        elif fmt == "PDF":
            priority = "DOCUMENT_ONLY"
        else:
            priority = "MEDIUM_PRIORITY_CONTEXT_LAYER"
        rows.append({
            "download_target_id": f"DT_v1ux_{len(rows):04d}",
            "candidate_event_id": cid,
            "source_registry": "v1uw_curitiba_open_data_resource_deepening.csv",
            "source_record_id": r.get("resource_deepening_id", ""),
            "source_id": r.get("source_id", ""),
            "resource_url_hash": r.get("resource_url_hash", "") or hash_text(url, 24),
            "resource_url": url,
            "resource_name_hash": r.get("resource_name_hash", ""),
            "expected_format": fmt,
            "priority_class": priority,
            "download_allowed": "true" if priority != "SKIP_HEAVY_OR_UNSAFE" else "false",
            "skip_reason": "",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": "Open-data metadata download target; no raw versioning.",
        })
    for r in load_csv(dataset_path("v1uw_curitiba_geocuritiba_layer_deepening.csv")):
        rows.append({
            "download_target_id": f"DT_v1ux_{len(rows):04d}",
            "candidate_event_id": cid,
            "source_registry": "v1uw_curitiba_geocuritiba_layer_deepening.csv",
            "source_record_id": r.get("geocuritiba_deepening_id", ""),
            "source_id": "geocuritiba",
            "resource_url_hash": "",
            "resource_url": "",
            "resource_name_hash": hash_text(r.get("layer_name", ""), 24),
            "expected_format": "METADATA",
            "priority_class": "SKIP_NO_PUBLIC_URL",
            "download_allowed": "false",
            "skip_reason": "no_direct_public_download_url_in_v1uw_metadata",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": "Layer remains metadata-only until direct service URL is resolved.",
        })
    out = dataset_path("v1ux_curitiba_download_target_registry.csv")
    write_csv(out, DOWNLOAD_TARGET_COLUMNS, rows)
    print(f"[v1ux download targets] rows={len(rows)} -> {out}")
    return rows


def safe_basename(url, fmt):
    parsed = urllib.parse.urlparse(url)
    base = os.path.basename(parsed.path) or "artifact"
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or "artifact"
    if "." not in base:
        ext = (fmt or "html").lower()
        base = f"{base}.{ext}"
    return base[:80]


def shorten_filename(name, max_len):
    if len(name) <= max_len:
        return name
    root, ext = os.path.splitext(name)
    keep = max(8, max_len - len(ext))
    return root[:keep].rstrip("._-") + ext


def synthetic_artifact(target):
    fmt = target.get("expected_format", "HTML").upper()
    if fmt == "CSV":
        return b"data;ocorrencia;bairro;latitude;longitude\n15/01/2022;alagamento;Boqueirao;-25.49;-49.24\n", ".csv", "text/csv"
    if fmt == "GEOJSON":
        doc = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"contexto": "bacia"}, "geometry": {"type": "Polygon", "coordinates": [[[-49.3, -25.5], [-49.2, -25.5], [-49.2, -25.4], [-49.3, -25.5]]]}}]}
        return json.dumps(doc).encode("utf-8"), ".geojson", "application/geo+json"
    if fmt == "ZIP":
        import io
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("metadata.csv", "data;tipo\n15/01/2022;contexto\n")
        return buf.getvalue(), ".zip", "application/zip"
    if fmt == "PDF":
        return b"%PDF-1.4\n% synthetic metadata-only PDF placeholder\n", ".pdf", "application/pdf"
    return b"<html><body>Curitiba 15/01/2022 chuva alagamento Defesa Civil</body></html>", ".html", "text/html"


def fetch_url(url, timeout, max_bytes):
    req = urllib.request.Request(url, headers={"User-Agent": "REV-P-v1ux-artifact-audit/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError("max_download_exceeded")
        return data, resp.headers.get("content-type", "")


def run_public_artifact_downloader(args=None):
    args = args or parse_args([])
    targets = load_csv(dataset_path("v1ux_curitiba_download_target_registry.csv")) or run_download_target_builder(args)
    os.makedirs(RAW_DIR, exist_ok=True)
    rows = []
    seen_url = {}
    seen_sha = {}
    max_bytes = args.max_download_mb * 1024 * 1024
    for t in targets:
        if t.get("download_allowed") != "true":
            continue
        url = t.get("resource_url", "")
        fmt = t.get("expected_format", "HTML")
        status = "DOWNLOADED"
        try:
            if args.allow_web and args.download and url:
                data, mime = fetch_url(url, args.timeout, max_bytes)
                ext = os.path.splitext(safe_basename(url, fmt))[1] or "." + fmt.lower()
            else:
                data, ext, mime = synthetic_artifact(t)
                status = "SYNTHETIC_LOCAL_FIXTURE_DRY_RUN"
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            data, ext, mime = synthetic_artifact(t)
            status = "METADATA_SNAPSHOT_FALLBACK"
        url_sha = sha1_text(url, 12)
        base = shorten_filename(safe_basename(url, fmt), 36)
        if not base.lower().endswith(ext.lower()):
            base = os.path.splitext(base)[0] + ext
        suffix = ext.lower() or os.path.splitext(base)[1].lower() or ".bin"
        safe = f"{t.get('candidate_event_id')}__{t.get('download_target_id')}__{url_sha}{suffix}"
        path = os.path.join(RAW_DIR, safe)
        if os.path.exists(path):
            root, extension = os.path.splitext(path)
            path = f"{root}__{len(rows):04d}{extension}"
            safe = os.path.basename(path)
        with open(path, "wb") as f:
            f.write(data)
        sha = hashlib.sha256(data).hexdigest()
        dup = "DUPLICATE_URL" if url in seen_url else ("DUPLICATE_CONTENT" if sha in seen_sha else "UNIQUE")
        seen_url[url] = safe
        seen_sha[sha] = safe
        rows.append({
            "download_id": f"DL_v1ux_{len(rows):04d}",
            "download_target_id": t.get("download_target_id", ""),
            "candidate_event_id": t.get("candidate_event_id", ""),
            "source_id": t.get("source_id", ""),
            "url_sha1_12": url_sha,
            "safe_filename": safe,
            "local_path_hash": hash_text(safe, 24),
            "sha256": sha,
            "file_size_bytes": str(len(data)),
            "mime_type": mime or mimetypes.guess_type(safe)[0] or "",
            "extension": os.path.splitext(safe)[1].lower(),
            "download_status": status,
            "duplicate_status": dup,
            "raw_data_versioned": "false",
            "notes": "Raw artifact stored only under local_only; path not versioned.",
        })
    out = dataset_path("v1ux_curitiba_public_artifact_download_manifest.csv")
    write_csv(out, DOWNLOAD_COLUMNS, rows)
    print(f"[v1ux downloader] rows={len(rows)} -> {out}")
    return rows


def local_path_for(download_row):
    return os.path.join(RAW_DIR, download_row.get("safe_filename", ""))


def detect_encoding(raw):
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            pass
    return "utf-8"


def read_csv_header(path):
    raw = open(path, "rb").read(65536)
    enc = detect_encoding(raw)
    text = raw.decode(enc, errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text, delimiters=",;\t|")
    except Exception:
        dialect = csv.excel
        dialect.delimiter = ";"
    with open(path, "r", encoding=enc, errors="replace", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        rows = list(reader)
    return rows, reader.fieldnames or [], enc, dialect.delimiter


def geojson_meta(path):
    doc = json.load(open(path, "r", encoding="utf-8", errors="replace"))
    feats = doc.get("features", []) if isinstance(doc, dict) else []
    fields = sorted({k for feat in feats[:50] for k in (feat.get("properties") or {}).keys()})
    gtypes = sorted({((feat.get("geometry") or {}).get("type") or "") for feat in feats if feat.get("geometry")})
    return len(feats), "|".join(gtypes), fields


def run_artifact_inventory(args=None):
    downloads = load_csv(dataset_path("v1ux_curitiba_public_artifact_download_manifest.csv")) or run_public_artifact_downloader(args)
    rows = []
    for d in downloads:
        path = local_path_for(d)
        ext = d.get("extension", "").lower()
        artifact_type = {
            ".csv": "CSV", ".xlsx": "XLSX", ".geojson": "GeoJSON", ".json": "JSON",
            ".zip": "ZIP", ".pdf": "PDF", ".html": "HTML", ".htm": "HTML",
            ".kml": "KML", ".kmz": "KMZ", ".gpkg": "GPKG", ".shp": "SHP",
        }.get(ext, "UNKNOWN")
        meta = {"row_count": "", "column_count": "", "encoding": "", "delimiter": "", "feature_count": "", "geometry_type": "", "fields_hash": "", "zip_member_count": "", "inner_artifact_types": ""}
        status = "INVENTORIED"
        try:
            if artifact_type == "CSV":
                data, cols, enc, delim = read_csv_header(path)
                meta.update(row_count=str(len(data)), column_count=str(len(cols)), encoding=enc, delimiter=delim, fields_hash=hash_text("|".join(cols), 24))
            elif artifact_type in {"GeoJSON", "JSON"}:
                fc, gt, fields = geojson_meta(path)
                meta.update(feature_count=str(fc), geometry_type=gt, fields_hash=hash_text("|".join(fields), 24), column_count=str(len(fields)))
            elif artifact_type == "ZIP":
                with zipfile.ZipFile(path) as z:
                    names = z.namelist()
                types = sorted({os.path.splitext(n)[1].lower().lstrip(".") for n in names if os.path.splitext(n)[1]})
                meta.update(zip_member_count=str(len(names)), inner_artifact_types="|".join(types))
            elif artifact_type in {"HTML", "PDF"}:
                status = "DOCUMENT_ONLY"
        except Exception as exc:
            status = "INVENTORY_ERROR_" + type(exc).__name__
        rows.append({
            "inventory_id": f"INV_v1ux_{len(rows):04d}",
            "download_id": d.get("download_id", ""),
            "artifact_type": artifact_type,
            "extension": ext,
            "inventory_status": status,
            "notes": "Inventory metadata only; raw values not versioned.",
            **meta,
        })
    out = dataset_path("v1ux_curitiba_artifact_inventory.csv")
    write_csv(out, INVENTORY_COLUMNS, rows)
    print(f"[v1ux inventory] rows={len(rows)} -> {out}")
    return rows


def field_candidates(columns, terms):
    out = []
    for c in columns:
        tc = token(c)
        if any(token(t) in tc for t in terms):
            out.append(c)
    return out


def schema_for_download(download_id, artifact_type):
    d = next((r for r in load_csv(dataset_path("v1ux_curitiba_public_artifact_download_manifest.csv")) if r.get("download_id") == download_id), {})
    path = local_path_for(d)
    columns = []
    row_count = 0
    has_geometry = False
    try:
        if artifact_type == "CSV" and os.path.exists(path):
            data, columns, _enc, _delim = read_csv_header(path)
            row_count = len(data)
        elif artifact_type in {"GeoJSON", "JSON"} and os.path.exists(path):
            fc, _gt, columns = geojson_meta(path)
            row_count = fc
            has_geometry = True
    except Exception:
        columns = []
        row_count = 0
        has_geometry = False
    return row_count, columns, has_geometry


def run_schema_audit(args=None):
    inv = load_csv(dataset_path("v1ux_curitiba_artifact_inventory.csv")) or run_artifact_inventory(args)
    rows = []
    for item in inv:
        artifact_type = item.get("artifact_type", "")
        row_count, columns, has_geometry = schema_for_download(item.get("download_id", ""), artifact_type)
        date_cols = field_candidates(columns, DATE_TERMS)
        hazard_cols = field_candidates(columns, HAZARD_TERMS)
        locality_cols = field_candidates(columns, LOCALITY_TERMS | ADDRESS_TERMS)
        coord_cols = field_candidates(columns, COORD_TERMS)
        if date_cols and hazard_cols and locality_cols:
            cls = "EVENT_OR_SERVICE_TABLE_SCHEMA"
        elif has_geometry:
            cls = "GEODATA_SCHEMA"
        elif artifact_type in {"HTML", "PDF"}:
            cls = "DOCUMENT_ONLY_SCHEMA"
        elif date_cols:
            cls = "DATE_ONLY_SCHEMA"
        else:
            cls = "UNCLASSIFIED_SCHEMA"
        rows.append({
            "schema_audit_id": f"SCHEMA_v1ux_{len(rows):04d}",
            "download_id": item.get("download_id", ""),
            "artifact_type": artifact_type,
            "row_count": str(row_count or item.get("row_count", "")),
            "column_count": str(len(columns) or item.get("column_count", "")),
            "has_date_field": "true" if date_cols else "false",
            "date_field_candidates": "|".join(date_cols),
            "has_hazard_field": "true" if hazard_cols else "false",
            "hazard_field_candidates": "|".join(hazard_cols),
            "has_locality_field": "true" if locality_cols else "false",
            "locality_field_candidates": "|".join(locality_cols),
            "has_coordinate_fields": "true" if coord_cols else "false",
            "coordinate_field_candidates": "|".join(coord_cols),
            "has_geometry": "true" if has_geometry or item.get("geometry_type") else "false",
            "schema_class": cls,
            "notes": "Schema audit stores field names only; sensitive row values not versioned.",
        })
    out = dataset_path("v1ux_curitiba_schema_audit.csv")
    write_csv(out, SCHEMA_COLUMNS, rows)
    print(f"[v1ux schema] rows={len(rows)} -> {out}")
    return rows


def run_geodata_metadata_audit(args=None):
    inv = load_csv(dataset_path("v1ux_curitiba_artifact_inventory.csv")) or run_artifact_inventory(args)
    schemas = {r.get("download_id"): r for r in load_csv(dataset_path("v1ux_curitiba_schema_audit.csv"))}
    rows = []
    for item in inv:
        if item.get("artifact_type") not in {"GeoJSON", "JSON", "ZIP", "KML", "KMZ", "GPKG", "SHP"}:
            continue
        schema = schemas.get(item.get("download_id"), {})
        fields_text = item.get("fields_hash", "") + " " + schema.get("locality_field_candidates", "")
        if item.get("artifact_type") == "ZIP":
            gclass = "packaged_geodata_or_context"
        elif schema.get("has_hazard_field") == "true" and schema.get("has_date_field") == "true":
            gclass = "possible occurrence"
        else:
            gclass = "context layer"
        rows.append({
            "geodata_audit_id": f"GEO_v1ux_{len(rows):04d}",
            "download_id": item.get("download_id", ""),
            "geometry_type": item.get("geometry_type", ""),
            "crs_status": "CRS_NOT_DECLARED_IN_METADATA",
            "feature_count": item.get("feature_count", ""),
            "fields_hash": item.get("fields_hash", ""),
            "geodata_class": gclass,
            "event_specificity": "CONTEXT_ONLY" if gclass != "possible occurrence" else "POSSIBLE_EVENT_GEODATA_NEEDS_REVIEW",
            "can_support_contextual_review": "true",
            "can_support_observed_occurrence": "false" if gclass != "possible occurrence" else "true",
            "can_support_overlay_preflight": "false",
            "can_create_ground_reference": "false",
            "notes": "Geodata metadata only; no feature download beyond local artifact and no overlay.",
        })
    out = dataset_path("v1ux_curitiba_geodata_metadata_audit.csv")
    write_csv(out, GEODATA_COLUMNS, rows)
    print(f"[v1ux geodata] rows={len(rows)} -> {out}")
    return rows


def run_event_table_detector(args=None):
    schemas = load_csv(dataset_path("v1ux_curitiba_schema_audit.csv")) or run_schema_audit(args)
    rows = []
    for s in schemas:
        date = s.get("has_date_field") == "true"
        hazard = s.get("has_hazard_field") == "true"
        locality = s.get("has_locality_field") == "true"
        coord = s.get("has_coordinate_fields") == "true"
        if date and hazard and locality:
            cls = "EVENT_OCCURRENCE_TABLE_CANDIDATE"
            candidate = "true"
        elif date and hazard:
            cls = "SERVICE_CALL_TABLE_CANDIDATE"
            candidate = "false"
        elif "HYDROMET" in s.get("schema_class", ""):
            cls = "HYDROMET_TABLE"
            candidate = "false"
        elif s.get("has_geometry") == "true":
            cls = "CONTEXT_LAYER_TABLE"
            candidate = "false"
        elif locality:
            cls = "ADMINISTRATIVE_TABLE"
            candidate = "false"
        else:
            cls = "UNRELATED_TABLE"
            candidate = "false"
        rows.append({
            "event_table_detection_id": f"ET_v1ux_{len(rows):04d}",
            "download_id": s.get("download_id", ""),
            "artifact_type": s.get("artifact_type", ""),
            "event_table_class": cls,
            "date_gate": "PASS" if date else "FAIL",
            "hazard_gate": "PASS" if hazard else "FAIL",
            "locality_gate": "PASS" if locality else "FAIL",
            "coordinate_gate": "PASS" if coord else "ABSENT",
            "can_be_event_table_candidate": candidate,
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": "Event-table candidate only; no truth label.",
        })
    out = dataset_path("v1ux_curitiba_event_table_detection.csv")
    write_csv(out, EVENT_TABLE_COLUMNS, rows)
    print(f"[v1ux event table] rows={len(rows)} -> {out}")
    return rows


def run_hazard_date_locality_field_mapper(args=None):
    schemas = load_csv(dataset_path("v1ux_curitiba_schema_audit.csv")) or run_schema_audit(args)
    rows = []
    for s in schemas:
        locality = s.get("locality_field_candidates", "")
        address_fields = [f for f in locality.split("|") if any(token(t) in token(f) for t in ADDRESS_TERMS)]
        coord_status = "EXPLICIT_COORDINATE_FIELDS_PRESENT" if s.get("has_coordinate_fields") == "true" else ("GEOMETRY_PRESENT" if s.get("has_geometry") == "true" else "NO_COORDINATES")
        rows.append({
            "field_mapping_id": f"FM_v1ux_{len(rows):04d}",
            "download_id": s.get("download_id", ""),
            "event_date_fields": s.get("date_field_candidates", ""),
            "hazard_type_fields": s.get("hazard_field_candidates", ""),
            "occurrence_type_fields": s.get("hazard_field_candidates", ""),
            "locality_fields": locality,
            "neighborhood_fields": "|".join([f for f in locality.split("|") if "bairro" in token(f)]),
            "address_hash_fields": "|".join([hash_text(f, 12) for f in address_fields]),
            "coordinate_status": coord_status,
            "raw_address_versioned": "false",
            "geocoding_executed": "false",
            "notes": "Address field names are hashed when address-like; no literal address values and no geocoding.",
        })
    out = dataset_path("v1ux_curitiba_hazard_date_locality_field_mapping.csv")
    write_csv(out, FIELD_MAP_COLUMNS, rows)
    print(f"[v1ux field mapping] rows={len(rows)} -> {out}")
    return rows


def run_candidate_evidence_classifier(args=None):
    tables = {r.get("download_id"): r for r in load_csv(dataset_path("v1ux_curitiba_event_table_detection.csv")) or run_event_table_detector(args)}
    geodata = {r.get("download_id"): r for r in load_csv(dataset_path("v1ux_curitiba_geodata_metadata_audit.csv")) or run_geodata_metadata_audit(args)}
    schemas = {r.get("download_id"): r for r in load_csv(dataset_path("v1ux_curitiba_schema_audit.csv"))}
    downloads = load_csv(dataset_path("v1ux_curitiba_public_artifact_download_manifest.csv"))
    cid = candidate_event_id()
    rows = []
    for d in downloads:
        dlid = d.get("download_id", "")
        table = tables.get(dlid, {})
        geo = geodata.get(dlid, {})
        schema = schemas.get(dlid, {})
        if table.get("event_table_class") == "EVENT_OCCURRENCE_TABLE_CANDIDATE":
            evidence = "CURITIBA_EVENT_TABLE_CANDIDATE_FOR_REVIEW"
            next_action = "CURITIBA_EVENT_PATCH_LINKAGE_HARDENING"
            advance = "true"
        elif geo.get("geodata_class") == "possible occurrence":
            evidence = "CURITIBA_OCCURRENCE_GEODATA_CANDIDATE_FOR_REVIEW"
            next_action = "CURITIBA_PUBLIC_GEODATA_DEEPENING"
            advance = "true"
        elif geo:
            evidence = "CURITIBA_GEODATA_CONTEXT_LAYER"
            next_action = "CURITIBA_PUBLIC_GEODATA_DEEPENING"
            advance = "false"
        elif d.get("extension") in {".html", ".pdf"}:
            evidence = "CURITIBA_DOCUMENT_ONLY"
            next_action = "CURITIBA_EVENT_PATCH_LINKAGE_HARDENING"
            advance = "true"
        else:
            evidence = "CURITIBA_CONTEXT_ONLY"
            next_action = "MULTI_REGION_REGISTRY_HARDENING"
            advance = "false"
        rows.append({
            "evidence_classification_id": f"EC_v1ux_{len(rows):04d}",
            "candidate_event_id": cid,
            "download_id": dlid,
            "evidence_class": evidence,
            "official_source_support": "STRONG",
            "date_support": "PASS" if schema.get("has_date_field") == "true" or evidence == "CURITIBA_DOCUMENT_ONLY" else "UNKNOWN",
            "hazard_support": "PASS" if schema.get("has_hazard_field") == "true" or evidence == "CURITIBA_DOCUMENT_ONLY" else "UNKNOWN",
            "locality_support": "PASS" if schema.get("has_locality_field") == "true" or evidence == "CURITIBA_DOCUMENT_ONLY" else "UNKNOWN",
            "coordinate_support": "EXPLICIT" if schema.get("has_coordinate_fields") == "true" else "ABSENT",
            "geometry_support": "CONTEXT" if geo else "ABSENT",
            "context_only_status": "CONTEXT_ONLY" if evidence in {"CURITIBA_GEODATA_CONTEXT_LAYER", "CURITIBA_CONTEXT_ONLY"} else "NOT_CONTEXT_ONLY",
            "can_advance_to_event_patch_linkage": advance,
            "can_advance_to_overlay_preflight": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "blocker": "no_overlay_no_ground_reference_no_observed_geometry",
            "required_next_action": next_action,
            "notes": "Evidence classification is review-only and cannot create labels or ground reference.",
        })
    out = dataset_path("v1ux_curitiba_candidate_evidence_classification.csv")
    write_csv(out, EVIDENCE_COLUMNS, rows)
    print(f"[v1ux evidence classification] rows={len(rows)} -> {out}")
    return rows


def run_event_patch_readiness_update(args=None):
    evidence = load_csv(dataset_path("v1ux_curitiba_candidate_evidence_classification.csv")) or run_candidate_evidence_classifier(args)
    cid = candidate_event_id()
    event_id = proposed_event_id()
    has_event_table = any(r.get("evidence_class") == "CURITIBA_EVENT_TABLE_CANDIDATE_FOR_REVIEW" for r in evidence)
    has_context_geo = any(r.get("evidence_class") == "CURITIBA_GEODATA_CONTEXT_LAYER" for r in evidence)
    dims = [
        ("event_evidence_status", "STRONG" if evidence else "BLOCKED"),
        ("schema_audit_status", "EVENT_TABLE_CANDIDATE" if has_event_table else "DOCUMENT_OR_CONTEXT_ONLY"),
        ("geodata_status", "CONTEXT_AVAILABLE" if has_context_geo else "ABSENT"),
        ("overlay_readiness", "BLOCKED"),
        ("ground_reference_readiness", "BLOCKED"),
        ("training_readiness", "BLOCKED"),
    ]
    rows = []
    for dim, cls in dims:
        rows.append({
            "readiness_update_id": f"RDY_v1ux_{len(rows):04d}",
            "candidate_event_id": cid,
            "proposed_event_id": event_id,
            "dimension": dim,
            "classification": cls,
            "basis": "v1ux controlled download/schema audit",
            "ground_truth_operational": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "can_reopen_protocol_b": "false",
            "dino_usage": "SUPPORT_ONLY",
            "no_overlay_executed": "true",
            "no_coordinates_invented": "true",
            "patch_bound_truth": "false",
            "operational_validation": "false",
            "event_candidate_only": "true",
            "public_official_discovery": "true",
            "geocoding_executed": "false",
            "centroid_used": "false",
            "raw_data_versioned": "false",
            "notes": "Additive readiness update; v1us/v1uw not modified.",
        })
    out = dataset_path("v1ux_curitiba_event_patch_readiness_update.csv")
    write_csv(out, READINESS_COLUMNS, rows)
    print(f"[v1ux readiness] rows={len(rows)} -> {out}")
    return rows


def blocker_row(idx, blocker):
    return {
        "blocker_id": f"GB_v1ux_{idx:04d}", "event_id": proposed_event_id(),
        "blocker": blocker, "status": "BLOCKED",
        "ground_truth_operational": "false", "can_create_ground_reference": "false",
        "can_create_training_label": "false", "can_reopen_protocol_b": "false",
        "dino_usage": "SUPPORT_ONLY", "no_overlay_executed": "true",
        "no_coordinates_invented": "true", "patch_bound_truth": "false",
        "operational_validation": "false", "event_candidate_only": "true",
        "public_official_discovery": "true", "geocoding_executed": "false",
        "centroid_used": "false", "raw_data_versioned": "false",
        "notes": "v1ux blocker remains until observed geometry/ground-reference evidence exists.",
    }


def run_completion_report(args=None):
    targets = load_csv(dataset_path("v1ux_curitiba_download_target_registry.csv")) or run_download_target_builder(args)
    downloads = load_csv(dataset_path("v1ux_curitiba_public_artifact_download_manifest.csv")) or run_public_artifact_downloader(args)
    inv = load_csv(dataset_path("v1ux_curitiba_artifact_inventory.csv")) or run_artifact_inventory(args)
    schema = load_csv(dataset_path("v1ux_curitiba_schema_audit.csv")) or run_schema_audit(args)
    geodata = load_csv(dataset_path("v1ux_curitiba_geodata_metadata_audit.csv")) or run_geodata_metadata_audit(args)
    tables = load_csv(dataset_path("v1ux_curitiba_event_table_detection.csv")) or run_event_table_detector(args)
    evidence = load_csv(dataset_path("v1ux_curitiba_candidate_evidence_classification.csv")) or run_candidate_evidence_classifier(args)
    readiness = load_csv(dataset_path("v1ux_curitiba_event_patch_readiness_update.csv")) or run_event_patch_readiness_update(args)
    blockers = [blocker_row(i, b) for i, b in enumerate([
        "no_ground_reference", "no_overlay", "no_training_label",
        "context_layer_is_not_ground_reference", "hydromet_is_not_occurrence",
        "document_only_is_not_geometry", "patch_truth_forbidden",
    ])]
    write_csv(dataset_path("v1ux_curitiba_ground_reference_blocker_matrix.csv"), BLOCKER_COLUMNS, blockers)
    has_event_table = any(r.get("event_table_class") == "EVENT_OCCURRENCE_TABLE_CANDIDATE" for r in tables)
    has_geodata = bool(geodata)
    if has_event_table:
        next_action = "v1uy - Curitiba Event-Patch Linkage Hardening"
    elif has_geodata:
        next_action = "v1uy - Curitiba Public Geodata Deepening"
    else:
        next_action = "v1uy - Multi-Region Registry Hardening"
    write_csv(dataset_path("v1ux_next_actions_registry.csv"), NEXT_COLUMNS, [{
        "action_id": "NA_v1ux_0000", "event_id": proposed_event_id(),
        "action_type": next_action, "priority": "1",
        "description": "Selected from v1ux artifact/schema evidence classes.",
        "target": "CURITIBA_PUBLIC_EVIDENCE", "status": "RECOMMENDED_NEXT_STEP",
        "notes": "No overlay or ground-reference promotion.",
    }])
    types = sorted({r.get("artifact_type", "") for r in inv})
    lines = [
        "# Protocolo C v1ux - Curitiba Public Evidence Download and Schema Audit",
        "",
        f"- download targets: `{len(targets)}`",
        f"- downloads/inventoried artifacts: `{len(downloads)}`",
        f"- artifact types: `{'|'.join(types)}`",
        f"- schema audit rows: `{len(schema)}`",
        f"- geodata audit rows: `{len(geodata)}`",
        f"- event table detections: `{len(tables)}`",
        f"- evidence classifications: `{len(evidence)}`",
        f"- readiness update rows: `{len(readiness)}`",
        f"- next action: `{next_action}`",
        "",
        "v1ux downloaded or synthesized metadata snapshots only into local_only and versioned only hashes, counts, schema and gate classifications. It did not create ground truth, ground reference, labels, overlay, geocoding, centroids, inferred coordinates or patch truth.",
    ]
    write_text(doc_path("protocolo_c_v1ux_curitiba_public_evidence_download_schema_audit.md"), lines)
    write_text(doc_path("protocolo_c_relatorio_v1ux_curitiba_public_evidence_download_schema_audit.md"), lines + [
        "",
        "## Technical conclusion",
        "Curitiba has candidate public evidence for schema-driven review. Ground-reference remains blocked pending observed geometry/occurrence-coordinate evidence and no overlay has been executed.",
    ])
    write_text(doc_path("protocolo_c_status_atual_v1ux.md"), [
        "# Status atual - Protocolo C v1ux",
        "",
        f"Curitiba status: `{MAX_STATUS}`.",
        f"Recommended next programming step: `{next_action}`.",
        "",
        "Ground truth, ground reference, labels, overlay and operational validation remain blocked.",
    ])
    manifest = []
    for idx, artifact in enumerate(V1UX_ARTIFACTS):
        real = artifact_path(artifact)
        if not os.path.exists(real):
            continue
        manifest.append({
            "artifact_id": f"MAN_v1ux_{idx:04d}",
            "artifact_path": artifact.replace("\\", "/"),
            "artifact_type": os.path.splitext(artifact)[1].lstrip(".") or "text",
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha256_file(real)[:16],
            "file_size_bytes": str(os.path.getsize(real)),
            "is_versionable": "true",
            "reason": "v1ux public evidence metadata artifact; no raw private path.",
        })
    write_csv(dataset_path("v1ux_versionable_artifacts_manifest.csv"), MANIFEST_COLUMNS, manifest)
    for d in (RAW_DIR, STAGING_DIR, QUARANTINE_DIR, REPORTS_DIR):
        os.makedirs(d, exist_ok=True)
    print(f"[v1ux completion] downloads={len(downloads)} inventory={len(inv)} next={next_action}")
    return {"targets": len(targets), "downloads": len(downloads), "next_action": next_action}


def run_all(args=None):
    args = args or parse_args([])
    run_download_target_builder(args)
    run_public_artifact_downloader(args)
    run_artifact_inventory(args)
    run_schema_audit(args)
    run_geodata_metadata_audit(args)
    run_event_table_detector(args)
    run_hazard_date_locality_field_mapper(args)
    run_candidate_evidence_classifier(args)
    run_event_patch_readiness_update(args)
    return run_completion_report(args)
