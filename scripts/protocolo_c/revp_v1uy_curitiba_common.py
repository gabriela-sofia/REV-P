#!/usr/bin/env python3
"""v1uy Curitiba public geodata deepening.

Metadata-only endpoint probing and layer classification for Curitiba public
geodata candidates. Raw responses and samples stay in local_only; versionable
outputs contain hashes, counts, flags, and blocked review decisions only.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zipfile

PROTOCOL_VERSION = "v1uy"
DATASET_DIR = "datasets/protocolo_c"
DOCS_DIR = "docs/metodologia_cientifica"
CONFIG_DIR = "configs/protocolo_c"
V1UX_RAW_DIR = "local_only/protocolo_c/curitiba_public_evidence_download/raw/v1ux"
RAW_DIR = "local_only/protocolo_c/curitiba_public_geodata_deepening/raw/v1uy"
STAGING_DIR = "local_only/protocolo_c/curitiba_public_geodata_deepening/staging/v1uy"
QUARANTINE_DIR = "local_only/protocolo_c/curitiba_public_geodata_deepening/quarantine/v1uy"
REPORTS_DIR = "local_only/protocolo_c/curitiba_public_geodata_deepening/reports/v1uy"
MAX_STATUS = "CURITIBA_PUBLIC_GEODATA_DEEPENED_EVENT_CANDIDATE"

GUARDRAIL_COLUMNS = [
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "can_reopen_protocol_b", "dino_usage",
    "no_overlay_executed", "no_coordinates_invented", "patch_bound_truth",
    "operational_validation", "event_candidate_only",
    "public_official_discovery", "geocoding_executed", "centroid_used",
    "raw_data_versioned",
]

MISMATCH_COLUMNS = [
    "mismatch_id", "candidate_event_id", "download_id", "expected_type",
    "detected_type", "mismatch_status", "original_url_hash",
    "alternative_url_hash", "alternative_url_type", "recoverable",
    "recovery_strategy", *GUARDRAIL_COLUMNS, "notes",
]
PROBE_COLUMNS = [
    "endpoint_probe_id", "candidate_event_id", "source_record_id",
    "endpoint_url_hash", "endpoint_type", "http_status", "content_type",
    "service_type", "probe_status", "supports_metadata",
    "supports_feature_query", "supports_count_query", "feature_count_hint",
    "geometry_type_hint", "spatial_reference_hint",
    "can_support_contextual_review", "can_support_occurrence_review",
    *GUARDRAIL_COLUMNS, "notes",
]
METADATA_COLUMNS = [
    "layer_metadata_id", "candidate_event_id", "endpoint_probe_id",
    "layer_name_hash", "service_name_hash", "geometry_type",
    "spatial_reference", "extent_hash", "field_count", "fields_hash",
    "has_date_field", "has_hazard_field", "has_locality_field",
    "has_coordinate_fields", "metadata_class", *GUARDRAIL_COLUMNS, "notes",
]
SAMPLE_COLUMNS = [
    "feature_sample_id", "candidate_event_id", "endpoint_probe_id",
    "sample_status", "feature_count", "sampled_feature_count",
    "sample_local_hash", "has_geometry", "has_date_values",
    "has_hazard_values", "has_locality_values",
    "geometry_sample_versioned", "raw_feature_versioned",
    *GUARDRAIL_COLUMNS, "notes",
]
CLASSIFICATION_COLUMNS = [
    "layer_classification_id", "candidate_event_id", "endpoint_probe_id",
    "layer_metadata_id", "layer_class", "event_specificity", "has_geometry",
    "has_date_field", "has_hazard_field", "has_locality_field",
    "context_only_status", "can_support_contextual_review",
    "can_support_observed_occurrence", "can_create_ground_reference",
    *[c for c in GUARDRAIL_COLUMNS if c != "can_create_ground_reference"],
    "notes",
]
OCCURRENCE_COLUMNS = [
    "occurrence_layer_audit_id", "candidate_event_id",
    "layer_classification_id", "official_source_support", "date_support",
    "hazard_support", "locality_support", "geometry_support",
    "event_specificity", "context_only_status",
    "can_advance_to_controlled_download",
    "can_advance_to_overlay_preflight", "can_create_ground_reference",
    "can_create_training_label", "blocker", "required_next_action",
    *[c for c in GUARDRAIL_COLUMNS if c not in {"can_create_ground_reference", "can_create_training_label"}],
    "notes",
]
PLAN_COLUMNS = [
    "download_plan_id", "candidate_event_id", "endpoint_probe_id",
    "layer_class", "plan_status", "recommended_filter", "max_features",
    "allowed_fields_hash", "requires_redaction", "overclaim_risk",
    "can_execute_now", "recommended_next_version",
    *GUARDRAIL_COLUMNS, "notes",
]
READINESS_COLUMNS = [
    "readiness_update_id", "candidate_event_id", "proposed_event_id",
    "dimension", "classification", "basis", *GUARDRAIL_COLUMNS, "notes",
]
BLOCKER_COLUMNS = [
    "blocker_id", "event_id", "blocker", "status", *GUARDRAIL_COLUMNS,
    "notes",
]
NEXT_COLUMNS = [
    "action_id", "event_id", "action_type", "priority", "description",
    "target", "status", "notes",
]
MANIFEST_COLUMNS = [
    "artifact_id", "artifact_path", "artifact_type", "protocol_version",
    "sha256_prefix", "file_size_bytes", "is_versionable", "reason",
]

V1UY_ARTIFACTS = [
    "configs/protocolo_c/v1uy_curitiba_content_mismatch_policy.yaml",
    "configs/protocolo_c/v1uy_curitiba_endpoint_probe_policy.yaml",
    "configs/protocolo_c/v1uy_curitiba_layer_classification_policy.yaml",
    "configs/protocolo_c/v1uy_curitiba_feature_schema_policy.yaml",
    "configs/protocolo_c/v1uy_curitiba_occurrence_layer_policy.yaml",
    "configs/protocolo_c/v1uy_curitiba_next_action_policy.yaml",
    "datasets/protocolo_c/v1uy_curitiba_content_mismatch_resolution.csv",
    "datasets/protocolo_c/v1uy_curitiba_geodata_endpoint_probe.csv",
    "datasets/protocolo_c/v1uy_curitiba_layer_metadata_extraction.csv",
    "datasets/protocolo_c/v1uy_curitiba_feature_schema_sampling.csv",
    "datasets/protocolo_c/v1uy_curitiba_context_layer_classification.csv",
    "datasets/protocolo_c/v1uy_curitiba_possible_occurrence_layer_audit.csv",
    "datasets/protocolo_c/v1uy_curitiba_controlled_feature_download_plan.csv",
    "datasets/protocolo_c/v1uy_curitiba_event_patch_readiness_update.csv",
    "datasets/protocolo_c/v1uy_curitiba_ground_reference_blocker_matrix.csv",
    "datasets/protocolo_c/v1uy_next_actions_registry.csv",
    "docs/metodologia_cientifica/protocolo_c_v1uy_curitiba_public_geodata_deepening.md",
    "docs/metodologia_cientifica/protocolo_c_relatorio_v1uy_curitiba_public_geodata_deepening.md",
    "docs/metodologia_cientifica/protocolo_c_status_atual_v1uy.md",
]

DATE_TERMS = {"data", "date", "dt", "dia", "created", "updated"}
HAZARD_TERMS = {"alagamento", "inundacao", "enchente", "chuva", "risco", "ocorrencia", "tipo", "defesa_civil"}
LOCALITY_TERMS = {"bairro", "regional", "localidade", "municipio", "cidade", "logradouro", "endereco"}
COORD_TERMS = {"lat", "latitude", "lon", "lng", "long", "longitude", "x", "y", "geometry", "geom", "shape"}
DRAINAGE_TERMS = {"drenagem", "bacia", "hidrograf", "rio", "canal", "galeria"}
ADMIN_TERMS = {"bairro", "regional", "administr", "limite", "municip"}
RISK_TERMS = {"risco", "suscet", "vulnerab", "perigo", "amea"}
INFRA_TERMS = {"equip", "infra", "obra", "rede", "ponto"}
OCCURRENCE_TERMS = {"ocorr", "atendimento", "chamado", "defesa", "alag", "inund", "evento"}


def norm(value):
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower().strip()


def token(value):
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


def artifact_path(path):
    base = os.path.basename(path)
    if path.startswith("datasets/protocolo_c/"):
        return dataset_path(base)
    if path.startswith("configs/protocolo_c/"):
        return config_path(base)
    if path.startswith("docs/metodologia_cientifica/"):
        return doc_path(base)
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-web", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def guardrails():
    return {
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
    }


def write_policy_configs():
    policies = {
        "v1uy_curitiba_content_mismatch_policy.yaml": [
            "raw_data_versioned: false", "bypass_access_controls: false",
            "alternative_url_publication: hash_only",
        ],
        "v1uy_curitiba_endpoint_probe_policy.yaml": [
            "feature_mass_download: false", "metadata_probe_only: true",
            "count_query_allowed: true",
        ],
        "v1uy_curitiba_layer_classification_policy.yaml": [
            "context_is_not_occurrence: true", "ground_reference_allowed: false",
        ],
        "v1uy_curitiba_feature_schema_policy.yaml": [
            "max_public_sample_features: 0", "raw_samples_local_only: true",
            "geometry_publication_allowed: false",
        ],
        "v1uy_curitiba_occurrence_layer_policy.yaml": [
            "requires_date_hazard_geometry_event_specificity: true",
            "auto_promotion_allowed: false",
        ],
        "v1uy_curitiba_next_action_policy.yaml": [
            "max_status: CURITIBA_PUBLIC_GEODATA_DEEPENED_EVENT_CANDIDATE",
            "overlay_allowed: false",
        ],
    }
    for name, lines in policies.items():
        write_text(config_path(name), lines)


def candidate_event_id():
    rows = load_csv(dataset_path("v1uw_curitiba_event_candidate_status.csv"))
    return rows[0].get("candidate_event_id", "CE_v1uv_0000") if rows else "CE_v1uv_0000"


def proposed_event_id():
    rows = load_csv(dataset_path("v1uw_curitiba_event_candidate_status.csv"))
    return rows[0].get("proposed_event_id", "CUR_2022_01_15") if rows else "CUR_2022_01_15"


def target_by_download_id():
    targets = {r.get("download_target_id"): r for r in load_csv(dataset_path("v1ux_curitiba_download_target_registry.csv"))}
    out = {}
    for row in load_csv(dataset_path("v1ux_curitiba_public_artifact_download_manifest.csv")):
        out[row.get("download_id")] = targets.get(row.get("download_target_id"), {})
    return out


def download_by_id():
    return {r.get("download_id"): r for r in load_csv(dataset_path("v1ux_curitiba_public_artifact_download_manifest.csv"))}


def local_v1ux_path(download):
    return os.path.join(V1UX_RAW_DIR, download.get("safe_filename", ""))


def read_bytes(path, max_bytes=512 * 1024):
    try:
        with open(path, "rb") as f:
            return f.read(max_bytes)
    except OSError:
        return b""


def detect_content(data):
    if not data:
        return "empty file"
    prefix = data[:4096].lstrip()
    low = prefix[:4096].lower()
    if data.startswith(b"PK\x03\x04"):
        return "binary zip"
    if low.startswith(b"{") or low.startswith(b"["):
        try:
            json.loads(data[:65536].decode("utf-8", errors="replace"))
            return "metadata JSON"
        except Exception:
            return "mislabeled content"
    if low.startswith(b"<!doctype html") or low.startswith(b"<html") or b"<html" in low:
        if b"captcha" in low or b"access denied" in low or b"forbidden" in low:
            return "redirect/captcha/block"
        return "portal page"
    if b"\x00" in prefix:
        return "binary real"
    return "text/plain"


def extract_alternative_urls(data):
    text = data.decode("utf-8", errors="ignore")
    urls = re.findall(r"https?://[^\"' <>)]+", text)
    hrefs = re.findall(r"href=[\"']([^\"']+)[\"']", text, flags=re.I)
    urls.extend(hrefs)
    out = []
    for url in urls:
        clean = url.strip().rstrip(".,;")
        if any(term in clean.lower() for term in [".geojson", ".json", ".zip", "featureserver", "mapserver", "wfs", "ows"]):
            out.append(clean)
    return out[:5]


def mismatch_status(expected, detected, alternatives):
    if detected == "empty file":
        return "BROKEN_OR_EMPTY_CONTENT"
    if detected == "redirect/captcha/block":
        return "ACCESS_BLOCKED"
    if alternatives:
        if any(any(s in a.lower() for s in [".geojson", ".zip"]) for a in alternatives):
            return "RECOVERABLE_DIRECT_DOWNLOAD_LINK"
        return "RECOVERABLE_METADATA_ENDPOINT"
    if detected == "portal page":
        return "HTML_PORTAL_PAGE_NOT_GEODATA"
    if detected == "metadata JSON":
        return "RECOVERABLE_METADATA_ENDPOINT"
    if detected in {"text/plain", "mislabeled content"}:
        return "MIME_TYPE_MISMATCH"
    return "NOT_RECOVERABLE"


def run_content_mismatch_resolver(args=None):
    write_policy_configs()
    cid = candidate_event_id()
    downloads = download_by_id()
    targets = target_by_download_id()
    rows = []
    inventory = load_csv(dataset_path("v1ux_curitiba_artifact_inventory.csv"))
    for item in inventory:
        status = item.get("inventory_status", "")
        if "JSONDecodeError" not in status and "BadZipFile" not in status:
            continue
        download_id = item.get("download_id", "")
        download = downloads.get(download_id, {})
        target = targets.get(download_id, {})
        data = read_bytes(local_v1ux_path(download))
        detected = detect_content(data)
        alternatives = extract_alternative_urls(data)
        alt = alternatives[0] if alternatives else ""
        mstatus = mismatch_status(item.get("artifact_type", ""), detected, alternatives)
        rows.append({
            "mismatch_id": f"MM_v1uy_{len(rows):04d}",
            "candidate_event_id": cid,
            "download_id": download_id,
            "expected_type": item.get("artifact_type", ""),
            "detected_type": detected,
            "mismatch_status": mstatus,
            "original_url_hash": target.get("resource_url_hash", ""),
            "alternative_url_hash": hash_text(alt, 24) if alt else "",
            "alternative_url_type": classify_endpoint_type(alt) if alt else "",
            "recoverable": "true" if mstatus.startswith("RECOVERABLE") else "false",
            "recovery_strategy": "probe_alternative_metadata" if alt else "keep_as_mismatch_and_probe_source_catalog",
            **guardrails(),
            "notes": "Content mismatch resolved from local raw bytes only; no bypass or raw publication.",
        })
    out = dataset_path("v1uy_curitiba_content_mismatch_resolution.csv")
    write_csv(out, MISMATCH_COLUMNS, rows)
    print(f"[v1uy mismatch] rows={len(rows)} -> {out}")
    return rows


def classify_endpoint_type(url):
    low = (url or "").lower()
    if not url:
        return "NO_ENDPOINT"
    if "featureserver" in low:
        return "ARCGIS_FEATURESERVER"
    if "mapserver" in low:
        return "ARCGIS_MAPSERVER"
    if "wfs" in low or "service=wfs" in low:
        return "GEOSERVER_WFS"
    if low.endswith(".geojson") or ".geojson" in low:
        return "STATIC_GEOJSON"
    if low.endswith(".zip") or ".zip" in low:
        return "STATIC_ZIP"
    if "dadosabertos" in low:
        return "OPEN_DATA_CATALOG"
    if "geocuritiba" in low:
        return "GEOCURITIBA_PORTAL"
    return "WEB_PAGE"


def fetch_head_or_sample(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": "REV-P-v1uy-metadata-probe/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return str(resp.status), resp.headers.get("content-type", ""), resp.read(128 * 1024)


def endpoint_candidates():
    candidates = []
    seen = set()
    cid = candidate_event_id()
    targets = load_csv(dataset_path("v1ux_curitiba_download_target_registry.csv"))
    for row in targets:
        expected = row.get("expected_format", "")
        if expected not in {"GeoJSON", "ZIP", "METADATA"} and "GEODATA" not in row.get("priority_class", ""):
            continue
        url = row.get("resource_url", "")
        key = (row.get("source_record_id", ""), url, expected)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "candidate_event_id": cid,
            "source_record_id": row.get("source_record_id", ""),
            "url": url,
            "endpoint_type": classify_endpoint_type(url),
            "source": "v1ux_target",
        })
    for row in load_csv(dataset_path("v1uv_curitiba_geocuritiba_registry.csv")):
        url = row.get("layer_url") or row.get("service_url", "")
        key = (row.get("geocuritiba_record_id", ""), url, "geocuritiba")
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "candidate_event_id": cid,
            "source_record_id": row.get("geocuritiba_record_id", ""),
            "url": url,
            "endpoint_type": classify_endpoint_type(url),
            "source": "v1uv_geocuritiba",
        })
    for row in load_csv(dataset_path("v1uv_curitiba_open_data_registry.csv")):
        url = row.get("package_url", "")
        key = (row.get("open_data_record_id", ""), url, "open_data")
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "candidate_event_id": cid,
            "source_record_id": row.get("open_data_record_id", ""),
            "url": url,
            "endpoint_type": classify_endpoint_type(url),
            "source": "v1uv_open_data",
        })
    return candidates


def classify_probe_response(endpoint_type, content_type, data):
    detected = detect_content(data)
    low = data[:4096].lower()
    if endpoint_type in {"ARCGIS_FEATURESERVER", "ARCGIS_MAPSERVER"}:
        return "ARCGIS_REST", "PROBED_METADATA", "true", "true", "true", detected
    if endpoint_type == "GEOSERVER_WFS":
        return "GEOSERVER_WFS", "PROBED_METADATA", "true", "true", "true", detected
    if endpoint_type == "STATIC_GEOJSON" and detected == "metadata JSON":
        return "STATIC_GEOJSON", "PROBED_METADATA", "true", "false", "false", detected
    if "html" in content_type.lower() or detected == "portal page":
        return "HTML_PORTAL", "PORTAL_PAGE_NOT_QUERYABLE_GEODATA", "false", "false", "false", detected
    if b"arcgis" in low or b"featureserver" in low:
        return "ARCGIS_REST_HINT", "PROBED_METADATA", "true", "false", "false", detected
    return "UNKNOWN", "NOT_QUERYABLE_OR_UNVERIFIED", "false", "false", "false", detected


def run_geodata_endpoint_probe(args=None):
    args = args or parse_args([])
    rows = []
    for cand in endpoint_candidates():
        url = cand.get("url", "")
        http_status = ""
        ctype = ""
        service = "NO_ENDPOINT"
        pstatus = "NOT_PROBED_NO_URL"
        supports_metadata = "false"
        supports_feature = "false"
        supports_count = "false"
        feature_count = ""
        geometry = ""
        spatial = ""
        notes = "No direct endpoint URL in prior metadata."
        if url:
            if args.allow_web:
                try:
                    http_status, ctype, data = fetch_head_or_sample(url, args.timeout)
                    service, pstatus, supports_metadata, supports_feature, supports_count, detected = classify_probe_response(cand.get("endpoint_type", ""), ctype, data)
                    notes = f"Metadata probe only; detected {detected}."
                except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                    service = cand.get("endpoint_type", "")
                    pstatus = "PROBE_ERROR_" + type(exc).__name__
                    notes = "Probe failed without bypass or feature download."
            else:
                service = cand.get("endpoint_type", "")
                pstatus = "NOT_PROBED_WEB_DISABLED"
                notes = "Web probe disabled; endpoint retained as hashed candidate."
        rows.append({
            "endpoint_probe_id": f"EP_v1uy_{len(rows):04d}",
            "candidate_event_id": cand.get("candidate_event_id", ""),
            "source_record_id": cand.get("source_record_id", ""),
            "endpoint_url_hash": hash_text(url, 24) if url else "",
            "endpoint_type": cand.get("endpoint_type", ""),
            "http_status": http_status,
            "content_type": ctype,
            "service_type": service,
            "probe_status": pstatus,
            "supports_metadata": supports_metadata,
            "supports_feature_query": supports_feature,
            "supports_count_query": supports_count,
            "feature_count_hint": feature_count,
            "geometry_type_hint": geometry,
            "spatial_reference_hint": spatial,
            "can_support_contextual_review": "true" if url or cand.get("source_record_id", "") else "false",
            "can_support_occurrence_review": "false",
            **guardrails(),
            "notes": notes,
        })
    out = dataset_path("v1uy_curitiba_geodata_endpoint_probe.csv")
    write_csv(out, PROBE_COLUMNS, rows)
    print(f"[v1uy endpoint probe] rows={len(rows)} -> {out}")
    return rows


def source_lookup():
    lookup = {}
    for row in load_csv(dataset_path("v1uv_curitiba_geocuritiba_registry.csv")):
        lookup[row.get("geocuritiba_record_id")] = row
    for row in load_csv(dataset_path("v1uw_curitiba_geocuritiba_layer_deepening.csv")):
        lookup[row.get("geocuritiba_deepening_id")] = row
    for row in load_csv(dataset_path("v1uw_curitiba_open_data_resource_deepening.csv")):
        lookup[row.get("resource_deepening_id")] = row
    return lookup


def field_names_from_source(row):
    fields = row.get("fields", "") or row.get("layer_name", "") or row.get("resource_format", "")
    if "|" in fields:
        return [f for f in fields.split("|") if f]
    if "," in fields:
        return [f.strip() for f in fields.split(",") if f.strip()]
    return [fields] if fields else []


def has_any(fields, terms):
    hay = " ".join(token(f) for f in fields)
    return any(token(t) in hay for t in terms)


def run_layer_metadata_extractor(args=None):
    probes = load_csv(dataset_path("v1uy_curitiba_geodata_endpoint_probe.csv")) or run_geodata_endpoint_probe(args)
    lookup = source_lookup()
    rows = []
    for probe in probes:
        src = lookup.get(probe.get("source_record_id"), {})
        fields = field_names_from_source(src)
        layer_name = src.get("layer_name") or src.get("resource_format") or probe.get("endpoint_type", "")
        service_name = src.get("source_id") or probe.get("service_type", "")
        geometry = src.get("geometry_type") or probe.get("geometry_type_hint", "")
        spatial = src.get("spatial_reference") or probe.get("spatial_reference_hint", "")
        supports = probe.get("supports_metadata") == "true" or bool(src)
        if has_any(fields + [layer_name], OCCURRENCE_TERMS):
            meta_class = "OCCURRENCE_SCHEMA_HINT"
        elif has_any(fields + [layer_name], DRAINAGE_TERMS | ADMIN_TERMS | RISK_TERMS | INFRA_TERMS):
            meta_class = "CONTEXT_LAYER_METADATA"
        elif supports:
            meta_class = "SPARSE_METADATA_ONLY"
        else:
            meta_class = "NO_METADATA_AVAILABLE"
        rows.append({
            "layer_metadata_id": f"LM_v1uy_{len(rows):04d}",
            "candidate_event_id": probe.get("candidate_event_id", ""),
            "endpoint_probe_id": probe.get("endpoint_probe_id", ""),
            "layer_name_hash": hash_text(layer_name, 24),
            "service_name_hash": hash_text(service_name, 24),
            "geometry_type": geometry,
            "spatial_reference": spatial,
            "extent_hash": hash_text(src.get("extent", ""), 24) if src.get("extent") else "",
            "field_count": str(len(fields)),
            "fields_hash": hash_text("|".join(sorted(fields)), 24),
            "has_date_field": "true" if has_any(fields, DATE_TERMS) else "false",
            "has_hazard_field": "true" if has_any(fields + [layer_name], HAZARD_TERMS) else "false",
            "has_locality_field": "true" if has_any(fields + [layer_name], LOCALITY_TERMS) else "false",
            "has_coordinate_fields": "true" if geometry or has_any(fields, COORD_TERMS) else "false",
            "metadata_class": meta_class,
            **guardrails(),
            "notes": "Layer metadata only; field names are hashed and no features are published.",
        })
    out = dataset_path("v1uy_curitiba_layer_metadata_extraction.csv")
    write_csv(out, METADATA_COLUMNS, rows)
    print(f"[v1uy layer metadata] rows={len(rows)} -> {out}")
    return rows


def run_feature_schema_sampler(args=None):
    probes = load_csv(dataset_path("v1uy_curitiba_geodata_endpoint_probe.csv")) or run_geodata_endpoint_probe(args)
    meta = {r.get("endpoint_probe_id"): r for r in load_csv(dataset_path("v1uy_curitiba_layer_metadata_extraction.csv")) or run_layer_metadata_extractor(args)}
    os.makedirs(STAGING_DIR, exist_ok=True)
    rows = []
    for probe in probes:
        m = meta.get(probe.get("endpoint_probe_id"), {})
        can_query = probe.get("supports_feature_query") == "true" or probe.get("supports_count_query") == "true"
        sample_status = "SCHEMA_ONLY_NO_FEATURE_DOWNLOAD"
        sampled = "0"
        local_hash = ""
        if can_query and m:
            safe_doc = {
                "endpoint_probe_id": probe.get("endpoint_probe_id"),
                "fields_hash": m.get("fields_hash", ""),
                "geometry_redacted": True,
            }
            local_name = f"{probe.get('endpoint_probe_id')}_schema_sample.json"
            local_path = os.path.join(STAGING_DIR, local_name)
            write_text(local_path, [json.dumps(safe_doc, sort_keys=True)])
            local_hash = hash_text(local_name + m.get("fields_hash", ""), 24)
            sample_status = "LOCAL_REDACTED_SCHEMA_SAMPLE"
        elif m:
            sample_status = "METADATA_FIELDS_ONLY"
        rows.append({
            "feature_sample_id": f"FS_v1uy_{len(rows):04d}",
            "candidate_event_id": probe.get("candidate_event_id", ""),
            "endpoint_probe_id": probe.get("endpoint_probe_id", ""),
            "sample_status": sample_status,
            "feature_count": probe.get("feature_count_hint", ""),
            "sampled_feature_count": sampled,
            "sample_local_hash": local_hash,
            "has_geometry": "true" if m.get("has_coordinate_fields") == "true" else "false",
            "has_date_values": "false",
            "has_hazard_values": "false",
            "has_locality_values": "false",
            "geometry_sample_versioned": "false",
            "raw_feature_versioned": "false",
            **guardrails(),
            "notes": "No complete feature values or geometry are versioned.",
        })
    out = dataset_path("v1uy_curitiba_feature_schema_sampling.csv")
    write_csv(out, SAMPLE_COLUMNS, rows)
    print(f"[v1uy schema sample] rows={len(rows)} -> {out}")
    return rows


def class_from_text(text, has_date, has_hazard, has_locality):
    t = token(text)
    if has_date and has_hazard and has_locality and any(x in t for x in ["ocorr", "atendimento", "defesa", "alag", "inund"]):
        return "EVENT_SPECIFIC_OCCURRENCE_LAYER_CANDIDATE"
    if any(x in t for x in ["ocorr", "atendimento", "defesa", "alag", "inund"]):
        return "POSSIBLE_OCCURRENCE_LAYER"
    if any(x in t for x in ["drenagem", "bacia", "hidrograf", "rio", "canal"]):
        return "DRAINAGE_CONTEXT_LAYER"
    if any(x in t for x in ["risco", "suscet", "vulnerab"]):
        return "RISK_OR_SUSCEPTIBILITY_CONTEXT_LAYER"
    if any(x in t for x in ["bairro", "regional", "administr", "limite"]):
        return "ADMINISTRATIVE_CONTEXT_LAYER"
    if any(x in t for x in ["infra", "equip", "rede", "obra"]):
        return "INFRASTRUCTURE_CONTEXT_LAYER"
    if any(x in t for x in ["chuva", "precipit", "hidro", "estacao"]):
        return "HYDROMET_CONTEXT_LAYER"
    return "UNKNOWN_CONTEXT_LAYER"


def run_context_layer_classifier(args=None):
    meta_rows = load_csv(dataset_path("v1uy_curitiba_layer_metadata_extraction.csv")) or run_layer_metadata_extractor(args)
    probes = {r.get("endpoint_probe_id"): r for r in load_csv(dataset_path("v1uy_curitiba_geodata_endpoint_probe.csv"))}
    lookup = source_lookup()
    rows = []
    for m in meta_rows:
        probe = probes.get(m.get("endpoint_probe_id"), {})
        src = lookup.get(probe.get("source_record_id"), {})
        text = " ".join([
            src.get("layer_name", ""), src.get("layer_id", ""),
            src.get("layer_class", ""), src.get("resource_class", ""),
            m.get("metadata_class", ""),
        ])
        has_date = m.get("has_date_field") == "true"
        has_hazard = m.get("has_hazard_field") == "true"
        has_locality = m.get("has_locality_field") == "true"
        layer_class = class_from_text(text, has_date, has_hazard, has_locality)
        occurrence = layer_class in {"POSSIBLE_OCCURRENCE_LAYER", "EVENT_SPECIFIC_OCCURRENCE_LAYER_CANDIDATE"}
        rows.append({
            "layer_classification_id": f"LC_v1uy_{len(rows):04d}",
            "candidate_event_id": m.get("candidate_event_id", ""),
            "endpoint_probe_id": m.get("endpoint_probe_id", ""),
            "layer_metadata_id": m.get("layer_metadata_id", ""),
            "layer_class": layer_class,
            "event_specificity": "POSSIBLE_EVENT_LAYER_NEEDS_CONTROLLED_DOWNLOAD" if occurrence else "CONTEXT_LAYER_NOT_EVENT",
            "has_geometry": "true" if m.get("has_coordinate_fields") == "true" else "false",
            "has_date_field": m.get("has_date_field", "false"),
            "has_hazard_field": m.get("has_hazard_field", "false"),
            "has_locality_field": m.get("has_locality_field", "false"),
            "context_only_status": "NOT_CONTEXT_ONLY_NEEDS_AUDIT" if occurrence else "CONTEXT_ONLY",
            "can_support_contextual_review": "true",
            "can_support_observed_occurrence": "true" if occurrence else "false",
            "can_create_ground_reference": "false",
            **{k: v for k, v in guardrails().items() if k != "can_create_ground_reference"},
            "notes": "Context layers are not promoted to observed occurrence or ground reference.",
        })
    out = dataset_path("v1uy_curitiba_context_layer_classification.csv")
    write_csv(out, CLASSIFICATION_COLUMNS, rows)
    print(f"[v1uy layer class] rows={len(rows)} -> {out}")
    return rows


def run_possible_occurrence_layer_audit(args=None):
    classes = load_csv(dataset_path("v1uy_curitiba_context_layer_classification.csv")) or run_context_layer_classifier(args)
    rows = []
    candidates = [r for r in classes if r.get("layer_class") in {"POSSIBLE_OCCURRENCE_LAYER", "EVENT_SPECIFIC_OCCURRENCE_LAYER_CANDIDATE"}]
    for row in candidates:
        date = row.get("has_date_field") == "true"
        hazard = row.get("has_hazard_field") == "true"
        locality = row.get("has_locality_field") == "true"
        geometry = row.get("has_geometry") == "true"
        complete = date and hazard and locality and geometry and row.get("context_only_status") != "CONTEXT_ONLY"
        rows.append({
            "occurrence_layer_audit_id": f"OA_v1uy_{len(rows):04d}",
            "candidate_event_id": row.get("candidate_event_id", ""),
            "layer_classification_id": row.get("layer_classification_id", ""),
            "official_source_support": "PUBLIC_OFFICIAL_SOURCE",
            "date_support": "PASS" if date else "BLOCKED_NO_DATE_FIELD",
            "hazard_support": "PASS" if hazard else "BLOCKED_NO_HAZARD_FIELD",
            "locality_support": "PASS" if locality else "BLOCKED_NO_LOCALITY_FIELD",
            "geometry_support": "PASS" if geometry else "BLOCKED_NO_GEOMETRY_METADATA",
            "event_specificity": row.get("event_specificity", ""),
            "context_only_status": row.get("context_only_status", ""),
            "can_advance_to_controlled_download": "true" if complete else "false",
            "can_advance_to_overlay_preflight": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "blocker": "" if complete else "missing_required_occurrence_layer_gate",
            "required_next_action": "v1uz - Curitiba Controlled Feature Download and Occurrence Gate Audit" if complete else "v1uz - Hold Curitiba Context Only",
            **{k: v for k, v in guardrails().items() if k not in {"can_create_ground_reference", "can_create_training_label"}},
            "notes": "Occurrence-layer audit is candidate-only; no overlay or label.",
        })
    out = dataset_path("v1uy_curitiba_possible_occurrence_layer_audit.csv")
    write_csv(out, OCCURRENCE_COLUMNS, rows)
    print(f"[v1uy occurrence audit] rows={len(rows)} -> {out}")
    return rows


def run_controlled_feature_download_planner(args=None):
    audits = load_csv(dataset_path("v1uy_curitiba_possible_occurrence_layer_audit.csv")) or run_possible_occurrence_layer_audit(args)
    classes = {r.get("layer_classification_id"): r for r in load_csv(dataset_path("v1uy_curitiba_context_layer_classification.csv"))}
    rows = []
    approved = [r for r in audits if r.get("can_advance_to_controlled_download") == "true"]
    if approved:
        for audit in approved:
            cls = classes.get(audit.get("layer_classification_id"), {})
            rows.append({
                "download_plan_id": f"DP_v1uy_{len(rows):04d}",
                "candidate_event_id": audit.get("candidate_event_id", ""),
                "endpoint_probe_id": cls.get("endpoint_probe_id", ""),
                "layer_class": cls.get("layer_class", ""),
                "plan_status": "CONTROLLED_DOWNLOAD_CANDIDATE_FOR_V1UZ",
                "recommended_filter": "event_date_window_and_hazard_terms_required",
                "max_features": "100",
                "allowed_fields_hash": hash_text("date|hazard|locality|objectid", 24),
                "requires_redaction": "true",
                "overclaim_risk": "HIGH_UNTIL_CONTROLLED_AUDIT",
                "can_execute_now": "false",
                "recommended_next_version": "v1uz",
                **guardrails(),
                "notes": "Planning only; no feature download executed.",
            })
    else:
        rows.append({
            "download_plan_id": "DP_v1uy_0000",
            "candidate_event_id": candidate_event_id(),
            "endpoint_probe_id": "",
            "layer_class": "NO_OCCURRENCE_LAYER_READY",
            "plan_status": "NO_CONTROLLED_DOWNLOAD_RECOMMENDED",
            "recommended_filter": "",
            "max_features": "0",
            "allowed_fields_hash": "",
            "requires_redaction": "false",
            "overclaim_risk": "OVERCLAIM_BLOCKED_BY_CONTEXT_ONLY_EVIDENCE",
            "can_execute_now": "false",
            "recommended_next_version": "v1uz",
            **guardrails(),
            "notes": "No occurrence layer passed gates; keep Curitiba context-only.",
        })
    out = dataset_path("v1uy_curitiba_controlled_feature_download_plan.csv")
    write_csv(out, PLAN_COLUMNS, rows)
    print(f"[v1uy download plan] rows={len(rows)} -> {out}")
    return rows


def run_event_patch_readiness_update(args=None):
    classes = load_csv(dataset_path("v1uy_curitiba_context_layer_classification.csv")) or run_context_layer_classifier(args)
    plans = load_csv(dataset_path("v1uy_curitiba_controlled_feature_download_plan.csv")) or run_controlled_feature_download_planner(args)
    has_candidate = any(p.get("plan_status") == "CONTROLLED_DOWNLOAD_CANDIDATE_FOR_V1UZ" for p in plans)
    has_context = bool(classes)
    dims = [
        ("geodata_deepening_status", "CONTEXT_LAYER_DEEPENED" if has_context else "NO_QUERYABLE_GEODATA"),
        ("controlled_feature_download_status", "CANDIDATE_FOR_V1UZ" if has_candidate else "BLOCKED_NO_OCCURRENCE_LAYER_READY"),
        ("event_patch_linkage_status", "CONTEXTUAL_SUPPORT_IMPROVED" if has_context else "UNCHANGED"),
        ("overlay_readiness", "BLOCKED"),
        ("ground_reference_readiness", "BLOCKED"),
        ("training_readiness", "BLOCKED"),
    ]
    rows = []
    for dim, cls in dims:
        rows.append({
            "readiness_update_id": f"RDY_v1uy_{len(rows):04d}",
            "candidate_event_id": candidate_event_id(),
            "proposed_event_id": proposed_event_id(),
            "dimension": dim,
            "classification": cls,
            "basis": "v1uy public geodata deepening",
            **guardrails(),
            "notes": "Additive readiness update; prior v1us/v1uw/v1ux outputs not modified.",
        })
    out = dataset_path("v1uy_curitiba_event_patch_readiness_update.csv")
    write_csv(out, READINESS_COLUMNS, rows)
    print(f"[v1uy readiness] rows={len(rows)} -> {out}")
    return rows


def run_ground_reference_blocker_builder(args=None):
    blockers = [
        "no_observed_geometry", "no_occurrence_table",
        "no_controlled_feature_download", "context_layer_only",
        "no_overlay", "no_ground_reference", "no_training_label",
        "hydromet_is_not_occurrence", "patch_truth_forbidden",
    ]
    rows = []
    for blocker in blockers:
        rows.append({
            "blocker_id": f"GB_v1uy_{len(rows):04d}",
            "event_id": proposed_event_id(),
            "blocker": blocker,
            "status": "BLOCKED",
            **guardrails(),
            "notes": "v1uy cannot create ground reference without controlled observed occurrence evidence.",
        })
    out = dataset_path("v1uy_curitiba_ground_reference_blocker_matrix.csv")
    write_csv(out, BLOCKER_COLUMNS, rows)
    print(f"[v1uy blockers] rows={len(rows)} -> {out}")
    return rows


def select_next_action(plans, audits, classes):
    if any(p.get("plan_status") == "CONTROLLED_DOWNLOAD_CANDIDATE_FOR_V1UZ" for p in plans):
        return "v1uz - Curitiba Controlled Feature Download and Occurrence Gate Audit"
    if any(r.get("layer_class") != "UNKNOWN_CONTEXT_LAYER" for r in classes):
        return "v1uz - Hold Curitiba Context Only"
    if audits:
        return "v1uz - Multi-Region Registry Hardening"
    return "v1uz - Sentinel Date Recovery for Event-Patch Packages"


def run_completion_report(args=None):
    write_policy_configs()
    mismatches = load_csv(dataset_path("v1uy_curitiba_content_mismatch_resolution.csv")) or run_content_mismatch_resolver(args)
    probes = load_csv(dataset_path("v1uy_curitiba_geodata_endpoint_probe.csv")) or run_geodata_endpoint_probe(args)
    metadata = load_csv(dataset_path("v1uy_curitiba_layer_metadata_extraction.csv")) or run_layer_metadata_extractor(args)
    samples = load_csv(dataset_path("v1uy_curitiba_feature_schema_sampling.csv")) or run_feature_schema_sampler(args)
    classes = load_csv(dataset_path("v1uy_curitiba_context_layer_classification.csv")) or run_context_layer_classifier(args)
    audits = load_csv(dataset_path("v1uy_curitiba_possible_occurrence_layer_audit.csv")) or run_possible_occurrence_layer_audit(args)
    plans = load_csv(dataset_path("v1uy_curitiba_controlled_feature_download_plan.csv")) or run_controlled_feature_download_planner(args)
    readiness = load_csv(dataset_path("v1uy_curitiba_event_patch_readiness_update.csv")) or run_event_patch_readiness_update(args)
    blockers = load_csv(dataset_path("v1uy_curitiba_ground_reference_blocker_matrix.csv")) or run_ground_reference_blocker_builder(args)
    next_action = select_next_action(plans, audits, classes)
    write_csv(dataset_path("v1uy_next_actions_registry.csv"), NEXT_COLUMNS, [{
        "action_id": "NA_v1uy_0000",
        "event_id": proposed_event_id(),
        "action_type": next_action,
        "priority": "1",
        "description": "Selected from v1uy geodata endpoint and layer-classification evidence.",
        "target": "CURITIBA_PUBLIC_GEODATA",
        "status": "RECOMMENDED_NEXT_STEP",
        "notes": "No overlay, labels, ground truth or ground reference.",
    }])
    class_counts = {}
    for row in classes:
        class_counts[row.get("layer_class", "")] = class_counts.get(row.get("layer_class", ""), 0) + 1
    class_summary = "|".join(f"{k}={v}" for k, v in sorted(class_counts.items())) or "none"
    plan_status = "|".join(sorted({p.get("plan_status", "") for p in plans})) or "none"
    lines = [
        "# Protocolo C v1uy - Curitiba Public Geodata Deepening",
        "",
        f"- content mismatches resolved: `{len(mismatches)}`",
        f"- endpoints probed: `{len(probes)}`",
        f"- layer metadata rows: `{len(metadata)}`",
        f"- feature schema samples: `{len(samples)}`",
        f"- layer classes: `{class_summary}`",
        f"- possible occurrence audit rows: `{len(audits)}`",
        f"- controlled download plan status: `{plan_status}`",
        f"- readiness update rows: `{len(readiness)}`",
        f"- blocker rows: `{len(blockers)}`",
        f"- next action: `{next_action}`",
        "",
        "v1uy deepened public geodata metadata and endpoint evidence only. It did not execute overlay, geocoding, centroid use, label creation, ground truth, ground reference, operational validation, or raw data versioning.",
        "",
        "Ground reference remains blocked because v1uy did not establish controlled observed occurrence geometry tied to the Curitiba event candidate.",
    ]
    write_text(doc_path("protocolo_c_v1uy_curitiba_public_geodata_deepening.md"), lines)
    write_text(doc_path("protocolo_c_relatorio_v1uy_curitiba_public_geodata_deepening.md"), lines + [
        "",
        "## Technical result",
        "The current public resources improve contextual understanding but remain insufficient for overlay preflight or ground-reference creation.",
    ])
    write_text(doc_path("protocolo_c_status_atual_v1uy.md"), [
        "# Status atual - Protocolo C v1uy",
        "",
        f"Curitiba status: `{MAX_STATUS}`.",
        f"Recommended next programming step: `{next_action}`.",
        "",
        "Ground truth, ground reference, labels, overlay, coordinates inferred from context, and operational validation remain blocked.",
    ])
    manifest = []
    for idx, artifact in enumerate(V1UY_ARTIFACTS):
        real = artifact_path(artifact)
        if not os.path.exists(real):
            continue
        manifest.append({
            "artifact_id": f"MAN_v1uy_{idx:04d}",
            "artifact_path": artifact.replace("\\", "/"),
            "artifact_type": os.path.splitext(artifact)[1].lstrip(".") or "text",
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha256_file(real)[:16],
            "file_size_bytes": str(os.path.getsize(real)),
            "is_versionable": "true",
            "reason": "v1uy metadata-only artifact; no raw private path.",
        })
    write_csv(dataset_path("v1uy_versionable_artifacts_manifest.csv"), MANIFEST_COLUMNS, manifest)
    for folder in (RAW_DIR, STAGING_DIR, QUARANTINE_DIR, REPORTS_DIR):
        os.makedirs(folder, exist_ok=True)
    print(f"[v1uy completion] probes={len(probes)} classes={len(classes)} next={next_action}")
    return {"probes": len(probes), "classes": len(classes), "next_action": next_action}


def run_all(args=None):
    args = args or parse_args([])
    run_content_mismatch_resolver(args)
    run_geodata_endpoint_probe(args)
    run_layer_metadata_extractor(args)
    run_feature_schema_sampler(args)
    run_context_layer_classifier(args)
    run_possible_occurrence_layer_audit(args)
    run_controlled_feature_download_planner(args)
    run_event_patch_readiness_update(args)
    run_ground_reference_blocker_builder(args)
    return run_completion_report(args)
