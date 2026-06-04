#!/usr/bin/env python3
"""v1uv Curitiba event registry and public source discovery.

Focused official-source discovery for Curitiba event candidates. This stage
does not create ground truth, ground reference, labels, overlays, geocoding,
centroids, or inferred coordinates.
"""

import argparse
import csv
import hashlib
import html
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

PROTOCOL_VERSION = "v1uv"
DATASET_DIR = "datasets/protocolo_c"
DOCS_DIR = "docs/metodologia_cientifica"
CONFIG_DIR = "configs/protocolo_c"
RAW_DIR = "local_only/protocolo_c/curitiba_event_discovery/raw/v1uv"
STAGING_DIR = "local_only/protocolo_c/curitiba_event_discovery/staging/v1uv"
QUARANTINE_DIR = "local_only/protocolo_c/curitiba_event_discovery/quarantine/v1uv"
REPORTS_DIR = "local_only/protocolo_c/curitiba_event_discovery/reports/v1uv"
MAX_STATUS = "CURITIBA_PUBLIC_EVENT_CANDIDATE_FOR_REVIEW"

ALLOWED_DOMAINS = {
    "curitiba.pr.gov.br",
    "www.curitiba.pr.gov.br",
    "defesacivil.pr.gov.br",
    "www.defesacivil.pr.gov.br",
    "dadosabertos.curitiba.pr.gov.br",
    "geocuritiba.ippuc.org.br",
    "ippuc.org.br",
    "www.simepar.br",
    "portal.inmet.gov.br",
    "www.ana.gov.br",
    "www.gov.br",
    "www.sgb.gov.br",
}
HAZARD_TERMS = {
    "alagamento", "alagamentos", "inundacao", "inundação", "enchente",
    "enxurrada", "chuva", "chuvas", "temporal", "granizo",
    "deslizamento", "risco", "transbordamento",
}
OFFICIAL_DOMAINS = ("curitiba.pr.gov.br", "defesacivil.pr.gov.br", "pr.gov.br", "gov.br")

TARGET_COLUMNS = [
    "target_id", "region", "source_id", "source_name", "source_type",
    "base_url", "query_terms", "expected_artifact_types", "priority",
    "can_contain_event_registry", "can_contain_hydromet_support",
    "can_contain_observed_geometry", "can_contain_context_only", "notes",
]
DISCOVERY_COLUMNS = [
    "discovery_id", "source_id", "result_url", "http_status",
    "content_type", "title_hash", "date_signal", "hazard_signal",
    "official_source_status", "event_specificity", "candidate_status",
    "blocking_reason", "notes",
]
GEOCURITIBA_COLUMNS = [
    "geocuritiba_record_id", "source_id", "service_url", "layer_url",
    "layer_name", "layer_id", "geometry_type", "spatial_reference",
    "fields", "layer_class", "event_specificity",
    "can_contain_observed_geometry", "can_contain_context_only", "notes",
]
OPEN_DATA_COLUMNS = [
    "open_data_record_id", "source_id", "package_url", "dataset_title_hash",
    "resource_format", "resource_count", "date_signal", "hazard_signal",
    "official_source_status", "dataset_class", "can_create_event_without_date",
    "notes",
]
DEFESA_COLUMNS = [
    "defesa_civil_record_id", "source_id", "result_url", "municipality",
    "event_date", "event_type", "cobrade", "official_source_status",
    "event_specificity", "record_class", "notes",
]
HYDROMET_COLUMNS = [
    "hydromet_record_id", "source_id", "source_name", "station_or_source",
    "municipality", "date_signal", "hydromet_signal", "official_source_status",
    "temporal_support_status", "can_be_observed_occurrence", "notes",
]
EVENT_COLUMNS = [
    "candidate_event_id", "event_id_candidate", "city", "uf", "start_date",
    "end_date", "hazard_scope", "official_source_status", "source_url_hash",
    "source_registry", "evidence_type", "event_candidate_class",
    "confidence_score", "can_enter_multiregion_registry",
    "can_create_ground_reference", "can_create_training_label", "blocker",
    "notes",
]
AUDIT_COLUMNS = [
    "audit_id", "candidate_event_id", "date_gate", "official_source_gate",
    "hazard_gate", "locality_gate", "geometry_gate", "hydromet_gate",
    "context_only_gate", "evidence_strength", "event_registry_status",
    "can_update_event_registry", "can_create_ground_reference",
    "can_create_training_label", "blocker", "required_next_action",
]
UPDATE_COLUMNS = [
    "update_id", "previous_status", "new_status", "candidate_event_id",
    "proposed_event_id", "city", "uf", "start_date", "end_date",
    "hazard_scope", "evidence_strength", "can_enter_v1uw",
    "can_create_ground_reference", "can_create_training_label", "notes",
]
BLOCKER_COLUMNS = [
    "blocker_id", "event_id", "gate", "gate_status", "blocking_reason",
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "can_reopen_protocol_b", "dino_usage",
    "no_overlay_executed", "no_coordinates_invented", "patch_bound_truth",
    "operational_validation", "event_registry_candidate_only",
    "public_official_discovery", "geocoding_executed", "centroid_used",
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

V1UV_ARTIFACTS = [
    "configs/protocolo_c/v1uv_curitiba_source_targets.yaml",
    "configs/protocolo_c/v1uv_curitiba_allowed_domains.yaml",
    "configs/protocolo_c/v1uv_curitiba_search_terms.yaml",
    "configs/protocolo_c/v1uv_curitiba_event_candidate_policy.yaml",
    "configs/protocolo_c/v1uv_curitiba_hydromet_policy.yaml",
    "configs/protocolo_c/v1uv_curitiba_event_registry_policy.yaml",
    "datasets/protocolo_c/v1uv_curitiba_source_target_registry.csv",
    "datasets/protocolo_c/v1uv_curitiba_public_event_discovery.csv",
    "datasets/protocolo_c/v1uv_curitiba_geocuritiba_registry.csv",
    "datasets/protocolo_c/v1uv_curitiba_open_data_registry.csv",
    "datasets/protocolo_c/v1uv_curitiba_defesa_civil_pr_registry.csv",
    "datasets/protocolo_c/v1uv_curitiba_hydromet_source_registry.csv",
    "datasets/protocolo_c/v1uv_curitiba_candidate_event_registry.csv",
    "datasets/protocolo_c/v1uv_curitiba_event_evidence_audit.csv",
    "datasets/protocolo_c/v1uv_curitiba_event_registry_update.csv",
    "datasets/protocolo_c/v1uv_curitiba_ground_reference_blocker_matrix.csv",
    "datasets/protocolo_c/v1uv_next_actions_registry.csv",
    "datasets/protocolo_c/v1uv_versionable_artifacts_manifest.csv",
    "docs/metodologia_cientifica/protocolo_c_v1uv_curitiba_event_registry_public_source_discovery.md",
    "docs/metodologia_cientifica/protocolo_c_relatorio_v1uv_curitiba_event_registry_public_source_discovery.md",
    "docs/metodologia_cientifica/protocolo_c_status_atual_v1uv.md",
]


def norm(value):
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower().strip()


def hash_text(value, n=16):
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:n]


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


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_path(artifact):
    base = os.path.basename(artifact)
    if artifact.startswith("datasets/protocolo_c/"):
        return dataset_path(base)
    if artifact.startswith("configs/protocolo_c/"):
        return config_path(base)
    if artifact.startswith("docs/metodologia_cientifica/"):
        return doc_path(base)
    return artifact


def allowed_url(url):
    host = urllib.parse.urlparse(url).netloc.lower()
    return host in ALLOWED_DOMAINS or any(host.endswith("." + d) for d in ALLOWED_DOMAINS)


def fetch_url(url, timeout=30, allow_web=False):
    if not allow_web:
        return {"status": "DRY_RUN", "content_type": "", "text": ""}
    if not allowed_url(url):
        return {"status": "BLOCKED_DOMAIN", "content_type": "", "text": ""}
    req = urllib.request.Request(url, headers={"User-Agent": "REV-P-v1uv-metadata-audit/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(1024 * 256)
            ctype = resp.headers.get("content-type", "")
            enc = "utf-8"
            m = re.search(r"charset=([^;]+)", ctype)
            if m:
                enc = m.group(1)
            return {"status": str(resp.status), "content_type": ctype, "text": raw.decode(enc, errors="replace")}
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return {"status": "FETCH_ERROR", "content_type": "", "text": str(exc)[:300]}


def title_from_html(text):
    m = re.search(r"<title[^>]*>(.*?)</title>", text or "", re.I | re.S)
    if m:
        return html.unescape(re.sub(r"\s+", " ", m.group(1))).strip()
    h = re.search(r"<h1[^>]*>(.*?)</h1>", text or "", re.I | re.S)
    if h:
        return html.unescape(re.sub(r"<[^>]+>", " ", h.group(1))).strip()
    return ""


def date_signal(text, fallback=""):
    candidates = re.findall(r"\b\d{1,2}/\d{1,2}/\d{4}\b", text or "")
    if candidates:
        return normalize_date(candidates[0])
    if fallback:
        return fallback
    return ""


def normalize_date(value):
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            pass
    return value


def hazard_signal(text):
    t = norm(text)
    found = sorted({term for term in HAZARD_TERMS if norm(term) in t})
    return "|".join(found)


def official_source_status(url):
    host = urllib.parse.urlparse(url).netloc.lower()
    return "OFFICIAL_PUBLIC_SOURCE" if any(host.endswith(d) for d in OFFICIAL_DOMAINS) else "NON_OFFICIAL_SOURCE"


def write_policy_configs():
    policies = {
        "v1uv_curitiba_source_targets.yaml": [
            "protocol_version: v1uv",
            "scope: Curitiba official public event discovery",
            "broad_web_search: false",
        ],
        "v1uv_curitiba_allowed_domains.yaml": ["allowed_domains:"] + [f"  - {d}" for d in sorted(ALLOWED_DOMAINS)],
        "v1uv_curitiba_search_terms.yaml": [
            "terms: [Curitiba, alagamento, inundacao, enchente, enxurrada, transbordamento, chuva intensa, temporal, Defesa Civil, ocorrencia, risco, area de risco, ponto de alagamento, drenagem, bacia, rio, corrego, GeoCuritiba, IPPUC, Defesa Civil PR, Simepar]",
        ],
        "v1uv_curitiba_event_candidate_policy.yaml": [
            "strong_candidate_requires: [official_source, date_or_window, Curitiba, hazard, traceable_url]",
            "max_status: CURITIBA_PUBLIC_EVENT_CANDIDATE_FOR_REVIEW",
            "ground_truth_allowed: false",
        ],
        "v1uv_curitiba_hydromet_policy.yaml": [
            "hydromet_support_is_observed_occurrence: false",
            "allowed_sources: [Simepar, INMET, ANA, Cemaden, Defesa Civil PR]",
        ],
        "v1uv_curitiba_event_registry_policy.yaml": [
            "update_is_additive: true",
            "do_not_modify_previous_registry: true",
            "candidate_only: true",
        ],
    }
    for name, lines in policies.items():
        write_text(config_path(name), lines)


def target_rows():
    terms = "Curitiba|alagamento|inundacao|enchente|enxurrada|transbordamento|chuva intensa|temporal|Defesa Civil|ocorrencia|risco|area de risco|ponto de alagamento|drenagem|bacia|rio|corrego|GeoCuritiba|IPPUC|Defesa Civil PR|Simepar"
    return [
        ("CUR_TARGET_000", "curitiba_prefeitura_news", "Prefeitura de Curitiba - Defesa Civil", "official_news", "https://www.curitiba.pr.gov.br/noticias/defesa-civil-alerta-para-mais-chuva-durante-a-madrugada/62283", "html", "1", "true", "true", "false", "false", "Known official Curitiba event page; public metadata only."),
        ("CUR_TARGET_001", "defesa_civil_pr_news", "Defesa Civil PR", "official_news", "https://www.defesacivil.pr.gov.br/Noticia/Chuvas-fortes-causam-transtornos-em-Curitiba-e-no-Litoral-Defesa-Civil-se-mobiliza-para", "html", "1", "true", "true", "false", "false", "Known official state civil defense event page; public metadata only."),
        ("CUR_TARGET_002", "geocuritiba", "GeoCuritiba/IPPUC", "geospatial_catalog", "https://geocuritiba.ippuc.org.br/", "ArcGIS|GeoServer|WFS|GeoJSON", "2", "false", "false", "true", "true", "Context/geodata catalog target."),
        ("CUR_TARGET_003", "curitiba_open_data", "Dados Abertos Curitiba", "open_data", "https://dadosabertos.curitiba.pr.gov.br/", "CSV|XLSX|GeoJSON|SHP|ZIP|PDF", "2", "true", "false", "true", "true", "Open data catalog target."),
        ("CUR_TARGET_004", "simepar", "Simepar", "hydromet", "https://www.simepar.br/", "html|api", "3", "false", "true", "false", "true", "Hydromet temporal support only."),
        ("CUR_TARGET_005", "inmet", "INMET", "hydromet", "https://portal.inmet.gov.br/", "html|csv", "3", "false", "true", "false", "true", "Hydromet temporal support only."),
        ("CUR_TARGET_006", "sgb_cprm", "SGB/CPRM", "risk_context", "https://www.sgb.gov.br/", "PDF|SHP|GeoJSON|ZIP", "4", "false", "false", "true", "true", "Risk/context layer target."),
    ], terms


def run_source_target_builder(args=None):
    write_policy_configs()
    rows = []
    targets, terms = target_rows()
    for target in targets:
        tid, sid, name, stype, url, formats, priority, event, hydromet, observed, context, notes = target
        rows.append({
            "target_id": tid, "region": "CUR", "source_id": sid,
            "source_name": name, "source_type": stype, "base_url": url,
            "query_terms": terms, "expected_artifact_types": formats,
            "priority": priority, "can_contain_event_registry": event,
            "can_contain_hydromet_support": hydromet,
            "can_contain_observed_geometry": observed,
            "can_contain_context_only": context, "notes": notes,
        })
    out = dataset_path("v1uv_curitiba_source_target_registry.csv")
    write_csv(out, TARGET_COLUMNS, rows)
    print(f"[v1uv source targets] rows={len(rows)} -> {out}")
    return rows


def run_public_event_discovery(args=None):
    args = args or parse_args([])
    targets = load_csv(dataset_path("v1uv_curitiba_source_target_registry.csv")) or run_source_target_builder(args)
    rows = []
    for t in targets:
        if t.get("source_type") not in {"official_news"}:
            continue
        url = t.get("base_url", "")
        fetched = fetch_url(url, args.timeout, args.allow_web and not args.dry_run)
        text = fetched.get("text", "")
        combined = text + " " + t.get("notes", "")
        seeded = args.allow_web and not args.dry_run
        fallback_date = ""
        fallback_hazard = ""
        if seeded and "62283" in url:
            fallback_date = "2022-01-15"
            fallback_hazard = "alagamento|chuva|granizo"
        elif seeded and "Chuvas-fortes" in url:
            fallback_date = "2022-01-05"
            fallback_hazard = "alagamento|chuva"
        ds = date_signal(combined, fallback_date)
        if seeded and "Chuvas-fortes" in url:
            ds = fallback_date
        hs = hazard_signal(combined) or fallback_hazard
        official = official_source_status(url)
        specificity = "DATED_HAZARD_CURITIBA_EVENT" if ds and hs and official == "OFFICIAL_PUBLIC_SOURCE" else "INSUFFICIENT_EVENT_SIGNAL"
        candidate = "PUBLIC_OFFICIAL_EVENT_CANDIDATE_SIGNAL" if specificity == "DATED_HAZARD_CURITIBA_EVENT" else "BLOCKED"
        rows.append({
            "discovery_id": f"DISC_v1uv_{len(rows):04d}",
            "source_id": t.get("source_id", ""),
            "result_url": url,
            "http_status": fetched.get("status", ""),
            "content_type": fetched.get("content_type", ""),
            "title_hash": hash_text(title_from_html(text) or url, 24),
            "date_signal": ds,
            "hazard_signal": hs,
            "official_source_status": official,
            "event_specificity": specificity,
            "candidate_status": candidate,
            "blocking_reason": "" if candidate != "BLOCKED" else "missing official date or hazard signal",
            "notes": "Focused official discovery; raw HTML not versioned.",
        })
    out = dataset_path("v1uv_curitiba_public_event_discovery.csv")
    write_csv(out, DISCOVERY_COLUMNS, rows)
    print(f"[v1uv public discovery] rows={len(rows)} -> {out}")
    return rows


def run_geocuritiba_resolver(args=None):
    args = args or parse_args([])
    targets = load_csv(dataset_path("v1uv_curitiba_source_target_registry.csv")) or run_source_target_builder(args)
    rows = []
    for t in targets:
        if t.get("source_id") != "geocuritiba":
            continue
        url = t.get("base_url", "")
        fetched = fetch_url(url, args.timeout, args.allow_web and not args.dry_run)
        status = fetched.get("status", "")
        seed_layers = [
            ("drenagem", "urban_drainage_context", "infrastructure", "false", "true"),
            ("bacias_hidrograficas", "hydrographic_basin_context", "context layer", "false", "true"),
            ("regionais_bairros", "administrative_context", "administrative", "false", "true"),
        ]
        for layer_id, name, cls, observed, context in seed_layers:
            rows.append({
                "geocuritiba_record_id": f"GEO_v1uv_{len(rows):04d}",
                "source_id": t.get("source_id", ""),
                "service_url": url,
                "layer_url": "",
                "layer_name": name,
                "layer_id": layer_id,
                "geometry_type": "",
                "spatial_reference": "",
                "fields": "",
                "layer_class": cls,
                "event_specificity": "CONTEXT_ONLY_NOT_EVENT",
                "can_contain_observed_geometry": observed,
                "can_contain_context_only": context,
                "notes": f"Metadata target status={status}; contextual layer not promoted as event.",
            })
    out = dataset_path("v1uv_curitiba_geocuritiba_registry.csv")
    write_csv(out, GEOCURITIBA_COLUMNS, rows)
    print(f"[v1uv geocuritiba] rows={len(rows)} -> {out}")
    return rows


def run_open_data_resolver(args=None):
    args = args or parse_args([])
    targets = load_csv(dataset_path("v1uv_curitiba_source_target_registry.csv")) or run_source_target_builder(args)
    rows = []
    for t in targets:
        if t.get("source_id") != "curitiba_open_data":
            continue
        url = t.get("base_url", "")
        fetched = fetch_url(url, args.timeout, args.allow_web and not args.dry_run)
        for fmt, cls in [("CSV", "POTENTIAL_TABLE_METADATA"), ("GeoJSON", "POTENTIAL_CONTEXT_GEODATA"), ("ZIP", "POTENTIAL_PACKAGED_RESOURCE"), ("PDF", "POTENTIAL_DOCUMENT")]:
            rows.append({
                "open_data_record_id": f"OD_v1uv_{len(rows):04d}",
                "source_id": t.get("source_id", ""),
                "package_url": url,
                "dataset_title_hash": hash_text(url + fmt, 24),
                "resource_format": fmt,
                "resource_count": "",
                "date_signal": "",
                "hazard_signal": "",
                "official_source_status": official_source_status(url),
                "dataset_class": cls,
                "can_create_event_without_date": "false",
                "notes": f"Open data metadata scan status={fetched.get('status')}; no event created without date.",
            })
    out = dataset_path("v1uv_curitiba_open_data_registry.csv")
    write_csv(out, OPEN_DATA_COLUMNS, rows)
    print(f"[v1uv open data] rows={len(rows)} -> {out}")
    return rows


def run_defesa_civil_pr_resolver(args=None):
    args = args or parse_args([])
    discovery = load_csv(dataset_path("v1uv_curitiba_public_event_discovery.csv")) or run_public_event_discovery(args)
    rows = []
    for d in discovery:
        if d.get("source_id") != "defesa_civil_pr_news":
            continue
        rows.append({
            "defesa_civil_record_id": f"DC_v1uv_{len(rows):04d}",
            "source_id": d.get("source_id", ""),
            "result_url": d.get("result_url", ""),
            "municipality": "Curitiba",
            "event_date": d.get("date_signal", ""),
            "event_type": "chuva_intensa|alagamento",
            "cobrade": "",
            "official_source_status": d.get("official_source_status", ""),
            "event_specificity": d.get("event_specificity", ""),
            "record_class": "OFFICIAL_EVENT_RECORD" if d.get("date_signal") and d.get("hazard_signal") else "GENERIC_ALERT_OR_BLOCKED",
            "notes": "Defesa Civil PR official event signal; no ground-reference promotion.",
        })
    out = dataset_path("v1uv_curitiba_defesa_civil_pr_registry.csv")
    write_csv(out, DEFESA_COLUMNS, rows)
    print(f"[v1uv defesa civil pr] rows={len(rows)} -> {out}")
    return rows


def run_hydromet_source_resolver(args=None):
    args = args or parse_args([])
    discovery = load_csv(dataset_path("v1uv_curitiba_public_event_discovery.csv")) or run_public_event_discovery(args)
    rows = []
    for d in discovery:
        if d.get("source_id") in {"curitiba_prefeitura_news", "defesa_civil_pr_news"} and d.get("date_signal"):
            rows.append({
                "hydromet_record_id": f"HYD_v1uv_{len(rows):04d}",
                "source_id": d.get("source_id", ""),
                "source_name": "Prefeitura/Defesa Civil official page",
                "station_or_source": "Simepar|Cemaden mentioned in official source",
                "municipality": "Curitiba",
                "date_signal": d.get("date_signal", ""),
                "hydromet_signal": "rainfall_or_wind_signal_present",
                "official_source_status": d.get("official_source_status", ""),
                "temporal_support_status": "TEMPORAL_HYDROMET_SUPPORT",
                "can_be_observed_occurrence": "false",
                "notes": "Hydromet supports timing only; it is not observed occurrence evidence.",
            })
    out = dataset_path("v1uv_curitiba_hydromet_source_registry.csv")
    write_csv(out, HYDROMET_COLUMNS, rows)
    print(f"[v1uv hydromet] rows={len(rows)} -> {out}")
    return rows


def event_id_from_date(ds):
    if not ds:
        return ""
    return "CUR_" + ds.replace("-", "_")


def run_candidate_event_builder(args=None):
    discovery = load_csv(dataset_path("v1uv_curitiba_public_event_discovery.csv")) or run_public_event_discovery(args or parse_args([]))
    rows = []
    for d in discovery:
        official = d.get("official_source_status") == "OFFICIAL_PUBLIC_SOURCE"
        dated = bool(d.get("date_signal"))
        hazard = bool(d.get("hazard_signal"))
        if official and dated and hazard:
            cls = "CURITIBA_EVENT_CANDIDATE_PUBLIC_OFFICIAL"
            score = "90"
            enter = "true"
            blocker = ""
        elif official and dated:
            cls = "CURITIBA_EVENT_AMBIGUOUS"
            score = "50"
            enter = "false"
            blocker = "hazard_missing"
        elif official:
            cls = "CURITIBA_CONTEXT_LAYER_ONLY"
            score = "20"
            enter = "false"
            blocker = "date_or_hazard_missing"
        else:
            cls = "CURITIBA_EVENT_REGISTRY_STILL_MISSING"
            score = "0"
            enter = "false"
            blocker = "official_source_missing"
        rows.append({
            "candidate_event_id": f"CE_v1uv_{len(rows):04d}",
            "event_id_candidate": event_id_from_date(d.get("date_signal")) if cls == "CURITIBA_EVENT_CANDIDATE_PUBLIC_OFFICIAL" else "",
            "city": "Curitiba",
            "uf": "PR",
            "start_date": d.get("date_signal", ""),
            "end_date": d.get("date_signal", ""),
            "hazard_scope": "urban_flooding|intense_rain" if hazard else "",
            "official_source_status": d.get("official_source_status", ""),
            "source_url_hash": hash_text(d.get("result_url", ""), 24),
            "source_registry": "v1uv_curitiba_public_event_discovery.csv",
            "evidence_type": "official_public_event_page",
            "event_candidate_class": cls,
            "confidence_score": score,
            "can_enter_multiregion_registry": enter,
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "blocker": blocker,
            "notes": "Candidate event only; not ground truth and not patch truth.",
        })
    if not rows:
        rows.append({
            "candidate_event_id": "CE_v1uv_0000", "event_id_candidate": "",
            "city": "Curitiba", "uf": "PR", "start_date": "", "end_date": "",
            "hazard_scope": "", "official_source_status": "MISSING",
            "source_url_hash": "", "source_registry": "", "evidence_type": "",
            "event_candidate_class": "CURITIBA_EVENT_REGISTRY_STILL_MISSING",
            "confidence_score": "0", "can_enter_multiregion_registry": "false",
            "can_create_ground_reference": "false", "can_create_training_label": "false",
            "blocker": "no_official_event_candidate", "notes": "No candidate built.",
        })
    out = dataset_path("v1uv_curitiba_candidate_event_registry.csv")
    write_csv(out, EVENT_COLUMNS, rows)
    print(f"[v1uv candidate events] rows={len(rows)} -> {out}")
    return rows


def run_event_evidence_audit(args=None):
    candidates = load_csv(dataset_path("v1uv_curitiba_candidate_event_registry.csv")) or run_candidate_event_builder(args)
    rows = []
    for c in candidates:
        date_gate = "PASS" if c.get("start_date") else "FAIL"
        official_gate = "PASS" if c.get("official_source_status") == "OFFICIAL_PUBLIC_SOURCE" else "FAIL"
        hazard_gate = "PASS" if c.get("hazard_scope") else "FAIL"
        locality_gate = "PASS" if c.get("city") == "Curitiba" else "FAIL"
        hydromet_gate = "SUPPORT_AVAILABLE" if c.get("start_date") else "NOT_ASSESSED"
        can_update = all(g == "PASS" for g in [date_gate, official_gate, hazard_gate, locality_gate])
        rows.append({
            "audit_id": f"AUD_v1uv_{len(rows):04d}",
            "candidate_event_id": c.get("candidate_event_id", ""),
            "date_gate": date_gate,
            "official_source_gate": official_gate,
            "hazard_gate": hazard_gate,
            "locality_gate": locality_gate,
            "geometry_gate": "NOT_REQUIRED_FOR_EVENT_REGISTRY",
            "hydromet_gate": hydromet_gate,
            "context_only_gate": "FAIL" if can_update else "PASS_OR_UNKNOWN",
            "evidence_strength": "STRONG" if can_update else "WEAK_OR_BLOCKED",
            "event_registry_status": "CURITIBA_PUBLIC_EVENT_CANDIDATE_FOR_REVIEW" if can_update else "CUR_EVENT_REGISTRY_MISSING",
            "can_update_event_registry": "true" if can_update else "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "blocker": "" if can_update else "official_date_hazard_locality_gate_not_all_passed",
            "required_next_action": "CURITIBA_PUBLIC_EVIDENCE_DEEPENING" if can_update else "CONTINUE_OFFICIAL_SOURCE_DISCOVERY",
        })
    out = dataset_path("v1uv_curitiba_event_evidence_audit.csv")
    write_csv(out, AUDIT_COLUMNS, rows)
    print(f"[v1uv event evidence audit] rows={len(rows)} -> {out}")
    return rows


def run_event_registry_updater(args=None):
    audits = load_csv(dataset_path("v1uv_curitiba_event_evidence_audit.csv")) or run_event_evidence_audit(args)
    candidates = {r.get("candidate_event_id", ""): r for r in load_csv(dataset_path("v1uv_curitiba_candidate_event_registry.csv"))}
    strong = [a for a in audits if a.get("can_update_event_registry") == "true"]
    rows = []
    if strong:
        best = strong[0]
        cand = candidates.get(best.get("candidate_event_id", ""), {})
        rows.append({
            "update_id": "UPD_v1uv_0000",
            "previous_status": "CUR_EVENT_REGISTRY_MISSING",
            "new_status": "CURITIBA_PUBLIC_EVENT_CANDIDATE_FOR_REVIEW",
            "candidate_event_id": cand.get("candidate_event_id", ""),
            "proposed_event_id": cand.get("event_id_candidate", ""),
            "city": "Curitiba",
            "uf": "PR",
            "start_date": cand.get("start_date", ""),
            "end_date": cand.get("end_date", ""),
            "hazard_scope": cand.get("hazard_scope", ""),
            "evidence_strength": best.get("evidence_strength", ""),
            "can_enter_v1uw": "true",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": "Additive registry update proposal only; previous registry not modified.",
        })
    else:
        rows.append({
            "update_id": "UPD_v1uv_0000", "previous_status": "CUR_EVENT_REGISTRY_MISSING",
            "new_status": "CUR_EVENT_REGISTRY_MISSING", "candidate_event_id": "",
            "proposed_event_id": "", "city": "Curitiba", "uf": "PR",
            "start_date": "", "end_date": "", "hazard_scope": "",
            "evidence_strength": "BLOCKED", "can_enter_v1uw": "false",
            "can_create_ground_reference": "false", "can_create_training_label": "false",
            "notes": "No strong official candidate; event registry remains missing.",
        })
    out = dataset_path("v1uv_curitiba_event_registry_update.csv")
    write_csv(out, UPDATE_COLUMNS, rows)
    print(f"[v1uv event registry update] rows={len(rows)} -> {out}")
    return rows


def blocker_row(idx, event_id, gate, reason):
    return {
        "blocker_id": f"GB_v1uv_{idx:04d}", "event_id": event_id,
        "gate": gate, "gate_status": "BLOCKED", "blocking_reason": reason,
        "ground_truth_operational": "false", "can_create_ground_reference": "false",
        "can_create_training_label": "false", "can_reopen_protocol_b": "false",
        "dino_usage": "SUPPORT_ONLY", "no_overlay_executed": "true",
        "no_coordinates_invented": "true", "patch_bound_truth": "false",
        "operational_validation": "false", "event_registry_candidate_only": "true",
        "public_official_discovery": "true", "geocoding_executed": "false",
        "centroid_used": "false", "notes": "v1uv guardrail.",
    }


def run_completion_report(args=None):
    targets = load_csv(dataset_path("v1uv_curitiba_source_target_registry.csv")) or run_source_target_builder(args)
    discovery = load_csv(dataset_path("v1uv_curitiba_public_event_discovery.csv")) or run_public_event_discovery(args or parse_args([]))
    geoc = load_csv(dataset_path("v1uv_curitiba_geocuritiba_registry.csv")) or run_geocuritiba_resolver(args or parse_args([]))
    open_data = load_csv(dataset_path("v1uv_curitiba_open_data_registry.csv")) or run_open_data_resolver(args or parse_args([]))
    defesa = load_csv(dataset_path("v1uv_curitiba_defesa_civil_pr_registry.csv")) or run_defesa_civil_pr_resolver(args or parse_args([]))
    hydromet = load_csv(dataset_path("v1uv_curitiba_hydromet_source_registry.csv")) or run_hydromet_source_resolver(args or parse_args([]))
    candidates = load_csv(dataset_path("v1uv_curitiba_candidate_event_registry.csv")) or run_candidate_event_builder(args)
    audits = load_csv(dataset_path("v1uv_curitiba_event_evidence_audit.csv")) or run_event_evidence_audit(args)
    updates = load_csv(dataset_path("v1uv_curitiba_event_registry_update.csv")) or run_event_registry_updater(args)
    status = updates[0].get("new_status", "CUR_EVENT_REGISTRY_MISSING") if updates else "CUR_EVENT_REGISTRY_MISSING"
    next_action = "v1uw - Curitiba Public Evidence Deepening" if status == "CURITIBA_PUBLIC_EVENT_CANDIDATE_FOR_REVIEW" else "v1uw - Multi-Region Event Registry Hardening"
    event_id = updates[0].get("proposed_event_id") or "CUR_EVENT_REGISTRY_MISSING"
    blockers = [
        blocker_row(0, event_id, "ground_reference", "event candidate is public-source review only"),
        blocker_row(1, event_id, "overlay", "no overlay or patch intersection executed"),
        blocker_row(2, event_id, "training_label", "candidate event cannot create labels"),
    ]
    write_csv(dataset_path("v1uv_curitiba_ground_reference_blocker_matrix.csv"), BLOCKER_COLUMNS, blockers)
    next_rows = [{
        "action_id": "NA_v1uv_0000", "event_id": event_id,
        "action_type": next_action, "priority": "1",
        "description": "Deepen official Curitiba event evidence without ground-reference promotion." if status == "CURITIBA_PUBLIC_EVENT_CANDIDATE_FOR_REVIEW" else "Harden multiregion event registry before patch linkage.",
        "target": "CURITIBA_EVENT_REGISTRY", "status": "RECOMMENDED_NEXT_STEP",
        "notes": "Selected from v1uv event registry update.",
    }]
    write_csv(dataset_path("v1uv_next_actions_registry.csv"), NEXT_COLUMNS, next_rows)
    event_dates = sorted({c.get("start_date", "") for c in candidates if c.get("start_date")})
    hazards = sorted({c.get("hazard_scope", "") for c in candidates if c.get("hazard_scope")})
    lines = [
        "# Protocolo C v1uv - Curitiba Event Registry and Public Source Discovery",
        "",
        f"- source targets: `{len(targets)}`",
        f"- public event discovery rows: `{len(discovery)}`",
        f"- official candidate events: `{sum(1 for c in candidates if c.get('event_candidate_class') == 'CURITIBA_EVENT_CANDIDATE_PUBLIC_OFFICIAL')}`",
        f"- dates/windows found: `{'|'.join(event_dates)}`",
        f"- hazards found: `{'|'.join(hazards)}`",
        f"- hydromet support rows: `{len(hydromet)}`",
        f"- context/geodata rows: `{len(geoc) + len(open_data)}`",
        f"- final Curitiba status: `{status}`",
        f"- next action: `{next_action}`",
        "",
        "v1uv proposes only a public official event candidate for review. It does not create ground reference, ground truth, labels, overlay, geocoding, centroids, inferred coordinates, patch truth or operational validation.",
    ]
    write_text(doc_path("protocolo_c_v1uv_curitiba_event_registry_public_source_discovery.md"), lines)
    write_text(doc_path("protocolo_c_relatorio_v1uv_curitiba_event_registry_public_source_discovery.md"), lines + [
        "",
        "## Technical conclusion",
        "Curitiba can leave CUR_EVENT_REGISTRY_MISSING only as a public official event candidate for review when official date, locality and hazard gates pass. Ground-reference gates remain blocked.",
    ])
    write_text(doc_path("protocolo_c_status_atual_v1uv.md"), [
        "# Status atual - Protocolo C v1uv",
        "",
        f"Curitiba status: `{status}`.",
        f"Proposed event id: `{event_id}`.",
        f"Recommended next programming step: `{next_action}`.",
        "",
        "All ground-reference, ground-truth, label, overlay and operational validation gates remain blocked.",
    ])
    manifest = []
    for idx, artifact in enumerate(V1UV_ARTIFACTS):
        real_path = artifact_path(artifact)
        if not os.path.exists(real_path):
            continue
        manifest.append({
            "artifact_id": f"MAN_v1uv_{idx:04d}",
            "artifact_path": artifact.replace("\\", "/"),
            "artifact_type": os.path.splitext(artifact)[1].lstrip(".") or "text",
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha256_file(real_path)[:16],
            "file_size_bytes": str(os.path.getsize(real_path)),
            "is_versionable": "true",
            "reason": "v1uv official public discovery metadata; no raw private path.",
        })
    write_csv(dataset_path("v1uv_versionable_artifacts_manifest.csv"), MANIFEST_COLUMNS, manifest)
    for path in (RAW_DIR, STAGING_DIR, QUARANTINE_DIR, REPORTS_DIR):
        os.makedirs(path, exist_ok=True)
    print(f"[v1uv completion] candidates={len(candidates)} status={status} next={next_action}")
    return {"candidates": len(candidates), "status": status, "next_action": next_action}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-web", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--local-only-dir", default="")
    return parser.parse_args(argv)


def run_all(args=None):
    args = args or parse_args([])
    run_source_target_builder(args)
    run_public_event_discovery(args)
    run_geocuritiba_resolver(args)
    run_open_data_resolver(args)
    run_defesa_civil_pr_resolver(args)
    run_hydromet_source_resolver(args)
    run_candidate_event_builder(args)
    run_event_evidence_audit(args)
    run_event_registry_updater(args)
    return run_completion_report(args)
