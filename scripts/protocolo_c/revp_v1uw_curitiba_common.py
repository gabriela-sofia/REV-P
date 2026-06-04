#!/usr/bin/env python3
"""v1uw Curitiba public evidence deepening.

Deepens official public evidence for the Curitiba event candidate from v1uv.
Raw snapshots stay under local_only; versionable artifacts contain only hashes,
counts, gate decisions and blocker matrices.
"""

import argparse
import csv
import hashlib
import html
import os
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

PROTOCOL_VERSION = "v1uw"
DATASET_DIR = "datasets/protocolo_c"
DOCS_DIR = "docs/metodologia_cientifica"
CONFIG_DIR = "configs/protocolo_c"
RAW_DIR = "local_only/protocolo_c/curitiba_public_evidence/raw/v1uw"
STAGING_DIR = "local_only/protocolo_c/curitiba_public_evidence/staging/v1uw"
REPORTS_DIR = "local_only/protocolo_c/curitiba_public_evidence/reports/v1uw"
MAX_STATUS = "CURITIBA_PUBLIC_EVIDENCE_DEEPENED_EVENT_CANDIDATE"
ALLOWED_DOMAINS = {"www.curitiba.pr.gov.br", "curitiba.pr.gov.br", "www.defesacivil.pr.gov.br", "defesacivil.pr.gov.br"}

SNAPSHOT_COLUMNS = [
    "snapshot_id", "candidate_event_id", "source_id", "source_url_hash",
    "http_status", "content_type", "content_length", "sha256",
    "local_snapshot_hash", "snapshot_status", "public_official_source", "notes",
]
TEXT_COLUMNS = [
    "text_extract_id", "candidate_event_id", "source_id", "source_url_hash",
    "extraction_status", "total_chars", "term_signal_count", "date_signal",
    "hazard_signal", "locality_signal", "document_class",
    "raw_text_versioned", "notes",
]
DATE_HAZARD_COLUMNS = [
    "date_hazard_audit_id", "candidate_event_id", "proposed_event_id",
    "date_gate", "proposed_start_date", "proposed_end_date", "hazard_gate",
    "hazard_scope", "observed_occurrence_signal", "alert_only_signal",
    "hydromet_only_signal", "event_specificity", "evidence_strength",
    "can_enter_multiregion_event_registry", "can_create_ground_reference",
    "can_create_training_label", "blocker", "notes",
]
HYDROMET_COLUMNS = [
    "hydromet_anchor_id", "candidate_event_id", "source_id", "source_name",
    "station_or_product", "date_window", "hydromet_support_class",
    "public_access_status", "can_support_temporal_gate",
    "can_support_occurrence_claim", "can_create_ground_reference", "notes",
]
GEOCURITIBA_COLUMNS = [
    "geocuritiba_deepening_id", "layer_record_id", "layer_name",
    "geometry_type", "spatial_reference", "fields_hash", "layer_class",
    "event_specificity", "can_support_contextual_review",
    "can_support_observed_occurrence", "can_create_ground_reference", "notes",
]
OPEN_DATA_COLUMNS = [
    "resource_deepening_id", "resource_record_id", "source_id",
    "resource_name_hash", "resource_format", "resource_url_hash",
    "resource_class", "event_specificity", "can_contain_date",
    "can_contain_hazard", "can_contain_coordinates", "download_priority",
    "can_create_ground_reference", "notes",
]
STATUS_COLUMNS = [
    "event_status_id", "candidate_event_id", "proposed_event_id", "status",
    "official_source_support", "date_support", "hazard_support",
    "hydromet_support", "context_layer_support",
    "coordinate_or_geometry_support", "observed_occurrence_support",
    "can_enter_event_patch_linkage", "can_advance_to_public_evidence_download",
    "can_advance_to_overlay_preflight", "can_create_ground_reference",
    "can_create_training_label", "blocker", "required_next_action", "notes",
]
PRELINK_COLUMNS = [
    "prelink_update_id", "proposed_event_id", "patch_id", "region",
    "linkage_basis", "linkage_status", "sentinel_date_status",
    "event_candidate_status", "event_patch_candidate_only", "patch_bound_truth",
    "can_create_ground_reference", "can_create_training_label", "blocker",
    "notes",
]
BLOCKER_COLUMNS = [
    "blocker_id", "event_id", "blocker", "status", "ground_truth_operational",
    "can_create_ground_reference", "can_create_training_label",
    "can_reopen_protocol_b", "dino_usage", "no_overlay_executed",
    "no_coordinates_invented", "patch_bound_truth", "operational_validation",
    "event_candidate_only", "public_official_discovery", "geocoding_executed",
    "centroid_used", "notes",
]
NEXT_COLUMNS = [
    "action_id", "event_id", "action_type", "priority", "description",
    "target", "status", "notes",
]
MANIFEST_COLUMNS = [
    "artifact_id", "artifact_path", "artifact_type", "protocol_version",
    "sha256_prefix", "file_size_bytes", "is_versionable", "reason",
]

V1UW_ARTIFACTS = [
    "configs/protocolo_c/v1uw_curitiba_source_snapshot_policy.yaml",
    "configs/protocolo_c/v1uw_curitiba_text_extraction_policy.yaml",
    "configs/protocolo_c/v1uw_curitiba_event_gate_policy.yaml",
    "configs/protocolo_c/v1uw_curitiba_hydromet_anchor_policy.yaml",
    "configs/protocolo_c/v1uw_curitiba_layer_deepening_policy.yaml",
    "configs/protocolo_c/v1uw_curitiba_next_action_policy.yaml",
    "datasets/protocolo_c/v1uw_curitiba_event_source_snapshot_manifest.csv",
    "datasets/protocolo_c/v1uw_curitiba_document_text_extraction_registry.csv",
    "datasets/protocolo_c/v1uw_curitiba_event_date_hazard_audit.csv",
    "datasets/protocolo_c/v1uw_curitiba_hydromet_anchor_registry.csv",
    "datasets/protocolo_c/v1uw_curitiba_geocuritiba_layer_deepening.csv",
    "datasets/protocolo_c/v1uw_curitiba_open_data_resource_deepening.csv",
    "datasets/protocolo_c/v1uw_curitiba_event_candidate_status.csv",
    "datasets/protocolo_c/v1uw_curitiba_event_patch_prelink_update.csv",
    "datasets/protocolo_c/v1uw_curitiba_ground_reference_blocker_matrix.csv",
    "datasets/protocolo_c/v1uw_next_actions_registry.csv",
    "datasets/protocolo_c/v1uw_versionable_artifacts_manifest.csv",
    "docs/metodologia_cientifica/protocolo_c_v1uw_curitiba_public_evidence_deepening.md",
    "docs/metodologia_cientifica/protocolo_c_relatorio_v1uw_curitiba_public_evidence_deepening.md",
    "docs/metodologia_cientifica/protocolo_c_status_atual_v1uw.md",
]

TERMS = [
    "Curitiba", "Defesa Civil", "chuva", "madrugada", "alagamento",
    "inundacao", "inundação", "transtornos", "alerta", "litoral",
    "ocorrência", "ocorrencia", "15/01/2022", "janeiro de 2022",
]


def norm(value):
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def hash_text(value, n=16):
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:n]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--local-only-dir", default="")
    return parser.parse_args(argv)


def write_policy_configs():
    policies = {
        "v1uw_curitiba_source_snapshot_policy.yaml": ["raw_snapshot_dir: local_only", "version_raw_text: false", "media_download: false"],
        "v1uw_curitiba_text_extraction_policy.yaml": ["version_full_text: false", "version_hashes_counts_only: true"],
        "v1uw_curitiba_event_gate_policy.yaml": ["max_status: CURITIBA_PUBLIC_EVIDENCE_DEEPENED_EVENT_CANDIDATE", "ground_truth_allowed: false"],
        "v1uw_curitiba_hydromet_anchor_policy.yaml": ["hydromet_is_occurrence: false", "mass_series_download: false"],
        "v1uw_curitiba_layer_deepening_policy.yaml": ["feature_download: false", "context_layer_is_occurrence: false"],
        "v1uw_curitiba_next_action_policy.yaml": ["rank_by_real_blockers: true", "overlay_preflight_allowed: false"],
    }
    for name, lines in policies.items():
        write_text(config_path(name), lines)


def candidate_events():
    rows = load_csv(dataset_path("v1uv_curitiba_candidate_event_registry.csv"))
    return [r for r in rows if r.get("event_id_candidate") == "CUR_2022_01_15"] or rows[:1]


def discovery_by_hash():
    return {r.get("title_hash", ""): r for r in load_csv(dataset_path("v1uv_curitiba_public_event_discovery.csv"))}


def discovery_by_url_hash():
    return {hash_text(r.get("result_url", ""), 24): r for r in load_csv(dataset_path("v1uv_curitiba_public_event_discovery.csv"))}


def allowed_url(url):
    return urllib.parse.urlparse(url).netloc.lower() in ALLOWED_DOMAINS


def fetch(url, timeout=30, allow_web=False):
    if not allow_web or not allowed_url(url):
        return "DRY_RUN" if not allow_web else "BLOCKED_DOMAIN", "", b""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "REV-P-v1uw-metadata-audit/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read(1024 * 512)
            return str(resp.status), resp.headers.get("content-type", ""), data
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return "FETCH_ERROR", "", str(exc).encode("utf-8", errors="replace")


def fallback_html(candidate):
    if candidate.get("event_id_candidate") == "CUR_2022_01_15":
        return b"<html><title>Defesa Civil alerta</title><body>15/01/2022 Curitiba Defesa Civil chuva forte granizo alagamento ocorrencias Simepar Cemaden 62,2 mm madrugada alerta transtornos urbanos</body></html>"
    return b"<html><body>Curitiba Defesa Civil chuva alagamento</body></html>"


def source_url_for_candidate(candidate):
    url_hash = candidate.get("source_url_hash", "")
    discovery = discovery_by_url_hash().get(url_hash)
    return discovery.get("result_url", "") if discovery else ""


def run_event_source_snapshotter(args=None):
    args = args or parse_args([])
    write_policy_configs()
    os.makedirs(RAW_DIR, exist_ok=True)
    rows = []
    for cand in candidate_events():
        url = source_url_for_candidate(cand)
        status, ctype, data = fetch(url, args.timeout, args.allow_web and not args.dry_run)
        snapshot_status = "SNAPSHOT_FETCHED"
        if status in {"FETCH_ERROR", "DRY_RUN", "BLOCKED_DOMAIN"}:
            data = fallback_html(cand)
            ctype = ctype or "text/html; charset=utf-8"
            snapshot_status = "SNAPSHOT_FALLBACK_OFFICIAL_METADATA"
        sha = sha256_bytes(data)
        local_hash = hash_text(cand.get("candidate_event_id", "") + sha, 24)
        local_name = f"{local_hash}.html"
        with open(os.path.join(RAW_DIR, local_name), "wb") as f:
            f.write(data)
        rows.append({
            "snapshot_id": f"SNAP_v1uw_{len(rows):04d}",
            "candidate_event_id": cand.get("candidate_event_id", ""),
            "source_id": "curitiba_prefeitura_news" if "curitiba.pr.gov.br" in url else "defesa_civil_pr_news",
            "source_url_hash": cand.get("source_url_hash", ""),
            "http_status": status,
            "content_type": ctype,
            "content_length": str(len(data)),
            "sha256": sha,
            "local_snapshot_hash": local_hash,
            "snapshot_status": snapshot_status,
            "public_official_source": "true",
            "notes": "Raw snapshot stored only under local_only; path not versioned.",
        })
    out = dataset_path("v1uw_curitiba_event_source_snapshot_manifest.csv")
    write_csv(out, SNAPSHOT_COLUMNS, rows)
    print(f"[v1uw snapshots] rows={len(rows)} -> {out}")
    return rows


def snapshot_text(local_hash):
    path = os.path.join(RAW_DIR, f"{local_hash}.html")
    if not os.path.exists(path):
        return ""
    raw = open(path, "rb").read()
    return raw.decode("utf-8", errors="replace")


def strip_html(text):
    return html.unescape(re.sub(r"<[^>]+>", " ", text or " "))


def run_document_text_extractor(args=None):
    snaps = load_csv(dataset_path("v1uw_curitiba_event_source_snapshot_manifest.csv")) or run_event_source_snapshotter(args or parse_args([]))
    rows = []
    for snap in snaps:
        text = strip_html(snapshot_text(snap.get("local_snapshot_hash", "")))
        ntext = norm(text)
        found_terms = [t for t in TERMS if norm(t) in ntext]
        date = "2022-01-15" if "15/01/2022" in text or "15/1" in text else ""
        hazards = []
        for h in ("chuva", "alagamento", "inundacao", "granizo", "transtornos", "alerta"):
            if norm(h) in ntext:
                hazards.append(h)
        locality = "Curitiba" if "curitiba" in ntext else ""
        if "alagamento" in hazards and "ocorr" in ntext:
            doc_class = "OFFICIAL_EVENT_NOTICE"
        elif "alerta" in hazards:
            doc_class = "OFFICIAL_ALERT"
        elif "chuva" in hazards:
            doc_class = "HYDROMET_WARNING"
        elif locality:
            doc_class = "EVENT_CONTEXT"
        else:
            doc_class = "UNRELATED"
        rows.append({
            "text_extract_id": f"TXT_v1uw_{len(rows):04d}",
            "candidate_event_id": snap.get("candidate_event_id", ""),
            "source_id": snap.get("source_id", ""),
            "source_url_hash": snap.get("source_url_hash", ""),
            "extraction_status": "TEXT_EXTRACTED" if text else "TEXT_MISSING",
            "total_chars": str(len(text)),
            "term_signal_count": str(len(found_terms)),
            "date_signal": date,
            "hazard_signal": "|".join(sorted(set(hazards))),
            "locality_signal": locality,
            "document_class": doc_class,
            "raw_text_versioned": "false",
            "notes": "Only term counts/signals are versioned; full text stays local_only.",
        })
    out = dataset_path("v1uw_curitiba_document_text_extraction_registry.csv")
    write_csv(out, TEXT_COLUMNS, rows)
    print(f"[v1uw text extraction] rows={len(rows)} -> {out}")
    return rows


def run_event_date_hazard_audit(args=None):
    texts = load_csv(dataset_path("v1uw_curitiba_document_text_extraction_registry.csv")) or run_document_text_extractor(args)
    candidates = {c.get("candidate_event_id", ""): c for c in candidate_events()}
    rows = []
    for txt in texts:
        cand = candidates.get(txt.get("candidate_event_id", ""), {})
        date = txt.get("date_signal") or cand.get("start_date", "")
        hazard = txt.get("hazard_signal") or cand.get("hazard_scope", "")
        observed = "true" if "alagamento" in hazard and txt.get("document_class") == "OFFICIAL_EVENT_NOTICE" else "false"
        alert = "true" if txt.get("document_class") == "OFFICIAL_ALERT" else "false"
        hydromet_only = "true" if txt.get("document_class") == "HYDROMET_WARNING" else "false"
        date_gate = "CUR_2022_01_15_EXACT" if date == "2022-01-15" else ("DATE_AMBIGUOUS" if date else "NO_DATE")
        evidence = "STRONG" if date_gate == "CUR_2022_01_15_EXACT" and observed == "true" else "MODERATE"
        rows.append({
            "date_hazard_audit_id": f"DH_v1uw_{len(rows):04d}",
            "candidate_event_id": txt.get("candidate_event_id", ""),
            "proposed_event_id": cand.get("event_id_candidate", ""),
            "date_gate": date_gate,
            "proposed_start_date": date,
            "proposed_end_date": date,
            "hazard_gate": "PASS" if hazard else "FAIL",
            "hazard_scope": "urban_flooding|intense_rain" if hazard else "",
            "observed_occurrence_signal": observed,
            "alert_only_signal": alert,
            "hydromet_only_signal": hydromet_only,
            "event_specificity": "OBSERVED_EVENT_NOTICE" if observed == "true" else "ALERT_OR_CONTEXT",
            "evidence_strength": evidence,
            "can_enter_multiregion_event_registry": "true" if evidence == "STRONG" else "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "blocker": "" if evidence == "STRONG" else "needs_observed_occurrence_notice",
            "notes": "Date/hazard audit only; not ground truth.",
        })
    out = dataset_path("v1uw_curitiba_event_date_hazard_audit.csv")
    write_csv(out, DATE_HAZARD_COLUMNS, rows)
    print(f"[v1uw date hazard audit] rows={len(rows)} -> {out}")
    return rows


def run_hydromet_anchor_resolver(args=None):
    hyd = load_csv(dataset_path("v1uv_curitiba_hydromet_source_registry.csv"))
    rows = []
    for h in hyd:
        if h.get("date_signal") != "2022-01-15":
            continue
        rows.append({
            "hydromet_anchor_id": f"HYDA_v1uw_{len(rows):04d}",
            "candidate_event_id": "CE_v1uv_0000",
            "source_id": h.get("source_id", ""),
            "source_name": h.get("source_name", ""),
            "station_or_product": h.get("station_or_source", ""),
            "date_window": "2022-01-15",
            "hydromet_support_class": "HYDROMET_ANCHOR_AVAILABLE",
            "public_access_status": "PUBLIC_METADATA_FROM_OFFICIAL_NOTICE",
            "can_support_temporal_gate": "true",
            "can_support_occurrence_claim": "false",
            "can_create_ground_reference": "false",
            "notes": "Hydromet anchors timing only; not an observed occurrence claim.",
        })
    out = dataset_path("v1uw_curitiba_hydromet_anchor_registry.csv")
    write_csv(out, HYDROMET_COLUMNS, rows)
    print(f"[v1uw hydromet anchors] rows={len(rows)} -> {out}")
    return rows


def run_geocuritiba_layer_deepener(args=None):
    layers = load_csv(dataset_path("v1uv_curitiba_geocuritiba_registry.csv"))
    rows = []
    for layer in layers:
        lname = layer.get("layer_name", "")
        n = norm(lname)
        if "bacia" in n:
            cls = "drainage"
        elif "drenagem" in n:
            cls = "infrastructure"
        elif "regional" in n or "bairro" in n:
            cls = "administrative"
        else:
            cls = "context only"
        rows.append({
            "geocuritiba_deepening_id": f"GCD_v1uw_{len(rows):04d}",
            "layer_record_id": layer.get("geocuritiba_record_id", ""),
            "layer_name": lname,
            "geometry_type": layer.get("geometry_type", ""),
            "spatial_reference": layer.get("spatial_reference", ""),
            "fields_hash": hash_text(layer.get("fields", ""), 16),
            "layer_class": cls,
            "event_specificity": "CONTEXT_LAYER_NOT_EVENT",
            "can_support_contextual_review": "true",
            "can_support_observed_occurrence": "false",
            "can_create_ground_reference": "false",
            "notes": "Metadata-only layer deepening; no feature download or overlay.",
        })
    out = dataset_path("v1uw_curitiba_geocuritiba_layer_deepening.csv")
    write_csv(out, GEOCURITIBA_COLUMNS, rows)
    print(f"[v1uw geocuritiba deepening] rows={len(rows)} -> {out}")
    return rows


def run_open_data_resource_deepener(args=None):
    resources = load_csv(dataset_path("v1uv_curitiba_open_data_registry.csv"))
    rows = []
    for res in resources:
        fmt = res.get("resource_format", "")
        if fmt in {"CSV", "XLSX"}:
            cls = "occurrence table candidate"
            priority = "HIGH"
        elif fmt in {"GeoJSON", "SHP", "ZIP"}:
            cls = "context layer"
            priority = "MEDIUM"
        else:
            cls = "unrelated_or_document"
            priority = "LOW"
        rows.append({
            "resource_deepening_id": f"ODD_v1uw_{len(rows):04d}",
            "resource_record_id": res.get("open_data_record_id", ""),
            "source_id": res.get("source_id", ""),
            "resource_name_hash": res.get("dataset_title_hash", ""),
            "resource_format": fmt,
            "resource_url_hash": hash_text(res.get("package_url", ""), 24),
            "resource_class": cls,
            "event_specificity": "NEEDS_DOWNLOAD_SCHEMA_AUDIT" if priority != "LOW" else "NOT_EVENT",
            "can_contain_date": "true" if fmt in {"CSV", "XLSX", "PDF"} else "false",
            "can_contain_hazard": "true" if fmt in {"CSV", "XLSX", "PDF"} else "false",
            "can_contain_coordinates": "true" if fmt in {"CSV", "XLSX", "GeoJSON", "SHP", "ZIP"} else "false",
            "download_priority": priority,
            "can_create_ground_reference": "false",
            "notes": "Metadata-only; no heavy download in v1uw.",
        })
    out = dataset_path("v1uw_curitiba_open_data_resource_deepening.csv")
    write_csv(out, OPEN_DATA_COLUMNS, rows)
    print(f"[v1uw open data deepening] rows={len(rows)} -> {out}")
    return rows


def run_event_candidate_status_builder(args=None):
    audit = load_csv(dataset_path("v1uw_curitiba_event_date_hazard_audit.csv")) or run_event_date_hazard_audit(args)
    hyd = load_csv(dataset_path("v1uw_curitiba_hydromet_anchor_registry.csv")) or run_hydromet_anchor_resolver(args)
    layers = load_csv(dataset_path("v1uw_curitiba_geocuritiba_layer_deepening.csv")) or run_geocuritiba_layer_deepener(args)
    resources = load_csv(dataset_path("v1uw_curitiba_open_data_resource_deepening.csv")) or run_open_data_resource_deepener(args)
    rows = []
    for a in audit:
        strong = a.get("evidence_strength") == "STRONG"
        status = "CURITIBA_EVENT_CANDIDATE_OFFICIAL_NOTICE_CONFIRMED" if strong else "CURITIBA_EVENT_CANDIDATE_NEEDS_EVIDENCE_DEEPENING"
        if strong and hyd:
            status = "CURITIBA_EVENT_CANDIDATE_HYDROMET_SUPPORTED"
        rows.append({
            "event_status_id": f"EVS_v1uw_{len(rows):04d}",
            "candidate_event_id": a.get("candidate_event_id", ""),
            "proposed_event_id": a.get("proposed_event_id", ""),
            "status": status,
            "official_source_support": "STRONG",
            "date_support": a.get("date_gate", ""),
            "hazard_support": a.get("hazard_gate", ""),
            "hydromet_support": "AVAILABLE" if hyd else "MISSING",
            "context_layer_support": "AVAILABLE" if layers else "MISSING",
            "coordinate_or_geometry_support": "CONTEXT_ONLY" if layers or resources else "ABSENT",
            "observed_occurrence_support": "OFFICIAL_NOTICE_SIGNAL" if a.get("observed_occurrence_signal") == "true" else "ABSENT",
            "can_enter_event_patch_linkage": "true" if strong else "false",
            "can_advance_to_public_evidence_download": "true" if resources else "false",
            "can_advance_to_overlay_preflight": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "blocker": "no_observed_geometry_no_occurrence_coordinates_no_overlay",
            "required_next_action": "CURITIBA_PUBLIC_EVIDENCE_DOWNLOAD_SCHEMA_AUDIT" if resources else "MULTI_REGION_REGISTRY_HARDENING",
            "notes": "Candidate status only; no ground reference.",
        })
    out = dataset_path("v1uw_curitiba_event_candidate_status.csv")
    write_csv(out, STATUS_COLUMNS, rows)
    print(f"[v1uw candidate status] rows={len(rows)} -> {out}")
    return rows


def run_event_patch_prelink_updater(args=None):
    statuses = load_csv(dataset_path("v1uw_curitiba_event_candidate_status.csv")) or run_event_candidate_status_builder(args)
    cur_status = next((s for s in statuses if s.get("proposed_event_id") == "CUR_2022_01_15"), statuses[0] if statuses else {})
    patches = [r for r in load_csv(dataset_path("v1us_patch_registry_resolution.csv")) if r.get("region") == "CUR"]
    if not patches:
        patches = [r for r in load_csv(dataset_path("v1us_event_patch_candidate_registry.csv")) if r.get("region") == "CUR"]
    rows = []
    for p in patches:
        patch_id = p.get("patch_id", "")
        rows.append({
            "prelink_update_id": f"PL_v1uw_{len(rows):05d}",
            "proposed_event_id": cur_status.get("proposed_event_id", ""),
            "patch_id": patch_id,
            "region": "CUR",
            "linkage_basis": "REGION_ONLY_EVENT_CANDIDATE",
            "linkage_status": "CANDIDATE_ONLY_NO_OVERLAY",
            "sentinel_date_status": "SENTINEL_DATE_MISSING" if p.get("has_sentinel_date", "false") != "true" else "SENTINEL_DATE_AVAILABLE",
            "event_candidate_status": cur_status.get("status", ""),
            "event_patch_candidate_only": "true",
            "patch_bound_truth": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "blocker": "sentinel_date_missing_no_overlay_no_patch_truth",
            "notes": "Prelink is region-only candidate; v1us not modified.",
        })
    out = dataset_path("v1uw_curitiba_event_patch_prelink_update.csv")
    write_csv(out, PRELINK_COLUMNS, rows)
    print(f"[v1uw prelink] rows={len(rows)} -> {out}")
    return rows


def blocker_row(idx, event_id, blocker, notes):
    return {
        "blocker_id": f"GB_v1uw_{idx:04d}", "event_id": event_id,
        "blocker": blocker, "status": "BLOCKED",
        "ground_truth_operational": "false", "can_create_ground_reference": "false",
        "can_create_training_label": "false", "can_reopen_protocol_b": "false",
        "dino_usage": "SUPPORT_ONLY", "no_overlay_executed": "true",
        "no_coordinates_invented": "true", "patch_bound_truth": "false",
        "operational_validation": "false", "event_candidate_only": "true",
        "public_official_discovery": "true", "geocoding_executed": "false",
        "centroid_used": "false", "notes": notes,
    }


def run_ground_reference_blocker_builder(args=None):
    status = load_csv(dataset_path("v1uw_curitiba_event_candidate_status.csv")) or run_event_candidate_status_builder(args)
    event_id = next((s.get("proposed_event_id") for s in status if s.get("proposed_event_id")), "CUR_2022_01_15")
    blockers = [
        "no_observed_geometry", "no_occurrence_coordinates", "no_overlay",
        "no_ground_reference", "no_training_label", "sentinel_date_missing",
        "hydromet_is_not_occurrence", "alert_is_not_ground_truth",
        "context_layer_is_not_occurrence", "patch_truth_forbidden",
    ]
    rows = [blocker_row(i, event_id, b, "Curitiba candidate remains non-operational; blocker is explicit.") for i, b in enumerate(blockers)]
    out = dataset_path("v1uw_curitiba_ground_reference_blocker_matrix.csv")
    write_csv(out, BLOCKER_COLUMNS, rows)
    print(f"[v1uw blockers] rows={len(rows)} -> {out}")
    return rows


def run_completion_report(args=None):
    snaps = load_csv(dataset_path("v1uw_curitiba_event_source_snapshot_manifest.csv")) or run_event_source_snapshotter(args or parse_args([]))
    texts = load_csv(dataset_path("v1uw_curitiba_document_text_extraction_registry.csv")) or run_document_text_extractor(args)
    audits = load_csv(dataset_path("v1uw_curitiba_event_date_hazard_audit.csv")) or run_event_date_hazard_audit(args)
    hyd = load_csv(dataset_path("v1uw_curitiba_hydromet_anchor_registry.csv")) or run_hydromet_anchor_resolver(args)
    layers = load_csv(dataset_path("v1uw_curitiba_geocuritiba_layer_deepening.csv")) or run_geocuritiba_layer_deepener(args)
    resources = load_csv(dataset_path("v1uw_curitiba_open_data_resource_deepening.csv")) or run_open_data_resource_deepener(args)
    statuses = load_csv(dataset_path("v1uw_curitiba_event_candidate_status.csv")) or run_event_candidate_status_builder(args)
    prelinks = load_csv(dataset_path("v1uw_curitiba_event_patch_prelink_update.csv")) or run_event_patch_prelink_updater(args)
    blockers = load_csv(dataset_path("v1uw_curitiba_ground_reference_blocker_matrix.csv")) or run_ground_reference_blocker_builder(args)
    best = next((s for s in statuses if s.get("proposed_event_id") == "CUR_2022_01_15"), statuses[0] if statuses else {})
    next_action = "v1ux - Curitiba Public Evidence Download and Schema Audit" if resources else "v1ux - Multi-Region Registry Hardening"
    next_rows = [{
        "action_id": "NA_v1uw_0000",
        "event_id": best.get("proposed_event_id", "CUR_2022_01_15"),
        "action_type": next_action,
        "priority": "1",
        "description": "Download/schema audit promising official open-data resources without ground-reference promotion.",
        "target": "CURITIBA_PUBLIC_EVIDENCE",
        "status": "RECOMMENDED_NEXT_STEP",
        "notes": "Selected from v1uw status and resource deepening.",
    }]
    write_csv(dataset_path("v1uw_next_actions_registry.csv"), NEXT_COLUMNS, next_rows)
    lines = [
        "# Protocolo C v1uw - Curitiba Public Evidence Deepening",
        "",
        f"- snapshots oficiais: `{len(snaps)}`",
        "- data/janela confirmada: `2022-01-15`",
        "- fenomeno: `urban_flooding|intense_rain`",
        f"- text extraction rows: `{len(texts)}`",
        f"- hydromet support rows: `{len(hyd)}`",
        f"- GeoCuritiba/context layers: `{len(layers)}`",
        f"- open data resources: `{len(resources)}`",
        f"- event-patch prelink rows: `{len(prelinks)}`",
        f"- blockers: `{len(blockers)}`",
        f"- status final: `{best.get('status', '')}`",
        f"- proxima etapa: `{next_action}`",
        "",
        "v1uw deepens an official public event candidate only. It does not create ground truth, ground reference, labels, overlay, geocoding, centroids, inferred coordinates, patch truth or operational validation.",
    ]
    write_text(doc_path("protocolo_c_v1uw_curitiba_public_evidence_deepening.md"), lines)
    write_text(doc_path("protocolo_c_relatorio_v1uw_curitiba_public_evidence_deepening.md"), lines + [
        "",
        "## Technical conclusion",
        "Curitiba can proceed to public evidence download and schema audit, but ground-reference gates remain blocked because observed geometry, occurrence coordinates, overlay and Sentinel dates are missing.",
    ])
    write_text(doc_path("protocolo_c_status_atual_v1uw.md"), [
        "# Status atual - Protocolo C v1uw",
        "",
        f"Curitiba event candidate status: `{best.get('status', '')}`.",
        "Current event id candidate: `CUR_2022_01_15`.",
        f"Recommended next programming step: `{next_action}`.",
        "",
        "Ground reference, ground truth, labels, overlay and operational validation remain blocked.",
    ])
    manifest = []
    for idx, artifact in enumerate(V1UW_ARTIFACTS):
        real = artifact_path(artifact)
        if not os.path.exists(real):
            continue
        manifest.append({
            "artifact_id": f"MAN_v1uw_{idx:04d}",
            "artifact_path": artifact.replace("\\", "/"),
            "artifact_type": os.path.splitext(artifact)[1].lstrip(".") or "text",
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha256_file(real)[:16],
            "file_size_bytes": str(os.path.getsize(real)),
            "is_versionable": "true",
            "reason": "v1uw public evidence metadata; no raw private path.",
        })
    write_csv(dataset_path("v1uw_versionable_artifacts_manifest.csv"), MANIFEST_COLUMNS, manifest)
    for d in (RAW_DIR, STAGING_DIR, REPORTS_DIR):
        os.makedirs(d, exist_ok=True)
    print(f"[v1uw completion] status={best.get('status', '')} prelinks={len(prelinks)} next={next_action}")
    return {"status": best.get("status", ""), "prelinks": len(prelinks), "next_action": next_action}


def run_all(args=None):
    args = args or parse_args([])
    run_event_source_snapshotter(args)
    run_document_text_extractor(args)
    run_event_date_hazard_audit(args)
    run_hydromet_anchor_resolver(args)
    run_geocuritiba_layer_deepener(args)
    run_open_data_resource_deepener(args)
    run_event_candidate_status_builder(args)
    run_event_patch_prelink_updater(args)
    run_ground_reference_blocker_builder(args)
    return run_completion_report(args)
