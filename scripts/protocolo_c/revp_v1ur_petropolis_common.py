#!/usr/bin/env python3
"""v1ur Petropolis public geodata path recovery.

Focused public-path recovery from v1uq missing-geodata signals. It may find and
inventory geodata candidates, but it never creates ground truth, ground
reference, labels, overlay, or inferred coordinates.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import unicodedata
import zipfile
from collections import Counter
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

try:
    import urllib.request
except Exception:  # pragma: no cover
    urllib = None

PROTOCOL_VERSION = "v1ur"
DATASET_DIR = "datasets/protocolo_c"
DOCS_DIR = "docs/metodologia_cientifica"
CONFIG_DIR = "configs/protocolo_c"
LOCAL_RAW_DIR = "local_only/protocolo_c/petropolis_geodata_path_recovery/raw/v1ur"
LOCAL_STAGING_DIR = "local_only/protocolo_c/petropolis_geodata_path_recovery/staging/v1ur"
LOCAL_QUARANTINE_DIR = "local_only/protocolo_c/petropolis_geodata_path_recovery/quarantine/v1ur"
LOCAL_REPORTS_DIR = "local_only/protocolo_c/petropolis_geodata_path_recovery/reports/v1ur"
MAX_STATUS = "PETROPOLIS_PUBLIC_GEODATA_PATH_CANDIDATE_FOR_REVIEW"

SEED_COLUMNS = [
    "seed_id", "event_id", "source_asset_id", "page_number", "signal_class",
    "signal_strength", "referenced_artifact_type", "seed_terms",
    "source_context_hash", "priority", "target_source_family",
    "can_resolve_by_public_search", "can_create_ground_reference",
    "can_create_training_label", "notes",
]
MISSING_GEODATA_COLUMNS = [
    "missing_geodata_id", "event_id", "asset_id", "page_number",
    "signal_class", "signal_strength", "referenced_artifact_type",
    "public_path_hint", "can_be_resolved_by_public_search",
    "recommended_next_query", "notes",
]
RELATED_COLUMNS = [
    "related_item_id", "event_id", "seed_id", "item_url", "title", "year",
    "relation_type", "matched_terms_hash", "is_public", "event_specificity",
    "candidate_relevance", "has_bitstreams", "notes",
]
BITSTREAM_COLUMNS = [
    "bitstream_record_id", "event_id", "item_url", "bitstream_url",
    "bitstream_name", "format_hint", "content_length", "is_public",
    "is_geodata_candidate", "is_pdf_only", "event_specificity",
    "blocking_reason", "notes",
]
LAYER_COLUMNS = [
    "layer_search_id", "event_id", "service_url", "layer_url",
    "layer_name", "layer_id", "geometry_type", "spatial_reference", "extent",
    "fields", "matched_terms_hash", "layer_class", "candidate_relevance",
    "can_query_features", "sample_downloaded", "can_create_ground_reference",
    "notes",
]
QUERY_COLUMNS = [
    "query_id", "event_id", "seed_id", "query_terms", "target_domain",
    "result_url", "http_status", "content_type", "link_text_hash",
    "detected_artifact_type", "candidate_relevance", "public_access_status",
    "blocking_reason", "notes",
]
CANDIDATE_COLUMNS = [
    "candidate_url_id", "event_id", "source_registry", "source_record_id",
    "url", "url_sha1_12", "candidate_class", "artifact_type",
    "event_specificity", "phenomenon_specificity", "download_priority",
    "can_contain_geometry", "can_contain_observed_occurrence",
    "can_contain_context_only", "blocking_reason", "notes",
]
DOWNLOAD_COLUMNS = [
    "download_id", "event_id", "candidate_url_id", "url", "safe_filename",
    "local_path_hash", "sha256", "file_size_bytes", "mime_type", "extension",
    "download_status", "duplicate_status", "license_status", "notes",
]
INVENTORY_COLUMNS = [
    "inventory_id", "event_id", "download_id", "asset_type",
    "container_type", "internal_path", "has_geometry", "geometry_type", "crs",
    "feature_count", "has_date_field", "has_phenomenon_field",
    "has_locality_field", "has_coordinate_fields", "is_pdf_only",
    "inventory_status", "notes",
]
AUDIT_COLUMNS = [
    "geodata_candidate_id", "event_id", "download_id", "inventory_id",
    "candidate_class", "public_official_source", "event_specificity",
    "phenomenon_class", "temporal_compatibility", "has_geometry",
    "geometry_type", "crs_status", "context_only_status",
    "can_advance_to_overlay_preflight", "can_create_ground_reference",
    "can_create_training_label", "blocker", "required_next_action", "notes",
]
EVENT_STATUS_COLUMNS = [
    "event_id", "v1ur_status", "has_public_path_candidate",
    "has_downloaded_artifact", "has_geodata_inventory", "has_observed_geometry",
    "has_context_layer_only", "ground_truth_operational",
    "can_create_ground_reference", "can_create_training_label",
    "can_advance_to_overlay_preflight", "main_blocker",
    "recommended_next_action", "notes",
]
BLOCKER_COLUMNS = [
    "blocker_id", "event_id", "gate", "gate_status", "blocking_reason",
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "notes",
]
NEXT_COLUMNS = ["action_id", "event_id", "action_type", "priority", "description", "target", "status", "notes"]
MANIFEST_COLUMNS = [
    "artifact_id", "artifact_path", "artifact_type", "protocol_version",
    "sha256_prefix", "file_size_bytes", "is_versionable", "reason",
]

V1UR_ARTIFACTS = [
    "configs/protocolo_c/v1ur_petropolis_geodata_recovery_policy.yaml",
    "configs/protocolo_c/v1ur_petropolis_allowed_domains.yaml",
    "configs/protocolo_c/v1ur_petropolis_query_templates.yaml",
    "configs/protocolo_c/v1ur_petropolis_candidate_url_policy.yaml",
    "configs/protocolo_c/v1ur_petropolis_download_policy.yaml",
    "configs/protocolo_c/v1ur_petropolis_geodata_audit_policy.yaml",
    "datasets/protocolo_c/v1ur_petropolis_geodata_signal_seed_registry.csv",
    "datasets/protocolo_c/v1ur_petropolis_rigeo_related_item_registry.csv",
    "datasets/protocolo_c/v1ur_petropolis_sgb_bitstream_deep_registry.csv",
    "datasets/protocolo_c/v1ur_petropolis_geosgb_layer_search_registry.csv",
    "datasets/protocolo_c/v1ur_petropolis_public_query_registry.csv",
    "datasets/protocolo_c/v1ur_petropolis_candidate_url_registry.csv",
    "datasets/protocolo_c/v1ur_petropolis_geodata_download_manifest.csv",
    "datasets/protocolo_c/v1ur_petropolis_geodata_inventory.csv",
    "datasets/protocolo_c/v1ur_petropolis_geodata_candidate_audit.csv",
    "datasets/protocolo_c/v1ur_petropolis_event_status_registry.csv",
    "datasets/protocolo_c/v1ur_petropolis_ground_reference_blocker_matrix.csv",
    "datasets/protocolo_c/v1ur_next_actions_registry.csv",
    "datasets/protocolo_c/v1ur_versionable_artifacts_manifest.csv",
    "docs/metodologia_cientifica/protocolo_c_v1ur_petropolis_public_geodata_path_recovery.md",
    "docs/metodologia_cientifica/protocolo_c_relatorio_v1ur_petropolis_public_geodata_path_recovery.md",
    "docs/metodologia_cientifica/protocolo_c_status_atual_v1ur.md",
]

ALLOWED_DOMAINS = [
    "rigeo.sgb.gov.br", "geoportal.sgb.gov.br", "sgb.gov.br",
    "cprm.gov.br", "rj.gov.br", "petropolis.rj.gov.br",
    "copernicus.eu", "disasterscharter.org",
]
SAFE_TOKENS = [
    "Petropolis", "Petrópolis", "SGB", "CPRM", "RIGeo", "geodados",
    "shapefile", "SIG", "camada", "vetor", "mapa", "risco",
    "inundacao", "inundação", "deslizamento", "movimento de massa",
    "cicatriz", "base cartografica", "base cartográfica", "anexo digital",
]
GEODATA_EXTS = {"zip", "shp", "gpkg", "geojson", "kml", "kmz", "csv", "xlsx", "xls"}


def norm_text(value):
    normalized = unicodedata.normalize("NFKD", value or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def bool_text(value):
    return "true" if bool(value) else "false"


def write_csv(path, columns, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_text(path, lines):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha1_12(value):
    return hashlib.sha1((value or "").encode("utf-8")).hexdigest()[:12]


def hash_terms(value):
    if isinstance(value, list):
        text = "|".join(sorted(set(value)))
    else:
        text = value or ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16] if text else ""


def safe_basename(url, fallback="asset"):
    name = os.path.basename(urlparse(url).path) or fallback
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:160] or fallback


def classify_format(url, name=""):
    text = norm_text(f"{url} {name}")
    for ext in ["geojson", "gpkg", "kmz", "kml", "shp", "zip", "xlsx", "xls", "csv", "pdf", "json"]:
        if f".{ext}" in text or text.endswith(ext):
            return ext
    if "featureserver" in text:
        return "FeatureServer"
    if "mapserver" in text:
        return "MapServer"
    if "wfs" in text:
        return "WFS"
    return "html"


def allowed_url(url):
    host = urlparse(url).netloc.lower()
    return any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)


def fetch_bytes(url, timeout=30, max_bytes=2_000_000):
    if urllib is None:
        return b"", "", "0"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "REV-P-Academic-Research/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read(max_bytes)
            ctype = resp.headers.get("Content-Type", "")
            clen = resp.headers.get("Content-Length", "")
            return data, ctype, clen
    except Exception:
        return b"", "", "0"


def fetch_text(url, timeout=30):
    data, ctype, clen = fetch_bytes(url, timeout=timeout)
    return data.decode("utf-8", errors="replace") if data else "", ctype, clen


def fetch_json(url, timeout=30):
    sep = "&" if "?" in url else "?"
    text, _, _ = fetch_text(url if "f=pjson" in url else f"{url}{sep}f=pjson", timeout=timeout)
    try:
        return json.loads(text) if text else None
    except Exception:
        return None


class LinkExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.title = ""
        self._href = None
        self._text = []
        self._title = False

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v:
                    self._href = v
                    self._text = []
        if tag in {"title", "h1", "h2", "h3"}:
            self._title = True

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data.strip())
        if self._title and data.strip() and not self.title:
            self.title = data.strip()

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            self.links.append((self._href, " ".join(t for t in self._text if t)))
            self._href = None
            self._text = []
        if tag in {"title", "h1", "h2", "h3"}:
            self._title = False


def parse_links(html, base):
    parser = LinkExtractor()
    parser.feed(html or "")
    return parser.title, [(urljoin(base, href), text) for href, text in parser.links]


def token_terms(*values):
    text = norm_text(" ".join(v or "" for v in values))
    terms = []
    for token in SAFE_TOKENS:
        if norm_text(token) in text:
            terms.append(token)
    if "petropolis" not in norm_text("|".join(terms)):
        terms.append("Petropolis")
    if "sgb" not in norm_text("|".join(terms)):
        terms.append("SGB")
    return terms[:14]


def run_geodata_signal_seed_builder():
    source = load_csv(os.path.join(DATASET_DIR, "v1uq_petropolis_missing_geodata_signal_audit.csv"))
    rows, seen = [], set()
    for r in source:
        key = (r.get("event_id"), r.get("asset_id"), r.get("page_number"), r.get("signal_class"), r.get("referenced_artifact_type"), r.get("recommended_next_query"))
        if key in seen:
            continue
        seen.add(key)
        terms = token_terms(r.get("signal_class"), r.get("referenced_artifact_type"), r.get("recommended_next_query"), r.get("public_path_hint"))
        family = "SGB_RIGEO" if "SGB" in terms or "RIGeo" in terms else "GEOSGB"
        priority = "1" if r.get("signal_strength") == "STRONG" else "2" if r.get("signal_strength") == "MODERATE" else "3"
        rows.append({
            "seed_id": f"SEED_v1ur_{len(rows):04d}",
            "event_id": r.get("event_id", ""),
            "source_asset_id": r.get("asset_id", ""),
            "page_number": r.get("page_number", ""),
            "signal_class": r.get("signal_class", ""),
            "signal_strength": r.get("signal_strength", ""),
            "referenced_artifact_type": r.get("referenced_artifact_type", ""),
            "seed_terms": "|".join(terms),
            "source_context_hash": hash_terms("|".join(str(x) for x in key)),
            "priority": priority,
            "target_source_family": family,
            "can_resolve_by_public_search": "true",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": "Seed derived from v1uq missing-geodata signal; no geometry inferred.",
        })
    out = os.path.join(DATASET_DIR, "v1ur_petropolis_geodata_signal_seed_registry.csv")
    write_csv(out, SEED_COLUMNS, rows)
    print(f"[v1ur seeds] rows={len(rows)} -> {out}")
    return rows


def rigeo_item_url():
    rows = load_csv(os.path.join(DATASET_DIR, "v1up_petropolis_sgb_rigeo_registry.csv"))
    for r in rows:
        if r.get("item_url") and "handle/doc" in r["item_url"]:
            return r["item_url"]
    return "https://rigeo.sgb.gov.br/handle/doc/22668"


def run_rigeo_related_item_resolver(allow_web=False, timeout=30, fixture_html=""):
    seeds = load_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_geodata_signal_seed_registry.csv"))
    item = rigeo_item_url()
    html = ""
    if fixture_html and os.path.exists(fixture_html):
        html = open(fixture_html, encoding="utf-8").read()
    elif allow_web:
        html, _, _ = fetch_text(item, timeout=timeout)
    title, links = parse_links(html, item) if html else ("Avaliacao tecnica pos-desastre: Petropolis, RJ", [])
    rows = []
    seen = set()
    base_seed = seeds[0]["seed_id"] if seeds else ""
    candidates = [(item, title, "KNOWN_V1UP_ITEM")]
    for url, text in links:
        if "handle/doc" in url and url not in seen:
            candidates.append((url, text or safe_basename(url), "INTERNAL_HANDLE_LINK"))
    for url, text, relation in candidates:
        seen.add(url)
        rows.append({
            "related_item_id": f"REL_v1ur_{len(rows):04d}",
            "event_id": "PET_2022_02_15" if "petropolis" in norm_text(f"{url} {text} {title}") else "",
            "seed_id": base_seed,
            "item_url": url,
            "title": text or title,
            "year": "2022" if "2022" in norm_text(f"{text} {title} {url}") else "",
            "relation_type": relation,
            "matched_terms_hash": hash_terms(token_terms(text, title, url)),
            "is_public": bool_text(allowed_url(url)),
            "event_specificity": "EVENT_SPECIFIC_PETROPOLIS_2022" if "petropolis" in norm_text(f"{url} {text} {title}") else "UNCONFIRMED",
            "candidate_relevance": "HIGH" if url == item else "MEDIUM",
            "has_bitstreams": bool_text("bitstream" in norm_text(html) or url == item),
            "notes": "Related item candidate only; no item invented.",
        })
    out = os.path.join(DATASET_DIR, "v1ur_petropolis_rigeo_related_item_registry.csv")
    write_csv(out, RELATED_COLUMNS, rows)
    print(f"[v1ur related RIGeo] rows={len(rows)} -> {out}")
    return rows


def run_sgb_bitstream_deep_resolver(allow_web=False, timeout=30, fixture_html=""):
    related = load_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_rigeo_related_item_registry.csv"))
    rows = []
    for rel in related:
        item = rel.get("item_url", "")
        html = ""
        if fixture_html and os.path.exists(fixture_html):
            html = open(fixture_html, encoding="utf-8").read()
        elif allow_web and item:
            html, _, _ = fetch_text(item, timeout=timeout)
        title, links = parse_links(html, item) if html else ("", [])
        if not links and item == rigeo_item_url():
            for r in load_csv(os.path.join(DATASET_DIR, "v1up_petropolis_sgb_rigeo_registry.csv")):
                if r.get("bitstream_url"):
                    links.append((r["bitstream_url"], r.get("bitstream_name", "")))
        for url, text in links:
            if not any(x in norm_text(f"{url} {text}") for x in ["bitstream", "download", ".zip", ".pdf", ".geojson", ".gpkg", ".kml", ".kmz", ".csv", ".xlsx", ".shp"]):
                continue
            fmt = classify_format(url, text)
            geo = fmt in GEODATA_EXTS and fmt != "pdf"
            rows.append({
                "bitstream_record_id": f"BIT_v1ur_{len(rows):04d}",
                "event_id": rel.get("event_id", ""),
                "item_url": item,
                "bitstream_url": url,
                "bitstream_name": text or safe_basename(url),
                "format_hint": fmt,
                "content_length": "",
                "is_public": bool_text(allowed_url(url)),
                "is_geodata_candidate": bool_text(geo),
                "is_pdf_only": bool_text(fmt == "pdf"),
                "event_specificity": rel.get("event_specificity", ""),
                "blocking_reason": "pdf_document_only" if fmt == "pdf" else "",
                "notes": "Deep bitstream inventory only; download handled downstream.",
            })
    out = os.path.join(DATASET_DIR, "v1ur_petropolis_sgb_bitstream_deep_registry.csv")
    write_csv(out, BITSTREAM_COLUMNS, rows)
    print(f"[v1ur bitstreams] rows={len(rows)} -> {out}")
    return rows


def classify_layer(name, fields=""):
    text = norm_text(f"{name} {fields}")
    risk = any(t in text for t in ["risco", "suscetibilidade", "susceptibility"])
    observed = any(t in text for t in ["cicatriz", "ocorrencia", "campo", "pos desastre", "pós desastre", "deslizamento", "inundacao", "alagamento"]) and not risk
    if risk:
        return "RISK_CONTEXT_LAYER", "MEDIUM"
    if observed:
        return "OBSERVED_OR_FIELD_MAPPING_LAYER_CANDIDATE", "HIGH"
    return "OFF_TARGET_OR_GENERIC_LAYER", "LOW"


def run_geosgb_layer_search(allow_web=False, timeout=30, services_fixture="", layers_fixture=""):
    base = "https://geoportal.sgb.gov.br/server/rest/services"
    services_doc = None
    layers_doc = None
    if services_fixture and os.path.exists(services_fixture):
        services_doc = json.load(open(services_fixture, encoding="utf-8"))
    elif allow_web:
        services_doc = fetch_json(base, timeout=timeout)
    if layers_fixture and os.path.exists(layers_fixture):
        layers_doc = json.load(open(layers_fixture, encoding="utf-8"))
    rows = []
    if not services_doc:
        rows.append({
            "layer_search_id": "LAYER_v1ur_0000", "event_id": "",
            "service_url": base, "layer_url": "", "layer_name": "",
            "layer_id": "", "geometry_type": "", "spatial_reference": "",
            "extent": "", "fields": "", "matched_terms_hash": "",
            "layer_class": "BLOCKED", "candidate_relevance": "LOW",
            "can_query_features": "false", "sample_downloaded": "false",
            "can_create_ground_reference": "false",
            "notes": "DRY_RUN_ENDPOINT_NOT_QUERIED" if not allow_web else "ENDPOINT_UNREACHABLE",
        })
    else:
        for svc in services_doc.get("services", [])[:100]:
            sname = svc.get("name", "")
            stype = svc.get("type", "MapServer")
            service_url = f"{base.rstrip('/')}/{sname}/{stype}"
            doc = layers_doc or (fetch_json(service_url, timeout=timeout) if allow_web else None)
            for layer in (doc or {}).get("layers", [])[:80]:
                lname = layer.get("name", "")
                fields = "|".join(f.get("name", "") for f in layer.get("fields", []) if isinstance(f, dict))
                cls, rel = classify_layer(lname, fields)
                if rel == "LOW" and "petropolis" not in norm_text(lname):
                    continue
                extent = layer.get("extent", {})
                sr = ""
                if isinstance(extent, dict):
                    sr = str((extent.get("spatialReference") or {}).get("wkid", ""))
                lid = str(layer.get("id", ""))
                rows.append({
                    "layer_search_id": f"LAYER_v1ur_{len(rows):04d}",
                    "event_id": "PET_2022_02_15" if "petropolis" in norm_text(lname) or rel != "LOW" else "",
                    "service_url": service_url,
                    "layer_url": f"{service_url}/{lid}" if lid else service_url,
                    "layer_name": lname,
                    "layer_id": lid,
                    "geometry_type": layer.get("geometryType", ""),
                    "spatial_reference": sr,
                    "extent": json.dumps(extent, ensure_ascii=True)[:250] if extent else "",
                    "fields": fields,
                    "matched_terms_hash": hash_terms(token_terms(lname, fields)),
                    "layer_class": cls,
                    "candidate_relevance": rel,
                    "can_query_features": bool_text(bool(layer.get("geometryType"))),
                    "sample_downloaded": "false",
                    "can_create_ground_reference": "false",
                    "notes": "Metadata only; no feature download.",
                })
    out = os.path.join(DATASET_DIR, "v1ur_petropolis_geosgb_layer_search_registry.csv")
    write_csv(out, LAYER_COLUMNS, rows)
    print(f"[v1ur GeoSGB layers] rows={len(rows)} -> {out}")
    return rows


def run_public_query_runner(allow_web=False, timeout=30, fixture_html=""):
    seeds = load_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_geodata_signal_seed_registry.csv"))
    targets = [
        ("rigeo.sgb.gov.br", rigeo_item_url()),
        ("geoportal.sgb.gov.br", "https://geoportal.sgb.gov.br/server/rest/services"),
        ("petropolis.rj.gov.br", "https://www.petropolis.rj.gov.br"),
    ]
    rows = []
    for seed in seeds[:12]:
        terms = seed.get("seed_terms", "")
        for domain, url in targets:
            html = ""
            ctype = ""
            status = "NOT_REQUESTED"
            if fixture_html and os.path.exists(fixture_html):
                html = open(fixture_html, encoding="utf-8").read()
                status = "200"
                ctype = "text/html"
            elif allow_web and allowed_url(url):
                html, ctype, _ = fetch_text(url, timeout=timeout)
                status = "200" if html else "FETCH_FAILED"
            if not allow_web and not fixture_html:
                rows.append({
                    "query_id": f"QUERY_v1ur_{len(rows):04d}", "event_id": seed.get("event_id", ""),
                    "seed_id": seed.get("seed_id", ""), "query_terms": terms,
                    "target_domain": domain, "result_url": url, "http_status": "NOT_REQUESTED",
                    "content_type": "", "link_text_hash": "", "detected_artifact_type": "html",
                    "candidate_relevance": "LOW", "public_access_status": "DRY_RUN",
                    "blocking_reason": "allow_web_not_enabled", "notes": "Dry-run public query row.",
                })
                continue
            title, links = parse_links(html, url)
            for link, text in links[:80]:
                if not allowed_url(link):
                    continue
                fmt = classify_format(link, text)
                if fmt == "html" and not any(t in norm_text(f"{link} {text}") for t in ["arcgis", "featureserver", "mapserver", "geoserver", "wfs"]):
                    continue
                rows.append({
                    "query_id": f"QUERY_v1ur_{len(rows):04d}",
                    "event_id": seed.get("event_id", ""),
                    "seed_id": seed.get("seed_id", ""),
                    "query_terms": terms,
                    "target_domain": domain,
                    "result_url": link,
                    "http_status": status,
                    "content_type": ctype,
                    "link_text_hash": hash_terms(text),
                    "detected_artifact_type": fmt,
                    "candidate_relevance": "HIGH" if fmt in GEODATA_EXTS else "MEDIUM",
                    "public_access_status": "PUBLIC" if status == "200" else "UNRESOLVED",
                    "blocking_reason": "",
                    "notes": "Allowlisted public query result.",
                })
    out = os.path.join(DATASET_DIR, "v1ur_petropolis_public_query_registry.csv")
    write_csv(out, QUERY_COLUMNS, rows)
    print(f"[v1ur public query] rows={len(rows)} -> {out}")
    return rows


def add_candidate(rows, event_id, source, record_id, url, artifact, relevance, context=False):
    if not url:
        return
    text = norm_text(url)
    if "risco" in text or "suscetibilidade" in text or context:
        cls = "RISK_CONTEXT_LAYER" if "risco" in text else "SUSCEPTIBILITY_CONTEXT_LAYER" if "suscet" in text else "PDF_DOCUMENT_ONLY" if artifact == "pdf" else "GEODATA_PACKAGE_CANDIDATE"
    elif artifact in {"MapServer", "FeatureServer"}:
        cls = "ARCGIS_LAYER_METADATA_CANDIDATE"
    elif artifact == "WFS":
        cls = "WFS_LAYER_CANDIDATE"
    elif artifact in GEODATA_EXTS:
        cls = "GEODATA_PACKAGE_CANDIDATE"
    elif artifact == "pdf":
        cls = "PDF_DOCUMENT_ONLY"
    else:
        cls = "BLOCKED"
    rows.append({
        "candidate_url_id": f"CAND_v1ur_{len(rows):04d}",
        "event_id": event_id,
        "source_registry": source,
        "source_record_id": record_id,
        "url": url,
        "url_sha1_12": sha1_12(url),
        "candidate_class": cls,
        "artifact_type": artifact,
        "event_specificity": "PETROPOLIS_2022_COMPATIBLE" if event_id == "PET_2022_02_15" else "UNCONFIRMED",
        "phenomenon_specificity": "GEODATA_PATH_SIGNAL" if cls != "PDF_DOCUMENT_ONLY" else "DOCUMENT_ONLY",
        "download_priority": "1" if cls == "GEODATA_PACKAGE_CANDIDATE" else "2" if "LAYER" in cls else "9",
        "can_contain_geometry": bool_text(cls in {"GEODATA_PACKAGE_CANDIDATE", "ARCGIS_LAYER_METADATA_CANDIDATE", "WFS_LAYER_CANDIDATE", "RISK_CONTEXT_LAYER", "SUSCEPTIBILITY_CONTEXT_LAYER"}),
        "can_contain_observed_occurrence": bool_text(cls in {"GEODATA_PACKAGE_CANDIDATE", "ARCGIS_LAYER_METADATA_CANDIDATE", "WFS_LAYER_CANDIDATE"} and not ("risk" in norm_text(url) or "suscet" in norm_text(url))),
        "can_contain_context_only": bool_text("CONTEXT" in cls or cls in {"PDF_DOCUMENT_ONLY", "RISK_CONTEXT_LAYER", "SUSCEPTIBILITY_CONTEXT_LAYER"}),
        "blocking_reason": "document_only" if cls == "PDF_DOCUMENT_ONLY" else "context_only" if "CONTEXT" in cls else "",
        "notes": f"Consolidated from {source}.",
    })


def run_candidate_url_classifier():
    rows = []
    for r in load_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_sgb_bitstream_deep_registry.csv")):
        add_candidate(rows, r.get("event_id", ""), "sgb_bitstream_deep", r.get("bitstream_record_id", ""), r.get("bitstream_url", ""), r.get("format_hint", ""), r.get("event_specificity", ""))
    for r in load_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_geosgb_layer_search_registry.csv")):
        add_candidate(rows, r.get("event_id", ""), "geosgb_layer_search", r.get("layer_search_id", ""), r.get("layer_url", ""), classify_format(r.get("layer_url", ""), r.get("layer_name", "")), r.get("candidate_relevance", ""), context="CONTEXT" in r.get("layer_class", ""))
    for r in load_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_public_query_registry.csv")):
        add_candidate(rows, r.get("event_id", ""), "public_query", r.get("query_id", ""), r.get("result_url", ""), r.get("detected_artifact_type", ""), r.get("candidate_relevance", ""))
    seen = {}
    dedup = []
    for row in rows:
        key = row["url_sha1_12"]
        if key in seen:
            row["candidate_class"] = "DUPLICATE"
            row["blocking_reason"] = f"duplicate_of={seen[key]}"
        else:
            seen[key] = row["candidate_url_id"]
            dedup.append(row)
    out = os.path.join(DATASET_DIR, "v1ur_petropolis_candidate_url_registry.csv")
    write_csv(out, CANDIDATE_COLUMNS, dedup)
    print(f"[v1ur candidates] rows={len(dedup)} -> {out}")
    return dedup


def run_geodata_downloader(allow_web=False, download=False, timeout=60, max_download_mb=300, local_only_dir=None):
    raw_dir = local_only_dir or LOCAL_RAW_DIR
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(LOCAL_STAGING_DIR, exist_ok=True)
    os.makedirs(LOCAL_QUARANTINE_DIR, exist_ok=True)
    os.makedirs(LOCAL_REPORTS_DIR, exist_ok=True)
    rows, seen_hash = [], {}
    max_bytes = max_download_mb * 1024 * 1024
    candidates = [r for r in load_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_candidate_url_registry.csv")) if r.get("download_priority") in {"1", "2"} and r.get("candidate_class") not in {"PDF_DOCUMENT_ONLY", "DUPLICATE", "OFF_TARGET", "BLOCKED"}]
    for cand in candidates:
        url = cand["url"]
        ext = cand.get("artifact_type") or classify_format(url)
        base = safe_basename(url, f"candidate.{ext}")
        safe = f"{cand['event_id']}__{cand['source_registry']}__{cand['source_record_id']}__{cand['url_sha1_12']}__{base}"
        status = "MANIFEST_ONLY"
        sha = ""; size = "0"; mime = ""; dup = ""; notes = "Raw artifact path intentionally not versioned."
        if allow_web and download and allowed_url(url) and ext not in {"MapServer", "FeatureServer", "WFS"}:
            data, mime, _ = fetch_bytes(url, timeout=timeout, max_bytes=max_bytes + 1)
            if not data:
                status = "FAILED"
                notes = "Fetch failed or empty response."
            elif len(data) > max_bytes:
                status = "BLOCKED_MAX_SIZE"
            else:
                dest = os.path.join(raw_dir, safe)
                with open(dest, "wb") as f:
                    f.write(data)
                sha = hashlib.sha256(data).hexdigest()
                size = str(len(data))
                dup = seen_hash.get(sha, "")
                if not dup:
                    seen_hash[sha] = cand["candidate_url_id"]
                status = "DOWNLOADED"
        rows.append({
            "download_id": f"DL_v1ur_{len(rows):04d}",
            "event_id": cand.get("event_id", ""),
            "candidate_url_id": cand.get("candidate_url_id", ""),
            "url": url,
            "safe_filename": safe,
            "local_path_hash": hash_terms(safe),
            "sha256": sha,
            "file_size_bytes": size,
            "mime_type": mime,
            "extension": ext,
            "download_status": status,
            "duplicate_status": f"duplicate_of={dup}" if dup else "unique_or_not_downloaded",
            "license_status": "PUBLIC_SOURCE_UNVERIFIED_LICENSE",
            "notes": notes,
        })
    out = os.path.join(DATASET_DIR, "v1ur_petropolis_geodata_download_manifest.csv")
    write_csv(out, DOWNLOAD_COLUMNS, rows)
    print(f"[v1ur downloads] rows={len(rows)} downloaded={sum(1 for r in rows if r['download_status']=='DOWNLOADED')} -> {out}")
    return rows


def csv_field_flags(fields):
    lower = norm_text("|".join(fields))
    return (
        any(t in lower for t in ["data", "date", "dt"]),
        any(t in lower for t in ["fenomeno", "tipo", "ocorrencia", "desliz", "alag"]),
        any(t in lower for t in ["bairro", "local", "logradouro", "endereco"]),
        any(t in lower for t in ["lat", "lon", "latitude", "longitude", "x", "y"]),
    )


def inspect_geojson_bytes(data):
    try:
        doc = json.loads(data.decode("utf-8", errors="replace"))
    except Exception:
        return False, "", "", "0", [], ""
    features = doc.get("features", []) if doc.get("type") == "FeatureCollection" else []
    gtypes = sorted({(f.get("geometry") or {}).get("type", "") for f in features})
    fields = sorted({k for f in features for k in (f.get("properties") or {}).keys()})
    return bool(features), "|".join(gtypes), "EPSG:4326_ASSUMED_BY_GEOJSON_SPEC", str(len(features)), fields, ""


def run_geodata_inventory(local_only_dir=None):
    raw_dir = local_only_dir or LOCAL_RAW_DIR
    rows = []
    for dl in load_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_geodata_download_manifest.csv")):
        if dl.get("download_status") != "DOWNLOADED":
            continue
        path = os.path.join(raw_dir, dl.get("safe_filename", ""))
        if not os.path.exists(path):
            continue
        ext = dl.get("extension", "").lower()
        data = open(path, "rb").read()
        def add(asset_type, container, internal, has_geo, gtype, crs, count, fields, pdf=False, status="INVENTORIED", notes=""):
            has_date, has_phen, has_loc, has_coord = csv_field_flags(fields)
            rows.append({
                "inventory_id": f"INV_v1ur_{len(rows):04d}",
                "event_id": dl.get("event_id", ""),
                "download_id": dl.get("download_id", ""),
                "asset_type": asset_type,
                "container_type": container,
                "internal_path": internal,
                "has_geometry": bool_text(has_geo),
                "geometry_type": gtype,
                "crs": crs,
                "feature_count": count,
                "has_date_field": bool_text(has_date),
                "has_phenomenon_field": bool_text(has_phen),
                "has_locality_field": bool_text(has_loc),
                "has_coordinate_fields": bool_text(has_coord),
                "is_pdf_only": bool_text(pdf),
                "inventory_status": status,
                "notes": notes,
            })
        if ext == "geojson":
            has_geo, gtype, crs, count, fields, notes = inspect_geojson_bytes(data)
            add("geojson", "file", "", has_geo, gtype, crs, count, fields, False, notes=notes)
        elif ext == "csv":
            text = data.decode("utf-8", errors="replace").splitlines()
            fields = next(csv.reader([text[0]]), []) if text else []
            add("csv", "file", "", False, "", "", "0", fields, False, notes="CSV fields profiled; coordinates not inferred.")
        elif ext == "zip" or zipfile.is_zipfile(path):
            try:
                with zipfile.ZipFile(path) as zf:
                    for name in zf.namelist():
                        fmt = classify_format(name)
                        if fmt == "geojson":
                            has_geo, gtype, crs, count, fields, notes = inspect_geojson_bytes(zf.read(name))
                            add("geojson", "zip", name, has_geo, gtype, crs, count, fields, False, notes=notes)
                        elif fmt in {"shp", "gpkg", "kml", "kmz"}:
                            add(fmt, "zip", name, True, "UNREAD_METADATA", "", "0", [], False, notes="Geodata file detected in ZIP; no overlay.")
                        elif fmt == "csv":
                            first = zf.read(name).decode("utf-8", errors="replace").splitlines()
                            fields = next(csv.reader([first[0]]), []) if first else []
                            add("csv", "zip", name, False, "", "", "0", fields, False, notes="CSV in ZIP profiled.")
                        elif fmt == "pdf":
                            add("pdf", "zip", name, False, "", "", "0", [], True, notes="PDF document only.")
            except Exception:
                add("zip", "file", "", False, "", "", "0", [], False, "FAILED", "ZIP inventory failed.")
        elif ext == "pdf":
            add("pdf", "file", "", False, "", "", "0", [], True, notes="PDF document only.")
    out = os.path.join(DATASET_DIR, "v1ur_petropolis_geodata_inventory.csv")
    write_csv(out, INVENTORY_COLUMNS, rows)
    print(f"[v1ur inventory] rows={len(rows)} -> {out}")
    return rows


def run_geodata_candidate_audit():
    rows = []
    cand_by_id = {r["candidate_url_id"]: r for r in load_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_candidate_url_registry.csv"))}
    dl_by_id = {r["download_id"]: r for r in load_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_geodata_download_manifest.csv"))}
    for inv in load_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_geodata_inventory.csv")):
        dl = dl_by_id.get(inv.get("download_id", ""), {})
        cand = cand_by_id.get(dl.get("candidate_url_id", ""), {})
        text = norm_text(f"{cand.get('url','')} {inv.get('internal_path','')} {inv.get('asset_type','')}")
        if inv.get("is_pdf_only") == "true":
            cls = "DOCUMENT_ONLY"
            blocker = "document_only"
        elif "risco" in text or "suscet" in text:
            cls = "RISK_CONTEXT_LAYER" if "risco" in text else "SUSCEPTIBILITY_CONTEXT_LAYER"
            blocker = "context_layer_not_observed_occurrence"
        elif inv.get("has_geometry") == "true" and ("cicatriz" in text or "desliz" in text):
            cls = "LANDSLIDE_SCAR_CANDIDATE"
            blocker = "REVIEW_ONLY_NO_OVERLAY"
        elif inv.get("has_geometry") == "true" and ("inund" in text or "alag" in text):
            cls = "FLOOD_AREA_CANDIDATE"
            blocker = "REVIEW_ONLY_NO_OVERLAY"
        elif inv.get("has_geometry") == "true":
            cls = "OBSERVED_OCCURRENCE_GEOMETRY_CANDIDATE"
            blocker = "INSUFFICIENT_EVENT_PHENOMENON_METADATA"
        else:
            cls = "INSUFFICIENT_METADATA"
            blocker = "geometry_missing_or_unreadable"
        rows.append({
            "geodata_candidate_id": f"GEOAUD_v1ur_{len(rows):04d}",
            "event_id": inv.get("event_id", ""),
            "download_id": inv.get("download_id", ""),
            "inventory_id": inv.get("inventory_id", ""),
            "candidate_class": cls,
            "public_official_source": bool_text(allowed_url(cand.get("url", ""))),
            "event_specificity": cand.get("event_specificity", "UNCONFIRMED"),
            "phenomenon_class": "LANDSLIDE_OR_MASS_MOVEMENT" if "desliz" in text or "cicatriz" in text else "FLOOD_OR_INUNDATION" if "inund" in text or "alag" in text else "UNCONFIRMED",
            "temporal_compatibility": "UNCONFIRMED",
            "has_geometry": inv.get("has_geometry", "false"),
            "geometry_type": inv.get("geometry_type", ""),
            "crs_status": "CRS_PRESENT" if inv.get("crs") else "CRS_MISSING",
            "context_only_status": bool_text(cls in {"RISK_CONTEXT_LAYER", "SUSCEPTIBILITY_CONTEXT_LAYER", "DOCUMENT_ONLY"}),
            "can_advance_to_overlay_preflight": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "blocker": blocker,
            "required_next_action": "v1us - Petropolis Geodata Candidate Overlay Preflight" if cls in {"LANDSLIDE_SCAR_CANDIDATE", "FLOOD_AREA_CANDIDATE", "OBSERVED_OCCURRENCE_GEOMETRY_CANDIDATE"} else "v1us - Petropolis Risk/Susceptibility Context Layer Consolidation" if "CONTEXT" in cls else "v1us - Event-Patch Package Linkage Engine",
            "notes": "Review-only geodata candidate audit; no overlay or label.",
        })
    out = os.path.join(DATASET_DIR, "v1ur_petropolis_geodata_candidate_audit.csv")
    write_csv(out, AUDIT_COLUMNS, rows)
    print(f"[v1ur geodata audit] rows={len(rows)} -> {out}")
    return rows


def run_event_status_updater():
    cands = load_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_candidate_url_registry.csv"))
    downloads = load_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_geodata_download_manifest.csv"))
    inv = load_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_geodata_inventory.csv"))
    audits = load_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_geodata_candidate_audit.csv"))
    rows, blockers = [], []
    for event_id in ["PET_2022_02_15", "PET_2024_03_21_28"]:
        ev_cands = [r for r in cands if r.get("event_id") == event_id]
        ev_dl = [r for r in downloads if r.get("event_id") == event_id and r.get("download_status") == "DOWNLOADED"]
        ev_inv = [r for r in inv if r.get("event_id") == event_id]
        ev_aud = [r for r in audits if r.get("event_id") == event_id]
        obs = any(r.get("candidate_class") in {"OBSERVED_OCCURRENCE_GEOMETRY_CANDIDATE", "LANDSLIDE_SCAR_CANDIDATE", "FLOOD_AREA_CANDIDATE", "FIELD_MAPPING_CANDIDATE"} for r in ev_aud)
        context = any("CONTEXT" in r.get("candidate_class", "") for r in ev_aud) or any(r.get("candidate_class") in {"RISK_CONTEXT_LAYER", "SUSCEPTIBILITY_CONTEXT_LAYER"} for r in ev_cands)
        if obs:
            status = "OBSERVED_GEOMETRY_CANDIDATE_FOUND"
            blocker = "REVIEW_ONLY_OVERLAY_PREFLIGHT_NOT_EXECUTED"
            action = "v1us - Petropolis Geodata Candidate Overlay Preflight"
        elif ev_inv and context:
            status = "RISK_CONTEXT_LAYER_ONLY"
            blocker = "CONTEXT_LAYER_NOT_OBSERVED_OCCURRENCE"
            action = "v1us - Petropolis Risk/Susceptibility Context Layer Consolidation"
        elif ev_dl:
            status = "DOCUMENT_ONLY_NO_GEODATA"
            blocker = "GEOMETRY_STILL_MISSING"
            action = "v1us - Event-Patch Package Linkage Engine"
        elif ev_cands:
            status = "PUBLIC_GEODATA_CANDIDATE_FOUND"
            blocker = "DOWNLOAD_OR_METADATA_REVIEW_PENDING"
            action = "v1us - Event-Patch Package Linkage Engine"
        else:
            status = "BLOCKED_NO_PUBLIC_PATH"
            blocker = "NO_PUBLIC_PATH_RECOVERED"
            action = "v1us - Curitiba Event Registry and Public Source Discovery"
        rows.append({
            "event_id": event_id,
            "v1ur_status": status,
            "has_public_path_candidate": bool_text(bool(ev_cands)),
            "has_downloaded_artifact": bool_text(bool(ev_dl)),
            "has_geodata_inventory": bool_text(any(r.get("has_geometry") == "true" for r in ev_inv)),
            "has_observed_geometry": bool_text(obs),
            "has_context_layer_only": bool_text(context and not obs),
            "ground_truth_operational": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "can_advance_to_overlay_preflight": "false",
            "main_blocker": blocker,
            "recommended_next_action": action,
            "notes": "v1ur status only; prior registries unchanged.",
        })
        for gate, ok, reason in [
            ("public_path_candidate", bool(ev_cands), "no_public_path_candidate"),
            ("downloaded_artifact", bool(ev_dl), "download_not_available_or_not_requested"),
            ("geodata_inventory", any(r.get("has_geometry") == "true" for r in ev_inv), "geodata_inventory_missing"),
            ("observed_geometry", obs, "observed_geometry_not_confirmed"),
            ("overlay_preflight", False, "overlay_forbidden_in_v1ur"),
            ("ground_reference", False, "ground_reference_forbidden_in_v1ur"),
            ("training_label", False, "training_label_forbidden_in_v1ur"),
        ]:
            blockers.append({
                "blocker_id": f"BLOCK_v1ur_{event_id}_{gate}",
                "event_id": event_id,
                "gate": gate,
                "gate_status": "PASS_REVIEW_ONLY" if ok else "BLOCKED",
                "blocking_reason": "" if ok else reason,
                "ground_truth_operational": "false",
                "can_create_ground_reference": "false",
                "can_create_training_label": "false",
                "notes": "Ground reference remains blocked.",
            })
    write_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_event_status_registry.csv"), EVENT_STATUS_COLUMNS, rows)
    write_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_ground_reference_blocker_matrix.csv"), BLOCKER_COLUMNS, blockers)
    print(f"[v1ur event status] rows={len(rows)} blockers={len(blockers)}")
    return rows


def write_policy_configs():
    configs = {
        "v1ur_petropolis_geodata_recovery_policy.yaml": ["protocol_version: v1ur", "status_max: PETROPOLIS_PUBLIC_GEODATA_PATH_CANDIDATE_FOR_REVIEW", "geodata_path_recovery_only: true"],
        "v1ur_petropolis_allowed_domains.yaml": ["protocol_version: v1ur", "allowed_domains:"] + [f"  - {d}" for d in ALLOWED_DOMAINS],
        "v1ur_petropolis_query_templates.yaml": ["protocol_version: v1ur", "templates:", "  - Petropolis SGB RIGeo SIG shapefile geodados", "  - Petropolis cicatriz deslizamento GeoSGB"],
        "v1ur_petropolis_candidate_url_policy.yaml": ["protocol_version: v1ur", "deduplicate_by: url_sha1_12", "risk_susceptibility_context_only: true"],
        "v1ur_petropolis_download_policy.yaml": ["protocol_version: v1ur", "raw_storage_scope: local_only", "max_download_mb_default: 300", "overwrite_by_basename: false"],
        "v1ur_petropolis_geodata_audit_policy.yaml": ["protocol_version: v1ur", "can_create_ground_reference: false", "can_create_training_label: false", "overlay_allowed: false"],
    }
    for name, lines in configs.items():
        write_text(os.path.join(CONFIG_DIR, name), lines)


def choose_next_action(status_rows):
    if any(r.get("has_observed_geometry") == "true" for r in status_rows):
        return "v1us - Petropolis Geodata Candidate Overlay Preflight"
    if any(r.get("has_context_layer_only") == "true" for r in status_rows):
        return "v1us - Petropolis Risk/Susceptibility Context Layer Consolidation"
    if any(r.get("has_public_path_candidate") == "true" for r in status_rows):
        return "v1us - Event-Patch Package Linkage Engine"
    return "v1us - Curitiba Event Registry and Public Source Discovery"


def run_completion_report():
    write_policy_configs()
    seeds = load_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_geodata_signal_seed_registry.csv"))
    related = load_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_rigeo_related_item_registry.csv"))
    bits = load_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_sgb_bitstream_deep_registry.csv"))
    layers = load_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_geosgb_layer_search_registry.csv"))
    queries = load_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_public_query_registry.csv"))
    cands = load_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_candidate_url_registry.csv"))
    downloads = load_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_geodata_download_manifest.csv"))
    inv = load_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_geodata_inventory.csv"))
    audits = load_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_geodata_candidate_audit.csv"))
    status_rows = load_csv(os.path.join(DATASET_DIR, "v1ur_petropolis_event_status_registry.csv"))
    next_action = choose_next_action(status_rows)
    write_csv(os.path.join(DATASET_DIR, "v1ur_next_actions_registry.csv"), NEXT_COLUMNS, [{
        "action_id": "ACT_v1ur_0000", "event_id": "PET_2022_02_15" if "Petropolis" in next_action else "",
        "action_type": "PROGRAMMING_DEEPENING", "priority": "1",
        "description": next_action, "target": "PET", "status": "PENDING",
        "notes": "Selected from v1ur geodata path gates; still non-operational.",
    }])
    manifest = []
    for idx, path in enumerate(V1UR_ARTIFACTS):
        exists = os.path.exists(path)
        manifest.append({
            "artifact_id": f"ART_v1ur_{idx:04d}", "artifact_path": path.replace("\\", "/"),
            "artifact_type": "config" if path.startswith("configs/") else "doc" if path.startswith("docs/") else "dataset",
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha256_file(path)[:16] if exists else "MISSING",
            "file_size_bytes": str(os.path.getsize(path) if exists else 0),
            "is_versionable": bool_text(exists),
            "reason": "Safe v1ur engineering artifact" if exists else "File not found",
        })
    write_csv(os.path.join(DATASET_DIR, "v1ur_versionable_artifacts_manifest.csv"), MANIFEST_COLUMNS, manifest)
    downloaded = sum(1 for r in downloads if r.get("download_status") == "DOWNLOADED")
    geodata = sum(1 for r in inv if r.get("has_geometry") == "true")
    observed = sum(1 for r in audits if r.get("candidate_class") in {"OBSERVED_OCCURRENCE_GEOMETRY_CANDIDATE", "LANDSLIDE_SCAR_CANDIDATE", "FLOOD_AREA_CANDIDATE"})
    pet2022 = next((r for r in status_rows if r.get("event_id") == "PET_2022_02_15"), {})
    pet2024 = next((r for r in status_rows if r.get("event_id") == "PET_2024_03_21_28"), {})
    write_text(os.path.join(DOCS_DIR, "protocolo_c_v1ur_petropolis_public_geodata_path_recovery.md"), [
        "# Protocolo C v1ur - Petropolis Public Geodata Path Recovery", "",
        "## Engineering Scope", "- Converts v1uq missing-geodata signals into public path recovery seeds.",
        "- Resolves allowlisted public item pages, bitstreams, service metadata, candidate URLs, downloads and inventories.",
        "- Does not execute overlay, infer coordinates, create labels, create ground truth, or create ground reference.",
    ])
    report = [
        "# Relatorio tecnico v1ur - Petropolis Public Geodata Path Recovery", "",
        f"- missing_geodata_signals_used: {len(seeds)}",
        f"- related_items_found: {len(related)}",
        f"- bitstreams_found: {len(bits)}",
        f"- geosgb_layers_found: {len(layers)}",
        f"- public_query_rows: {len(queries)}",
        f"- candidate_urls: {len(cands)}",
        f"- downloads_realized: {downloaded}",
        f"- geodata_inventory_rows: {len(inv)}",
        f"- geodata_with_geometry_rows: {geodata}",
        f"- observed_geometry_review_candidates: {observed}",
        f"- PET_2022_status: {pet2022.get('v1ur_status', '')}",
        f"- PET_2024_status: {pet2024.get('v1ur_status', '')}",
        f"- next_programming_step: {next_action}",
        "", "## Guardrails",
        "- ground_truth_operational=false", "- can_create_ground_reference=false",
        "- can_create_training_label=false", "- can_reopen_protocol_b=false",
        "- dino_usage=SUPPORT_ONLY", "- no_overlay_executed=true",
        "- no_coordinates_invented=true", "- patch_bound_truth=false",
        "- operational_validation=false", "- public_official_discovery=true",
        "- geodata_path_recovery_only=true",
    ]
    write_text(os.path.join(DOCS_DIR, "protocolo_c_relatorio_v1ur_petropolis_public_geodata_path_recovery.md"), report)
    write_text(os.path.join(DOCS_DIR, "protocolo_c_status_atual_v1ur.md"), [
        "# Status Atual - Protocolo C v1ur", "", f"status_max={MAX_STATUS}",
        f"PET_2022_02_15={pet2022.get('v1ur_status', '')}",
        f"PET_2024_03_21_28={pet2024.get('v1ur_status', '')}",
        f"candidate_urls={len(cands)}", f"downloads_realized={downloaded}",
        f"geodata_with_geometry_rows={geodata}", f"observed_geometry_review_candidates={observed}",
        f"recommended_next_action={next_action}",
        "ground_truth_operational=false", "can_create_ground_reference=false",
        "can_create_training_label=false", "can_reopen_protocol_b=false",
        "dino_usage=SUPPORT_ONLY", "no_overlay_executed=true",
        "no_coordinates_invented=true", "patch_bound_truth=false",
        "operational_validation=false", "public_official_discovery=true",
        "geodata_path_recovery_only=true",
    ])
    print(f"[v1ur completion] next_action={next_action}")
    return {"seeds": len(seeds), "candidate_urls": len(cands), "downloads": downloaded, "geodata": geodata, "observed": observed, "next_action": next_action}


def parser_for(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--allow-web", action="store_true")
    p.add_argument("--download", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--max-download-mb", type=int, default=300)
    p.add_argument("--local-only-dir", default="")
    p.add_argument("--fixture-html", default="")
    p.add_argument("--services-fixture", default="")
    p.add_argument("--layers-fixture", default="")
    return p


def main_for(kind):
    args = parser_for(f"v1ur {kind}").parse_args()
    allow_web = bool(args.allow_web and not args.dry_run)
    if kind == "geodata_signal_seed_builder":
        return run_geodata_signal_seed_builder()
    if kind == "rigeo_related_item_resolver":
        return run_rigeo_related_item_resolver(allow_web=allow_web, timeout=args.timeout, fixture_html=args.fixture_html)
    if kind == "sgb_bitstream_deep_resolver":
        return run_sgb_bitstream_deep_resolver(allow_web=allow_web, timeout=args.timeout, fixture_html=args.fixture_html)
    if kind == "geosgb_layer_search":
        return run_geosgb_layer_search(allow_web=allow_web, timeout=args.timeout, services_fixture=args.services_fixture, layers_fixture=args.layers_fixture)
    if kind == "public_query_runner":
        return run_public_query_runner(allow_web=allow_web, timeout=args.timeout, fixture_html=args.fixture_html)
    if kind == "candidate_url_classifier":
        return run_candidate_url_classifier()
    if kind == "geodata_downloader":
        return run_geodata_downloader(allow_web=allow_web, download=args.download, timeout=args.timeout, max_download_mb=args.max_download_mb, local_only_dir=args.local_only_dir or None)
    if kind == "geodata_inventory":
        return run_geodata_inventory(local_only_dir=args.local_only_dir or None)
    if kind == "geodata_candidate_audit":
        return run_geodata_candidate_audit()
    if kind == "event_status_updater":
        return run_event_status_updater()
    if kind == "completion_report":
        return run_completion_report()
    raise ValueError(kind)
