#!/usr/bin/env python3
"""v1up Petropolis public geometry deepening.

Focused, source-specific, non-operational pipeline for public official or
institutional evidence discovery. Raw downloads are kept outside versionable
outputs, and every promotion gate fails closed.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import unicodedata
import zipfile
from collections import Counter
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

try:
    import urllib.request
except ImportError:  # pragma: no cover
    urllib = None

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

PROTOCOL_VERSION = "v1up"
DATASET_DIR = "datasets/protocolo_c"
DOCS_DIR = "docs/metodologia_cientifica"
CONFIG_DIR = "configs/protocolo_c"
LOCAL_RAW_DIR = "local_only/protocolo_c/petropolis_public_geometry/raw/v1up"
LOCAL_STAGING_DIR = "local_only/protocolo_c/petropolis_public_geometry/staging/v1up"
LOCAL_QUARANTINE_DIR = "local_only/protocolo_c/petropolis_public_geometry/quarantine/v1up"
LOCAL_REPORTS_DIR = "local_only/protocolo_c/petropolis_public_geometry/reports/v1up"
MAX_STATUS = "PETROPOLIS_PUBLIC_GEOMETRY_CANDIDATE_FOR_REVIEW"

PET_EVENTS = [
    {
        "event_id": "PET_2022_02_15",
        "start_date": "2022-02-15",
        "end_date": "2022-02-15",
        "event_terms": "Petropolis|Quitandinha|Rio Quitandinha|15/02/2022|fevereiro 2022",
    },
    {
        "event_id": "PET_2024_03_21_28",
        "start_date": "2024-03-21",
        "end_date": "2024-03-28",
        "event_terms": "Petropolis|Quitandinha|marco 2024|chuvas 2024",
    },
]

SOURCE_COLUMNS = [
    "target_id", "event_id", "source_id", "source_name", "source_type",
    "base_url", "query_terms", "expected_artifact_types", "priority",
    "can_contain_observed_geometry", "can_contain_phenomenon_separation",
    "notes",
]

RIGEO_COLUMNS = [
    "rigeo_record_id", "event_id", "item_url", "title",
    "publication_year", "bitstream_url", "bitstream_name", "format_hint",
    "is_public", "is_event_specific", "is_geometry_candidate",
    "is_context_only", "blocking_reason", "notes",
]

GEOSGB_COLUMNS = [
    "service_record_id", "event_id", "service_url", "service_type",
    "layer_id", "layer_name", "geometry_type", "spatial_reference",
    "fields", "extent", "relevance_score", "is_event_specific",
    "is_observed_occurrence_candidate", "is_susceptibility_or_risk_context",
    "blocking_reason", "notes",
]

PORTAL_COLUMNS = [
    "portal_record_id", "event_id", "source_id", "source_name", "page_url",
    "title", "artifact_url", "artifact_name", "format_hint",
    "is_public", "requires_authentication", "is_event_specific",
    "is_geometry_candidate", "is_locality_only", "is_context_only",
    "blocking_reason", "notes",
]

CEMADEN_COLUMNS = [
    "cemaden_record_id", "event_id", "source_url", "title",
    "artifact_url", "artifact_name", "format_hint", "evidence_class",
    "is_public", "is_event_specific", "is_geometry_candidate",
    "blocking_reason", "notes",
]

COPERNICUS_COLUMNS = [
    "activation_record_id", "event_id", "source_id", "activation_url",
    "activation_title", "activation_date", "product_name", "product_date",
    "artifact_url", "artifact_type", "is_public", "is_event_specific",
    "is_off_target", "is_quickview", "is_vector_package_candidate",
    "blocking_reason", "notes",
]

DOWNLOAD_COLUMNS = [
    "download_id", "event_id", "source_id", "record_id", "url",
    "url_sha1_12", "safe_filename", "basename", "format_hint",
    "download_status", "downloaded", "sha256", "size_bytes",
    "duplicate_of_sha256", "collision_status", "storage_scope",
    "blocking_reason", "notes",
]

INVENTORY_COLUMNS = [
    "asset_id", "event_id", "source_id", "record_id", "safe_filename",
    "format_hint", "sha256", "size_bytes", "artifact_class",
    "contained_files", "has_pdf_text", "has_internal_links",
    "has_geodata", "geometry_type", "crs", "feature_count", "bounds",
    "fields", "has_date_field", "has_phenomenon_field",
    "has_locality_field", "has_coordinate_fields", "inventory_status",
    "notes",
]

PHENOMENON_COLUMNS = [
    "phenomenon_id", "event_id", "asset_id", "source_id",
    "phenomenon_class", "flood_signal", "landslide_signal",
    "mixed_signal", "separation_status", "evidence_strength",
    "can_support_phenomenon_gate", "can_create_ground_reference",
    "can_create_training_label", "notes",
]

AUDIT_COLUMNS = [
    "audit_id", "event_id", "asset_id", "source_id", "candidate_type",
    "public_official_traceable", "event_specificity",
    "event_date_available", "phenomenon_available",
    "phenomenon_separated", "geometry_or_coordinates_available",
    "crs_available", "not_context_only", "no_overlay_executed",
    "label_forbidden", "audit_status", "status_max",
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "can_reopen_protocol_b", "dino_usage",
    "no_coordinates_invented", "patch_bound_truth",
    "operational_validation", "formal_request_required",
    "public_official_discovery", "blocking_reason", "notes",
]

EVENT_STATUS_COLUMNS = [
    "event_id", "v1up_status", "has_public_artifact",
    "has_downloaded_artifact", "has_geodata", "has_observed_geometry_candidate",
    "phenomenon_separation_status", "ground_truth_operational",
    "can_create_ground_reference", "can_create_training_label",
    "can_advance_to_overlay_preflight", "main_blocker",
    "recommended_next_action", "notes",
]

BLOCKER_COLUMNS = [
    "blocker_id", "event_id", "gate", "gate_status", "blocking_reason",
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "notes",
]

NEXT_ACTION_COLUMNS = [
    "action_id", "event_id", "action_type", "priority", "description",
    "target", "status", "notes",
]

MANIFEST_COLUMNS = [
    "artifact_id", "artifact_path", "artifact_type", "protocol_version",
    "sha256_prefix", "file_size_bytes", "is_versionable", "reason",
]

V1UP_ARTIFACTS = [
    "configs/protocolo_c/v1up_petropolis_source_targets.yaml",
    "configs/protocolo_c/v1up_petropolis_allowed_domains.yaml",
    "configs/protocolo_c/v1up_petropolis_download_policy.yaml",
    "configs/protocolo_c/v1up_petropolis_phenomenon_terms.yaml",
    "configs/protocolo_c/v1up_petropolis_observed_geometry_policy.yaml",
    "configs/protocolo_c/v1up_petropolis_candidate_scoring_policy.yaml",
    "datasets/protocolo_c/v1up_petropolis_source_target_registry.csv",
    "datasets/protocolo_c/v1up_petropolis_sgb_rigeo_registry.csv",
    "datasets/protocolo_c/v1up_petropolis_geosgb_service_registry.csv",
    "datasets/protocolo_c/v1up_petropolis_rj_public_portal_registry.csv",
    "datasets/protocolo_c/v1up_petropolis_cemaden_registry.csv",
    "datasets/protocolo_c/v1up_petropolis_copernicus_charter_registry.csv",
    "datasets/protocolo_c/v1up_petropolis_download_manifest.csv",
    "datasets/protocolo_c/v1up_petropolis_artifact_inventory.csv",
    "datasets/protocolo_c/v1up_petropolis_phenomenon_separation_registry.csv",
    "datasets/protocolo_c/v1up_petropolis_observed_geometry_candidate_audit.csv",
    "datasets/protocolo_c/v1up_petropolis_event_status_registry.csv",
    "datasets/protocolo_c/v1up_petropolis_ground_reference_blocker_matrix.csv",
    "datasets/protocolo_c/v1up_next_actions_registry.csv",
    "datasets/protocolo_c/v1up_versionable_artifacts_manifest.csv",
    "docs/metodologia_cientifica/protocolo_c_v1up_petropolis_public_geometry_deepening.md",
    "docs/metodologia_cientifica/protocolo_c_relatorio_v1up_petropolis_public_geometry_deepening.md",
    "docs/metodologia_cientifica/protocolo_c_status_atual_v1up.md",
]


def bool_text(value):
    return "true" if bool(value) else "false"


def norm_text(value):
    normalized = unicodedata.normalize("NFKD", value or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def safe_rel(path):
    return path.replace("\\", "/")


def write_csv(path, columns, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
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
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def safe_basename(url, fallback):
    parsed = urlparse(url)
    name = os.path.basename(parsed.path) or fallback
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:160] or fallback


def read_bytes_url(url, timeout=30, max_bytes=2_000_000):
    if urllib is None:
        return b""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "REV-P-Academic-Research/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read(max_bytes)
    except Exception:
        return b""


def fetch_text(url, timeout=30, max_bytes=2_000_000):
    data = read_bytes_url(url, timeout=timeout, max_bytes=max_bytes)
    if not data:
        return ""
    return data.decode("utf-8", errors="replace")


def fetch_json(url, timeout=30):
    text = fetch_text(url if "?" in url else f"{url}?f=pjson", timeout=timeout)
    if not text:
        return None
    try:
        return json.loads(text)
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
                    break
        if tag in {"title", "h1", "h2", "h3"}:
            self._title = True

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data.strip())
        if self._title and not self.title:
            txt = data.strip()
            if txt:
                self.title = txt

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            self.links.append((self._href, " ".join(t for t in self._text if t)))
            self._href = None
            self._text = []
        if tag in {"title", "h1", "h2", "h3"}:
            self._title = False


def parse_links(html, base_url):
    parser = LinkExtractor()
    parser.feed(html or "")
    return parser.title, [(urljoin(base_url, href), text) for href, text in parser.links]


def classify_format(url, name=""):
    text = f"{url} {name}".lower()
    for ext in [".geojson", ".gpkg", ".kmz", ".kml", ".shp", ".zip", ".xlsx", ".xls", ".csv", ".pdf", ".json"]:
        if ext in text:
            return ext.lstrip(".")
    if "mapserver" in text or "featureserver" in text:
        return "arcgis_rest"
    if "quickview" in text or "image" in text:
        return "quickview"
    return "html"


def is_geodata_format(fmt, name=""):
    text = f"{fmt} {name}".lower()
    return any(t in text for t in ["geojson", "gpkg", "kmz", "kml", "shp", "feature", "mapserver"])


def signal_terms(text):
    lower = norm_text(text)
    flood_terms = ["inundacao", "inundacoes", "alagamento", "alagamentos", "enchente", "enxurrada", "transbordamento", "flood"]
    landslide_terms = ["deslizamento", "deslizamentos", "movimento de massa", "movimentos de massa", "escorregamento", "cicatriz", "landslide"]
    risk_terms = ["risco", "suscetibilidade", "susceptibility", "setor de risco", "areas de risco", "risk"]
    hydro_terms = ["chuva", "precipitacao", "pluviometro", "alerta", "cemaden", "hydromet"]
    return {
        "flood": any(t in lower for t in flood_terms),
        "landslide": any(t in lower for t in landslide_terms),
        "risk": any(t in lower for t in risk_terms),
        "hydromet": any(t in lower for t in hydro_terms),
    }


def target_rows():
    sources = [
        ("SGB_RIGEO", "SGB/CPRM RIGeo", "document_repository", "https://rigeo.sgb.gov.br/handle/doc/22668", "pdf|zip|kmz|kml", "1", "true", "true", "Official SGB post-disaster item for Petropolis 2022."),
        ("GEOSGB_ARCGIS", "GeoSGB ArcGIS REST", "arcgis_rest", "https://geoportal.sgb.gov.br/server/rest/services", "MapServer|FeatureServer|fields|extent", "2", "true", "true", "SGB geospatial service inventory; no feature harvest."),
        ("DRM_RJ", "DRM-RJ public documents", "public_portal", "https://www.rj.gov.br/drm", "pdf|zip|map|metadata", "2", "true", "true", "State geohazard reports and technical documents."),
        ("PMP_DEFESA_CIVIL", "Prefeitura/Defesa Civil Petropolis", "public_portal", "https://www.petropolis.rj.gov.br/pmp/index.php/defesa-civil", "html|pdf|csv|xlsx|geojson|zip", "2", "true", "true", "Municipal bulletins and public emergency pages."),
        ("CEMADEN", "Cemaden", "hydromet_public", "https://www.gov.br/cemaden", "alert|hydromet|station|report", "3", "false", "true", "Hydrometeorological and alert support only unless observed geometry is explicit."),
        ("COPERNICUS_EMS", "Copernicus public pages", "activation_catalog", "https://global-flood.emergency.copernicus.eu/react/news/99-floods-and-landslides-in-rio-de-janeiro-state-brazil-february-to-march-2022/", "quickview|map|metadata", "3", "false", "true", "Public context pages; quickviews do not promote."),
        ("CHARTER_751", "International Charter activation 751", "activation_catalog", "https://disasterscharter.org/activations/flood-flash-in-brazil-activation-751-", "quickview|product_page|metadata", "2", "true", "true", "Real activation for Petropolis 2022; product links require audit."),
    ]
    query_base = [
        "Petropolis", "Quitandinha", "Rio Quitandinha", "15/02/2022",
        "marco 2024", "inundacao", "alagamento", "enchente",
        "transbordamento", "deslizamento", "movimento de massa",
        "cicatriz", "risco geologico", "pos-desastre", "SGB", "CPRM",
        "RIGeo", "DRM-RJ", "NADE", "Cemaden",
    ]
    rows = []
    seq = 0
    for ev in PET_EVENTS:
        for source in sources:
            rows.append({
                "target_id": f"TGT_v1up_{seq:04d}",
                "event_id": ev["event_id"],
                "source_id": source[0],
                "source_name": source[1],
                "source_type": source[2],
                "base_url": source[3],
                "query_terms": "|".join(query_base + ev["event_terms"].split("|")),
                "expected_artifact_types": source[4],
                "priority": source[5],
                "can_contain_observed_geometry": source[6],
                "can_contain_phenomenon_separation": source[7],
                "notes": source[8],
            })
            seq += 1
    return rows


def run_source_target_builder(out_path=None):
    rows = target_rows()
    out_path = out_path or os.path.join(DATASET_DIR, "v1up_petropolis_source_target_registry.csv")
    write_csv(out_path, SOURCE_COLUMNS, rows)
    print(f"[v1up source targets] rows={len(rows)} -> {out_path}")
    return rows


def run_sgb_rigeo_resolver(allow_web=False, timeout=30, fixture_html="", out_path=None):
    rows = []
    item_url = "https://rigeo.sgb.gov.br/handle/doc/22668"
    html = ""
    if fixture_html and os.path.exists(fixture_html):
        with open(fixture_html, "r", encoding="utf-8") as f:
            html = f.read()
    elif allow_web:
        html = fetch_text(item_url, timeout=timeout)

    if not html:
        rows.append({
            "rigeo_record_id": "RIGEO_v1up_0000",
            "event_id": "PET_2022_02_15",
            "item_url": item_url,
            "title": "Avaliacao tecnica pos-desastre: Petropolis, RJ",
            "publication_year": "2022",
            "bitstream_url": "",
            "bitstream_name": "",
            "format_hint": "",
            "is_public": "true",
            "is_event_specific": "true",
            "is_geometry_candidate": "false",
            "is_context_only": "false",
            "blocking_reason": "DRY_RUN_NOT_RESOLVED" if not allow_web else "ITEM_FETCH_FAILED",
            "notes": "Known public RIGeo target registered without bitstream resolution.",
        })
    else:
        title, links = parse_links(html, item_url)
        seq = 0
        for url, text in links:
            joined = f"{url} {text}".lower()
            if not any(t in joined for t in ["bitstream", "download", ".pdf", ".zip", ".kmz", ".kml", ".geojson", ".gpkg", ".shp"]):
                continue
            fmt = classify_format(url, text)
            sig = signal_terms(f"{title} {text} {url}")
            is_geo = is_geodata_format(fmt, text) or ("anexos" in joined and "zip" in fmt)
            context = (fmt == "pdf") or sig["risk"]
            rows.append({
                "rigeo_record_id": f"RIGEO_v1up_{seq:04d}",
                "event_id": "PET_2022_02_15",
                "item_url": item_url,
                "title": title or "Avaliacao tecnica pos-desastre: Petropolis, RJ",
                "publication_year": "2022",
                "bitstream_url": url,
                "bitstream_name": text or safe_basename(url, "rigeo_asset"),
                "format_hint": fmt,
                "is_public": "true",
                "is_event_specific": "true",
                "is_geometry_candidate": bool_text(is_geo),
                "is_context_only": bool_text(context and not is_geo),
                "blocking_reason": "technical_pdf_not_vector" if fmt == "pdf" else "",
                "notes": "RIGeo/SGB public item; geometry candidate remains review-only.",
            })
            seq += 1
        if not rows:
            rows.append({
                "rigeo_record_id": "RIGEO_v1up_0000",
                "event_id": "PET_2022_02_15",
                "item_url": item_url,
                "title": title or "Avaliacao tecnica pos-desastre: Petropolis, RJ",
                "publication_year": "2022",
                "bitstream_url": "",
                "bitstream_name": "",
                "format_hint": "",
                "is_public": "true",
                "is_event_specific": "true",
                "is_geometry_candidate": "false",
                "is_context_only": "false",
                "blocking_reason": "NO_BITSTREAM_LINKS_DETECTED",
                "notes": "Item resolved but no downloadable links parsed.",
            })

    rows.append({
        "rigeo_record_id": f"RIGEO_v1up_{len(rows):04d}",
        "event_id": "PET_2024_03_21_28",
        "item_url": "https://rigeo.sgb.gov.br",
        "title": "Petropolis 2024 RIGeo targeted search",
        "publication_year": "",
        "bitstream_url": "",
        "bitstream_name": "",
        "format_hint": "",
        "is_public": "true",
        "is_event_specific": "false",
        "is_geometry_candidate": "false",
        "is_context_only": "false",
        "blocking_reason": "NO_EVENT_SPECIFIC_PUBLIC_RIGEO_ITEM_RESOLVED",
        "notes": "No PET_2024 item is promoted without an explicit public item URL.",
    })
    out_path = out_path or os.path.join(DATASET_DIR, "v1up_petropolis_sgb_rigeo_registry.csv")
    write_csv(out_path, RIGEO_COLUMNS, rows)
    print(f"[v1up RIGeo] rows={len(rows)} -> {out_path}")
    return rows


def classify_layer_name(name, fields=""):
    text = f"{name} {fields}"
    sig = signal_terms(text)
    observed = (sig["landslide"] or sig["flood"]) and not sig["risk"]
    context = sig["risk"]
    score = 0
    if observed:
        score += 20
    if context:
        score += 5
    if "petropolis" in norm_text(text):
        score += 10
    return observed, context, score


def layer_meta(layer):
    lid = str(layer.get("id", ""))
    name = layer.get("name", "")
    gtype = layer.get("geometryType", "")
    sr = ""
    extent = layer.get("extent") or {}
    if isinstance(extent, dict):
        sr_info = extent.get("spatialReference") or {}
        sr = str(sr_info.get("wkid", sr_info.get("latestWkid", "")))
    fields = "|".join((f.get("name", "") for f in layer.get("fields", [])[:30] if isinstance(f, dict)))
    return lid, name, gtype, sr, fields, json.dumps(extent, ensure_ascii=True)[:250] if extent else ""


def run_geosgb_service_resolver(allow_web=False, timeout=30, services_fixture="", layers_fixture="", out_path=None):
    base = "https://geoportal.sgb.gov.br/server/rest/services"
    rows = []
    services_doc = None
    layers_doc = None
    if services_fixture and os.path.exists(services_fixture):
        with open(services_fixture, "r", encoding="utf-8") as f:
            services_doc = json.load(f)
    elif allow_web:
        services_doc = fetch_json(base, timeout=timeout)
    if layers_fixture and os.path.exists(layers_fixture):
        with open(layers_fixture, "r", encoding="utf-8") as f:
            layers_doc = json.load(f)

    if not services_doc:
        rows.append({
            "service_record_id": "GEOSGB_v1up_0000",
            "event_id": "",
            "service_url": base,
            "service_type": "ArcGIS_REST",
            "layer_id": "",
            "layer_name": "",
            "geometry_type": "",
            "spatial_reference": "",
            "fields": "",
            "extent": "",
            "relevance_score": "0",
            "is_event_specific": "false",
            "is_observed_occurrence_candidate": "false",
            "is_susceptibility_or_risk_context": "false",
            "blocking_reason": "DRY_RUN_ENDPOINT_NOT_QUERIED" if not allow_web else "ENDPOINT_UNREACHABLE",
            "notes": "Fail-closed service probe.",
        })
    else:
        seq = 0
        for svc in services_doc.get("services", [])[:80]:
            name = svc.get("name", "")
            stype = svc.get("type", "MapServer")
            service_url = f"{base.rstrip('/')}/{name}/{stype}"
            doc = layers_doc
            if doc is None and allow_web:
                doc = fetch_json(service_url, timeout=timeout)
            layers = (doc or {}).get("layers", [])
            if not layers:
                observed, context, score = classify_layer_name(name)
                if score == 0 and "petropolis" not in norm_text(name):
                    continue
                rows.append({
                    "service_record_id": f"GEOSGB_v1up_{seq:04d}",
                    "event_id": "PET_2022_02_15" if "petropolis" in norm_text(name) else "",
                    "service_url": service_url,
                    "service_type": stype,
                    "layer_id": "",
                    "layer_name": name,
                    "geometry_type": "",
                    "spatial_reference": "",
                    "fields": "",
                    "extent": "",
                    "relevance_score": str(score),
                    "is_event_specific": bool_text("petropolis" in norm_text(name)),
                    "is_observed_occurrence_candidate": bool_text(observed),
                    "is_susceptibility_or_risk_context": bool_text(context),
                    "blocking_reason": "NO_LAYER_METADATA",
                    "notes": "Service name only; no feature harvest.",
                })
                seq += 1
                continue
            for layer in layers[:80]:
                lid, lname, gtype, sr, fields, extent = layer_meta(layer)
                observed, context, score = classify_layer_name(lname, fields)
                if score == 0 and "petropolis" not in norm_text(lname):
                    continue
                rows.append({
                    "service_record_id": f"GEOSGB_v1up_{seq:04d}",
                    "event_id": "PET_2022_02_15" if "2022" in norm_text(lname) or "petropolis" in norm_text(lname) else "",
                    "service_url": service_url,
                    "service_type": stype,
                    "layer_id": lid,
                    "layer_name": lname,
                    "geometry_type": gtype,
                    "spatial_reference": sr,
                    "fields": fields,
                    "extent": extent,
                    "relevance_score": str(score),
                    "is_event_specific": bool_text("petropolis" in norm_text(lname)),
                    "is_observed_occurrence_candidate": bool_text(observed),
                    "is_susceptibility_or_risk_context": bool_text(context),
                    "blocking_reason": "risk_or_susceptibility_context" if context else "",
                    "notes": "Metadata only; no feature download.",
                })
                seq += 1
        if not rows:
            rows.append({
                "service_record_id": "GEOSGB_v1up_0000",
                "event_id": "",
                "service_url": base,
                "service_type": "ArcGIS_REST",
                "layer_id": "",
                "layer_name": "",
                "geometry_type": "",
                "spatial_reference": "",
                "fields": "",
                "extent": "",
                "relevance_score": "0",
                "is_event_specific": "false",
                "is_observed_occurrence_candidate": "false",
                "is_susceptibility_or_risk_context": "false",
                "blocking_reason": "NO_RELEVANT_PETROPOLIS_LAYER_FOUND",
                "notes": "Service queried, relevant layer not resolved.",
            })
    out_path = out_path or os.path.join(DATASET_DIR, "v1up_petropolis_geosgb_service_registry.csv")
    write_csv(out_path, GEOSGB_COLUMNS, rows)
    print(f"[v1up GeoSGB] rows={len(rows)} -> {out_path}")
    return rows


def portal_targets():
    return [
        ("DRM_RJ", "DRM-RJ", "https://www.rj.gov.br/drm/sites/default/files/arquivos_paginas/RL_09.2022.01-MTDLG-PETROPOLIS.pdf", "PET_2022_02_15"),
        ("PMP_DEFESA_CIVIL", "Prefeitura Petropolis", "https://www.petropolis.rj.gov.br/pmp/index.php/noticias/item/22231-defesa-civil-boletim-de-ocorrencias", "PET_2024_03_21_28"),
        ("PMP_DEFESA_CIVIL", "Prefeitura Petropolis", "https://www.petropolis.rj.gov.br/pmp/index.php/noticias/item/21067-nota-oficial-defesa-civil-deslizamento-quitandinha", "PET_2024_03_21_28"),
        ("PMP_TRANSPARENCIA", "Prefeitura Petropolis Transparencia Emergencial", "https://web2.petropolis.rj.gov.br/gap/transparencia-emergencial/", "PET_2022_02_15"),
    ]


def run_rj_public_portal_resolver(allow_web=False, timeout=30, fixture_html="", out_path=None):
    rows = []
    seq = 0
    fixture = ""
    if fixture_html and os.path.exists(fixture_html):
        with open(fixture_html, "r", encoding="utf-8") as f:
            fixture = f.read()
    for source_id, source_name, page_url, event_id in portal_targets():
        html = fixture or (fetch_text(page_url, timeout=timeout) if allow_web else "")
        fmt = classify_format(page_url)
        title = source_name
        links = []
        if html:
            title, links = parse_links(html, page_url)
        candidates = [(page_url, safe_basename(page_url, "portal_asset"), fmt)] if fmt != "html" else []
        for url, text in links[:100]:
            link_fmt = classify_format(url, text)
            if link_fmt in {"pdf", "zip", "csv", "xlsx", "xls", "geojson", "kmz", "kml", "gpkg"}:
                candidates.append((url, text or safe_basename(url, "portal_asset"), link_fmt))
        if not candidates:
            sig = signal_terms(f"{title} {page_url}")
            rows.append({
                "portal_record_id": f"PORTAL_v1up_{seq:04d}",
                "event_id": event_id,
                "source_id": source_id,
                "source_name": source_name,
                "page_url": page_url,
                "title": title,
                "artifact_url": "",
                "artifact_name": "",
                "format_hint": "html",
                "is_public": "true",
                "requires_authentication": "false",
                "is_event_specific": bool_text("petropolis" in norm_text(f"{title} {page_url}")),
                "is_geometry_candidate": "false",
                "is_locality_only": bool_text(sig["flood"] or sig["landslide"]),
                "is_context_only": "true",
                "blocking_reason": "DOCUMENT_OR_PAGE_ONLY_NO_VECTOR",
                "notes": "Public page resolved or targeted; no authentication.",
            })
            seq += 1
        else:
            for url, name, fmt in candidates:
                sig = signal_terms(f"{title} {name} {url}")
                is_geo = is_geodata_format(fmt, name) and not sig["risk"]
                rows.append({
                    "portal_record_id": f"PORTAL_v1up_{seq:04d}",
                    "event_id": event_id,
                    "source_id": source_id,
                    "source_name": source_name,
                    "page_url": page_url,
                    "title": title,
                    "artifact_url": url,
                    "artifact_name": name,
                    "format_hint": fmt,
                    "is_public": "true",
                    "requires_authentication": "false",
                    "is_event_specific": bool_text("petropolis" in norm_text(f"{title} {name} {url}")),
                    "is_geometry_candidate": bool_text(is_geo),
                    "is_locality_only": bool_text((sig["flood"] or sig["landslide"]) and not is_geo),
                    "is_context_only": bool_text(fmt == "pdf" or sig["risk"]),
                    "blocking_reason": "risk_or_document_context" if fmt == "pdf" or sig["risk"] else "",
                    "notes": "Public RJ/Petropolis portal artifact; no authentication.",
                })
                seq += 1
    out_path = out_path or os.path.join(DATASET_DIR, "v1up_petropolis_rj_public_portal_registry.csv")
    write_csv(out_path, PORTAL_COLUMNS, rows)
    print(f"[v1up RJ portal] rows={len(rows)} -> {out_path}")
    return rows


def run_cemaden_resolver(allow_web=False, timeout=30, fixture_html="", out_path=None):
    targets = [
        ("https://www.gov.br/cemaden", "Cemaden public portal", ""),
        ("http://www2.cemaden.gov.br/mapainterativo/#", "Cemaden interactive map reference", ""),
    ]
    rows = []
    for idx, (url, title, artifact) in enumerate(targets):
        html = ""
        if fixture_html and os.path.exists(fixture_html):
            with open(fixture_html, "r", encoding="utf-8") as f:
                html = f.read()
        elif allow_web:
            html = fetch_text(url, timeout=timeout)
        sig = signal_terms(f"{title} {url} {html[:5000]}")
        rows.append({
            "cemaden_record_id": f"CEMADEN_v1up_{idx:04d}",
            "event_id": "PET_2022_02_15" if idx == 1 else "",
            "source_url": url,
            "title": title,
            "artifact_url": artifact,
            "artifact_name": "",
            "format_hint": "html",
            "evidence_class": "temporal_hydromet_support" if sig["hydromet"] or idx == 1 else "alert_context",
            "is_public": "true",
            "is_event_specific": "false",
            "is_geometry_candidate": "false",
            "blocking_reason": "HYDROMET_OR_ALERT_CONTEXT_NOT_OBSERVED_GEOMETRY",
            "notes": "Cemaden is retained as support only unless an observed geometry artifact is explicit.",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1up_petropolis_cemaden_registry.csv")
    write_csv(out_path, CEMADEN_COLUMNS, rows)
    print(f"[v1up Cemaden] rows={len(rows)} -> {out_path}")
    return rows


def run_copernicus_charter_resolver(allow_web=False, timeout=30, fixture_html="", out_path=None):
    targets = [
        ("CHARTER_751", "https://disasterscharter.org/activations/flood-flash-in-brazil-activation-751-", "PET_2022_02_15"),
        ("COPERNICUS_REACT", "https://global-flood.emergency.copernicus.eu/react/news/99-floods-and-landslides-in-rio-de-janeiro-state-brazil-february-to-march-2022/", "PET_2022_02_15"),
        ("COPERNICUS_OBSERVER", "https://www.copernicus.eu/pt-pt/node/11470", "PET_2022_02_15"),
    ]
    rows = []
    seq = 0
    fixture = ""
    if fixture_html and os.path.exists(fixture_html):
        with open(fixture_html, "r", encoding="utf-8") as f:
            fixture = f.read()
    for source_id, url, event_id in targets:
        html = fixture or (fetch_text(url, timeout=timeout) if allow_web else "")
        title, links = parse_links(html, url) if html else (source_id, [])
        page_text = f"{title} {html[:20000]}"
        event_specific = "petropolis" in norm_text(page_text) and "2022" in norm_text(page_text)
        product_rows = []
        if source_id == "CHARTER_751" and html:
            product_titles = re.findall(r"<h3[^>]*>(.*?)</h3>\s*[^<]*(?:<[^>]+>)*\s*([0-9]{2} [A-Za-z]+ 2022)", html, flags=re.I | re.S)
            for pidx, (name, pdate) in enumerate(product_titles):
                clean_name = re.sub(r"<[^>]+>", "", name).strip()
                product_rows.append((clean_name, pdate, "", "product_page"))
        for link, text in links[:80]:
            if "sitemap" in norm_text(f"{link} {text}"):
                continue
            if any(t in norm_text(f"{link} {text}") for t in ["quickview", "product", "map", "vector", "download"]):
                product_rows.append((text or safe_basename(link, "activation_product"), "", link, classify_format(link, text)))
        if not product_rows:
            product_rows = [(title, "", "", "metadata_only")]
        for name, pdate, artifact_url, atype in product_rows:
            is_quick = atype == "quickview" or "image" in norm_text(atype) or "quickview" in norm_text(name)
            vector = any(t in norm_text(f"{atype} {artifact_url} {name}") for t in ["vector", "shp", "geojson", "kmz", "kml"])
            rows.append({
                "activation_record_id": f"ACT_v1up_{seq:04d}",
                "event_id": event_id if event_specific else "",
                "source_id": source_id,
                "activation_url": url,
                "activation_title": title,
                "activation_date": "2022-02-16" if source_id == "CHARTER_751" else "",
                "product_name": name,
                "product_date": pdate,
                "artifact_url": artifact_url,
                "artifact_type": atype,
                "is_public": "true",
                "is_event_specific": bool_text(event_specific),
                "is_off_target": bool_text(not event_specific),
                "is_quickview": bool_text(is_quick or atype in {"metadata_only", "html"}),
                "is_vector_package_candidate": bool_text(vector and event_specific),
                "blocking_reason": "quickview_or_metadata_not_promoted" if not vector else "",
                "notes": "Activation discovery only; no ID invented.",
            })
            seq += 1
    out_path = out_path or os.path.join(DATASET_DIR, "v1up_petropolis_copernicus_charter_registry.csv")
    write_csv(out_path, COPERNICUS_COLUMNS, rows)
    print(f"[v1up Copernicus/Charter] rows={len(rows)} -> {out_path}")
    return rows


def downloadable_records():
    records = []
    for r in load_csv(os.path.join(DATASET_DIR, "v1up_petropolis_sgb_rigeo_registry.csv")):
        if r.get("bitstream_url") and r.get("is_public") == "true" and r.get("event_id"):
            fmt = r.get("format_hint", "")
            if fmt in {"pdf", "zip", "kmz", "kml", "geojson", "gpkg", "csv", "xlsx", "xls"}:
                records.append(("SGB_RIGEO", r.get("rigeo_record_id", ""), r.get("event_id", ""), r.get("bitstream_url", ""), r.get("bitstream_name", ""), fmt))
    for r in load_csv(os.path.join(DATASET_DIR, "v1up_petropolis_rj_public_portal_registry.csv")):
        if r.get("artifact_url") and r.get("is_public") == "true":
            fmt = r.get("format_hint", "")
            if fmt in {"pdf", "zip", "kmz", "kml", "geojson", "gpkg", "csv", "xlsx", "xls"} and r.get("is_geometry_candidate") == "true":
                records.append((r.get("source_id", ""), r.get("portal_record_id", ""), r.get("event_id", ""), r.get("artifact_url", ""), r.get("artifact_name", ""), fmt))
    return records


def run_focused_downloader(allow_web=False, download=False, timeout=60, max_download_mb=200, out_path=None):
    os.makedirs(LOCAL_RAW_DIR, exist_ok=True)
    os.makedirs(LOCAL_STAGING_DIR, exist_ok=True)
    os.makedirs(LOCAL_QUARANTINE_DIR, exist_ok=True)
    os.makedirs(LOCAL_REPORTS_DIR, exist_ok=True)
    rows = []
    seen_hash = {}
    max_bytes = int(max_download_mb * 1024 * 1024)
    for idx, (source_id, record_id, event_id, url, name, fmt) in enumerate(downloadable_records()):
        url_hash = sha1_12(url)
        base = safe_basename(url, safe_basename(name, "asset"))
        safe_name = f"{event_id}__{source_id}__{record_id}__{url_hash}__{base}"
        dest = os.path.join(LOCAL_RAW_DIR, safe_name)
        status = "MANIFEST_ONLY"
        downloaded = "false"
        sha = ""
        size = "0"
        dup = ""
        collision = "none"
        blocker = "DOWNLOAD_NOT_REQUESTED"
        if allow_web and download:
            data = read_bytes_url(url, timeout=timeout, max_bytes=max_bytes + 1)
            if not data:
                status = "FAILED"
                blocker = "FETCH_FAILED"
            elif len(data) > max_bytes:
                status = "BLOCKED"
                blocker = "MAX_DOWNLOAD_MB_EXCEEDED"
            else:
                if os.path.exists(dest):
                    collision = "safe_filename_already_exists"
                with open(dest, "wb") as f:
                    f.write(data)
                sha = hashlib.sha256(data).hexdigest()
                size = str(len(data))
                dup = seen_hash.get(sha, "")
                if not dup:
                    seen_hash[sha] = safe_name
                status = "DOWNLOADED"
                downloaded = "true"
                blocker = ""
        rows.append({
            "download_id": f"DL_v1up_{idx:04d}",
            "event_id": event_id,
            "source_id": source_id,
            "record_id": record_id,
            "url": url,
            "url_sha1_12": url_hash,
            "safe_filename": safe_name,
            "basename": base,
            "format_hint": fmt,
            "download_status": status,
            "downloaded": downloaded,
            "sha256": sha,
            "size_bytes": size,
            "duplicate_of_sha256": dup,
            "collision_status": collision,
            "storage_scope": "RAW_LOCAL_SCOPE_REDACTED",
            "blocking_reason": blocker,
            "notes": "Raw artifact path intentionally not versioned.",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1up_petropolis_download_manifest.csv")
    write_csv(out_path, DOWNLOAD_COLUMNS, rows)
    print(f"[v1up downloader] rows={len(rows)} downloaded={sum(1 for r in rows if r['downloaded']=='true')} -> {out_path}")
    return rows


def detect_csv_fields(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
    except Exception:
        header = []
    fields = "|".join(header)
    lower = norm_text(fields)
    return fields, {
        "date": any(t in lower for t in ["data", "date", "dt"]),
        "phenomenon": any(t in lower for t in ["fenomeno", "tipo", "evento", "ocorrencia", "desliz", "alag"]),
        "locality": any(t in lower for t in ["bairro", "local", "logradouro", "endereco", "rua"]),
        "coordinate": any(t in lower for t in ["lat", "lon", "latitude", "longitude", "x", "y"]),
    }


def detect_geojson(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception:
        return "", "", "0", "", ""
    features = doc.get("features", []) if doc.get("type") == "FeatureCollection" else []
    gtypes = sorted({((ft.get("geometry") or {}).get("type") or "") for ft in features})
    fields = sorted({k for ft in features for k in (ft.get("properties") or {}).keys()})
    return "|".join(gtypes), "EPSG:4326_ASSUMED_BY_GEOJSON_SPEC", str(len(features)), "", "|".join(fields)


def inventory_one(row):
    filename = row.get("safe_filename", "")
    path = os.path.join(LOCAL_RAW_DIR, filename)
    fmt = row.get("format_hint", "").lower()
    if not os.path.exists(path):
        return None
    size = os.path.getsize(path)
    sha = sha256_file(path)
    contained = ""
    has_pdf_text = "false"
    has_links = "false"
    has_geo = "false"
    gtype = ""
    crs = ""
    feature_count = "0"
    bounds = ""
    fields = ""
    artifact_class = fmt
    notes = ""
    if fmt == "zip" or zipfile.is_zipfile(path):
        artifact_class = "zip_package"
        try:
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
            contained = "|".join(names[:80])
            has_geo = bool_text(any(classify_format(n) in {"shp", "geojson", "gpkg", "kml", "kmz"} for n in names))
            notes = "ZIP content listed; geodata not extracted into versionable output."
        except Exception as exc:
            notes = f"zip_inventory_failed:{type(exc).__name__}"
    elif fmt == "geojson" or filename.lower().endswith(".geojson"):
        artifact_class = "geojson"
        gtype, crs, feature_count, bounds, fields = detect_geojson(path)
        has_geo = bool_text(bool(gtype))
    elif fmt in {"csv", "xlsx", "xls"} or filename.lower().endswith(".csv"):
        artifact_class = "table"
        if filename.lower().endswith(".csv"):
            fields, flags = detect_csv_fields(path)
        else:
            flags = {"date": False, "phenomenon": False, "locality": False, "coordinate": False}
        return {
            "asset_id": f"ASSET_{row.get('download_id','')}",
            "event_id": row.get("event_id", ""),
            "source_id": row.get("source_id", ""),
            "record_id": row.get("record_id", ""),
            "safe_filename": filename,
            "format_hint": fmt,
            "sha256": sha,
            "size_bytes": str(size),
            "artifact_class": artifact_class,
            "contained_files": "",
            "has_pdf_text": "false",
            "has_internal_links": "false",
            "has_geodata": "false",
            "geometry_type": "",
            "crs": "",
            "feature_count": "0",
            "bounds": "",
            "fields": fields,
            "has_date_field": bool_text(flags["date"]),
            "has_phenomenon_field": bool_text(flags["phenomenon"]),
            "has_locality_field": bool_text(flags["locality"]),
            "has_coordinate_fields": bool_text(flags["coordinate"]),
            "inventory_status": "INVENTORIED",
            "notes": "Tabular fields profiled; coordinates not inferred.",
        }
    elif fmt == "pdf":
        artifact_class = "technical_pdf"
        data = open(path, "rb").read(200000)
        has_links = bool_text(b"http" in data)
        has_pdf_text = "unknown"
        notes = "PDF is document evidence, not vector geometry."
    return {
        "asset_id": f"ASSET_{row.get('download_id','')}",
        "event_id": row.get("event_id", ""),
        "source_id": row.get("source_id", ""),
        "record_id": row.get("record_id", ""),
        "safe_filename": filename,
        "format_hint": fmt,
        "sha256": sha,
        "size_bytes": str(size),
        "artifact_class": artifact_class,
        "contained_files": contained,
        "has_pdf_text": has_pdf_text,
        "has_internal_links": has_links,
        "has_geodata": has_geo,
        "geometry_type": gtype,
        "crs": crs,
        "feature_count": feature_count,
        "bounds": bounds,
        "fields": fields,
        "has_date_field": "false",
        "has_phenomenon_field": "false",
        "has_locality_field": "false",
        "has_coordinate_fields": "false",
        "inventory_status": "INVENTORIED",
        "notes": notes,
    }


def run_artifact_inventory(out_path=None):
    rows = []
    for dl in load_csv(os.path.join(DATASET_DIR, "v1up_petropolis_download_manifest.csv")):
        if dl.get("downloaded") != "true":
            continue
        item = inventory_one(dl)
        if item:
            rows.append(item)
    out_path = out_path or os.path.join(DATASET_DIR, "v1up_petropolis_artifact_inventory.csv")
    write_csv(out_path, INVENTORY_COLUMNS, rows)
    print(f"[v1up inventory] rows={len(rows)} -> {out_path}")
    return rows


def classify_phenomenon(text, has_geo=False):
    sig = signal_terms(text)
    if sig["risk"] and not has_geo:
        return "RISK_OR_SUSCEPTIBILITY_CONTEXT"
    if sig["flood"] and sig["landslide"]:
        return "MIXED_HYDRO_GEO"
    if sig["flood"] and "alag" in norm_text(text):
        return "URBAN_FLOODING"
    if sig["flood"]:
        return "FLOOD_OR_INUNDATION"
    if sig["landslide"]:
        return "LANDSLIDE_OR_MASS_MOVEMENT"
    if sig["hydromet"]:
        return "HYDROMET_CONTEXT"
    return "NO_PHENOMENON_SIGNAL"


def evidence_assets_for_phenomenon():
    rows = []
    rows.extend(load_csv(os.path.join(DATASET_DIR, "v1up_petropolis_artifact_inventory.csv")))
    for reg, rid, source_id, url_field, name_field in [
        ("v1up_petropolis_sgb_rigeo_registry.csv", "rigeo_record_id", "SGB_RIGEO", "bitstream_url", "bitstream_name"),
        ("v1up_petropolis_rj_public_portal_registry.csv", "portal_record_id", "", "artifact_url", "artifact_name"),
        ("v1up_petropolis_copernicus_charter_registry.csv", "activation_record_id", "", "artifact_url", "product_name"),
        ("v1up_petropolis_cemaden_registry.csv", "cemaden_record_id", "CEMADEN", "artifact_url", "title"),
    ]:
        for r in load_csv(os.path.join(DATASET_DIR, reg)):
            rows.append({
                "asset_id": r.get(rid, ""),
                "event_id": r.get("event_id", ""),
                "source_id": r.get("source_id", source_id) or source_id,
                "text": " ".join([r.get("title", ""), r.get("activation_title", ""), r.get(name_field, ""), r.get(url_field, ""), r.get("notes", "")]),
                "has_geodata": r.get("has_geodata", r.get("is_geometry_candidate", "false")),
            })
    return rows


def run_phenomenon_separator(out_path=None):
    rows = []
    for idx, asset in enumerate(evidence_assets_for_phenomenon()):
        if not asset.get("event_id"):
            continue
        text = asset.get("text") or " ".join(str(v) for v in asset.values())
        sig = signal_terms(text)
        cls = classify_phenomenon(text, asset.get("has_geodata") == "true")
        separated = cls in {"FLOOD_OR_INUNDATION", "URBAN_FLOODING", "LANDSLIDE_OR_MASS_MOVEMENT"}
        if cls == "MIXED_HYDRO_GEO":
            sep_status = "MIXED_NOT_SEPARATED"
        elif separated:
            sep_status = "PHENOMENON_CLASS_SEPARATED"
        elif cls.endswith("CONTEXT"):
            sep_status = "CONTEXT_ONLY"
        else:
            sep_status = "NO_USABLE_SIGNAL"
        strength = "STRONG" if separated and asset.get("has_geodata") == "true" else "MODERATE" if separated else "WEAK"
        rows.append({
            "phenomenon_id": f"PHEN_v1up_{idx:04d}",
            "event_id": asset.get("event_id", ""),
            "asset_id": asset.get("asset_id", ""),
            "source_id": asset.get("source_id", ""),
            "phenomenon_class": cls,
            "flood_signal": bool_text(sig["flood"]),
            "landslide_signal": bool_text(sig["landslide"]),
            "mixed_signal": bool_text(sig["flood"] and sig["landslide"]),
            "separation_status": sep_status,
            "evidence_strength": strength,
            "can_support_phenomenon_gate": bool_text(separated),
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": "Phenomenon support is review-only and cannot create labels.",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1up_petropolis_phenomenon_separation_registry.csv")
    write_csv(out_path, PHENOMENON_COLUMNS, rows)
    print(f"[v1up phenomenon] rows={len(rows)} -> {out_path}")
    return rows


def run_observed_geometry_candidate_audit(out_path=None):
    inv = load_csv(os.path.join(DATASET_DIR, "v1up_petropolis_artifact_inventory.csv"))
    phen = {r.get("asset_id"): r for r in load_csv(os.path.join(DATASET_DIR, "v1up_petropolis_phenomenon_separation_registry.csv"))}
    rows = []
    for idx, asset in enumerate(inv):
        ph = phen.get(asset.get("asset_id"), {})
        has_geo = asset.get("has_geodata") == "true" or asset.get("has_coordinate_fields") == "true"
        context_only = asset.get("artifact_class") in {"technical_pdf"} or ph.get("phenomenon_class") == "RISK_OR_SUSCEPTIBILITY_CONTEXT"
        separated = ph.get("separation_status") == "PHENOMENON_CLASS_SEPARATED"
        public_trace = asset.get("source_id") in {"SGB_RIGEO", "DRM_RJ", "PMP_DEFESA_CIVIL", "CHARTER_751", "COPERNICUS_EMS", "COPERNICUS_REACT"}
        candidate_type = "document_without_geometry"
        if has_geo and not context_only:
            candidate_type = "observed_geometry_or_coordinate_candidate"
        elif asset.get("artifact_class") == "zip_package" and asset.get("has_geodata") == "true":
            candidate_type = "geodata_package_for_review"
        gates_ok = public_trace and has_geo and separated and not context_only
        status = "PETROPOLIS_OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW" if gates_ok else "BLOCKED_REVIEW_ONLY"
        blockers = []
        if not has_geo:
            blockers.append("geometry_or_coordinates_missing")
        if context_only:
            blockers.append("context_only_or_pdf")
        if not separated:
            blockers.append("phenomenon_not_separated")
        rows.append({
            "audit_id": f"AUDIT_v1up_{idx:04d}",
            "event_id": asset.get("event_id", ""),
            "asset_id": asset.get("asset_id", ""),
            "source_id": asset.get("source_id", ""),
            "candidate_type": candidate_type,
            "public_official_traceable": bool_text(public_trace),
            "event_specificity": "true" if asset.get("event_id") else "false",
            "event_date_available": "true" if asset.get("event_id") == "PET_2022_02_15" else "false",
            "phenomenon_available": bool_text(bool(ph.get("phenomenon_class"))),
            "phenomenon_separated": bool_text(separated),
            "geometry_or_coordinates_available": bool_text(has_geo),
            "crs_available": bool_text(bool(asset.get("crs"))),
            "not_context_only": bool_text(not context_only),
            "no_overlay_executed": "true",
            "label_forbidden": "true",
            "audit_status": status,
            "status_max": MAX_STATUS,
            "ground_truth_operational": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "can_reopen_protocol_b": "false",
            "dino_usage": "SUPPORT_ONLY",
            "no_coordinates_invented": "true",
            "patch_bound_truth": "false",
            "operational_validation": "false",
            "formal_request_required": "false",
            "public_official_discovery": "true",
            "blocking_reason": "|".join(blockers),
            "notes": "Candidate status is review-only; no overlay or ground reference.",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1up_petropolis_observed_geometry_candidate_audit.csv")
    write_csv(out_path, AUDIT_COLUMNS, rows)
    print(f"[v1up observed audit] rows={len(rows)} -> {out_path}")
    return rows


def run_event_status_updater(out_path=None, blocker_out_path=None):
    audits = load_csv(os.path.join(DATASET_DIR, "v1up_petropolis_observed_geometry_candidate_audit.csv"))
    phens = load_csv(os.path.join(DATASET_DIR, "v1up_petropolis_phenomenon_separation_registry.csv"))
    inv = load_csv(os.path.join(DATASET_DIR, "v1up_petropolis_artifact_inventory.csv"))
    all_public = []
    for name in ["v1up_petropolis_sgb_rigeo_registry.csv", "v1up_petropolis_rj_public_portal_registry.csv", "v1up_petropolis_copernicus_charter_registry.csv", "v1up_petropolis_cemaden_registry.csv"]:
        all_public.extend(load_csv(os.path.join(DATASET_DIR, name)))
    rows = []
    blockers = []
    for ev in ["PET_2022_02_15", "PET_2024_03_21_28"]:
        ev_audits = [r for r in audits if r.get("event_id") == ev]
        ev_phens = [r for r in phens if r.get("event_id") == ev]
        ev_inv = [r for r in inv if r.get("event_id") == ev]
        public_count = sum(1 for r in all_public if r.get("event_id") == ev)
        has_geo = any(r.get("has_geodata") == "true" for r in ev_inv)
        has_candidate = any(r.get("audit_status") == "PETROPOLIS_OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW" for r in ev_audits)
        separated = any(r.get("separation_status") == "PHENOMENON_CLASS_SEPARATED" for r in ev_phens)
        mixed = any(r.get("separation_status") == "MIXED_NOT_SEPARATED" for r in ev_phens)
        if has_candidate:
            status = "PUBLIC_GEOMETRY_CANDIDATE_FOUND"
            blocker = "GROUND_REFERENCE_BLOCKED_PENDING_HUMAN_REVIEW_AND_PATCH_LINKAGE"
            next_action = "v1uq - Petropolis Observed Geometry Review Preflight"
        elif separated:
            status = "PHENOMENON_SEPARATION_IMPROVED"
            blocker = "BLOCKED_GEOMETRY_MISSING"
            next_action = "v1uq - Petropolis Phenomenon Separation Deep Audit"
        elif has_geo:
            status = "RISK_CONTEXT_ONLY"
            blocker = "BLOCKED_PHENOMENON_SEPARATION"
            next_action = "v1uq - Petropolis Phenomenon Separation Deep Audit"
        elif public_count:
            status = "DOCUMENT_ONLY_NO_GEOMETRY"
            blocker = "BLOCKED_GEOMETRY_MISSING" if not mixed else "BLOCKED_PHENOMENON_SEPARATION"
            next_action = "v1uq - Petropolis Phenomenon Separation Deep Audit"
        else:
            status = "BLOCKED_NO_PUBLIC_ARTIFACT"
            blocker = "BLOCKED_NO_PUBLIC_ARTIFACT"
            next_action = "v1uq - Curitiba Event Registry and Public Source Discovery"
        rows.append({
            "event_id": ev,
            "v1up_status": status,
            "has_public_artifact": bool_text(public_count > 0),
            "has_downloaded_artifact": bool_text(bool(ev_inv)),
            "has_geodata": bool_text(has_geo),
            "has_observed_geometry_candidate": bool_text(has_candidate),
            "phenomenon_separation_status": "IMPROVED" if separated else "MIXED_OR_CONTEXT_ONLY" if mixed else "NOT_PROVEN",
            "ground_truth_operational": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "can_advance_to_overlay_preflight": "false",
            "main_blocker": blocker,
            "recommended_next_action": next_action,
            "notes": "v1up status registry only; older registries not modified.",
        })
        for gate, ok, reason in [
            ("public_official_traceable", public_count > 0, "no_public_artifact"),
            ("observed_geometry_or_coordinates", has_geo, "geometry_or_coordinates_missing"),
            ("phenomenon_separated", separated, "phenomenon_not_separated"),
            ("overlay_preflight", False, "overlay_forbidden_in_v1up"),
            ("ground_reference", False, "ground_reference_forbidden_in_v1up"),
            ("training_label", False, "training_label_forbidden_in_v1up"),
        ]:
            blockers.append({
                "blocker_id": f"BLOCK_v1up_{ev}_{gate}",
                "event_id": ev,
                "gate": gate,
                "gate_status": "PASS_REVIEW_ONLY" if ok else "BLOCKED",
                "blocking_reason": "" if ok else reason,
                "ground_truth_operational": "false",
                "can_create_ground_reference": "false",
                "can_create_training_label": "false",
                "notes": "Gate matrix preserves non-operational status.",
            })
    out_path = out_path or os.path.join(DATASET_DIR, "v1up_petropolis_event_status_registry.csv")
    blocker_out_path = blocker_out_path or os.path.join(DATASET_DIR, "v1up_petropolis_ground_reference_blocker_matrix.csv")
    write_csv(out_path, EVENT_STATUS_COLUMNS, rows)
    write_csv(blocker_out_path, BLOCKER_COLUMNS, blockers)
    print(f"[v1up event status] rows={len(rows)} blockers={len(blockers)} -> {out_path}")
    return rows


def write_policy_configs():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    configs = {
        "v1up_petropolis_source_targets.yaml": [
            "protocol_version: v1up",
            "scope: Petropolis public official geometry deepening",
            "events: [PET_2022_02_15, PET_2024_03_21_28]",
            "status_max: PETROPOLIS_PUBLIC_GEOMETRY_CANDIDATE_FOR_REVIEW",
        ],
        "v1up_petropolis_allowed_domains.yaml": [
            "protocol_version: v1up",
            "allowed_domains:",
            "  - rigeo.sgb.gov.br",
            "  - geoportal.sgb.gov.br",
            "  - sgb.gov.br",
            "  - rj.gov.br",
            "  - petropolis.rj.gov.br",
            "  - gov.br",
            "  - global-flood.emergency.copernicus.eu",
            "  - copernicus.eu",
            "  - disasterscharter.org",
            "authentication_allowed: false",
        ],
        "v1up_petropolis_download_policy.yaml": [
            "protocol_version: v1up",
            "raw_storage_scope: local_only",
            "version_raw_artifacts: false",
            "overwrite_by_basename: false",
            "max_download_mb_default: 200",
        ],
        "v1up_petropolis_phenomenon_terms.yaml": [
            "protocol_version: v1up",
            "flood_terms: [inundacao, alagamento, enchente, enxurrada, transbordamento, flood]",
            "landslide_terms: [deslizamento, movimento de massa, escorregamento, cicatriz, landslide]",
            "context_terms: [risco, suscetibilidade, susceptibility]",
        ],
        "v1up_petropolis_observed_geometry_policy.yaml": [
            "protocol_version: v1up",
            "risk_or_susceptibility_is_context_only: true",
            "pdf_without_vector_is_document_only: true",
            "quickview_promotes: false",
            "geocode_textual_locality: false",
            "infer_coordinates: false",
        ],
        "v1up_petropolis_candidate_scoring_policy.yaml": [
            "protocol_version: v1up",
            "required_gates: [public_official_traceable, event_specificity, event_date_available, phenomenon_available, phenomenon_separated, geometry_or_coordinates_available, crs_available, not_context_only]",
            "ground_truth_operational: false",
            "can_create_ground_reference: false",
            "can_create_training_label: false",
        ],
    }
    for name, lines in configs.items():
        write_text(os.path.join(CONFIG_DIR, name), lines)


def make_manifest():
    rows = []
    for idx, path in enumerate(V1UP_ARTIFACTS):
        exists = os.path.exists(path)
        rows.append({
            "artifact_id": f"ART_v1up_{idx:04d}",
            "artifact_path": safe_rel(path),
            "artifact_type": "config" if path.startswith("configs/") else "doc" if path.startswith("docs/") else "dataset",
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha256_file(path)[:16] if exists else "MISSING",
            "file_size_bytes": str(os.path.getsize(path) if exists else 0),
            "is_versionable": bool_text(exists),
            "reason": "Safe v1up engineering artifact" if exists else "File not found",
        })
    return rows


def choose_next_action(status_rows):
    if any(r.get("has_observed_geometry_candidate") == "true" for r in status_rows):
        return "v1uq - Petropolis Observed Geometry Review Preflight"
    if any(r.get("has_public_artifact") == "true" and r.get("has_geodata") != "true" for r in status_rows):
        return "v1uq - Petropolis Phenomenon Separation Deep Audit"
    if any(r.get("phenomenon_separation_status") != "IMPROVED" for r in status_rows):
        return "v1uq - Petropolis Phenomenon Separation Deep Audit"
    if any(r.get("has_geodata") == "true" for r in status_rows):
        return "v1uq - Event-Patch Package Linkage Engine"
    return "v1uq - Curitiba Event Registry and Public Source Discovery"


def run_completion_report():
    write_policy_configs()
    status_rows = load_csv(os.path.join(DATASET_DIR, "v1up_petropolis_event_status_registry.csv"))
    downloads = load_csv(os.path.join(DATASET_DIR, "v1up_petropolis_download_manifest.csv"))
    inventory = load_csv(os.path.join(DATASET_DIR, "v1up_petropolis_artifact_inventory.csv"))
    audit = load_csv(os.path.join(DATASET_DIR, "v1up_petropolis_observed_geometry_candidate_audit.csv"))
    phens = load_csv(os.path.join(DATASET_DIR, "v1up_petropolis_phenomenon_separation_registry.csv"))
    sources = {
        "RIGeo/SGB": len(load_csv(os.path.join(DATASET_DIR, "v1up_petropolis_sgb_rigeo_registry.csv"))),
        "GeoSGB": len(load_csv(os.path.join(DATASET_DIR, "v1up_petropolis_geosgb_service_registry.csv"))),
        "RJ portals": len(load_csv(os.path.join(DATASET_DIR, "v1up_petropolis_rj_public_portal_registry.csv"))),
        "Cemaden": len(load_csv(os.path.join(DATASET_DIR, "v1up_petropolis_cemaden_registry.csv"))),
        "Copernicus/Charter": len(load_csv(os.path.join(DATASET_DIR, "v1up_petropolis_copernicus_charter_registry.csv"))),
    }
    next_action = choose_next_action(status_rows)
    next_rows = [{
        "action_id": "ACT_v1up_0000",
        "event_id": "PET_2022_02_15" if "Petropolis" in next_action else "",
        "action_type": "PROGRAMMING_DEEPENING",
        "priority": "1",
        "description": next_action,
        "target": "PET",
        "status": "PENDING",
        "notes": "Selected from v1up status gates; still non-operational.",
    }]
    write_csv(os.path.join(DATASET_DIR, "v1up_next_actions_registry.csv"), NEXT_ACTION_COLUMNS, next_rows)
    write_csv(os.path.join(DATASET_DIR, "v1up_versionable_artifacts_manifest.csv"), MANIFEST_COLUMNS, make_manifest())

    downloaded = sum(1 for r in downloads if r.get("downloaded") == "true")
    geodata = sum(1 for r in inventory if r.get("has_geodata") == "true")
    candidates = sum(1 for r in audit if r.get("audit_status") == "PETROPOLIS_OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW")
    pet2022 = next((r for r in status_rows if r.get("event_id") == "PET_2022_02_15"), {})
    pet2024 = next((r for r in status_rows if r.get("event_id") == "PET_2024_03_21_28"), {})
    sep_counts = Counter(r.get("separation_status", "") for r in phens)

    method = [
        "# Protocolo C v1up - Petropolis Public Geometry Deepening",
        "",
        "## Engineering Scope",
        "- Focused public-source discovery for Petropolis events only.",
        "- Resolves SGB/RIGeo, GeoSGB, RJ public portals, Cemaden, Copernicus and Charter records.",
        "- Keeps raw artifacts outside versionable outputs and records only sanitized filenames and source URLs.",
        "- Does not geocode textual localities, infer coordinates, execute overlay, or create labels.",
        "",
        "## Gates",
        "- Risk or susceptibility layers are context only.",
        "- Technical PDFs without vector packages are document evidence only.",
        "- Quickviews never promote to observed geometry.",
        "- Candidate status is review-only and capped at PETROPOLIS_PUBLIC_GEOMETRY_CANDIDATE_FOR_REVIEW.",
    ]
    report = [
        "# Relatorio tecnico v1up - Petropolis Public Geometry Deepening",
        "",
        f"- source_targets: {sum(sources.values())}",
        f"- downloads_realized: {downloaded}",
        f"- inventoried_artifacts: {len(inventory)}",
        f"- geodata_packages_or_files_detected: {geodata}",
        f"- observed_geometry_review_candidates: {candidates}",
        f"- PET_2022_status: {pet2022.get('v1up_status', '')}",
        f"- PET_2024_status: {pet2024.get('v1up_status', '')}",
        f"- phenomenon_separation_counts: {dict(sep_counts)}",
        f"- next_programming_step: {next_action}",
        "",
        "## Source Families Accessed",
    ]
    report += [f"- {name}: {count} registry rows" for name, count in sources.items()]
    report += [
        "",
        "## Guardrails",
        "- ground_truth_operational=false",
        "- can_create_ground_reference=false",
        "- can_create_training_label=false",
        "- can_reopen_protocol_b=false",
        "- dino_usage=SUPPORT_ONLY",
        "- no_overlay_executed=true",
        "- no_coordinates_invented=true",
        "- patch_bound_truth=false",
        "- operational_validation=false",
        "- formal_request_required=false",
        "- public_official_discovery=true",
    ]
    status_doc = [
        "# Status Atual - Protocolo C v1up",
        "",
        f"status_max={MAX_STATUS}",
        f"PET_2022_02_15={pet2022.get('v1up_status', '')}",
        f"PET_2024_03_21_28={pet2024.get('v1up_status', '')}",
        f"downloads_realized={downloaded}",
        f"geodata_detected={geodata}",
        f"observed_geometry_review_candidates={candidates}",
        f"recommended_next_action={next_action}",
        "ground_truth_operational=false",
        "can_create_ground_reference=false",
        "can_create_training_label=false",
        "can_reopen_protocol_b=false",
        "dino_usage=SUPPORT_ONLY",
        "no_overlay_executed=true",
        "no_coordinates_invented=true",
        "patch_bound_truth=false",
        "operational_validation=false",
        "formal_request_required=false",
        "public_official_discovery=true",
    ]
    write_text(os.path.join(DOCS_DIR, "protocolo_c_v1up_petropolis_public_geometry_deepening.md"), method)
    write_text(os.path.join(DOCS_DIR, "protocolo_c_relatorio_v1up_petropolis_public_geometry_deepening.md"), report)
    write_text(os.path.join(DOCS_DIR, "protocolo_c_status_atual_v1up.md"), status_doc)
    print(f"[v1up completion] next_action={next_action}")
    return {
        "downloads_realized": downloaded,
        "geodata_detected": geodata,
        "observed_geometry_review_candidates": candidates,
        "next_action": next_action,
    }


def parser_with_common(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--allow-web", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--max-download-mb", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fixture-html", default="")
    parser.add_argument("--services-fixture", default="")
    parser.add_argument("--layers-fixture", default="")
    return parser


def main_for(kind):
    parser = parser_with_common(f"v1up {kind}")
    args = parser.parse_args()
    allow_web = bool(args.allow_web and not args.dry_run)
    if kind == "source_target_builder":
        return run_source_target_builder()
    if kind == "sgb_rigeo_resolver":
        return run_sgb_rigeo_resolver(allow_web=allow_web, timeout=args.timeout, fixture_html=args.fixture_html)
    if kind == "geosgb_service_resolver":
        return run_geosgb_service_resolver(allow_web=allow_web, timeout=args.timeout, services_fixture=args.services_fixture, layers_fixture=args.layers_fixture)
    if kind == "rj_public_portal_resolver":
        return run_rj_public_portal_resolver(allow_web=allow_web, timeout=args.timeout, fixture_html=args.fixture_html)
    if kind == "cemaden_resolver":
        return run_cemaden_resolver(allow_web=allow_web, timeout=args.timeout, fixture_html=args.fixture_html)
    if kind == "copernicus_charter_resolver":
        return run_copernicus_charter_resolver(allow_web=allow_web, timeout=args.timeout, fixture_html=args.fixture_html)
    if kind == "focused_downloader":
        return run_focused_downloader(allow_web=allow_web, download=args.download, timeout=args.timeout, max_download_mb=args.max_download_mb)
    if kind == "artifact_inventory":
        return run_artifact_inventory()
    if kind == "phenomenon_separator":
        return run_phenomenon_separator()
    if kind == "observed_geometry_candidate_audit":
        return run_observed_geometry_candidate_audit()
    if kind == "event_status_updater":
        return run_event_status_updater()
    if kind == "completion_report":
        return run_completion_report()
    raise ValueError(kind)
