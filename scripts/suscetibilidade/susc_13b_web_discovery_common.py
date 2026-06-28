"""
SUSC-13B-AUTO shared web-discovery helpers (offline-first, network opt-in).

Pure Python (urllib only). Builds deterministic query plans and a registry of
candidate official endpoints (CKAN, ArcGIS REST, WFS/GeoServer, open-data
portals). Live probing is OFF by default and only runs when the environment
variable ``SUSC_13B_NETWORK=1`` is set. When network is disabled (the default,
and the case for CI / offline runs), every fetch records an explicit
``network_disabled`` status and nothing is fabricated.

Governance: every artifact stays review-only. No invented direct download URL,
no API key, no Google Maps, no generic geocoding, no aggressive scraping,
robots.txt is honored before any HTML crawl, files over 250MB and rasters are
blocked.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from susc_io import sha256_bytes  # noqa: E402

# ---------------------------------------------------------------------------
# Network policy
# ---------------------------------------------------------------------------

NETWORK_ENV = "SUSC_13B_NETWORK"
USER_AGENT = "REV-P-SUSC-13B/1.0 (review-only research; respects robots.txt)"
TIMEOUT = 20
RETRIES = 2
CRAWL_DELAY = 1.0  # minimum delay between HTML requests to the same host
MAX_BYTES = 250 * 1024 * 1024  # 250 MB hard cap (task rule 18)
PROBE_MAX_BYTES = 8 * 1024 * 1024  # metadata probes are small

RASTER_EXTS = {".tif", ".tiff", ".geotiff", ".jp2", ".img", ".nc", ".hdf",
               ".grib", ".grib2", ".ecw", ".sid", ".dem"}
EXECUTABLE_EXTS = {".exe", ".dll", ".bin", ".msi", ".bat", ".sh", ".com"}
# Vector / tabular / documentary formats we are allowed to fetch.
ALLOWED_DATA_EXTS = {".csv", ".tsv", ".xlsx", ".xls", ".geojson", ".json",
                     ".kml", ".kmz", ".wkt", ".txt", ".xml", ".gml", ".gpkg",
                     ".zip", ".pdf"}


def network_enabled() -> bool:
    return os.environ.get(NETWORK_ENV, "0").strip().lower() in {"1", "true", "yes"}


# ---------------------------------------------------------------------------
# Region / institution catalog (institutional facts, never invented downloads)
# ---------------------------------------------------------------------------

REGIONS = {
    "recife": {"municipality": "Recife", "uf": "PE", "event_type": "urban_flooding"},
    "petropolis": {"municipality": "Petropolis", "uf": "RJ", "event_type": "flash_flood"},
    "curitiba": {"municipality": "Curitiba", "uf": "PR", "event_type": "urban_flooding"},
}

# Keyword templates (FASE 1). {city} is substituted per region. site:* templates
# are kept as documented search intents, not executed as live scraping.
KEYWORD_TEMPLATES = [
    "alagamento {city}",
    "inundacao {city}",
    "enchente {city}",
    "enxurrada {city}",
    "pontos de alagamento {city}",
    "ocorrencias alagamento {city}",
    "Defesa Civil ocorrencias {city}",
    "Defesa Civil alagamento {city}",
    "areas alagadas {city}",
    "mancha de inundacao {city}",
    "flood extent {city}",
    "flood footprint {city}",
    "shapefile alagamento {city}",
    "geojson alagamento {city}",
    "csv alagamento {city}",
    "xlsx ocorrencias defesa civil {city}",
    "FeatureServer alagamento {city}",
    "MapServer alagamento {city}",
    "WFS alagamento {city}",
    "GetCapabilities alagamento {city}",
]

# site-scoped intents (documented, not executed as open scraping)
SITE_TEMPLATES = {
    "recife": [
        "site:*.gov.br alagamento Recife",
        "site:*.recife.pe.gov.br alagamento",
        "site:*.pe.gov.br Recife pontos de alagamento",
    ],
    "petropolis": [
        "site:*.gov.br alagamento Petropolis",
        "site:*.petropolis.rj.gov.br alagamento",
        "site:*.rj.gov.br Petropolis inundacao 2022 shapefile",
    ],
    "curitiba": [
        "site:*.gov.br alagamento Curitiba",
        "site:*.curitiba.pr.gov.br alagamento",
        "site:*.pr.gov.br alagamento Curitiba",
    ],
}

# Candidate official endpoints. These are well-known institutional service roots
# (already referenced by the SUSC-13A registry). They are NOT invented direct
# download files: each is probed for resources only when network is enabled.
# discovery_method drives how a probe is executed.
ENDPOINTS = [
    # ---- Recife ----
    ("recife", "Prefeitura do Recife - Dados Abertos", "http://dados.recife.pe.gov.br/api/3/action/package_search", "ckan_api"),
    ("recife", "Prefeitura do Recife - Dados Abertos (portal)", "http://dados.recife.pe.gov.br", "data_portal_html"),
    ("recife", "APAC Pernambuco", "https://www.apac.pe.gov.br", "data_portal_html"),
    ("recife", "Governo de Pernambuco - Dados Abertos", "http://www.dados.pe.gov.br/api/3/action/package_search", "ckan_api"),
    ("recife", "ESIG Recife (GeoServer/ArcGIS)", "https://www.recife.pe.gov.br/arcgis/rest/services", "arcgis_rest"),
    ("recife", "CPRM/SGB GeoSGB", "https://geoportal.sgb.gov.br/server/rest/services", "arcgis_rest"),
    ("recife", "ANA - Dados Abertos", "https://dadosabertos.ana.gov.br/api/3/action/package_search", "ckan_api"),
    ("recife", "S2iD Defesa Civil Nacional", "https://s2id.mi.gov.br", "data_portal_html"),
    ("recife", "CEMADEN", "https://www.cemaden.gov.br", "data_portal_html"),
    # ---- Petropolis ----
    ("petropolis", "Prefeitura de Petropolis", "https://www.petropolis.rj.gov.br", "data_portal_html"),
    ("petropolis", "Governo do RJ - Dados Abertos", "http://www.dados.rj.gov.br/api/3/action/package_search", "ckan_api"),
    ("petropolis", "INEA-RJ (GeoServer)", "https://inea.maps.arcgis.com", "arcgis_rest"),
    ("petropolis", "DRM-RJ", "https://www.drm.rj.gov.br", "data_portal_html"),
    ("petropolis", "CPRM/SGB GeoSGB", "https://geoportal.sgb.gov.br/server/rest/services", "arcgis_rest"),
    ("petropolis", "ANA - Dados Abertos", "https://dadosabertos.ana.gov.br/api/3/action/package_search", "ckan_api"),
    ("petropolis", "S2iD Defesa Civil Nacional", "https://s2id.mi.gov.br", "data_portal_html"),
    ("petropolis", "CEMADEN", "https://www.cemaden.gov.br", "data_portal_html"),
    # ---- Curitiba ----
    ("curitiba", "GeoCuritiba / IPPUC (ArcGIS)", "https://geocuritiba.ippuc.org.br/server/rest/services", "arcgis_rest"),
    ("curitiba", "Curitiba - Dados Abertos", "https://dadosabertos.curitiba.pr.gov.br/api/3/action/package_search", "ckan_api"),
    ("curitiba", "Prefeitura de Curitiba", "https://www.curitiba.pr.gov.br", "data_portal_html"),
    ("curitiba", "Governo do Parana - Dados Abertos", "http://www.dadosabertos.pr.gov.br/api/3/action/package_search", "ckan_api"),
    ("curitiba", "IAT/Aguas Parana (GeoServer WFS)", "https://geoportal.iat.pr.gov.br/geoserver/ows", "wfs_getcapabilities"),
    ("curitiba", "ANA - Dados Abertos", "https://dadosabertos.ana.gov.br/api/3/action/package_search", "ckan_api"),
    ("curitiba", "S2iD Defesa Civil Nacional", "https://s2id.mi.gov.br", "data_portal_html"),
    ("curitiba", "CEMADEN", "https://www.cemaden.gov.br", "data_portal_html"),
]

# Keywords that flag a resource/layer/link as flood-relevant.
FLOOD_KEYWORDS = [
    "alagamento", "alagamentos", "inundacao", "inundação", "inundacoes",
    "enchente", "enchentes", "enxurrada", "transbordamento", "area alagada",
    "areas alagadas", "mancha", "ocorrencia", "ocorrência", "ocorrencias",
    "defesa civil", "defesa-civil", "defesa_civil", "risco hidrologico",
    "hidrologico", "ponto critico", "pontos criticos", "drenagem", "flood",
    "flooding", "flood extent", "flood footprint",
]
# Keywords that downgrade relevance to risk/alert/context.
RISK_KEYWORDS = ["risco", "risk", "suscet", "suscept", "alerta", "alert",
                 "aviso", "warning", "atencao", "monitoramento", "previs",
                 "simulado", "template"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except ValueError:
        return ""


def is_official_domain(url: str) -> bool:
    d = domain_of(url)
    return d.endswith(".gov.br") or d.endswith(".org.br") or ".gov.br" in d or ".ippuc.org.br" in d


def matched_keywords(text: str) -> list[str]:
    low = (text or "").lower()
    return [k for k in FLOOD_KEYWORDS if k in low]


def looks_like_risk(text: str) -> bool:
    low = (text or "").lower()
    return any(k in low for k in RISK_KEYWORDS)


def classify_url_ext(url: str) -> str:
    low = (url or "").lower().split("?")[0]
    for e in RASTER_EXTS:
        if low.endswith(e):
            return "raster_blocked"
    for e in EXECUTABLE_EXTS:
        if low.endswith(e):
            return "executable_blocked"
    for e in sorted(ALLOWED_DATA_EXTS, key=len, reverse=True):
        if low.endswith(e):
            return e
    return "unknown_ext"


def priority_score(*, is_official: bool, has_geometry_hint: bool,
                   has_temporal_hint: bool, is_occurrence: bool,
                   is_risk_or_alert: bool, is_news: bool) -> tuple[float, str]:
    """Return (score 0..1, tier). Mirrors FASE 1 scoring rules."""
    if is_news:
        return 0.10, "context_only"
    if is_risk_or_alert:
        return 0.30, "low"
    if is_official and has_geometry_hint and is_occurrence and has_temporal_hint:
        return 0.95, "max"
    if is_official and has_geometry_hint and is_occurrence:
        return 0.80, "high"
    if is_official and is_occurrence:
        return 0.55, "medium"
    if is_official and has_geometry_hint:
        return 0.60, "medium"
    if is_official:
        return 0.45, "medium_low"
    return 0.20, "low"


# ---------------------------------------------------------------------------
# Offline-safe HTTP
# ---------------------------------------------------------------------------

def http_get(url: str, *, max_bytes: int = PROBE_MAX_BYTES,
             accept: str | None = None) -> dict:
    """Offline-safe GET. Never raises. Records an explicit status.

    When network is disabled, returns status ``network_disabled`` and no body.
    """
    rec: dict[str, object] = {
        "url": url, "status": "not_attempted", "http_code": "", "error": "",
        "content_type": "", "content_length": "", "bytes": b"", "sha256": "",
    }
    if not (url.startswith("http://") or url.startswith("https://")):
        rec["status"] = "invalid_url"
        return rec
    if not network_enabled():
        rec["status"] = "network_disabled"
        rec["error"] = f"{NETWORK_ENV} not set; live probing skipped (offline-safe)"
        return rec
    try:
        import urllib.request
    except Exception as exc:  # pragma: no cover
        rec["status"] = "failed_no_urllib"
        rec["error"] = str(exc)[:120]
        return rec
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    last = ""
    for _ in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                clen = resp.headers.get("Content-Length")
                rec["content_type"] = resp.headers.get("Content-Type", "")
                rec["content_length"] = clen or ""
                if clen and int(clen) > MAX_BYTES:
                    rec["status"] = "blocked_too_large"
                    rec["error"] = f"content-length {clen} > {MAX_BYTES}"
                    return rec
                data = resp.read(max_bytes + 1)
                if len(data) > max_bytes:
                    rec["status"] = "truncated_too_large"
                    rec["error"] = f"stream exceeded {max_bytes}"
                    return rec
                rec["status"] = "ok"
                rec["http_code"] = getattr(resp, "status", 200) or 200
                rec["bytes"] = data
                rec["sha256"] = sha256_bytes(data)
                return rec
        except Exception as exc:  # urllib.error.* and network errors
            last = str(exc)[:160]
            continue
    rec["status"] = "failed"
    rec["error"] = last or "unknown_error"
    return rec


def robots_allows(url: str, path: str = "/") -> tuple[bool, str]:
    """Best-effort robots.txt check. When network is disabled, crawling is not
    allowed (returns False). Honors a global Disallow for our UA / '*'."""
    if not network_enabled():
        return False, "network_disabled"
    base = f"{urlparse(url).scheme}://{domain_of(url)}"
    res = http_get(base + "/robots.txt", max_bytes=256 * 1024)
    if res["status"] != "ok":
        # Conservative: if robots cannot be read, do not crawl.
        return False, f"robots_unreadable:{res['status']}"
    try:
        text = res["bytes"].decode("utf-8", "ignore")  # type: ignore[union-attr]
    except Exception:
        return False, "robots_decode_error"
    disallows: list[str] = []
    active = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().lower()
        val = val.strip()
        if key == "user-agent":
            active = val == "*" or "rev-p" in val.lower()
        elif key == "disallow" and active:
            disallows.append(val)
    for rule in disallows:
        if rule and path.startswith(rule):
            return False, f"disallow:{rule}"
    return True, "allowed"


def polite_sleep() -> None:
    if network_enabled():
        time.sleep(CRAWL_DELAY)
