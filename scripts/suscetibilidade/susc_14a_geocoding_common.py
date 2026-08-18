"""
SUSC-14A shared helpers for official geocoding reference rescue (review-only).

Pure Python (urllib only). Reuses the SUSC-13B network policy (offline-first,
network opt-in via SUSC_13B_NETWORK=1). Provides:

  - a deterministic catalog of official/traceable spatial-reference endpoints
    per city (street axes, neighborhoods, address points, critical flood
    points, drainage, risk maps) used by FASE 2 discovery;
  - flood-occurrence keyword sets and exclusion sets (FASE 5);
  - robust Brazilian-CSV reading (utf-8 / latin-1) and address normalization;
  - pure-Python fuzzy similarity (difflib).

Governance: nothing here creates ground truth, training material or score v7.
Google Maps, generic geocoding and administrative centroids are never used as
observed geometry. OSM/Nominatim is not consulted; only official/traceable
layers can turn a record into a candidate.
"""

from __future__ import annotations

import csv
import io
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import susc_13b_web_discovery_common as wd  # noqa: E402  (network policy reuse)

ROOT = HERE.parents[1]

# Reuse the 13B network gate so 14A live probing also requires SUSC_13B_NETWORK=1.
network_enabled = wd.network_enabled
http_get = wd.http_get
NETWORK_ENV = wd.NETWORK_ENV

# Already-acquired official raw data (13C live) is reused as a local, traceable
# reference/occurrence source so 14A can run meaningfully offline.
LOCAL_OFFICIAL_DIR = ROOT / "datasets" / "suscetibilidade" / "observed_event_sources_susc13c_live"

# ---------------------------------------------------------------------------
# FASE 2 - official spatial-reference endpoint catalog (institutional facts).
# These are well-known institutional service roots; no direct download URL is
# invented. Each row documents what kind of spatial reference the source can
# provide, so the geocoder knows whether it yields street/address/neighborhood/
# flood-point/drainage geometry.
# ---------------------------------------------------------------------------

REFERENCE_ENDPOINTS = [
    # region, institution, source_name, source_url, source_type, access_method,
    # capability flags: street, address, neighborhood, flood_points, flood_polygons, drainage
    ("recife", "Prefeitura do Recife - Dados Abertos", "Logradouros / eixos de via (CKAN)",
     "http://dados.recife.pe.gov.br/api/3/action/package_search", "official", "ckan_api",
     dict(street=1, address=1, neighborhood=1, flood_points=1, flood_polygons=0, drainage=1)),
    ("recife", "Prefeitura do Recife - ESIG", "Bairros / lotes / logradouros (ArcGIS REST)",
     "https://www.recife.pe.gov.br/arcgis/rest/services", "official", "arcgis_rest",
     dict(street=1, address=1, neighborhood=1, flood_points=1, flood_polygons=0, drainage=1)),
    ("recife", "Prefeitura do Recife - Defesa Civil", "Pontos criticos de alagamento / areas de risco",
     "http://dados.recife.pe.gov.br/api/3/action/package_search", "official", "ckan_api",
     dict(street=0, address=1, neighborhood=1, flood_points=1, flood_polygons=0, drainage=0)),
    ("recife", "APAC Pernambuco", "Macro/microdrenagem, canais, rios (portal)",
     "https://www.apac.pe.gov.br", "state", "data_portal_html",
     dict(street=0, address=0, neighborhood=0, flood_points=1, flood_polygons=1, drainage=1)),
    # ---- Curitiba ----
    ("curitiba", "GeoCuritiba / IPPUC", "Eixos de rua / logradouros / bairros (ArcGIS REST)",
     "https://geocuritiba.ippuc.org.br/server/rest/services", "official", "arcgis_rest",
     dict(street=1, address=1, neighborhood=1, flood_points=1, flood_polygons=0, drainage=1)),
    ("curitiba", "Curitiba - Dados Abertos / SIAC 156", "Pontos de alagamento / drenagem / bacias (CKAN)",
     "https://dadosabertos.curitiba.pr.gov.br/api/3/action/package_search", "official", "ckan_api",
     dict(street=1, address=1, neighborhood=1, flood_points=1, flood_polygons=0, drainage=1)),
    ("curitiba", "IAT / Aguas Parana", "Hidrografia / bacias / drenagem (WFS GeoServer)",
     "https://geoportal.iat.pr.gov.br/geoserver/ows", "state", "wfs_getcapabilities",
     dict(street=0, address=0, neighborhood=0, flood_points=1, flood_polygons=1, drainage=1)),
    # ---- Petropolis ----
    ("petropolis", "CPRM/SGB GeoSGB", "Setores de risco / hidrografia (ArcGIS REST)",
     "https://geoportal.sgb.gov.br/server/rest/services", "federal", "arcgis_rest",
     dict(street=0, address=0, neighborhood=1, flood_points=1, flood_polygons=1, drainage=1)),
    ("petropolis", "INEA-RJ", "Areas inundaveis / hidrografia (ArcGIS)",
     "https://inea.maps.arcgis.com", "state", "arcgis_rest",
     dict(street=0, address=0, neighborhood=1, flood_points=1, flood_polygons=1, drainage=1)),
    ("petropolis", "DRM-RJ", "Setores afetados / relatorios tecnicos 2022 (portal)",
     "https://www.drm.rj.gov.br", "state", "data_portal_html",
     dict(street=1, address=0, neighborhood=1, flood_points=1, flood_polygons=1, drainage=0)),
    ("petropolis", "Prefeitura de Petropolis", "Bairros/distritos, logradouros, localidades atingidas (portal)",
     "https://www.petropolis.rj.gov.br", "official", "data_portal_html",
     dict(street=1, address=1, neighborhood=1, flood_points=1, flood_polygons=0, drainage=1)),
    ("petropolis", "Governo do RJ - Dados Abertos", "Logradouros / bairros / hidrografia RJ (CKAN)",
     "http://www.dados.rj.gov.br/api/3/action/package_search", "state", "ckan_api",
     dict(street=1, address=1, neighborhood=1, flood_points=0, flood_polygons=0, drainage=1)),
]

# ---------------------------------------------------------------------------
# Flood occurrence vocabulary (FASE 5). Only genuine flood terms keep a record;
# the exclusion set rejects/downgrades structural inspection, landslide-only,
# social registry, alerts, prevention, etc.
# ---------------------------------------------------------------------------

FLOOD_TERMS = [
    "alagamento", "alagamentos", "alagad", "inundacao", "inundac", "inundaç",
    "enchente", "enxurrada", "transbordament", "agua invadiu", "rua alagada",
    "imovel alagado", "imoveis alagad", "lamina d", "ponto de alagamento",
    "ponto critico de alagamento",
]

EXCLUDE_TERMS = [
    "vistoria", "analise visual", "analise estrutural", "imoveis com danos",
    "imovel com dano", "queda de arvore", "queda de muro", "deslizamento",
    "escorregamento", "movimento de massa", "animal", "incendio", "cadastro social",
    "atendimento social", "alerta", "aviso", "previsao", "prevencao", "simulado",
    "rufa", "remocao", "doacao", "abrigo", "vendaval", "raio", "desabamento",
]

# Terms that, even inside an occurrence text, signal risk/suscept context only.
RISK_CONTEXT_TERMS = ["risco", "suscet", "suscept", "area de risco", "grau de risco"]

STREET_ABBREV = {
    r"\bav\b": "avenida", r"\bav\.\b": "avenida", r"\br\b": "rua", r"\br\.\b": "rua",
    r"\bpc\b": "praca", r"\bpc\.\b": "praca", r"\btv\b": "travessa",
    r"\best\b": "estrada", r"\brod\b": "rodovia", r"\bal\b": "alameda",
    r"\blgo\b": "largo", r"\bvl\b": "vila", r"\bdr\b": "doutor", r"\bdr\.\b": "doutor",
    r"\bprof\b": "professor", r"\bgen\b": "general", r"\bcel\b": "coronel",
}


def strip_accents(text: str) -> str:
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_text(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace/punctuation. Deterministic."""
    t = strip_accents(str(text or "")).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_street(text: str) -> str:
    """Normalize a street/address string: expand abbreviations, drop house-number
    markers ('no', 'num', 'numero'), drop any token containing a digit (house
    numbers such as '24a'), and any trailing ' e adjacencias' clause."""
    t = normalize_text(text)
    for pat, rep in STREET_ABBREV.items():
        t = re.sub(pat, rep, t)
    t = re.sub(r"\be\s+adjacencias\b.*$", " ", t)
    t = re.sub(r"\b(no|num|numero|nro)\b", " ", t)
    # drop any whitespace-delimited token that contains a digit (house numbers)
    t = " ".join(tok for tok in t.split() if not any(c.isdigit() for c in tok))
    t = re.sub(r"\s+", " ", t).strip()
    return t


# Precompute normalized term lists once (these are scanned over ~1M occurrence
# rows; re-normalizing each term per row is the dominant cost otherwise).
_FLOOD_NORM = tuple(dict.fromkeys(normalize_text(x) for x in FLOOD_TERMS if normalize_text(x)))
_EXCLUDE_NORM = tuple(dict.fromkeys(normalize_text(x) for x in EXCLUDE_TERMS if normalize_text(x)))
_RISK_NORM = tuple(dict.fromkeys(normalize_text(x) for x in RISK_CONTEXT_TERMS if normalize_text(x)))


def is_flood_text(*texts: str) -> bool:
    """True only when an explicit flood term is present. A genuine flood term
    overrides exclusion (e.g. 'vistoria de alagamento'); otherwise rows with only
    inspection/landslide/social/alert terms are dropped by the absence of a flood
    term."""
    low = normalize_text(" ".join(t for t in texts if t))
    return any(x in low for x in _FLOOD_NORM)


def exclusion_reason(*texts: str) -> str:
    """First matched exclusion term (for audit of why a non-flood row was dropped)."""
    low = normalize_text(" ".join(t for t in texts if t))
    for x in _EXCLUDE_NORM:
        if x in low:
            return x
    return "no_flood_term"


def looks_risk_only(*texts: str) -> bool:
    low = normalize_text(" ".join(t for t in texts if t))
    has_risk = any(x in low for x in _RISK_NORM)
    has_flood = any(x in low for x in _FLOOD_NORM)
    return has_risk and not has_flood


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def read_brazilian_csv(path: Path) -> list[dict]:
    """Read a possibly latin-1 / ';'-delimited Brazilian official CSV robustly.

    Returns list[dict] keyed by accent-stripped, normalized header tokens so the
    Defesa Civil 'M\xeas' / 'Endere\xe7o' headers are reachable consistently.
    """
    raw = path.read_bytes()
    text = None
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            decoded = raw.decode(enc)
            # latin-1 always succeeds; prefer the one with fewest replacement chars
            if enc == "utf-8" and "�" in decoded:
                continue
            text = decoded
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("latin-1", "ignore")
    sample = text[:4096]
    counts = {d: sample.count(d) for d in (";", ",", "\t")}
    delim = max(counts, key=lambda d: counts[d]) if any(counts.values()) else ","
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    rows = list(reader)
    if not rows:
        return []
    header = [normalize_text(h) for h in rows[0]]
    out = []
    for r in rows[1:]:
        if not any(c.strip() for c in r):
            continue
        out.append({header[i]: (r[i] if i < len(r) else "") for i in range(len(header))})
    return out


def header_pick(row: dict, *candidates: str) -> str:
    """Return the value of the first matching normalized header candidate."""
    for c in candidates:
        cn = normalize_text(c)
        if cn in row and str(row[cn]).strip():
            return str(row[cn]).strip()
    return ""


# Region centroids are intentionally NOT exported as event coordinates: a
# municipality/neighborhood centroid must never become a patch-level event.
REGION_EVENT_TYPE = {"recife": "urban_flooding", "petropolis": "flash_flood", "curitiba": "urban_flooding"}
MUNICIPALITY = {"recife": "Recife", "petropolis": "Petropolis", "curitiba": "Curitiba"}
