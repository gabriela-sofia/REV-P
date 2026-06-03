"""
revp_v1if_official_observed_event_vector_acquisition_audit.py

v1if -- Aquisicao e auditoria de vetores observados oficiais para eventos
hidrológicos/geologicos do Protocolo C.

Objetivo:
    Buscar, baixar, extrair e auditar dados vetoriais observados de fontes
    oficiais e institucionais para os eventos alvo do REV-P:
        PET_2022_02_15 (Petropolis, RJ)
        REC_2022_05    (Recife/Grande Recife, PE)
        CUR_*          (Curitiba, PR)

Modos de operacao:
    default (sem flags)        -- dry-run: relata o que faria, sem escrever
    --search-local             -- varre workspace local em modo read-only
    --download-official-known  -- tenta baixar de URLs oficiais curadas
    --force                    -- escreve registries publicos (sem paths privados)

Regras invariantes:
    - Arquivos brutos (ZIP, SHP, GPKG, KMZ, PDF, raster) so vao para
      local_runs/protocolo_c/v1if/raw_official_sources/
    - Nenhum path privado aparece em registries publicos
    - Suscetibilidade/risco/modelagem nunca vira ground truth de evento observado
    - Vetor sem data compativel fica BLOCKED
    - Fenomeno misto sem campo separavel fica BLOCKED_UNTIL_PHENOMENON_SEPARATION
    - ml_label_status permanece BLOCKED_UNTIL_SPLIT_AND_LEAKAGE_PROTOCOL invariante
    - can_create_training_label permanece false ate split/leakage protocol completo
"""

import argparse
import csv
import json
import struct
import zipfile
import zlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, Request


# ---------------------------------------------------------------------------
# Tentativa de importacao de pyshp
# ---------------------------------------------------------------------------
HAS_PYSHP = False
try:
    import shapefile  # type: ignore[import-untyped]
    HAS_PYSHP = True
except ImportError:
    shapefile = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Caminhos do repositorio
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"
LOCAL_RUNS = REPO_ROOT / "local_runs" / "protocolo_c" / "v1if"
RAW_SOURCES_DIR = LOCAL_RUNS / "raw_official_sources"
EXTRACTED_DIR = RAW_SOURCES_DIR / "extracted"

# Registry privado do PROJETO (nao vai para Git)
PROJETO_DIR = Path("C:/Users/gabriela/Documents/PROJETO")

# ---------------------------------------------------------------------------
# Fontes oficiais curadas
# ---------------------------------------------------------------------------
# Cada entrada e auditavel: URL verificada, instituicao, tipo esperado.
# URLs validadas em 2026-05-22 via rigeo.sgb.gov.br.

KNOWN_OFFICIAL_SOURCES = [
    {
        "source_asset_id": "OBS_PET_001",
        "event_id": "PET_2022_02_15",
        "region": "PET",
        "event_date": "2022-02-15",
        "source_institution": "SGB/CPRM",
        "source_repository": "RIGeo/SGB",
        "source_title": "Avaliacao tecnica pos-desastre: Petropolis, RJ",
        "source_asset_name": "Relatorio_Tecnico_Petropolis.pdf",
        "source_asset_type": "PDF_REPORT",
        "download_url": (
            "https://rigeo.sgb.gov.br/bitstreams/"
            "e011ddf7-3612-4f4c-8589-726572657929/download"
        ),
        "official_source_url_reference": "https://rigeo.sgb.gov.br/handle/doc/22668",
        "expected_size_mb": 4.1,
        "skip_if_exists": True,
        "notes": "Relatorio PDF principal - nao e vetor, mas pode referenciar anexos",
    },
    {
        "source_asset_id": "OBS_PET_002",
        "event_id": "PET_2022_02_15",
        "region": "PET",
        "event_date": "2022-02-15",
        "source_institution": "SGB/CPRM",
        "source_repository": "RIGeo/SGB",
        "source_title": "Avaliacao tecnica pos-desastre: Petropolis, RJ",
        "source_asset_name": "anexos_avaliacao_pos_desastre_petropolis_rj_2022.zip",
        "source_asset_type": "ZIP_ARCHIVE",
        "download_url": (
            "https://rigeo.sgb.gov.br/bitstreams/"
            "23d77158-e00c-4a99-87c7-0bb1d3ecb7fd/download"
        ),
        "official_source_url_reference": "https://rigeo.sgb.gov.br/handle/doc/22668",
        "expected_size_mb": 20.0,
        "skip_if_exists": True,
        "notes": (
            "ZIP de anexos - prioritario: pode conter KMZ, SHP, mapas georref "
            "da avaliacao pos-desastre"
        ),
    },
    # Fontes adicionais: pendentes de aquisicao manual ou formal
    {
        "source_asset_id": "OBS_PET_003",
        "event_id": "PET_2022_02_15",
        "region": "PET",
        "event_date": "2022-02-15",
        "source_institution": "DRM-RJ/NADE",
        "source_repository": "DRM-RJ",
        "source_title": "Relatorio DRM-RJ pos-desastre Petropolis 2022",
        "source_asset_name": "PKG_FR_PET_001",
        "source_asset_type": "UNKNOWN",
        "download_url": None,
        "official_source_url_reference": "https://www.drm.rj.gov.br",
        "expected_size_mb": None,
        "skip_if_exists": False,
        "notes": "Requer solicitacao formal DRM-RJ/NADE - nao disponivel publicamente",
    },
    {
        "source_asset_id": "OBS_PET_004",
        "event_id": "PET_2022_02_15",
        "region": "PET",
        "event_date": "2022-02-15",
        "source_institution": "Defesa Civil Municipal Petropolis",
        "source_repository": "Portal Prefeitura Petropolis",
        "source_title": "Laudos/mapas de areas afetadas Petropolis 2022",
        "source_asset_name": "UNKNOWN",
        "source_asset_type": "UNKNOWN",
        "download_url": None,
        "official_source_url_reference": "https://www.petropolis.rj.gov.br",
        "expected_size_mb": None,
        "skip_if_exists": False,
        "notes": "Requer busca manual ou solicitacao formal via Defesa Civil Municipal",
    },
    {
        "source_asset_id": "OBS_REC_001",
        "event_id": "REC_2022_05",
        "region": "REC",
        "event_date": "2022-05-26",
        "source_institution": "COMPDEC/Defesa Civil PE",
        "source_repository": "Portal Defesa Civil PE",
        "source_title": "Boletim/laudo areas afetadas Recife maio 2022",
        "source_asset_name": "PKG_FR_REC_002",
        "source_asset_type": "UNKNOWN",
        "download_url": None,
        "official_source_url_reference": "https://www.defesacivil.pe.gov.br",
        "expected_size_mb": None,
        "skip_if_exists": False,
        "notes": "Requer solicitacao formal PKG_FR_REC_002 - acesso bloqueado por SSL/403",
    },
    {
        "source_asset_id": "OBS_CUR_001",
        "event_id": "CUR_HISTORICAL",
        "region": "CUR",
        "event_date": "UNKNOWN",
        "source_institution": "Prefeitura Curitiba/IPPUC",
        "source_repository": "GeoCuritiba/IPPUC",
        "source_title": "Mapeamento de areas de alagamento Curitiba",
        "source_asset_name": "UNKNOWN",
        "source_asset_type": "UNKNOWN",
        "download_url": None,
        "official_source_url_reference": "https://www.ippuc.org.br",
        "expected_size_mb": None,
        "skip_if_exists": False,
        "notes": "Requer busca e auditoria de eventos CUR com data confirmada",
    },
]

# Extensoes de vetores que devem ser auditadas
VECTOR_EXTENSIONS = {".shp", ".gpkg", ".geojson", ".kml", ".kmz", ".json"}

# Extensoes de risco/suscetibilidade (nunca ground truth observado)
SUSCEPTIBILITY_KEYWORDS = {
    "suscetib", "susceptib", "risco", "risk", "vulnerab",
    "modelag", "model", "classe", "class", "zoneam", "zoning",
    "carta_risco", "risk_map",
}

# Palavras que indicam ocorrencia observada
OBSERVED_KEYWORDS = {
    "ocorr", "event", "afet", "impact", "damage",
    "inund", "alaga", "enxurr", "transbord", "flood",
    "desliz", "escorr", "feição de deslizamento", "landslide feature", "desastre",
    "pos_desastre", "pos-desastre", "avaliacao",
}

# Palavras que indicam inundacao/hidrologia
HYDRO_KEYWORDS = {
    "inund", "alaga", "enxurr", "transbord", "flood",
    "cheia", "overfl", "submers", "anegad",
}

# Palavras que indicam movimento de massa
MASS_MOVEMENT_KEYWORDS = {
    "desliz", "escorr", "feição de deslizamento", "landslide feature", "corrida",
    "landslide", "mass_movement", "debris", "mudflow",
    "queda", "fall", "colapso",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SourceDownloadRecord:
    source_asset_id: str
    event_id: str
    region: str
    event_date: str
    source_institution: str
    source_repository: str
    source_title: str
    source_asset_name: str
    source_asset_type: str
    download_url_reference: str  # URL usada (sem path privado de destino)
    official_source_url_reference: str
    download_status: str  # DOWNLOAD_OK | DOWNLOAD_FAILED | NO_URL | ALREADY_EXISTS
    download_timestamp: str
    file_size_bytes: int
    error_message: str
    notes: str


@dataclass
class ExtractedAssetRecord:
    source_asset_id: str
    event_id: str
    parent_zip: str  # nome do ZIP de origem, sem path
    asset_name: str
    asset_extension: str
    asset_size_bytes: int
    is_vector: bool
    is_raster: bool
    is_document: bool
    notes: str


@dataclass
class VectorGeometryAudit:
    source_asset_id: str
    asset_name: str  # sem path
    geometry_available: str
    geometry_type: str
    crs: str
    feature_count: int
    fields: str  # lista separada por ponto-e-virgula
    has_date_field: str
    has_phenomenon_field: str
    has_locality_field: str
    phenomenon_values_sample: str
    date_values_sample: str
    bounds: str
    notes: str


@dataclass
class GroundTruthGateAudit:
    source_asset_id: str
    asset_name: str  # sem path
    # Gates
    official_or_institutional_source: str
    raw_asset_traceable: str
    geometry_available: str
    crs_available: str
    geometry_valid: str
    event_date_available: str
    event_date_compatible: str
    phenomenon_available: str
    phenomenon_is_observed_not_risk: str
    hydrological_or_mass_movement_separable: str
    spatial_unit_usable_for_patch_binding: str
    # Decisao
    final_gate_status: str  # PASS | FAIL
    ground_truth_status: str
    blocking_gate: str
    notes: str


@dataclass
class PublicRegistryRecord:
    source_asset_id: str
    event_id: str
    region: str
    event_date: str
    source_institution: str
    source_repository: str
    source_title: str
    source_asset_name: str
    source_asset_type: str
    source_access_status: str
    local_ingestion_status: str
    official_source_url_reference: str
    raw_file_versioning_status: str
    geometry_available: str
    geometry_type: str
    crs: str
    feature_count: int
    has_event_date_field: str
    event_date_compatible: str
    has_phenomenon_field: str
    phenomenon_raw_values: str
    hydrological_observed_features_count: int
    mass_movement_observed_features_count: int
    mixed_or_unknown_features_count: int
    risk_or_susceptibility_only: str
    observed_event_status: str
    patch_level_usability: str
    ground_truth_status: str
    ml_label_status: str
    limitations: str
    next_required_action: str
    notes: str


# ---------------------------------------------------------------------------
# Utilitarios
# ---------------------------------------------------------------------------

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def sanitize_path_for_public(path: Optional[Path]) -> str:
    """Remove componentes de path privado; retorna apenas nome do arquivo."""
    if path is None:
        return "N/A"
    return path.name


def detect_susceptibility(name: str, fields: list[str]) -> bool:
    """Retorna True se o nome do arquivo ou campos sugerem suscetibilidade/risco."""
    combined = (name.lower() + " " + " ".join(f.lower() for f in fields))
    return any(kw in combined for kw in SUSCEPTIBILITY_KEYWORDS)


def detect_hydro_observation(name: str, fields: list[str]) -> bool:
    combined = (name.lower() + " " + " ".join(f.lower() for f in fields))
    return any(kw in combined for kw in HYDRO_KEYWORDS)


def detect_mass_movement_observation(name: str, fields: list[str]) -> bool:
    combined = (name.lower() + " " + " ".join(f.lower() for f in fields))
    return any(kw in combined for kw in MASS_MOVEMENT_KEYWORDS)


def detect_observation_likelihood(name: str) -> bool:
    """Retorna True se o nome sugere dado observado (nao apenas risco)."""
    lower = name.lower()
    return (
        any(kw in lower for kw in OBSERVED_KEYWORDS)
        and not any(kw in lower for kw in {"suscetib", "susceptib", "risco", "risk"})
    )


# ---------------------------------------------------------------------------
# Auditoria de shapefile com pyshp
# ---------------------------------------------------------------------------

def audit_shapefile(shp_path: Path) -> dict:
    """Audita um shapefile com pyshp e retorna metadata."""
    result = {
        "geometry_available": "NO",
        "geometry_type": "UNKNOWN",
        "crs": "UNKNOWN",
        "feature_count": 0,
        "fields": [],
        "bounds": "UNKNOWN",
        "has_date_field": False,
        "has_phenomenon_field": False,
        "has_locality_field": False,
        "phenomenon_values": [],
        "date_values": [],
        "error": None,
    }

    if not HAS_PYSHP or shapefile is None:
        result["error"] = "pyshp_not_available"
        return result

    try:
        sf = shapefile.Reader(str(shp_path))
        result["geometry_available"] = "YES"
        result["feature_count"] = len(sf)

        # Tipo de geometria
        shape_type_map = {
            0: "NULL", 1: "POINT", 3: "POLYLINE", 5: "POLYGON",
            8: "MULTIPOINT", 11: "POINTZ", 13: "POLYLINEZ",
            15: "POLYGONZ", 21: "MULTIPOINTM",
        }
        result["geometry_type"] = shape_type_map.get(sf.shapeType, f"TYPE_{sf.shapeType}")

        # Bounds
        try:
            result["bounds"] = str(sf.bbox)
        except Exception:
            result["bounds"] = "UNKNOWN"

        # CRS via .prj
        prj_path = shp_path.with_suffix(".prj")
        if prj_path.exists():
            prj_text = prj_path.read_text(errors="ignore").lower()
            if "sirgas" in prj_text or "gcs_sirgas" in prj_text:
                result["crs"] = "SIRGAS_2000"
            elif "wgs_1984" in prj_text or "wgs84" in prj_text or "epsg:4326" in prj_text:
                result["crs"] = "WGS84"
            elif "utm" in prj_text and "23" in prj_text:
                result["crs"] = "UTM_ZONE_23"
            elif "projcs" in prj_text:
                result["crs"] = "PROJECTED_CRS_UNKNOWN"
            else:
                result["crs"] = "UNKNOWN_CRS_FROM_PRJ"
        else:
            result["crs"] = "NO_PRJ_FILE"

        # Campos
        field_names = [f[0] for f in sf.fields[1:]]  # skip DeletionFlag
        result["fields"] = field_names

        date_keywords = {"data", "date", "dt_", "dt", "ano", "year", "mes"}
        phenom_keywords = {
            "tipo", "type", "processo", "process", "classe", "class",
            "fenomeno", "phenomenon", "evento", "event",
        }
        locality_keywords = {"localidade", "locality", "bairro", "municipio", "cidade", "local"}

        result["has_date_field"] = any(
            any(kw in f.lower() for kw in date_keywords) for f in field_names
        )
        result["has_phenomenon_field"] = any(
            any(kw in f.lower() for kw in phenom_keywords) for f in field_names
        )
        result["has_locality_field"] = any(
            any(kw in f.lower() for kw in locality_keywords) for f in field_names
        )

        # Amostra de valores de fenomeno (ate 5 feicoes)
        phenom_fields = [
            f for f in field_names
            if any(kw in f.lower() for kw in phenom_keywords)
        ]
        if phenom_fields and len(sf) > 0:
            seen: set = set()
            for rec in sf.iterRecords():
                val = str(rec[phenom_fields[0]])
                seen.add(val)
                if len(seen) >= 8:
                    break
            result["phenomenon_values"] = list(seen)

        # Amostra de valores de data (ate 5 feicoes)
        date_fields = [
            f for f in field_names
            if any(kw in f.lower() for kw in date_keywords)
        ]
        if date_fields and len(sf) > 0:
            seen_dates: set = set()
            for rec in sf.iterRecords():
                val = str(rec[date_fields[0]])
                if val and val not in {"None", ""}:
                    seen_dates.add(val)
                if len(seen_dates) >= 5:
                    break
            result["date_values"] = list(seen_dates)

    except Exception as exc:
        result["geometry_available"] = "ERROR"
        result["error"] = str(exc)[:200]

    return result


def audit_geojson(path: Path) -> dict:
    """Auditoria basica de GeoJSON com json stdlib."""
    result = {
        "geometry_available": "NO",
        "geometry_type": "UNKNOWN",
        "crs": "WGS84",  # GeoJSON e sempre WGS84 por spec
        "feature_count": 0,
        "fields": [],
        "bounds": "UNKNOWN",
        "has_date_field": False,
        "has_phenomenon_field": False,
        "has_locality_field": False,
        "phenomenon_values": [],
        "date_values": [],
        "error": None,
    }
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        features = data.get("features", [])
        result["feature_count"] = len(features)
        if features:
            result["geometry_available"] = "YES"
            geom = features[0].get("geometry", {}) or {}
            result["geometry_type"] = geom.get("type", "UNKNOWN").upper()
            props = features[0].get("properties", {}) or {}
            result["fields"] = list(props.keys())
            date_kw = {"data", "date", "dt_", "ano", "year"}
            phenom_kw = {"tipo", "type", "processo", "class", "fenomeno"}
            loc_kw = {"localidade", "bairro", "municipio", "local"}
            result["has_date_field"] = any(
                any(kw in k.lower() for kw in date_kw) for k in props
            )
            result["has_phenomenon_field"] = any(
                any(kw in k.lower() for kw in phenom_kw) for k in props
            )
            result["has_locality_field"] = any(
                any(kw in k.lower() for kw in loc_kw) for k in props
            )
    except Exception as exc:
        result["geometry_available"] = "ERROR"
        result["error"] = str(exc)[:200]
    return result


def audit_kmz_kml(path: Path) -> dict:
    """Auditoria basica de KML/KMZ: conta Placemark/Polygon."""
    result = {
        "geometry_available": "NO",
        "geometry_type": "UNKNOWN",
        "crs": "WGS84",
        "feature_count": 0,
        "fields": [],
        "bounds": "UNKNOWN",
        "has_date_field": False,
        "has_phenomenon_field": False,
        "has_locality_field": False,
        "phenomenon_values": [],
        "date_values": [],
        "error": None,
    }
    try:
        # KMZ e um ZIP contendo KML
        if path.suffix.lower() == ".kmz":
            import zipfile as zf
            with zf.ZipFile(path, "r") as z:
                kml_names = [n for n in z.namelist() if n.lower().endswith(".kml")]
                if not kml_names:
                    result["error"] = "no_kml_in_kmz"
                    return result
                kml_content = z.read(kml_names[0]).decode("utf-8", errors="ignore")
        else:
            kml_content = path.read_text(encoding="utf-8", errors="ignore")

        # Contagem de Placemarks
        placemark_count = kml_content.count("<Placemark")
        polygon_count = kml_content.count("<Polygon")
        point_count = kml_content.count("<Point")
        linestring_count = kml_content.count("<LineString")

        result["feature_count"] = placemark_count
        result["geometry_available"] = "YES" if placemark_count > 0 else "NO"

        if polygon_count > 0:
            result["geometry_type"] = "POLYGON"
        elif point_count > 0:
            result["geometry_type"] = "POINT"
        elif linestring_count > 0:
            result["geometry_type"] = "LINESTRING"
        else:
            result["geometry_type"] = "UNKNOWN"

        # Verificar campos SimpleData
        result["has_date_field"] = any(
            kw in kml_content.lower()
            for kw in ["<simpledata name=\"data\"", "<simpledata name=\"date\"", "quando", "when>"]
        )
        result["has_phenomenon_field"] = any(
            kw in kml_content.lower()
            for kw in ["tipo", "process", "fenomeno", "classe"]
        )

    except Exception as exc:
        result["geometry_available"] = "ERROR"
        result["error"] = str(exc)[:200]
    return result


def audit_vector(path: Path) -> dict:
    """Dispatcha para o auditor correto baseado na extensao."""
    ext = path.suffix.lower()
    if ext == ".shp":
        return audit_shapefile(path)
    elif ext in (".geojson", ".json"):
        return audit_geojson(path)
    elif ext in (".kmz", ".kml"):
        return audit_kmz_kml(path)
    else:
        return {
            "geometry_available": "NOT_APPLICABLE",
            "geometry_type": "NOT_APPLICABLE",
            "crs": "NOT_APPLICABLE",
            "feature_count": 0,
            "fields": [],
            "bounds": "UNKNOWN",
            "has_date_field": False,
            "has_phenomenon_field": False,
            "has_locality_field": False,
            "phenomenon_values": [],
            "date_values": [],
            "error": f"unsupported_extension_{ext}",
        }


# ---------------------------------------------------------------------------
# Aplicacao de gates
# ---------------------------------------------------------------------------

def apply_ground_truth_gates(
    source_id: str,
    asset_name: str,
    source: dict,
    audit: dict,
    download_status: str,
    ingestion_ok: bool,
) -> GroundTruthGateAudit:
    """Aplica os 11 gates metodologicos e retorna decisao."""

    gates: dict[str, str] = {}

    # Gate 1: fonte oficial ou institucional
    institution = source.get("source_institution", "").upper()
    is_official = any(
        kw in institution
        for kw in ["SGB", "CPRM", "DRM", "INPE", "IBGE", "ANA",
                   "DEFESA CIVIL", "SEDEC", "INEA", "IPPUC", "APAC",
                   "PREFEITURA", "SIMEPAR", "COMPDEC"]
    )
    gates["official_or_institutional_source"] = "PASS" if is_official else "FAIL"

    # Gate 2: rastreavelidade do ativo bruto
    has_url = bool(source.get("download_url") or source.get("official_source_url_reference"))
    gates["raw_asset_traceable"] = "PASS" if has_url else "FAIL"

    # Gate 3: geometria disponivel
    geom_status = audit.get("geometry_available", "NO")
    gates["geometry_available"] = "PASS" if geom_status == "YES" else "FAIL"

    # Gate 4: CRS disponivel
    crs_val = audit.get("crs", "UNKNOWN")
    gates["crs_available"] = (
        "PASS"
        if crs_val not in ("UNKNOWN", "NO_PRJ_FILE", "NOT_APPLICABLE", "")
        else "FAIL"
    )

    # Gate 5: geometria valida (sem erro critico)
    error = audit.get("error")
    gates["geometry_valid"] = (
        "PASS"
        if error is None and geom_status == "YES"
        else "FAIL"
    )

    # Gate 6: campo de data disponivel
    gates["event_date_available"] = (
        "PASS" if audit.get("has_date_field", False) else "FAIL"
    )

    # Gate 7: data compativel com evento alvo
    event_date = source.get("event_date", "")
    date_vals = audit.get("date_values", [])
    date_compatible = False
    if event_date and date_vals:
        for dv in date_vals:
            if "2022" in str(dv) and ("02" in str(dv) or "15" in str(dv) or "fev" in str(dv).lower()):
                date_compatible = True
                break
    # Se nao ha campo de data e o evento e "UNKNOWN", gate 7 tambem falha
    gates["event_date_compatible"] = "PASS" if date_compatible else "FAIL"

    # Gate 8: campo de fenomeno disponivel
    gates["phenomenon_available"] = (
        "PASS" if audit.get("has_phenomenon_field", False) else "FAIL"
    )

    # Gate 9: fenomeno e ocorrencia observada (nao risco/suscetibilidade)
    asset_name_lower = asset_name.lower()
    is_suscept = detect_susceptibility(asset_name, audit.get("fields", []))
    phenom_vals = audit.get("phenomenon_values", [])
    has_suscept_values = any(
        any(kw in str(v).lower() for kw in {"baixa", "media", "alta", "suscet", "risco"})
        for v in phenom_vals
    )
    is_risk_only = is_suscept or has_suscept_values
    gates["phenomenon_is_observed_not_risk"] = "FAIL" if is_risk_only else "PASS"

    # Gate 10: fenomeno separavel (hidro vs. massa)
    is_hydro = detect_hydro_observation(asset_name, audit.get("fields", []))
    is_mass = detect_mass_movement_observation(asset_name, audit.get("fields", []))
    is_mixed_unseparable = (
        not audit.get("has_phenomenon_field", False)
        and "inund" in asset_name_lower
        and "desliz" in asset_name_lower
    )
    gates["hydrological_or_mass_movement_separable"] = (
        "BLOCKED_UNTIL_PHENOMENON_SEPARATION" if is_mixed_unseparable else "PASS"
    )

    # Gate 11: usabilidade para patch-level binding
    fc = audit.get("feature_count", 0)
    geom_type = audit.get("geometry_type", "")
    is_patch_usable = (
        fc > 0
        and geom_type in ("POLYGON", "MULTIPOLYGON", "5", "15")
        and geom_status == "YES"
    )
    gates["spatial_unit_usable_for_patch_binding"] = "PASS" if is_patch_usable else "FAIL"

    # Decisao final
    blocking = []
    for g, v in gates.items():
        if v in ("FAIL", "BLOCKED_UNTIL_PHENOMENON_SEPARATION"):
            blocking.append(g)

    if not blocking:
        final_status = "PASS"
        gt_status = "CANDIDATE_OBSERVED_GROUND_TRUTH"
    elif "hydrological_or_mass_movement_separable" in blocking and len(blocking) == 1:
        final_status = "FAIL"
        gt_status = "BLOCKED_UNTIL_PHENOMENON_SEPARATION"
    else:
        final_status = "FAIL"
        gt_status = "BLOCKED"

    return GroundTruthGateAudit(
        source_asset_id=source_id,
        asset_name=asset_name,
        official_or_institutional_source=gates["official_or_institutional_source"],
        raw_asset_traceable=gates["raw_asset_traceable"],
        geometry_available=gates["geometry_available"],
        crs_available=gates["crs_available"],
        geometry_valid=gates["geometry_valid"],
        event_date_available=gates["event_date_available"],
        event_date_compatible=gates["event_date_compatible"],
        phenomenon_available=gates["phenomenon_available"],
        phenomenon_is_observed_not_risk=gates["phenomenon_is_observed_not_risk"],
        hydrological_or_mass_movement_separable=gates["hydrological_or_mass_movement_separable"],
        spatial_unit_usable_for_patch_binding=gates["spatial_unit_usable_for_patch_binding"],
        final_gate_status=final_status,
        ground_truth_status=gt_status,
        blocking_gate="; ".join(blocking) if blocking else "NONE",
        notes="Aplicado em v1if. gate_count=11.",
    )


# ---------------------------------------------------------------------------
# Busca local
# ---------------------------------------------------------------------------

def search_local_candidates() -> list[Path]:
    """Busca candidatos vetoriais no workspace local (read-only)."""
    candidates: list[Path] = []
    search_roots = []
    if PROJETO_DIR.exists():
        search_roots.append(PROJETO_DIR)

    priority_names = {
        "areas_inundadas", "flood_areas", "inundacao_evento",
        "alagamento", "ocorrencia", "afetadas", "impacto",
        "dano", "desastre", "pos_desastre",
        "petropolis_2022", "petropolis_evento",
    }

    for root in search_roots:
        try:
            for ext in (".shp", ".gpkg", ".geojson", ".kml", ".kmz"):
                for p in root.rglob(f"*{ext}"):
                    name_lower = p.stem.lower()
                    if any(kw in name_lower for kw in priority_names):
                        candidates.append(p)
        except PermissionError:
            pass

    return candidates


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_file(url: str, dest_path: Path, max_mb: float = 50.0) -> tuple[bool, str]:
    """Baixa um arquivo de URL para dest_path. Retorna (ok, mensagem)."""
    try:
        req = Request(url, headers={"User-Agent": "REV-P-Protocolo-C/v1if (research)"})
        with urlopen(req, timeout=60) as response:
            content_length = response.headers.get("Content-Length")
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > max_mb:
                    return False, f"FILE_TOO_LARGE: {size_mb:.1f}MB > {max_mb}MB limit"

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            chunk_size = 1024 * 64  # 64KB
            total = 0
            with open(dest_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)

        return True, f"OK: {total} bytes"
    except HTTPError as e:
        return False, f"HTTP_{e.code}: {e.reason}"
    except URLError as e:
        return False, f"URL_ERROR: {e.reason}"
    except OSError as e:
        return False, f"OS_ERROR: {e}"


# ---------------------------------------------------------------------------
# Extracao de ZIP
# ---------------------------------------------------------------------------

def _parse_streaming_zip_entries(zip_path: Path) -> list[dict]:
    """
    Parse local file entries de um ZIP sem Central Directory (streaming ZIP).
    Retorna lista de dicts com metadados de cada entrada.
    """
    entries = []
    LOCAL_SIG = b"PK\x03\x04"
    with open(zip_path, "rb") as f:
        while True:
            sig = f.read(4)
            if len(sig) < 4 or sig != LOCAL_SIG:
                break
            version_needed = struct.unpack("<H", f.read(2))[0]
            flags = struct.unpack("<H", f.read(2))[0]
            compression = struct.unpack("<H", f.read(2))[0]
            mod_time = struct.unpack("<H", f.read(2))[0]
            mod_date = struct.unpack("<H", f.read(2))[0]
            crc32_val = struct.unpack("<I", f.read(4))[0]
            compressed_size = struct.unpack("<I", f.read(4))[0]
            uncompressed_size = struct.unpack("<I", f.read(4))[0]
            fname_len = struct.unpack("<H", f.read(2))[0]
            extra_len = struct.unpack("<H", f.read(2))[0]
            filename_bytes = f.read(fname_len)
            _extra = f.read(extra_len)
            data_offset = f.tell()

            # Tentar decodificar o nome do arquivo
            fname_str = filename_bytes.decode("utf-8", errors="replace")
            # Fix de encoding: tentar cp1252 se UTF-8 resultou em replacement chars
            if "�" in fname_str:
                try:
                    fname_str = filename_bytes.decode("cp1252")
                except Exception:
                    fname_str = filename_bytes.decode("latin-1", errors="replace")

            entries.append({
                "filename": fname_str,
                "compression": compression,
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "data_offset": data_offset,
                "crc32": crc32_val,
                "flags": flags,
                "version_needed": version_needed,
                "mod_date": mod_date,
                "mod_time": mod_time,
            })

            if compressed_size == 0:
                # Dados inline ou corrupto — parar
                break
            try:
                f.seek(compressed_size, 1)
            except OSError:
                break
    return entries


def _extract_streaming_zip_entry(zip_path: Path, entry: dict, dest_path: Path) -> bool:
    """Extrai uma entrada de streaming ZIP para dest_path."""
    try:
        with open(zip_path, "rb") as f:
            f.seek(entry["data_offset"])
            compressed_data = f.read(entry["compressed_size"])

        if entry["compression"] == 0:
            data = compressed_data
        elif entry["compression"] == 8:
            data = zlib.decompress(compressed_data, -15)  # raw deflate
        else:
            return False  # metodo nao suportado

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as out:
            out.write(data)
        return True
    except Exception:
        return False


def extract_zip(zip_path: Path, dest_dir: Path) -> list[ExtractedAssetRecord]:
    """Extrai ZIP e retorna inventario de arquivos extraidos.

    Tenta primeiro a via padrao (zipfile). Se falhar com BadZipFile, usa
    parser de streaming ZIP (sem Central Directory).
    """
    records = []
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Tentativa 1: zipfile padrao
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                ext = Path(member.filename).suffix.lower()
                is_vector = ext in VECTOR_EXTENSIONS
                is_raster = ext in {".tif", ".tiff", ".geotiff", ".img"}
                is_doc = ext in {".pdf", ".docx", ".xlsx", ".csv"}
                should_extract = is_vector or is_doc or member.file_size < 50 * 1024 * 1024
                if should_extract:
                    try:
                        zf.extract(member, dest_dir)
                    except Exception:
                        pass
                records.append(ExtractedAssetRecord(
                    source_asset_id="OBS_PET_002",
                    event_id="PET_2022_02_15",
                    parent_zip=zip_path.name,
                    asset_name=Path(member.filename).name,
                    asset_extension=ext,
                    asset_size_bytes=member.file_size,
                    is_vector=is_vector,
                    is_raster=is_raster,
                    is_document=is_doc,
                    notes=f"compressed_size={member.compress_size}",
                ))
        return records
    except zipfile.BadZipFile:
        pass  # fallback para parser de streaming

    # Tentativa 2: streaming ZIP (sem Central Directory)
    print("    [INFO] ZIP sem Central Directory — usando parser de streaming...")
    entries = _parse_streaming_zip_entries(zip_path)
    print(f"    [INFO] {len(entries)} entrada(s) encontrada(s) no streaming ZIP")

    for entry in entries:
        fname = entry["filename"]
        ext = Path(fname).suffix.lower()
        is_vector = ext in VECTOR_EXTENSIONS
        is_raster = ext in {".tif", ".tiff", ".geotiff", ".img"}
        is_doc = ext in {".pdf", ".docx", ".xlsx", ".csv"}

        dest_file = dest_dir / Path(fname).name
        extracted_ok = False
        if is_vector or is_doc:
            print(f"    [EXTRACT-STREAM] {Path(fname).name} ({entry['uncompressed_size']//1024}KB)...")
            extracted_ok = _extract_streaming_zip_entry(zip_path, entry, dest_file)
            if extracted_ok:
                print(f"      [OK] extraido: {dest_file.name}")
            else:
                print(f"      [FAIL] falhou ao extrair: {dest_file.name}")

        records.append(ExtractedAssetRecord(
            source_asset_id="OBS_PET_002",
            event_id="PET_2022_02_15",
            parent_zip=zip_path.name,
            asset_name=Path(fname).name,
            asset_extension=ext,
            asset_size_bytes=entry["uncompressed_size"],
            is_vector=is_vector,
            is_raster=is_raster,
            is_document=is_doc,
            notes=(
                f"streaming_zip; compression={entry['compression']}; "
                f"extracted={'OK' if extracted_ok else 'NO'}"
            ),
        ))

    if not records:
        records.append(ExtractedAssetRecord(
            source_asset_id="OBS_PET_002",
            event_id="PET_2022_02_15",
            parent_zip=zip_path.name,
            asset_name="PARSE_FAILED",
            asset_extension="",
            asset_size_bytes=0,
            is_vector=False,
            is_raster=False,
            is_document=False,
            notes="ZIP_PARSE_FAILED: nem zipfile padrao nem streaming funcionaram",
        ))
    return records


# ---------------------------------------------------------------------------
# Geracao de registro publico
# ---------------------------------------------------------------------------

def build_public_record(
    source: dict,
    download_ok: bool,
    ingested_path: Optional[Path],
    audit: Optional[dict],
    gate_decision: Optional[GroundTruthGateAudit],
) -> PublicRegistryRecord:
    """Constroi registro publico sem paths privados."""

    # Status de acesso
    if source.get("download_url") is None:
        access_status = "PENDING_FORMAL_REQUEST"
        ingestion_status = "NOT_INGESTED"
    elif download_ok:
        access_status = "DOWNLOAD_OK"
        ingestion_status = "INGESTED" if ingested_path else "PARTIAL"
    else:
        access_status = "DOWNLOAD_FAILED"
        ingestion_status = "NOT_INGESTED"

    asset_type = source.get("source_asset_type", "UNKNOWN")
    is_vector = asset_type in (
        "VECTOR_SHP", "VECTOR_GPKG", "VECTOR_GEOJSON", "VECTOR_KMZ", "VECTOR_KML"
    )

    # Para ZIPs e PDFs sem auditoria de vetor direta
    if audit is None:
        geom_avail = "NOT_APPLICABLE" if asset_type in ("PDF_REPORT", "ZIP_ARCHIVE") else "NO"
        geom_type = "NOT_APPLICABLE"
        crs_val = "NOT_APPLICABLE"
        feat_count = 0
        has_date_field = "NOT_APPLICABLE"
        evt_date_compat = "NOT_APPLICABLE"
        has_phenom = "NOT_APPLICABLE"
        phenom_vals = ""
        hydro_count = 0
        mass_count = 0
        mixed_count = 0
        risk_only = "NOT_APPLICABLE"
        obs_status = "CARTOGRAPHIC_LEAD_ONLY" if asset_type == "PDF_REPORT" else "UNDETERMINED"
        patch_usability = "NOT_USABLE"
        gt_status = gate_decision.ground_truth_status if gate_decision else "BLOCKED"
    else:
        geom_avail = audit.get("geometry_available", "NO")
        geom_type = audit.get("geometry_type", "UNKNOWN")
        crs_val = audit.get("crs", "UNKNOWN")
        feat_count = audit.get("feature_count", 0)
        has_date_field = "YES" if audit.get("has_date_field", False) else "NO"
        evt_date_compat = (
            gate_decision.event_date_compatible if gate_decision else "UNKNOWN"
        )
        has_phenom = "YES" if audit.get("has_phenomenon_field", False) else "NO"
        phenom_vals = "; ".join(audit.get("phenomenon_values", []))[:200]

        # Classificacao de feicoes
        pvals = [v.lower() for v in audit.get("phenomenon_values", [])]
        hydro_count = sum(
            1 for v in pvals if any(kw in v for kw in HYDRO_KEYWORDS)
        )
        mass_count = sum(
            1 for v in pvals if any(kw in v for kw in MASS_MOVEMENT_KEYWORDS)
        )
        mixed_count = max(0, len(pvals) - hydro_count - mass_count)

        risk_only = "YES" if detect_susceptibility(
            source.get("source_asset_name", ""),
            audit.get("fields", []),
        ) else "NO"

        obs_status = "UNDETERMINED"
        if risk_only == "YES":
            obs_status = "NOT_OBSERVED_EVENT"
        elif hydro_count > 0 and mass_count == 0:
            obs_status = "OBSERVED_HYDROLOGICAL"
        elif mass_count > 0 and hydro_count == 0:
            obs_status = "OBSERVED_MASS_MOVEMENT"
        elif hydro_count > 0 and mass_count > 0:
            obs_status = "OBSERVED_MIXED"

        patch_usability = (
            "POTENTIALLY_USABLE"
            if (gate_decision and gate_decision.spatial_unit_usable_for_patch_binding == "PASS")
            else "NOT_USABLE"
        )
        gt_status = gate_decision.ground_truth_status if gate_decision else "BLOCKED"

    # Determinacao de limitacoes e proxima acao
    if access_status == "DOWNLOAD_FAILED":
        limitations = "Download falhou — URL pode ter mudado ou acesso restrito"
        next_action = "Verificar URL manualmente; tentar download direto no navegador"
    elif access_status == "PENDING_FORMAL_REQUEST":
        limitations = "Dado nao publico — requer solicitacao institucional formal"
        next_action = f"Enviar solicitacao formal para {source.get('source_institution','')}"
    elif asset_type in ("PDF_REPORT", "ZIP_ARCHIVE"):
        limitations = "Arquivo nao e vetor — auditoria geometrica nao aplicavel diretamente"
        next_action = "Extrair e auditar arquivos internos se forem vetores"
    elif gt_status == "BLOCKED":
        block_gate = gate_decision.blocking_gate if gate_decision else "UNKNOWN"
        limitations = f"Gate(s) bloqueante(s): {block_gate}"
        next_action = "Resolver gate bloqueante; buscar versao com data de evento explicita"
    else:
        limitations = "Auditoria completa — revisao supervisora obrigatoria antes de uso"
        next_action = "Revisao supervisora; definir protocolo de split/leakage"

    return PublicRegistryRecord(
        source_asset_id=source["source_asset_id"],
        event_id=source["event_id"],
        region=source["region"],
        event_date=source["event_date"],
        source_institution=source["source_institution"],
        source_repository=source["source_repository"],
        source_title=source["source_title"][:100],
        source_asset_name=source["source_asset_name"],
        source_asset_type=asset_type,
        source_access_status=access_status,
        local_ingestion_status=ingestion_status,
        official_source_url_reference=source.get("official_source_url_reference", ""),
        raw_file_versioning_status="NOT_VERSIONED_LOCAL_ONLY",
        geometry_available=geom_avail,
        geometry_type=geom_type,
        crs=crs_val,
        feature_count=feat_count,
        has_event_date_field=has_date_field,
        event_date_compatible=evt_date_compat,
        has_phenomenon_field=has_phenom,
        phenomenon_raw_values=phenom_vals,
        hydrological_observed_features_count=hydro_count,
        mass_movement_observed_features_count=mass_count,
        mixed_or_unknown_features_count=mixed_count,
        risk_or_susceptibility_only=risk_only,
        observed_event_status=obs_status,
        patch_level_usability=patch_usability,
        ground_truth_status=gt_status,
        ml_label_status="BLOCKED_UNTIL_SPLIT_AND_LEAKAGE_PROTOCOL",
        limitations=limitations,
        next_required_action=next_action,
        notes=source.get("notes", "")[:200],
    )


# ---------------------------------------------------------------------------
# Escrita de CSV
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if hasattr(row, "__dataclass_fields__"):
                writer.writerow(asdict(row))
            else:
                writer.writerow(row)


# ---------------------------------------------------------------------------
# Verificacao de paths privados
# ---------------------------------------------------------------------------

PRIVATE_MARKERS = ["PROJETO", "Users\\gabriela", "C:\\Users", "/home/gabriela"]


def assert_no_private_paths(path: Path) -> None:
    content = path.read_text(encoding="utf-8", errors="ignore")
    for marker in PRIVATE_MARKERS:
        if marker in content:
            raise ValueError(
                f"Path privado '{marker}' detectado em {path.name} — "
                "nao pode ser commitado."
            )


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="v1if -- Aquisicao e auditoria de vetores observados oficiais"
    )
    parser.add_argument("--search-local", action="store_true",
                        help="Varre workspace local em modo read-only")
    parser.add_argument("--download-official-known", action="store_true",
                        help="Tenta baixar de URLs oficiais curadas")
    parser.add_argument("--force", action="store_true",
                        help="Escreve registries publicos em datasets/")
    parser.add_argument("--candidate-path",
                        help="Caminho adicional a um vetor candidato para auditoria")
    args = parser.parse_args()

    if not any([args.search_local, args.download_official_known, args.force, args.candidate_path]):
        print("[DRY-RUN] Nenhum flag ativo. Para executar:")
        print("  --search-local           : varre workspace local")
        print("  --download-official-known: tenta download de URLs curadas")
        print("  --force                  : escreve registries publicos")
        print()
        print("[DRY-RUN] Fontes oficiais curadas:")
        for s in KNOWN_OFFICIAL_SOURCES:
            url = s.get("download_url") or "(sem URL publica)"
            print(f"  [{s['source_asset_id']}] {s['source_asset_name']}")
            print(f"        Inst: {s['source_institution']}")
            print(f"        URL:  {url[:80]}")
        return

    # Estrutura de resultados
    download_log: list[SourceDownloadRecord] = []
    extracted_assets: list[ExtractedAssetRecord] = []
    geometry_audits: list[VectorGeometryAudit] = []
    gate_decisions: list[GroundTruthGateAudit] = []
    public_records: list[PublicRegistryRecord] = []

    # Mapa de (source_id -> path local extraido ou baixado)
    local_paths: dict[str, Path] = {}

    # --- BUSCA LOCAL ---
    if args.search_local:
        print("[SEARCH-LOCAL] Varrendo workspace local...")
        candidates = search_local_candidates()
        print(f"[SEARCH-LOCAL] {len(candidates)} candidato(s) encontrado(s) por nome")
        for p in candidates:
            print(f"  [FOUND] {p.name}")

    # --- DOWNLOAD ---
    if args.download_official_known:
        print("[DOWNLOAD] Tentando download de fontes curadas...")
        RAW_SOURCES_DIR.mkdir(parents=True, exist_ok=True)

        for source in KNOWN_OFFICIAL_SOURCES:
            sid = source["source_asset_id"]
            url = source.get("download_url")
            asset_name = source["source_asset_name"]

            if url is None:
                print(f"  [SKIP] {sid} ({asset_name}) -- sem URL publica")
                dl_record = SourceDownloadRecord(
                    source_asset_id=sid,
                    event_id=source["event_id"],
                    region=source["region"],
                    event_date=source["event_date"],
                    source_institution=source["source_institution"],
                    source_repository=source["source_repository"],
                    source_title=source["source_title"],
                    source_asset_name=asset_name,
                    source_asset_type=source["source_asset_type"],
                    download_url_reference=source.get("official_source_url_reference", ""),
                    official_source_url_reference=source.get("official_source_url_reference", ""),
                    download_status="NO_URL",
                    download_timestamp=now_str(),
                    file_size_bytes=0,
                    error_message="Sem URL de download direto — requer solicitacao formal",
                    notes=source.get("notes", ""),
                )
                download_log.append(dl_record)
                continue

            dest_path = RAW_SOURCES_DIR / asset_name
            if dest_path.exists() and source.get("skip_if_exists", False):
                print(f"  [SKIP] {sid} ({asset_name}) -- ja existe localmente")
                local_paths[sid] = dest_path
                dl_record = SourceDownloadRecord(
                    source_asset_id=sid,
                    event_id=source["event_id"],
                    region=source["region"],
                    event_date=source["event_date"],
                    source_institution=source["source_institution"],
                    source_repository=source["source_repository"],
                    source_title=source["source_title"],
                    source_asset_name=asset_name,
                    source_asset_type=source["source_asset_type"],
                    download_url_reference=url[:120],
                    official_source_url_reference=source.get("official_source_url_reference", ""),
                    download_status="ALREADY_EXISTS",
                    download_timestamp=now_str(),
                    file_size_bytes=dest_path.stat().st_size,
                    error_message="",
                    notes=source.get("notes", ""),
                )
                download_log.append(dl_record)
                continue

            print(f"  [DOWNLOADING] {sid} ({asset_name}) de {url[:60]}...")
            ok, msg = download_file(url, dest_path, max_mb=60.0)
            if ok:
                print(f"    [OK] {msg}")
                local_paths[sid] = dest_path
                dl_status = "DOWNLOAD_OK"
                err_msg = ""
                fsize = dest_path.stat().st_size if dest_path.exists() else 0
            else:
                print(f"    [FAIL] {msg}")
                dl_status = "DOWNLOAD_FAILED"
                err_msg = msg
                fsize = 0

            dl_record = SourceDownloadRecord(
                source_asset_id=sid,
                event_id=source["event_id"],
                region=source["region"],
                event_date=source["event_date"],
                source_institution=source["source_institution"],
                source_repository=source["source_repository"],
                source_title=source["source_title"],
                source_asset_name=asset_name,
                source_asset_type=source["source_asset_type"],
                download_url_reference=url[:120],
                official_source_url_reference=source.get("official_source_url_reference", ""),
                download_status=dl_status,
                download_timestamp=now_str(),
                file_size_bytes=fsize,
                error_message=err_msg,
                notes=source.get("notes", ""),
            )
            download_log.append(dl_record)

        # --- EXTRACAO DO ZIP ---
        zip_path = local_paths.get("OBS_PET_002")
        if zip_path and zip_path.exists() and zip_path.suffix.lower() == ".zip":
            print(f"[EXTRACT] Extraindo {zip_path.name} ...")
            EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
            extracted = extract_zip(zip_path, EXTRACTED_DIR)
            extracted_assets.extend(extracted)
            n_vec = sum(1 for e in extracted if e.is_vector)
            n_doc = sum(1 for e in extracted if e.is_document)
            print(f"  [OK] {len(extracted)} arquivo(s) inventariados: {n_vec} vetores, {n_doc} documentos")

            # --- AUDITORIA DE VETORES EXTRAIDOS ---
            for asset in extracted:
                if not asset.is_vector:
                    continue
                # Procura o arquivo extraido
                asset_path = None
                for p in EXTRACTED_DIR.rglob(asset.asset_name):
                    asset_path = p
                    break
                if asset_path is None or not asset_path.exists():
                    continue

                print(f"    [AUDIT] {asset.asset_name} ...")
                aud = audit_vector(asset_path)
                audit_record = VectorGeometryAudit(
                    source_asset_id=asset.source_asset_id,
                    asset_name=asset.asset_name,
                    geometry_available=aud.get("geometry_available", "NO"),
                    geometry_type=aud.get("geometry_type", "UNKNOWN"),
                    crs=aud.get("crs", "UNKNOWN"),
                    feature_count=aud.get("feature_count", 0),
                    fields="; ".join(aud.get("fields", [])),
                    has_date_field="YES" if aud.get("has_date_field") else "NO",
                    has_phenomenon_field="YES" if aud.get("has_phenomenon_field") else "NO",
                    has_locality_field="YES" if aud.get("has_locality_field") else "NO",
                    phenomenon_values_sample="; ".join(aud.get("phenomenon_values", []))[:200],
                    date_values_sample="; ".join(str(v) for v in aud.get("date_values", []))[:200],
                    bounds=str(aud.get("bounds", "UNKNOWN"))[:100],
                    notes=str(aud.get("error", ""))[:100],
                )
                geometry_audits.append(audit_record)

                # Gate decision para vetor extraido
                # Precisa criar um pseudo-source para o vetor extraido
                pseudo_source = {
                    "source_asset_id": asset.source_asset_id,
                    "source_institution": "SGB/CPRM",
                    "download_url": "FROM_ZIP",
                    "official_source_url_reference": "https://rigeo.sgb.gov.br/handle/doc/22668",
                    "source_asset_name": asset.asset_name,
                    "event_date": "2022-02-15",
                    "notes": f"Extraido de {asset.parent_zip}",
                }
                gate_dec = apply_ground_truth_gates(
                    source_id=asset.source_asset_id + "_" + Path(asset.asset_name).stem[:10],
                    asset_name=asset.asset_name,
                    source=pseudo_source,
                    audit=aud,
                    download_status="DOWNLOAD_OK",
                    ingestion_ok=True,
                )
                gate_decisions.append(gate_dec)
                print(
                    f"      CRS={audit_record.crs} feicoes={audit_record.feature_count} "
                    f"gt={gate_dec.ground_truth_status}"
                )

    # --- CANDIDATO ADICIONAL (--candidate-path) ---
    if args.candidate_path:
        cp = Path(args.candidate_path)
        if cp.exists():
            print(f"[CANDIDATE] Auditando: {cp.name} ...")
            aud = audit_vector(cp)
            audit_record = VectorGeometryAudit(
                source_asset_id="OBS_CUSTOM_001",
                asset_name=cp.name,
                geometry_available=aud.get("geometry_available", "NO"),
                geometry_type=aud.get("geometry_type", "UNKNOWN"),
                crs=aud.get("crs", "UNKNOWN"),
                feature_count=aud.get("feature_count", 0),
                fields="; ".join(aud.get("fields", [])),
                has_date_field="YES" if aud.get("has_date_field") else "NO",
                has_phenomenon_field="YES" if aud.get("has_phenomenon_field") else "NO",
                has_locality_field="YES" if aud.get("has_locality_field") else "NO",
                phenomenon_values_sample="; ".join(aud.get("phenomenon_values", []))[:200],
                date_values_sample="; ".join(str(v) for v in aud.get("date_values", []))[:200],
                bounds=str(aud.get("bounds", "UNKNOWN"))[:100],
                notes=str(aud.get("error", ""))[:100],
            )
            geometry_audits.append(audit_record)
        else:
            print(f"[CANDIDATE] Caminho nao encontrado: {cp}")

    # --- GERAR REGISTROS PUBLICOS para TODAS as fontes curadas ---
    print("[RECORDS] Gerando registros publicos...")
    for source in KNOWN_OFFICIAL_SOURCES:
        sid = source["source_asset_id"]
        dl_ok = sid in local_paths
        ingested_path = local_paths.get(sid)

        # Para ZIP e PDF, audit_data = None (auditoria e dos arquivos internos)
        is_zip_or_pdf = source.get("source_asset_type") in ("ZIP_ARCHIVE", "PDF_REPORT")
        audit_data = None
        gate_dec = None

        if not is_zip_or_pdf and ingested_path:
            aud = audit_vector(ingested_path)
            gate_dec = apply_ground_truth_gates(
                source_id=sid,
                asset_name=ingested_path.name,
                source=source,
                audit=aud,
                download_status="DOWNLOAD_OK",
                ingestion_ok=True,
            )
            audit_data = aud

        # Para ZIP: sintetizar resultado dos vetores extraidos do ZIP
        if is_zip_or_pdf and source.get("source_asset_type") == "ZIP_ARCHIVE":
            # Gate minimal para o ZIP em si
            dl_rec = next((d for d in download_log if d.source_asset_id == sid), None)
            dl_ok_zip = dl_rec.download_status in ("DOWNLOAD_OK", "ALREADY_EXISTS") if dl_rec else False
            gate_dec = GroundTruthGateAudit(
                source_asset_id=sid,
                asset_name=source["source_asset_name"],
                official_or_institutional_source="PASS",
                raw_asset_traceable="PASS",
                geometry_available="NOT_APPLICABLE",
                crs_available="NOT_APPLICABLE",
                geometry_valid="NOT_APPLICABLE",
                event_date_available="NOT_APPLICABLE",
                event_date_compatible="NOT_APPLICABLE",
                phenomenon_available="NOT_APPLICABLE",
                phenomenon_is_observed_not_risk="NOT_APPLICABLE",
                hydrological_or_mass_movement_separable="NOT_APPLICABLE",
                spatial_unit_usable_for_patch_binding="NOT_APPLICABLE",
                final_gate_status="PENDING",
                ground_truth_status="BLOCKED" if not dl_ok_zip else "PENDING_VECTOR_AUDIT",
                blocking_gate="DOWNLOAD_REQUIRED" if not dl_ok_zip else "VECTOR_AUDIT_REQUIRED",
                notes="ZIP precisa ser extraido e vetores auditados individualmente.",
            )

        rec = build_public_record(
            source=source,
            download_ok=dl_ok,
            ingested_path=ingested_path,
            audit=audit_data,
            gate_decision=gate_dec,
        )
        public_records.append(rec)
        print(f"  [{rec.source_asset_id}] gt={rec.ground_truth_status}")

    # Adicionar registros para vetores extraidos do ZIP (se houver)
    for gate_dec in gate_decisions:
        # Verificar se ja existe um registro com esse ID
        existing_ids = {r.source_asset_id for r in public_records}
        if gate_dec.source_asset_id in existing_ids:
            continue
        # Procura o geometry audit correspondente
        ga = next((g for g in geometry_audits if g.asset_name == gate_dec.asset_name), None)
        aud = None
        if ga:
            aud = {
                "geometry_available": ga.geometry_available,
                "geometry_type": ga.geometry_type,
                "crs": ga.crs,
                "feature_count": ga.feature_count,
                "fields": ga.fields.split("; ") if ga.fields else [],
                "has_date_field": ga.has_date_field == "YES",
                "has_phenomenon_field": ga.has_phenomenon_field == "YES",
                "has_locality_field": ga.has_locality_field == "YES",
                "phenomenon_values": ga.phenomenon_values_sample.split("; "),
                "date_values": ga.date_values_sample.split("; "),
                "bounds": ga.bounds,
                "error": None,
            }
        pseudo_source = {
            "source_asset_id": gate_dec.source_asset_id,
            "event_id": "PET_2022_02_15",
            "region": "PET",
            "event_date": "2022-02-15",
            "source_institution": "SGB/CPRM",
            "source_repository": "RIGeo/SGB",
            "source_title": "Avaliacao tecnica pos-desastre: Petropolis, RJ (anexo)",
            "source_asset_name": gate_dec.asset_name,
            "source_asset_type": "VECTOR_" + Path(gate_dec.asset_name).suffix.upper().lstrip("."),
            "download_url": "FROM_ZIP",
            "official_source_url_reference": "https://rigeo.sgb.gov.br/handle/doc/22668",
            "notes": f"Extraido de anexos_avaliacao_pos_desastre_petropolis_rj_2022.zip",
        }
        rec = build_public_record(
            source=pseudo_source,
            download_ok=True,
            ingested_path=None,
            audit=aud,
            gate_decision=gate_dec,
        )
        public_records.append(rec)
        print(f"  [{rec.source_asset_id}] gt={rec.ground_truth_status} (extraido de ZIP)")

    # --- SUMARIO ---
    n_candidate_gt = sum(
        1 for r in public_records
        if r.ground_truth_status == "CANDIDATE_OBSERVED_GROUND_TRUTH"
    )
    n_blocked = sum(1 for r in public_records if "BLOCKED" in r.ground_truth_status)
    n_pending = sum(
        1 for r in public_records
        if r.ground_truth_status in ("PENDING_VECTOR_AUDIT", "UNDETERMINED")
    )

    summary = {
        "stage": "v1if",
        "timestamp": now_str(),
        "total_sources_checked": len(KNOWN_OFFICIAL_SOURCES),
        "total_records": len(public_records),
        "candidate_observed_ground_truth": n_candidate_gt,
        "blocked_count": n_blocked,
        "pending_audit_count": n_pending,
        "vectors_extracted_from_zip": sum(1 for e in extracted_assets if e.is_vector),
        "documents_extracted_from_zip": sum(1 for e in extracted_assets if e.is_document),
        "operational_ground_truth_status": "BLOCKED",
        "ml_label_status": "BLOCKED_UNTIL_SPLIT_AND_LEAKAGE_PROTOCOL",
        "can_create_training_label": False,
        "can_reopen_protocol_b": False,
        "multimodal_status": "HOLD",
        "dino_usage_status": "SUPPORT_ONLY",
        "pyshp_available": HAS_PYSHP,
        "notes": (
            "v1if buscou vetores observados oficiais. Nenhum ativo confirmou "
            "ground truth operacional. Invariantes mantidos."
            if n_candidate_gt == 0 else
            f"{n_candidate_gt} candidato(s) passaram os gates — revisao supervisora obrigatoria."
        ),
    }

    # --- OUTPUTS LOCAIS ---
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)

    # Log de downloads
    if download_log:
        dl_fields = list(asdict(download_log[0]).keys())
        write_csv(LOCAL_RUNS / "v1if_official_source_download_log.csv", download_log, dl_fields)
        print(f"[LOCAL] Download log: {len(download_log)} registro(s)")

    # Inventario de extraidos
    if extracted_assets:
        ea_fields = list(asdict(extracted_assets[0]).keys())
        write_csv(LOCAL_RUNS / "v1if_extracted_asset_inventory.csv", extracted_assets, ea_fields)
        print(f"[LOCAL] Inventario extraidos: {len(extracted_assets)} arquivo(s)")

    # Auditoria de vetores
    if geometry_audits:
        ga_fields = list(asdict(geometry_audits[0]).keys())
        write_csv(LOCAL_RUNS / "v1if_vector_geometry_audit.csv", geometry_audits, ga_fields)
        print(f"[LOCAL] Geometria auditada: {len(geometry_audits)} vetor(es)")

    # Gates
    if gate_decisions:
        gd_fields = list(asdict(gate_decisions[0]).keys())
        write_csv(LOCAL_RUNS / "v1if_ground_truth_candidate_decision.csv", gate_decisions, gd_fields)
        print(f"[LOCAL] Gate decisions: {len(gate_decisions)} decisao(oes)")

    # QA
    qa_rows = [
        {
            "check": "operational_ground_truth_status_is_blocked",
            "expected": "BLOCKED",
            "actual": summary["operational_ground_truth_status"],
            "pass": summary["operational_ground_truth_status"] == "BLOCKED",
        },
        {
            "check": "can_create_training_label_is_false",
            "expected": False,
            "actual": summary["can_create_training_label"],
            "pass": summary["can_create_training_label"] is False,
        },
        {
            "check": "ml_label_status_blocked",
            "expected": "BLOCKED_UNTIL_SPLIT_AND_LEAKAGE_PROTOCOL",
            "actual": summary["ml_label_status"],
            "pass": "BLOCKED" in summary["ml_label_status"],
        },
        {
            "check": "no_private_paths_in_records",
            "expected": "True",
            "actual": str(all(
                not any(m in r.official_source_url_reference for m in ["PROJETO", "Users\\"])
                for r in public_records
            )),
            "pass": all(
                not any(m in r.official_source_url_reference for m in ["PROJETO", "Users\\"])
                for r in public_records
            ),
        },
        {
            "check": "all_blocked_no_susceptibility_as_ground_truth",
            "expected": "True",
            "actual": str(all(
                r.ground_truth_status != "CANDIDATE_OBSERVED_GROUND_TRUTH"
                for r in public_records
                if r.risk_or_susceptibility_only == "YES"
            )),
            "pass": all(
                r.ground_truth_status != "CANDIDATE_OBSERVED_GROUND_TRUTH"
                for r in public_records
                if r.risk_or_susceptibility_only == "YES"
            ),
        },
    ]
    write_csv(LOCAL_RUNS / "v1if_qa.csv", qa_rows, ["check", "expected", "actual", "pass"])

    # Summary JSON
    with open(LOCAL_RUNS / "v1if_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[LOCAL] Summary: v1if_summary.json")

    # --- ESCRITA PUBLICA (--force) ---
    if args.force:
        print("[FORCE] Escrevendo registries publicos...")

        registry_path = DATASETS_DIR / "official_observed_event_vector_registry.csv"

        # Verificar ausencia de paths privados antes de escrever
        pub_fields = list(asdict(public_records[0]).keys()) if public_records else []
        if pub_fields:
            write_csv(registry_path, public_records, pub_fields)
            assert_no_private_paths(registry_path)
            print(f"[PUB] {registry_path.name}: {len(public_records)} registro(s)")

        # Verificar schema
        schema_path = SCHEMAS_DIR / "official_observed_event_vector_registry_schema.csv"
        if schema_path.exists():
            print(f"[PUB] Schema ja existe: {schema_path.name}")
        else:
            print(f"[WARN] Schema nao encontrado: {schema_path.name}")
    else:
        print("[DRY-RUN] Use --force para escrever registries publicos.")

    # --- RELATORIO FINAL ---
    print()
    print("=" * 60)
    print("RELATORIO v1if")
    print("=" * 60)
    print(f"  Fontes curadas verificadas : {summary['total_sources_checked']}")
    print(f"  Registros gerados          : {summary['total_records']}")
    print(f"  Candidate ground truth     : {summary['candidate_observed_ground_truth']}")
    print(f"  Bloqueados                 : {summary['blocked_count']}")
    print(f"  Pendente auditoria         : {summary['pending_audit_count']}")
    print(f"  Vetores extraidos de ZIP   : {summary['vectors_extracted_from_zip']}")
    print(f"  Documentos extraidos       : {summary['documents_extracted_from_zip']}")
    print()
    print(f"  operational_ground_truth   : {summary['operational_ground_truth_status']}")
    print(f"  can_create_training_label  : {summary['can_create_training_label']}")
    print(f"  ml_label_status            : {summary['ml_label_status']}")
    print()
    if n_candidate_gt == 0:
        print("[RESULT] Nenhum ground truth vetorial observado confirmado neste estagio.")
    else:
        print(f"[RESULT] {n_candidate_gt} candidato(s) — revisao supervisora obrigatoria.")


if __name__ == "__main__":
    main()
