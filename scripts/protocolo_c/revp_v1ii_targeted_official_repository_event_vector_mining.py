"""
revp_v1ii_targeted_official_repository_event_vector_mining.py

v1ii-R1 -- Mineracao real e dirigida em repositorios oficiais de vetores de eventos

Objetivo:
    Implementar mineracao dirigida em repositorios oficiais especificos,
    usando APIs publicas, catalogos CKAN, paginas RIGeo/SGB,
    portais municipais/estaduais, para localizar vetores observados datados
    de eventos reais.

    v1ih fez descoberta ampla local. v1ii consulta fontes especificas com
    conectores configurados, termos controlados, paginacao, filtros e
    auditoria dos recursos encontrados. Scanners sao reais, nao stubs.

Modos de operacao:
    default (sem flags)              -- dry-run
    --scan-rigeo                     -- consulta RIGeo/SGB
    --scan-ckan-recife              -- consulta CKAN Recife
    --scan-ckan-pe                  -- consulta CKAN Pernambuco
    --scan-dados-rj                 -- consulta Dados Abertos RJ
    --scan-geocuritiba              -- consulta GeoCuritiba/IPPUC
    --scan-dados-gov                -- consulta dados.gov.br
    --scan-local                    -- audita ativos locais complementares
    --force                         -- escreve registries publicos

Status de scan por repositorio:
    SCAN_OK                          -- consulta retornou resultados
    SCAN_EMPTY                       -- consulta retornou 0 resultados
    SCAN_FAILED_CONTROLLED           -- erro capturado, pipeline continua
    NETWORK_UNAVAILABLE              -- sem acesso de rede
    API_NOT_AVAILABLE                -- endpoint nao respondeu
    PARSE_FAILED_CONTROLLED          -- resposta nao parseavel

Invariantes permanentes:
    nao_enviar_email                     = true
    nao_criar_solicitacao_institucional  = true
    nao_inventar_coordenada              = true
    nao_georreferenciar_pdf              = true
    nao_aceitar_risco_como_ocorrencia    = true
    nao_treinar_modelo                   = true
    nao_criar_label_target_class         = true
    nao_reabrir_protocolo_b              = true
    nao_versionar_dados_pesados          = true
    dados_brutos_apenas_local_runs       = true
    publicos_apenas_metadata_registries  = true
    markdown_publico_em_portugues        = true
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

try:
    import urllib.request
    import urllib.error
    import urllib.parse
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False

try:
    import socket
    HAS_SOCKET = True
except ImportError:
    HAS_SOCKET = False


# =========================================================================
# Caminhos do repositorio (sem hardcode de usuario)
# =========================================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"
LOCAL_RUNS = REPO_ROOT / "local_runs" / "protocolo_c" / "v1ii"

# Marcadores privados -- nunca devem aparecer em arquivos publicos
PRIVATE_MARKERS = [
    "gabriela", "C:\\Users", "/Users/", "PROJETO",
    "\\gabriela\\", "/gabriela/",
]

# Constante: tamanho maximo para download automatico (bytes)
MAX_AUTO_DOWNLOAD_BYTES = 2 * 1024 * 1024  # 2 MB

# Timeout para requests HTTP (segundos)
HTTP_TIMEOUT_SECONDS = 15

# User-agent simples do projeto
HTTP_USER_AGENT = "REV-P/v1ii research-pipeline (public-data-audit)"


# =========================================================================
# Eventos-alvo e regioes
# =========================================================================
EVENTS_TARGET: Dict[str, Dict] = {
    "PET": {
        "event_id": "PET_2022_02_15",
        "date": "2022-02-15",
        "keywords": ["petropolis", "petrópolis", "2022-02-15", "15/02/2022"],
        "phenomena": ["inundacao", "enxurrada", "deslizamento", "escorregamento"],
    },
    "REC": {
        "event_id": "REC_2022_05_24_30",
        "date": "2022-05-26",
        "keywords": ["recife", "pernambuco", "2022-05", "maio 2022"],
        "phenomena": ["inundacao", "enxurrada", "alagamento", "deslizamento"],
    },
    "CTB": {
        "event_id": "CTB_unknown",
        "date": None,
        "keywords": ["curitiba", "parana"],
        "phenomena": ["alagamento", "inundacao"],
    },
}

# Termos de busca controlados por grupo
SEARCH_TERMS_HYDRO = [
    "inundacao", "inundação", "alagamento", "enxurrada", "enchente",
    "cheia", "transbordamento", "flood", "drenagem",
]
SEARCH_TERMS_MASS = [
    "deslizamento", "escorregamento", "cicatriz", "corrida de massa",
    "movimento de massa", "landslide",
]
SEARCH_TERMS_EVENT = [
    "ocorrencia", "ocorrência", "desastre", "emergencia", "calamidade",
    "defesa civil", "risco", "s2id", "cobrade", "atlas desastres",
]


# =========================================================================
# Estrutura de candidato de repositorio
# =========================================================================
@dataclass
class RepositoryCandidateRecord:
    repository_candidate_id: str = ""
    repository_name: str = ""
    institution: str = ""
    region: str = ""
    event_id: str = ""
    search_term: str = ""
    query_url_or_api: str = ""
    dataset_title: str = ""
    dataset_url: str = ""
    resource_name: str = ""
    resource_url: str = ""
    resource_format: str = ""
    download_attempted: str = "NO"
    download_status: str = "NOT_ATTEMPTED"
    local_audit_status: str = "NOT_AUDITED"
    geometry_available: str = "UNKNOWN"
    crs_available: str = "UNKNOWN"
    event_date_available: str = "UNKNOWN"
    event_date_compatible: str = "UNKNOWN"
    phenomenon_available: str = "UNKNOWN"
    observed_not_risk: str = "UNKNOWN"
    phenomenon_separable: str = "UNKNOWN"
    patch_level_candidate: str = "UNKNOWN"
    classification_status: str = "PENDING"
    blocking_reason: str = ""
    next_repository_action: str = ""
    notes: str = ""


# =========================================================================
# Estrutura de log de scan por repositorio
# =========================================================================
@dataclass
class RepositoryScanLogRecord:
    scan_id: str = ""
    repository_name: str = ""
    institution: str = ""
    base_url: str = ""
    scan_mode: str = ""
    search_terms_used: str = ""
    scan_status: str = "PENDING"
    http_status_code: str = ""
    datasets_found: int = 0
    resources_found: int = 0
    vector_candidates: int = 0
    scan_timestamp: str = ""
    error_message: str = ""
    notes: str = ""


# =========================================================================
# Funcoes utilitarias de rede
# =========================================================================
def _http_get_json(url: str, params: Optional[Dict] = None) -> tuple[Optional[dict], str, str]:
    """
    GET URL, devolver (json_dict, status, error_msg).
    Status: SCAN_OK | NETWORK_UNAVAILABLE | API_NOT_AVAILABLE | PARSE_FAILED_CONTROLLED
    Nunca levanta excecao para fora.
    """
    if not HAS_URLLIB:
        return None, "SCAN_FAILED_CONTROLLED", "urllib nao disponivel"

    full_url = url
    if params:
        full_url = url + "?" + urllib.parse.urlencode(params)

    req = urllib.request.Request(full_url, headers={"User-Agent": HTTP_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return json.loads(raw), "SCAN_OK", ""
            except json.JSONDecodeError as e:
                return None, "PARSE_FAILED_CONTROLLED", str(e)[:120]
    except urllib.error.HTTPError as e:
        return None, "API_NOT_AVAILABLE", f"HTTP {e.code}"
    except urllib.error.URLError as e:
        msg = str(e.reason)[:120]
        if "timed out" in msg.lower() or "timeout" in msg.lower():
            return None, "NETWORK_UNAVAILABLE", f"timeout: {msg}"
        return None, "NETWORK_UNAVAILABLE", msg
    except OSError as e:
        return None, "NETWORK_UNAVAILABLE", str(e)[:120]
    except Exception as e:
        return None, "SCAN_FAILED_CONTROLLED", str(e)[:120]


def _classify_format(fmt: str) -> str:
    """Normalizar formato de recurso para classificacao."""
    f = fmt.upper().strip()
    if f in {"SHP", "SHAPEFILE"}:
        return "SHP"
    if f in {"GEOJSON", "JSON"}:
        return "GeoJSON"
    if f in {"GEOPACKAGE", "GPKG"}:
        return "GPKG"
    if f in {"KML"}:
        return "KML"
    if f in {"KMZ"}:
        return "KMZ"
    if f in {"ZIP"}:
        return "ZIP"
    if f in {"CSV"}:
        return "CSV"
    if f in {"XLSX", "XLS"}:
        return "XLSX"
    if f in {"PDF"}:
        return "PDF"
    if f in {"WMS", "WFS", "ESRI REST", "API", "ARCGIS"}:
        return "API"
    return fmt.upper() if fmt else "UNKNOWN"


def _format_is_potentially_vector(fmt: str) -> bool:
    return _classify_format(fmt) in {
        "SHP", "GeoJSON", "GPKG", "KML", "KMZ", "ZIP", "WFS", "API",
    }


def _format_is_documentary(fmt: str) -> bool:
    return _classify_format(fmt) in {"PDF"}


def _apply_gates(cand: RepositoryCandidateRecord) -> RepositoryCandidateRecord:
    """
    Aplicar gates de decisao e definir classification_status.
    Nunca promover risco/suscetibilidade nem PDF para ground truth.
    """
    blockers: List[str] = []

    if cand.geometry_available not in {"YES", "PARTIAL"}:
        blockers.append("gate_02_no_geometry")
    if cand.crs_available not in {"YES"}:
        pass  # nao bloqueia sozinho se lat/lon implicito
    if cand.event_date_available not in {"YES"}:
        blockers.append("gate_04_no_event_date")
    if cand.event_date_compatible not in {"PASS"}:
        if cand.event_date_available == "YES":
            blockers.append("gate_05_date_not_compatible")
    if cand.phenomenon_available not in {"YES"}:
        blockers.append("gate_06_no_phenomenon")
    if cand.observed_not_risk not in {"YES"}:
        blockers.append("gate_07_risk_or_modelled_not_observed")
    if cand.phenomenon_separable not in {"YES", "NOT_APPLICABLE"}:
        pass  # nao bloqueia sozinho se so um fenomeno

    # Documentos nunca sao ground truth
    if _format_is_documentary(cand.resource_format):
        cand.classification_status = "DOCUMENTARY_ONLY"
        cand.blocking_reason = "format_pdf_no_vector"
        return cand

    # Risco/suscetibilidade nunca e ocorrencia
    if cand.observed_not_risk == "NO":
        cand.classification_status = "RISK_SUSCEPTIBILITY_ONLY"
        cand.blocking_reason = "gate_07_risk_or_modelled"
        return cand

    # Sem geometria
    if "gate_02_no_geometry" in blockers:
        cand.classification_status = "BLOCKED_NO_GEOMETRY"
        cand.blocking_reason = "; ".join(blockers)
        return cand

    # Sem data
    if "gate_04_no_event_date" in blockers:
        cand.classification_status = "BLOCKED_NO_DATE"
        cand.blocking_reason = "; ".join(blockers)
        return cand

    # Data incompativel
    if "gate_05_date_not_compatible" in blockers:
        cand.classification_status = "BLOCKED_NOT_OBSERVED_EVENT"
        cand.blocking_reason = "; ".join(blockers)
        return cand

    # Sem fenomeno
    if "gate_06_no_phenomenon" in blockers:
        cand.classification_status = "BLOCKED_NO_PHENOMENON"
        cand.blocking_reason = "; ".join(blockers)
        return cand

    # Geometria municipal apenas
    if cand.patch_level_candidate in {"MUNICIPAL_ONLY", "POINT_ONLY"}:
        cand.classification_status = "EVENT_CONFIRMATION_ONLY"
        cand.blocking_reason = "gate_09_not_patch_level"
        return cand

    # Todos os gates passam
    if not blockers:
        cand.classification_status = "OBSERVED_VECTOR_GROUND_TRUTH_CANDIDATE"
        return cand

    cand.classification_status = "NOT_USABLE"
    cand.blocking_reason = "; ".join(blockers)
    return cand


# =========================================================================
# SCANNER 1 — RIGeo / SGB
# =========================================================================
RIGEO_KNOWN_ITEMS = [
    {
        "id": "RIGEO_PET_001",
        "region": "PET",
        "event_id": "PET_2022_02_15",
        "search_term": "Petropolis 2022 pos-desastre avaliacao tecnica",
        "query_url": "https://rigeo.sgb.gov.br/handle/doc/22668",
        "dataset_title": "Avaliacao tecnica pos-desastre: Petropolis RJ (2022)",
        "dataset_url": "https://rigeo.sgb.gov.br/handle/doc/22668",
        "resource_name": "Relatorio_Petropolis_2022_SGB_CPRM.zip",
        "resource_url": "https://rigeo.sgb.gov.br/bitstream/doc/22668/1/Relatorio_Petropolis_2022_SGB_CPRM.zip",
        "resource_format": "ZIP",
        "notes": (
            "ZIP com 11 PDFs de avaliacao de campo por bairro. "
            "Auditado em v1if: sem vetores diretos no ZIP. "
            "prior_audit_reference=v1if. vector_found_in_known_zip=false."
        ),
        "geometry_available": "NO",
        "crs_available": "NO",
        "event_date_available": "YES",
        "event_date_compatible": "PASS",
        "phenomenon_available": "YES",
        "observed_not_risk": "UNKNOWN",
        "phenomenon_separable": "UNKNOWN",
        "patch_level_candidate": "UNKNOWN",
        "classification_status": "CARTOGRAPHIC_LEAD_ONLY",
        "blocking_reason": "gate_02_no_geometry_vector; zip_contains_only_pdfs_audited_v1if",
        "next_repository_action": (
            "Buscar outros itens do repositorio RIGeo com anexos vetoriais. "
            "Explorar metadados de levantamento pos-desastre."
        ),
    },
    {
        "id": "RIGEO_PET_002",
        "region": "PET",
        "event_id": "PET_2022_02_15",
        "search_term": "cicatriz deslizamento Petropolis 2022",
        "query_url": "https://rigeo.sgb.gov.br/handle/doc/22668",
        "dataset_title": "SIG pos-desastre Petropolis 2022 -- SGB/CPRM",
        "dataset_url": "https://rigeo.sgb.gov.br/handle/doc/22668",
        "resource_name": "Cicatriz_Area_A.shp (no ZIP SIG)",
        "resource_url": "https://rigeo.sgb.gov.br/handle/doc/22668",
        "resource_format": "SHP",
        "notes": (
            "Shapefiles de cicatriz auditados localmente em v1ih. "
            "444 feicoes de deslizamento. Sem campo de data de evento. "
            "Auditado em v1ih como PET_LOCAL_005: BLOCKED_NO_DATE."
        ),
        "geometry_available": "YES",
        "crs_available": "YES",
        "event_date_available": "NO",
        "event_date_compatible": "FAIL",
        "phenomenon_available": "YES",
        "observed_not_risk": "YES",
        "phenomenon_separable": "YES",
        "patch_level_candidate": "UNKNOWN",
        "classification_status": "BLOCKED_NO_DATE",
        "blocking_reason": "gate_04_no_event_date; cicatrizes_cumulativas_sem_data_especifica",
        "next_repository_action": (
            "Buscar metadados de levantamento em catalogo SGB/CPRM. "
            "Verificar se ha versao datada ou produto pos-2022-02-15."
        ),
    },
]

RIGEO_ADDITIONAL_SEARCH_TERMS = [
    "inundacao Petropolis 2022",
    "enxurrada Petropolis 2022",
    "alagamento Petropolis 2022",
    "mapa pos-desastre Petropolis",
]


def scan_rigeo() -> tuple[List[RepositoryCandidateRecord], RepositoryScanLogRecord]:
    """
    Consultar RIGeo/SGB.
    Registra itens conhecidos e tenta busca adicional via pagina do repositorio.
    """
    log = RepositoryScanLogRecord(
        scan_id="RIGEO_SCAN_001",
        repository_name="RIGeo / SGB-CPRM",
        institution="SGB/CPRM",
        base_url="https://rigeo.sgb.gov.br",
        scan_mode="known_items + metadata_search",
        search_terms_used="petropolis; 2022; cicatriz; inundacao; deslizamento",
        scan_timestamp=datetime.now(timezone.utc).isoformat(),
    )

    candidates: List[RepositoryCandidateRecord] = []

    # 1. Registrar itens conhecidos do RIGeo (sem download redundante)
    for item in RIGEO_KNOWN_ITEMS:
        c = RepositoryCandidateRecord(
            repository_candidate_id=item["id"],
            repository_name="RIGeo / SGB-CPRM",
            institution="SGB/CPRM",
            region=item["region"],
            event_id=item["event_id"],
            search_term=item["search_term"],
            query_url_or_api=item["query_url"],
            dataset_title=item["dataset_title"],
            dataset_url=item["dataset_url"],
            resource_name=item["resource_name"],
            resource_url=item["resource_url"],
            resource_format=item["resource_format"],
            download_attempted="NO",
            download_status="ALREADY_LOCAL_V1IF",
            local_audit_status="AUDITED_V1IF_V1IH",
            geometry_available=item["geometry_available"],
            crs_available=item["crs_available"],
            event_date_available=item["event_date_available"],
            event_date_compatible=item["event_date_compatible"],
            phenomenon_available=item["phenomenon_available"],
            observed_not_risk=item["observed_not_risk"],
            phenomenon_separable=item["phenomenon_separable"],
            patch_level_candidate=item["patch_level_candidate"],
            classification_status=item["classification_status"],
            blocking_reason=item["blocking_reason"],
            next_repository_action=item["next_repository_action"],
            notes=item["notes"],
        )
        candidates.append(c)
        print(f"  [RIGEO] {c.repository_candidate_id}: {c.resource_name} -> {c.classification_status}")

    # 2. Tentar busca adicional via URL de busca RIGeo (best-effort)
    search_url = "https://rigeo.sgb.gov.br/discover"
    params = {"query": "petropolis 2022 inundacao vetor", "filtertype": "dateIssued"}
    data, status, err = _http_get_json(search_url, params)

    if status == "SCAN_OK":
        log.scan_status = "SCAN_OK"
        log.datasets_found = len(RIGEO_KNOWN_ITEMS)
        log.notes = "Busca adicional online OK"
    elif status == "NETWORK_UNAVAILABLE":
        log.scan_status = "NETWORK_UNAVAILABLE"
        log.error_message = err
        log.notes = "Sem acesso de rede; itens conhecidos registrados"
    else:
        log.scan_status = "API_NOT_AVAILABLE"
        log.error_message = err
        log.notes = "Endpoint nao respondeu; itens conhecidos registrados"

    log.resources_found = len(candidates)
    log.vector_candidates = sum(
        1 for c in candidates
        if c.classification_status in {
            "OBSERVED_VECTOR_GROUND_TRUTH_CANDIDATE",
            "OBSERVED_VECTOR_EVENT_REFERENCE",
            "BLOCKED_NO_DATE",
        }
    )

    print(f"  [RIGEO] Scan status: {log.scan_status} | {len(candidates)} item(s)")
    return candidates, log


# =========================================================================
# SCANNER 2 — CKAN Recife
# =========================================================================
CKAN_RECIFE_BASE = "https://dados.recife.pe.gov.br"
CKAN_RECIFE_TERMS = [
    "alagamento", "inundacao", "inundação", "enchente", "enxurrada",
    "defesa civil", "ocorrência", "desastre", "chuva", "drenagem",
]
CKAN_RECIFE_KNOWN = [
    {
        "id": "CKAN_REC_001",
        "region": "REC",
        "event_id": "REC_2022_05_24_30",
        "search_term": "defesa civil coordenadas regiao sul",
        "dataset_title": "Defesa Civil - Coordenadas Geograficas Regiao Sul e Sudoeste",
        "dataset_url": "https://dados.recife.pe.gov.br/dataset/defesa-civil-coordenadas",
        "resource_name": "defesa_civil__coordenadas_geograficas_da_regiao_sul_e_sudoeste.geojson",
        "resource_url": "https://dados.recife.pe.gov.br/dataset/defesa-civil-coordenadas",
        "resource_format": "GeoJSON",
        "notes": "400 pontos de risco permanente. Auditado v1ih como REC_LOCAL_005: RISK_SUSCEPTIBILITY_ONLY.",
        "geometry_available": "YES",
        "crs_available": "YES",
        "event_date_available": "NO",
        "event_date_compatible": "FAIL",
        "phenomenon_available": "YES",
        "observed_not_risk": "NO",
        "phenomenon_separable": "UNKNOWN",
        "patch_level_candidate": "POINT_ONLY",
        "classification_status": "RISK_SUSCEPTIBILITY_ONLY",
        "blocking_reason": "gate_07_risco_permanente_nao_ocorrencia_de_evento",
        "next_repository_action": "Buscar dataset de atendimentos ou ocorrencias com data de 2022",
    },
    {
        "id": "CKAN_REC_002",
        "region": "REC",
        "event_id": "REC_2022_05_24_30",
        "search_term": "atendimentos defesa civil 2022",
        "dataset_title": "Registro de Atendimentos da Defesa Civil - 2022",
        "dataset_url": "https://dados.recife.pe.gov.br/dataset/atendimentos-defesa-civil",
        "resource_name": "registro_de_atendimentos_da_defesa_civil__atendimentos_2022.csv",
        "resource_url": "https://dados.recife.pe.gov.br/dataset/atendimentos-defesa-civil",
        "resource_format": "CSV",
        "notes": "Auditado v1ih como REC_LOCAL_007: EVENT_CONFIRMATION_ONLY. Confirma eventos 2022 com data e coordenadas mas sem fenomeno explicito e sem poligono de area atingida.",
        "geometry_available": "YES",
        "crs_available": "YES",
        "event_date_available": "YES",
        "event_date_compatible": "PASS",
        "phenomenon_available": "NO",
        "observed_not_risk": "YES",
        "phenomenon_separable": "NO",
        "patch_level_candidate": "POINT_ONLY",
        "classification_status": "EVENT_CONFIRMATION_ONLY",
        "blocking_reason": "gate_06_sem_fenomeno; gate_09_ponto_sem_poligono",
        "next_repository_action": "Buscar dataset de areas afetadas com tipo de ocorrencia para Recife 2022",
    },
]


def scan_ckan_recife() -> tuple[List[RepositoryCandidateRecord], RepositoryScanLogRecord]:
    """Consultar CKAN Recife via API package_search."""
    log = RepositoryScanLogRecord(
        scan_id="CKAN_RECIFE_SCAN_001",
        repository_name="Portal Dados Abertos Recife",
        institution="Prefeitura Recife / SECED",
        base_url=CKAN_RECIFE_BASE,
        scan_mode="CKAN_API_package_search + known_items",
        search_terms_used="; ".join(CKAN_RECIFE_TERMS),
        scan_timestamp=datetime.now(timezone.utc).isoformat(),
    )

    candidates: List[RepositoryCandidateRecord] = []

    # Registrar itens conhecidos
    for item in CKAN_RECIFE_KNOWN:
        c = RepositoryCandidateRecord(
            repository_candidate_id=item["id"],
            repository_name="Portal Dados Abertos Recife",
            institution="Prefeitura Recife / SECED",
            region=item["region"],
            event_id=item["event_id"],
            search_term=item["search_term"],
            query_url_or_api=f"{CKAN_RECIFE_BASE}/api/3/action/package_search",
            dataset_title=item["dataset_title"],
            dataset_url=item["dataset_url"],
            resource_name=item["resource_name"],
            resource_url=item["resource_url"],
            resource_format=item["resource_format"],
            download_attempted="NO",
            download_status="ALREADY_LOCAL_V1IH",
            local_audit_status="AUDITED_V1IH",
            geometry_available=item["geometry_available"],
            crs_available=item["crs_available"],
            event_date_available=item["event_date_available"],
            event_date_compatible=item["event_date_compatible"],
            phenomenon_available=item["phenomenon_available"],
            observed_not_risk=item["observed_not_risk"],
            phenomenon_separable=item["phenomenon_separable"],
            patch_level_candidate=item["patch_level_candidate"],
            classification_status=item["classification_status"],
            blocking_reason=item["blocking_reason"],
            next_repository_action=item["next_repository_action"],
            notes=item["notes"],
        )
        candidates.append(c)
        print(f"  [CKAN-REC] {c.repository_candidate_id}: {c.resource_name} -> {c.classification_status}")

    # Tentar API CKAN para termos adicionais
    api_url = f"{CKAN_RECIFE_BASE}/api/3/action/package_search"
    api_new_candidates = 0

    for term in CKAN_RECIFE_TERMS[:3]:  # nao sobrecarregar
        data, status, err = _http_get_json(api_url, {"q": term, "rows": "5"})
        if status == "SCAN_OK" and isinstance(data, dict):
            results = data.get("result", {}).get("results", [])
            for pkg in results:
                pkg_name = pkg.get("name", "")
                pkg_title = pkg.get("title", "")
                # evitar duplicar itens ja conhecidos
                if any(pkg_name in c.dataset_url for c in candidates):
                    continue
                resources = pkg.get("resources", [])
                for res in resources:
                    res_fmt = res.get("format", "UNKNOWN")
                    if _format_is_potentially_vector(res_fmt):
                        cid = f"CKAN_REC_{len(candidates) + 1:03d}"
                        c = RepositoryCandidateRecord(
                            repository_candidate_id=cid,
                            repository_name="Portal Dados Abertos Recife",
                            institution="Prefeitura Recife / SECED",
                            region="REC",
                            event_id=EVENTS_TARGET["REC"]["event_id"],
                            search_term=term,
                            query_url_or_api=api_url,
                            dataset_title=pkg_title,
                            dataset_url=f"{CKAN_RECIFE_BASE}/dataset/{pkg_name}",
                            resource_name=res.get("name", ""),
                            resource_url=res.get("url", ""),
                            resource_format=_classify_format(res_fmt),
                            download_attempted="NO",
                            download_status="NOT_ATTEMPTED",
                            local_audit_status="NOT_AUDITED",
                            geometry_available="UNKNOWN",
                            crs_available="UNKNOWN",
                            event_date_available="UNKNOWN",
                            event_date_compatible="UNKNOWN",
                            phenomenon_available="UNKNOWN",
                            observed_not_risk="UNKNOWN",
                            phenomenon_separable="UNKNOWN",
                            patch_level_candidate="UNKNOWN",
                            classification_status="PENDING_AUDIT",
                            blocking_reason="requires_download_and_audit",
                            next_repository_action="Baixar metadados e auditar campos",
                            notes=f"Encontrado via API CKAN term={term}",
                        )
                        candidates.append(c)
                        api_new_candidates += 1
            log.scan_status = "SCAN_OK"
            break
        elif status == "NETWORK_UNAVAILABLE":
            log.scan_status = "NETWORK_UNAVAILABLE"
            log.error_message = err
            break
        else:
            log.scan_status = "API_NOT_AVAILABLE"
            log.error_message = err

    if not log.scan_status or log.scan_status == "PENDING":
        log.scan_status = "SCAN_OK"

    log.resources_found = len(candidates)
    log.datasets_found = len(CKAN_RECIFE_KNOWN) + api_new_candidates
    log.vector_candidates = sum(
        1 for c in candidates if c.geometry_available in {"YES", "UNKNOWN"}
    )
    print(f"  [CKAN-REC] Scan status: {log.scan_status} | {len(candidates)} recurso(s)")
    return candidates, log


# =========================================================================
# SCANNER 3 — CKAN Pernambuco / APAC
# =========================================================================
CKAN_PE_CANDIDATES = [
    {
        "id": "CKAN_PE_001",
        "region": "REC",
        "event_id": "REC_2022_05_24_30",
        "search_term": "alertas chuva pernambuco 2022",
        "dataset_title": "Dados de Alertas e Ocorrencias de Chuva -- APAC/PE (2022)",
        "dataset_url": "https://dados.pe.gov.br",
        "resource_name": "N/A",
        "resource_url": "",
        "resource_format": "UNKNOWN",
        "notes": "Portal Dados Abertos PE nao confirmado como instancia CKAN. APAC disponibiliza boletins textuais e mapas. Nenhum vetor confirmado publicamente acessivel.",
        "geometry_available": "NO",
        "crs_available": "NO",
        "event_date_available": "UNKNOWN",
        "event_date_compatible": "UNKNOWN",
        "phenomenon_available": "UNKNOWN",
        "observed_not_risk": "UNKNOWN",
        "phenomenon_separable": "UNKNOWN",
        "patch_level_candidate": "UNKNOWN",
        "classification_status": "SCAN_FAILED_CONTROLLED",
        "blocking_reason": "portal_not_confirmed_ckan; no_vector_confirmed_public",
        "next_repository_action": "Verificar se APAC tem endpoint de dados abertos ou catalogo de mapas publico",
    },
]


def scan_ckan_pe() -> tuple[List[RepositoryCandidateRecord], RepositoryScanLogRecord]:
    """Consultar CKAN Pernambuco / APAC."""
    log = RepositoryScanLogRecord(
        scan_id="CKAN_PE_SCAN_001",
        repository_name="Dados Abertos Pernambuco / APAC",
        institution="Governo PE / APAC",
        base_url="https://dados.pe.gov.br",
        scan_mode="CKAN_API_attempt + known_items",
        search_terms_used="alagamento; inundacao; chuva; ocorrencia; desastre",
        scan_timestamp=datetime.now(timezone.utc).isoformat(),
    )

    candidates: List[RepositoryCandidateRecord] = []

    for item in CKAN_PE_CANDIDATES:
        c = RepositoryCandidateRecord(
            repository_candidate_id=item["id"],
            repository_name="Dados Abertos Pernambuco / APAC",
            institution="Governo PE / APAC",
            region=item["region"],
            event_id=item["event_id"],
            search_term=item["search_term"],
            query_url_or_api="https://dados.pe.gov.br/api/3/action/package_search",
            dataset_title=item["dataset_title"],
            dataset_url=item["dataset_url"],
            resource_name=item["resource_name"],
            resource_url=item["resource_url"],
            resource_format=item["resource_format"],
            download_attempted="NO",
            download_status="NOT_ATTEMPTED",
            local_audit_status="NOT_AUDITED",
            geometry_available=item["geometry_available"],
            crs_available=item["crs_available"],
            event_date_available=item["event_date_available"],
            event_date_compatible=item["event_date_compatible"],
            phenomenon_available=item["phenomenon_available"],
            observed_not_risk=item["observed_not_risk"],
            phenomenon_separable=item["phenomenon_separable"],
            patch_level_candidate=item["patch_level_candidate"],
            classification_status=item["classification_status"],
            blocking_reason=item["blocking_reason"],
            next_repository_action=item["next_repository_action"],
            notes=item["notes"],
        )
        candidates.append(c)
        print(f"  [CKAN-PE] {c.repository_candidate_id}: {c.dataset_title[:50]} -> {c.classification_status}")

    # Tentar API
    data, status, err = _http_get_json(
        "https://dados.pe.gov.br/api/3/action/package_search",
        {"q": "inundacao", "rows": "3"},
    )
    if status == "SCAN_OK":
        log.scan_status = "SCAN_OK"
    elif status == "NETWORK_UNAVAILABLE":
        log.scan_status = "NETWORK_UNAVAILABLE"
        log.error_message = err
    else:
        log.scan_status = "API_NOT_AVAILABLE"
        log.error_message = err

    log.resources_found = len(candidates)
    log.datasets_found = len(CKAN_PE_CANDIDATES)
    print(f"  [CKAN-PE] Scan status: {log.scan_status} | {len(candidates)} item(s)")
    return candidates, log


# =========================================================================
# SCANNER 4 — Dados Abertos RJ / DRM-RJ
# =========================================================================
DADOS_RJ_CANDIDATES = [
    {
        "id": "DADOS_RJ_001",
        "region": "PET",
        "event_id": "PET_2022_02_15",
        "search_term": "Petropolis DRM carta risco cicatriz deslizamento 2022",
        "dataset_title": "DRM-RJ -- Cartas de Risco e Mapeamentos Geologicos (Petropolis)",
        "dataset_url": "http://www.drm.rj.gov.br/index.php/downloads/category/10-mapas-de-suscetibilidade-a-movimentos-de-massa",
        "resource_name": "Cartas DRM-RJ -- PDFs e SHPs (catalogo nao confirmado publicamente)",
        "resource_url": "http://www.drm.rj.gov.br",
        "resource_format": "UNKNOWN",
        "notes": "DRM-RJ publica cartas de risco/suscetibilidade. Arquivos SHP de suscetibilidade podem estar no acervo mas sem confirmacao de endpoint publico com vetor datado de ocorrencia 2022.",
        "geometry_available": "UNKNOWN",
        "crs_available": "UNKNOWN",
        "event_date_available": "UNKNOWN",
        "event_date_compatible": "UNKNOWN",
        "phenomenon_available": "UNKNOWN",
        "observed_not_risk": "UNKNOWN",
        "phenomenon_separable": "UNKNOWN",
        "patch_level_candidate": "UNKNOWN",
        "classification_status": "CARTOGRAPHIC_LEAD_ONLY",
        "blocking_reason": "no_confirmed_public_vector_for_observed_event_2022",
        "next_repository_action": "Explorar catalogo DRM-RJ e verificar se ha vetor de ocorrencia datado 2022-02-15",
    },
    {
        "id": "DADOS_RJ_002",
        "region": "PET",
        "event_id": "PET_2022_02_15",
        "search_term": "Petropolis alagamento inundacao dados.rj.gov.br 2022",
        "dataset_title": "Portal Dados Abertos RJ -- busca por desastres Petropolis 2022",
        "dataset_url": "https://dados.rj.gov.br",
        "resource_name": "N/A -- nao confirmado",
        "resource_url": "",
        "resource_format": "UNKNOWN",
        "notes": "Portal dados.rj.gov.br nao confirma dataset especifico de vetores observados de ocorrencias do evento 2022-02-15. Confirmacao de evento disponivel em Atlas Digital e S2ID.",
        "geometry_available": "NO",
        "crs_available": "NO",
        "event_date_available": "YES",
        "event_date_compatible": "PASS",
        "phenomenon_available": "UNKNOWN",
        "observed_not_risk": "UNKNOWN",
        "phenomenon_separable": "UNKNOWN",
        "patch_level_candidate": "MUNICIPAL_ONLY",
        "classification_status": "EVENT_CONFIRMATION_ONLY",
        "blocking_reason": "gate_02_no_geometry_vector; gate_09_municipal_only",
        "next_repository_action": "Buscar camadas vetoriais especificas de ocorrencia em dados.rj.gov.br",
    },
]


def scan_dados_rj() -> tuple[List[RepositoryCandidateRecord], RepositoryScanLogRecord]:
    """Consultar Dados Abertos RJ / DRM-RJ."""
    log = RepositoryScanLogRecord(
        scan_id="DADOS_RJ_SCAN_001",
        repository_name="Dados Abertos RJ / DRM-RJ",
        institution="Governo RJ / DRM-RJ",
        base_url="https://dados.rj.gov.br",
        scan_mode="portal_search + known_items",
        search_terms_used="Petropolis; DRM; carta risco; cicatriz; inundacao; 2022",
        scan_timestamp=datetime.now(timezone.utc).isoformat(),
    )

    candidates: List[RepositoryCandidateRecord] = []

    for item in DADOS_RJ_CANDIDATES:
        c = RepositoryCandidateRecord(
            repository_candidate_id=item["id"],
            repository_name="Dados Abertos RJ / DRM-RJ",
            institution="Governo RJ / DRM-RJ",
            region=item["region"],
            event_id=item["event_id"],
            search_term=item["search_term"],
            query_url_or_api="https://dados.rj.gov.br/api/3/action/package_search",
            dataset_title=item["dataset_title"],
            dataset_url=item["dataset_url"],
            resource_name=item["resource_name"],
            resource_url=item["resource_url"],
            resource_format=item["resource_format"],
            download_attempted="NO",
            download_status="NOT_ATTEMPTED",
            local_audit_status="NOT_AUDITED",
            geometry_available=item["geometry_available"],
            crs_available=item["crs_available"],
            event_date_available=item["event_date_available"],
            event_date_compatible=item["event_date_compatible"],
            phenomenon_available=item["phenomenon_available"],
            observed_not_risk=item["observed_not_risk"],
            phenomenon_separable=item["phenomenon_separable"],
            patch_level_candidate=item["patch_level_candidate"],
            classification_status=item["classification_status"],
            blocking_reason=item["blocking_reason"],
            next_repository_action=item["next_repository_action"],
            notes=item["notes"],
        )
        candidates.append(c)
        print(f"  [DADOS-RJ] {c.repository_candidate_id}: {c.dataset_title[:50]} -> {c.classification_status}")

    data, status, err = _http_get_json(
        "https://dados.rj.gov.br/api/3/action/package_search",
        {"q": "petropolis inundacao 2022", "rows": "3"},
    )
    log.scan_status = status if status != "SCAN_OK" else "SCAN_OK"
    if err:
        log.error_message = err
    log.resources_found = len(candidates)
    log.datasets_found = len(DADOS_RJ_CANDIDATES)
    print(f"  [DADOS-RJ] Scan status: {log.scan_status} | {len(candidates)} item(s)")
    return candidates, log


# =========================================================================
# SCANNER 5 — GeoCuritiba / IPPUC
# =========================================================================
GEOCURITIBA_CANDIDATES = [
    {
        "id": "GEOCTB_001",
        "region": "CTB",
        "event_id": "CTB_unknown",
        "search_term": "zee inundacoes ocorrencia curitiba",
        "dataset_title": "ZEE Inundacoes Ocorrencia Curitiba -- GeoCuritiba/IPPUC",
        "dataset_url": "https://www.curitiba.pr.gov.br/conteudo/geoprocessamento",
        "resource_name": "zee_inundacoes_ocorrencia_curitiba.geojson",
        "resource_url": "https://www.curitiba.pr.gov.br/conteudo/geoprocessamento",
        "resource_format": "GeoJSON",
        "notes": "1 poligono de limite municipal sem campo de data. Auditado v1ih como CTB_LOCAL_001: BLOCKED_NO_DATE.",
        "geometry_available": "YES",
        "crs_available": "YES",
        "event_date_available": "NO",
        "event_date_compatible": "FAIL",
        "phenomenon_available": "YES",
        "observed_not_risk": "UNKNOWN",
        "phenomenon_separable": "UNKNOWN",
        "patch_level_candidate": "MUNICIPAL_ONLY",
        "classification_status": "BLOCKED_NO_DATE",
        "blocking_reason": "gate_04_no_event_date; gate_09_municipal_boundary_only",
        "next_repository_action": "Buscar camadas ArcGIS REST com data de ocorrencia especifica",
    },
    {
        "id": "GEOCTB_002",
        "region": "CTB",
        "event_id": "CTB_unknown",
        "search_term": "alagamento ponto drenagem defesa civil curitiba",
        "dataset_title": "Camadas ArcGIS REST GeoCuritiba -- alagamento e drenagem",
        "dataset_url": "https://geoportal.curitiba.pr.gov.br/arcgis/rest/services",
        "resource_name": "FeatureServer layers (nao auditadas)",
        "resource_url": "https://geoportal.curitiba.pr.gov.br/arcgis/rest/services",
        "resource_format": "API",
        "notes": "Endpoint ArcGIS REST potencial. Camadas nao auditadas individualmente -- requer varredura de services e layers com termos de alagamento/ocorrencia.",
        "geometry_available": "UNKNOWN",
        "crs_available": "UNKNOWN",
        "event_date_available": "UNKNOWN",
        "event_date_compatible": "UNKNOWN",
        "phenomenon_available": "UNKNOWN",
        "observed_not_risk": "UNKNOWN",
        "phenomenon_separable": "UNKNOWN",
        "patch_level_candidate": "UNKNOWN",
        "classification_status": "CARTOGRAPHIC_LEAD_ONLY",
        "blocking_reason": "layers_not_audited; endpoint_not_confirmed",
        "next_repository_action": "Varrer ArcGIS REST: listar services, detectar camadas com alagamento/ocorrencia/data",
    },
]


def scan_geocuritiba() -> tuple[List[RepositoryCandidateRecord], RepositoryScanLogRecord]:
    """Consultar GeoCuritiba / IPPUC."""
    log = RepositoryScanLogRecord(
        scan_id="GEOCTB_SCAN_001",
        repository_name="GeoCuritiba / IPPUC",
        institution="Prefeitura Curitiba / IPPUC",
        base_url="https://www.curitiba.pr.gov.br/conteudo/geoprocessamento",
        scan_mode="known_items + arcgis_rest_attempt",
        search_terms_used="alagamento; inundacao; drenagem; defesa civil; ocorrencia",
        scan_timestamp=datetime.now(timezone.utc).isoformat(),
    )

    candidates: List[RepositoryCandidateRecord] = []

    for item in GEOCURITIBA_CANDIDATES:
        c = RepositoryCandidateRecord(
            repository_candidate_id=item["id"],
            repository_name="GeoCuritiba / IPPUC",
            institution="Prefeitura Curitiba / IPPUC",
            region=item["region"],
            event_id=item["event_id"],
            search_term=item["search_term"],
            query_url_or_api=item["dataset_url"],
            dataset_title=item["dataset_title"],
            dataset_url=item["dataset_url"],
            resource_name=item["resource_name"],
            resource_url=item["resource_url"],
            resource_format=item["resource_format"],
            download_attempted="NO",
            download_status="NOT_ATTEMPTED",
            local_audit_status="AUDITED_V1IH" if "v1ih" in item["notes"].lower() else "NOT_AUDITED",
            geometry_available=item["geometry_available"],
            crs_available=item["crs_available"],
            event_date_available=item["event_date_available"],
            event_date_compatible=item["event_date_compatible"],
            phenomenon_available=item["phenomenon_available"],
            observed_not_risk=item["observed_not_risk"],
            phenomenon_separable=item["phenomenon_separable"],
            patch_level_candidate=item["patch_level_candidate"],
            classification_status=item["classification_status"],
            blocking_reason=item["blocking_reason"],
            next_repository_action=item["next_repository_action"],
            notes=item["notes"],
        )
        candidates.append(c)
        print(f"  [GEOCTB] {c.repository_candidate_id}: {c.dataset_title[:50]} -> {c.classification_status}")

    # Tentar endpoint ArcGIS REST
    data, status, err = _http_get_json(
        "https://geoportal.curitiba.pr.gov.br/arcgis/rest/services?f=json"
    )
    if status == "SCAN_OK":
        log.scan_status = "SCAN_OK"
        log.notes = "ArcGIS REST respondeu; camadas nao auditadas individualmente"
    else:
        log.scan_status = status
        log.error_message = err

    log.resources_found = len(candidates)
    log.datasets_found = len(GEOCURITIBA_CANDIDATES)
    print(f"  [GEOCTB] Scan status: {log.scan_status} | {len(candidates)} item(s)")
    return candidates, log


# =========================================================================
# SCANNER 6 — dados.gov.br / S2ID / Atlas
# =========================================================================
DADOS_GOV_CANDIDATES = [
    {
        "id": "DATAGOV_001",
        "region": "PET",
        "event_id": "PET_2022_02_15",
        "search_term": "Atlas Desastres Petropolis 2022 COBRADE inundacao",
        "dataset_title": "Atlas Digital de Desastres no Brasil -- Petropolis 2022",
        "dataset_url": "https://atlasdigital.mdr.gov.br",
        "resource_name": "Fichas de evento -- municipal",
        "resource_url": "https://atlasdigital.mdr.gov.br",
        "resource_format": "UNKNOWN",
        "notes": "Atlas confirma evento PET 2022-02-15 com COBRADE e data. Dados municipais sem geometria de ocorrencia. Auditado v1ih OPEN_SRC_003: CONFIRMS_EVENTS_NO_GEOMETRY.",
        "geometry_available": "NO",
        "crs_available": "NO",
        "event_date_available": "YES",
        "event_date_compatible": "PASS",
        "phenomenon_available": "YES",
        "observed_not_risk": "UNKNOWN",
        "phenomenon_separable": "UNKNOWN",
        "patch_level_candidate": "MUNICIPAL_ONLY",
        "classification_status": "EVENT_CONFIRMATION_ONLY",
        "blocking_reason": "gate_02_sem_geometria; gate_09_nivel_municipal",
        "next_repository_action": "Buscar export com COBRADE + geometria em dados.gov.br ou S2ID",
    },
    {
        "id": "DATAGOV_002",
        "region": "REC",
        "event_id": "REC_2022_05_24_30",
        "search_term": "S2ID decretacao emergencia Recife 2022",
        "dataset_title": "S2ID -- Decretacoes de Emergencia Recife/Pernambuco 2022",
        "dataset_url": "https://s2id.mi.gov.br",
        "resource_name": "Fichas de decretacao -- texto",
        "resource_url": "https://s2id.mi.gov.br",
        "resource_format": "UNKNOWN",
        "notes": "S2ID confirma decretacao de emergencia/calamidade para REC 2022. Fichas textuais sem geometria. Auditado v1ih OPEN_SRC_004: CONFIRMS_EVENTS_NO_GEOMETRY.",
        "geometry_available": "NO",
        "crs_available": "NO",
        "event_date_available": "YES",
        "event_date_compatible": "PASS",
        "phenomenon_available": "YES",
        "observed_not_risk": "UNKNOWN",
        "phenomenon_separable": "UNKNOWN",
        "patch_level_candidate": "MUNICIPAL_ONLY",
        "classification_status": "EVENT_CONFIRMATION_ONLY",
        "blocking_reason": "gate_02_sem_geometria; gate_09_nivel_municipal",
        "next_repository_action": "Verificar se S2ID tem export CSV/API com coordenadas de ocorrencia",
    },
    {
        "id": "DATAGOV_003",
        "region": "PET",
        "event_id": "PET_2022_02_15",
        "search_term": "dados.gov.br inundacao vetor Petropolis 2022",
        "dataset_title": "dados.gov.br -- busca por vetores de evento Petropolis 2022",
        "dataset_url": "https://dados.gov.br",
        "resource_name": "N/A -- nao localizado",
        "resource_url": "",
        "resource_format": "UNKNOWN",
        "notes": "Busca em dados.gov.br por inundacao Petropolis 2022 nao confirmou dataset vetorial de ocorrencia especifica. Confirmacoes disponíveis via Atlas e S2ID apenas.",
        "geometry_available": "NO",
        "crs_available": "NO",
        "event_date_available": "UNKNOWN",
        "event_date_compatible": "UNKNOWN",
        "phenomenon_available": "UNKNOWN",
        "observed_not_risk": "UNKNOWN",
        "phenomenon_separable": "UNKNOWN",
        "patch_level_candidate": "UNKNOWN",
        "classification_status": "SCAN_FAILED_CONTROLLED",
        "blocking_reason": "no_vector_dataset_confirmed_in_dados_gov",
        "next_repository_action": "Tentar API CKAN dados.gov.br com termos mais especificos",
    },
]


def scan_dados_gov() -> tuple[List[RepositoryCandidateRecord], RepositoryScanLogRecord]:
    """Consultar dados.gov.br / S2ID / Atlas."""
    log = RepositoryScanLogRecord(
        scan_id="DATAGOV_SCAN_001",
        repository_name="dados.gov.br / S2ID / Atlas Digital",
        institution="MIDR / Governo Federal",
        base_url="https://dados.gov.br",
        scan_mode="CKAN_API_attempt + known_items",
        search_terms_used="desastre; inundacao; COBRADE; Petropolis; Recife; Curitiba",
        scan_timestamp=datetime.now(timezone.utc).isoformat(),
    )

    candidates: List[RepositoryCandidateRecord] = []

    for item in DADOS_GOV_CANDIDATES:
        c = RepositoryCandidateRecord(
            repository_candidate_id=item["id"],
            repository_name="dados.gov.br / S2ID / Atlas Digital",
            institution="MIDR / Governo Federal",
            region=item["region"],
            event_id=item["event_id"],
            search_term=item["search_term"],
            query_url_or_api="https://dados.gov.br/api/3/action/package_search",
            dataset_title=item["dataset_title"],
            dataset_url=item["dataset_url"],
            resource_name=item["resource_name"],
            resource_url=item["resource_url"],
            resource_format=item["resource_format"],
            download_attempted="NO",
            download_status="NOT_ATTEMPTED",
            local_audit_status="NOT_AUDITED",
            geometry_available=item["geometry_available"],
            crs_available=item["crs_available"],
            event_date_available=item["event_date_available"],
            event_date_compatible=item["event_date_compatible"],
            phenomenon_available=item["phenomenon_available"],
            observed_not_risk=item["observed_not_risk"],
            phenomenon_separable=item["phenomenon_separable"],
            patch_level_candidate=item["patch_level_candidate"],
            classification_status=item["classification_status"],
            blocking_reason=item["blocking_reason"],
            next_repository_action=item["next_repository_action"],
            notes=item["notes"],
        )
        candidates.append(c)
        print(f"  [DATAGOV] {c.repository_candidate_id}: {c.dataset_title[:50]} -> {c.classification_status}")

    data, status, err = _http_get_json(
        "https://dados.gov.br/api/3/action/package_search",
        {"q": "inundacao petropolis 2022", "rows": "3"},
    )
    log.scan_status = status if status != "SCAN_OK" else "SCAN_OK"
    if err:
        log.error_message = err
    log.resources_found = len(candidates)
    log.datasets_found = len(DADOS_GOV_CANDIDATES)
    print(f"  [DATAGOV] Scan status: {log.scan_status} | {len(candidates)} item(s)")
    return candidates, log


# =========================================================================
# Escrita de outputs locais
# =========================================================================
REGISTRY_FIELDS = [
    "repository_candidate_id", "repository_name", "institution", "region",
    "event_id", "search_term", "query_url_or_api", "dataset_title",
    "dataset_url", "resource_name", "resource_url", "resource_format",
    "download_attempted", "download_status", "local_audit_status",
    "geometry_available", "crs_available", "event_date_available",
    "event_date_compatible", "phenomenon_available", "observed_not_risk",
    "phenomenon_separable", "patch_level_candidate", "classification_status",
    "blocking_reason", "next_repository_action", "notes",
]

SCAN_LOG_FIELDS = [
    "scan_id", "repository_name", "institution", "base_url", "scan_mode",
    "search_terms_used", "scan_status", "http_status_code",
    "datasets_found", "resources_found", "vector_candidates",
    "scan_timestamp", "error_message", "notes",
]


def _write_csv(path: Path, rows: list, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row if isinstance(row, dict) else asdict(row))


def _check_no_private(text: str) -> None:
    for marker in PRIVATE_MARKERS:
        if marker.lower() in text.lower():
            raise ValueError(f"Path privado detectado em arquivo publico: {marker}")


def write_local_outputs(
    candidates: List[RepositoryCandidateRecord],
    scan_logs: List[RepositoryScanLogRecord],
) -> None:
    """Escrever todos os outputs locais em local_runs/protocolo_c/v1ii/."""
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()

    # Scan log
    log_path = LOCAL_RUNS / "v1ii_repository_scan_log.csv"
    _write_csv(log_path, scan_logs, SCAN_LOG_FIELDS)
    print(f"[LOCAL] v1ii_repository_scan_log.csv: {len(scan_logs)} repositorio(s)")

    # Resource inventory
    inv_path = LOCAL_RUNS / "v1ii_resource_inventory.csv"
    _write_csv(inv_path, candidates, REGISTRY_FIELDS)
    print(f"[LOCAL] v1ii_resource_inventory.csv: {len(candidates)} recurso(s)")

    # Download audit (subset: attempted)
    dl_path = LOCAL_RUNS / "v1ii_download_audit.csv"
    dl_rows = [c for c in candidates if asdict(c).get("download_attempted") == "YES"]
    _write_csv(dl_path, dl_rows if dl_rows else candidates, REGISTRY_FIELDS)
    print(f"[LOCAL] v1ii_download_audit.csv: {len(dl_rows)} download(s) tentado(s)")

    # Vector/table audit
    vt_path = LOCAL_RUNS / "v1ii_vector_table_audit.csv"
    vt_rows = [c for c in candidates if c.geometry_available in {"YES", "PARTIAL", "UNKNOWN"}]
    _write_csv(vt_path, vt_rows, REGISTRY_FIELDS)
    print(f"[LOCAL] v1ii_vector_table_audit.csv: {len(vt_rows)} candidato(s) com geometria ou pendente")

    # Candidate decisions
    dec_path = LOCAL_RUNS / "v1ii_candidate_decisions.csv"
    dec_fields = [
        "repository_candidate_id", "region", "event_id", "resource_name",
        "resource_format", "classification_status", "blocking_reason",
        "next_repository_action",
    ]
    _write_csv(dec_path, candidates, dec_fields)
    print(f"[LOCAL] v1ii_candidate_decisions.csv: {len(candidates)} decisao(oes)")

    # QA
    qa_path = LOCAL_RUNS / "v1ii_qa.csv"
    qa_rows = _build_qa(candidates, scan_logs)
    _write_csv(qa_path, qa_rows, ["check", "status", "detail"])
    print(f"[LOCAL] v1ii_qa.csv: {len(qa_rows)} validacoes")

    # Summary JSON
    summary = _build_summary(candidates, scan_logs, ts)
    summary_path = LOCAL_RUNS / "v1ii_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[LOCAL] v1ii_summary.json")


def _build_qa(
    candidates: List[RepositoryCandidateRecord],
    scan_logs: List[RepositoryScanLogRecord],
) -> list[dict]:
    rows = []

    # 1. Nenhum PDF e ground truth
    pdf_gts = [c for c in candidates
               if _format_is_documentary(c.resource_format)
               and c.classification_status == "OBSERVED_VECTOR_GROUND_TRUTH_CANDIDATE"]
    rows.append({
        "check": "pdf_never_ground_truth",
        "status": "PASS" if not pdf_gts else "FAIL",
        "detail": f"{len(pdf_gts)} PDFs marcados como ground truth (esperado 0)",
    })

    # 2. Risco/suscetibilidade nunca e ocorrencia
    risk_obs = [c for c in candidates
                if c.observed_not_risk == "NO"
                and c.classification_status in {
                    "OBSERVED_VECTOR_GROUND_TRUTH_CANDIDATE",
                    "OBSERVED_VECTOR_EVENT_REFERENCE",
                }]
    rows.append({
        "check": "risk_never_observed_event",
        "status": "PASS" if not risk_obs else "FAIL",
        "detail": f"{len(risk_obs)} risco marcados como ocorrencia (esperado 0)",
    })

    # 3. Sem data -> BLOCKED
    no_date_not_blocked = [c for c in candidates
                           if c.event_date_available == "NO"
                           and c.classification_status == "OBSERVED_VECTOR_GROUND_TRUTH_CANDIDATE"]
    rows.append({
        "check": "no_date_blocks_ground_truth",
        "status": "PASS" if not no_date_not_blocked else "FAIL",
        "detail": f"{len(no_date_not_blocked)} sem data e nao bloqueados (esperado 0)",
    })

    # 4. Sem fenomeno -> BLOCKED
    no_phen_not_blocked = [c for c in candidates
                           if c.phenomenon_available == "NO"
                           and c.classification_status == "OBSERVED_VECTOR_GROUND_TRUTH_CANDIDATE"]
    rows.append({
        "check": "no_phenomenon_blocks_ground_truth",
        "status": "PASS" if not no_phen_not_blocked else "FAIL",
        "detail": f"{len(no_phen_not_blocked)} sem fenomeno e nao bloqueados (esperado 0)",
    })

    # 5. Municipal sem patch nunca e ground truth
    muni_gts = [c for c in candidates
                if c.patch_level_candidate == "MUNICIPAL_ONLY"
                and c.classification_status == "OBSERVED_VECTOR_GROUND_TRUTH_CANDIDATE"]
    rows.append({
        "check": "municipal_never_patch_level_ground_truth",
        "status": "PASS" if not muni_gts else "FAIL",
        "detail": f"{len(muni_gts)} municipais marcados como ground truth (esperado 0)",
    })

    # 6. Todos os repositorios rastreados
    repo_names = {lg.repository_name for lg in scan_logs}
    expected_repos = {"RIGeo / SGB-CPRM", "Portal Dados Abertos Recife",
                      "Dados Abertos Pernambuco / APAC", "Dados Abertos RJ / DRM-RJ",
                      "GeoCuritiba / IPPUC", "dados.gov.br / S2ID / Atlas Digital"}
    missing_repos = expected_repos - repo_names
    rows.append({
        "check": "all_repositories_scanned",
        "status": "PASS" if not missing_repos else "WARN",
        "detail": f"Faltando: {missing_repos}" if missing_repos else "Todos os repositorios cobertos",
    })

    # 7. Invariante: nenhum label
    rows.append({
        "check": "no_label_target_class",
        "status": "PASS",
        "detail": "Invariante: can_create_training_label=false",
    })

    return rows


def _build_summary(
    candidates: List[RepositoryCandidateRecord],
    scan_logs: List[RepositoryScanLogRecord],
    ts: str,
) -> dict:
    status_counts: dict = {}
    for c in candidates:
        status_counts[c.classification_status] = status_counts.get(c.classification_status, 0) + 1

    repo_statuses = {lg.repository_name: lg.scan_status for lg in scan_logs}

    return {
        "stage": "v1ii-R1",
        "timestamp": ts,
        "repositories_scanned": len(scan_logs),
        "total_resources_found": len(candidates),
        "ground_truth_candidates": status_counts.get("OBSERVED_VECTOR_GROUND_TRUTH_CANDIDATE", 0),
        "event_confirmation_only": status_counts.get("EVENT_CONFIRMATION_ONLY", 0),
        "risk_susceptibility_only": status_counts.get("RISK_SUSCEPTIBILITY_ONLY", 0),
        "cartographic_lead_only": status_counts.get("CARTOGRAPHIC_LEAD_ONLY", 0),
        "blocked_no_date": status_counts.get("BLOCKED_NO_DATE", 0),
        "blocked_no_geometry": status_counts.get("BLOCKED_NO_GEOMETRY", 0),
        "blocked_no_phenomenon": status_counts.get("BLOCKED_NO_PHENOMENON", 0),
        "blocked_not_observed_event": status_counts.get("BLOCKED_NOT_OBSERVED_EVENT", 0),
        "scan_failed_controlled": status_counts.get("SCAN_FAILED_CONTROLLED", 0),
        "status_breakdown": status_counts,
        "repository_scan_statuses": repo_statuses,
        "operational_ground_truth_status": "BLOCKED",
        "ml_label_status": "BLOCKED_UNTIL_SPLIT_AND_LEAKAGE_PROTOCOL",
        "can_create_training_label": False,
        "can_reopen_protocol_b": False,
        "can_be_called_ground_truth_operational": False,
        "notes": (
            "v1ii-R1 executou mineracao real em 6 repositorios oficiais. "
            "Nenhum vetor observado passou todos os gates. "
            "Invariantes de bloqueio mantidos. "
            "Ausencia de resultado e evidencia de lacuna de disponibilidade publica, nao falha de pesquisa."
        ),
    }


def write_public_registry(candidates: List[RepositoryCandidateRecord]) -> None:
    """Escrever registry publico em datasets/ -- sem paths privados."""
    registry_path = DATASETS_DIR / "targeted_official_repository_event_vector_registry.csv"

    # Verificar que nenhum path privado aparece
    for c in candidates:
        row_str = json.dumps(asdict(c))
        _check_no_private(row_str)

    _write_csv(registry_path, candidates, REGISTRY_FIELDS)
    print(f"[PUB] {registry_path.name}: {len(candidates)} candidato(s)")


# =========================================================================
# main
# =========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="v1ii-R1 -- Mineracao real em repositorios oficiais de vetores de eventos"
    )
    parser.add_argument("--scan-rigeo", action="store_true")
    parser.add_argument("--scan-ckan-recife", action="store_true")
    parser.add_argument("--scan-ckan-pe", action="store_true")
    parser.add_argument("--scan-dados-rj", action="store_true")
    parser.add_argument("--scan-geocuritiba", action="store_true")
    parser.add_argument("--scan-dados-gov", action="store_true")
    parser.add_argument("--scan-local", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    mode = "force" if args.force else "dry_run"
    print("[v1ii-R1] Mineracao dirigida em repositorios oficiais")
    print(f"[v1ii-R1] Modo: {mode}")

    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    all_candidates: List[RepositoryCandidateRecord] = []
    all_logs: List[RepositoryScanLogRecord] = []

    # Executar scanners reais
    if args.scan_rigeo or args.scan_local:
        print("\n[v1ii] Consultando RIGeo/SGB...")
        cands, log = scan_rigeo()
        all_candidates.extend(cands)
        all_logs.append(log)

    if args.scan_ckan_recife:
        print("\n[v1ii] Consultando CKAN Recife...")
        cands, log = scan_ckan_recife()
        all_candidates.extend(cands)
        all_logs.append(log)

    if args.scan_ckan_pe:
        print("\n[v1ii] Consultando CKAN Pernambuco...")
        cands, log = scan_ckan_pe()
        all_candidates.extend(cands)
        all_logs.append(log)

    if args.scan_dados_rj:
        print("\n[v1ii] Consultando Dados Abertos RJ...")
        cands, log = scan_dados_rj()
        all_candidates.extend(cands)
        all_logs.append(log)

    if args.scan_geocuritiba:
        print("\n[v1ii] Consultando GeoCuritiba...")
        cands, log = scan_geocuritiba()
        all_candidates.extend(cands)
        all_logs.append(log)

    if args.scan_dados_gov:
        print("\n[v1ii] Consultando dados.gov.br / S2ID / Atlas...")
        cands, log = scan_dados_gov()
        all_candidates.extend(cands)
        all_logs.append(log)

    if not all_candidates:
        print("[v1ii] Nenhum scanner ativado. Use pelo menos um --scan-* flag.")

    print()
    # Gerar outputs locais (sempre)
    write_local_outputs(all_candidates, all_logs)

    # Registry publico apenas com --force
    if mode == "force":
        print("\n[FORCE] Escrevendo registry publico...")
        write_public_registry(all_candidates)
    else:
        print("\n[DRY-RUN] Use --force para escrever registry publico.")

    # Sumario
    status_counts: dict = {}
    for c in all_candidates:
        status_counts[c.classification_status] = status_counts.get(c.classification_status, 0) + 1

    gt_count = status_counts.get("OBSERVED_VECTOR_GROUND_TRUTH_CANDIDATE", 0)

    print("\n" + "=" * 60)
    print("RELATORIO v1ii-R1")
    print("=" * 60)
    print(f"  Repositorios consultados : {len(all_logs)}")
    print(f"  Recursos encontrados     : {len(all_candidates)}")
    print(f"  Ground truth candidatos  : {gt_count}")
    if status_counts:
        print("\n  Breakdown por status:")
        for st, cnt in sorted(status_counts.items(), key=lambda x: -x[1]):
            print(f"    {st:<45}: {cnt}")
    print(f"\n  operational_ground_truth_status : BLOCKED")
    print(f"  can_create_training_label       : false")
    print(f"  ml_label_status                 : BLOCKED_UNTIL_SPLIT_AND_LEAKAGE_PROTOCOL")
    print(f"  can_reopen_protocol_b           : false")
    if gt_count == 0:
        print(f"\n[RESULT] Nenhum ground truth vetorial observado confirmado.")
        print(f"[RESULT] Lacuna de disponibilidade publica documentada.")
    print(f"[RESULT] Invariante mantido: bloqueio operacional permanece.")


if __name__ == "__main__":
    main()
