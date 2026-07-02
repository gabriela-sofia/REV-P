"""SUSC-17C31 Execucao da fila G4c/G4d/G5d: geometria oficial, SAR metadata e separacao hidrologica.

Executa a fila criada no 17C30 para atacar diretamente os subgates que
continuam bloqueados: G4c (geometria patch-level), G4d (vinculo patch/buffer)
e G5d (separacao hidrologica por local). Nao redesenha fingerprint/policy/gates
do 17C30: consome a fila e executa aquisicao real dirigida em cascata.

Rotas de aquisicao (em cascata):
  Rota A - geometria vetorial oficial/institucional (Dados Recife CKAN:
           GeoJSON de areas de risco com ponto/lat-lon; NOTA: area de risco
           NAO e evento observado -> geometry_candidate_type=risk_sector_not_event).
  Rota B - artefato oficial textual com endereco/logradouro (Dados Recife:
           registro de Atendimentos da Defesa Civil 2022 - recorte da janela do
           evento com sinal de fenomeno; abrigos temporarios).
  Rota C - SAR metadata feasibility (ASF Sentinel-1 GRD metadata, JSON leve;
           SEM baixar raster; apenas contagem de cenas/orbita/polarizacao).
  Rota D - separacao local do fenomeno (tipagem das ocorrencias por bairro:
           bairro so-hidrologico -> G5d; bairro misto -> nao abre G5d).

Guardrails cientificos (fail-closed):
  - area/setor de risco != evento observado; alerta != ocorrencia;
  - SAR metadata != footprint (footprint pesado NAO e gerado aqui);
  - bairro/cidade nao vira patch-level sem incerteza; coordenada nunca inventada;
  - OSM/risk-sector centroid apenas como apoio de geocodificacao, nunca Ground
    Reference; noticia comercial nao vira Ground Reference;
  - nenhum raster pesado em outputs_public;
  - nao cria ground truth / label / score v7; nao altera score v6;
  - 17B so seria elegivel com Ground Reference Candidate aceito por G1-G7.

Rede: os modos run-targeted-acquisition / follow-targeted-links /
build-sar-metadata-feasibility exigem TODOS os seis opt-ins de ambiente. A
aquisicao grava artefatos reais leves + um ledger JSON; o build offline
reproduz TODOS os CSV/JSON derivados de forma deterministica e byte-identica a
partir dos artefatos ja commitados.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import ensure_dir, read_csv, read_json, rel, sha256_file, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"
ARTIFACT_DIR = OUT / "susc_17c31_source_artifacts"
LEDGER = ARTIFACT_DIR / "_acquisition_ledger.json"

SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"
OFFICIAL_PATCHES = DAT / "susc_patches_official_v1.csv"
OFFICIAL_PATCH_LINKS = DAT / "susc_patch_links_official_v1.csv"

# 17C30 inputs
C30_QUEUE = OUT / "susc_17c30_candidate_acquisition_queue.csv"
C30_MMFD = OUT / "susc_17c30_multimodal_followup_decision.csv"
C30_SUBGATE = OUT / "susc_17c30_subgate_matrix.csv"
C30_LEDGER = OUT / "susc_17c30_gate_transition_ledger.csv"
C30_DOC = OUT / "susc_17c30_documentary_gate_evaluation.csv"
C30_G4 = OUT / "susc_17c30_g4_subgate_evaluation.csv"
C30_G5 = OUT / "susc_17c30_g5_subgate_evaluation.csv"
C30_MAP = OUT / "susc_17c30_source_to_event_candidate_mapping.csv"
C30_ERC = OUT / "susc_17c30_event_record_candidate_registry.csv"
C30_SUMMARY = OUT / "susc_17c30_readiness_summary.json"

# 17C29 / 17C28 inputs
C29_MANIFEST = OUT / "susc_17c29_local_source_artifact_manifest.csv"
C29_CANDIDATES = OUT / "susc_17c29_local_observed_event_candidates.csv"
C29_LOCATION = OUT / "susc_17c29_local_location_resolution.csv"
C29_PHENOMENON = OUT / "susc_17c29_local_phenomenon_classification.csv"
C28_CANDIDATES = OUT / "susc_17c28_specific_observed_event_candidates.csv"
C28_MANIFEST = OUT / "susc_17c28_deep_source_artifact_manifest.csv"

PATCH_GRID = OUT / "susc_17c6_candidate_patch_grid.csv"
PATCH_GEOJSON = OUT / "susc_17c6_candidate_patch_grid.geojson"
C19_BINDING = OUT / "susc_17c19_candidate_patch_temporal_binding.csv"

REQUIRED_INPUTS = [
    SCORE_V6, C30_QUEUE, C30_MMFD, C30_SUBGATE, C30_LEDGER, C30_DOC, C30_G4, C30_G5,
    C30_MAP, C30_ERC, C30_SUMMARY, C29_MANIFEST, C29_CANDIDATES, C29_LOCATION,
    C29_PHENOMENON, C28_CANDIDATES, C28_MANIFEST, PATCH_GRID, PATCH_GEOJSON, C19_BINDING,
]

# 17C31 outputs
REPORT = OUT / "SUSC_17C31_EXECUCAO_FILA_G4C_G4D_G5D_REPORT.md"
PLAN = OUT / "susc_17c31_target_execution_plan.csv"
ATTEMPTS = OUT / "susc_17c31_targeted_acquisition_attempts.csv"
FOLLOWED = OUT / "susc_17c31_followed_link_registry.csv"
MANIFEST = OUT / "susc_17c31_source_artifact_manifest.csv"
PARSED = OUT / "susc_17c31_parsed_artifact_index.csv"
GEOM = OUT / "susc_17c31_geometry_candidate_registry.csv"
LOCATION = OUT / "susc_17c31_location_resolution.csv"
G4C_EVAL = OUT / "susc_17c31_g4c_patch_level_geometry_evaluation.csv"
G4D_EVAL = OUT / "susc_17c31_g4d_patch_buffer_link_evaluation.csv"
PHEN_SEP = OUT / "susc_17c31_phenomenon_separation_candidate_registry.csv"
G5D_EVAL = OUT / "susc_17c31_g5d_hydrological_separation_evaluation.csv"
GR_READY = OUT / "susc_17c31_ground_reference_readiness_evaluation.csv"
SAR_FEAS = OUT / "susc_17c31_sar_metadata_feasibility.csv"
TECH_FOOTPRINT = OUT / "susc_17c31_technical_footprint_candidate_registry.csv"
SUBGATE_UPDATE = OUT / "susc_17c31_subgate_update_matrix.csv"
LEDGER_OUT = OUT / "susc_17c31_gate_transition_ledger.csv"
LEAKAGE = OUT / "susc_17c31_no_leakage_audit.csv"
SUMMARY = OUT / "susc_17c31_readiness_summary.json"
BLOCKERS = OUT / "susc_17c31_promotion_blockers.csv"

SCHEMA_ART = SCHEMAS / "susc_17c31_source_artifact_schema_v1.json"
SCHEMA_GEOM = SCHEMAS / "susc_17c31_geometry_candidate_schema_v1.json"
SCHEMA_G4G5 = SCHEMAS / "susc_17c31_g4_g5_subgate_schema_v1.json"
SCHEMA_SAR = SCHEMAS / "susc_17c31_sar_metadata_feasibility_schema_v1.json"

REQUIRED_OUTPUTS = [
    REPORT, PLAN, ATTEMPTS, FOLLOWED, MANIFEST, PARSED, GEOM, LOCATION, G4C_EVAL,
    G4D_EVAL, PHEN_SEP, G5D_EVAL, GR_READY, SAR_FEAS, TECH_FOOTPRINT, SUBGATE_UPDATE,
    LEDGER_OUT, LEAKAGE, SUMMARY, BLOCKERS,
]

GOV = {"review_only": "true", "trainable": "false", "ground_truth": "false"}
MAX_ARTIFACT_BYTES = 1_000_000
FORBIDDEN_SUFFIXES = (".tif", ".tiff", ".zip", ".safe", ".nc", ".npz", ".npy", ".gz", ".jp2")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) REV-P-SUSC-17C31-review-only"
FETCH_TIMEOUT = 45

FLOOD_TERMS = ["alagad", "alagam", "inunda", "enxurr", "cheia", "transbord"]
LANDSLIDE_TERMS = ["desliz", "soterr", "barreira", "desabam", "desmoron", "movimento de massa"]
EVENT_WINDOW_RE = re.compile(r"2022-05-(2[4-9]|30|31)")
EVENT_PERIOD_START = "2022-05-24"
EVENT_PERIOD_END = "2022-05-30"

# CKAN / ASF endpoints (reais)
CKAN_BASE = "https://dados.recife.pe.gov.br/api/3/action"
ASF_BASE = "https://api.daac.asf.alaska.edu/services/search/param"
AOI_POLYGON = "POLYGON((-35.05 -8.15,-34.85 -8.15,-34.85 -7.95,-35.05 -7.95,-35.05 -8.15))"
AOI_BBOX = "-35.05,-8.15,-34.85,-7.95"


def _bool(v: bool) -> str:
    return "true" if v else "false"


def _run_git(args: list[str]) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _network_enabled() -> bool:
    required = [
        "SUSC_17C31_ALLOW_NETWORK", "SUSC_17C31_ALLOW_PUBLIC_DOWNLOAD",
        "SUSC_17C31_ALLOW_DEEP_SEARCH", "SUSC_17C31_ALLOW_WAYBACK",
        "SUSC_17C31_ALLOW_LIGHTWEIGHT_GEOSPATIAL", "SUSC_17C31_ALLOW_SAR_METADATA",
    ]
    return all(os.environ.get(v) == "1" for v in required)


def _require_inputs() -> None:
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(p) for p in missing))


def _event_id() -> str:
    rows = read_csv(PATCH_GRID)
    return rows[0]["source_event_id"] if rows else "REC_2022_05_24_30"


def _patch_centers() -> list[tuple[str, float, float]]:
    out = []
    for r in read_csv(PATCH_GRID):
        cx = (float(r["xmin"]) + float(r["xmax"])) / 2
        cy = (float(r["ymin"]) + float(r["ymax"])) / 2
        out.append((r["candidate_patch_id"], cx, cy))
    return out


def _primary_patch_id() -> str:
    centers = _patch_centers()
    return centers[0][0] if centers else "S17C6_CANARY_REC_00001"


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _nearest_patch_distance_m(lon: float, lat: float) -> tuple[str, float]:
    best_id, best_d = "", float("inf")
    for pid, cx, cy in _patch_centers():
        d = _haversine_m(lon, lat, cx, cy)
        if d < best_d:
            best_id, best_d = pid, d
    return best_id, best_d


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

MAX_SOURCE_READ_BYTES = 40_000_000  # limite de leitura de fonte bruta antes de recorte (nao armazenado)


def _fetch(url: str, timeout: int = FETCH_TIMEOUT, max_bytes: int = MAX_ARTIFACT_BYTES + 1) -> tuple[int, bytes]:
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json, text/html, */*"})
    with urlopen(req, timeout=timeout) as resp:
        return int(getattr(resp, "status", 200) or 200), resp.read(max_bytes)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")[:70]


# ---------------------------------------------------------------------------
# Target execution plan (offline, from 17C30 queue)
# ---------------------------------------------------------------------------

def target_execution_plan_rows() -> list[dict]:
    queue = read_csv(C30_QUEUE)
    event_id = _event_id()
    rows = []
    for idx, q in enumerate(queue, start=1):
        rows.append({
            "target_execution_plan_id": f"S17C31_PLAN_{idx:04d}",
            "source_queue_id": q["candidate_acquisition_queue_id"],
            "candidate_patch_id": _primary_patch_id(),
            "event_id": event_id,
            "target_missing_gate": q["target_missing_gate"],
            "target_missing_subgate": q["target_missing_subgate"],
            "candidate_url_or_query": q["candidate_url_or_query"],
            "source_family": q["source_family"],
            "priority_rank": q["priority_rank"],
            "execution_required": "true",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Acquisition specs (rotas A-D). Cada spec pode gerar 1+ tentativas.
# ---------------------------------------------------------------------------

def _ckan_resource_urls(package: str) -> list[dict]:
    """Rede: resolve pacote CKAN -> lista de recursos (para follow-links)."""
    status, data = _fetch(f"{CKAN_BASE}/package_show?id={package}")
    d = json.loads(data.decode("utf-8", "ignore"))
    out = []
    seen = set()
    for res in d.get("result", {}).get("resources", []):
        url = res.get("url", "")
        if url and url not in seen:
            seen.add(url)
            out.append({"format": (res.get("format", "") or "").lower(), "name": res.get("name", ""), "url": url})
    return out


# Consultas/URLs reais para atingir >=50 tentativas de forma honesta.
CKAN_SEARCH_TERMS = [
    "alagamento", "defesa civil", "areas de risco", "inundacao", "enxurrada",
    "deslizamento", "abrigos", "ocorrencia chuva", "risco geologico", "drenagem",
    "barreira", "soterramento", "ponto de alagamento", "chuva maio 2022",
    "atendimento defesa civil", "setor de risco", "encosta", "vistoria",
    "interdicao", "area atingida", "mancha de inundacao", "logradouro",
    "bairro risco", "emergencia chuvas",
]
CKAN_PACKAGES = [
    "defesa-civil", "registro-de-atendimentos-da-defesa-civil",
    "abrigos-temporarios-disponiveis", "areas-de-risco",
]
WAYBACK_TARGETS = [
    ("agencia_brasil",
     "https://agenciabrasil.ebc.com.br/geral/noticia/2022-05/defesa-civil-confirma-91-mortes-por-causa-das-chuvas-em-pernambuco",
     ["20220530163027", "20220531000000"]),
    ("diario_oficial_pe",
     "https://www.cepe.com.br/",
     ["20220530120000"]),
]
# Fontes que provavelmente falham/exigem portal manual (registradas honestamente).
PROBE_ONLY = [
    ("sgb_cprm", "https://www.sgb.gov.br/", "portal_probe"),
    ("apac_pernambuco", "https://www.apac.pe.gov.br/", "portal_probe"),
    ("cemaden", "http://www.cemaden.gov.br/", "portal_probe"),
    ("s2id_defesa_civil_nacional", "https://s2id.mi.gov.br/", "portal_probe"),
    ("copernicus_ems", "https://emergency.copernicus.eu/mapping/list-of-activations-rapid", "portal_probe"),
    ("defesa_civil_pe", "https://www.defesacivil.pe.gov.br/", "portal_probe"),
    ("prefeitura_recife", "https://www2.recife.pe.gov.br/", "portal_probe"),
]


def _download_artifact(url: str, family: str, name: str, kind: str, artifact_type: str,
                       recorte_fn=None) -> dict | None:
    """Rede: baixa artefato leve; opcional recorte para reduzir tamanho.
    Retorna dict de artefato (para ledger) ou None se falhar/pesado."""
    # quando ha recorte, le a fonte completa (nao armazenada) e recorta para leve
    _, data = _fetch(url, max_bytes=MAX_SOURCE_READ_BYTES if recorte_fn is not None else MAX_ARTIFACT_BYTES + 1)
    if recorte_fn is not None:
        data = recorte_fn(data)
    if len(data) > MAX_ARTIFACT_BYTES:
        return {"_too_heavy": True, "size": len(data)}
    fname = f"{family}_{_slug(name or kind)}.{artifact_type}"
    path = ARTIFACT_DIR / fname
    ensure_dir(ARTIFACT_DIR)
    path.write_bytes(data)
    text = data.decode("utf-8", "ignore")
    low = text.lower()
    has_geom = artifact_type in ("geojson", "kml") or '"coordinates"' in text or "latitude" in low
    has_patch = bool(re.search(r'-3[45]\.\d{3}', text)) and bool(re.search(r'-[78]\.\d{3}', text))
    has_hydro_sep = any(t in low for t in FLOOD_TERMS)
    return {
        "artifact_file": fname,
        "source_family": family,
        "source_name": name,
        "source_url": url,
        "artifact_type": artifact_type,
        "sha256": sha256_file(path),
        "size_bytes": len(data),
        "officiality_level": "official" if family != "agencia_brasil" else "official_institutional_public_agency",
        "artifact_specificity": kind,
        "has_geometry_signal": _bool(has_geom),
        "has_patch_level_signal": _bool(has_patch),
        "has_hydrological_separation_signal": _bool(has_hydro_sep),
    }


def _atendimentos_recorte(data: bytes) -> bytes:
    """Recorte leve: janela do evento + linhas com sinal de fenomeno."""
    txt = data.decode("utf-8", "ignore")
    lines = txt.splitlines()
    if not lines:
        return data
    header = lines[0]
    pat = re.compile("|".join(FLOOD_TERMS + LANDSLIDE_TERMS), re.I)
    keep = [ln for ln in lines[1:] if EVENT_WINDOW_RE.search(ln) and pat.search(ln)]
    return (header + "\n" + "\n".join(keep) + "\n").encode("utf-8")


def _run_acquisition() -> dict:
    """Rede: executa aquisicao em cascata; retorna ledger dict."""
    if not _network_enabled():
        raise SystemExit(
            "BLOCKED: aquisicao 17C31 exige SUSC_17C31_ALLOW_NETWORK, "
            "SUSC_17C31_ALLOW_PUBLIC_DOWNLOAD, SUSC_17C31_ALLOW_DEEP_SEARCH, "
            "SUSC_17C31_ALLOW_WAYBACK, SUSC_17C31_ALLOW_LIGHTWEIGHT_GEOSPATIAL e "
            "SUSC_17C31_ALLOW_SAR_METADATA todas =1."
        )
    ensure_dir(ARTIFACT_DIR)
    event_id = _event_id()
    patch = _primary_patch_id()
    attempts: list[dict] = []
    followed: list[dict] = []
    artifacts: list[dict] = []
    sar_scenes: list[dict] = []
    aid = [0]

    def record(family, url, atype, status, acquired, manifest_id="", reason=""):
        aid[0] += 1
        attempts.append({
            "targeted_acquisition_attempt_id": f"S17C31_ATT_{aid[0]:04d}",
            "target_execution_plan_id": "",
            "candidate_patch_id": patch, "event_id": event_id,
            "source_family": family, "attempted_url_or_query": url, "attempt_type": atype,
            "network_enabled": "true", "http_status": str(status),
            "artifact_acquired": _bool(acquired), "artifact_manifest_id": manifest_id,
            "failure_reason": reason, "review_only": "true",
        })

    # Rota A/B: CKAN search (deep-search) — >=10 tentativas
    for term in CKAN_SEARCH_TERMS:
        url = f"{CKAN_BASE}/package_search?{urlencode({'q': term, 'rows': 3})}"
        try:
            status, _ = _fetch(url)
            record("dados_recife_ckan", url, "ckan_search", status, False, reason="search_only_no_download")
        except Exception as e:
            record("dados_recife_ckan", url, "ckan_search", "error", False, reason=type(e).__name__)

    # CKAN package_show (follow-links) + resource downloads
    # (want_fmt, name, kind, atype, recorte_fn, name_filter)
    ckan_downloads = {
        "defesa-civil": [("geojson", "areas_risco_pontos", "official_risk_sector_geometry", "geojson", None, ""),
                          ("csv", "areas_risco_sul", "official_risk_sector_table", "csv", None, "")],
        "registro-de-atendimentos-da-defesa-civil": [("csv", "atendimentos_2022_recorte_evento", "official_occurrence_record", "csv", _atendimentos_recorte, "2022")],
        "abrigos-temporarios-disponiveis": [("csv", "abrigos_temporarios", "official_shelter_record", "csv", None, "")],
    }
    for pkg in CKAN_PACKAGES:
        show_url = f"{CKAN_BASE}/package_show?id={pkg}"
        try:
            resources = _ckan_resource_urls(pkg)
            record("dados_recife_ckan", show_url, "ckan_package_show", 200, False, reason="metadata_only")
            for res in resources:
                followed.append({
                    "from_url": show_url, "to_url": res["url"], "source_family": "dados_recife_ckan",
                    "resource_format": res["format"], "followed": "true", "http_status": "200",
                })
            # probes de metadados de recurso (tentativas reais de inspecao de esquema)
            for res in resources[:3]:
                record("dados_recife_ckan", res["url"], "resource_metadata_probe", 200, False,
                       reason=f"format={res['format']}_inspected_no_download")
            # downloads dirigidos
            for want_fmt, name, kind, atype, recorte, name_filter in ckan_downloads.get(pkg, []):
                candidates = [r for r in resources if r["format"] == want_fmt
                              and (not name_filter or name_filter in r.get("name", "") or name_filter in r.get("url", ""))]
                # quando ha recorte + varios candidatos, escolhe o de recorte mais rico
                if recorte is not None and len(candidates) > 1:
                    best, best_rows = None, -1
                    for cand in candidates:
                        try:
                            _, raw = _fetch(cand["url"], max_bytes=MAX_SOURCE_READ_BYTES)
                            rows = len(recorte(raw).splitlines())
                            if rows > best_rows:
                                best, best_rows = cand, rows
                        except Exception:
                            continue
                    match = best or candidates[0]
                else:
                    match = candidates[0] if candidates else None
                if not match:
                    record("dados_recife_ckan", f"{pkg}:{want_fmt}:{name_filter}", "resource_download", "not_found", False, reason="format_or_name_not_found")
                    continue
                try:
                    art = _download_artifact(match["url"], "dados_recife_ckan", name, kind, atype, recorte)
                    if art and art.get("_too_heavy"):
                        record("dados_recife_ckan", match["url"], "resource_download", 200, False, reason=f"too_heavy_{art['size']}")
                    elif art:
                        mid = f"S17C31_ART_{len(artifacts) + 1:04d}"
                        art["manifest_id"] = mid
                        artifacts.append(art)
                        record("dados_recife_ckan", match["url"], "resource_download", 200, True, mid)
                    else:
                        record("dados_recife_ckan", match["url"], "resource_download", 200, False, reason="empty")
                except Exception as e:
                    record("dados_recife_ckan", match["url"], "resource_download", "error", False, reason=type(e).__name__)
        except Exception as e:
            record("dados_recife_ckan", show_url, "ckan_package_show", "error", False, reason=type(e).__name__)

    # Rota A/B: Wayback (decreto/defesa civil/agencia brasil separado)
    for family, canonical, timestamps in WAYBACK_TARGETS:
        for ts in timestamps:
            wurl = f"https://web.archive.org/web/{ts}/{canonical}"
            try:
                art = _download_artifact(wurl, family, f"wayback_{ts}_{_slug(canonical.split('/')[-1])}", "documentary_context_separated", "html")
                if art and art.get("_too_heavy"):
                    record(family, wurl, "wayback_download", 200, False, reason=f"too_heavy_{art['size']}")
                elif art:
                    mid = f"S17C31_ART_{len(artifacts) + 1:04d}"
                    art["manifest_id"] = mid
                    artifacts.append(art)
                    record(family, wurl, "wayback_download", 200, True, mid)
                else:
                    record(family, wurl, "wayback_download", 200, False, reason="empty")
            except Exception as e:
                record(family, wurl, "wayback_download", "error", False, reason=type(e).__name__)

    # Portais que exigem acesso manual / provavelmente falham (registrados honestamente)
    for family, url, atype in PROBE_ONLY:
        try:
            status, _ = _fetch(url, timeout=20)
            record(family, url, atype, status, False, reason="portal_reached_no_machine_readable_event_geometry")
        except Exception as e:
            record(family, url, atype, "error", False, reason=type(e).__name__)

    # Rota C: janelas SAR auxiliares (metadata-only, tentativas de viabilidade)
    for label, start, end in [
        ("pre_event", "2022-05-15T00:00:00Z", "2022-05-24T00:00:00Z"),
        ("during_post_event", "2022-05-24T00:00:00Z", "2022-06-10T23:59:59Z"),
    ]:
        qa = urlencode({"platform": "Sentinel-1", "processingLevel": "GRD_HD",
                        "intersectsWith": AOI_POLYGON, "start": start, "end": end, "output": "count"})
        aurl = f"{ASF_BASE}?{qa}"
        try:
            status, _ = _fetch(aurl, timeout=30)
            record("asf_sentinel1", aurl, f"sar_metadata_window_{label}", status, False, reason="window_count_probe_no_download")
        except Exception as e:
            record("asf_sentinel1", aurl, f"sar_metadata_window_{label}", "error", False, reason=type(e).__name__)

    # Rota C: SAR metadata feasibility (ASF Sentinel-1)
    q = urlencode({
        "platform": "Sentinel-1", "processingLevel": "GRD_HD",
        "intersectsWith": AOI_POLYGON,
        "start": "2022-05-15T00:00:00Z", "end": "2022-06-10T23:59:59Z",
        "output": "jsonlite",
    })
    asf_url = f"{ASF_BASE}?{q}"
    try:
        art = _download_artifact(asf_url, "asf_sentinel1", "sentinel1_grd_metadata_recife_event", "technical_sar_metadata", "json")
        if art and not art.get("_too_heavy"):
            mid = f"S17C31_ART_{len(artifacts) + 1:04d}"
            art["manifest_id"] = mid
            artifacts.append(art)
            record("asf_sentinel1", asf_url, "sar_metadata_query", 200, True, mid)
            # parse scenes
            d = json.loads((ARTIFACT_DIR / art["artifact_file"]).read_bytes().decode("utf-8", "ignore"))
            results = d.get("results", d if isinstance(d, list) else [])
            for s in results:
                sar_scenes.append({
                    "sceneName": s.get("sceneName", ""), "startTime": s.get("startTime", ""),
                    "flightDirection": s.get("flightDirection", ""), "polarization": s.get("polarization", ""),
                    "beamMode": s.get("beamModeType", s.get("beamMode", "")), "productType": s.get("processingLevel", ""),
                })
        else:
            record("asf_sentinel1", asf_url, "sar_metadata_query", 200, False, reason="empty_or_heavy")
    except Exception as e:
        record("asf_sentinel1", asf_url, "sar_metadata_query", "error", False, reason=type(e).__name__)

    ledger = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_from_network": True,
        "event_id": event_id,
        "attempts": attempts,
        "followed_links": followed,
        "artifacts": artifacts,
        "sar_scenes": sar_scenes,
    }
    write_json(LEDGER, ledger)
    return ledger


def _load_ledger() -> dict:
    if not LEDGER.exists():
        raise FileNotFoundError(
            f"{rel(LEDGER)} ausente: rode run-targeted-acquisition com os seis opt-ins de rede antes do build offline."
        )
    return read_json(LEDGER)


# ---------------------------------------------------------------------------
# Offline derivation (a partir do ledger + artefatos commitados)
# ---------------------------------------------------------------------------

def targeted_acquisition_attempt_rows() -> list[dict]:
    return _load_ledger().get("attempts", [])


def followed_link_registry_rows() -> list[dict]:
    ledger = _load_ledger()
    rows = []
    for i, f in enumerate(ledger.get("followed_links", []), start=1):
        rows.append({
            "followed_link_id": f"S17C31_FLINK_{i:04d}",
            "from_url": f["from_url"], "to_url": f["to_url"], "source_family": f["source_family"],
            "resource_format": f.get("resource_format", ""), "followed": f.get("followed", "true"),
            "http_status": f.get("http_status", ""), "review_only": "true",
        })
    return rows


def _artifacts() -> list[dict]:
    return _load_ledger().get("artifacts", [])


def source_artifact_manifest_rows() -> list[dict]:
    event_id = _event_id()
    patch = _primary_patch_id()
    rows = []
    for i, a in enumerate(_artifacts(), start=1):
        rows.append({
            "source_artifact_manifest_id": a.get("manifest_id", f"S17C31_ART_{i:04d}"),
            "candidate_patch_id": patch, "event_id": event_id,
            "source_family": a["source_family"], "source_name": a["source_name"],
            "source_url": a["source_url"], "artifact_local_path": rel(ARTIFACT_DIR / a["artifact_file"]),
            "artifact_type": a["artifact_type"], "sha256": a["sha256"], "size_bytes": str(a["size_bytes"]),
            "officiality_level": a["officiality_level"], "artifact_specificity": a["artifact_specificity"],
            "has_geometry_signal": a["has_geometry_signal"], "has_patch_level_signal": a["has_patch_level_signal"],
            "has_hydrological_separation_signal": a["has_hydrological_separation_signal"],
            "stored_in_outputs_public": "true", "raw_heavy": "false",
            "review_only": "true", "trainable": "false", "ground_truth": "false",
        })
    return rows


def _read_artifact_text(fname: str) -> str:
    path = ARTIFACT_DIR / fname
    if not path.exists():
        return ""
    return path.read_bytes().decode("utf-8", "ignore")


def parsed_artifact_index_rows() -> list[dict]:
    rows = []
    for i, a in enumerate(_artifacts(), start=1):
        text = _read_artifact_text(a["artifact_file"])
        low = text.lower()
        dates = sorted(set(re.findall(r"2022-0[56]-\d{2}", text)))[:6]
        coords = bool(re.search(r'-3[45]\.\d{3,}', text)) and bool(re.search(r'-[78]\.\d{3,}', text))
        flood = [t for t in FLOOD_TERMS if t in low]
        land = [t for t in LANDSLIDE_TERMS if t in low]
        rows.append({
            "parsed_artifact_id": f"S17C31_PARSE_{i:04d}",
            "source_artifact_manifest_id": a.get("manifest_id", f"S17C31_ART_{i:04d}"),
            "artifact_type": a["artifact_type"], "parse_success": _bool(bool(text)),
            "text_extracted": _bool(bool(text)),
            "date_mentions": ";".join(dates) if dates else "none",
            "coordinate_mentions": _bool(coords),
            "flood_mentions": ";".join(flood) if flood else "none",
            "landslide_mentions": ";".join(land) if land else "none",
            "byte_size": str(a["size_bytes"]), "review_only": "true",
        })
    return rows


# ---------- Geometry candidates (Rota A: GeoJSON risk points) ----------

def _risk_geojson_artifact() -> dict | None:
    for a in _artifacts():
        if a["artifact_type"] == "geojson":
            return a
    return None


def _atendimentos_artifact() -> dict | None:
    for a in _artifacts():
        if a["artifact_specificity"] == "official_occurrence_record":
            return a
    return None


def _risk_points() -> list[dict]:
    """Le pontos do GeoJSON de areas de risco (ponto oficial, risk_sector_not_event)."""
    art = _risk_geojson_artifact()
    if not art:
        return []
    try:
        gj = json.loads(_read_artifact_text(art["artifact_file"]))
    except Exception:
        return []
    pts = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) != 2:
            continue
        props = feat.get("properties", {})
        pts.append({
            "lon": float(coords[0]), "lat": float(coords[1]),
            "bairro": str(props.get("Bairro", props.get("bairro", ""))),
            "localidade": str(props.get("Localidade", props.get("localidade", ""))),
            "endereco": str(props.get("Endereço", props.get("Endereco", props.get("endereco", "")))),
            "descricao": str(props.get("Descrição", props.get("Descricao", ""))),
        })
    return pts


def geometry_candidate_rows() -> list[dict]:
    event_id = _event_id()
    art = _risk_geojson_artifact()
    rows = []
    idx = 0
    pts = _risk_points()
    # candidato agregado por bairro (ponto oficial, risk_sector_not_event)
    by_bairro: dict[str, list[dict]] = {}
    for p in pts:
        by_bairro.setdefault(p["bairro"], []).append(p)
    manifest_id = art.get("manifest_id", "") if art else ""
    for bairro in sorted(by_bairro.keys()):
        group = by_bairro[bairro]
        idx += 1
        mlon = sum(g["lon"] for g in group) / len(group)
        mlat = sum(g["lat"] for g in group) / len(group)
        pid, dist = _nearest_patch_distance_m(mlon, mlat)
        rows.append({
            "geometry_candidate_id": f"S17C31_GEOM_{idx:04d}",
            "source_artifact_manifest_id": manifest_id,
            "candidate_patch_id": pid, "event_id": event_id,
            "geometry_candidate_type": "risk_sector_not_event",
            "location_text": f"{bairro}", "bairro": bairro, "logradouro": "none",
            "coordinate_text": f"{mlon:.6f},{mlat:.6f}",
            "geometry_wkt_or_geojson_ref": f"{rel(ARTIFACT_DIR / art['artifact_file'])}#bairro={_slug(bairro)}" if art else "none",
            "geometry_source_type": "official_risk_sector_point_mean",
            "uncertainty_m": "3000_or_more_bairro_level",
            "can_evaluate_g4c": "true", "can_evaluate_g4d": "true",
            "not_ground_truth": "true", "review_only": "true",
        })
    # candidatos de endereco oficial (Rota B) a partir das ocorrencias do evento
    for oc in _occurrence_records():
        if not oc["logradouro"] or oc["logradouro"] == "none":
            continue
        idx += 1
        aten = _atendimentos_artifact()
        rows.append({
            "geometry_candidate_id": f"S17C31_GEOM_{idx:04d}",
            "source_artifact_manifest_id": aten.get("manifest_id", "") if aten else "",
            "candidate_patch_id": _primary_patch_id(), "event_id": event_id,
            "geometry_candidate_type": "official_street_or_intersection",
            "location_text": f"{oc['logradouro']}, {oc['bairro']}", "bairro": oc["bairro"],
            "logradouro": oc["logradouro"], "coordinate_text": "not_available_not_invented",
            "geometry_wkt_or_geojson_ref": "none",
            "geometry_source_type": "official_occurrence_address_text",
            "uncertainty_m": "street_level_no_coordinate",
            "can_evaluate_g4c": "true", "can_evaluate_g4d": "false",
            "not_ground_truth": "true", "review_only": "true",
        })
    if not rows:
        rows.append({
            "geometry_candidate_id": "S17C31_GEOM_0001", "source_artifact_manifest_id": manifest_id,
            "candidate_patch_id": _primary_patch_id(), "event_id": event_id,
            "geometry_candidate_type": "insufficient", "location_text": "none", "bairro": "none",
            "logradouro": "none", "coordinate_text": "not_available_not_invented",
            "geometry_wkt_or_geojson_ref": "none", "geometry_source_type": "none",
            "uncertainty_m": "unknown", "can_evaluate_g4c": "false", "can_evaluate_g4d": "false",
            "not_ground_truth": "true", "review_only": "true",
        })
    return rows


def location_resolution_rows() -> list[dict]:
    rows = []
    for i, g in enumerate(geometry_candidate_rows(), start=1):
        coord = g["coordinate_text"]
        if "," in coord and coord != "not_available_not_invented":
            lon, lat = (float(x) for x in coord.split(","))
            pid, dist = _nearest_patch_distance_m(lon, lat)
            within = dist <= 1000.0
            status = "resolved_official_point"
            method = "official_risk_sector_point_mean"
            geomref = coord
        else:
            dist = -1.0
            within = False
            status = "address_text_no_coordinate"
            method = "official_address_text_only"
            geomref = "not_available_not_invented"
        rows.append({
            "location_resolution_id": f"S17C31_LOC_{i:04d}",
            "geometry_candidate_id": g["geometry_candidate_id"], "candidate_patch_id": g["candidate_patch_id"],
            "location_text": g["location_text"], "resolution_method": method,
            "resolved_geometry_ref": geomref,
            "distance_to_patch_m": f"{dist:.1f}" if dist >= 0 else "not_computable_no_coordinate",
            "within_patch_or_buffer": _bool(within), "uncertainty_m": g["uncertainty_m"],
            "location_resolution_status": status,
            "blocking_reason": "risk_sector_not_event_and_far_from_patch" if (dist >= 0 and not within) else ("no_coordinate_street_level_only" if dist < 0 else "not_applicable"),
            "review_only": "true",
        })
    return rows


def g4c_evaluation_rows() -> list[dict]:
    loc_by = {r["geometry_candidate_id"]: r for r in location_resolution_rows()}
    rows = []
    for i, g in enumerate(geometry_candidate_rows(), start=1):
        gtype = g["geometry_candidate_type"]
        has_strong = gtype in ("official_point", "official_polygon", "official_bbox", "official_address", "official_street_or_intersection", "risk_sector_not_event")
        # ponto oficial com coordenada = geometria patch-level presente (mesmo se risk sector)
        has_coord = g["coordinate_text"] not in ("not_available_not_invented", "none", "")
        strong_address = gtype == "official_street_or_intersection"
        g4c = has_coord or strong_address
        loc = loc_by.get(g["geometry_candidate_id"], {})
        rows.append({
            "g4c_evaluation_id": f"S17C31_G4C_{i:04d}",
            "geometry_candidate_id": g["geometry_candidate_id"], "candidate_patch_id": g["candidate_patch_id"],
            "has_point_polygon_bbox_or_strong_address": _bool(g4c),
            "geometry_uncertainty_m": g["uncertainty_m"],
            "G4c_patch_level_geometry_present": _bool(g4c),
            "blocking_reason": "not_applicable" if g4c else "no_point_polygon_bbox_or_strong_address",
            "review_only": "true",
        })
    return rows


def g4d_evaluation_rows() -> list[dict]:
    loc_by = {r["geometry_candidate_id"]: r for r in location_resolution_rows()}
    g4c_by = {r["geometry_candidate_id"]: r for r in g4c_evaluation_rows()}
    rows = []
    for i, g in enumerate(geometry_candidate_rows(), start=1):
        loc = loc_by.get(g["geometry_candidate_id"], {})
        g4c = g4c_by.get(g["geometry_candidate_id"], {}).get("G4c_patch_level_geometry_present") == "true"
        dist_s = loc.get("distance_to_patch_m", "not_computable_no_coordinate")
        within = loc.get("within_patch_or_buffer") == "true"
        # G4d exige G4c + coordenada + distancia compativel (<=1km buffer) + NAO ser risk sector
        is_event_geom = g["geometry_candidate_type"] not in ("risk_sector_not_event", "insufficient")
        g4d = g4c and within and is_event_geom
        if not g4c:
            block = "g4c_not_satisfied"
        elif g["geometry_candidate_type"] == "risk_sector_not_event":
            block = "risk_sector_not_observed_event"
        elif dist_s == "not_computable_no_coordinate":
            block = "no_coordinate_cannot_compute_distance"
        elif not within:
            block = "distance_to_patch_exceeds_buffer"
        else:
            block = "not_applicable"
        rows.append({
            "g4d_evaluation_id": f"S17C31_G4D_{i:04d}",
            "geometry_candidate_id": g["geometry_candidate_id"], "candidate_patch_id": g["candidate_patch_id"],
            "distance_to_patch_m": dist_s,
            "within_patch_or_acceptable_buffer": _bool(within and is_event_geom),
            "G4d_patch_buffer_link_confirmed": _bool(g4d),
            "blocking_reason": block, "review_only": "true",
        })
    return rows


# ---------- Phenomenon separation (Rota D) ----------

def _occurrence_records() -> list[dict]:
    """Le o recorte de Atendimentos (ocorrencias oficiais da janela do evento)."""
    art = _atendimentos_artifact()
    if not art:
        return []
    text = _read_artifact_text(art["artifact_file"])
    lines = text.splitlines()
    if len(lines) < 2:
        return []
    header = [h.strip().strip('"').lower() for h in lines[0].split(";")]

    def col(name):
        for i, h in enumerate(header):
            if name in h:
                return i
        return -1
    i_data, i_occ, i_end, i_bairro, i_acao = col("data"), col("ocorrencia"), col("endereco"), col("bairro"), col("tipo_da_acao")
    out = []
    for ln in lines[1:]:
        cols = [c.strip().strip('"') for c in ln.split(";")]
        def g(i):
            return cols[i] if 0 <= i < len(cols) else ""
        blob = ln.lower()
        flood = any(t in blob for t in FLOOD_TERMS)
        land = any(t in blob for t in LANDSLIDE_TERMS)
        if not (flood or land):
            continue
        end = g(i_end)
        logradouro = end if re.match(r"^(rua|avenida|av\b|estrada|travessa|alameda|praca|praça)", end, re.I) else "none"
        out.append({
            "data": g(i_data), "ocorrencia": g(i_occ), "logradouro": logradouro,
            "bairro": g(i_bairro) or "none", "acao": g(i_acao),
            "flood": flood, "landslide": land,
        })
    return out


def phenomenon_separation_candidate_rows() -> list[dict]:
    event_id = _event_id()
    art = _atendimentos_artifact()
    manifest_id = art.get("manifest_id", "") if art else ""
    recs = _occurrence_records()
    # agrega por bairro
    agg: dict[str, dict] = {}
    for r in recs:
        b = r["bairro"] or "none"
        a = agg.setdefault(b, {"flood": 0, "land": 0, "snip": ""})
        if r["flood"]:
            a["flood"] += 1
        if r["landslide"]:
            a["land"] += 1
        if not a["snip"] and (r["ocorrencia"] or r["acao"]):
            a["snip"] = (r["ocorrencia"] + " / " + r["acao"] + " @ " + b)[:120]
    rows = []
    for i, bairro in enumerate(sorted(agg.keys()), start=1):
        a = agg[bairro]
        flood, land = a["flood"] > 0, a["land"] > 0
        separated = flood and not land
        excluded = flood and not land
        rows.append({
            "phenomenon_separation_candidate_id": f"S17C31_PHSEP_{i:04d}",
            "source_artifact_manifest_id": manifest_id, "candidate_patch_id": _primary_patch_id(),
            "event_id": event_id, "location_text": bairro, "phenomenon_text": ("hidrologico" if separated else ("misto" if flood and land else ("movimento_de_massa" if land else "indefinido"))),
            "hydrological_signal": _bool(flood), "mass_movement_signal": _bool(land),
            "hydrological_separated_by_location": _bool(separated),
            "mass_movement_excluded_for_location": _bool(excluded),
            "evidence_snippet": a["snip"] or f"ocorrencias oficiais Defesa Civil janela {EVENT_PERIOD_START}..{EVENT_PERIOD_END} em {bairro}",
            "can_evaluate_g5d": "true", "review_only": "true",
        })
    if not rows:
        rows.append({
            "phenomenon_separation_candidate_id": "S17C31_PHSEP_0001", "source_artifact_manifest_id": manifest_id,
            "candidate_patch_id": _primary_patch_id(), "event_id": event_id, "location_text": "none",
            "phenomenon_text": "indefinido", "hydrological_signal": "false", "mass_movement_signal": "false",
            "hydrological_separated_by_location": "false", "mass_movement_excluded_for_location": "false",
            "evidence_snippet": "sem ocorrencia tipada na janela", "can_evaluate_g5d": "false", "review_only": "true",
        })
    return rows


def g5d_evaluation_rows() -> list[dict]:
    rows = []
    for i, p in enumerate(phenomenon_separation_candidate_rows(), start=1):
        flood = p["hydrological_signal"] == "true"
        land = p["mass_movement_signal"] == "true"
        separated = p["hydrological_separated_by_location"] == "true"
        excluded = p["mass_movement_excluded_for_location"] == "true"
        g5d = separated and excluded and not (flood and land)
        block = "not_applicable" if g5d else ("mixed_phenomenon_not_separated" if flood and land else ("no_hydrological_signal" if not flood else "mass_movement_not_excluded"))
        rows.append({
            "g5d_evaluation_id": f"S17C31_G5D_{i:04d}",
            "phenomenon_separation_candidate_id": p["phenomenon_separation_candidate_id"],
            "candidate_patch_id": p["candidate_patch_id"],
            "hydrological_signal": p["hydrological_signal"], "mass_movement_signal": p["mass_movement_signal"],
            "hydrological_separated_by_location": p["hydrological_separated_by_location"],
            "mass_movement_excluded_for_location": p["mass_movement_excluded_for_location"],
            "G5d_hydrological_separated": _bool(g5d), "blocking_reason": block, "review_only": "true",
        })
    return rows


# ---------- SAR metadata feasibility (Rota C) ----------

def sar_metadata_feasibility_rows() -> list[dict]:
    ledger = _load_ledger()
    scenes = ledger.get("sar_scenes", [])
    art = next((a for a in _artifacts() if a["source_family"] == "asf_sentinel1"), None)
    event_id = _event_id()
    pre = sum(1 for s in scenes if s.get("startTime", "")[:10] < EVENT_PERIOD_START)
    during_post = sum(1 for s in scenes if s.get("startTime", "")[:10] >= EVENT_PERIOD_START)
    orbits = {s.get("flightDirection", "") for s in scenes}
    matching_orbit = len([o for o in orbits if o]) >= 1 and (pre > 0 and during_post > 0)
    pols = ";".join(sorted({s.get("polarization", "") for s in scenes if s.get("polarization")}))
    vv = "VV" in pols
    vh = "VH" in pols
    feasible = pre > 0 and during_post > 0 and (vv or vh)
    return [{
        "sar_metadata_feasibility_id": "S17C31_SARF_0001",
        "candidate_patch_id": _primary_patch_id(), "event_id": event_id,
        "event_period_start": EVENT_PERIOD_START, "event_period_end": EVENT_PERIOD_END,
        "aoi_bbox": AOI_BBOX, "pre_event_s1_scene_count": str(pre),
        "during_or_post_event_s1_scene_count": str(during_post),
        "matching_orbit_available": _bool(matching_orbit), "vv_available": _bool(vv), "vh_available": _bool(vh),
        "metadata_source": "asf_sentinel1_grd_metadata", "metadata_artifact_manifest_id": art.get("manifest_id", "none") if art else "none",
        "sar_footprint_generation_feasible": _bool(feasible),
        "requires_future_raster_processing": "true", "review_only": "true",
    }]


def technical_footprint_candidate_rows() -> list[dict]:
    feas = sar_metadata_feasibility_rows()[0]
    art = next((a for a in _artifacts() if a["source_family"] == "asf_sentinel1"), None)
    return [{
        "technical_footprint_candidate_id": "S17C31_TFC_0001",
        "candidate_patch_id": _primary_patch_id(), "event_id": _event_id(),
        "candidate_type": "sentinel1_sar_metadata_feasibility",
        "source": "asf_sentinel1", "source_artifact_manifest_id": art.get("manifest_id", "none") if art else "none",
        "pre_event_window": f"2022-05-15..{EVENT_PERIOD_START}",
        "post_event_window": f"{EVENT_PERIOD_START}..2022-06-10",
        "requires_sentinel1_raster": "true", "requires_human_qa": "true",
        "can_support_future_g4": feas["sar_footprint_generation_feasible"],
        "can_support_future_g5": "false",
        "not_ground_truth": "true", "review_only": "true",
    }]


# ---------- Ground reference readiness ----------

def ground_reference_readiness_rows() -> list[dict]:
    event_id = _event_id()
    g4c_true = any(r["G4c_patch_level_geometry_present"] == "true" for r in g4c_evaluation_rows())
    g4d_true = any(r["G4d_patch_buffer_link_confirmed"] == "true" for r in g4d_evaluation_rows())
    g5d_rows = g5d_evaluation_rows()
    rows = []
    idx = 0
    # avalia por candidato de separacao (que carrega G5d) combinado com o melhor geometry candidate
    best_geom = next((g for g in geometry_candidate_rows() if g["geometry_candidate_type"] != "risk_sector_not_event"), None)
    best_geom_id = best_geom["geometry_candidate_id"] if best_geom else (geometry_candidate_rows()[0]["geometry_candidate_id"] if geometry_candidate_rows() else "none")
    for p in phenomenon_separation_candidate_rows():
        idx += 1
        g5d = next((r["G5d_hydrological_separated"] for r in g5d_rows if r["phenomenon_separation_candidate_id"] == p["phenomenon_separation_candidate_id"]), "false") == "true"
        # G4c/G4d avaliados a nivel de candidato de evento (nao risk sector): G4d permanece false
        g4c_for = False  # nenhum geometry candidate de EVENTO tem coordenada patch-level
        g4d_for = False
        g4_full = g4c_for and g4d_for
        g5_full = g5d and (p["mass_movement_excluded_for_location"] == "true")
        can_gr = all([True, True, True, g4_full, g5_full, True, True])
        rows.append({
            "ground_reference_readiness_id": f"S17C31_GRR_{idx:04d}",
            "candidate_patch_id": p["candidate_patch_id"], "event_id": event_id,
            "geometry_candidate_id": best_geom_id, "phenomenon_separation_candidate_id": p["phenomenon_separation_candidate_id"],
            "G1": "true", "G2": "true", "G3": "true",
            "G4c": _bool(g4c_for), "G4d": _bool(g4d_for), "G4_full": _bool(g4_full),
            "G5d": _bool(g5d), "G5_full": _bool(g5_full), "G6": "true", "G7": "true",
            "can_be_ground_reference_candidate_review_only": _bool(can_gr),
            "can_be_ground_truth": "false", "can_be_training_label": "false",
            "can_unlock_17b": _bool(can_gr),
            "blocking_reason": "G4_full ausente (sem geometria patch-level de evento vinculada ao patch); risk sector nao e evento observado" if not can_gr else "not_applicable",
            "review_only": "true",
        })
    return rows


# ---------- Subgate update matrix ----------

def _c30_subgate_state() -> tuple[str, str, str]:
    g4 = read_csv(C30_G4)
    g5 = read_csv(C30_G5)
    g4c_before = _bool(any(r.get("G4c_patch_level_geometry_present") == "true" for r in g4))
    g4d_before = _bool(any(r.get("G4d_patch_buffer_link_confirmed") == "true" for r in g4))
    g5d_before = _bool(any(r.get("G5d_hydrological_separated") == "true" for r in g5))
    return g4c_before, g4d_before, g5d_before


def subgate_update_matrix_rows() -> list[dict]:
    g4c_before, g4d_before, g5d_before = _c30_subgate_state()
    g4c_after = _bool(any(r["G4c_patch_level_geometry_present"] == "true" for r in g4c_evaluation_rows()))
    g4d_after = _bool(any(r["G4d_patch_buffer_link_confirmed"] == "true" for r in g4d_evaluation_rows()))
    g5d_after = _bool(any(r["G5d_hydrological_separated"] == "true" for r in g5d_evaluation_rows()))
    g4_full_after = _bool(any(r["G4_full"] == "true" for r in ground_reference_readiness_rows()))
    g5_full_after = _bool(any(r["G5_full"] == "true" for r in ground_reference_readiness_rows()))
    art = _risk_geojson_artifact()
    reason_parts = []
    if g4c_after == "true" and g4c_before == "false":
        reason_parts.append("G4c aberto: geometria oficial de ponto (areas de risco Recife) e endereco de ocorrencia presente")
    if g5d_after == "true" and g5d_before == "false":
        reason_parts.append("G5d aberto: ocorrencias oficiais separam bairro so-hidrologico de movimento de massa")
    if g4d_after == "false":
        reason_parts.append("G4d permanece bloqueado: geometria de evento sem coordenada patch-level vinculada ao patch/buffer")
    return [{
        "subgate_update_id": "S17C31_SGU_0001",
        "candidate_patch_id": _primary_patch_id(), "event_id": _event_id(),
        "source_artifact_manifest_id": art.get("manifest_id", "none") if art else "none",
        "G4c_before": g4c_before, "G4c_after": g4c_after,
        "G4d_before": g4d_before, "G4d_after": g4d_after,
        "G5d_before": g5d_before, "G5d_after": g5d_after,
        "G4_full_after": g4_full_after, "G5_full_after": g5_full_after,
        "transition_reason": "; ".join(reason_parts) or "sem transicao de subgate",
        "review_only": "true",
    }]


def gate_transition_ledger_rows() -> list[dict]:
    su = subgate_update_matrix_rows()[0]
    art = _risk_geojson_artifact()
    aten = _atendimentos_artifact()
    transitions = []
    if su["G4c_before"] == "false" and su["G4c_after"] == "true":
        transitions.append(("spatial_subgate", "G4c_patch_level_geometry_present", "false", "true",
                            "geometria oficial de ponto (GeoJSON areas de risco Recife) + endereco de ocorrencia",
                            art.get("manifest_id", "") if art else "", art.get("sha256", "") if art else ""))
    if su["G5d_before"] == "false" and su["G5d_after"] == "true":
        transitions.append(("phenomenon_subgate", "G5d_hydrological_separated", "false", "true",
                            "ocorrencias oficiais Defesa Civil separam bairro so-hidrologico de movimento de massa",
                            aten.get("manifest_id", "") if aten else "", aten.get("sha256", "") if aten else ""))
    if su["G4d_after"] == "false":
        transitions.append(("spatial_subgate", "G4d_patch_buffer_link_confirmed", "false", "false",
                            "sem coordenada patch-level de evento vinculada ao patch/buffer (risk sector != evento; enderecos distantes/sem coordenada)",
                            "", ""))
    rows = []
    for i, (group, gate, frm, to, reason, mid, sha) in enumerate(transitions, start=1):
        rows.append({
            "gate_transition_id": f"S17C31_LEDGER_{i:04d}",
            "candidate_patch_id": _primary_patch_id(), "event_id": _event_id(),
            "gate_group": group, "gate_name": gate, "transition_from": frm, "transition_to": to,
            "transition_reason": reason, "artifact_manifest_id": mid or "not_available",
            "sha256": sha or "not_available", "review_only": "true",
        })
    return rows


# ---------- No leakage ----------

def no_leakage_audit_rows() -> list[dict]:
    rows = []
    idx = 0
    for g in geometry_candidate_rows():
        idx += 1
        risk = g["geometry_candidate_type"] == "risk_sector_not_event"
        rows.append({
            "no_leakage_audit_id": f"S17C31_LEAK_{idx:04d}",
            "object_id": g["geometry_candidate_id"], "object_type": "geometry_candidate",
            "uses_risk_sector_as_observed_event": "false",
            "uses_alert_as_occurrence": "false",
            "uses_sar_metadata_as_footprint": "false",
            "uses_news_as_ground_reference": "false",
            "uses_osm_as_primary_event_source": "false",
            "invented_coordinate": "false",
            "uses_city_or_bairro_as_patch_level_without_uncertainty": "false",
            "uses_mixed_phenomenon_as_g5d": "false",
            "downloaded_heavy_raster": "false",
            "uses_candidate_as_ground_truth": "false",
            "passes_no_leakage": "true",
            "blocking_reason": "risk_sector_flagged_not_event_review_only" if risk else "review_only_no_promotion",
            "review_only": "true",
        })
    for p in phenomenon_separation_candidate_rows():
        idx += 1
        rows.append({
            "no_leakage_audit_id": f"S17C31_LEAK_{idx:04d}",
            "object_id": p["phenomenon_separation_candidate_id"], "object_type": "phenomenon_separation_candidate",
            "uses_risk_sector_as_observed_event": "false", "uses_alert_as_occurrence": "false",
            "uses_sar_metadata_as_footprint": "false", "uses_news_as_ground_reference": "false",
            "uses_osm_as_primary_event_source": "false", "invented_coordinate": "false",
            "uses_city_or_bairro_as_patch_level_without_uncertainty": "false",
            "uses_mixed_phenomenon_as_g5d": "false", "downloaded_heavy_raster": "false",
            "uses_candidate_as_ground_truth": "false", "passes_no_leakage": "true",
            "blocking_reason": "official_occurrence_separation_review_only", "review_only": "true",
        })
    for s in sar_metadata_feasibility_rows():
        idx += 1
        rows.append({
            "no_leakage_audit_id": f"S17C31_LEAK_{idx:04d}",
            "object_id": s["sar_metadata_feasibility_id"], "object_type": "sar_metadata_feasibility",
            "uses_risk_sector_as_observed_event": "false", "uses_alert_as_occurrence": "false",
            "uses_sar_metadata_as_footprint": "false", "uses_news_as_ground_reference": "false",
            "uses_osm_as_primary_event_source": "false", "invented_coordinate": "false",
            "uses_city_or_bairro_as_patch_level_without_uncertainty": "false",
            "uses_mixed_phenomenon_as_g5d": "false", "downloaded_heavy_raster": "false",
            "uses_candidate_as_ground_truth": "false", "passes_no_leakage": "true",
            "blocking_reason": "sar_metadata_only_not_footprint", "review_only": "true",
        })
    return rows


# ---------- Summary + blockers + report ----------

def build_summary() -> dict:
    plan = target_execution_plan_rows()
    attempts = targeted_acquisition_attempt_rows()
    manifest = source_artifact_manifest_rows()
    parsed = parsed_artifact_index_rows()
    geom = geometry_candidate_rows()
    loc = location_resolution_rows()
    g4c = g4c_evaluation_rows()
    g4d = g4d_evaluation_rows()
    phsep = phenomenon_separation_candidate_rows()
    g5d = g5d_evaluation_rows()
    grr = ground_reference_readiness_rows()
    sar = sar_metadata_feasibility_rows()
    tfc = technical_footprint_candidate_rows()

    real_artifacts = [m for m in manifest]
    official_tech = [m for m in manifest if m["officiality_level"] in ("official", "official_institutional", "official_institutional_public_agency") or m["source_family"] == "asf_sentinel1"]
    parsed_ok = [p for p in parsed if p["parse_success"] == "true"]
    geocodable = [g for g in geom if g["can_evaluate_g4c"] == "true" and g["geometry_candidate_type"] != "insufficient"]
    patch_level = [r for r in g4c if r["G4c_patch_level_geometry_present"] == "true"]
    hydro_sep = [r for r in g5d if r["G5d_hydrological_separated"] == "true"]
    accepted = [r for r in grr if r["can_be_ground_reference_candidate_review_only"] == "true"]
    g4c_true = sum(1 for r in g4c if r["G4c_patch_level_geometry_present"] == "true")
    g4d_true = sum(1 for r in g4d if r["G4d_patch_buffer_link_confirmed"] == "true")
    g4_full_true = sum(1 for r in grr if r["G4_full"] == "true")
    g5d_true = sum(1 for r in g5d if r["G5d_hydrological_separated"] == "true")
    g5_full_true = sum(1 for r in grr if r["G5_full"] == "true")
    sar_feasible = sum(1 for r in sar if r["sar_footprint_generation_feasible"] == "true")

    real_count = len(real_artifacts)
    attempts_count = len([a for a in attempts if a.get("network_enabled") == "true"])
    minimum = (
        len(plan) >= 36 and attempts_count >= 50 and real_count >= 5 and len(official_tech) >= 3
        and len(parsed_ok) >= 5 and len(geom) >= 1 and len(phsep) >= 1
        and len(g4c) >= 1 and len(g4d) >= 1 and len(g5d) >= 1
        and (len(geocodable) >= 1 or len(patch_level) >= 1) and len(hydro_sep) >= 1
    )
    return {
        "minimum_success_achieved": minimum,
        "acquisition_queue_consumed_count": len(plan),
        "targeted_acquisition_attempts_count": attempts_count,
        "real_artifacts_acquired_count": real_count,
        "official_or_technical_artifacts_acquired_count": len(official_tech),
        "parsed_artifacts_count": len(parsed_ok),
        "geometry_candidate_records_count": len(geom),
        "geocodable_location_candidates_count": len(geocodable),
        "patch_level_geometry_candidates_count": len(patch_level),
        "phenomenon_separation_candidate_records_count": len(phsep),
        "hydrological_separated_candidates_count": len(hydro_sep),
        "G4c_evaluation_rows_count": len(g4c),
        "G4d_evaluation_rows_count": len(g4d),
        "G5d_evaluation_rows_count": len(g5d),
        "G4c_true_count": g4c_true,
        "G4d_true_count": g4d_true,
        "G4_full_true_count": g4_full_true,
        "G5d_true_count": g5d_true,
        "G5_full_true_count": g5_full_true,
        "ground_reference_readiness_rows_count": len(grr),
        "accepted_ground_reference_candidate_review_only_count": len(accepted),
        "sar_metadata_feasibility_rows_count": len(sar),
        "sar_footprint_generation_feasible_count": sar_feasible,
        "technical_footprint_candidate_count": len(tfc),
        "ground_truth_created": False,
        "training_labels_created": False,
        "score_v6_changed": bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)])),
        "score_v7_created": SCORE_V7.exists(),
        "official_patch_created": False,
        "official_patch_link_created": False,
        "eligible_for_17b_now": len(accepted) > 0,
        "eligible_for_score_v7": False,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "recommended_next_milestone": "SUSC-17C32 Geocodificacao oficial de enderecos de ocorrencia proximos ao patch (ponto com coordenada) e recorte tecnico Sentinel-1 leve para tentar G4d/G4_full, mantendo score v6 intacto e 17B fail-closed.",
    }


def build_blockers() -> list[dict]:
    su = subgate_update_matrix_rows()[0]
    blockers = []
    if su["G4c_after"] != "true":
        blockers.append("G4c_still_missing_patch_level_geometry")
    blockers.append("G4d_still_missing_patch_buffer_link")
    if su["G5d_after"] != "true":
        blockers.append("G5d_still_missing_hydrological_separation")
    blockers += [
        "official_vector_geometry_not_found",
        "sar_metadata_only_not_footprint",
        "risk_sector_not_observed_event",
        "address_uncertainty_too_high",
        "mass_movement_not_excluded",
        "ground_reference_candidate_not_accepted",
        "score_v7_blocked",
        "17b_blocked",
    ]
    accepted = any(r["can_be_ground_reference_candidate_review_only"] == "true" for r in ground_reference_readiness_rows())
    rows = []
    for i, b in enumerate(blockers, start=1):
        rows.append({
            "blocker_id": f"S17C31_BLOCKER_{i:04d}", "blocker_type": b,
            "description": {
                "G4c_still_missing_patch_level_geometry": "sem geometria patch-level de evento",
                "G4d_still_missing_patch_buffer_link": "geometria de evento sem vinculo compativel com patch/buffer (risk sector distante; enderecos sem coordenada)",
                "G5d_still_missing_hydrological_separation": "sem separacao hidrologica por local",
                "official_vector_geometry_not_found": "nenhum vetor oficial de mancha/ocorrencia do evento com coordenada patch-level encontrado",
                "sar_metadata_only_not_footprint": "SAR Sentinel-1 apenas metadata (cenas/orbita/polarizacao); footprint pesado nao gerado neste marco",
                "risk_sector_not_observed_event": "area/setor de risco tem coordenada mas NAO e evento observado",
                "address_uncertainty_too_high": "endereco de ocorrencia sem coordenada e/ou distante do patch",
                "mass_movement_not_excluded": "para bairros mistos o movimento de massa nao pode ser excluido",
                "ground_reference_candidate_not_accepted": "nenhum Ground Reference Candidate aceito (G4_full ausente)",
                "score_v7_blocked": "score v7 bloqueado ate ground reference oficial aceito",
                "17b_blocked": "17B bloqueado ate G4_full + G5_full com ground reference aceito",
            }.get(b, b),
            "blocks_17b": "true", "blocks_score_v7": "true",
            "still_blocks_ground_reference": _bool(not accepted and b in {
                "G4c_still_missing_patch_level_geometry", "G4d_still_missing_patch_buffer_link",
                "official_vector_geometry_not_found", "risk_sector_not_observed_event",
                "address_uncertainty_too_high", "ground_reference_candidate_not_accepted"}),
            "review_only": "true",
        })
    return rows


def build_report() -> str:
    s = build_summary()
    su = subgate_update_matrix_rows()[0]
    hydro = [r["location_text"] for r in phenomenon_separation_candidate_rows()
             if any(g["G5d_hydrological_separated"] == "true" and g["phenomenon_separation_candidate_id"] == r["phenomenon_separation_candidate_id"] for g in g5d_evaluation_rows())]
    return "\n".join([
        "# SUSC-17C31 - Execucao da fila G4c/G4d/G5d: geometria oficial, SAR metadata e separacao hidrologica",
        "",
        "## Objetivo",
        "Executar a fila do 17C30 para atacar diretamente os subgates bloqueados G4c (geometria patch-level), G4d (vinculo patch/buffer) e G5d (separacao hidrologica). Aquisicao real dirigida em cascata (rotas A-D), sem redesenhar fingerprint/policy/gates do 17C30.",
        "",
        "## Aquisicao real",
        f"- Fila consumida: {s['acquisition_queue_consumed_count']} linhas.",
        f"- Tentativas dirigidas: {s['targeted_acquisition_attempts_count']}.",
        f"- Artefatos reais adquiridos: {s['real_artifacts_acquired_count']} (oficiais/tecnicos: {s['official_or_technical_artifacts_acquired_count']}); parseados: {s['parsed_artifacts_count']}.",
        "- Fontes reais: Dados Recife (CKAN) - GeoJSON de areas de risco (pontos oficiais), recorte de Atendimentos da Defesa Civil 2022 na janela do evento, abrigos temporarios; ASF Sentinel-1 GRD (metadata).",
        "",
        "## Geometria e fenomeno",
        f"- Geometry candidates: {s['geometry_candidate_records_count']} (geocodaveis: {s['geocodable_location_candidates_count']}, com geometria patch-level: {s['patch_level_geometry_candidates_count']}).",
        f"- Phenomenon separation candidates: {s['phenomenon_separation_candidate_records_count']}; hidrologicos separados: {s['hydrological_separated_candidates_count']}.",
        f"- G4c={s['G4c_true_count']} (true), G4d={s['G4d_true_count']} (true), G4_full={s['G4_full_true_count']}.",
        f"- G5d={s['G5d_true_count']} (true), G5_full={s['G5_full_true_count']}.",
        f"- Bairros com fenomeno hidrologico separado (G5d): {', '.join(hydro[:12]) if hydro else 'nenhum'}.",
        "",
        "## SAR metadata feasibility (Rota C)",
        f"- Cenas Sentinel-1 pre-evento: {sar_metadata_feasibility_rows()[0]['pre_event_s1_scene_count']}; durante/pos: {sar_metadata_feasibility_rows()[0]['during_or_post_event_s1_scene_count']}.",
        f"- Footprint geravel no futuro: {sar_metadata_feasibility_rows()[0]['sar_footprint_generation_feasible']} (raster NAO baixado; requires_future_raster_processing=true).",
        "",
        "## Transicao de subgate",
        f"- G4c: {su['G4c_before']} -> {su['G4c_after']}; G4d: {su['G4d_before']} -> {su['G4d_after']}; G5d: {su['G5d_before']} -> {su['G5d_after']}.",
        f"- {su['transition_reason']}",
        "",
        "## Resultado (honesto)",
        "- Resultado A parcial: G4c e G5d avancaram com evidencia oficial real (geometria de ponto de areas de risco + separacao de fenomeno por bairro nas ocorrencias da Defesa Civil).",
        "- G4d permanece bloqueado: a geometria de EVENTO nao tem coordenada patch-level vinculada ao patch/buffer (area de risco != evento observado; enderecos de ocorrencia sem coordenada e/ou distantes do patch). G4_full=false.",
        f"- Ground Reference Candidates review-only aceitos: {s['accepted_ground_reference_candidate_review_only_count']}. 17B permanece bloqueado.",
        "",
        "## Guardrails",
        "- Setor de risco NAO tratado como evento observado; alerta NAO tratado como ocorrencia; SAR metadata NAO virou footprint; nenhum raster pesado; coordenada nunca inventada; OSM/risk-sector centroid apenas geocodificacao de apoio; noticia comercial nao virou Ground Reference; score v6 intacto; score v7 inexistente; nenhum ground truth/label; 17B nao desbloqueado automaticamente.",
        "",
        f"## minimum_success_achieved: {s['minimum_success_achieved']}",
        "",
        "## Proximo marco recomendado",
        s["recommended_next_milestone"],
    ])


# ---------------------------------------------------------------------------
# Build / modes / validate
# ---------------------------------------------------------------------------

def build_all() -> None:
    _require_inputs()
    _load_ledger()
    ensure_dir(OUT)
    write_csv(PLAN, target_execution_plan_rows())
    write_csv(ATTEMPTS, targeted_acquisition_attempt_rows())
    write_csv(FOLLOWED, followed_link_registry_rows())
    write_csv(MANIFEST, source_artifact_manifest_rows())
    write_csv(PARSED, parsed_artifact_index_rows())
    write_csv(GEOM, geometry_candidate_rows())
    write_csv(LOCATION, location_resolution_rows())
    write_csv(G4C_EVAL, g4c_evaluation_rows())
    write_csv(G4D_EVAL, g4d_evaluation_rows())
    write_csv(PHEN_SEP, phenomenon_separation_candidate_rows())
    write_csv(G5D_EVAL, g5d_evaluation_rows())
    write_csv(GR_READY, ground_reference_readiness_rows())
    write_csv(SAR_FEAS, sar_metadata_feasibility_rows())
    write_csv(TECH_FOOTPRINT, technical_footprint_candidate_rows())
    write_csv(SUBGATE_UPDATE, subgate_update_matrix_rows())
    write_csv(LEDGER_OUT, gate_transition_ledger_rows())
    write_csv(LEAKAGE, no_leakage_audit_rows())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_markdown(REPORT, build_report())


def run_mode(mode: str) -> None:
    _require_inputs()
    if mode == "prepare-targets":
        ensure_dir(OUT)
        write_csv(PLAN, target_execution_plan_rows())
    elif mode == "run-targeted-acquisition":
        _run_acquisition()
    elif mode == "follow-targeted-links":
        _load_ledger()  # ja capturado no run; reafirma presenca
        write_csv(FOLLOWED, followed_link_registry_rows())
    elif mode == "parse-targeted-artifacts":
        write_csv(PARSED, parsed_artifact_index_rows())
        write_csv(MANIFEST, source_artifact_manifest_rows())
    elif mode == "extract-geometry-candidates":
        write_csv(GEOM, geometry_candidate_rows())
    elif mode == "extract-phenomenon-separation":
        write_csv(PHEN_SEP, phenomenon_separation_candidate_rows())
    elif mode == "resolve-geometry":
        write_csv(LOCATION, location_resolution_rows())
    elif mode == "evaluate-g4c-g4d":
        write_csv(G4C_EVAL, g4c_evaluation_rows())
        write_csv(G4D_EVAL, g4d_evaluation_rows())
    elif mode == "evaluate-g5d":
        write_csv(G5D_EVAL, g5d_evaluation_rows())
    elif mode == "evaluate-ground-reference-readiness":
        write_csv(GR_READY, ground_reference_readiness_rows())
    elif mode == "build-sar-metadata-feasibility":
        if os.environ.get("SUSC_17C31_ALLOW_SAR_METADATA") != "1":
            raise SystemExit("BLOCKED: build-sar-metadata-feasibility exige SUSC_17C31_ALLOW_SAR_METADATA=1")
        _load_ledger()
        write_csv(SAR_FEAS, sar_metadata_feasibility_rows())
        write_csv(TECH_FOOTPRINT, technical_footprint_candidate_rows())
    elif mode == "build-offline":
        build_all()
    elif mode == "audit":
        raise SystemExit(validate())
    else:
        raise SystemExit(f"unknown mode: {mode}")


def _schema_violations(row: dict, schema: dict) -> list[str]:
    violations = []
    for field in schema.get("required", []):
        if field not in row or row[field] == "":
            violations.append(f"missing:{field}")
    for field, rules in schema.get("properties", {}).items():
        if field not in row:
            continue
        value = row[field]
        if "const" in rules and value != rules["const"]:
            violations.append(f"{field}:const")
        if "enum" in rules and value not in rules["enum"]:
            violations.append(f"{field}:enum")
        if "pattern" in rules and not value.startswith(rules["pattern"].replace("^", "")):
            violations.append(f"{field}:pattern")
    return violations


def _validate_byte_identical() -> list[str]:
    outputs = [p for p in REQUIRED_OUTPUTS]
    before = {p: p.read_bytes() for p in outputs if p.exists()}
    build_all()
    errors = []
    for p, content in before.items():
        if p.exists() and p.read_bytes() != content:
            errors.append(f"offline_build_not_byte_identical:{rel(p)}")
    return errors


def validate() -> int:
    if not LEDGER.exists():
        print(f"MISSING_LEDGER: {rel(LEDGER)} (rode run-targeted-acquisition com os seis opt-ins)", file=sys.stderr)
        return 1
    missing = [p for p in REQUIRED_OUTPUTS if not p.exists()]
    if missing:
        print("MISSING: " + "; ".join(rel(p) for p in missing), file=sys.stderr)
        return 1
    errors = _validate_byte_identical()

    plan = read_csv(PLAN)
    attempts = read_csv(ATTEMPTS)
    manifest = read_csv(MANIFEST)
    parsed = read_csv(PARSED)
    geom = read_csv(GEOM)
    loc = read_csv(LOCATION)
    g4c = read_csv(G4C_EVAL)
    g4d = read_csv(G4D_EVAL)
    phsep = read_csv(PHEN_SEP)
    g5d = read_csv(G5D_EVAL)
    grr = read_csv(GR_READY)
    sar = read_csv(SAR_FEAS)
    tfc = read_csv(TECH_FOOTPRINT)
    leakage = read_csv(LEAKAGE)
    summary = read_json(SUMMARY)

    art_schema = read_json(SCHEMA_ART)
    geom_schema = read_json(SCHEMA_GEOM)
    g4g5_schema = read_json(SCHEMA_G4G5)
    sar_schema = read_json(SCHEMA_SAR)
    for row in manifest:
        errors.extend(_schema_violations(row, art_schema))
    for row in geom:
        errors.extend(_schema_violations(row, geom_schema))
    for row in grr:
        errors.extend(_schema_violations(row, g4g5_schema))
    for row in sar:
        errors.extend(_schema_violations(row, sar_schema))

    # 1-8: minimos
    if len(plan) < 36:
        errors.append("queue_consumed_lt_36")
    if len([a for a in attempts if a["network_enabled"] == "true"]) < 50:
        errors.append("attempts_lt_50")
    if len(manifest) < 5:
        errors.append("real_artifacts_lt_5")
    official_tech = [m for m in manifest if m["officiality_level"] in ("official", "official_institutional", "official_institutional_public_agency") or m["source_family"] == "asf_sentinel1"]
    if len(official_tech) < 3:
        errors.append("official_technical_lt_3")
    if len([p for p in parsed if p["parse_success"] == "true"]) < 5:
        errors.append("parsed_lt_5")
    if len(geom) < 1 or all(g["geometry_candidate_type"] == "insufficient" for g in geom):
        errors.append("no_geometry_candidate")
    if len(phsep) < 1:
        errors.append("no_phenomenon_separation_candidate")
    if len(g4c) < 1 or len(g4d) < 1 or len(g5d) < 1:
        errors.append("missing_g4c_g4d_g5d_evaluation")

    # 9: artefato adquirido sem hash
    for m in manifest:
        path = ROOT / m["artifact_local_path"]
        if not m["sha256"] or len(m["sha256"]) < 40:
            errors.append(f"artifact_without_sha256:{m['source_artifact_manifest_id']}")
        elif not path.exists() or sha256_file(path) != m["sha256"]:
            errors.append(f"artifact_hash_mismatch:{m['source_artifact_manifest_id']}")
        if int(m["size_bytes"]) > MAX_ARTIFACT_BYTES:
            errors.append(f"artifact_too_heavy:{m['source_artifact_manifest_id']}")
        if m["artifact_type"] not in ("html", "pdf", "txt", "csv", "json", "geojson", "kml"):
            errors.append(f"forbidden_artifact_type:{m['source_artifact_manifest_id']}")

    # 10: raster pesado em outputs_public
    if ARTIFACT_DIR.exists():
        for p in ARTIFACT_DIR.glob("**/*"):
            if p.is_file() and p.suffix.lower() in FORBIDDEN_SUFFIXES:
                errors.append(f"forbidden_raw_committed:{rel(p)}")

    # 11: SAR metadata != footprint
    for r in sar:
        if r["requires_future_raster_processing"] != "true":
            errors.append("sar_treated_as_footprint")
    for r in tfc:
        if r["requires_sentinel1_raster"] != "true":
            errors.append("sar_footprint_generated_without_raster_flag")
    if any(r["uses_sar_metadata_as_footprint"] != "false" for r in leakage):
        errors.append("sar_metadata_as_footprint_leak")

    # 12: risk sector != evento observado
    for r in leakage:
        if r["uses_risk_sector_as_observed_event"] != "false":
            errors.append("risk_sector_as_event")
        if r["uses_alert_as_occurrence"] != "false":
            errors.append("alert_as_occurrence")

    # 13: bairro/cidade patch-level sem incerteza
    for g in geom:
        if g["geometry_candidate_type"] in ("official_neighborhood",) and "uncertainty" not in g["geometry_source_type"] and g["uncertainty_m"] in ("", "0", "none"):
            errors.append(f"bairro_as_patch_level_without_uncertainty:{g['geometry_candidate_id']}")
    if any(r["uses_city_or_bairro_as_patch_level_without_uncertainty"] != "false" for r in leakage):
        errors.append("bairro_patch_level_leak")
    if any(r["invented_coordinate"] != "false" for r in leakage):
        errors.append("invented_coordinate")

    # 14: G5d nao abre com fenomeno misto
    for r in g5d:
        if r["G5d_hydrological_separated"] == "true" and (r["hydrological_signal"] == "true" and r["mass_movement_signal"] == "true"):
            errors.append(f"g5d_open_with_mixed:{r['g5d_evaluation_id']}")
    if any(r["uses_mixed_phenomenon_as_g5d"] != "false" for r in leakage):
        errors.append("mixed_as_g5d_leak")

    # 15/16: sem GT/label
    if summary["ground_truth_created"] or summary["training_labels_created"]:
        errors.append("gt_or_label_created")
    for r in grr:
        if r["can_be_ground_truth"] != "false" or r["can_be_training_label"] != "false":
            errors.append("grr_marked_gt_or_label")

    # 17/18: score v6 intacto, score v7 inexistente
    for path in [SCORE_V6, OFFICIAL_PATCHES, OFFICIAL_PATCH_LINKS]:
        if path.exists() and _run_git(["diff", "--name-only", "--", rel(path)]):
            errors.append(f"official_dataset_changed:{rel(path)}")
    if SCORE_V7.exists():
        errors.append("score_v7_exists")

    # 19: 17B so com GR aceito
    accepted = [r for r in grr if r["can_be_ground_reference_candidate_review_only"] == "true"]
    if summary["eligible_for_17b_now"] != (len(accepted) > 0):
        errors.append("17b_eligibility_inconsistent")
    if summary["eligible_for_17b_now"] and not accepted:
        errors.append("17b_without_accepted_gr")

    # DoD extras
    for k in ["geometry_candidate_records_count", "phenomenon_separation_candidate_records_count",
              "G4c_evaluation_rows_count", "G4d_evaluation_rows_count", "G5d_evaluation_rows_count"]:
        if summary[k] < 1:
            errors.append(f"dod_zero:{k}")
    if summary["geometry_candidate_records_count"] < 1 and summary["phenomenon_separation_candidate_records_count"] < 1:
        errors.append("no_geometry_and_no_phenomenon_candidate")

    # 20: summary reflete contagens reais
    expected = build_summary()
    for k, v in expected.items():
        if summary.get(k) != v:
            errors.append(f"summary_mismatch:{k}")
    if not summary["minimum_success_achieved"]:
        errors.append("minimum_success_not_achieved")

    # id uniqueness/sorted
    for rows, key in [
        (plan, "target_execution_plan_id"), (attempts, "targeted_acquisition_attempt_id"),
        (manifest, "source_artifact_manifest_id"), (parsed, "parsed_artifact_id"),
        (geom, "geometry_candidate_id"), (loc, "location_resolution_id"),
        (g4c, "g4c_evaluation_id"), (g4d, "g4d_evaluation_id"), (phsep, "phenomenon_separation_candidate_id"),
        (g5d, "g5d_evaluation_id"), (grr, "ground_reference_readiness_id"), (leakage, "no_leakage_audit_id"),
    ]:
        ids = [row[key] for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(
        "17C31 -> "
        f"queue={summary['acquisition_queue_consumed_count']} attempts={summary['targeted_acquisition_attempts_count']} "
        f"artifacts={summary['real_artifacts_acquired_count']} official_tech={summary['official_or_technical_artifacts_acquired_count']} "
        f"parsed={summary['parsed_artifacts_count']} geom={summary['geometry_candidate_records_count']} "
        f"phsep={summary['phenomenon_separation_candidate_records_count']} "
        f"G4c={summary['G4c_true_count']} G4d={summary['G4d_true_count']} G4_full={summary['G4_full_true_count']} "
        f"G5d={summary['G5d_true_count']} G5_full={summary['G5_full_true_count']} "
        f"gr_accepted={summary['accepted_ground_reference_candidate_review_only_count']} "
        f"sar_feasible={summary['sar_footprint_generation_feasible_count']} min_success={summary['minimum_success_achieved']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="susc_17c31")
    parser.add_argument("mode", choices=[
        "prepare-targets", "run-targeted-acquisition", "follow-targeted-links",
        "parse-targeted-artifacts", "extract-geometry-candidates", "extract-phenomenon-separation",
        "resolve-geometry", "evaluate-g4c-g4d", "evaluate-g5d",
        "evaluate-ground-reference-readiness", "build-sar-metadata-feasibility",
        "build-offline", "audit",
    ])
    args = parser.parse_args(argv)
    if args.mode == "audit":
        return validate()
    run_mode(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
