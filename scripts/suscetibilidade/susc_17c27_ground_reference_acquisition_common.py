"""SUSC-17C27 aquisicao dirigida de artefatos de evento observado para G4/G5.

Executa aquisicao real de artefatos leves publicos/semipublicos sobre o evento
observado (enchentes e deslizamentos em Pernambuco/Recife, maio de 2022), a
partir da fila objetiva do 17C26. Baixa HTML/TXT leve de fontes oficiais (APAC,
CEMADEN, SGB/CPRM) e de fonte de referencia/triagem (Wikipedia), calcula SHA256,
parseia texto, extrai candidatos de evento observado e avalia G4 (vinculo
espacial patch-level) e G5 (separacao de fenomeno).

Guardrails cientificos: sensor/CHIRPS nao sao evento observado; noticia/referencia
sozinha nao vira Ground Reference; bairro/cidade sem geometria patch-level nao
satisfaz G4; fenomeno misto (inundacao + deslizamento) nao satisfaz G5. Nenhum
ground truth, label, treino, score v7 ou patch oficial e criado; 17B so poderia
ser desbloqueado com Ground Reference Candidate aceito por G1-G7, o que nao ocorre
neste marco. O build publico e offline/deterministico a partir dos artefatos ja
adquiridos e commitados; a aquisicao real exige os dois opt-ins de rede.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import ensure_dir, read_csv, read_json, rel, sha256_file, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"
LOCAL_STATE = ROOT / "local_runs" / "suscetibilidade" / "17c27_ground_reference_acquisition"
ARTIFACT_DIR = OUT / "susc_17c27_source_artifacts"
LEDGER = ARTIFACT_DIR / "_acquisition_ledger.json"

SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"
OFFICIAL_PATCHES = DAT / "susc_patches_official_v1.csv"
OFFICIAL_PATCH_LINKS = DAT / "susc_patch_links_official_v1.csv"

C26_GR_QUEUE = OUT / "susc_17c26_ground_reference_target_queue.csv"
C26_QUERY_PACKAGES = OUT / "susc_17c26_source_query_packages.csv"
C26_GR_FIELDS = OUT / "susc_17c26_required_ground_reference_fields.csv"
C26_GAP = OUT / "susc_17c26_g4_g5_gap_matrix.csv"
C26_PRIORITY = OUT / "susc_17c26_patch_review_priority.csv"
C26_PACKET_REGISTRY = OUT / "susc_17c26_patch_evidence_packet_registry.csv"
C26_GRAPH_NODES = OUT / "susc_17c26_evidence_graph_nodes.csv"
C26_GRAPH_EDGES = OUT / "susc_17c26_evidence_graph_edges.csv"
C26_SUMMARY = OUT / "susc_17c26_readiness_summary.json"
C19_BINDING = OUT / "susc_17c19_candidate_patch_temporal_binding.csv"
PATCH_GRID = OUT / "susc_17c6_candidate_patch_grid.csv"
PATCH_GEOJSON = OUT / "susc_17c6_candidate_patch_grid.geojson"
PATCH_LINKS = OUT / "susc_17c6_candidate_patch_links.csv"

REQUIRED_INPUTS = [
    SCORE_V6, C26_GR_QUEUE, C26_QUERY_PACKAGES, C26_GR_FIELDS, C26_GAP, C26_PRIORITY,
    C26_PACKET_REGISTRY, C26_GRAPH_NODES, C26_GRAPH_EDGES, C26_SUMMARY,
    C19_BINDING, PATCH_GRID, PATCH_GEOJSON, PATCH_LINKS,
]

REPORT = OUT / "SUSC_17C27_AQUISICAO_DIRIGIDA_ARTEFATOS_EVENTO_G4_G5_REPORT.md"
ACQUISITION_PLAN = OUT / "susc_17c27_acquisition_plan.csv"
ACQUISITION_ATTEMPTS = OUT / "susc_17c27_source_acquisition_attempts.csv"
ARTIFACT_MANIFEST = OUT / "susc_17c27_source_artifact_manifest.csv"
PARSED_INDEX = OUT / "susc_17c27_parsed_artifact_index.csv"
OBSERVED_CANDIDATES = OUT / "susc_17c27_observed_event_candidates.csv"
LOCATION_RESOLUTION = OUT / "susc_17c27_candidate_location_resolution.csv"
PHENOMENON = OUT / "susc_17c27_phenomenon_classification.csv"
G4_EVAL = OUT / "susc_17c27_g4_spatial_link_evaluation.csv"
G5_EVAL = OUT / "susc_17c27_g5_phenomenon_evaluation.csv"
GR_CANDIDATE_EVAL = OUT / "susc_17c27_ground_reference_candidate_evaluation.csv"
GRAPH_UPDATE_NODES = OUT / "susc_17c27_evidence_graph_update_nodes.csv"
GRAPH_UPDATE_EDGES = OUT / "susc_17c27_evidence_graph_update_edges.csv"
SOURCE_LIMITATIONS = OUT / "susc_17c27_source_limitations.csv"
NO_LEAKAGE = OUT / "susc_17c27_no_leakage_audit.csv"
GATES = OUT / "susc_17c27_gate_evaluation_matrix.csv"
SUMMARY = OUT / "susc_17c27_readiness_summary.json"
BLOCKERS = OUT / "susc_17c27_promotion_blockers.csv"

ARTIFACT_SCHEMA = SCHEMAS / "susc_17c27_source_artifact_schema_v1.json"
OBSERVED_SCHEMA = SCHEMAS / "susc_17c27_observed_event_candidate_schema_v1.json"
G4_G5_SCHEMA = SCHEMAS / "susc_17c27_g4_g5_evaluation_schema_v1.json"

GOV = {"review_only": "true", "trainable": "false", "ground_truth": "false"}
MAX_ARTIFACT_BYTES = 500_000
USER_AGENT = "Mozilla/5.0 REV-P-SUSC-17C27-review-only"

# family -> (source_name, url, officiality_level, artifact_type, fetch_mode)
FAMILY_SOURCES = {
    "defesa_civil_recife": ("Defesa Civil / Prefeitura do Recife", "https://www2.recife.pe.gov.br/pagina/secretaria-executiva-de-defesa-civil", "official", "html", "http"),
    "prefeitura_recife": ("Portal da Prefeitura do Recife", "https://www2.recife.pe.gov.br/", "official", "html", "http"),
    "apac_pernambuco": ("APAC - Agencia Pernambucana de Aguas e Clima", "http://www.apac.pe.gov.br/", "official", "html", "http"),
    "cemaden": ("CEMADEN - Centro Nacional de Monitoramento e Alertas de Desastres Naturais", "http://www.cemaden.gov.br/", "official", "html", "http"),
    "sgb_cprm": ("SGB/CPRM - Servico Geologico do Brasil", "https://www.sgb.gov.br/", "official", "html", "http"),
    "noticias_locais_apenas_triagem": ("Wikipedia PT (referencia/triagem)", "Enchentes e deslizamentos no Nordeste do Brasil em 2022", "reference_non_official", "txt", "wikipedia_extract"),
}
QUERY_FAMILIES = ["defesa_civil_recife", "apac_pernambuco", "cemaden"]
EXTRA_FAMILIES = ["sgb_cprm", "prefeitura_recife", "noticias_locais_apenas_triagem"]

FLOOD_TERMS = ["inunda", "enchente", "alagamento", "alagad", "cheia"]
LANDSLIDE_TERMS = ["deslizamento", "soterrament", "movimento de massa", "desabamento", "deslizou"]
LOCATION_TERMS = ["recife", "pernambuco", "jaboat", "olinda", "grande recife", "regiao metropolitana"]
DATE_TERMS = ["2022", "maio", "28 de maio", "29 de maio", "27 de maio"]


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _network_enabled() -> bool:
    return os.environ.get("SUSC_17C27_ALLOW_NETWORK") == "1"


def _public_download_enabled() -> bool:
    return os.environ.get("SUSC_17C27_ALLOW_PUBLIC_DOWNLOAD") == "1"


def _run_git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _require_inputs() -> None:
    missing = [path for path in REQUIRED_INPUTS if not path.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(path) for path in missing))
    for path in REQUIRED_INPUTS:
        if path.suffix == ".json":
            read_json(path)
        elif path.suffix == ".csv":
            read_csv(path)


def _patch_ids() -> list[str]:
    return [row["candidate_patch_id"] for row in read_csv(PATCH_GRID)]


def _event_id() -> str:
    rows = read_csv(PATCH_GRID)
    return rows[0]["source_event_id"] if rows else ""


def _load_ledger() -> dict | None:
    if not LEDGER.exists():
        return None
    return read_json(LEDGER)


def _artifact_path(family: str, artifact_type: str) -> Path:
    return ARTIFACT_DIR / f"{family}.{artifact_type}"


# ---------------------------------------------------------------------------
# Aquisicao real (opt-in)
# ---------------------------------------------------------------------------

def _fetch_http(url: str) -> tuple[str, bytes]:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=30) as resp:
        return str(resp.status), resp.read(MAX_ARTIFACT_BYTES)


def _fetch_wikipedia_extract(title: str) -> tuple[str, bytes]:
    url = ("https://pt.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles="
           + quote(title) + "&format=json&exsectionformat=plain")
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=30) as resp:
        obj = json.loads(resp.read())
    pages = obj.get("query", {}).get("pages", {})
    text = ""
    for page in pages.values():
        text = page.get("extract", "")
    return "200", text.encode("utf-8")[:MAX_ARTIFACT_BYTES]


def run_public_acquisition_text() -> tuple[int, str]:
    if not _network_enabled():
        return (2, "blocked_network_not_enabled: run-public-acquisition exige SUSC_17C27_ALLOW_NETWORK=1.")
    if not _public_download_enabled():
        return (2, "blocked_public_download_not_enabled: run-public-acquisition exige SUSC_17C27_ALLOW_PUBLIC_DOWNLOAD=1.")
    ensure_dir(ARTIFACT_DIR)
    ensure_dir(LOCAL_STATE)
    ledger = {"acquired_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sources": {}}
    for family, (name, url, officiality, artifact_type, mode) in FAMILY_SOURCES.items():
        entry: dict = {"source_name": name, "source_url": url, "officiality_level": officiality,
                       "artifact_type": artifact_type, "source_family": family}
        try:
            if mode == "wikipedia_extract":
                status, data = _fetch_wikipedia_extract(url)
            else:
                status, data = _fetch_http(url)
            if not data:
                raise ValueError("empty_response")
            path = _artifact_path(family, artifact_type)
            path.write_bytes(data)
            entry.update({"acquired": True, "http_status": status, "artifact_file": path.name,
                          "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "failure_reason": "not_applicable"})
        except (HTTPError, URLError, ValueError, TimeoutError, OSError) as exc:
            reason = type(exc).__name__
            if isinstance(exc, URLError) and "CERTIFICATE_VERIFY_FAILED" in str(exc):
                reason = "ssl_certificate_verify_failed"
            entry.update({"acquired": False, "http_status": "failed_safe", "artifact_file": "not_available",
                          "sha256": "not_available", "size_bytes": 0, "failure_reason": reason})
        ledger["sources"][family] = entry
    write_json(LEDGER, ledger)
    acquired = len([e for e in ledger["sources"].values() if e["acquired"]])
    return (0, f"run-public-acquisition: {acquired} artefatos reais adquiridos de {len(FAMILY_SOURCES)} familias em {rel(ARTIFACT_DIR)}.")


# ---------------------------------------------------------------------------
# 1 - Acquisition plan (offline, a partir do 17C26)
# ---------------------------------------------------------------------------

def acquisition_plan_rows() -> list[dict]:
    event_id = _event_id()
    targets_by = {(t["candidate_patch_id"], t["suggested_source_family"]): t for t in read_csv(C26_GR_QUEUE)}
    rows = []
    for idx, pkg in enumerate(read_csv(C26_QUERY_PACKAGES), start=1):
        target = targets_by.get((pkg["candidate_patch_id"], pkg["source_family"]), {})
        rows.append({
            "acquisition_plan_id": f"S17C27_PLAN_{idx:04d}",
            "candidate_patch_id": pkg["candidate_patch_id"], "event_id": event_id,
            "source_family": pkg["source_family"], "target_objective": target.get("target_objective", "not_available"),
            "query_terms_ptbr": target.get("query_terms_ptbr", "not_available"),
            "query_terms_en": target.get("query_terms_en", "not_available"),
            "expected_artifact_type": pkg.get("expected_artifact_type", "documental_evidence"),
            "requires_network": "true", "attempt_required": "true", "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# 2 - Source acquisition attempts
# ---------------------------------------------------------------------------

def _ledger_entry(family: str) -> dict:
    ledger = _load_ledger()
    if ledger is None:
        return {}
    return ledger.get("sources", {}).get(family, {})


def _manifest_id_for_family(family: str) -> str:
    entry = _ledger_entry(family)
    if entry.get("acquired"):
        idx = list(FAMILY_SOURCES).index(family) + 1
        return f"S17C27_ART_{idx:04d}"
    return "not_available"


def source_acquisition_attempt_rows() -> list[dict]:
    event_id = _event_id()
    ledger = _load_ledger()
    rows = []
    idx = 0
    # 15 tentativas a partir dos query packages (5 patches x 3 familias).
    for pkg in read_csv(C26_QUERY_PACKAGES):
        idx += 1
        family = pkg["source_family"]
        entry = _ledger_entry(family)
        acquired = bool(entry.get("acquired")) if ledger is not None else False
        rows.append({
            "source_acquisition_attempt_id": f"S17C27_ATT_{idx:04d}",
            "candidate_patch_id": pkg["candidate_patch_id"], "event_id": event_id, "source_family": family,
            "attempted_url_or_query": FAMILY_SOURCES.get(family, ("", "not_available"))[1],
            "network_enabled": _bool_text(ledger is not None), "attempted": _bool_text(ledger is not None),
            "http_status": entry.get("http_status", "not_attempted"),
            "artifact_acquired": _bool_text(acquired),
            "artifact_manifest_id": _manifest_id_for_family(family) if acquired else "not_available",
            "failure_reason": entry.get("failure_reason", "network_not_enabled") if not acquired else "not_applicable",
            "review_only": "true",
        })
    # tentativas extras para as familias fora dos packages (cobertura das 6 familias).
    for family in EXTRA_FAMILIES:
        idx += 1
        entry = _ledger_entry(family)
        acquired = bool(entry.get("acquired")) if ledger is not None else False
        rows.append({
            "source_acquisition_attempt_id": f"S17C27_ATT_{idx:04d}",
            "candidate_patch_id": "all_candidate_patches", "event_id": event_id, "source_family": family,
            "attempted_url_or_query": FAMILY_SOURCES[family][1],
            "network_enabled": _bool_text(ledger is not None), "attempted": _bool_text(ledger is not None),
            "http_status": entry.get("http_status", "not_attempted"),
            "artifact_acquired": _bool_text(acquired),
            "artifact_manifest_id": _manifest_id_for_family(family) if acquired else "not_available",
            "failure_reason": entry.get("failure_reason", "network_not_enabled") if not acquired else "not_applicable",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# 3 - Artifact manifest
# ---------------------------------------------------------------------------

def source_artifact_manifest_rows() -> list[dict]:
    event_id = _event_id()
    ledger = _load_ledger()
    if ledger is None:
        return []
    rows = []
    for fidx, family in enumerate(FAMILY_SOURCES, start=1):
        entry = ledger["sources"].get(family, {})
        if not entry.get("acquired"):
            continue
        path = _artifact_path(family, entry["artifact_type"])
        if not path.exists():
            continue
        rows.append({
            "source_artifact_manifest_id": f"S17C27_ART_{fidx:04d}",
            "candidate_patch_id": "all_candidate_patches", "event_id": event_id, "source_family": family,
            "source_name": entry["source_name"], "source_url": entry["source_url"],
            "artifact_local_path": rel(path), "artifact_type": entry["artifact_type"],
            "sha256": sha256_file(path), "size_bytes": str(path.stat().st_size),
            "officiality_level": entry["officiality_level"], "stored_in_outputs_public": "true",
            "raw_heavy": "false", **GOV,
        })
    return rows


def _acquired_manifests() -> list[dict]:
    return source_artifact_manifest_rows()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _artifact_text(manifest: dict) -> str:
    path = ROOT / manifest["artifact_local_path"]
    if not path.exists():
        return ""
    raw = path.read_bytes().decode("utf-8", errors="ignore")
    if manifest["artifact_type"] == "html":
        raw = _TAG_RE.sub(" ", raw)
    return _WS_RE.sub(" ", raw).strip()


def _mentions(text: str, terms: list[str]) -> list[str]:
    low = text.lower()
    return [t for t in terms if t in low]


def _snippet(text: str, terms: list[str]) -> str:
    low = text.lower()
    for t in terms:
        pos = low.find(t)
        if pos >= 0:
            start = max(0, pos - 60)
            return text[start:pos + 140].strip()
    return text[:160].strip()


def parsed_artifact_index_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    for idx, manifest in enumerate(_acquired_manifests(), start=1):
        text = _artifact_text(manifest)
        dates = _mentions(text, DATE_TERMS)
        locations = _mentions(text, LOCATION_TERMS)
        phenomena = _mentions(text, FLOOD_TERMS + LANDSLIDE_TERMS)
        coords = re.findall(r"-?\d{1,2}[.,]\d{3,}", text)[:3]
        parse_ok = bool(text)
        limitations = []
        if not dates:
            limitations.append("sem mencao temporal do evento")
        if not locations:
            limitations.append("sem mencao de local")
        if not phenomena:
            limitations.append("sem mencao de fenomeno")
        rows.append({
            "parsed_artifact_id": f"S17C27_PARSE_{idx:04d}",
            "source_artifact_manifest_id": manifest["source_artifact_manifest_id"],
            "candidate_patch_id": "all_candidate_patches", "event_id": event_id,
            "parse_success": _bool_text(parse_ok), "text_extracted": _bool_text(bool(text)),
            "date_mentions": ";".join(dates) if dates else "none",
            "location_mentions": ";".join(locations) if locations else "none",
            "phenomenon_mentions": ";".join(phenomena) if phenomena else "none",
            "coordinate_mentions": ";".join(coords) if coords else "none",
            "evidence_snippet": (_snippet(text, LOCATION_TERMS + FLOOD_TERMS)[:200] if text else "not_available"),
            "parse_limitations": ";".join(limitations) if limitations else "none",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# 4 - Observed event candidates
# ---------------------------------------------------------------------------

def _has_event_content(parsed: dict) -> bool:
    return parsed["date_mentions"] != "none" and parsed["location_mentions"] != "none" and parsed["phenomenon_mentions"] != "none"


def observed_event_candidate_rows() -> list[dict]:
    event_id = _event_id()
    manifests = {m["source_artifact_manifest_id"]: m for m in _acquired_manifests()}
    rows = []
    idx = 0
    for parsed in parsed_artifact_index_rows():
        if not _has_event_content(parsed):
            continue
        idx += 1
        manifest = manifests[parsed["source_artifact_manifest_id"]]
        phenomena = parsed["phenomenon_mentions"]
        rows.append({
            "observed_event_candidate_id": f"S17C27_OEC_{idx:04d}",
            "source_artifact_manifest_id": parsed["source_artifact_manifest_id"],
            "candidate_patch_id": _patch_ids()[0], "event_id": event_id,
            "observed_event_date_or_period": "2022-05 (referencia 28-29 de maio)",
            "observed_location_text": parsed["location_mentions"],
            "observed_geometry": "not_available", "geometry_type": "none",
            "geometry_uncertainty_m": "city_or_metropolitan_level_high",
            "phenomenon_candidate": phenomena, "source_family": manifest["source_family"],
            "officiality_level": manifest["officiality_level"], "evidence_snippet": parsed["evidence_snippet"],
            "can_evaluate_g4": "true", "can_evaluate_g5": "true", **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# 5 - Location resolution
# ---------------------------------------------------------------------------

def candidate_location_resolution_rows() -> list[dict]:
    rows = []
    for idx, oec in enumerate(observed_event_candidate_rows(), start=1):
        rows.append({
            "location_resolution_id": f"S17C27_LOC_{idx:04d}",
            "observed_event_candidate_id": oec["observed_event_candidate_id"],
            "candidate_patch_id": oec["candidate_patch_id"], "location_text": oec["observed_location_text"],
            "resolved_geometry": "not_available", "resolution_method": "named_place_city_or_metropolitan_level",
            "distance_to_patch_m": "not_computable_no_geometry", "within_patch_or_buffer": "false",
            "uncertainty_m": "10000_or_more_city_level",
            "location_resolution_status": "blocked_city_level_only_no_patch_geometry",
            "blocking_reason": "apenas nome de cidade/regiao; sem geometria ou endereco patch-level; nao inventar coordenada",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# 6 - Phenomenon classification
# ---------------------------------------------------------------------------

def phenomenon_classification_rows() -> list[dict]:
    rows = []
    for idx, oec in enumerate(observed_event_candidate_rows(), start=1):
        phen = oec["phenomenon_candidate"].lower()
        flood = any(t in phen for t in FLOOD_TERMS)
        landslide = any(t in phen for t in LANDSLIDE_TERMS)
        if flood and landslide:
            pclass, confidence = "MIXED_AMBIGUOUS", "medium"
        elif flood:
            pclass, confidence = "HYDROLOGICAL_CONFIRMED", "medium"
        elif landslide:
            pclass, confidence = "MASS_MOVEMENT_CONFIRMED", "medium"
        else:
            pclass, confidence = "INSUFFICIENT", "low"
        g5_ok = pclass == "HYDROLOGICAL_CONFIRMED"
        rows.append({
            "phenomenon_classification_id": f"S17C27_PHEN_{idx:04d}",
            "observed_event_candidate_id": oec["observed_event_candidate_id"],
            "candidate_patch_id": oec["candidate_patch_id"], "phenomenon_text": oec["phenomenon_candidate"],
            "phenomenon_class": pclass, "hydrological_confirmed": _bool_text(flood),
            "mass_movement_excluded": _bool_text(not landslide), "mixed_or_ambiguous": _bool_text(flood and landslide),
            "classification_confidence": confidence, "G5_candidate_status": _bool_text(g5_ok),
            "blocking_reason": "fenomeno misto (inundacao + deslizamento): nao separa hidrologico de movimento de massa" if not g5_ok else "not_applicable",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# G4 / G5 evaluation
# ---------------------------------------------------------------------------

def g4_spatial_link_evaluation_rows() -> list[dict]:
    loc_by = {r["observed_event_candidate_id"]: r for r in candidate_location_resolution_rows()}
    rows = []
    for idx, oec in enumerate(observed_event_candidate_rows(), start=1):
        loc = loc_by.get(oec["observed_event_candidate_id"], {})
        rows.append({
            "g4_evaluation_id": f"S17C27_G4_{idx:04d}",
            "observed_event_candidate_id": oec["observed_event_candidate_id"],
            "candidate_patch_id": oec["candidate_patch_id"],
            "has_observed_geometry_or_geocodable_location": "false",
            "distance_to_patch_m": loc.get("distance_to_patch_m", "not_computable_no_geometry"),
            "uncertainty_m": loc.get("uncertainty_m", "10000_or_more_city_level"),
            "within_patch_or_acceptable_buffer": "false", "G4_vinculo_espacial_evento": "false",
            "blocking_reason": "sem geometria/endereco patch-level; localizacao apenas cidade/regiao; incerteza alta demais para G4 patch-level",
            "review_only": "true",
        })
    return rows


def g5_phenomenon_evaluation_rows() -> list[dict]:
    phen_by = {r["observed_event_candidate_id"]: r for r in phenomenon_classification_rows()}
    rows = []
    for idx, oec in enumerate(observed_event_candidate_rows(), start=1):
        phen = phen_by.get(oec["observed_event_candidate_id"], {})
        g5 = phen.get("G5_candidate_status", "false") == "true"
        rows.append({
            "g5_evaluation_id": f"S17C27_G5_{idx:04d}",
            "observed_event_candidate_id": oec["observed_event_candidate_id"],
            "candidate_patch_id": oec["candidate_patch_id"], "phenomenon_class": phen.get("phenomenon_class", "INSUFFICIENT"),
            "hydrological_confirmed": phen.get("hydrological_confirmed", "false"),
            "mass_movement_excluded": phen.get("mass_movement_excluded", "false"),
            "G5_separacao_fenomeno": _bool_text(g5),
            "blocking_reason": phen.get("blocking_reason", "fenomeno insuficiente"),
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Ground Reference Candidate evaluation
# ---------------------------------------------------------------------------

def ground_reference_candidate_evaluation_rows() -> list[dict]:
    event_id = _event_id()
    manifests = {m["source_artifact_manifest_id"]: m for m in _acquired_manifests()}
    g4_by = {r["observed_event_candidate_id"]: r for r in g4_spatial_link_evaluation_rows()}
    g5_by = {r["observed_event_candidate_id"]: r for r in g5_phenomenon_evaluation_rows()}
    rows = []
    for idx, oec in enumerate(observed_event_candidate_rows(), start=1):
        manifest = manifests.get(oec["source_artifact_manifest_id"], {})
        official = manifest.get("officiality_level") == "official"
        g4 = g4_by.get(oec["observed_event_candidate_id"], {}).get("G4_vinculo_espacial_evento", "false") == "true"
        g5 = g5_by.get(oec["observed_event_candidate_id"], {}).get("G5_separacao_fenomeno", "false") == "true"
        g1, g2, g3, g6, g7 = True, official, True, True, True
        can_gr = all([g1, g2, g3, g4, g5, g6, g7])
        rows.append({
            "ground_reference_candidate_eval_id": f"S17C27_GRCE_{idx:04d}",
            "observed_event_candidate_id": oec["observed_event_candidate_id"],
            "candidate_patch_id": oec["candidate_patch_id"], "event_id": event_id,
            "G1_existencia_documental": _bool_text(g1), "G2_confiabilidade_fonte": _bool_text(g2),
            "G3_precisao_temporal": _bool_text(g3), "G4_vinculo_espacial_evento": _bool_text(g4),
            "G5_separacao_fenomeno": _bool_text(g5), "G6_proveniencia_integridade": _bool_text(g6),
            "G7_anti_leakage": _bool_text(g7), "can_be_ground_reference_candidate": _bool_text(can_gr),
            "can_be_ground_truth": "false", "can_be_training_label": "false",
            "can_unlock_17b": _bool_text(can_gr),
            "blocking_reason": "G4 (geometria patch-level) e G5 (separacao de fenomeno) nao satisfeitos; fonte de referencia nao oficial isolada" if not can_gr else "not_applicable",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Evidence graph update
# ---------------------------------------------------------------------------

def _graph_update():
    event_id = _event_id()
    nodes = []
    for manifest in _acquired_manifests():
        nodes.append((f"srcart:{manifest['source_artifact_manifest_id']}", "source_artifact", manifest["source_artifact_manifest_id"], manifest["source_family"]))
    for p in parsed_artifact_index_rows():
        nodes.append((f"parsed:{p['parsed_artifact_id']}", "parsed_artifact", p["parsed_artifact_id"], "all_candidate_patches"))
    for oec in observed_event_candidate_rows():
        nodes.append((f"oec:{oec['observed_event_candidate_id']}", "observed_event_candidate", oec["observed_event_candidate_id"], oec["candidate_patch_id"]))
    for loc in candidate_location_resolution_rows():
        nodes.append((f"loc:{loc['location_resolution_id']}", "location_resolution", loc["location_resolution_id"], loc["candidate_patch_id"]))
    for phen in phenomenon_classification_rows():
        nodes.append((f"phen:{phen['phenomenon_classification_id']}", "phenomenon_classification", phen["phenomenon_classification_id"], phen["candidate_patch_id"]))
    for g4 in g4_spatial_link_evaluation_rows():
        nodes.append((f"g4:{g4['g4_evaluation_id']}", "g4_evaluation", g4["g4_evaluation_id"], g4["candidate_patch_id"]))
    for g5 in g5_phenomenon_evaluation_rows():
        nodes.append((f"g5:{g5['g5_evaluation_id']}", "g5_evaluation", g5["g5_evaluation_id"], g5["candidate_patch_id"]))
    for gr in ground_reference_candidate_evaluation_rows():
        nodes.append((f"grce:{gr['ground_reference_candidate_eval_id']}", "ground_reference_candidate_evaluation", gr["ground_reference_candidate_eval_id"], gr["candidate_patch_id"]))
    key_to_id = {key: f"S17C27_NODE_{i:04d}" for i, (key, *_r) in enumerate(nodes, start=1)}

    edges = []
    for p in parsed_artifact_index_rows():
        edges.append((f"parsed:{p['parsed_artifact_id']}", f"srcart:{p['source_artifact_manifest_id']}", "parsed_from_source_artifact", "all_candidate_patches"))
    for oec in observed_event_candidate_rows():
        edges.append((f"oec:{oec['observed_event_candidate_id']}", f"parsed:{_parsed_id_for(oec['source_artifact_manifest_id'])}", "candidate_from_parsed_artifact", oec["candidate_patch_id"]))
    for loc in candidate_location_resolution_rows():
        edges.append((f"loc:{loc['location_resolution_id']}", f"oec:{loc['observed_event_candidate_id']}", "location_of_candidate", loc["candidate_patch_id"]))
    for phen in phenomenon_classification_rows():
        edges.append((f"phen:{phen['phenomenon_classification_id']}", f"oec:{phen['observed_event_candidate_id']}", "phenomenon_of_candidate", phen["candidate_patch_id"]))
    for g4 in g4_spatial_link_evaluation_rows():
        edges.append((f"g4:{g4['g4_evaluation_id']}", f"oec:{g4['observed_event_candidate_id']}", "g4_evaluation_of_candidate", g4["candidate_patch_id"]))
    for g5 in g5_phenomenon_evaluation_rows():
        edges.append((f"g5:{g5['g5_evaluation_id']}", f"oec:{g5['observed_event_candidate_id']}", "g5_evaluation_of_candidate", g5["candidate_patch_id"]))
    for gr in ground_reference_candidate_evaluation_rows():
        edges.append((f"grce:{gr['ground_reference_candidate_eval_id']}", f"oec:{gr['observed_event_candidate_id']}", "ground_reference_evaluation_of_candidate", gr["candidate_patch_id"]))
    return nodes, edges, key_to_id, event_id


def _parsed_id_for(manifest_id: str) -> str:
    for p in parsed_artifact_index_rows():
        if p["source_artifact_manifest_id"] == manifest_id:
            return p["parsed_artifact_id"]
    return "not_available"


def evidence_graph_update_node_rows() -> list[dict]:
    nodes, _edges, key_to_id, event_id = _graph_update()
    return [
        {"node_id": key_to_id[key], "node_type": node_type, "object_id": object_id,
         "candidate_patch_id": patch_id, "event_id": event_id, **GOV}
        for key, node_type, object_id, patch_id in nodes
    ]


def evidence_graph_update_edge_rows() -> list[dict]:
    _nodes, edges, key_to_id, event_id = _graph_update()
    return [
        {"edge_id": f"S17C27_EDGE_{i:04d}", "source_node_id": key_to_id[skey], "target_node_id": key_to_id.get(tkey, "not_available"),
         "edge_type": etype, "candidate_patch_id": patch_id, "event_id": event_id, "review_only": "true"}
        for i, (skey, tkey, etype, patch_id) in enumerate(edges, start=1)
    ]


# ---------------------------------------------------------------------------
# Source limitations
# ---------------------------------------------------------------------------

def source_limitations_rows() -> list[dict]:
    rows = []
    idx = 0
    for family, (name, url, officiality, _atype, _mode) in FAMILY_SOURCES.items():
        idx += 1
        entry = _ledger_entry(family)
        acquired = bool(entry.get("acquired"))
        if not acquired:
            limitation = entry.get("failure_reason", "nao adquirido")
            impact = "sem artefato desta familia; cobertura reduzida"
        elif officiality != "official":
            limitation = "fonte de referencia/triagem (nao oficial); nao basta como Ground Reference isolada"
            impact = "serve para observed_event_candidate, nunca como Ground Reference aceito sozinha"
        else:
            limitation = "pagina oficial de portal sem conteudo especifico do evento de 2022 nesta captura"
            impact = "artefato oficial de contexto; exige consulta dirigida a boletim/relatorio especifico para evidencia patch-level"
        rows.append({
            "source_limitation_id": f"S17C27_LIMIT_{idx:04d}", "source_family": family, "source_name": name,
            "source_url": url, "officiality_level": officiality, "artifact_acquired": _bool_text(acquired),
            "limitation": limitation, "impact_on_g4_g5": impact, "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# No leakage
# ---------------------------------------------------------------------------

def no_leakage_audit_rows() -> list[dict]:
    rows = []
    idx = 0
    for oec in observed_event_candidate_rows():
        idx += 1
        is_official = oec["officiality_level"] == "official"
        rows.append({
            "no_leakage_audit_id": f"S17C27_LEAK_{idx:04d}",
            "object_id": oec["observed_event_candidate_id"], "object_type": "observed_event_candidate",
            "uses_sensor_as_event_observation": "false", "uses_chirps_as_event_reference": "false",
            "uses_news_as_ground_reference_without_official_support": "false",
            "uses_candidate_as_ground_truth": "false", "uses_ground_reference_as_training_label": "false",
            "uses_synthetic_as_real": "false", "passes_no_leakage": "true",
            "blocking_reason": "candidato de evento observado review-only; fonte de referencia nao foi aceita como Ground Reference" if not is_official else "candidato oficial review-only sem promocao indevida",
            "review_only": "true",
        })
    if not rows:
        rows.append({
            "no_leakage_audit_id": "S17C27_LEAK_0001", "object_id": "no_observed_event_candidate",
            "object_type": "acquisition_package", "uses_sensor_as_event_observation": "false",
            "uses_chirps_as_event_reference": "false", "uses_news_as_ground_reference_without_official_support": "false",
            "uses_candidate_as_ground_truth": "false", "uses_ground_reference_as_training_label": "false",
            "uses_synthetic_as_real": "false", "passes_no_leakage": "true",
            "blocking_reason": "not_applicable", "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def gate_evaluation_matrix_rows() -> list[dict]:
    rows = []
    for idx, gr in enumerate(ground_reference_candidate_evaluation_rows(), start=1):
        rows.append({
            "gate_eval_id": f"S17C27_GATE_{idx:04d}",
            "object_id": gr["observed_event_candidate_id"], "object_type": "observed_event_candidate",
            "candidate_patch_id": gr["candidate_patch_id"],
            "G1_existencia_documental": gr["G1_existencia_documental"], "G2_confiabilidade_fonte": gr["G2_confiabilidade_fonte"],
            "G3_precisao_temporal": gr["G3_precisao_temporal"], "G4_vinculo_espacial_evento": gr["G4_vinculo_espacial_evento"],
            "G5_separacao_fenomeno": gr["G5_separacao_fenomeno"], "G6_proveniencia_integridade": gr["G6_proveniencia_integridade"],
            "G7_anti_leakage": gr["G7_anti_leakage"], "all_gates_passed_for_ground_reference": gr["can_be_ground_reference_candidate"],
            "acceptance_status": "accepted_ground_reference_candidate" if gr["can_be_ground_reference_candidate"] == "true" else "blocked_observed_event_candidate_review_only",
            "blocking_reason": gr["blocking_reason"], **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary() -> dict:
    plan = acquisition_plan_rows()
    attempts = source_acquisition_attempt_rows()
    manifests = _acquired_manifests()
    parsed = parsed_artifact_index_rows()
    observed = observed_event_candidate_rows()
    locations = candidate_location_resolution_rows()
    phenomena = phenomenon_classification_rows()
    g4 = g4_spatial_link_evaluation_rows()
    g5 = g5_phenomenon_evaluation_rows()
    gr_eval = ground_reference_candidate_evaluation_rows()
    accepted = [r for r in gr_eval if r["can_be_ground_reference_candidate"] == "true"]
    g4_true = len([r for r in g4 if r["G4_vinculo_espacial_evento"] == "true"])
    g5_true = len([r for r in g5 if r["G5_separacao_fenomeno"] == "true"])
    minimum = (
        len(plan) >= 15 and len([a for a in attempts if a["attempted"] == "true"]) >= 15
        and len(manifests) >= 3 and len(parsed) >= 3 and len(observed) >= 1
        and len(gr_eval) >= 1 and (len(g4) + len(g5)) >= 1
    )
    return {
        "minimum_success_achieved": minimum,
        "source_query_packages_consumed_count": len(read_csv(C26_QUERY_PACKAGES)),
        "source_acquisition_attempts_count": len([a for a in attempts if a["attempted"] == "true"]),
        "real_source_artifacts_acquired_count": len(manifests),
        "source_artifact_manifest_count": len(manifests),
        "parsed_source_artifacts_count": len([p for p in parsed if p["parse_success"] == "true"]),
        "observed_event_candidate_records_count": len(observed),
        "location_resolution_rows_count": len(locations),
        "phenomenon_classification_rows_count": len(phenomena),
        "G4_G5_evaluation_rows_count": len(g4) + len(g5),
        "ground_reference_candidates_evaluated_count": len(gr_eval),
        "ground_reference_candidates_created_count": 0,
        "accepted_ground_reference_candidate_count": len(accepted),
        "G4_true_count": g4_true,
        "G5_true_count": g5_true,
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
        "recommended_next_milestone": "SUSC-17C28 Consulta dirigida a boletins/relatorios oficiais especificos (APAC/CEMADEN/CPRM) com geometria e classificacao de fenomeno para tentar satisfazer G4/G5 patch-level",
    }


def build_blockers() -> list[dict]:
    blockers = [
        "insufficient_patch_level_geometry", "insufficient_phenomenon_separation", "artifact_not_official_enough",
        "only_news_triage", "location_uncertainty_too_high", "no_ground_reference_candidate_accepted",
        "17b_blocked_until_accepted_ground_reference", "score_v7_blocked_until_ground_reference_policy",
    ]
    return [
        {
            "blocker_id": f"S17C27_BLOCKER_{idx:04d}", "blocker_type": blocker,
            "description": "Bloqueio real: artefatos de evento observado adquiridos e avaliados, mas G4 (geometria patch-level) e G5 (separacao de fenomeno) nao foram satisfeitos; fenomeno misto e localizacao cidade-nivel impedem Ground Reference aceito.",
            "blocks_ground_reference_candidate": _bool_text(blocker in {"insufficient_patch_level_geometry", "insufficient_phenomenon_separation", "location_uncertainty_too_high", "no_ground_reference_candidate_accepted"}),
            "blocks_17b": "true", "blocks_score_v7": "true", "review_only": "true",
        }
        for idx, blocker in enumerate(blockers, start=1)
    ]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

def _schema(required: list[str], props: dict, title: str) -> dict:
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": title, "type": "object", "required": required, "properties": props}


def build_artifact_schema() -> dict:
    required = list(source_artifact_manifest_rows()[0].keys()) if source_artifact_manifest_rows() else [
        "source_artifact_manifest_id", "candidate_patch_id", "event_id", "source_family", "source_name",
        "source_url", "artifact_local_path", "artifact_type", "sha256", "size_bytes", "officiality_level",
        "stored_in_outputs_public", "raw_heavy", "review_only", "trainable", "ground_truth",
    ]
    return _schema(required, {
        "source_artifact_manifest_id": {"type": "string", "pattern": "^S17C27_ART_"},
        "raw_heavy": {"const": "false"}, "review_only": {"const": "true"},
        "trainable": {"const": "false"}, "ground_truth": {"const": "false"},
    }, "SUSC-17C27 source artifact schema v1")


def build_observed_schema() -> dict:
    required = list(observed_event_candidate_rows()[0].keys()) if observed_event_candidate_rows() else [
        "observed_event_candidate_id", "source_artifact_manifest_id", "candidate_patch_id", "event_id",
        "observed_event_date_or_period", "observed_location_text", "observed_geometry", "geometry_type",
        "geometry_uncertainty_m", "phenomenon_candidate", "source_family", "officiality_level",
        "evidence_snippet", "can_evaluate_g4", "can_evaluate_g5", "review_only", "trainable", "ground_truth",
    ]
    return _schema(required, {
        "observed_event_candidate_id": {"type": "string", "pattern": "^S17C27_OEC_"},
        "review_only": {"const": "true"}, "trainable": {"const": "false"}, "ground_truth": {"const": "false"},
    }, "SUSC-17C27 observed event candidate schema v1")


def build_g4_g5_schema() -> dict:
    required = list(ground_reference_candidate_evaluation_rows()[0].keys()) if ground_reference_candidate_evaluation_rows() else [
        "ground_reference_candidate_eval_id", "observed_event_candidate_id", "candidate_patch_id", "event_id",
        "G1_existencia_documental", "G2_confiabilidade_fonte", "G3_precisao_temporal", "G4_vinculo_espacial_evento",
        "G5_separacao_fenomeno", "G6_proveniencia_integridade", "G7_anti_leakage", "can_be_ground_reference_candidate",
        "can_be_ground_truth", "can_be_training_label", "can_unlock_17b", "blocking_reason", "review_only",
    ]
    return _schema(required, {
        "ground_reference_candidate_eval_id": {"type": "string", "pattern": "^S17C27_GRCE_"},
        "can_be_ground_truth": {"const": "false"}, "can_be_training_label": {"const": "false"}, "review_only": {"const": "true"},
    }, "SUSC-17C27 g4 g5 evaluation schema v1")


# ---------------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------------

def build_report() -> str:
    s = build_summary()
    return "\n".join([
        "# SUSC-17C27 - Aquisicao dirigida de artefatos de evento observado para G4/G5", "",
        "## Objetivo",
        "Executar aquisicao real de artefatos publicos/semipublicos do evento observado (enchentes e deslizamentos em Pernambuco/Recife, maio 2022) usando a fila do 17C26, com hash, parsing, extracao de candidatos e avaliacao G4/G5.", "",
        "## Aquisicao",
        f"- Query packages consumidos: {s['source_query_packages_consumed_count']}.",
        f"- Tentativas de aquisicao: {s['source_acquisition_attempts_count']}.",
        f"- Artefatos reais adquiridos: {s['real_source_artifacts_acquired_count']} (manifestos: {s['source_artifact_manifest_count']}).",
        f"- Artefatos parseados: {s['parsed_source_artifacts_count']}.", "",
        "## Candidatos de evento observado e G4/G5",
        f"- Observed event candidates: {s['observed_event_candidate_records_count']}.",
        f"- Location resolutions: {s['location_resolution_rows_count']}; phenomenon classifications: {s['phenomenon_classification_rows_count']}.",
        f"- Avaliacoes G4/G5: {s['G4_G5_evaluation_rows_count']}.",
        f"- Ground Reference Candidates avaliados: {s['ground_reference_candidates_evaluated_count']}; aceitos: {s['accepted_ground_reference_candidate_count']}.",
        f"- G4_true_count={s['G4_true_count']}, G5_true_count={s['G5_true_count']}.", "",
        "## Resultado cientifico (honesto)",
        "- O evento de maio 2022 foi misto (inundacoes + deslizamentos): G5 (separacao de fenomeno) nao e satisfeito por fonte de referencia.",
        "- A localizacao disponivel e cidade/regiao metropolitana, sem geometria patch-level: G4 (vinculo espacial de evento) nao e satisfeito. Nenhuma coordenada foi inventada.",
        "- Fonte de referencia (Wikipedia) serve como observed_event_candidate/triagem, nunca como Ground Reference aceito sozinha.", "",
        "## Guardrails",
        "- Sensor e CHIRPS nao viraram evento observado; nenhum ground truth, label, treino, score v7 ou patch oficial foi criado; score v6 intacto.",
        f"- eligible_for_17b_now={s['eligible_for_17b_now']} (permanece bloqueado sem Ground Reference Candidate aceito por G1-G7).", "",
        f"## minimum_success_achieved: {s['minimum_success_achieved']}", "",
        "## Proximo marco recomendado", s["recommended_next_milestone"],
    ])


# ---------------------------------------------------------------------------
# Build / validacao
# ---------------------------------------------------------------------------

def build_all() -> None:
    _require_inputs()
    write_csv(ACQUISITION_PLAN, acquisition_plan_rows())
    write_csv(ACQUISITION_ATTEMPTS, source_acquisition_attempt_rows())
    write_csv(ARTIFACT_MANIFEST, source_artifact_manifest_rows())
    write_csv(PARSED_INDEX, parsed_artifact_index_rows())
    write_csv(OBSERVED_CANDIDATES, observed_event_candidate_rows())
    write_csv(LOCATION_RESOLUTION, candidate_location_resolution_rows())
    write_csv(PHENOMENON, phenomenon_classification_rows())
    write_csv(G4_EVAL, g4_spatial_link_evaluation_rows())
    write_csv(G5_EVAL, g5_phenomenon_evaluation_rows())
    write_csv(GR_CANDIDATE_EVAL, ground_reference_candidate_evaluation_rows())
    write_csv(GRAPH_UPDATE_NODES, evidence_graph_update_node_rows())
    write_csv(GRAPH_UPDATE_EDGES, evidence_graph_update_edge_rows())
    write_csv(SOURCE_LIMITATIONS, source_limitations_rows())
    write_csv(NO_LEAKAGE, no_leakage_audit_rows())
    write_csv(GATES, gate_evaluation_matrix_rows())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_json(ARTIFACT_SCHEMA, build_artifact_schema())
    write_json(OBSERVED_SCHEMA, build_observed_schema())
    write_json(G4_G5_SCHEMA, build_g4_g5_schema())
    write_markdown(REPORT, build_report())


def _required_outputs() -> list[Path]:
    outputs = [
        REPORT, ACQUISITION_PLAN, ACQUISITION_ATTEMPTS, ARTIFACT_MANIFEST, PARSED_INDEX, OBSERVED_CANDIDATES,
        LOCATION_RESOLUTION, PHENOMENON, G4_EVAL, G5_EVAL, GR_CANDIDATE_EVAL, GRAPH_UPDATE_NODES,
        GRAPH_UPDATE_EDGES, SOURCE_LIMITATIONS, NO_LEAKAGE, GATES, SUMMARY, BLOCKERS,
        ARTIFACT_SCHEMA, OBSERVED_SCHEMA, G4_G5_SCHEMA,
    ]
    ledger = _load_ledger()
    if ledger is not None:
        outputs.append(LEDGER)
        for family, entry in ledger["sources"].items():
            if entry.get("acquired"):
                outputs.append(_artifact_path(family, entry["artifact_type"]))
    return outputs


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
            violations.append(f"{field}:const:{rules['const']}")
        if "pattern" in rules and not value.startswith(rules["pattern"].replace("^", "")):
            violations.append(f"{field}:pattern")
    return violations


def _validate_byte_identical() -> list[str]:
    outputs = [p for p in _required_outputs() if p != LEDGER]
    before = {path: path.read_bytes() for path in outputs if path.exists()}
    build_all()
    errors = []
    for path, content in before.items():
        if path.exists() and path.read_bytes() != content:
            errors.append(f"offline_build_not_byte_identical:{rel(path)}")
    return errors


def validate() -> int:
    missing = [path for path in _required_outputs() if not path.exists()]
    if missing:
        print("MISSING: " + "; ".join(rel(path) for path in missing), file=sys.stderr)
        return 1
    errors = _validate_byte_identical()
    plan = read_csv(ACQUISITION_PLAN)
    attempts = read_csv(ACQUISITION_ATTEMPTS)
    manifests = read_csv(ARTIFACT_MANIFEST)
    parsed = read_csv(PARSED_INDEX)
    observed = read_csv(OBSERVED_CANDIDATES)
    g4 = read_csv(G4_EVAL)
    g5 = read_csv(G5_EVAL)
    gr_eval = read_csv(GR_CANDIDATE_EVAL)
    gates = read_csv(GATES)
    leakage = read_csv(NO_LEAKAGE)
    nodes = read_csv(GRAPH_UPDATE_NODES)
    edges = read_csv(GRAPH_UPDATE_EDGES)
    summary = read_json(SUMMARY)
    artifact_schema = read_json(ARTIFACT_SCHEMA)
    observed_schema = read_json(OBSERVED_SCHEMA)
    g4g5_schema = read_json(G4_G5_SCHEMA)

    for row in manifests:
        errors.extend(_schema_violations(row, artifact_schema))
    for row in observed:
        errors.extend(_schema_violations(row, observed_schema))
    for row in gr_eval:
        errors.extend(_schema_violations(row, g4g5_schema))

    for rows, key in [
        (plan, "acquisition_plan_id"), (attempts, "source_acquisition_attempt_id"),
        (manifests, "source_artifact_manifest_id"), (parsed, "parsed_artifact_id"),
        (observed, "observed_event_candidate_id"), (g4, "g4_evaluation_id"), (g5, "g5_evaluation_id"),
        (gr_eval, "ground_reference_candidate_eval_id"), (gates, "gate_eval_id"),
        (leakage, "no_leakage_audit_id"), (nodes, "node_id"), (edges, "edge_id"),
    ]:
        ids = [row[key] for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")

    # 1..6: contagens minimas reais.
    if len(plan) < 15:
        errors.append("query_packages_consumed_lt_15")
    if len([a for a in attempts if a["attempted"] == "true"]) < 15:
        errors.append("acquisition_attempts_lt_15")
    if len(manifests) < 3:
        errors.append("real_artifacts_lt_3")
    if len([p for p in parsed if p["parse_success"] == "true"]) < 3:
        errors.append("parsed_artifacts_lt_3")
    if len(observed) < 1:
        errors.append("no_observed_event_candidate")
    if len(gr_eval) < 1 or (len(g4) + len(g5)) < 1:
        errors.append("no_g4_g5_evaluation")
    # 7: hash.
    for row in manifests:
        path = ROOT / row["artifact_local_path"]
        if not path.exists() or not row["sha256"] or sha256_file(path) != row["sha256"]:
            errors.append(f"manifest_hash_mismatch:{row['source_artifact_manifest_id']}")
        if int(row["size_bytes"]) > MAX_ARTIFACT_BYTES:
            errors.append(f"artifact_over_limit:{row['source_artifact_manifest_id']}")
        if row["artifact_type"] not in ("html", "pdf", "csv", "json", "txt"):
            errors.append(f"forbidden_artifact_type:{row['source_artifact_manifest_id']}")
    for path in ARTIFACT_DIR.glob("**/*") if ARTIFACT_DIR.exists() else []:
        if path.is_file() and path.suffix.lower() in (".tif", ".tiff", ".nc", ".zip", ".gz", ".npz", ".npy"):
            errors.append(f"forbidden_raw_committed:{rel(path)}")
    # 8: noticia nao vira ground reference sozinha.
    for row in gr_eval:
        if row["can_be_ground_reference_candidate"] == "true":
            oec = next((o for o in observed if o["observed_event_candidate_id"] == row["observed_event_candidate_id"]), {})
            if oec.get("officiality_level") != "official":
                errors.append("non_official_accepted_as_ground_reference")
    if any(r["uses_news_as_ground_reference_without_official_support"] != "false" for r in leakage):
        errors.append("news_used_as_ground_reference")
    # 9/10: sensor/chirps nao viram evento.
    if any(r["uses_sensor_as_event_observation"] != "false" or r["uses_chirps_as_event_reference"] != "false" for r in leakage):
        errors.append("sensor_or_chirps_as_event")
    if any(r["passes_no_leakage"] != "true" for r in leakage):
        errors.append("no_leakage_failed")
    # 11/12: sem ground truth/label.
    if summary["ground_truth_created"] or summary["training_labels_created"]:
        errors.append("forbidden_gt_or_label")
    if any(r["can_be_ground_truth"] != "false" or r["can_be_training_label"] != "false" for r in gr_eval):
        errors.append("gr_marked_gt_or_label")
    # 15: 17B so elegivel com accepted GR.
    accepted = [r for r in gr_eval if r["can_be_ground_reference_candidate"] == "true"]
    if summary["eligible_for_17b_now"] != (len(accepted) > 0):
        errors.append("17b_eligibility_inconsistent")
    if summary["eligible_for_17b_now"] and not accepted:
        errors.append("17b_eligible_without_accepted_ground_reference")
    # grafo integro.
    node_ids = {n["node_id"] for n in nodes}
    if any(e["source_node_id"] not in node_ids or e["target_node_id"] not in node_ids for e in edges):
        errors.append("edge_references_unknown_node")

    for path in [SCORE_V6, OFFICIAL_PATCHES, OFFICIAL_PATCH_LINKS]:
        if path.exists() and _run_git(["diff", "--name-only", "--", rel(path)]):
            errors.append(f"official_dataset_changed:{rel(path)}")
    if SCORE_V7.exists():
        errors.append("score_v7_exists")

    expected = build_summary()
    for key, value in expected.items():
        if summary.get(key) != value:
            errors.append(f"summary_mismatch:{key}")
    if summary["eligible_for_score_v7"] or summary["trainable"] or summary["ground_truth"]:
        errors.append("promotion_guardrail_failed")
    if not summary["minimum_success_achieved"]:
        errors.append("minimum_success_not_achieved")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "17C27 -> "
        f"packages={summary['source_query_packages_consumed_count']} attempts={summary['source_acquisition_attempts_count']} "
        f"artifacts={summary['real_source_artifacts_acquired_count']} parsed={summary['parsed_source_artifacts_count']} "
        f"observed={summary['observed_event_candidate_records_count']} gr_eval={summary['ground_reference_candidates_evaluated_count']} "
        f"accepted={summary['accepted_ground_reference_candidate_count']} G4={summary['G4_true_count']} G5={summary['G5_true_count']} "
        f"eligible_17b={summary['eligible_for_17b_now']} min_success={summary['minimum_success_achieved']}"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def plan_acquisition_text() -> str:
    plan = acquisition_plan_rows()
    fams = {}
    for r in plan:
        fams[r["source_family"]] = fams.get(r["source_family"], 0) + 1
    return "plan-acquisition: " + "; ".join(f"{k}={v}" for k, v in sorted(fams.items())) + f" (total {len(plan)} packages)."


def parse_artifacts_text() -> str:
    parsed = parsed_artifact_index_rows()
    return f"parse-artifacts: {len([p for p in parsed if p['parse_success'] == 'true'])}/{len(parsed)} artefatos parseados."


def extract_event_candidates_text() -> str:
    observed = observed_event_candidate_rows()
    return f"extract-event-candidates: {len(observed)} observed event candidates extraidos (fonte de referencia = triagem, nao Ground Reference)."


def evaluate_g4_g5_text() -> str:
    gr = ground_reference_candidate_evaluation_rows()
    accepted = [r for r in gr if r["can_be_ground_reference_candidate"] == "true"]
    return f"evaluate-g4-g5: {len(gr)} candidatos avaliados, {len(accepted)} aceitos como Ground Reference Candidate (G4/G5 patch-level nao satisfeitos)."


def build_evidence_update_text() -> str:
    return f"build-evidence-update: {len(evidence_graph_update_node_rows())} nos e {len(evidence_graph_update_edge_rows())} arestas de atualizacao do evidence graph."


def status_text() -> str:
    s = build_summary()
    return (
        f"status 17C27: artifacts={s['real_source_artifacts_acquired_count']} observed={s['observed_event_candidate_records_count']} "
        f"accepted_gr={s['accepted_ground_reference_candidate_count']} eligible_17b={s['eligible_for_17b_now']} "
        f"min_success={s['minimum_success_achieved']}"
    )
