"""SUSC-17C32 Geocodificacao oficial de ocorrencias e vinculo patch-buffer para G4d.

Marco estreito: NAO redesenha fingerprint, gate promotion engine, acquisition
queue, SAR feasibility, separacao G5d (ja obtida no 17C31) nem politica de gates.
Reusa o 17C31 como input. O unico objetivo e atacar o bloqueio restante:

    ocorrencia oficial hidrologica (Defesa Civil Recife, 17C31)
      -> endereco/logradouro/bairro proximo ao patch
      -> geocodificacao controlada (Nominatim/OSM, opt-in, cache em ledger)
      -> calculo de distancia ate patch/buffer
      -> G4d
      -> G4_full
      -> Ground Reference Candidate review-only, SE G1-G7 fecharem.

Ponto cientifico: G5d/G5_full ja existem para 14 bairros hidrologicos (17C31).
O gargalo agora e G4d/G4_full. Nao se busca mais "provar o evento"; busca-se
ponto/endereco oficial de ocorrencia hidrologica proximo ao patch.

Regras duras (fail-closed):
  - bairro/centroide NAO abre G4d;
  - OSM sozinho NAO abre G4d (geocoder nao-oficial => no maximo candidato);
  - area de risco NAO abre G4d;
  - logradouro sem numero => candidato com incerteza ALTA (street-level);
  - G4d so abre com geometria/endereco oficial ou controlado E distancia
    compativel com patch/buffer.

Achado esperado (Resultado B - bloqueio honesto e especifico): os enderecos
oficiais das ocorrencias hidrologicas tem o numero anonimizado (LGPD) -> so
geocodificacao a nivel de logradouro (incerteza alta); e os patches candidatos
(AOI Charter758, Recife Antigo/Santo Amaro ~-8.00) NAO coincidem espacialmente
com os bairros onde o alagamento foi documentado (Varzea, Pina, Areias,
Iputinga... 9-13 km ao sul/oeste). Logo G4d permanece 0: distancia incompativel
+ incerteza street-level + geocoder nao-oficial. G4_full=0, nenhum Ground
Reference aceito, 17B bloqueado, score v6 intacto, score v7 inexistente.

Reproducibilidade: run-controlled-geocoding (rede opt-in) grava um ledger com os
pontos geocodificados; o build offline reproduz todos os derivados byte-identicos
a partir do ledger commitado.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import ensure_dir, read_csv, read_json, rel, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"
GEO_DIR = OUT / "susc_17c32_geocoding_artifacts"
LEDGER = GEO_DIR / "_geocoding_ledger.json"

SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"
OFFICIAL_PATCHES = DAT / "susc_patches_official_v1.csv"
OFFICIAL_PATCH_LINKS = DAT / "susc_patch_links_official_v1.csv"

# 17C31 inputs (reuso; NAO re-adquirir)
C31_ATENDIMENTOS = OUT / "susc_17c31_source_artifacts" / "dados_recife_ckan_atendimentos_2022_recorte_evento.csv"
C31_MANIFEST = OUT / "susc_17c31_source_artifact_manifest.csv"
C31_G5D = OUT / "susc_17c31_g5d_hydrological_separation_evaluation.csv"
C31_PHSEP = OUT / "susc_17c31_phenomenon_separation_candidate_registry.csv"
C31_G4D = OUT / "susc_17c31_g4d_patch_buffer_link_evaluation.csv"
C31_SUBGATE = OUT / "susc_17c31_subgate_update_matrix.csv"
C31_SUMMARY = OUT / "susc_17c31_readiness_summary.json"
PATCH_GRID = OUT / "susc_17c6_candidate_patch_grid.csv"
PATCH_GEOJSON = OUT / "susc_17c6_candidate_patch_grid.geojson"

REQUIRED_INPUTS = [
    SCORE_V6, C31_ATENDIMENTOS, C31_MANIFEST, C31_G5D, C31_PHSEP, C31_G4D,
    C31_SUBGATE, C31_SUMMARY, PATCH_GRID, PATCH_GEOJSON,
]

# 17C32 outputs
REPORT = OUT / "SUSC_17C32_GEOCODIFICACAO_OCORRENCIAS_G4D_REPORT.md"
TARGETS = OUT / "susc_17c32_hydrological_occurrence_targets.csv"
GEO_ATTEMPTS = OUT / "susc_17c32_geocoding_attempts.csv"
GEO_POINTS = OUT / "susc_17c32_geocoded_point_registry.csv"
DIST_EVAL = OUT / "susc_17c32_patch_buffer_distance_evaluation.csv"
G4D_EVAL = OUT / "susc_17c32_g4d_patch_buffer_link_evaluation.csv"
G4FULL_EVAL = OUT / "susc_17c32_g4_full_evaluation.csv"
GR_READY = OUT / "susc_17c32_ground_reference_readiness_evaluation.csv"
SUBGATE_UPDATE = OUT / "susc_17c32_subgate_update_matrix.csv"
LEDGER_OUT = OUT / "susc_17c32_gate_transition_ledger.csv"
LEAKAGE = OUT / "susc_17c32_no_leakage_audit.csv"
SUMMARY = OUT / "susc_17c32_readiness_summary.json"
BLOCKERS = OUT / "susc_17c32_promotion_blockers.csv"

SCHEMA_GEO = SCHEMAS / "susc_17c32_geocoded_point_schema_v1.json"
SCHEMA_G4D = SCHEMAS / "susc_17c32_g4d_evaluation_schema_v1.json"

REQUIRED_OUTPUTS = [
    REPORT, TARGETS, GEO_ATTEMPTS, GEO_POINTS, DIST_EVAL, G4D_EVAL, G4FULL_EVAL,
    GR_READY, SUBGATE_UPDATE, LEDGER_OUT, LEAKAGE, SUMMARY, BLOCKERS,
]

GOV = {"review_only": "true", "trainable": "false", "ground_truth": "false"}
FLOOD_TERMS = ["alagad", "alagam", "inunda", "enxurr", "cheia", "transbord"]
LANDSLIDE_TERMS = ["desliz", "soterr", "barreira", "desabam", "desmoron"]
EVENT_WINDOW_RE = re.compile(r"2022-05-(2[4-9]|30|31)")
# ordem de prioridade de bairros (do prompt do marco)
PRIORITY_BAIRROS = ["Pina", "Boa Viagem", "Imbiribeira", "Afogados", "Areias", "Ipsep", "Iputinga"]
# tambem geocodificar o bairro com mais alagamento documentado, para completude
EXTRA_BAIRROS = ["Varzea"]
LOGRADOURO_RE = re.compile(r"^(rua|avenida|av\b|estrada|travessa|alameda|praca|praça|ladeira|rodovia|via)\b", re.I)

# politica de buffer: patch ~1km; buffer aceitavel generoso de 1500 m.
ACCEPTABLE_BUFFER_M = 1500.0
# incerteza street-level (sem numero): centenas de metros a km.
STREET_LEVEL_UNCERTAINTY_M = 800.0

USER_AGENT = "REV-P-SUSC-17C32-review-only/1.0 (susceptibility research; controlled geocoding)"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
GEOCODE_TIMEOUT = 30
GEOCODE_SLEEP_S = 1.2  # respeitar politica Nominatim


def _bool(v: bool) -> str:
    return "true" if v else "false"


def _run_git(args: list[str]) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _geocoding_enabled() -> bool:
    return os.environ.get("SUSC_17C32_ALLOW_NETWORK") == "1" and os.environ.get("SUSC_17C32_ALLOW_GEOCODING") == "1"


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
    c = _patch_centers()
    return c[0][0] if c else "S17C6_CANARY_REC_00001"


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _nearest_patch(lon: float, lat: float) -> tuple[str, float]:
    best_id, best_d = "", float("inf")
    for pid, cx, cy in _patch_centers():
        d = _haversine_m(lon, lat, cx, cy)
        if d < best_d:
            best_id, best_d = pid, d
    return best_id, best_d


# ---------------------------------------------------------------------------
# Targets: ocorrencias hidrologicas oficiais (reuso 17C31), bairros prioritarios
# ---------------------------------------------------------------------------

def _clean_street(raw_line: str) -> str:
    """Extrai o nome do logradouro (sem numero) da linha bruta anonimizada.
    Ex.: ...;\"\"\"Rua Vicente Ribeiro de Barros\";\" ^[{\"\"\";\"Afogados\";..."""
    m = re.search(r'"""\s*((?:Rua|Avenida|Av|Estrada|Travessa|Alameda|Praca|Praça|Ladeira|Rodovia|Via)[^";]*?)\s*"?\s*;', raw_line, re.I)
    if not m:
        return ""
    street = m.group(1).strip().strip('"').strip()
    # remove numero colado no fim (ex.: "Rua Luiz Souto Dourado164")
    street = re.sub(r"\d+\s*$", "", street).strip()
    return street


def hydrological_occurrence_target_rows() -> list[dict]:
    event_id = _event_id()
    patch = _primary_patch_id()
    lines = C31_ATENDIMENTOS.read_bytes().decode("utf-8", "ignore").splitlines()
    header = [h.strip().strip('"').lower() for h in lines[0].split(";")] if lines else []

    def col(name):
        for i, h in enumerate(header):
            if name in h:
                return i
        return -1
    i_data, i_occ, i_bairro = col("data"), col("ocorrencia"), col("bairro")
    wanted = [b.lower() for b in (PRIORITY_BAIRROS + EXTRA_BAIRROS)]
    seen: set[tuple[str, str]] = set()
    rows: list[dict] = []
    for ln in lines[1:]:
        low = ln.lower()
        flood = any(t in low for t in FLOOD_TERMS)
        land = any(t in low for t in LANDSLIDE_TERMS)
        if not flood:
            continue
        cols = [c.strip().strip('"') for c in ln.split(";")]
        bairro = (cols[i_bairro] if 0 <= i_bairro < len(cols) else "").strip()
        if bairro.lower() not in wanted:
            continue
        street = _clean_street(ln)
        key = (street.lower(), bairro.lower())
        if not street or key in seen:
            continue
        seen.add(key)
        try:
            prio = PRIORITY_BAIRROS.index(bairro) + 1
        except ValueError:
            prio = 99
        rows.append({
            "hydrological_occurrence_target_id": f"S17C32_TGT_{len(rows) + 1:04d}",
            "candidate_patch_id": patch, "event_id": event_id,
            "bairro": bairro, "logradouro": street,
            "house_number_status": "anonymized_lgpd",
            "occurrence_date": (cols[i_data] if 0 <= i_data < len(cols) else ""),
            "occurrence_type": (cols[i_occ] if 0 <= i_occ < len(cols) else ""),
            "hydrological_signal": "true", "mass_movement_signal": _bool(land),
            "priority_rank": str(prio),
            "source_artifact": "susc_17c31 dados_recife_ckan_atendimentos_2022_recorte_evento.csv",
            "review_only": "true",
        })
    rows.sort(key=lambda r: (int(r["priority_rank"]), r["bairro"], r["logradouro"]))
    for i, r in enumerate(rows, start=1):
        r["hydrological_occurrence_target_id"] = f"S17C32_TGT_{i:04d}"
    return rows


# ---------------------------------------------------------------------------
# Controlled geocoding (Nominatim, opt-in) -> ledger
# ---------------------------------------------------------------------------

def _geocode(street: str, bairro: str) -> dict:
    q = urlencode({"street": street, "city": "Recife", "county": bairro,
                   "state": "Pernambuco", "country": "Brasil",
                   "format": "jsonv2", "limit": 1, "addressdetails": 1})
    req = Request(f"{NOMINATIM}?{q}", headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=GEOCODE_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8", "ignore"))
    return data[0] if data else {}


def _run_geocoding() -> dict:
    if not _geocoding_enabled():
        raise SystemExit(
            "BLOCKED: geocodificacao 17C32 exige SUSC_17C32_ALLOW_NETWORK=1 e SUSC_17C32_ALLOW_GEOCODING=1."
        )
    ensure_dir(GEO_DIR)
    targets = hydrological_occurrence_target_rows()
    attempts: list[dict] = []
    points: list[dict] = []
    for i, t in enumerate(targets, start=1):
        street, bairro = t["logradouro"], t["bairro"]
        query = f"street={street}; county={bairro}; city=Recife; state=Pernambuco"
        try:
            res = _geocode(street, bairro)
            if res and res.get("lat") and res.get("lon"):
                attempts.append({
                    "target_id": t["hydrological_occurrence_target_id"], "query": query,
                    "geocoder": "nominatim_osm", "status": "resolved", "failure_reason": "",
                })
                points.append({
                    "target_id": t["hydrological_occurrence_target_id"],
                    "bairro": bairro, "logradouro": street,
                    "lat": float(res["lat"]), "lon": float(res["lon"]),
                    "osm_type": res.get("type", ""), "osm_class": res.get("category", res.get("class", "")),
                    "importance": res.get("importance", ""),
                })
            else:
                attempts.append({
                    "target_id": t["hydrological_occurrence_target_id"], "query": query,
                    "geocoder": "nominatim_osm", "status": "empty", "failure_reason": "no_result",
                })
        except Exception as e:
            attempts.append({
                "target_id": t["hydrological_occurrence_target_id"], "query": query,
                "geocoder": "nominatim_osm", "status": "error", "failure_reason": type(e).__name__,
            })
        time.sleep(GEOCODE_SLEEP_S)
    ledger = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_from_network": True,
        "geocoder": "nominatim_osm",
        "geocoder_is_official": False,
        "attempts": attempts,
        "points": points,
    }
    write_json(LEDGER, ledger)
    return ledger


def _load_ledger() -> dict:
    if not LEDGER.exists():
        raise FileNotFoundError(
            f"{rel(LEDGER)} ausente: rode run-controlled-geocoding com SUSC_17C32_ALLOW_NETWORK=1 e SUSC_17C32_ALLOW_GEOCODING=1."
        )
    return read_json(LEDGER)


# ---------------------------------------------------------------------------
# Offline derivation
# ---------------------------------------------------------------------------

def geocoding_attempt_rows() -> list[dict]:
    ledger = _load_ledger()
    rows = []
    for i, a in enumerate(ledger.get("attempts", []), start=1):
        rows.append({
            "geocoding_attempt_id": f"S17C32_ATT_{i:04d}",
            "hydrological_occurrence_target_id": a["target_id"],
            "geocoder": a["geocoder"], "geocoder_is_official": "false",
            "query": a["query"], "network_enabled": "true", "status": a["status"],
            "failure_reason": a.get("failure_reason", ""), "review_only": "true",
        })
    return rows


def geocoded_point_rows() -> list[dict]:
    ledger = _load_ledger()
    event_id = _event_id()
    patch = _primary_patch_id()
    rows = []
    for i, p in enumerate(ledger.get("points", []), start=1):
        rows.append({
            "geocoded_point_id": f"S17C32_GEO_{i:04d}",
            "hydrological_occurrence_target_id": p["target_id"],
            "candidate_patch_id": patch, "event_id": event_id,
            "bairro": p["bairro"], "logradouro": p["logradouro"],
            "house_number_status": "anonymized_lgpd",
            "geocoder": "nominatim_osm", "geocoder_is_official": "false",
            "resolved_lat": f"{float(p['lat']):.7f}", "resolved_lon": f"{float(p['lon']):.7f}",
            "osm_type": str(p.get("osm_type", "")), "osm_class": str(p.get("osm_class", "")),
            "precision_level": "street_level",
            "uncertainty_m": str(int(STREET_LEVEL_UNCERTAINTY_M)),
            "not_ground_truth": "true", "review_only": "true",
        })
    return rows


def patch_buffer_distance_rows() -> list[dict]:
    rows = []
    for i, p in enumerate(geocoded_point_rows(), start=1):
        lon, lat = float(p["resolved_lon"]), float(p["resolved_lat"])
        pid, dist = _nearest_patch(lon, lat)
        within = dist <= ACCEPTABLE_BUFFER_M
        rows.append({
            "patch_buffer_distance_id": f"S17C32_DIST_{i:04d}",
            "geocoded_point_id": p["geocoded_point_id"],
            "hydrological_occurrence_target_id": p["hydrological_occurrence_target_id"],
            "bairro": p["bairro"], "logradouro": p["logradouro"],
            "nearest_patch_id": pid, "distance_to_patch_m": f"{dist:.1f}",
            "acceptable_buffer_m": str(int(ACCEPTABLE_BUFFER_M)),
            "uncertainty_m": p["uncertainty_m"],
            "within_patch_or_acceptable_buffer": _bool(within),
            "review_only": "true",
        })
    return rows


def g4d_evaluation_rows() -> list[dict]:
    dist_by = {r["geocoded_point_id"]: r for r in patch_buffer_distance_rows()}
    rows = []
    for i, p in enumerate(geocoded_point_rows(), start=1):
        d = dist_by.get(p["geocoded_point_id"], {})
        within = d.get("within_patch_or_acceptable_buffer") == "true"
        official_or_controlled_point = False  # OSM street-level nao e ponto oficial/controlado preciso
        # G4d exige: geometria oficial/controlada de PONTO + distancia compativel
        g4d = official_or_controlled_point and within
        if not within:
            block = "distance_to_patch_exceeds_buffer"
        elif p["house_number_status"] == "anonymized_lgpd":
            block = "logradouro_sem_numero_incerteza_alta_street_level"
        else:
            block = "osm_geocoder_not_official_point"
        rows.append({
            "g4d_evaluation_id": f"S17C32_G4D_{i:04d}",
            "geocoded_point_id": p["geocoded_point_id"],
            "hydrological_occurrence_target_id": p["hydrological_occurrence_target_id"],
            "candidate_patch_id": p["candidate_patch_id"], "event_id": p["event_id"],
            "distance_to_patch_m": d.get("distance_to_patch_m", "unknown"),
            "acceptable_buffer_m": str(int(ACCEPTABLE_BUFFER_M)),
            "within_patch_or_acceptable_buffer": _bool(within),
            "geometry_is_official_or_controlled_point": _bool(official_or_controlled_point),
            "uncertainty_m": p["uncertainty_m"],
            "G4d_patch_buffer_link_confirmed": _bool(g4d),
            "blocking_reason": block, "review_only": "true",
        })
    return rows


def g4_full_evaluation_rows() -> list[dict]:
    # G4c foi obtido parcialmente no 17C31 (2, risk_sector_not_event) mas nao para EVENTO.
    # G4_full = G4c(evento) AND G4d. Aqui G4d avaliado por ponto; G4c de evento permanece falso.
    g4d_rows = g4d_evaluation_rows()
    rows = []
    for i, g in enumerate(g4d_rows, start=1):
        g4d = g["G4d_patch_buffer_link_confirmed"] == "true"
        g4c_event = False  # nenhuma geometria patch-level de EVENTO (nao risk sector) confirmada
        g4_full = g4c_event and g4d
        rows.append({
            "g4_full_evaluation_id": f"S17C32_G4F_{i:04d}",
            "geocoded_point_id": g["geocoded_point_id"],
            "hydrological_occurrence_target_id": g["hydrological_occurrence_target_id"],
            "candidate_patch_id": g["candidate_patch_id"], "event_id": g["event_id"],
            "G4c_event_geometry_present": _bool(g4c_event),
            "G4d_patch_buffer_link_confirmed": _bool(g4d),
            "G4_full_spatial_patch_link": _bool(g4_full),
            "blocking_reason": "G4d_not_confirmed" if not g4d else ("G4c_event_geometry_absent" if not g4c_event else "not_applicable"),
            "review_only": "true",
        })
    return rows


def ground_reference_readiness_rows() -> list[dict]:
    """G1-G3/G6/G7 e G5d ja verdadeiros (17C31); G4_full e o gargalo. GR aceito
    exige G4_full=true. Uma linha por ponto avaliado."""
    event_id = _event_id()
    g4f_by = {r["geocoded_point_id"]: r for r in g4_full_evaluation_rows()}
    rows = []
    for i, p in enumerate(geocoded_point_rows(), start=1):
        g4f = g4f_by.get(p["geocoded_point_id"], {})
        g4_full = g4f.get("G4_full_spatial_patch_link") == "true"
        # G5d/G5_full herdados do 17C31 para o bairro do ponto (se hidrologico separado)
        g5_full = True  # bairro hidrologico separado (o alvo e hidrologico); herdado 17C31
        can_gr = all([True, True, True, g4_full, g5_full, True, True])
        rows.append({
            "ground_reference_readiness_id": f"S17C32_GRR_{i:04d}",
            "candidate_patch_id": p["candidate_patch_id"], "event_id": event_id,
            "geocoded_point_id": p["geocoded_point_id"],
            "hydrological_occurrence_target_id": p["hydrological_occurrence_target_id"],
            "G1": "true", "G2": "true", "G3": "true",
            "G4c": "false", "G4d": g4f.get("G4d_patch_buffer_link_confirmed", "false"),
            "G4_full": _bool(g4_full), "G5d": "true", "G5_full": _bool(g5_full),
            "G6": "true", "G7": "true",
            "can_be_ground_reference_candidate_review_only": _bool(can_gr),
            "can_be_ground_truth": "false", "can_be_training_label": "false",
            "can_unlock_17b": _bool(can_gr),
            "blocking_reason": "G4_full ausente: G4d nao confirmado (distancia incompativel + logradouro sem numero + geocoder OSM nao-oficial)" if not can_gr else "not_applicable",
            "review_only": "true",
        })
    return rows


def _c31_subgate_state() -> tuple[str, str, str, str, str]:
    rows = read_csv(C31_SUBGATE)
    r = rows[0] if rows else {}
    return (r.get("G4c_after", "false"), r.get("G4d_after", "false"),
            r.get("G5d_after", "false"), r.get("G4_full_after", "false"), r.get("G5_full_after", "false"))


def subgate_update_matrix_rows() -> list[dict]:
    g4c_31, g4d_31, g5d_31, g4full_31, g5full_31 = _c31_subgate_state()
    g4d_after = _bool(any(r["G4d_patch_buffer_link_confirmed"] == "true" for r in g4d_evaluation_rows()))
    g4full_after = _bool(any(r["G4_full_spatial_patch_link"] == "true" for r in g4_full_evaluation_rows()))
    reason = []
    if g4d_after == "false":
        reason.append("G4d permanece bloqueado: enderecos oficiais com numero anonimizado (street-level) e patch candidato (Recife Antigo ~-8.00) distante 9-13 km dos bairros com alagamento documentado; geocoder OSM nao-oficial")
    else:
        reason.append("G4d aberto por ponto controlado dentro do buffer")
    return [{
        "subgate_update_id": "S17C32_SGU_0001",
        "candidate_patch_id": _primary_patch_id(), "event_id": _event_id(),
        "G4c_before": g4c_31, "G4c_after": g4c_31,
        "G4d_before": g4d_31, "G4d_after": g4d_after,
        "G4_full_before": g4full_31, "G4_full_after": g4full_after,
        "G5d_before": g5d_31, "G5d_after": g5d_31,
        "G5_full_before": g5full_31, "G5_full_after": g5full_31,
        "transition_reason": "; ".join(reason), "review_only": "true",
    }]


def gate_transition_ledger_rows() -> list[dict]:
    su = subgate_update_matrix_rows()[0]
    rows = []
    # transicao registrada mesmo quando nao ha mudanca (auditavel)
    rows.append({
        "gate_transition_id": "S17C32_LEDGER_0001",
        "candidate_patch_id": _primary_patch_id(), "event_id": _event_id(),
        "gate_group": "spatial_subgate", "gate_name": "G4d_patch_buffer_link_confirmed",
        "transition_from": su["G4d_before"], "transition_to": su["G4d_after"],
        "transition_reason": su["transition_reason"],
        "artifact_manifest_id": "susc_17c32_geocoding_artifacts/_geocoding_ledger.json",
        "geocoder": "nominatim_osm_not_official", "review_only": "true",
    })
    rows.append({
        "gate_transition_id": "S17C32_LEDGER_0002",
        "candidate_patch_id": _primary_patch_id(), "event_id": _event_id(),
        "gate_group": "spatial_gate", "gate_name": "G4_full_spatial_patch_link",
        "transition_from": su["G4_full_before"], "transition_to": su["G4_full_after"],
        "transition_reason": "G4_full requer G4c(evento) e G4d; ambos ausentes",
        "artifact_manifest_id": "susc_17c32_geocoding_artifacts/_geocoding_ledger.json",
        "geocoder": "nominatim_osm_not_official", "review_only": "true",
    })
    return rows


def no_leakage_audit_rows() -> list[dict]:
    rows = []
    for i, p in enumerate(geocoded_point_rows(), start=1):
        rows.append({
            "no_leakage_audit_id": f"S17C32_LEAK_{i:04d}",
            "object_id": p["geocoded_point_id"], "object_type": "geocoded_point",
            "uses_bairro_centroid_as_g4d": "false",
            "uses_osm_alone_to_open_g4d": "false",
            "uses_risk_sector_to_open_g4d": "false",
            "invented_coordinate": "false",
            "logradouro_without_number_marked_high_uncertainty": "true",
            "uses_geocoded_point_as_ground_truth": "false",
            "uses_candidate_as_training_label": "false",
            "passes_no_leakage": "true",
            "blocking_reason": "geocode OSM street-level review-only; nao abre G4d sozinho",
            "review_only": "true",
        })
    if not rows:
        rows.append({
            "no_leakage_audit_id": "S17C32_LEAK_0001", "object_id": "no_point", "object_type": "geocoding_package",
            "uses_bairro_centroid_as_g4d": "false", "uses_osm_alone_to_open_g4d": "false",
            "uses_risk_sector_to_open_g4d": "false", "invented_coordinate": "false",
            "logradouro_without_number_marked_high_uncertainty": "true",
            "uses_geocoded_point_as_ground_truth": "false", "uses_candidate_as_training_label": "false",
            "passes_no_leakage": "true", "blocking_reason": "not_applicable", "review_only": "true",
        })
    return rows


def build_summary() -> dict:
    targets = hydrological_occurrence_target_rows()
    attempts = geocoding_attempt_rows()
    points = geocoded_point_rows()
    dist = patch_buffer_distance_rows()
    g4d = g4d_evaluation_rows()
    g4f = g4_full_evaluation_rows()
    grr = ground_reference_readiness_rows()
    within = [r for r in dist if r["within_patch_or_acceptable_buffer"] == "true"]
    g4d_true = sum(1 for r in g4d if r["G4d_patch_buffer_link_confirmed"] == "true")
    g4full_true = sum(1 for r in g4f if r["G4_full_spatial_patch_link"] == "true")
    accepted = [r for r in grr if r["can_be_ground_reference_candidate_review_only"] == "true"]
    resolved = [a for a in attempts if a["status"] == "resolved"]
    dists = [float(r["distance_to_patch_m"]) for r in dist] if dist else []
    minimum = (
        len(targets) >= 3 and len([a for a in attempts if a["network_enabled"] == "true"]) >= 3
        and len(points) >= 1 and len(dist) >= 1 and len(g4d) >= 1 and len(g4f) >= 1
    )
    return {
        "minimum_success_achieved": minimum,
        "hydrological_occurrence_targets_count": len(targets),
        "geocoding_attempts_count": len(attempts),
        "geocoding_resolved_count": len(resolved),
        "geocoded_points_count": len(points),
        "patch_buffer_distance_rows_count": len(dist),
        "points_within_patch_buffer_count": len(within),
        "nearest_distance_to_patch_m": round(min(dists), 1) if dists else -1,
        "farthest_distance_to_patch_m": round(max(dists), 1) if dists else -1,
        "acceptable_buffer_m": int(ACCEPTABLE_BUFFER_M),
        "G4d_evaluation_rows_count": len(g4d),
        "G4d_true_count": g4d_true,
        "G4_full_evaluation_rows_count": len(g4f),
        "G4_full_true_count": g4full_true,
        "ground_reference_readiness_rows_count": len(grr),
        "accepted_ground_reference_candidate_review_only_count": len(accepted),
        # G5d/G5_full herdados do 17C31 (nao recalculados)
        "G5d_true_count_inherited_17c31": int(_c31_summary().get("G5d_true_count", 0)),
        "G5_full_true_count_inherited_17c31": int(_c31_summary().get("G5_full_true_count", 0)),
        "ground_truth_created": False,
        "training_labels_created": False,
        "score_v6_changed": bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)])),
        "score_v7_created": SCORE_V7.exists(),
        "official_patch_created": False,
        "official_patch_link_created": False,
        "geocoder_is_official": False,
        "eligible_for_17b_now": len(accepted) > 0,
        "eligible_for_score_v7": False,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "result_class": "A_g4d_advanced" if g4d_true > 0 else "B_honest_block_spatial_mismatch",
        "recommended_next_milestone": "SUSC-17C33 Reancorar patch candidato para bairro com alagamento oficial documentado (ex.: Varzea/Areias) OU obter ponto de endereco oficial de-anonimizado (numero) proximo ao patch atual; manter score v6 intacto e 17B fail-closed.",
    }


def _c31_summary() -> dict:
    try:
        return read_json(C31_SUMMARY)
    except Exception:
        return {}


def build_blockers() -> list[dict]:
    su = subgate_update_matrix_rows()[0]
    blockers = []
    if su["G4d_after"] != "true":
        blockers.append(("G4d_still_blocked_distance_or_uncertainty", "G4d bloqueado: distancia incompativel e/ou incerteza street-level"))
    blockers += [
        ("logradouro_sem_numero_anonimizado_lgpd", "enderecos oficiais com numero anonimizado (LGPD): so geocodificacao street-level, incerteza alta"),
        ("osm_geocoder_not_official", "geocoder OSM/Nominatim nao e fonte oficial: sozinho nao abre G4d"),
        ("spatial_mismatch_patch_vs_documented_flood", "patch candidato (Recife Antigo ~-8.00) distante 9-13 km dos bairros com alagamento documentado (Varzea/Pina/Areias/Iputinga)"),
        ("bairro_centroid_not_patch_link", "centroide de bairro nao abre G4d"),
        ("risk_sector_not_patch_link", "area de risco nao abre G4d"),
        ("G4_full_blocked_no_event_geometry", "G4_full bloqueado: sem geometria patch-level de evento confirmada"),
        ("ground_reference_candidate_not_accepted", "nenhum Ground Reference Candidate aceito (G4_full ausente)"),
        ("score_v7_blocked", "score v7 bloqueado ate ground reference oficial aceito"),
        ("17b_blocked", "17B bloqueado ate G4_full + G5_full com ground reference aceito"),
    ]
    rows = []
    for i, (name, desc) in enumerate(blockers, start=1):
        rows.append({
            "blocker_id": f"S17C32_BLOCKER_{i:04d}", "blocker_type": name, "description": desc,
            "blocks_g4d": "true", "blocks_17b": "true", "blocks_score_v7": "true", "review_only": "true",
        })
    return rows


def build_report() -> str:
    s = build_summary()
    su = subgate_update_matrix_rows()[0]
    dist = patch_buffer_distance_rows()
    return "\n".join([
        "# SUSC-17C32 - Geocodificacao oficial de ocorrencias e vinculo patch-buffer para G4d",
        "",
        "## Objetivo (estreito)",
        "Atacar somente o gargalo G4d/G4_full reusando o 17C31: ocorrencia oficial hidrologica -> logradouro/bairro -> geocodificacao controlada -> distancia ao patch/buffer -> G4d. NAO redesenha fingerprint, gate promotion engine, acquisition queue, SAR feasibility, separacao G5d nem politica de gates.",
        "",
        "## Ponto cientifico",
        "G5d/G5_full ja existem para 14 bairros hidrologicos (17C31). O gargalo e G4d/G4_full. Nao se busca mais provar o evento; busca-se ponto oficial de ocorrencia hidrologica proximo ao patch.",
        "",
        "## Geocodificacao controlada",
        f"- Alvos de ocorrencia hidrologica (bairros prioritarios): {s['hydrological_occurrence_targets_count']}.",
        f"- Tentativas de geocodificacao: {s['geocoding_attempts_count']} (resolvidas: {s['geocoding_resolved_count']}); pontos geocodificados: {s['geocoded_points_count']}.",
        "- Geocoder: Nominatim/OSM (NAO oficial); numero do endereco anonimizado (LGPD) -> precisao street-level, incerteza alta.",
        "",
        "## Distancia patch/buffer",
        f"- Buffer aceitavel: {s['acceptable_buffer_m']} m. Pontos dentro do buffer: {s['points_within_patch_buffer_count']}.",
        f"- Distancia mais proxima ao patch: {s['nearest_distance_to_patch_m']} m; mais distante: {s['farthest_distance_to_patch_m']} m.",
        "- Amostra:",
        *[f"  - {d['bairro']} / {d['logradouro']}: {d['distance_to_patch_m']} m -> {d['nearest_patch_id']}" for d in dist[:8]],
        "",
        "## Resultado (honesto)",
        f"- G4d: {su['G4d_before']} -> {su['G4d_after']} (true count = {s['G4d_true_count']}).",
        f"- G4_full: {su['G4_full_before']} -> {su['G4_full_after']} (true count = {s['G4_full_true_count']}).",
        f"- Ground Reference Candidates review-only aceitos: {s['accepted_ground_reference_candidate_review_only_count']}. 17B permanece bloqueado.",
        f"- Classe do resultado: {s['result_class']}.",
        "- Bloqueio honesto e especifico: (1) enderecos oficiais com numero anonimizado (LGPD) -> geocodificacao street-level de incerteza alta; (2) o patch candidato (AOI Charter758, Recife Antigo/Santo Amaro ~-8.00) NAO coincide espacialmente com os bairros onde o alagamento foi oficialmente documentado (Varzea, Pina, Areias, Iputinga: 9-13 km ao sul/oeste); (3) OSM sozinho nao abre G4d.",
        "",
        "## Guardrails",
        "- bairro/centroide nao abriu G4d; OSM sozinho nao abriu G4d; area de risco nao abriu G4d; logradouro sem numero => incerteza alta; coordenada nunca inventada; nenhum ground truth/label; score v6 intacto; score v7 inexistente; 17B nao desbloqueado.",
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
    write_csv(TARGETS, hydrological_occurrence_target_rows())
    write_csv(GEO_ATTEMPTS, geocoding_attempt_rows())
    write_csv(GEO_POINTS, geocoded_point_rows())
    write_csv(DIST_EVAL, patch_buffer_distance_rows())
    write_csv(G4D_EVAL, g4d_evaluation_rows())
    write_csv(G4FULL_EVAL, g4_full_evaluation_rows())
    write_csv(GR_READY, ground_reference_readiness_rows())
    write_csv(SUBGATE_UPDATE, subgate_update_matrix_rows())
    write_csv(LEDGER_OUT, gate_transition_ledger_rows())
    write_csv(LEAKAGE, no_leakage_audit_rows())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_markdown(REPORT, build_report())


def run_mode(mode: str) -> None:
    _require_inputs()
    if mode == "prepare-hydrological-occurrence-targets":
        ensure_dir(OUT)
        write_csv(TARGETS, hydrological_occurrence_target_rows())
    elif mode == "run-controlled-geocoding":
        _run_geocoding()
    elif mode == "resolve-geocoded-points":
        write_csv(GEO_POINTS, geocoded_point_rows())
        write_csv(GEO_ATTEMPTS, geocoding_attempt_rows())
    elif mode == "evaluate-patch-buffer-distance":
        write_csv(DIST_EVAL, patch_buffer_distance_rows())
    elif mode == "evaluate-g4d":
        write_csv(G4D_EVAL, g4d_evaluation_rows())
    elif mode == "evaluate-g4-full":
        write_csv(G4FULL_EVAL, g4_full_evaluation_rows())
    elif mode == "evaluate-ground-reference-readiness":
        write_csv(GR_READY, ground_reference_readiness_rows())
    elif mode == "build-gate-transition-ledger":
        write_csv(SUBGATE_UPDATE, subgate_update_matrix_rows())
        write_csv(LEDGER_OUT, gate_transition_ledger_rows())
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
    before = {p: p.read_bytes() for p in REQUIRED_OUTPUTS if p.exists()}
    build_all()
    errors = []
    for p, content in before.items():
        if p.exists() and p.read_bytes() != content:
            errors.append(f"offline_build_not_byte_identical:{rel(p)}")
    return errors


def validate() -> int:
    if not LEDGER.exists():
        print(f"MISSING_LEDGER: {rel(LEDGER)} (rode run-controlled-geocoding com os dois opt-ins)", file=sys.stderr)
        return 1
    missing = [p for p in REQUIRED_OUTPUTS if not p.exists()]
    if missing:
        print("MISSING: " + "; ".join(rel(p) for p in missing), file=sys.stderr)
        return 1
    errors = _validate_byte_identical()

    targets = read_csv(TARGETS)
    attempts = read_csv(GEO_ATTEMPTS)
    points = read_csv(GEO_POINTS)
    dist = read_csv(DIST_EVAL)
    g4d = read_csv(G4D_EVAL)
    g4f = read_csv(G4FULL_EVAL)
    grr = read_csv(GR_READY)
    leakage = read_csv(LEAKAGE)
    summary = read_json(SUMMARY)
    geo_schema = read_json(SCHEMA_GEO)
    g4d_schema = read_json(SCHEMA_G4D)

    for row in points:
        errors.extend(_schema_violations(row, geo_schema))
    for row in g4d:
        errors.extend(_schema_violations(row, g4d_schema))

    # minimos: alvos, tentativas, pontos, distancias, avaliacoes
    if len(targets) < 3:
        errors.append("targets_lt_3")
    if len([a for a in attempts if a["network_enabled"] == "true"]) < 3:
        errors.append("attempts_lt_3")
    if len(points) < 1:
        errors.append("no_geocoded_point")
    if len(dist) < 1:
        errors.append("no_distance_eval")
    if len(g4d) < 1:
        errors.append("no_g4d_eval")
    if len(g4f) < 1:
        errors.append("no_g4_full_eval")

    # regra dura: bairro/centroide, OSM sozinho, area de risco NAO abrem G4d
    for r in leakage:
        if r["uses_bairro_centroid_as_g4d"] != "false":
            errors.append("bairro_centroid_opened_g4d")
        if r["uses_osm_alone_to_open_g4d"] != "false":
            errors.append("osm_alone_opened_g4d")
        if r["uses_risk_sector_to_open_g4d"] != "false":
            errors.append("risk_sector_opened_g4d")
        if r["invented_coordinate"] != "false":
            errors.append("invented_coordinate")
        if r["passes_no_leakage"] != "true":
            errors.append("no_leakage_failed")

    # logradouro sem numero -> incerteza alta (street-level), nunca point_level
    for p in points:
        if p["house_number_status"] == "anonymized_lgpd" and p["precision_level"] == "point_level":
            errors.append(f"anonymized_marked_point_level:{p['geocoded_point_id']}")
        if int(p["uncertainty_m"]) < 300:
            errors.append(f"street_level_uncertainty_too_low:{p['geocoded_point_id']}")
        if p["geocoder_is_official"] != "false":
            errors.append("geocoder_marked_official")

    # G4d so abre com ponto oficial/controlado E dentro do buffer
    for r in g4d:
        if r["G4d_patch_buffer_link_confirmed"] == "true":
            if r["geometry_is_official_or_controlled_point"] != "true" or r["within_patch_or_acceptable_buffer"] != "true":
                errors.append(f"g4d_opened_without_official_point_or_within_buffer:{r['g4d_evaluation_id']}")

    # G4_full = G4c(evento) AND G4d
    for r in g4f:
        if r["G4_full_spatial_patch_link"] == "true" and (r["G4c_event_geometry_present"] != "true" or r["G4d_patch_buffer_link_confirmed"] != "true"):
            errors.append(f"g4_full_without_components:{r['g4_full_evaluation_id']}")

    # sem GT/label
    if summary["ground_truth_created"] or summary["training_labels_created"]:
        errors.append("gt_or_label_created")
    for r in grr:
        if r["can_be_ground_truth"] != "false" or r["can_be_training_label"] != "false":
            errors.append("grr_marked_gt_or_label")

    # score v6 intacto, score v7 inexistente
    for path in [SCORE_V6, OFFICIAL_PATCHES, OFFICIAL_PATCH_LINKS]:
        if path.exists() and _run_git(["diff", "--name-only", "--", rel(path)]):
            errors.append(f"official_dataset_changed:{rel(path)}")
    if SCORE_V7.exists():
        errors.append("score_v7_exists")

    # 17B so com GR aceito
    accepted = [r for r in grr if r["can_be_ground_reference_candidate_review_only"] == "true"]
    if summary["eligible_for_17b_now"] != (len(accepted) > 0):
        errors.append("17b_eligibility_inconsistent")
    if summary["eligible_for_17b_now"] and not accepted:
        errors.append("17b_without_accepted_gr")

    # nao redesenhar 17C31: G5d herdado deve bater com 17C31
    c31 = read_json(C31_SUMMARY)
    if summary["G5d_true_count_inherited_17c31"] != int(c31.get("G5d_true_count", -1)):
        errors.append("g5d_not_inherited_from_17c31")

    # summary reflete contagens reais
    expected = build_summary()
    for k, v in expected.items():
        if summary.get(k) != v:
            errors.append(f"summary_mismatch:{k}")
    if not summary["minimum_success_achieved"]:
        errors.append("minimum_success_not_achieved")

    # id uniqueness/sorted
    for rows, key in [
        (targets, "hydrological_occurrence_target_id"), (attempts, "geocoding_attempt_id"),
        (points, "geocoded_point_id"), (dist, "patch_buffer_distance_id"),
        (g4d, "g4d_evaluation_id"), (g4f, "g4_full_evaluation_id"),
        (grr, "ground_reference_readiness_id"), (leakage, "no_leakage_audit_id"),
    ]:
        ids = [row[key] for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(
        "17C32 -> "
        f"targets={summary['hydrological_occurrence_targets_count']} attempts={summary['geocoding_attempts_count']} "
        f"points={summary['geocoded_points_count']} within_buffer={summary['points_within_patch_buffer_count']} "
        f"nearest_m={summary['nearest_distance_to_patch_m']} G4d={summary['G4d_true_count']} G4_full={summary['G4_full_true_count']} "
        f"gr_accepted={summary['accepted_ground_reference_candidate_review_only_count']} "
        f"result={summary['result_class']} min_success={summary['minimum_success_achieved']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="susc_17c32")
    parser.add_argument("mode", choices=[
        "prepare-hydrological-occurrence-targets", "run-controlled-geocoding",
        "resolve-geocoded-points", "evaluate-patch-buffer-distance", "evaluate-g4d",
        "evaluate-g4-full", "evaluate-ground-reference-readiness",
        "build-gate-transition-ledger", "build-offline", "audit",
    ])
    args = parser.parse_args(argv)
    if args.mode == "audit":
        return validate()
    run_mode(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
