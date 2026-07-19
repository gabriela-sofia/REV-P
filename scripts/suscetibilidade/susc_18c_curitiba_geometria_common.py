"""SUSC-18C aquisicao e resolucao de geometria oficial de ocorrencia em Curitiba.

O 18B mostrou que Curitiba e o caminho mais curto para uma segunda regiao com
referencia observacional forte somente revisao: o evento datado CUR_2022_01_15
(S17C_REF_0060) tem fonte oficial, ancora hidrometeorologica e 54 patches
candidatos region-only, faltando apenas a geometria de ocorrencia. O trabalho de
protocolo anterior (v2ca) ja inventariou 119 fontes de Curitiba e concluiu, sem
inventar nada, CONTEXT_ONLY_NO_GEOMETRY / NO_LOCAL_GEOMETRY_OR_POINTS: nao ha ponto
nem poligono de ocorrencia local.

Esta etapa entrega a maquinaria completa que transforma geometria de ocorrencia em
referencia forte, pronta para consumir o dado assim que ele chegar:

  auditoria local -> aquisicao (offline-first, rede opcional) -> normalizacao de
  geometria (ocorrencia x area administrativa) -> overlay com os patches oficiais
  (poligonos reais) -> features diretas -> referencia observacional forte somente
  revisao -> gate 17B.

Como nao ha geometria de ocorrencia local e a rede fica desabilitada por padrao,
o resultado honesto e: geometria ausente, maquinaria pronta e um pacote formal
executavel de aquisicao externa (oficio, schema de ingestao e planilha de
resposta) para a Defesa Civil / IPPUC / GeoCuritiba.

Nunca cria ground truth, treino, score_v7 nem benchmark 17B. O score_v6 nunca e
alterado. Bairro/rua/texto, centroide de bairro/municipio e area administrativa
nunca viram geometria forte. Alerta/risco nunca vira ocorrencia. Nao baixa raster
pesado. Nada e inventado.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import ensure_dir, read_csv, read_json, rel, sha256_file, write_csv, write_json, write_markdown  # noqa: E402

DATASETS = ROOT / "datasets"
DAT_SUSC = DATASETS / "suscetibilidade"
DAT_PROTC = DATASETS / "protocolo_c"
LOCAL_RUNS = ROOT / "local_runs"
OUT_DATA_17C = ROOT / "outputs_public" / "data" / "susc_17c_strong_reference_acquisition_canary"
OUT_DATA_18B = ROOT / "outputs_public" / "data" / "susc_18b_execucao_geometrias_regionais_separacao_fenomeno"
OUT_DATA = ROOT / "outputs_public" / "data" / "susc_18c_aquisicao_geometria_oficial_curitiba"
CARDS_DIR = OUT_DATA / "cartoes_curitiba"
OUT_REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

# --- Entradas herdadas -----------------------------------------------------
TARGET_PACK_17C = OUT_DATA_17C / "susc_17c_source_target_pack.csv"
CUR_EXEC_18B = OUT_DATA_18B / "curitiba_execucao_geometria.csv"
MATRIZ_18B = OUT_DATA_18B / "matriz_referencia_observacional_18b.csv"
SUMMARY_18B = OUT_DATA_18B / "summary.json"
CUR_PRELINK = DAT_PROTC / "v1uw_curitiba_event_patch_prelink_update.csv"
PATCH_BOUNDARIES_DIR = LOCAL_RUNS / "ground_truth" / "v2ca" / "recovered_patch_boundaries"
CUR_GEOM_INVENTORY_V2CA = LOCAL_RUNS / "ground_truth" / "v2ca" / "curitiba_event_geometry_inventory_v2ca.csv"
CUR_SOURCE_INVENTORY_V2CA = LOCAL_RUNS / "ground_truth" / "v2ca" / "curitiba_event_source_inventory_v2ca.csv"
CUR_BINDINGS_V2CA = LOCAL_RUNS / "ground_truth" / "v2ca" / "curitiba_patch_event_binding_candidates_v2ca.csv"
CUR_SUMMARY_V2CA = LOCAL_RUNS / "ground_truth" / "v2ca" / "curitiba_acquisition_summary_v2ca.json"
SCORE_V6 = DAT_SUSC / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT_SUSC / "susc_score_v7_candidate_by_patch_v1.csv"

# Ingestao opcional de resposta oficial (quando a geometria chegar). Offline por padrao.
INGEST_DIR = LOCAL_RUNS / "suscetibilidade" / "18c_ingest"
INGEST_CSV = INGEST_DIR / "curitiba_ocorrencias_geometria.csv"
INGEST_JSON = INGEST_DIR / "curitiba_ocorrencias_geometria.json"
INGEST_GEOJSON = INGEST_DIR / "curitiba_ocorrencias_geometria.geojson"
INGEST_WKT = INGEST_DIR / "curitiba_ocorrencias_geometria.wkt"
ACQUIRE_DIR = LOCAL_RUNS / "suscetibilidade" / "18c_aquisicao"
NETWORK_ENABLED = os.environ.get("SUSC_18C_NETWORK", "") == "1"

# --- Saidas publicas -------------------------------------------------------
PREFLIGHT_JSON = OUT_DATA / "preflight.json"
AUDITORIA = OUT_DATA / "auditoria_local_geometria_curitiba.csv"
AQUISICAO = OUT_DATA / "aquisicao_fontes_oficiais_curitiba.csv"
MANIFESTO = OUT_DATA / "manifesto_aquisicao_curitiba.csv"
FILA_AQUISICAO = OUT_DATA / "fila_aquisicao_externa_curitiba.csv"
GEOM_NORM = OUT_DATA / "curitiba_geometrias_ocorrencia_normalizadas.csv"
GEOM_NORM_GEOJSON = OUT_DATA / "curitiba_geometrias_ocorrencia_normalizadas.geojson"
VINCULOS = OUT_DATA / "curitiba_vinculos_evento_patch.csv"
FEATURES = OUT_DATA / "curitiba_features_por_vinculo.csv"
FILA_FEATURES = OUT_DATA / "curitiba_fila_extracao_features.csv"
REFERENCIA = OUT_DATA / "curitiba_referencia_observacional.csv"
SOLICITACAO_MD = OUT_DATA / "solicitacao_geometria_ocorrencia_curitiba.md"
SCHEMA_RESPOSTA = OUT_DATA / "schema_resposta_esperada_curitiba.json"
MODELO_RESPOSTA = OUT_DATA / "modelo_planilha_resposta_curitiba.csv"
GATE_CUR = OUT_DATA / "gate_curitiba_pos_18c.csv"
GATE_17B = OUT_DATA / "gate_prontidao_17b_pos_18c.csv"
RESUMO_STATUS = OUT_DATA / "resumo_por_status.csv"
SUMMARY = OUT_DATA / "summary.json"
REPORT = OUT_REPORTS / "SUSC_18C_AQUISICAO_GEOMETRIA_OFICIAL_CURITIBA.md"
SCHEMA = SCHEMAS / "susc_18c_curitiba_geometria_schema_v1.json"

REQUIRED_INPUTS = [TARGET_PACK_17C, CUR_EXEC_18B, MATRIZ_18B, SUMMARY_18B, SCORE_V6]
REQUIRED_OUTPUTS = [
    PREFLIGHT_JSON, AUDITORIA, AQUISICAO, MANIFESTO, FILA_AQUISICAO, GEOM_NORM,
    VINCULOS, FEATURES, FILA_FEATURES, REFERENCIA, SOLICITACAO_MD, SCHEMA_RESPOSTA,
    MODELO_RESPOSTA, GATE_CUR, GATE_17B, RESUMO_STATUS, SUMMARY, REPORT, SCHEMA,
]

CURITIBA_MAIN_EVENT = "S17C_REF_0060"

# --- Enums -----------------------------------------------------------------
GEOM_STATUS_ALLOWED = [
    "geometria_oficial_ponto", "geometria_oficial_poligono", "geometria_oficial_bbox",
    "geometria_tecnica_footprint", "bloqueada_area_administrativa", "bloqueada_textual",
    "bloqueada_sem_data", "bloqueada_sem_fenomeno", "ausente",
]
GEOM_STATUS_FORTE = {
    "geometria_oficial_ponto", "geometria_oficial_poligono",
    "geometria_oficial_bbox", "geometria_tecnica_footprint",
}
CLASSE_VINCULO_ALLOWED = [
    "exact_polygon_overlap", "point_within_patch", "bbox_overlap",
    "near_patch_buffer_10m", "near_patch_buffer_30m", "near_patch_buffer_50m",
    "same_region_only", "insufficient_for_patch_link",
]
CLASSE_VINCULO_FORTE = {"exact_polygon_overlap", "point_within_patch", "bbox_overlap"}
STATUS_REF_ALLOWED = [
    "referencia_observacional_forte_somente_revisao",
    "referencia_observacional_parcial",
    "evidencia_contextual",
    "bloqueado_sem_geometria",
    "bloqueado_sem_patch_link",
    "bloqueado_sem_features",
    "fila_executavel",
]
USO_PERMITIDO_ALLOWED = [
    "avaliacao_somente_revisao", "contexto_documental",
    "fila_obtencao_geometria", "fila_extracao_features", "fila_aquisicao_externa",
    "solicitacao_externa_geometria_oficial",
]
ACESSO_ALLOWED = [
    "network_disabled_offline_first", "acquired_light_artifact", "access_failed", "external_queue",
]
GATE_FINAL_ALLOWED = [
    "18C_CURITIBA_REFERENCIA_FORTE_SOMENTE_REVISAO",
    "18C_CURITIBA_GEOMETRIA_RESOLVIDA_SEM_FEATURES_COMPLETAS",
    "18C_CURITIBA_GEOMETRIA_AUSENTE_COM_SOLICITACAO_FORMAL",
    "18C_CURITIBA_FILA_EXECUTAVEL_DE_AQUISICAO",
    "18C_BLOQUEADO_FAIL_CLOSED",
]
STATUS_17B_ALLOWED = [
    "17B_APROXIMACAO_COM_SEGUNDA_REGIAO_FORTE",
    "17B_APROXIMACAO_REGIONAL_COM_REFERENCIAS_PARCIAS",
    "17B_BLOQUEADO_POR_GEOMETRIA",
    "17B_BLOQUEADO_POR_AMOSTRA",
    "17B_BLOQUEADO_FAIL_CLOSED",
]

PUBLIC_FORBIDDEN_RE = re.compile(r"\b(?:agentic|agente|codex|llm|ia)\b", re.IGNORECASE)
NO_GT_REASON = (
    "referencia observacional de ocorrencia somente revisao; nao confirma verdade de referencia, "
    "nao alimenta treino e nao autoriza score_v7"
)
HARD_DEFAULTS = {"ground_truth": "false", "eligible_for_training": "false",
                 "score_v7_allowed": "false", "review_only": "true"}

# Fontes oficiais leves candidatas (nunca raster pesado)
OFFICIAL_SOURCES = [
    ("geocuritiba_alagamento", "GeoCuritiba - camadas de alagamento/drenagem",
     "https://geocuritiba.ippuc.org.br/portal/apps/sites/#/geocuritiba", "vetor_wfs_ou_geojson"),
    ("ippuc_dados", "IPPUC - dados geoespaciais municipais",
     "https://ippuc.org.br/geodownloads/geodados.htm", "vetor_shapefile_leve"),
    ("defesa_civil_curitiba", "Defesa Civil de Curitiba - ocorrencias",
     "https://www.curitiba.pr.gov.br/conteudo/defesa-civil", "csv_ou_ocorrencia_documental"),
    ("dados_abertos_curitiba", "Portal de Dados Abertos de Curitiba",
     "https://www.curitiba.pr.gov.br/dadosabertos/", "csv_geojson_json"),
]

# --- Field lists -----------------------------------------------------------
AUDITORIA_FIELDS = [
    "arquivo", "tipo", "tamanho", "origem", "contem_curitiba", "contem_evento",
    "contem_data", "contem_lat_lon", "contem_wkt", "contem_geojson",
    "contem_bairro_rua", "contem_geometria_oficial", "utilizavel", "motivo_uso_ou_bloqueio",
]
AQUISICAO_FIELDS = [
    "source_id", "nome_fonte", "url", "tipo_fonte", "status_acesso", "artefato_local",
    "sha256", "tamanho_bytes", "contem_geometria", "contem_data", "contem_tipo_fenomeno",
    "utilizavel_para_patch_link", "bloqueio", "proxima_acao",
]
MANIFESTO_FIELDS = [
    "source_id", "url", "data_aquisicao", "artefato_local", "sha256", "tamanho_bytes", "observacao",
]
FILA_AQ_FIELDS = [
    "task_id", "source_id", "nome_fonte", "url", "artefato_esperado", "campos_esperados",
    "formato_esperado", "expected_output_path", "command_hint", "criterio_de_sucesso",
]
GEOM_NORM_FIELDS = [
    "geometry_id", "candidate_event_id", "data_evento", "tipo_fenomeno", "geometry_type",
    "crs", "lon", "lat", "bbox", "uncertainty_m", "geometry_source_type", "is_event_specific",
    "geometria_de_ocorrencia", "status_geometria", "review_only", "justificativa_tecnica",
]
VINCULOS_FIELDS = [
    "patch_link_id", "candidate_event_id", "geometry_id", "geometry_type", "patch_id",
    "classe_vinculo", "metrica", "vinculo_forte", "review_only", "justificativa_tecnica",
]
FEATURES_FIELDS = [
    "patch_link_id", "patch_id", "possui_fisico", "possui_espectral", "possui_chuva",
    "fonte_fisico", "fonte_espectral", "fonte_chuva", "lacunas", "acao_minima",
]
FILA_FEATURES_FIELDS = [
    "task_id", "candidate_event_id", "patch_id", "familia_feature", "pre_requisito",
    "fonte_sugerida", "expected_output_path", "command_hint", "criterio_de_sucesso",
]
REFERENCIA_FIELDS = [
    "item_id", "candidate_event_id", "data_evento", "tipo_fenomeno", "fonte", "autoridade_fonte",
    "geometry_id", "geometry_type", "patch_id", "classe_vinculo",
    "possui_fisico", "possui_espectral", "possui_chuva",
    "qualidade_fonte", "qualidade_temporal", "qualidade_geometrica", "qualidade_vinculo", "qualidade_fenomeno",
    "status_referencia_observacional", "uso_permitido",
    "ground_truth", "eligible_for_training", "score_v7_allowed", "review_only",
    "not_ground_truth_reason", "justificativa_tecnica",
]
GATE_FIELDS = ["criterio", "valor_observado", "limiar", "passou", "observacao"]
RESUMO_STATUS_FIELDS = ["status_referencia_observacional", "quantidade"]


# --- Helpers ---------------------------------------------------------------
def _bool(v):
    return "true" if v else "false"


def _run_git(args):
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _read(path):
    return read_csv(path) if path.exists() else []


def _require_inputs():
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(p) for p in missing))
    if SCORE_V7.exists():
        raise AssertionError("score_v7 existe e e proibido para SUSC-18C")


def _has(v):
    return v not in {"", "not_available", "unknown", None}


def _is_false(v):
    return str(v).strip().lower() in {"false", "0", "nao", "no"}


def _is_true(v):
    return str(v).strip().lower() in {"true", "1", "sim", "yes"}


def _score_v6_changed():
    return bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]))


def _target_by_id():
    return {r.get("candidate_event_id", ""): r for r in _read(TARGET_PACK_17C)}


def _v2ca_summary():
    if CUR_SUMMARY_V2CA.exists():
        return read_json(CUR_SUMMARY_V2CA)
    return {
        "sources_inventoried": len(_read(CUR_SOURCE_INVENTORY_V2CA)),
        "patches_with_boundary": len(load_cur_patch_polygons()),
        "patch_event_bindings_created": len(_read(CUR_BINDINGS_V2CA)),
        "ready_for_overlay_count": 0,
        "blocked_reason": "CURITIBA_GEOMETRY_OR_POINT_EVIDENCE_NOT_READY",
        "next_required_step": "acquire_curitiba_event_geometry_or_point_evidence",
    }


def _cur_2022_patch_prelink_count():
    rows = [r for r in _read(CUR_PRELINK) if r.get("proposed_event_id") == "CUR_2022_01_15"]
    return len({r.get("patch_id") for r in rows if r.get("patch_id")})


# --- Geometria: primitivas pure-Python -------------------------------------
def _ring_bbox(ring):
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    return min(lons), min(lats), max(lons), max(lats)


def _point_in_ring(lon, lat, ring):
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _deg_dist_m(lon1, lat1, lon2, lat2):
    mlat = math.radians((lat1 + lat2) / 2.0)
    dx = (lon2 - lon1) * 111320.0 * math.cos(mlat)
    dy = (lat2 - lat1) * 110540.0
    return math.hypot(dx, dy)


def _point_to_bbox_dist_m(lon, lat, bbox):
    minlon, minlat, maxlon, maxlat = bbox
    clon = min(max(lon, minlon), maxlon)
    clat = min(max(lat, minlat), maxlat)
    return _deg_dist_m(lon, lat, clon, clat)


def _bbox_overlap(b1, b2):
    return not (b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1])


def _coords_bbox(coords):
    vals = []

    def walk(obj):
        if isinstance(obj, (list, tuple)) and len(obj) >= 2 and all(isinstance(x, (int, float)) for x in obj[:2]):
            vals.append((float(obj[0]), float(obj[1])))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item)

    walk(coords)
    if not vals:
        return ""
    lons = [v[0] for v in vals]
    lats = [v[1] for v in vals]
    return f"{min(lons)},{min(lats)},{max(lons)},{max(lats)}"


def load_cur_patch_polygons():
    """Carrega os poligonos oficiais dos patches de Curitiba (EPSG:4326)."""
    polys = {}
    if not PATCH_BOUNDARIES_DIR.exists():
        return polys
    for path in sorted(PATCH_BOUNDARIES_DIR.glob("patch_boundary_CUR_*_recovered_v2ca.geojson")):
        try:
            obj = read_json(path)
            geom = obj.get("geometry", {})
            if geom.get("type") != "Polygon":
                continue
            ring = [(float(c[0]), float(c[1])) for c in geom["coordinates"][0]]
            polys[obj.get("properties", {}).get("patch_id", path.stem)] = ring
        except Exception:
            continue
    return polys


# --- Normalizacao de geometria de ocorrencia -------------------------------
def classify_geometria_record(rec):
    """Classifica um registro de geometria candidato. So aceita ocorrencia/evento;
    area administrativa, centroide de bairro, texto e alerta/risco sao bloqueados."""
    src = (rec.get("geometry_source_type") or rec.get("geometry_source") or rec.get("fonte") or "").lower()
    kind = (rec.get("geom_kind") or rec.get("geometry_type") or "").lower()
    kind = {
        "ponto": "point", "Point": "point", "point": "point",
        "poligono": "polygon", "polygon": "polygon",
        "retangulo": "bbox", "bbox": "bbox",
        "technical_footprint": "footprint", "footprint": "footprint",
    }.get(kind, kind)
    textual_hint = " ".join(str(rec.get(k, "")) for k in ("bairro", "rua", "logradouro", "endereco", "address"))
    if _is_true(rec.get("is_neighborhood_centroid")) or "centroid" in src or "centroide" in src:
        return "bloqueada_area_administrativa"
    if _is_true(rec.get("is_risk_area_general")) or any(t in src for t in ("risco", "risk", "administr", "setor", "bairro", "municip")):
        return "bloqueada_area_administrativa"
    if _is_false(rec.get("is_event_specific")):
        return "bloqueada_area_administrativa"
    if kind in {"textual", "endereco", "address", "rua", "logradouro"}:
        return "bloqueada_textual"
    if textual_hint.strip() and kind not in {"point", "polygon", "bbox", "footprint"}:
        return "bloqueada_textual"
    if not _has(rec.get("data_evento", "")):
        return "bloqueada_sem_data"
    if (rec.get("tipo_fenomeno", "") or "") not in {"flood_inundation_alagamento", "inundacao_alagamento_enxurrada"}:
        return "bloqueada_sem_fenomeno"
    return {
        "point": "geometria_oficial_ponto", "polygon": "geometria_oficial_poligono",
        "bbox": "geometria_oficial_bbox", "footprint": "geometria_tecnica_footprint",
    }.get(kind, "ausente")


def normalize_geometry(rec, idx=0):
    """Normaliza um registro aceito para EPSG:4326 e gera geometry_id + uncertainty_m."""
    status = classify_geometria_record(rec)
    lon = rec.get("lon", "")
    lat = rec.get("lat", "")
    bbox = rec.get("bbox", "")
    unc = rec.get("precisao_m", "")
    uncertainty = unc if _has(unc) else ("30" if status == "geometria_oficial_ponto" else "60")
    cid = rec.get("candidate_event_id", "")
    return {
        "geometry_id": f"S18C_CUR_GEOM_{idx:04d}",
        "candidate_event_id": cid, "data_evento": rec.get("data_evento", "not_available"),
        "tipo_fenomeno": rec.get("tipo_fenomeno", ""),
        "geometry_type": {"geometria_oficial_ponto": "point", "geometria_oficial_poligono": "polygon",
                          "geometria_oficial_bbox": "bbox", "geometria_tecnica_footprint": "technical_footprint"}.get(status, "none"),
        "crs": "EPSG:4326", "lon": lon, "lat": lat, "bbox": bbox, "uncertainty_m": uncertainty,
        "geometry_source_type": rec.get("geometry_source_type") or rec.get("geometry_source", ""),
        "is_event_specific": str(rec.get("is_event_specific", "true")).lower(),
        "geometria_de_ocorrencia": _bool(status in GEOM_STATUS_FORTE),
        "status_geometria": status, "review_only": "true",
        "justificativa_tecnica": (
            "geometria oficial de ocorrencia normalizada para EPSG:4326; validada como evento, nao area administrativa"
            if status in GEOM_STATUS_FORTE else
            f"geometria candidata bloqueada ({status}); nao usada como geometria forte"
        ),
        "_status": status,
    }


def _feature_to_record(feature):
    props = feature.get("properties", {})
    geom = feature.get("geometry", {}) or {}
    rec = dict(props)
    geom_type = geom.get("type", "").lower()
    coords = geom.get("coordinates")
    if geom_type == "point" and isinstance(coords, list) and len(coords) >= 2:
        rec.setdefault("geometry_type", "point")
        rec.setdefault("lon", coords[0])
        rec.setdefault("lat", coords[1])
    elif geom_type == "polygon":
        rec.setdefault("geometry_type", "polygon")
        rec.setdefault("bbox", _coords_bbox(coords))
    elif geom_type in {"multipolygon", "linestring", "multilinestring"}:
        rec.setdefault("geometry_type", "polygon")
        rec.setdefault("bbox", _coords_bbox(coords))
    return rec


def _json_records(path):
    obj = read_json(path)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and obj.get("type") == "FeatureCollection":
        return [_feature_to_record(f) for f in obj.get("features", [])]
    if isinstance(obj, dict) and obj.get("type") == "Feature":
        return [_feature_to_record(obj)]
    if isinstance(obj, dict) and isinstance(obj.get("records"), list):
        return obj["records"]
    if isinstance(obj, dict):
        return [obj]
    return []


def _wkt_records(path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip() or line.lower().startswith("candidate_event_id"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5:
            continue
        cid, data_evento, tipo_fenomeno, geometry_source, wkt = parts[:5]
        rec = {
            "candidate_event_id": cid, "data_evento": data_evento,
            "tipo_fenomeno": tipo_fenomeno, "geometry_source": geometry_source,
            "crs": parts[5] if len(parts) > 5 else "EPSG:4326",
        }
        up = wkt.upper()
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", wkt)]
        if up.startswith("POINT") and len(nums) >= 2:
            rec.update({"geometry_type": "point", "lon": nums[0], "lat": nums[1]})
        elif up.startswith(("POLYGON", "MULTIPOLYGON")) and len(nums) >= 4:
            pairs = list(zip(nums[0::2], nums[1::2]))
            rec.update({"geometry_type": "polygon", "bbox": _coords_bbox(pairs)})
        rows.append(rec)
    return rows


def _ingest_records():
    """Le registros de geometria de ocorrencia se houver resposta oficial ingerida.
    Offline por padrao: sem arquivo, retorna lista vazia (nada e inventado)."""
    rows = []
    if INGEST_CSV.exists():
        rows.extend(_read(INGEST_CSV))
    for path in (INGEST_JSON, INGEST_GEOJSON):
        if path.exists():
            rows.extend(_json_records(path))
    if INGEST_WKT.exists():
        rows.extend(_wkt_records(INGEST_WKT))
    return rows


# --- Overlay com patches oficiais ------------------------------------------
def overlay_geometry(norm, patches):
    """Cruza uma geometria normalizada com os poligonos oficiais dos patches.
    Retorna lista de (patch_id, classe_vinculo, metrica)."""
    links = []
    status = norm.get("_status") or norm.get("status_geometria")
    if status not in GEOM_STATUS_FORTE:
        return links
    if status in {"geometria_oficial_poligono", "geometria_oficial_bbox", "geometria_tecnica_footprint"}:
        gb = _parse_bbox(norm.get("bbox", ""))
        if gb:
            for pid, ring in patches.items():
                pb = _ring_bbox(ring)
                if _bbox_overlap(gb, pb):
                    classe = "exact_polygon_overlap" if status == "geometria_oficial_poligono" else "bbox_overlap"
                    links.append((pid, classe, "bbox_intersect"))
        return links
    # ponto
    try:
        lon = float(norm.get("lon", ""))
        lat = float(norm.get("lat", ""))
    except (TypeError, ValueError):
        return links
    for pid, ring in patches.items():
        if _point_in_ring(lon, lat, ring):
            links.append((pid, "point_within_patch", "0m"))
            continue
        d = _point_to_bbox_dist_m(lon, lat, _ring_bbox(ring))
        if d <= 10:
            links.append((pid, "near_patch_buffer_10m", f"{d:.1f}m"))
        elif d <= 30:
            links.append((pid, "near_patch_buffer_30m", f"{d:.1f}m"))
        elif d <= 50:
            links.append((pid, "near_patch_buffer_50m", f"{d:.1f}m"))
    return links


def _parse_bbox(s):
    if not _has(s):
        return None
    try:
        parts = [float(x) for x in str(s).replace(";", ",").split(",")]
        if len(parts) == 4:
            return (min(parts[0], parts[2]), min(parts[1], parts[3]), max(parts[0], parts[2]), max(parts[1], parts[3]))
    except ValueError:
        return None
    return None


# --- Tarefa 2: auditoria local ---------------------------------------------
def auditoria_rows():
    rows = []

    def add(path, origem, contem_geom_oficial, utilizavel, motivo):
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        rows.append({
            "arquivo": rel(path), "tipo": path.suffix.lstrip(".") or "dir", "tamanho": str(size),
            "origem": origem, "contem_curitiba": "true", "contem_evento": "true",
            "contem_data": "true", "contem_lat_lon": "false", "contem_wkt": "false",
            "contem_geojson": _bool(path.suffix == ".geojson"), "contem_bairro_rua": "false",
            "contem_geometria_oficial": _bool(contem_geom_oficial), "utilizavel": _bool(utilizavel),
            "motivo_uso_ou_bloqueio": motivo,
        })

    # inventario de geometria de evento de Curitiba do v2ca (conclusao anterior)
    if CUR_GEOM_INVENTORY_V2CA.exists():
        for r in _read(CUR_GEOM_INVENTORY_V2CA):
            size = CUR_GEOM_INVENTORY_V2CA.stat().st_size
            rows.append({
                "arquivo": rel(CUR_GEOM_INVENTORY_V2CA), "tipo": "csv", "tamanho": str(size),
                "origem": "v2ca_curitiba_event_geometry_inventory", "contem_curitiba": "true",
                "contem_evento": "true", "contem_data": _bool(_has(r.get("event_id"))),
                "contem_lat_lon": "false", "contem_wkt": "false", "contem_geojson": "false",
                "contem_bairro_rua": "false", "contem_geometria_oficial": "false", "utilizavel": "false",
                "motivo_uso_ou_bloqueio": (
                    f"{r.get('event_id')}: {r.get('geometry_source_type')} / {r.get('blocked_reason')}; "
                    "geometria nao inventada, sem ponto/poligono de ocorrencia"
                ),
            })
    if CUR_SOURCE_INVENTORY_V2CA.exists():
        rows.append({
            "arquivo": rel(CUR_SOURCE_INVENTORY_V2CA), "tipo": "csv",
            "tamanho": str(CUR_SOURCE_INVENTORY_V2CA.stat().st_size),
            "origem": "v2ca_curitiba_event_source_inventory", "contem_curitiba": "true",
            "contem_evento": "true", "contem_data": "false", "contem_lat_lon": "false",
            "contem_wkt": "false", "contem_geojson": "false", "contem_bairro_rua": "false",
            "contem_geometria_oficial": "false", "utilizavel": "false",
            "motivo_uso_ou_bloqueio": (
                f"{_v2ca_summary().get('sources_inventoried', len(_read(CUR_SOURCE_INVENTORY_V2CA)))} fontes "
                "locais auditadas; nenhuma trouxe ponto ou poligono de ocorrencia"
            ),
        })
    if CUR_BINDINGS_V2CA.exists():
        rows.append({
            "arquivo": rel(CUR_BINDINGS_V2CA), "tipo": "csv",
            "tamanho": str(CUR_BINDINGS_V2CA.stat().st_size),
            "origem": "v2ca_curitiba_patch_event_binding_candidates", "contem_curitiba": "true",
            "contem_evento": "true", "contem_data": "true", "contem_lat_lon": "false",
            "contem_wkt": "false", "contem_geojson": "false", "contem_bairro_rua": "false",
            "contem_geometria_oficial": "false", "utilizavel": "true",
            "motivo_uso_ou_bloqueio": (
                f"{_v2ca_summary().get('patch_event_bindings_created', len(_read(CUR_BINDINGS_V2CA)))} "
                "bindings candidatos; 0 prontos para overlay sem geometria de ocorrencia"
            ),
        })
    # patches oficiais de Curitiba (geometria de patch, nao de ocorrencia)
    polys = load_cur_patch_polygons()
    if polys:
        rows.append({
            "arquivo": rel(PATCH_BOUNDARIES_DIR), "tipo": "geojson_dir", "tamanho": str(len(polys)),
            "origem": "v2ca_recovered_patch_boundaries", "contem_curitiba": "true", "contem_evento": "false",
            "contem_data": "false", "contem_lat_lon": "true", "contem_wkt": "false", "contem_geojson": "true",
            "contem_bairro_rua": "false", "contem_geometria_oficial": "false", "utilizavel": "true",
            "motivo_uso_ou_bloqueio": (
                f"{len(polys)} poligonos de patch oficiais (EPSG:4326) uteis para overlay quando a geometria "
                "de ocorrencia chegar; nao sao geometria de ocorrencia por si"
            ),
        })
    # execucao 18B (confirmacao de ausencia)
    if CUR_EXEC_18B.exists():
        add(CUR_EXEC_18B, "susc_18b_execucao_geometria", False, False,
            "18B ja concluiu: geometria de ocorrencia ausente; patches candidatos apenas region-only")
    return rows


# --- Tarefa 3: aquisicao de fontes oficiais --------------------------------
def aquisicao_rows():
    rows = []
    ensure_dir(ACQUIRE_DIR)
    for source_id, nome, url, tipo in OFFICIAL_SOURCES:
        acesso = _try_acquire(source_id, url) if NETWORK_ENABLED else "network_disabled_offline_first"
        artefato, sha, size = "none", "not_available", "0"
        bloqueio = ("rede desabilitada por padrao (offline-first); habilite SUSC_18C_NETWORK=1 para aquisicao leve"
                    if acesso == "network_disabled_offline_first" else "sem artefato de ocorrencia adquirido")
        rows.append({
            "source_id": source_id, "nome_fonte": nome, "url": url, "tipo_fonte": tipo,
            "status_acesso": acesso, "artefato_local": artefato, "sha256": sha, "tamanho_bytes": size,
            "contem_geometria": "false", "contem_data": "false", "contem_tipo_fenomeno": "false",
            "utilizavel_para_patch_link": "false", "bloqueio": bloqueio,
            "proxima_acao": "solicitar camada de ocorrencia datada via oficio formal ou aquisicao leve autorizada",
        })
    return rows


def _try_acquire(source_id, url):
    # aquisicao leve real apenas quando explicitamente habilitada; nunca raster pesado.
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "REV-P-academico"})
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            data = resp.read(1_000_000)
        ensure_dir(ACQUIRE_DIR)
        out = ACQUIRE_DIR / f"{source_id}_metadata.bin"
        out.write_bytes(data)
        return "acquired_light_artifact"
    except Exception:
        return "access_failed"


def manifesto_rows(aquisicao):
    rows = []
    for r in aquisicao:
        if r["status_acesso"] == "acquired_light_artifact" and r["artefato_local"] != "none":
            path = Path(r["artefato_local"])
            rows.append({
                "source_id": r["source_id"], "url": r["url"], "data_aquisicao": _run_git(["log", "-1", "--format=%cI"]) or "not_available",
                "artefato_local": r["artefato_local"],
                "sha256": sha256_file(path) if path.exists() else "not_available",
                "tamanho_bytes": r["tamanho_bytes"], "observacao": "artefato leve oficial preservado",
            })
    if not rows:
        rows.append({
            "source_id": "none", "url": "not_available", "data_aquisicao": "not_available",
            "artefato_local": "none", "sha256": "not_available", "tamanho_bytes": "0",
            "observacao": "nenhum artefato leve adquirido (offline-first); aquisicao pendente de oficio ou rede autorizada",
        })
    return rows


def fila_aquisicao_rows():
    rows = []
    for i, (source_id, nome, url, tipo) in enumerate(OFFICIAL_SOURCES, start=1):
        rows.append({
            "task_id": f"S18C_AQ_{i:04d}", "source_id": source_id, "nome_fonte": nome, "url": url,
            "artefato_esperado": "camada de ocorrencia de alagamento/inundacao datada do evento de janeiro de 2022",
            "campos_esperados": "geometria(ponto/poligono/bbox);crs;data_ocorrencia;tipo_fenomeno;fonte;precisao_m",
            "formato_esperado": "geojson;shapefile_leve;csv;json;xlsx",
            "expected_output_path": rel(INGEST_CSV),
            "command_hint": "adquirir camada leve oficial (sem raster pesado); ingerir na planilha de resposta e reexecutar o 18C",
            "criterio_de_sucesso": "geometria oficial de ocorrencia datada, vinculavel a patch por overlay",
        })
    return rows


# --- Normalizacao (execucao) -----------------------------------------------
def _blocked_geometry_rows():
    rows = []
    for i, c in enumerate(_cur_flood_events(), start=1):
        cid = c.get("candidate_event_id", "")
        rows.append({
            "geometry_id": "not_available",
            "candidate_event_id": cid,
            "data_evento": c.get("event_date_candidate", "not_available"),
            "tipo_fenomeno": c.get("phenomenon_type", ""),
            "geometry_type": "none",
            "crs": "not_available",
            "lon": "",
            "lat": "",
            "bbox": "",
            "uncertainty_m": "not_available",
            "geometry_source_type": "ausente",
            "is_event_specific": "false",
            "geometria_de_ocorrencia": "false",
            "status_geometria": "ausente",
            "review_only": "true",
            "justificativa_tecnica": (
                f"{cid}: nenhuma geometria oficial de ocorrencia foi encontrada localmente; "
                "bairro, rua, centroide, area administrativa, alerta e risco nao sao aceitos como geometria forte"
            ),
            "_status": "ausente",
        })
    return rows


def build_geometries():
    norms = []
    for i, rec in enumerate(_ingest_records(), start=1):
        norms.append(normalize_geometry(rec, i))
    return norms or _blocked_geometry_rows()


def geom_norm_rows(norms):
    rows = []
    for n in norms:
        row = {k: n.get(k, "") for k in GEOM_NORM_FIELDS}
        rows.append(row)
    return rows


def write_geojson(norms):
    features = []
    for n in norms:
        if n["_status"] == "geometria_oficial_ponto" and _has(n["lon"]) and _has(n["lat"]):
            features.append({
                "type": "Feature",
                "properties": {"geometry_id": n["geometry_id"], "candidate_event_id": n["candidate_event_id"],
                               "data_evento": n["data_evento"], "status_geometria": n["_status"], "review_only": True},
                "geometry": {"type": "Point", "coordinates": [float(n["lon"]), float(n["lat"])]},
            })
    if features:
        write_json(GEOM_NORM_GEOJSON, {"type": "FeatureCollection", "features": features})
        return len(features)
    return 0


# --- Overlay (execucao) ----------------------------------------------------
def build_vinculos(norms):
    patches = load_cur_patch_polygons()
    rows = []
    counter = 0
    for n in norms:
        links = overlay_geometry(n, patches)
        if not links:
            counter += 1
            blocked = n.get("_status") not in GEOM_STATUS_FORTE
            rows.append({
                "patch_link_id": f"S18C_LINK_{counter:04d}", "candidate_event_id": n["candidate_event_id"],
                "geometry_id": n["geometry_id"], "geometry_type": n["geometry_type"], "patch_id": "not_available",
                "classe_vinculo": "insufficient_for_patch_link",
                "metrica": "sem_geometria_de_ocorrencia" if blocked else "sem_overlap",
                "vinculo_forte": "false", "review_only": "true",
                "justificativa_tecnica": (
                    "patch-link bloqueado: sem geometry_id oficial de ocorrencia; same_region_only nao e vinculo forte"
                    if blocked else
                    "geometria resolvida nao intersecta nem se aproxima de patch oficial"
                ),
            })
            continue
        for pid, classe, metric in links:
            counter += 1
            rows.append({
                "patch_link_id": f"S18C_LINK_{counter:04d}", "candidate_event_id": n["candidate_event_id"],
                "geometry_id": n["geometry_id"], "geometry_type": n["geometry_type"], "patch_id": pid,
                "classe_vinculo": classe, "metrica": metric,
                "vinculo_forte": _bool(classe in CLASSE_VINCULO_FORTE), "review_only": "true",
                "justificativa_tecnica": (
                    f"overlay review-only: {classe} entre geometria de ocorrencia e patch oficial {pid}"
                    if classe in CLASSE_VINCULO_FORTE else
                    f"candidato QA tecnico ({classe}); nao forte automatico"
                ),
            })
    return rows, patches


# --- Features (Curitiba) ---------------------------------------------------
def features_rows(vinculos):
    rows = []
    seen = set()
    for v in vinculos:
        pid = v["patch_id"]
        if pid in {"", "not_available"} or pid in seen:
            continue
        seen.add(pid)
        rows.append({
            "patch_link_id": v["patch_link_id"], "patch_id": pid,
            "possui_fisico": "false", "possui_espectral": "false", "possui_chuva": "false",
            "fonte_fisico": "ausente", "fonte_espectral": "ausente", "fonte_chuva": "ausente",
            "lacunas": "fisico;espectral;chuva",
            "acao_minima": "extrair features diretas pre-evento por patch (DEM/espectral/chuva) apos overlay",
        })
    if not rows and vinculos:
        rows.append({
            "patch_link_id": vinculos[0]["patch_link_id"], "patch_id": "not_available",
            "possui_fisico": "false", "possui_espectral": "false", "possui_chuva": "false",
            "fonte_fisico": "ausente", "fonte_espectral": "ausente", "fonte_chuva": "ausente",
            "lacunas": "sem_geometria_de_ocorrencia;sem_patch_link;fisico;espectral;chuva",
            "acao_minima": "obter geometria oficial de ocorrencia antes de extrair features por patch",
        })
    return rows


def fila_features_rows(vinculos):
    rows = []
    counter = 0
    seen = set()
    for v in vinculos:
        pid = v["patch_id"]
        if pid in {"", "not_available"} or pid in seen:
            continue
        seen.add(pid)
        for familia in ("fisico", "espectral", "chuva"):
            counter += 1
            rows.append({
                "task_id": f"S18C_FEAT_{counter:04d}", "candidate_event_id": v["candidate_event_id"],
                "patch_id": pid, "familia_feature": familia,
                "pre_requisito": "patch-link resolvido por overlay de geometria de ocorrencia",
                "fonte_sugerida": {
                    "fisico": "DEM Copernicus GLO-30 + hidrografia (metodo 17F/17G)",
                    "espectral": "Sentinel-2 COG pre-evento (metodo 17C20)",
                    "chuva": "CHIRPS janela pre-evento (metodo 17C25)",
                }[familia],
                "expected_output_path": f"local_runs/suscetibilidade/18c_regional/{pid.lower()}_{familia}.csv",
                "command_hint": "extrair a familia na janela pre-evento por patch",
                "criterio_de_sucesso": f"features de {familia} por patch, somente pre-evento, com fonte",
            })
    return rows


# --- Referencia observacional Curitiba -------------------------------------
def _cur_flood_events():
    return sorted(
        [r for r in _read(TARGET_PACK_17C)
         if r.get("region") == "CUR" and r.get("phenomenon_type") == "flood_inundation_alagamento"],
        key=lambda r: r.get("candidate_event_id", ""),
    )


def _q_fonte(tier):
    return {"official": 85, "technical": 80, "documentary": 55, "internal_registry": 30}.get(tier, 20)


def _q_temporal(prec):
    return {"exact_day": 90, "range": 80, "month_only": 35}.get(prec, 0)


def referencia_rows(norms, vinculos, features):
    by_event_geom = {n["candidate_event_id"]: n for n in norms if n["_status"] in GEOM_STATUS_FORTE}
    strong_links_by_event = {}
    for v in vinculos:
        if v["vinculo_forte"] == "true":
            strong_links_by_event.setdefault(v["candidate_event_id"], []).append(v)
    feats_by_patch = {f["patch_id"]: f for f in features}
    rows = []
    for i, c in enumerate(_cur_flood_events(), start=1):
        cid = c.get("candidate_event_id", "")
        geom = by_event_geom.get(cid)
        strong = strong_links_by_event.get(cid, [])
        patch_id = strong[0]["patch_id"] if strong else "not_available"
        classe = strong[0]["classe_vinculo"] if strong else "same_region_only"
        feat = feats_by_patch.get(patch_id, {})
        has_fis = feat.get("possui_fisico") == "true"
        has_esp = feat.get("possui_espectral") == "true"
        has_chu = feat.get("possui_chuva") == "true"
        status, uso = _ref_status(c, geom, strong, has_fis and has_esp and has_chu)
        rows.append({
            "item_id": f"S18C_REF_{i:04d}", "candidate_event_id": cid,
            "data_evento": c.get("event_date_candidate", "not_available"),
            "tipo_fenomeno": c.get("phenomenon_type", ""), "fonte": c.get("source_name", ""),
            "autoridade_fonte": c.get("authority_tier", ""),
            "geometry_id": geom["geometry_id"] if geom else "not_available",
            "geometry_type": geom["geometry_type"] if geom else "none",
            "patch_id": patch_id, "classe_vinculo": classe,
            "possui_fisico": _bool(has_fis), "possui_espectral": _bool(has_esp), "possui_chuva": _bool(has_chu),
            "qualidade_fonte": str(_q_fonte(c.get("authority_tier", ""))),
            "qualidade_temporal": str(_q_temporal(c.get("event_date_precision", ""))),
            "qualidade_geometrica": "85" if geom else ("20" if _addr_only(c) else "10"),
            "qualidade_vinculo": "85" if strong else "0",
            "qualidade_fenomeno": "85",
            "status_referencia_observacional": status, "uso_permitido": uso,
            "ground_truth": HARD_DEFAULTS["ground_truth"],
            "eligible_for_training": HARD_DEFAULTS["eligible_for_training"],
            "score_v7_allowed": HARD_DEFAULTS["score_v7_allowed"], "review_only": HARD_DEFAULTS["review_only"],
            "not_ground_truth_reason": NO_GT_REASON,
            "justificativa_tecnica": _ref_justificativa(c, geom, strong, status),
        })
    return rows


def _addr_only(c):
    return "address_text" in c.get("geometry_status", "") or (c.get("has_address") == "true" and c.get("has_point") != "true")


def _ref_status(c, geom, strong, features_ok):
    if geom and strong and features_ok:
        return "referencia_observacional_forte_somente_revisao", "avaliacao_somente_revisao"
    if geom and strong:
        return "bloqueado_sem_features", "fila_extracao_features"
    if geom and not strong:
        return "bloqueado_sem_patch_link", "fila_obtencao_geometria"
    if _addr_only(c):
        return "bloqueado_sem_geometria", "solicitacao_externa_geometria_oficial"
    if c.get("authority_tier") == "official" and _has(c.get("event_date_candidate", "")):
        return "bloqueado_sem_geometria", "solicitacao_externa_geometria_oficial"
    return "bloqueado_sem_geometria", "solicitacao_externa_geometria_oficial"


def _ref_justificativa(c, geom, strong, status):
    base = (f"evento {c.get('candidate_event_id','')} ({c.get('event_date_candidate','')}); "
            f"fonte {c.get('authority_tier','')}")
    extra = {
        "referencia_observacional_forte_somente_revisao": "geometria de ocorrencia + vinculo forte + features completas review-only",
        "bloqueado_sem_features": "geometria e vinculo forte resolvidos; faltam features diretas pre-evento",
        "bloqueado_sem_patch_link": "geometria resolvida, mas sem overlap com patch oficial",
        "referencia_observacional_parcial": "inundacao datada oficial; geometria de ocorrencia pendente de aquisicao formal",
        "bloqueado_sem_geometria": "geometria oficial de ocorrencia ausente; solicitacao formal externa emitida",
    }.get(status, "candidato review-only")
    return f"{base}; {extra}"


# --- Pacote formal de solicitacao ------------------------------------------
def solicitacao_markdown():
    return """# Solicitacao tecnica de geometria de ocorrencia - eventos de alagamento em Curitiba

## Finalidade

Solicitacao de dados espaciais de ocorrencia de alagamento/inundacao/enxurrada em Curitiba, para uso
academico em analise observacional de suscetibilidade, em carater somente revisao. Os dados nao serao
usados como verdade de referencia operacional, nem para treino de modelo, e nenhum score oficial sera
alterado.

## Destinatarios sugeridos

- Defesa Civil Municipal de Curitiba
- Instituto de Pesquisa e Planejamento Urbano de Curitiba (IPPUC) / GeoCuritiba
- Portal de Dados Abertos de Curitiba

## Eventos de interesse

- Evento principal: temporal e alagamentos de 15 e 16 de janeiro de 2022 (referencia interna CUR_2022_01_15).
- Eventos adicionais: outubro de 2023 e fevereiro de 2024.

## Dados solicitados por ocorrencia

1. Data da ocorrencia (dia exato ou intervalo curto).
2. Tipo de fenomeno (alagamento, inundacao ou enxurrada), confirmado explicitamente.
3. Geometria da ocorrencia: ponto, poligono ou retangulo envolvente (bbox) da area efetivamente atingida.
4. Endereco textual de referencia, se houver (apenas complementar; nao substitui a geometria).
5. Fonte e responsavel tecnico pelo registro.
6. Precisao espacial estimada (em metros).
7. Observacao sobre o metodo de coleta (vistoria em campo, registro de chamado, sensoriamento etc.).
8. Autorizacao de uso academico dos dados.

## Requisitos tecnicos

- Sistema de referencia espacial informado (preferencialmente EPSG:4326; qualquer CRS oficial e aceito e sera convertido).
- Formatos leves: GeoJSON, shapefile, CSV com colunas de coordenada, JSON ou XLSX.
- Nao e necessario enviar imagens de satelite nem arquivos pesados.
- A geometria deve representar a ocorrencia observada, e nao area de risco generica, setor administrativo ou limite de bairro.

## Modelo de resposta

Segue planilha modelo (`modelo_planilha_resposta_curitiba.csv`) e o schema de ingestao
(`schema_resposta_esperada_curitiba.json`) para padronizar o retorno.
"""


def schema_resposta_obj():
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Schema de resposta esperada - geometria de ocorrencia de Curitiba",
        "type": "object",
        "required": ["candidate_event_id", "data_evento", "tipo_fenomeno", "geometry_type", "geometry_source", "crs"],
        "properties": {
            "candidate_event_id": {"type": "string", "examples": ["S17C_REF_0060"]},
            "data_evento": {"type": "string", "description": "dia exato ou intervalo curto (ISO)"},
            "tipo_fenomeno": {"enum": ["flood_inundation_alagamento", "inundacao_alagamento_enxurrada"]},
            "geometry_source": {"type": "string", "description": "fonte responsavel ou metodo do registro"},
            "geometry_source_type": {"type": "string", "description": "alias tecnico opcional de geometry_source"},
            "geometry_type": {"enum": ["point", "polygon", "bbox", "footprint", "textual"]},
            "geom_kind": {"enum": ["point", "polygon", "bbox", "footprint", "textual"], "description": "alias opcional de geometry_type"},
            "lon": {"type": "number"}, "lat": {"type": "number"},
            "bbox": {"type": "string", "description": "minlon,minlat,maxlon,maxlat"},
            "crs": {"type": "string", "default": "EPSG:4326"},
            "fonte": {"type": "string"}, "autoridade_fonte": {"type": "string"},
            "is_event_specific": {"type": "boolean"},
            "is_risk_area_general": {"type": "boolean"},
            "is_neighborhood_centroid": {"type": "boolean"},
            "precisao_m": {"type": "number"},
        },
        "ingestion_path": rel(INGEST_CSV),
        "notes": "geometria de ocorrencia apenas; area administrativa, centroide de bairro e alerta/risco sao rejeitados",
    }


def modelo_resposta_rows():
    return [{
        "candidate_event_id": "S17C_REF_0060", "data_evento": "2022-01-15..2022-01-16",
        "tipo_fenomeno": "flood_inundation_alagamento", "geometry_source": "PREENCHER_fonte_responsavel",
        "geometry_source_type": "PREENCHER_origem_do_registro", "geometry_type": "point|polygon|bbox",
        "geom_kind": "alias_opcional", "lon": "PREENCHER", "lat": "PREENCHER",
        "bbox": "minlon,minlat,maxlon,maxlat", "crs": "EPSG:4326", "fonte": "PREENCHER",
        "autoridade_fonte": "official", "is_event_specific": "true", "is_risk_area_general": "false",
        "is_neighborhood_centroid": "false", "precisao_m": "PREENCHER",
    }]


MODELO_RESPOSTA_FIELDS = list(modelo_resposta_rows()[0].keys())


# --- Gates -----------------------------------------------------------------
def gate_curitiba_rows(norms, vinculos, features, ref_rows):
    geom_ok = any(n["_status"] in GEOM_STATUS_FORTE for n in norms)
    strong_links = sum(1 for v in vinculos if v["vinculo_forte"] == "true")
    feats_ok = any(f["possui_fisico"] == "true" for f in features)
    forte = any(r["status_referencia_observacional"] == "referencia_observacional_forte_somente_revisao" for r in ref_rows)
    rows = [
        ("geometria_oficial_de_ocorrencia_encontrada", _bool(geom_ok), "true", geom_ok, "ponto/poligono/bbox/footprint oficial de ocorrencia"),
        ("patch_link_forte_criado", str(strong_links), ">=1", strong_links >= 1, "overlay forte com patch oficial"),
        ("features_diretas_disponiveis", _bool(feats_ok), "true", feats_ok, "fisico/espectral/chuva por patch"),
        ("curitiba_segunda_regiao_forte", _bool(forte), "true", forte, "referencia forte somente revisao em Curitiba"),
        ("ground_truth_zero", "0", "0", True, "sem ground truth"),
        ("trainable_zero", "0", "0", True, "sem treino"),
        ("score_v7_zero", "0", "0", True, "sem score_v7"),
        ("score_v6_intacto", _bool(not _score_v6_changed()), "true", not _score_v6_changed(), "score_v6 nunca alterado"),
    ]
    return [{"criterio": c, "valor_observado": v, "limiar": t, "passou": _bool(p), "observacao": o} for c, v, t, p, o in rows]


def gate_17b_rows(curitiba_forte):
    # herdado: Recife 1 evento/1 regiao/5 vinculos fortes
    eventos = 1 + (1 if curitiba_forte else 0)
    regioes = 1 + (1 if curitiba_forte else 0)
    vinculos = 5  # Curitiba forte adicionaria vinculos, mas so quando features completas
    rows = [
        ("minimo_3_eventos_distintos_fortes", str(eventos), "3", eventos >= 3, "eventos com geometria forte"),
        ("minimo_2_regioes_fortes", str(regioes), "2", regioes >= 2, "regioes com referencia forte"),
        ("minimo_20_patch_links_fortes", str(vinculos), "20", vinculos >= 20, "patch-links fortes review-only"),
        ("separacao_temporal_possivel", "false", "true", False, "amostra forte ainda de um unico evento"),
        ("features_diretas_suficientes", "true", "true", True, "Recife com features completas"),
        ("controles_nao_supervisionados", "true", "true", True, "controles seguem nao supervisionados"),
        ("ground_truth_zero", "0", "0", True, "sem ground truth"),
        ("trainable_zero", "0", "0", True, "sem treino"),
        ("score_v7_zero", "0", "0", True, "sem score_v7"),
        ("score_v6_intacto", _bool(not _score_v6_changed()), "true", not _score_v6_changed(), "score_v6 intacto"),
    ]
    return [{"criterio": c, "valor_observado": v, "limiar": t, "passou": _bool(p), "observacao": o} for c, v, t, p, o in rows]


def _final_status(norms, vinculos, features, ref_rows):
    geom_ok = any(n["_status"] in GEOM_STATUS_FORTE for n in norms)
    strong_links = any(v["vinculo_forte"] == "true" for v in vinculos)
    feats_ok = any(f["possui_fisico"] == "true" for f in features)
    forte = any(r["status_referencia_observacional"] == "referencia_observacional_forte_somente_revisao" for r in ref_rows)
    if forte:
        return "18C_CURITIBA_REFERENCIA_FORTE_SOMENTE_REVISAO"
    if geom_ok and (strong_links or not feats_ok):
        return "18C_CURITIBA_GEOMETRIA_RESOLVIDA_SEM_FEATURES_COMPLETAS"
    # sem geometria: sempre emitimos solicitacao formal + fila
    return "18C_CURITIBA_GEOMETRIA_AUSENTE_COM_SOLICITACAO_FORMAL"


def _status_17b(curitiba_forte, geom_ok):
    if curitiba_forte:
        return "17B_APROXIMACAO_COM_SEGUNDA_REGIAO_FORTE"
    if not geom_ok:
        return "17B_BLOQUEADO_POR_GEOMETRIA"
    return "17B_APROXIMACAO_REGIONAL_COM_REFERENCIAS_PARCIAS"


# --- Resumo / summary / preflight ------------------------------------------
def resumo_status_rows(ref_rows):
    c = Counter(r["status_referencia_observacional"] for r in ref_rows)
    return [{"status_referencia_observacional": s, "quantidade": str(c[s])} for s in STATUS_REF_ALLOWED if c.get(s, 0)]


def summary_obj(auditoria, aquisicao, norms, vinculos, features, ref_rows, fila_aq, fila_feat, final_status):
    inherited = read_json(SUMMARY_18B)
    v2ca = _v2ca_summary()
    geom_ok = any(n["_status"] in GEOM_STATUS_FORTE for n in norms)
    strong_links = sum(1 for v in vinculos if v["vinculo_forte"] == "true")
    forte = any(r["status_referencia_observacional"] == "referencia_observacional_forte_somente_revisao" for r in ref_rows)
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "herdado_18b_status_final": inherited.get("status_final_18b", ""),
        "herdado_18b_status_17b": inherited.get("status_17b", ""),
        "network_enabled": NETWORK_ENABLED,
        "auditoria_arquivos": len(auditoria),
        "v2ca_fontes_locais_auditadas": int(v2ca.get("sources_inventoried", 0) or 0),
        "v2ca_geometry_status": "CONTEXT_ONLY_NO_GEOMETRY",
        "v2ca_blocked_reason": v2ca.get("blocked_reason", "CURITIBA_GEOMETRY_OR_POINT_EVIDENCE_NOT_READY"),
        "v2ca_next_required_step": v2ca.get("next_required_step", "acquire_curitiba_event_geometry_or_point_evidence"),
        "v2ca_patch_event_bindings": int(v2ca.get("patch_event_bindings_created", 0) or 0),
        "v2ca_ready_for_overlay_count": int(v2ca.get("ready_for_overlay_count", 0) or 0),
        "prelink_region_only_cur_2022_01_15": _cur_2022_patch_prelink_count(),
        "fontes_oficiais_tentadas": len(aquisicao),
        "artefatos_leves_adquiridos": sum(1 for r in aquisicao if r["status_acesso"] == "acquired_light_artifact"),
        "geometrias_ocorrencia_resolvidas": sum(1 for n in norms if n["_status"] in GEOM_STATUS_FORTE),
        "patch_polys_disponiveis": len(load_cur_patch_polygons()),
        "patch_links_curitiba": len(vinculos),
        "patch_links_fortes_curitiba": strong_links,
        "curitiba_segunda_regiao_forte": forte,
        "geometria_encontrada": geom_ok,
        "solicitacao_formal_criada": True,
        "fila_aquisicao_externa": len(fila_aq),
        "fila_extracao_features": len(fila_feat),
        "ground_truth_true_count": 0,
        "eligible_for_training_true_count": 0,
        "score_v7_allowed_true_count": 0,
        "benchmark_17b_criado": False,
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
        "status_final_18c": final_status,
        "status_17b": _status_17b(forte, geom_ok),
        "review_only": True, "ground_truth": False, "trainable": False,
}


def _existing_18c_files():
    candidates = [
        Path("scripts/suscetibilidade/susc_18c_curitiba_geometria_common.py"),
        Path("scripts/suscetibilidade/build_susc_18c_aquisicao_geometria_oficial_curitiba.py"),
        Path("scripts/suscetibilidade/validate_susc_18c_aquisicao_geometria_oficial_curitiba.py"),
        Path("tests/suscetibilidade/test_susc_18c_aquisicao_geometria_oficial_curitiba.py"),
        SCHEMA,
        REPORT,
    ]
    candidates.extend(sorted(OUT_DATA.glob("*")))
    candidates.extend(sorted(CARDS_DIR.glob("*")))
    return [rel(p) for p in candidates if p.exists()]


def preflight_obj():
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "staged_files": _run_git(["diff", "--cached", "--name-only"]).splitlines(),
        "dirty_lines": len(_run_git(["status", "--short"]).splitlines()),
        "git_status_short": _run_git(["status", "--short"]).splitlines(),
        "arquivos_18c_existentes": _existing_18c_files(),
        "network_enabled": NETWORK_ENABLED,
        "inputs": [
            {"role": "target_pack_17c", "path": rel(TARGET_PACK_17C), "exists": TARGET_PACK_17C.exists()},
            {"role": "curitiba_execucao_18b", "path": rel(CUR_EXEC_18B), "exists": CUR_EXEC_18B.exists()},
            {"role": "matriz_18b", "path": rel(MATRIZ_18B), "exists": MATRIZ_18B.exists()},
            {"role": "summary_18b", "path": rel(SUMMARY_18B), "exists": SUMMARY_18B.exists()},
            {"role": "curitiba_prelink_v1uw", "path": rel(CUR_PRELINK), "exists": CUR_PRELINK.exists()},
            {"role": "patch_boundaries_cur", "path": rel(PATCH_BOUNDARIES_DIR), "exists": PATCH_BOUNDARIES_DIR.exists()},
            {"role": "curitiba_geom_inventory_v2ca", "path": rel(CUR_GEOM_INVENTORY_V2CA), "exists": CUR_GEOM_INVENTORY_V2CA.exists()},
            {"role": "curitiba_source_inventory_v2ca", "path": rel(CUR_SOURCE_INVENTORY_V2CA), "exists": CUR_SOURCE_INVENTORY_V2CA.exists()},
            {"role": "curitiba_summary_v2ca", "path": rel(CUR_SUMMARY_V2CA), "exists": CUR_SUMMARY_V2CA.exists()},
            {"role": "ingest_response", "path": rel(INGEST_CSV), "exists": INGEST_CSV.exists()},
            {"role": "ingest_response_json", "path": rel(INGEST_JSON), "exists": INGEST_JSON.exists()},
            {"role": "ingest_response_geojson", "path": rel(INGEST_GEOJSON), "exists": INGEST_GEOJSON.exists()},
            {"role": "ingest_response_wkt", "path": rel(INGEST_WKT), "exists": INGEST_WKT.exists()},
            {"role": "score_v6", "path": rel(SCORE_V6), "exists": SCORE_V6.exists()},
            {"role": "score_v7_proibido", "path": rel(SCORE_V7), "exists": SCORE_V7.exists()},
        ],
    }


# --- Schema ----------------------------------------------------------------
def _schema():
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-18C aquisicao e resolucao de geometria oficial de ocorrencia em Curitiba",
        "type": "object",
        "properties": {
            "status_geometria": {"enum": GEOM_STATUS_ALLOWED},
            "classe_vinculo": {"enum": CLASSE_VINCULO_ALLOWED},
            "status_referencia_observacional": {"enum": STATUS_REF_ALLOWED},
            "uso_permitido": {"enum": USO_PERMITIDO_ALLOWED},
            "status_acesso": {"enum": ACESSO_ALLOWED},
            "status_final_18c": {"enum": GATE_FINAL_ALLOWED},
            "status_17b": {"enum": STATUS_17B_ALLOWED},
            "ground_truth": {"const": "false"}, "eligible_for_training": {"const": "false"},
            "score_v7_allowed": {"const": "false"}, "review_only": {"const": "true"},
        },
        "enums": {
            "status_geometria": GEOM_STATUS_ALLOWED, "classe_vinculo": CLASSE_VINCULO_ALLOWED,
            "status_referencia_observacional": STATUS_REF_ALLOWED, "uso_permitido": USO_PERMITIDO_ALLOWED,
            "status_acesso": ACESSO_ALLOWED, "status_final_18c": GATE_FINAL_ALLOWED, "status_17b": STATUS_17B_ALLOWED,
        },
        "hard_defaults": HARD_DEFAULTS,
        "ingestion_path": rel(INGEST_CSV),
        "ingestion_formats": [rel(INGEST_CSV), rel(INGEST_JSON), rel(INGEST_GEOJSON), rel(INGEST_WKT)],
        "sections": [
            "auditoria_local", "geometria_normalizada", "vinculo_evento_patch",
            "referencia_observacional", "gates",
        ],
        "network_opt_in_env": "SUSC_18C_NETWORK",
        "required_outputs": [rel(p) for p in REQUIRED_OUTPUTS],
    }


def write_schema():
    write_json(SCHEMA, _schema())


# --- Cartoes ---------------------------------------------------------------
def card_markdown(ref, norms, vinculos):
    cid = ref["candidate_event_id"]
    geom = next((n for n in norms if n["candidate_event_id"] == cid and n["_status"] in GEOM_STATUS_FORTE), None)
    links = [v for v in vinculos if v["candidate_event_id"] == cid and v["vinculo_forte"] == "true"]
    return f"""# Cartao Curitiba {ref['item_id']}

## Evento

- `{cid}` | data {ref['data_evento']} | fenomeno {ref['tipo_fenomeno']}

## Fonte

- {ref['fonte']} (autoridade {ref['autoridade_fonte']})

## Auditoria local

- Geometria de ocorrencia local: {'encontrada' if geom else 'ausente (v2ca ja concluira CONTEXT_ONLY_NO_GEOMETRY)'}
- Patches oficiais disponiveis para overlay: {len(load_cur_patch_polygons())} poligonos

## Aquisicao tentada

- Fontes oficiais leves: GeoCuritiba, IPPUC, Defesa Civil, Dados Abertos (offline-first por padrao)

## Geometria

- {'geometria de ocorrencia resolvida: ' + geom['geometry_id'] if geom else 'sem geometria de ocorrencia; endereco/bairro/centroide nao sao usados'}

## Vinculo patch

- {'vinculos fortes: ' + ', '.join(v['patch_id'] for v in links) if links else 'sem vinculo forte (sem geometria de ocorrencia)'}

## Features

- fisico={ref['possui_fisico']}; espectral={ref['possui_espectral']}; chuva={ref['possui_chuva']}

## Bloqueios

- Status: **{ref['status_referencia_observacional']}**

## Solicitacao externa

- Pacote formal emitido: oficio + schema de ingestao + planilha modelo (`solicitacao_geometria_ocorrencia_curitiba.md`)

## Impacto no 17B

{('Curitiba com referencia forte tornaria o 17B uma aproximacao com segunda regiao forte.' if geom and links
  else 'Sem geometria de ocorrencia, Curitiba ainda nao contribui com vinculo forte; 17B bloqueado por geometria.')}

## Por que nao e ground truth

Referencia observacional de ocorrencia somente revisao; nao constitui verdade de referencia.

## Por que nao e treinavel

Sem rotulo validado; referencia review-only nunca alimenta treino supervisionado.

## Por que nao cria score_v7

Nenhum score oficial ou score_v7 e criado; o score_v6 permanece intacto.
"""


# --- Relatorio -------------------------------------------------------------
def report_markdown(summary, gate_cur, gate17b, ref_rows):
    gate_cur_lines = "\n".join(f"- {r['criterio']}: passou={r['passou']} ({r['valor_observado']} / {r['limiar']})" for r in gate_cur)
    gate17b_lines = "\n".join(f"- {r['criterio']}: passou={r['passou']} ({r['valor_observado']} / {r['limiar']})" for r in gate17b)
    ref_lines = "\n".join(
        f"- {r['item_id']} {r['candidate_event_id']} ({r['data_evento']}): {r['status_referencia_observacional']}"
        for r in ref_rows)
    return f"""# SUSC-18C Aquisicao e resolucao de geometria oficial de ocorrencia em Curitiba

## Estado herdado do 18B

- Branch: `{summary['branch']}`
- HEAD: `{summary['head']}`
- Status final 18B herdado: {summary['herdado_18b_status_final']}
- Status 17B herdado: {summary['herdado_18b_status_17b']}
- `score_v6` alterado: {summary['score_v6_changed']}
- `score_v7` criado: {summary['score_v7_created']}

## Por que Curitiba e o caminho mais curto para a segunda regiao forte

O evento de janeiro de 2022 (CUR_2022_01_15 / S17C_REF_0060) ja tem data oficial, fonte administrativa e
54 patches candidatos region-only identificados. Falta apenas a geometria de ocorrencia para executar o
overlay e, com features, virar referencia forte. Nenhum outro candidato regional esta tao proximo.

## Auditoria local

Auditoria consolidada de {summary['auditoria_arquivos']} entradas. O inventario de geometria de evento do trabalho de
protocolo anterior (v2ca) ja havia concluido, sem inventar nada, CONTEXT_ONLY_NO_GEOMETRY /
NO_LOCAL_GEOMETRY_OR_POINTS para os eventos de Curitiba. Nao ha ponto nem poligono de ocorrencia local.
O v2ca auditou {summary['v2ca_fontes_locais_auditadas']} fontes locais, criou
{summary['v2ca_patch_event_bindings']} bindings candidatos e manteve ready_for_overlay_count=
{summary['v2ca_ready_for_overlay_count']}. Ha, porem, {summary['patch_polys_disponiveis']} poligonos
oficiais de patch (EPSG:4326), prontos para o overlay assim que a geometria de ocorrencia chegar.

## Aquisicao oficial tentada

Fontes oficiais leves candidatas: GeoCuritiba, IPPUC, Defesa Civil de Curitiba e Portal de Dados Abertos.
Aquisicao offline-first por padrao (rede desabilitada; opcional via variavel de ambiente dedicada). Artefatos
leves adquiridos nesta execucao: {summary['artefatos_leves_adquiridos']}.

## Geometrias encontradas ou ausentes

- Geometrias de ocorrencia resolvidas: {summary['geometrias_ocorrencia_resolvidas']}.
- Status v2ca herdado: {summary['v2ca_geometry_status']} / {summary['v2ca_blocked_reason']}.
- Sem geometria de ocorrencia, endereco/rua/bairro, centroide de bairro/municipio e area administrativa nao
  sao promovidos a geometria forte.

## Vinculos patch

- Patch-links de Curitiba gerados: {summary['patch_links_curitiba']} (fortes: {summary['patch_links_fortes_curitiba']}).
- Patches candidatos region-only no prelink para CUR_2022_01_15: {summary['prelink_region_only_cur_2022_01_15']}.
- Quando `geometrias_ocorrencia_resolvidas=0`, os patch-links gerados sao linhas de controle
  nao fortes: `insufficient_for_patch_link`, sem `geometry_id` real e sem `patch_id` adjudicado.
  Eles existem para manter o contrato tabular e explicar o bloqueio, nao para afirmar overlay.
- A maquinaria de overlay (poligonos reais dos patches + teste ponto-em-poligono) esta pronta e validada;
  produz vinculo forte assim que houver geometria de ocorrencia.

## Features disponiveis

- Curitiba ainda sem features diretas por patch; extracao em fila ({summary['fila_extracao_features']} tarefas),
  condicionada ao overlay.

## Referencia observacional Curitiba

{ref_lines}

## Solicitacao formal

Pacote formal executavel emitido: oficio de solicitacao (`solicitacao_geometria_ocorrencia_curitiba.md`),
schema de ingestao (`schema_resposta_esperada_curitiba.json`) e planilha modelo de resposta
(`modelo_planilha_resposta_curitiba.csv`). A resposta oficial, uma vez ingerida, aciona a maquinaria completa
em uma unica execucao.

## Gate Curitiba

{gate_cur_lines}

## Gate 17B pos-18C

{gate17b_lines}

- Status 17B: **{summary['status_17b']}**
- Nenhum benchmark 17B foi criado.

## Conclusao

- O que avancou de verdade: a maquinaria completa de geometria -> overlay -> vinculo -> referencia foi
  construida e validada com os poligonos oficiais reais dos patches de Curitiba, e um pacote formal
  executavel de aquisicao foi emitido. O bloqueio de Curitiba esta reduzido a um unico insumo: a geometria
  oficial de ocorrencia.
- O que segue bloqueado: sem esse insumo, Curitiba ainda nao vira segunda regiao forte; o 17B permanece
  bloqueado por geometria.
- Proxima acao pesada: **SUSC-18D**: protocolar a solicitacao formal e, ao receber a camada oficial de
  ocorrencia, ingeri-la para acionar overlay e extracao de features, consolidando Curitiba como segunda
  regiao forte somente revisao.

## Garantias

- ground_truth=false; eligible_for_training=false; score_v7_allowed=false; review_only=true.
- score_v6 intacto ({summary['score_v6_changed']} para alterado); nenhum benchmark 17B criado; nenhuma coordenada inventada.
"""


# --- Orquestracao ----------------------------------------------------------
def build_all():
    _require_inputs()
    ensure_dir(OUT_DATA)
    ensure_dir(CARDS_DIR)
    ensure_dir(OUT_REPORTS)
    ensure_dir(SCHEMAS)
    write_schema()

    auditoria = auditoria_rows()
    aquisicao = aquisicao_rows()
    manifesto = manifesto_rows(aquisicao)
    fila_aq = fila_aquisicao_rows()
    norms = build_geometries()
    geom_norm = geom_norm_rows(norms)
    n_geojson = write_geojson(norms)
    vinculos, _patches = build_vinculos(norms)
    features = features_rows(vinculos)
    fila_feat = fila_features_rows(vinculos)
    ref_rows = referencia_rows(norms, vinculos, features)
    final_status = _final_status(norms, vinculos, features, ref_rows)
    curitiba_forte = any(r["status_referencia_observacional"] == "referencia_observacional_forte_somente_revisao" for r in ref_rows)
    gate_cur = gate_curitiba_rows(norms, vinculos, features, ref_rows)
    gate17b = gate_17b_rows(curitiba_forte)
    summary = summary_obj(auditoria, aquisicao, norms, vinculos, features, ref_rows, fila_aq, fila_feat, final_status)

    write_json(PREFLIGHT_JSON, preflight_obj())
    write_csv(AUDITORIA, auditoria, AUDITORIA_FIELDS)
    write_csv(AQUISICAO, aquisicao, AQUISICAO_FIELDS)
    write_csv(MANIFESTO, manifesto, MANIFESTO_FIELDS)
    write_csv(FILA_AQUISICAO, fila_aq, FILA_AQ_FIELDS)
    write_csv(GEOM_NORM, geom_norm, GEOM_NORM_FIELDS)
    write_csv(VINCULOS, vinculos, VINCULOS_FIELDS)
    write_csv(FEATURES, features, FEATURES_FIELDS)
    write_csv(FILA_FEATURES, fila_feat, FILA_FEATURES_FIELDS)
    write_csv(REFERENCIA, ref_rows, REFERENCIA_FIELDS)
    write_markdown(SOLICITACAO_MD, solicitacao_markdown())
    write_json(SCHEMA_RESPOSTA, schema_resposta_obj())
    write_csv(MODELO_RESPOSTA, modelo_resposta_rows(), MODELO_RESPOSTA_FIELDS)
    write_csv(GATE_CUR, gate_cur, GATE_FIELDS)
    write_csv(GATE_17B, gate17b, GATE_FIELDS)
    write_csv(RESUMO_STATUS, resumo_status_rows(ref_rows), RESUMO_STATUS_FIELDS)
    write_json(SUMMARY, summary)
    for ref in ref_rows:
        write_markdown(CARDS_DIR / f"{ref['item_id']}.md", card_markdown(ref, norms, vinculos))
    write_markdown(REPORT, report_markdown(summary, gate_cur, gate17b, ref_rows))
    summary["_n_geojson"] = n_geojson
    return summary


# --- Validacao -------------------------------------------------------------
def _public_output_paths():
    paths = [REPORT, SOLICITACAO_MD]
    paths.extend(OUT_DATA.glob("*.csv"))
    paths.extend(OUT_DATA.glob("*.json"))
    paths.extend(OUT_DATA.glob("*.md"))
    paths.extend(CARDS_DIR.glob("*.md"))
    return [p for p in paths if p.exists()]


def public_text_violations_text(text):
    return sorted({m.group(0) for m in PUBLIC_FORBIDDEN_RE.finditer(text)})


def public_text_violations_files(paths=None):
    errors = []
    for path in paths or _public_output_paths():
        hits = public_text_violations_text(path.read_text(encoding="utf-8", errors="ignore"))
        if hits:
            errors.append(f"{rel(path)}:{','.join(hits)}")
    return errors


def validate_referencia_rows(rows):
    errors = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("item_id", f"row{idx}")
        for field in ("ground_truth", "eligible_for_training", "score_v7_allowed"):
            if row.get(field) == "true":
                errors.append(f"{rid}:{field}_true_proibido")
        if row.get("review_only") != "true":
            errors.append(f"{rid}:review_only_deve_ser_true")
        if row.get("status_referencia_observacional") not in STATUS_REF_ALLOWED:
            errors.append(f"{rid}:status_fora_enum")
        if row.get("uso_permitido") not in USO_PERMITIDO_ALLOWED:
            errors.append(f"{rid}:uso_fora_enum")
        if row.get("classe_vinculo") not in CLASSE_VINCULO_ALLOWED:
            errors.append(f"{rid}:classe_vinculo_fora_enum")
        forte = row.get("status_referencia_observacional") == "referencia_observacional_forte_somente_revisao"
        if forte and row.get("classe_vinculo") == "same_region_only":
            errors.append(f"{rid}:forte_sem_vinculo_forte")
        if forte and row.get("geometry_id") in {"", "not_available"}:
            errors.append(f"{rid}:forte_sem_geometry_id")
        if row.get("justificativa_tecnica", "").strip() == "":
            errors.append(f"{rid}:justificativa_vazia")
    return errors


def validate_vinculos_rows(rows):
    errors = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("patch_link_id", f"link{idx}")
        if row.get("classe_vinculo") not in CLASSE_VINCULO_ALLOWED:
            errors.append(f"{rid}:classe_vinculo_fora_enum")
        classe_forte = row.get("classe_vinculo") in CLASSE_VINCULO_FORTE
        geometry_absent = row.get("geometry_id") in {"", "not_available"}
        if classe_forte and geometry_absent:
            errors.append(f"{rid}:classe_forte_sem_geometry_id")
        if classe_forte and row.get("patch_id") in {"", "not_available"}:
            errors.append(f"{rid}:classe_forte_sem_patch_id")
        if row.get("vinculo_forte") == "true":
            if row.get("geometry_id") in {"", "not_available"}:
                errors.append(f"{rid}:forte_sem_geometry_id")
            if row.get("patch_id") in {"", "not_available"}:
                errors.append(f"{rid}:forte_sem_patch_id")
        if geometry_absent and row.get("vinculo_forte") == "true":
            errors.append(f"{rid}:sem_geometria_marcado_forte")
        if geometry_absent and row.get("classe_vinculo") in CLASSE_VINCULO_FORTE:
            errors.append(f"{rid}:sem_geometria_com_classe_forte")
    return errors


def validate_geom_norm_rows(rows):
    errors = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("geometry_id", f"g{idx}")
        st = row.get("status_geometria", "")
        if st not in GEOM_STATUS_ALLOWED:
            errors.append(f"{rid}:status_geometria_fora_enum:{st}")
        # geometria forte obriga geometria_de_ocorrencia=true (nunca area administrativa/textual)
        if st in GEOM_STATUS_FORTE and row.get("geometria_de_ocorrencia") != "true":
            errors.append(f"{rid}:geometria_forte_nao_de_ocorrencia")
    return errors


def validate_features_rows(rows):
    errors = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("patch_link_id", f"f{idx}")
        for fam, fonte in (("possui_fisico", "fonte_fisico"), ("possui_espectral", "fonte_espectral"), ("possui_chuva", "fonte_chuva")):
            if row.get(fam) == "true" and (not row.get(fonte, "").strip() or row.get(fonte) == "ausente"):
                errors.append(f"{rid}:{fam}_sem_fonte")
    return errors


def validate():
    summary = build_all()
    errors = []
    ref = _read(REFERENCIA)
    if not ref:
        errors.append("referencia_curitiba_vazia")
    errors.extend(validate_referencia_rows(ref))
    errors.extend(validate_vinculos_rows(_read(VINCULOS)))
    errors.extend(validate_geom_norm_rows(_read(GEOM_NORM)))
    errors.extend(validate_features_rows(_read(FEATURES)))
    errors.extend(public_text_violations_files())

    geom_ok = summary["geometria_encontrada"]
    if summary["geometrias_ocorrencia_resolvidas"] == 0 and summary["patch_links_fortes_curitiba"] != 0:
        errors.append("geom_resolvida_zero_com_patch_link_forte")
    # se geometria nao existir, solicitacao formal deve existir
    if not geom_ok and not (SOLICITACAO_MD.exists() and SCHEMA_RESPOSTA.exists() and MODELO_RESPOSTA.exists()):
        errors.append("solicitacao_externa_ausente_sem_geometria")
    if summary["benchmark_17b_criado"]:
        errors.append("benchmark_17b_criado_proibido")
    if summary["status_final_18c"] not in GATE_FINAL_ALLOWED:
        errors.append(f"status_final_fora_enum:{summary['status_final_18c']}")
    if summary["status_17b"] not in STATUS_17B_ALLOWED:
        errors.append(f"status_17b_fora_enum:{summary['status_17b']}")
    if summary["status_final_18c"] == "18C_BLOQUEADO_FAIL_CLOSED":
        errors.append("sem_avanco_funcional_entregue")
    for field in ("ground_truth_true_count", "eligible_for_training_true_count", "score_v7_allowed_true_count"):
        if summary[field] != 0:
            errors.append(f"{field}_nonzero")
    if summary["score_v6_changed"]:
        errors.append("score_v6_changed_forbidden")
    if summary["score_v7_created"]:
        errors.append("score_v7_created_forbidden")

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(
        "18C aquisicao geometria oficial Curitiba validada: "
        f"geom_resolvida={summary['geometrias_ocorrencia_resolvidas']} "
        f"patch_links={summary['patch_links_curitiba']} fortes={summary['patch_links_fortes_curitiba']} "
        f"patch_polys={summary['patch_polys_disponiveis']} status18C={summary['status_final_18c']} "
        f"status17B={summary['status_17b']}"
    )
    return 0


def run_all():
    summary = build_all()
    print(
        "18C aquisicao geometria oficial Curitiba gerada: "
        f"auditoria={summary['auditoria_arquivos']} geom_resolvida={summary['geometrias_ocorrencia_resolvidas']} "
        f"patch_polys={summary['patch_polys_disponiveis']} patch_links={summary['patch_links_curitiba']} "
        f"solicitacao_formal={summary['solicitacao_formal_criada']} "
        f"status18C={summary['status_final_18c']} status17B={summary['status_17b']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_all())
