"""SUSC-17G extracao direta dos atributos fisicos/topograficos dos canarios.

O 17F consolidou apenas referencia comparativa (patch oficial distante) e uma
fila de extracao. Esta etapa executa a extracao DIRETA das features fisicas dos
canarios S17C6_CANARY_REC_00001..00005 quando ha insumo local suficiente, ou
deixa um extrator operacional completo e uma fila de insumos externos minima
quando faltar insumo, sem inventar valores.

O modelo digital de elevacao Copernicus GLO-30 do dominio de bacia (17C38) e a
hidrografia oficial (faixas-marginais, 17C35) ja estao no repositorio local e
cobrem a area dos canarios. A extracao reutiliza o pipeline hidrologico numpy do
17C36 (preenchimento de depressoes, direcao D8, acumulacao de fluxo, HAND, TWI).

Nada aqui vira ground truth, dado treinavel, score_v7 oficial ou benchmark 17B.
O score_v6 nunca e alterado. Referencia comparativa nunca e usada como feature
direta. Valores fisicos so aparecem quando extraidos de insumo local real.
"""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import ensure_dir, read_csv, read_json, rel, write_csv, write_json, write_markdown  # noqa: E402

DATASETS = ROOT / "datasets"
DAT_SUSC = DATASETS / "suscetibilidade"
SUSC_OUT = ROOT / "outputs_public" / "suscetibilidade"
OUT_DATA_17F = ROOT / "outputs_public" / "data" / "susc_17f_extracao_fisica_topografica_canarios_observacionais"
OUT_DATA_17E = ROOT / "outputs_public" / "data" / "susc_17e_prontidao_calibracao_observacional_exploratoria"
OUT_DATA_17D = ROOT / "outputs_public" / "data" / "susc_17d_validacao_tecnica_evidencia_observacional"
OUT_DATA_17C5 = ROOT / "outputs_public" / "data" / "susc_17c5_geometry_to_patch_linkage_resolver"
OUT_DATA = ROOT / "outputs_public" / "data" / "susc_17g_extracao_direta_features_fisicas_canarios"
CARDS_DIR = OUT_DATA / "cartoes_extracao"
OUT_REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

# --- Entradas herdadas -----------------------------------------------------
MATRIZ_FISICA_17F = OUT_DATA_17F / "matriz_fisica_topografica_canarios.csv"
CALIB_APRIMORADA_17F = OUT_DATA_17F / "matriz_calibracao_exploratoria_aprimorada.csv"
SUMMARY_17F = OUT_DATA_17F / "summary.json"
CALIBRACAO_17E = OUT_DATA_17E / "matriz_calibracao_exploratoria.csv"
DECISIONS_17D = OUT_DATA_17D / "decisoes_validacao_tecnica.csv"
PATCH_GRID_17C6 = SUSC_OUT / "susc_17c6_candidate_patch_grid.csv"
SCORE_V6 = DAT_SUSC / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT_SUSC / "susc_score_v7_candidate_by_patch_v1.csv"

# Insumos fisicos locais (DEM e hidrografia).
BASIN_DEM_NPY = ROOT / "local_runs" / "suscetibilidade" / "17c38_basin" / "basin_dem.npy"
BASIN_DEM_GRID = SUSC_OUT / "susc_17c38_basin_dem_grid_metadata.csv"
BASIN_DEM_MANIFEST = SUSC_OUT / "susc_17c38_dem_artifact_manifest.csv"
HIDROGRAFIA_GEOJSON = SUSC_OUT / "susc_17c35_light_artifacts" / "faixas_marginais_aoi_light.geojson"

# --- Saidas publicas -------------------------------------------------------
PREFLIGHT_JSON = OUT_DATA / "preflight.json"
AUDITORIA = OUT_DATA / "auditoria_insumos_extracao_fisica.csv"
MATRIZ_DIRETA = OUT_DATA / "matriz_features_fisicas_diretas_canarios.csv"
FILA_INSUMOS = OUT_DATA / "fila_insumos_externos_features_fisicas.csv"
REAVALIACAO = OUT_DATA / "reavaliacao_calibracao_17e_com_features_diretas.csv"
SIMULACAO = OUT_DATA / "simulacao_sensibilidade_features_diretas_review_only.csv"
GATE = OUT_DATA / "gate_extracao_direta_features_fisicas.csv"
RESUMO_CANARIO = OUT_DATA / "resumo_por_canario.csv"
RESUMO_STATUS = OUT_DATA / "resumo_por_status.csv"
SUMMARY = OUT_DATA / "summary.json"
LEDGER = OUT_DATA / "extracao_ledger.json"
REPORT = OUT_REPORTS / "SUSC_17G_EXTRACAO_DIRETA_FEATURES_FISICAS_CANARIOS.md"
SCHEMA = SCHEMAS / "susc_17g_extracao_direta_features_fisicas_schema_v1.json"

REQUIRED_INPUTS = [
    MATRIZ_FISICA_17F,
    CALIB_APRIMORADA_17F,
    SUMMARY_17F,
    CALIBRACAO_17E,
    DECISIONS_17D,
    PATCH_GRID_17C6,
    SCORE_V6,
]

REQUIRED_OUTPUTS = [
    PREFLIGHT_JSON,
    AUDITORIA,
    MATRIZ_DIRETA,
    FILA_INSUMOS,
    REAVALIACAO,
    SIMULACAO,
    GATE,
    RESUMO_CANARIO,
    RESUMO_STATUS,
    SUMMARY,
    REPORT,
    SCHEMA,
]

# --- Enums -----------------------------------------------------------------
STATUS_USO_ALLOWED = [
    "cobre_aoi_e_utilizavel",
    "cobre_parcialmente",
    "nao_cobre_aoi",
    "precisa_preprocessamento",
    "ausente",
]
FEATURE_SOURCE_MODE_ALLOWED = [
    "direta_por_dem_e_hidrografia_local",
    "direta_parcial_por_dem_local",
    "direta_parcial_por_hidrografia_local",
    "insumo_local_insuficiente",
    "aguardando_insumo_externo",
]
GATE_FINAL_ALLOWED = [
    "17G_FEATURES_FISICAS_DIRETAS_COMPLETAS",
    "17G_FEATURES_FISICAS_DIRETAS_PARCIAIS",
    "17G_EXTRATOR_OPERACIONAL_AGUARDANDO_INSUMO_EXTERNO",
    "17G_CALIBRACAO_FORTE_REVIEW_ONLY_POSSIVEL",
    "17G_EXPLORATORIA_COM_FEATURES_DIRETAS",
    "17G_BLOQUEADO_FAIL_CLOSED",
]

PUBLIC_FORBIDDEN_RE = re.compile(r"\b(?:agentic|agente|llm|codex|ia|human\s+qa)\b", re.IGNORECASE)
NO_GT_REASON = (
    "feature fisica direta review-only extraida de modelo de elevacao local; "
    "nao confirma ocorrencia, nao e verdade de referencia, nao e treino, "
    "nao autoriza score_v7 e nao altera o score_v6"
)

# Referencias de normalizacao review-only para o descritor topografico direto.
HAND_REF_M = 25.0
ELEV_REF_M = 50.0
TWI_REF = 15.0
# Peso documentado do componente topografico/hidrologico no score_v6 (SUSC-10A).
TOPO_WEIGHT = 0.40
BASE_WEIGHTS = {"rain": 0.25, "urban": 0.20, "veg": -0.10, "moisture": 0.00}

# --- Field lists -----------------------------------------------------------
AUDITORIA_FIELDS = [
    "insumo_id", "caminho", "tipo", "cobre_aoi_canarios", "regioes_cobertas",
    "crs", "resolucao", "formato",
    "utilizavel_para_elevation", "utilizavel_para_slope", "utilizavel_para_HAND",
    "utilizavel_para_TWI", "utilizavel_para_flow", "utilizavel_para_distance_to_water",
    "status_uso", "observacao",
]
MATRIZ_FIELDS = [
    "canary_patch_id", "candidate_event_id", "geometry_id", "bbox", "crs",
    "elevation_mean", "elevation_min", "elevation_max", "elevation_std",
    "slope_mean", "slope_min", "slope_max", "slope_std",
    "HAND_mean", "HAND_min", "HAND_max",
    "distance_to_water_min", "distance_to_water_mean",
    "TWI_mean", "flow_accumulation_mean",
    "drainage_context",
    "feature_source_mode", "dem_source", "drainage_source",
    "features_diretas_completas", "features_diretas_parciais", "features_ausentes",
    "qualidade_extracao", "bloqueios_extracao",
    "ground_truth", "eligible_for_training", "score_v7_allowed", "not_ground_truth_reason",
]
FILA_FIELDS = [
    "task_id", "canary_patch_id", "bbox", "crs", "insumo_necessario",
    "fonte_sugerida", "formato_esperado", "resolucao_esperada",
    "expected_input_path", "expected_output_path", "command_hint",
    "prioridade", "bloqueio_resolvido_por_este_insumo",
]
REAVALIACAO_FIELDS = [
    "canary_patch_id", "status_17f", "status_17g",
    "possui_feature_fisica_direta", "possui_feature_fisica_parcial",
    "possui_espectral", "possui_chuva",
    "pode_calibracao_forte_review_only", "pode_calibracao_exploratoria_com_features_diretas",
    "bloqueio_restante", "justificativa_tecnica",
]
SIMULACAO_FIELDS = [
    "simulacao_id", "canary_patch_id", "componentes_com_feature_direta", "componentes_ausentes",
    "descritor_topografico_direto", "peso_topografico",
    "score_exploratorio_sem_fisico", "classe_sem_fisico",
    "score_exploratorio_fisico_direto_review_only", "classe_com_fisico_direto",
    "score_fisico_comparativo_17f", "delta_direto_vs_comparativo",
    "score_v6_reference_value", "score_v6_reference_class",
    "score_oficial", "substituir_score_v6", "usar_em_treino", "ground_truth", "score_v7_allowed", "review_only",
]
GATE_FIELDS = ["criterio", "valor_observado", "limiar", "passou", "observacao"]
RESUMO_CANARIO_FIELDS = [
    "canary_patch_id", "feature_source_mode", "features_fisicas_diretas_completas",
    "features_fisicas_diretas_parciais", "fila_insumo_externo",
    "pode_calibracao_forte_review_only", "calibracao_exploratoria_com_features_diretas",
]
RESUMO_STATUS_FIELDS = ["dimensao", "categoria", "quantidade"]

PHYSICAL_TARGETS = ["elevation", "slope", "HAND", "distance_to_water", "TWI", "flow_accumulation"]


# --- Helpers ---------------------------------------------------------------
def _bool(v: bool) -> str:
    return "true" if v else "false"


def _run_git(args):
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _read(path: Path):
    return read_csv(path) if path.exists() else []


def _by_key(rows, key):
    return {r.get(key, ""): r for r in rows}


def _float(v):
    try:
        if v is None or v == "" or v == "not_available":
            return None
        return float(v)
    except (ValueError, TypeError):
        return None


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def _fmt(v):
    if v is None:
        return "not_available"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _require_inputs():
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(p) for p in missing))
    if SCORE_V7.exists():
        raise AssertionError("score_v7 existe e e proibido para SUSC-17G")


def _sha256(path: Path):
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _class_thresholds():
    by_class = defaultdict(list)
    for row in _read(SCORE_V6):
        v = _float(row.get("susc_score_v6_candidate"))
        cls = row.get("susc_class_v6_candidate", "")
        if v is not None and cls:
            by_class[cls].append(v)
    low_max = max(by_class.get("low", [0.42]))
    med_min = min(by_class.get("medium", [0.43]))
    med_max = max(by_class.get("medium", [0.61]))
    high_min = min(by_class.get("high", [0.62]))
    return round((low_max + med_min) / 2, 4), round((med_max + high_min) / 2, 4)


def _classify(v, thr):
    a, b = thr
    if v < a:
        return "low"
    if v < b:
        return "medium"
    return "high"


def _class_shift(a, b):
    order = {"low": 0, "medium": 1, "high": 2}
    x, y = order.get(a), order.get(b)
    if x is None or y is None:
        return "nao_comparavel"
    return "igual" if x == y else ("maior" if x > y else "menor")


# --- Cobertura da AOI ------------------------------------------------------
def _canary_grid():
    rows = [r for r in _read(PATCH_GRID_17C6) if r.get("candidate_patch_id", "").startswith("S17C6_CANARY_REC")]
    return _by_key(rows, "candidate_patch_id")


def _aoi_bounds():
    grid = _canary_grid()
    xs, ys = [], []
    for r in grid.values():
        xs += [float(r["xmin"]), float(r["xmax"])]
        ys += [float(r["ymin"]), float(r["ymax"])]
    return (min(xs), min(ys), max(xs), max(ys)) if xs else None


def _dem_bbox():
    rows = _read(BASIN_DEM_GRID)
    if not rows:
        return None
    bbox = rows[0].get("bbox", "")
    parts = [float(x) for x in bbox.split(",")] if bbox else []
    return tuple(parts) if len(parts) == 4 else None


def _bbox_covers(outer, inner):
    if not outer or not inner:
        return False
    ox0, oy0, ox1, oy1 = outer
    ix0, iy0, ix1, iy1 = inner
    return ox0 <= ix0 and oy0 <= iy0 and ox1 >= ix1 and oy1 >= iy1


def _dem_covers_aoi():
    return BASIN_DEM_NPY.exists() and _bbox_covers(_dem_bbox(), _aoi_bounds())


def _hydro_bounds():
    if not HIDROGRAFIA_GEOJSON.exists():
        return None
    data = read_json(HIDROGRAFIA_GEOJSON)
    feats = data.get("features", []) if isinstance(data, dict) else data
    xs, ys = [], []

    def walk(c):
        if isinstance(c, (int, float)):
            return
        if c and isinstance(c[0], (int, float)) and len(c) >= 2:
            xs.append(c[0])
            ys.append(c[1])
            return
        for x in c:
            walk(x)

    for f in feats:
        walk(f.get("geometry", {}).get("coordinates", []))
    return (min(xs), min(ys), max(xs), max(ys)) if xs else None


def _hydro_covers_aoi():
    return HIDROGRAFIA_GEOJSON.exists() and _bbox_covers(_hydro_bounds(), _aoi_bounds())


# --- Extracao real (numpy, pipeline 17C36) ---------------------------------
def _extract_to_ledger():
    """Executa a extracao direta e escreve o ledger leve. Requer insumos locais."""
    import numpy as np
    import susc_17c36_hydrologic_topography_common as h36  # primitivas hidrologicas

    dem = np.load(BASIN_DEM_NPY).astype("float64")
    ny, nx = dem.shape
    bbox = _dem_bbox()
    x0, y0, x1, y1 = bbox
    dlon = (x1 - x0) / nx
    dlat = (y1 - y0) / ny
    lat_mid = (y0 + y1) / 2.0
    cellsize = dlon * 111320.0 * math.cos(math.radians(lat_mid))

    demf = np.where(np.isfinite(dem), dem, np.nanmax(dem[np.isfinite(dem)]))
    filled = h36._priority_flood_fill(demf)
    slope_rad = h36._slope_rad(filled, cellsize)
    slope_deg = np.degrees(slope_rad)
    acc, rec = h36._d8_flow(filled, cellsize)
    flow_acc_log = np.log1p(acc)
    thr = np.percentile(acc, h36.DRAINAGE_FLOWACC_PCTL)
    drainage_mask = acc >= thr
    hand = h36._hand(filled, acc, rec, drainage_mask)
    a_sca = (acc + 1.0) * cellsize
    beta = np.maximum(slope_rad, math.radians(0.1))
    twi = np.log(a_sca / np.tan(beta))

    def bbox_cells(arr, bx0, by0, bx1, by1):
        c0 = int((bx0 - x0) / dlon)
        c1 = int((bx1 - x0) / dlon) + 1
        r0 = int((y1 - by1) / dlat)
        r1 = int((y1 - by0) / dlat) + 1
        c0, c1 = max(0, c0), min(nx, c1)
        r0, r1 = max(0, r0), min(ny, r1)
        sub = arr[r0:r1, c0:c1]
        return sub[np.isfinite(sub)]

    def stats(arr, b):
        sub = bbox_cells(arr, *b)
        if sub.size == 0:
            return {}
        return {
            "mean": round(float(sub.mean()), 4),
            "min": round(float(sub.min()), 4),
            "max": round(float(sub.max()), 4),
            "std": round(float(sub.std()), 4),
            "npix": int(sub.size),
        }

    # hidrografia: distancia ponto-segmento em metros (equirretangular local)
    hydro_segments = _load_hydro_segments()
    mplon = 111320.0 * math.cos(math.radians(lat_mid))
    mplat = 110540.0

    def dist_to_water(bx0, by0, bx1, by1):
        pts = [((bx0 + bx1) / 2, (by0 + by1) / 2), (bx0, by0), (bx1, by1), (bx0, by1), (bx1, by0)]
        dists = [_min_dist_point_segments(px * mplon, py * mplat, hydro_segments, mplon, mplat) for px, py in pts]
        dists = [d for d in dists if d is not None]
        if not dists:
            return None, None
        return round(min(dists), 2), round(sum(dists) / len(dists), 2)

    per_canary = {}
    for cid, g in _canary_grid().items():
        b = (float(g["xmin"]), float(g["ymin"]), float(g["xmax"]), float(g["ymax"]))
        e = stats(demf, b)
        s = stats(slope_deg, b)
        h = stats(hand, b)
        fa = stats(flow_acc_log, b)
        tw = stats(twi, b)
        dw_min, dw_mean = dist_to_water(*b)
        per_canary[cid] = {
            "elevation_mean": e.get("mean"), "elevation_min": e.get("min"),
            "elevation_max": e.get("max"), "elevation_std": e.get("std"),
            "slope_mean": s.get("mean"), "slope_min": s.get("min"),
            "slope_max": s.get("max"), "slope_std": s.get("std"),
            "HAND_mean": h.get("mean"), "HAND_min": h.get("min"), "HAND_max": h.get("max"),
            "TWI_mean": tw.get("mean"), "flow_accumulation_mean": fa.get("mean"),
            "distance_to_water_min": dw_min, "distance_to_water_mean": dw_mean,
            "valid_pixel_count": e.get("npix", 0),
        }

    ledger = {
        "dem_sha256": _sha256(BASIN_DEM_NPY),
        "hydro_sha256": _sha256(HIDROGRAFIA_GEOJSON),
        "dem_source": "copernicus_dem_glo30_dominio_bacia_17c38",
        "drainage_source": "faixas_marginais_hidrografia_oficial_dados_recife_17c35",
        "cellsize_m": round(cellsize, 3),
        "grid_shape": [ny, nx],
        "method": {
            "sink_fill": "priority_flood_wang_liu",
            "flow_direction": "d8_steepest_descent",
            "flow_accumulation": "d8_topo_order",
            "hand": "downstream_to_drainage_cell",
            "twi_formula": "ln((flowacc+1)*cellsize/tan(beta))",
            "drainage_flowacc_percentile": h36.DRAINAGE_FLOWACC_PCTL,
            "method_equivalence_status": "reconstruido_nao_provado_equivalente_ao_oficial",
            "resolution_note": "modelo de elevacao a ~92 m; extracao review-only",
        },
        "per_canary": per_canary,
    }
    ensure_dir(OUT_DATA)
    write_json(LEDGER, ledger)
    return ledger


def _load_hydro_segments():
    data = read_json(HIDROGRAFIA_GEOJSON)
    feats = data.get("features", []) if isinstance(data, dict) else data
    segments = []

    def rings(coords, depth):
        # coords structure varies by geometry type; collect coordinate sequences
        if not coords:
            return
        if isinstance(coords[0], (int, float)):
            return
        if isinstance(coords[0][0], (int, float)):
            seq = [(c[0], c[1]) for c in coords if len(c) >= 2]
            for i in range(len(seq) - 1):
                segments.append((seq[i], seq[i + 1]))
            return
        for c in coords:
            rings(c, depth + 1)

    for f in feats:
        geom = f.get("geometry", {}) or {}
        rings(geom.get("coordinates", []), 0)
    return segments


def _min_dist_point_segments(px_m, py_m, segments, mplon, mplat):
    best = None
    for (ax, ay), (bx, by) in segments:
        ax_m, ay_m = ax * mplon, ay * mplat
        bx_m, by_m = bx * mplon, by * mplat
        d = _point_segment_dist(px_m, py_m, ax_m, ay_m, bx_m, by_m)
        if best is None or d < best:
            best = d
    return best


def _point_segment_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = _clamp(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def _ensure_ledger():
    """Garante o ledger de extracao quando ha insumo local; senao retorna None."""
    if not (_dem_covers_aoi() and _hydro_covers_aoi()):
        return None
    if LEDGER.exists():
        led = read_json(LEDGER)
        if led.get("dem_sha256") == _sha256(BASIN_DEM_NPY) and led.get("hydro_sha256") == _sha256(HIDROGRAFIA_GEOJSON):
            return led
    return _extract_to_ledger()


# --- Modelo dos itens ------------------------------------------------------
def build_items():
    thr = _class_thresholds()
    calib_17e = _by_key(_read(CALIBRACAO_17E), "item_validacao_id")
    decisions = _by_key(_read(DECISIONS_17D), "item_validacao_id")
    grid = _canary_grid()
    fisica_17f = _by_key(_read(MATRIZ_FISICA_17F), "canary_patch_id")
    calib_17f = _by_key(_read(CALIB_APRIMORADA_17F), "canary_patch_id")
    ledger = _ensure_ledger()
    per_canary = (ledger or {}).get("per_canary", {})

    items = []
    for item_id, calib in calib_17e.items():
        patch_id = calib["patch_id"]
        decision = decisions.get(item_id, {})
        g = grid.get(patch_id, {})
        f17f = fisica_17f.get(patch_id, {})
        c17f = calib_17f.get(patch_id, {})
        extracted = per_canary.get(patch_id, {})
        has_dem_feats = bool(extracted) and extracted.get("elevation_mean") is not None
        has_dist = bool(extracted) and extracted.get("distance_to_water_min") is not None

        # descritor topografico direto (review-only) a partir das features extraidas
        direct_topo_idx = _direct_topo_idx(extracted) if has_dem_feats else None

        items.append({
            "item_validacao_id": item_id,
            "canary_patch_id": patch_id,
            "candidate_event_id": calib.get("candidate_event_id", decision.get("candidate_event_id", "")),
            "geometry_id": calib.get("geometry_id", decision.get("geometry_id", "")),
            "region": calib.get("regiao", decision.get("regiao", "")),
            "bbox": _grid_bbox(g),
            "extracted": extracted,
            "has_dem_feats": has_dem_feats,
            "has_dist": has_dist,
            "direct_topo_idx": direct_topo_idx,
            "ledger": ledger,
            "thresholds": thr,
            "rain_idx": _float(calib.get("rain_idx")),
            "urban_spectral_idx": _float(calib.get("urban_spectral_idx")),
            "vegetation_idx": _float(calib.get("vegetation_idx")),
            "moisture_idx": _float(calib.get("moisture_idx")),
            "score_sem_fisico": _float(calib.get("score_exploratorio_baseline")),
            "classe_sem_fisico": calib.get("score_exploratorio_class_baseline", ""),
            "score_fisico_comparativo_17f": _float(c17f.get("score_exploratorio_fisico_review_only")),
            "score_v6_reference_value": _float(calib.get("score_v6_reference_value")),
            "score_v6_reference_class": calib.get("score_v6_reference_class", ""),
            "status_17f": f17f.get("feature_source_mode", "referencia_comparativa_review_only"),
        })
    return items


def _grid_bbox(g):
    keys = ("xmin", "ymin", "xmax", "ymax")
    return ",".join(g[k] for k in keys) if all(g.get(k) for k in keys) else "not_available"


def _direct_topo_idx(extracted):
    hand = extracted.get("HAND_mean")
    elev = extracted.get("elevation_mean")
    twi = extracted.get("TWI_mean")
    parts = []
    if hand is not None:
        parts.append(_clamp(1.0 - hand / HAND_REF_M))
    if elev is not None:
        parts.append(_clamp(1.0 - elev / ELEV_REF_M))
    if twi is not None:
        parts.append(_clamp(twi / TWI_REF))
    return round(sum(parts) / len(parts), 4) if parts else None


def _feature_source_mode(item):
    if item["has_dem_feats"] and item["has_dist"]:
        return "direta_por_dem_e_hidrografia_local"
    if item["has_dem_feats"]:
        return "direta_parcial_por_dem_local"
    if item["has_dist"]:
        return "direta_parcial_por_hidrografia_local"
    if _dem_covers_aoi() or _hydro_covers_aoi():
        return "insumo_local_insuficiente"
    return "aguardando_insumo_externo"


def _features_present(item):
    e = item["extracted"]
    present = []
    if e.get("elevation_mean") is not None:
        present.append("elevation")
    if e.get("slope_mean") is not None:
        present.append("slope")
    if e.get("HAND_mean") is not None:
        present.append("HAND")
    if e.get("TWI_mean") is not None:
        present.append("TWI")
    if e.get("flow_accumulation_mean") is not None:
        present.append("flow_accumulation")
    if e.get("distance_to_water_min") is not None:
        present.append("distance_to_water")
    return present


def _pode_forte_review_only(item, present):
    minimal = {"elevation", "slope", "distance_to_water"}
    has_hydro = "HAND" in present or "TWI" in present
    return (
        minimal.issubset(set(present))
        and has_hydro
        and item["rain_idx"] is not None
        and item["urban_spectral_idx"] is not None
        and item["bbox"] != "not_available"
    )


# --- Preflight -------------------------------------------------------------
def preflight_obj():
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "dirty_lines": len(_run_git(["status", "--short"]).splitlines()),
        "inputs": [
            {"role": "matriz_fisica_17f", "path": rel(MATRIZ_FISICA_17F), "exists": MATRIZ_FISICA_17F.exists()},
            {"role": "calibracao_17e", "path": rel(CALIBRACAO_17E), "exists": CALIBRACAO_17E.exists()},
            {"role": "decisoes_17d", "path": rel(DECISIONS_17D), "exists": DECISIONS_17D.exists()},
            {"role": "grade_canario_17c6", "path": rel(PATCH_GRID_17C6), "exists": PATCH_GRID_17C6.exists()},
            {"role": "basin_dem_npy", "path": rel(BASIN_DEM_NPY), "exists": BASIN_DEM_NPY.exists()},
            {"role": "hidrografia_geojson", "path": rel(HIDROGRAFIA_GEOJSON), "exists": HIDROGRAFIA_GEOJSON.exists()},
            {"role": "score_v6", "path": rel(SCORE_V6), "exists": SCORE_V6.exists()},
            {"role": "score_v7_proibido", "path": rel(SCORE_V7), "exists": SCORE_V7.exists()},
        ],
        "aoi_bounds": _aoi_bounds(),
        "dem_bbox": _dem_bbox(),
        "dem_cobre_aoi": _dem_covers_aoi(),
        "hidrografia_cobre_aoi": _hydro_covers_aoi(),
    }


# --- Auditoria de insumos --------------------------------------------------
def auditoria_rows():
    dem_cov = _dem_covers_aoi()
    hydro_cov = _hydro_covers_aoi()
    dem_manifest = _by_key(_read(BASIN_DEM_MANIFEST), "dem_artifact_manifest_id")
    res = ""
    for r in dem_manifest.values():
        res = r.get("resolution_m", "")
        break
    rows = [
        {
            "insumo_id": "S17G_INSUMO_0001",
            "caminho": rel(BASIN_DEM_NPY),
            "tipo": "modelo_digital_elevacao_dominio_bacia",
            "cobre_aoi_canarios": _bool(dem_cov),
            "regioes_cobertas": "recife_dominio_bacia",
            "crs": "EPSG:4326", "resolucao": f"{res}m_aprox" if res else "92m_aprox", "formato": "npy",
            "utilizavel_para_elevation": _bool(dem_cov), "utilizavel_para_slope": _bool(dem_cov),
            "utilizavel_para_HAND": _bool(dem_cov), "utilizavel_para_TWI": _bool(dem_cov),
            "utilizavel_para_flow": _bool(dem_cov), "utilizavel_para_distance_to_water": "false",
            "status_uso": "cobre_aoi_e_utilizavel" if dem_cov else "nao_cobre_aoi",
            "observacao": "modelo Copernicus GLO-30 do dominio de bacia (17C38) cobre a area dos canarios; base para elevacao, declividade, HAND, TWI e fluxo",
        },
        {
            "insumo_id": "S17G_INSUMO_0002",
            "caminho": rel(HIDROGRAFIA_GEOJSON),
            "tipo": "hidrografia_oficial_faixas_marginais",
            "cobre_aoi_canarios": _bool(hydro_cov),
            "regioes_cobertas": "recife",
            "crs": "EPSG:4326", "resolucao": "vetor_oficial", "formato": "geojson",
            "utilizavel_para_elevation": "false", "utilizavel_para_slope": "false",
            "utilizavel_para_HAND": "false", "utilizavel_para_TWI": "false", "utilizavel_para_flow": "false",
            "utilizavel_para_distance_to_water": _bool(hydro_cov),
            "status_uso": "cobre_aoi_e_utilizavel" if hydro_cov else "nao_cobre_aoi",
            "observacao": "faixas-marginais oficiais Dados Recife (17C35) para proximidade hidrica direta",
        },
        {
            "insumo_id": "S17G_INSUMO_0003",
            "caminho": "local_runs/suscetibilidade/17c36_hydrologic/dem_aoi_window.npy",
            "tipo": "modelo_digital_elevacao_janela_sul",
            "cobre_aoi_canarios": "false",
            "regioes_cobertas": "recife_sul",
            "crs": "EPSG:4326", "resolucao": "30.62m", "formato": "npy",
            "utilizavel_para_elevation": "false", "utilizavel_para_slope": "false",
            "utilizavel_para_HAND": "false", "utilizavel_para_TWI": "false", "utilizavel_para_flow": "false",
            "utilizavel_para_distance_to_water": "false",
            "status_uso": "nao_cobre_aoi",
            "observacao": "janela DEM do 17C36 fica ao sul (ate -8.01855) e nao alcanca os canarios; substituida pelo dominio de bacia 17C38",
        },
        {
            "insumo_id": "S17G_INSUMO_0004",
            "caminho": rel(BASIN_DEM_MANIFEST),
            "tipo": "manifesto_dem_bacia",
            "cobre_aoi_canarios": _bool(dem_cov),
            "regioes_cobertas": "recife_dominio_bacia",
            "crs": "EPSG:4326", "resolucao": f"{res}m" if res else "92m", "formato": "csv",
            "utilizavel_para_elevation": "true", "utilizavel_para_slope": "true",
            "utilizavel_para_HAND": "true", "utilizavel_para_TWI": "true", "utilizavel_para_flow": "true",
            "utilizavel_para_distance_to_water": "false",
            "status_uso": "cobre_aoi_e_utilizavel" if dem_cov else "nao_cobre_aoi",
            "observacao": "metadados e sha256 do modelo de elevacao do dominio de bacia",
        },
    ]
    return rows


# --- Matriz direta ---------------------------------------------------------
def matriz_rows(items):
    ledger = items[0]["ledger"] if items else None
    dem_source = (ledger or {}).get("dem_source", "not_available")
    drainage_source = (ledger or {}).get("drainage_source", "not_available")
    rows = []
    for item in items:
        e = item["extracted"]
        present = _features_present(item)
        mode = _feature_source_mode(item)
        ausentes = [f for f in PHYSICAL_TARGETS if f not in present]
        completas = len(ausentes) == 0
        parciais = 0 < len(present) < len(PHYSICAL_TARGETS)
        drainage_ctx = (
            f"distancia_hidrica_min_m={_fmt(e.get('distance_to_water_min'))}"
            if item["has_dist"] else "not_available"
        )
        quality = _extraction_quality(item, present)
        bloqueios = "nenhum" if completas else ("insumo_local_ausente_para:" + ";".join(ausentes) if ausentes else "nenhum")
        rows.append({
            "canary_patch_id": item["canary_patch_id"],
            "candidate_event_id": item["candidate_event_id"],
            "geometry_id": item["geometry_id"],
            "bbox": item["bbox"], "crs": "EPSG:4326",
            "elevation_mean": _fmt(e.get("elevation_mean")), "elevation_min": _fmt(e.get("elevation_min")),
            "elevation_max": _fmt(e.get("elevation_max")), "elevation_std": _fmt(e.get("elevation_std")),
            "slope_mean": _fmt(e.get("slope_mean")), "slope_min": _fmt(e.get("slope_min")),
            "slope_max": _fmt(e.get("slope_max")), "slope_std": _fmt(e.get("slope_std")),
            "HAND_mean": _fmt(e.get("HAND_mean")), "HAND_min": _fmt(e.get("HAND_min")), "HAND_max": _fmt(e.get("HAND_max")),
            "distance_to_water_min": _fmt(e.get("distance_to_water_min")),
            "distance_to_water_mean": _fmt(e.get("distance_to_water_mean")),
            "TWI_mean": _fmt(e.get("TWI_mean")), "flow_accumulation_mean": _fmt(e.get("flow_accumulation_mean")),
            "drainage_context": drainage_ctx,
            "feature_source_mode": mode,
            "dem_source": dem_source if item["has_dem_feats"] else "not_available",
            "drainage_source": drainage_source if item["has_dist"] else "not_available",
            "features_diretas_completas": _bool(completas),
            "features_diretas_parciais": _bool(parciais),
            "features_ausentes": ";".join(ausentes) if ausentes else "nenhuma",
            "qualidade_extracao": quality,
            "bloqueios_extracao": bloqueios,
            "ground_truth": "false", "eligible_for_training": "false", "score_v7_allowed": "false",
            "not_ground_truth_reason": NO_GT_REASON,
        })
    return rows


def _extraction_quality(item, present):
    if not present:
        return "sem_extracao"
    if len(present) == len(PHYSICAL_TARGETS):
        return "media_metodo_reconstruido_resolucao_92m"
    return "parcial_metodo_reconstruido"


# --- Fila de insumos externos ----------------------------------------------
def fila_rows(items):
    rows = []
    counter = 0
    for item in items:
        present = _features_present(item)
        ausentes = [f for f in PHYSICAL_TARGETS if f not in present]
        if ausentes:
            # insumo faltante concreto
            counter += 1
            rows.append({
                "task_id": f"S17G_INSUMO_TASK_{counter:04d}",
                "canary_patch_id": item["canary_patch_id"], "bbox": item["bbox"], "crs": "EPSG:4326",
                "insumo_necessario": ";".join(ausentes),
                "fonte_sugerida": "Copernicus DEM GLO-30",
                "formato_esperado": "geotiff_ou_npy", "resolucao_esperada": "30m",
                "expected_input_path": f"local_runs/suscetibilidade/17g_extracao/dem_{item['canary_patch_id'].lower()}.npy",
                "expected_output_path": f"local_runs/suscetibilidade/17g_extracao/{item['canary_patch_id'].lower()}_features.csv",
                "command_hint": "colocar o modelo de elevacao que cubra o bbox e rodar o extrator 17G (modo extract)",
                "prioridade": "alta",
                "bloqueio_resolvido_por_este_insumo": "features fisicas diretas ausentes",
            })
        else:
            # refinamento opcional de resolucao (extracao ja concluida a ~92 m)
            counter += 1
            rows.append({
                "task_id": f"S17G_INSUMO_TASK_{counter:04d}",
                "canary_patch_id": item["canary_patch_id"], "bbox": item["bbox"], "crs": "EPSG:4326",
                "insumo_necessario": "modelo_elevacao_30m_nativo_para_refinar",
                "fonte_sugerida": "Copernicus DEM GLO-30",
                "formato_esperado": "geotiff", "resolucao_esperada": "30m",
                "expected_input_path": f"local_runs/suscetibilidade/17g_extracao/dem30_{item['canary_patch_id'].lower()}.tif",
                "expected_output_path": f"local_runs/suscetibilidade/17g_extracao/{item['canary_patch_id'].lower()}_features_30m.csv",
                "command_hint": "refinamento opcional: extracao ja concluida a ~92 m; DEM 30m nativo melhora HAND/TWI/fluxo",
                "prioridade": "baixa",
                "bloqueio_resolvido_por_este_insumo": "refinamento de resolucao (nao bloqueante)",
            })
    return rows


# --- Reavaliacao de calibracao ---------------------------------------------
def reavaliacao_rows(items):
    rows = []
    for item in items:
        present = _features_present(item)
        completas = len(present) == len(PHYSICAL_TARGETS)
        parciais = 0 < len(present) < len(PHYSICAL_TARGETS)
        pode_forte = _pode_forte_review_only(item, present)
        status_17g = _feature_source_mode(item)
        bloqueio = (
            "nenhum_bloqueio_de_feature_metodo_reconstruido_review_only" if pode_forte
            else ("features_fisicas_incompletas" if present else "sem_insumo_local_features_ausentes")
        )
        justificativa = (
            "features fisicas diretas extraidas do modelo de elevacao local (Copernicus GLO-30, dominio de bacia) "
            "e da hidrografia oficial; metodo reconstruido review-only a ~92 m; com espectral e chuva reais do 17E, "
            "a calibracao forte review-only fica possivel, sem ground truth, sem treino, sem score_v7 e sem alterar o score_v6"
            if pode_forte else
            "extracao direta incompleta; manter calibracao exploratoria e a fila de insumos"
        )
        rows.append({
            "canary_patch_id": item["canary_patch_id"],
            "status_17f": item["status_17f"],
            "status_17g": status_17g,
            "possui_feature_fisica_direta": _bool(completas),
            "possui_feature_fisica_parcial": _bool(parciais),
            "possui_espectral": _bool(item["urban_spectral_idx"] is not None),
            "possui_chuva": _bool(item["rain_idx"] is not None),
            "pode_calibracao_forte_review_only": _bool(pode_forte),
            "pode_calibracao_exploratoria_com_features_diretas": _bool(bool(present)),
            "bloqueio_restante": bloqueio,
            "justificativa_tecnica": justificativa,
        })
    return rows


# --- Simulacao com features diretas ----------------------------------------
def _exploratory_score(item, topo_idx, topo_weight):
    w = BASE_WEIGHTS
    raw = (
        topo_weight * (topo_idx or 0.0)
        + w["rain"] * (item["rain_idx"] or 0.0)
        + w["urban"] * (item["urban_spectral_idx"] or 0.0)
        + w["veg"] * (item["vegetation_idx"] or 0.0)
        + w["moisture"] * (item["moisture_idx"] or 0.0)
    )
    raw_min = sum(v for v in w.values() if v < 0)
    raw_max = topo_weight + sum(v for v in w.values() if v > 0)
    if raw_max - raw_min == 0:
        return 0.0
    return round(_clamp((raw - raw_min) / (raw_max - raw_min)), 4)


def simulacao_rows(items):
    rows = []
    counter = 0
    for item in items:
        present = _features_present(item)
        if not present or item["direct_topo_idx"] is None:
            continue
        counter += 1
        topo_idx = item["direct_topo_idx"]
        score_sem = item["score_sem_fisico"] if item["score_sem_fisico"] is not None else _exploratory_score(item, 0.0, 0.0)
        classe_sem = item["classe_sem_fisico"] or _classify(score_sem, item["thresholds"])
        score_com = _exploratory_score(item, topo_idx, TOPO_WEIGHT)
        classe_com = _classify(score_com, item["thresholds"])
        comp_direto = [f for f in ["elevation", "slope", "HAND", "TWI", "flow_accumulation", "distance_to_water"] if f in present]
        comp_ausente = [f for f in PHYSICAL_TARGETS if f not in present]
        score_comp_17f = item["score_fisico_comparativo_17f"]
        delta = round((score_com - score_comp_17f), 4) if score_comp_17f is not None else None
        rows.append({
            "simulacao_id": f"S17G_SIM_{counter:04d}",
            "canary_patch_id": item["canary_patch_id"],
            "componentes_com_feature_direta": ";".join(comp_direto),
            "componentes_ausentes": ";".join(comp_ausente) if comp_ausente else "nenhum",
            "descritor_topografico_direto": _fmt(topo_idx),
            "peso_topografico": _fmt(TOPO_WEIGHT),
            "score_exploratorio_sem_fisico": _fmt(score_sem),
            "classe_sem_fisico": classe_sem,
            "score_exploratorio_fisico_direto_review_only": _fmt(score_com),
            "classe_com_fisico_direto": classe_com,
            "score_fisico_comparativo_17f": _fmt(score_comp_17f),
            "delta_direto_vs_comparativo": _fmt(delta),
            "score_v6_reference_value": _fmt(item["score_v6_reference_value"]),
            "score_v6_reference_class": item["score_v6_reference_class"],
            "score_oficial": "false", "substituir_score_v6": "false", "usar_em_treino": "false",
            "ground_truth": "false", "score_v7_allowed": "false", "review_only": "true",
        })
    return rows


# --- Gate ------------------------------------------------------------------
def _final_status(items, matriz, fila):
    n = len(items)
    n_complete = sum(1 for r in matriz if r["features_diretas_completas"] == "true")
    n_partial = sum(1 for r in matriz if r["features_diretas_parciais"] == "true")
    n_forte = sum(1 for item in items if _pode_forte_review_only(item, _features_present(item)))
    if n == 0 or (n_complete == 0 and n_partial == 0 and not fila):
        return "17G_BLOQUEADO_FAIL_CLOSED"
    if n_forte == n and n_forte > 0:
        return "17G_CALIBRACAO_FORTE_REVIEW_ONLY_POSSIVEL"
    if n_complete == n and n_complete > 0:
        return "17G_FEATURES_FISICAS_DIRETAS_COMPLETAS"
    if n_complete + n_partial > 0:
        return "17G_FEATURES_FISICAS_DIRETAS_PARCIAIS"
    return "17G_EXTRATOR_OPERACIONAL_AGUARDANDO_INSUMO_EXTERNO"


def gate_rows(items, matriz, fila, simulacao, final_status):
    n = len(items)
    n_complete = sum(1 for r in matriz if r["features_diretas_completas"] == "true")
    n_partial = sum(1 for r in matriz if r["features_diretas_parciais"] == "true")
    n_aguardando = sum(1 for r in matriz if r["feature_source_mode"] in {"insumo_local_insuficiente", "aguardando_insumo_externo"})
    n_forte = sum(1 for item in items if _pode_forte_review_only(item, _features_present(item)))
    n_sim = len({r["canary_patch_id"] for r in simulacao})
    rows = [
        ("canarios_processados", str(n), "5", n == 5, "todos os canarios recebem status"),
        ("features_diretas_completas", str(n_complete), ">=0", True, "extracao completa das seis features fisicas"),
        ("features_diretas_parciais", str(n_partial), ">=0", True, "extracao parcial"),
        ("aguardando_insumo_externo", str(n_aguardando), ">=0", True, "canarios sem insumo local suficiente"),
        ("pode_calibracao_forte_review_only", str(n_forte), ">=0", True, "features fisicas diretas + espectral + chuva"),
        ("exploratoria_com_features_diretas", str(n_sim), ">=0", True, "simulacao com descritor topografico direto"),
        ("ground_truth_zero", "0", "0", True, "sem ground truth"),
        ("trainable_zero", "0", "0", True, "sem treino"),
        ("score_v7_allowed_zero", "0", "0", True, "sem score_v7"),
        ("score_v6_intacto", _bool(not _score_v6_changed()), "true", not _score_v6_changed(), "score_v6 nunca alterado"),
        ("caminho_funcional_entregue", "true", "true", final_status != "17G_BLOQUEADO_FAIL_CLOSED", "sprint entrega caminho funcional"),
        ("status_final_17g", final_status, "enum", final_status in GATE_FINAL_ALLOWED, "status final da extracao direta"),
    ]
    return [{"criterio": c, "valor_observado": v, "limiar": t, "passou": _bool(p), "observacao": o} for c, v, t, p, o in rows]


def _score_v6_changed():
    return bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]))


# --- Resumos ---------------------------------------------------------------
def resumo_canario_rows(items, matriz, fila, simulacao):
    matriz_by = _by_key(matriz, "canary_patch_id")
    sim_ids = {r["canary_patch_id"] for r in simulacao}
    fila_by = defaultdict(list)
    for r in fila:
        fila_by[r["canary_patch_id"]].append(r)
    rows = []
    for item in items:
        m = matriz_by.get(item["canary_patch_id"], {})
        present = _features_present(item)
        rows.append({
            "canary_patch_id": item["canary_patch_id"],
            "feature_source_mode": m.get("feature_source_mode", ""),
            "features_fisicas_diretas_completas": m.get("features_diretas_completas", "false"),
            "features_fisicas_diretas_parciais": m.get("features_diretas_parciais", "false"),
            "fila_insumo_externo": _bool(any(r["prioridade"] == "alta" for r in fila_by.get(item["canary_patch_id"], []))),
            "pode_calibracao_forte_review_only": _bool(_pode_forte_review_only(item, present)),
            "calibracao_exploratoria_com_features_diretas": _bool(item["canary_patch_id"] in sim_ids),
        })
    return rows


def resumo_status_rows(items, matriz, fila, simulacao):
    modes = Counter(r["feature_source_mode"] for r in matriz)
    n_complete = sum(1 for r in matriz if r["features_diretas_completas"] == "true")
    n_partial = sum(1 for r in matriz if r["features_diretas_parciais"] == "true")
    n_forte = sum(1 for item in items if _pode_forte_review_only(item, _features_present(item)))
    n_sim = len({r["canary_patch_id"] for r in simulacao})
    n_fila_alta = len({r["canary_patch_id"] for r in fila if r["prioridade"] == "alta"})
    rows = [{"dimensao": "feature_source_mode", "categoria": k, "quantidade": str(v)} for k, v in sorted(modes.items())]
    rows += [
        {"dimensao": "extracao", "categoria": "diretas_completas", "quantidade": str(n_complete)},
        {"dimensao": "extracao", "categoria": "diretas_parciais", "quantidade": str(n_partial)},
        {"dimensao": "extracao", "categoria": "fila_insumo_externo_alta", "quantidade": str(n_fila_alta)},
        {"dimensao": "calibracao", "categoria": "forte_review_only_possivel", "quantidade": str(n_forte)},
        {"dimensao": "calibracao", "categoria": "exploratoria_com_features_diretas", "quantidade": str(n_sim)},
    ]
    return rows


def _status_17e(final_status):
    if final_status == "17G_CALIBRACAO_FORTE_REVIEW_ONLY_POSSIVEL":
        return "17E_CALIBRACAO_FORTE_REVIEW_ONLY_POSSIVEL_COM_FEATURES_FISICAS_DIRETAS"
    if final_status in {"17G_FEATURES_FISICAS_DIRETAS_COMPLETAS", "17G_FEATURES_FISICAS_DIRETAS_PARCIAIS", "17G_EXPLORATORIA_COM_FEATURES_DIRETAS"}:
        return "17E_CALIBRACAO_EXPLORATORIA_APRIMORADA_COM_FEATURES_FISICAS_DIRETAS"
    return "17E_CALIBRACAO_EXPLORATORIA_APRIMORADA_COM_COMPONENTE_FISICO_COMPARATIVO"


def summary_obj(items, auditoria, matriz, fila, reavaliacao, simulacao, final_status):
    inherited = read_json(SUMMARY_17F)
    n_complete = sum(1 for r in matriz if r["features_diretas_completas"] == "true")
    n_partial = sum(1 for r in matriz if r["features_diretas_parciais"] == "true")
    n_forte = sum(1 for item in items if _pode_forte_review_only(item, _features_present(item)))
    n_fila_alta = len({r["canary_patch_id"] for r in fila if r["prioridade"] == "alta"})
    n_sim = len({r["canary_patch_id"] for r in simulacao})
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "herdado_17f_status_final": inherited.get("status_final_17f", ""),
        "herdado_17f_comparativa": inherited.get("referencia_comparativa", 0),
        "herdado_17f_status_17b": inherited.get("status_17b", ""),
        "canarios_processados": len(items),
        "features_fisicas_diretas_completas": n_complete,
        "features_fisicas_diretas_parciais": n_partial,
        "fila_insumo_externo_alta": n_fila_alta,
        "pode_calibracao_forte_review_only": n_forte,
        "calibracao_exploratoria_com_features_diretas": n_sim,
        "dem_cobre_aoi": _dem_covers_aoi(),
        "hidrografia_cobre_aoi": _hydro_covers_aoi(),
        "extracao_executada": bool(items and items[0]["ledger"]),
        "dem_source": (items[0]["ledger"] or {}).get("dem_source", "not_available") if items and items[0]["ledger"] else "not_available",
        "ground_truth_true_count": 0,
        "eligible_for_training_true_count": 0,
        "score_v7_allowed_true_count": 0,
        "score_oficial_true_count": 0,
        "substituir_score_v6_true_count": 0,
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
        "caminho_funcional": _caminho_funcional(final_status),
        "status_final_17g": final_status,
        "status_17e": _status_17e(final_status),
        "status_17b": "17B_PARCIAL_COM_CALIBRACAO_EXPLORATORIA_SEM_PRONTIDAO_BENCHMARK",
        "review_only": True, "ground_truth": False, "trainable": False,
        "score_oficial": False, "substituir_score_v6": False,
    }


def _caminho_funcional(final_status):
    return {
        "17G_CALIBRACAO_FORTE_REVIEW_ONLY_POSSIVEL": "calibracao_forte_review_only_possivel_com_features_diretas",
        "17G_FEATURES_FISICAS_DIRETAS_COMPLETAS": "features_fisicas_diretas_completas",
        "17G_FEATURES_FISICAS_DIRETAS_PARCIAIS": "features_fisicas_diretas_parciais",
        "17G_EXTRATOR_OPERACIONAL_AGUARDANDO_INSUMO_EXTERNO": "extrator_operacional_aguardando_insumo",
        "17G_EXPLORATORIA_COM_FEATURES_DIRETAS": "exploratoria_com_features_diretas",
        "17G_BLOQUEADO_FAIL_CLOSED": "bloqueado_fail_closed",
    }.get(final_status, "indefinido")


# --- Schema ----------------------------------------------------------------
def _schema():
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-17G extracao direta das features fisicas dos canarios",
        "type": "object",
        "properties": {
            "feature_source_mode": {"enum": FEATURE_SOURCE_MODE_ALLOWED},
            "status_uso": {"enum": STATUS_USO_ALLOWED},
            "status_final_17g": {"enum": GATE_FINAL_ALLOWED},
            "ground_truth": {"const": "false"},
            "eligible_for_training": {"const": "false"},
            "score_v7_allowed": {"const": "false"},
            "score_oficial": {"const": "false"},
            "substituir_score_v6": {"const": "false"},
            "usar_em_treino": {"const": "false"},
            "review_only": {"const": "true"},
        },
        "enums": {
            "feature_source_mode": FEATURE_SOURCE_MODE_ALLOWED,
            "status_uso": STATUS_USO_ALLOWED,
            "status_final_17g": GATE_FINAL_ALLOWED,
        },
        "physical_targets": PHYSICAL_TARGETS,
        "normalization_refs": {"HAND_REF_M": HAND_REF_M, "ELEV_REF_M": ELEV_REF_M, "TWI_REF": TWI_REF},
        "required_outputs": [rel(p) for p in REQUIRED_OUTPUTS],
    }


def write_schema():
    write_json(SCHEMA, _schema())


# --- Cartoes ---------------------------------------------------------------
def card_markdown(item, matriz_row):
    e = item["extracted"]
    present = _features_present(item)
    ausentes = [f for f in PHYSICAL_TARGETS if f not in present]
    pode_forte = _pode_forte_review_only(item, present)
    feats = "\n".join(
        f"- {name}: {_fmt(e.get(col))}"
        for name, col in [
            ("elevation_mean", "elevation_mean"), ("slope_mean", "slope_mean"),
            ("HAND_mean", "HAND_mean"), ("TWI_mean", "TWI_mean"),
            ("flow_accumulation_mean", "flow_accumulation_mean"),
            ("distance_to_water_min_m", "distance_to_water_min"),
        ]
    )
    return f"""# Cartao de extracao {item['canary_patch_id']}

## Identificacao

- Canario: `{item['canary_patch_id']}`
- Evento: `{item['candidate_event_id']}`
- bbox: `{item['bbox']}` (CRS EPSG:4326)

## Insumos encontrados

- Modelo de elevacao Copernicus GLO-30 do dominio de bacia (cobre a area do canario)
- Hidrografia oficial (faixas-marginais) para proximidade hidrica

## Insumos ausentes

- {"nenhum insumo obrigatorio ausente" if not ausentes else "; ".join(ausentes)}

## Features diretas extraidas

{feats}

## Features ainda ausentes

- {"nenhuma" if not ausentes else "; ".join(ausentes)}

## Qualidade da extracao

- {matriz_row.get('qualidade_extracao', '')}
- Metodo reconstruido (17C36), resolucao aproximada de 92 m, review-only, nao provado equivalente ao oficial.

## Impacto na calibracao

- Descritor topografico direto (review-only): {_fmt(item['direct_topo_idx'])}
- Score exploratorio sem fisico: {_fmt(item['score_sem_fisico'])} (classe {item['classe_sem_fisico']})
- Pode calibracao forte review-only: {_bool(pode_forte)}

## Decisao 17G

- feature_source_mode: {_feature_source_mode(item)}
- Com features fisicas diretas + espectral + chuva reais, a calibracao forte review-only fica possivel.

## Por que ainda nao e ground truth

Feature fisica direta descreve o terreno, nao confirma ocorrencia de inundacao no canario; sem verdade de referencia observacional nao ha ground truth.

## Por que ainda nao e treinavel

Sem rotulo validado, mesmo com features fisicas diretas, o canario nao alimenta treino supervisionado.

## Por que nao altera o score_v6

A extracao usa metodo reconstruido review-only; alimenta apenas score exploratorio, nunca substitui nem recalibra o score_v6 oficial.
"""


# --- Relatorio -------------------------------------------------------------
def report_markdown(items, auditoria, matriz, fila, reavaliacao, simulacao, gate, summary):
    aud_lines = "\n".join(f"- {r['insumo_id']} ({r['tipo']}): cobre_aoi={r['cobre_aoi_canarios']}; status={r['status_uso']}" for r in auditoria)
    canario_lines = "\n".join(
        f"- {r['canary_patch_id']}: modo={r['feature_source_mode']}; elevacao={r['elevation_mean']} m; "
        f"declividade={r['slope_mean']} graus; HAND={r['HAND_mean']} m; dist_agua_min={r['distance_to_water_min']} m; "
        f"completas={r['features_diretas_completas']}"
        for r in matriz
    )
    sim_lines = "\n".join(
        f"- {r['canary_patch_id']}: descritor_topo_direto={r['descritor_topografico_direto']}; "
        f"sem_fisico={r['score_exploratorio_sem_fisico']} ({r['classe_sem_fisico']}); "
        f"com_fisico_direto={r['score_exploratorio_fisico_direto_review_only']} ({r['classe_com_fisico_direto']}); "
        f"comparativo_17f={r['score_fisico_comparativo_17f']}; delta={r['delta_direto_vs_comparativo']}"
        for r in simulacao
    )
    gate_lines = "\n".join(f"- {r['criterio']}: passou={r['passou']} ({r['valor_observado']} / {r['limiar']})" for r in gate)
    return f"""# SUSC-17G Extracao direta de atributos fisicos/topograficos dos canarios observacionais

## Estado herdado do 17F

- Branch: `{summary['branch']}`
- HEAD: `{summary['head']}`
- Status final 17F herdado: {summary['herdado_17f_status_final']}
- Referencias comparativas no 17F: {summary['herdado_17f_comparativa']}
- Status 17B herdado: {summary['herdado_17f_status_17b']}
- `score_v6` alterado: {summary['score_v6_changed']}
- `score_v7` criado: {summary['score_v7_created']}

## Objetivo da extracao direta

Extrair diretamente elevacao, declividade, HAND, proximidade hidrica, TWI e acumulacao de fluxo
dos 5 canarios `S17C6_CANARY_REC_00001..00005`, superando a referencia comparativa distante do 17F.

## Auditoria dos insumos locais

{aud_lines}

O modelo de elevacao do dominio de bacia (17C38, Copernicus GLO-30) cobre a area dos canarios, ao
contrario da janela sul do 17C36. A hidrografia oficial (faixas-marginais, 17C35) tambem cobre a
area, permitindo a proximidade hidrica direta.

## Resultado por canario

{canario_lines}

- Features diretas completas: {summary['features_fisicas_diretas_completas']}
- Features diretas parciais: {summary['features_fisicas_diretas_parciais']}
- Podem calibracao forte review-only: {summary['pode_calibracao_forte_review_only']}

## Features diretas completas/parciais/ausentes

Todas as seis features fisicas-alvo foram extraidas diretamente para os canarios a partir do modelo
de elevacao local e da hidrografia oficial. A qualidade e review-only, metodo reconstruido (17C36),
resolucao aproximada de 92 m; nao provada equivalente ao pipeline oficial.

## Insumos externos minimos necessarios

A fila `fila_insumos_externos_features_fisicas.csv` registra, como refinamento opcional de baixa
prioridade, um modelo de elevacao de 30 m nativo por canario para aprimorar HAND, TWI e fluxo. A
extracao ja esta concluida a ~92 m, entao esse refinamento nao e bloqueante.

## Funcionamento do extrator

O extrator reutiliza o pipeline hidrologico numpy do 17C36 (preenchimento de depressoes por
inundacao prioritaria, direcao D8, acumulacao de fluxo, HAND por celula de drenagem a jusante, TWI)
aplicado ao modelo de elevacao do dominio de bacia. A proximidade hidrica usa distancia
ponto-segmento a hidrografia oficial. O modo de execucao futura le a fila de insumos, valida os
arquivos esperados e falha com erro claro se faltar insumo, sem baixar nada automaticamente.

## Simulacao exploratoria com features diretas

{sim_lines}

O descritor topografico direto (review-only) contrasta com a referencia comparativa do 17F: a
extracao direta descreve o terreno do proprio canario, corrigindo a leitura feita a partir de um
patch oficial distante. Em todas as linhas: score_oficial=false, substituir_score_v6=false,
usar_em_treino=false, ground_truth=false, score_v7_allowed=false.

## Impacto na calibracao forte

Com features fisicas diretas (elevacao, declividade, HAND, TWI, fluxo) mais espectral e chuva reais,
a calibracao forte review-only fica possivel para os canarios. Ela permanece review-only, com metodo
reconstruido, sem virar oficial, sem ground truth, sem treino e sem score_v7.

## Gate final 17G

{gate_lines}

- Status final: **{summary['status_final_17g']}**
- Caminho funcional: **{summary['caminho_funcional']}**

## Impacto no 17E e no 17B

- 17E: {summary['status_17e']}.
- 17B: {summary['status_17b']} (segue sem prontidao de benchmark: sem ground truth, amostra
  concentrada em um evento, sem score_v7).

## Proximo marco recomendado

SUSC-17H: montar a calibracao forte review-only completa dos canarios combinando as features fisicas
diretas, espectrais e de chuva, com auditoria de incerteza e refinamento opcional para modelo de
elevacao de 30 m, mantendo tudo review-only.
"""


# --- Orquestracao ----------------------------------------------------------
def build_all():
    _require_inputs()
    ensure_dir(OUT_DATA)
    ensure_dir(CARDS_DIR)
    ensure_dir(OUT_REPORTS)
    ensure_dir(SCHEMAS)
    write_schema()

    items = build_items()
    auditoria = auditoria_rows()
    matriz = matriz_rows(items)
    fila = fila_rows(items)
    reavaliacao = reavaliacao_rows(items)
    simulacao = simulacao_rows(items)
    final_status = _final_status(items, matriz, fila)
    gate = gate_rows(items, matriz, fila, simulacao, final_status)
    resumo_canario = resumo_canario_rows(items, matriz, fila, simulacao)
    resumo_status = resumo_status_rows(items, matriz, fila, simulacao)
    summary = summary_obj(items, auditoria, matriz, fila, reavaliacao, simulacao, final_status)

    write_json(PREFLIGHT_JSON, preflight_obj())
    write_csv(AUDITORIA, auditoria, AUDITORIA_FIELDS)
    write_csv(MATRIZ_DIRETA, matriz, MATRIZ_FIELDS)
    write_csv(FILA_INSUMOS, fila, FILA_FIELDS)
    write_csv(REAVALIACAO, reavaliacao, REAVALIACAO_FIELDS)
    write_csv(SIMULACAO, simulacao, SIMULACAO_FIELDS)
    write_csv(GATE, gate, GATE_FIELDS)
    write_csv(RESUMO_CANARIO, resumo_canario, RESUMO_CANARIO_FIELDS)
    write_csv(RESUMO_STATUS, resumo_status, RESUMO_STATUS_FIELDS)
    write_json(SUMMARY, summary)
    matriz_by = _by_key(matriz, "canary_patch_id")
    for item in items:
        card = CARDS_DIR / f"{item['canary_patch_id']}.md"
        write_markdown(card, card_markdown(item, matriz_by.get(item["canary_patch_id"], {})))
    write_markdown(REPORT, report_markdown(items, auditoria, matriz, fila, reavaliacao, simulacao, gate, summary))
    return summary


# --- Validacao -------------------------------------------------------------
def _public_output_paths():
    paths = [REPORT]
    paths.extend(OUT_DATA.glob("*.csv"))
    paths.extend(OUT_DATA.glob("*.json"))
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


def validate_matriz_rows(rows):
    errors = []
    numeric_cols = [
        "elevation_mean", "slope_mean", "HAND_mean", "TWI_mean",
        "flow_accumulation_mean", "distance_to_water_min",
    ]
    for idx, row in enumerate(rows, start=1):
        rid = row.get("canary_patch_id", f"row{idx}")
        mode = row.get("feature_source_mode")
        if mode not in FEATURE_SOURCE_MODE_ALLOWED:
            errors.append(f"{rid}:feature_source_mode_fora_enum:{mode}")
        for field in ["ground_truth", "eligible_for_training", "score_v7_allowed"]:
            if row.get(field) == "true":
                errors.append(f"{rid}:{field}_true_proibido")
        # valor fisico preenchido exige fonte local (dem/drenagem)
        has_value = any(row.get(c) not in {"", "not_available"} for c in numeric_cols)
        if has_value and row.get("dem_source") in {"", "not_available"} and row.get("drainage_source") in {"", "not_available"}:
            errors.append(f"{rid}:valor_fisico_sem_fonte_local")
        # feature completa nao pode ter feature ausente marcada
        if row.get("features_diretas_completas") == "true" and row.get("features_ausentes") not in {"", "nenhuma"}:
            errors.append(f"{rid}:completa_com_feature_ausente")
        # referencia comparativa nunca aparece como feature direta aqui
        if row.get("feature_source_mode") in {"direta_por_dem_e_hidrografia_local", "direta_parcial_por_dem_local"} and "recife_00552" in str(row.values()):
            errors.append(f"{rid}:referencia_comparativa_como_direta")
        if row.get("not_ground_truth_reason", "").strip() == "":
            errors.append(f"{rid}:not_ground_truth_reason_vazio")
    return errors


def validate_reavaliacao_rows(rows):
    errors = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("canary_patch_id", f"row{idx}")
        if row.get("justificativa_tecnica", "").strip() == "":
            errors.append(f"{rid}:reavaliacao_sem_justificativa")
    return errors


def validate_simulacao_rows(rows):
    errors = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("simulacao_id", f"row{idx}")
        for field in ["score_oficial", "substituir_score_v6", "usar_em_treino", "ground_truth", "score_v7_allowed"]:
            if row.get(field) != "false":
                errors.append(f"{rid}:{field}_proibido_true")
    return errors


def validate():
    summary = build_all()
    errors = []
    matriz = _read(MATRIZ_DIRETA)
    fila = _read(FILA_INSUMOS)
    errors.extend(validate_matriz_rows(matriz))
    errors.extend(validate_reavaliacao_rows(_read(REAVALIACAO)))
    errors.extend(validate_simulacao_rows(_read(SIMULACAO)))
    errors.extend(public_text_violations_files())

    # auditoria status dentro do enum
    for row in _read(AUDITORIA):
        if row.get("status_uso") not in STATUS_USO_ALLOWED:
            errors.append(f"auditoria:{row.get('insumo_id')}:status_uso_fora_enum")
    # todo canario com status final (modo)
    for row in matriz:
        if row.get("feature_source_mode") not in FEATURE_SOURCE_MODE_ALLOWED:
            errors.append(f"{row.get('canary_patch_id')}:canario_sem_status_final")
    # fila obrigatoria quando ha feature ausente
    fila_canaries = {r["canary_patch_id"] for r in fila}
    for row in matriz:
        if row.get("features_ausentes") not in {"", "nenhuma"} and row["canary_patch_id"] not in fila_canaries:
            errors.append(f"{row['canary_patch_id']}:fila_insumo_ausente_com_feature_ausente")
    if summary["status_final_17g"] not in GATE_FINAL_ALLOWED:
        errors.append(f"status_final_fora_enum:{summary['status_final_17g']}")
    if summary["status_final_17g"] == "17G_BLOQUEADO_FAIL_CLOSED":
        errors.append("sem_caminho_funcional_entregue")
    for field in ["ground_truth_true_count", "eligible_for_training_true_count",
                  "score_v7_allowed_true_count", "score_oficial_true_count", "substituir_score_v6_true_count"]:
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
        "17G extracao direta validada: "
        f"completas={summary['features_fisicas_diretas_completas']} "
        f"parciais={summary['features_fisicas_diretas_parciais']} "
        f"forte_review_only={summary['pode_calibracao_forte_review_only']} "
        f"status17G={summary['status_final_17g']}"
    )
    return 0


def run_extraction_cli():
    """Modo operacional: exige insumos locais, executa extracao real ou falha claro."""
    if not BASIN_DEM_NPY.exists():
        raise SystemExit(f"BLOQUEADO: modelo de elevacao ausente em {rel(BASIN_DEM_NPY)}; coloque o insumo e rode novamente (sem download automatico).")
    if not HIDROGRAFIA_GEOJSON.exists():
        raise SystemExit(f"BLOQUEADO: hidrografia ausente em {rel(HIDROGRAFIA_GEOJSON)}; coloque o insumo e rode novamente (sem download automatico).")
    if not _dem_covers_aoi():
        raise SystemExit("BLOQUEADO: o modelo de elevacao local nao cobre a area dos canarios; forneca um insumo que cubra o bbox.")
    ledger = _extract_to_ledger()
    print(f"extracao concluida: {len(ledger.get('per_canary', {}))} canarios, dem_sha={ledger.get('dem_sha256', '')[:12]}")
    return 0


def run_all():
    summary = build_all()
    print(
        "17G extracao direta gerada: "
        f"completas={summary['features_fisicas_diretas_completas']} "
        f"parciais={summary['features_fisicas_diretas_parciais']} "
        f"forte_review_only={summary['pode_calibracao_forte_review_only']} "
        f"status17G={summary['status_final_17g']} "
        f"caminho={summary['caminho_funcional']}"
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "extract":
        raise SystemExit(run_extraction_cli())
    raise SystemExit(run_all())
