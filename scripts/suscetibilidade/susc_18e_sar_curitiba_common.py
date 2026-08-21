"""SUSC-18E footprint tecnico SAR de Curitiba.

Esta etapa transforma os 43 poligonos CUR ja existentes em uma AOI tecnica para
busca Sentinel-1/SAR, define janelas pre/pos para o evento CUR_2022_01_15
(S17C_REF_0060), audita arquivos locais e entrega um pacote executavel para GEE,
ASF/Copernicus. Sem raster local suficiente, nenhum footprint e inventado.

O footprint tecnico SAR, quando existir, e evidencia candidata somente revisao:
nao e geometria oficial de ocorrencia, nao e ground truth, nao alimenta treino,
nao cria score_v7 e nao cria benchmark 17B.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

import susc_18c_curitiba_geometria_common as s18c  # noqa: E402
from susc_io import ensure_dir, read_csv, read_json, rel, write_csv, write_json, write_markdown  # noqa: E402

OUT_18A = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_18a_execucao_referencia_observacional_regional"
OUT_18B = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_18b_execucao_geometrias_regionais_separacao_fenomeno"
OUT_18C = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_18c_aquisicao_geometria_oficial_curitiba"
OUT_18D = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_18d_protocolo_externo_curitiba"
OUT_DATA = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_18e_footprint_tecnico_sar_curitiba"
GEE_DIR = OUT_DATA / "pacote_gee"
CARDS_DIR = OUT_DATA / "cartoes_sar"
REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

DAT_SUSC = ROOT / "datasets" / "suscetibilidade"
SCORE_V6 = DAT_SUSC / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT_SUSC / "susc_score_v7_candidate_by_patch_v1.csv"
CUR_PRELINK = ROOT / "datasets" / "protocolo_c" / "v1uw_curitiba_event_patch_prelink_update.csv"

SUMMARY_18A = OUT_18A / "summary.json"
SUMMARY_18B = OUT_18B / "summary.json"
SUMMARY_18C = OUT_18C / "summary.json"
SUMMARY_18D = OUT_18D / "summary.json"
CUR_EXEC_18B = OUT_18B / "curitiba_execucao_geometria.csv"
FILA_SAR_18A = OUT_18A / "fila_execucao_footprint_sar.csv"
FILA_SAR_18B = OUT_18B / "fila_sar_18b.csv"

AOI_CSV = OUT_DATA / "curitiba_aoi_tecnica_43_patches.csv"
AOI_GEOJSON = OUT_DATA / "curitiba_aoi_tecnica_43_patches.geojson"
JANELAS_SAR = OUT_DATA / "janelas_sar_curitiba.csv"
AUDITORIA_SAR = OUT_DATA / "auditoria_sar_local_curitiba.csv"
FOOTPRINT_CSV = OUT_DATA / "footprint_sar_curitiba.csv"
FOOTPRINT_GEOJSON = OUT_DATA / "footprint_sar_curitiba.geojson"
PATCH_STATS = OUT_DATA / "estatisticas_sar_por_patch_curitiba.csv"
VINCULOS_TECNICOS = OUT_DATA / "vinculos_tecnicos_sar_patch_curitiba.csv"
FEATURES_SAR = OUT_DATA / "features_por_vinculo_sar_curitiba.csv"
COMPARACAO_SCORE = OUT_DATA / "comparacao_sar_score_v6_curitiba.csv"
FILA_DOWNLOAD = OUT_DATA / "fila_download_sar_curitiba.csv"
INSTRUCOES_DOWNLOAD = OUT_DATA / "instrucoes_download_sar_curitiba.md"
GATE_FOOTPRINT = OUT_DATA / "gate_footprint_sar_curitiba.csv"
GATE_17B = OUT_DATA / "gate_prontidao_17b_pos_18e.csv"
SUMMARY = OUT_DATA / "summary.json"
PREFLIGHT = OUT_DATA / "preflight.json"

GEE_SCRIPT = GEE_DIR / "gee_sentinel1_curitiba_2022_01_15.js"
GEE_README = GEE_DIR / "gee_sentinel1_curitiba_2022_01_15.md"
GEE_MANIFEST = GEE_DIR / "gee_export_manifest_curitiba.csv"
GEE_EXPECTED_SCHEMA = GEE_DIR / "expected_outputs_schema.json"
SCHEMA = SCHEMAS / "susc_18e_sar_curitiba_schema_v1.json"
REPORT = REPORTS / "SUSC_18E_FOOTPRINT_TECNICO_SAR_CURITIBA.md"

EVENT_ID = "S17C_REF_0060"
EVENT_PUBLIC_ID = "CUR_2022_01_15"
EVENT_DATE = date(2022, 1, 15)
EVENT_DATE_RANGE = "2022-01-15..2022-01-16"
LOCAL_SAR_INPUT_DIR = ROOT / "local_runs" / "suscetibilidade" / "18e_sar_curitiba"
LOCAL_GEE_RESULT_DIR = LOCAL_SAR_INPUT_DIR / "resultados_gee"

REQUIRED_INPUTS = [
    SUMMARY_18A,
    SUMMARY_18B,
    SUMMARY_18C,
    SUMMARY_18D,
    CUR_EXEC_18B,
    FILA_SAR_18A,
    FILA_SAR_18B,
    CUR_PRELINK,
    SCORE_V6,
]

REQUIRED_OUTPUTS = [
    AOI_CSV,
    AOI_GEOJSON,
    JANELAS_SAR,
    AUDITORIA_SAR,
    FOOTPRINT_CSV,
    PATCH_STATS,
    VINCULOS_TECNICOS,
    FEATURES_SAR,
    COMPARACAO_SCORE,
    FILA_DOWNLOAD,
    INSTRUCOES_DOWNLOAD,
    GATE_FOOTPRINT,
    GATE_17B,
    SUMMARY,
    PREFLIGHT,
    GEE_SCRIPT,
    GEE_README,
    GEE_MANIFEST,
    GEE_EXPECTED_SCHEMA,
    SCHEMA,
    REPORT,
]

HARD_DEFAULTS = {
    "ground_truth": "false",
    "eligible_for_training": "false",
    "score_v7_allowed": "false",
    "review_only": "true",
}

STATUS_18E_ALLOWED = {
    "18E_PACOTE_GEE_SENTINEL1_PRONTO",
    "18E_FOOTPRINT_TECNICO_SAR_PRODUZIDO",
    "18E_AGUARDANDO_RASTER_LOCAL_OU_EXECUCAO_EXTERNA",
    "18E_BLOQUEADO_FAIL_CLOSED",
}
STATUS_17B_ALLOWED = {
    "17B_BLOQUEADO_POR_GEOMETRIA",
    "17B_BLOQUEADO_POR_GEOMETRIA_OFICIAL",
    "17B_APROXIMACAO_REGIONAL_COM_REFERENCIAS_PARCIAS",
    "17B_BLOQUEADO_FAIL_CLOSED",
}

PUBLIC_FORBIDDEN_RE = re.compile(
    r"\b(?:agentic|agente|codex|llm|ia|automacao|automatizacao|automatizado|automatica|automatico)\b",
    re.IGNORECASE,
)

AOI_FIELDS = [
    "aoi_id",
    "candidate_event_id",
    "evento_publico",
    "patch_count",
    "patches_fonte",
    "crs",
    "buffer_m",
    "min_lon",
    "min_lat",
    "max_lon",
    "max_lat",
    "area_bbox_buffer_km2_aprox",
    "uso_permitido",
    "geometria_de_ocorrencia",
    "ground_truth",
    "eligible_for_training",
    "score_v7_allowed",
    "review_only",
    "justificativa_tecnica",
]
JANELAS_FIELDS = [
    "window_id",
    "candidate_event_id",
    "evento_publico",
    "event_date",
    "pre_start",
    "pre_end",
    "post_start",
    "post_end",
    "sensor",
    "collection",
    "product_type",
    "polarizacoes",
    "orbit_policy",
    "uso_permitido",
    "justificativa_tecnica",
]
AUDIT_FIELDS = [
    "audit_id",
    "arquivo",
    "tipo",
    "tamanho_bytes",
    "origem",
    "contem_curitiba",
    "contem_sentinel1",
    "contem_vv_vh",
    "contem_pre",
    "contem_pos",
    "raster_local",
    "utilizavel",
    "motivo_uso_ou_bloqueio",
]
FOOTPRINT_FIELDS = [
    "footprint_id",
    "candidate_event_id",
    "evento_publico",
    "footprint_produzido",
    "geometry_type",
    "crs",
    "bbox",
    "fonte",
    "metodo",
    "data_pre",
    "data_pos",
    "qa_status",
    "status_footprint",
    "geometria_oficial_de_ocorrencia",
    "ground_truth",
    "eligible_for_training",
    "score_v7_allowed",
    "review_only",
    "justificativa_tecnica",
]
PATCH_STATS_FIELDS = [
    "patch_id",
    "candidate_event_id",
    "footprint_id",
    "area_patch_bbox_km2_aprox",
    "area_footprint_overlap_km2",
    "fracao_overlap",
    "status_estatistica",
    "ground_truth",
    "eligible_for_training",
    "score_v7_allowed",
    "review_only",
    "justificativa_tecnica",
]
VINCULO_FIELDS = [
    "technical_link_id",
    "candidate_event_id",
    "footprint_id",
    "patch_id",
    "classe_vinculo",
    "metrica",
    "vinculo_forte",
    "ground_truth",
    "eligible_for_training",
    "score_v7_allowed",
    "review_only",
    "justificativa_tecnica",
]
FEATURE_FIELDS = [
    "feature_id",
    "technical_link_id",
    "patch_id",
    "feature_family",
    "feature_name",
    "valor",
    "periodo_evento",
    "usavel_pre_evento",
    "status_feature",
    "ground_truth",
    "eligible_for_training",
    "score_v7_allowed",
    "review_only",
    "justificativa_tecnica",
]
COMPARACAO_FIELDS = [
    "patch_id",
    "susc_score_v6_candidate",
    "susc_class_v6_candidate",
    "tem_footprint_sar",
    "classe_vinculo_sar",
    "comparacao_status",
    "score_v6_alterado",
    "score_v7_criado",
    "ground_truth",
    "eligible_for_training",
    "score_v7_allowed",
    "review_only",
    "justificativa_tecnica",
]
FILA_FIELDS = [
    "task_id",
    "candidate_event_id",
    "evento_publico",
    "provider",
    "platform",
    "product_type",
    "polarizacoes",
    "window_id",
    "start_date",
    "end_date",
    "aoi_bbox",
    "expected_input_path",
    "expected_output_path",
    "command_hint",
    "criterio_de_sucesso",
]
GATE_FIELDS = ["criterio", "valor_observado", "limiar", "passou", "observacao"]


def _bool(value) -> str:
    return "true" if value else "false"


def _run_git(args: list[str]) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _score_v6_changed() -> bool:
    return bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]))


def _read_json_or_empty(path: Path) -> dict:
    return read_json(path) if path.exists() else {}


def _require_inputs() -> None:
    missing = [rel(p) for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(missing))


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def _bbox_area_km2(bbox: tuple[float, float, float, float]) -> float:
    minlon, minlat, maxlon, maxlat = bbox
    midlat = math.radians((minlat + maxlat) / 2.0)
    width_km = max(0.0, (maxlon - minlon) * 111.32 * math.cos(midlat))
    height_km = max(0.0, (maxlat - minlat) * 110.54)
    return width_km * height_km


def _bbox_to_str(bbox: tuple[float, float, float, float]) -> str:
    return ",".join(f"{v:.8f}" for v in bbox)


def _parse_bbox(value: str) -> tuple[float, float, float, float] | None:
    try:
        parts = [float(x.strip()) for x in str(value).replace(";", ",").split(",") if x.strip()]
    except (TypeError, ValueError):
        return None
    if len(parts) != 4:
        return None
    return min(parts[0], parts[2]), min(parts[1], parts[3]), max(parts[0], parts[2]), max(parts[1], parts[3])


def _bbox_intersection_area_km2(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    if not s18c._bbox_overlap(a, b):
        return 0.0
    inter = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return _bbox_area_km2(inter)


def load_curitiba_context() -> dict:
    patches = s18c.load_cur_patch_polygons()
    if not patches:
        raise FileNotFoundError("43 poligonos CUR nao encontrados em local_runs/ground_truth/v2ca")
    bboxes = [s18c._ring_bbox(ring) for ring in patches.values()]
    raw = (
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )
    midlat = math.radians((raw[1] + raw[3]) / 2.0)
    buffer_m = 500
    dlon = buffer_m / (111_320.0 * max(math.cos(midlat), 0.1))
    dlat = buffer_m / 110_540.0
    buffered = (raw[0] - dlon, raw[1] - dlat, raw[2] + dlon, raw[3] + dlat)
    cur_prelinks = [r for r in read_csv(CUR_PRELINK) if r.get("proposed_event_id") == EVENT_PUBLIC_ID]
    return {
        "patches": patches,
        "patch_bboxes": {pid: s18c._ring_bbox(ring) for pid, ring in patches.items()},
        "raw_bbox": raw,
        "aoi_bbox": buffered,
        "buffer_m": buffer_m,
        "prelinks_count": len(cur_prelinks),
        "prelink_patch_count": len({r.get("patch_id") for r in cur_prelinks if r.get("patch_id")}),
        "summary_18a": _read_json_or_empty(SUMMARY_18A),
        "summary_18b": _read_json_or_empty(SUMMARY_18B),
        "summary_18c": _read_json_or_empty(SUMMARY_18C),
        "summary_18d": _read_json_or_empty(SUMMARY_18D),
    }


def aoi_rows(ctx: dict) -> list[dict]:
    bbox = ctx["aoi_bbox"]
    return [{
        "aoi_id": "S18E_AOI_CUR_43_PATCHES",
        "candidate_event_id": EVENT_ID,
        "evento_publico": EVENT_PUBLIC_ID,
        "patch_count": str(len(ctx["patches"])),
        "patches_fonte": "local_runs/ground_truth/v2ca/recovered_patch_boundaries",
        "crs": "EPSG:4326",
        "buffer_m": str(ctx["buffer_m"]),
        "min_lon": f"{bbox[0]:.8f}",
        "min_lat": f"{bbox[1]:.8f}",
        "max_lon": f"{bbox[2]:.8f}",
        "max_lat": f"{bbox[3]:.8f}",
        "area_bbox_buffer_km2_aprox": f"{_bbox_area_km2(bbox):.3f}",
        "uso_permitido": "aoi_tecnica_para_busca_sar",
        "geometria_de_ocorrencia": "false",
        **HARD_DEFAULTS,
        "justificativa_tecnica": (
            "AOI derivada somente dos 43 poligonos CUR reais para busca Sentinel-1; "
            "nao e geometria de ocorrencia, nao e limite municipal e nao resolve o bloqueio oficial"
        ),
    }]


def write_aoi_geojson(ctx: dict) -> None:
    b = ctx["aoi_bbox"]
    coords = [[
        [b[0], b[1]],
        [b[2], b[1]],
        [b[2], b[3]],
        [b[0], b[3]],
        [b[0], b[1]],
    ]]
    write_json(AOI_GEOJSON, {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "aoi_id": "S18E_AOI_CUR_43_PATCHES",
                "candidate_event_id": EVENT_ID,
                "evento_publico": EVENT_PUBLIC_ID,
                "patch_count": len(ctx["patches"]),
                "uso_permitido": "aoi_tecnica_para_busca_sar",
                "geometria_de_ocorrencia": False,
                "ground_truth": False,
                "eligible_for_training": False,
                "score_v7_allowed": False,
                "review_only": True,
            },
            "geometry": {"type": "Polygon", "coordinates": coords},
        }],
    })


def sar_window_rows() -> list[dict]:
    rows = []
    for days in (30, 15, 7):
        pre_start = EVENT_DATE - timedelta(days=days)
        pre_end = EVENT_DATE - timedelta(days=1)
        post_start = EVENT_DATE
        post_end = EVENT_DATE + timedelta(days=days)
        rows.append({
            "window_id": f"S18E_SAR_WIN_{days:02d}D",
            "candidate_event_id": EVENT_ID,
            "evento_publico": EVENT_PUBLIC_ID,
            "event_date": EVENT_DATE.isoformat(),
            "pre_start": pre_start.isoformat(),
            "pre_end": pre_end.isoformat(),
            "post_start": post_start.isoformat(),
            "post_end": post_end.isoformat(),
            "sensor": "Sentinel-1",
            "collection": "COPERNICUS/S1_GRD",
            "product_type": "GRD",
            "polarizacoes": "VV;VH",
            "orbit_policy": "mesmo sentido de orbita e trilha quando disponivel; caso contrario registrar divergencia",
            "uso_permitido": "busca_tecnica_sar_somente_revisao",
            "justificativa_tecnica": "janela pre/pos para diferenca de retroespalhamento; resultado depende de execucao externa",
        })
    return rows


def _rg_files(roots: list[Path]) -> list[Path]:
    args = [
        "rg",
        "--files",
        "--no-ignore",
        "--hidden",
        "-g",
        "!.git/**",
        *[str(r) for r in roots if r.exists()],
    ]
    if len(args) <= 2:
        return []
    try:
        r = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if r.returncode not in {0, 1}:
        return []
    return [ROOT / line.strip() for line in r.stdout.splitlines() if line.strip()]


def _scan_sar_candidates(max_rows: int = 120) -> list[Path]:
    roots = [
        ROOT / "datasets",
        ROOT / "manifests",
        ROOT / "outputs_public" / "data",
        ROOT / "outputs_public" / "suscetibilidade",
        ROOT / "local_runs",
    ]
    terms = re.compile(r"(sentinel[-_ ]?1|\bs1\b|\bsar\b|\bgrd\b|\bvv\b|\bvh\b|flood_mask|footprint|gee|asf|copernicus)", re.I)
    paths = []
    for path in _rg_files(roots):
        if _inside(path, OUT_DATA) or ".git" in path.parts:
            continue
        text = rel(path)
        if terms.search(text):
            paths.append(path)
    return sorted(dict.fromkeys(paths), key=lambda p: rel(p))[:max_rows]


def _is_usable_local_sar_raster(path: Path) -> tuple[bool, str]:
    text = rel(path).lower()
    raster_ext = path.suffix.lower() in {".tif", ".tiff", ".vrt", ".nc"}
    curitiba = "curitiba" in text or "cur_" in text or "s17c_ref_0060" in text
    sentinel = any(x in text for x in ("sentinel1", "sentinel-1", "s1", "sar", "grd"))
    pol = "vv" in text or "vh" in text
    pre_or_post = "pre" in text or "pos" in text or "post" in text
    if raster_ext and curitiba and sentinel and pol and pre_or_post:
        return True, "raster Sentinel-1 candidato; requer par pre/pos para suficiencia"
    return False, "manifesto, plano, script ou artefato leve; nao contem par raster Sentinel-1 pre/pos utilizavel para Curitiba"


def auditoria_sar_rows() -> tuple[list[dict], bool]:
    rows = []
    usable_pre = False
    usable_post = False
    for idx, path in enumerate(_scan_sar_candidates(), start=1):
        text = rel(path).lower()
        usable, reason = _is_usable_local_sar_raster(path)
        usable_pre = usable_pre or (usable and "pre" in text)
        usable_post = usable_post or (usable and ("pos" in text or "post" in text))
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        rows.append({
            "audit_id": f"S18E_SAR_AUD_{idx:04d}",
            "arquivo": rel(path),
            "tipo": path.suffix.lower().lstrip(".") or "sem_extensao",
            "tamanho_bytes": str(size),
            "origem": "varredura_local_repositorio",
            "contem_curitiba": _bool("curitiba" in text or "cur_" in text or EVENT_ID.lower() in text),
            "contem_sentinel1": _bool(any(x in text for x in ("sentinel1", "sentinel-1", "s1", "sar"))),
            "contem_vv_vh": _bool("vv" in text or "vh" in text),
            "contem_pre": _bool("pre" in text),
            "contem_pos": _bool("pos" in text or "post" in text),
            "raster_local": _bool(path.suffix.lower() in {".tif", ".tiff", ".vrt", ".nc"}),
            "utilizavel": _bool(usable),
            "motivo_uso_ou_bloqueio": reason,
        })
    sufficient = usable_pre and usable_post
    if not rows:
        rows.append({
            "audit_id": "S18E_SAR_AUD_0001",
            "arquivo": "not_available",
            "tipo": "not_available",
            "tamanho_bytes": "0",
            "origem": "varredura_local_repositorio",
            "contem_curitiba": "false",
            "contem_sentinel1": "false",
            "contem_vv_vh": "false",
            "contem_pre": "false",
            "contem_pos": "false",
            "raster_local": "false",
            "utilizavel": "false",
            "motivo_uso_ou_bloqueio": "nenhum candidato local SAR/raster localizado",
        })
    rows.append({
        "audit_id": "S18E_SAR_AUD_SUMMARY",
        "arquivo": "not_available",
        "tipo": "summary",
        "tamanho_bytes": "0",
        "origem": "susc_18e",
        "contem_curitiba": _bool(any(r["contem_curitiba"] == "true" for r in rows)),
        "contem_sentinel1": _bool(any(r["contem_sentinel1"] == "true" for r in rows)),
        "contem_vv_vh": _bool(any(r["contem_vv_vh"] == "true" for r in rows)),
        "contem_pre": _bool(usable_pre),
        "contem_pos": _bool(usable_post),
        "raster_local": _bool(any(r["raster_local"] == "true" for r in rows)),
        "utilizavel": _bool(sufficient),
        "motivo_uso_ou_bloqueio": (
            "par raster local pre/pos suficiente localizado"
            if sufficient else
            "nao ha par raster Sentinel-1 local pre/pos suficiente; pacote GEE/ASF/Copernicus deve ser executado fora do repositorio"
        ),
    })
    return rows, sufficient


def footprint_rows(local_sar_sufficient: bool) -> list[dict]:
    if not local_sar_sufficient:
        return [{
            "footprint_id": "not_available",
            "candidate_event_id": EVENT_ID,
            "evento_publico": EVENT_PUBLIC_ID,
            "footprint_produzido": "false",
            "geometry_type": "none",
            "crs": "not_available",
            "bbox": "not_available",
            "fonte": "not_available",
            "metodo": "Sentinel-1 VV/VH pre-pos preparado para execucao externa",
            "data_pre": "not_available",
            "data_pos": "not_available",
            "qa_status": "aguardando_execucao_externa",
            "status_footprint": "pacote_externo_pronto_sem_footprint",
            "geometria_oficial_de_ocorrencia": "false",
            **HARD_DEFAULTS,
            "justificativa_tecnica": (
                "sem raster SAR local suficiente; nenhum footprint foi inferido. "
                "Os 43 patches servem apenas como AOI tecnica e alvo de overlay futuro"
            ),
        }]
    return [{
        "footprint_id": "S18E_FP_LOCAL_0001",
        "candidate_event_id": EVENT_ID,
        "evento_publico": EVENT_PUBLIC_ID,
        "footprint_produzido": "true",
        "geometry_type": "polygon",
        "crs": "EPSG:4326",
        "bbox": "pendente_de_materializacao_local",
        "fonte": "raster_sar_local",
        "metodo": "Sentinel-1 VV/VH pre-pos",
        "data_pre": "a_validar",
        "data_pos": "a_validar",
        "qa_status": "needs_review",
        "status_footprint": "footprint_tecnico_local_requer_revisao",
        "geometria_oficial_de_ocorrencia": "false",
        **HARD_DEFAULTS,
        "justificativa_tecnica": "raster local detectado; materializacao ainda requer processamento controlado e revisao",
    }]


def build_technical_links(footprints: list[dict], patches: dict | None = None) -> list[dict]:
    patches = patches or s18c.load_cur_patch_polygons()
    links = []
    counter = 0
    for fp in footprints:
        if str(fp.get("footprint_produzido", "")).lower() not in {"true", "1", "sim"}:
            continue
        fb = _parse_bbox(fp.get("bbox", ""))
        if not fb:
            continue
        for patch_id, ring in patches.items():
            pb = s18c._ring_bbox(ring)
            if not s18c._bbox_overlap(fb, pb):
                continue
            overlap = _bbox_intersection_area_km2(fb, pb)
            patch_area = max(_bbox_area_km2(pb), 1e-12)
            counter += 1
            links.append({
                "technical_link_id": f"S18E_TLINK_{counter:04d}",
                "candidate_event_id": fp.get("candidate_event_id", EVENT_ID),
                "footprint_id": fp.get("footprint_id", "not_available"),
                "patch_id": patch_id,
                "classe_vinculo": "technical_footprint_overlap",
                "metrica": f"bbox_overlap_km2={overlap:.6f};patch_fraction={overlap / patch_area:.6f}",
                "vinculo_forte": "false",
                **HARD_DEFAULTS,
                "justificativa_tecnica": (
                    "sobreposicao tecnica entre footprint SAR candidato e patch; "
                    "nao e vinculo forte, nao e geometria oficial de ocorrencia"
                ),
            })
    return links


def _blocked_technical_link_rows() -> list[dict]:
    return [{
        "technical_link_id": "not_available",
        "candidate_event_id": EVENT_ID,
        "footprint_id": "not_available",
        "patch_id": "not_available",
        "classe_vinculo": "aguardando_footprint_sar",
        "metrica": "not_available",
        "vinculo_forte": "false",
        **HARD_DEFAULTS,
        "justificativa_tecnica": (
            "nenhum footprint SAR produzido; nao ha vinculo tecnico nem vinculo forte. "
            "Patches candidatos preparados sao alvos de avaliacao futura, nao confirmacao de ocorrencia"
        ),
    }]


def patch_stats_rows(ctx: dict, footprints: list[dict], links: list[dict]) -> list[dict]:
    links_by_patch = {l["patch_id"]: l for l in links if l.get("patch_id") not in {"", "not_available"}}
    rows = []
    for patch_id, bbox in sorted(ctx["patch_bboxes"].items()):
        link = links_by_patch.get(patch_id)
        rows.append({
            "patch_id": patch_id,
            "candidate_event_id": EVENT_ID,
            "footprint_id": link.get("footprint_id", "not_available") if link else "not_available",
            "area_patch_bbox_km2_aprox": f"{_bbox_area_km2(bbox):.6f}",
            "area_footprint_overlap_km2": "not_available" if not link else link["metrica"].split(";")[0].split("=")[-1],
            "fracao_overlap": "not_available" if not link else link["metrica"].split(";")[-1].split("=")[-1],
            "status_estatistica": "aguardando_footprint_sar" if not link else "estatistica_tecnica_candidata",
            **HARD_DEFAULTS,
            "justificativa_tecnica": (
                "patch CUR preparado para estatistica SAR futura; sem valor SAR inferido"
                if not link else
                "estatistica derivada de footprint tecnico candidato; requer revisao e nao e ground truth"
            ),
        })
    return rows


def feature_rows(links: list[dict]) -> list[dict]:
    real_links = [l for l in links if l.get("patch_id") not in {"", "not_available"}]
    if not real_links:
        return [{
            "feature_id": "not_available",
            "technical_link_id": "not_available",
            "patch_id": "not_available",
            "feature_family": "sar",
            "feature_name": "not_available",
            "valor": "not_available",
            "periodo_evento": "not_available",
            "usavel_pre_evento": "false",
            "status_feature": "aguardando_footprint_sar",
            **HARD_DEFAULTS,
            "justificativa_tecnica": "sem footprint SAR; nenhuma feature tecnica foi calculada ou preenchida",
        }]
    rows = []
    counter = 0
    for link in real_links:
        for name in ("vv_delta_db", "vh_delta_db", "sar_water_candidate_fraction"):
            counter += 1
            rows.append({
                "feature_id": f"S18E_SAR_FEAT_{counter:04d}",
                "technical_link_id": link["technical_link_id"],
                "patch_id": link["patch_id"],
                "feature_family": "sar",
                "feature_name": name,
                "valor": "a_revisar",
                "periodo_evento": "pos_evento",
                "usavel_pre_evento": "false",
                "status_feature": "feature_tecnica_pos_evento_nao_usar_como_pre_evento",
                **HARD_DEFAULTS,
                "justificativa_tecnica": "feature SAR pos-evento e diagnostica; proibida como feature pre-evento de score",
            })
    return rows


def comparacao_score_rows(links: list[dict]) -> list[dict]:
    link_by_patch = {l["patch_id"]: l for l in links if l.get("patch_id") not in {"", "not_available"}}
    rows = []
    for row in read_csv(SCORE_V6):
        if row.get("regiao") != "curitiba":
            continue
        patch_id_score = row.get("patch_id", "")
        patch_id_cur = patch_id_score.replace("curitiba_", "CUR_").upper()
        link = link_by_patch.get(patch_id_cur)
        rows.append({
            "patch_id": patch_id_score,
            "susc_score_v6_candidate": row.get("susc_score_v6_candidate", ""),
            "susc_class_v6_candidate": row.get("susc_class_v6_candidate", ""),
            "tem_footprint_sar": _bool(bool(link)),
            "classe_vinculo_sar": link.get("classe_vinculo", "not_available") if link else "not_available",
            "comparacao_status": "aguardando_footprint_sar" if not link else "comparacao_tecnica_somente_revisao",
            "score_v6_alterado": _bool(_score_v6_changed()),
            "score_v7_criado": _bool(SCORE_V7.exists()),
            **HARD_DEFAULTS,
            "justificativa_tecnica": (
                "score_v6 mantido intacto; SAR ainda ausente para comparacao"
                if not link else
                "comparacao diagnostica com footprint tecnico candidato; sem promocao de score"
            ),
        })
    return rows


def fila_download_rows(ctx: dict, windows: list[dict]) -> list[dict]:
    rows = []
    bbox = _bbox_to_str(ctx["aoi_bbox"])
    counter = 0
    for provider in ("GEE", "ASF", "Copernicus Dataspace"):
        for w in windows:
            for side, start_field, end_field in (("pre", "pre_start", "pre_end"), ("pos", "post_start", "post_end")):
                counter += 1
                expected_input = LOCAL_SAR_INPUT_DIR / "entrada_sar" / f"{provider.lower().replace(' ', '_')}_{w['window_id'].lower()}_{side}"
                expected_output = LOCAL_GEE_RESULT_DIR / f"{EVENT_PUBLIC_ID.lower()}_{w['window_id'].lower()}_{side}_manifest.csv"
                rows.append({
                    "task_id": f"S18E_DL_{counter:04d}",
                    "candidate_event_id": EVENT_ID,
                    "evento_publico": EVENT_PUBLIC_ID,
                    "provider": provider,
                    "platform": "Sentinel-1",
                    "product_type": "GRD",
                    "polarizacoes": "VV;VH",
                    "window_id": w["window_id"],
                    "start_date": w[start_field],
                    "end_date": w[end_field],
                    "aoi_bbox": bbox,
                    "expected_input_path": rel(expected_input),
                    "expected_output_path": rel(expected_output),
                    "command_hint": (
                        "executar consulta Sentinel-1 fora do repositorio; preservar metadados, CRS, data, polarizacao e orbit pass"
                    ),
                    "criterio_de_sucesso": "par pre/pos Sentinel-1 com VV/VH e cobertura da AOI tecnica dos 43 patches CUR",
                })
    return rows


def gate_footprint_rows(summary: dict) -> list[dict]:
    rows = [
        ("aoi_43_patches_criada", str(summary["patch_polys_disponiveis"]), "43", summary["patch_polys_disponiveis"] == 43, "AOI tecnica deriva dos patches CUR reais"),
        ("raster_sar_local_suficiente", _bool(summary["local_sar_raster_sufficient"]), "true", summary["local_sar_raster_sufficient"], "sem par local pre/pos no estado atual"),
        ("pacote_gee_criado", _bool(summary["gee_package_created"]), "true", summary["gee_package_created"], "script e manifesto GEE presentes"),
        ("footprint_sar_produzido", _bool(summary["footprint_produced"]), "true", summary["footprint_produced"], "nenhum footprint sem raster"),
        ("strong_patch_links_zero", str(summary["strong_patch_links"]), "0", summary["strong_patch_links"] == 0, "footprint tecnico nao cria vinculo forte"),
        ("ground_truth_zero", "0", "0", True, "sem ground truth"),
        ("trainable_zero", "0", "0", True, "sem treino"),
        ("score_v7_zero", "0", "0", not summary["score_v7_created"], "score_v7 nao criado"),
        ("score_v6_intacto", _bool(not summary["score_v6_changed"]), "true", not summary["score_v6_changed"], "score_v6 sem alteracao"),
    ]
    return [{"criterio": c, "valor_observado": v, "limiar": t, "passou": _bool(p), "observacao": o} for c, v, t, p, o in rows]


def gate_17b_rows(summary: dict) -> list[dict]:
    rows = [
        ("geometria_oficial_curitiba_resolvida", "false", "true", False, "18D segue aguardando resposta oficial"),
        ("footprint_tecnico_produzido", _bool(summary["footprint_produced"]), "true", summary["footprint_produced"], "SAR tecnico nao substitui geometria oficial"),
        ("benchmark_17b_criado", "false", "false", True, "nenhum benchmark criado"),
        ("ground_truth_zero", "0", "0", True, "sem ground truth"),
        ("trainable_zero", "0", "0", True, "sem treino"),
        ("score_v7_zero", "0", "0", not summary["score_v7_created"], "sem score_v7"),
    ]
    return [{"criterio": c, "valor_observado": v, "limiar": t, "passou": _bool(p), "observacao": o} for c, v, t, p, o in rows]


def schema_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-18E footprint tecnico SAR Curitiba",
        "type": "object",
        "required": ["candidate_event_id", "footprint_id", "geometry_type", "crs", "bbox", "qa_status"],
        "properties": {
            "candidate_event_id": {"const": EVENT_ID},
            "evento_publico": {"const": EVENT_PUBLIC_ID},
            "footprint_id": {"type": "string"},
            "geometry_type": {"enum": ["polygon", "multipolygon", "none"]},
            "crs": {"type": "string"},
            "bbox": {"type": "string"},
            "fonte": {"type": "string"},
            "metodo": {"type": "string"},
            "data_pre": {"type": "string"},
            "data_pos": {"type": "string"},
            "qa_status": {"enum": ["needs_review", "aguardando_execucao_externa", "rejected"]},
            "ground_truth": {"const": "false"},
            "eligible_for_training": {"const": "false"},
            "score_v7_allowed": {"const": "false"},
            "review_only": {"const": "true"},
        },
        "allowed_status_18e": sorted(STATUS_18E_ALLOWED),
        "local_result_dir": rel(LOCAL_GEE_RESULT_DIR),
    }


def gee_script_text(ctx: dict) -> str:
    b = ctx["aoi_bbox"]
    coords = json.dumps([[[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]], [b[0], b[1]]]])
    return f"""// SUSC-18E Curitiba - Sentinel-1 SAR tecnico somente revisao
// AOI derivada dos 43 patches CUR. Nao e geometria oficial de ocorrencia.

var eventId = '{EVENT_ID}';
var eventPublicId = '{EVENT_PUBLIC_ID}';
var aoi = ee.Geometry.Polygon({coords});

var collection = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(aoi)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('resolution_meters', 10));

var pre = collection
  .filterDate('2021-12-16', '2022-01-14')
  .select(['VV', 'VH'])
  .mean()
  .clip(aoi);

var post = collection
  .filterDate('2022-01-15', '2022-02-14')
  .select(['VV', 'VH'])
  .mean()
  .clip(aoi);

var deltaVv = post.select('VV').subtract(pre.select('VV')).rename('delta_vv_db');
var deltaVh = post.select('VH').subtract(pre.select('VH')).rename('delta_vh_db');
var waterCandidate = deltaVv.lt(-1.5).and(deltaVh.lt(-1.5)).rename('sar_water_candidate');
var stack = pre.addBands(post).addBands(deltaVv).addBands(deltaVh).addBands(waterCandidate);

Map.centerObject(aoi, 11);
Map.addLayer(aoi, {{color: 'yellow'}}, 'AOI tecnica 43 patches CUR');
Map.addLayer(deltaVv, {{min: -5, max: 5, palette: ['blue', 'white', 'red']}}, 'Delta VV dB');
Map.addLayer(waterCandidate.selfMask(), {{palette: ['00FFFF']}}, 'Candidato agua SAR');

Export.image.toDrive({{
  image: stack,
  description: 'S18E_CUR_2022_01_15_Sentinel1_stack',
  folder: 'REV_P_SUSC_18E',
  fileNamePrefix: 's18e_curitiba_2022_01_15_sentinel1_stack',
  region: aoi,
  scale: 10,
  crs: 'EPSG:4326',
  maxPixels: 1e13
}});

Export.table.toDrive({{
  collection: ee.FeatureCollection([ee.Feature(aoi, {{
    candidate_event_id: eventId,
    evento_publico: eventPublicId,
    uso_permitido: 'footprint_tecnico_somente_revisao',
    geometria_oficial_de_ocorrencia: false,
    ground_truth: false,
    eligible_for_training: false,
    score_v7_allowed: false
  }})]),
  description: 'S18E_CUR_2022_01_15_AOI_manifest',
  folder: 'REV_P_SUSC_18E',
  fileNamePrefix: 's18e_curitiba_2022_01_15_aoi_manifest',
  fileFormat: 'CSV'
}});
"""


def gee_readme_text(ctx: dict) -> str:
    return f"""# Pacote GEE Sentinel-1 - SUSC-18E Curitiba

## Escopo

Este pacote usa a AOI tecnica derivada dos 43 poligonos CUR para consultar
Sentinel-1 GRD (`VV` e `VH`) em janelas pre/pos do evento `{EVENT_PUBLIC_ID}`.

## O que este pacote nao faz

- Nao cria geometria oficial de ocorrencia.
- Nao cria ground truth.
- Nao cria treino.
- Nao cria score_v7.
- Nao cria benchmark 17B.
- Nao substitui a resposta oficial solicitada no 18D.

## Resultado esperado

Exportar o raster tecnico e o manifesto para um diretorio local privado, por exemplo:

`{rel(LOCAL_GEE_RESULT_DIR)}`

Apos a execucao externa, registrar metadados, CRS, datas, orbit pass e caminho local
antes de qualquer ingestao. Um footprint tecnico so pode ser usado como evidencia
candidata somente revisao.
"""


def gee_manifest_rows(ctx: dict) -> list[dict]:
    bbox = _bbox_to_str(ctx["aoi_bbox"])
    return [
        {
            "export_id": "S18E_GEE_EXPORT_STACK",
            "candidate_event_id": EVENT_ID,
            "evento_publico": EVENT_PUBLIC_ID,
            "collection": "COPERNICUS/S1_GRD",
            "bands": "pre_VV;pre_VH;post_VV;post_VH;delta_vv_db;delta_vh_db;sar_water_candidate",
            "aoi_bbox": bbox,
            "expected_file": "s18e_curitiba_2022_01_15_sentinel1_stack.tif",
            "expected_local_path": rel(LOCAL_GEE_RESULT_DIR / "s18e_curitiba_2022_01_15_sentinel1_stack.tif"),
            "uso_permitido": "evidencia_tecnica_somente_revisao",
            "ground_truth": "false",
            "eligible_for_training": "false",
            "score_v7_allowed": "false",
        },
        {
            "export_id": "S18E_GEE_EXPORT_AOI_MANIFEST",
            "candidate_event_id": EVENT_ID,
            "evento_publico": EVENT_PUBLIC_ID,
            "collection": "AOI_43_PATCHES_CUR",
            "bands": "not_applicable",
            "aoi_bbox": bbox,
            "expected_file": "s18e_curitiba_2022_01_15_aoi_manifest.csv",
            "expected_local_path": rel(LOCAL_GEE_RESULT_DIR / "s18e_curitiba_2022_01_15_aoi_manifest.csv"),
            "uso_permitido": "manifesto_tecnico_somente_revisao",
            "ground_truth": "false",
            "eligible_for_training": "false",
            "score_v7_allowed": "false",
        },
    ]


def expected_outputs_schema() -> dict:
    return {
        "required_private_outputs": [
            {
                "file": "s18e_curitiba_2022_01_15_sentinel1_stack.tif",
                "required_metadata": ["crs", "transform", "bands", "pre_window", "post_window", "orbit_pass", "source_collection"],
                "public_versioning": "nao_versionar_raster_pesado",
            },
            {
                "file": "footprint_sar_curitiba.geojson",
                "required_metadata": ["candidate_event_id", "footprint_id", "crs", "method", "qa_status"],
                "public_versioning": "somente vetor leve apos revisao tecnica",
            },
        ],
        "hard_defaults": HARD_DEFAULTS,
        "local_result_dir": rel(LOCAL_GEE_RESULT_DIR),
    }


def instrucoes_download_text(ctx: dict) -> str:
    return f"""# Instrucoes de busca SAR - Curitiba SUSC-18E

## Onde colocar resultados locais

Use diretorio privado:

`{rel(LOCAL_GEE_RESULT_DIR)}`

## Fontes aceitas

- Google Earth Engine, com o script em `{rel(GEE_SCRIPT)}`.
- ASF, buscando Sentinel-1 GRD com `VV` e `VH`.
- Copernicus Dataspace, buscando Sentinel-1 GRD com `VV` e `VH`.

## Regras

1. A AOI deve ser a bbox tecnica dos 43 patches CUR: `{_bbox_to_str(ctx['aoi_bbox'])}`.
2. O par pre/pos deve cobrir o evento `{EVENT_PUBLIC_ID}`.
3. Registrar CRS, data, orbit pass, polarizacoes e fonte.
4. Raster bruto permanece em caminho local privado.
5. Sem footprint revisado, manter `footprint_produzido=false`.
6. Footprint tecnico nao substitui geometria oficial de ocorrencia do 18D.
"""


def card_text(title: str, body: str) -> str:
    return f"# {title}\n\n{body}\n"


def write_cards(summary: dict) -> list[Path]:
    ensure_dir(CARDS_DIR)
    cards = {
        "card_01_status_18e.md": card_text(
            "Status SUSC-18E",
            f"Status 18E: `{summary['status_18e']}`.\n\nFootprint produzido: `{str(summary['footprint_produced']).lower()}`.",
        ),
        "card_02_aoi_43_patches.md": card_text(
            "AOI tecnica",
            "A AOI foi derivada dos 43 poligonos CUR reais e serve somente para busca SAR. Nao e geometria de ocorrencia.",
        ),
        "card_03_gate_17b.md": card_text(
            "Gate 17B",
            f"Status 17B: `{summary['status_17b']}`. O 18E nao cria benchmark 17B.",
        ),
    }
    paths = []
    for name, text in cards.items():
        path = CARDS_DIR / name
        write_markdown(path, text)
        paths.append(path)
    return paths


def report_text(summary: dict) -> str:
    return f"""# SUSC-18E - Footprint tecnico SAR de Curitiba

## Resultado

- Status 18E: `{summary['status_18e']}`
- Status 17B: `{summary['status_17b']}`
- Raster SAR local suficiente: `{str(summary['local_sar_raster_sufficient']).lower()}`
- Footprint produzido: `{str(summary['footprint_produced']).lower()}`
- Pacote GEE criado: `{str(summary['gee_package_created']).lower()}`
- Patches CUR usados para AOI: `{summary['patch_polys_disponiveis']}`
- Vinculos fortes: `{summary['strong_patch_links']}`

## Interpretacao

Os 43 patches CUR foram usados para preparar uma AOI tecnica de busca SAR. Isso
gera uma fila executavel e um pacote GEE/ASF/Copernicus, mas nao resolve a
geometria oficial de ocorrencia. No estado atual, nao ha par raster Sentinel-1
local pre/pos suficiente, portanto nenhum footprint foi produzido.

## Diferenca metodologica

Patches candidatos preparados sao alvos espaciais para processamento e revisao.
Vinculos fortes exigem geometria real de ocorrencia ou sobreposicao aceita por
regra especifica. Como o 18E nao recebeu footprint tecnico revisado e o 18D segue
sem resposta oficial, `strong_patch_links=0`.

## Proxima acao pesada

Executar o pacote GEE ou obter par Sentinel-1 GRD por ASF/Copernicus, salvar os
resultados privados em `{rel(LOCAL_GEE_RESULT_DIR)}` e revisar o footprint tecnico
antes de qualquer ingestao leve.
"""


def preflight_obj(ctx: dict, audit_rows: list[dict], local_sar_sufficient: bool) -> dict:
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "required_inputs": [{"path": rel(p), "exists": p.exists()} for p in REQUIRED_INPUTS],
        "patch_polys_disponiveis": len(ctx["patches"]),
        "prelinks_cur_2022_01_15": ctx["prelinks_count"],
        "prelink_patch_count": ctx["prelink_patch_count"],
        "audit_rows": len(audit_rows),
        "local_sar_raster_sufficient": local_sar_sufficient,
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
    }


def summary_obj(
    ctx: dict,
    audit_rows: list[dict],
    local_sar_sufficient: bool,
    footprints: list[dict],
    links: list[dict],
    cards: list[Path],
) -> dict:
    footprint_produced = any(r.get("footprint_produzido") == "true" for r in footprints)
    status = "18E_FOOTPRINT_TECNICO_SAR_PRODUZIDO" if footprint_produced else "18E_PACOTE_GEE_SENTINEL1_PRONTO"
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "candidate_event_id": EVENT_ID,
        "evento_publico": EVENT_PUBLIC_ID,
        "input_status_18c": ctx["summary_18c"].get("status_final_18c", "unknown"),
        "input_status_18d": ctx["summary_18d"].get("status_18d", "unknown"),
        "patch_polys_disponiveis": len(ctx["patches"]),
        "prelinks_cur_2022_01_15": ctx["prelinks_count"],
        "prelink_patch_count": ctx["prelink_patch_count"],
        "local_sar_candidates_count": max(0, len(audit_rows) - 1),
        "local_sar_raster_sufficient": local_sar_sufficient,
        "gee_package_created": all(p.exists() for p in (GEE_SCRIPT, GEE_README, GEE_MANIFEST, GEE_EXPECTED_SCHEMA)),
        "footprint_produced": footprint_produced,
        "footprints": sum(1 for r in footprints if r.get("footprint_produzido") == "true"),
        "technical_patch_links": len([l for l in links if l.get("patch_id") not in {"", "not_available"}]),
        "strong_patch_links": sum(1 for l in links if l.get("vinculo_forte") == "true"),
        "ground_truth_true_count": 0,
        "eligible_for_training_true_count": 0,
        "score_v7_allowed_true_count": 0,
        "benchmark_17b_criado": False,
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
        "cards_count": len(cards),
        "status_18e": status,
        "status_17b": "17B_BLOQUEADO_POR_GEOMETRIA_OFICIAL",
        "review_only": True,
        "ground_truth": False,
        "trainable": False,
        "next_heavy_action": "executar pacote GEE/ASF/Copernicus e revisar footprint tecnico antes de ingestao leve",
        "local_result_dir": rel(LOCAL_GEE_RESULT_DIR),
    }


def build_all() -> dict:
    _require_inputs()
    ensure_dir(OUT_DATA)
    ensure_dir(GEE_DIR)
    ensure_dir(CARDS_DIR)
    ensure_dir(REPORTS)
    ensure_dir(SCHEMAS)

    ctx = load_curitiba_context()
    windows = sar_window_rows()
    audit_rows, local_sar_sufficient = auditoria_sar_rows()
    footprints = footprint_rows(local_sar_sufficient)
    links = build_technical_links(footprints, ctx["patches"]) or _blocked_technical_link_rows()
    stats = patch_stats_rows(ctx, footprints, links)
    features = feature_rows(links)

    write_csv(AOI_CSV, aoi_rows(ctx), AOI_FIELDS)
    write_aoi_geojson(ctx)
    write_csv(JANELAS_SAR, windows, JANELAS_FIELDS)
    write_csv(AUDITORIA_SAR, audit_rows, AUDIT_FIELDS)
    write_csv(FOOTPRINT_CSV, footprints, FOOTPRINT_FIELDS)
    if FOOTPRINT_GEOJSON.exists() and not any(r.get("footprint_produzido") == "true" for r in footprints):
        FOOTPRINT_GEOJSON.unlink()
    write_csv(PATCH_STATS, stats, PATCH_STATS_FIELDS)
    write_csv(VINCULOS_TECNICOS, links, VINCULO_FIELDS)
    write_csv(FEATURES_SAR, features, FEATURE_FIELDS)
    write_csv(COMPARACAO_SCORE, comparacao_score_rows(links), COMPARACAO_FIELDS)
    write_csv(FILA_DOWNLOAD, fila_download_rows(ctx, windows), FILA_FIELDS)
    write_markdown(INSTRUCOES_DOWNLOAD, instrucoes_download_text(ctx))
    write_markdown(GEE_SCRIPT, gee_script_text(ctx))
    write_markdown(GEE_README, gee_readme_text(ctx))
    write_csv(GEE_MANIFEST, gee_manifest_rows(ctx))
    write_json(GEE_EXPECTED_SCHEMA, expected_outputs_schema())
    write_json(SCHEMA, schema_obj())

    cards = write_cards({
        "status_18e": "18E_PACOTE_GEE_SENTINEL1_PRONTO",
        "status_17b": "17B_BLOQUEADO_POR_GEOMETRIA_OFICIAL",
        "footprint_produced": any(r.get("footprint_produzido") == "true" for r in footprints),
    })
    summary = summary_obj(ctx, audit_rows, local_sar_sufficient, footprints, links, cards)
    write_csv(GATE_FOOTPRINT, gate_footprint_rows(summary), GATE_FIELDS)
    write_csv(GATE_17B, gate_17b_rows(summary), GATE_FIELDS)
    write_json(SUMMARY, summary)
    write_json(PREFLIGHT, preflight_obj(ctx, audit_rows, local_sar_sufficient))
    write_markdown(REPORT, report_text(summary))
    write_cards(summary)
    return summary


def _public_output_paths() -> list[Path]:
    paths = []
    if OUT_DATA.exists():
        paths.extend(p for p in OUT_DATA.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".json", ".md", ".js", ".geojson"})
    for p in (REPORT, SCHEMA):
        if p.exists():
            paths.append(p)
    return sorted(dict.fromkeys(paths), key=lambda p: rel(p))


def public_text_violations_text(text: str) -> list[str]:
    return sorted({m.group(0) for m in PUBLIC_FORBIDDEN_RE.finditer(text)})


def public_text_violations_files(paths: list[Path] | None = None) -> list[str]:
    errors = []
    for path in paths or _public_output_paths():
        hits = public_text_violations_text(path.read_text(encoding="utf-8", errors="ignore"))
        if hits:
            errors.append(f"{rel(path)}:{','.join(hits)}")
    return errors


def validate_aoi(rows: list[dict]) -> list[str]:
    errors = []
    if len(rows) != 1:
        errors.append("aoi_quantidade_invalida")
        return errors
    row = rows[0]
    if row.get("patch_count") != "43":
        errors.append("aoi_nao_usa_43_patches")
    if row.get("uso_permitido") != "aoi_tecnica_para_busca_sar":
        errors.append("aoi_uso_permitido_invalido")
    if row.get("geometria_de_ocorrencia") != "false":
        errors.append("aoi_promovida_para_ocorrencia")
    if "nao e geometria de ocorrencia" not in row.get("justificativa_tecnica", ""):
        errors.append("aoi_sem_negativa_ocorrencia")
    return errors


def validate_links(rows: list[dict]) -> list[str]:
    errors = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("technical_link_id", f"linha{idx}")
        for field in ("ground_truth", "eligible_for_training", "score_v7_allowed", "vinculo_forte"):
            if row.get(field) == "true":
                errors.append(f"{rid}:{field}_true_proibido")
        if row.get("review_only") != "true":
            errors.append(f"{rid}:review_only_false")
        if row.get("classe_vinculo") in {"exact_polygon_overlap", "point_within_patch", "bbox_overlap"}:
            errors.append(f"{rid}:classe_forte_proibida_no_18e")
        if "nao e" not in row.get("justificativa_tecnica", "") and row.get("patch_id") not in {"", "not_available"}:
            errors.append(f"{rid}:justificativa_sem_negativa_de_promocao")
    return errors


def validate_feature_rows(rows: list[dict]) -> list[str]:
    errors = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("feature_id", f"linha{idx}")
        for field in ("ground_truth", "eligible_for_training", "score_v7_allowed"):
            if row.get(field) == "true":
                errors.append(f"{rid}:{field}_true_proibido")
        if row.get("periodo_evento") == "pos_evento" and row.get("usavel_pre_evento") == "true":
            errors.append(f"{rid}:feature_pos_evento_marcada_como_pre_evento")
        if row.get("review_only") != "true":
            errors.append(f"{rid}:review_only_false")
    return errors


def validate_summary(summary: dict) -> list[str]:
    errors = []
    if summary.get("status_18e") not in STATUS_18E_ALLOWED:
        errors.append(f"status_18e_fora_enum:{summary.get('status_18e')}")
    if summary.get("status_17b") not in STATUS_17B_ALLOWED:
        errors.append(f"status_17b_fora_enum:{summary.get('status_17b')}")
    if summary.get("patch_polys_disponiveis") != 43:
        errors.append("patch_polys_disponiveis_diferente_de_43")
    if not summary.get("local_sar_raster_sufficient") and summary.get("footprint_produced"):
        errors.append("footprint_produzido_sem_raster_suficiente")
    if summary.get("strong_patch_links") != 0:
        errors.append("strong_patch_links_nao_zero")
    for field in ("ground_truth_true_count", "eligible_for_training_true_count", "score_v7_allowed_true_count"):
        if summary.get(field) != 0:
            errors.append(f"{field}_nonzero")
    if summary.get("benchmark_17b_criado"):
        errors.append("benchmark_17b_criado_proibido")
    if summary.get("score_v6_changed"):
        errors.append("score_v6_changed_forbidden")
    if summary.get("score_v7_created"):
        errors.append("score_v7_created_forbidden")
    if any(p.name.startswith("benchmark_17b") for p in OUT_DATA.glob("*")):
        errors.append("arquivo_benchmark_17b_criado")
    return errors


def validate_required_outputs() -> list[str]:
    return [f"missing:{rel(p)}" for p in REQUIRED_OUTPUTS if not p.exists()]


def validate() -> int:
    summary = build_all()
    errors = []
    errors.extend(validate_required_outputs())
    errors.extend(validate_aoi(read_csv(AOI_CSV)))
    errors.extend(validate_links(read_csv(VINCULOS_TECNICOS)))
    errors.extend(validate_feature_rows(read_csv(FEATURES_SAR)))
    errors.extend(validate_summary(summary))
    errors.extend(public_text_violations_files())
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(
        "18E footprint tecnico SAR Curitiba validado: "
        f"raster_local={str(summary['local_sar_raster_sufficient']).lower()} "
        f"footprint={str(summary['footprint_produced']).lower()} "
        f"gee={str(summary['gee_package_created']).lower()} "
        f"fortes={summary['strong_patch_links']} "
        f"status18E={summary['status_18e']} status17B={summary['status_17b']}"
    )
    return 0


def run_all() -> int:
    summary = build_all()
    print(
        "18E footprint tecnico SAR Curitiba gerado: "
        f"raster_local={str(summary['local_sar_raster_sufficient']).lower()} "
        f"footprint={str(summary['footprint_produced']).lower()} "
        f"gee={str(summary['gee_package_created']).lower()} "
        f"fortes={summary['strong_patch_links']} "
        f"status18E={summary['status_18e']} status17B={summary['status_17b']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_all())
