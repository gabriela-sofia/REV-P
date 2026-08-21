"""SUSC-18G recuperacao e compactacao vetorial SAR de Curitiba.

Continua do 18F: patch_stats real existe, exports GEE foram concluidos, mas o
vetor completo nao foi recuperado porque excedeu o limite de feicoes. Esta etapa
tenta recuperar arquivo leve em rotas locais/Drive e, se necessario, gera vetor
compacto via Earth Engine, mantendo tudo somente revisao.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

import susc_18c_curitiba_geometria_common as s18c  # noqa: E402
import susc_18e2_execucao_gee_common as s18e2  # noqa: E402
import susc_18f_sar_ingestao_common as s18f  # noqa: E402
from susc_io import ensure_dir, read_csv, read_json, rel, sha256_file, write_csv, write_json, write_markdown  # noqa: E402

OUT_18G = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_18g_recuperacao_compactacao_vetorial_sar_curitiba"
CARDS = OUT_18G / "cartoes_sar"
REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"
REPORT = REPORTS / "SUSC_18G_RECUPERACAO_COMPACTACAO_VETORIAL_SAR_CURITIBA.md"
SCHEMA = SCHEMAS / "susc_18g_sar_vector_schema_v1.json"

LOCAL = s18e2.LOCAL_RESULTS
LOCAL_VECTOR = s18e2.LOCAL_FLOOD_VECTOR
LOCAL_COMPACT_VECTOR = LOCAL / "flood_vector_compact_curitiba_2022_01_15.geojson"
LOCAL_COMPACT_METADATA = LOCAL / "flood_vector_compact_metadata.json"
LOCAL_COMPACT_PATCH_SUMMARY = LOCAL / "flood_vector_compact_patch_summary.csv"
LOCAL_TASKS = s18e2.LOCAL_TASKS
LOCAL_PATCH_STATS = s18e2.LOCAL_PATCH_STATS
LOCAL_SCENES = s18e2.LOCAL_SCENES
LOCAL_METADATA = s18e2.LOCAL_METADATA
LOCAL_MASK = s18e2.LOCAL_FLOOD_MASK

OUT_18F = s18f.OUT_18F
OUT_18E2 = s18e2.OUT_DATA
OUT_18E = s18e2.OUT_18E
OUT_18C = s18e2.OUT_18C
AOI_CSV = s18e2.AOI_CSV_18E
AOI_GEOJSON = s18e2.AOI_GEOJSON_18E
WINDOWS = s18e2.WINDOWS_18E
GEE_SCRIPT = s18e2.GEE_SCRIPT_18E
SCORE_V6 = s18e2.SCORE_V6
SCORE_V7 = s18e2.SCORE_V7

STATUS_EXPORTS = OUT_18G / "status_exports_gee_18g.csv"
RECUP_DRIVE = OUT_18G / "recuperacao_drive_flood_vector.csv"
AUDIT_VECTOR = OUT_18G / "auditoria_footprint_vetorial_compacto.csv"
VINCULOS = OUT_18G / "vinculos_vetoriais_sar_patch_curitiba_18g.csv"
MATRIZ = OUT_18G / "matriz_sar_curitiba_18g.csv"
GATE_18G = OUT_18G / "gate_footprint_vetorial_sar_curitiba_pos_18g.csv"
GATE_17B = OUT_18G / "gate_prontidao_17b_pos_18g.csv"
RESUMO = OUT_18G / "resumo_por_status.csv"
SUMMARY = OUT_18G / "summary.json"
PREFLIGHT = OUT_18G / "preflight.json"
PUBLIC_VECTOR = OUT_18G / "footprint_sar_curitiba_compacto.geojson"
PUBLIC_VECTOR_CSV = OUT_18G / "footprint_sar_curitiba_compacto.csv"

EVENT_ID = "S17C_REF_0060"
EVENT_PUBLIC_ID = "CUR_2022_01_15"
TASK_IDS = s18f.TASK_IDS
MAX_FEATURES = 5000
COMPACT_SAFE_LIMIT = 43

HARD_DEFAULTS = {
    "ground_truth": "false",
    "eligible_for_training": "false",
    "score_v7_allowed": "false",
    "review_only": "true",
}

STATUS_18G_ALLOWED = {
    "18G_FOOTPRINT_VETORIAL_COMPACTO_VALIDADO",
    "18G_REFERENCIA_TECNICA_SAR_FORTE_COM_OVERLAY",
    "18G_REFERENCIA_TECNICA_SAR_PARCIAL_POR_PATCH_STATS",
    "18G_EXPORT_DRIVE_CONCLUIDO_NAO_RECUPERADO",
    "18G_BLOQUEADO_POR_LIMITE_FEICOES",
    "18G_BLOQUEADO_FAIL_CLOSED",
}
STATUS_17B_ALLOWED = {
    "17B_APROXIMACAO_COM_EVIDENCIA_TECNICA_CURITIBA",
    "17B_APROXIMACAO_COM_SEGUNDA_REGIAO_TECNICA",
    "17B_BLOQUEADO_POR_GEOMETRIA_OFICIAL",
    "17B_BLOQUEADO_POR_AMOSTRA",
    "17B_BLOQUEADO_FAIL_CLOSED",
}
MATRIX_STATUS_ALLOWED = {
    "referencia_tecnica_sar_forte_com_vetor_somente_revisao",
    "referencia_tecnica_sar_parcial_por_patch_stats",
    "footprint_vetorial_compacto_sem_overlap",
    "aguardando_recuperacao_drive",
    "aguardando_compactacao",
    "bloqueado_por_limite_feicoes",
    "bloqueado_sem_vetor_valido",
}

PUBLIC_FORBIDDEN_RE = re.compile(r"\b(?:agentic|agente|codex|llm|ia)\b", re.IGNORECASE)
SECRET_RE = s18e2.SECRET_RE

REQUIRED_INPUTS = [
    OUT_18F / "summary.json",
    OUT_18F / "estatisticas_sar_por_patch_curitiba_18f.csv",
    OUT_18E2 / "summary.json",
    OUT_18E / "summary.json",
    OUT_18C / "summary.json",
    LOCAL_PATCH_STATS,
    LOCAL_TASKS,
    LOCAL_METADATA,
    LOCAL_SCENES,
    AOI_CSV,
    AOI_GEOJSON,
    WINDOWS,
    GEE_SCRIPT,
    SCORE_V6,
]
REQUIRED_OUTPUTS = [STATUS_EXPORTS, RECUP_DRIVE, AUDIT_VECTOR, VINCULOS, MATRIZ, GATE_18G, GATE_17B, RESUMO, SUMMARY, PREFLIGHT, SCHEMA, REPORT]

STATUS_EXPORT_FIELDS = ["task_id", "tipo_export", "status_atual", "destino", "nome_esperado", "existe_localmente", "recuperavel_por_api", "recuperavel_por_drive", "bloqueio"]
RECUP_FIELDS = ["rota", "comando_testado", "status", "arquivo_encontrado", "caminho_local", "tamanho_bytes", "sha256", "bloqueio", "acao_corretiva"]
AUDIT_FIELDS = ["geometry_id", "fonte", "metodo_recuperacao", "arquivo_local", "tipo_geometria", "numero_feicoes", "area_total_m2", "bbox", "crs", "dentro_aoi", "intersecta_aoi", "valido", "incerteza_m", "limiar_usado", "filtros_usados", "limitacoes", "uso_permitido", "ground_truth", "eligible_for_training", "score_v7_allowed", "review_only", "justificativa_tecnica"]
VINC_FIELDS = ["technical_link_id", "candidate_event_id", "geometry_id", "patch_id", "classe_vinculo_vetorial", "area_overlap_m2", "overlap_ratio", "distance_m", "vinculo_tecnico_forte", "ground_truth", "eligible_for_training", "score_v7_allowed", "review_only", "justificativa_tecnica"]
MATRIX_FIELDS = ["item_id", "patch_id", "patch_stats_disponivel", "vetor_disponivel", "classe_vinculo_vetorial", "area_overlap_m2", "overlap_ratio", "flood_mask_mean", "pixel_count", "score_v6", "classe_score_v6", "status_referencia_tecnica", "ground_truth", "eligible_for_training", "score_v7_allowed", "review_only", "justificativa_tecnica"]
GATE_FIELDS = ["criterio", "valor_observado", "limiar", "passou", "status", "observacao"]
RESUMO_FIELDS = ["status_referencia_tecnica", "quantidade"]
VECTOR_CSV_FIELDS = ["geometry_id", "candidate_event_id", "evento_publico", "arquivo_local", "numero_feicoes", "area_total_m2", "bbox", "metodo_compactacao", "uso_permitido", "ground_truth", "eligible_for_training", "score_v7_allowed", "review_only", "justificativa_tecnica"]


def _bool(value) -> str:
    return "true" if value else "false"


def _run_git(args: list[str]) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _score_v6_changed() -> bool:
    return bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]))


def _sha(path: Path) -> str:
    return sha256_file(path) if path.exists() and path.is_file() else "not_available"


def _safe(value) -> str:
    return s18e2._safe_text(value)


def require_inputs() -> None:
    missing = [rel(p) for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(missing))


def ensure_dirs() -> None:
    ensure_dir(OUT_18G)
    ensure_dir(CARDS)
    ensure_dir(REPORTS)
    ensure_dir(SCHEMAS)
    ensure_dir(LOCAL)


def task_rows_from_cli() -> list[dict]:
    rows, _ = s18f.task_status_from_cli()
    return rows


def status_exports_rows(task_rows: list[dict]) -> list[dict]:
    names = {
        "flood_mask": "flood_mask_curitiba_2022_01_15.tif",
        "flood_vector": "flood_vector_curitiba_2022_01_15.geojson",
        "patch_stats": "patch_stats_curitiba_2022_01_15.csv",
    }
    paths = {"flood_mask": LOCAL_MASK, "flood_vector": LOCAL_VECTOR, "patch_stats": LOCAL_PATCH_STATS}
    rows = []
    for row in task_rows:
        artifact = row["artefato"]
        exists = paths[artifact].exists()
        rows.append({
            "task_id": row["task_id"],
            "tipo_export": artifact,
            "status_atual": row["status_atual"],
            "destino": row.get("destination", ""),
            "nome_esperado": names[artifact],
            "existe_localmente": _bool(exists),
            "recuperavel_por_api": _bool(artifact == "flood_vector" and not exists),
            "recuperavel_por_drive": _bool(row["status_atual"] == "SUCCEEDED" and not exists),
            "bloqueio": "not_available" if exists else ("export_concluido_nao_recuperado_localmente" if row["status_atual"] == "SUCCEEDED" else "aguardando_conclusao_export"),
        })
    return rows


def candidate_drive_paths() -> list[Path]:
    names = [
        "flood_vector_curitiba_2022_01_15.geojson",
        "S18E2_CUR_2022_01_15_flood_vector.geojson",
        "flood_vector_curitiba_2022_01_15.csv",
    ]
    roots = [
        Path.home() / "Downloads",
        Path.home() / "Desktop",
        Path.home() / "Documents",
        Path.home() / "Google Drive",
        Path.home() / "My Drive",
        ROOT / "local_runs",
    ]
    found = []
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            try:
                found.extend(root.rglob(name))
            except Exception:
                continue
    return sorted(dict.fromkeys([p for p in found if p.is_file()]), key=lambda p: str(p))


def copy_if_geojson(path: Path) -> bool:
    if path.suffix.lower() != ".geojson":
        return False
    try:
        obj = read_json(path)
        if obj.get("type") != "FeatureCollection":
            return False
        LOCAL_VECTOR.write_text(path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        return True
    except Exception:
        return False


def drive_recovery_rows() -> tuple[list[dict], bool]:
    rows = []
    found = candidate_drive_paths()
    copied = False
    if found:
        for p in found[:10]:
            ok = copy_if_geojson(p)
            copied = copied or ok
            rows.append({
                "rota": "busca_local_downloads_drive",
                "comando_testado": "varredura por nome esperado em Downloads/Documents/Google Drive/local_runs",
                "status": "arquivo_copiado" if ok else "arquivo_encontrado_nao_copiado",
                "arquivo_encontrado": p.name,
                "caminho_local": rel(LOCAL_VECTOR) if ok else "fora_do_repositorio",
                "tamanho_bytes": str(p.stat().st_size),
                "sha256": sha256_file(p),
                "bloqueio": "not_available" if ok else "arquivo_encontrado_sem_geojson_valido",
                "acao_corretiva": "validar arquivo local copiado" if ok else "baixar GeoJSON valido do Drive para local_runs",
            })
    else:
        rows.append({
            "rota": "busca_local_downloads_drive",
            "comando_testado": "varredura por nome esperado em Downloads/Documents/Google Drive/local_runs",
            "status": "nao_encontrado",
            "arquivo_encontrado": "not_available",
            "caminho_local": "not_available",
            "tamanho_bytes": "0",
            "sha256": "not_available",
            "bloqueio": "export_drive_concluido_nao_localizado_em_pastas_locais",
            "acao_corretiva": "baixar o GeoJSON exportado do Drive para local_runs",
        })
    for cli in ("gcloud", "drive", "rclone"):
        exists = shutil.which(cli) is not None
        rows.append({
            "rota": f"cli_{cli}",
            "comando_testado": f"{cli} --version",
            "status": "disponivel" if exists else "indisponivel",
            "arquivo_encontrado": "not_available",
            "caminho_local": "not_available",
            "tamanho_bytes": "0",
            "sha256": "not_available",
            "bloqueio": "not_available" if exists else "cliente_drive_indisponivel",
            "acao_corretiva": "usar cliente para baixar export GeoJSON" if exists else f"instalar/configurar {cli} se Drive nao estiver montado",
        })
    api_available = importlib.util.find_spec("googleapiclient") is not None
    rows.append({
        "rota": "google_drive_api",
        "comando_testado": "import googleapiclient",
        "status": "biblioteca_disponivel" if api_available else "biblioteca_ausente",
        "arquivo_encontrado": "not_available",
        "caminho_local": "not_available",
        "tamanho_bytes": "0",
        "sha256": "not_available",
        "bloqueio": "not_available" if api_available else "google_drive_api_indisponivel_no_ambiente",
        "acao_corretiva": "configurar credencial Drive sem expor segredo" if api_available else "baixar manualmente o GeoJSON concluido do Drive",
    })
    return rows, copied or LOCAL_VECTOR.exists()


def _write_local_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def _end_exclusive(date_text: str) -> str:
    return (datetime.strptime(date_text, "%Y-%m-%d").date().toordinal() + 1)


def run_compact_vector() -> dict:
    """Executa compactacao vetorial via EE e grava arquivos locais leves."""
    ensure_dirs()
    try:
        import ee  # type: ignore

        ee.Initialize()
        scene_rows = read_csv(LOCAL_SCENES)
        pre_ids = [r["system_id"] for r in scene_rows if r.get("periodo") == "pre_evento"]
        post_ids = [r["system_id"] for r in scene_rows if r.get("periodo") == "pos_evento"]
        if not pre_ids or not post_ids:
            return {"ok": False, "method": "compact_vector_by_patch", "block": "inventario_cenas_incompleto", "detail": "sem ids pre/pos"}
        pre = ee.ImageCollection([ee.Image(f"COPERNICUS/S1_GRD/{sid}") for sid in pre_ids])
        post = ee.ImageCollection([ee.Image(f"COPERNICUS/S1_GRD/{sid}") for sid in post_ids])
        bbox = s18e2.load_aoi_bbox()
        aoi = ee.Geometry.Rectangle(list(bbox), "EPSG:4326", False)
        pre_mean = pre.select(["VV", "VH"]).mean().clip(aoi)
        post_mean = post.select(["VV", "VH"]).mean().clip(aoi)
        delta_vv = post_mean.select("VV").subtract(pre_mean.select("VV"))
        delta_vh = post_mean.select("VH").subtract(pre_mean.select("VH"))
        raw_mask = delta_vv.lt(-1.5).And(delta_vh.lt(-1.5)).rename("flood_mask_candidate")
        compact_mask = raw_mask.updateMask(raw_mask).connectedPixelCount(64, True).gte(6).selfMask()
        patches = s18e2.patch_feature_collection(ee)
        area_image = compact_mask.multiply(ee.Image.pixelArea()).rename("area_detectada_m2")
        patch_stats = area_image.reduceRegions(
            collection=patches,
            reducer=ee.Reducer.sum(),
            scale=60,
            tileScale=4,
        )
        patch_stats = patch_stats.map(lambda f: f.set({
            "candidate_event_id": EVENT_ID,
            "evento_publico": EVENT_PUBLIC_ID,
            "compact_method": "compact_vector_by_patch",
            "threshold": "delta_vv<-1.5_and_delta_vh<-1.5",
            "min_connected_pixels": 6,
            "scale_m": 60,
            "ground_truth": False,
            "eligible_for_training": False,
            "score_v7_allowed": False,
            "review_only": True,
        }))
        info = patch_stats.getInfo()
        features = info.get("features", [])
        rows = []
        kept_features = []
        total_area = 0.0
        for feat in features:
            props = feat.get("properties", {})
            area = float(props.get("sum") or props.get("area_detectada_m2") or 0.0)
            props["area_detectada_m2"] = area
            props["sum"] = area
            props["uso_permitido"] = "footprint_tecnico_sar_somente_revisao"
            props["geometria_oficial_de_ocorrencia"] = False
            feat["properties"] = props
            if area > 0:
                kept_features.append(feat)
                total_area += area
            rows.append({
                "patch_id": props.get("patch_id", ""),
                "candidate_event_id": EVENT_ID,
                "evento_publico": EVENT_PUBLIC_ID,
                "area_detectada_m2": f"{area:.6f}",
                "compact_method": "compact_vector_by_patch",
                "threshold": "delta_vv<-1.5_and_delta_vh<-1.5",
                "scale_m": "60",
                "review_only": "true",
            })
        obj = {
            "type": "FeatureCollection",
            "name": "flood_vector_compact_curitiba_2022_01_15",
            "metadata": {
                "candidate_event_id": EVENT_ID,
                "evento_publico": EVENT_PUBLIC_ID,
                "compact_method": "compact_vector_by_patch",
                "feature_count": len(kept_features),
                "max_feature_limit": COMPACT_SAFE_LIMIT,
                "area_total_m2": total_area,
                "crs": "EPSG:4326",
                "uso_permitido": "footprint_tecnico_sar_somente_revisao",
                "geometria_oficial_de_ocorrencia": False,
                "ground_truth": False,
                "eligible_for_training": False,
                "score_v7_allowed": False,
                "review_only": True,
                "limitations": "geometria agregada por patch com area SAR detectada; nao e limite exato da mancha d'agua",
            },
            "features": kept_features,
        }
        write_json(LOCAL_COMPACT_VECTOR, obj)
        write_json(LOCAL_COMPACT_METADATA, obj["metadata"])
        _write_local_csv(LOCAL_COMPACT_PATCH_SUMMARY, rows, ["patch_id", "candidate_event_id", "evento_publico", "area_detectada_m2", "compact_method", "threshold", "scale_m", "review_only"])
        return {"ok": True, "method": "compact_vector_by_patch", "feature_count": len(kept_features), "area_total_m2": total_area, "block": "not_available", "detail": "vetor compacto gerado por patch"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "method": "compact_vector_by_patch", "feature_count": 0, "area_total_m2": 0.0, "block": "falha_compactacao_earth_engine", "detail": _safe(f"{exc.__class__.__name__}: {exc}")}


def _coords_bbox(coords) -> tuple[float, float, float, float] | None:
    vals: list[tuple[float, float]] = []

    def walk(obj):
        if isinstance(obj, (list, tuple)) and len(obj) >= 2 and all(isinstance(x, (int, float)) for x in obj[:2]):
            vals.append((float(obj[0]), float(obj[1])))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item)

    walk(coords)
    if not vals:
        return None
    return min(x for x, _ in vals), min(y for _, y in vals), max(x for x, _ in vals), max(y for _, y in vals)


def compact_vector_info() -> dict:
    path = LOCAL_COMPACT_VECTOR if LOCAL_COMPACT_VECTOR.exists() else LOCAL_VECTOR
    if not path.exists():
        return {"exists": False, "valid": False, "path": path, "feature_count": 0, "area_total_m2": 0.0, "bbox": None, "method": "not_available", "obj": None}
    try:
        obj = read_json(path)
        feats = obj.get("features", []) if obj.get("type") == "FeatureCollection" else []
        bboxes = []
        area = 0.0
        for feat in feats:
            bbox = _coords_bbox((feat.get("geometry") or {}).get("coordinates"))
            if bbox:
                bboxes.append(bbox)
            props = feat.get("properties", {})
            try:
                area += float(props.get("area_detectada_m2") or props.get("sum") or 0.0)
            except (TypeError, ValueError):
                pass
        bbox_all = None if not bboxes else (min(b[0] for b in bboxes), min(b[1] for b in bboxes), max(b[2] for b in bboxes), max(b[3] for b in bboxes))
        aoi = s18e2.load_aoi_bbox()
        valid = bool(feats) and bbox_all is not None and s18c._bbox_overlap(bbox_all, aoi) and len(feats) <= MAX_FEATURES
        method = (obj.get("metadata") or {}).get("compact_method", "drive_or_local_vector")
        return {"exists": True, "valid": valid, "path": path, "feature_count": len(feats), "area_total_m2": area, "bbox": bbox_all, "method": method, "obj": obj}
    except Exception:
        return {"exists": True, "valid": False, "path": path, "feature_count": 0, "area_total_m2": 0.0, "bbox": None, "method": "arquivo_invalido", "obj": None}


def audit_vector_rows(info: dict, compact_result: dict) -> list[dict]:
    bbox = info.get("bbox")
    return [{
        "geometry_id": "S18G_CUR_SAR_COMPACT_0001" if info["valid"] else "not_available",
        "fonte": "Earth Engine Sentinel-1 GRD",
        "metodo_recuperacao": info.get("method", compact_result.get("method", "not_available")),
        "arquivo_local": rel(info["path"]),
        "tipo_geometria": "FeatureCollection" if info["exists"] else "none",
        "numero_feicoes": str(info["feature_count"]),
        "area_total_m2": f"{info.get('area_total_m2', 0.0):.6f}",
        "bbox": "not_available" if not bbox else ",".join(f"{v:.8f}" for v in bbox),
        "crs": "EPSG:4326" if info["valid"] else "not_available",
        "dentro_aoi": _bool(info["valid"]),
        "intersecta_aoi": _bool(info["exists"] and bbox is not None),
        "valido": _bool(info["valid"]),
        "incerteza_m": "60",
        "limiar_usado": "delta_vv<-1.5_and_delta_vh<-1.5",
        "filtros_usados": "connectedPixelCount>=6;scale=60m;compact_vector_by_patch",
        "limitacoes": "vetor tecnico compacto agregado por patch; nao e geometria oficial nem limite exato de mancha",
        "uso_permitido": "footprint_tecnico_sar_somente_revisao",
        **HARD_DEFAULTS,
        "justificativa_tecnica": "footprint tecnico SAR compacto para revisao; nao e ground truth, nao e treino e nao substitui geometria oficial",
    }]


def _bbox_area_m2(bbox: tuple[float, float, float, float]) -> float:
    minlon, minlat, maxlon, maxlat = bbox
    midlat = math.radians((minlat + maxlat) / 2)
    return max(0.0, (maxlon - minlon) * 111_320 * math.cos(midlat)) * max(0.0, (maxlat - minlat) * 110_540)


def _inter(a, b):
    if not s18c._bbox_overlap(a, b):
        return None
    x = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    if x[2] <= x[0] or x[3] <= x[1]:
        return None
    return x


def vector_feature_bboxes(info: dict) -> list[tuple[str, tuple[float, float, float, float], float]]:
    out = []
    if not info.get("obj"):
        return out
    for feat in info["obj"].get("features", []):
        props = feat.get("properties", {})
        bbox = _coords_bbox((feat.get("geometry") or {}).get("coordinates"))
        if not bbox:
            continue
        try:
            area = float(props.get("area_detectada_m2") or props.get("sum") or 0.0)
        except (TypeError, ValueError):
            area = 0.0
        out.append((props.get("patch_id", ""), bbox, area))
    return out


def overlay_rows(info: dict, stats_rows: list[dict]) -> list[dict]:
    rows = []
    patches = s18c.load_cur_patch_polygons()
    if not info["valid"]:
        for idx, stat in enumerate(stats_rows, start=1):
            rows.append({
                "technical_link_id": f"S18G_STAT_{idx:04d}",
                "candidate_event_id": EVENT_ID,
                "geometry_id": "not_available",
                "patch_id": stat["patch_id"],
                "classe_vinculo_vetorial": "sar_patch_stat_available",
                "area_overlap_m2": "not_available",
                "overlap_ratio": "not_available",
                "distance_m": "not_available",
                "vinculo_tecnico_forte": "false",
                **HARD_DEFAULTS,
                "justificativa_tecnica": "patch_stats preservado sem vetor valido; nao e overlay geometrico nem ground truth",
            })
        return rows
    fboxes = vector_feature_bboxes(info)
    for idx, (patch_id, ring) in enumerate(sorted(patches.items()), start=1):
        pb = s18c._ring_bbox(ring)
        patch_area = max(_bbox_area_m2(pb), 1e-9)
        area_overlap = 0.0
        for f_patch, fb, area_detected in fboxes:
            if f_patch and f_patch != patch_id:
                continue
            inter = _inter(pb, fb)
            if inter:
                area_overlap += area_detected if f_patch == patch_id and area_detected > 0 else _bbox_area_m2(inter)
        ratio = min(1.0, area_overlap / patch_area)
        if ratio >= 0.05:
            cls = "technical_footprint_overlap"
        elif ratio > 0:
            cls = "technical_footprint_partial_overlap"
        else:
            cls = "no_technical_overlap"
        rows.append({
            "technical_link_id": f"S18G_LINK_{idx:04d}",
            "candidate_event_id": EVENT_ID,
            "geometry_id": "S18G_CUR_SAR_COMPACT_0001",
            "patch_id": patch_id,
            "classe_vinculo_vetorial": cls,
            "area_overlap_m2": f"{area_overlap:.6f}",
            "overlap_ratio": f"{ratio:.6f}",
            "distance_m": "0" if ratio > 0 else "not_available",
            "vinculo_tecnico_forte": _bool(ratio > 0),
            **HARD_DEFAULTS,
            "justificativa_tecnica": "overlay tecnico SAR com vetor compacto; nao e ground truth nem geometria oficial",
        })
    return rows


def score_v6_by_patch() -> dict:
    out = {}
    for row in read_csv(SCORE_V6):
        if row.get("regiao") == "curitiba":
            out[row.get("patch_id", "").replace("curitiba_", "CUR_").upper()] = row
    return out


def patch_stats_rows() -> list[dict]:
    return read_csv(LOCAL_PATCH_STATS)


def matrix_rows(stats_rows: list[dict], links: list[dict], info: dict) -> list[dict]:
    score = score_v6_by_patch()
    by_link = {l["patch_id"]: l for l in links}
    rows = []
    for idx, stat in enumerate(stats_rows, start=1):
        link = by_link.get(stat["patch_id"], {})
        s = score.get(stat["patch_id"], {})
        if info["valid"] and link.get("vinculo_tecnico_forte") == "true":
            status = "referencia_tecnica_sar_forte_com_vetor_somente_revisao"
        elif info["valid"]:
            status = "footprint_vetorial_compacto_sem_overlap"
        else:
            status = "referencia_tecnica_sar_parcial_por_patch_stats"
        rows.append({
            "item_id": f"S18G_MAT_{idx:04d}",
            "patch_id": stat["patch_id"],
            "patch_stats_disponivel": "true",
            "vetor_disponivel": _bool(info["valid"]),
            "classe_vinculo_vetorial": link.get("classe_vinculo_vetorial", "sar_patch_stat_available"),
            "area_overlap_m2": link.get("area_overlap_m2", "not_available"),
            "overlap_ratio": link.get("overlap_ratio", "not_available"),
            "flood_mask_mean": stat.get("flood_mask_mean", ""),
            "pixel_count": stat.get("pixel_count", ""),
            "score_v6": s.get("susc_score_v6_candidate", "not_available"),
            "classe_score_v6": s.get("susc_class_v6_candidate", "not_available"),
            "status_referencia_tecnica": status,
            **HARD_DEFAULTS,
            "justificativa_tecnica": "referencia tecnica SAR review-only; resultado pos-evento nao vira feature pre-evento nem geometria oficial",
        })
    return rows


def status_18g(info: dict, links: list[dict], stats_rows: list[dict], drive_found: bool, compact_result: dict) -> str:
    if info["valid"] and any(l.get("vinculo_tecnico_forte") == "true" for l in links):
        return "18G_REFERENCIA_TECNICA_SAR_FORTE_COM_OVERLAY"
    if info["valid"]:
        return "18G_FOOTPRINT_VETORIAL_COMPACTO_VALIDADO"
    if compact_result.get("block") == "falha_compactacao_earth_engine":
        return "18G_BLOQUEADO_POR_LIMITE_FEICOES"
    if drive_found is False and stats_rows:
        return "18G_REFERENCIA_TECNICA_SAR_PARCIAL_POR_PATCH_STATS"
    return "18G_EXPORT_DRIVE_CONCLUIDO_NAO_RECUPERADO"


def gates(summary: dict) -> tuple[list[dict], list[dict]]:
    rows18 = [
        ("vetor_compacto_valido", _bool(summary["compact_vector_valid"]), "true", summary["compact_vector_valid"], summary["status_18g"], "vetor compacto local valido"),
        ("numero_feicoes", str(summary["compact_feature_count"]), f"<={MAX_FEATURES}", summary["compact_feature_count"] <= MAX_FEATURES, summary["status_18g"], "limite de seguranca"),
        ("overlays_tecnicos", str(summary["technical_overlays"]), ">=1", summary["technical_overlays"] >= 1, summary["status_18g"], "overlay tecnico com patches CUR"),
        ("patch_stats_real", str(summary["patch_stats_rows"]), "43", summary["patch_stats_rows"] == 43, summary["status_18g"], "patch_stats real herdado do 18F"),
        ("ground_truth_zero", "0", "0", True, summary["status_18g"], "sem ground truth"),
        ("score_v7_zero", "0", "0", not summary["score_v7_created"], summary["status_18g"], "sem score_v7"),
    ]
    rows17 = [
        ("segunda_regiao_tecnica", _bool(summary["technical_strong_links"] > 0), "true", summary["technical_strong_links"] > 0, summary["status_17b"], "evidencia tecnica SAR de Curitiba"),
        ("geometria_oficial", "false", "true", False, summary["status_17b"], "nao substitui resposta oficial"),
        ("benchmark_17b_criado", "false", "false", True, summary["status_17b"], "nenhum benchmark criado"),
        ("ground_truth_zero", "0", "0", True, summary["status_17b"], "sem ground truth"),
    ]
    conv = lambda rows: [{"criterio": c, "valor_observado": v, "limiar": l, "passou": _bool(p), "status": s, "observacao": o} for c, v, l, p, s, o in rows]
    return conv(rows18), conv(rows17)


def resumo_rows(matrix: list[dict]) -> list[dict]:
    counts = {}
    for row in matrix:
        counts[row["status_referencia_tecnica"]] = counts.get(row["status_referencia_tecnica"], 0) + 1
    return [{"status_referencia_tecnica": k, "quantidade": str(v)} for k, v in sorted(counts.items())]


def write_public_vector(info: dict) -> None:
    if info["valid"]:
        write_json(PUBLIC_VECTOR, info["obj"])
        write_csv(PUBLIC_VECTOR_CSV, [{
            "geometry_id": "S18G_CUR_SAR_COMPACT_0001",
            "candidate_event_id": EVENT_ID,
            "evento_publico": EVENT_PUBLIC_ID,
            "arquivo_local": rel(info["path"]),
            "numero_feicoes": str(info["feature_count"]),
            "area_total_m2": f"{info['area_total_m2']:.6f}",
            "bbox": ",".join(f"{v:.8f}" for v in info["bbox"]),
            "metodo_compactacao": info["method"],
            "uso_permitido": "footprint_tecnico_sar_somente_revisao",
            **HARD_DEFAULTS,
            "justificativa_tecnica": "vetor compacto SAR para overlay tecnico; nao e geometria oficial",
        }], VECTOR_CSV_FIELDS)
    else:
        for p in (PUBLIC_VECTOR, PUBLIC_VECTOR_CSV):
            if p.exists():
                p.unlink()


def schema_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-18G recuperacao e compactacao vetorial SAR Curitiba",
        "type": "object",
        "required": ["candidate_event_id", "patch_id", "status_referencia_tecnica"],
        "properties": {
            "candidate_event_id": {"const": EVENT_ID},
            "patch_id": {"type": "string"},
            "ground_truth": {"const": "false"},
            "eligible_for_training": {"const": "false"},
            "score_v7_allowed": {"const": "false"},
            "review_only": {"const": "true"},
        },
        "max_features": MAX_FEATURES,
        "status_18g_allowed": sorted(STATUS_18G_ALLOWED),
        "status_17b_allowed": sorted(STATUS_17B_ALLOWED),
    }


def preflight_obj() -> dict:
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "required_inputs": [{"path": rel(p), "exists": p.exists()} for p in REQUIRED_INPUTS],
        "local_results": rel(LOCAL),
        "patch_count": len(s18c.load_cur_patch_polygons()),
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
    }


def summary_obj(task_rows, drive_rows, compact_result, info, links, matrix, stats_rows) -> dict:
    drive_accessible = any(r["status"] in {"arquivo_copiado", "arquivo_encontrado_nao_copiado", "disponivel", "biblioteca_disponivel"} for r in drive_rows)
    status = status_18g(info, links, stats_rows, drive_accessible, compact_result)
    status17 = "17B_APROXIMACAO_COM_SEGUNDA_REGIAO_TECNICA" if info["valid"] and any(l["vinculo_tecnico_forte"] == "true" for l in links) else "17B_APROXIMACAO_COM_EVIDENCIA_TECNICA_CURITIBA"
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "candidate_event_id": EVENT_ID,
        "evento_publico": EVENT_PUBLIC_ID,
        "input_status_18f": read_json(OUT_18F / "summary.json").get("status_18f", "unknown"),
        "task_status": {r["artefato"]: r["status_atual"] for r in task_rows},
        "drive_accessible": drive_accessible,
        "drive_vector_recovered": LOCAL_VECTOR.exists(),
        "compact_vector_produced": LOCAL_COMPACT_VECTOR.exists(),
        "compact_vector_valid": info["valid"],
        "compact_method": info["method"],
        "compact_feature_count": info["feature_count"],
        "compact_area_total_m2": info["area_total_m2"],
        "patch_stats_rows": len(stats_rows),
        "technical_overlays": len([l for l in links if l.get("classe_vinculo_vetorial", "").startswith("technical_footprint")]),
        "technical_strong_links": sum(1 for l in links if l.get("vinculo_tecnico_forte") == "true"),
        "matrix_rows": len(matrix),
        "status_18g": status,
        "status_17b": status17,
        "ground_truth_true_count": 0,
        "eligible_for_training_true_count": 0,
        "score_v7_allowed_true_count": 0,
        "benchmark_17b_criado": False,
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
        "review_only": True,
        "ground_truth": False,
        "trainable": False,
    }


def report_text(summary: dict, compact_result: dict) -> str:
    return f"""# SUSC-18G - Recuperacao e compactacao vetorial SAR de Curitiba

## Estado herdado do 18F

- Patch_stats real: `{summary['patch_stats_rows']}` linhas.
- Flood vector local herdado: `{str(summary['drive_vector_recovered']).lower()}`
- Status herdado 18F: `{summary['input_status_18f']}`

## Recuperacao e compactacao

- Drive acessivel ou rota local disponivel: `{str(summary['drive_accessible']).lower()}`
- Vetor compacto produzido: `{str(summary['compact_vector_produced']).lower()}`
- Vetor compacto valido: `{str(summary['compact_vector_valid']).lower()}`
- Metodo: `{summary['compact_method']}`
- Numero de feicoes: `{summary['compact_feature_count']}`
- Resultado da compactacao: `{compact_result.get('detail', 'not_available')}`

## Overlay tecnico

- Overlays patch criados: `{summary['technical_overlays']}`
- Vinculos tecnicos fortes somente revisao: `{summary['technical_strong_links']}`
- Status 18G: `{summary['status_18g']}`
- Status 17B: `{summary['status_17b']}`

## Guardrails

Sem ground truth, sem treino, sem score_v7, score_v6 intacto e sem benchmark 17B.
O footprint tecnico SAR compacto nao substitui geometria oficial de ocorrencia.

## Proxima acao pesada

Revisar o vetor compacto por patch e, se necessario, recuperar o GeoJSON exportado
do Drive para comparar com a compactacao local.
"""


def write_cards(summary: dict) -> None:
    cards = {
        "card_01_evento_tasks.md": f"Evento `{EVENT_PUBLIC_ID}`. Tasks: `{summary['task_status']}`.",
        "card_02_recuperacao_drive.md": f"Drive acessivel: `{str(summary['drive_accessible']).lower()}`. Vetor Drive recuperado: `{str(summary['drive_vector_recovered']).lower()}`.",
        "card_03_compactacao_ee.md": f"Vetor compacto: `{str(summary['compact_vector_valid']).lower()}`. Feicoes: `{summary['compact_feature_count']}`.",
        "card_04_overlay_patches.md": f"Overlays: `{summary['technical_overlays']}`. Fortes review-only: `{summary['technical_strong_links']}`.",
        "card_05_score_v6_limitacoes.md": "Comparacao com score_v6 permanece diagnostica; score_v6 intacto e score_v7 nao criado.",
        "card_06_impacto_17b.md": f"Status 17B: `{summary['status_17b']}`. Sem benchmark 17B.",
        "card_07_guardrails.md": "Nao e ground truth, nao e treinavel e nao substitui geometria oficial.",
    }
    for name, body in cards.items():
        write_markdown(CARDS / name, f"# {name.replace('_', ' ').replace('.md', '')}\n\n{body}\n")


def build_all(run_compact: bool = True) -> dict:
    require_inputs()
    ensure_dirs()
    task_rows = task_rows_from_cli()
    status_rows = status_exports_rows(task_rows)
    drive_rows, drive_ok = drive_recovery_rows()
    compact_result = {"ok": False, "method": "not_attempted", "feature_count": 0, "area_total_m2": 0.0, "block": "not_attempted", "detail": "compactacao nao executada"}
    if run_compact and not LOCAL_VECTOR.exists():
        compact_result = run_compact_vector()
    elif LOCAL_COMPACT_VECTOR.exists() or LOCAL_VECTOR.exists():
        compact_result = {"ok": True, "method": "arquivo_local_existente", "feature_count": 0, "area_total_m2": 0.0, "block": "not_available", "detail": "vetor local existente"}
    info = compact_vector_info()
    stats = patch_stats_rows()
    links = overlay_rows(info, stats)
    matrix = matrix_rows(stats, links, info)
    summary = summary_obj(task_rows, drive_rows, compact_result, info, links, matrix, stats)
    gate18, gate17 = gates(summary)

    write_csv(STATUS_EXPORTS, status_rows, STATUS_EXPORT_FIELDS)
    write_csv(RECUP_DRIVE, drive_rows, RECUP_FIELDS)
    write_csv(AUDIT_VECTOR, audit_vector_rows(info, compact_result), AUDIT_FIELDS)
    write_csv(VINCULOS, links, VINC_FIELDS)
    write_csv(MATRIZ, matrix, MATRIX_FIELDS)
    write_csv(GATE_18G, gate18, GATE_FIELDS)
    write_csv(GATE_17B, gate17, GATE_FIELDS)
    write_csv(RESUMO, resumo_rows(matrix), RESUMO_FIELDS)
    write_json(SCHEMA, schema_obj())
    write_json(PREFLIGHT, preflight_obj())
    write_json(SUMMARY, summary)
    write_public_vector(info)
    write_cards(summary)
    write_markdown(REPORT, report_text(summary, compact_result))
    return summary


def _public_paths() -> list[Path]:
    paths = []
    if OUT_18G.exists():
        paths.extend(p for p in OUT_18G.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".json", ".md", ".geojson"})
    for p in (REPORT, SCHEMA):
        if p.exists():
            paths.append(p)
    return sorted(dict.fromkeys(paths), key=lambda p: rel(p))


def public_text_violations_text(text: str) -> list[str]:
    return sorted({m.group(0) for m in PUBLIC_FORBIDDEN_RE.finditer(text)})


def validate_public_text() -> list[str]:
    errors = []
    for path in _public_paths():
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = public_text_violations_text(text)
        if hits:
            errors.append(f"{rel(path)}:vocabulario:{','.join(hits)}")
        if SECRET_RE.search(text):
            errors.append(f"{rel(path)}:credencial_ou_token")
    return errors


def validate_outputs(summary: dict) -> list[str]:
    errors = [f"missing:{rel(p)}" for p in REQUIRED_OUTPUTS if not p.exists()]
    if summary["status_18g"] not in STATUS_18G_ALLOWED:
        errors.append(f"status_18g_fora_enum:{summary['status_18g']}")
    if summary["status_17b"] not in STATUS_17B_ALLOWED:
        errors.append(f"status_17b_fora_enum:{summary['status_17b']}")
    if summary["score_v6_changed"]:
        errors.append("score_v6_changed_forbidden")
    if summary["score_v7_created"]:
        errors.append("score_v7_created_forbidden")
    if summary["benchmark_17b_criado"]:
        errors.append("benchmark_17b_criado_forbidden")
    if summary["compact_vector_valid"] and not LOCAL_COMPACT_VECTOR.exists() and not LOCAL_VECTOR.exists():
        errors.append("vetor_valido_sem_arquivo_real")
    if summary["compact_feature_count"] > MAX_FEATURES:
        errors.append("vetor_compacto_acima_limite_sem_justificativa")
    for path in [AUDIT_VECTOR, VINCULOS, MATRIZ]:
        for idx, row in enumerate(read_csv(path), start=1):
            rid = row.get("technical_link_id") or row.get("item_id") or row.get("geometry_id") or f"linha{idx}"
            for field in ("ground_truth", "eligible_for_training", "score_v7_allowed"):
                if row.get(field) == "true":
                    errors.append(f"{rel(path)}:{rid}:{field}_true")
            if not row.get("justificativa_tecnica", "").strip():
                errors.append(f"{rel(path)}:{rid}:justificativa_vazia")
            text = " ".join(str(v).lower() for v in row.values())
            if "geometria oficial" in text and "nao" not in text:
                errors.append(f"{rel(path)}:{rid}:footprint_promovido")
    vector_available = summary["compact_vector_valid"]
    for row in read_csv(VINCULOS):
        if row["classe_vinculo_vetorial"].startswith("technical_footprint") and not vector_available:
            errors.append(f"{row['technical_link_id']}:overlay_sem_vetor")
        if row["vinculo_tecnico_forte"] == "true" and (row["geometry_id"] in {"", "not_available"} or row["patch_id"] in {"", "not_available"}):
            errors.append(f"{row['technical_link_id']}:forte_sem_ids")
    if "pronto_para_17b" in json.dumps(read_json(SUMMARY), ensure_ascii=False).lower():
        errors.append("pronto_para_17b_proibido")
    errors.extend(validate_public_text())
    return errors


def validate() -> int:
    summary = build_all(run_compact=False)
    errors = validate_outputs(summary)
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(
        "18G recuperacao vetorial SAR Curitiba validada: "
        f"vetor={str(summary['compact_vector_valid']).lower()} "
        f"feicoes={summary['compact_feature_count']} overlays={summary['technical_overlays']} "
        f"fortes={summary['technical_strong_links']} status18G={summary['status_18g']} status17B={summary['status_17b']}"
    )
    return 0


def run_all() -> int:
    summary = build_all(run_compact=True)
    print(
        "18G recuperacao vetorial SAR Curitiba gerada: "
        f"vetor={str(summary['compact_vector_valid']).lower()} "
        f"feicoes={summary['compact_feature_count']} overlays={summary['technical_overlays']} "
        f"fortes={summary['technical_strong_links']} status18G={summary['status_18g']} status17B={summary['status_17b']}"
    )
    return 0


def run_compact_only() -> int:
    result = run_compact_vector()
    print(
        "18G compactacao vetorial Curitiba: "
        f"ok={str(result.get('ok')).lower()} metodo={result.get('method')} "
        f"feicoes={result.get('feature_count')} detalhe={result.get('detail')}"
    )
    return 0
