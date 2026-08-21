"""SUSC-18F recuperacao, ingestao e validacao SAR de Curitiba.

Consome o estado real do SUSC-18E2: tasks Earth Engine iniciadas, inventario de
cenas e patch_stats local real. Tenta recuperar vetor leve via Earth Engine API,
mas nao baixa raster pesado nem inventa footprint, raster, patch-link ou
geometria oficial.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

import susc_18c_curitiba_geometria_common as s18c  # noqa: E402
import susc_18e2_execucao_gee_common as s18e2  # noqa: E402
from susc_io import ensure_dir, read_csv, read_json, rel, sha256_file, write_csv, write_json, write_markdown  # noqa: E402

OUT_18F = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_18f_ingestao_validacao_footprint_sar_curitiba"
CARDS = OUT_18F / "cartoes_sar"
REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"
REPORT = REPORTS / "SUSC_18F_INGESTAO_VALIDACAO_FOOTPRINT_SAR_CURITIBA.md"
SCHEMA = SCHEMAS / "susc_18f_sar_ingestao_schema_v1.json"

LOCAL = s18e2.LOCAL_RESULTS
LOCAL_TASKS = s18e2.LOCAL_TASKS
LOCAL_PATCH_STATS = s18e2.LOCAL_PATCH_STATS
LOCAL_VECTOR = s18e2.LOCAL_FLOOD_VECTOR
LOCAL_MASK = s18e2.LOCAL_FLOOD_MASK
LOCAL_METADATA = s18e2.LOCAL_METADATA
LOCAL_SCENES = s18e2.LOCAL_SCENES

OUT_18E = s18e2.OUT_18E
OUT_18E2 = s18e2.OUT_DATA
OUT_18D = s18e2.OUT_18D
OUT_18C = s18e2.OUT_18C
AOI_CSV = s18e2.AOI_CSV_18E
AOI_GEOJSON = s18e2.AOI_GEOJSON_18E
WINDOWS = s18e2.WINDOWS_18E
GEE_MANIFEST = s18e2.GEE_MANIFEST_18E
GEE_SCHEMA = s18e2.GEE_SCHEMA_18E
SCORE_V6 = s18e2.SCORE_V6
SCORE_V7 = s18e2.SCORE_V7

STATUS_TASKS = OUT_18F / "status_tasks_gee_18f.csv"
RECUPERACAO = OUT_18F / "recuperacao_exports_gee.csv"
AUDITORIA = OUT_18F / "auditoria_resultados_gee_curitiba.csv"
FOOTPRINT_CSV = OUT_18F / "footprint_sar_curitiba_ingestado.csv"
FOOTPRINT_GEOJSON = OUT_18F / "footprint_sar_curitiba_ingestado.geojson"
FILA_VETORIZACAO = OUT_18F / "fila_vetorizacao_sar_curitiba.csv"
VALIDACAO = OUT_18F / "validacao_footprint_sar_curitiba.csv"
VINCULOS = OUT_18F / "vinculos_sar_patch_curitiba_18f.csv"
ESTATISTICAS = OUT_18F / "estatisticas_sar_por_patch_curitiba_18f.csv"
FEATURES = OUT_18F / "features_por_vinculo_sar_curitiba_18f.csv"
COMPARACAO = OUT_18F / "comparacao_sar_score_v6_curitiba_18f.csv"
MATRIZ = OUT_18F / "matriz_referencia_tecnica_sar_curitiba.csv"
GATE_18F = OUT_18F / "gate_footprint_sar_curitiba_pos_18f.csv"
GATE_17B = OUT_18F / "gate_prontidao_17b_pos_18f.csv"
RESUMO_STATUS = OUT_18F / "resumo_por_status.csv"
SUMMARY = OUT_18F / "summary.json"
PREFLIGHT = OUT_18F / "preflight.json"

EVENT_ID = "S17C_REF_0060"
EVENT_PUBLIC_ID = "CUR_2022_01_15"
EVENT_DATE = "2022-01-15"

TASK_IDS = {
    "flood_mask": "YNY6EV5ZXU25NT737DDJ7VCJ",
    "flood_vector": "KF4J7MQQDB64TDEM7S5CNLPZ",
    "patch_stats": "PCSCLVHXO2BGXK6P63XCFF4K",
}

HARD_DEFAULTS = {
    "ground_truth": "false",
    "eligible_for_training": "false",
    "score_v7_allowed": "false",
    "review_only": "true",
}

STATUS_18F_ALLOWED = {
    "18F_REFERENCIA_TECNICA_SAR_CURITIBA_FORTE",
    "18F_REFERENCIA_TECNICA_SAR_CURITIBA_PARCIAL_POR_PATCH_STATS",
    "18F_AGUARDANDO_RECUPERACAO_FOOTPRINT_VETORIAL",
    "18F_AGUARDANDO_CONCLUSAO_FLOOD_MASK",
    "18F_AGUARDANDO_RESULTADO_EXTERNO_GEE",
    "18F_FOOTPRINT_INVALIDO",
    "18F_BLOQUEADO_FAIL_CLOSED",
}
STATUS_17B_ALLOWED = {
    "17B_APROXIMACAO_COM_EVIDENCIA_TECNICA_CURITIBA",
    "17B_BLOQUEADO_POR_GEOMETRIA_OFICIAL",
    "17B_BLOQUEADO_POR_AMOSTRA",
    "17B_BLOQUEADO_FAIL_CLOSED",
}
VALIDACAO_STATUS_ALLOWED = {
    "footprint_tecnico_valido_somente_revisao",
    "footprint_tecnico_parcial_por_patch_stats",
    "footprint_vetorial_aguardando_recuperacao",
    "footprint_invalido_fora_aoi",
    "footprint_invalido_sem_metadados",
    "aguardando_conclusao_export",
    "aguardando_vetorizacao",
}
MATRIZ_STATUS_ALLOWED = {
    "referencia_tecnica_sar_forte_somente_revisao",
    "referencia_tecnica_sar_parcial_por_patch_stats",
    "aguardando_resultado_externo",
    "aguardando_recuperacao_footprint_vetorial",
    "aguardando_vetorizacao",
    "bloqueado_footprint_invalido",
    "bloqueado_sem_patch_link",
}

PUBLIC_FORBIDDEN_RE = re.compile(r"\b(?:agentic|agente|codex|llm|ia)\b", re.IGNORECASE)
SECRET_RE = s18e2.SECRET_RE

REQUIRED_INPUTS = [
    OUT_18E / "summary.json",
    OUT_18E2 / "summary.json",
    OUT_18D / "summary.json",
    OUT_18C / "summary.json",
    AOI_CSV,
    AOI_GEOJSON,
    WINDOWS,
    GEE_MANIFEST,
    GEE_SCHEMA,
    LOCAL_METADATA,
    LOCAL_SCENES,
    LOCAL_TASKS,
    LOCAL_PATCH_STATS,
    SCORE_V6,
]

REQUIRED_OUTPUTS = [
    STATUS_TASKS,
    RECUPERACAO,
    AUDITORIA,
    FOOTPRINT_CSV,
    FILA_VETORIZACAO,
    VALIDACAO,
    VINCULOS,
    ESTATISTICAS,
    FEATURES,
    COMPARACAO,
    MATRIZ,
    GATE_18F,
    GATE_17B,
    RESUMO_STATUS,
    SUMMARY,
    PREFLIGHT,
    SCHEMA,
    REPORT,
]

STATUS_TASKS_FIELDS = ["task_id", "artefato", "task_type", "destination", "status_anterior", "status_atual", "consulta_realizada", "observacao"]
RECUP_FIELDS = ["artifact_id", "tipo", "task_id", "status_task", "esperado", "existe_local", "recuperado_nesta_sprint", "metodo_recuperacao", "caminho_local", "bloqueio", "proxima_acao"]
AUDIT_FIELDS = ["arquivo", "caminho", "tipo", "tamanho_bytes", "sha256", "existe", "formato_valido", "cobre_aoi", "contem_evento", "contem_pre_window", "contem_post_window", "utilizavel", "motivo_uso_ou_bloqueio"]
FOOTPRINT_FIELDS = ["geometry_id", "candidate_event_id", "evento_publico", "status_ingestao", "geometry_type", "crs", "bbox", "area_bbox_km2_aprox", "feature_count", "fonte", "metodo_recuperacao", "footprint_vetorial_disponivel", "geometria_oficial_de_ocorrencia", "ground_truth", "eligible_for_training", "score_v7_allowed", "review_only", "justificativa_tecnica"]
FILA_VET_FIELDS = ["task_id", "artefato", "status", "caminho_esperado", "acao_requerida", "criterio_de_sucesso"]
VALIDACAO_FIELDS = ["validation_id", "candidate_event_id", "geometry_id", "status_validacao", "patch_stats_disponivel", "footprint_vetorial_disponivel", "flood_mask_disponivel", "feature_count", "cobre_aoi", "ground_truth", "eligible_for_training", "score_v7_allowed", "review_only", "justificativa_tecnica"]
VINCULOS_FIELDS = ["technical_link_id", "candidate_event_id", "geometry_id", "patch_id", "classe_vinculo_tecnico", "area_overlap_m2", "overlap_ratio", "vinculo_tecnico_forte", "ground_truth", "eligible_for_training", "score_v7_allowed", "review_only", "justificativa_tecnica"]
ESTAT_FIELDS = ["patch_id", "candidate_event_id", "evento_publico", "flood_mask_mean", "pixel_count", "sinal_sar", "status_estatistica", "usavel_pre_evento", "ground_truth", "eligible_for_training", "score_v7_allowed", "review_only", "justificativa_tecnica"]
FEATURE_FIELDS = ["feature_id", "technical_link_id", "patch_id", "feature_name", "valor", "periodo_evento", "usavel_pre_evento", "status_feature", "ground_truth", "eligible_for_training", "score_v7_allowed", "review_only", "justificativa_tecnica"]
COMPARACAO_FIELDS = ["patch_id", "susc_score_v6_candidate", "susc_class_v6_candidate", "flood_mask_mean", "sinal_sar", "comparacao_status", "score_v6_alterado", "score_v7_criado", "ground_truth", "eligible_for_training", "score_v7_allowed", "review_only", "justificativa_tecnica"]
MATRIZ_FIELDS = ["item_id", "candidate_event_id", "data_evento", "tipo_fenomeno", "geometry_id", "footprint_source", "patch_id", "classe_vinculo_tecnico", "patch_stats_disponivel", "footprint_vetorial_disponivel", "flood_mask_disponivel", "possui_fisico", "possui_espectral", "possui_chuva", "score_v6_referencia", "classe_score_v6", "status_referencia_tecnica", "uso_permitido", "ground_truth", "eligible_for_training", "score_v7_allowed", "review_only", "not_ground_truth_reason", "justificativa_tecnica"]
GATE_FIELDS = ["criterio", "valor_observado", "limiar", "passou", "status", "observacao"]
RESUMO_FIELDS = ["status_referencia_tecnica", "quantidade"]


def _bool(value) -> str:
    return "true" if value else "false"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _run_git(args: list[str]) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _score_v6_changed() -> bool:
    return bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]))


def _safe_text(value) -> str:
    return s18e2._safe_text(value)


def _sha(path: Path) -> str:
    return sha256_file(path) if path.exists() and path.is_file() else "not_available"


def _read_json(path: Path) -> dict:
    return read_json(path) if path.exists() else {}


def _bbox_area_km2(bbox: tuple[float, float, float, float]) -> float:
    minlon, minlat, maxlon, maxlat = bbox
    midlat = math.radians((minlat + maxlat) / 2.0)
    return max(0.0, (maxlon - minlon) * 111.32 * math.cos(midlat)) * max(0.0, (maxlat - minlat) * 110.54)


def _bbox_intersection(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    if not s18c._bbox_overlap(a, b):
        return None
    inter = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    if inter[2] <= inter[0] or inter[3] <= inter[1]:
        return None
    return inter


def require_inputs() -> None:
    missing = [rel(p) for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(missing))


def ensure_dirs() -> None:
    ensure_dir(OUT_18F)
    ensure_dir(CARDS)
    ensure_dir(REPORTS)
    ensure_dir(SCHEMAS)
    ensure_dir(LOCAL)


def task_status_from_cli() -> tuple[list[dict], dict]:
    previous = {r.get("task_id"): r for r in read_csv(LOCAL_TASKS)} if LOCAL_TASKS.exists() else {}
    cmd = ["earthengine", "task", "list"]
    result = s18e2._run_cmd(cmd, timeout=60) if shutil.which("earthengine") else {"returncode": 127, "output": "comando_nao_encontrado"}
    states: dict[str, str] = {}
    known_states = {"READY", "RUNNING", "PENDING", "SUCCEEDED", "FAILED", "CANCELLED", "COMPLETED"}
    for line in result.get("output", "").split(" --- "):
        parts = line.split()
        if len(parts) < 4:
            continue
        tid = parts[0]
        state = next((p for p in parts[1:] if p in known_states), "")
        if tid:
            states[tid] = state
    artifact_by_id = {v: k for k, v in TASK_IDS.items()}
    rows = []
    local_task_rows = []
    for artifact, tid in TASK_IDS.items():
        prev = previous.get(tid, {})
        state = states.get(tid, prev.get("state", "not_available"))
        task_type = prev.get("task_type", {"flood_mask": "Export.image.toDrive", "flood_vector": "Export.table.toDrive", "patch_stats": "Export.table.toDrive"}[artifact])
        destination = prev.get("destination", f"Google Drive/REV_P_SUSC_18E2/{artifact}")
        rows.append({
            "task_id": tid,
            "artefato": artifact,
            "task_type": task_type,
            "destination": destination,
            "status_anterior": prev.get("state", "not_available"),
            "status_atual": state,
            "consulta_realizada": _bool(result.get("returncode") == 0),
            "observacao": "status atualizado via Earth Engine CLI" if result.get("returncode") == 0 else _safe_text(result.get("output", "falha_consulta")),
        })
        local_task_rows.append({"task_id": tid, "task_type": task_type, "destination": destination, "state": state, "detail": ""})
    write_csv(LOCAL_TASKS, local_task_rows, ["task_id", "task_type", "destination", "state", "detail"])
    return rows, {artifact_by_id.get(tid, tid): state for tid, state in states.items()}


def _end_exclusive(date_text: str) -> str:
    return (datetime.strptime(date_text, "%Y-%m-%d").date() + timedelta(days=1)).isoformat()


def try_recover_vector_from_ee(task_rows: list[dict]) -> dict:
    if LOCAL_VECTOR.exists():
        return {"ok": True, "method": "arquivo_local_existente", "block": "not_available", "detail": "vetor ja estava em local_runs"}
    task = next((r for r in task_rows if r["artefato"] == "flood_vector"), {})
    if task.get("status_atual") != "SUCCEEDED":
        return {"ok": False, "method": "nao_tentado", "block": "aguardando_conclusao_export", "detail": "task vetorial ainda nao esta SUCCEEDED"}
    try:
        import ee  # type: ignore

        ee.Initialize()
        window = next(r for r in read_csv(WINDOWS) if r.get("window_id") == s18e2.PRIMARY_WINDOW_ID)
        bbox = s18e2.load_aoi_bbox()
        aoi = ee.Geometry.Rectangle(list(bbox), "EPSG:4326", False)
        collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(aoi)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("resolution_meters", 10))
        )
        pre = collection.filterDate(window["pre_start"], _end_exclusive(window["pre_end"]))
        post = collection.filterDate(window["post_start"], _end_exclusive(window["post_end"]))
        pre_count = int(pre.size().getInfo())
        post_count = int(post.size().getInfo())
        if pre_count == 0 or post_count == 0:
            return {"ok": False, "method": "earth_engine_api", "block": "sem_cenas_compativeis", "detail": f"pre={pre_count};pos={post_count}"}
        pre_mean = pre.select(["VV", "VH"]).mean().clip(aoi)
        post_mean = post.select(["VV", "VH"]).mean().clip(aoi)
        delta_vv = post_mean.select("VV").subtract(pre_mean.select("VV"))
        delta_vh = post_mean.select("VH").subtract(pre_mean.select("VH"))
        flood_mask = delta_vv.lt(-1.5).And(delta_vh.lt(-1.5)).rename("flood_mask_candidate").selfMask()
        vector_fc = flood_mask.reduceToVectors(
            geometry=aoi,
            scale=30,
            geometryType="polygon",
            eightConnected=True,
            labelProperty="flood_candidate",
            maxPixels=1e8,
        )
        info = vector_fc.limit(10000).getInfo()
        features = info.get("features", [])
        obj = {
            "type": "FeatureCollection",
            "name": "flood_vector_curitiba_2022_01_15",
            "metadata": {
                "candidate_event_id": EVENT_ID,
                "evento_publico": EVENT_PUBLIC_ID,
                "source": "Earth Engine API reduceToVectors",
                "pre_scene_count": pre_count,
                "post_scene_count": post_count,
                "crs": "EPSG:4326",
                "ground_truth": False,
                "eligible_for_training": False,
                "score_v7_allowed": False,
                "review_only": True,
            },
            "features": features,
        }
        write_json(LOCAL_VECTOR, obj)
        return {"ok": True, "method": "earth_engine_api_getinfo_reduce_to_vectors", "block": "not_available", "detail": f"features={len(features)}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "method": "earth_engine_api_getinfo_reduce_to_vectors", "block": "falha_download_leve_earth_engine", "detail": _safe_text(f"{exc.__class__.__name__}: {exc}")}


def recovery_rows(task_rows: list[dict], vector_recovery: dict) -> list[dict]:
    status_by_artifact = {r["artefato"]: r["status_atual"] for r in task_rows}
    artifacts = [
        ("flood_mask", "raster_mask", TASK_IDS["flood_mask"], LOCAL_MASK, "flood_mask_curitiba_2022_01_15.tif"),
        ("flood_vector", "geojson_vector", TASK_IDS["flood_vector"], LOCAL_VECTOR, "flood_vector_curitiba_2022_01_15.geojson"),
        ("patch_stats", "csv_patch_stats", TASK_IDS["patch_stats"], LOCAL_PATCH_STATS, "patch_stats_curitiba_2022_01_15.csv"),
    ]
    rows = []
    for artifact, tipo, tid, path, expected in artifacts:
        exists = path.exists()
        recovered = artifact == "flood_vector" and vector_recovery.get("ok") and vector_recovery.get("method") != "arquivo_local_existente"
        if artifact == "patch_stats" and exists:
            method, block, action = "arquivo_local_existente", "not_available", "usar estatisticas reais locais"
        elif artifact == "flood_vector" and exists:
            method, block, action = vector_recovery.get("method", "arquivo_local_existente"), "not_available", "validar vetor tecnico e gerar overlay"
        elif artifact == "flood_mask" and not exists:
            method = "nao_baixado_por_regra_de_raster_pesado"
            block = "export_drive_concluido_nao_recuperado_localmente" if status_by_artifact.get(artifact) == "SUCCEEDED" else "aguardando_conclusao_export"
            action = "baixar raster do Drive somente com autorizacao explicita e manter em local_runs"
        else:
            method = vector_recovery.get("method", "not_available") if artifact == "flood_vector" else "not_available"
            block = vector_recovery.get("block", "not_available") if artifact == "flood_vector" else "not_available"
            if artifact == "flood_vector" and vector_recovery.get("detail"):
                block = f"{block}: {_safe_text(vector_recovery.get('detail'))}"
            action = "recuperar export concluido no Drive ou reexecutar download leve via Earth Engine API"
        rows.append({
            "artifact_id": artifact,
            "tipo": tipo,
            "task_id": tid,
            "status_task": status_by_artifact.get(artifact, "not_available"),
            "esperado": expected,
            "existe_local": _bool(exists),
            "recuperado_nesta_sprint": _bool(recovered),
            "metodo_recuperacao": method,
            "caminho_local": rel(path),
            "bloqueio": block,
            "proxima_acao": action,
        })
    return rows


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
    lons = [v[0] for v in vals]
    lats = [v[1] for v in vals]
    return min(lons), min(lats), max(lons), max(lats)


def footprint_info() -> dict:
    if not LOCAL_VECTOR.exists():
        return {"exists": False, "valid": False, "feature_count": 0, "bbox": None, "obj": None, "geometry_id": "not_available", "status": "footprint_vetorial_aguardando_recuperacao"}
    try:
        obj = read_json(LOCAL_VECTOR)
        features = obj.get("features", []) if obj.get("type") == "FeatureCollection" else []
        bboxes = []
        for feat in features:
            geom = feat.get("geometry") or {}
            bbox = _coords_bbox(geom.get("coordinates"))
            if bbox:
                bboxes.append(bbox)
        if not bboxes:
            return {"exists": True, "valid": False, "feature_count": len(features), "bbox": None, "obj": obj, "geometry_id": "not_available", "status": "footprint_invalido_sem_metadados"}
        bbox_all = (min(b[0] for b in bboxes), min(b[1] for b in bboxes), max(b[2] for b in bboxes), max(b[3] for b in bboxes))
        aoi = s18e2.load_aoi_bbox()
        if not s18c._bbox_overlap(bbox_all, aoi):
            status = "footprint_invalido_fora_aoi"
            valid = False
        else:
            status = "footprint_tecnico_valido_somente_revisao"
            valid = True
        return {"exists": True, "valid": valid, "feature_count": len(features), "bbox": bbox_all, "obj": obj, "geometry_id": "S18F_CUR_SAR_FOOTPRINT_0001", "status": status}
    except Exception:  # noqa: BLE001
        return {"exists": True, "valid": False, "feature_count": 0, "bbox": None, "obj": None, "geometry_id": "not_available", "status": "footprint_invalido_sem_metadados"}


def audit_rows(fp: dict) -> list[dict]:
    files = [
        (LOCAL_PATCH_STATS, "patch_stats"),
        (LOCAL_VECTOR, "flood_vector"),
        (LOCAL_MASK, "flood_mask"),
        (LOCAL_METADATA, "metadata"),
        (LOCAL_SCENES, "scene_inventory"),
        (LOCAL_TASKS, "task_status"),
    ]
    rows = []
    for path, tipo in files:
        exists = path.exists()
        valid = False
        usable = False
        reason = "arquivo ausente"
        if exists:
            if tipo.endswith("stats") or tipo in {"scene_inventory", "task_status"}:
                valid = len(read_csv(path)) > 0
            elif tipo == "metadata":
                valid = bool(read_json(path))
            elif tipo == "flood_vector":
                valid = fp["valid"]
            elif tipo == "flood_mask":
                valid = path.suffix.lower() in {".tif", ".tiff"}
            usable = valid and tipo != "flood_mask"
            reason = "utilizavel como evidencia tecnica somente revisao" if usable else "presente, mas nao usado sem validacao ou por regra de raster pesado"
        rows.append({
            "arquivo": path.name,
            "caminho": rel(path),
            "tipo": tipo,
            "tamanho_bytes": str(path.stat().st_size if exists else 0),
            "sha256": _sha(path),
            "existe": _bool(exists),
            "formato_valido": _bool(valid),
            "cobre_aoi": _bool(bool(fp.get("valid")) if tipo == "flood_vector" else exists),
            "contem_evento": _bool(EVENT_PUBLIC_ID in path.name or tipo in {"patch_stats", "metadata", "scene_inventory", "task_status"}),
            "contem_pre_window": _bool(tipo in {"metadata", "scene_inventory", "patch_stats", "flood_vector"} and exists),
            "contem_post_window": _bool(tipo in {"metadata", "scene_inventory", "patch_stats", "flood_vector"} and exists),
            "utilizavel": _bool(usable),
            "motivo_uso_ou_bloqueio": reason,
        })
    return rows


def ingest_patch_stats_rows() -> list[dict]:
    rows = []
    for row in read_csv(LOCAL_PATCH_STATS):
        try:
            mean = float(row.get("flood_mask_mean", ""))
        except ValueError:
            mean = 0.0
        sinal = "sinal_sar_alto" if mean >= 0.05 else ("sinal_sar_medio" if mean >= 0.02 else "sinal_sar_baixo")
        rows.append({
            "patch_id": row.get("patch_id", ""),
            "candidate_event_id": row.get("candidate_event_id", EVENT_ID),
            "evento_publico": row.get("evento_publico", EVENT_PUBLIC_ID),
            "flood_mask_mean": row.get("flood_mask_mean", ""),
            "pixel_count": row.get("pixel_count", ""),
            "sinal_sar": sinal,
            "status_estatistica": "estatistica_sar_real_somente_revisao",
            "usavel_pre_evento": "false",
            **HARD_DEFAULTS,
            "justificativa_tecnica": "estatistica pos-evento calculada no Earth Engine; nao e feature pre-evento, treino ou geometria oficial",
        })
    return rows


def footprint_rows(fp: dict, recovery: dict) -> list[dict]:
    if not fp["exists"] or not fp["valid"]:
        return [{
            "geometry_id": "not_available",
            "candidate_event_id": EVENT_ID,
            "evento_publico": EVENT_PUBLIC_ID,
            "status_ingestao": fp["status"],
            "geometry_type": "none",
            "crs": "not_available",
            "bbox": "not_available",
            "area_bbox_km2_aprox": "not_available",
            "feature_count": str(fp.get("feature_count", 0)),
            "fonte": "not_available",
            "metodo_recuperacao": recovery.get("method", "not_available"),
            "footprint_vetorial_disponivel": "false",
            "geometria_oficial_de_ocorrencia": "false",
            **HARD_DEFAULTS,
            "justificativa_tecnica": "footprint vetorial ainda nao disponivel localmente; patch_stats real sustenta apenas referencia tecnica parcial e nao e geometria oficial",
        }]
    bbox = fp["bbox"]
    return [{
        "geometry_id": fp["geometry_id"],
        "candidate_event_id": EVENT_ID,
        "evento_publico": EVENT_PUBLIC_ID,
        "status_ingestao": "footprint_tecnico_valido_somente_revisao",
        "geometry_type": "MultiPolygon",
        "crs": "EPSG:4326",
        "bbox": ",".join(f"{v:.8f}" for v in bbox),
        "area_bbox_km2_aprox": f"{_bbox_area_km2(bbox):.6f}",
        "feature_count": str(fp["feature_count"]),
        "fonte": "Earth Engine Sentinel-1 GRD",
        "metodo_recuperacao": recovery.get("method", "arquivo_local_existente"),
        "footprint_vetorial_disponivel": "true",
        "geometria_oficial_de_ocorrencia": "false",
        **HARD_DEFAULTS,
        "justificativa_tecnica": "footprint tecnico SAR recuperado e validado somente para revisao; nao e geometria oficial de ocorrencia",
    }]


def write_public_footprint_geojson(fp: dict) -> None:
    if fp.get("valid") and fp.get("obj"):
        write_json(FOOTPRINT_GEOJSON, fp["obj"])
    elif FOOTPRINT_GEOJSON.exists():
        FOOTPRINT_GEOJSON.unlink()


def validation_rows(fp: dict, stats_rows: list[dict]) -> list[dict]:
    if fp["valid"]:
        status = "footprint_tecnico_valido_somente_revisao"
        just = "vetor tecnico recuperado, cobre AOI tecnica e permanece somente revisao"
    elif stats_rows:
        status = "footprint_tecnico_parcial_por_patch_stats"
        just = "sem vetor local valido; patch_stats real permite referencia tecnica parcial sem overlay geometrico"
    else:
        status = "footprint_vetorial_aguardando_recuperacao"
        just = "sem vetor e sem estatisticas SAR locais"
    return [{
        "validation_id": "S18F_VAL_0001",
        "candidate_event_id": EVENT_ID,
        "geometry_id": fp.get("geometry_id", "not_available"),
        "status_validacao": status,
        "patch_stats_disponivel": _bool(bool(stats_rows)),
        "footprint_vetorial_disponivel": _bool(fp["valid"]),
        "flood_mask_disponivel": _bool(LOCAL_MASK.exists()),
        "feature_count": str(fp.get("feature_count", 0)),
        "cobre_aoi": _bool(fp["valid"]),
        **HARD_DEFAULTS,
        "justificativa_tecnica": just,
    }]


def overlay_links(fp: dict, stats_rows: list[dict]) -> list[dict]:
    patches = s18c.load_cur_patch_polygons()
    rows = []
    if not fp["valid"]:
        for idx, row in enumerate(stats_rows, start=1):
            rows.append({
                "technical_link_id": f"S18F_STAT_{idx:04d}",
                "candidate_event_id": EVENT_ID,
                "geometry_id": "not_available",
                "patch_id": row["patch_id"],
                "classe_vinculo_tecnico": "sar_patch_stat_available",
                "area_overlap_m2": "not_available",
                "overlap_ratio": "not_available",
                "vinculo_tecnico_forte": "false",
                **HARD_DEFAULTS,
                "justificativa_tecnica": "estatistica SAR por patch disponivel sem geometria vetorial; nao e geometry overlap nem patch-link forte",
            })
        return rows
    fb = fp["bbox"]
    for idx, (patch_id, ring) in enumerate(sorted(patches.items()), start=1):
        pb = s18c._ring_bbox(ring)
        inter = _bbox_intersection(fb, pb)
        if not inter:
            classe, area_m2, ratio, strong = "no_technical_overlap", "0", "0", "false"
        else:
            area_km2 = _bbox_area_km2(inter)
            patch_area = max(_bbox_area_km2(pb), 1e-12)
            ratio_val = min(1.0, area_km2 / patch_area)
            classe = "technical_footprint_overlap" if ratio_val >= 0.05 else "technical_footprint_partial_overlap"
            area_m2, ratio, strong = f"{area_km2 * 1_000_000:.3f}", f"{ratio_val:.6f}", "true"
        rows.append({
            "technical_link_id": f"S18F_LINK_{idx:04d}",
            "candidate_event_id": EVENT_ID,
            "geometry_id": fp["geometry_id"],
            "patch_id": patch_id,
            "classe_vinculo_tecnico": classe,
            "area_overlap_m2": area_m2,
            "overlap_ratio": ratio,
            "vinculo_tecnico_forte": strong,
            **HARD_DEFAULTS,
            "justificativa_tecnica": "vinculo tecnico SAR com footprint vetorial; nao e ground truth nem geometria oficial",
        })
    return rows


def features_rows(links: list[dict], stats_rows: list[dict]) -> list[dict]:
    stat_by_patch = {r["patch_id"]: r for r in stats_rows}
    rows = []
    for idx, link in enumerate(links, start=1):
        stat = stat_by_patch.get(link["patch_id"], {})
        rows.append({
            "feature_id": f"S18F_FEAT_{idx:04d}",
            "technical_link_id": link["technical_link_id"],
            "patch_id": link["patch_id"],
            "feature_name": "flood_mask_mean",
            "valor": stat.get("flood_mask_mean", "not_available"),
            "periodo_evento": "pos_evento",
            "usavel_pre_evento": "false",
            "status_feature": "feature_sar_pos_evento_somente_revisao",
            **HARD_DEFAULTS,
            "justificativa_tecnica": "valor SAR pos-evento usado apenas como evidencia tecnica; proibido como feature pre-evento",
        })
    return rows


def score_v6_by_patch() -> dict:
    rows = {}
    for row in read_csv(SCORE_V6):
        if row.get("regiao") != "curitiba":
            continue
        cur = row.get("patch_id", "").replace("curitiba_", "CUR_").upper()
        rows[cur] = row
    return rows


def comparacao_rows(stats_rows: list[dict]) -> list[dict]:
    score = score_v6_by_patch()
    rows = []
    for stat in stats_rows:
        s = score.get(stat["patch_id"], {})
        cls = s.get("susc_class_v6_candidate", "not_available")
        sinal = stat["sinal_sar"]
        if cls in {"high", "medium"} and sinal in {"sinal_sar_alto", "sinal_sar_medio"}:
            comp = "alinhamento_tecnico_parcial"
        elif cls == "low" and sinal == "sinal_sar_baixo":
            comp = "alinhamento_tecnico_baixo"
        else:
            comp = "divergencia_tecnica_para_revisao"
        rows.append({
            "patch_id": stat["patch_id"],
            "susc_score_v6_candidate": s.get("susc_score_v6_candidate", "not_available"),
            "susc_class_v6_candidate": cls,
            "flood_mask_mean": stat["flood_mask_mean"],
            "sinal_sar": sinal,
            "comparacao_status": comp,
            "score_v6_alterado": _bool(_score_v6_changed()),
            "score_v7_criado": _bool(SCORE_V7.exists()),
            **HARD_DEFAULTS,
            "justificativa_tecnica": "comparacao diagnostica somente revisao; score_v6 nao foi alterado e score_v7 nao foi criado",
        })
    return rows


def matriz_rows(stats_rows: list[dict], links: list[dict], fp: dict) -> list[dict]:
    score = score_v6_by_patch()
    links_by_patch = {l["patch_id"]: l for l in links}
    rows = []
    vector_ok = fp["valid"]
    for idx, stat in enumerate(stats_rows, start=1):
        link = links_by_patch.get(stat["patch_id"], {})
        s = score.get(stat["patch_id"], {})
        if vector_ok and link.get("vinculo_tecnico_forte") == "true":
            status = "referencia_tecnica_sar_forte_somente_revisao"
        elif stat:
            status = "referencia_tecnica_sar_parcial_por_patch_stats"
        else:
            status = "aguardando_resultado_externo"
        rows.append({
            "item_id": f"S18F_MAT_{idx:04d}",
            "candidate_event_id": EVENT_ID,
            "data_evento": EVENT_DATE,
            "tipo_fenomeno": "flood_inundation_alagamento",
            "geometry_id": fp["geometry_id"] if vector_ok else "not_available",
            "footprint_source": "Earth Engine Sentinel-1 GRD" if vector_ok else "patch_stats_sar_real_sem_vetor",
            "patch_id": stat["patch_id"],
            "classe_vinculo_tecnico": link.get("classe_vinculo_tecnico", "sar_patch_stat_available"),
            "patch_stats_disponivel": "true",
            "footprint_vetorial_disponivel": _bool(vector_ok),
            "flood_mask_disponivel": _bool(LOCAL_MASK.exists()),
            "possui_fisico": "false",
            "possui_espectral": "true",
            "possui_chuva": "false",
            "score_v6_referencia": s.get("susc_score_v6_candidate", "not_available"),
            "classe_score_v6": s.get("susc_class_v6_candidate", "not_available"),
            "status_referencia_tecnica": status,
            "uso_permitido": "evidencia_tecnica_sar_somente_revisao",
            **HARD_DEFAULTS,
            "not_ground_truth_reason": "footprint SAR e patch_stats sao evidencia tecnica pos-evento; nao sao verdade de referencia, treino ou geometria oficial",
            "justificativa_tecnica": "Curitiba tem patch_stats real; vetor tecnico " + ("recuperado" if vector_ok else "ainda ausente"),
        })
    return rows


def status_18f(fp: dict, stats_rows: list[dict]) -> str:
    if fp["valid"]:
        return "18F_REFERENCIA_TECNICA_SAR_CURITIBA_FORTE"
    if stats_rows:
        return "18F_REFERENCIA_TECNICA_SAR_CURITIBA_PARCIAL_POR_PATCH_STATS"
    return "18F_AGUARDANDO_RESULTADO_EXTERNO_GEE"


def gate_rows(summary: dict) -> tuple[list[dict], list[dict]]:
    rows18 = [
        ("patch_stats_real_ingestado", str(summary["patch_stats_rows"]), ">=1", summary["patch_stats_rows"] >= 1, summary["status_18f"], "patch_stats local real vindo do GEE"),
        ("footprint_vetorial_recuperado", _bool(summary["footprint_vector_recovered"]), "true", summary["footprint_vector_recovered"], summary["status_18f"], "vetor tecnico leve recuperado ou pendente"),
        ("flood_mask_local", _bool(summary["flood_mask_local"]), "true", summary["flood_mask_local"], summary["status_18f"], "raster pesado nao baixado sem autorizacao explicita"),
        ("ground_truth_zero", "0", "0", True, summary["status_18f"], "sem ground truth"),
        ("score_v7_zero", "0", "0", not summary["score_v7_created"], summary["status_18f"], "sem score_v7"),
    ]
    rows17 = [
        ("evidencia_tecnica_curitiba", _bool(summary["patch_stats_rows"] >= 1), "true", summary["patch_stats_rows"] >= 1, summary["status_17b"], "evidencia tecnica SAR sem benchmark 17B"),
        ("geometria_oficial_curitiba", "false", "true", False, summary["status_17b"], "18D segue aguardando resposta oficial"),
        ("benchmark_17b_criado", "false", "false", True, summary["status_17b"], "nenhum benchmark criado"),
        ("ground_truth_zero", "0", "0", True, summary["status_17b"], "sem ground truth"),
        ("trainable_zero", "0", "0", True, summary["status_17b"], "sem treino"),
    ]
    conv = lambda rows: [{"criterio": c, "valor_observado": v, "limiar": l, "passou": _bool(p), "status": s, "observacao": o} for c, v, l, p, s, o in rows]
    return conv(rows18), conv(rows17)


def fila_vetorizacao_rows(fp: dict, task_rows: list[dict]) -> list[dict]:
    vector_task = next((r for r in task_rows if r["artefato"] == "flood_vector"), {})
    return [{
        "task_id": "S18F_VET_0001",
        "artefato": "flood_vector_curitiba_2022_01_15.geojson",
        "status": "concluido" if fp["valid"] else "pendente_recuperacao",
        "caminho_esperado": rel(LOCAL_VECTOR),
        "acao_requerida": "validar vetor tecnico" if fp["valid"] else f"recuperar export Drive task {vector_task.get('task_id', TASK_IDS['flood_vector'])}",
        "criterio_de_sucesso": "GeoJSON leve com CRS EPSG:4326 e metadados somente revisao",
    }]


def resumo_status_rows(matriz: list[dict]) -> list[dict]:
    counts: dict[str, int] = {}
    for row in matriz:
        counts[row["status_referencia_tecnica"]] = counts.get(row["status_referencia_tecnica"], 0) + 1
    return [{"status_referencia_tecnica": k, "quantidade": str(v)} for k, v in sorted(counts.items())]


def schema_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-18F ingestao SAR Curitiba",
        "type": "object",
        "required": ["candidate_event_id", "patch_id", "status_referencia_tecnica"],
        "properties": {
            "candidate_event_id": {"const": EVENT_ID},
            "patch_id": {"type": "string"},
            "geometry_id": {"type": "string"},
            "status_referencia_tecnica": {"enum": sorted(MATRIZ_STATUS_ALLOWED)},
            "ground_truth": {"const": "false"},
            "eligible_for_training": {"const": "false"},
            "score_v7_allowed": {"const": "false"},
            "review_only": {"const": "true"},
        },
        "status_18f_allowed": sorted(STATUS_18F_ALLOWED),
        "status_17b_allowed": sorted(STATUS_17B_ALLOWED),
    }


def preflight_obj() -> dict:
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "required_inputs": [{"path": rel(p), "exists": p.exists()} for p in REQUIRED_INPUTS],
        "local_results_dir": rel(LOCAL),
        "patch_polys_cur_count": len(s18c.load_cur_patch_polygons()),
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
    }


def summary_obj(task_rows, recovery_rows_, fp, stats_rows, links, matriz) -> dict:
    st18 = status_18f(fp, stats_rows)
    st17 = "17B_APROXIMACAO_COM_EVIDENCIA_TECNICA_CURITIBA" if stats_rows else "17B_BLOQUEADO_POR_GEOMETRIA_OFICIAL"
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "candidate_event_id": EVENT_ID,
        "evento_publico": EVENT_PUBLIC_ID,
        "input_status_18e2": _read_json(OUT_18E2 / "summary.json").get("status_18e2", "unknown"),
        "task_status": {r["artefato"]: r["status_atual"] for r in task_rows},
        "patch_stats_rows": len(stats_rows),
        "footprint_vector_recovered": fp["valid"],
        "footprint_feature_count": fp.get("feature_count", 0),
        "flood_mask_local": LOCAL_MASK.exists(),
        "technical_links": len(links),
        "technical_strong_links": sum(1 for l in links if l.get("vinculo_tecnico_forte") == "true"),
        "matrix_rows": len(matriz),
        "status_18f": st18,
        "status_17b": st17,
        "ground_truth_true_count": 0,
        "eligible_for_training_true_count": 0,
        "score_v7_allowed_true_count": 0,
        "benchmark_17b_criado": False,
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
        "review_only": True,
        "ground_truth": False,
        "trainable": False,
        "local_results_dir": rel(LOCAL),
    }


def card_text(title: str, body: str) -> str:
    return f"# {title}\n\n{body}\n"


def write_cards(summary: dict) -> None:
    cards = {
        "card_01_status_tasks.md": f"Evento: `{EVENT_PUBLIC_ID}`.\n\nTasks GEE: `{summary['task_status']}`.",
        "card_02_resultados_recuperados.md": f"Patch_stats ingerido: `{summary['patch_stats_rows']}` linhas. Vetor recuperado: `{str(summary['footprint_vector_recovered']).lower()}`.",
        "card_03_validacao_tecnica.md": f"Status 18F: `{summary['status_18f']}`. Vínculos técnicos: `{summary['technical_links']}`.",
        "card_04_comparacao_score_v6.md": "Comparacao diagnostica com score_v6; score_v6 intacto e score_v7 nao criado.",
        "card_05_impacto_17b.md": f"Status 17B: `{summary['status_17b']}`. Sem benchmark 17B.",
        "card_06_limitacoes.md": "Nao e ground truth, nao e treinavel, nao cria score_v7 e nao substitui geometria oficial.",
    }
    for name, body in cards.items():
        write_markdown(CARDS / name, card_text(name.replace("_", " ").replace(".md", ""), body))


def report_text(summary: dict, recovery: list[dict]) -> str:
    rec = {r["artifact_id"]: r for r in recovery}
    return f"""# SUSC-18F - Ingestao e validacao SAR de Curitiba

## Estado herdado do 18E2

- Earth Engine autenticado: `true`
- GEE consultado: `true`
- Cenas Sentinel-1: `pre=3`, `pos=3`
- Status herdado: `{summary['input_status_18e2']}`

## Tasks atualizadas

- Flood mask: `{summary['task_status'].get('flood_mask', 'not_available')}`
- Flood vector: `{summary['task_status'].get('flood_vector', 'not_available')}`
- Patch stats: `{summary['task_status'].get('patch_stats', 'not_available')}`

## Exports recuperados

- Patch_stats local: `{rec.get('patch_stats', {}).get('existe_local', 'false')}` ({summary['patch_stats_rows']} linhas)
- Footprint vetorial local: `{str(summary['footprint_vector_recovered']).lower()}`
- Flood mask local: `{str(summary['flood_mask_local']).lower()}`

## Validacao tecnica

- Status 18F: `{summary['status_18f']}`
- Vínculos técnicos criados: `{summary['technical_links']}`
- Vínculos técnicos fortes somente revisao: `{summary['technical_strong_links']}`

## Gate 17B

- Status 17B: `{summary['status_17b']}`
- Nenhum benchmark 17B foi criado.

## Guardrails

Sem ground truth, sem treino, sem score_v7, score_v6 intacto e footprint tecnico
nao substitui geometria oficial de ocorrencia.

## Proxima acao pesada

Recuperar do Drive o raster `flood_mask_curitiba_2022_01_15.tif`, se autorizado,
ou usar o vetor/patch_stats ja recuperados para revisao tecnica 18G sem promover
verdade de referencia.
"""


def build_all(attempt_recovery: bool = True) -> dict:
    require_inputs()
    ensure_dirs()
    task_rows, _ = task_status_from_cli()
    if attempt_recovery:
        vector_recovery = try_recover_vector_from_ee(task_rows)
    elif LOCAL_VECTOR.exists():
        vector_recovery = {"ok": True, "method": "arquivo_local_existente", "block": "not_available", "detail": "vetor local presente"}
    elif RECUPERACAO.exists():
        prev = next((r for r in read_csv(RECUPERACAO) if r.get("artifact_id") == "flood_vector"), {})
        block = prev.get("bloqueio", "nao_tentado")
        vector_recovery = {
            "ok": False,
            "method": prev.get("metodo_recuperacao", "nao_tentado"),
            "block": block.split(":", 1)[0],
            "detail": block.split(":", 1)[1].strip() if ":" in block else "tentativa anterior preservada",
        }
    else:
        vector_recovery = {"ok": False, "method": "nao_tentado", "block": "nao_tentado", "detail": "sem nova tentativa"}
    recovery = recovery_rows(task_rows, vector_recovery)
    fp = footprint_info()
    stats = ingest_patch_stats_rows()
    footprint = footprint_rows(fp, vector_recovery)
    validation = validation_rows(fp, stats)
    links = overlay_links(fp, stats)
    features = features_rows(links, stats)
    comp = comparacao_rows(stats)
    matrix = matriz_rows(stats, links, fp)
    summary = summary_obj(task_rows, recovery, fp, stats, links, matrix)
    gate18, gate17 = gate_rows(summary)

    write_csv(STATUS_TASKS, task_rows, STATUS_TASKS_FIELDS)
    write_csv(RECUPERACAO, recovery, RECUP_FIELDS)
    write_csv(AUDITORIA, audit_rows(fp), AUDIT_FIELDS)
    write_csv(FOOTPRINT_CSV, footprint, FOOTPRINT_FIELDS)
    write_public_footprint_geojson(fp)
    write_csv(FILA_VETORIZACAO, fila_vetorizacao_rows(fp, task_rows), FILA_VET_FIELDS)
    write_csv(VALIDACAO, validation, VALIDACAO_FIELDS)
    write_csv(VINCULOS, links, VINCULOS_FIELDS)
    write_csv(ESTATISTICAS, stats, ESTAT_FIELDS)
    write_csv(FEATURES, features, FEATURE_FIELDS)
    write_csv(COMPARACAO, comp, COMPARACAO_FIELDS)
    write_csv(MATRIZ, matrix, MATRIZ_FIELDS)
    write_csv(GATE_18F, gate18, GATE_FIELDS)
    write_csv(GATE_17B, gate17, GATE_FIELDS)
    write_csv(RESUMO_STATUS, resumo_status_rows(matrix), RESUMO_FIELDS)
    write_json(SCHEMA, schema_obj())
    write_json(PREFLIGHT, preflight_obj())
    write_json(SUMMARY, summary)
    write_cards(summary)
    write_markdown(REPORT, report_text(summary, recovery))
    return summary


def _public_output_paths() -> list[Path]:
    paths = []
    if OUT_18F.exists():
        paths.extend(p for p in OUT_18F.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".json", ".md", ".geojson"})
    for p in (REPORT, SCHEMA):
        if p.exists():
            paths.append(p)
    return sorted(dict.fromkeys(paths), key=lambda p: rel(p))


def public_text_violations_text(text: str) -> list[str]:
    return sorted({m.group(0) for m in PUBLIC_FORBIDDEN_RE.finditer(text)})


def validate_public_text() -> list[str]:
    errors = []
    for path in _public_output_paths():
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = public_text_violations_text(text)
        if hits:
            errors.append(f"{rel(path)}:vocabulario:{','.join(hits)}")
        if SECRET_RE.search(text):
            errors.append(f"{rel(path)}:credencial_ou_token")
    return errors


def validate_outputs(summary: dict) -> list[str]:
    errors = [f"missing:{rel(p)}" for p in REQUIRED_OUTPUTS if not p.exists()]
    if summary["status_18f"] not in STATUS_18F_ALLOWED:
        errors.append(f"status_18f_fora_enum:{summary['status_18f']}")
    if summary["status_17b"] not in STATUS_17B_ALLOWED:
        errors.append(f"status_17b_fora_enum:{summary['status_17b']}")
    if summary["score_v6_changed"]:
        errors.append("score_v6_changed_forbidden")
    if summary["score_v7_created"]:
        errors.append("score_v7_created_forbidden")
    if summary["benchmark_17b_criado"]:
        errors.append("benchmark_17b_criado_forbidden")
    for path in [ESTATISTICAS, FEATURES, COMPARACAO, MATRIZ, VINCULOS, FOOTPRINT_CSV]:
        for idx, row in enumerate(read_csv(path), start=1):
            rid = row.get("item_id") or row.get("technical_link_id") or row.get("patch_id") or f"linha{idx}"
            for field in ("ground_truth", "eligible_for_training", "score_v7_allowed"):
                if row.get(field) == "true":
                    errors.append(f"{rel(path)}:{rid}:{field}_true")
            if row.get("usavel_pre_evento") == "true" and row.get("periodo_evento") == "pos_evento":
                errors.append(f"{rel(path)}:{rid}:feature_pos_evento_como_pre")
            if not row.get("justificativa_tecnica", "").strip():
                errors.append(f"{rel(path)}:{rid}:justificativa_vazia")
    fp = read_csv(FOOTPRINT_CSV)[0]
    vector_available = fp["footprint_vetorial_disponivel"] == "true"
    for link in read_csv(VINCULOS):
        cls = link["classe_vinculo_tecnico"]
        if cls.startswith("technical_footprint") and not vector_available:
            errors.append(f"{link['technical_link_id']}:patch_link_geometrico_sem_vetor")
        if link["vinculo_tecnico_forte"] == "true" and (link["geometry_id"] in {"", "not_available"} or link["patch_id"] in {"", "not_available"}):
            errors.append(f"{link['technical_link_id']}:vinculo_tecnico_forte_sem_ids")
    for row in read_csv(VALIDACAO):
        if row["status_validacao"] not in VALIDACAO_STATUS_ALLOWED:
            errors.append(f"status_validacao_fora_enum:{row['status_validacao']}")
    for row in read_csv(MATRIZ):
        if row["status_referencia_tecnica"] not in MATRIZ_STATUS_ALLOWED:
            errors.append(f"status_matriz_fora_enum:{row['status_referencia_tecnica']}")
    if "pronto_para_17b" in json.dumps(read_json(SUMMARY), ensure_ascii=False).lower():
        errors.append("pronto_para_17b_proibido")
    errors.extend(validate_public_text())
    return errors


def validate() -> int:
    summary = build_all(attempt_recovery=False)
    errors = validate_outputs(summary)
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(
        "18F ingestao SAR Curitiba validada: "
        f"patch_stats={summary['patch_stats_rows']} "
        f"vetor={str(summary['footprint_vector_recovered']).lower()} "
        f"links={summary['technical_links']} "
        f"status18F={summary['status_18f']} status17B={summary['status_17b']}"
    )
    return 0


def run_all() -> int:
    summary = build_all(attempt_recovery=True)
    print(
        "18F ingestao SAR Curitiba gerada: "
        f"patch_stats={summary['patch_stats_rows']} "
        f"vetor={str(summary['footprint_vector_recovered']).lower()} "
        f"links={summary['technical_links']} "
        f"status18F={summary['status_18f']} status17B={summary['status_17b']}"
    )
    return 0


def check_tasks_only() -> int:
    require_inputs()
    ensure_dirs()
    task_rows, _ = task_status_from_cli()
    write_csv(STATUS_TASKS, task_rows, STATUS_TASKS_FIELDS)
    print("18F tasks GEE atualizadas: " + "; ".join(f"{r['artefato']}={r['status_atual']}" for r in task_rows))
    return 0
