"""SUSC-18E2 execucao controlada Sentinel-1/GEE para Curitiba.

O modulo tenta executar o pacote Sentinel-1 criado no 18E por rotas locais
disponiveis. Quando Earth Engine nao esta autenticado ou nao ha permissao, ele
registra bloqueio tecnico reexecutavel. Nao inventa raster, footprint, patch-link
ou estatistica SAR.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

import susc_18c_curitiba_geometria_common as s18c  # noqa: E402
from susc_io import ensure_dir, read_csv, read_json, rel, sha256_file, write_csv, write_json, write_markdown  # noqa: E402

OUT_18E = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_18e_footprint_tecnico_sar_curitiba"
OUT_18D = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_18d_protocolo_externo_curitiba"
OUT_18C = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_18c_aquisicao_geometria_oficial_curitiba"
OUT_DATA = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_18e2_execucao_controlada_sentinel1_curitiba"
REPORTS = ROOT / "outputs_public" / "reports"

AOI_GEOJSON_18E = OUT_18E / "curitiba_aoi_tecnica_43_patches.geojson"
AOI_CSV_18E = OUT_18E / "curitiba_aoi_tecnica_43_patches.csv"
WINDOWS_18E = OUT_18E / "janelas_sar_curitiba.csv"
GEE_SCRIPT_18E = OUT_18E / "pacote_gee" / "gee_sentinel1_curitiba_2022_01_15.js"
GEE_README_18E = OUT_18E / "pacote_gee" / "gee_sentinel1_curitiba_2022_01_15.md"
GEE_MANIFEST_18E = OUT_18E / "pacote_gee" / "gee_export_manifest_curitiba.csv"
GEE_SCHEMA_18E = OUT_18E / "pacote_gee" / "expected_outputs_schema.json"
SUMMARY_18E = OUT_18E / "summary.json"
SUMMARY_18D = OUT_18D / "summary.json"
SUMMARY_18C = OUT_18C / "summary.json"
SCORE_V6 = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv"

LOCAL_RESULTS = ROOT / "local_runs" / "suscetibilidade" / "18e_sar_curitiba" / "resultados_gee"
LOCAL_METADATA = LOCAL_RESULTS / "execution_metadata.json"
LOCAL_LOG = LOCAL_RESULTS / "execution_log.txt"
LOCAL_SCENES = LOCAL_RESULTS / "gee_scene_inventory_curitiba_2022_01_15.csv"
LOCAL_TASKS = LOCAL_RESULTS / "gee_tasks_curitiba_2022_01_15.csv"
LOCAL_PATCH_STATS = LOCAL_RESULTS / "patch_stats_curitiba_2022_01_15.csv"
LOCAL_FLOOD_VECTOR = LOCAL_RESULTS / "flood_vector_curitiba_2022_01_15.geojson"
LOCAL_FLOOD_MASK = LOCAL_RESULTS / "flood_mask_curitiba_2022_01_15.tif"

AUDIT_ENV = OUT_DATA / "auditoria_ambiente_earth_engine.csv"
MANIFEST = OUT_DATA / "manifesto_resultados_gee.csv"
BLOCKS = OUT_DATA / "bloqueio_execucao_gee.csv"
HANDOFF = OUT_DATA / "handoff_18f.csv"
GATE = OUT_DATA / "gate_execucao_sentinel1_curitiba.csv"
SUMMARY = OUT_DATA / "summary.json"
PREFLIGHT = OUT_DATA / "preflight.json"
PUBLIC_SCENES = OUT_DATA / "inventario_cenas_sentinel1.csv"
PUBLIC_TASKS = OUT_DATA / "tarefas_gee_curitiba.csv"
PUBLIC_STATS = OUT_DATA / "estatisticas_execucao_sentinel1.csv"
REPORT = REPORTS / "SUSC_18E2_EXECUCAO_CONTROLADA_SENTINEL1_CURITIBA.md"

EVENT_ID = "S17C_REF_0060"
EVENT_PUBLIC_ID = "CUR_2022_01_15"
PRIMARY_WINDOW_ID = "S18E_SAR_WIN_30D"

REQUIRED_INPUTS = [
    SUMMARY_18E,
    SUMMARY_18D,
    SUMMARY_18C,
    AOI_GEOJSON_18E,
    AOI_CSV_18E,
    WINDOWS_18E,
    GEE_SCRIPT_18E,
    GEE_README_18E,
    GEE_MANIFEST_18E,
    GEE_SCHEMA_18E,
    SCORE_V6,
]

REQUIRED_PUBLIC_OUTPUTS = [AUDIT_ENV, MANIFEST, BLOCKS, HANDOFF, GATE, SUMMARY, PREFLIGHT, REPORT]

HARD_DEFAULTS = {
    "ground_truth": "false",
    "eligible_for_training": "false",
    "score_v7_allowed": "false",
    "review_only": "true",
}

ENV_STATUS_ALLOWED = {
    "autenticado_pronto_para_execucao",
    "biblioteca_ausente",
    "nao_autenticado",
    "permissao_insuficiente",
    "ambiente_sem_internet",
    "ambiente_indeterminado",
}

GATE_STATUS_ALLOWED = {
    "18E2_EXECUCAO_GEE_CONCLUIDA_COM_RESULTADOS",
    "18E2_TAREFAS_GEE_INICIADAS_AGUARDANDO_CONCLUSAO",
    "18E2_BLOQUEADO_POR_AUTENTICACAO",
    "18E2_BLOQUEADO_POR_PERMISSAO",
    "18E2_BLOQUEADO_POR_AMBIENTE",
    "18E2_SEM_CENAS_SENTINEL1_COMPATIVEIS",
    "18E2_BLOQUEADO_FAIL_CLOSED",
}

PUBLIC_FORBIDDEN_RE = re.compile(
    r"\b(?:agentic|agente|codex|llm|ia|automacao|automatizacao|automatizado|automatica|automatico)\b",
    re.IGNORECASE,
)
SECRET_RE = re.compile(
    r"(-----BEGIN [A-Z ]*PRIVATE KEY-----|ya29\\.|AIza[0-9A-Za-z_\\-]{20,}|private_key\\s*[:=]|client_secret\\s*[:=]|refresh_token\\s*[:=])",
    re.IGNORECASE,
)

AUDIT_FIELDS = [
    "item",
    "comando_testado",
    "disponivel",
    "status",
    "resultado_obtido",
    "detalhe_tecnico",
]
MANIFEST_FIELDS = [
    "arquivo",
    "caminho_local",
    "tipo",
    "tamanho_bytes",
    "sha256",
    "produzido_localmente",
    "exportado_gee",
    "task_id",
    "status",
    "observacao",
]
BLOCK_FIELDS = [
    "bloqueio_id",
    "etapa",
    "tipo_bloqueio",
    "detalhe_tecnico",
    "comando_testado",
    "resultado_obtido",
    "acao_corretiva_exata",
    "reexecutavel_apos_correcao",
]
HANDOFF_FIELDS = [
    "resultado_gee_disponivel",
    "flood_vector_disponivel",
    "flood_mask_disponivel",
    "patch_stats_disponivel",
    "execution_metadata_disponivel",
    "pronto_para_18f",
    "motivo_se_nao_pronto",
    "caminho_resultados_local_runs",
]
GATE_FIELDS = [
    "gate_id",
    "status_18e2",
    "rota_python_earth_engine",
    "rota_cli_earth_engine",
    "rota_js_gee",
    "earth_engine_autenticado",
    "gee_consultado",
    "cenas_pre",
    "cenas_pos",
    "export_iniciado",
    "task_ids",
    "pronto_para_18f",
    "ground_truth",
    "eligible_for_training",
    "score_v7_allowed",
    "benchmark_17b_criado",
    "observacao",
]


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
    text = "" if value is None else str(value)
    text = re.sub(r"(?i)(token|secret|private_key|refresh_token)=[^\\s;]+", r"\\1=<oculto>", text)
    text = text.replace(str(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")), "<caminho_oculto>") if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") else text
    return text.replace("\r", " ").replace("\n", " ")[:600]


def _run_cmd(args: list[str], timeout: int = 30) -> dict:
    try:
        r = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False)
        out = (r.stdout or r.stderr or "").strip()
        return {"returncode": r.returncode, "output": _safe_text(out)}
    except FileNotFoundError:
        return {"returncode": 127, "output": "comando_nao_encontrado"}
    except subprocess.TimeoutExpired:
        return {"returncode": 124, "output": "tempo_limite_excedido"}


def _sha_or_na(path: Path) -> str:
    return sha256_file(path) if path.exists() and path.is_file() else "not_available"


def _file_manifest_row(path: Path, tipo: str, exported: bool = False, task_id: str = "", status: str = "presente", obs: str = "") -> dict:
    exists = path.exists() and path.is_file()
    size = path.stat().st_size if exists else 0
    return {
        "arquivo": path.name,
        "caminho_local": rel(path),
        "tipo": tipo,
        "tamanho_bytes": str(size),
        "sha256": _sha_or_na(path),
        "produzido_localmente": _bool(exists),
        "exportado_gee": _bool(exported),
        "task_id": task_id or "not_available",
        "status": status if exists or exported else "nao_produzido",
        "observacao": obs or ("arquivo local real" if exists else "arquivo nao presente"),
    }


def _required_inputs_status() -> list[dict]:
    return [{"path": rel(p), "exists": p.exists()} for p in REQUIRED_INPUTS]


def require_inputs() -> None:
    missing = [rel(p) for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(missing))


def ensure_local_results_dir() -> Path:
    return ensure_dir(LOCAL_RESULTS)


def load_primary_window() -> dict:
    rows = read_csv(WINDOWS_18E)
    for row in rows:
        if row.get("window_id") == PRIMARY_WINDOW_ID:
            return row
    if rows:
        return rows[0]
    raise FileNotFoundError(rel(WINDOWS_18E))


def _end_exclusive(date_text: str) -> str:
    return (datetime.strptime(date_text, "%Y-%m-%d").date() + timedelta(days=1)).isoformat()


def load_aoi_bbox() -> tuple[float, float, float, float]:
    row = read_csv(AOI_CSV_18E)[0]
    return (float(row["min_lon"]), float(row["min_lat"]), float(row["max_lon"]), float(row["max_lat"]))


def js_preservation_status() -> dict:
    if not GEE_SCRIPT_18E.exists():
        return {"ok": False, "detail": "script_js_ausente"}
    text = GEE_SCRIPT_18E.read_text(encoding="utf-8", errors="ignore")
    required = ["COPERNICUS/S1_GRD", "Export.image.toDrive", "CUR_2022_01_15", "S17C_REF_0060"]
    missing = [x for x in required if x not in text]
    braces_ok = text.count("{") == text.count("}") and text.count("(") == text.count(")")
    return {
        "ok": not missing and braces_ok,
        "detail": "preservado" if not missing and braces_ok else f"pendencias:{','.join(missing) or 'balanceamento'}",
    }


def environment_audit_rows(init_status: dict | None = None) -> list[dict]:
    rows = []
    py = _run_cmd([sys.executable, "--version"])
    rows.append({
        "item": "python",
        "comando_testado": "python --version",
        "disponivel": _bool(py["returncode"] == 0),
        "status": "ambiente_indeterminado" if py["returncode"] != 0 else "autenticado_pronto_para_execucao",
        "resultado_obtido": py["output"],
        "detalhe_tecnico": "interpretador Python disponivel" if py["returncode"] == 0 else "interpretador Python indisponivel",
    })
    ee_present = importlib.util.find_spec("ee") is not None
    rows.append({
        "item": "pacote_ee",
        "comando_testado": "python -c \"import ee\"",
        "disponivel": _bool(ee_present),
        "status": "ambiente_indeterminado" if ee_present else "biblioteca_ausente",
        "resultado_obtido": "import_ok" if ee_present else "import_fail",
        "detalhe_tecnico": "pacote ee encontrado" if ee_present else "instalar earthengine-api",
    })
    rows.append({
        "item": "earthengine_api",
        "comando_testado": "python -c \"import ee\"",
        "disponivel": _bool(ee_present),
        "status": "ambiente_indeterminado" if ee_present else "biblioteca_ausente",
        "resultado_obtido": "disponivel" if ee_present else "ausente",
        "detalhe_tecnico": "biblioteca Python Earth Engine disponivel" if ee_present else "biblioteca Python Earth Engine ausente",
    })
    earthengine_path = shutil.which("earthengine")
    rows.append({
        "item": "earthengine_cli",
        "comando_testado": "earthengine --help",
        "disponivel": _bool(bool(earthengine_path)),
        "status": "ambiente_indeterminado" if earthengine_path else "biblioteca_ausente",
        "resultado_obtido": "comando_disponivel" if earthengine_path else "comando_ausente",
        "detalhe_tecnico": "CLI Earth Engine disponivel" if earthengine_path else "CLI Earth Engine ausente",
    })
    rows.append({
        "item": "google_application_credentials",
        "comando_testado": "$env:GOOGLE_APPLICATION_CREDENTIALS",
        "disponivel": _bool(bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))),
        "status": "ambiente_indeterminado",
        "resultado_obtido": "variavel_presente_valor_oculto" if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") else "variavel_ausente",
        "detalhe_tecnico": "valor nao impresso por seguranca",
    })
    gcloud_path = shutil.which("gcloud")
    rows.append({
        "item": "gcloud",
        "comando_testado": "gcloud --version",
        "disponivel": _bool(bool(gcloud_path)),
        "status": "ambiente_indeterminado",
        "resultado_obtido": "comando_disponivel" if gcloud_path else "comando_ausente",
        "detalhe_tecnico": "gcloud disponivel" if gcloud_path else "gcloud ausente",
    })
    internet = test_internet()
    rows.append({
        "item": "internet",
        "comando_testado": "GET https://earthengine.googleapis.com",
        "disponivel": _bool(internet["available"]),
        "status": "ambiente_indeterminado" if internet["available"] else "ambiente_sem_internet",
        "resultado_obtido": internet["result"],
        "detalhe_tecnico": internet["detail"],
    })
    init_status = init_status or try_ee_initialize()
    rows.append({
        "item": "ee_initialize",
        "comando_testado": "ee.Initialize()",
        "disponivel": _bool(init_status["ok"]),
        "status": init_status["status"],
        "resultado_obtido": init_status["result"],
        "detalhe_tecnico": init_status["detail"],
    })
    writable = test_local_writable()
    rows.append({
        "item": "pasta_local_destino",
        "comando_testado": f"write_test {rel(LOCAL_RESULTS)}",
        "disponivel": _bool(writable["ok"]),
        "status": "ambiente_indeterminado" if writable["ok"] else "ambiente_sem_internet",
        "resultado_obtido": writable["result"],
        "detalhe_tecnico": writable["detail"],
    })
    rows.append({
        "item": "permissao_exportacao",
        "comando_testado": "Export.image.toDrive/table.toDrive" if init_status["ok"] else "nao_testado_sem_autenticacao",
        "disponivel": "false" if not init_status["ok"] else "true",
        "status": "ambiente_indeterminado" if init_status["ok"] else init_status["status"],
        "resultado_obtido": "preparavel_apos_initialize" if init_status["ok"] else "nao_testado",
        "detalhe_tecnico": "permissao efetiva so e confirmada ao iniciar tarefa" if init_status["ok"] else "inicializacao Earth Engine nao concluida",
    })
    return rows


def test_internet() -> dict:
    try:
        with urllib.request.urlopen("https://earthengine.googleapis.com", timeout=8) as resp:
            return {"available": True, "result": f"http_{resp.status}", "detail": "endpoint acessivel"}
    except urllib.error.HTTPError as exc:
        return {"available": True, "result": f"http_{exc.code}", "detail": "endpoint respondeu; conectividade existe"}
    except Exception as exc:
        return {"available": False, "result": exc.__class__.__name__, "detail": _safe_text(str(exc))}


def test_local_writable() -> dict:
    ensure_local_results_dir()
    test_path = LOCAL_RESULTS / ".susc_18e2_write_test"
    try:
        test_path.write_text("ok", encoding="utf-8")
        test_path.unlink(missing_ok=True)
        return {"ok": True, "result": "gravavel", "detail": rel(LOCAL_RESULTS)}
    except Exception as exc:
        return {"ok": False, "result": "falha_gravacao", "detail": _safe_text(str(exc))}


def try_ee_initialize() -> dict:
    if importlib.util.find_spec("ee") is None:
        return {"ok": False, "status": "biblioteca_ausente", "result": "pacote_ee_ausente", "detail": "instalar earthengine-api"}
    try:
        import ee  # type: ignore

        ee.Initialize()
        try:
            ee.data.getAssetRoots()
        except Exception:
            pass
        return {"ok": True, "status": "autenticado_pronto_para_execucao", "result": "initialize_ok", "detail": "sessao Earth Engine inicializada"}
    except Exception as exc:
        text = _safe_text(str(exc))
        low = text.lower()
        if "permission" in low or "denied" in low or "forbidden" in low:
            status = "permissao_insuficiente"
        elif "credentials" in low or "authenticate" in low or "oauth" in low or "not found" in low or "please authorize" in low:
            status = "nao_autenticado"
        else:
            status = "ambiente_indeterminado"
        return {"ok": False, "status": status, "result": exc.__class__.__name__, "detail": text}


def route_b_cli_status() -> dict:
    help_status = _run_cmd(["earthengine", "--help"], timeout=30)
    task_status = _run_cmd(["earthengine", "task", "list"], timeout=45)
    available = help_status["returncode"] in {0, 1} and help_status["output"] != "comando_nao_encontrado"
    authenticated = task_status["returncode"] == 0 and "Traceback" not in task_status["output"]
    return {
        "available": available,
        "help_returncode": help_status["returncode"],
        "help_output": help_status["output"],
        "task_returncode": task_status["returncode"],
        "task_output": task_status["output"],
        "authenticated": authenticated,
        "limitation": "CLI local nao executa o JavaScript do Code Editor diretamente; rota Python e a rota executavel local",
    }


def refresh_task_states_from_cli(metadata: dict) -> dict:
    tasks = metadata.get("tasks") or []
    if not tasks:
        return metadata
    task_ids = {t.get("task_id") for t in tasks if t.get("task_id") not in {"", "not_available"}}
    if not task_ids:
        return metadata
    cli = route_b_cli_status()
    output = cli.get("task_output", "")
    states = {}
    known_states = {"READY", "RUNNING", "PENDING", "SUCCEEDED", "FAILED", "CANCELLED", "COMPLETED"}
    for line in output.split(" --- "):
        parts = line.split()
        if len(parts) < 4:
            continue
        tid = parts[0]
        state = next((part for part in parts[1:] if part in known_states), parts[-1])
        if tid in task_ids:
            states[tid] = state
    for task in tasks:
        tid = task.get("task_id")
        if tid in states:
            task["state"] = states[tid]
    metadata["tasks"] = tasks
    metadata["route_b_cli"] = cli
    _write_local_csv(LOCAL_TASKS, tasks, ["task_id", "task_type", "destination", "state", "detail"])
    write_json(LOCAL_METADATA, metadata)
    return metadata


def route_c_js_status() -> dict:
    js = js_preservation_status()
    return {
        "available": GEE_SCRIPT_18E.exists(),
        "preserved": js["ok"],
        "detail": js["detail"],
        "instruction": "abrir o script no GEE Code Editor somente se as rotas locais nao estiverem autenticadas",
    }


def patch_feature_collection(ee_module):
    features = []
    patches = s18c.load_cur_patch_polygons()
    for patch_id, ring in patches.items():
        coords = [[float(x), float(y)] for x, y in ring]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        geom = ee_module.Geometry.Polygon([coords], None, False)
        features.append(ee_module.Feature(geom, {"patch_id": patch_id}))
    return ee_module.FeatureCollection(features)


def _scene_inventory_from_collection(ee_module, collection, label: str) -> list[dict]:
    fc = collection.limit(200).map(
        lambda img: ee_module.Feature(None, {
            "periodo": label,
            "scene_id": img.get("system:index"),
            "system_id": img.id(),
            "orbit_pass": img.get("orbitProperties_pass"),
            "relative_orbit": img.get("relativeOrbitNumber_start"),
            "instrument_mode": img.get("instrumentMode"),
            "product_type": img.get("productType"),
            "platform": img.get("platform_number"),
            "time_start": img.date().format("YYYY-MM-dd"),
        })
    )
    info = fc.getInfo()
    rows = []
    for feat in info.get("features", []):
        props = feat.get("properties", {})
        rows.append({
            "periodo": props.get("periodo", label),
            "scene_id": props.get("scene_id", ""),
            "system_id": props.get("system_id", ""),
            "orbit_pass": props.get("orbit_pass", ""),
            "relative_orbit": props.get("relative_orbit", ""),
            "instrument_mode": props.get("instrument_mode", ""),
            "product_type": props.get("product_type", ""),
            "platform": props.get("platform", ""),
            "time_start": props.get("time_start", ""),
        })
    return rows


def _write_local_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def run_python_earth_engine() -> dict:
    """Rota A: tenta consulta real no Earth Engine. Nao baixa raster pesado."""
    ensure_local_results_dir()
    log_lines = [f"{_now()} inicio rota Python Earth Engine"]
    init = try_ee_initialize()
    metadata = {
        "created_at": _now(),
        "route_a_python": "tentada",
        "route_b_cli": "pendente",
        "route_c_js": "pendente",
        "authenticated": init["ok"],
        "ee_initialize_status": init["status"],
        "ee_initialize_result": init["result"],
        "ee_initialize_detail": init["detail"],
        "gee_queried": False,
        "pre_scene_count": "not_available",
        "post_scene_count": "not_available",
        "export_started": False,
        "tasks": [],
        "local_files": [],
        "status_18e2": "18E2_BLOQUEADO_POR_AUTENTICACAO" if init["status"] == "nao_autenticado" else "18E2_BLOQUEADO_POR_AMBIENTE",
        "block_type": init["status"],
        "block_detail": init["detail"],
        "ground_truth": False,
        "trainable": False,
        "score_v7_created": SCORE_V7.exists(),
        "benchmark_17b_criado": False,
    }
    if not init["ok"]:
        log_lines.append(f"{_now()} ee.Initialize bloqueado: {init['status']} {init['result']}")
        write_json(LOCAL_METADATA, metadata)
        write_markdown(LOCAL_LOG, "\n".join(log_lines))
        return metadata

    try:
        import ee  # type: ignore

        window = load_primary_window()
        bbox = load_aoi_bbox()
        aoi = ee.Geometry.Rectangle(list(bbox), "EPSG:4326", False)
        pre_start = window["pre_start"]
        pre_end = _end_exclusive(window["pre_end"])
        post_start = window["post_start"]
        post_end = _end_exclusive(window["post_end"])
        collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(aoi)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("resolution_meters", 10))
        )
        pre = collection.filterDate(pre_start, pre_end)
        post = collection.filterDate(post_start, post_end)
        pre_count = int(pre.size().getInfo())
        post_count = int(post.size().getInfo())
        metadata.update({
            "gee_queried": True,
            "pre_scene_count": pre_count,
            "post_scene_count": post_count,
            "query_window_id": window["window_id"],
            "query_pre_start": pre_start,
            "query_pre_end": window["pre_end"],
            "query_post_start": post_start,
            "query_post_end": window["post_end"],
        })
        log_lines.append(f"{_now()} consulta Sentinel-1 concluida pre={pre_count} pos={post_count}")
        scene_rows = _scene_inventory_from_collection(ee, pre, "pre_evento") + _scene_inventory_from_collection(ee, post, "pos_evento")
        _write_local_csv(LOCAL_SCENES, scene_rows, [
            "periodo", "scene_id", "system_id", "orbit_pass", "relative_orbit", "instrument_mode", "product_type", "platform", "time_start",
        ])
        metadata["local_files"].append(rel(LOCAL_SCENES))
        if pre_count == 0 or post_count == 0:
            metadata.update({
                "status_18e2": "18E2_SEM_CENAS_SENTINEL1_COMPATIVEIS",
                "block_type": "ausencia_de_cenas_sentinel1",
                "block_detail": "consulta real retornou zero cenas pre ou pos",
            })
            write_json(LOCAL_METADATA, metadata)
            write_markdown(LOCAL_LOG, "\n".join(log_lines))
            return metadata

        pre_mean = pre.select(["VV", "VH"]).mean().clip(aoi)
        post_mean = post.select(["VV", "VH"]).mean().clip(aoi)
        delta_vv = post_mean.select("VV").subtract(pre_mean.select("VV")).rename("delta_vv_db")
        delta_vh = post_mean.select("VH").subtract(pre_mean.select("VH")).rename("delta_vh_db")
        flood_mask = delta_vv.lt(-1.5).And(delta_vh.lt(-1.5)).rename("flood_mask_candidate").selfMask()
        patch_fc = patch_feature_collection(ee)
        patch_stats_fc = flood_mask.unmask(0).reduceRegions(
            collection=patch_fc,
            reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
            scale=30,
            tileScale=4,
        )
        stats_info = patch_stats_fc.getInfo()
        stats_rows = []
        for feat in stats_info.get("features", []):
            props = feat.get("properties", {})
            stats_rows.append({
                "patch_id": props.get("patch_id", ""),
                "candidate_event_id": EVENT_ID,
                "evento_publico": EVENT_PUBLIC_ID,
                "flood_mask_mean": props.get("mean", ""),
                "pixel_count": props.get("count", ""),
                "status_estatistica": "estatistica_sar_real_somente_revisao",
                "ground_truth": "false",
                "eligible_for_training": "false",
                "score_v7_allowed": "false",
                "review_only": "true",
                "justificativa_tecnica": "estatistica calculada por Earth Engine sobre footprint tecnico candidato; nao e geometria oficial",
            })
        _write_local_csv(LOCAL_PATCH_STATS, stats_rows, [
            "patch_id", "candidate_event_id", "evento_publico", "flood_mask_mean", "pixel_count", "status_estatistica",
            "ground_truth", "eligible_for_training", "score_v7_allowed", "review_only", "justificativa_tecnica",
        ])
        metadata["local_files"].append(rel(LOCAL_PATCH_STATS))

        vector_fc = flood_mask.reduceToVectors(
            geometry=aoi,
            scale=30,
            geometryType="polygon",
            eightConnected=True,
            labelProperty="flood_candidate",
            maxPixels=1e8,
        )
        tasks = []
        try:
            image_task = ee.batch.Export.image.toDrive(
                image=flood_mask.toByte(),
                description="S18E2_CUR_2022_01_15_flood_mask",
                folder="REV_P_SUSC_18E2",
                fileNamePrefix="flood_mask_curitiba_2022_01_15",
                region=aoi,
                scale=10,
                crs="EPSG:4326",
                maxPixels=1e13,
            )
            image_task.start()
            tasks.append({"task_id": image_task.id, "task_type": "Export.image.toDrive", "destination": "Google Drive/REV_P_SUSC_18E2/flood_mask_curitiba_2022_01_15", "state": image_task.status().get("state", "")})
        except Exception as exc:
            tasks.append({"task_id": "not_available", "task_type": "Export.image.toDrive", "destination": "Google Drive/REV_P_SUSC_18E2/flood_mask_curitiba_2022_01_15", "state": "START_FAILED", "detail": _safe_text(str(exc))})
        try:
            vector_task = ee.batch.Export.table.toDrive(
                collection=vector_fc,
                description="S18E2_CUR_2022_01_15_flood_vector",
                folder="REV_P_SUSC_18E2",
                fileNamePrefix="flood_vector_curitiba_2022_01_15",
                fileFormat="GeoJSON",
            )
            vector_task.start()
            tasks.append({"task_id": vector_task.id, "task_type": "Export.table.toDrive", "destination": "Google Drive/REV_P_SUSC_18E2/flood_vector_curitiba_2022_01_15", "state": vector_task.status().get("state", "")})
        except Exception as exc:
            tasks.append({"task_id": "not_available", "task_type": "Export.table.toDrive", "destination": "Google Drive/REV_P_SUSC_18E2/flood_vector_curitiba_2022_01_15", "state": "START_FAILED", "detail": _safe_text(str(exc))})
        try:
            stats_task = ee.batch.Export.table.toDrive(
                collection=patch_stats_fc,
                description="S18E2_CUR_2022_01_15_patch_stats",
                folder="REV_P_SUSC_18E2",
                fileNamePrefix="patch_stats_curitiba_2022_01_15",
                fileFormat="CSV",
            )
            stats_task.start()
            tasks.append({"task_id": stats_task.id, "task_type": "Export.table.toDrive", "destination": "Google Drive/REV_P_SUSC_18E2/patch_stats_curitiba_2022_01_15", "state": stats_task.status().get("state", "")})
        except Exception as exc:
            tasks.append({"task_id": "not_available", "task_type": "Export.table.toDrive", "destination": "Google Drive/REV_P_SUSC_18E2/patch_stats_curitiba_2022_01_15", "state": "START_FAILED", "detail": _safe_text(str(exc))})
        _write_local_csv(LOCAL_TASKS, tasks, ["task_id", "task_type", "destination", "state", "detail"])
        metadata["local_files"].append(rel(LOCAL_TASKS))
        metadata["tasks"] = tasks
        metadata["export_started"] = any(t.get("task_id") not in {"", "not_available"} and t.get("state") != "START_FAILED" for t in tasks)
        metadata["status_18e2"] = (
            "18E2_TAREFAS_GEE_INICIADAS_AGUARDANDO_CONCLUSAO"
            if metadata["export_started"] else
            "18E2_BLOQUEADO_POR_PERMISSAO"
        )
        metadata["block_type"] = "not_available" if metadata["export_started"] else "permissao_insuficiente"
        metadata["block_detail"] = "tarefas iniciadas; aguardar conclusao dos exports" if metadata["export_started"] else "falha ao iniciar exports"
        log_lines.append(f"{_now()} tarefas export iniciadas={metadata['export_started']}")
    except Exception as exc:
        metadata.update({
            "status_18e2": "18E2_BLOQUEADO_POR_AMBIENTE",
            "block_type": "ambiente_indeterminado",
            "block_detail": _safe_text(str(exc)),
            "traceback_class": exc.__class__.__name__,
        })
        log_lines.append(f"{_now()} falha ambiente: {exc.__class__.__name__} {_safe_text(str(exc))}")
        log_lines.append(_safe_text(traceback.format_exc()))
    write_json(LOCAL_METADATA, metadata)
    write_markdown(LOCAL_LOG, "\n".join(log_lines))
    return metadata


def run_all_routes() -> dict:
    require_inputs()
    ensure_local_results_dir()
    metadata = run_python_earth_engine()
    metadata["route_b_cli"] = route_b_cli_status()
    metadata["route_c_js"] = route_c_js_status()
    write_json(LOCAL_METADATA, metadata)
    build_public_outputs(metadata)
    return metadata


def load_execution_metadata() -> dict:
    if LOCAL_METADATA.exists():
        return read_json(LOCAL_METADATA)
    return {
        "created_at": _now(),
        "route_a_python": "nao_executada",
        "route_b_cli": route_b_cli_status(),
        "route_c_js": route_c_js_status(),
        "authenticated": False,
        "ee_initialize_status": "ambiente_indeterminado",
        "gee_queried": False,
        "pre_scene_count": "not_available",
        "post_scene_count": "not_available",
        "export_started": False,
        "tasks": [],
        "local_files": [],
        "status_18e2": "18E2_BLOQUEADO_FAIL_CLOSED",
        "block_type": "execucao_nao_realizada",
        "block_detail": "executor ainda nao foi rodado",
        "ground_truth": False,
        "trainable": False,
        "score_v7_created": SCORE_V7.exists(),
        "benchmark_17b_criado": False,
    }


def manifest_rows(metadata: dict) -> list[dict]:
    rows = [
        _file_manifest_row(LOCAL_METADATA, "metadata", False, status="presente", obs="metadados da tentativa real"),
        _file_manifest_row(LOCAL_LOG, "log", False, status="presente", obs="log tecnico sem credenciais"),
        _file_manifest_row(LOCAL_SCENES, "scene_inventory", False, status="presente" if LOCAL_SCENES.exists() else "nao_produzido", obs="inventario de cenas se consulta GEE ocorreu"),
        _file_manifest_row(LOCAL_PATCH_STATS, "patch_stats", False, status="presente" if LOCAL_PATCH_STATS.exists() else "nao_produzido", obs="estatisticas reais somente se GEE calculou"),
        _file_manifest_row(LOCAL_FLOOD_VECTOR, "flood_vector", False, status="nao_produzido", obs="vetor local ausente; export pode ter sido iniciado"),
        _file_manifest_row(LOCAL_FLOOD_MASK, "flood_mask", False, status="nao_produzido", obs="raster local ausente; download pesado nao foi forcado"),
    ]
    for task in metadata.get("tasks", []) or []:
        rows.append({
            "arquivo": "export_gee",
            "caminho_local": task.get("destination", "not_available"),
            "tipo": task.get("task_type", "task"),
            "tamanho_bytes": "0",
            "sha256": "not_available",
            "produzido_localmente": "false",
            "exportado_gee": _bool(task.get("task_id") not in {"", "not_available"}),
            "task_id": task.get("task_id", "not_available") or "not_available",
            "status": task.get("state", "not_available") or "not_available",
            "observacao": "tarefa Earth Engine registrada; arquivo local so existe apos export concluido e copia local",
        })
    return rows


def block_rows(metadata: dict) -> list[dict]:
    status = metadata.get("status_18e2", "")
    if status in {"18E2_EXECUCAO_GEE_CONCLUIDA_COM_RESULTADOS", "18E2_TAREFAS_GEE_INICIADAS_AGUARDANDO_CONCLUSAO"}:
        return [{
            "bloqueio_id": "not_available",
            "etapa": "execucao",
            "tipo_bloqueio": "not_available",
            "detalhe_tecnico": "nenhum bloqueio terminal registrado",
            "comando_testado": "python scripts\\suscetibilidade\\run_susc_18e2_gee_sentinel1_curitiba.py",
            "resultado_obtido": status,
            "acao_corretiva_exata": "aguardar conclusao dos exports se houver tarefas pendentes",
            "reexecutavel_apos_correcao": "true",
        }]
    block_type = metadata.get("block_type") or metadata.get("ee_initialize_status") or "ambiente_indeterminado"
    if block_type == "nao_autenticado":
        action = "executar `earthengine authenticate`; ou configurar `GOOGLE_APPLICATION_CREDENTIALS`; depois reexecutar `python scripts\\suscetibilidade\\run_susc_18e2_gee_sentinel1_curitiba.py`"
    elif block_type == "biblioteca_ausente":
        action = "instalar `earthengine-api`; depois reexecutar o executor 18E2"
    elif block_type == "permissao_insuficiente":
        action = "habilitar Earth Engine no projeto e conceder permissao de exportacao; depois reexecutar o executor 18E2"
    elif block_type == "ausencia_de_cenas_sentinel1":
        action = "revisar janela temporal/AOI e repetir consulta Sentinel-1"
    else:
        action = "corrigir ambiente local indicado no detalhe tecnico e reexecutar o executor 18E2"
    return [{
        "bloqueio_id": "S18E2_BLOCK_0001",
        "etapa": "rota_a_python_earth_engine",
        "tipo_bloqueio": block_type,
        "detalhe_tecnico": _safe_text(metadata.get("block_detail") or metadata.get("ee_initialize_detail") or "bloqueio tecnico"),
        "comando_testado": "python scripts\\suscetibilidade\\run_susc_18e2_gee_sentinel1_curitiba.py",
        "resultado_obtido": _safe_text(metadata.get("ee_initialize_result") or status),
        "acao_corretiva_exata": action,
        "reexecutavel_apos_correcao": "true",
    }]


def handoff_rows(metadata: dict) -> list[dict]:
    vector = LOCAL_FLOOD_VECTOR.exists()
    mask = LOCAL_FLOOD_MASK.exists()
    stats = LOCAL_PATCH_STATS.exists()
    meta = LOCAL_METADATA.exists()
    ready = bool(vector or stats)
    if ready:
        reason = "resultado real disponivel para revisao 18F"
    elif metadata.get("export_started"):
        reason = "aguardando_conclusao_export"
    elif metadata.get("ee_initialize_status") == "nao_autenticado":
        reason = "bloqueado_por_autenticacao"
    else:
        reason = metadata.get("block_type", "sem_resultado_real")
    return [{
        "resultado_gee_disponivel": _bool(vector or mask or stats),
        "flood_vector_disponivel": _bool(vector),
        "flood_mask_disponivel": _bool(mask),
        "patch_stats_disponivel": _bool(stats),
        "execution_metadata_disponivel": _bool(meta),
        "pronto_para_18f": _bool(ready),
        "motivo_se_nao_pronto": "not_available" if ready else reason,
        "caminho_resultados_local_runs": rel(LOCAL_RESULTS),
    }]


def gate_rows(metadata: dict, handoff: list[dict]) -> list[dict]:
    task_ids = [t.get("task_id", "") for t in metadata.get("tasks", []) if t.get("task_id") not in {"", "not_available"}]
    return [{
        "gate_id": "S18E2_GATE_0001",
        "status_18e2": metadata.get("status_18e2", "18E2_BLOQUEADO_FAIL_CLOSED"),
        "rota_python_earth_engine": metadata.get("route_a_python", "tentada"),
        "rota_cli_earth_engine": "tentada" if metadata.get("route_b_cli") else "nao_testada",
        "rota_js_gee": "validada" if metadata.get("route_c_js", {}).get("preserved") else "bloqueada",
        "earth_engine_autenticado": _bool(metadata.get("authenticated")),
        "gee_consultado": _bool(metadata.get("gee_queried")),
        "cenas_pre": str(metadata.get("pre_scene_count", "not_available")),
        "cenas_pos": str(metadata.get("post_scene_count", "not_available")),
        "export_iniciado": _bool(metadata.get("export_started")),
        "task_ids": ";".join(task_ids) if task_ids else "not_available",
        "pronto_para_18f": handoff[0]["pronto_para_18f"],
        "ground_truth": "false",
        "eligible_for_training": "false",
        "score_v7_allowed": "false",
        "benchmark_17b_criado": "false",
        "observacao": (
            "execucao controlada registrada; resultado local real exigido para 18F"
            if handoff[0]["pronto_para_18f"] == "false" else
            "resultado real local disponivel para revisao 18F"
        ),
    }]


def preflight_obj(metadata: dict) -> dict:
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "git_status_short_count": len(_run_git(["status", "--short"]).splitlines()),
        "required_inputs": _required_inputs_status(),
        "local_results_dir": rel(LOCAL_RESULTS),
        "local_results_dir_exists": LOCAL_RESULTS.exists(),
        "patch_polys_cur_count": len(s18c.load_cur_patch_polygons()),
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
        "metadata_status_18e2": metadata.get("status_18e2", "unknown"),
    }


def summary_obj(metadata: dict, env_rows: list[dict], handoff: list[dict]) -> dict:
    cli = metadata.get("route_b_cli", {}) or {}
    js = metadata.get("route_c_js", {}) or {}
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "candidate_event_id": EVENT_ID,
        "evento_publico": EVENT_PUBLIC_ID,
        "input_status_18e": read_json(SUMMARY_18E).get("status_18e", "unknown") if SUMMARY_18E.exists() else "unknown",
        "input_status_18d": read_json(SUMMARY_18D).get("status_18d", "unknown") if SUMMARY_18D.exists() else "unknown",
        "earth_engine_authenticated": bool(metadata.get("authenticated")),
        "gee_queried": bool(metadata.get("gee_queried")),
        "python_route_tested": True,
        "cli_route_tested": bool(cli),
        "cli_available": bool(cli.get("available")),
        "js_route_validated": bool(js.get("preserved")),
        "pre_scene_count": metadata.get("pre_scene_count", "not_available"),
        "post_scene_count": metadata.get("post_scene_count", "not_available"),
        "export_started": bool(metadata.get("export_started")),
        "task_ids": [t.get("task_id") for t in metadata.get("tasks", []) if t.get("task_id") not in {"", "not_available"}],
        "local_files_created": [p for p in metadata.get("local_files", []) if (ROOT / p).exists()],
        "flood_mask_local": LOCAL_FLOOD_MASK.exists(),
        "flood_vector_local": LOCAL_FLOOD_VECTOR.exists(),
        "patch_stats_local": LOCAL_PATCH_STATS.exists(),
        "execution_metadata_local": LOCAL_METADATA.exists(),
        "ready_for_18f": handoff[0]["pronto_para_18f"] == "true",
        "status_18e2": metadata.get("status_18e2", "18E2_BLOQUEADO_FAIL_CLOSED"),
        "status_17b": "17B_BLOQUEADO_POR_GEOMETRIA_OFICIAL",
        "environment_statuses": sorted({r["status"] for r in env_rows}),
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


def report_text(summary: dict, metadata: dict, blocks: list[dict], handoff: list[dict]) -> str:
    tasks = summary["task_ids"]
    task_text = ", ".join(tasks) if tasks else "not_available"
    block = blocks[0] if blocks else {}
    return f"""# SUSC-18E2 - Execucao controlada Sentinel-1 Curitiba

## Estado herdado do 18E

- Pacote GEE Sentinel-1 criado: `true`
- Raster SAR local suficiente no 18E: `false`
- Footprint produzido no 18E: `false`
- 17B: `17B_BLOQUEADO_POR_GEOMETRIA_OFICIAL`

## Rotas testadas

- Rota A - Python Earth Engine API: `tentada`
- Rota B - Earth Engine CLI: `{'tentada' if summary['cli_route_tested'] else 'nao_testada'}`
- Rota C - pacote JS GEE: `{'validada' if summary['js_route_validated'] else 'bloqueada'}`

## Autenticacao e consulta

- Earth Engine autenticado: `{str(summary['earth_engine_authenticated']).lower()}`
- GEE consultado: `{str(summary['gee_queried']).lower()}`
- Cenas Sentinel-1 pre-evento: `{summary['pre_scene_count']}`
- Cenas Sentinel-1 pos-evento: `{summary['post_scene_count']}`

## Resultados

- Export iniciado: `{str(summary['export_started']).lower()}`
- Task IDs: `{task_text}`
- Flood mask local: `{str(summary['flood_mask_local']).lower()}`
- Flood vector local: `{str(summary['flood_vector_local']).lower()}`
- Patch stats local: `{str(summary['patch_stats_local']).lower()}`
- Arquivos locais: `{rel(LOCAL_RESULTS)}`

## Bloqueio tecnico

- Tipo: `{block.get('tipo_bloqueio', 'not_available')}`
- Detalhe: `{block.get('detalhe_tecnico', 'not_available')}`
- Acao corretiva: `{block.get('acao_corretiva_exata', 'not_available')}`

## Handoff 18F

- Pronto para 18F: `{handoff[0]['pronto_para_18f']}`
- Motivo: `{handoff[0]['motivo_se_nao_pronto']}`

## Proxima acao pesada

Corrigir o bloqueio tecnico, reexecutar `python scripts\\suscetibilidade\\run_susc_18e2_gee_sentinel1_curitiba.py`
e, se houver tarefas iniciadas, aguardar a conclusao do export antes de ingerir resultado local.
"""


def copy_optional_local_to_public() -> None:
    if LOCAL_SCENES.exists():
        write_csv(PUBLIC_SCENES, read_csv(LOCAL_SCENES))
    elif PUBLIC_SCENES.exists():
        PUBLIC_SCENES.unlink()
    if LOCAL_TASKS.exists():
        write_csv(PUBLIC_TASKS, read_csv(LOCAL_TASKS))
    elif PUBLIC_TASKS.exists():
        PUBLIC_TASKS.unlink()
    if LOCAL_PATCH_STATS.exists():
        write_csv(PUBLIC_STATS, read_csv(LOCAL_PATCH_STATS))
    elif PUBLIC_STATS.exists():
        PUBLIC_STATS.unlink()


def build_public_outputs(metadata: dict | None = None) -> dict:
    require_inputs()
    ensure_dir(OUT_DATA)
    ensure_dir(REPORTS)
    ensure_local_results_dir()
    metadata = metadata or load_execution_metadata()
    metadata = refresh_task_states_from_cli(metadata)
    env_rows = environment_audit_rows({
        "ok": bool(metadata.get("authenticated")),
        "status": metadata.get("ee_initialize_status", "ambiente_indeterminado"),
        "result": metadata.get("ee_initialize_result", "not_available"),
        "detail": metadata.get("ee_initialize_detail", metadata.get("block_detail", "")),
    })
    manifest = manifest_rows(metadata)
    blocks = block_rows(metadata)
    handoff = handoff_rows(metadata)
    gate = gate_rows(metadata, handoff)
    summary = summary_obj(metadata, env_rows, handoff)
    write_csv(AUDIT_ENV, env_rows, AUDIT_FIELDS)
    write_csv(MANIFEST, manifest, MANIFEST_FIELDS)
    write_csv(BLOCKS, blocks, BLOCK_FIELDS)
    write_csv(HANDOFF, handoff, HANDOFF_FIELDS)
    write_csv(GATE, gate, GATE_FIELDS)
    write_json(PREFLIGHT, preflight_obj(metadata))
    write_json(SUMMARY, summary)
    copy_optional_local_to_public()
    write_markdown(REPORT, report_text(summary, metadata, blocks, handoff))
    return summary


def _public_output_paths() -> list[Path]:
    paths = []
    if OUT_DATA.exists():
        paths.extend(p for p in OUT_DATA.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".json", ".md"})
    if REPORT.exists():
        paths.append(REPORT)
    return sorted(dict.fromkeys(paths), key=lambda p: rel(p))


def public_text_violations_text(text: str) -> list[str]:
    return sorted({m.group(0) for m in PUBLIC_FORBIDDEN_RE.finditer(text)})


def secret_violations_text(text: str) -> list[str]:
    return sorted({m.group(0)[:30] for m in SECRET_RE.finditer(text)})


def public_text_violations_files(paths: list[Path] | None = None) -> list[str]:
    errors = []
    for path in paths or _public_output_paths():
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = public_text_violations_text(text)
        secrets = secret_violations_text(text)
        if hits:
            errors.append(f"{rel(path)}:vocabulario:{','.join(hits)}")
        if secrets:
            errors.append(f"{rel(path)}:credencial_ou_token")
    return errors


def validate_rows_no_promotion(paths: list[Path]) -> list[str]:
    errors = []
    for path in paths:
        if path.suffix.lower() == ".csv":
            rows = read_csv(path)
            for idx, row in enumerate(rows, start=1):
                rid = row.get("gate_id") or row.get("bloqueio_id") or row.get("arquivo") or f"linha{idx}"
                for field in ("ground_truth", "eligible_for_training", "score_v7_allowed"):
                    if str(row.get(field, "")).lower() == "true":
                        errors.append(f"{rel(path)}:{rid}:{field}_true")
                text = " ".join(str(v).lower() for v in row.values())
                if "aoi tecnica" in text and "geometria de ocorrencia" in text and "nao" not in text:
                    errors.append(f"{rel(path)}:{rid}:aoi_promovida_para_ocorrencia")
                if "footprint tecnico" in text and "geometria oficial" in text and "nao" not in text:
                    errors.append(f"{rel(path)}:{rid}:footprint_promovido_para_geometria_oficial")
        elif path.suffix.lower() == ".json":
            obj = read_json(path)
            text = json.dumps(obj, ensure_ascii=False).lower()
            if '"ground_truth": true' in text or '"trainable": true' in text:
                errors.append(f"{rel(path)}:ground_truth_ou_trainable_true")
            if '"score_v7_created": true' in text or '"score_v7_allowed": true' in text:
                errors.append(f"{rel(path)}:score_v7_true")
            if "benchmark_17b_criado" in obj and obj.get("benchmark_17b_criado") is True:
                errors.append(f"{rel(path)}:benchmark_17b_criado_true")
    return errors


def validate_handoff_and_gate(summary: dict) -> list[str]:
    errors = []
    handoff = read_csv(HANDOFF)
    gate = read_csv(GATE)
    blocks = read_csv(BLOCKS)
    if not handoff or not gate:
        return ["handoff_ou_gate_ausente"]
    h = handoff[0]
    g = gate[0]
    if g.get("status_18e2") not in GATE_STATUS_ALLOWED:
        errors.append(f"status_18e2_fora_enum:{g.get('status_18e2')}")
    if h.get("pronto_para_18f") == "true" and not (LOCAL_FLOOD_VECTOR.exists() or LOCAL_PATCH_STATS.exists()):
        errors.append("pronto_para_18f_sem_resultado_real")
    if h.get("patch_stats_disponivel") == "true" and not LOCAL_PATCH_STATS.exists():
        errors.append("patch_stats_disponivel_sem_arquivo_real")
    if h.get("flood_vector_disponivel") == "true" and not LOCAL_FLOOD_VECTOR.exists():
        errors.append("flood_vector_disponivel_sem_arquivo_real")
    if g.get("status_18e2") == "18E2_TAREFAS_GEE_INICIADAS_AGUARDANDO_CONCLUSAO" and g.get("task_ids") in {"", "not_available"}:
        errors.append("status_tarefa_iniciada_sem_task_id")
    if g.get("status_18e2", "").startswith("18E2_BLOQUEADO"):
        real_blocks = [b for b in blocks if b.get("bloqueio_id") != "not_available"]
        if not real_blocks:
            errors.append("bloqueio_sem_linha_especifica")
        for b in real_blocks:
            if not b.get("comando_testado") or b.get("comando_testado") == "not_available":
                errors.append("bloqueio_generico_sem_comando")
            if not b.get("detalhe_tecnico"):
                errors.append("bloqueio_sem_detalhe")
    if summary.get("score_v6_changed"):
        errors.append("score_v6_changed_forbidden")
    if summary.get("score_v7_created"):
        errors.append("score_v7_created_forbidden")
    if summary.get("benchmark_17b_criado"):
        errors.append("benchmark_17b_criado_forbidden")
    return errors


def validate() -> int:
    summary = build_public_outputs()
    errors = []
    errors.extend([f"missing:{rel(p)}" for p in REQUIRED_PUBLIC_OUTPUTS if not p.exists()])
    errors.extend(validate_rows_no_promotion(_public_output_paths()))
    errors.extend(validate_handoff_and_gate(summary))
    errors.extend(public_text_violations_files())
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(
        "18E2 execucao controlada Sentinel-1 Curitiba validada: "
        f"autenticado={str(summary['earth_engine_authenticated']).lower()} "
        f"gee_consultado={str(summary['gee_queried']).lower()} "
        f"pre={summary['pre_scene_count']} pos={summary['post_scene_count']} "
        f"export={str(summary['export_started']).lower()} "
        f"status18E2={summary['status_18e2']} pronto18F={str(summary['ready_for_18f']).lower()}"
    )
    return 0


def run_public_only() -> int:
    summary = build_public_outputs()
    print(
        "18E2 outputs publicos atualizados: "
        f"autenticado={str(summary['earth_engine_authenticated']).lower()} "
        f"gee_consultado={str(summary['gee_queried']).lower()} "
        f"status18E2={summary['status_18e2']}"
    )
    return 0
