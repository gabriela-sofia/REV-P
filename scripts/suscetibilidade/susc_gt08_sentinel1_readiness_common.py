"""SUSC-GT08 - Preparacao Integrada de AOI e Busca Sentinel-1 para Canarios.

Transforma os 5 canarios QA/SAR (GT06/GT07) em um pacote tecnico completo para futura
busca Sentinel-1 metadata-only (apenas metadados, sem baixar imagem): resolve AOI/bbox
offline a partir de artefatos versionados, gera especificacoes de consulta Sentinel-1,
faz replay sobre catalogos Sentinel-1 locais ja versionados (se existirem), monta o
contrato de pareamento pre/pos-evento, define criterios de aceitacao/rejeicao de cenas e
o plano de execucao SAR futura.

E um pacote offline e review-only: NAO usa rede, NAO consulta STAC/API remota, NAO baixa
Sentinel nem raster, NAO roda SAR nem GEE, NAO cria footprint, NAO cria geometria real,
NAO executa QA, NAO marca accepted/rejected, NAO altera o score_v6, NAO cria score_v7,
NAO treina modelo e NAO promove nada a positive_strong nem a ground truth. Toda consulta
Sentinel-1 aqui e apenas especificacao futura (metadata_only=true, no_download=true,
remote_execution_allowed_now=false). O footprint futuro pos-evento sera REFERENCIA de
avaliacao, nunca feature pre-evento.

Ver [[susc_gt07_human_qa_canary_protocol]]. Identificadores tecnicos em ingles sao
mantidos por compatibilidade; o relatorio publico explica cada campo em portugues.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import (  # noqa: E402
    ensure_dir,
    rel,
    sha256_file,
    write_csv,
    write_json,
    write_markdown,
)
import susc_gt07_human_qa_canary_protocol_common as gt07  # noqa: E402

gt06 = gt07.gt06
gt02 = gt07.gt02

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS_DIR = ROOT / "schemas" / "suscetibilidade"

GT07_QUEUE = OUT / "susc_gt07_fila_qa_canarios.csv"
GT07_MANIFEST = OUT / "susc_gt07_manifesto.json"
GT06_PACK = OUT / "susc_gt06_pacote_canarios_sar.csv"
GT06_CANARY = OUT / "susc_gt06_canarios_prioritarios.csv"
GT05_TARGETS = OUT / "susc_gt05_pacote_aquisicao_geometria_oficial.csv"
GT04_WINDOWS = OUT / "susc_gt04_janelas_pre_pos_evento.csv"
PATCH_REGISTRY = OUT / "susc_18a_unified_patch_registry.csv"

REQUIRED_INPUTS = [GT07_QUEUE, GT07_MANIFEST, GT06_PACK, GT06_CANARY, GT05_TARGETS, GT04_WINDOWS]

# Catalogos Sentinel-1 locais candidatos (versionados). Nao falhar se ausentes.
LOCAL_S1_CATALOG_GLOBS = [
    "susc_17c31_source_artifacts/asf_sentinel1_*grd*metadata*.json",
    "susc_17c31_source_artifacts/asf_sentinel1_*.json",
]

REPORT = OUT / "SUSC_GT08_PREPARACAO_AOI_BUSCA_SENTINEL1_CANARIOS.md"
TARGETS_CSV = OUT / "susc_gt08_alvos_sentinel1_canarios.csv"
AOI_CSV = OUT / "susc_gt08_aoi_resolvido_canarios.csv"
QUERY_CSV = OUT / "susc_gt08_especificacoes_consulta_sentinel1.csv"
CATALOG_CSV = OUT / "susc_gt08_catalogos_locais_sentinel1.csv"
SCENES_CSV = OUT / "susc_gt08_cenas_candidatas_locais_sentinel1.csv"
PAIRING_CSV = OUT / "susc_gt08_contrato_pareamento_pre_pos.csv"
PLAN_CSV = OUT / "susc_gt08_plano_execucao_sar_futura.csv"
BLOCKERS_CSV = OUT / "susc_gt08_bloqueios_sentinel1.csv"
SUMMARY_CSV = OUT / "susc_gt08_resumo_sentinel1_readiness.csv"
MANIFEST_JSON = OUT / "susc_gt08_manifesto.json"

SCHEMA_TARGET = SCHEMAS_DIR / "susc_gt08_sentinel1_target.schema.json"
SCHEMA_AOI = SCHEMAS_DIR / "susc_gt08_aoi_resolution.schema.json"
SCHEMA_QUERY = SCHEMAS_DIR / "susc_gt08_query_spec.schema.json"
SCHEMA_SCENE = SCHEMAS_DIR / "susc_gt08_scene_candidate.schema.json"
SCHEMA_PAIR = SCHEMAS_DIR / "susc_gt08_pairing_contract.schema.json"
SCHEMA_SUMMARY = SCHEMAS_DIR / "susc_gt08_summary.schema.json"

PUBLIC_ARTIFACTS = [
    REPORT, TARGETS_CSV, AOI_CSV, QUERY_CSV, CATALOG_CSV, SCENES_CSV, PAIRING_CSV,
    PLAN_CSV, BLOCKERS_CSV, SUMMARY_CSV,
]
REQUIRED_OUTPUTS = PUBLIC_ARTIFACTS + [
    MANIFEST_JSON, SCHEMA_TARGET, SCHEMA_AOI, SCHEMA_QUERY, SCHEMA_SCENE,
    SCHEMA_PAIR, SCHEMA_SUMMARY,
]

MARCO = "SUSC-GT08"
TITULO_PUBLICO = "Preparação Integrada de AOI e Busca Sentinel-1 para Canários"
TITULO_H1_TOKEN = "AOI e Busca Sentinel-1"

PUBLIC_FORBIDDEN_RE = gt07.PUBLIC_FORBIDDEN_RE
OPERATIONAL_FORECAST_RE = gt07.OPERATIONAL_FORECAST_RE

GLOBAL_GUARDRAILS = {
    "eligible_for_training": "false", "eligible_for_ground_truth": "false",
    "trainable": "false", "score_v7_candidate": "false", "review_only": "true",
}

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
AOI_STATUSES = [
    "aoi_forte_por_bbox_ou_geometria_patch",
    "aoi_parcial_por_centroide_ou_ponto",
    "aoi_contextual_insuficiente",
    "aoi_ausente",
    "aoi_invalida",
]
SCENE_CLASSES = [
    "pre_event_candidate", "post_event_candidate", "outside_window",
    "missing_metadata", "wrong_product", "wrong_mode", "insufficient_overlap",
    "usable_metadata_only_candidate",
]
PAIRING_STATUSES = [
    "par_completo_pre_pos", "falta_cena_pre", "falta_cena_pos",
    "sem_par_valido", "bloqueado_por_aoi_ou_janela",
]
STRONG_AOI = {"aoi_forte_por_bbox_ou_geometria_patch", "aoi_parcial_por_centroide_ou_ponto"}
USABLE_WINDOW = gt06.USABLE_WINDOW

# Politica de consulta Sentinel-1 (fixa; apenas especificacao futura).
QUERY_POLICY = {
    "collection_policy": "sentinel1_grd",
    "platform_policy": "any_sentinel1",
    "instrument": "SAR",
    "product_type": "GRD",
    "acquisition_mode": "IW",
    "polarization_required": "VV,VH",
    "orbit_direction": "any",
    "metadata_only": "true",
    "no_download": "true",
    "remote_execution_allowed_now": "false",
}

# Plano de execucao SAR futura (11 passos), sem execucao agora.
SAR_PLAN_STEPS = [
    ("confirmar AOI do canario", "AOI confirmado", "false", "false", "false"),
    ("buscar metadados Sentinel-1 (metadata-only, sem download)", "lista de cenas por metadados", "true", "true", "false"),
    ("selecionar cena pre-evento", "cena pre-evento selecionada", "true", "true", "false"),
    ("selecionar cena pos-evento", "cena pos-evento selecionada", "true", "true", "false"),
    ("baixar cenas apenas em marco futuro", "cenas baixadas (marco futuro)", "true", "true", "false"),
    ("pre-processar SAR", "SAR pre-processado", "false", "true", "false"),
    ("aplicar mascara de agua permanente", "mascara de agua aplicada", "false", "true", "false"),
    ("aplicar filtro HAND/slope", "filtros topograficos aplicados", "false", "false", "false"),
    ("aplicar exclusoes urbanas", "ambiguidade urbana tratada", "false", "true", "false"),
    ("gerar footprint candidato", "footprint candidato (referencia de avaliacao, nao feature pre-evento)", "false", "true", "false"),
    ("executar QA humano", "QA humano concluido", "false", "false", "true"),
]

_clean = gt07._clean
_present = gt07._present
FLOOD_COMPATIBLE = gt06.FLOOD_COMPATIBLE
PETROPOLIS_BLOCKING = gt02.PETROPOLIS_BLOCKING_PHENOMENA


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _run_git(args: list[str]) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def require_inputs() -> None:
    missing = [rel(p) for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("entradas GT07/GT06/GT05/GT04 ausentes: " + "; ".join(missing))


def _seq(x: str) -> str:
    m = re.search(r"(\d+)$", x)
    return m.group(1) if m else x


def _b(v) -> str:
    return "true" if v else "false"


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# A) AOI Resolver offline
# ---------------------------------------------------------------------------
def _parse_bbox(s: str):
    """bbox 'min_lon,min_lat,max_lon,max_lat' -> tupla de floats validada."""
    parts = [p.strip() for p in _clean(s).split(",")]
    if len(parts) != 4:
        return None
    vals = []
    for p in parts:
        v = _num(p)
        if v is None:
            return None
        vals.append(v)
    min_lon, min_lat, max_lon, max_lat = vals
    if min_lon >= max_lon or min_lat >= max_lat:
        return None
    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180 and -90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        return None
    return (min_lon, min_lat, max_lon, max_lat)


def resolve_aoi(patch_row: dict, queue_row: dict) -> dict:
    """Resolve AOI a partir do registro de patch (bbox/centroide) versionado."""
    bbox_raw = _clean(patch_row.get("bbox"))
    bbox = _parse_bbox(bbox_raw)
    clon, clat = _num(patch_row.get("centroid_lon")), _num(patch_row.get("centroid_lat"))
    geom_av = _clean(patch_row.get("geometry_available")).lower() == "true"
    if bbox_raw and bbox is None:
        status = "aoi_invalida"
        aoi_bbox, aoi_src = bbox_raw, "susc_18a_unified_patch_registry(bbox_invalido)"
    elif bbox is not None:
        status = "aoi_forte_por_bbox_ou_geometria_patch"
        aoi_bbox = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        aoi_src = "susc_18a_unified_patch_registry(bbox)"
    elif clon is not None and clat is not None:
        status = "aoi_parcial_por_centroide_ou_ponto"
        aoi_bbox = f"{clon},{clat},{clon},{clat}"
        aoi_src = "susc_18a_unified_patch_registry(centroide)"
    elif _present(queue_row.get("city")) or _present(queue_row.get("region")):
        status = "aoi_contextual_insuficiente"
        aoi_bbox, aoi_src = "not_available", "cidade/regiao(contexto)"
    else:
        status = "aoi_ausente"
        aoi_bbox, aoi_src = "not_available", "none"
    return {
        "aoi_status": status, "aoi_bbox": aoi_bbox, "aoi_source": aoi_src,
        "centroid_lon": _clean(patch_row.get("centroid_lon")) or "not_available",
        "centroid_lat": _clean(patch_row.get("centroid_lat")) or "not_available",
        "geometry_available": _b(geom_av), "bbox_tuple": bbox,
    }


# ---------------------------------------------------------------------------
# C) Descoberta e normalizacao de catalogo Sentinel-1 local
# ---------------------------------------------------------------------------
def _wkt_bbox(wkt: str):
    nums = re.findall(r"-?\d+\.?\d*", wkt or "")
    if len(nums) < 8:
        return None
    fl = [float(x) for x in nums]
    lons = fl[0::2]
    lats = fl[1::2]
    return (min(lons), min(lats), max(lons), max(lats))


def _bbox_intersects(a, b) -> bool:
    if not a or not b:
        return False
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def discover_local_catalogs() -> list[Path]:
    """Descobre catalogos Sentinel-1 locais versionados. Deterministico."""
    seen = set()
    catalogs = []
    for pattern in LOCAL_S1_CATALOG_GLOBS:
        for path in sorted(OUT.glob(pattern)):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            catalogs.append(path)
    return catalogs


def load_scenes(path: Path) -> list[dict]:
    """Normaliza cenas de um catalogo ASF (JSON com 'results')."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    results = data.get("results", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    scenes = []
    for r in results:
        if not isinstance(r, dict):
            continue
        gran = _clean(r.get("granuleName") or r.get("sceneName") or r.get("fileName"))
        start = _clean(r.get("startTime") or r.get("acquisition_datetime"))
        acq_date = start[:10] if len(start) >= 10 else ""
        scenes.append({
            "scene_id": gran or "not_available",
            "product_id": _clean(r.get("productID") or r.get("product_id")) or "not_available",
            "acquisition_datetime": start or "not_available",
            "acquisition_date": acq_date,
            "platform": _clean(r.get("dataset") or r.get("missionName") or r.get("platform")) or "not_available",
            "product_type": _clean(r.get("productType") or r.get("product_type")) or "not_available",
            "acquisition_mode": _clean(r.get("beamMode") or r.get("acquisition_mode")) or "not_available",
            "polarization": _clean(r.get("polarization")) or "not_available",
            "orbit_direction": _clean(r.get("flightDirection") or r.get("orbit_direction")) or "not_available",
            "orbit_path": _clean(r.get("path")) or "not_available",
            "orbit_frame": _clean(r.get("frame")) or "not_available",
            "scene_bbox": _wkt_bbox(r.get("wkt") or r.get("wkt_unwrapped") or ""),
        })
    return sorted(scenes, key=lambda s: s["scene_id"])


def _catalog_region_hint(name: str) -> str:
    n = name.lower()
    for city in ("recife", "curitiba", "petropolis"):
        if city in n:
            return city
    return "indefinido"


# ---------------------------------------------------------------------------
# Classificacao de cena por canario
# ---------------------------------------------------------------------------
def classify_scene(scene: dict, aoi_bbox_tuple, pre_win, post_win) -> str:
    if not _present(scene.get("scene_id")) or not scene.get("acquisition_date"):
        return "missing_metadata"
    if not _clean(scene.get("product_type")).upper().startswith("GRD"):
        return "wrong_product"
    if _clean(scene.get("acquisition_mode")).upper() != "IW":
        return "wrong_mode"
    if not _bbox_intersects(scene.get("scene_bbox"), aoi_bbox_tuple):
        return "insufficient_overlap"
    d = scene["acquisition_date"]
    if pre_win[0] and pre_win[1] and pre_win[0] <= d <= pre_win[1]:
        return "pre_event_candidate"
    if post_win[0] and post_win[1] and post_win[0] <= d <= post_win[1]:
        return "post_event_candidate"
    return "outside_window"


# ---------------------------------------------------------------------------
# Construcao das linhas
# ---------------------------------------------------------------------------
def build_rows() -> dict:
    queue = _read_csv(GT07_QUEUE)
    windows = {w["target_id"]: w for w in _read_csv(GT04_WINDOWS)}
    patches = {p["unified_patch_id"]: p for p in _read_csv(PATCH_REGISTRY)} if PATCH_REGISTRY.exists() else {}

    # C) catalogos locais
    catalog_paths = discover_local_catalogs()
    catalog_rows = []
    all_scenes = []  # (catalog_id, scene)
    for i, cp in enumerate(catalog_paths, start=1):
        scenes = load_scenes(cp)
        cid = f"CAT_{i:03d}"
        catalog_rows.append({
            "catalog_id": cid, "path": rel(cp),
            "format": cp.suffix.lstrip(".").lower() or "json",
            "region_hint": _catalog_region_hint(cp.name),
            "scene_count": str(len(scenes)),
            "matches_canary_event": _b(_catalog_region_hint(cp.name) == "recife"),
            "no_download_in_this_milestone": "true", "review_only": "true",
            "blocking_reason": "" if scenes else "catalogo sem cenas legiveis",
        })
        for s in scenes:
            all_scenes.append((cid, _catalog_region_hint(cp.name), s))
    if not catalog_rows:
        catalog_rows.append({
            "catalog_id": "CAT_000", "path": "not_available", "format": "none",
            "region_hint": "indefinido", "scene_count": "0",
            "matches_canary_event": "false", "no_download_in_this_milestone": "true",
            "review_only": "true",
            "blocking_reason": "nenhum catalogo Sentinel-1 local versionado encontrado",
        })

    targets, aoi_rows, query_rows, scene_rows, pairing_rows, plan_rows, blocker_rows = [], [], [], [], [], [], []
    b_idx = 1

    for q in queue:
        tid = q["target_id"]
        patch_id = q.get("patch_id", "")
        prow = patches.get(patch_id, {})
        w = windows.get(tid, {})
        pre_win = (_clean(w.get("pre_event_window_start")), _clean(w.get("pre_event_window_end")))
        post_win = (_clean(w.get("post_event_window_start")), _clean(w.get("post_event_window_end")))
        has_window = _clean(w.get("window_status")).lower() in USABLE_WINDOW or (pre_win[0] and post_win[0])
        aoi = resolve_aoi(prow, q)
        aoi_ok = aoi["aoi_status"] in STRONG_AOI
        sid = f"S1_{_seq(tid)}"

        aoi_rows.append({
            "aoi_resolution_id": f"AOI_{_seq(tid)}", "sentinel1_target_id": sid, "target_id": tid,
            "patch_id": patch_id or "not_available",
            "aoi_status": aoi["aoi_status"], "aoi_bbox": aoi["aoi_bbox"], "aoi_source": aoi["aoi_source"],
            "centroid_lon": aoi["centroid_lon"], "centroid_lat": aoi["centroid_lat"],
            "geometry_available": aoi["geometry_available"],
            "sufficient_for_search": _b(aoi_ok),
            "review_only": "true", "no_geometry_created_in_this_milestone": "true",
            "rationale_pt": ("AOI forte por bbox do patch (registro versionado)" if aoi["aoi_status"] == "aoi_forte_por_bbox_ou_geometria_patch"
                             else "AOI resolvido em nivel " + aoi["aoi_status"]),
        })

        # B) query spec
        query_rows.append({
            "query_spec_id": f"QS_{_seq(tid)}", "sentinel1_target_id": sid, "target_id": tid,
            "event_id": q.get("event_id", "not_available"),
            **QUERY_POLICY,
            "pre_event_window": f"{pre_win[0]}..{pre_win[1]}",
            "post_event_window": f"{post_win[0]}..{post_win[1]}",
            "aoi_bbox": aoi["aoi_bbox"], "aoi_source": aoi["aoi_source"],
            "query_status": "pronta_para_consulta_futura" if (aoi_ok and has_window) else "bloqueada",
            "query_blocking_reason": "" if (aoi_ok and has_window) else "AOI insuficiente ou janela ausente",
            "review_only": "true",
        })

        # cenas candidatas (por canario x cena de catalogo que casa a regiao)
        pre_ids, post_ids = [], []
        canary_region = "recife"  # canarios sao Recife (GT06)
        for cid, region_hint, scene in all_scenes:
            if region_hint not in (canary_region, "indefinido"):
                continue  # so catalogos da regiao do canario
            cls = classify_scene(scene, aoi["bbox_tuple"], pre_win, post_win)
            usable = cls in ("pre_event_candidate", "post_event_candidate")
            if cls == "pre_event_candidate":
                pre_ids.append(scene)
            elif cls == "post_event_candidate":
                post_ids.append(scene)
            scene_rows.append({
                "scene_candidate_id": f"SC_{len(scene_rows)+1:05d}", "sentinel1_target_id": sid,
                "target_id": tid, "catalog_id": cid, "scene_id": scene["scene_id"],
                "product_id": scene["product_id"], "acquisition_datetime": scene["acquisition_datetime"],
                "platform": scene["platform"], "product_type": scene["product_type"],
                "acquisition_mode": scene["acquisition_mode"], "polarization": scene["polarization"],
                "orbit_direction": scene["orbit_direction"], "orbit_path": scene["orbit_path"],
                "orbit_frame": scene["orbit_frame"],
                "scene_classification": cls,
                "usable_metadata_only_candidate": _b(usable),
                "aoi_overlap": _b(_bbox_intersects(scene.get("scene_bbox"), aoi["bbox_tuple"])),
                "metadata_only": "true", "no_download_in_this_milestone": "true", "review_only": "true",
            })

        # D) contrato de pareamento pre/pos
        pair = _pair_contract(pre_ids, post_ids, aoi_ok, has_window)
        pairing_rows.append({
            "pairing_id": f"PR_{_seq(tid)}", "sentinel1_target_id": sid, "target_id": tid,
            "pre_candidate_count": str(len(pre_ids)), "post_candidate_count": str(len(post_ids)),
            "pairing_status": pair["status"],
            "selected_pre_scene_id": pair["pre_id"], "selected_post_scene_id": pair["post_id"],
            "same_orbit_pair": _b(pair["same_orbit"]),
            "requires_pre_and_post": "true", "requires_aoi_intersection": "true",
            "required_product": "GRD", "required_mode": "IW", "required_polarization": "VV,VH",
            "pairing_blocking_reason": pair["reason"],
            "metadata_only": "true", "no_download_in_this_milestone": "true", "review_only": "true",
        })

        # E) plano de execucao SAR futura (11 passos)
        for order, (act, art, net, s1, qa) in enumerate(SAR_PLAN_STEPS, start=1):
            plan_rows.append({
                "sar_plan_id": f"SPL_{len(plan_rows)+1:05d}", "sentinel1_target_id": sid, "step_order": str(order),
                "action_pt": act, "expected_artifact_pt": art,
                "success_criteria_pt": f"{art} registrado e auditavel, mantendo review_only",
                "fail_closed_outcome_pt": "sem o artefato, o canario nao avanca; sem promocao e sem treino",
                "requires_network_future": net, "requires_sentinel1_future": s1,
                "requires_manual_qa_future": qa,
                "download_only_in_future_milestone": _b(order == 5),
                "no_execution_now": "true",
            })

        # alvo Sentinel-1 (resumo por canario)
        ready = aoi_ok and has_window
        targets.append({
            "sentinel1_target_id": sid, "qa_queue_id": q.get("qa_queue_id", ""), "target_id": tid,
            "event_id": q.get("event_id", "not_available"), "patch_id": patch_id or "not_available",
            "city": q.get("city", "not_available"), "region": q.get("region", "not_available"),
            "phenomenon_type": q.get("phenomenon_type", "not_available"),
            "resolved_event_date": q.get("resolved_event_date", "not_available"),
            "temporal_precision": q.get("temporal_precision", ""),
            "aoi_status": aoi["aoi_status"], "aoi_bbox": aoi["aoi_bbox"],
            "pre_event_window": f"{pre_win[0]}..{pre_win[1]}",
            "post_event_window": f"{post_win[0]}..{post_win[1]}",
            "local_scene_count": str(len(pre_ids) + len(post_ids)),
            "pre_candidate_count": str(len(pre_ids)), "post_candidate_count": str(len(post_ids)),
            "pairing_status": pair["status"],
            "ready_for_metadata_search": _b(ready),
            "can_execute_search_now": "false", "can_download_now": "false",
            "expected_future_footprint_pt": "footprint SAR pos-evento futuro (referencia de avaliacao, nunca feature pre-evento)",
            "remains_review_only": "true", "trainable": "false",
            "eligible_for_training": "false", "eligible_for_ground_truth": "false",
            "score_v7_candidate": "false",
            "rationale_pt": _target_rationale(aoi["aoi_status"], has_window, pair["status"]),
        })

        # bloqueios
        for bl in _target_blockers(aoi["aoi_status"], has_window, pair, sid):
            bl["sentinel1_blocker_id"] = f"S1B_{b_idx:05d}"
            blocker_rows.append(bl)
            b_idx += 1

    return {
        "targets": targets, "aoi": aoi_rows, "query": query_rows, "catalog": catalog_rows,
        "scenes": scene_rows, "pairing": pairing_rows, "plan": plan_rows, "blockers": blocker_rows,
    }


def _pair_contract(pre, post, aoi_ok, has_window) -> dict:
    if not aoi_ok or not has_window:
        return {"status": "bloqueado_por_aoi_ou_janela", "pre_id": "not_available",
                "post_id": "not_available", "same_orbit": False,
                "reason": "AOI insuficiente ou janela ausente para pareamento"}
    if not pre and not post:
        return {"status": "sem_par_valido", "pre_id": "not_available", "post_id": "not_available",
                "same_orbit": False, "reason": "sem cena pre nem pos candidata no catalogo local"}
    if not pre:
        return {"status": "falta_cena_pre", "pre_id": "not_available",
                "post_id": post[0]["scene_id"], "same_orbit": False,
                "reason": "sem cena pre-evento candidata"}
    if not post:
        return {"status": "falta_cena_pos", "pre_id": pre[0]["scene_id"],
                "post_id": "not_available", "same_orbit": False,
                "reason": "sem cena pos-evento candidata"}
    # preferir mesma orbita (path/frame/direcao)
    for a in pre:
        for b in post:
            if (a["orbit_path"], a["orbit_frame"], a["orbit_direction"]) == (b["orbit_path"], b["orbit_frame"], b["orbit_direction"]):
                return {"status": "par_completo_pre_pos", "pre_id": a["scene_id"],
                        "post_id": b["scene_id"], "same_orbit": True,
                        "reason": ""}
    return {"status": "par_completo_pre_pos", "pre_id": pre[0]["scene_id"],
            "post_id": post[0]["scene_id"], "same_orbit": False,
            "reason": "par pre/pos disponivel, mas em orbitas diferentes"}


def _target_rationale(aoi_status, has_window, pairing_status) -> str:
    if aoi_status not in STRONG_AOI:
        return "AOI insuficiente; resolver geometria antes da busca Sentinel-1"
    if not has_window:
        return "sem janela pre/pos-evento; resolver data antes da busca"
    if pairing_status == "par_completo_pre_pos":
        return "AOI forte, janela definida e par pre/pos localizado no catalogo local; pronto para consulta metadata-only futura"
    return "AOI forte e janela definida; pronto para consulta metadata-only futura (par ainda a completar)"


def _target_blockers(aoi_status, has_window, pair, sid) -> list[dict]:
    out = []

    def add(btype, sev, field, reason, search=True, sar=True):
        out.append({
            "sentinel1_target_id": sid, "blocker_type": btype, "severity": sev, "field": field,
            "reason_pt": reason, "prevents_metadata_search": _b(search),
            "prevents_sar_future": _b(sar), "prevents_positive_strong": "true",
            "how_to_resolve_pt": _how_resolve(btype),
        })

    if aoi_status not in STRONG_AOI:
        add("aoi_insuficiente", "alta", "aoi_status", "AOI insuficiente para recorte Sentinel-1")
    if not has_window:
        add("janela_ausente", "alta", "pre_event_window", "sem janela pre/pos-evento")
    if pair["status"] in ("falta_cena_pre", "falta_cena_pos", "sem_par_valido"):
        add("par_pre_pos_incompleto", "media", "pairing_status",
            "par pre/pos ainda nao localizado no catalogo local", search=False)
    return out


def _how_resolve(btype: str) -> str:
    return {
        "aoi_insuficiente": "obter bbox/geometria do patch antes da busca",
        "janela_ausente": "resolver data e janela pre/pos-evento",
        "par_pre_pos_incompleto": "buscar metadados Sentinel-1 (metadata-only) em marco futuro para completar o par",
    }.get(btype, "resolver em marco futuro")


# ---------------------------------------------------------------------------
# Resumo
# ---------------------------------------------------------------------------
def summary_rows(bundle: dict) -> list[dict]:
    order = {s: i for i, s in enumerate(AOI_STATUSES)}
    targets = bundle["targets"]
    agg = {}
    for t in targets:
        s = t["aoi_status"]
        a = agg.setdefault(s, {"count": 0, "ready": 0, "par": 0, "scenes": 0})
        a["count"] += 1
        a["ready"] += t["ready_for_metadata_search"] == "true"
        a["par"] += t["pairing_status"] == "par_completo_pre_pos"
        a["scenes"] += int(t["local_scene_count"])
    rows = []
    for s in sorted(agg, key=lambda x: order.get(x, 99)):
        a = agg[s]
        rows.append({
            "aoi_status": s, "count": str(a["count"]),
            "ready_for_metadata_search_count": str(a["ready"]),
            "par_completo_pre_pos_count": str(a["par"]),
            "local_scene_candidate_count": str(a["scenes"]),
            "can_execute_search_now_count": "0", "can_download_now_count": "0",
            "training_allowed_count": "0", "ground_truth_allowed_count": "0",
            "score_v7_candidate_count": "0",
        })
    return rows


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
_BOOL = {"enum": ["true", "false"]}
_FALSE = {"const": "false"}
_TRUE = {"const": "true"}


def schema_target_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt08_sentinel1_target.schema.json",
        "title": "SUSC-GT08 Alvo Sentinel-1", "type": "object",
        "required": ["sentinel1_target_id", "target_id", "aoi_status", "pairing_status",
                     "ready_for_metadata_search", "can_execute_search_now", "can_download_now",
                     "remains_review_only", "trainable", "eligible_for_training",
                     "eligible_for_ground_truth", "score_v7_candidate", "rationale_pt"],
        "properties": {
            "sentinel1_target_id": {"type": "string", "pattern": r"^S1_\d+$"},
            "target_id": {"type": "string", "pattern": r"^TGT_\d+$"},
            "aoi_status": {"enum": AOI_STATUSES}, "pairing_status": {"enum": PAIRING_STATUSES},
            "ready_for_metadata_search": _BOOL, "can_execute_search_now": _FALSE,
            "can_download_now": _FALSE, "remains_review_only": _TRUE, "trainable": _FALSE,
            "eligible_for_training": _FALSE, "eligible_for_ground_truth": _FALSE,
            "score_v7_candidate": _FALSE, "rationale_pt": {"type": "string", "minLength": 1},
        },
    }


def schema_aoi_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt08_aoi_resolution.schema.json",
        "title": "SUSC-GT08 Resolucao de AOI", "type": "object",
        "required": ["aoi_resolution_id", "sentinel1_target_id", "aoi_status",
                     "sufficient_for_search", "review_only", "no_geometry_created_in_this_milestone"],
        "properties": {
            "aoi_resolution_id": {"type": "string", "pattern": r"^AOI_\d+$"},
            "sentinel1_target_id": {"type": "string", "pattern": r"^S1_\d+$"},
            "aoi_status": {"enum": AOI_STATUSES}, "sufficient_for_search": _BOOL,
            "review_only": _TRUE, "no_geometry_created_in_this_milestone": _TRUE,
        },
    }


def schema_query_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt08_query_spec.schema.json",
        "title": "SUSC-GT08 Especificacao de Consulta Sentinel-1", "type": "object",
        "required": ["query_spec_id", "sentinel1_target_id", "collection_policy", "instrument",
                     "product_type", "acquisition_mode", "polarization_required",
                     "metadata_only", "no_download", "remote_execution_allowed_now", "review_only"],
        "properties": {
            "query_spec_id": {"type": "string", "pattern": r"^QS_\d+$"},
            "sentinel1_target_id": {"type": "string", "pattern": r"^S1_\d+$"},
            "collection_policy": {"const": "sentinel1_grd"},
            "instrument": {"const": "SAR"}, "product_type": {"const": "GRD"},
            "acquisition_mode": {"const": "IW"}, "polarization_required": {"const": "VV,VH"},
            "metadata_only": _TRUE, "no_download": _TRUE,
            "remote_execution_allowed_now": _FALSE, "review_only": _TRUE,
        },
    }


def schema_scene_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt08_scene_candidate.schema.json",
        "title": "SUSC-GT08 Cena Candidata", "type": "object",
        "required": ["scene_candidate_id", "sentinel1_target_id", "scene_id",
                     "scene_classification", "metadata_only", "no_download_in_this_milestone", "review_only"],
        "properties": {
            "scene_candidate_id": {"type": "string", "pattern": r"^SC_\d+$"},
            "sentinel1_target_id": {"type": "string", "pattern": r"^S1_\d+$"},
            "scene_id": {"type": "string", "minLength": 1},
            "scene_classification": {"enum": SCENE_CLASSES},
            "usable_metadata_only_candidate": _BOOL, "aoi_overlap": _BOOL,
            "metadata_only": _TRUE, "no_download_in_this_milestone": _TRUE, "review_only": _TRUE,
        },
    }


def schema_pair_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt08_pairing_contract.schema.json",
        "title": "SUSC-GT08 Contrato de Pareamento", "type": "object",
        "required": ["pairing_id", "sentinel1_target_id", "pairing_status",
                     "requires_pre_and_post", "requires_aoi_intersection",
                     "metadata_only", "no_download_in_this_milestone", "review_only"],
        "properties": {
            "pairing_id": {"type": "string", "pattern": r"^PR_\d+$"},
            "sentinel1_target_id": {"type": "string", "pattern": r"^S1_\d+$"},
            "pairing_status": {"enum": PAIRING_STATUSES},
            "requires_pre_and_post": _TRUE, "requires_aoi_intersection": _TRUE,
            "metadata_only": _TRUE, "no_download_in_this_milestone": _TRUE, "review_only": _TRUE,
        },
    }


def schema_summary_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "susc_gt08_summary.schema.json",
        "title": "SUSC-GT08 Resumo de Prontidao Sentinel-1", "type": "object",
        "required": ["aoi_status", "count", "can_execute_search_now_count", "can_download_now_count",
                     "training_allowed_count", "ground_truth_allowed_count", "score_v7_candidate_count"],
        "properties": {
            "aoi_status": {"enum": AOI_STATUSES}, "count": {"type": "string", "pattern": r"^\d+$"},
            "ready_for_metadata_search_count": {"type": "string", "pattern": r"^\d+$"},
            "par_completo_pre_pos_count": {"type": "string", "pattern": r"^\d+$"},
            "local_scene_candidate_count": {"type": "string", "pattern": r"^\d+$"},
            "can_execute_search_now_count": {"const": "0"}, "can_download_now_count": {"const": "0"},
            "training_allowed_count": {"const": "0"}, "ground_truth_allowed_count": {"const": "0"},
            "score_v7_candidate_count": {"const": "0"},
        },
    }


# ---------------------------------------------------------------------------
# Manifesto / proximo marco
# ---------------------------------------------------------------------------
def _next_milestone(targets: list[dict]) -> str:
    aoi_ausente = sum(1 for t in targets if t["aoi_status"] in ("aoi_ausente", "aoi_contextual_insuficiente", "aoi_invalida"))
    pares = sum(1 for t in targets if t["pairing_status"] == "par_completo_pre_pos")
    ready = sum(1 for t in targets if t["ready_for_metadata_search"] == "true")
    if targets and aoi_ausente > ready:
        return "GT09 - Resolucao de AOI"
    if pares >= 1:
        return "GT09 - Pacote Visual Manual"
    if ready >= 1:
        return "GT09 - Consulta Metadata-Only Sentinel-1"
    return "GT09 - Resolucao de AOI"


def manifest_obj(bundle: dict) -> dict:
    targets = bundle["targets"]
    dist = {}
    for t in targets:
        dist[t["aoi_status"]] = dist.get(t["aoi_status"], 0) + 1
    total_scenes = len(bundle["scenes"])
    artifacts = [
        {"path": rel(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in PUBLIC_ARTIFACTS + [SCHEMA_TARGET, SCHEMA_AOI, SCHEMA_QUERY, SCHEMA_SCENE, SCHEMA_PAIR, SCHEMA_SUMMARY]
        if p.exists()
    ]
    return {
        "marco": MARCO, "titulo_publico": TITULO_PUBLICO,
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "gt07_source_commit": _run_git(["log", "-1", "--format=%h", "--", rel(GT07_QUEUE)]) or "unknown",
        "canarios": len(targets), "distribuicao_aoi": dist,
        "catalogos_locais": len([c for c in bundle["catalog"] if c["path"] != "not_available"]),
        "cenas_locais_avaliadas": total_scenes,
        "pre_candidatas": sum(1 for s in bundle["scenes"] if s["scene_classification"] == "pre_event_candidate"),
        "post_candidatas": sum(1 for s in bundle["scenes"] if s["scene_classification"] == "post_event_candidate"),
        "canarios_prontos_busca": sum(1 for t in targets if t["ready_for_metadata_search"] == "true"),
        "canarios_com_par_completo": sum(1 for t in targets if t["pairing_status"] == "par_completo_pre_pos"),
        "download_executado": 0, "sar_executado": 0, "footprint_criado": 0,
        "geometria_criada": 0, "positive_strong_promovidos": 0, "qa_executado": 0,
        "internet_usada": 0, "stac_api_usada": 0,
        "global_guardrails": GLOBAL_GUARDRAILS,
        "training_true_count": sum(1 for t in targets if t["eligible_for_training"] == "true"),
        "ground_truth_true_count": sum(1 for t in targets if t["eligible_for_ground_truth"] == "true"),
        "score_v7_candidate_true_count": sum(1 for t in targets if t["score_v7_candidate"] == "true"),
        "can_execute_search_now_true_count": sum(1 for t in targets if t["can_execute_search_now"] == "true"),
        "can_download_now_true_count": sum(1 for t in targets if t["can_download_now"] == "true"),
        "score_v6_changed": gt02.gt01._score_v6_changed(),
        "score_v7_created": gt02.gt01._score_v7_exists(),
        "proximo_marco": _next_milestone(targets),
        "artifacts": artifacts,
    }


# ---------------------------------------------------------------------------
# Relatorio publico (portugues)
# ---------------------------------------------------------------------------
def report_text(manifest: dict, bundle: dict) -> str:
    targets = bundle["targets"]
    linhas_alvo = "\n".join(
        f"| {t['sentinel1_target_id']} | {t['patch_id']} | {t['city']}/{t['region']} | {t['aoi_status']} | "
        f"{t['pre_candidate_count']}/{t['post_candidate_count']} | {t['pairing_status']} | {t['ready_for_metadata_search']} |"
        for t in targets
    )
    cat = bundle["catalog"]
    linhas_cat = "\n".join(
        f"| {c['catalog_id']} | {c['path'].split('/')[-1]} | {c['region_hint']} | {c['scene_count']} | {c['matches_canary_event']} |"
        for c in cat
    )
    # exemplos de cenas pre/pos
    pre_ex = next((s for s in bundle["scenes"] if s["scene_classification"] == "pre_event_candidate"), None)
    post_ex = next((s for s in bundle["scenes"] if s["scene_classification"] == "post_event_candidate"), None)
    ex_pre = f"`{pre_ex['scene_id']}` ({pre_ex['acquisition_datetime']})" if pre_ex else "nenhuma"
    ex_post = f"`{post_ex['scene_id']}` ({post_ex['acquisition_datetime']})" if post_ex else "nenhuma"

    return f"""# {MARCO} - {TITULO_PUBLICO}

## 1. Escopo do marco

Este marco transforma os canários selecionados nos marcos anteriores em um **pacote técnico
completo para busca Sentinel-1 metadata-only** (apenas metadados, sem baixar imagem):
resolve AOI (Area of Interest, área de interesse) offline, gera especificações de consulta,
faz replay de catálogos Sentinel-1 locais já versionados, monta o contrato de pareamento
pré/pós-evento e o plano de execução SAR futura. É **offline e review-only**: não usa
internet, não consulta STAC/API remota, não baixa Sentinel nem raster, não roda SAR nem GEE,
não cria footprint, não cria geometria real, não executa QA, não altera o `score_v6`, não
cria `score_v7`, não treina modelo e não promove nada a ground truth nem a `positive_strong`.

## 2. Relação com GT01 a GT07

O GT01–GT07 formalizaram a política, aplicaram-na, montaram a fila, resolveram datas,
prepararam geometria, selecionaram os canários e prepararam o QA humano. O GT08 avança a
engenharia até a fronteira da aquisição, **sem cruzá-la**.

## 3. Por que SAR entra como canário e nada é baixado/processado

O SAR é apenas um teste de aderência em **poucos** canários; a generalização do REV-P segue
por features escaláveis por patch. Toda consulta Sentinel-1 aqui é **especificação futura**
(`metadata_only=true`, `no_download=true`, `remote_execution_allowed_now=false`). O footprint
futuro pós-evento será **referência de avaliação**, nunca feature pré-evento.

## 4. Canários e AOI

Entraram **{manifest['canarios']}** canários. Distribuição de AOI:
{"; ".join(f"{k}={v}" for k, v in manifest['distribuicao_aoi'].items())}.

| alvo | patch | cidade/bairro | AOI | pré/pós locais | pareamento | pronto p/ busca |
| --- | --- | --- | --- | --- | --- | --- |
{linhas_alvo}

A AOI forte vem do **bbox** (retângulo envolvente) dos patches no registro versionado
`susc_18a_unified_patch_registry.csv`; nenhuma geometria nova foi criada.

## 5. Catálogos Sentinel-1 locais

Catálogos locais versionados encontrados: **{manifest['catalogos_locais']}**;
cenas avaliadas: **{manifest['cenas_locais_avaliadas']}**.

| catálogo | arquivo | região | cenas | casa evento do canário |
| --- | --- | --- | --- | --- |
{linhas_cat}

## 6. Especificações de consulta Sentinel-1

Uma especificação por canário em `susc_gt08_especificacoes_consulta_sentinel1.csv`, com
política fixa: coleção `sentinel1_grd`, instrumento `SAR`, produto `GRD`, modo `IW`,
polarização `VV,VH`, órbita `any`, janelas pré/pós do GT04, `metadata_only=true`,
`no_download=true` e `remote_execution_allowed_now=false`.

## 7. Cenas candidatas locais (replay)

Do catálogo local, foram classificadas **{manifest['pre_candidatas']}** cenas
`pre_event_candidate` e **{manifest['post_candidatas']}** `post_event_candidate` (as demais
como `outside_window`, `wrong_product`, `wrong_mode` ou `insufficient_overlap`). Exemplos:
pré = {ex_pre}; pós = {ex_post}. Cada cena é apenas metadado (`metadata_only=true`,
`no_download_in_this_milestone=true`).

## 8. Contrato de pareamento pré/pós

Em `susc_gt08_contrato_pareamento_pre_pos.csv`: exige ≥1 cena pré e ≥1 pós, ambas
intersectando a AOI, produto GRD, modo IW e polarização VV/VH, preferindo a mesma órbita.
Canários com par completo pré/pós: **{manifest['canarios_com_par_completo']}**.

## 9. Plano de execução SAR futura

Em `susc_gt08_plano_execucao_sar_futura.csv`, onze passos por canário (confirmar AOI; buscar
metadados; selecionar pré; selecionar pós; **baixar apenas em marco futuro**; pré-processar
SAR; máscara de água permanente; filtro HAND/slope; exclusões urbanas; footprint candidato;
QA humano). Todos com `no_execution_now=true`.

## 10. Bloqueios

Em `susc_gt08_bloqueios_sentinel1.csv`: AOI insuficiente, janela ausente e par pré/pós
incompleto quando aplicável.

## 11. Por que nada foi baixado ou processado

Este marco é **metadata-only**: prepara a busca e o pareamento a partir do que já está
versionado, sem rede e sem download. A aquisição real fica para marcos futuros controlados.

## 12. Confirmação explícita dos bloqueios

**Não** houve internet, STAC/API remota, download de Sentinel/raster
(`download_executado={manifest['download_executado']}`), SAR/GEE executado
(`sar_executado={manifest['sar_executado']}`), footprint criado
(`footprint_criado={manifest['footprint_criado']}`), geometria criada, QA executado,
alteração do `score_v6` (`score_v6_changed={str(manifest['score_v6_changed']).lower()}`) nem
promoção a `positive_strong`
(`positive_strong_promovidos={manifest['positive_strong_promovidos']}`). Buscas/downloads
agora: `can_execute_search_now_true_count={manifest['can_execute_search_now_true_count']}`,
`can_download_now_true_count={manifest['can_download_now_true_count']}`. Contagens de
controle: `eligible_for_training=true` → {manifest['training_true_count']};
`eligible_for_ground_truth=true` → {manifest['ground_truth_true_count']};
`score_v7_candidate=true` → {manifest['score_v7_candidate_true_count']}.

O REV-P não prevê enchentes operacionalmente: produz análise estrutural review-only com
evidência observacional auditável.

## 13. Próximo passo recomendado

**{manifest['proximo_marco']}**. Com AOI forte e cenas locais pré/pós já identificadas para
os canários de Recife, o próximo passo natural é consolidar o pacote de revisão visual dos
pares localizados, mantendo tudo review-only e sem download.
"""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_all() -> dict:
    require_inputs()
    ensure_dir(OUT)
    ensure_dir(SCHEMAS_DIR)

    bundle = build_rows()
    summary = summary_rows(bundle)

    write_csv(TARGETS_CSV, bundle["targets"])
    write_csv(AOI_CSV, bundle["aoi"])
    write_csv(QUERY_CSV, bundle["query"])
    write_csv(CATALOG_CSV, bundle["catalog"])
    write_csv(SCENES_CSV, bundle["scenes"])
    write_csv(PAIRING_CSV, bundle["pairing"])
    write_csv(PLAN_CSV, bundle["plan"])
    write_csv(BLOCKERS_CSV, bundle["blockers"])
    write_csv(SUMMARY_CSV, summary)

    write_json(SCHEMA_TARGET, schema_target_obj())
    write_json(SCHEMA_AOI, schema_aoi_obj())
    write_json(SCHEMA_QUERY, schema_query_obj())
    write_json(SCHEMA_SCENE, schema_scene_obj())
    write_json(SCHEMA_PAIR, schema_pair_obj())
    write_json(SCHEMA_SUMMARY, schema_summary_obj())

    manifest = manifest_obj(bundle)
    write_markdown(REPORT, report_text(manifest, bundle))
    manifest["artifacts"] = [
        {"path": rel(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in PUBLIC_ARTIFACTS + [SCHEMA_TARGET, SCHEMA_AOI, SCHEMA_QUERY, SCHEMA_SCENE, SCHEMA_PAIR, SCHEMA_SUMMARY]
        if p.exists()
    ]
    write_json(MANIFEST_JSON, manifest)
    return manifest


# ---------------------------------------------------------------------------
# Validacao
# ---------------------------------------------------------------------------
def _public_paths() -> list[Path]:
    return sorted((p for p in REQUIRED_OUTPUTS if p.suffix.lower() in {".csv", ".json", ".md"} and p.exists()),
                  key=lambda p: rel(p))


def public_text_violations_text(text: str) -> list[str]:
    return sorted({m.group(0) for m in PUBLIC_FORBIDDEN_RE.finditer(text)})


def operational_forecast_violations(text: str) -> list[str]:
    return sorted({m.group(0).strip() for m in OPERATIONAL_FORECAST_RE.finditer(text)})


def _report_h1(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def title_is_portuguese(h1: str) -> bool:
    return TITULO_H1_TOKEN in h1


def validate_outputs(manifest: dict) -> list[str]:
    errors = [f"missing:{rel(p)}" for p in REQUIRED_OUTPUTS if not p.exists()]
    if errors:
        return errors

    if manifest["score_v6_changed"]:
        errors.append("score_v6_changed_forbidden")
    if manifest["score_v7_created"]:
        errors.append("score_v7_created_forbidden")
    if manifest["positive_strong_promovidos"] != 0:
        errors.append("positive_strong_promovido_forbidden")
    for k in ("download_executado", "sar_executado", "footprint_criado", "geometria_criada", "qa_executado", "internet_usada", "stac_api_usada"):
        if manifest[k] != 0:
            errors.append(f"{k}_forbidden")
    if manifest["can_execute_search_now_true_count"] != 0 or manifest["can_download_now_true_count"] != 0:
        errors.append("execucao_ou_download_agora_forbidden")

    targets = _read_csv(TARGETS_CSV)
    aoi = {a["sentinel1_target_id"]: a for a in _read_csv(AOI_CSV)}
    query = _read_csv(QUERY_CSV)
    scenes = _read_csv(SCENES_CSV)
    summary = _read_csv(SUMMARY_CSV)

    for t in targets:
        sid = t["sentinel1_target_id"]
        for fld in ("eligible_for_training", "eligible_for_ground_truth", "trainable", "score_v7_candidate"):
            if t.get(fld) != "false":
                errors.append(f"alvo:{sid}:{fld}_nao_false")
        if t.get("remains_review_only") != "true":
            errors.append(f"alvo:{sid}:review_only_nao_true")
        if t.get("can_execute_search_now") != "false" or t.get("can_download_now") != "false":
            errors.append(f"alvo:{sid}:execucao_agora_forbidden")
        if t.get("aoi_status") not in AOI_STATUSES:
            errors.append(f"alvo:{sid}:aoi_status_invalido")
        if t.get("pairing_status") not in PAIRING_STATUSES:
            errors.append(f"alvo:{sid}:pairing_status_invalido")
        blob = t.get("rationale_pt", "").lower()
        if "positive_strong" in blob:
            errors.append(f"alvo:{sid}:rationale_positive_strong")
        # canario sem AOI suficiente ou sem janela nao pode estar pronto para busca
        if t.get("ready_for_metadata_search") == "true":
            a = aoi.get(sid, {})
            if a.get("sufficient_for_search") != "true":
                errors.append(f"alvo:{sid}:pronto_sem_aoi")
            if not _present(t.get("pre_event_window", "").split("..")[0]):
                errors.append(f"alvo:{sid}:pronto_sem_janela")

    # query spec: metadata_only/no_download/remote_execution
    for q in query:
        if q.get("metadata_only") != "true":
            errors.append(f"query:{q.get('query_spec_id')}:metadata_only_nao_true")
        if q.get("no_download") != "true":
            errors.append(f"query:{q.get('query_spec_id')}:no_download_nao_true")
        if q.get("remote_execution_allowed_now") != "false":
            errors.append(f"query:{q.get('query_spec_id')}:remote_execution_permitida")

    # cenas: candidata valida exige scene_id/product_id; pre/post exige dentro da janela
    for s in scenes:
        cls = s.get("scene_classification")
        if cls not in SCENE_CLASSES:
            errors.append(f"cena:{s.get('scene_candidate_id')}:classe_invalida")
        if cls in ("pre_event_candidate", "post_event_candidate"):
            if not _present(s.get("scene_id")) or s.get("scene_id") == "not_available":
                errors.append(f"cena:{s.get('scene_candidate_id')}:candidata_sem_scene_id")
            if s.get("aoi_overlap") != "true":
                errors.append(f"cena:{s.get('scene_candidate_id')}:candidata_sem_overlap")
        if s.get("metadata_only") != "true" or s.get("no_download_in_this_milestone") != "true":
            errors.append(f"cena:{s.get('scene_candidate_id')}:download_flag_incorreta")

    # resumo: contagens proibidas 0
    for r in summary:
        for fld in ("can_execute_search_now_count", "can_download_now_count", "training_allowed_count",
                    "ground_truth_allowed_count", "score_v7_candidate_count"):
            if r.get(fld) != "0":
                errors.append(f"resumo:{r['aoi_status']}:{fld}_nao_zero")

    # texto publico
    for path in _public_paths():
        txt = path.read_text(encoding="utf-8", errors="ignore")
        if operational_forecast_violations(txt):
            errors.append(f"{rel(path)}:claim_previsao_operacional")
        if public_text_violations_text(txt):
            errors.append(f"{rel(path)}:vocabulario_publico_proibido")
    report_txt = REPORT.read_text(encoding="utf-8", errors="ignore")
    if "review-only" not in report_txt.lower() and "review_only" not in report_txt.lower():
        errors.append("relatorio_omite_review_only")
    if not title_is_portuguese(_report_h1(report_txt)):
        errors.append("titulo_principal_nao_portugues")

    ignored = gt02.gt01.git_ignored(REQUIRED_OUTPUTS)
    for p in REQUIRED_OUTPUTS:
        if rel(p) in ignored:
            errors.append(f"artefato_ignorado_pelo_git:{rel(p)}")

    return errors


def validate() -> int:
    manifest = build_all()
    errors = validate_outputs(manifest)
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(
        "GT08 prontidao Sentinel-1 validada: "
        f"canarios={manifest['canarios']} aoi={manifest['distribuicao_aoi']} "
        f"cenas_locais={manifest['cenas_locais_avaliadas']} pre={manifest['pre_candidatas']} "
        f"pos={manifest['post_candidatas']} pares={manifest['canarios_com_par_completo']} "
        f"proximo={manifest['proximo_marco']} score_v6_changed={str(manifest['score_v6_changed']).lower()}"
    )
    return 0


def run_all() -> int:
    manifest = build_all()
    print(
        "GT08 prontidao Sentinel-1 gerada: "
        f"canarios={manifest['canarios']} aoi={manifest['distribuicao_aoi']} "
        f"cenas_locais={manifest['cenas_locais_avaliadas']} pares={manifest['canarios_com_par_completo']} "
        f"proximo={manifest['proximo_marco']}"
    )
    return 0
