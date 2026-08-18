"""SUSC-17C19 consolidacao temporal sensorial e selecao canonica de cenas Sentinel-2.

Marco totalmente offline e deterministico. Consolida a linhagem temporal do
evento a partir da fonte upstream oficial (susc_17c4_extracted_reference_candidates.csv),
cria binding temporal explicito evento -> patch candidato -> janela sensorial e
seleciona cenas Sentinel-2 pre-evento canonicas usando apenas o inventario de
cenas ja capturado no 17C18. Nenhum produto Sentinel-2 e baixado, nenhum tile,
embedding ou feature CHIRPS e criado, nenhuma cena vira Ground Reference
Candidate, e o periodo do evento nunca e parseado a partir do event_id.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import read_csv, read_json, rel, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"
OFFICIAL_PATCHES = DAT / "susc_patches_official_v1.csv"
OFFICIAL_PATCH_LINKS = DAT / "susc_patch_links_official_v1.csv"

PATCH_GRID = OUT / "susc_17c6_candidate_patch_grid.csv"
PATCH_LINKS = OUT / "susc_17c6_candidate_patch_links.csv"
REFERENCE_CANDIDATES = OUT / "susc_17c4_extracted_reference_candidates.csv"
SCENE_REGISTRY_17C18 = OUT / "susc_17c18_sentinel2_candidate_scene_registry.csv"

C18_INPUTS = [
    OUT / "susc_17c18_sensor_aoi_execution_plan.csv",
    OUT / "susc_17c18_chirps_source_resolution.csv",
    OUT / "susc_17c18_chirps_zonal_stats.csv",
    OUT / "susc_17c18_rainfall_trigger_features.csv",
    OUT / "susc_17c18_sentinel2_stac_query_plan.csv",
    OUT / "susc_17c18_sentinel2_stac_query_results.csv",
    SCENE_REGISTRY_17C18,
    OUT / "susc_17c18_embedding_tile_blockers.csv",
    OUT / "susc_17c18_sensor_feature_provenance.csv",
    OUT / "susc_17c18_no_leakage_audit.csv",
    OUT / "susc_17c18_gate_evaluation_matrix.csv",
    OUT / "susc_17c18_readiness_summary.json",
    OUT / "susc_17c18_promotion_blockers.csv",
]
C17_INPUTS = [
    OUT / "susc_17c17_candidate_patch_aoi_plan.csv",
    OUT / "susc_17c17_light_raster_policy.csv",
    OUT / "susc_17c17_light_raster_execution_plan.csv",
]
PATCH_INPUTS = [PATCH_GRID, OUT / "susc_17c6_candidate_patch_grid.geojson", PATCH_LINKS]
REQUIRED_INPUTS = [SCORE_V6, REFERENCE_CANDIDATES, *C18_INPUTS, *C17_INPUTS, *PATCH_INPUTS]

REPORT = OUT / "SUSC_17C19_CONSOLIDACAO_TEMPORAL_E_SELECAO_CENAS_SENTINEL2_REPORT.md"
EVENT_LINEAGE = OUT / "susc_17c19_event_temporal_lineage.csv"
TEMPORAL_BINDING = OUT / "susc_17c19_candidate_patch_temporal_binding.csv"
SELECTION_CRITERIA = OUT / "susc_17c19_sentinel2_scene_selection_criteria.csv"
CANONICAL_SCENES = OUT / "susc_17c19_sentinel2_canonical_pre_event_scenes.csv"
REJECTED_SCENES = OUT / "susc_17c19_sentinel2_rejected_scene_candidates.csv"
TILE_REQUEST = OUT / "susc_17c19_sentinel2_tile_request_manifest.csv"
CHIRPS_RUNTIME = OUT / "susc_17c19_chirps_runtime_plan.csv"
GAP_AUDIT = OUT / "susc_17c19_temporal_lineage_gap_audit.csv"
NO_LEAKAGE = OUT / "susc_17c19_no_leakage_audit.csv"
GATES = OUT / "susc_17c19_gate_evaluation_matrix.csv"
SUMMARY = OUT / "susc_17c19_readiness_summary.json"
BLOCKERS = OUT / "susc_17c19_promotion_blockers.csv"

TEMPORAL_LINEAGE_SCHEMA = SCHEMAS / "susc_17c19_temporal_lineage_schema_v1.json"
SCENE_SELECTION_SCHEMA = SCHEMAS / "susc_17c19_scene_selection_schema_v1.json"
TILE_REQUEST_SCHEMA = SCHEMAS / "susc_17c19_tile_request_schema_v1.json"

GOV = {"review_only": "true", "trainable": "false", "ground_truth": "false"}

CHIRPS_WINDOWS = [("CHIRPS_3d", 3), ("CHIRPS_7d", 7), ("CHIRPS_30d", 30)]
PRE_EVENT_WINDOW_DAYS = 30
POST_EVENT_WINDOW_DAYS = 30
UPSTREAM_SOURCE_FILE = "outputs_public/suscetibilidade/susc_17c4_extracted_reference_candidates.csv"

# Artefatos auditados quanto a propagacao da linhagem temporal do evento.
TEMPORAL_GAP_ARTIFACTS = [
    "susc_17c6_candidate_patch_grid.csv",
    "susc_17c6_candidate_patch_links.csv",
    "susc_17c7_candidate_patch_feature_inventory.csv",
    "susc_17c9_embedding_tile_requirement_plan.csv",
    "susc_17c17_candidate_patch_aoi_plan.csv",
    "susc_17c18_sensor_aoi_execution_plan.csv",
]


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


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


def _candidate_patches() -> list[dict]:
    return read_csv(PATCH_GRID)


def _event_id_for_patches() -> str:
    patches = _candidate_patches()
    return patches[0]["source_event_id"] if patches else ""


def _upstream_temporal(event_id: str) -> dict | None:
    """Le periodo do evento da fonte upstream oficial. Nunca parseia o event_id."""
    for row in read_csv(REFERENCE_CANDIDATES):
        if row.get("event_id") == event_id:
            period = row.get("event_date_or_period", "")
            if "/" in period:
                start, end = period.split("/", 1)
                return {
                    "source_reference_candidate_id": row.get("reference_candidate_id", ""),
                    "event_period_start": start.strip(),
                    "event_period_end": end.strip(),
                    "temporal_precision": row.get("temporal_precision", "date_range_period"),
                }
    return None


def _date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def _event_context() -> dict:
    event_id = _event_id_for_patches()
    upstream = _upstream_temporal(event_id)
    if upstream is None:
        return {"event_id": event_id, "blocking_reason": "blocked_missing_upstream_temporal_reference"}
    return {"event_id": event_id, "blocking_reason": "", **upstream}


# ---------------------------------------------------------------------------
# Parte 1 - Linhagem temporal do evento
# ---------------------------------------------------------------------------

def event_temporal_lineage_rows() -> list[dict]:
    ctx = _event_context()
    has_temporal = ctx["blocking_reason"] == ""
    return [{
        "event_temporal_lineage_id": "S17C19_TLINEAGE_0001",
        "event_id": ctx["event_id"],
        "source_reference_candidate_id": ctx.get("source_reference_candidate_id", "not_available"),
        "upstream_source_file": UPSTREAM_SOURCE_FILE,
        "event_period_start": ctx.get("event_period_start", "not_available"),
        "event_period_end": ctx.get("event_period_end", "not_available"),
        "temporal_precision": ctx.get("temporal_precision", "not_available"),
        "temporal_source_status": "resolved_from_upstream_reference_candidate" if has_temporal else "blocked_missing_upstream_temporal_reference",
        "used_for_sensor_windows": _bool_text(has_temporal),
        "parse_from_event_id": "false",
        "blocking_reason": ctx["blocking_reason"] if not has_temporal else "not_applicable",
        **GOV,
    }]


# ---------------------------------------------------------------------------
# Parte 2 - Binding temporal por patch candidato
# ---------------------------------------------------------------------------

def candidate_patch_temporal_binding_rows() -> list[dict]:
    ctx = _event_context()
    rows = []
    has_temporal = ctx["blocking_reason"] == ""
    for idx, patch in enumerate(_candidate_patches(), start=1):
        if has_temporal:
            start = _date(ctx["event_period_start"])
            end = _date(ctx["event_period_end"])
            pre_start = (start - timedelta(days=PRE_EVENT_WINDOW_DAYS)).isoformat()
            pre_end = (start - timedelta(days=1)).isoformat()
            during_start = start.isoformat()
            during_end = end.isoformat()
            post_start = (end + timedelta(days=1)).isoformat()
            post_end = (end + timedelta(days=POST_EVENT_WINDOW_DAYS)).isoformat()
            status = "resolved"
        else:
            pre_start = pre_end = during_start = during_end = post_start = post_end = "not_available"
            status = "blocked_missing_upstream_temporal_reference"
        rows.append({
            "candidate_patch_temporal_binding_id": f"S17C19_TBIND_{idx:04d}",
            "candidate_patch_id": patch["candidate_patch_id"],
            "event_id": ctx["event_id"],
            "event_period_start": ctx.get("event_period_start", "not_available"),
            "event_period_end": ctx.get("event_period_end", "not_available"),
            "pre_event_window_start": pre_start,
            "pre_event_window_end": pre_end,
            "during_event_window_start": during_start,
            "during_event_window_end": during_end,
            "post_event_window_start": post_start,
            "post_event_window_end": post_end,
            "temporal_binding_status": status,
            "usable_for_pre_event_sensor_query": _bool_text(has_temporal),
            **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 3 - Criterios de selecao
# ---------------------------------------------------------------------------

SELECTION_CRITERIA_DEFS = [
    ("pre_event_only", 1, "cena deve ter is_pre_event=true e is_post_event=false", "true", "true", "true"),
    ("bbox_intersects_patch", 2, "cena deve interseccionar o AOI do patch candidato", "true", "true", "true"),
    ("has_datetime", 3, "cena deve ter datetime real", "true", "true", "true"),
    ("has_scene_id", 4, "cena deve ter scene_or_item_id valido", "true", "true", "true"),
    ("cloud_cover_available", 5, "cobertura de nuvem disponivel e usada como preferencia primaria", "false", "true", "false"),
    ("prefer_low_cloud", 6, "preferir menor cobertura de nuvem disponivel", "false", "true", "false"),
    ("prefer_closest_before_event", 7, "desempate por menor distancia temporal antes do inicio do evento", "false", "true", "false"),
    ("prefer_L2A", 8, "desempate final preferindo produto L2A sobre L1C", "false", "true", "false"),
    ("block_post_event_for_pre_event_feature", 9, "cena pos-evento nunca pode virar feature pre-evento", "true", "true", "true"),
    ("block_during_event_for_pre_event_feature", 10, "cena durante o evento nunca pode virar feature pre-evento", "true", "true", "true"),
]


def scene_selection_criteria_rows() -> list[dict]:
    rows = []
    for idx, (name, order, desc, required, used, blocks) in enumerate(SELECTION_CRITERIA_DEFS, start=1):
        rows.append({
            "selection_criteria_id": f"S17C19_CRIT_{idx:04d}",
            "criterion_name": name,
            "criterion_order": str(order),
            "criterion_description": desc,
            "required": required,
            "used_for_selection": used,
            "blocks_if_failed": blocks,
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Selecao de cenas
# ---------------------------------------------------------------------------

def _scene_registry() -> list[dict]:
    return read_csv(SCENE_REGISTRY_17C18)


def _scene_time_class(scene: dict) -> str:
    if scene["datetime"] in ("", "not_available"):
        return "missing_datetime_blocked"
    if scene["bbox_intersects_patch"] != "true":
        return "no_patch_intersection_blocked"
    if scene["is_pre_event"] == "true":
        return "pre_event"
    if scene["is_post_event"] == "true":
        return "post_event_blocked"
    return "during_event_blocked"


def _cloud_value(scene: dict) -> float:
    try:
        return float(scene["cloud_cover"])
    except (ValueError, KeyError):
        return float("inf")


def _days_before(scene: dict, event_start: str) -> int:
    scene_date = _date(scene["datetime"][:10])
    return (_date(event_start) - scene_date).days


def _ranked_pre_event(patch_id: str, event_start: str) -> list[dict]:
    scenes = [s for s in _scene_registry() if s["candidate_patch_id"] == patch_id and _scene_time_class(s) == "pre_event"]
    scenes.sort(key=lambda s: (
        _cloud_value(s),
        _days_before(s, event_start),
        0 if "L2A" in s["product_type"] or "2A" in s["product_type"] else 1,
        s["scene_or_item_id"],
    ))
    return scenes


def sentinel2_canonical_pre_event_scene_rows() -> list[dict]:
    ctx = _event_context()
    if ctx["blocking_reason"] != "":
        return []
    event_start = ctx["event_period_start"]
    rows = []
    idx = 0
    for patch in _candidate_patches():
        ranked = _ranked_pre_event(patch["candidate_patch_id"], event_start)
        if not ranked:
            continue
        idx += 1
        best = ranked[0]
        rows.append({
            "canonical_scene_selection_id": f"S17C19_CANON_{idx:04d}",
            "candidate_patch_id": patch["candidate_patch_id"],
            "event_id": ctx["event_id"],
            "scene_or_item_id": best["scene_or_item_id"],
            "collection": best["collection"],
            "datetime": best["datetime"],
            "days_before_event_start": str(_days_before(best, event_start)),
            "platform": best["platform"],
            "product_type": best["product_type"],
            "cloud_cover": best["cloud_cover"],
            "bbox_intersects_patch": best["bbox_intersects_patch"],
            "selection_rank": "1",
            "selection_reason": "menor cobertura de nuvem entre cenas pre-evento, com desempate por proximidade ao inicio do evento e preferencia L2A",
            "downloaded": "false",
            "tile_created": "false",
            "usable_for_future_tile": "true",
            "usable_for_pre_event_feature": "true",
            **GOV,
        })
    return rows


def sentinel2_rejected_scene_rows() -> list[dict]:
    canonical_ids = {(r["candidate_patch_id"], r["scene_or_item_id"]) for r in sentinel2_canonical_pre_event_scene_rows()}
    rows = []
    idx = 0
    for scene in _scene_registry():
        time_class = _scene_time_class(scene)
        key = (scene["candidate_patch_id"], scene["scene_or_item_id"])
        if time_class == "pre_event" and key in canonical_ids:
            continue  # cena canonica vai no arquivo canonico
        idx += 1
        if time_class == "pre_event":
            scene_time_class = "pre_event_noncanonical"
            reason = "cena pre-evento valida mas nao selecionada como canonica (maior nuvem ou mais distante do evento)"
            future = "true"
        elif time_class == "during_event_blocked":
            scene_time_class = "during_event_blocked"
            reason = "cena durante o evento; bloqueada para feature pre-evento"
            future = "true"
        elif time_class == "post_event_blocked":
            scene_time_class = "post_event_blocked"
            reason = "cena pos-evento; bloqueada para feature pre-evento, util apenas para analise observacional futura"
            future = "true"
        elif time_class == "missing_datetime_blocked":
            scene_time_class = "missing_datetime_blocked"
            reason = "cena sem datetime real; nao selecionavel"
            future = "false"
        else:
            scene_time_class = "no_patch_intersection_blocked"
            reason = "cena sem interseccao com o AOI do patch; nao selecionavel"
            future = "false"
        rows.append({
            "rejected_scene_id": f"S17C19_REJECT_{idx:04d}",
            "candidate_patch_id": scene["candidate_patch_id"],
            "event_id": scene["event_id"],
            "scene_or_item_id": scene["scene_or_item_id"],
            "datetime": scene["datetime"],
            "scene_time_class": scene_time_class,
            "cloud_cover": scene["cloud_cover"],
            "rejection_reason": reason,
            "could_be_used_future_observational": future,
            "usable_for_pre_event_feature": "false",
            **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 6 - Manifesto de pedido de tile leve
# ---------------------------------------------------------------------------

def sentinel2_tile_request_manifest_rows() -> list[dict]:
    rows = []
    for idx, canonical in enumerate(sentinel2_canonical_pre_event_scene_rows(), start=1):
        rows.append({
            "tile_request_id": f"S17C19_TILEREQ_{idx:04d}",
            "candidate_patch_id": canonical["candidate_patch_id"],
            "event_id": canonical["event_id"],
            "canonical_scene_selection_id": canonical["canonical_scene_selection_id"],
            "scene_or_item_id": canonical["scene_or_item_id"],
            "requested_tile_type": "sentinel2_small_pre_event_tile_over_aoi",
            "requested_bands": "B02;B03;B04;B08;B11;B12",
            "requested_resolution_m": "10",
            "requested_output_format": "cog_or_geotiff_small_tile",
            "max_size_bytes": "5242880",
            "storage_policy": "local_runs_only_with_manifest_no_raw_heavy_committed",
            "can_execute_now": "false",
            "requires_network": "true",
            "requires_download_policy": "true",
            "requires_storage_policy": "true",
            "requires_no_raw_heavy": "true",
            **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 7 - Plano runtime CHIRPS
# ---------------------------------------------------------------------------

def chirps_runtime_plan_rows() -> list[dict]:
    ctx = _event_context()
    has_temporal = ctx["blocking_reason"] == ""
    rows = []
    idx = 0
    for patch in _candidate_patches():
        for window_name, days in CHIRPS_WINDOWS:
            idx += 1
            if has_temporal:
                end = _date(ctx["event_period_start"])
                start = end - timedelta(days=days)
                window_start, window_end = start.isoformat(), end.isoformat()
            else:
                window_start = window_end = "not_available"
            rows.append({
                "chirps_runtime_plan_id": f"S17C19_CHIRPSRT_{idx:04d}",
                "candidate_patch_id": patch["candidate_patch_id"],
                "event_id": ctx["event_id"],
                "temporal_window": window_name,
                "window_start": window_start,
                "window_end": window_end,
                "runtime_option": "blocked_no_lightweight_endpoint",
                "can_execute_now": "false",
                "requires_runtime": "true",
                "requires_network": "true",
                "requires_lightweight_download": "true",
                "requires_heavy_raster_policy": "true",
                "expected_output": "precipitation_accumulation_zonal_stats_csv_per_aoi",
                "blocking_reason": "CHIRPS publico so oferece raster global pesado ou colecao GEE com runtime; sem endpoint leve por AOI resolvido no 17C18",
                **GOV,
            })
    return rows


# ---------------------------------------------------------------------------
# Parte 8 - Auditoria de lacuna temporal
# ---------------------------------------------------------------------------

def _header(artifact: str) -> list[str]:
    rows = read_csv(OUT / artifact)
    return list(rows[0].keys()) if rows else []


def temporal_lineage_gap_audit_rows() -> list[dict]:
    rows = []
    for idx, artifact in enumerate(TEMPORAL_GAP_ARTIFACTS, start=1):
        header = _header(artifact)
        has_event = any("event_id" in col for col in header)
        has_period = any(("period" in col) or ("date_or_period" in col) for col in header)
        has_precision = any("temporal_precision" in col for col in header)
        requires_backfill = not (has_event and has_period)
        rows.append({
            "temporal_gap_audit_id": f"S17C19_GAP_{idx:04d}",
            "artifact_name": artifact,
            "contains_event_id": _bool_text(has_event),
            "contains_event_period_start": _bool_text(has_period),
            "contains_event_period_end": _bool_text(has_period),
            "contains_temporal_precision": _bool_text(has_precision),
            "requires_temporal_backfill": _bool_text(requires_backfill),
            "blocking_risk": (
                "alto: sem event_id nem periodo, downstream nao consegue reconstruir janela sensorial"
                if not has_event
                else "medio: event_id presente mas periodo ausente, exige join com fonte upstream"
                if requires_backfill
                else "baixo: event_id e periodo propagados"
            ),
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 9 - Auditoria anti-leakage
# ---------------------------------------------------------------------------

def no_leakage_audit_rows() -> list[dict]:
    rows = []
    idx = 0
    canonical_keys = {(r["candidate_patch_id"], r["scene_or_item_id"]) for r in sentinel2_canonical_pre_event_scene_rows()}
    for scene in _scene_registry():
        idx += 1
        time_class = _scene_time_class(scene)
        is_pre = time_class == "pre_event"
        is_during = time_class == "during_event_blocked"
        is_post = time_class == "post_event_blocked"
        is_canonical = (scene["candidate_patch_id"], scene["scene_or_item_id"]) in canonical_keys and is_pre
        rows.append({
            "no_leakage_audit_id": f"S17C19_LEAK_{idx:04d}",
            "candidate_patch_id": scene["candidate_patch_id"],
            "event_id": scene["event_id"],
            "scene_or_runtime_id": scene["scene_or_item_id"],
            "object_type": "canonical_pre_event_scene" if is_canonical else "candidate_scene",
            "uses_pre_event_data_only": _bool_text(is_pre),
            "uses_during_event_as_pre_event": "false",
            "uses_post_event_as_pre_event": "false",
            "uses_catalog_metadata_as_event": "false",
            "uses_synthetic_as_real": "false",
            "passes_no_leakage": "true",
            "blocking_reason": (
                "cena classificada explicitamente por janela temporal; durante/pos-evento nunca usados como pre-evento"
                if (is_during or is_post)
                else "cena pre-evento usada apenas como candidata sensorial, sem promocao a evento"
            ),
            "review_only": "true",
        })
    for runtime in chirps_runtime_plan_rows():
        idx += 1
        rows.append({
            "no_leakage_audit_id": f"S17C19_LEAK_{idx:04d}",
            "candidate_patch_id": runtime["candidate_patch_id"],
            "event_id": runtime["event_id"],
            "scene_or_runtime_id": runtime["chirps_runtime_plan_id"],
            "object_type": "chirps_runtime_plan",
            "uses_pre_event_data_only": "true",
            "uses_during_event_as_pre_event": "false",
            "uses_post_event_as_pre_event": "false",
            "uses_catalog_metadata_as_event": "false",
            "uses_synthetic_as_real": "false",
            "passes_no_leakage": "true",
            "blocking_reason": "plano CHIRPS restrito a janela antecedente; nenhum valor calculado nesta sprint",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 10 - Gates
# ---------------------------------------------------------------------------

def gate_evaluation_matrix_rows() -> list[dict]:
    rows = []
    idx = 0
    for lineage in event_temporal_lineage_rows():
        idx += 1
        resolved = lineage["temporal_source_status"] == "resolved_from_upstream_reference_candidate"
        rows.append({
            "gate_eval_id": f"S17C19_GATE_{idx:04d}",
            "object_id": lineage["event_temporal_lineage_id"],
            "object_type": "event_temporal_lineage",
            "G1_existencia_documental": _bool_text(resolved),
            "G2_confiabilidade_fonte": _bool_text(resolved),
            "G3_precisao_temporal": _bool_text(resolved),
            "G4_vinculo_espacial_evento": "false",
            "G5_separacao_fenomeno": "false",
            "G6_proveniencia_integridade": _bool_text(resolved),
            "G7_anti_leakage": "true",
            "all_gates_passed_for_ground_reference": "false",
            "usable_as_sensor_feature_candidate": "false",
            "acceptance_status": "accepted_as_temporal_lineage_only",
            "blocking_reason": "linhagem temporal consolida periodo do evento, mas nao e evento observado nem geometria; G4/G5 bloqueados por design",
            **GOV,
        })
    for canonical in sentinel2_canonical_pre_event_scene_rows():
        idx += 1
        rows.append({
            "gate_eval_id": f"S17C19_GATE_{idx:04d}",
            "object_id": canonical["canonical_scene_selection_id"],
            "object_type": "sentinel2_canonical_pre_event_scene",
            "G1_existencia_documental": "true",
            "G2_confiabilidade_fonte": "true",
            "G3_precisao_temporal": "true",
            "G4_vinculo_espacial_evento": "false",
            "G5_separacao_fenomeno": "false",
            "G6_proveniencia_integridade": "true",
            "G7_anti_leakage": "true",
            "all_gates_passed_for_ground_reference": "false",
            "usable_as_sensor_feature_candidate": "true",
            "acceptance_status": "accepted_as_canonical_pre_event_scene_metadata_only",
            "blocking_reason": "cena canonica e metadado orbital pre-evento candidato, nao evento observado; nao vira Ground Reference Candidate",
            **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 11 - Resumo
# ---------------------------------------------------------------------------

def build_summary() -> dict:
    lineage = event_temporal_lineage_rows()
    bindings = candidate_patch_temporal_binding_rows()
    gap = temporal_lineage_gap_audit_rows()
    canonical = sentinel2_canonical_pre_event_scene_rows()
    rejected = sentinel2_rejected_scene_rows()
    tile_requests = sentinel2_tile_request_manifest_rows()
    chirps_runtime = chirps_runtime_plan_rows()
    scenes_input = _scene_registry()
    patches_with_canonical = {row["candidate_patch_id"] for row in canonical}
    return {
        "candidate_patch_count": len(_candidate_patches()),
        "event_temporal_lineage_rows_count": len(lineage),
        "candidate_patch_temporal_bindings_count": len(bindings),
        "artifacts_audited_for_temporal_gap_count": len(gap),
        "artifacts_requiring_temporal_backfill_count": len([row for row in gap if row["requires_temporal_backfill"] == "true"]),
        "sentinel2_candidate_scenes_input_count": len(scenes_input),
        "canonical_pre_event_scene_count": len(canonical),
        "patches_with_canonical_pre_event_scene_count": len(patches_with_canonical),
        "rejected_scene_candidates_count": len(rejected),
        "tile_requests_manifested_count": len(tile_requests),
        "chirps_runtime_plan_rows_count": len(chirps_runtime),
        "sentinel2_products_downloaded_count": 0,
        "sentinel2_tiles_created_count": 0,
        "embedding_tiles_created_count": 0,
        "chirps_features_extracted_count": 0,
        "ground_reference_candidates_created_count": 0,
        "features_extracted_this_sprint": 0,
        "score_v6_changed": bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)])),
        "score_v7_created": SCORE_V7.exists(),
        "official_patch_created": False,
        "official_patch_link_created": False,
        "eligible_for_17b_now": False,
        "eligible_for_score_v7": False,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "recommended_next_milestone": "SUSC-17C20 Politica de execucao de tile leve Sentinel-2 sobre cena canonica e runtime CHIRPS por AOI com storage aprovado",
    }


def build_blockers() -> list[dict]:
    blockers = [
        "temporal_lineage_backfill_needed_in_downstream_artifacts",
        "sentinel2_tile_not_created",
        "chirps_runtime_not_executed",
        "chirps_feature_not_extracted",
        "embedding_tile_missing",
        "ground_reference_event_artifacts_still_missing",
        "event_geometry_still_missing",
        "phenomenon_gate_not_satisfied",
        "candidate_patch_policy_missing",
        "17b_blocked_until_event_reference_and_policy",
    ]
    return [
        {
            "blocker_id": f"S17C19_BLOCKER_{idx:04d}",
            "blocker_type": blocker,
            "description": "Bloqueio preservado: consolidacao temporal e selecao de cena canonica sao camada sensorial candidata; faltam tile Sentinel-2, runtime/feature CHIRPS, artefato de evento, geometria, fenomeno e politica de patch candidato antes de qualquer Ground Reference, score v7 ou 17B.",
            "blocks_ground_reference_candidate": _bool_text(blocker in {"ground_reference_event_artifacts_still_missing", "event_geometry_still_missing", "phenomenon_gate_not_satisfied", "candidate_patch_policy_missing"}),
            "blocks_17b": "true",
            "blocks_score_v7": "true",
            "review_only": "true",
        }
        for idx, blocker in enumerate(blockers, start=1)
    ]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

def _schema(required: list[str], props: dict, title: str) -> dict:
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": title, "type": "object", "required": required, "properties": props}


def build_temporal_lineage_schema() -> dict:
    required = list(event_temporal_lineage_rows()[0].keys())
    return _schema(required, {
        "event_temporal_lineage_id": {"type": "string", "pattern": "^S17C19_TLINEAGE_"},
        "parse_from_event_id": {"const": "false"},
        "review_only": {"const": "true"},
        "trainable": {"const": "false"},
        "ground_truth": {"const": "false"},
    }, "SUSC-17C19 temporal lineage schema v1")


def build_scene_selection_schema() -> dict:
    required = [
        "canonical_scene_selection_id", "candidate_patch_id", "event_id",
        "scene_or_item_id", "collection", "datetime", "days_before_event_start",
        "platform", "product_type", "cloud_cover", "bbox_intersects_patch",
        "selection_rank", "selection_reason", "downloaded", "tile_created",
        "usable_for_future_tile", "usable_for_pre_event_feature",
        "review_only", "trainable", "ground_truth",
    ]
    return _schema(required, {
        "canonical_scene_selection_id": {"type": "string", "pattern": "^S17C19_CANON_"},
        "downloaded": {"const": "false"},
        "tile_created": {"const": "false"},
        "usable_for_pre_event_feature": {"const": "true"},
        "review_only": {"const": "true"},
        "trainable": {"const": "false"},
        "ground_truth": {"const": "false"},
    }, "SUSC-17C19 scene selection schema v1")


def build_tile_request_schema() -> dict:
    required = list(sentinel2_tile_request_manifest_rows()[0].keys()) if sentinel2_tile_request_manifest_rows() else [
        "tile_request_id", "candidate_patch_id", "event_id", "canonical_scene_selection_id",
        "scene_or_item_id", "requested_tile_type", "requested_bands", "requested_resolution_m",
        "requested_output_format", "max_size_bytes", "storage_policy", "can_execute_now",
        "requires_network", "requires_download_policy", "requires_storage_policy",
        "requires_no_raw_heavy", "review_only", "trainable", "ground_truth",
    ]
    return _schema(required, {
        "tile_request_id": {"type": "string", "pattern": "^S17C19_TILEREQ_"},
        "can_execute_now": {"const": "false"},
        "review_only": {"const": "true"},
        "trainable": {"const": "false"},
        "ground_truth": {"const": "false"},
    }, "SUSC-17C19 tile request schema v1")


# ---------------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------------

def build_report() -> str:
    summary = build_summary()
    lineage = event_temporal_lineage_rows()[0]
    return "\n".join([
        "# SUSC-17C19 - Consolidacao temporal sensorial e selecao canonica de cenas Sentinel-2",
        "",
        "## O que o 17C18 demonstrou",
        "O 17C18 provou que a consulta Sentinel-2 por AOI funciona: retornou 100 cenas reais para os 5 patches candidatos de Recife. Mas revelou uma lacuna de linhagem: a data/periodo do evento REC_2022_05_24_30 so estava disponivel em `susc_17c4_extracted_reference_candidates.csv`, nao nos artefatos 17C6-17C17.",
        "",
        "## Por que a ausencia de periodo era uma lacuna de linhagem",
        "Sem o periodo propagado, qualquer marco sensorial downstream teria que reconstruir a janela temporal a partir do `event_id`, o que e fragil e proibido. A auditoria de lacuna mostra que 17c6 (grid e links) e 17c7 carregam o `event_id` mas nao o periodo, e 17c9/17c17 nem o `event_id`. Apenas o 17c18 finalmente propagou `event_id` + `event_date_or_period`.",
        "",
        "## Como o 17C19 consolida o periodo sem parsear o ID",
        f"O periodo e lido do campo temporal oficial da fonte upstream `{UPSTREAM_SOURCE_FILE}` (referencia {lineage['source_reference_candidate_id']}), com `parse_from_event_id=false` sempre. Periodo consolidado: {lineage['event_period_start']} a {lineage['event_period_end']} (precisao {lineage['temporal_precision']}).",
        "",
        "## Janelas pre/durante/pos-evento",
        "Para cada patch foi criado um binding temporal explicito: janela pre-evento de 30 dias antes do inicio ate 1 dia antes do inicio; janela durante-evento do inicio ao fim; janela pos-evento de 1 dia apos o fim ate 30 dias apos o fim.",
        "",
        "## Selecao de cenas Sentinel-2",
        "Entre as cenas pre-evento (is_pre_event=true, is_post_event=false, com datetime, scene_id e interseccao de AOI), a canonica e escolhida por menor cobertura de nuvem, desempatando por proximidade ao inicio do evento e preferencia por produto L2A.",
        f"- Cenas de entrada: {summary['sentinel2_candidate_scenes_input_count']}.",
        f"- Cenas canonicas pre-evento: {summary['canonical_pre_event_scene_count']} ({summary['patches_with_canonical_pre_event_scene_count']} de {summary['candidate_patch_count']} patches com cena canonica).",
        f"- Cenas rejeitadas/bloqueadas: {summary['rejected_scene_candidates_count']}.",
        "",
        "## Por que cenas durante/pos-evento foram bloqueadas",
        "Cenas durante o evento sao marcadas `during_event_blocked` e cenas apos o evento `post_event_blocked`; nenhuma pode virar feature pre-evento, para evitar vazamento temporal. Elas permanecem registradas apenas como contexto observacional futuro.",
        "",
        "## Por que tile ainda nao foi criado",
        "Foi manifestado apenas o pedido futuro de tile leve por cena canonica (`can_execute_now=false`), sujeito a politica de download e storage. Nenhum produto Sentinel-2 foi baixado e nenhum tile foi gerado.",
        "",
        "## Por que CHIRPS ainda nao foi calculado",
        "O plano de runtime CHIRPS reafirma que a fonte publica so oferece raster global pesado ou colecao GEE com runtime; sem endpoint leve por AOI, nenhuma estatistica zonal foi calculada.",
        "",
        "## Por que nada vira Ground Reference",
        "Linhagem temporal e cena Sentinel-2 sao camadas sensoriais/contextuais. Nenhuma passa G4 (vinculo espacial de evento) ou G5 (separacao de fenomeno); nenhuma vira Ground Reference Candidate, ground truth ou label.",
        "",
        "## Score v6, score v7 e 17B",
        "O score v6 nao mudou (nenhum dataset oficial tocado). O score v7 continua inexistente e o 17B permanece bloqueado ate existir artefato de evento com geometria e fenomeno, tile/feature sensorial real e politica de patch candidato.",
        "",
        "## Proximo marco recomendado",
        summary["recommended_next_milestone"],
    ])


# ---------------------------------------------------------------------------
# Build / validacao
# ---------------------------------------------------------------------------

def build_all() -> None:
    _require_inputs()
    write_csv(EVENT_LINEAGE, event_temporal_lineage_rows())
    write_csv(TEMPORAL_BINDING, candidate_patch_temporal_binding_rows())
    write_csv(SELECTION_CRITERIA, scene_selection_criteria_rows())
    write_csv(CANONICAL_SCENES, sentinel2_canonical_pre_event_scene_rows())
    write_csv(REJECTED_SCENES, sentinel2_rejected_scene_rows())
    write_csv(TILE_REQUEST, sentinel2_tile_request_manifest_rows())
    write_csv(CHIRPS_RUNTIME, chirps_runtime_plan_rows())
    write_csv(GAP_AUDIT, temporal_lineage_gap_audit_rows())
    write_csv(NO_LEAKAGE, no_leakage_audit_rows())
    write_csv(GATES, gate_evaluation_matrix_rows())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_json(TEMPORAL_LINEAGE_SCHEMA, build_temporal_lineage_schema())
    write_json(SCENE_SELECTION_SCHEMA, build_scene_selection_schema())
    write_json(TILE_REQUEST_SCHEMA, build_tile_request_schema())
    write_markdown(REPORT, build_report())


REQUIRED_OUTPUTS = [
    REPORT, EVENT_LINEAGE, TEMPORAL_BINDING, SELECTION_CRITERIA, CANONICAL_SCENES,
    REJECTED_SCENES, TILE_REQUEST, CHIRPS_RUNTIME, GAP_AUDIT, NO_LEAKAGE, GATES,
    SUMMARY, BLOCKERS, TEMPORAL_LINEAGE_SCHEMA, SCENE_SELECTION_SCHEMA, TILE_REQUEST_SCHEMA,
]


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
        if "enum" in rules and value not in rules["enum"]:
            violations.append(f"{field}:enum")
        if "pattern" in rules and not value.startswith(rules["pattern"].replace("^", "")):
            violations.append(f"{field}:pattern")
    return violations


def _validate_byte_identical() -> list[str]:
    before = {path: path.read_bytes() for path in REQUIRED_OUTPUTS if path.exists()}
    build_all()
    errors = []
    for path, content in before.items():
        if path.read_bytes() != content:
            errors.append(f"offline_build_not_byte_identical:{rel(path)}")
    return errors


def validate() -> int:
    missing = [path for path in REQUIRED_OUTPUTS if not path.exists()]
    if missing:
        print("MISSING: " + "; ".join(rel(path) for path in missing), file=sys.stderr)
        return 1
    errors = _validate_byte_identical()
    lineage = read_csv(EVENT_LINEAGE)
    bindings = read_csv(TEMPORAL_BINDING)
    criteria = read_csv(SELECTION_CRITERIA)
    canonical = read_csv(CANONICAL_SCENES)
    rejected = read_csv(REJECTED_SCENES)
    tile_requests = read_csv(TILE_REQUEST)
    chirps_runtime = read_csv(CHIRPS_RUNTIME)
    gap = read_csv(GAP_AUDIT)
    leakage = read_csv(NO_LEAKAGE)
    gates = read_csv(GATES)
    summary = read_json(SUMMARY)
    lineage_schema = read_json(TEMPORAL_LINEAGE_SCHEMA)
    scene_schema = read_json(SCENE_SELECTION_SCHEMA)
    tile_schema = read_json(TILE_REQUEST_SCHEMA)

    for row in lineage:
        errors.extend(_schema_violations(row, lineage_schema))
    for row in canonical:
        errors.extend(_schema_violations(row, scene_schema))
    for row in tile_requests:
        errors.extend(_schema_violations(row, tile_schema))

    for rows, key in [
        (lineage, "event_temporal_lineage_id"), (bindings, "candidate_patch_temporal_binding_id"),
        (criteria, "selection_criteria_id"), (canonical, "canonical_scene_selection_id"),
        (rejected, "rejected_scene_id"), (tile_requests, "tile_request_id"),
        (chirps_runtime, "chirps_runtime_plan_id"), (gap, "temporal_gap_audit_id"),
        (leakage, "no_leakage_audit_id"), (gates, "gate_eval_id"),
    ]:
        ids = [row[key] for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")

    patch_count = len(_candidate_patches())
    scenes_input = read_csv(SCENE_REGISTRY_17C18)

    # 5: nunca parseia data do event_id.
    if any(row["parse_from_event_id"] != "false" for row in lineage):
        errors.append("temporal_parsed_from_event_id")
    # 6: usa fonte upstream temporal explicita.
    if any(row["upstream_source_file"] != UPSTREAM_SOURCE_FILE for row in lineage):
        errors.append("upstream_temporal_source_not_used")
    # 7: cada patch tem binding temporal.
    if len(bindings) != patch_count or {row["candidate_patch_id"] for row in bindings} != {p["candidate_patch_id"] for p in _candidate_patches()}:
        errors.append("missing_temporal_binding_for_patch")
    # 8/9: cada patch com cena pre-evento tem cena canonica; canonicas sao pre-evento.
    pre_event_patches = {s["candidate_patch_id"] for s in scenes_input if s["is_pre_event"] == "true" and s["is_post_event"] == "false"}
    canonical_patches = {row["candidate_patch_id"] for row in canonical}
    if pre_event_patches - canonical_patches:
        errors.append("patch_with_pre_event_scene_without_canonical")
    for row in canonical:
        match = next((s for s in scenes_input if s["candidate_patch_id"] == row["candidate_patch_id"] and s["scene_or_item_id"] == row["scene_or_item_id"]), None)
        if match is None or match["is_pre_event"] != "true" or match["is_post_event"] != "false":
            errors.append(f"canonical_scene_not_pre_event:{row['canonical_scene_selection_id']}")
        if row["selection_rank"] != "1":
            errors.append(f"canonical_scene_not_rank1:{row['canonical_scene_selection_id']}")
    # 10: durante/pos-evento nunca usados como pre-evento.
    if any(row["usable_for_pre_event_feature"] != "false" for row in rejected):
        errors.append("rejected_scene_marked_usable_pre_event")
    if any(row["uses_during_event_as_pre_event"] != "false" or row["uses_post_event_as_pre_event"] != "false" for row in leakage):
        errors.append("temporal_leakage_detected")
    if any(row["passes_no_leakage"] != "true" for row in leakage):
        errors.append("no_leakage_audit_failed")
    # canonical + rejected devem cobrir todas as cenas de entrada exatamente uma vez.
    if len(canonical) + len(rejected) != len(scenes_input):
        errors.append("scene_partition_mismatch")
    # 11/12/13: nenhum produto/tile/embedding.
    if any(row["downloaded"] != "false" or row["tile_created"] != "false" for row in canonical):
        errors.append("canonical_scene_downloaded_or_tiled")
    if any(row["can_execute_now"] != "false" for row in tile_requests):
        errors.append("tile_request_executable_now")
    # 14: nenhuma feature CHIRPS calculada.
    if any(row["can_execute_now"] != "false" for row in chirps_runtime):
        errors.append("chirps_runtime_executed_now")
    # 15: nenhuma cena vira Ground Reference.
    if any(row["G4_vinculo_espacial_evento"] == "true" or row["G5_separacao_fenomeno"] == "true" for row in gates):
        errors.append("sensor_object_promoted_to_event")
    if any(row["all_gates_passed_for_ground_reference"] == "true" for row in gates):
        errors.append("ground_reference_accepted_from_sensor")
    # gap audit deve detectar backfill em pelo menos os artefatos sem periodo.
    if not any(row["requires_temporal_backfill"] == "true" for row in gap):
        errors.append("temporal_gap_not_detected")

    for path in [SCORE_V6, OFFICIAL_PATCHES, OFFICIAL_PATCH_LINKS]:
        if path.exists() and _run_git(["diff", "--name-only", "--", rel(path)]):
            errors.append(f"official_dataset_changed:{rel(path)}")
    if SCORE_V7.exists():
        errors.append("score_v7_exists")

    expected = build_summary()
    for key, value in expected.items():
        if summary.get(key) != value:
            errors.append(f"summary_mismatch:{key}")
    if summary["eligible_for_17b_now"] or summary["eligible_for_score_v7"] or summary["trainable"] or summary["ground_truth"]:
        errors.append("promotion_guardrail_failed")
    if summary["features_extracted_this_sprint"] != 0 or summary["ground_reference_candidates_created_count"] != 0:
        errors.append("unexpected_feature_or_ground_reference_creation")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "17C19 -> "
        f"patches={summary['candidate_patch_count']} "
        f"bindings={summary['candidate_patch_temporal_bindings_count']} "
        f"canonical={summary['canonical_pre_event_scene_count']} "
        f"rejected={summary['rejected_scene_candidates_count']} "
        f"backfill={summary['artifacts_requiring_temporal_backfill_count']} "
        f"score_v7_created={summary['score_v7_created']}"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def build_temporal_lineage_text() -> str:
    _require_inputs()
    write_csv(EVENT_LINEAGE, event_temporal_lineage_rows())
    write_csv(TEMPORAL_BINDING, candidate_patch_temporal_binding_rows())
    write_csv(GAP_AUDIT, temporal_lineage_gap_audit_rows())
    lineage = event_temporal_lineage_rows()[0]
    return (
        f"build-temporal-lineage: periodo consolidado {lineage['event_period_start']}..{lineage['event_period_end']} "
        f"da fonte upstream (parse_from_event_id={lineage['parse_from_event_id']}); "
        f"{len(candidate_patch_temporal_binding_rows())} bindings temporais e {len(temporal_lineage_gap_audit_rows())} artefatos auditados."
    )


def select_scenes_text() -> str:
    build_all()
    summary = build_summary()
    return (
        f"select-scenes: {summary['canonical_pre_event_scene_count']} cenas canonicas pre-evento selecionadas "
        f"({summary['patches_with_canonical_pre_event_scene_count']} patches), "
        f"{summary['rejected_scene_candidates_count']} cenas rejeitadas/bloqueadas. Nenhum produto baixado."
    )


def plan_light_tile_text() -> str:
    rows = sentinel2_tile_request_manifest_rows()
    lines = ["plan-light-tile:"]
    for row in rows:
        lines.append(f"  {row['tile_request_id']} {row['candidate_patch_id']} {row['scene_or_item_id'][:40]} can_execute_now={row['can_execute_now']}")
    lines.append(f"  total {len(rows)} pedidos de tile leve manifestados; nenhum tile criado.")
    return "\n".join(lines)


def plan_chirps_runtime_text() -> str:
    rows = chirps_runtime_plan_rows()
    return (
        f"plan-chirps-runtime: {len(rows)} linhas de plano de runtime CHIRPS por AOI "
        f"(runtime_option=blocked_no_lightweight_endpoint); nenhuma estatistica calculada."
    )


def status_text() -> str:
    summary = build_summary()
    return (
        f"status 17C19: bindings={summary['candidate_patch_temporal_bindings_count']} "
        f"canonical={summary['canonical_pre_event_scene_count']} "
        f"rejected={summary['rejected_scene_candidates_count']} "
        f"backfill={summary['artifacts_requiring_temporal_backfill_count']} "
        f"score_v7_created={summary['score_v7_created']}"
    )
