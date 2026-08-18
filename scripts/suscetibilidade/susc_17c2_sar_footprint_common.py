"""SUSC-17C2 Sentinel-1/SAR Footprint Execution (review-only, fail-closed).

Executes (or honestly blocks) the SAR canaries prioritized by SUSC-17C to
produce dated, auditable candidate flood footprints. It NEVER changes score v6,
creates score v7, training data, supervised labels, models, ground truth or true
negatives. A candidate footprint is observational candidate evidence only:
``technical_remote_sensing_flood_footprint_candidate`` at ``qa_status=needs_review``,
never eligible for strong evaluation / calibration / 17B before human QA accepts
it. No raster is downloaded or committed. No geometry, date, source or footprint
is invented: blocked executions produce no footprint.

Runtime policy (offline-first, deterministic): real SAR execution requires GEE/
STAC credentials AND the opt-in ``SUSC_17C2_SAR_RUNTIME=1``. Without them the SAR
canaries are ``blocked_no_runtime_access`` with ready task specs. The
International Charter canary follows an official-geometry review path (rule 20):
its real official bbox (already committed in SUSC-13A) is materialized as a light
vector candidate footprint - not invented, not SAR-derived, still review-only.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import read_csv, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

AUDIT = OUT / "SUSC_WORKTREE_AUDIT_BEFORE_17C2.md"
FOOTPRINT_SCHEMA = SCHEMAS / "susc_17c2_candidate_footprint_schema_v1.json"
PATCH_LINK_SCHEMA = SCHEMAS / "susc_17c2_candidate_patch_link_schema_v1.json"

# --- inputs (read-only) ---------------------------------------------------- #
PRIORITY_17C = OUT / "susc_17c_priority_canary_events.csv"
SAR_FEAS_17C = OUT / "susc_17c_sar_feasibility_pack.csv"
TASK_PLAN_17C = OUT / "susc_17c_sentinel1_task_plan.json"
EVENTS_13A = DAT / "susc_13a_strong_observed_events_parsed_v1.csv"
WINDOW_PLAN = DAT / "susc_16a_sentinel_event_window_plan_v1.csv"
FEATURES = DAT / "susc_features_by_patch_v1.csv"
SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"

GEE_STUB = "scripts/suscetibilidade/susc_16a_gee_sentinel1_flood_mapping_stub.js"
STAC_STUB = "scripts/suscetibilidade/susc_16a_stac_sentinel1_query_stub.py"

# --- outputs --------------------------------------------------------------- #
EXEC_REGISTRY = OUT / "susc_17c2_canary_execution_registry.csv"
QUERY_PLAN = OUT / "susc_17c2_sentinel1_query_plan.csv"
PROCESSING_PLAN = OUT / "susc_17c2_sar_processing_plan.csv"
FOOTPRINT_REGISTRY = OUT / "susc_17c2_candidate_footprint_registry.csv"
FOOTPRINT_GEOJSON = OUT / "susc_17c2_candidate_footprints.geojson"
PATCH_LINKS = OUT / "susc_17c2_candidate_patch_links.csv"
QA_UPDATE = OUT / "susc_17c2_human_qa_update.csv"
SUMMARY = OUT / "susc_17c2_execution_summary.json"
REPORT = OUT / "SUSC_17C2_SENTINEL1_SAR_FOOTPRINT_EXECUTION_REPORT.md"

REQUIRED_INPUTS = [AUDIT, FOOTPRINT_SCHEMA, PATCH_LINK_SCHEMA, PRIORITY_17C,
                   SAR_FEAS_17C, EVENTS_13A, WINDOW_PLAN, FEATURES, SCORE_V6]

MANDATORY = (
    "O SUSC-17C2 executa de forma controlada e review-only os canaries SAR do "
    "SUSC-17C. Footprint SAR candidato e evidencia observacional candidata, nunca "
    "ground truth, nunca label, nunca score; so vira forte com QA humano. O 17C2 "
    "nao altera o score v6 oficial, nao cria score v7, nao cria treino, modelo ou "
    "ground truth, nao baixa raster bruto, nao inventa geometria e nao executa o "
    "benchmark 17B."
)

CHARTER_EVENT_ID = "S17C_E_SUSC13A_00001"
DECISION_VALUES = (
    "accepted;rejected;ambiguous;needs_more_source;phenomenon_mismatch;"
    "temporal_mismatch;spatial_mismatch;processing_artifact"
)
NOT_GT = "technical_sar_or_official_flood_footprint_candidate_is_review_evidence_not_ground_truth"


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _require_inputs() -> None:
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(str(p.relative_to(ROOT)) for p in missing))


def _na(value) -> str:
    value = ("" if value is None else str(value)).strip()
    return value if value else "not_available"


def _sar_runtime_available() -> bool:
    """Deterministic, offline-first: real SAR needs credentials AND explicit opt-in."""
    if os.environ.get("SUSC_17C2_SAR_RUNTIME") != "1":
        return False
    return bool(os.environ.get("EARTHENGINE_TOKEN") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))


def _parse_bbox(bbox: str) -> tuple[float, float, float, float] | None:
    parts = [p for p in (bbox or "").replace(",", ";").split(";") if p.strip()]
    try:
        vals = [float(p) for p in parts[:4]]
    except (ValueError, TypeError):
        return None
    if len(vals) != 4:
        return None
    lon_min, lat_min, lon_max, lat_max = vals
    if lon_max < lon_min or lat_max < lat_min:
        return None
    return lon_min, lat_min, lon_max, lat_max


# --------------------------------------------------------------------------- #
# context loaders
# --------------------------------------------------------------------------- #

def _load_context() -> dict:
    priority = read_csv(PRIORITY_17C)
    sar_by_id = {r["candidate_event_id"]: r for r in read_csv(SAR_FEAS_17C)}
    charter = next((r for r in read_csv(EVENTS_13A) if r.get("event_id") == "SUSC13A_00001"), {})
    windows_by_id = {r["event_window_id"]: r for r in read_csv(WINDOW_PLAN)}
    return {"priority": priority, "sar": sar_by_id, "charter": charter, "windows": windows_by_id}


def _aoi_for(canary: dict, ctx: dict) -> tuple[str, str]:
    """Returns (aoi_source, aoi_bbox) using only real committed geometry."""
    eid = canary["candidate_event_id"]
    if eid == CHARTER_EVENT_ID:
        bbox = (ctx["charter"].get("bbox") or "").strip()
        return ("susc_13a_official_international_charter_bbox", bbox or "not_available")
    win_id = eid.replace("S17C_W_", "")
    win = ctx["windows"].get(win_id, {})
    return ("susc_16a_sentinel_event_window_plan_aoi_bbox", (win.get("aoi_bbox") or "").strip() or "not_available")


# --------------------------------------------------------------------------- #
# 1. canary execution registry
# --------------------------------------------------------------------------- #

EXEC_FIELDS = [
    "execution_id", "candidate_event_id", "city", "event_date", "pre_event_window",
    "post_event_window", "aoi_source", "source_id", "execution_mode",
    "execution_status", "blocking_reason", "review_only", "trainable", "ground_truth",
]


def _exec_rows(ctx: dict) -> list[dict]:
    runtime = _sar_runtime_available()
    rows = []
    for canary in ctx["priority"]:
        eid = canary["candidate_event_id"]
        sar = ctx["sar"].get(eid, {})
        aoi_source, aoi_bbox = _aoi_for(canary, ctx)
        has_aoi = _parse_bbox(aoi_bbox) is not None
        if eid == CHARTER_EVENT_ID:
            # official geometry review path (rule 20): real bbox materialized.
            mode, status, blocking = "local_existing_artifact", "executed_candidate_created", "not_available"
            if not has_aoi:
                status, blocking = "blocked_missing_aoi", "official_geometry_bbox_unavailable"
        else:
            # SAR path: needs runtime; otherwise blocked honestly.
            mode = "gee_task_spec"
            if not has_aoi:
                status, blocking = "blocked_missing_aoi", "sentinel_window_aoi_unavailable"
            elif not runtime:
                status, blocking = "blocked_no_runtime_access", "no_gee_or_stac_credential_and_no_opt_in"
            else:
                status, blocking = "planned_ready", "not_available"
        rows.append({
            "execution_id": f"EXEC_{eid}",
            "candidate_event_id": eid,
            "city": canary.get("city", ""),
            "event_date": canary.get("event_date", ""),
            "pre_event_window": _na(sar.get("pre_event_window")),
            "post_event_window": _na(sar.get("post_event_window")),
            "aoi_source": aoi_source,
            "source_id": canary.get("source_id", ""),
            "execution_mode": mode,
            "execution_status": status,
            "blocking_reason": blocking,
            "review_only": "true", "trainable": "false", "ground_truth": "false",
        })
    rows.sort(key=lambda r: r["execution_id"])
    return rows


# --------------------------------------------------------------------------- #
# 2. sentinel-1 query plan + 3. SAR processing plan
# --------------------------------------------------------------------------- #

QUERY_FIELDS = [
    "execution_id", "candidate_event_id", "collection", "platform", "instrument",
    "product_type", "polarizations", "orbit_pass_policy", "pre_window_start",
    "pre_window_end", "post_window_start", "post_window_end", "aoi_bbox",
    "query_status", "query_blocking_reason",
]

PROC_FIELDS = [
    "execution_id", "candidate_event_id", "method", "speckle_filter_policy",
    "change_detection_method", "permanent_water_mask_policy", "hand_mask_policy",
    "slope_mask_policy", "urban_ambiguity_guard", "polygonization_policy",
    "min_area_policy", "qa_required",
]


def _split_window(window: str) -> tuple[str, str]:
    if window and window != "not_available" and "/" in window:
        a, b = window.split("/", 1)
        return a, b
    return "not_available", "not_available"


def build_query_and_processing(ctx: dict, exec_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    queries, procs = [], []
    for ex in exec_rows:
        eid = ex["candidate_event_id"]
        _, aoi_bbox = _aoi_for({"candidate_event_id": eid}, ctx)
        pre_s, pre_e = _split_window(ex["pre_event_window"])
        post_s, post_e = _split_window(ex["post_event_window"])
        if eid == CHARTER_EVENT_ID:
            q_status, q_block = "not_required_official_geometry_path", "official_geometry_review_path_no_sar_required"
        elif ex["execution_status"] == "planned_ready":
            q_status, q_block = "planned_ready", "not_available"
        elif ex["execution_status"] == "blocked_no_runtime_access":
            q_status, q_block = "blocked_no_runtime_access", "no_gee_or_stac_credential_and_no_opt_in"
        else:
            q_status, q_block = "blocked", ex["blocking_reason"]
        queries.append({
            "execution_id": ex["execution_id"], "candidate_event_id": eid,
            "collection": "COPERNICUS/S1_GRD", "platform": "Sentinel-1",
            "instrument": "C-SAR", "product_type": "GRD",
            "polarizations": "VV;VH",
            "orbit_pass_policy": "prefer_same_relative_orbit_and_pass_for_pre_post_pairing",
            "pre_window_start": pre_s, "pre_window_end": pre_e,
            "post_window_start": post_s, "post_window_end": post_e,
            "aoi_bbox": _na(aoi_bbox), "query_status": q_status,
            "query_blocking_reason": q_block,
        })
        procs.append({
            "execution_id": ex["execution_id"], "candidate_event_id": eid,
            "method": "pre_post_event_sentinel1_change_detection"
            if eid != CHARTER_EVENT_ID else "official_charter_flood_extent_review",
            "speckle_filter_policy": "refined_lee_or_gamma_map_before_ratio",
            "change_detection_method": "log_ratio_VV_then_VH_confirmation_pre_baseline_vs_post_event",
            "permanent_water_mask_policy": "exclude_jrc_global_surface_water_permanent",
            "hand_mask_policy": "exclude_high_HAND_above_threshold",
            "slope_mask_policy": "exclude_steep_slope_above_threshold",
            "urban_ambiguity_guard": "flag_layover_shadow_and_double_bounce_urban_zones",
            "polygonization_policy": "vectorize_connected_components_review_only",
            "min_area_policy": "drop_below_min_mapping_unit_then_human_qa",
            "qa_required": "true",
        })
    queries.sort(key=lambda r: r["execution_id"])
    procs.sort(key=lambda r: r["execution_id"])
    return queries, procs


# --------------------------------------------------------------------------- #
# 4/5. candidate footprint registry + geojson
# --------------------------------------------------------------------------- #

FOOTPRINT_FIELDS = [
    "candidate_footprint_id", "candidate_event_id", "execution_id", "event_date",
    "geometry_type", "geometry_source", "evidence_class", "source_authority",
    "annotation_method", "uncertainty_m", "spatial_resolution", "temporal_resolution",
    "pre_event_window", "post_event_window", "qa_status", "eligible_for_reference_registry",
    "eligible_for_strong_evaluation", "eligible_for_calibration", "eligible_for_17b",
    "review_only", "trainable", "ground_truth", "not_ground_truth_reason",
]


def _bbox_polygon(bbox: tuple[float, float, float, float]) -> list:
    lon_min, lat_min, lon_max, lat_max = bbox
    return [[
        [lon_min, lat_min], [lon_max, lat_min], [lon_max, lat_max],
        [lon_min, lat_max], [lon_min, lat_min],
    ]]


def build_footprints(ctx: dict, exec_rows: list[dict]) -> tuple[list[dict], dict]:
    rows = []
    features = []
    for ex in exec_rows:
        if ex["execution_status"] != "executed_candidate_created":
            continue  # blocked executions produce NO footprint (no invention)
        eid = ex["candidate_event_id"]
        _, aoi_bbox = _aoi_for({"candidate_event_id": eid}, ctx)
        bbox = _parse_bbox(aoi_bbox)
        if bbox is None:
            continue
        fid = f"S17C2_FP_{eid}"
        charter = ctx["charter"]
        period_start = (charter.get("event_period_start") or charter.get("event_date") or "").strip()
        period_end = (charter.get("event_period_end") or "").strip()
        temporal_resolution = (f"date_range_{period_start}_to_{period_end}"
                               if period_start and period_end else "date_range_period")
        row = {
            "candidate_footprint_id": fid,
            "candidate_event_id": eid,
            "execution_id": ex["execution_id"],
            "event_date": ex["event_date"],
            "geometry_type": "Polygon",
            "geometry_source": "susc_13a_official_international_charter_bbox",
            "evidence_class": "technical_remote_sensing_flood_footprint_candidate",
            "source_authority": "international_charter_space_and_major_disasters_official_remote_sensing",
            "annotation_method": "official_charter_flood_extent_bbox_materialized_no_raster",
            "uncertainty_m": "not_available",
            "spatial_resolution": "coarse_bbox_polygon_neighborhood_scale",
            "temporal_resolution": temporal_resolution,
            "pre_event_window": ex["pre_event_window"],
            "post_event_window": ex["post_event_window"],
            "qa_status": "needs_review",
            "eligible_for_reference_registry": "false",
            "eligible_for_strong_evaluation": "false",
            "eligible_for_calibration": "false",
            "eligible_for_17b": "false",
            "review_only": "true", "trainable": "false", "ground_truth": "false",
            "not_ground_truth_reason": NOT_GT,
        }
        rows.append(row)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": _bbox_polygon(bbox)},
            "properties": {
                "candidate_footprint_id": fid,
                "candidate_event_id": eid,
                "event_date": ex["event_date"],
                "evidence_class": "technical_remote_sensing_flood_footprint_candidate",
                "qa_status": "needs_review",
                "ground_truth": False,
                "review_only": True,
                "annotation_method": row["annotation_method"],
                "uncertainty_m": "not_available",
                "not_ground_truth_reason": NOT_GT,
            },
        })
    rows.sort(key=lambda r: r["candidate_footprint_id"])
    features.sort(key=lambda f: f["properties"]["candidate_footprint_id"])
    blocked = [ex for ex in exec_rows if ex["execution_status"] != "executed_candidate_created"]
    geojson = {
        "type": "FeatureCollection",
        "metadata": {
            "protocol": "susc_17c2_candidate_footprints",
            "review_only": True,
            "ground_truth": False,
            "footprint_count": len(features),
            "note": (
                "Footprints candidatos review-only. Nenhum raster bruto. Execucoes "
                "bloqueadas nao geram geometria." if features else
                "Nenhum footprint candidato produzido: execucoes SAR bloqueadas por falta "
                "de runtime e sem geometria oficial materializavel adicional."),
            "blocked_executions": [ex["execution_id"] for ex in blocked],
        },
        "features": features,
    }
    return rows, geojson


# --------------------------------------------------------------------------- #
# 6. candidate patch links (real bbox intersection, never invented)
# --------------------------------------------------------------------------- #

PATCH_LINK_FIELDS = [
    "candidate_patch_link_id", "candidate_footprint_id", "candidate_event_id",
    "patch_id", "link_method", "overlap_ratio", "distance_m", "link_quality",
    "qa_status", "eligible_for_17b", "review_only", "trainable", "ground_truth",
]


def _patch_bbox(row: dict) -> tuple[float, float, float, float] | None:
    try:
        return (float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"]))
    except (KeyError, ValueError, TypeError):
        return None


def _overlap_ratio(fb, pb) -> float:
    ix0, iy0 = max(fb[0], pb[0]), max(fb[1], pb[1])
    ix1, iy1 = min(fb[2], pb[2]), min(fb[3], pb[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = (pb[2] - pb[0]) * (pb[3] - pb[1])
    return inter / area if area > 0 else 0.0


def build_patch_links(ctx: dict, footprints: list[dict]) -> list[dict]:
    if not footprints:
        return []
    patches = read_csv(FEATURES)
    rows = []
    for fp in footprints:
        eid = fp["candidate_event_id"]
        _, aoi_bbox = _aoi_for({"candidate_event_id": eid}, ctx)
        fb = _parse_bbox(aoi_bbox)
        if fb is None:
            continue
        for p in patches:
            pb = _patch_bbox(p)
            if pb is None:
                continue
            ratio = _overlap_ratio(fb, pb)
            if ratio <= 0:
                continue
            quality = ("high_spatial_overlap" if ratio >= 0.5
                       else "moderate_spatial_overlap" if ratio >= 0.1
                       else "low_spatial_overlap")
            rows.append({
                "candidate_patch_link_id": f"S17C2_PL_{fp['candidate_footprint_id']}_{p['patch_id']}",
                "candidate_footprint_id": fp["candidate_footprint_id"],
                "candidate_event_id": eid,
                "patch_id": p["patch_id"],
                "link_method": "footprint_bbox_vs_patch_bbox_intersection",
                "overlap_ratio": f"{ratio:.6f}",
                "distance_m": "not_available",
                "link_quality": quality,
                "qa_status": "needs_review",
                "eligible_for_17b": "false",
                "review_only": "true", "trainable": "false", "ground_truth": "false",
            })
    rows.sort(key=lambda r: r["candidate_patch_link_id"])
    return rows


# --------------------------------------------------------------------------- #
# 7. human QA update
# --------------------------------------------------------------------------- #

QA_FIELDS = [
    "qa_item_id", "candidate_event_id", "candidate_footprint_id", "qa_question",
    "expected_decision_values", "default_status", "blocking_if_unreviewed",
    "qa_artifact_to_review", "review_only", "trainable", "ground_truth",
]


def build_qa_update(exec_rows: list[dict], footprints: list[dict]) -> list[dict]:
    fp_by_event = {f["candidate_event_id"]: f for f in footprints}
    rows = []
    for ex in exec_rows:
        eid = ex["candidate_event_id"]
        fp = fp_by_event.get(eid)
        if fp is not None:
            question = ("O footprint candidato (geometria oficial/SAR) representa uma inundacao "
                        "observada real na data e local, sem ser artefato de processamento?")
            artifact = _rel(FOOTPRINT_GEOJSON)
            fid = fp["candidate_footprint_id"]
        else:
            question = ("A execucao ficou bloqueada (sem runtime SAR). Decidir proxima fonte/caminho "
                        "de aquisicao (runtime SAR, geometria oficial alternativa ou rejeicao).")
            artifact = _rel(EXEC_REGISTRY)
            fid = "not_available"
        rows.append({
            "qa_item_id": f"QA17C2_{eid}",
            "candidate_event_id": eid,
            "candidate_footprint_id": fid,
            "qa_question": question,
            "expected_decision_values": DECISION_VALUES,
            "default_status": "needs_review",
            "blocking_if_unreviewed": "true",
            "qa_artifact_to_review": artifact,
            "review_only": "true", "trainable": "false", "ground_truth": "false",
        })
    rows.sort(key=lambda r: r["qa_item_id"])
    return rows


# --------------------------------------------------------------------------- #
# 8. summary + report
# --------------------------------------------------------------------------- #

def _readiness(footprints, exec_rows, sar_runtime) -> str:
    if footprints:
        return "FOOTPRINT_CANDIDATES_CREATED_NEEDS_QA"
    if any(ex["execution_status"] == "planned_ready" for ex in exec_rows):
        return "SAR_TASKS_READY_NEEDS_EXECUTION"
    if any(ex["execution_status"] == "blocked_missing_aoi" for ex in exec_rows):
        return "BLOCKED_NO_VALID_AOI"
    if not sar_runtime:
        return "BLOCKED_NO_RUNTIME_ACCESS"
    return "BLOCKED_NO_FOOTPRINTS_CREATED"


def build_summary(exec_rows, footprints, patch_links, qa_rows) -> int:
    status_counts = Counter(ex["execution_status"] for ex in exec_rows)
    official_geometry_review = sum(
        1 for ex in exec_rows
        if ex["candidate_event_id"] == CHARTER_EVENT_ID
        and ex["execution_status"] == "executed_candidate_created")
    # technical footprints created via actual SAR change-detection execution (0 offline).
    technical_footprint_created = sum(
        1 for f in footprints if f["geometry_source"].startswith("sar_change_detection"))
    blocks_17b = ["no_qa_accepted_yet"]
    if not footprints:
        blocks_17b.append("no_candidate_footprint_created")
    if not patch_links:
        blocks_17b.append("no_candidate_patch_link_charter_footprint_outside_patch_grid")
    if technical_footprint_created == 0:
        blocks_17b.append("no_sar_footprint_executed_no_runtime")
    blocks_17b.append("strong_references_still_concentrated_in_recife")
    summary = {
        "total_canaries": len(exec_rows),
        "executed_candidate_created_count": status_counts.get("executed_candidate_created", 0),
        "planned_ready_count": status_counts.get("planned_ready", 0),
        "blocked_count": sum(v for k, v in status_counts.items() if k.startswith("blocked")),
        "technical_footprint_created_count": technical_footprint_created,
        "official_geometry_review_count": official_geometry_review,
        "candidate_footprint_count": len(footprints),
        "candidate_patch_link_count": len(patch_links),
        "qa_items_count": len(qa_rows),
        "qa_accepted_count": 0,
        "eligible_for_17b_now": False,
        "blocks_17b": blocks_17b,
        "sar_runtime_available": _sar_runtime_available(),
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "score_v6_changed": False,
        "score_v7_created": False,
        "readiness_status": _readiness(footprints, exec_rows, _sar_runtime_available()),
        "next_recommended_milestone": _next_milestone(footprints, exec_rows),
    }
    write_json(SUMMARY, summary)
    print(f"summary -> {_rel(SUMMARY)}")
    return 0


def _next_milestone(footprints, exec_rows) -> str:
    if footprints:
        return ("SUSC-17D Human QA Protocol para revisar os footprints/geometria oficial "
                "candidatos; em paralelo SUSC-17C3 SAR Runtime Integration para destravar "
                "os canaries SAR bloqueados por falta de runtime")
    if any(ex["execution_status"] == "blocked_no_runtime_access" for ex in exec_rows):
        return "SUSC-17C3 SAR Runtime Integration (tudo planejado, falta runtime GEE/STAC)"
    return "SUSC-17C3 Official Geometry Review"


def build_report(exec_rows, footprints) -> int:
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    exec_lines = "\n".join(
        f"- `{ex['candidate_event_id']}` ({ex['city']}, {ex['event_date']}): "
        f"`{ex['execution_mode']}` -> `{ex['execution_status']}`"
        + (f" ({ex['blocking_reason']})" if ex["blocking_reason"] != "not_available" else "")
        for ex in exec_rows)
    fp_line = (footprints[0]["candidate_footprint_id"] if footprints else "nenhum")
    report = f"""# SUSC-17C2 - Sentinel-1/SAR Footprint Execution (review-only)

Status: review-only. `trainable=false`; `ground_truth=false`; `score_v6_changed=false`; `score_v7_created=false`; `technical_footprint_created_count={s['technical_footprint_created_count']}`; `qa_accepted_count=0`; `readiness_status={s['readiness_status']}`.

{MANDATORY}

## 1. O que o 17C2 tentou executar

Processar os 5 canaries priorizados pelo 17C: 1 via geometria oficial (International Charter) e 4 via Sentinel-1/SAR (janelas Recife 2014).

## 2. Canaries processados ou bloqueados

{exec_lines}

## 3. SAR real foi executado ou so planejado

SAR real NAO foi executado. As bibliotecas GEE/STAC existem no ambiente, mas sem credencial e sem opt-in `SUSC_17C2_SAR_RUNTIME=1`. Os 4 canaries SAR ficam `blocked_no_runtime_access` com `query_plan`/`processing_plan` prontos e compativeis com os stubs `{GEE_STUB}` e `{STAC_STUB}`.

## 4. Algum footprint candidato foi criado

{s['candidate_footprint_count']} footprint candidato: `{fp_line}` (International Charter, caminho de geometria oficial). Materializado a partir do bbox oficial real ja commitado no SUSC-13A, como vetor leve, sem raster e sem invencao. `technical_footprint_created_count={s['technical_footprint_created_count']}` porque nenhuma deteccao de mudanca SAR foi de fato executada.

## 5. Por que footprints candidatos nao sao ground truth

Sao geometrias candidatas (oficiais coarse ou, no futuro, derivadas de SAR), sem confirmacao QA, sem validacao patch-a-patch independente, com incerteza nao quantificada. `evidence_class=technical_remote_sensing_flood_footprint_candidate`, `qa_status=needs_review`, `ground_truth=false`, `not_ground_truth_reason` preenchido.

## 6. Por que o QA humano ainda bloqueia o 17B

Nenhum footprint foi aceito (`qa_accepted_count=0`). `eligible_for_strong_evaluation`, `eligible_for_calibration` e `eligible_for_17b` permanecem `false` ate QA `accepted`. Alem disso o footprint do International Charter cai fora da grade de patches atual ({s['candidate_patch_link_count']} patch-links), entao nao alimenta o 17B mesmo apos QA sem ajuste de cobertura.

## 7. Como usar cada output no proximo marco

- `canary_execution_registry`: rastreia o estado de cada canary.
- `sentinel1_query_plan` + `sar_processing_plan`: specs prontas para o `SUSC-17C3 SAR Runtime Integration`.
- `candidate_footprint_registry` + `candidate_footprints.geojson`: entram no `SUSC-17D Human QA Protocol`.
- `candidate_patch_links`: vazio aqui (cobertura), mas e o gancho para o 17B quando houver footprint dentro da grade.
- `human_qa_update`: fila de decisoes humanas.

## 8. Readiness

- QA humano: PRONTO para revisar 1 footprint candidato e {s['blocked_count']} bloqueios SAR.
- 17B benchmark: BLOQUEADO ({json.dumps(s['blocks_17b'], ensure_ascii=False)}).
- Expansao regional: ainda concentrado em Recife; depende de novos canaries datados em Curitiba/Petropolis.

## 9. Proximo marco recomendado

{s['next_recommended_milestone']}.
"""
    write_markdown(REPORT, report)
    print(f"report -> {_rel(REPORT)}")
    return 0


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

def run_all() -> int:
    _require_inputs()
    ctx = _load_context()
    exec_rows = _exec_rows(ctx)
    write_csv(EXEC_REGISTRY, exec_rows, EXEC_FIELDS)
    queries, procs = build_query_and_processing(ctx, exec_rows)
    write_csv(QUERY_PLAN, queries, QUERY_FIELDS)
    write_csv(PROCESSING_PLAN, procs, PROC_FIELDS)
    footprints, geojson = build_footprints(ctx, exec_rows)
    write_csv(FOOTPRINT_REGISTRY, footprints, FOOTPRINT_FIELDS)
    write_json(FOOTPRINT_GEOJSON, geojson)
    patch_links = build_patch_links(ctx, footprints)
    write_csv(PATCH_LINKS, patch_links, PATCH_LINK_FIELDS)
    qa_rows = build_qa_update(exec_rows, footprints)
    write_csv(QA_UPDATE, qa_rows, QA_FIELDS)
    build_summary(exec_rows, footprints, patch_links, qa_rows)
    build_report(exec_rows, footprints)
    print(f"exec -> {_rel(EXEC_REGISTRY)} ({len(exec_rows)} canaries); "
          f"footprints={len(footprints)}; patch_links={len(patch_links)}; qa={len(qa_rows)}")
    return 0


# --------------------------------------------------------------------------- #
# validation (fail-closed)
# --------------------------------------------------------------------------- #

def schema_violations(row: dict, schema: dict) -> list[str]:
    out = []
    props = schema.get("properties", {})
    for field in schema.get("required", []):
        if not str(row.get(field, "")).strip():
            out.append(f"required empty: {field}")
    for field, spec in props.items():
        if field not in row:
            continue
        value = row.get(field, "")
        if "const" in spec and value != spec["const"]:
            out.append(f"{field}={value} must be {spec['const']}")
        if "enum" in spec and value not in spec["enum"]:
            out.append(f"{field}={value} not in enum")
    return out


def footprint_row_violations(row: dict) -> list[str]:
    """Fail-closed guard for a candidate footprint (used by validator + tests)."""
    out = []
    if row.get("review_only") != "true":
        out.append("review_only not true")
    if row.get("trainable") != "false":
        out.append("trainable true")
    if row.get("ground_truth") != "false":
        out.append("ground_truth true")
    if row.get("score_v7_created", "false") == "true":
        out.append("score_v7_created true")
    if row.get("official_score_changed", "false") == "true":
        out.append("official_score_changed true")
    accepted = row.get("qa_status") == "accepted"
    if row.get("eligible_for_strong_evaluation") == "true" and not accepted:
        out.append("strong eligible before QA accepted")
    if row.get("eligible_for_calibration") == "true" and not accepted:
        out.append("calibration eligible before QA accepted")
    if row.get("eligible_for_17b") == "true" and not accepted:
        out.append("17b eligible before QA accepted")
    if not str(row.get("not_ground_truth_reason", "")).strip():
        out.append("missing not_ground_truth_reason")
    if row.get("evidence_class") != "technical_remote_sensing_flood_footprint_candidate":
        out.append("evidence_class not candidate")
    return out


def validate() -> int:
    errors: list[str] = []
    required_files = [
        AUDIT, FOOTPRINT_SCHEMA, PATCH_LINK_SCHEMA, EXEC_REGISTRY, QUERY_PLAN,
        PROCESSING_PLAN, FOOTPRINT_REGISTRY, FOOTPRINT_GEOJSON, PATCH_LINKS,
        QA_UPDATE, SUMMARY, REPORT,
    ]
    for path in required_files:
        if not path.exists():
            errors.append(f"missing: {path.relative_to(ROOT)}")
    if errors:
        print("FAILED: SUSC-17C2 validation")
        for error in errors:
            print("  ERROR:", error)
        return 1

    fp_schema = json.loads(FOOTPRINT_SCHEMA.read_text(encoding="utf-8"))
    pl_schema = json.loads(PATCH_LINK_SCHEMA.read_text(encoding="utf-8"))
    exec_rows = read_csv(EXEC_REGISTRY)
    footprints = read_csv(FOOTPRINT_REGISTRY)
    patch_links = read_csv(PATCH_LINKS)
    qa_rows = read_csv(QA_UPDATE)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    geojson = json.loads(FOOTPRINT_GEOJSON.read_text(encoding="utf-8"))

    # unique + deterministic IDs across all registries
    for label, rows, key in (
        ("exec", exec_rows, "execution_id"),
        ("footprint", footprints, "candidate_footprint_id"),
        ("patch_link", patch_links, "candidate_patch_link_id"),
        ("qa", qa_rows, "qa_item_id"),
    ):
        ids = [r[key] for r in rows]
        if len(ids) != len(set(ids)):
            errors.append(f"{label}: ids not unique")
        if ids != sorted(ids):
            errors.append(f"{label}: ids not sorted (non-deterministic)")

    # every prioritized canary appears in execution registry (rule 19)
    priority_ids = {r["candidate_event_id"] for r in read_csv(PRIORITY_17C)}
    exec_ids = {r["candidate_event_id"] for r in exec_rows}
    if priority_ids - exec_ids:
        errors.append(f"missing canaries in exec registry: {sorted(priority_ids - exec_ids)}")

    # International Charter via official geometry review path, not forced SAR (rule 20)
    charter = next((r for r in exec_rows if r["candidate_event_id"] == CHARTER_EVENT_ID), {})
    if charter and charter.get("execution_mode") not in ("local_existing_artifact", "planned_not_executed"):
        errors.append("International Charter not on official-geometry review path")

    # governance + schema on footprints
    for line_no, row in enumerate(footprints, start=2):
        for v in schema_violations(row, fp_schema):
            errors.append(f"{FOOTPRINT_REGISTRY.name}:{line_no}: {v}")
        for v in footprint_row_violations(row):
            errors.append(f"{FOOTPRINT_REGISTRY.name}:{line_no}: {v}")

    for line_no, row in enumerate(patch_links, start=2):
        for v in schema_violations(row, pl_schema):
            errors.append(f"{PATCH_LINKS.name}:{line_no}: {v}")
        if row.get("eligible_for_17b") != "false":
            errors.append(f"{PATCH_LINKS.name}:{line_no}: eligible_for_17b not false")

    # executed_candidate_created <=> footprint in registry + geojson (rules 15/16)
    created_events = {ex["candidate_event_id"] for ex in exec_rows
                      if ex["execution_status"] == "executed_candidate_created"}
    fp_events = {f["candidate_event_id"] for f in footprints}
    if created_events != fp_events:
        errors.append(f"executed_candidate_created mismatch footprints: {created_events} vs {fp_events}")
    geojson_ids = {f["properties"]["candidate_footprint_id"] for f in geojson.get("features", [])}
    registry_ids = {f["candidate_footprint_id"] for f in footprints}
    if geojson_ids != registry_ids:
        errors.append("geojson features and footprint registry disagree")
    # blocked executions invent no geometry (rule 16/22)
    for ex in exec_rows:
        if ex["execution_status"].startswith("blocked") and ex["candidate_event_id"] in fp_events:
            errors.append(f"blocked execution produced a footprint: {ex['candidate_event_id']}")

    # geojson features fail-closed properties
    for feat in geojson.get("features", []):
        props = feat.get("properties", {})
        if props.get("ground_truth") is not False or props.get("review_only") is not True:
            errors.append("geojson feature governance wrong")
        if props.get("qa_status") != "needs_review":
            errors.append("geojson feature qa_status not needs_review")
        if not feat.get("geometry", {}).get("coordinates"):
            errors.append("geojson feature without coordinates")

    # every footprint has a QA item (rule 18); every blocked canary too has one
    qa_events = {q["candidate_event_id"] for q in qa_rows}
    for f in footprints:
        if f["candidate_event_id"] not in qa_events:
            errors.append(f"footprint {f['candidate_footprint_id']} has no QA item")
    for q in qa_rows:
        if q.get("default_status") != "needs_review":
            errors.append("qa default_status not needs_review")
        if q.get("ground_truth") != "false":
            errors.append("qa ground_truth true")

    # blocked executions must carry a blocking_reason (rule: todo bloqueio tem razao)
    for ex in exec_rows:
        if ex["execution_status"].startswith("blocked") and ex["blocking_reason"] in ("", "not_available"):
            errors.append(f"blocked execution without reason: {ex['execution_id']}")
        if ex.get("ground_truth") != "false" or ex.get("review_only") != "true":
            errors.append(f"exec governance wrong: {ex['execution_id']}")

    # summary governance + honesty
    if summary.get("score_v7_created") is not False:
        errors.append("summary score_v7_created true")
    if summary.get("score_v6_changed") is not False:
        errors.append("summary score_v6_changed true")
    if summary.get("trainable") is not False or summary.get("ground_truth") is not False:
        errors.append("summary trainable/ground_truth wrong")
    if summary.get("eligible_for_17b_now") is not False:
        errors.append("summary eligible_for_17b_now true")
    if summary.get("qa_accepted_count") not in (0, "0"):
        errors.append("summary qa_accepted_count != 0")
    if summary.get("candidate_footprint_count") != len(footprints):
        errors.append("summary footprint count mismatch")
    if summary.get("candidate_patch_link_count") != len(patch_links):
        errors.append("summary patch link count mismatch")
    if summary.get("qa_items_count") != len(qa_rows):
        errors.append("summary qa items mismatch")
    if not footprints and summary.get("technical_footprint_created_count") not in (0, "0"):
        errors.append("empty geojson but technical_footprint_created_count != 0")  # rule 17
    if summary.get("readiness_status") not in (
            "FOOTPRINT_CANDIDATES_CREATED_NEEDS_QA", "SAR_TASKS_READY_NEEDS_EXECUTION",
            "BLOCKED_NO_RUNTIME_ACCESS", "BLOCKED_NO_VALID_AOI",
            "BLOCKED_NO_FOOTPRINTS_CREATED", "OFFICIAL_GEOMETRY_REVIEW_READY"):
        errors.append("summary readiness_status invalid")

    # no heavy raster artifacts created/staged (rule 14)
    for path in OUT.glob("susc_17c2_*"):
        if path.suffix.lower() in (".tif", ".tiff", ".img", ".jp2", ".nc", ".grd", ".npy", ".npz"):
            errors.append(f"heavy raster artifact present: {path.name}")

    # cross guards shared with the SUSC chain
    if SCORE_V7.exists():
        errors.append("score v7 artifact exists")
    if not SCORE_V6.exists():
        errors.append("score v6 official source missing")
    if MANDATORY not in REPORT.read_text(encoding="utf-8"):
        errors.append("mandatory report phrase missing")

    if errors:
        print("FAILED: SUSC-17C2 validation")
        for error in errors:
            print("  ERROR:", error)
        return 1
    print("PASSED: SUSC-17C2 validations passed")
    return 0
