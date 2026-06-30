"""SUSC-17C5 Patch Grid Expansion Review (review-only, fail-closed).

Audits whether the real Charter 758 geometry can inform future patch-level
coverage without silently expanding the official SUSC patch grid, creating
official patches, creating patch-links, changing score v6, creating score v7,
or promoting any candidate evidence into training, labels or ground truth.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import read_csv, read_json, rel, sha256_file, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

AUDIT = OUT / "SUSC_WORKTREE_AUDIT_BEFORE_17C5.md"
FEATURES = DAT / "susc_features_by_patch_v1.csv"
SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"

GEOMS_17C4 = OUT / "susc_17c4_candidate_geometries.geojson"
REFS_17C4 = OUT / "susc_17c4_extracted_reference_candidates.csv"
PATCH_LINKS_17C4 = OUT / "susc_17c4_candidate_patch_links.csv"
SUMMARY_17C4 = OUT / "susc_17c4_summary.json"
COVERAGE_17C3 = OUT / "susc_17c3_patch_coverage_audit.csv"
DECISIONS_17C3 = OUT / "susc_17c3_next_action_decision_table.csv"

GRID_SCHEMA = SCHEMAS / "susc_17c5_patch_grid_inventory_schema_v1.json"
OPTION_SCHEMA = SCHEMAS / "susc_17c5_expansion_option_schema_v1.json"

REPORT = OUT / "SUSC_17C5_PATCH_GRID_EXPANSION_REVIEW_REPORT.md"
PATCH_GRID_INVENTORY = OUT / "susc_17c5_patch_grid_inventory.csv"
COVERAGE_AUDIT = OUT / "susc_17c5_charter758_grid_coverage_audit.csv"
NAMESPACE_AUDIT = OUT / "susc_17c5_protocolc_to_susc_patch_namespace_audit.csv"
AOI_GEOJSON = OUT / "susc_17c5_candidate_expansion_aoi.geojson"
OPTIONS = OUT / "susc_17c5_patch_grid_expansion_options.csv"
RISK_POLICY = OUT / "susc_17c5_expansion_risk_policy.json"
NEXT_ACTIONS = OUT / "susc_17c5_next_action_decision_table.csv"
SUMMARY = OUT / "susc_17c5_summary.json"

REQUIRED_INPUTS = [
    AUDIT, FEATURES, SCORE_V6, GEOMS_17C4, REFS_17C4, PATCH_LINKS_17C4,
    SUMMARY_17C4, COVERAGE_17C3, DECISIONS_17C3, GRID_SCHEMA, OPTION_SCHEMA,
]

EXPECTED_REGIONS = [
    ("Recife", "recife", "SUSC_GRID_RECIFE_V1"),
    ("Curitiba", "curitiba", "SUSC_GRID_CURITIBA_V1"),
    ("Petropolis", "petropolis", "SUSC_GRID_PETROPOLIS_V1"),
]

CHARTER_REF_ID = "S17C4_REF_REC_2022_05_24_30"
CHARTER_EVENT_ID = "REC_2022_05_24_30"
PROTOCOLC_PATCH_ID = "REC_00019"

NOT_OFFICIAL_PATCH_NOTE = (
    "candidate review geometry only; not an official patch, not a footprint, "
    "not a patch-link and not ground truth"
)


def _require_inputs() -> None:
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(p) for p in missing))


def _gov_str() -> dict:
    return {"review_only": "true", "trainable": "false", "ground_truth": "false"}


def _gov_bool() -> dict:
    return {"review_only": True, "trainable": False, "ground_truth": False}


def _na(value) -> str:
    value = "" if value is None else str(value).strip()
    return value if value else "not_available"


def _float(row: dict, key: str):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def _fmt_float(value, digits: int = 6) -> str:
    if value is None:
        return "not_available"
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".")


def _fmt_bbox(bbox) -> str:
    if not bbox:
        return "not_available"
    return ";".join(_fmt_float(v, 6) for v in bbox)


def _bbox_polygon(bbox):
    x0, y0, x1, y1 = bbox
    return {
        "type": "Polygon",
        "coordinates": [[
            [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0],
        ]],
    }


def _iter_coords(obj):
    if isinstance(obj, (list, tuple)):
        if obj and isinstance(obj[0], (int, float)) and len(obj) >= 2:
            yield (float(obj[0]), float(obj[1]))
        else:
            for item in obj:
                yield from _iter_coords(item)


def _geometry_bbox(geom):
    pts = list(_iter_coords(geom.get("coordinates", []))) if geom else []
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _geometry_vertex_count(geom) -> int:
    return sum(1 for _ in _iter_coords(geom.get("coordinates", []))) if geom else 0


def _bbox_intersects(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _bbox_gap_meters(a, b) -> float:
    dx = max(0.0, a[0] - b[2], b[0] - a[2])
    dy = max(0.0, a[1] - b[3], b[1] - a[3])
    lat = (a[1] + a[3]) / 2.0
    mx = dx * 111320.0 * math.cos(math.radians(lat))
    my = dy * 110540.0
    return math.hypot(mx, my)


def _load_charter_feature() -> dict:
    geojson = read_json(GEOMS_17C4)
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        if props.get("reference_candidate_id") == CHARTER_REF_ID:
            return feature
    raise ValueError(f"{CHARTER_REF_ID} not found in {rel(GEOMS_17C4)}")


def _load_charter_reference() -> dict:
    for row in read_csv(REFS_17C4):
        if row.get("reference_candidate_id") == CHARTER_REF_ID:
            return row
    raise ValueError(f"{CHARTER_REF_ID} not found in {rel(REFS_17C4)}")


def _patch_rows() -> list[dict]:
    return read_csv(FEATURES)


def _patch_bbox(row: dict):
    vals = [_float(row, k) for k in ("xmin", "ymin", "xmax", "ymax")]
    if any(v is None for v in vals):
        return None
    return tuple(vals)


def _patch_groups() -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in _patch_rows():
        groups[row.get("regiao", "")].append(row)
    return groups


def _grid_bbox(rows: list[dict]):
    boxes = [_patch_bbox(r) for r in rows]
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _mean_patch_size(rows: list[dict]) -> tuple[float | None, float | None]:
    boxes = [_patch_bbox(r) for r in rows]
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None, None
    widths = [b[2] - b[0] for b in boxes]
    heights = [b[3] - b[1] for b in boxes]
    return sum(widths) / len(widths), sum(heights) / len(heights)


PATCH_GRID_FIELDS = [
    "patch_grid_id", "region", "source_dataset", "patch_id_prefix",
    "patch_count", "crs", "xmin", "ymin", "xmax", "ymax",
    "mean_patch_width_deg", "mean_patch_height_deg", "has_patch_bbox",
    "has_patch_geometry", "grid_status", "review_only", "trainable",
    "ground_truth",
]


def build_patch_grid_inventory() -> list[dict]:
    groups = _patch_groups()
    rows = []
    for region_name, key, grid_id in EXPECTED_REGIONS:
        patches = groups.get(key, [])
        bbox = _grid_bbox(patches)
        mean_w, mean_h = _mean_patch_size(patches)
        prefix = key if patches else "not_available"
        rows.append({
            "patch_grid_id": grid_id,
            "region": region_name,
            "source_dataset": rel(FEATURES),
            "patch_id_prefix": prefix,
            "patch_count": str(len(patches)),
            "crs": "EPSG:4326" if patches else "not_available",
            "xmin": _fmt_float(bbox[0]) if bbox else "not_available",
            "ymin": _fmt_float(bbox[1]) if bbox else "not_available",
            "xmax": _fmt_float(bbox[2]) if bbox else "not_available",
            "ymax": _fmt_float(bbox[3]) if bbox else "not_available",
            "mean_patch_width_deg": _fmt_float(mean_w, 9),
            "mean_patch_height_deg": _fmt_float(mean_h, 9),
            "has_patch_bbox": "true" if bbox else "false",
            "has_patch_geometry": "false",
            "grid_status": "existing_patch_grid" if patches else "missing_patch_grid",
            **_gov_str(),
        })
    rows.sort(key=lambda r: r["patch_grid_id"])
    return rows


COVERAGE_FIELDS = [
    "coverage_audit_id", "reference_candidate_id", "event_id", "geometry_source",
    "geometry_type", "geometry_bbox", "geometry_crs", "target_patch_grid_id",
    "target_region", "intersecting_patch_count", "nearest_patch_id",
    "nearest_patch_distance_m", "nearest_grid_edge_distance_m", "coverage_status",
    "blocking_reason", "recommended_action", "review_only", "trainable",
    "ground_truth",
]


def build_charter_coverage_audit(grid_inventory: list[dict]) -> list[dict]:
    feature = _load_charter_feature()
    ref = _load_charter_reference()
    geom = feature.get("geometry", {})
    fb = _geometry_bbox(geom)
    recife_grid = [g for g in grid_inventory if g["patch_grid_id"] == "SUSC_GRID_RECIFE_V1"][0]
    recife_patches = [p for p in _patch_groups().get("recife", []) if _patch_bbox(p) is not None]
    inter = []
    distances = []
    for patch in recife_patches:
        pb = _patch_bbox(patch)
        if _bbox_intersects(fb, pb):
            inter.append(patch)
        distances.append((patch.get("patch_id", "not_available"), _bbox_gap_meters(fb, pb)))
    nearest_patch_id, nearest_m = min(distances, key=lambda x: (x[1], x[0])) if distances else ("not_available", None)
    grid_bbox = tuple(float(recife_grid[k]) for k in ("xmin", "ymin", "xmax", "ymax")) if recife_patches else None
    grid_edge_m = _bbox_gap_meters(fb, grid_bbox) if grid_bbox else None
    if not recife_patches:
        status = "no_patch_grid"
        reason = "no_susc_recife_patch_grid_available"
        action = "blocked_until_grid_expansion_protocol"
    elif not fb:
        status = "invalid_geometry"
        reason = "charter758_geometry_missing_or_invalid"
        action = "human_qa_reference_geometry_only"
    elif inter:
        status = "intersects_patch_grid"
        reason = "not_blocked_by_grid"
        action = "human_qa_reference_geometry_only"
    elif nearest_m is not None and nearest_m <= 2000.0:
        status = "outside_patch_grid_adjacent"
        reason = "candidate_geometry_outside_current_susc_grid_near_edge"
        action = "prepare_patch_grid_candidate_generation"
    elif nearest_m is not None:
        status = "outside_patch_grid_far"
        reason = "candidate_geometry_outside_current_susc_grid_far_from_edge"
        action = "blocked_until_grid_expansion_protocol"
    else:
        status = "needs_grid_expansion_review"
        reason = "coverage_distance_unavailable"
        action = "blocked_until_grid_expansion_protocol"
    return [{
        "coverage_audit_id": "S17C5_COV_CHARTER758_RECIFE",
        "reference_candidate_id": ref["reference_candidate_id"],
        "event_id": ref["event_id"],
        "geometry_source": ref["geometry_source"],
        "geometry_type": geom.get("type", "not_available"),
        "geometry_bbox": _fmt_bbox(fb),
        "geometry_crs": "EPSG:4326",
        "target_patch_grid_id": recife_grid["patch_grid_id"],
        "target_region": "Recife",
        "intersecting_patch_count": str(len(inter)),
        "nearest_patch_id": nearest_patch_id,
        "nearest_patch_distance_m": f"{nearest_m:.1f}" if nearest_m is not None else "not_available",
        "nearest_grid_edge_distance_m": f"{grid_edge_m:.1f}" if grid_edge_m is not None else "not_available",
        "coverage_status": status,
        "blocking_reason": reason,
        "recommended_action": action,
        **_gov_str(),
    }]


NAMESPACE_FIELDS = [
    "namespace_audit_id", "source_patch_id", "source_namespace",
    "target_namespace", "event_id", "geometry_source", "source_patch_available",
    "target_susc_patch_match", "match_method", "match_status",
    "blocking_reason", "recommended_action",
]


def build_namespace_audit() -> list[dict]:
    protocolc_sources = [
        ROOT / "datasets" / "v2bd_REC_00019_reference_inventory.csv",
        ROOT / "datasets" / "v2bd_REC_00019_patch_boundary_candidate_registry.csv",
        ROOT / "datasets" / "external_sources" / "recife_minimal_tp" / "derived" / "patch_boundary_REC_00019_from_lineage.geojson",
    ]
    source_available = any(p.exists() for p in protocolc_sources)
    susc_patch_ids = {p.get("patch_id", "") for p in _patch_groups().get("recife", [])}
    exact_match = PROTOCOLC_PATCH_ID in susc_patch_ids
    row = {
        "namespace_audit_id": "S17C5_NS_REC_00019_PROTOCOLC_TO_SUSC",
        "source_patch_id": PROTOCOLC_PATCH_ID,
        "source_namespace": "Protocolo_C_REC",
        "target_namespace": "SUSC_recife_lowercase_patch_grid",
        "event_id": CHARTER_EVENT_ID,
        "geometry_source": ";".join(rel(p) for p in protocolc_sources if p.exists()) or "not_available",
        "source_patch_available": "true" if source_available else "false",
        "target_susc_patch_match": "true" if exact_match else "false",
        "match_method": "exact_string_match_only_no_namespace_bridge",
        "match_status": "exact_match" if exact_match else ("namespace_incompatible" if source_available else "source_patch_not_found"),
        "blocking_reason": "not_blocked" if exact_match else "REC_00019_is_not_a_SUSC_recife_patch_id_and_no_bridge_manifest_exists",
        "recommended_action": "do_not_create_automatic_equivalence",
    }
    return [row]


def build_candidate_expansion_aoi(grid_inventory: list[dict], coverage: list[dict]) -> dict:
    feature = _load_charter_feature()
    geom = feature["geometry"]
    fb = _geometry_bbox(geom)
    recife_grid = [g for g in grid_inventory if g["patch_grid_id"] == "SUSC_GRID_RECIFE_V1"][0]
    grid_bbox = tuple(float(recife_grid[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
    envelope = (
        min(fb[0], grid_bbox[0]),
        min(fb[1], grid_bbox[1]),
        max(fb[2], grid_bbox[2]),
        max(fb[3], grid_bbox[3]),
    )
    mean_w = float(recife_grid["mean_patch_width_deg"])
    mean_h = float(recife_grid["mean_patch_height_deg"])
    review_buffer = (
        envelope[0] - mean_w,
        envelope[1] - mean_h,
        envelope[2] + mean_w,
        envelope[3] + mean_h,
    )
    common_props = {
        "event_id": CHARTER_EVENT_ID,
        "review_only": True,
        "ground_truth": False,
        "not_official_patch": True,
        "not_footprint": True,
        "not_patch_link": True,
        "note": NOT_OFFICIAL_PATCH_NOTE,
    }
    features = [
        {
            "type": "Feature",
            "geometry": geom,
            "properties": {
                **common_props,
                "feature_role": "observed_reference_geometry",
                "source": rel(GEOMS_17C4),
                "vertex_count": _geometry_vertex_count(geom),
            },
        },
        {
            "type": "Feature",
            "geometry": _bbox_polygon(grid_bbox),
            "properties": {
                **common_props,
                "feature_role": "current_susc_grid_extent",
                "source": rel(FEATURES),
                "bbox": _fmt_bbox(grid_bbox),
            },
        },
        {
            "type": "Feature",
            "geometry": _bbox_polygon(envelope),
            "properties": {
                **common_props,
                "feature_role": "candidate_expansion_envelope",
                "source": "derived_from_current_susc_grid_extent_and_charter758_bbox",
                "bbox": _fmt_bbox(envelope),
            },
        },
        {
            "type": "Feature",
            "geometry": _bbox_polygon(review_buffer),
            "properties": {
                **common_props,
                "feature_role": "candidate_review_buffer",
                "source": "candidate_envelope_plus_one_mean_recife_patch_step",
                "bbox": _fmt_bbox(review_buffer),
                "buffer_basis": "one_mean_current_susc_recife_patch_width_height",
            },
        },
    ]
    return {
        "type": "FeatureCollection",
        "metadata": {
            "protocol": "susc_17c5_candidate_expansion_aoi",
            "review_only": True,
            "ground_truth": False,
            "not_official_patch": True,
            "not_footprint": True,
            "not_patch_link": True,
            "feature_count": len(features),
            "coverage_status": coverage[0]["coverage_status"],
        },
        "features": features,
    }


OPTION_FIELDS = [
    "expansion_option_id", "option_name", "description", "target_event_id",
    "target_region", "requires_new_patches", "requires_score_recompute",
    "requires_feature_extraction", "requires_sentinel_assets",
    "requires_hydrology_features", "requires_human_qa",
    "methodological_risk", "benefit", "blocking_reason",
    "recommended_next_milestone", "allowed_now", "review_only", "trainable",
    "ground_truth",
]


def build_expansion_options(namespace: list[dict]) -> list[dict]:
    bridge_allowed = namespace[0]["target_susc_patch_match"] == "true"
    rows = [
        {
            "expansion_option_id": "S17C5_OPT_001",
            "option_name": "do_not_expand_context_only",
            "description": "Keep Charter 758 as contextual observed candidate evidence only, without patch-level coverage.",
            "target_event_id": CHARTER_EVENT_ID,
            "target_region": "Recife",
            "requires_new_patches": "false",
            "requires_score_recompute": "false",
            "requires_feature_extraction": "false",
            "requires_sentinel_assets": "false",
            "requires_hydrology_features": "false",
            "requires_human_qa": "true",
            "methodological_risk": "low",
            "benefit": "preserves score and avoids false patch-link",
            "blocking_reason": "does_not_unlock_17b",
            "recommended_next_milestone": "SUSC-17C6 Formal Request Package",
            "allowed_now": "true",
            **_gov_str(),
        },
        {
            "expansion_option_id": "S17C5_OPT_002",
            "option_name": "expand_grid_candidate_review_only",
            "description": "Prepare a future candidate patch-grid generation protocol around the observed geometry and current grid edge.",
            "target_event_id": CHARTER_EVENT_ID,
            "target_region": "Recife",
            "requires_new_patches": "true",
            "requires_score_recompute": "false",
            "requires_feature_extraction": "true",
            "requires_sentinel_assets": "true",
            "requires_hydrology_features": "true",
            "requires_human_qa": "true",
            "methodological_risk": "medium",
            "benefit": "can test future patch-level coverage without using REC_00019 as SUSC patch",
            "blocking_reason": "new_candidate_patches_require_explicit_grid_expansion_protocol_and_manifest",
            "recommended_next_milestone": "SUSC-17C6 Patch Grid Candidate Generation",
            "allowed_now": "false",
            **_gov_str(),
        },
        {
            "expansion_option_id": "S17C5_OPT_003",
            "option_name": "create_protocolc_to_susc_bridge_only",
            "description": "Create only a namespace bridge if a real audited match exists between REC_00019 and a SUSC patch.",
            "target_event_id": CHARTER_EVENT_ID,
            "target_region": "Recife",
            "requires_new_patches": "false",
            "requires_score_recompute": "false",
            "requires_feature_extraction": "false",
            "requires_sentinel_assets": "false",
            "requires_hydrology_features": "false",
            "requires_human_qa": "true",
            "methodological_risk": "high",
            "benefit": "could preserve namespace separation if a real bridge manifest exists",
            "blocking_reason": "no_real_susc_match_for_REC_00019" if not bridge_allowed else "bridge_requires_human_qa_manifest",
            "recommended_next_milestone": "blocked_until_grid_expansion_protocol",
            "allowed_now": "false" if not bridge_allowed else "false",
            **_gov_str(),
        },
        {
            "expansion_option_id": "S17C5_OPT_004",
            "option_name": "acquire_compdec_pkg_before_expansion",
            "description": "Request COMPDEC/DRM official packages before deciding whether grid expansion is necessary.",
            "target_event_id": CHARTER_EVENT_ID,
            "target_region": "Recife",
            "requires_new_patches": "false",
            "requires_score_recompute": "false",
            "requires_feature_extraction": "false",
            "requires_sentinel_assets": "false",
            "requires_hydrology_features": "false",
            "requires_human_qa": "true",
            "methodological_risk": "low",
            "benefit": "may provide official coordinates or package context before any grid expansion",
            "blocking_reason": "human_formal_request_required",
            "recommended_next_milestone": "SUSC-17C6 Formal Request Package",
            "allowed_now": "true",
            **_gov_str(),
        },
        {
            "expansion_option_id": "S17C5_OPT_005",
            "option_name": "run_sar_runtime_on_existing_grid_only",
            "description": "Run SAR only on existing grid cells; this cannot cover Charter 758 while it remains outside the grid.",
            "target_event_id": CHARTER_EVENT_ID,
            "target_region": "Recife",
            "requires_new_patches": "false",
            "requires_score_recompute": "false",
            "requires_feature_extraction": "true",
            "requires_sentinel_assets": "true",
            "requires_hydrology_features": "false",
            "requires_human_qa": "true",
            "methodological_risk": "medium",
            "benefit": "could test existing-grid SAR canary only after runtime exists",
            "blocking_reason": "sar_runtime_unavailable_and_no_patch_intersection_for_charter758",
            "recommended_next_milestone": "SUSC-17C6 SAR Runtime Integration",
            "allowed_now": "false",
            **_gov_str(),
        },
    ]
    rows.sort(key=lambda r: r["expansion_option_id"])
    return rows


def build_risk_policy() -> dict:
    return {
        "policy_id": "SUSC_17C5_EXPANSION_RISK_POLICY_V1",
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "score_v6_changed": False,
        "score_v7_created": False,
        "blocked_actions": {
            "silent_grid_expansion": True,
            "patch_creation_without_manifest": True,
            "patch_link_without_official_patch": True,
            "protocolc_susc_namespace_mixing": True,
            "score_recompute_without_protocol": True,
            "charter_as_ground_truth": True,
            "candidate_aoi_as_strong_evidence": True,
            "benchmark_17b_without_qa_accepted_and_valid_patch_link": True,
        },
        "allowed_actions_now": [
            "review_only_inventory",
            "review_only_candidate_expansion_aoi",
            "formal_request_preparation",
            "human_qa_reference_geometry_only",
        ],
        "required_before_official_grid_change": [
            "explicit_grid_expansion_protocol",
            "candidate_patch_manifest",
            "feature_extraction_plan",
            "hydrology_feature_plan",
            "sentinel_asset_plan",
            "human_qa_acceptance",
            "separate_score_recompute_protocol",
        ],
    }


NEXT_ACTION_FIELDS = [
    "decision_id", "event_id", "decision_scope", "current_status",
    "next_action", "why", "required_inputs", "expected_outputs",
    "can_unlock_17b", "review_only", "trainable", "ground_truth",
]


def build_next_action_decision_table(coverage: list[dict]) -> list[dict]:
    cov = coverage[0]
    rows = [
        {
            "decision_id": "S17C5_DEC_001",
            "event_id": CHARTER_EVENT_ID,
            "decision_scope": "patch_grid_coverage",
            "current_status": cov["coverage_status"],
            "next_action": "prepare_patch_grid_candidate_generation",
            "why": "Charter 758 is real candidate geometry but intersects 0 current SUSC Recife patches.",
            "required_inputs": "grid_expansion_protocol;candidate_patch_manifest;human_qa;feature_plan",
            "expected_outputs": "candidate_patch_grid_review_package_not_official",
            "can_unlock_17b": "false",
            **_gov_str(),
        },
        {
            "decision_id": "S17C5_DEC_002",
            "event_id": CHARTER_EVENT_ID,
            "decision_scope": "formal_packages",
            "current_status": "P0_COMPDEC_and_DRM_packages_missing",
            "next_action": "request_compdec_pkg_first",
            "why": "P0 official packages remain absent and may constrain future grid expansion.",
            "required_inputs": "PKG_FR_REC_002;PKG_FR_PET_001",
            "expected_outputs": "official_package_inventory_or_blocker_update",
            "can_unlock_17b": "false",
            **_gov_str(),
        },
        {
            "decision_id": "S17C5_DEC_003",
            "event_id": CHARTER_EVENT_ID,
            "decision_scope": "namespace_bridge",
            "current_status": "REC_00019_protocolc_namespace_only",
            "next_action": "blocked_until_grid_expansion_protocol",
            "why": "REC_00019 is not a SUSC recife_* patch and no bridge manifest exists.",
            "required_inputs": "audited_namespace_bridge_manifest_if_any",
            "expected_outputs": "namespace_bridge_audit_or_explicit_no_match",
            "can_unlock_17b": "false",
            **_gov_str(),
        },
        {
            "decision_id": "S17C5_DEC_004",
            "event_id": CHARTER_EVENT_ID,
            "decision_scope": "17b_benchmark",
            "current_status": "blocked_no_valid_patch_link_no_qa",
            "next_action": "blocked_until_grid_expansion_protocol",
            "why": "17B requires QA accepted geometry plus a valid patch-link; both are absent now.",
            "required_inputs": "qa_accepted_reference_geometry;valid_susc_patch_link",
            "expected_outputs": "17b_eligibility_recheck",
            "can_unlock_17b": "false",
            **_gov_str(),
        },
    ]
    rows.sort(key=lambda r: r["decision_id"])
    return rows


def build_summary(grid_inventory, coverage, namespace, options, aoi) -> dict:
    recife = [g for g in grid_inventory if g["patch_grid_id"] == "SUSC_GRID_RECIFE_V1"][0]
    cov = coverage[0]
    allowed_now = sum(1 for o in options if o["allowed_now"] == "true")
    blocked = sum(1 for o in options if o["allowed_now"] == "false")
    return {
        "patch_grids_inventoried": len(grid_inventory),
        "recife_patch_count": int(recife["patch_count"]),
        "charter758_intersecting_patch_count": int(cov["intersecting_patch_count"]),
        "charter758_nearest_patch_distance_m": float(cov["nearest_patch_distance_m"]),
        "charter758_coverage_status": cov["coverage_status"],
        "protocolc_patch_id": PROTOCOLC_PATCH_ID,
        "protocolc_to_susc_match_status": namespace[0]["match_status"],
        "candidate_expansion_aoi_created": len(aoi.get("features", [])) == 4,
        "official_patch_created": False,
        "official_patch_link_created": False,
        "allowed_now_options_count": allowed_now,
        "blocked_options_count": blocked,
        "eligible_for_17b_now": False,
        "blocks_17b": [
            "no_qa_accepted_yet",
            "no_valid_susc_patch_link",
            "candidate_geometry_outside_current_susc_patch_grid",
            "REC_00019_namespace_incompatible_with_SUSC_recife_patch_grid",
            "P0_official_packages_still_missing",
            "sar_runtime_unavailable",
        ],
        "recommended_next_milestone": "SUSC-17C6 Patch Grid Candidate Generation",
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "score_v6_changed": False,
        "score_v7_created": False,
    }


def build_report(grid_inventory, coverage, namespace, options, summary) -> int:
    recife = [g for g in grid_inventory if g["patch_grid_id"] == "SUSC_GRID_RECIFE_V1"][0]
    allowed = [o for o in options if o["allowed_now"] == "true"]
    blocked = [o for o in options if o["allowed_now"] == "false"]
    report = f"""# SUSC-17C5 - Patch Grid Expansion Review

Status: review-only. `trainable=false`; `ground_truth=false`; `score_v6_changed=false`; `score_v7_created=false`; `official_patch_created=false`; `official_patch_link_created=false`; `eligible_for_17b_now=false`.

## 1. O que o SUSC-17C4 descobriu

O SUSC-17C4 encontrou uma geometria oficial candidata real do Charter 758 para `{CHARTER_EVENT_ID}`: `MultiPolygon`, CRS EPSG:4326, `qa_status=needs_review`, `review_only=true`, `ground_truth=false`. A tabela 17C4 de patch-links permaneceu vazia.

## 2. Por que Charter 758 e real, mas ainda nao utilizavel para 17B

A geometria existe e e auditavel como candidata observacional, mas ainda nao tem QA aceito e nao intersecta a grade SUSC Recife atual. Sem `qa_status=accepted` e sem patch-link SUSC valido, o benchmark 17B continua bloqueado.

## 3. Por que cair fora da grade impede patch-link

A grade SUSC Recife atual tem `{recife['patch_count']}` patches, bbox `{recife['xmin']};{recife['ymin']};{recife['xmax']};{recife['ymax']}`. O Charter 758 intersecta `{coverage[0]['intersecting_patch_count']}` patches, com patch mais proximo `{coverage[0]['nearest_patch_id']}` a `{coverage[0]['nearest_patch_distance_m']}` m. Criar patch-link com intersecao zero inventaria cobertura.

## 4. Por que REC_00019 nao pode ser assumido como patch SUSC

REC_00019 pertence ao namespace historico do Protocolo C. A grade SUSC usa ids `recife_*`; nao ha match de string nem manifesto de ponte entre namespaces. Status: `{namespace[0]['match_status']}`.

## 5. Expandir, manter ou apenas auditar

Nesta sprint, apenas auditar. A expansao candidata e metodologicamente aceitavel como proximo marco se for feita por protocolo proprio, manifesto de patches candidatos, extracao de features e QA humana. Ela nao e permitida como alteracao silenciosa da grade oficial.

## 6. Riscos metodologicos

Os riscos principais sao misturar namespace Protocolo C e SUSC, transformar AOI candidato em patch oficial, criar patch-link sem patch real, recomputar score sem protocolo separado, usar Charter como ground truth e executar 17B antes de QA + patch-link valido.

## 7. Artefatos necessarios para patches candidatos futuros

Sao necessarios protocolo de expansao de grade, manifesto de patches candidatos, regra de alinhamento ao grid atual, features hidrologicas e Sentinel para os candidatos, plano de QA humana e protocolo separado de recomputo de score se algum dia houver promocao.

## 8. Por que score v6 nao pode ser recalculado nesta sprint

O objetivo e revisar cobertura e risco, nao alterar o dataset oficial. Recalcular score v6 contaminaria a linha de base antes de existir patch candidato validado, features extraidas e protocolo de score proprio. O score v7 permanece inexistente.

## 9. Opcoes permitidas e bloqueadas

Permitidas agora: {", ".join(o["option_name"] for o in allowed)}.

Bloqueadas agora: {", ".join(o["option_name"] for o in blocked)}.

## 10. Proximo marco recomendado

`{summary['recommended_next_milestone']}`. Em paralelo, manter a solicitacao formal de pacotes P0, especialmente `PKG_FR_REC_002`, para reduzir risco antes de qualquer expansao oficial futura.
"""
    write_markdown(REPORT, report)
    return 0


def run_all() -> int:
    _require_inputs()
    grid_inventory = build_patch_grid_inventory()
    write_csv(PATCH_GRID_INVENTORY, grid_inventory, PATCH_GRID_FIELDS)
    coverage = build_charter_coverage_audit(grid_inventory)
    write_csv(COVERAGE_AUDIT, coverage, COVERAGE_FIELDS)
    namespace = build_namespace_audit()
    write_csv(NAMESPACE_AUDIT, namespace, NAMESPACE_FIELDS)
    aoi = build_candidate_expansion_aoi(grid_inventory, coverage)
    write_json(AOI_GEOJSON, aoi)
    options = build_expansion_options(namespace)
    write_csv(OPTIONS, options, OPTION_FIELDS)
    policy = build_risk_policy()
    write_json(RISK_POLICY, policy)
    decisions = build_next_action_decision_table(coverage)
    write_csv(NEXT_ACTIONS, decisions, NEXT_ACTION_FIELDS)
    summary = build_summary(grid_inventory, coverage, namespace, options, aoi)
    write_json(SUMMARY, summary)
    build_report(grid_inventory, coverage, namespace, options, summary)
    print(
        "17C5 -> "
        f"patch_grids={len(grid_inventory)} "
        f"recife_patches={summary['recife_patch_count']} "
        f"intersections={summary['charter758_intersecting_patch_count']} "
        f"nearest_m={summary['charter758_nearest_patch_distance_m']:.1f} "
        f"coverage={summary['charter758_coverage_status']}"
    )
    return 0


def _schema_violations(row: dict, schema: dict) -> list[str]:
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


def _gov_violations(row: dict) -> list[str]:
    out = []
    if row.get("review_only") != "true":
        out.append("review_only not true")
    if row.get("trainable") != "false":
        out.append("trainable not false")
    if row.get("ground_truth") != "false":
        out.append("ground_truth not false")
    return out


def _git_diff_name_only(path: Path) -> str:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", rel(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip()


def validate() -> int:
    errors: list[str] = []
    required_outputs = [
        REPORT, PATCH_GRID_INVENTORY, COVERAGE_AUDIT, NAMESPACE_AUDIT,
        AOI_GEOJSON, OPTIONS, RISK_POLICY, NEXT_ACTIONS, SUMMARY,
    ]
    required_files = [*REQUIRED_INPUTS, *required_outputs]
    for path in required_files:
        if not path.exists():
            errors.append(f"missing: {rel(path)}")
    if errors:
        print("FAILED: SUSC-17C5 validation")
        for err in errors:
            print("  ERROR:", err)
        return 1

    grid_schema = read_json(GRID_SCHEMA)
    option_schema = read_json(OPTION_SCHEMA)
    grid_inventory = read_csv(PATCH_GRID_INVENTORY)
    coverage = read_csv(COVERAGE_AUDIT)
    namespace = read_csv(NAMESPACE_AUDIT)
    options = read_csv(OPTIONS)
    decisions = read_csv(NEXT_ACTIONS)
    summary = read_json(SUMMARY)
    aoi = read_json(AOI_GEOJSON)
    policy = read_json(RISK_POLICY)

    for label, rows, key in (
        ("grid_inventory", grid_inventory, "patch_grid_id"),
        ("coverage", coverage, "coverage_audit_id"),
        ("namespace", namespace, "namespace_audit_id"),
        ("options", options, "expansion_option_id"),
        ("decisions", decisions, "decision_id"),
    ):
        ids = [r[key] for r in rows]
        if len(ids) != len(set(ids)):
            errors.append(f"{label}: ids not unique")
        if ids != sorted(ids):
            errors.append(f"{label}: ids not sorted")

    for line_no, row in enumerate(grid_inventory, start=2):
        for err in _schema_violations(row, grid_schema):
            errors.append(f"{PATCH_GRID_INVENTORY.name}:{line_no}: {err}")
        for err in _gov_violations(row):
            errors.append(f"{PATCH_GRID_INVENTORY.name}:{line_no}: {err}")
        if row["grid_status"] == "existing_patch_grid" and row["has_patch_bbox"] != "true":
            errors.append(f"{PATCH_GRID_INVENTORY.name}:{line_no}: existing grid without bbox")

    for line_no, row in enumerate(options, start=2):
        for err in _schema_violations(row, option_schema):
            errors.append(f"{OPTIONS.name}:{line_no}: {err}")
        for err in _gov_violations(row):
            errors.append(f"{OPTIONS.name}:{line_no}: {err}")
        if row["requires_score_recompute"] == "true" and row["allowed_now"] != "false":
            errors.append(f"{OPTIONS.name}:{line_no}: score recompute option allowed now")
        if row["requires_new_patches"] == "true" and row["allowed_now"] != "false":
            errors.append(f"{OPTIONS.name}:{line_no}: new patch option allowed now")

    for row in coverage:
        for err in _gov_violations(row):
            errors.append(f"{COVERAGE_AUDIT.name}: {err}")
        if row["intersecting_patch_count"] == "0" and row["recommended_action"] == "human_qa_reference_geometry_only":
            errors.append("zero-intersection geometry incorrectly routed to existing-patch QA only")
        if row["coverage_status"] not in {
            "intersects_patch_grid", "outside_patch_grid_adjacent", "outside_patch_grid_far",
            "no_patch_grid", "unknown_crs", "invalid_geometry", "needs_grid_expansion_review",
        }:
            errors.append(f"invalid coverage_status: {row['coverage_status']}")

    ns = namespace[0]
    if ns["source_patch_id"] == PROTOCOLC_PATCH_ID:
        if ns["target_susc_patch_match"] != "false":
            errors.append("REC_00019 treated as SUSC match")
        if ns["match_status"] not in ("namespace_incompatible", "no_susc_match", "source_patch_not_found"):
            errors.append("REC_00019 namespace status not fail-closed")

    aoi_roles = [f.get("properties", {}).get("feature_role") for f in aoi.get("features", [])]
    expected_roles = [
        "observed_reference_geometry", "current_susc_grid_extent",
        "candidate_expansion_envelope", "candidate_review_buffer",
    ]
    if aoi_roles != expected_roles:
        errors.append("AOI feature roles not deterministic or incomplete")
    for feature in aoi.get("features", []):
        props = feature.get("properties", {})
        if props.get("review_only") is not True or props.get("ground_truth") is not False:
            errors.append("AOI feature governance wrong")
        if props.get("not_official_patch") is not True or props.get("not_footprint") is not True:
            errors.append("AOI feature promoted to patch or footprint")

    blocked = policy.get("blocked_actions", {})
    for key in (
        "silent_grid_expansion", "patch_creation_without_manifest",
        "patch_link_without_official_patch", "protocolc_susc_namespace_mixing",
        "score_recompute_without_protocol", "charter_as_ground_truth",
        "candidate_aoi_as_strong_evidence",
        "benchmark_17b_without_qa_accepted_and_valid_patch_link",
    ):
        if blocked.get(key) is not True:
            errors.append(f"policy does not block {key}")

    if summary.get("charter758_intersecting_patch_count") != 0:
        errors.append("summary intersection count is not fail-closed zero")
    if summary.get("official_patch_created") is not False:
        errors.append("summary official_patch_created not false")
    if summary.get("official_patch_link_created") is not False:
        errors.append("summary official_patch_link_created not false")
    if summary.get("eligible_for_17b_now") is not False:
        errors.append("summary eligible_for_17b_now not false")
    if summary.get("review_only") is not True or summary.get("trainable") is not False or summary.get("ground_truth") is not False:
        errors.append("summary governance wrong")
    if summary.get("score_v6_changed") is not False or summary.get("score_v7_created") is not False:
        errors.append("summary score flags wrong")
    if not summary.get("candidate_expansion_aoi_created"):
        errors.append("summary candidate_expansion_aoi_created false")

    for row in decisions:
        for err in _gov_violations(row):
            errors.append(f"{NEXT_ACTIONS.name}: {err}")
        if row["can_unlock_17b"] != "false":
            errors.append("decision claims it can unlock 17B now")

    if SCORE_V7.exists():
        errors.append("score v7 artifact exists")
    if not SCORE_V6.exists():
        errors.append("score v6 source missing")
    if _git_diff_name_only(SCORE_V6):
        errors.append("score v6 has git diff")
    if _git_diff_name_only(FEATURES):
        errors.append("official patch grid dataset has git diff")

    report_text = REPORT.read_text(encoding="utf-8")
    for phrase in (
        "REC_00019 pertence ao namespace historico do Protocolo C",
        "Criar patch-link com intersecao zero inventaria cobertura",
        "O score v7 permanece inexistente",
    ):
        if phrase not in report_text:
            errors.append(f"report missing phrase: {phrase}")

    if errors:
        print("FAILED: SUSC-17C5 validation")
        for err in errors:
            print("  ERROR:", err)
        return 1
    print("PASSED: SUSC-17C5 validations passed")
    return 0
