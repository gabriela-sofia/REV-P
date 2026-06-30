"""SUSC-17C6 Canario de Aplicabilidade Multimodal Review-Only.

Builds a deterministic, review-only multimodal canary around the real Charter
758 candidate geometry without creating official patches, official patch-links,
score v7, labels, training data, models or ground truth.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import read_csv, read_json, rel, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

AUDIT = OUT / "SUSC_WORKTREE_AUDIT_BEFORE_17C6.md"
FEATURES = DAT / "susc_features_by_patch_v1.csv"
SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"

GRID_17C5 = OUT / "susc_17c5_patch_grid_inventory.csv"
COVERAGE_17C5 = OUT / "susc_17c5_charter758_grid_coverage_audit.csv"
AOI_17C5 = OUT / "susc_17c5_candidate_expansion_aoi.geojson"
OPTIONS_17C5 = OUT / "susc_17c5_patch_grid_expansion_options.csv"
POLICY_17C5 = OUT / "susc_17c5_expansion_risk_policy.json"
SUMMARY_17C5 = OUT / "susc_17c5_summary.json"
GEOMS_17C4 = OUT / "susc_17c4_candidate_geometries.geojson"
REFS_17C4 = OUT / "susc_17c4_extracted_reference_candidates.csv"

MATRIX_SCHEMA = SCHEMAS / "susc_17c6_multimodal_canary_schema_v1.json"
EMBEDDING_SCHEMA = SCHEMAS / "susc_17c6_embedding_contract_schema_v1.json"

REPORT = OUT / "SUSC_17C6_MULTIMODAL_APPLICABILITY_CANARY_REPORT.md"
CANDIDATE_GRID = OUT / "susc_17c6_candidate_patch_grid.csv"
CANDIDATE_GRID_GEOJSON = OUT / "susc_17c6_candidate_patch_grid.geojson"
CANDIDATE_LINKS = OUT / "susc_17c6_candidate_patch_links.csv"
FEATURE_CONTRACT = OUT / "susc_17c6_multimodal_feature_contract.csv"
CANARY_MATRIX = OUT / "susc_17c6_multimodal_canary_matrix.csv"
EMBEDDING_CONTRACT = OUT / "susc_17c6_embedding_contract.json"
SYNTHETIC_EMBEDDING = OUT / "susc_17c6_synthetic_embedding_smoke_test.csv"
SUMMARY = OUT / "susc_17c6_applicability_readiness_summary.json"
BLOCKERS = OUT / "susc_17c6_promotion_blockers.csv"

CHARTER_REF_ID = "S17C4_REF_REC_2022_05_24_30"
CHARTER_EVENT_ID = "REC_2022_05_24_30"
MILESTONE_CATEGORY = "canario_de_aplicabilidade_multimodal_review_only"
MILESTONE_NAME = "SUSC-17C6 — Canário de Aplicabilidade Multimodal Review-Only"
REC_00019 = "REC_00019"

REQUIRED_INPUTS = [
    AUDIT, FEATURES, SCORE_V6,
    GRID_17C5, COVERAGE_17C5, AOI_17C5, OPTIONS_17C5, POLICY_17C5, SUMMARY_17C5,
    GEOMS_17C4, REFS_17C4, MATRIX_SCHEMA, EMBEDDING_SCHEMA,
]

REQUIRED_OUTPUTS = [
    REPORT, CANDIDATE_GRID, CANDIDATE_GRID_GEOJSON, CANDIDATE_LINKS,
    FEATURE_CONTRACT, CANARY_MATRIX, EMBEDDING_CONTRACT, SYNTHETIC_EMBEDDING,
    SUMMARY, BLOCKERS,
]

MULTIMODAL_LAYERS = [
    "physical_static",
    "urban_territorial",
    "sentinel2_spectral",
    "rainfall_trigger",
    "documentary_evidence",
    "sar_observational",
    "embedding_representation",
]


def _require_inputs() -> None:
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(p) for p in missing))


def _gov_str() -> dict:
    return {"review_only": "true", "trainable": "false", "ground_truth": "false"}


def _fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "not_available"
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".")


def _float(row: dict, key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def _iter_points(obj):
    if isinstance(obj, (list, tuple)):
        if len(obj) >= 2 and isinstance(obj[0], (int, float)) and isinstance(obj[1], (int, float)):
            yield (float(obj[0]), float(obj[1]))
        else:
            for item in obj:
                yield from _iter_points(item)


def _geometry_bbox(geom: dict) -> tuple[float, float, float, float]:
    pts = list(_iter_points(geom.get("coordinates", [])))
    if not pts:
        raise ValueError("reference geometry has no coordinates")
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _rings(geom: dict) -> list[list[tuple[float, float]]]:
    if geom.get("type") == "Polygon":
        return [[(float(x), float(y)) for x, y in geom["coordinates"][0]]]
    if geom.get("type") == "MultiPolygon":
        return [[(float(x), float(y)) for x, y in poly[0]] for poly in geom["coordinates"] if poly]
    return []


def _polygon_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        area += (x1 * y2) - (x2 * y1)
    return abs(area) / 2.0


def _clip_edge(poly: list[tuple[float, float]], edge: str, value: float) -> list[tuple[float, float]]:
    if not poly:
        return []

    def inside(p):
        x, y = p
        if edge == "left":
            return x >= value
        if edge == "right":
            return x <= value
        if edge == "bottom":
            return y >= value
        return y <= value

    def intersect(a, b):
        ax, ay = a
        bx, by = b
        if edge in {"left", "right"}:
            if bx == ax:
                return (value, ay)
            t = (value - ax) / (bx - ax)
            return (value, ay + t * (by - ay))
        if by == ay:
            return (ax, value)
        t = (value - ay) / (by - ay)
        return (ax + t * (bx - ax), value)

    out = []
    prev = poly[-1]
    prev_in = inside(prev)
    for cur in poly:
        cur_in = inside(cur)
        if cur_in:
            if not prev_in:
                out.append(intersect(prev, cur))
            out.append(cur)
        elif prev_in:
            out.append(intersect(prev, cur))
        prev = cur
        prev_in = cur_in
    return out


def _clip_to_rect(poly: list[tuple[float, float]], bbox: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    out = poly[:]
    for edge, value in (
        ("left", bbox[0]),
        ("right", bbox[2]),
        ("bottom", bbox[1]),
        ("top", bbox[3]),
    ):
        out = _clip_edge(out, edge, value)
    return out


def _polygon_rect_intersection_area(rings: list[list[tuple[float, float]]], bbox: tuple[float, float, float, float]) -> float:
    return sum(_polygon_area(_clip_to_rect(ring, bbox)) for ring in rings)


def _bbox_polygon(bbox: tuple[float, float, float, float]) -> dict:
    x0, y0, x1, y1 = bbox
    return {
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
    }


def _bbox_gap_meters(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    dx = max(0.0, a[0] - b[2], b[0] - a[2])
    dy = max(0.0, a[1] - b[3], b[1] - a[3])
    lat = (a[1] + a[3]) / 2.0
    mx = dx * 111320.0 * math.cos(math.radians(lat))
    my = dy * 110540.0
    return math.hypot(mx, my)


def _patch_bbox(row: dict) -> tuple[float, float, float, float] | None:
    vals = [_float(row, k) for k in ("xmin", "ymin", "xmax", "ymax")]
    if any(v is None for v in vals):
        return None
    return tuple(vals)  # type: ignore[return-value]


def _load_charter_feature() -> dict:
    for feature in read_json(GEOMS_17C4).get("features", []):
        if feature.get("properties", {}).get("reference_candidate_id") == CHARTER_REF_ID:
            return feature
    raise ValueError(f"{CHARTER_REF_ID} not found")


def _load_charter_reference() -> dict:
    for row in read_csv(REFS_17C4):
        if row.get("reference_candidate_id") == CHARTER_REF_ID:
            return row
    raise ValueError(f"{CHARTER_REF_ID} not found")


def _recife_grid_meta() -> dict:
    for row in read_csv(GRID_17C5):
        if row.get("patch_grid_id") == "SUSC_GRID_RECIFE_V1":
            return row
    raise ValueError("SUSC_GRID_RECIFE_V1 not found")


def _nearest_official_patch(bbox: tuple[float, float, float, float]) -> tuple[str, float]:
    candidates = []
    for row in read_csv(FEATURES):
        if row.get("regiao") != "recife":
            continue
        pb = _patch_bbox(row)
        if pb is not None:
            candidates.append((row.get("patch_id", "not_available"), _bbox_gap_meters(bbox, pb)))
    if not candidates:
        return "not_available", 0.0
    return min(candidates, key=lambda item: (item[1], item[0]))


CANDIDATE_GRID_FIELDS = [
    "candidate_patch_id", "candidate_grid_id", "region", "source_event_id",
    "source_reference_candidate_id", "xmin", "ymin", "xmax", "ymax", "crs",
    "generation_method", "intersects_reference_geometry", "overlap_ratio_patch",
    "overlap_ratio_reference", "distance_to_reference_m", "nearest_official_patch_id",
    "nearest_official_patch_distance_m", "official_patch", "review_only",
    "trainable", "ground_truth", "not_official_patch_reason",
]


def build_candidate_patch_grid() -> tuple[list[dict], dict]:
    feature = _load_charter_feature()
    geom = feature["geometry"]
    ref_bbox = _geometry_bbox(geom)
    rings = _rings(geom)
    ref_area = sum(_polygon_area(ring) for ring in rings)
    grid = _recife_grid_meta()
    step_x = float(grid["mean_patch_width_deg"])
    step_y = float(grid["mean_patch_height_deg"])
    origin_x = float(grid["xmin"])
    origin_y = float(grid["ymin"])
    col0 = math.floor((ref_bbox[0] - origin_x) / step_x)
    col1 = math.floor((ref_bbox[2] - origin_x) / step_x)
    row0 = math.floor((ref_bbox[1] - origin_y) / step_y)
    row1 = math.floor((ref_bbox[3] - origin_y) / step_y)
    rows = []
    geo_features = []
    idx = 1
    for row in range(row0, row1 + 1):
        for col in range(col0, col1 + 1):
            bbox = (
                origin_x + col * step_x,
                origin_y + row * step_y,
                origin_x + (col + 1) * step_x,
                origin_y + (row + 1) * step_y,
            )
            inter_area = _polygon_rect_intersection_area(rings, bbox)
            if inter_area <= 0.0:
                continue
            patch_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            nearest_id, nearest_m = _nearest_official_patch(bbox)
            patch_id = f"S17C6_CANARY_REC_{idx:05d}"
            record = {
                "candidate_patch_id": patch_id,
                "candidate_grid_id": "S17C6_CANARY_GRID_RECIFE_CHARTER758_V1",
                "region": "Recife",
                "source_event_id": CHARTER_EVENT_ID,
                "source_reference_candidate_id": CHARTER_REF_ID,
                "xmin": _fmt(bbox[0], 9),
                "ymin": _fmt(bbox[1], 9),
                "xmax": _fmt(bbox[2], 9),
                "ymax": _fmt(bbox[3], 9),
                "crs": "EPSG:4326",
                "generation_method": "aligned_to_susc_recife_mean_patch_step_from_charter758_geometry",
                "intersects_reference_geometry": "true",
                "overlap_ratio_patch": _fmt(inter_area / patch_area if patch_area else 0.0, 6),
                "overlap_ratio_reference": _fmt(inter_area / ref_area if ref_area else 0.0, 6),
                "distance_to_reference_m": "0",
                "nearest_official_patch_id": nearest_id,
                "nearest_official_patch_distance_m": f"{nearest_m:.1f}",
                "official_patch": "false",
                **_gov_str(),
                "not_official_patch_reason": "candidate_patch_for_multimodal_canary_only_not_in_official_susc_grid",
            }
            rows.append(record)
            geo_features.append({
                "type": "Feature",
                "geometry": _bbox_polygon(bbox),
                "properties": record,
            })
            idx += 1
    geojson = {
        "type": "FeatureCollection",
        "metadata": {
            "milestone": "SUSC-17C6",
            "candidate_grid_id": "S17C6_CANARY_GRID_RECIFE_CHARTER758_V1",
            "source_event_id": CHARTER_EVENT_ID,
            "source_reference_candidate_id": CHARTER_REF_ID,
            "crs": "EPSG:4326",
            "official_patch_created": False,
            "review_only": True,
            "trainable": False,
            "ground_truth": False,
            "feature_count": len(geo_features),
        },
        "features": geo_features,
    }
    return rows, geojson


CANDIDATE_LINK_FIELDS = [
    "candidate_patch_link_id", "event_id", "reference_candidate_id",
    "candidate_patch_id", "link_method", "overlap_ratio_patch",
    "overlap_ratio_reference", "distance_m", "link_quality", "qa_status",
    "official_patch_link", "eligible_for_17b_now", "eligible_for_calibration",
    "review_only", "trainable", "ground_truth", "not_official_link_reason",
]


def build_candidate_links(candidate_grid: list[dict]) -> list[dict]:
    rows = []
    for idx, row in enumerate(candidate_grid, start=1):
        rows.append({
            "candidate_patch_link_id": f"S17C6_LINK_REC_CHARTER758_{idx:05d}",
            "event_id": CHARTER_EVENT_ID,
            "reference_candidate_id": CHARTER_REF_ID,
            "candidate_patch_id": row["candidate_patch_id"],
            "link_method": "reference_geometry_intersection_with_candidate_patch_only",
            "overlap_ratio_patch": row["overlap_ratio_patch"],
            "overlap_ratio_reference": row["overlap_ratio_reference"],
            "distance_m": "0",
            "link_quality": "candidate_spatial_intersection_needs_human_qa",
            "qa_status": "needs_review",
            "official_patch_link": "false",
            "eligible_for_17b_now": "false",
            "eligible_for_calibration": "false",
            **_gov_str(),
            "not_official_link_reason": "candidate_patch_is_not_official_and_geometry_qa_not_accepted",
        })
    return rows


FEATURE_CONTRACT_FIELDS = [
    "feature_name", "layer", "required_for_future_score",
    "required_for_future_embedding", "required_for_review_only_evaluation",
    "current_source_status", "available_for_official_patches",
    "available_for_candidate_patches", "candidate_patch_status",
    "synthetic_placeholder_allowed", "usable_for_science", "blocking_reason",
]


def _feature_rows() -> list[dict]:
    return read_csv(FEATURES)


def _feature_columns() -> set[str]:
    rows = _feature_rows()
    if not rows:
        return set()
    return set(rows[0].keys())


def _contract_row(name, layer, columns, future_score, future_embedding, review_required, synthetic_allowed, candidate_status, blocking):
    cols = _feature_columns()
    official_available = bool(columns) and all(c in cols for c in columns)
    return {
        "feature_name": name,
        "layer": layer,
        "required_for_future_score": "true" if future_score else "false",
        "required_for_future_embedding": "true" if future_embedding else "false",
        "required_for_review_only_evaluation": "true" if review_required else "false",
        "current_source_status": "available_for_official_patches" if official_available else "not_found_in_current_feature_table",
        "available_for_official_patches": "true" if official_available else "false",
        "available_for_candidate_patches": "false",
        "candidate_patch_status": candidate_status,
        "synthetic_placeholder_allowed": "true" if synthetic_allowed else "false",
        "usable_for_science": "false",
        "blocking_reason": blocking,
    }


def build_feature_contract() -> list[dict]:
    rows = [
        _contract_row("HAND", "physical_static", ["hand_mean"], True, False, True, True, "not_available", "candidate_patch_features_not_extracted"),
        _contract_row("elevation", "physical_static", ["elevation_mean"], True, False, True, True, "not_available", "candidate_patch_features_not_extracted"),
        _contract_row("slope", "physical_static", ["slope_mean"], True, False, True, True, "not_available", "candidate_patch_features_not_extracted"),
        _contract_row("DEM", "physical_static", ["elevation_mean"], True, False, True, True, "not_available", "candidate_patch_dem_not_extracted"),
        _contract_row("drainage", "physical_static", ["distance_to_water_mean", "flow_acc_log_mean"], True, False, True, True, "not_available", "candidate_patch_drainage_not_extracted"),
        _contract_row("distance_to_water", "physical_static", ["distance_to_water_mean"], True, False, True, True, "not_available", "candidate_patch_features_not_extracted"),
        _contract_row("flow_accumulation", "physical_static", ["flow_acc_log_mean"], True, False, True, True, "not_available", "candidate_patch_features_not_extracted"),
        _contract_row("TWI", "physical_static", ["twi_mean"], True, False, True, True, "not_available", "candidate_patch_features_not_extracted"),
        _contract_row("MapBiomas", "urban_territorial", ["urban_prop", "vegetation_prop"], True, False, True, True, "not_available", "candidate_patch_landcover_not_extracted"),
        _contract_row("urban_prop", "urban_territorial", ["urban_prop"], True, False, True, True, "not_available", "candidate_patch_features_not_extracted"),
        _contract_row("vegetation_prop", "urban_territorial", ["vegetation_prop"], True, False, True, True, "not_available", "candidate_patch_features_not_extracted"),
        _contract_row("water_prop", "urban_territorial", ["water_occurrence_patch"], True, False, True, True, "not_available", "candidate_patch_features_not_extracted"),
        _contract_row("NDBI", "urban_territorial", ["ndbi_mean"], True, False, True, True, "not_available", "candidate_patch_features_not_extracted"),
        _contract_row("imperviousness_proxy", "urban_territorial", ["urban_prop", "ndbi_mean"], True, False, True, True, "not_available", "candidate_patch_features_not_extracted"),
        _contract_row("Sentinel-2", "sentinel2_spectral", ["B2_mean", "B3_mean", "B4_mean", "B8_mean"], True, True, True, True, "not_available", "candidate_patch_sentinel_tile_not_materialized"),
        _contract_row("NDVI", "sentinel2_spectral", ["ndvi_mean"], True, True, True, True, "not_available", "candidate_patch_features_not_extracted"),
        _contract_row("NDWI", "sentinel2_spectral", [], True, True, True, True, "not_available", "ndwi_column_not_found_for_candidate_or_official_patch_table"),
        _contract_row("MNDWI", "sentinel2_spectral", ["mndwi_mean"], True, True, True, True, "not_available", "candidate_patch_features_not_extracted"),
        _contract_row("NDBI_spectral", "sentinel2_spectral", ["ndbi_mean"], True, True, True, True, "not_available", "candidate_patch_features_not_extracted"),
        _contract_row("acquisition_date", "sentinel2_spectral", ["reference_date"], True, True, True, False, "not_available", "candidate_patch_scene_lineage_not_resolved"),
        _contract_row("cloud_policy", "sentinel2_spectral", [], True, True, True, False, "not_available", "candidate_patch_cloud_policy_not_resolved"),
        _contract_row("CHIRPS_3d", "rainfall_trigger", ["chirps_3d_mm"], True, False, True, True, "not_available", "candidate_patch_rainfall_not_extracted"),
        _contract_row("CHIRPS_7d", "rainfall_trigger", ["chirps_7d_mm"], True, False, True, True, "not_available", "candidate_patch_rainfall_not_extracted"),
        _contract_row("CHIRPS_30d", "rainfall_trigger", ["chirps_30d_mm"], True, False, True, True, "not_available", "candidate_patch_rainfall_not_extracted"),
        _contract_row("runoff_context_7d", "rainfall_trigger", ["runoff_context_7d"], True, False, True, True, "not_available", "candidate_patch_runoff_not_extracted"),
        {
            "feature_name": "source_authority",
            "layer": "documentary_evidence",
            "required_for_future_score": "false",
            "required_for_future_embedding": "false",
            "required_for_review_only_evaluation": "true",
            "current_source_status": "available_from_17c4_charter758_reference_candidate",
            "available_for_official_patches": "false",
            "available_for_candidate_patches": "true",
            "candidate_patch_status": "available_review_only",
            "synthetic_placeholder_allowed": "false",
            "usable_for_science": "false",
            "blocking_reason": "documentary_evidence_requires_qa_before_scientific_use",
        },
        {
            "feature_name": "evidence_class",
            "layer": "documentary_evidence",
            "required_for_future_score": "false",
            "required_for_future_embedding": "false",
            "required_for_review_only_evaluation": "true",
            "current_source_status": "available_from_17c4_charter758_reference_candidate",
            "available_for_official_patches": "false",
            "available_for_candidate_patches": "true",
            "candidate_patch_status": "available_review_only",
            "synthetic_placeholder_allowed": "false",
            "usable_for_science": "false",
            "blocking_reason": "documentary_evidence_requires_qa_before_scientific_use",
        },
        _contract_row("source_confidence", "documentary_evidence", [], False, False, True, False, "not_available", "candidate_reference_qa_status_is_needs_review"),
        _contract_row("uncertainty_m", "documentary_evidence", [], False, False, True, False, "not_available", "uncertainty_not_available_in_17c4_reference_candidate"),
        {
            "feature_name": "qa_status",
            "layer": "documentary_evidence",
            "required_for_future_score": "true",
            "required_for_future_embedding": "false",
            "required_for_review_only_evaluation": "true",
            "current_source_status": "available_from_17c4_charter758_reference_candidate",
            "available_for_official_patches": "false",
            "available_for_candidate_patches": "true",
            "candidate_patch_status": "needs_review",
            "synthetic_placeholder_allowed": "false",
            "usable_for_science": "false",
            "blocking_reason": "qa_not_accepted",
        },
        _contract_row("SAR_pre_window", "sar_observational", [], True, True, False, False, "not_available", "sar_runtime_unavailable"),
        _contract_row("SAR_post_window", "sar_observational", [], True, True, False, False, "not_available", "sar_runtime_unavailable"),
        _contract_row("expected_footprint_class", "sar_observational", [], True, False, False, False, "not_available", "no_sar_footprint_created"),
        _contract_row("runtime_status", "sar_observational", [], False, False, True, False, "not_available", "sar_runtime_unavailable"),
        _contract_row("DINOv2_input_tile", "embedding_representation", [], False, True, True, True, "not_available", "no_real_embedding_tile"),
        _contract_row("SatMAE_input_tile", "embedding_representation", [], False, True, True, True, "not_available", "no_real_embedding_tile"),
        _contract_row("embedding_dimension", "embedding_representation", [], False, True, True, True, "synthetic_contract_only", "no_real_embedding_vector"),
        _contract_row("model_name", "embedding_representation", [], False, True, True, False, "contract_only", "model_not_executed_for_candidate_patch"),
        _contract_row("embedding_source", "embedding_representation", [], False, True, True, False, "synthetic_smoke_test_only", "synthetic_embedding_not_scientific"),
        _contract_row("embedding_status", "embedding_representation", [], False, True, True, False, "synthetic_smoke_test_only", "synthetic_embedding_not_scientific"),
    ]
    rows.sort(key=lambda r: r["feature_name"])
    return rows


CANARY_MATRIX_FIELDS = [
    "canary_row_id", "candidate_patch_id", "event_id", "reference_candidate_id",
    "has_reference_geometry", "has_candidate_patch_link",
    "has_physical_static_features", "has_urban_territorial_features",
    "has_sentinel2_spectral_features", "has_rainfall_trigger_features",
    "has_documentary_evidence", "has_sar_observational_footprint",
    "has_embedding_input_tile", "has_embedding_vector",
    "multimodal_readiness_score_review_only", "missing_layers",
    "synthetic_placeholder_used", "usable_for_science", "eligible_for_17b_now",
    "eligible_for_training", "eligible_for_score_v7", "review_only",
    "trainable", "ground_truth",
]


def build_canary_matrix(candidate_grid: list[dict], links: list[dict]) -> list[dict]:
    linked = {row["candidate_patch_id"] for row in links}
    rows = []
    missing = [
        "physical_static",
        "urban_territorial",
        "sentinel2_spectral",
        "rainfall_trigger",
        "sar_observational",
        "embedding_representation",
    ]
    for idx, patch in enumerate(candidate_grid, start=1):
        readiness = 1 / len(MULTIMODAL_LAYERS)
        rows.append({
            "canary_row_id": f"S17C6_CANARY_ROW_{idx:05d}",
            "candidate_patch_id": patch["candidate_patch_id"],
            "event_id": CHARTER_EVENT_ID,
            "reference_candidate_id": CHARTER_REF_ID,
            "has_reference_geometry": "true",
            "has_candidate_patch_link": "true" if patch["candidate_patch_id"] in linked else "false",
            "has_physical_static_features": "false",
            "has_urban_territorial_features": "false",
            "has_sentinel2_spectral_features": "false",
            "has_rainfall_trigger_features": "false",
            "has_documentary_evidence": "true",
            "has_sar_observational_footprint": "false",
            "has_embedding_input_tile": "false",
            "has_embedding_vector": "false",
            "multimodal_readiness_score_review_only": _fmt(readiness, 6),
            "missing_layers": ";".join(missing),
            "synthetic_placeholder_used": "true",
            "usable_for_science": "false",
            "eligible_for_17b_now": "false",
            "eligible_for_training": "false",
            "eligible_for_score_v7": "false",
            **_gov_str(),
        })
    return rows


def build_embedding_contract() -> dict:
    return {
        "contract_id": "SUSC_17C6_EMBEDDING_CONTRACT_V1",
        "milestone_category": MILESTONE_CATEGORY,
        "supported_models": ["DINOv2", "SatMAE", "Scale-MAE"],
        "expected_input": "patch_tile_sentinel2_or_rgb_multiband",
        "expected_shape": {
            "tile_pixels": "model_specific_square_tile",
            "bands": "RGB_or_multiband_with_documented_band_order",
        },
        "expected_crs_policy": "candidate_patch_tile_must_preserve_source_crs_and_document_reprojection_if_any",
        "expected_resolution_policy": "resolution_must_be_recorded_per_tile_and_not_inferred_from_filename_only",
        "expected_embedding_dim": {
            "DINOv2": 768,
            "SatMAE": 768,
            "Scale-MAE": 1024,
        },
        "normalization_policy": "model_specific_normalization_must_be_recorded_before_scientific_use",
        "cloud_policy": "sentinel2_tile_requires explicit cloud mask or documented cloud risk before scientific use",
        "missing_tile_policy": "do_not_run_real_embedding_without_real_local_tile",
        "synthetic_smoke_test_policy": "deterministic_hash_only_for_schema_join_and_interface_validation",
        "not_scientific_until_real_tile": True,
        "real_embedding_created": False,
        "synthetic_embedding_smoke_test_created": True,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
    }


SYNTHETIC_EMBEDDING_FIELDS = [
    "synthetic_embedding_id", "candidate_patch_id", "embedding_model_declared",
    "embedding_dim_declared", "embedding_source", "synthetic_embedding",
    "embedding_hash", "usable_for_science", "usable_for_training",
    "review_only", "ground_truth", "blocking_reason",
]


def build_synthetic_embedding_smoke(candidate_grid: list[dict]) -> list[dict]:
    rows = []
    for idx, patch in enumerate(candidate_grid, start=1):
        seed = f"SUSC-17C6|{patch['candidate_patch_id']}|DINOv2|synthetic_interface_only"
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        rows.append({
            "synthetic_embedding_id": f"S17C6_SYN_EMB_{idx:05d}",
            "candidate_patch_id": patch["candidate_patch_id"],
            "embedding_model_declared": "DINOv2_interface_contract",
            "embedding_dim_declared": "8",
            "embedding_source": "deterministic_hash_no_real_tile_no_model_execution",
            "synthetic_embedding": "true",
            "embedding_hash": digest,
            "usable_for_science": "false",
            "usable_for_training": "false",
            "review_only": "true",
            "ground_truth": "false",
            "blocking_reason": "synthetic_embedding_not_scientific_no_real_tile_no_model_execution",
        })
    return rows


BLOCKER_FIELDS = [
    "blocker_id", "blocker_code", "description", "blocks_17b",
    "blocks_training", "blocks_score_v7", "required_resolution",
    "review_only", "trainable", "ground_truth",
]


def build_promotion_blockers() -> list[dict]:
    items = [
        ("candidate_patches_not_official", "Patches candidatos nao pertencem a grade oficial SUSC.", "official_patch_manifest_or_policy"),
        ("candidate_patch_features_not_extracted", "Features fisicas, urbanas, espectrais e chuva ainda nao foram extraidas para candidatos.", "feature_extraction_pipeline_for_candidate_patches"),
        ("no_real_embedding_tile", "Nao ha tile real local para DINOv2, SatMAE ou Scale-MAE neste marco.", "real_tile_materialization_with_lineage"),
        ("synthetic_embedding_not_scientific", "O smoke test de embedding e apenas hash sintetico deterministico.", "real_model_execution_on_real_tile_with_provenance"),
        ("no_sar_runtime", "Runtime SAR segue indisponivel.", "sar_runtime_available_and_footprint_pipeline_validated"),
        ("no_qa_accepted", "A geometria Charter 758 ainda esta em QA needs_review.", "human_qa_accepted_reference_geometry"),
        ("no_score_policy_for_candidate_grid", "Nao existe politica de score para grade candidata.", "separate_score_policy_for_candidate_grid"),
        ("p0_official_packages_missing", "Pacotes formais DRM-RJ PKG_FR_PET_001 e COMPDEC PKG_FR_REC_002 seguem ausentes.", "formal_packages_or_explicit_blocker_update"),
        ("17b_blocked_until_official_patch_or_policy", "17B exige patch oficial ou politica explicita, QA aceito, features reais e evidencia suficiente.", "official_patch_or_policy_plus_qa_features_and_observational_evidence"),
    ]
    rows = []
    for idx, (code, desc, resolution) in enumerate(items, start=1):
        rows.append({
            "blocker_id": f"S17C6_BLOCKER_{idx:03d}",
            "blocker_code": code,
            "description": desc,
            "blocks_17b": "true",
            "blocks_training": "true",
            "blocks_score_v7": "true",
            "required_resolution": resolution,
            **_gov_str(),
        })
    return rows


def build_summary(candidate_grid, links, contract_rows, matrix_rows, synthetic_rows, blockers) -> dict:
    missing_layers = sorted({
        layer for row in matrix_rows for layer in row["missing_layers"].split(";") if layer
    })
    candidate_available_layers = sorted({
        row["layer"] for row in contract_rows if row["available_for_candidate_patches"] == "true"
    })
    official_available_layers = sorted({
        row["layer"] for row in contract_rows if row["available_for_official_patches"] == "true"
    })
    return {
        "milestone_category": MILESTONE_CATEGORY,
        "milestone_name": MILESTONE_NAME,
        "candidate_patch_count": len(candidate_grid),
        "candidate_patch_link_count": len(links),
        "reference_geometry_used": True,
        "official_patch_created": False,
        "official_patch_link_created": False,
        "score_v6_changed": False,
        "score_v7_created": False,
        "real_dino_embedding_created": False,
        "synthetic_embedding_smoke_test_created": bool(synthetic_rows),
        "multimodal_contract_created": True,
        "multimodal_layers": MULTIMODAL_LAYERS,
        "available_layers_for_candidate_patches": candidate_available_layers,
        "available_layers_for_official_patches": official_available_layers,
        "all_layers_available_for_candidate_patches": False,
        "missing_layers": missing_layers,
        "missing_layers_count": len(missing_layers),
        "synthetic_values_marked_scientific": False,
        "eligible_for_17b_now": False,
        "eligible_for_training": False,
        "eligible_for_score_v7": False,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "promotion_blocker_count": len(blockers),
        "recommended_next_milestone": "SUSC-17C7 Plano de Extracao de Features para Patches Candidatos",
    }


def build_report(summary: dict, candidate_grid: list[dict], links: list[dict]) -> None:
    layers = ", ".join(summary["multimodal_layers"])
    available_candidate = ", ".join(summary["available_layers_for_candidate_patches"]) or "nenhuma camada completa; apenas evidencia documental review-only"
    missing = ", ".join(summary["missing_layers"])
    text = f"""# SUSC-17C6 — Canário de Aplicabilidade Multimodal Review-Only

Status: somente para revisao. `review_only=true`; `trainable=false`; `ground_truth=false`; `score_v6_changed=false`; `score_v7_created=false`; `official_patch_created=false`; `official_patch_link_created=false`; `eligible_for_17b_now=false`.

## Objetivo do marco

Este marco testa se a esteira tecnica multimodal do REV-P consegue representar um patch candidato por multiplas camadas de evidencia, sem criar verdade-terreno, sem treinar modelo e sem promover score v7.

## Cadeia validada

Geometria observacional -> patch candidato -> matriz multimodal -> contrato de embeddings -> saida somente para revisao.

## Por que o teste e sintetico e somente para revisao

Os patches criados aqui sao candidatos e nao entram na grade oficial SUSC. As features fisicas, urbanas, espectrais, chuva, SAR e embeddings reais ainda nao foram extraidas para esses candidatos. O smoke test de embedding e apenas um hash deterministico para validar schema, join e interface.

## Geometria Charter 758

O Charter 758 entra como geometria observacional candidata real do evento `{CHARTER_EVENT_ID}`. O SUSC-17C5 registrou que ela intersecta zero patches da grade SUSC Recife atual e fica a `1398.4 m` do patch oficial mais proximo. Por isso este marco cria uma grade candidata separada, com prefixo proprio, sem alterar a grade oficial.

`REC_00019` nao e usado como patch SUSC neste marco. Ele permanece no namespace historico do Protocolo C e nao vira equivalencia automatica com a grade SUSC.

## Patches candidatos

Foram criados `{summary['candidate_patch_count']}` patches candidatos e `{summary['candidate_patch_link_count']}` vinculos candidatos patch-evento. Todos usam prefixo `S17C6_CANARY_REC_`, `official_patch=false`, `official_patch_link=false`, `review_only=true`, `trainable=false` e `ground_truth=false`.

## Contrato de camadas multimodais

Camadas registradas: {layers}.

Camadas disponiveis de verdade para os patches candidatos: {available_candidate}. Camadas missing/not_available: {missing}.

## O que foi real

- Geometria Charter 758 candidata, herdada do SUSC-17C4.
- Grade base SUSC Recife, herdada do SUSC-17C5 apenas como referencia de alinhamento.
- Contrato baseado nas colunas e artefatos ja existentes no repositorio.
- Patches candidatos geometricos somente para revisao.

## O que foi sintetico

- Placeholders de disponibilidade multimodal para testar fluxo e missingness.
- Smoke test de interface de embeddings com hash deterministico, sem tile real e sem execucao DINOv2/SatMAE/Scale-MAE.

## Papel futuro da matriz multimodal

A matriz canaria registra, por patch candidato, quais camadas existem, quais faltam e por que o registro nao pode virar evidencia cientifica. O readiness score e operacional, nao e score de suscetibilidade, nao valida o score v6 e nao autoriza score v7.

## Bloqueadores para virar evidencia cientifica

Faltam patches oficiais ou politica explicita para grade candidata, extracao real de features, tile real, embedding real, QA aceito, runtime SAR, politica de score e pacotes formais P0.

## Por que 17B e score v7 seguem bloqueados

O 17B exige vinculo com patch oficial ou politica aprovada, QA aceito, features reais e evidencia observacional suficiente. Nada disso foi promovido neste marco. O score v7 continua inexistente e inelegivel.

## Proximo marco recomendado

`{summary['recommended_next_milestone']}`. Se o maior bloqueio operacional for entrada real de embedding, a alternativa e `SUSC-17C7 Preparacao de Tile Real / Entrada DINO`. Se o maior bloqueio institucional continuar sendo P0, a alternativa e `SUSC-17C7 Pacote de Solicitacao Formal`.
"""
    write_markdown(REPORT, text)


def run_all() -> int:
    _require_inputs()
    candidate_grid, candidate_geojson = build_candidate_patch_grid()
    write_csv(CANDIDATE_GRID, candidate_grid, CANDIDATE_GRID_FIELDS)
    write_json(CANDIDATE_GRID_GEOJSON, candidate_geojson)
    links = build_candidate_links(candidate_grid)
    write_csv(CANDIDATE_LINKS, links, CANDIDATE_LINK_FIELDS)
    contract = build_feature_contract()
    write_csv(FEATURE_CONTRACT, contract, FEATURE_CONTRACT_FIELDS)
    matrix = build_canary_matrix(candidate_grid, links)
    write_csv(CANARY_MATRIX, matrix, CANARY_MATRIX_FIELDS)
    write_json(EMBEDDING_CONTRACT, build_embedding_contract())
    synthetic = build_synthetic_embedding_smoke(candidate_grid)
    write_csv(SYNTHETIC_EMBEDDING, synthetic, SYNTHETIC_EMBEDDING_FIELDS)
    blockers = build_promotion_blockers()
    write_csv(BLOCKERS, blockers, BLOCKER_FIELDS)
    summary = build_summary(candidate_grid, links, contract, matrix, synthetic, blockers)
    write_json(SUMMARY, summary)
    build_report(summary, candidate_grid, links)
    print(
        "17C6 -> "
        f"candidate_patches={summary['candidate_patch_count']} "
        f"candidate_links={summary['candidate_patch_link_count']} "
        f"missing_layers={summary['missing_layers_count']} "
        f"score_v7_created={summary['score_v7_created']}"
    )
    return 0


def _schema_violations(row: dict, schema: dict) -> list[str]:
    out = []
    props = schema.get("properties", {})
    for field in schema.get("required", []):
        if field not in row or not str(row.get(field, "")).strip():
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


def _gov_row_violations(row: dict, label: str) -> list[str]:
    out = []
    if row.get("review_only") != "true":
        out.append(f"{label}: review_only not true")
    if row.get("trainable") != "false":
        out.append(f"{label}: trainable not false")
    if row.get("ground_truth") != "false":
        out.append(f"{label}: ground_truth not false")
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


def _file_hashes(paths: list[Path]) -> dict[str, str]:
    return {rel(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def validate() -> int:
    errors: list[str] = []
    for path in [*REQUIRED_INPUTS, *REQUIRED_OUTPUTS]:
        if not path.exists():
            errors.append(f"missing: {rel(path)}")
    if errors:
        print("FAILED: SUSC-17C6 validation")
        for err in errors:
            print("  ERROR:", err)
        return 1

    matrix_schema = read_json(MATRIX_SCHEMA)
    embedding_schema = read_json(EMBEDDING_SCHEMA)
    candidate_grid = read_csv(CANDIDATE_GRID)
    links = read_csv(CANDIDATE_LINKS)
    contract = read_csv(FEATURE_CONTRACT)
    matrix = read_csv(CANARY_MATRIX)
    synthetic = read_csv(SYNTHETIC_EMBEDDING)
    blockers = read_csv(BLOCKERS)
    summary = read_json(SUMMARY)
    embedding_contract = read_json(EMBEDDING_CONTRACT)
    geojson = read_json(CANDIDATE_GRID_GEOJSON)

    for label, rows, key in (
        ("candidate_grid", candidate_grid, "candidate_patch_id"),
        ("candidate_links", links, "candidate_patch_link_id"),
        ("contract", contract, "feature_name"),
        ("matrix", matrix, "canary_row_id"),
        ("synthetic", synthetic, "synthetic_embedding_id"),
        ("blockers", blockers, "blocker_id"),
    ):
        ids = [row[key] for row in rows]
        if len(ids) != len(set(ids)):
            errors.append(f"{label}: ids not unique")
        if ids != sorted(ids):
            errors.append(f"{label}: ids not sorted")

    for row in candidate_grid:
        errors.extend(_gov_row_violations(row, CANDIDATE_GRID.name))
        if not row["candidate_patch_id"].startswith("S17C6_CANARY_REC_"):
            errors.append("candidate patch id prefix invalid")
        if row["candidate_patch_id"] == REC_00019:
            errors.append("REC_00019 appears as candidate patch id")
        if row["official_patch"] != "false":
            errors.append("candidate patch promoted to official")
        if row["intersects_reference_geometry"] != "true":
            errors.append("candidate patch without reference intersection")

    for row in links:
        errors.extend(_gov_row_violations(row, CANDIDATE_LINKS.name))
        if row["official_patch_link"] != "false":
            errors.append("candidate link promoted to official")
        if row["eligible_for_17b_now"] != "false":
            errors.append("candidate link eligible for 17B")
        if row["eligible_for_calibration"] != "false":
            errors.append("candidate link eligible for calibration")
        if row["candidate_patch_id"] == REC_00019:
            errors.append("REC_00019 appears as linked SUSC patch")

    for row in contract:
        if row["layer"] not in MULTIMODAL_LAYERS:
            errors.append(f"unexpected layer: {row['layer']}")
        if row["available_for_candidate_patches"] == "true" and row["layer"] != "documentary_evidence":
            errors.append(f"candidate layer unexpectedly available: {row['layer']}")
        if row["candidate_patch_status"] in {"synthetic", "placeholder"} and row["usable_for_science"] != "false":
            errors.append("synthetic feature marked usable for science")

    matrix_required_layers = set(MULTIMODAL_LAYERS)
    if set(row["layer"] for row in contract) != matrix_required_layers:
        errors.append("contract does not contain all mandatory layers")

    for line_no, row in enumerate(matrix, start=2):
        for err in _schema_violations(row, matrix_schema):
            errors.append(f"{CANARY_MATRIX.name}:{line_no}: {err}")
        errors.extend(_gov_row_violations(row, CANARY_MATRIX.name))
        if row["usable_for_science"] != "false":
            errors.append("matrix row usable_for_science not false")
        if row["eligible_for_17b_now"] != "false":
            errors.append("matrix row eligible_for_17b_now not false")
        if row["eligible_for_training"] != "false":
            errors.append("matrix row eligible_for_training not false")
        if row["eligible_for_score_v7"] != "false":
            errors.append("matrix row eligible_for_score_v7 not false")
        missing = set(row["missing_layers"].split(";"))
        for layer in ("physical_static", "urban_territorial", "sentinel2_spectral", "rainfall_trigger", "sar_observational", "embedding_representation"):
            if layer not in missing:
                errors.append(f"{CANARY_MATRIX.name}:{line_no}: missing layer not listed: {layer}")

    for key in embedding_schema.get("required", []):
        if key not in embedding_contract:
            errors.append(f"embedding contract missing {key}")
    if embedding_contract.get("review_only") is not True:
        errors.append("embedding contract review_only not true")
    if embedding_contract.get("trainable") is not False:
        errors.append("embedding contract trainable not false")
    if embedding_contract.get("ground_truth") is not False:
        errors.append("embedding contract ground_truth not false")
    if embedding_contract.get("not_scientific_until_real_tile") is not True:
        errors.append("embedding contract scientific gate not closed")
    if embedding_contract.get("real_embedding_created") is not False:
        errors.append("embedding contract says real embedding created")

    for row in synthetic:
        if row["synthetic_embedding"] != "true":
            errors.append("synthetic embedding row not synthetic")
        if row["usable_for_science"] != "false":
            errors.append("synthetic embedding usable_for_science not false")
        if row["usable_for_training"] != "false":
            errors.append("synthetic embedding usable_for_training not false")
        if row["review_only"] != "true" or row["ground_truth"] != "false":
            errors.append("synthetic embedding governance wrong")
        if len(row["embedding_hash"]) != 64:
            errors.append("synthetic embedding hash is not sha256 hex")

    expected_blockers = {
        "candidate_patches_not_official",
        "candidate_patch_features_not_extracted",
        "no_real_embedding_tile",
        "synthetic_embedding_not_scientific",
        "no_sar_runtime",
        "no_qa_accepted",
        "no_score_policy_for_candidate_grid",
        "p0_official_packages_missing",
        "17b_blocked_until_official_patch_or_policy",
    }
    if {row["blocker_code"] for row in blockers} != expected_blockers:
        errors.append("promotion blockers set mismatch")

    if geojson.get("metadata", {}).get("official_patch_created") is not False:
        errors.append("candidate grid geojson promotes official patch")
    if len(geojson.get("features", [])) != len(candidate_grid):
        errors.append("candidate grid geojson feature count mismatch")

    if summary.get("milestone_category") != MILESTONE_CATEGORY:
        errors.append("summary milestone_category mismatch")
    for key in (
        "official_patch_created", "official_patch_link_created", "score_v6_changed",
        "score_v7_created", "real_dino_embedding_created", "eligible_for_17b_now",
        "eligible_for_training", "eligible_for_score_v7", "trainable", "ground_truth",
        "synthetic_values_marked_scientific",
    ):
        if summary.get(key) is not False:
            errors.append(f"summary {key} not false")
    if summary.get("review_only") is not True:
        errors.append("summary review_only not true")
    if summary.get("candidate_patch_count") != len(candidate_grid):
        errors.append("summary candidate patch count mismatch")
    if summary.get("candidate_patch_link_count") != len(links):
        errors.append("summary candidate link count mismatch")
    if summary.get("missing_layers_count") != len(summary.get("missing_layers", [])):
        errors.append("summary missing layer count mismatch")

    if SCORE_V7.exists():
        errors.append("score v7 artifact exists")
    if _git_diff_name_only(SCORE_V6):
        errors.append("score v6 has git diff")
    if _git_diff_name_only(FEATURES):
        errors.append("official patch grid dataset has git diff")

    before = _file_hashes(REQUIRED_OUTPUTS)
    run_all()
    after = _file_hashes(REQUIRED_OUTPUTS)
    if before != after:
        errors.append("build is not byte-identical across two runs")

    report_text = REPORT.read_text(encoding="utf-8")
    for phrase in (
        "# SUSC-17C6 — Canário de Aplicabilidade Multimodal Review-Only",
        "somente para revisao",
        "O score v7 continua inexistente",
        "REC_00019",
    ):
        if phrase not in report_text:
            errors.append(f"report missing phrase: {phrase}")

    if errors:
        print("FAILED: SUSC-17C6 validation")
        for err in errors:
            print("  ERROR:", err)
        return 1
    print("PASSED: SUSC-17C6 validations passed")
    return 0
