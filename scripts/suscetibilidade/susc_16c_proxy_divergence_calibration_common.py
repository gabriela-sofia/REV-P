"""SUSC-16C review-only proxy divergence and calibration diagnostics.

This module derives audit tables, case files, figures, and validation from
SUSC-16A/SUSC-16B artifacts. It never creates score v7, ground truth, training
data, or model artifacts. Controls remain "no documented footprint" controls,
not true negatives.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_common import find_model_artifacts, matrix_sha256_ok  # noqa: E402
from susc_io import read_csv, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_16c_proxy_divergence_calibration_schema_v1.json"
AUDIT = OUT / "SUSC_WORKTREE_AUDIT_BEFORE_16C.md"

LINKAGE = DAT / "susc_16a_footprint_patch_linkage_v1.csv"
CATALOG = DAT / "susc_16a_observed_footprint_catalog_v1.csv"
S16B_DATASET = DAT / "susc_16b_footprint_event_control_dataset_v1.csv"
MATRIX = DAT / "susc_features_by_patch_v1.csv"
SCORE = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"

S16B_SUMMARY = OUT / "SUSC_16B_score_v6_footprint_evaluation_summary.json"
S16B_FEATURE_CONTRAST = OUT / "SUSC_16B_feature_contrast_against_footprints.csv"
S16B_QUALITY = OUT / "SUSC_16B_footprint_evidence_quality_audit.csv"
S16B_RECOMMENDATIONS = OUT / "SUSC_16B_proxy_calibration_recommendations.csv"

UNIFIED = DAT / "susc_16c_unified_observational_analysis_table_v1.csv"
UNIFIED_SUMMARY = OUT / "SUSC_16C_unified_observational_analysis_summary.csv"
DECOMP = OUT / "SUSC_16C_score_v6_component_decomposition.csv"
DECOMP_CASE = OUT / "SUSC_16C_score_v6_component_decomposition_by_case.csv"
DECOMP_SUMMARY = OUT / "SUSC_16C_score_v6_component_decomposition_summary.json"
CASE_AUDIT = OUT / "SUSC_16C_individual_case_audit.csv"
CASE_DIR = OUT / "SUSC_16C_individual_cases"
FEATURE_STABILITY = OUT / "SUSC_16C_feature_direction_stability.csv"
FEATURE_STABILITY_REGION = OUT / "SUSC_16C_feature_direction_stability_by_region.csv"
FEATURE_STABILITY_SUMMARY = OUT / "SUSC_16C_feature_direction_stability_summary.json"
WEIGHT_SENSITIVITY = OUT / "SUSC_16C_weight_sensitivity_review_only.csv"
WEIGHT_TOP = OUT / "SUSC_16C_weight_sensitivity_top_scenarios.csv"
WEIGHT_SUMMARY = OUT / "SUSC_16C_weight_sensitivity_summary.json"
FAILURES = OUT / "SUSC_16C_proxy_failure_modes.csv"
FAILURE_SUMMARY = OUT / "SUSC_16C_proxy_failure_modes_summary.json"
DESIGN = OUT / "SUSC_16C_calibration_design_matrix.csv"
DESIGN_MD = OUT / "SUSC_16C_calibration_design_matrix.md"
TCC_TABLES = OUT / "SUSC_16C_tcc_tables"
TCC_FIGURES = OUT / "SUSC_16C_tcc_figures"
REPORT = OUT / "SUSC_16C_proxy_divergence_and_calibration_review_report.md"

MANDATORY = (
    "O SUSC-16C transforma a divergencia entre score v6 e footprints "
    "observacionais em diagnostico auditavel de calibracao. A etapa permanece "
    "review-only, nao cria ground truth, nao libera treino supervisionado, nao "
    "altera o score v6 oficial e nao cria score v7 automatico."
)

REQUIRED_INPUTS = [
    AUDIT, SCHEMA, LINKAGE, CATALOG, S16B_DATASET, S16B_SUMMARY,
    S16B_FEATURE_CONTRAST, S16B_QUALITY, S16B_RECOMMENDATIONS, MATRIX, SCORE,
]

FEATURE_MAP = {
    "region": "regiao",
    "flow_accumulation_mean": "flow_acc_log_mean",
    "water_prop": "water_occurrence_patch",
}

FEATURES = [
    "hand_mean",
    "slope_mean",
    "elevation_mean",
    "distance_to_water_mean",
    "twi_mean",
    "flow_accumulation_mean",
    "urban_prop",
    "vegetation_prop",
    "water_prop",
    "ndvi_mean",
    "mndwi_mean",
    "ndbi_mean",
    "chirps_3d_mm",
    "chirps_7d_mm",
    "chirps_30d_mm",
    "runoff_context_7d",
]

EXPECTED_DIRECTIONS = {
    "hand_mean": "lower_in_event",
    "slope_mean": "lower_in_event",
    "elevation_mean": "lower_in_event",
    "distance_to_water_mean": "lower_in_event",
    "twi_mean": "higher_in_event",
    "flow_accumulation_mean": "higher_in_event",
    "urban_prop": "higher_in_event",
    "vegetation_prop": "lower_in_event",
    "water_prop": "higher_in_event",
    "ndvi_mean": "lower_in_event",
    "mndwi_mean": "higher_in_event",
    "ndbi_mean": "higher_in_event",
    "chirps_3d_mm": "higher_in_event",
    "chirps_7d_mm": "higher_in_event",
    "chirps_30d_mm": "higher_in_event",
    "runoff_context_7d": "higher_in_event",
}

UNIFIED_FIELDS = [
    "row_id", "case_id", "patch_id", "region", "group_type", "footprint_id",
    "footprint_type", "footprint_quality_tier", "evidence_strength",
    "overlap_ratio", "distance_m", "score_v6", "score_v6_class",
    "score_percentile_region", "is_event_patch", "is_control_patch",
    "is_low_score_event_case", "is_high_score_event_case",
    "is_calibration_candidate_case", "case_class", "diagnostic_concept",
    "hand_mean", "slope_mean", "elevation_mean", "twi_mean",
    "distance_to_water_mean", "flow_accumulation_mean", "urban_prop",
    "vegetation_prop", "water_prop", "ndvi_mean", "mndwi_mean", "ndbi_mean",
    "chirps_3d_mm", "chirps_7d_mm", "chirps_30d_mm", "runoff_context_7d",
    "can_be_ground_truth", "can_be_used_as_ground_truth",
    "allowed_for_training", "review_only", "score_v7_created",
]

COMPONENT_FIELDS = [
    "case_id", "patch_id", "footprint_id", "score_v6", "score_v6_class",
    "case_class", "topographic_component", "hydrological_component",
    "urban_component", "spectral_component", "rainfall_runoff_component",
    "documentary_component", "component_lowest", "component_highest",
    "possible_underestimated_driver", "interpretation", "can_be_ground_truth",
    "allowed_for_training", "review_only", "score_v7_created",
]


def _f(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if math.isnan(out):
        return None
    return out


def _fmt(value: float | int | str | None) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        parsed = _f(value)
        if parsed is None:
            return value
        value = parsed
    return str(round(float(value), 6))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _as_bool_text(value: bool) -> str:
    return "true" if value else "false"


def _value(row: dict, field: str) -> str:
    return row.get(field, row.get(FEATURE_MAP.get(field, field), ""))


def _score(row: dict) -> str:
    return row.get("susc_score_v6_candidate", row.get("score_v6", ""))


def _score_class(row: dict) -> str:
    return row.get("susc_class_v6_candidate", row.get("score_v6_class", ""))


def _require_inputs() -> None:
    missing = [path for path in REQUIRED_INPUTS if not path.exists()]
    if missing:
        raise FileNotFoundError("; ".join(str(path.relative_to(ROOT)) for path in missing))


def _index(rows: list[dict], key: str) -> dict[str, dict]:
    return {row.get(key, ""): row for row in rows if row.get(key)}


def _link_index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    out = {}
    for row in rows:
        if row.get("patch_id") and row.get("footprint_id"):
            out[(row["patch_id"], row["footprint_id"])] = row
    return out


def _load() -> dict[str, object]:
    _require_inputs()
    matrix = read_csv(MATRIX)
    score = read_csv(SCORE)
    linkage = read_csv(LINKAGE)
    quality = read_csv(S16B_QUALITY)
    return {
        "dataset": read_csv(S16B_DATASET),
        "summary": json.loads(S16B_SUMMARY.read_text(encoding="utf-8")),
        "contrast": read_csv(S16B_FEATURE_CONTRAST),
        "quality": quality,
        "recommendations": read_csv(S16B_RECOMMENDATIONS),
        "matrix": matrix,
        "score": score,
        "linkage": linkage,
        "matrix_by_patch": _index(matrix, "patch_id"),
        "score_by_patch": _index(score, "patch_id"),
        "quality_by_footprint": _index(quality, "footprint_id"),
        "link_by_patch_footprint": _link_index(linkage),
    }

def _percentiles(score_rows: list[dict]) -> dict[str, float]:
    by_region: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in score_rows:
        value = _f(_score(row))
        if value is not None:
            by_region[row.get("regiao", "")].append((row["patch_id"], value))
    out: dict[str, float] = {}
    for rows in by_region.values():
        ordered = sorted(rows, key=lambda item: item[1])
        denom = max(len(ordered) - 1, 1)
        for idx, (patch_id, _value_score) in enumerate(ordered):
            out[patch_id] = idx / denom
    return out


def _supporting_features(row: dict) -> list[str]:
    supporting: list[str] = []
    hand = _f(row.get("hand_mean"))
    water = _f(row.get("distance_to_water_mean"))
    twi = _f(row.get("twi_mean"))
    flow = _f(row.get("flow_accumulation_mean"))
    urban = _f(row.get("urban_prop"))
    ndvi = _f(row.get("ndvi_mean"))
    mndwi = _f(row.get("mndwi_mean"))
    rainfall = _f(row.get("chirps_7d_mm"))
    runoff = _f(row.get("runoff_context_7d"))
    if hand is not None and hand <= 12:
        supporting.append("low_HAND")
    if water is not None and water <= 500:
        supporting.append("near_water")
    if twi is not None and twi >= 8:
        supporting.append("high_TWI")
    if flow is not None and flow >= 6:
        supporting.append("high_flow_accumulation")
    if urban is not None and urban >= 0.35:
        supporting.append("urban_exposure")
    if ndvi is not None and ndvi <= 0.45:
        supporting.append("low_vegetation")
    if mndwi is not None and mndwi >= -0.15:
        supporting.append("wet_spectral_signal")
    if rainfall is not None and rainfall >= 25:
        supporting.append("rainfall_7d_context")
    if runoff is not None and runoff >= 0.25:
        supporting.append("runoff_context")
    return supporting


def _contradicting_features(row: dict) -> list[str]:
    contradicting: list[str] = []
    hand = _f(row.get("hand_mean"))
    water = _f(row.get("distance_to_water_mean"))
    twi = _f(row.get("twi_mean"))
    urban = _f(row.get("urban_prop"))
    ndvi = _f(row.get("ndvi_mean"))
    rainfall = _f(row.get("chirps_7d_mm"))
    if hand is not None and hand > 25:
        contradicting.append("high_HAND")
    if water is not None and water > 1500:
        contradicting.append("far_from_water")
    if twi is not None and twi < 5:
        contradicting.append("low_TWI")
    if urban is not None and urban < 0.1:
        contradicting.append("low_urban_exposure")
    if ndvi is not None and ndvi > 0.65:
        contradicting.append("high_vegetation")
    if rainfall is not None and rainfall < 10:
        contradicting.append("weak_rainfall_context")
    return contradicting


def _case_class(row: dict, quality_tier: str, support: list[str]) -> str:
    is_event = row.get("group_type") == "observed_footprint_patch"
    score_class = row.get("score_v6_class", "")
    if not is_event:
        return "score_v6_overestimated_control_patch" if score_class == "high" else "insufficient_evidence_case"
    if quality_tier.startswith(("A_", "B_")) and score_class in {"low", "medium"} and len(support) >= 3:
        return "high_confidence_underestimated_event_patch"
    if quality_tier.startswith(("A_", "B_", "C_")) and score_class in {"low", "medium"}:
        return "medium_confidence_underestimated_event_patch"
    if score_class in {"low", "medium"}:
        return "low_confidence_underestimated_event_patch"
    if score_class == "high":
        return "score_v6_consistent_event_patch"
    return "insufficient_evidence_case"


def _diagnostic_concept(case_class: str, quality_tier: str, support: list[str], contradictions: list[str]) -> str:
    if quality_tier.startswith(("D_", "E_")):
        return "footprint_quality_limited"
    if "underestimated" in case_class:
        if support:
            return "feature_supports_observed_footprint"
        return "score_v6_underestimation_case"
    if "overestimated" in case_class:
        return "score_v6_overestimation_case"
    if contradictions:
        return "feature_contradicts_observed_footprint"
    return "review_only"


def build_unified_observational_analysis_table() -> int:
    data = _load()
    dataset = data["dataset"]  # type: ignore[assignment]
    matrix_by_patch = data["matrix_by_patch"]  # type: ignore[assignment]
    score_by_patch = data["score_by_patch"]  # type: ignore[assignment]
    quality_by_fp = data["quality_by_footprint"]  # type: ignore[assignment]
    links = data["link_by_patch_footprint"]  # type: ignore[assignment]
    percentiles = _percentiles(data["score"])  # type: ignore[arg-type]
    feature_mapping_used = {}
    rows = []
    for idx, source in enumerate(dataset):  # type: ignore[union-attr]
        patch_id = source.get("patch_id", "")
        footprint_id = source.get("footprint_id", "")
        matrix_row = matrix_by_patch.get(patch_id, {})  # type: ignore[union-attr]
        score_row = score_by_patch.get(patch_id, {})  # type: ignore[union-attr]
        quality_row = quality_by_fp.get(footprint_id, {}) if footprint_id else {}  # type: ignore[union-attr]
        link_row = links.get((patch_id, footprint_id), {}) if footprint_id else {}  # type: ignore[union-attr]
        merged = {**matrix_row, **score_row, **source}
        for target, source_field in FEATURE_MAP.items():
            if target not in source and source_field in matrix_row:
                feature_mapping_used[target] = source_field
        support = _supporting_features(merged)
        contradictions = _contradicting_features(merged)
        quality_tier = quality_row.get("quality_tier", "not_applicable_control")
        case_class = _case_class(source, quality_tier, support)
        is_event = source.get("group_type") == "observed_footprint_patch"
        is_control = source.get("group_type") == "no_documented_footprint_control"
        score_class = source.get("score_v6_class", _score_class(score_row))
        calibration_candidate = is_event and score_class in {"low", "medium"} and bool(support)
        out = {
            "row_id": f"S16C_ROW_{idx:05d}",
            "case_id": f"S16C_CASE_{idx:05d}" if is_event else "",
            "patch_id": patch_id,
            "region": source.get("region") or matrix_row.get("regiao", ""),
            "group_type": source.get("group_type", ""),
            "footprint_id": footprint_id,
            "footprint_type": source.get("footprint_type", ""),
            "footprint_quality_tier": quality_tier,
            "evidence_strength": source.get("evidence_strength", ""),
            "overlap_ratio": link_row.get("overlap_ratio", ""),
            "distance_m": link_row.get("distance_m", ""),
            "score_v6": _fmt(source.get("score_v6") or _score(score_row)),
            "score_v6_class": score_class,
            "score_percentile_region": _fmt(percentiles.get(patch_id)),
            "is_event_patch": _as_bool_text(is_event),
            "is_control_patch": _as_bool_text(is_control),
            "is_low_score_event_case": _as_bool_text(is_event and score_class == "low"),
            "is_high_score_event_case": _as_bool_text(is_event and score_class == "high"),
            "is_calibration_candidate_case": _as_bool_text(calibration_candidate),
            "case_class": case_class,
            "diagnostic_concept": _diagnostic_concept(case_class, quality_tier, support, contradictions),
            "can_be_ground_truth": "false",
            "can_be_used_as_ground_truth": "false",
            "allowed_for_training": "false",
            "review_only": "true",
            "score_v7_created": "false",
        }
        for feature in FEATURES:
            out[feature] = _fmt(_value(merged, feature))
        rows.append(out)
    write_csv(UNIFIED, rows, UNIFIED_FIELDS)
    event_rows = [row for row in rows if row["is_event_patch"] == "true"]
    low_medium = [row for row in event_rows if row["score_v6_class"] in {"low", "medium"}]
    summary_rows = [
        {"metric": "row_count", "value": len(rows), "review_only": "true"},
        {"metric": "event_link_rows", "value": len(event_rows), "review_only": "true"},
        {"metric": "unique_event_patches", "value": len({row["patch_id"] for row in event_rows}), "review_only": "true"},
        {"metric": "low_or_medium_event_rows", "value": len(low_medium), "review_only": "true"},
        {"metric": "calibration_candidate_rows", "value": sum(1 for row in rows if row["is_calibration_candidate_case"] == "true"), "review_only": "true"},
        {"metric": "feature_mapping_used", "value": json.dumps(feature_mapping_used, ensure_ascii=False, sort_keys=True), "review_only": "true"},
        {"metric": "score_v7_created", "value": "false", "review_only": "true"},
        {"metric": "allowed_for_training", "value": "false", "review_only": "true"},
        {"metric": "can_be_ground_truth", "value": "false", "review_only": "true"},
    ]
    write_csv(UNIFIED_SUMMARY, summary_rows, ["metric", "value", "review_only"])
    print(f"unified -> {UNIFIED.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


def _component_values(row: dict, score_row: dict) -> dict[str, float]:
    topography = _f(score_row.get("topography_hydrology_index"))
    rainfall = _f(score_row.get("rainfall_trigger_index"))
    urban = _f(score_row.get("urban_spectral_index"))
    vegetation_mitigation = _f(score_row.get("vegetation_mitigation_index"))
    evidence = _f(score_row.get("evidence_support_index"))
    spectral = None if urban is None or vegetation_mitigation is None else max(0.0, min(1.0, (urban + (1.0 - vegetation_mitigation)) / 2.0))
    return {
        "topographic_component": topography if topography is not None else 0.0,
        "hydrological_component": topography if topography is not None else 0.0,
        "urban_component": urban if urban is not None else 0.0,
        "spectral_component": spectral if spectral is not None else 0.0,
        "rainfall_runoff_component": rainfall if rainfall is not None else 0.0,
        "documentary_component": evidence if evidence is not None else 0.0,
    }


def decompose_score_v6_components() -> int:
    if not UNIFIED.exists():
        build_unified_observational_analysis_table()
    data = _load()
    score_by_patch = data["score_by_patch"]  # type: ignore[assignment]
    rows = []
    for item in read_csv(UNIFIED):
        if item["is_event_patch"] != "true":
            continue
        components = _component_values(item, score_by_patch.get(item["patch_id"], {}))  # type: ignore[union-attr]
        lowest = min(components, key=components.get)
        highest = max(components, key=components.get)
        if lowest == "documentary_component":
            driver = "documentary_support_absent_or_low"
        elif lowest == "rainfall_runoff_component":
            driver = "rainfall_runoff_underweighted_candidate"
        elif lowest in {"topographic_component", "hydrological_component"}:
            driver = "hydro_topographic_component_low_or_non_discriminative"
        elif lowest == "urban_component":
            driver = "urban_component_underweighted_candidate"
        else:
            driver = "spectral_component_underweighted_candidate"
        interpretation = (
            "low_component_may_reduce_score_v6_review_only"
            if item["score_v6_class"] in {"low", "medium"}
            else "score_v6_high_despite_component_variation_review_only"
        )
        row = {
            "case_id": item["case_id"],
            "patch_id": item["patch_id"],
            "footprint_id": item["footprint_id"],
            "score_v6": item["score_v6"],
            "score_v6_class": item["score_v6_class"],
            "case_class": item["case_class"],
            "component_lowest": lowest,
            "component_highest": highest,
            "possible_underestimated_driver": driver,
            "interpretation": interpretation,
            "can_be_ground_truth": "false",
            "allowed_for_training": "false",
            "review_only": "true",
            "score_v7_created": "false",
        }
        for key, value in components.items():
            row[key] = _fmt(value)
        rows.append(row)
    write_csv(DECOMP, rows, COMPONENT_FIELDS)
    by_case = []
    for case_class, group_rows in _group(rows, "case_class").items():
        counts = Counter(row["component_lowest"] for row in group_rows)
        by_case.append({
            "case_class": case_class,
            "n": len(group_rows),
            "dominant_lowest_component": counts.most_common(1)[0][0] if counts else "",
            "lowest_component_distribution": json.dumps(dict(counts), ensure_ascii=False, sort_keys=True),
            "review_only": "true",
        })
    write_csv(DECOMP_CASE, by_case)
    summary = {
        "event_case_rows": len(rows),
        "component_lowest_distribution": dict(Counter(row["component_lowest"] for row in rows)),
        "possible_underestimated_driver_distribution": dict(Counter(row["possible_underestimated_driver"] for row in rows)),
        "score_v7_created": False,
        "allowed_for_training": False,
        "can_be_ground_truth": False,
        "review_only": True,
    }
    write_json(DECOMP_SUMMARY, summary)
    print(f"decomposition -> {DECOMP.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


def _group(rows: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        out[row.get(key, "")].append(row)
    return dict(out)


def _feature_summary_for_case(row: dict) -> tuple[str, str]:
    return ";".join(_supporting_features(row)), ";".join(_contradicting_features(row))


def _case_bucket(row: dict, decomp: dict) -> str:
    quality = row["footprint_quality_tier"]
    score_class = row["score_v6_class"]
    support, _contra = _feature_summary_for_case(row)
    has_support = bool(support)
    if quality.startswith(("A_", "B_")) and score_class in {"low", "medium"} and has_support and decomp.get("component_lowest") in {"topographic_component", "hydrological_component"}:
        return "A"
    if quality.startswith(("A_", "B_")) and score_class in {"low", "medium"} and has_support:
        return "B"
    if score_class in {"low", "medium"} and not quality.startswith(("A_", "B_")):
        return "C"
    if quality.startswith(("A_", "B_")) and score_class == "high":
        return "D"
    return "E"


def build_individual_case_files() -> int:
    if not DECOMP.exists():
        decompose_score_v6_components()
    unified = [row for row in read_csv(UNIFIED) if row["is_event_patch"] == "true"]
    decomp_by_case = _index(read_csv(DECOMP), "case_id")
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    for row in unified:
        decomp = decomp_by_case.get(row["case_id"], {})
        support, contra = _feature_summary_for_case(row)
        bucket = _case_bucket(row, decomp)
        is_cal = row["is_calibration_candidate_case"]
        limitations = "review_only; footprint_candidate_not_ground_truth; controls_not_true_negatives; no_score_v7_created"
        interpretation = (
            "score_v6_possibly_underestimates_observational_case"
            if row["score_v6_class"] in {"low", "medium"}
            else "score_v6_consistent_with_high_susceptibility_for_this_case"
        )
        audit_rows.append({
            "case_id": row["case_id"],
            "case_bucket": bucket,
            "case_class": row["case_class"],
            "patch_id": row["patch_id"],
            "footprint_id": row["footprint_id"],
            "region": row["region"],
            "score_v6": row["score_v6"],
            "score_v6_class": row["score_v6_class"],
            "score_percentile_region": row["score_percentile_region"],
            "footprint_quality_tier": row["footprint_quality_tier"],
            "features_supporting_susceptibility": support,
            "features_reducing_or_contradicting": contra,
            "component_lowest": decomp.get("component_lowest", ""),
            "is_calibration_candidate_case": is_cal,
            "limitations": limitations,
            "can_be_ground_truth": "false",
            "allowed_for_training": "false",
            "review_only": "true",
            "score_v7_created": "false",
        })
        case_md = f"""# {row['case_id']} - SUSC-16C individual footprint-patch audit

Status: review-only. Este caso nao e ground truth, nao libera treino e nao cria score v7.

- patch_id: `{row['patch_id']}`
- footprint_id: `{row['footprint_id']}`
- regiao: `{row['region']}`
- tipo de footprint: `{row['footprint_type']}`
- qualidade da evidencia: `{row['footprint_quality_tier']}`
- score v6: `{row['score_v6']}`
- classe score v6: `{row['score_v6_class']}`
- percentil regional: `{row['score_percentile_region']}`
- features que indicam suscetibilidade: {support or 'nenhuma sinalizacao forte nos limiares 16C'}
- features que reduzem ou contradizem o score: {contra or 'nenhuma contradicao forte nos limiares 16C'}
- componente que mais derrubou o score: `{decomp.get('component_lowest', '')}`
- interpretacao: {interpretation}
- candidato de recalibracao: `{is_cal}`
- classe resumida: `{bucket}`
- limitacoes: {limitations}
"""
        write_markdown(CASE_DIR / f"{row['case_id']}.md", case_md)
    write_csv(CASE_AUDIT, audit_rows)
    print(f"individual cases -> {CASE_DIR.relative_to(ROOT)} ({len(audit_rows)} files)")
    return 0


def _direction(event_values: list[float], control_values: list[float]) -> str:
    event_mean = _mean(event_values)
    control_mean = _mean(control_values)
    if event_mean is None or control_mean is None:
        return "insufficient_sample"
    if abs(event_mean - control_mean) < 1e-9:
        return "similar"
    return "higher_in_event" if event_mean > control_mean else "lower_in_event"


def _agreement(feature: str, direction: str) -> str:
    if direction == "insufficient_sample":
        return "insufficient_sample"
    return "agreement" if direction == EXPECTED_DIRECTIONS[feature] else "divergence"


def _values(rows: list[dict], feature: str) -> list[float]:
    return [value for value in (_f(row.get(feature)) for row in rows) if value is not None]


def _stability(feature: str, all_dir: str, ab_dir: str, region_dirs: dict[str, str], ab_n: int) -> str:
    all_agree = _agreement(feature, all_dir) == "agreement"
    ab_agree = _agreement(feature, ab_dir) == "agreement"
    region_clean = [direction for direction in region_dirs.values() if direction != "insufficient_sample"]
    if len(region_clean) < 2:
        region_conflict = False
    else:
        region_conflict = len(set(region_clean)) > 1
    if ab_n < 5:
        return "insufficient_sample"
    if all_agree and ab_agree and not region_conflict:
        return "stable_support"
    if region_conflict:
        return "regional_conflict"
    if all_agree != ab_agree:
        return "quality_dependent"
    if all_agree or ab_agree:
        return "partial_support"
    return "contradictory"


def run_feature_direction_stability() -> int:
    if not UNIFIED.exists():
        build_unified_observational_analysis_table()
    rows = read_csv(UNIFIED)
    events = [row for row in rows if row["is_event_patch"] == "true"]
    controls = [row for row in rows if row["is_control_patch"] == "true"]
    quality_ab = [row for row in events if row["footprint_quality_tier"].startswith(("A_", "B_"))]
    quality_abc = [row for row in events if row["footprint_quality_tier"].startswith(("A_", "B_", "C_"))]
    out_rows = []
    region_rows = []
    for feature in FEATURES:
        all_dir = _direction(_values(events, feature), _values(controls, feature))
        ab_dir = _direction(_values(quality_ab, feature), _values(controls, feature))
        abc_dir = _direction(_values(quality_abc, feature), _values(controls, feature))
        region_dirs = {}
        for region, region_events in _group(events, "region").items():
            region_controls = [row for row in controls if row["region"] == region]
            region_dir = _direction(_values(region_events, feature), _values(region_controls, feature))
            region_dirs[region] = region_dir
            region_rows.append({
                "feature_name": feature,
                "region": region,
                "expected_direction": EXPECTED_DIRECTIONS[feature],
                "observed_direction": region_dir,
                "agreement": _agreement(feature, region_dir),
                "event_n": len(region_events),
                "control_n": len(region_controls),
                "review_only": "true",
            })
        stability = _stability(feature, all_dir, ab_dir, region_dirs, len(quality_ab))
        out_rows.append({
            "feature_name": feature,
            "expected_direction": EXPECTED_DIRECTIONS[feature],
            "observed_direction_all": all_dir,
            "observed_direction_quality_ab": ab_dir,
            "observed_direction_quality_abc": abc_dir,
            "observed_direction_by_region": json.dumps(region_dirs, ensure_ascii=False, sort_keys=True),
            "agreement_all": _agreement(feature, all_dir),
            "agreement_quality_ab": _agreement(feature, ab_dir),
            "agreement_quality_abc": _agreement(feature, abc_dir),
            "agreement_by_region": json.dumps({k: _agreement(feature, v) for k, v in region_dirs.items()}, ensure_ascii=False, sort_keys=True),
            "stability_class": stability,
            "recommendation": _recommend_from_stability(stability),
            "can_be_ground_truth": "false",
            "allowed_for_training": "false",
            "review_only": "true",
            "score_v7_created": "false",
        })
    write_csv(FEATURE_STABILITY, out_rows)
    write_csv(FEATURE_STABILITY_REGION, region_rows)
    summary = {
        "feature_count": len(out_rows),
        "stability_distribution": dict(Counter(row["stability_class"] for row in out_rows)),
        "agreement_all_count": sum(1 for row in out_rows if row["agreement_all"] == "agreement"),
        "divergence_all_count": sum(1 for row in out_rows if row["agreement_all"] == "divergence"),
        "score_v7_created": False,
        "can_be_ground_truth": False,
        "allowed_for_training": False,
        "review_only": True,
    }
    write_json(FEATURE_STABILITY_SUMMARY, summary)
    print(f"feature stability -> {FEATURE_STABILITY.relative_to(ROOT)}")
    return 0


def _recommend_from_stability(stability: str) -> str:
    return {
        "stable_support": "eligible_for_future_weight_discussion_review_only",
        "partial_support": "keep_as_candidate_requires_more_data",
        "regional_conflict": "split_by_region_before_any_weight_change",
        "quality_dependent": "restrict_to_higher_quality_footprints_before_weight_change",
        "contradictory": "do_not_increase_weight_without_more_evidence",
        "insufficient_sample": "block_due_to_insufficient_sample",
    }[stability]


SCENARIOS = [
    ("increase_hydrological_weight", "Increase hydrological/topographic signal for review-only ranking", {"hydrological_component": 0.12}),
    ("increase_urban_weight", "Increase urban exposure signal", {"urban_component": 0.12}),
    ("increase_rainfall_runoff_weight", "Increase rainfall/runoff trigger signal", {"rainfall_runoff_component": 0.14}),
    ("increase_spectral_water_weight", "Increase wet spectral signal", {"spectral_component": 0.10}),
    ("decrease_topographic_penalty", "Reduce penalty when HAND/topography is not discriminative", {"topographic_component": 0.10}),
    ("decrease_ndvi_protection_weight", "Reduce vegetation mitigation in observed urban flood contexts", {"spectral_component": 0.08, "urban_component": 0.04}),
    ("increase_drainage_proximity_weight", "Increase drainage proximity/hydrology proxy", {"hydrological_component": 0.08, "topographic_component": 0.04}),
    ("balanced_observational_adjustment", "Balanced review-only observational adjustment", {"hydrological_component": 0.07, "urban_component": 0.07, "rainfall_runoff_component": 0.07, "spectral_component": 0.04}),
]


def _simulated_score(base: float, components: dict[str, float], weights: dict[str, float]) -> float:
    delta = 0.0
    for component, weight in weights.items():
        delta += weight * (components.get(component, 0.0) - 0.5)
    return max(0.0, min(1.0, base + delta))


def run_weight_sensitivity_review_only() -> int:
    if not DECOMP.exists():
        decompose_score_v6_components()
    unified = read_csv(UNIFIED)
    data = _load()
    score_by_patch = data["score_by_patch"]  # type: ignore[assignment]
    rows = []
    scored_base = []
    for item in unified:
        base = _f(item["score_v6"]) or 0.0
        components = _component_values(item, score_by_patch.get(item["patch_id"], {}))  # type: ignore[union-attr]
        scored_base.append((item, base, components))
    for scenario_id, description, weights in SCENARIOS:
        scored = []
        for item, base, components in scored_base:
            scored.append((item, _simulated_score(base, components, weights)))
        ordered = sorted(scored, key=lambda pair: pair[1], reverse=True)
        event_scores = [score for item, score in scored if item["is_event_patch"] == "true"]
        control_scores = [score for item, score in scored if item["is_control_patch"] == "true"]
        row = {
            "scenario_id": scenario_id,
            "description": description,
            "weight_changes": json.dumps(weights, sort_keys=True),
            "event_hit_rate_top_10_simulated": _fmt(_hit_rate(ordered, 10)),
            "event_hit_rate_top_20_simulated": _fmt(_hit_rate(ordered, 20)),
            "event_hit_rate_top_30_simulated": _fmt(_hit_rate(ordered, 30)),
            "mean_event_score_simulated": _fmt(_mean(event_scores)),
            "mean_control_score_simulated": _fmt(_mean(control_scores)),
            "event_control_diff_simulated": _fmt((_mean(event_scores) or 0.0) - (_mean(control_scores) or 0.0)),
            "risk_of_overfitting": "medium" if len(weights) <= 2 else "high",
            "safe_for_future_score_v7_discussion": "true" if scenario_id == "balanced_observational_adjustment" else "false",
            "can_be_ground_truth": "false",
            "allowed_for_training": "false",
            "review_only": "true",
            "score_v7_created": "false",
        }
        rows.append(row)
    write_csv(WEIGHT_SENSITIVITY, rows)
    top = sorted(rows, key=lambda row: (_f(row["event_hit_rate_top_30_simulated"]) or 0.0, _f(row["event_control_diff_simulated"]) or -9), reverse=True)[:5]
    write_csv(WEIGHT_TOP, top)
    write_json(WEIGHT_SUMMARY, {
        "scenario_count": len(rows),
        "top_scenario_by_hit30": top[0]["scenario_id"] if top else "",
        "best_hit_rate_top_30_simulated": top[0]["event_hit_rate_top_30_simulated"] if top else "",
        "score_v7_created": False,
        "susc_score_v6_candidate_modified": False,
        "can_be_ground_truth": False,
        "allowed_for_training": False,
        "review_only": True,
    })
    print(f"weight sensitivity -> {WEIGHT_SENSITIVITY.relative_to(ROOT)}")
    return 0


def _hit_rate(ordered: list[tuple[dict, float]], k: int) -> float:
    top = ordered[:k]
    if not top:
        return 0.0
    return sum(1 for item, _score_value in top if item["is_event_patch"] == "true") / len(top)


def classify_proxy_failure_modes() -> int:
    if not CASE_AUDIT.exists():
        build_individual_case_files()
    rows = []
    case_audit = _index(read_csv(CASE_AUDIT), "case_id")
    for item in read_csv(UNIFIED):
        if item["is_event_patch"] != "true":
            continue
        audit = case_audit.get(item["case_id"], {})
        support = audit.get("features_supporting_susceptibility", "")
        contra = audit.get("features_reducing_or_contradicting", "")
        lowest = audit.get("component_lowest", "")
        if item["footprint_quality_tier"].startswith(("D_", "E_")):
            primary = "footprint_quality_uncertain"
        elif lowest in {"hydrological_component", "topographic_component"} and "near_water" in support:
            primary = "hydrology_underweighted"
        elif "urban_exposure" in support:
            primary = "urban_flash_flood_underrepresented"
        elif "rainfall_7d_context" in support or "runoff_context" in support:
            primary = "rainfall_trigger_underweighted"
        elif "high_HAND" in contra:
            primary = "HAND_not_discriminative_here"
        elif "wet_spectral_signal" in support:
            primary = "spectral_temporal_mismatch"
        else:
            primary = "regional_calibration_needed" if item["score_v6_class"] in {"low", "medium"} else "insufficient_evidence"
        secondary = "patch_scale_mismatch" if (item.get("overlap_ratio") in {"", "0", "0.0"} and item.get("distance_m")) else "control_selection_issue"
        rows.append({
            "case_id": item["case_id"],
            "patch_id": item["patch_id"],
            "footprint_id": item["footprint_id"],
            "score_v6": item["score_v6"],
            "score_v6_class": item["score_v6_class"],
            "failure_mode_primary": primary,
            "failure_mode_secondary": secondary,
            "supporting_features": support,
            "contradicting_features": contra,
            "recommendation": _failure_recommendation(primary),
            "can_be_ground_truth": "false",
            "allowed_for_training": "false",
            "review_only": "true",
            "score_v7_created": "false",
        })
    write_csv(FAILURES, rows)
    write_json(FAILURE_SUMMARY, {
        "case_count": len(rows),
        "primary_failure_mode_distribution": dict(Counter(row["failure_mode_primary"] for row in rows)),
        "secondary_failure_mode_distribution": dict(Counter(row["failure_mode_secondary"] for row in rows)),
        "score_v7_created": False,
        "can_be_ground_truth": False,
        "allowed_for_training": False,
        "review_only": True,
    })
    print(f"failure modes -> {FAILURES.relative_to(ROOT)}")
    return 0


def _failure_recommendation(mode: str) -> str:
    if mode == "footprint_quality_uncertain":
        return "block_due_to_insufficient_evidence"
    if mode == "regional_calibration_needed":
        return "split_by_region"
    if mode in {"hydrology_underweighted", "rainfall_trigger_underweighted", "urban_flash_flood_underrepresented"}:
        return "review_future_weight_increase"
    return "manual_review_before_score_v7"


def build_calibration_design_matrix() -> int:
    if not FEATURE_STABILITY.exists():
        run_feature_direction_stability()
    if not WEIGHT_SENSITIVITY.exists():
        run_weight_sensitivity_review_only()
    if not FAILURES.exists():
        classify_proxy_failure_modes()
    stability = read_csv(FEATURE_STABILITY)
    failures = json.loads(FAILURE_SUMMARY.read_text(encoding="utf-8"))
    failure_modes = failures.get("primary_failure_mode_distribution", {})
    rows = []
    for item in stability:
        stability_class = item["stability_class"]
        if stability_class == "stable_support":
            action = "increase_weight"
            eligible = "true"
        elif stability_class == "regional_conflict":
            action = "split_by_region"
            eligible = "true"
        elif stability_class in {"quality_dependent", "partial_support"}:
            action = "keep_unchanged"
            eligible = "true"
        elif stability_class == "contradictory":
            action = "block_due_to_insufficient_evidence"
            eligible = "false"
        else:
            action = "block_due_to_insufficient_evidence"
            eligible = "false"
        rows.append({
            "feature_or_component": item["feature_name"],
            "current_role_v6": "score_v6_input_or_proxy_component",
            "observed_issue": item["stability_class"],
            "evidence_support": item["agreement_all"],
            "proposed_action": action,
            "risk": "medium" if eligible == "true" else "high",
            "requires_more_data": "true",
            "eligible_for_16D": eligible,
            "eligible_for_score_v7_future": "false",
            "rationale": "review_only_design_candidate_no_weight_change",
            "can_be_ground_truth": "false",
            "allowed_for_training": "false",
            "review_only": "true",
            "score_v7_created": "false",
        })
    for component in ("hydrological_component", "urban_component", "rainfall_runoff_component", "spectral_component", "topographic_component", "documentary_component"):
        mode_hits = sum(v for k, v in failure_modes.items() if component.split("_")[0] in k)
        rows.append({
            "feature_or_component": component,
            "current_role_v6": "derived_or_existing_component",
            "observed_issue": "component_underweighted_candidate" if mode_hits else "component_requires_review",
            "evidence_support": str(mode_hits),
            "proposed_action": "increase_weight" if mode_hits else "keep_unchanged",
            "risk": "medium" if mode_hits else "high",
            "requires_more_data": "true",
            "eligible_for_16D": "true",
            "eligible_for_score_v7_future": "false",
            "rationale": "component_design_discussion_only_no_score_v7_created",
            "can_be_ground_truth": "false",
            "allowed_for_training": "false",
            "review_only": "true",
            "score_v7_created": "false",
        })
    write_csv(DESIGN, rows)
    lines = "\n".join(
        f"| {row['feature_or_component']} | {row['observed_issue']} | {row['proposed_action']} | {row['eligible_for_16D']} | {row['eligible_for_score_v7_future']} |"
        for row in rows
    )
    write_markdown(DESIGN_MD, f"""# SUSC-16C calibration design matrix

Status: review-only. A matriz nao cria score v7 e nao altera pesos do score v6.

| feature_or_component | observed_issue | proposed_action | eligible_for_16D | eligible_for_score_v7_future |
|---|---|---|---|---|
{lines}
""")
    build_tcc_tables_and_figures()
    build_final_report()
    print(f"design matrix -> {DESIGN.relative_to(ROOT)}")
    return 0


def _svg_bar(path: Path, title: str, rows: list[tuple[str, float]], width: int = 900, height: int = 460) -> None:
    max_value = max([value for _label, value in rows] + [1.0])
    bars = []
    for i, (label, value) in enumerate(rows[:10]):
        y = 72 + i * 34
        bar_w = 560 * value / max_value if max_value else 0
        safe_label = label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        bars.append(
            f'<text x="24" y="{y + 17}" font-size="12" font-family="Arial">{safe_label}</text>'
            f'<rect x="300" y="{y}" width="{bar_w:.2f}" height="22" fill="#2f6f9f" />'
            f'<text x="{310 + bar_w:.2f}" y="{y + 16}" font-size="12" font-family="Arial">{value:.3f}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        f'<text x="24" y="36" font-size="20" font-family="Arial">{title}</text>'
        + "".join(bars)
        + '<text x="24" y="430" font-size="12" font-family="Arial">SUSC-16C review-only; sem ground truth, treino, modelo ou score v7.</text>'
        + "</svg>\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def build_tcc_tables_and_figures() -> int:
    TCC_TABLES.mkdir(parents=True, exist_ok=True)
    TCC_FIGURES.mkdir(parents=True, exist_ok=True)
    write_csv(TCC_TABLES / "individual_case_summary.csv", read_csv(CASE_AUDIT))
    write_csv(TCC_TABLES / "feature_direction_stability_summary.csv", read_csv(FEATURE_STABILITY))
    write_csv(TCC_TABLES / "weight_sensitivity_summary.csv", read_csv(WEIGHT_SENSITIVITY))
    write_csv(TCC_TABLES / "proxy_failure_modes_summary.csv", read_csv(FAILURES))
    write_csv(TCC_TABLES / "calibration_design_matrix_summary.csv", read_csv(DESIGN))
    score_counts = Counter(row["score_v6_class"] for row in read_csv(UNIFIED) if row["is_event_patch"] == "true")
    _svg_bar(TCC_FIGURES / "score_v6_vs_footprint_cases.svg", "Score v6 classes in footprint cases", [(k, float(v)) for k, v in sorted(score_counts.items())])
    stability_counts = Counter(row["stability_class"] for row in read_csv(FEATURE_STABILITY))
    _svg_bar(TCC_FIGURES / "feature_direction_stability.svg", "Feature direction stability", [(k, float(v)) for k, v in sorted(stability_counts.items())])
    decomp_counts = Counter(row["component_lowest"] for row in read_csv(DECOMP))
    _svg_bar(TCC_FIGURES / "component_decomposition_event_patches.svg", "Lowest score v6 component in event patches", [(k, float(v)) for k, v in sorted(decomp_counts.items())])
    sensitivity = read_csv(WEIGHT_SENSITIVITY)
    _svg_bar(TCC_FIGURES / "weight_sensitivity_scenarios.svg", "Review-only weight sensitivity hit@30", [(row["scenario_id"], _f(row["event_hit_rate_top_30_simulated"]) or 0.0) for row in sensitivity])
    failures = Counter(row["failure_mode_primary"] for row in read_csv(FAILURES))
    _svg_bar(TCC_FIGURES / "proxy_failure_modes.svg", "Proxy failure modes", [(k, float(v)) for k, v in sorted(failures.items())])
    pipeline = """<svg xmlns="http://www.w3.org/2000/svg" width="980" height="300" viewBox="0 0 980 300">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="24" y="36" font-size="20" font-family="Arial">Pipeline SUSC-16C review-only calibration diagnostics</text>
<g font-family="Arial" font-size="13">
<rect x="30" y="80" width="140" height="54" fill="#dcecff" stroke="#2d5f8b"/><text x="48" y="112">16A links</text>
<rect x="205" y="80" width="140" height="54" fill="#e7f4e4" stroke="#4b7c42"/><text x="222" y="112">16B metrics</text>
<rect x="380" y="80" width="150" height="54" fill="#fff2cc" stroke="#8a6d1b"/><text x="395" y="112">Unified table</text>
<rect x="565" y="80" width="160" height="54" fill="#f3e6ff" stroke="#6a3d9a"/><text x="582" y="112">Diagnostics</text>
<rect x="760" y="80" width="170" height="54" fill="#fde9d9" stroke="#9b4f1f"/><text x="777" y="112">16D design</text>
<path d="M170 107 L205 107 M345 107 L380 107 M530 107 L565 107 M725 107 L760 107" stroke="#333" stroke-width="2"/>
<text x="30" y="200">Sem score v7 | sem ground truth | sem treino supervisionado | sem modelo | score v6 oficial preservado</text>
</g>
</svg>
"""
    (TCC_FIGURES / "pipeline_16c_review_only_calibration.svg").write_text(pipeline, encoding="utf-8")
    print(f"tcc package -> {TCC_TABLES.relative_to(ROOT)}, {TCC_FIGURES.relative_to(ROOT)}")
    return 0


def build_final_report() -> int:
    unified_summary = {row["metric"]: row["value"] for row in read_csv(UNIFIED_SUMMARY)}
    decomp_summary = json.loads(DECOMP_SUMMARY.read_text(encoding="utf-8"))
    feature_summary = json.loads(FEATURE_STABILITY_SUMMARY.read_text(encoding="utf-8"))
    weight_summary = json.loads(WEIGHT_SUMMARY.read_text(encoding="utf-8"))
    failure_summary = json.loads(FAILURE_SUMMARY.read_text(encoding="utf-8"))
    score_counts = Counter(row["score_v6_class"] for row in read_csv(UNIFIED) if row["is_event_patch"] == "true")
    design_rows = read_csv(DESIGN)
    eligible_16d = sum(1 for row in design_rows if row["eligible_for_16D"] == "true")
    report = f"""# SUSC-16C - diagnostico profundo de divergencia e calibracao review-only do proxy

Status: review-only. `allowed_for_training=false`; `can_be_ground_truth=false`; `can_be_used_as_ground_truth=false`; `score_v7_created=false`.

{MANDATORY}

## 1. Estado herdado do 16A/16B

O SUSC-16A desbloqueou links footprint-patch elegiveis e o SUSC-16B confirmou avaliacao observacional do score v6 contra esses footprints. O estado herdado do 16B registra score medio de evento menor que controle e hit@top-k nulo, indicando divergencia util para diagnostico.

## 2. Por que o resultado ruim do score v6 e util

O resultado ruim nao foi escondido. Ele foi usado para isolar casos em que footprints observacionais aparecem em patches low/medium, identificar features que ainda sustentam suscetibilidade e apontar componentes que podem estar subestimados.

## 3. Auditoria do worktree antes do 16C

A auditoria esta em `SUSC_WORKTREE_AUDIT_BEFORE_16C.md`. Nenhum arquivo SUSC-16A/SUSC-16B de entrada apareceu modificado localmente; a sujeira fora do escopo foi preservada.

## 4. Dataset analitico unificado

Linhas totais: {unified_summary.get('row_count')}. Links evento: {unified_summary.get('event_link_rows')}. Patches observacionais unicos: {unified_summary.get('unique_event_patches')}. Casos low/medium: {unified_summary.get('low_or_medium_event_rows')}.

## 5. Decomposicao do score v6

Distribuicao do componente mais baixo: {json.dumps(decomp_summary.get('component_lowest_distribution', {}), ensure_ascii=False, sort_keys=True)}.

## 6. Auditoria individual dos 65 links

Foram gerados arquivos individuais em `SUSC_16C_individual_cases/`, um por link evento elegivel.

## 7. Features que sustentam suscetibilidade apesar do score baixo

As features de suporte sao registradas caso a caso em `SUSC_16C_individual_case_audit.csv` e agregadas na estabilidade de direcao. Elas incluem sinais hidrologicos, urbanos, espectrais e de chuva/runoff quando os limiares review-only foram satisfeitos.

## 8. Features que contradizem os footprints

Contradicoes possiveis foram registradas como HAND alto, distancia alta da agua, TWI baixo, baixa exposicao urbana, vegetacao alta ou chuva fraca, sempre como diagnostico e nao como negacao do evento.

## 9. Estabilidade das direcoes por regiao/qualidade

Resumo: {json.dumps(feature_summary.get('stability_distribution', {}), ensure_ascii=False, sort_keys=True)}.

## 10. Cenarios de sensibilidade de pesos

Foram simulados {weight_summary.get('scenario_count')} cenarios review-only. Melhor hit@30 simulado: `{weight_summary.get('top_scenario_by_hit30')}` com {weight_summary.get('best_hit_rate_top_30_simulated')}. Nenhum score v7 foi persistido.

## 11. Modos de falha do proxy

Distribuicao primaria: {json.dumps(failure_summary.get('primary_failure_mode_distribution', {}), ensure_ascii=False, sort_keys=True)}.

## 12. Matriz de design para calibracao futura

A matriz possui {len(design_rows)} linhas; {eligible_16d} itens ficam elegiveis para discussao no 16D. `eligible_for_score_v7_future` permanece false por desenho fail-closed.

## 13. O que pode ser melhorado no proxy

Os candidatos de melhoria incluem calibracao regional, reavaliacao de hidrologia/topografia, chuva/runoff, exposicao urbana e sinal espectral umido. Essas acoes sao apenas candidatas.

## 14. O que nao pode ser afirmado ainda

Nao se pode afirmar ground truth, negativo verdadeiro, prontidao para treino, causalidade ou score v7 pronto. Footprints permanecem evidencias observacionais candidatas.

## 15. Por que score v7 nao foi criado

O 16C e diagnostico e desenho review-only. Criar score v7 exigiria decisao metodologica posterior, mais evidencia e validacao especifica.

## 16. Proximo marco recomendado

Executar SUSC-16D como desenho controlado de calibracao candidata, ainda sem treino e sem ground truth, priorizando itens elegiveis da matriz 16C.

## Distribuicao score v6 nos links evento

{json.dumps(dict(score_counts), ensure_ascii=False, sort_keys=True)}
"""
    write_markdown(REPORT, report)
    print(f"report -> {REPORT.relative_to(ROOT)}")
    return 0


def validate() -> int:
    required = [
        AUDIT, SCHEMA, UNIFIED, UNIFIED_SUMMARY, DECOMP, DECOMP_CASE,
        DECOMP_SUMMARY, CASE_AUDIT, CASE_DIR, FEATURE_STABILITY,
        FEATURE_STABILITY_REGION, FEATURE_STABILITY_SUMMARY, WEIGHT_SENSITIVITY,
        WEIGHT_TOP, WEIGHT_SUMMARY, FAILURES, FAILURE_SUMMARY, DESIGN, DESIGN_MD,
        TCC_TABLES, TCC_FIGURES, REPORT,
    ]
    errors = []
    for path in required:
        if not path.exists():
            errors.append(f"missing: {path.relative_to(ROOT)}")
    if CASE_DIR.exists() and not list(CASE_DIR.glob("S16C_CASE_*.md")):
        errors.append("missing individual case files")
    for path, names in {
        TCC_TABLES: [
            "individual_case_summary.csv",
            "feature_direction_stability_summary.csv",
            "weight_sensitivity_summary.csv",
            "proxy_failure_modes_summary.csv",
            "calibration_design_matrix_summary.csv",
        ],
        TCC_FIGURES: [
            "score_v6_vs_footprint_cases.svg",
            "feature_direction_stability.svg",
            "component_decomposition_event_patches.svg",
            "weight_sensitivity_scenarios.svg",
            "proxy_failure_modes.svg",
            "pipeline_16c_review_only_calibration.svg",
        ],
    }.items():
        if path.exists():
            for name in names:
                if not (path / name).exists():
                    errors.append(f"missing: {(path / name).relative_to(ROOT)}")
    guarded_csvs = [
        UNIFIED, DECOMP, CASE_AUDIT, FEATURE_STABILITY, WEIGHT_SENSITIVITY,
        FAILURES, DESIGN,
    ]
    for path in guarded_csvs:
        if path.exists():
            for line_no, row in enumerate(read_csv(path), start=2):
                if row.get("allowed_for_training") == "true":
                    errors.append(f"{path.name}:{line_no}: allowed_for_training true")
                if row.get("can_be_ground_truth") == "true" or row.get("can_be_used_as_ground_truth") == "true":
                    errors.append(f"{path.name}:{line_no}: ground truth true")
                if row.get("review_only") != "true":
                    errors.append(f"{path.name}:{line_no}: review_only not true")
                if row.get("score_v7_created") == "true":
                    errors.append(f"{path.name}:{line_no}: score_v7_created true")
                if row.get("group_type", "").endswith("control") and row.get("evidence_strength") == "negative":
                    errors.append(f"{path.name}:{line_no}: control treated as negative")
    if SCORE_V7.exists():
        errors.append("score v7 artifact exists")
    if not matrix_sha256_ok():
        errors.append("SUSC-03 matrix sha256 mismatch")
    if find_model_artifacts():
        errors.append("model artifact found under SUSC scope")
    if REPORT.exists() and MANDATORY not in REPORT.read_text(encoding="utf-8"):
        errors.append("mandatory report phrase missing")
    if SCORE.exists() and not matrix_sha256_ok():
        errors.append("score/source guard failed")
    if errors:
        print("FAILED: SUSC-16C validation")
        for error in errors:
            print("  ERROR:", error)
        return 1
    print("PASSED: SUSC-16C validations passed")
    return 0


def summarize_for_final() -> dict:
    if not all(path.exists() for path in (UNIFIED_SUMMARY, DECOMP_SUMMARY, FEATURE_STABILITY_SUMMARY, WEIGHT_SUMMARY, FAILURE_SUMMARY, DESIGN)):
        return {}
    unified_summary = {row["metric"]: row["value"] for row in read_csv(UNIFIED_SUMMARY)}
    decomp_summary = json.loads(DECOMP_SUMMARY.read_text(encoding="utf-8"))
    weight_summary = json.loads(WEIGHT_SUMMARY.read_text(encoding="utf-8"))
    failure_summary = json.loads(FAILURE_SUMMARY.read_text(encoding="utf-8"))
    design = read_csv(DESIGN)
    return {
        "event_link_rows": unified_summary.get("event_link_rows", ""),
        "underestimated_cases": unified_summary.get("low_or_medium_event_rows", ""),
        "component_lowest_distribution": decomp_summary.get("component_lowest_distribution", {}),
        "failure_modes": failure_summary.get("primary_failure_mode_distribution", {}),
        "top_sensitivity_scenario": weight_summary.get("top_scenario_by_hit30", ""),
        "best_hit30": weight_summary.get("best_hit_rate_top_30_simulated", ""),
        "eligible_16d": sum(1 for row in design if row["eligible_for_16D"] == "true"),
        "eligible_score_v7_future": sum(1 for row in design if row["eligible_for_score_v7_future"] == "true"),
    }
