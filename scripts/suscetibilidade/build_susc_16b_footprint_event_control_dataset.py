"""SUSC-16B footprint score v6 observational evaluation.

The module is intentionally review-only. It derives event/control rows,
metrics, feature contrasts, quality audits, recommendations, TCC tables/figures,
and the final report from SUSC-16A linkage plus the existing score v6 matrix.
It does not create score v7, ground truth, negative labels, training data, or a
model artifact.
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
SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_16b_footprint_score_evaluation_schema_v1.json"

LINKAGE = DAT / "susc_16a_footprint_patch_linkage_v1.csv"
CATALOG = DAT / "susc_16a_observed_footprint_catalog_v1.csv"
MATRIX = DAT / "susc_features_by_patch_v1.csv"
SCORE = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"

DATASET = DAT / "susc_16b_footprint_event_control_dataset_v1.csv"
DATASET_SUMMARY = OUT / "SUSC_16B_footprint_event_control_dataset_summary.csv"
METRICS = OUT / "SUSC_16B_score_v6_footprint_evaluation_metrics.csv"
METRICS_REGION = OUT / "SUSC_16B_score_v6_footprint_evaluation_by_region.csv"
METRICS_SUMMARY = OUT / "SUSC_16B_score_v6_footprint_evaluation_summary.json"
FEATURE_CONTRAST = OUT / "SUSC_16B_feature_contrast_against_footprints.csv"
FEATURE_CONTRAST_REGION = OUT / "SUSC_16B_feature_contrast_by_region.csv"
FEATURE_AUDIT = OUT / "SUSC_16B_feature_direction_audit.json"
LOW_SCORE_CASES = OUT / "SUSC_16B_low_score_footprint_case_audit.csv"
LOW_SCORE_REPORT = OUT / "SUSC_16B_low_score_footprint_case_report.md"
QUALITY_AUDIT = OUT / "SUSC_16B_footprint_evidence_quality_audit.csv"
QUALITY_SUMMARY = OUT / "SUSC_16B_footprint_evidence_quality_summary.json"
RECOMMENDATIONS = OUT / "SUSC_16B_proxy_calibration_recommendations.csv"
RECOMMENDATIONS_MD = OUT / "SUSC_16B_proxy_calibration_recommendations.md"
READINESS = OUT / "SUSC_16B_readiness_for_16C_and_score_v7.csv"
READINESS_MD = OUT / "SUSC_16B_readiness_for_16C_and_score_v7.md"
TABLE_DIR = OUT / "SUSC_16B_tcc_tables"
FIG_DIR = OUT / "SUSC_16B_tcc_figures"
REPORT = OUT / "SUSC_16B_score_v6_footprint_observational_evaluation_report.md"

MANDATORY = (
    "O SUSC-16B avalia o score v6 contra footprints observacionais elegíveis "
    "em modo review-only. A etapa não cria ground truth, não libera treino "
    "supervisionado, não cria score v7 e trata controles como ausência "
    "documentada, não como negativos verdadeiros."
)

DATASET_FIELDS = [
    "row_id", "group_type", "patch_id", "region", "footprint_id", "event_date",
    "footprint_type", "evidence_strength", "score_v6", "score_v6_class",
    "hand_mean", "hand_min", "slope_mean", "elevation_mean", "twi_mean",
    "distance_to_water_mean", "flow_accumulation_mean", "ndvi_mean",
    "mndwi_mean", "ndbi_mean", "urban_prop", "vegetation_prop", "water_prop",
    "chirps_3d_mm", "chirps_7d_mm", "chirps_30d_mm", "runoff_context_7d",
    "control_selection_reason", "can_be_ground_truth", "allowed_for_training",
    "review_only",
]

FEATURE_MAP = {
    "region": "regiao",
    "flow_accumulation_mean": "flow_acc_log_mean",
    "water_prop": "water_occurrence_patch",
    "hand_min": "hand_mean",
}

EXPECTED_DIRECTIONS = {
    "hand_mean": "lower_in_event",
    "slope_mean": "lower_in_event",
    "distance_to_water_mean": "lower_in_event",
    "twi_mean": "higher_in_event",
    "flow_accumulation_mean": "higher_in_event",
    "urban_prop": "higher_in_event",
    "ndbi_mean": "higher_in_event",
    "ndvi_mean": "lower_in_event",
    "mndwi_mean": "higher_in_event",
    "chirps_3d_mm": "higher_in_event",
    "chirps_7d_mm": "higher_in_event",
    "chirps_30d_mm": "higher_in_event",
    "runoff_context_7d": "higher_in_event",
}


def _f(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def _fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(round(value, 6))


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


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = _mean(values)
    if mean is None:
        return None
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _value(row: dict, field: str) -> str:
    return row.get(field, row.get(FEATURE_MAP.get(field, field), ""))


def _score_value(row: dict) -> str:
    return row.get("susc_score_v6_candidate", row.get("score_v6", ""))


def _score_class(row: dict) -> str:
    return row.get("susc_class_v6_candidate", row.get("score_v6_class", ""))


def _load_inputs() -> tuple[list[dict], list[dict], list[dict], dict[str, dict], dict[str, dict]]:
    for path in (SCHEMA, LINKAGE, CATALOG, MATRIX, SCORE):
        if not path.exists():
            raise FileNotFoundError(path)
    links = read_csv(LINKAGE)
    catalog = read_csv(CATALOG)
    matrix = read_csv(MATRIX)
    matrix_by_patch = {row["patch_id"]: row for row in matrix}
    score_by_patch = {row["patch_id"]: row for row in read_csv(SCORE)}
    return links, catalog, matrix, matrix_by_patch, score_by_patch


def _eligible_links(links: list[dict]) -> list[dict]:
    return [
        row for row in links
        if row.get("eligible_for_observational_evaluation") == "true"
        and row.get("patch_id")
        and row.get("patch_id") != "REGION_LEVEL_NO_PATCH_RESOLUTION"
    ]


def _event_dataset_row(index: int, link: dict, matrix: dict, score: dict) -> dict:
    row = {
        "row_id": f"S16B_EVT_{index:05d}",
        "group_type": "observed_footprint_patch",
        "patch_id": link["patch_id"],
        "region": link.get("region") or matrix.get("regiao", ""),
        "footprint_id": link.get("footprint_id", ""),
        "event_date": link.get("event_date", ""),
        "footprint_type": link.get("footprint_type", ""),
        "evidence_strength": link.get("evidence_strength", ""),
        "score_v6": link.get("score_v6") or _score_value(score),
        "score_v6_class": link.get("score_v6_class") or _score_class(score),
        "control_selection_reason": "observed_footprint_patch_from_SUSC_16A_linkage",
        "can_be_ground_truth": "false",
        "allowed_for_training": "false",
        "review_only": "true",
    }
    for field in DATASET_FIELDS:
        if field not in row:
            row[field] = _value(matrix, field)
    return row


def _control_dataset_row(index: int, patch: dict, score: dict, reason: str) -> dict:
    row = {
        "row_id": f"S16B_CTL_{index:05d}",
        "group_type": "no_documented_footprint_control",
        "patch_id": patch["patch_id"],
        "region": patch.get("regiao", ""),
        "footprint_id": "",
        "event_date": "",
        "footprint_type": "",
        "evidence_strength": "no_documented_eligible_footprint",
        "score_v6": _score_value(score),
        "score_v6_class": _score_class(score),
        "control_selection_reason": reason,
        "can_be_ground_truth": "false",
        "allowed_for_training": "false",
        "review_only": "true",
    }
    for field in DATASET_FIELDS:
        if field not in row:
            row[field] = _value(patch, field)
    return row


def build_event_control_dataset() -> int:
    links, _catalog, matrix, matrix_by_patch, score_by_patch = _load_inputs()
    eligible = _eligible_links(links)
    event_patch_ids = {row["patch_id"] for row in eligible}
    event_regions = {row.get("region") or matrix_by_patch.get(row["patch_id"], {}).get("regiao", "") for row in eligible}
    rows: list[dict] = []
    for link in eligible:
        rows.append(_event_dataset_row(
            len(rows),
            link,
            matrix_by_patch.get(link["patch_id"], {}),
            score_by_patch.get(link["patch_id"], {}),
        ))

    control_candidates = [
        row for row in matrix
        if row["patch_id"] not in event_patch_ids
        and row.get("regiao", "") in event_regions
    ]
    if len(control_candidates) < len(event_patch_ids):
        reason = "same_region_no_documented_footprint_control_insufficient_for_1_to_1"
    else:
        reason = "same_region_no_documented_footprint_control_1_to_1"
    ordered_controls = sorted(
        control_candidates,
        key=lambda row: (
            row.get("regiao", ""),
            -(_f(score_by_patch.get(row["patch_id"], {}).get("susc_score_v6_candidate")) or -1.0),
            row["patch_id"],
        ),
    )
    for patch in ordered_controls[:len(event_patch_ids)]:
        rows.append(_control_dataset_row(len(rows), patch, score_by_patch.get(patch["patch_id"], {}), reason))

    write_csv(DATASET, rows, DATASET_FIELDS)
    summary_rows = [
        {"metric": "event_link_rows", "value": sum(1 for row in rows if row["group_type"] == "observed_footprint_patch"), "review_only": "true"},
        {"metric": "unique_event_patches", "value": len(event_patch_ids), "review_only": "true"},
        {"metric": "control_patch_rows", "value": sum(1 for row in rows if row["group_type"] == "no_documented_footprint_control"), "review_only": "true"},
        {"metric": "same_region_control_candidates", "value": len(control_candidates), "review_only": "true"},
        {"metric": "control_semantics", "value": "no_documented_footprint_control_not_true_negative", "review_only": "true"},
        {"metric": "feature_mapping", "value": json.dumps(FEATURE_MAP, ensure_ascii=False, sort_keys=True), "review_only": "true"},
        {"metric": "score_v7_created", "value": "false", "review_only": "true"},
        {"metric": "can_be_ground_truth", "value": "false", "review_only": "true"},
        {"metric": "allowed_for_training", "value": "false", "review_only": "true"},
    ]
    write_csv(DATASET_SUMMARY, summary_rows, ["metric", "value", "review_only"])
    print(f"dataset -> {DATASET.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


def _dataset_groups() -> tuple[list[dict], list[dict]]:
    rows = read_csv(DATASET)
    events = [row for row in rows if row.get("group_type") == "observed_footprint_patch"]
    controls = [row for row in rows if row.get("group_type") == "no_documented_footprint_control"]
    return events, controls


def _score_list(rows: list[dict]) -> list[float]:
    return [v for v in (_f(row.get("score_v6")) for row in rows) if v is not None]


def run_score_evaluation() -> int:
    if not DATASET.exists():
        build_event_control_dataset()
    events, controls = _dataset_groups()
    scores_all = read_csv(SCORE)
    event_patch_ids = {row["patch_id"] for row in events}
    control_patch_ids = {row["patch_id"] for row in controls}
    event_scores = _score_list(events)
    control_scores = _score_list(controls)
    all_ranked = sorted(
        scores_all,
        key=lambda row: (-(_f(row.get("susc_score_v6_candidate")) or -1.0), row["patch_id"]),
    )

    def top_stats(k: int) -> tuple[float, float]:
        top = all_ranked[:k]
        observed = sum(1 for row in top if row["patch_id"] in event_patch_ids)
        hit = observed / k if k else 0.0
        baseline = len(event_patch_ids) / len(scores_all) if scores_all else 0.0
        enrichment = hit / baseline if baseline else 0.0
        return round(hit, 6), round(enrichment, 6)

    hit10, enr10 = top_stats(10)
    hit20, enr20 = top_stats(20)
    hit30, enr30 = top_stats(30)
    mean_event = _mean(event_scores)
    mean_control = _mean(control_scores)
    diff = mean_event - mean_control if mean_event is not None and mean_control is not None else None
    class_event = Counter(row.get("score_v6_class", "") for row in events)
    class_control = Counter(row.get("score_v6_class", "") for row in controls)
    flags = []
    if hit10 == 0 or hit20 == 0 or hit30 == 0:
        flags.append("score_v6_underestimation_possible")
    if diff is not None and diff <= 0:
        flags.append("score_v6_direction_conflict")

    metrics = {
        "event_patch_count": len(events),
        "control_patch_count": len(controls),
        "unique_event_patches": len(event_patch_ids),
        "unique_control_patches": len(control_patch_ids),
        "mean_score_event": mean_event,
        "median_score_event": _median(event_scores),
        "mean_score_control": mean_control,
        "median_score_control": _median(control_scores),
        "score_diff_event_minus_control": diff,
        "event_score_class_distribution": dict(class_event),
        "control_score_class_distribution": dict(class_control),
        "hit_rate_top_10": hit10,
        "hit_rate_top_20": hit20,
        "hit_rate_top_30": hit30,
        "event_enrichment_top_10": enr10,
        "event_enrichment_top_20": enr20,
        "event_enrichment_top_30": enr30,
        "low_score_event_patch_count": class_event.get("low", 0),
        "medium_score_event_patch_count": class_event.get("medium", 0),
        "high_score_event_patch_count": class_event.get("high", 0),
        "interpretation_flags": flags,
        "score_v7_created": False,
        "can_be_ground_truth": False,
        "allowed_for_training": False,
        "review_only": True,
    }
    rows = [
        {
            "metric": key,
            "value": json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else _fmt(value) if isinstance(value, float) else value,
            "review_only": "true",
        }
        for key, value in metrics.items()
    ]
    write_csv(METRICS, rows, ["metric", "value", "review_only"])

    region_rows = []
    for region in sorted({row.get("region", "") for row in events + controls}):
        ev = [row for row in events if row.get("region", "") == region]
        ct = [row for row in controls if row.get("region", "") == region]
        evs, cts = _score_list(ev), _score_list(ct)
        em, cm = _mean(evs), _mean(cts)
        region_rows.append({
            "region": region,
            "event_patch_count": len(ev),
            "control_patch_count": len(ct),
            "unique_event_patches": len({row["patch_id"] for row in ev}),
            "unique_control_patches": len({row["patch_id"] for row in ct}),
            "mean_score_event": _fmt(em),
            "mean_score_control": _fmt(cm),
            "score_diff_event_minus_control": _fmt(em - cm if em is not None and cm is not None else None),
            "review_only": "true",
        })
    write_csv(METRICS_REGION, region_rows)
    write_json(METRICS_SUMMARY, metrics)
    print("score evaluation ->", METRICS.relative_to(ROOT))
    return 0


def _feature_values(rows: list[dict], feature: str) -> list[float]:
    values = [_f(row.get(feature)) for row in rows]
    return [value for value in values if value is not None]


def _direction(diff: float | None, expected: str) -> tuple[str, str, str]:
    if diff is None:
        return "", "inconclusive", "feature_direction_inconclusive"
    observed = "higher_in_event" if diff > 0 else "lower_in_event" if diff < 0 else "equal"
    if observed == "equal":
        return observed, "inconclusive", "feature_direction_inconclusive"
    status = "feature_direction_agreement" if observed == expected else "feature_direction_divergence"
    return observed, "true" if observed == expected else "false", status


def run_feature_contrast() -> int:
    if not DATASET.exists():
        build_event_control_dataset()
    events, controls = _dataset_groups()
    rows = []
    for feature, expected in EXPECTED_DIRECTIONS.items():
        ev, ct = _feature_values(events, feature), _feature_values(controls, feature)
        em, cm = _mean(ev), _mean(ct)
        diff = em - cm if em is not None and cm is not None else None
        esd, csd = _std(ev), _std(ct)
        pooled = math.sqrt(((esd or 0.0) ** 2 + (csd or 0.0) ** 2) / 2) if esd is not None and csd is not None else None
        effect = diff / pooled if diff is not None and pooled not in (None, 0.0) else None
        observed, agreement, status = _direction(diff, expected)
        rows.append({
            "feature_name": feature,
            "mapped_source_field": FEATURE_MAP.get(feature, feature),
            "event_mean": _fmt(em),
            "control_mean": _fmt(cm),
            "event_median": _fmt(_median(ev)),
            "control_median": _fmt(_median(ct)),
            "difference_event_minus_control": _fmt(diff),
            "expected_direction_for_susceptibility": expected,
            "observed_direction": observed,
            "direction_agreement": status,
            "effect_size_simple": _fmt(effect),
            "sample_size_event": len(ev),
            "sample_size_control": len(ct),
            "interpretation": status if status != "feature_direction_agreement" else "feature_direction_agreement_review_only",
            "can_be_ground_truth": "false",
            "allowed_for_training": "false",
            "review_only": "true",
        })
    write_csv(FEATURE_CONTRAST, rows)

    region_rows = []
    for region in sorted({row.get("region", "") for row in events + controls}):
        ev_region = [row for row in events if row.get("region", "") == region]
        ct_region = [row for row in controls if row.get("region", "") == region]
        for feature, expected in EXPECTED_DIRECTIONS.items():
            ev, ct = _feature_values(ev_region, feature), _feature_values(ct_region, feature)
            em, cm = _mean(ev), _mean(ct)
            diff = em - cm if em is not None and cm is not None else None
            observed, _agree, status = _direction(diff, expected)
            region_rows.append({
                "region": region,
                "feature_name": feature,
                "event_mean": _fmt(em),
                "control_mean": _fmt(cm),
                "difference_event_minus_control": _fmt(diff),
                "expected_direction_for_susceptibility": expected,
                "observed_direction": observed,
                "direction_agreement": status,
                "sample_size_event": len(ev),
                "sample_size_control": len(ct),
                "review_only": "true",
            })
    write_csv(FEATURE_CONTRAST_REGION, region_rows)

    agree = [row["feature_name"] for row in rows if row["direction_agreement"] == "feature_direction_agreement"]
    diverge = [row["feature_name"] for row in rows if row["direction_agreement"] == "feature_direction_divergence"]
    audit = {
        "artifact": "SUSC-16B feature direction audit",
        "features_agreeing": agree,
        "features_diverging": diverge,
        "feature_mapping": FEATURE_MAP,
        "event_patch_count": len(events),
        "control_patch_count": len(controls),
        "can_be_ground_truth": False,
        "allowed_for_training": False,
        "review_only": True,
        "score_v7_created": False,
    }
    write_json(FEATURE_AUDIT, audit)
    print("feature contrast ->", FEATURE_CONTRAST.relative_to(ROOT))
    return 0


def build_low_score_case_audit() -> int:
    if not FEATURE_CONTRAST.exists():
        run_feature_contrast()
    events, controls = _dataset_groups()
    contrast = {row["feature_name"]: row for row in read_csv(FEATURE_CONTRAST)}
    control_medians = {feature: _median(_feature_values(controls, feature)) for feature in EXPECTED_DIRECTIONS}
    score_by_patch = {row["patch_id"]: row for row in read_csv(SCORE)}
    case_rows = []
    for row in events:
        if row.get("score_v6_class") not in {"low", "medium"}:
            continue
        susceptible, reducers = [], []
        for feature, expected in EXPECTED_DIRECTIONS.items():
            value = _f(row.get(feature))
            median = control_medians.get(feature)
            if value is None or median is None:
                continue
            indicates = (expected == "lower_in_event" and value < median) or (expected == "higher_in_event" and value > median)
            (susceptible if indicates else reducers).append(feature)
        score_row = score_by_patch.get(row["patch_id"], {})
        drivers = score_row.get("top5_feature_contributions", "")
        if len(susceptible) >= 4 and row.get("score_v6_class") == "low":
            reason = "score_weight_underestimates_local_factor"
        elif row.get("event_date") == "":
            reason = "spectral_feature_temporal_mismatch"
        elif "chirps" in ";".join(reducers) or "runoff" in ";".join(reducers):
            reason = "rainfall_driver_not_weighted_enough"
        elif row.get("overlap_ratio", "") == "" and row.get("distance_m", "") == "":
            reason = "patch_scale_mismatch"
        else:
            reason = "needs_manual_review"
        case_rows.append({
            "patch_id": row["patch_id"],
            "footprint_id": row["footprint_id"],
            "region": row["region"],
            "score_v6": row["score_v6"],
            "score_v6_class": row["score_v6_class"],
            "footprint_type": row["footprint_type"],
            "evidence_strength": row["evidence_strength"],
            "overlap_ratio": next((link.get("overlap_ratio", "") for link in read_csv(LINKAGE) if link.get("patch_id") == row["patch_id"] and link.get("footprint_id") == row["footprint_id"]), ""),
            "distance_m": next((link.get("distance_m", "") for link in read_csv(LINKAGE) if link.get("patch_id") == row["patch_id"] and link.get("footprint_id") == row["footprint_id"]), ""),
            "main_low_score_drivers": drivers,
            "features_that_indicate_susceptibility": ";".join(susceptible),
            "features_that_reduce_score": ";".join(reducers),
            "possible_reason": reason,
            "can_be_ground_truth": "false",
            "allowed_for_training": "false",
            "review_only": "true",
        })
    write_csv(LOW_SCORE_CASES, case_rows)
    counts = Counter(row["possible_reason"] for row in case_rows)
    reason_lines = "\n".join(f"- {key}: {value}" for key, value in sorted(counts.items()))
    write_markdown(LOW_SCORE_REPORT, f"""# SUSC-16B - casos footprint com score baixo/medio

Status: review-only. Estes casos nao sao ground truth e nao liberam treino.

Casos auditados: {len(case_rows)}.

## Possiveis razoes
{reason_lines or "- Nenhum caso baixo/medio encontrado."}

## Leitura
Quando o score v6 baixo/medio diverge de um footprint elegivel, a divergencia
fica registrada para revisao. Nenhum peso foi alterado e nenhum score v7 foi
criado neste marco.
""")
    print("low score cases ->", LOW_SCORE_CASES.relative_to(ROOT))
    return 0


def audit_footprint_quality() -> int:
    _links, catalog, _matrix, _matrix_by_patch, _score_by_patch = _load_inputs()
    rows = []
    for item in catalog:
        eligible = item.get("eligible_for_patch_observational_evaluation") == "true"
        is_flood = item.get("footprint_type") == "official_flood_footprint"
        is_risk = "risk" in item.get("footprint_type", "")
        is_susc = "suscept" in item.get("footprint_type", "") or "susc" in item.get("footprint_type", "")
        direct = item.get("geometry_type", "") not in {"", "unknown"} and bool(item.get("bbox") or item.get("geojson_ref") or item.get("wkt"))
        if not eligible:
            tier = "E_insufficient"
            warning = "not_eligible_for_patch_observational_evaluation"
        elif item.get("source_family") == "remote_sensing":
            tier = "C_remote_sensing_candidate"
            warning = "candidate_remote_sensing_review_only"
        elif is_flood and direct:
            tier = "B_official_or_technical_flood_footprint"
            warning = "moderate_candidate_not_ground_truth"
        elif is_risk or is_susc:
            tier = "D_contextual_or_risk_reference"
            warning = "context_only_no_strong_conclusion"
        else:
            tier = "E_insufficient"
            warning = "insufficient_geometry_or_source_specificity"
        rows.append({
            "footprint_id": item.get("footprint_id", ""),
            "source_type": item.get("source_family", ""),
            "source_confidence": "medium" if eligible else "low",
            "parse_confidence": "medium" if direct else "low",
            "is_direct_event_geometry": str(direct).lower(),
            "is_flood_footprint": str(is_flood).lower(),
            "is_risk_area": str(is_risk).lower(),
            "is_susceptibility_reference": str(is_susc).lower(),
            "event_date_available": str(bool(item.get("event_date"))).lower(),
            "geometry_quality": "polygon_or_bbox_available" if direct else "insufficient_or_context_only",
            "eligible_for_observational_evaluation": str(eligible).lower(),
            "quality_tier": tier,
            "quality_warning": warning,
            "can_be_ground_truth": "false",
            "allowed_for_training": "false",
            "review_only": "true",
        })
    write_csv(QUALITY_AUDIT, rows)
    counts = Counter(row["quality_tier"] for row in rows)
    eligible_counts = Counter(row["quality_tier"] for row in rows if row["eligible_for_observational_evaluation"] == "true")
    summary = {
        "artifact": "SUSC-16B footprint evidence quality audit",
        "footprint_count": len(rows),
        "eligible_footprint_count": sum(1 for row in rows if row["eligible_for_observational_evaluation"] == "true"),
        "quality_tier_distribution": dict(counts),
        "eligible_quality_tier_distribution": dict(eligible_counts),
        "exploratory_because_candidate_quality": any(tier in eligible_counts for tier in ("C_remote_sensing_candidate", "D_contextual_or_risk_reference")),
        "can_be_ground_truth": False,
        "allowed_for_training": False,
        "review_only": True,
        "score_v7_created": False,
    }
    write_json(QUALITY_SUMMARY, summary)
    print("quality audit ->", QUALITY_AUDIT.relative_to(ROOT))
    return 0


def build_recommendations() -> int:
    if not FEATURE_CONTRAST.exists():
        run_feature_contrast()
    if not QUALITY_SUMMARY.exists():
        audit_footprint_quality()
    contrast = read_csv(FEATURE_CONTRAST)
    quality = json.loads(QUALITY_SUMMARY.read_text(encoding="utf-8"))
    eligible_quality = quality.get("eligible_quality_tier_distribution", {})
    quality_abc = sum(eligible_quality.get(tier, 0) for tier in (
        "A_direct_observed_flood_geometry",
        "B_official_or_technical_flood_footprint",
        "C_remote_sensing_candidate",
    ))
    rows = []
    for row in contrast:
        sample = min(int(row.get("sample_size_event") or 0), int(row.get("sample_size_control") or 0))
        agrees = row.get("direction_agreement") == "feature_direction_agreement"
        evidence_strength = "moderate_review_only" if quality_abc >= 10 and sample >= 20 else "limited_review_only"
        if sample < 20 or quality_abc < 10:
            rec = "block_change_insufficient_evidence"
        elif agrees:
            rec = "increase_weight_candidate" if abs(_f(row.get("effect_size_simple")) or 0.0) >= 0.25 else "keep_weight_candidate"
        else:
            rec = "review_feature_candidate"
        safe = agrees and evidence_strength == "moderate_review_only" and sample >= 20
        rows.append({
            "feature_name": row["feature_name"],
            "current_role_in_score_v6": "score_v6_input_or_proxy_component",
            "observed_direction": row["observed_direction"],
            "expected_direction": row["expected_direction_for_susceptibility"],
            "agreement_status": row["direction_agreement"],
            "evidence_strength": evidence_strength,
            "sample_size": sample,
            "recommendation": rec,
            "rationale": "review_only_candidate; no automatic weight change",
            "safe_for_score_v7_discussion": str(safe).lower(),
            "can_be_ground_truth": "false",
            "allowed_for_training": "false",
            "review_only": "true",
        })
    write_csv(RECOMMENDATIONS, rows)
    counts = Counter(row["recommendation"] for row in rows)
    safe_count = sum(1 for row in rows if row["safe_for_score_v7_discussion"] == "true")
    lines = "\n".join(f"| {key} | {value} |" for key, value in sorted(counts.items()))
    write_markdown(RECOMMENDATIONS_MD, f"""# SUSC-16B - recomendacoes review-only para futura calibracao

Status: review-only. Nenhum score v7 foi criado.

| recomendacao | n |
|---|---:|
{lines}

Features seguras apenas para discussao futura de score v7: {safe_count}.

As recomendacoes sao candidatas de auditoria. Elas nao alteram pesos, nao
criam modelo, nao criam ground truth e nao liberam treino supervisionado.
""")
    print("recommendations ->", RECOMMENDATIONS.relative_to(ROOT))
    return 0


def run_readiness() -> int:
    if not METRICS_SUMMARY.exists():
        run_score_evaluation()
    if not FEATURE_AUDIT.exists():
        run_feature_contrast()
    if not QUALITY_SUMMARY.exists():
        audit_footprint_quality()
    if not RECOMMENDATIONS.exists():
        build_recommendations()
    metrics = json.loads(METRICS_SUMMARY.read_text(encoding="utf-8"))
    quality = json.loads(QUALITY_SUMMARY.read_text(encoding="utf-8"))
    feature_audit = json.loads(FEATURE_AUDIT.read_text(encoding="utf-8"))
    links_ok = int(metrics["event_patch_count"]) >= 50
    footprints_ok = int(quality["eligible_footprint_count"]) >= 10
    contrast_ok = FEATURE_CONTRAST.exists()
    quality_ok = QUALITY_AUDIT.exists()
    ready_16c = links_ok and footprints_ok and contrast_ok and quality_ok
    quality_dist = quality.get("eligible_quality_tier_distribution", {})
    ab = quality_dist.get("A_direct_observed_flood_geometry", 0) + quality_dist.get("B_official_or_technical_flood_footprint", 0)
    event_count_ok = int(metrics["unique_event_patches"]) >= 20
    dominant_ab = ab > int(quality["eligible_footprint_count"]) / 2
    strong_conflict = bool(feature_audit.get("features_diverging")) or "score_v6_direction_conflict" in metrics.get("interpretation_flags", [])
    ready_v7 = event_count_ok and dominant_ab and not strong_conflict and False
    rows = [{
        "criterion": "ready_for_16C_proxy_calibration_review",
        "value": str(ready_16c).lower(),
        "reason": ">=50 eligible links, >=10 eligible footprints, feature contrast and quality audit executed",
        "review_only": "true",
    }, {
        "criterion": "ready_for_score_v7_candidate",
        "value": str(ready_v7).lower(),
        "reason": "default false in SUSC-16B; score/features conflicts or review-only status block automatic v7",
        "review_only": "true",
    }, {
        "criterion": "score_v7_created",
        "value": "false",
        "reason": "SUSC-16B is evaluation, not calibration",
        "review_only": "true",
    }]
    write_csv(READINESS, rows)
    write_markdown(READINESS_MD, f"""# SUSC-16B - readiness para 16C e score v7

- ready_for_16C_proxy_calibration_review: {str(ready_16c).lower()}
- ready_for_score_v7_candidate: {str(ready_v7).lower()}
- score_v7_created: false

O 16B executou avaliacao observacional review-only. Mesmo quando criterios
quantitativos sao suficientes para discussao metodologica, este marco nao cria
score v7, nao altera pesos e nao libera treino supervisionado.
""")
    build_tcc_tables_and_figures()
    _write_final_report()
    print("readiness ->", READINESS.relative_to(ROOT))
    return 0


def _svg_bar(path: Path, title: str, rows: list[tuple[str, float]], width: int = 760, height: int = 420) -> None:
    max_value = max([value for _label, value in rows] + [1.0])
    bars = []
    for i, (label, value) in enumerate(rows):
        y = 70 + i * 54
        bar_w = 520 * value / max_value if max_value else 0
        bars.append(
            f'<text x="24" y="{y + 18}" font-size="13">{label}</text>'
            f'<rect x="210" y="{y}" width="{bar_w:.2f}" height="24" fill="#3478b8" />'
            f'<text x="{220 + bar_w:.2f}" y="{y + 18}" font-size="13">{value:.3f}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<rect width="100%" height="100%" fill="#ffffff" />'
        f'<text x="24" y="36" font-size="20" font-family="Arial">{title}</text>'
        + "".join(bars)
        + '<text x="24" y="392" font-size="12" font-family="Arial">SUSC-16B review-only; controles nao sao negativos verdadeiros.</text>'
        + "</svg>\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def build_tcc_tables_and_figures() -> int:
    for fn in (run_score_evaluation, run_feature_contrast, build_low_score_case_audit, audit_footprint_quality, build_recommendations):
        fn()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(METRICS_SUMMARY.read_text(encoding="utf-8"))
    write_csv(TABLE_DIR / "event_vs_control_score_summary.csv", [
        {"group": "observed_footprint_patch", "mean_score": _fmt(metrics.get("mean_score_event")), "median_score": _fmt(metrics.get("median_score_event")), "n": metrics.get("event_patch_count"), "review_only": "true"},
        {"group": "no_documented_footprint_control", "mean_score": _fmt(metrics.get("mean_score_control")), "median_score": _fmt(metrics.get("median_score_control")), "n": metrics.get("control_patch_count"), "review_only": "true"},
    ])
    write_csv(TABLE_DIR / "feature_contrast_summary.csv", read_csv(FEATURE_CONTRAST))
    write_csv(TABLE_DIR / "low_score_event_cases.csv", read_csv(LOW_SCORE_CASES))
    quality_summary = json.loads(QUALITY_SUMMARY.read_text(encoding="utf-8"))
    write_csv(TABLE_DIR / "footprint_quality_summary.csv", [
        {"quality_tier": key, "count": value, "review_only": "true"}
        for key, value in quality_summary.get("eligible_quality_tier_distribution", {}).items()
    ])
    write_csv(TABLE_DIR / "calibration_recommendations_summary.csv", read_csv(RECOMMENDATIONS))
    _svg_bar(FIG_DIR / "score_event_vs_control.svg", "Score v6 medio: evento x controle", [
        ("observed_footprint_patch", float(metrics.get("mean_score_event") or 0.0)),
        ("no_documented_footprint_control", float(metrics.get("mean_score_control") or 0.0)),
    ])
    event_classes = metrics.get("event_score_class_distribution", {})
    control_classes = metrics.get("control_score_class_distribution", {})
    _svg_bar(FIG_DIR / "score_class_distribution_event_vs_control.svg", "Distribuicao de classes score v6", [
        (f"evento {key}", float(event_classes.get(key, 0))) for key in ("low", "medium", "high")
    ] + [(f"controle {key}", float(control_classes.get(key, 0))) for key in ("low", "medium", "high")])
    audit = json.loads(FEATURE_AUDIT.read_text(encoding="utf-8"))
    _svg_bar(FIG_DIR / "feature_direction_agreement.svg", "Concordancia de direcao das features", [
        ("concordam", float(len(audit.get("features_agreeing", [])))),
        ("divergem", float(len(audit.get("features_diverging", [])))),
    ])
    pipeline = """<svg xmlns="http://www.w3.org/2000/svg" width="900" height="260" viewBox="0 0 900 260">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="24" y="36" font-size="20" font-family="Arial">Pipeline SUSC-16B review-only</text>
<g font-family="Arial" font-size="14">
<rect x="30" y="80" width="150" height="54" fill="#dcecff" stroke="#2d5f8b"/><text x="46" y="112">SUSC-16A links</text>
<rect x="230" y="80" width="150" height="54" fill="#e7f4e4" stroke="#4b7c42"/><text x="252" y="112">Score v6</text>
<rect x="430" y="80" width="170" height="54" fill="#fff2cc" stroke="#8a6d1b"/><text x="452" y="112">Evento-controle</text>
<rect x="650" y="80" width="190" height="54" fill="#f3e6ff" stroke="#6a3d9a"/><text x="672" y="112">Diagnostico 16B</text>
<path d="M180 107 L230 107 M380 107 L430 107 M600 107 L650 107" stroke="#333" stroke-width="2"/>
<text x="30" y="190">Sem ground truth | sem treino | sem score v7 | controles = ausencia documentada</text>
</g>
</svg>
"""
    (FIG_DIR / "pipeline_16b_review_only_evaluation.svg").write_text(pipeline, encoding="utf-8")
    print("tcc package ->", TABLE_DIR.relative_to(ROOT), FIG_DIR.relative_to(ROOT))
    return 0


def _write_final_report() -> None:
    metrics = json.loads(METRICS_SUMMARY.read_text(encoding="utf-8"))
    feature_audit = json.loads(FEATURE_AUDIT.read_text(encoding="utf-8"))
    quality = json.loads(QUALITY_SUMMARY.read_text(encoding="utf-8"))
    rec_counts = Counter(row["recommendation"] for row in read_csv(RECOMMENDATIONS))
    report = f"""# SUSC-16B - avaliacao observacional review-only do score v6 contra footprints elegiveis

Status: review-only. `allowed_for_training=false`; `can_be_ground_truth=false`; `score_v7_created=false`.

{MANDATORY}

## 1. O que o SUSC-16A desbloqueou
O SUSC-16A desbloqueou {metrics['event_patch_count']} links footprint-patch elegiveis, {metrics['unique_event_patches']} patches observacionais unicos e {quality['eligible_footprint_count']} footprints elegiveis para avaliacao observacional.

## 2. Por que 16B e avaliacao, nao calibracao
O 16B mede concordancia/divergencia entre score v6, features e footprints. Ele nao altera pesos, nao cria score v7, nao cria ground truth e nao treina modelo.

## 3. Dataset evento-controle
O dataset usa `observed_footprint_patch` para links elegiveis e `no_documented_footprint_control` para patches sem footprint elegivel documentado na mesma regiao quando disponivel. Controles nao sao negativos verdadeiros.

## 4. Metricas score v6 contra footprints
- Score medio evento: {_fmt(metrics.get('mean_score_event'))}
- Score medio controle: {_fmt(metrics.get('mean_score_control'))}
- Diferenca evento-controle: {_fmt(metrics.get('score_diff_event_minus_control'))}

## 5. Resultado hit@top-k
hit@10={metrics['hit_rate_top_10']}; hit@20={metrics['hit_rate_top_20']}; hit@30={metrics['hit_rate_top_30']}. Baixa presenca em top-k foi registrada como possivel subestimacao do score v6, sem correcao automatica.

## 6. Contraste de features
Features que concordam: {', '.join(feature_audit.get('features_agreeing', [])) or 'nenhuma'}.
Features que divergem: {', '.join(feature_audit.get('features_diverging', [])) or 'nenhuma'}.

## 7. Divergencias principais
Divergencias sao registradas como diagnostico review-only. Se controles tiverem score medio superior ou eventos aparecerem pouco no top-k, isso indica conflito de direcao ou possivel subestimacao.

## 8. Auditoria da qualidade dos footprints
Distribuicao dos tiers elegiveis: {json.dumps(quality.get('eligible_quality_tier_distribution', {}), ensure_ascii=False, sort_keys=True)}. Tiers D/E nao sustentam conclusao forte.

## 9. Casos de footprint com score baixo
Os casos baixo/medio estao em `SUSC_16B_low_score_footprint_case_audit.csv` com possiveis razoes auditaveis.

## 10. Recomendacoes de calibracao review-only
Recomendacoes candidatas: {json.dumps(dict(rec_counts), ensure_ascii=False, sort_keys=True)}. Nenhuma recomendacao altera pesos neste marco.

## 11. Readiness para 16C
O 16B esta pronto para revisao de calibracao proxy 16C quando os criterios de links, footprints, contraste de features e auditoria de qualidade estao completos.

## 12. Por que score v7 ainda nao e criado
Score v7 permanece bloqueado neste marco por governanca: 16B e avaliacao, nao calibracao. Tambem ha divergencias/limitacoes que exigem revisao antes de qualquer candidato.

## 13. Limitacoes
Footprints permanecem candidatos observacionais, muitos sem data explicita. Controles representam ausencia documentada no SUSC-16A, nao ausencia real. A avaliacao e exploratoria e review-only.

## 14. Proximo marco
Executar SUSC-16C como revisao de calibracao proxy, mantendo fail-closed, sem treino e sem score v7 automatico ate que a evidencia seja suficiente e auditavel.
"""
    write_markdown(REPORT, report)


def build_final_report() -> int:
    build_tcc_tables_and_figures()
    run_readiness()
    print("report ->", REPORT.relative_to(ROOT))
    return 0


def validate() -> int:
    required = [
        SCHEMA, DATASET, DATASET_SUMMARY, METRICS, METRICS_REGION, METRICS_SUMMARY,
        FEATURE_CONTRAST, FEATURE_CONTRAST_REGION, FEATURE_AUDIT, LOW_SCORE_CASES,
        LOW_SCORE_REPORT, QUALITY_AUDIT, QUALITY_SUMMARY, RECOMMENDATIONS,
        RECOMMENDATIONS_MD, READINESS, READINESS_MD, TABLE_DIR, FIG_DIR, REPORT,
    ]
    errors: list[str] = []
    for path in required:
        if not path.exists():
            errors.append(f"missing: {path.relative_to(ROOT)}")
    if TABLE_DIR.exists():
        for name in (
            "event_vs_control_score_summary.csv", "feature_contrast_summary.csv",
            "low_score_event_cases.csv", "footprint_quality_summary.csv",
            "calibration_recommendations_summary.csv",
        ):
            if not (TABLE_DIR / name).exists():
                errors.append(f"missing table: {name}")
    if FIG_DIR.exists():
        for name in (
            "score_event_vs_control.svg", "score_class_distribution_event_vs_control.svg",
            "feature_direction_agreement.svg", "pipeline_16b_review_only_evaluation.svg",
        ):
            if not (FIG_DIR / name).exists():
                errors.append(f"missing figure: {name}")
    for path in (DATASET, LOW_SCORE_CASES, QUALITY_AUDIT, RECOMMENDATIONS):
        if path.exists():
            for idx, row in enumerate(read_csv(path), start=2):
                if row.get("can_be_ground_truth") == "true":
                    errors.append(f"{path.name}:{idx}: can_be_ground_truth true")
                if row.get("allowed_for_training") == "true":
                    errors.append(f"{path.name}:{idx}: allowed_for_training true")
                if row.get("review_only") != "true":
                    errors.append(f"{path.name}:{idx}: review_only not true")
                if "control" in row.get("group_type", "") and row.get("evidence_strength") == "negative":
                    errors.append(f"{path.name}:{idx}: control marked negative")
    if SCORE_V7.exists():
        errors.append("score v7 candidate artifact exists")
    if not matrix_sha256_ok():
        errors.append("SUSC-03 matrix sha256 mismatch")
    if find_model_artifacts():
        errors.append("model artifact found")
    if REPORT.exists() and MANDATORY not in REPORT.read_text(encoding="utf-8"):
        errors.append("mandatory report phrase missing")
    if errors:
        print("FAILED: SUSC-16B validation")
        for error in errors:
            print("  ERROR:", error)
        return 1
    print("PASSED: SUSC-16B validations passed")
    return 0


def main() -> int:
    return build_event_control_dataset()


if __name__ == "__main__":
    raise SystemExit(main())
