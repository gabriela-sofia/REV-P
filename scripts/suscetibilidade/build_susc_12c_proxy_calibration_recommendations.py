"""SUSC-12C proxy calibration recommendations."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, read_json, write_csv, write_json, write_markdown  # noqa: E402

SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_12c_proxy_calibration_schema_v1.json"
CONTRAST = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12B_observed_event_feature_contrast.csv"
CONTRAST_SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_12B_observed_event_contrast_summary.json"

OUT = ROOT / "outputs_public" / "suscetibilidade"
RECS = OUT / "SUSC_12C_proxy_calibration_recommendations.csv"
CHANGELOG = OUT / "SUSC_12C_score_v6_to_v7_change_log.csv"
SUMMARY = OUT / "SUSC_12C_proxy_calibration_summary.json"
REPORT = OUT / "SUSC_12C_proxy_calibration_report.md"

FIELDS = [
    "feature",
    "expected_direction",
    "direction_agreement",
    "standardized_difference",
    "confidence_flag",
    "recommendation",
    "deterministic_rule",
    "can_be_ground_truth",
    "allowed_for_training",
    "review_only",
]


def _abs_float(value: str) -> float:
    try:
        return abs(float(value))
    except ValueError:
        return 0.0


def _recommend(row: dict, low_sample: bool) -> tuple[str, str]:
    if low_sample:
        return "needs_more_observed_events", "sample_size_event_below_20_no_automatic_weight_change"
    if row["blocked_in_v6"] == "true" and row["direction_agreement"] != "true":
        return "exclude_until_more_evidence", "blocked_in_v6_and_no_observed_support"
    if row["direction_agreement"] == "true" and _abs_float(row["standardized_difference"]) >= 0.25:
        return "increase_weight", "direction_agrees_and_effect_nontrivial"
    if row["direction_agreement"] == "true":
        return "keep_weight", "direction_agrees_but_effect_small"
    if row["direction_agreement"] == "false":
        return "decrease_weight", "direction_diverges_from_expected"
    return "exclude_until_more_evidence", "ambiguous_or_inconclusive_direction"


def main() -> int:
    print("=" * 60)
    print("SUSC-12C Proxy Calibration Recommendations")
    print("=" * 60)
    for path in (SCHEMA, CONTRAST, CONTRAST_SUMMARY):
        if not path.exists():
            print(f"STOP: required input not found: {path}")
            return 1

    contrast = read_csv(CONTRAST)
    contrast_summary = read_json(CONTRAST_SUMMARY)
    low_sample = bool(contrast_summary.get("low_sample_size", True))
    rows = []
    for row in contrast:
        rec, rule = _recommend(row, low_sample)
        rows.append({
            "feature": row["feature"],
            "expected_direction": row["expected_direction"],
            "direction_agreement": row["direction_agreement"],
            "standardized_difference": row["standardized_difference"],
            "confidence_flag": row["confidence_flag"],
            "recommendation": rec,
            "deterministic_rule": rule,
            "can_be_ground_truth": "false",
            "allowed_for_training": "false",
            "review_only": "true",
        })
    write_csv(RECS, rows, FIELDS)

    changelog = [{
        "change_id": "SUSC12C_NO_V7_0001",
        "change_type": "no_score_v7_created",
        "reason": "low_sample_size_and_review_only_observed_evidence",
        "event_patch_count": contrast_summary.get("event_patch_count", 0),
        "control_patch_count": contrast_summary.get("control_patch_count", 0),
        "deterministic_decision": "keep_score_v6_candidate_no_weight_change",
        "can_be_ground_truth": "false",
        "allowed_for_training": "false",
        "review_only": "true",
    }]
    write_csv(CHANGELOG, changelog)

    counts = Counter(r["recommendation"] for r in rows)
    summary = {
        "artifact": "SUSC-12C proxy calibration recommendations",
        "recommendation_counts": dict(counts),
        "score_v7_created": False,
        "score_v7_ready": False,
        "not_ready_reason": "low_sample_size_observed_event_patch_count_below_20",
        "event_patch_count": contrast_summary.get("event_patch_count", 0),
        "control_patch_count": contrast_summary.get("control_patch_count", 0),
        "can_be_ground_truth": False,
        "allowed_for_training": False,
        "model_persisted": False,
        "review_only": True,
    }
    write_json(SUMMARY, summary)

    lines = "\n".join(f"| {k} | {v} |" for k, v in sorted(counts.items()))
    md = f"""# SUSC-12C - Proxy Calibration Recommendations

Status: **review-only** | `score_v7_created=false`

As recomendacoes abaixo usam o contraste SUSC-12B apenas como diagnostico.
Nenhum peso foi alterado automaticamente porque a amostra observacional e
pequena (`event_patch_count={summary['event_patch_count']}`) e permanece
review-only.

| recomendacao | n |
|---|---:|
{lines}

Decisao: manter `score_v6_candidate` sem criar `score_v7_candidate`.

Limites: controles sao `no_documented_event_control`; nao ha ground truth,
treino supervisionado ou predicao operacional.
"""
    write_markdown(REPORT, md)

    print("recommendations:", dict(counts))
    print("score_v7_created: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
