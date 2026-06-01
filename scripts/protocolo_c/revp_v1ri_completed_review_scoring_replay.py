"""REV-P v1ri — Completed double-review scoring replay.

Aggregates validated A/B responses per sample and computes review support
scores. Compatible with v1qx but separate so P0 is never overwritten.
Fail-closed (headers only) when no validated responses exist. Review-only;
never creates labels, targets or ground truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1rg_v1rm_review_response_common import (
    DATASETS,
    DOCS,
    SCHEMAS,
    _p,
    assert_clean_rows,
    compute_completed_score,
    filled_response_rows,
    gather_responses,
    read_csv_safe,
    responses_path,
    write_csv_with_header,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

IN_SAMPLE = _p("REVP_V1RI_IN_SAMPLE", DATASETS / "protocol_c_event_patch_review_sample_v1qv.csv")
IN_VALIDATION_SUMMARY = _p("REVP_V1RI_IN_VALIDATION_SUMMARY", DATASETS / "protocol_c_review_response_validation_summary_v1rh.csv")
OUT_SCORES = _p("REVP_V1RI_OUT_SCORES", DATASETS / "protocol_c_completed_review_scores_v1ri.csv")
OUT_DISAGREE = _p("REVP_V1RI_OUT_DISAGREE", DATASETS / "protocol_c_completed_review_disagreements_v1ri.csv")
OUT_SUMMARY = _p("REVP_V1RI_OUT_SUMMARY", DATASETS / "protocol_c_completed_review_scoring_summary_v1ri.csv")
SCHEMA_SCORES = _p("REVP_V1RI_SCHEMA_SCORES", SCHEMAS / "protocol_c_completed_review_scores_v1ri_schema.csv")
SCHEMA_DISAGREE = _p("REVP_V1RI_SCHEMA_DISAGREE", SCHEMAS / "protocol_c_completed_review_disagreements_v1ri_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1RI_SCHEMA_SUMMARY", SCHEMAS / "protocol_c_completed_review_scoring_summary_v1ri_schema.csv")
DOC = _p("REVP_V1RI_DOC", DOCS / "revp_v1ri_completed_review_scoring_replay.md")

SCORE_FIELDS = [
    "completed_score_id", "review_sample_id", "event_id", "patch_id", "region",
    "reviewer_a_present", "reviewer_b_present", "reviewer_agreement_score",
    "disagreement_flag", "disagreement_type", "evidence_support_score",
    "temporal_support_score", "spatial_support_score", "source_support_score",
    "composite_review_score", "recommended_decision", "supervisor_review_required",
    "review_only", "can_create_operational_label", "can_train_model",
    "target_created", "ground_truth_operational", "formal_negative",
    "blocked_reason", "notes",
]

DISAGREE_FIELDS = [
    "disagreement_id", "review_sample_id", "decision_a", "decision_b",
    "disagreement_type", "reviewer_agreement_score", "needs_third_reviewer",
    "review_only", "notes",
]

SUMMARY_FIELDS = ["stat_key", "stat_value"]

NOT_COMPLETED = "COMPLETED_REVIEW_NOT_AVAILABLE_FAIL_CLOSED"
SCORED = "COMPLETED_REVIEW_SCORED_REVIEW_ONLY"

VALIDATION_PASS = "REVIEW_RESPONSES_VALIDATION_PASS_REVIEW_ONLY"


def _validation_passed() -> bool:
    for r in read_csv_safe(IN_VALIDATION_SUMMARY):
        if r.get("stat_key") == "validation_status":
            return r.get("stat_value") == VALIDATION_PASS
    return False


def run(datasets: Path | None = None) -> dict[str, Any]:
    sample = read_csv_safe(IN_SAMPLE)
    sample_by_id = {s.get("review_sample_id", ""): s for s in sample}

    path = responses_path()
    score_rows: list[dict[str, Any]] = []
    disagree_rows: list[dict[str, Any]] = []
    status = NOT_COMPLETED

    if path is not None and _validation_passed():
        filled = filled_response_rows(read_csv_safe(path))
        if filled:
            gathered = gather_responses(filled)
            for rsid, slots in sorted(gathered.items()):
                meta = sample_by_id.get(rsid, {})
                sr, dr = compute_completed_score(rsid, slots, meta)
                score_rows.append(sr)
                if dr:
                    disagree_rows.append(dr)
            if score_rows:
                status = SCORED

    assert_clean_rows(score_rows, "v1ri_scores")
    write_csv_with_header(OUT_SCORES, score_rows, SCORE_FIELDS)
    write_csv_with_header(OUT_DISAGREE, disagree_rows, DISAGREE_FIELDS)
    write_schema_safe(SCHEMA_SCORES, SCORE_FIELDS, "v1ri_scores")
    write_schema_safe(SCHEMA_DISAGREE, DISAGREE_FIELDS, "v1ri_disagreement")

    completed = sum(1 for r in score_rows if r["reviewer_a_present"] == "true" and r["reviewer_b_present"] == "true")
    c3 = sum(1 for r in score_rows if r["recommended_decision"] == "C3_REFERENCE_CANDIDATE_NEEDS_SUPERVISOR")
    summary = [
        {"stat_key": "scoring_status", "stat_value": status},
        {"stat_key": "scored_samples", "stat_value": str(len(score_rows))},
        {"stat_key": "completed_double_reviews", "stat_value": str(completed)},
        {"stat_key": "disagreement_cases", "stat_value": str(len(disagree_rows))},
        {"stat_key": "c3_candidate_signals", "stat_value": str(c3)},
        {"stat_key": "stage", "stat_value": "v1ri"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    write_schema_safe(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1ri_summary")

    write_doc(
        DOC,
        "v1ri — Completed Double-Review Scoring Replay",
        [
            "## Objetivo",
            "Agregar respostas A/B validadas por sample e calcular suportes de revisao "
            "(evidencia, temporal, espacial, fonte, concordancia) e composite. Sem respostas "
            "validadas, fail-closed (COMPLETED_REVIEW_NOT_AVAILABLE_FAIL_CLOSED).",
            "## Resultado",
            f"Status: {status}. Samples pontuados: {len(score_rows)}. "
            f"Reviews completos (A/B): {completed}. Desacordos: {len(disagree_rows)}. "
            f"Sinais C3 candidate: {c3}.",
            "## Guardrails",
            "Exige A/B; revisao unilateral conta como incompleta. Desacordo bloqueia C3. "
            "Nunca cria label/target/ground truth. dino_validates_event=false.",
        ],
    )
    print(f"[v1ri] status={status} scored={len(score_rows)} completed={completed} c3={c3}")
    return {"status": status, "scored": len(score_rows), "completed": completed, "c3": c3}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1ri completed review scoring").parse_args()
    run()
