"""REV-P v1qx — Observational evidence scoring model.

Scores completed double-review responses without supervising anything.
Responses are read from REVP_PROTOCOL_C_REVIEW_RESPONSES_PATH; if absent,
the stage is fail-closed (headers only, status REVIEW_NOT_COMPLETED_FAIL_CLOSED).
Scores are review-only signals, never supervised targets.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from revp_v1lj_v1lq_common import DATASETS, DOCS, SCHEMAS
from revp_v1qu_v1qz_ground_reference_common import (
    SECONDARY_FAMILIES,
    UNKNOWN_SOURCE,
    _p,
    assert_clean_rows,
    classify_source_family,
    composite_score,
    decision_from_scores,
    guardrail_row,
    normalize_region,
    read_csv_safe,
    score_independence,
    score_provenance,
    score_review_agreement,
    score_source_reliability,
    score_spatial_precision,
    score_temporal_precision,
    write_csv_with_header,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

IN_SAMPLE = _p("REVP_V1QX_IN_SAMPLE", DATASETS / "protocol_c_event_patch_review_sample_v1qv.csv")
OUT_SCORES = _p("REVP_V1QX_OUT_SCORES", DATASETS / "protocol_c_observational_evidence_scores_v1qx.csv")
OUT_DISAGREE = _p("REVP_V1QX_OUT_DISAGREE", DATASETS / "protocol_c_observational_disagreement_registry_v1qx.csv")
OUT_SUMMARY = _p("REVP_V1QX_OUT_SUMMARY", DATASETS / "protocol_c_observational_scoring_summary_v1qx.csv")
SCHEMA_SCORES = _p("REVP_V1QX_SCHEMA_SCORES", SCHEMAS / "protocol_c_observational_evidence_scores_v1qx_schema.csv")
SCHEMA_DISAGREE = _p("REVP_V1QX_SCHEMA_DISAGREE", SCHEMAS / "protocol_c_observational_disagreement_registry_v1qx_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1QX_SCHEMA_SUMMARY", SCHEMAS / "protocol_c_observational_scoring_summary_v1qx_schema.csv")
DOC = _p("REVP_V1QX_DOC", DOCS / "revp_v1qx_observational_evidence_scoring_model.md")

SCORE_FIELDS = [
    "score_id", "review_sample_id", "event_id", "patch_id", "region",
    "source_reliability_score", "temporal_precision_score", "spatial_precision_score",
    "provenance_score", "independence_score", "review_agreement_score",
    "composite_observational_score", "disagreement_flag", "disagreement_type",
    "scoring_status", "recommended_protocol_c_decision",
    "review_only", "dino_validates_event", "can_create_operational_label",
    "can_train_model", "target_created", "ground_truth_operational",
    "blocked_reason", "notes",
]

DISAGREE_FIELDS = [
    "disagreement_id", "review_sample_id", "decision_a", "decision_b",
    "disagreement_type", "review_agreement_score", "needs_third_reviewer",
    "review_only", "notes",
]

SUMMARY_FIELDS = ["stat_key", "stat_value"]

NOT_COMPLETED = "REVIEW_NOT_COMPLETED_FAIL_CLOSED"
SCORED = "SCORED_REVIEW_ONLY"


def _yes(value: str | None) -> bool:
    return str(value or "").strip().lower() in ("sim", "yes", "true", "1", "y")


def _responses_path() -> Path | None:
    env = os.environ.get("REVP_PROTOCOL_C_REVIEW_RESPONSES_PATH")
    if env and Path(env).exists():
        return Path(env)
    return None


def _gather(responses: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    """sample_id -> reviewer_slot -> {question_key: response_value}."""
    out: dict[str, dict[str, dict[str, str]]] = {}
    for r in responses:
        rsid = r.get("review_sample_id", "")
        slot = r.get("reviewer_slot", "")
        q = r.get("question_key", "")
        val = r.get("response_value", "")
        if not rsid or not slot or not q:
            continue
        out.setdefault(rsid, {}).setdefault(slot, {})[q] = val
    return out


def score_sample(rsid: str, slots: dict[str, dict[str, str]], sample_meta: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    a = slots.get("REVIEWER_A", {})
    b = slots.get("REVIEWER_B", {})

    # Source family from reviewer quality answer or sample evidence
    quality_txt = (a.get("source_quality", "") + " " + b.get("source_quality", "")
                   + " " + sample_meta.get("evidence_status", ""))
    family = classify_source_family(quality_txt)
    has_ref = bool(sample_meta.get("source_requirement_status", ""))

    # Temporal precision: from reviewer timing support + sample temporal status
    if _yes(a.get("timing_supported")) and _yes(b.get("timing_supported")):
        temporal_status = "DAY"
    elif _yes(a.get("timing_supported")) or _yes(b.get("timing_supported")):
        temporal_status = "MONTH"
    else:
        temporal_status = sample_meta.get("temporal_status", "UNKNOWN")

    if _yes(a.get("location_supported")) and _yes(b.get("location_supported")):
        spatial_status = "ADDRESS"
    elif _yes(a.get("location_supported")) or _yes(b.get("location_supported")):
        spatial_status = "ADMINISTRATIVE"
    else:
        spatial_status = "NONE"

    n_independent = sum(1 for slot in (a, b) if _yes(slot.get("independent_source_present")))

    rel = score_source_reliability(family)
    temp = score_temporal_precision(temporal_status)
    spat = score_spatial_precision(spatial_status)
    prov = score_provenance(family, has_ref)
    indep = score_independence(n_independent)
    agree = score_review_agreement(a.get("recommended_decision", ""), b.get("recommended_decision", ""))

    scores = {
        "source_reliability_score": rel,
        "temporal_precision_score": temp,
        "spatial_precision_score": spat,
        "provenance_score": prov,
        "independence_score": indep,
        "review_agreement_score": agree,
    }
    comp = composite_score(scores)
    scores["composite"] = comp

    decision = decision_from_scores(scores, family)

    # Block reasons
    blocked_reason = ""
    if family in ("", UNKNOWN_SOURCE) or family in SECONDARY_FAMILIES:
        if decision == "BLOCKED_INSUFFICIENT_EVIDENCE":
            blocked_reason = "SOURCE_WEAK_OR_SECONDARY"
    if temp < 0.6 and "C3" not in decision:
        blocked_reason = blocked_reason or "LOW_TEMPORAL_PRECISION"
    if spat < 0.5 and "C3" not in decision:
        blocked_reason = blocked_reason or "LOW_SPATIAL_PRECISION"

    da = str(a.get("recommended_decision", "")).strip().upper()
    db = str(b.get("recommended_decision", "")).strip().upper()
    disagree = bool(da and db and da != db)
    disagreement_type = ""
    if disagree:
        disagreement_type = "DECISION_MISMATCH"
    elif not (da and db):
        disagreement_type = "INCOMPLETE_REVIEW"

    score_row = {
        "score_id": f"V1QX_SC_{rsid}",
        "review_sample_id": rsid,
        "event_id": sample_meta.get("event_id", ""),
        "patch_id": sample_meta.get("patch_id", ""),
        "region": normalize_region(sample_meta.get("region", "")),
        "source_reliability_score": f"{rel:.3f}",
        "temporal_precision_score": f"{temp:.3f}",
        "spatial_precision_score": f"{spat:.3f}",
        "provenance_score": f"{prov:.3f}",
        "independence_score": f"{indep:.3f}",
        "review_agreement_score": f"{agree:.3f}",
        "composite_observational_score": f"{comp:.3f}",
        "disagreement_flag": "true" if disagree else "false",
        "disagreement_type": disagreement_type,
        "scoring_status": SCORED,
        "recommended_protocol_c_decision": decision,
        "blocked_reason": blocked_reason,
        "notes": "",
    }
    score_row.update(guardrail_row())

    disagree_row = None
    if disagree:
        disagree_row = {
            "disagreement_id": f"V1QX_DIS_{rsid}",
            "review_sample_id": rsid, "decision_a": da, "decision_b": db,
            "disagreement_type": disagreement_type,
            "review_agreement_score": f"{agree:.3f}",
            "needs_third_reviewer": "true",
            "review_only": "true", "notes": "",
        }
    return score_row, disagree_row


def run(datasets: Path | None = None) -> dict[str, Any]:
    sample = read_csv_safe(IN_SAMPLE)
    sample_by_id = {s.get("review_sample_id", ""): s for s in sample}

    responses_path = _responses_path()
    score_rows: list[dict[str, Any]] = []
    disagree_rows: list[dict[str, Any]] = []
    status = NOT_COMPLETED

    if responses_path is not None:
        responses = read_csv_safe(responses_path)
        # only count responses that actually carry a non-empty value
        filled = [r for r in responses if str(r.get("response_value", "")).strip()]
        if filled:
            status = SCORED
            gathered = _gather(filled)
            for rsid, slots in sorted(gathered.items()):
                meta = sample_by_id.get(rsid, {})
                sr, dr = score_sample(rsid, slots, meta)
                score_rows.append(sr)
                if dr:
                    disagree_rows.append(dr)

    assert_clean_rows(score_rows, "v1qx_scores")
    write_csv_with_header(OUT_SCORES, score_rows, SCORE_FIELDS)
    write_csv_with_header(OUT_DISAGREE, disagree_rows, DISAGREE_FIELDS)
    write_schema_safe(SCHEMA_SCORES, SCORE_FIELDS, "v1qx_scores")
    write_schema_safe(SCHEMA_DISAGREE, DISAGREE_FIELDS, "v1qx_disagreement")

    n_c3 = sum(1 for r in score_rows if "C3" in r["recommended_protocol_c_decision"])
    summary = [
        {"stat_key": "scoring_status", "stat_value": status},
        {"stat_key": "completed_reviews_scored", "stat_value": str(len(score_rows))},
        {"stat_key": "disagreement_cases", "stat_value": str(len(disagree_rows))},
        {"stat_key": "c3_candidate_signals", "stat_value": str(n_c3)},
        {"stat_key": "responses_path_present", "stat_value": "true" if responses_path else "false"},
        {"stat_key": "stage", "stat_value": "v1qx"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    write_schema_safe(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1qx_summary")

    write_doc(
        DOC,
        "v1qx — Observational Evidence Scoring Model",
        [
            "## Objetivo",
            "Pontuar respostas de revisao dupla concluidas, sem supervisionar. Sem respostas "
            "preenchidas em REVP_PROTOCOL_C_REVIEW_RESPONSES_PATH, o estagio e fail-closed "
            "(REVIEW_NOT_COMPLETED_FAIL_CLOSED) com apenas cabecalho.",
            "## Scores",
            "source_reliability, temporal_precision, spatial_precision, provenance, "
            "independence, review_agreement -> composite. Sao sinais de revisao, "
            "nunca targets supervisionados.",
            "## Resultado",
            f"Status: {status}. Reviews pontuados: {len(score_rows)}. "
            f"Desacordos: {len(disagree_rows)}.",
            "## Guardrails",
            "Fonte fraca/secundaria nunca fecha gate C3. Baixa precisao temporal/espacial "
            "bloqueia C3. dino_validates_event=false. Nenhum label/target/ground truth.",
        ],
    )
    print(f"[v1qx] status={status} scored={len(score_rows)} disagreements={len(disagree_rows)}")
    return {"status": status, "scored": len(score_rows), "disagreements": len(disagree_rows)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qx observational scoring").parse_args()
    run()
