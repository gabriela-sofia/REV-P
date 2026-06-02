"""REV-P v1tp — Automated Reviewer A/B adjudication.

Two independent automatic review profiles per case:
  - Reviewer A (conservative): demands independent observational source,
    penalises weak temporal/spatial precision, blocks overclaim.
  - Reviewer B (integrator): values cross-evidence consistency and DINO/patch
    context, still blocks overclaim and never creates ground truth.
Neither may create labels, targets, ground truth, C3, C4 or formal negatives.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1tn_v1tw_automated_review_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
    guardrail_row_review, scan_guardrails, hash_short,
    reviewer_dimensions, classify_automated_review_decision,
    next_required_action, classify_case_readiness, build_reviewer_rubric,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_DEC  = _p("REVP_V1TP_OUT_DEC",  DATASETS / "protocol_c_automated_reviewer_ab_decisions_v1tp.csv")
OUT_RUB  = _p("REVP_V1TP_OUT_RUB",  DATASETS / "protocol_c_automated_reviewer_ab_rubric_v1tp.csv")
OUT_SUM  = _p("REVP_V1TP_OUT_SUM",  DATASETS / "protocol_c_automated_reviewer_ab_summary_v1tp.csv")
SCHEMA_D = _p("REVP_V1TP_SCHEMA_D", SCHEMAS  / "protocol_c_automated_reviewer_ab_decisions_v1tp_schema.csv")
SCHEMA_R = _p("REVP_V1TP_SCHEMA_R", SCHEMAS  / "protocol_c_automated_reviewer_ab_rubric_v1tp_schema.csv")
SCHEMA_S = _p("REVP_V1TP_SCHEMA_S", SCHEMAS  / "protocol_c_automated_reviewer_ab_summary_v1tp_schema.csv")
DOC      = _p("REVP_V1TP_DOC",      DOCS     / "revp_v1tp_automated_reviewer_ab_adjudication.md")

DEC_FIELDS = [
    "automated_review_id", "case_id", "reviewer_slot", "reviewer_profile",
    "external_source_sufficient_for_review_only",
    "temporal_context_sufficient_for_review_only",
    "spatial_context_sufficient_for_review_only",
    "hydromet_context_useful", "dino_role_correctly_limited",
    "patch_linkage_sufficient_for_review_only", "overclaim_risk",
    "evidence_chain_completeness_score", "review_only_confidence_score",
    "recommended_review_only_status", "required_next_action",
    "automated_review", "internal_review_automated_for_review_only",
    "requires_external_observational_evidence_for_operational_claim",
    "review_only", "can_create_operational_label", "can_train_model",
    "target_created", "ground_truth_operational", "formal_negative",
    "automatic_c3_promotion", "c4_opened",
    "dino_validates_event", "hydromet_validates_event",
    "hydromet_is_negative_evidence", "absence_as_negative", "notes",
]
RUB_FIELDS = ["reviewer_profile", "criterion_key", "criterion_description",
              "review_only", "automated_review"]
SUM_FIELDS = ["stat_key", "stat_value"]

PROFILES = [("A", "conservative"), ("B", "integrator")]


def _decision_for(case: dict[str, str], slot: str, profile: str) -> dict[str, Any]:
    ext = case.get("external_evidence_status", "")
    hyd = case.get("hydromet_status", "")
    window = case.get("event_window", "")
    patch = case.get("patch_link_status", "")
    dino = case.get("dino_status", "")
    # temporal status: derive from window presence (case index already vetted it)
    temporal_status = "DATE_PARSED_OK" if window else ""

    dims = reviewer_dimensions(
        profile, ext, hyd, window, temporal_status, patch, dino)
    decision = classify_automated_review_decision(profile, dims)
    readiness = classify_case_readiness(ext, hyd, window, temporal_status, patch)

    row: dict[str, Any] = {
        "automated_review_id": f"V1TP_{slot}_{hash_short(case.get('case_id',''), 10)}",
        "case_id": case.get("case_id", ""),
        "reviewer_slot": slot, "reviewer_profile": profile,
        "recommended_review_only_status": decision,
        "required_next_action": next_required_action(readiness),
        "notes": "",
    }
    row.update(dims)
    row.update(guardrail_row_review())
    return row


def run() -> dict[str, Any]:
    cases = read_csv_safe(DATASETS / "protocol_c_unified_evidence_case_index_v1tn.csv")
    cases = [c for c in cases if not c.get("case_id", "").startswith("FAIL_CLOSED")]

    rows: list[dict[str, Any]] = []
    for c in cases:
        for slot, profile in PROFILES:
            rows.append(_decision_for(c, slot, profile))

    if not rows:
        base = guardrail_row_review()
        for slot, profile in PROFILES:
            rows.append({
                "automated_review_id": f"FAIL_CLOSED_{slot}", "case_id": "",
                "reviewer_slot": slot, "reviewer_profile": profile,
                "external_source_sufficient_for_review_only": "false",
                "temporal_context_sufficient_for_review_only": "false",
                "spatial_context_sufficient_for_review_only": "false",
                "hydromet_context_useful": "false",
                "dino_role_correctly_limited": "true",
                "patch_linkage_sufficient_for_review_only": "false",
                "overclaim_risk": "LOW",
                "evidence_chain_completeness_score": "0.00",
                "review_only_confidence_score": "0.00",
                "recommended_review_only_status": "AUTOMATED_REVIEW_BLOCKED_INSUFFICIENT_EVIDENCE",
                "required_next_action": "GATHER_MINIMUM_EVIDENCE_BEFORE_REVIEW",
                "notes": "no cases", **base,
            })

    viol = scan_guardrails(rows, "v1tp")
    if viol:
        raise ValueError(f"Guardrail violations v1tp: {viol[:3]}")

    write_csv_with_header(OUT_DEC, rows, DEC_FIELDS)
    write_schema(SCHEMA_D, DEC_FIELDS, "v1tp_decisions")

    rubric = build_reviewer_rubric()
    write_csv_with_header(OUT_RUB, rubric, RUB_FIELDS)
    write_schema(SCHEMA_R, RUB_FIELDS, "v1tp_rubric")

    a_rows = [r for r in rows if r["reviewer_slot"] == "A"]
    b_rows = [r for r in rows if r["reviewer_slot"] == "B"]
    validated = sum(1 for r in rows if r["recommended_review_only_status"]
                    == "AUTOMATED_REVIEW_VALIDATED_FOR_REVIEW_ONLY_USE")
    summary = [
        {"stat_key": "decisions_total",       "stat_value": str(len(rows))},
        {"stat_key": "reviewer_a_decisions",  "stat_value": str(len(a_rows))},
        {"stat_key": "reviewer_b_decisions",  "stat_value": str(len(b_rows))},
        {"stat_key": "validated_review_only", "stat_value": str(validated)},
        {"stat_key": "labels_created",        "stat_value": "0"},
        {"stat_key": "targets_created",       "stat_value": "0"},
        {"stat_key": "stage",                 "stat_value": "v1tp"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1tp_summary")

    write_doc(DOC, "v1tp — Automated Reviewer A/B Adjudication", [
        "## Objetivo",
        "Dois perfis independentes de revisão automatizada por caso. Reviewer A "
        "conservador (exige fonte observacional independente, penaliza baixa "
        "precisão, bloqueia overclaim). Reviewer B integrador (valoriza "
        "consistência cruzada e contexto DINO/patch, ainda bloqueia overclaim).",
        f"## Resultado\nDecisões: {len(rows)} (A={len(a_rows)}, B={len(b_rows)}). "
        f"Validadas review-only: {validated}.",
        "## Limitação",
        "Nenhum revisor cria label, target, ground truth operacional, C3 "
        "automático, C4 ou negativo formal. DINO/hidromet são contexto.",
    ])
    print(f"[v1tp] decisions={len(rows)} a={len(a_rows)} b={len(b_rows)} validated={validated}")
    return {"decisions": len(rows), "a": len(a_rows), "b": len(b_rows)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1tp reviewer A/B").parse_args()
    run()
