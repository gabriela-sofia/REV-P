"""REV-P v1tt — TCC tables for automated review.

Emits four TCC-facing tables: case status, review outcomes, blockers, and claim
safety. Tables make explicit what is validated for review-only, what is still
blocked, why C3 is not automatic, why C4 stays closed, and why DINO/hydromet
are context only. Review-only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1tn_v1tw_automated_review_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
    scan_guardrails,
    classify_review_only_validation_status,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_STATUS  = _p("REVP_V1TT_OUT_STATUS",  DATASETS / "protocol_c_tcc_table_automated_review_case_status_v1tt.csv")
OUT_OUTCOME = _p("REVP_V1TT_OUT_OUTCOME", DATASETS / "protocol_c_tcc_table_automated_review_outcomes_v1tt.csv")
OUT_BLOCK   = _p("REVP_V1TT_OUT_BLOCK",   DATASETS / "protocol_c_tcc_table_review_blockers_v1tt.csv")
OUT_SAFETY  = _p("REVP_V1TT_OUT_SAFETY",  DATASETS / "protocol_c_tcc_table_claim_safety_v1tt.csv")
SCHEMA_ST   = _p("REVP_V1TT_SCHEMA_ST",   SCHEMAS  / "protocol_c_tcc_table_automated_review_case_status_v1tt_schema.csv")
SCHEMA_OU   = _p("REVP_V1TT_SCHEMA_OU",   SCHEMAS  / "protocol_c_tcc_table_automated_review_outcomes_v1tt_schema.csv")
SCHEMA_BL   = _p("REVP_V1TT_SCHEMA_BL",   SCHEMAS  / "protocol_c_tcc_table_review_blockers_v1tt_schema.csv")
SCHEMA_SA   = _p("REVP_V1TT_SCHEMA_SA",   SCHEMAS  / "protocol_c_tcc_table_claim_safety_v1tt_schema.csv")
DOC         = _p("REVP_V1TT_DOC",         DOCS     / "revp_v1tt_automated_review_tcc_tables.md")

STATUS_FIELDS = ["case_id", "region", "hazard_type", "case_readiness_status",
                 "supervisor_decision", "review_only_validation_status",
                 "ready_for_tcc_discussion",
                 "review_only", "automated_review",
                 "automatic_c3_promotion", "c4_opened"]
OUTCOME_FIELDS = ["case_id", "reviewer_a_status", "reviewer_b_status",
                  "consensus_status", "supervisor_decision",
                  "final_for_review_only_use",
                  "review_only", "automated_review",
                  "ground_truth_operational", "formal_negative"]
BLOCK_FIELDS = ["case_id", "blocking_factors", "next_required_action",
                "why_c3_not_automatic", "why_c4_closed",
                "review_only", "automated_review"]
SAFETY_FIELDS = ["case_id", "claim_safety_status",
                 "dino_role", "hydromet_role", "absence_role",
                 "external_required_for_operational_claim",
                 "review_only", "automated_review",
                 "dino_validates_event", "hydromet_validates_event",
                 "hydromet_is_negative_evidence", "absence_as_negative"]


def run() -> dict[str, Any]:
    cases = read_csv_safe(DATASETS / "protocol_c_unified_evidence_case_index_v1tn.csv")
    decisions = read_csv_safe(DATASETS / "protocol_c_automated_reviewer_ab_decisions_v1tp.csv")
    consensus = read_csv_safe(DATASETS / "protocol_c_review_consensus_divergence_adjudication_v1tq.csv")
    supervisor = read_csv_safe(DATASETS / "protocol_c_automated_supervisor_adjudication_v1tr.csv")

    dec_by_case: dict[str, dict[str, str]] = {}
    for d in decisions:
        dec_by_case.setdefault(d.get("case_id", ""), {})[d.get("reviewer_slot", "")] = \
            d.get("recommended_review_only_status", "")
    con_by_case = {c.get("case_id", ""): c for c in consensus}
    sup_by_case = {s.get("case_id", ""): s for s in supervisor}

    status_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    safety_rows: list[dict[str, Any]] = []

    for c in cases:
        cid = c.get("case_id", "")
        if cid.startswith("FAIL_CLOSED"):
            continue
        sup = sup_by_case.get(cid, {})
        con = con_by_case.get(cid, {})
        sup_decision = sup.get("supervisor_decision", "N/A")
        validation = classify_review_only_validation_status(sup_decision)

        status_rows.append({
            "case_id": cid, "region": c.get("region", ""),
            "hazard_type": c.get("hazard_type", ""),
            "case_readiness_status": c.get("case_readiness_status", ""),
            "supervisor_decision": sup_decision,
            "review_only_validation_status": validation,
            "ready_for_tcc_discussion": sup.get("ready_for_tcc_discussion", "false"),
            "review_only": "true", "automated_review": "true",
            "automatic_c3_promotion": "false", "c4_opened": "false",
        })
        outcome_rows.append({
            "case_id": cid,
            "reviewer_a_status": dec_by_case.get(cid, {}).get("A", ""),
            "reviewer_b_status": dec_by_case.get(cid, {}).get("B", ""),
            "consensus_status": con.get("consensus_status", ""),
            "supervisor_decision": sup_decision,
            "final_for_review_only_use": sup.get("final_for_review_only_use", "false"),
            "review_only": "true", "automated_review": "true",
            "ground_truth_operational": "false", "formal_negative": "false",
        })
        block_rows.append({
            "case_id": cid, "blocking_factors": c.get("blocking_factors", ""),
            "next_required_action": c.get("next_required_action", ""),
            "why_c3_not_automatic": (
                "C3 exige supervisão e fonte observacional externa; revisão "
                "automatizada é review-only e não promove C3 automaticamente."),
            "why_c4_closed": (
                "C4 (negativo formal) permanece fechado; ausência de chuva/registro "
                "não é negativo formal."),
            "review_only": "true", "automated_review": "true",
        })
        safety_rows.append({
            "case_id": cid,
            "claim_safety_status": "REVIEW_ONLY_SAFE_NO_OPERATIONAL_CLAIM",
            "dino_role": "REPRESENTATION_CONTEXT_ONLY",
            "hydromet_role": "HYDROMETEOROLOGICAL_CONTEXT_ONLY",
            "absence_role": "ABSENCE_NOT_NEGATIVE",
            "external_required_for_operational_claim": "true",
            "review_only": "true", "automated_review": "true",
            "dino_validates_event": "false", "hydromet_validates_event": "false",
            "hydromet_is_negative_evidence": "false", "absence_as_negative": "false",
        })

    if not status_rows:
        status_rows = [{"case_id": "FAIL_CLOSED_NO_CASES", "region": "",
                        "hazard_type": "", "case_readiness_status": "",
                        "supervisor_decision": "", "review_only_validation_status":
                        "NOT_VALIDATED_FOR_REVIEW_ONLY_USE",
                        "ready_for_tcc_discussion": "false", "review_only": "true",
                        "automated_review": "true",
                        "automatic_c3_promotion": "false", "c4_opened": "false"}]
        outcome_rows = [{"case_id": "FAIL_CLOSED_NO_CASES", "reviewer_a_status": "",
                         "reviewer_b_status": "", "consensus_status": "",
                         "supervisor_decision": "", "final_for_review_only_use": "false",
                         "review_only": "true", "automated_review": "true",
                         "ground_truth_operational": "false", "formal_negative": "false"}]
        block_rows = [{"case_id": "FAIL_CLOSED_NO_CASES", "blocking_factors": "NO_INPUTS",
                       "next_required_action": "GATHER_MINIMUM_EVIDENCE_BEFORE_REVIEW",
                       "why_c3_not_automatic": "n/a", "why_c4_closed": "n/a",
                       "review_only": "true", "automated_review": "true"}]
        safety_rows = [{"case_id": "FAIL_CLOSED_NO_CASES",
                        "claim_safety_status": "REVIEW_ONLY_SAFE_NO_OPERATIONAL_CLAIM",
                        "dino_role": "REPRESENTATION_CONTEXT_ONLY",
                        "hydromet_role": "HYDROMETEOROLOGICAL_CONTEXT_ONLY",
                        "absence_role": "ABSENCE_NOT_NEGATIVE",
                        "external_required_for_operational_claim": "true",
                        "review_only": "true", "automated_review": "true",
                        "dino_validates_event": "false", "hydromet_validates_event": "false",
                        "hydromet_is_negative_evidence": "false", "absence_as_negative": "false"}]

    tables = [
        ("v1tt_status", status_rows, OUT_STATUS, STATUS_FIELDS, SCHEMA_ST),
        ("v1tt_outcome", outcome_rows, OUT_OUTCOME, OUTCOME_FIELDS, SCHEMA_OU),
        ("v1tt_block", block_rows, OUT_BLOCK, BLOCK_FIELDS, SCHEMA_BL),
        ("v1tt_safety", safety_rows, OUT_SAFETY, SAFETY_FIELDS, SCHEMA_SA),
    ]
    for label, rws, out, fields, schema in tables:
        viol = scan_guardrails(rws, label)
        if viol:
            raise ValueError(f"Guardrail violations {label}: {viol[:3]}")
        write_csv_with_header(out, rws, fields)
        write_schema(schema, fields, label)

    write_doc(DOC, "v1tt — Automated Review TCC Tables", [
        "## Objetivo",
        "Quatro tabelas para o TCC: case status, outcomes, blockers e claim "
        "safety. Explicitam o que foi validado review-only, o que segue "
        "bloqueado, por que C3 não é automático e por que C4 segue fechado.",
        f"## Resultado\nCasos: {len(status_rows)}.",
        "## Limitação",
        "DINO/hidromet são contexto; ausência não é negativo. Nenhuma tabela "
        "cria label/target/ground truth/negativo formal ou promove C3/C4.",
    ])
    print(f"[v1tt] status={len(status_rows)} outcomes={len(outcome_rows)} "
          f"blockers={len(block_rows)} safety={len(safety_rows)}")
    return {"status": len(status_rows), "outcomes": len(outcome_rows),
            "blockers": len(block_rows), "safety": len(safety_rows)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1tt TCC tables").parse_args()
    run()
