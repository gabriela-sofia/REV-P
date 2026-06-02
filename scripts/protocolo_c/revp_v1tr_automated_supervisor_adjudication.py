"""REV-P v1tr — Automated supervisor adjudication.

Runs the internal automated supervisor: evaluates consensus/divergence, checks
guardrails, and declares a final review-only decision plus TCC readiness. Never
emits an operational decision; external observational source remains required
for operational claims.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1tn_v1tw_automated_review_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
    guardrail_row_review, scan_guardrails, hash_short, parse_float_safe,
    classify_supervisor_precheck, classify_supervisor_decision,
    supervisor_final_for_review_only, supervisor_ready_for_tcc,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_SUP  = _p("REVP_V1TR_OUT_SUP",  DATASETS / "protocol_c_automated_supervisor_adjudication_v1tr.csv")
OUT_SUM  = _p("REVP_V1TR_OUT_SUM",  DATASETS / "protocol_c_automated_supervisor_adjudication_summary_v1tr.csv")
SCHEMA_A = _p("REVP_V1TR_SCHEMA_A", SCHEMAS  / "protocol_c_automated_supervisor_adjudication_v1tr_schema.csv")
SCHEMA_S = _p("REVP_V1TR_SCHEMA_S", SCHEMAS  / "protocol_c_automated_supervisor_adjudication_summary_v1tr_schema.csv")
DOC      = _p("REVP_V1TR_DOC",      DOCS     / "revp_v1tr_automated_supervisor_adjudication.md")

SUP_FIELDS = [
    "supervisor_adjudication_id", "case_id", "consensus_status",
    "supervisor_precheck_status", "supervisor_decision",
    "final_for_review_only_use", "ready_for_tcc_discussion",
    "evidence_chain_completeness_score",
    "automated_supervisor_adjudication", "operational_validation",
    "external_observational_source_required_for_operational_claim",
    "does_not_validate_event", "supervisor_final_operational_decision_allowed",
    "review_only", "automated_review",
    "internal_review_automated_for_review_only",
    "requires_external_observational_evidence_for_operational_claim",
    "automatic_c3_promotion", "c4_opened",
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "formal_negative",
    "dino_validates_event", "hydromet_validates_event",
    "hydromet_is_negative_evidence", "absence_as_negative", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def run() -> dict[str, Any]:
    consensus = read_csv_safe(DATASETS / "protocol_c_review_consensus_divergence_adjudication_v1tq.csv")
    cases = read_csv_safe(DATASETS / "protocol_c_unified_evidence_case_index_v1tn.csv")
    decisions = read_csv_safe(DATASETS / "protocol_c_automated_reviewer_ab_decisions_v1tp.csv")

    completeness_by_case: dict[str, float] = {}
    for d in decisions:
        cid = d.get("case_id", "")
        v = parse_float_safe(d.get("evidence_chain_completeness_score", "0"), 0.0)
        completeness_by_case[cid] = max(completeness_by_case.get(cid, 0.0), v)

    ext_by_case: dict[str, bool] = {}
    for c in cases:
        ext_by_case[c.get("case_id", "")] = str(
            c.get("external_evidence_status", "")).startswith("EXTERNAL_CANDIDATE_PRESENT")

    rows: list[dict[str, Any]] = []
    for con in consensus:
        cid = con.get("case_id", "")
        if not cid or cid.startswith("FAIL_CLOSED"):
            continue
        cstatus = con.get("consensus_status", "")
        guardrail_clean = str(con.get("formal_negative", "false")) != "true"
        completeness = completeness_by_case.get(cid, 0.0)
        has_ext = ext_by_case.get(cid, False)

        precheck = classify_supervisor_precheck(guardrail_clean)
        decision = classify_supervisor_decision(
            cstatus, completeness, precheck == "SUPERVISOR_PRECHECK_PASS", has_ext)

        row: dict[str, Any] = {
            "supervisor_adjudication_id": f"V1TR_{hash_short(cid, 10)}",
            "case_id": cid, "consensus_status": cstatus,
            "supervisor_precheck_status": precheck,
            "supervisor_decision": decision,
            "final_for_review_only_use": supervisor_final_for_review_only(decision),
            "ready_for_tcc_discussion": supervisor_ready_for_tcc(decision),
            "evidence_chain_completeness_score": f"{completeness:.2f}",
            "automated_supervisor_adjudication": "true",
            "operational_validation": "false",
            "external_observational_source_required_for_operational_claim":
                "true" if not has_ext else "false",
            "does_not_validate_event": "true",
            "supervisor_final_operational_decision_allowed": "false",
            "notes": "",
        }
        row.update(guardrail_row_review())
        rows.append(row)

    if not rows:
        base = guardrail_row_review()
        rows = [{
            "supervisor_adjudication_id": "FAIL_CLOSED_NO_CONSENSUS",
            "case_id": "", "consensus_status": "",
            "supervisor_precheck_status": "SUPERVISOR_PRECHECK_PASS",
            "supervisor_decision": "AUTOMATED_SUPERVISOR_BLOCKED_INSUFFICIENT_EVIDENCE",
            "final_for_review_only_use": "false",
            "ready_for_tcc_discussion": "false",
            "evidence_chain_completeness_score": "0.00",
            "automated_supervisor_adjudication": "true", "operational_validation": "false",
            "external_observational_source_required_for_operational_claim": "true",
            "does_not_validate_event": "true",
            "supervisor_final_operational_decision_allowed": "false",
            "notes": "no consensus", **base,
        }]

    viol = scan_guardrails(rows, "v1tr")
    if viol:
        raise ValueError(f"Guardrail violations v1tr: {viol[:3]}")
    # Hard invariants: supervisor never validates operationally.
    for r in rows:
        if r.get("operational_validation") == "true" or \
           r.get("supervisor_final_operational_decision_allowed") == "true":
            raise ValueError("v1tr supervisor emitted operational validation")

    write_csv_with_header(OUT_SUP, rows, SUP_FIELDS)
    write_schema(SCHEMA_A, SUP_FIELDS, "v1tr_supervisor")

    validated = sum(1 for r in rows if r["final_for_review_only_use"] == "true")
    tcc_ready = sum(1 for r in rows if r["ready_for_tcc_discussion"] == "true")
    waiting = sum(1 for r in rows if r["supervisor_decision"]
                  == "AUTOMATED_SUPERVISOR_WAITING_EXTERNAL_OBSERVATIONAL_SOURCE")
    summary = [
        {"stat_key": "supervisor_rows",          "stat_value": str(len(rows))},
        {"stat_key": "validated_for_review_only","stat_value": str(validated)},
        {"stat_key": "ready_for_tcc_discussion", "stat_value": str(tcc_ready)},
        {"stat_key": "waiting_external_source",  "stat_value": str(waiting)},
        {"stat_key": "operational_validation",   "stat_value": "false"},
        {"stat_key": "stage",                    "stat_value": "v1tr"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1tr_summary")

    write_doc(DOC, "v1tr — Automated Supervisor Adjudication", [
        "## Objetivo",
        "Supervisor automatizado interno: avalia consenso/divergência, checa "
        "guardrails e "
        "declara decisão final apenas para uso review-only e prontidão para TCC.",
        f"## Resultado\nLinhas: {len(rows)}. Validadas review-only: {validated}. "
        f"Prontas p/ TCC: {tcc_ready}. Aguardando fonte externa: {waiting}.",
        "## Limitação",
        "operational_validation=false sempre; supervisor_final_operational_decision_"
        "allowed=false sempre. Não valida evento operacionalmente; fonte "
        "observacional externa exigida para afirmação operacional.",
    ])
    print(f"[v1tr] rows={len(rows)} validated={validated} tcc={tcc_ready} waiting={waiting}")
    return {"rows": len(rows), "validated": validated, "tcc": tcc_ready}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1tr supervisor adjudication").parse_args()
    run()
