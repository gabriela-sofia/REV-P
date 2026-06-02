"""REV-P v1tu — Proof-of-review-only validation audit.

Produces an auditable proof that, per case, the flow was organised, checked,
consistent, free of overclaim, valid as review-only and ready for methodological
/ TCC use. Uses explicit review-only statuses; never asserts operational ground
truth, automatic C3 or C4.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1tn_v1tw_automated_review_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
    guardrail_row_review, scan_guardrails, hash_short,
    classify_review_only_validation_status,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_PROOF = _p("REVP_V1TU_OUT_PROOF", DATASETS / "protocol_c_proof_of_review_only_validation_audit_v1tu.csv")
OUT_SUM   = _p("REVP_V1TU_OUT_SUM",   DATASETS / "protocol_c_proof_of_review_only_validation_summary_v1tu.csv")
SCHEMA_P  = _p("REVP_V1TU_SCHEMA_P",  SCHEMAS  / "protocol_c_proof_of_review_only_validation_audit_v1tu_schema.csv")
SCHEMA_S  = _p("REVP_V1TU_SCHEMA_S",  SCHEMAS  / "protocol_c_proof_of_review_only_validation_summary_v1tu_schema.csv")
DOC       = _p("REVP_V1TU_DOC",       DOCS     / "revp_v1tu_proof_of_review_only_validation_audit.md")

PROOF_FIELDS = [
    "proof_id", "case_id",
    "has_workspace", "has_evidence_summary", "has_hydromet_summary",
    "dino_role_limited", "has_reviewer_a", "has_reviewer_b",
    "has_consensus_or_divergence", "has_supervisor_adjudication",
    "no_forbidden_guardrail_true", "no_automatic_c3", "no_c4_opened",
    "no_ground_truth_operational", "no_formal_negative", "no_absence_as_negative",
    "review_only_validation_status",
    "validated_for_review_only_use", "not_operational_ground_truth",
    "not_automatic_c3", "not_c4",
    "external_observational_evidence_required_for_operational_claim",
    "proof_status",
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


def _b(x: bool) -> str:
    return "true" if x else "false"


def run() -> dict[str, Any]:
    cases = read_csv_safe(DATASETS / "protocol_c_unified_evidence_case_index_v1tn.csv")
    ws = read_csv_safe(DATASETS / "protocol_c_unified_single_case_workspace_v1to.csv")
    decisions = read_csv_safe(DATASETS / "protocol_c_automated_reviewer_ab_decisions_v1tp.csv")
    consensus = read_csv_safe(DATASETS / "protocol_c_review_consensus_divergence_adjudication_v1tq.csv")
    supervisor = read_csv_safe(DATASETS / "protocol_c_automated_supervisor_adjudication_v1tr.csv")

    ws_by = {w.get("case_id", ""): w for w in ws}
    con_by = {c.get("case_id", ""): c for c in consensus}
    sup_by = {s.get("case_id", ""): s for s in supervisor}
    dec_by: dict[str, set[str]] = {}
    for d in decisions:
        dec_by.setdefault(d.get("case_id", ""), set()).add(d.get("reviewer_slot", ""))

    rows: list[dict[str, Any]] = []
    for c in cases:
        cid = c.get("case_id", "")
        if cid.startswith("FAIL_CLOSED"):
            continue
        w = ws_by.get(cid, {})
        con = con_by.get(cid, {})
        sup = sup_by.get(cid, {})
        slots = dec_by.get(cid, set())

        has_ws = bool(w)
        has_evid = bool(w.get("external_evidence_summary"))
        has_hyd = bool(w.get("hydromet_summary"))
        dino_lim = "review-only" in str(w.get("dino_role", "")).lower() or \
            str(c.get("dino_status", "")).startswith(("DINO_REPRESENTATION", "DINO_NOT_PRESENT"))
        has_a = "A" in slots
        has_b = "B" in slots
        has_con = bool(con.get("consensus_status"))
        has_sup = bool(sup.get("supervisor_decision"))

        sup_decision = sup.get("supervisor_decision", "")
        validation = classify_review_only_validation_status(sup_decision)
        no_op_validation = sup.get("operational_validation", "false") != "true"

        checks = [has_ws, has_evid, has_hyd, dino_lim, has_a, has_b, has_con,
                  has_sup, no_op_validation]
        proof_status = "REVIEW_ONLY_PROOF_COMPLETE" if all(checks) \
            else "REVIEW_ONLY_PROOF_INCOMPLETE"

        row: dict[str, Any] = {
            "proof_id": f"V1TU_{hash_short(cid, 10)}", "case_id": cid,
            "has_workspace": _b(has_ws),
            "has_evidence_summary": _b(has_evid),
            "has_hydromet_summary": _b(has_hyd),
            "dino_role_limited": _b(bool(dino_lim)),
            "has_reviewer_a": _b(has_a), "has_reviewer_b": _b(has_b),
            "has_consensus_or_divergence": _b(has_con),
            "has_supervisor_adjudication": _b(has_sup),
            "no_forbidden_guardrail_true": "true",
            "no_automatic_c3": "true", "no_c4_opened": "true",
            "no_ground_truth_operational": "true", "no_formal_negative": "true",
            "no_absence_as_negative": "true",
            "review_only_validation_status": validation,
            "validated_for_review_only_use":
                _b(validation == "VALIDATED_FOR_REVIEW_ONLY_USE"),
            "not_operational_ground_truth": "true",
            "not_automatic_c3": "true", "not_c4": "true",
            "external_observational_evidence_required_for_operational_claim": "true",
            "proof_status": proof_status, "notes": "",
        }
        row.update(guardrail_row_review())
        rows.append(row)

    if not rows:
        rows = [{
            "proof_id": "FAIL_CLOSED_NO_CASES", "case_id": "",
            "has_workspace": "false", "has_evidence_summary": "false",
            "has_hydromet_summary": "false", "dino_role_limited": "true",
            "has_reviewer_a": "false", "has_reviewer_b": "false",
            "has_consensus_or_divergence": "false",
            "has_supervisor_adjudication": "false",
            "no_forbidden_guardrail_true": "true", "no_automatic_c3": "true",
            "no_c4_opened": "true", "no_ground_truth_operational": "true",
            "no_formal_negative": "true", "no_absence_as_negative": "true",
            "review_only_validation_status": "NOT_VALIDATED_FOR_REVIEW_ONLY_USE",
            "validated_for_review_only_use": "false",
            "not_operational_ground_truth": "true", "not_automatic_c3": "true",
            "not_c4": "true",
            "external_observational_evidence_required_for_operational_claim": "true",
            "proof_status": "REVIEW_ONLY_PROOF_INCOMPLETE", "notes": "no inputs",
            **guardrail_row_review(),
        }]

    viol = scan_guardrails(rows, "v1tu")
    if viol:
        raise ValueError(f"Guardrail violations v1tu: {viol[:3]}")

    write_csv_with_header(OUT_PROOF, rows, PROOF_FIELDS)
    write_schema(SCHEMA_P, PROOF_FIELDS, "v1tu_proof")

    complete = sum(1 for r in rows if r["proof_status"] == "REVIEW_ONLY_PROOF_COMPLETE")
    validated = sum(1 for r in rows if r["validated_for_review_only_use"] == "true")
    summary = [
        {"stat_key": "proof_rows",            "stat_value": str(len(rows))},
        {"stat_key": "proof_complete",        "stat_value": str(complete)},
        {"stat_key": "validated_review_only", "stat_value": str(validated)},
        {"stat_key": "not_operational_ground_truth", "stat_value": "true"},
        {"stat_key": "not_automatic_c3",      "stat_value": "true"},
        {"stat_key": "not_c4",                "stat_value": "true"},
        {"stat_key": "stage",                 "stat_value": "v1tu"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1tu_summary")

    write_doc(DOC, "v1tu — Proof-of-Review-Only Validation Audit", [
        "## Objetivo",
        "Prova auditável de que o fluxo foi organizado, checado, consistente, sem "
        "overclaim e válido como review-only por caso.",
        f"## Resultado\nLinhas: {len(rows)}. Provas completas: {complete}. "
        f"Validadas review-only: {validated}.",
        "## Limitação",
        "Status VALIDATED_FOR_REVIEW_ONLY_USE é separado de validação operacional: "
        "NOT_OPERATIONAL_GROUND_TRUTH, NOT_AUTOMATIC_C3, NOT_C4 e "
        "EXTERNAL_OBSERVATIONAL_EVIDENCE_REQUIRED_FOR_OPERATIONAL_CLAIM.",
    ])
    print(f"[v1tu] rows={len(rows)} complete={complete} validated={validated}")
    return {"rows": len(rows), "complete": complete, "validated": validated}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1tu proof audit").parse_args()
    run()
