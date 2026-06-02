"""REV-P v1ty — Final evidence matrix.

One row per case with evidence dimensions and the automated review-only outcome.
Review-only; DINO/hydromet are context; absence is never a negative.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1tx_v1ub_tcc_dossier_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
    guardrail_row_review, scan_guardrails, evidence_matrix_cells,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_MTX = _p("REVP_V1TY_OUT_MTX", DATASETS / "protocol_c_final_evidence_matrix_v1ty.csv")
OUT_SUM = _p("REVP_V1TY_OUT_SUM", DATASETS / "protocol_c_final_evidence_matrix_summary_v1ty.csv")
SCHEMA_M = _p("REVP_V1TY_SCHEMA_M", SCHEMAS / "protocol_c_final_evidence_matrix_v1ty_schema.csv")
SCHEMA_S = _p("REVP_V1TY_SCHEMA_S", SCHEMAS / "protocol_c_final_evidence_matrix_summary_v1ty_schema.csv")
DOC = _p("REVP_V1TY_DOC", DOCS / "revp_v1ty_final_evidence_matrix.md")

MTX_FIELDS = [
    "case_id", "region", "hazard_type",
    "external_present", "hydromet_context", "dino_context", "patch_link",
    "temporal_window", "case_readiness_status", "supervisor_decision",
    "validated_for_review_only_use", "review_only_validation_status",
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
    cases = read_csv_safe(DATASETS / "protocol_c_unified_evidence_case_index_v1tn.csv")
    sup = {r.get("case_id", ""): r for r in
           read_csv_safe(DATASETS / "protocol_c_automated_supervisor_adjudication_v1tr.csv")}
    proof = {r.get("case_id", ""): r for r in
             read_csv_safe(DATASETS / "protocol_c_proof_of_review_only_validation_audit_v1tu.csv")}

    rows: list[dict[str, Any]] = []
    for c in cases:
        cid = c.get("case_id", "")
        if cid.startswith("FAIL_CLOSED"):
            continue
        s = sup.get(cid, {})
        p = proof.get(cid, {})
        row: dict[str, Any] = {
            "case_id": cid, "region": c.get("region", ""),
            "hazard_type": c.get("hazard_type", ""),
            "case_readiness_status": c.get("case_readiness_status", ""),
            "supervisor_decision": s.get("supervisor_decision", ""),
            "validated_for_review_only_use": s.get("final_for_review_only_use", "false"),
            "review_only_validation_status": p.get("review_only_validation_status", ""),
            "notes": "",
        }
        row.update(evidence_matrix_cells(c))
        row.update(guardrail_row_review())
        rows.append(row)

    if not rows:
        row = {"case_id": "FAIL_CLOSED_NO_CASES", "region": "", "hazard_type": "",
               "external_present": "false", "hydromet_context": "false",
               "dino_context": "false", "patch_link": "false",
               "temporal_window": "false", "case_readiness_status": "",
               "supervisor_decision": "", "validated_for_review_only_use": "false",
               "review_only_validation_status": "", "notes": "no inputs"}
        row.update(guardrail_row_review())
        rows = [row]

    viol = scan_guardrails(rows, "v1ty")
    if viol:
        raise ValueError(f"Guardrail violations v1ty: {viol[:3]}")

    write_csv_with_header(OUT_MTX, rows, MTX_FIELDS)
    write_schema(SCHEMA_M, MTX_FIELDS, "v1ty_matrix")

    validated = sum(1 for r in rows if r["validated_for_review_only_use"] == "true")
    summary = [
        {"stat_key": "matrix_rows", "stat_value": str(len(rows))},
        {"stat_key": "validated_for_review_only", "stat_value": str(validated)},
        {"stat_key": "external_present_count",
         "stat_value": str(sum(1 for r in rows if r["external_present"] == "true"))},
        {"stat_key": "automatic_c3_promotion", "stat_value": "false"},
        {"stat_key": "c4_opened", "stat_value": "false"},
        {"stat_key": "stage", "stat_value": "v1ty"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1ty_summary")

    write_doc(DOC, "v1ty — Final Evidence Matrix", [
        "## Objetivo",
        "Matriz final por caso: dimensoes de evidencia (externa, hidromet, DINO, "
        "patch, temporal) e desfecho da revisao automatizada review-only.",
        f"## Resultado\nLinhas: {len(rows)}. Validadas review-only: {validated}.",
        "## Limitacao",
        "DINO/hidromet sao contexto; ausencia nao e negativo. Sem C3 automatico, "
        "sem C4, sem ground truth/rotulo/target/negativo formal.",
    ])
    print(f"[v1ty] rows={len(rows)} validated={validated}")
    return {"rows": len(rows), "validated": validated}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1ty final evidence matrix").parse_args()
    run()
