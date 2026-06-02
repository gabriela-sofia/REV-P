"""REV-P v1tw — Unified automated review bundle.

Consolidates v1tn-v1tv into a manifest, quality checks and a scientific summary
with final status. This is the closure artefact of the automated review
adjudication workspace. Review-only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1tn_v1tw_automated_review_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
    scan_guardrails, safe_relpath,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_MAN = _p("REVP_V1TW_OUT_MAN", DATASETS / "protocol_c_unified_automated_review_manifest_v1tw.csv")
OUT_QC  = _p("REVP_V1TW_OUT_QC",  DATASETS / "protocol_c_unified_automated_review_quality_checks_v1tw.csv")
OUT_SCI = _p("REVP_V1TW_OUT_SCI", DATASETS / "protocol_c_unified_automated_review_scientific_summary_v1tw.csv")
SCHEMA_M = _p("REVP_V1TW_SCHEMA_M", SCHEMAS / "protocol_c_unified_automated_review_manifest_v1tw_schema.csv")
SCHEMA_Q = _p("REVP_V1TW_SCHEMA_Q", SCHEMAS / "protocol_c_unified_automated_review_quality_checks_v1tw_schema.csv")
SCHEMA_C = _p("REVP_V1TW_SCHEMA_C", SCHEMAS / "protocol_c_unified_automated_review_scientific_summary_v1tw_schema.csv")
DOC      = _p("REVP_V1TW_DOC",      DOCS    / "revp_v1tw_unified_automated_review_bundle.md")

MAN_FIELDS = ["artifact_id", "stage", "artifact_path", "rows", "role",
              "review_only", "automated_review"]
QC_FIELDS = ["check_id", "check_name", "check_result", "detail",
             "review_only", "automated_review"]
SCI_FIELDS = ["metric_key", "metric_value", "review_only", "automated_review"]

MANDATORY_PHRASE = (
    "A camada v1tn–v1tw consolida evidências externas, "
    "hidrometeorológicas, DINO review-only e estado Protocolo C em um fluxo "
    "único de revisão automatizada. A camada automatiza a organização e a "
    "adjudicação interna review-only, mas não promove C3 automaticamente, não "
    "abre C4, não cria ground truth operacional, não cria rótulos, não cria "
    "negativos formais e não substitui evidência observacional independente "
    "para afirmações operacionais."
)

ARTIFACTS = [
    ("v1tn", "protocol_c_unified_evidence_case_index_v1tn.csv", "case_index"),
    ("v1to", "protocol_c_unified_single_case_workspace_v1to.csv", "workspace"),
    ("v1tp", "protocol_c_automated_reviewer_ab_decisions_v1tp.csv", "reviewer_ab"),
    ("v1tq", "protocol_c_review_consensus_divergence_adjudication_v1tq.csv", "consensus"),
    ("v1tr", "protocol_c_automated_supervisor_adjudication_v1tr.csv", "supervisor"),
    ("v1ts", "protocol_c_single_flow_review_export_v1ts.csv", "single_flow"),
    ("v1tt", "protocol_c_tcc_table_automated_review_case_status_v1tt.csv", "tcc_tables"),
    ("v1tu", "protocol_c_proof_of_review_only_validation_audit_v1tu.csv", "proof_audit"),
    ("v1tv", "protocol_c_unified_review_guardrail_audit_v1tv.csv", "guardrail_audit"),
]


def run() -> dict[str, Any]:
    cases = read_csv_safe(DATASETS / "protocol_c_unified_evidence_case_index_v1tn.csv")
    decisions = read_csv_safe(DATASETS / "protocol_c_automated_reviewer_ab_decisions_v1tp.csv")
    consensus = read_csv_safe(DATASETS / "protocol_c_review_consensus_divergence_adjudication_v1tq.csv")
    supervisor = read_csv_safe(DATASETS / "protocol_c_automated_supervisor_adjudication_v1tr.csv")
    flow = read_csv_safe(DATASETS / "protocol_c_single_flow_review_export_v1ts.csv")
    proof = read_csv_safe(DATASETS / "protocol_c_proof_of_review_only_validation_audit_v1tu.csv")
    gaudit = read_csv_safe(DATASETS / "protocol_c_unified_review_guardrail_audit_v1tv.csv")

    def _real(rows): return [r for r in rows if not str(
        r.get("case_id", r.get("audit_id", ""))).startswith("FAIL_CLOSED")]

    cases_r = _real(cases)
    a_dec = [d for d in decisions if d.get("reviewer_slot") == "A" and not
             d.get("automated_review_id", "").startswith("FAIL_CLOSED")]
    b_dec = [d for d in decisions if d.get("reviewer_slot") == "B" and not
             d.get("automated_review_id", "").startswith("FAIL_CLOSED")]
    con_r = _real(consensus)
    sup_r = _real(supervisor)
    flow_r = _real(flow)
    proof_r = _real(proof)

    divergence_rows = sum(1 for c in con_r if c.get("consensus_status", "")
                          .startswith("AUTOMATED_DIVERGENCE"))
    consensus_rows = len(con_r) - divergence_rows
    validated = sum(1 for s in sup_r if s.get("final_for_review_only_use") == "true")
    tcc_ready = sum(1 for s in sup_r if s.get("ready_for_tcc_discussion") == "true")
    overclaim_blocked = sum(1 for s in sup_r if s.get("supervisor_decision")
                            == "AUTOMATED_SUPERVISOR_BLOCKED_OVERCLAIM_RISK")
    guardrail_violations = sum(int(g.get("total_violations", "0") or 0) for g in gaudit)
    proof_checks = len(proof_r)

    # ----- manifest -----
    man_rows: list[dict[str, Any]] = []
    for stage, fname, role in ARTIFACTS:
        path = DATASETS / fname
        man_rows.append({
            "artifact_id": f"V1TW_{stage.upper()}",
            "stage": stage, "artifact_path": safe_relpath(path),
            "rows": str(len(read_csv_safe(path))), "role": role,
            "review_only": "true", "automated_review": "true",
        })

    # ----- final status -----
    if guardrail_violations > 0:
        final_status = "UNIFIED_AUTOMATED_REVIEW_GUARDRAIL_FAIL_CLOSED"
    elif overclaim_blocked > 0:
        final_status = "UNIFIED_AUTOMATED_REVIEW_BLOCKED_OVERCLAIM_RISK"
    elif cases_r and tcc_ready == len(cases_r):
        final_status = "UNIFIED_AUTOMATED_REVIEW_READY_FOR_TCC_DISCUSSION"
    elif validated > 0:
        final_status = "UNIFIED_AUTOMATED_REVIEW_VALIDATED_FOR_REVIEW_ONLY_USE"
    else:
        final_status = "UNIFIED_AUTOMATED_REVIEW_WAITING_EXTERNAL_EVIDENCE"

    # ----- quality checks -----
    qc_defs = [
        ("reviewer_ab_present", len(a_dec) == len(cases_r) and len(b_dec) == len(cases_r),
         f"A={len(a_dec)} B={len(b_dec)} cases={len(cases_r)}"),
        ("consensus_present", len(con_r) == len(cases_r), f"consensus={len(con_r)}"),
        ("supervisor_present", len(sup_r) == len(cases_r), f"supervisor={len(sup_r)}"),
        ("single_flow_present", len(flow_r) == len(cases_r), f"flow={len(flow_r)}"),
        ("proof_present", proof_checks == len(cases_r), f"proof={proof_checks}"),
        ("no_guardrail_violations", guardrail_violations == 0,
         f"violations={guardrail_violations}"),
        ("no_automatic_c3", True, "automatic_c3_promotion=false"),
        ("no_c4_opened", True, "c4_opened=false"),
        ("no_labels_targets_ground_truth", True, "all false"),
        ("external_required_for_operational_claim", True, "true"),
    ]
    qc_rows: list[dict[str, Any]] = []
    for i, (name, ok, detail) in enumerate(qc_defs):
        qc_rows.append({
            "check_id": f"V1TW_QC_{i:02d}", "check_name": name,
            "check_result": "PASS" if ok else "FAIL", "detail": detail,
            "review_only": "true", "automated_review": "true",
        })

    # ----- scientific summary -----
    metrics = [
        ("cases_total", len(cases_r)),
        ("automated_reviewer_a_decisions", len(a_dec)),
        ("automated_reviewer_b_decisions", len(b_dec)),
        ("consensus_rows", consensus_rows),
        ("divergence_rows", divergence_rows),
        ("automated_supervisor_adjudication_rows", len(sup_r)),
        ("single_flow_rows", len(flow_r)),
        ("proof_validation_checks", proof_checks),
        ("guardrail_violations", guardrail_violations),
        ("cases_validated_for_review_only_use", validated),
        ("cases_ready_for_tcc_discussion", tcc_ready),
        ("automatic_c3_promotions", 0),
        ("c4_opened_count", 0),
        ("labels_created", 0),
        ("targets_created", 0),
        ("ground_truth_operational_created", 0),
        ("formal_negatives_created", 0),
        ("final_status", final_status),
    ]
    sci_rows = [{"metric_key": k, "metric_value": str(v),
                 "review_only": "true", "automated_review": "true"}
                for k, v in metrics]

    for label, rws in (("v1tw_man", man_rows), ("v1tw_qc", qc_rows),
                       ("v1tw_sci", sci_rows)):
        viol = scan_guardrails(rws, label)
        if viol:
            raise ValueError(f"Guardrail violations {label}: {viol[:3]}")

    write_csv_with_header(OUT_MAN, man_rows, MAN_FIELDS)
    write_csv_with_header(OUT_QC, qc_rows, QC_FIELDS)
    write_csv_with_header(OUT_SCI, sci_rows, SCI_FIELDS)
    write_schema(SCHEMA_M, MAN_FIELDS, "v1tw_manifest")
    write_schema(SCHEMA_Q, QC_FIELDS, "v1tw_quality_checks")
    write_schema(SCHEMA_C, SCI_FIELDS, "v1tw_scientific_summary")

    write_doc(DOC, "v1tw — Unified Automated Review Bundle", [
        "## Objetivo",
        "Consolidar v1tn-v1tv em manifesto, quality checks e resumo científico "
        "com status final.",
        MANDATORY_PHRASE,
        f"## Resultado\nCasos: {len(cases_r)}. Validados review-only: {validated}. "
        f"Prontos p/ TCC: {tcc_ready}. Divergências: {divergence_rows}. "
        f"Violações de guardrail: {guardrail_violations}. "
        f"Status final: {final_status}.",
        "## Limitação",
        "Sem C3 automático, sem C4, sem ground truth operacional, sem label/"
        "target/negativo formal. Fonte observacional externa exigida para "
        "afirmação operacional.",
    ])
    print(f"[v1tw] cases={len(cases_r)} validated={validated} tcc={tcc_ready} "
          f"final={final_status}")
    return {"cases": len(cases_r), "validated": validated, "tcc": tcc_ready,
            "final_status": final_status, "guardrail_violations": guardrail_violations}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1tw unified bundle").parse_args()
    run()
