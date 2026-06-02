"""REV-P v1tq — review consensus / divergence adjudication.

Compares Reviewer A vs Reviewer B per case. Emits consensus when they agree,
classifies divergence type when they disagree, and flags whether an external
human supervisor or only the internal automated supervisor is required. Never promotes
C3, opens C4, or creates ground truth / formal negatives.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1tn_v1tw_automated_review_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
    guardrail_row_review, scan_guardrails, hash_short,
    classify_consensus, supervisor_adjudication_required,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_CON  = _p("REVP_V1TQ_OUT_CON",  DATASETS / "protocol_c_review_consensus_divergence_adjudication_v1tq.csv")
OUT_SUM  = _p("REVP_V1TQ_OUT_SUM",  DATASETS / "protocol_c_review_consensus_divergence_summary_v1tq.csv")
SCHEMA_C = _p("REVP_V1TQ_SCHEMA_C", SCHEMAS  / "protocol_c_review_consensus_divergence_adjudication_v1tq_schema.csv")
SCHEMA_S = _p("REVP_V1TQ_SCHEMA_S", SCHEMAS  / "protocol_c_review_consensus_divergence_summary_v1tq_schema.csv")
DOC      = _p("REVP_V1TQ_DOC",      DOCS     / "revp_v1tq_review_consensus_divergence_adjudication.md")

CON_FIELDS = [
    "consensus_id", "case_id", "reviewer_a_status", "reviewer_b_status",
    "consensus_status", "divergence_type", "supervisor_adjudication_required",
    "external_observational_evidence_required_for_operational_claim",
    "recommended_next_action", "review_only", "automated_review",
    "internal_review_automated_for_review_only",
    "requires_external_observational_evidence_for_operational_claim",
    "automatic_c3_promotion", "c4_opened",
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "formal_negative",
    "dino_validates_event", "hydromet_validates_event",
    "hydromet_is_negative_evidence", "absence_as_negative", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]

_NEXT = {
    "AUTOMATED_CONSENSUS_VALIDATED_FOR_REVIEW_ONLY_USE": "PROCEED_TO_AUTOMATED_SUPERVISOR_FINALISATION",
    "AUTOMATED_CONSENSUS_BLOCKED_NEEDS_EXTERNAL_SOURCE": "COLLECT_EXTERNAL_OBSERVATIONAL_SOURCE",
    "AUTOMATED_CONSENSUS_BLOCKED_TEMPORAL_SPATIAL": "RECOVER_TEMPORAL_OR_SPATIAL_PRECISION",
    "AUTOMATED_DIVERGENCE_REQUIRES_SUPERVISOR_ADJUDICATION": "ESCALATE_TO_AUTOMATED_SUPERVISOR_ADJUDICATION",
    "AUTOMATED_CONSENSUS_OVERCLAIM_RISK": "BLOCK_AND_REWORK_OVERCLAIM",
}


def run() -> dict[str, Any]:
    decisions = read_csv_safe(DATASETS / "protocol_c_automated_reviewer_ab_decisions_v1tp.csv")
    by_case: dict[str, dict[str, str]] = {}
    for d in decisions:
        cid = d.get("case_id", "")
        if not cid or cid.startswith("FAIL_CLOSED"):
            continue
        by_case.setdefault(cid, {})[d.get("reviewer_slot", "")] = \
            d.get("recommended_review_only_status", "")

    rows: list[dict[str, Any]] = []
    for cid, slots in by_case.items():
        a = slots.get("A", "")
        b = slots.get("B", "")
        consensus, divergence = classify_consensus(a, b)
        row: dict[str, Any] = {
            "consensus_id": f"V1TQ_{hash_short(cid, 10)}",
            "case_id": cid,
            "reviewer_a_status": a, "reviewer_b_status": b,
            "consensus_status": consensus, "divergence_type": divergence,
            "supervisor_adjudication_required":
                supervisor_adjudication_required(consensus),
            "external_observational_evidence_required_for_operational_claim": "true",
            "recommended_next_action": _NEXT.get(consensus, "REVIEW_MANUALLY"),
            "notes": "",
        }
        row.update(guardrail_row_review())
        rows.append(row)

    if not rows:
        rows = [{
            "consensus_id": "FAIL_CLOSED_NO_DECISIONS", "case_id": "",
            "reviewer_a_status": "", "reviewer_b_status": "",
            "consensus_status": "AUTOMATED_CONSENSUS_BLOCKED_NEEDS_EXTERNAL_SOURCE",
            "divergence_type": "NONE", "supervisor_adjudication_required": "false",
            "external_observational_evidence_required_for_operational_claim": "true",
            "recommended_next_action": "COLLECT_EXTERNAL_OBSERVATIONAL_SOURCE",
            "notes": "no decisions", **guardrail_row_review(),
        }]

    viol = scan_guardrails(rows, "v1tq")
    if viol:
        raise ValueError(f"Guardrail violations v1tq: {viol[:3]}")

    write_csv_with_header(OUT_CON, rows, CON_FIELDS)
    write_schema(SCHEMA_C, CON_FIELDS, "v1tq_consensus")

    consensus_rows = sum(1 for r in rows if not r["consensus_status"]
                         .startswith("AUTOMATED_DIVERGENCE"))
    divergence_rows = sum(1 for r in rows if r["consensus_status"]
                          .startswith("AUTOMATED_DIVERGENCE"))
    summary = [
        {"stat_key": "rows_total",            "stat_value": str(len(rows))},
        {"stat_key": "consensus_rows",        "stat_value": str(consensus_rows)},
        {"stat_key": "divergence_rows",       "stat_value": str(divergence_rows)},
        {"stat_key": "automatic_c3_promotion","stat_value": "false"},
        {"stat_key": "c4_opened",             "stat_value": "false"},
        {"stat_key": "stage",                 "stat_value": "v1tq"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1tq_summary")

    write_doc(DOC, "v1tq — Review Consensus / Divergence Adjudication", [
        "## Objetivo",
        "Comparar Reviewer A e Reviewer B por caso: consenso quando concordam, "
        "tipo de divergência quando discordam, e necessidade de supervisor.",
        f"## Resultado\nLinhas: {len(rows)}. Consenso: {consensus_rows}. "
        f"Divergência: {divergence_rows}.",
        "## Limitação",
        "Não promove C3, não abre C4, não cria ground truth nem negativo formal. "
        "Fonte observacional externa segue exigida para afirmação operacional.",
    ])
    print(f"[v1tq] rows={len(rows)} consensus={consensus_rows} divergence={divergence_rows}")
    return {"rows": len(rows), "consensus": consensus_rows, "divergence": divergence_rows}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1tq consensus/divergence").parse_args()
    run()
