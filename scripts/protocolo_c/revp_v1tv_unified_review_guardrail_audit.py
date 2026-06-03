"""REV-P v1tv — Unified automated review guardrail audit.

Audits v1tn-v1tu outputs for absolute paths, local-runs exposure, forbidden
guardrail-true flags, and forbidden assertion phrases (e.g. "review gate
completed", "operationally validated event", "ground truth confirmed").
Fail-closed: any violation is reported per file.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1tn_v1tw_automated_review_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
    guardrail_row_review, scan_guardrails, FORBIDDEN_TRUE_FLAGS, ABS_PATH_RE,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_AUDIT = _p("REVP_V1TV_OUT_AUDIT", DATASETS / "protocol_c_unified_review_guardrail_audit_v1tv.csv")
OUT_SUM   = _p("REVP_V1TV_OUT_SUM",   DATASETS / "protocol_c_unified_review_guardrail_summary_v1tv.csv")
SCHEMA_A  = _p("REVP_V1TV_SCHEMA_A",  SCHEMAS  / "protocol_c_unified_review_guardrail_audit_v1tv_schema.csv")
SCHEMA_S  = _p("REVP_V1TV_SCHEMA_S",  SCHEMAS  / "protocol_c_unified_review_guardrail_summary_v1tv_schema.csv")
DOC       = _p("REVP_V1TV_DOC",       DOCS     / "revp_v1tv_unified_review_guardrail_audit.md")

AUDIT_FIELDS = [
    "audit_id", "source_file", "rows_scanned",
    "abs_path_hits", "lr_exposure_hits", "forbidden_true_hits",
    "forbidden_phrase_hits", "total_violations", "audit_status",
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

AUDITED_FILES = [
    "protocol_c_unified_evidence_case_index_v1tn.csv",
    "protocol_c_unified_single_case_workspace_v1to.csv",
    "protocol_c_unified_single_case_workspace_sections_v1to.csv",
    "protocol_c_automated_reviewer_ab_decisions_v1tp.csv",
    "protocol_c_review_consensus_divergence_adjudication_v1tq.csv",
    "protocol_c_automated_supervisor_adjudication_v1tr.csv",
    "protocol_c_single_flow_review_export_v1ts.csv",
    "protocol_c_single_flow_review_export_sections_v1ts.csv",
    "protocol_c_tcc_table_automated_review_case_status_v1tt.csv",
    "protocol_c_tcc_table_automated_review_outcomes_v1tt.csv",
    "protocol_c_tcc_table_review_blockers_v1tt.csv",
    "protocol_c_tcc_table_claim_safety_v1tt.csv",
    "protocol_c_proof_of_review_only_validation_audit_v1tu.csv",
]

FORBIDDEN_PHRASES = [
    "review gate completed", "revisao supervisora concluida", "revisão supervisora concluída",
    "operationally validated event", "operationally validated",
    "evento validado operacionalmente", "ground truth confirmed",
    "ground truth confirmado", "validacao operacional confirmada",
    "validação operacional confirmada",
]

_FORBIDDEN_LITERAL = "local" + "_runs"


def _scan_file(rows: list[dict[str, str]]) -> dict[str, int]:
    abs_hits = lr_hits = ft_hits = ph_hits = 0
    for row in rows:
        for f in FORBIDDEN_TRUE_FLAGS:
            if str(row.get(f, "false")).strip().lower() == "true":
                ft_hits += 1
        for v in row.values():
            sv = str(v)
            lo = sv.lower()
            if ABS_PATH_RE.search(sv):
                abs_hits += 1
            if _FORBIDDEN_LITERAL in lo:
                lr_hits += 1
            for ph in FORBIDDEN_PHRASES:
                if ph in lo:
                    ph_hits += 1
    return {"abs": abs_hits, "lr": lr_hits, "ft": ft_hits, "ph": ph_hits}


def run() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_violations = 0
    for fname in AUDITED_FILES:
        data = read_csv_safe(DATASETS / fname)
        hits = _scan_file(data)
        viols = hits["abs"] + hits["lr"] + hits["ft"] + hits["ph"]
        total_violations += viols
        row: dict[str, Any] = {
            "audit_id": f"V1TV_{fname.split('_v1')[-1].replace('.csv','')}",
            "source_file": fname, "rows_scanned": str(len(data)),
            "abs_path_hits": str(hits["abs"]), "lr_exposure_hits": str(hits["lr"]),
            "forbidden_true_hits": str(hits["ft"]),
            "forbidden_phrase_hits": str(hits["ph"]),
            "total_violations": str(viols),
            "audit_status": "GUARDRAIL_CLEAN" if viols == 0
                            else "GUARDRAIL_VIOLATION_FAIL_CLOSED",
            "notes": "",
        }
        row.update(guardrail_row_review())
        rows.append(row)

    viol = scan_guardrails(rows, "v1tv")
    if viol:
        raise ValueError(f"Guardrail violations v1tv self-rows: {viol[:3]}")

    write_csv_with_header(OUT_AUDIT, rows, AUDIT_FIELDS)
    write_schema(SCHEMA_A, AUDIT_FIELDS, "v1tv_audit")

    clean = sum(1 for r in rows if r["audit_status"] == "GUARDRAIL_CLEAN")
    summary = [
        {"stat_key": "files_audited",    "stat_value": str(len(rows))},
        {"stat_key": "files_clean",      "stat_value": str(clean)},
        {"stat_key": "total_violations", "stat_value": str(total_violations)},
        {"stat_key": "audit_status",     "stat_value":
            "ALL_CLEAN" if total_violations == 0 else "FAIL_CLOSED"},
        {"stat_key": "stage",            "stat_value": "v1tv"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1tv_summary")

    write_doc(DOC, "v1tv — Unified Automated Review Guardrail Audit", [
        "## Objetivo",
        "Auditar v1tn-v1tu: path absoluto, local_runs, label, target, ground "
        "truth, negativo formal, dino/hidromet como prova, hidromet como "
        "negativo, ausência como negativo, C3 automático, C4 aberto e frases "
        "proibidas (revisão supervisora concluída, evento validado operacionalmente, "
        "ground truth confirmado).",
        f"## Resultado\nArquivos auditados: {len(rows)}. Limpos: {clean}. "
        f"Violações: {total_violations}.",
        "## Limitação",
        "Auditoria fail-closed; qualquer violação é reportada por arquivo.",
    ])
    print(f"[v1tv] files={len(rows)} clean={clean} violations={total_violations}")
    return {"files": len(rows), "clean": clean, "violations": total_violations}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1tv guardrail audit").parse_args()
    run()
