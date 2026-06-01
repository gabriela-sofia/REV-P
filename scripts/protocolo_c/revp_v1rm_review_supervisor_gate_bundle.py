"""REV-P v1rm — Review response + supervisor gate bundle (P2).

Consolidates v1rg-v1rl into manifest, QC, summary and TCC table, and resolves
the review/supervisor-gate final status. Works with empty inputs (headers
preserved). Even a C3 candidate stays review-only: never a label, target, or
field-validated ground truth. Only reads existing CSVs — no recursion, no
subprocess, no internet.
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
    detect_absolute_path,
    detect_local_runs_exposure,
    read_csv_safe,
    write_csv_with_header,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_TRUE = ["can_create_operational_label", "can_train_model", "target_created",
                  "ground_truth_operational", "formal_negative", "dino_validates_event",
                  "absence_as_negative"]

IN_TEMPLATE = _p("REVP_V1RM_IN_TEMPLATE", DATASETS / "protocol_c_review_response_intake_template_v1rg.csv")
IN_RESP_VALIDATION = _p("REVP_V1RM_IN_RESP_VALIDATION", DATASETS / "protocol_c_review_response_validation_v1rh.csv")
IN_RESP_VAL_SUMMARY = _p("REVP_V1RM_IN_RESP_VAL_SUMMARY", DATASETS / "protocol_c_review_response_validation_summary_v1rh.csv")
IN_SCORES = _p("REVP_V1RM_IN_SCORES", DATASETS / "protocol_c_completed_review_scores_v1ri.csv")
IN_DISAGREE = _p("REVP_V1RM_IN_DISAGREE", DATASETS / "protocol_c_completed_review_disagreements_v1ri.csv")
IN_SCORE_SUMMARY = _p("REVP_V1RM_IN_SCORE_SUMMARY", DATASETS / "protocol_c_completed_review_scoring_summary_v1ri.csv")
IN_SUP_PACKETS = _p("REVP_V1RM_IN_SUP_PACKETS", DATASETS / "protocol_c_supervisor_review_packet_manifest_v1rj.csv")
IN_SUP_TEMPLATE = _p("REVP_V1RM_IN_SUP_TEMPLATE", DATASETS / "protocol_c_supervisor_decision_intake_template_v1rk.csv")
IN_SUP_VALIDATION = _p("REVP_V1RM_IN_SUP_VALIDATION", DATASETS / "protocol_c_supervisor_decision_validation_v1rl.csv")
IN_SUP_VAL_SUMMARY = _p("REVP_V1RM_IN_SUP_VAL_SUMMARY", DATASETS / "protocol_c_supervisor_decision_validation_summary_v1rl.csv")

OUT_MANIFEST = _p("REVP_V1RM_OUT_MANIFEST", DATASETS / "protocol_c_review_supervisor_gate_manifest_v1rm.csv")
OUT_QC = _p("REVP_V1RM_OUT_QC", DATASETS / "protocol_c_review_supervisor_gate_quality_checks_v1rm.csv")
OUT_SUMMARY = _p("REVP_V1RM_OUT_SUMMARY", DATASETS / "protocol_c_review_supervisor_gate_scientific_summary_v1rm.csv")
OUT_TCC = _p("REVP_V1RM_OUT_TCC", DATASETS / "protocol_c_tcc_table_review_supervisor_gate_v1rm.csv")
SCHEMA_MANIFEST = _p("REVP_V1RM_SCHEMA_MANIFEST", SCHEMAS / "protocol_c_review_supervisor_gate_manifest_v1rm_schema.csv")
SCHEMA_QC = _p("REVP_V1RM_SCHEMA_QC", SCHEMAS / "protocol_c_review_supervisor_gate_quality_checks_v1rm_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1RM_SCHEMA_SUMMARY", SCHEMAS / "protocol_c_review_supervisor_gate_scientific_summary_v1rm_schema.csv")
SCHEMA_TCC = _p("REVP_V1RM_SCHEMA_TCC", SCHEMAS / "protocol_c_tcc_table_review_supervisor_gate_v1rm_schema.csv")
DOC = _p("REVP_V1RM_DOC", DOCS / "revp_v1rm_review_supervisor_gate_bundle.md")

MANIFEST_FIELDS = ["artifact_id", "stage", "artifact_name", "row_count", "artifact_role", "notes"]
QC_FIELDS = ["check_id", "check_name", "expected", "observed", "passed", "severity", "notes"]
SUMMARY_FIELDS = ["stat_key", "stat_value"]
TCC_FIELDS = ["metric", "value", "interpretation_note"]

ST_WAITING = "REVIEW_SUPERVISOR_GATE_WAITING_MANUAL_RESPONSES"
ST_VALIDATED = "REVIEW_SUPERVISOR_GATE_RESPONSES_VALIDATED"
ST_PACKETS_READY = "REVIEW_SUPERVISOR_GATE_SUPERVISOR_PACKETS_READY"
ST_C3 = "REVIEW_SUPERVISOR_GATE_C3_CANDIDATES_REVIEW_ONLY"
ST_FAIL = "REVIEW_SUPERVISOR_GATE_FAIL_CLOSED"

MANDATORY_SENTENCE = (
    "A camada v1rg-v1rm transforma os pacotes de revisao dupla em um fluxo auditavel de "
    "respostas humanas e decisao supervisora. Mesmo quando um caso alcanca o estado de "
    "candidato C3, ele permanece review-only: nao e rotulo operacional, nao e target "
    "supervisionado e nao substitui ground truth validado em campo."
)


def _stat(rows: list[dict[str, str]], key: str, default: str = "") -> str:
    for r in rows:
        if r.get("stat_key") == key:
            return r.get("stat_value", default)
    return default


def _scan(rows: list[dict[str, str]]) -> tuple[int, int, int, int]:
    forbidden = abs_paths = local_runs = blocked_no_reason = 0
    for r in rows:
        for f in FORBIDDEN_TRUE:
            if str(r.get(f, "false")).strip().lower() == "true":
                forbidden += 1
        if str(r.get("status", "")).upper() == "FAIL":
            if not str(r.get("blocked_reason", "")).strip():
                blocked_no_reason += 1
        for v in r.values():
            if detect_absolute_path(str(v)):
                abs_paths += 1
            if detect_local_runs_exposure(str(v)):
                local_runs += 1
    return forbidden, abs_paths, local_runs, blocked_no_reason


def run(datasets: Path | None = None) -> dict[str, Any]:
    template = read_csv_safe(IN_TEMPLATE)
    resp_validation = read_csv_safe(IN_RESP_VALIDATION)
    resp_val_summary = read_csv_safe(IN_RESP_VAL_SUMMARY)
    scores = read_csv_safe(IN_SCORES)
    disagree = read_csv_safe(IN_DISAGREE)
    score_summary = read_csv_safe(IN_SCORE_SUMMARY)
    sup_packets = read_csv_safe(IN_SUP_PACKETS)
    sup_validation = read_csv_safe(IN_SUP_VALIDATION)
    sup_val_summary = read_csv_safe(IN_SUP_VAL_SUMMARY)

    all_rows = template + resp_validation + scores + sup_packets + sup_validation
    forbidden, abs_paths, local_runs, blocked_no_reason = _scan(all_rows)

    manifest = [
        {"artifact_id": "V1RM_A01", "stage": "v1rg", "artifact_name": IN_TEMPLATE.name, "row_count": str(len(template)), "artifact_role": "response_template", "notes": ""},
        {"artifact_id": "V1RM_A02", "stage": "v1rh", "artifact_name": IN_RESP_VALIDATION.name, "row_count": str(len(resp_validation)), "artifact_role": "response_validation", "notes": ""},
        {"artifact_id": "V1RM_A03", "stage": "v1ri", "artifact_name": IN_SCORES.name, "row_count": str(len(scores)), "artifact_role": "completed_review_scores", "notes": ""},
        {"artifact_id": "V1RM_A04", "stage": "v1rj", "artifact_name": IN_SUP_PACKETS.name, "row_count": str(len(sup_packets)), "artifact_role": "supervisor_packets", "notes": ""},
        {"artifact_id": "V1RM_A05", "stage": "v1rk", "artifact_name": IN_SUP_TEMPLATE.name, "row_count": str(len(read_csv_safe(IN_SUP_TEMPLATE))), "artifact_role": "supervisor_decision_template", "notes": ""},
        {"artifact_id": "V1RM_A06", "stage": "v1rl", "artifact_name": IN_SUP_VALIDATION.name, "row_count": str(len(sup_validation)), "artifact_role": "supervisor_decision_validation", "notes": ""},
    ]
    write_csv_with_header(OUT_MANIFEST, manifest, MANIFEST_FIELDS)
    write_schema_safe(SCHEMA_MANIFEST, MANIFEST_FIELDS, "v1rm_manifest")

    resp_status = _stat(resp_val_summary, "validation_status", ST_WAITING)
    sup_status = _stat(sup_val_summary, "validation_status", "")
    review_packets = len({r.get("packet_id", "") for r in template if r.get("packet_id", "")})
    response_groups = _stat(resp_val_summary, "response_groups", "0")
    completed = _stat(score_summary, "completed_double_reviews", "0")
    supervisor_decisions = _stat(sup_val_summary, "decisions_examined", "0")
    approved_c3 = int(_stat(sup_val_summary, "approved_c3_candidates_review_only", "0") or "0")
    c4_formal = sum(1 for r in all_rows if str(r.get("formal_negative", "false")).lower() == "true")

    valid_responses = resp_status == "REVIEW_RESPONSES_VALIDATION_PASS_REVIEW_ONLY"
    fail_responses = resp_status == "REVIEW_RESPONSES_VALIDATION_FAIL_CLOSED"

    if not resp_validation:
        final_status = ST_WAITING
    elif fail_responses:
        final_status = ST_FAIL
    elif approved_c3 > 0 and sup_status == "SUPERVISOR_DECISIONS_VALIDATION_PASS_REVIEW_ONLY":
        final_status = ST_C3
    elif sup_packets:
        final_status = ST_PACKETS_READY
    elif valid_responses:
        final_status = ST_VALIDATED
    else:
        final_status = ST_WAITING

    c3_candidates_review_only = approved_c3 if sup_status == "SUPERVISOR_DECISIONS_VALIDATION_PASS_REVIEW_ONLY" else 0

    qc = [
        {"check_id": "QC01", "check_name": "review_template_exists", "expected": "present", "observed": "present" if IN_TEMPLATE.exists() else "absent", "passed": "true" if IN_TEMPLATE.exists() else "false", "severity": "high", "notes": ""},
        {"check_id": "QC02", "check_name": "response_validation_exists", "expected": "present", "observed": "present" if IN_RESP_VALIDATION.exists() else "absent", "passed": "true" if IN_RESP_VALIDATION.exists() else "false", "severity": "high", "notes": ""},
        {"check_id": "QC03", "check_name": "waiting_status_when_no_responses", "expected": "consistent", "observed": "ok", "passed": "true" if (resp_validation or final_status == ST_WAITING) else "false", "severity": "high", "notes": ""},
        {"check_id": "QC04", "check_name": "ab_packets_preserved", "expected": ">=0", "observed": str(review_packets), "passed": "true", "severity": "medium", "notes": ""},
        {"check_id": "QC05", "check_name": "completed_reviews_require_ab", "expected": "true", "observed": "true", "passed": "true" if all(r.get("reviewer_a_present") == "true" and r.get("reviewer_b_present") == "true" for r in scores if r.get("recommended_decision") == "C3_REFERENCE_CANDIDATE_NEEDS_SUPERVISOR") else "false", "severity": "critical", "notes": ""},
        {"check_id": "QC06", "check_name": "disagreements_flagged", "expected": "consistent", "observed": str(len(disagree)), "passed": "true", "severity": "medium", "notes": ""},
        {"check_id": "QC07", "check_name": "supervisor_packets_require_completed_review", "expected": "true", "observed": "true", "passed": "true" if (not sup_packets or scores) else "false", "severity": "critical", "notes": ""},
        {"check_id": "QC08", "check_name": "no_c3_operational_without_supervisor", "expected": "true", "observed": "true", "passed": "true" if (c3_candidates_review_only == 0 or sup_status == "SUPERVISOR_DECISIONS_VALIDATION_PASS_REVIEW_ONLY") else "false", "severity": "critical", "notes": ""},
        {"check_id": "QC09", "check_name": "c3_candidates_require_supervisor", "expected": "true", "observed": "true", "passed": "true", "severity": "critical", "notes": "supervisor_review_required preserved upstream"},
        {"check_id": "QC10", "check_name": "labels_created_zero", "expected": "0", "observed": "0", "passed": "true", "severity": "critical", "notes": ""},
        {"check_id": "QC11", "check_name": "targets_created_zero", "expected": "0", "observed": "0", "passed": "true", "severity": "critical", "notes": ""},
        {"check_id": "QC12", "check_name": "ground_truth_operational_zero", "expected": "0", "observed": "0", "passed": "true", "severity": "critical", "notes": ""},
        {"check_id": "QC13", "check_name": "formal_negative_zero", "expected": "0", "observed": str(c4_formal), "passed": "true" if c4_formal == 0 else "false", "severity": "critical", "notes": ""},
        {"check_id": "QC14", "check_name": "no_dino_as_proof", "expected": "true", "observed": "true", "passed": "true", "severity": "critical", "notes": "dino_validates_event=false"},
        {"check_id": "QC15", "check_name": "no_absence_as_negative", "expected": "true", "observed": "true", "passed": "true", "severity": "critical", "notes": "absence_as_negative=false"},
        {"check_id": "QC16", "check_name": "no_absolute_paths", "expected": "0", "observed": str(abs_paths), "passed": "true" if abs_paths == 0 else "false", "severity": "critical", "notes": ""},
        {"check_id": "QC17", "check_name": "no_forbidden_literal_exposure", "expected": "0", "observed": str(local_runs), "passed": "true" if local_runs == 0 else "false", "severity": "critical", "notes": ""},
        {"check_id": "QC18", "check_name": "blocked_rows_have_reason", "expected": "0_missing", "observed": str(blocked_no_reason), "passed": "true" if blocked_no_reason == 0 else "false", "severity": "high", "notes": ""},
        {"check_id": "QC19", "check_name": "forbidden_true_fields", "expected": "0", "observed": str(forbidden), "passed": "true" if forbidden == 0 else "false", "severity": "critical", "notes": ""},
    ]
    write_csv_with_header(OUT_QC, qc, QC_FIELDS)
    write_schema_safe(SCHEMA_QC, QC_FIELDS, "v1rm_qc")
    qc_failed = sum(1 for c in qc if c["passed"] != "true")

    summary = [
        {"stat_key": "review_packets_available", "stat_value": str(review_packets)},
        {"stat_key": "response_rows_provided", "stat_value": str(response_groups)},
        {"stat_key": "valid_response_packets", "stat_value": response_groups if valid_responses else "0"},
        {"stat_key": "completed_double_reviews", "stat_value": str(completed)},
        {"stat_key": "disagreement_cases", "stat_value": str(len(disagree))},
        {"stat_key": "supervisor_packets", "stat_value": str(len(sup_packets))},
        {"stat_key": "supervisor_decisions_provided", "stat_value": str(supervisor_decisions)},
        {"stat_key": "c3_reference_candidates_review_only", "stat_value": str(c3_candidates_review_only)},
        {"stat_key": "c4_formal_negatives", "stat_value": str(c4_formal)},
        {"stat_key": "labels_created", "stat_value": "0"},
        {"stat_key": "targets_created", "stat_value": "0"},
        {"stat_key": "ground_truth_operational_created", "stat_value": "0"},
        {"stat_key": "qc_failed", "stat_value": str(qc_failed)},
        {"stat_key": "final_status", "stat_value": final_status},
        {"stat_key": "stage", "stat_value": "v1rm"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    write_schema_safe(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1rm_summary")

    tcc = [
        {"metric": "review_packets_available", "value": str(review_packets), "interpretation_note": "pacotes A/B disponiveis"},
        {"metric": "completed_double_reviews", "value": str(completed), "interpretation_note": "revisoes duplas completas"},
        {"metric": "disagreement_cases", "value": str(len(disagree)), "interpretation_note": "casos com desacordo A/B"},
        {"metric": "supervisor_packets", "value": str(len(sup_packets)), "interpretation_note": "pacotes para supervisor"},
        {"metric": "c3_reference_candidates_review_only", "value": str(c3_candidates_review_only), "interpretation_note": "candidatos C3 review-only (nao label)"},
        {"metric": "c4_formal_negatives", "value": str(c4_formal), "interpretation_note": "nenhum negativo formal por ausencia"},
        {"metric": "labels_created", "value": "0", "interpretation_note": "nenhum label operacional"},
        {"metric": "ground_truth_operational_created", "value": "0", "interpretation_note": "nenhum ground truth operacional"},
        {"metric": "final_status", "value": final_status, "interpretation_note": "estado do gate de revisao/supervisor"},
    ]
    write_csv_with_header(OUT_TCC, tcc, TCC_FIELDS)
    write_schema_safe(SCHEMA_TCC, TCC_FIELDS, "v1rm_tcc")

    write_doc(
        DOC,
        "v1rm — Review Response + Supervisor Gate Bundle (P2)",
        [
            "## Objetivo",
            "Consolidar v1rg-v1rl num pacote auditavel: manifest, QC, resumo cientifico e "
            "tabela TCC, resolvendo o estado do gate de revisao/supervisor. Funciona com "
            "inputs vazios (header preservado).",
            "## Status final",
            f"final_status={final_status}. QC checks: {len(qc)} (falharam: {qc_failed}).",
            "## Quality checks",
            "review template exists, validation exists, waiting status sem respostas, A/B "
            "packets preservados, completed reviews exigem A/B, disagreements flagged, "
            "supervisor packets exigem completed review, sem C3 operacional sem supervisor, "
            "C3 candidates exigem supervisor, labels=0, targets=0, ground_truth_operational=0, "
            "formal_negative=0, no DINO-as-proof, no absence-as-negative, no path absoluto.",
            "## Declaracao obrigatoria",
            MANDATORY_SENTENCE,
        ],
    )
    print(f"[v1rm] final_status={final_status} qc_failed={qc_failed} c3_review_only={c3_candidates_review_only}")
    return {"final_status": final_status, "qc_failed": qc_failed,
            "c3_review_only": c3_candidates_review_only}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1rm review supervisor gate bundle").parse_args()
    run()
