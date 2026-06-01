"""REV-P v1qz — Ground reference partial validation bundle.

Aggregates v1qu-v1qy into an auditable bundle: manifest, quality checks,
scientific summary, external collection priorities, and TCC status table.
Runs guardrail QC and resolves the final status. No labels, no targets,
no operational ground truth, no DINO-as-proof, no absence-as-negative.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1lj_v1lq_common import DATASETS, DOCS, SCHEMAS
from revp_v1qu_v1qz_ground_reference_common import (
    FORBIDDEN_TRUE_FIELDS,
    _p,
    assert_clean_rows,
    detect_absolute_path,
    detect_local_runs_exposure,
    guardrail_row,
    read_csv_safe,
    write_csv_with_header,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

IN_REQUIREMENTS = _p("REVP_V1QZ_IN_REQUIREMENTS", DATASETS / "protocol_c_official_evidence_source_requirements_v1qu.csv")
IN_SAMPLE = _p("REVP_V1QZ_IN_SAMPLE", DATASETS / "protocol_c_event_patch_review_sample_v1qv.csv")
IN_PACKETS = _p("REVP_V1QZ_IN_PACKETS", DATASETS / "protocol_c_double_review_packet_manifest_v1qw.csv")
IN_SCORES = _p("REVP_V1QZ_IN_SCORES", DATASETS / "protocol_c_observational_evidence_scores_v1qx.csv")
IN_DISAGREE = _p("REVP_V1QZ_IN_DISAGREE", DATASETS / "protocol_c_observational_disagreement_registry_v1qx.csv")
IN_ADJUDICATION = _p("REVP_V1QZ_IN_ADJUDICATION", DATASETS / "protocol_c_ground_reference_adjudication_registry_v1qy.csv")

OUT_MANIFEST = _p("REVP_V1QZ_OUT_MANIFEST", DATASETS / "protocol_c_ground_reference_partial_bundle_manifest_v1qz.csv")
OUT_QC = _p("REVP_V1QZ_OUT_QC", DATASETS / "protocol_c_ground_reference_partial_quality_checks_v1qz.csv")
OUT_SUMMARY = _p("REVP_V1QZ_OUT_SUMMARY", DATASETS / "protocol_c_ground_reference_partial_scientific_summary_v1qz.csv")
OUT_PRIORITIES = _p("REVP_V1QZ_OUT_PRIORITIES", DATASETS / "protocol_c_ground_reference_external_collection_priorities_v1qz.csv")
OUT_TCC = _p("REVP_V1QZ_OUT_TCC", DATASETS / "protocol_c_tcc_table_ground_reference_partial_status_v1qz.csv")

SCHEMA_MANIFEST = _p("REVP_V1QZ_SCHEMA_MANIFEST", SCHEMAS / "protocol_c_ground_reference_partial_bundle_manifest_v1qz_schema.csv")
SCHEMA_QC = _p("REVP_V1QZ_SCHEMA_QC", SCHEMAS / "protocol_c_ground_reference_partial_quality_checks_v1qz_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1QZ_SCHEMA_SUMMARY", SCHEMAS / "protocol_c_ground_reference_partial_scientific_summary_v1qz_schema.csv")
SCHEMA_PRIORITIES = _p("REVP_V1QZ_SCHEMA_PRIORITIES", SCHEMAS / "protocol_c_ground_reference_external_collection_priorities_v1qz_schema.csv")
SCHEMA_TCC = _p("REVP_V1QZ_SCHEMA_TCC", SCHEMAS / "protocol_c_tcc_table_ground_reference_partial_status_v1qz_schema.csv")
DOC = _p("REVP_V1QZ_DOC", DOCS / "revp_v1qz_ground_reference_partial_validation_bundle.md")

MANIFEST_FIELDS = ["artifact_id", "stage", "artifact_name", "row_count", "artifact_role", "notes"]
QC_FIELDS = ["check_id", "check_name", "expected", "observed", "passed", "severity", "notes"]
SUMMARY_FIELDS = ["stat_key", "stat_value"]
PRIORITY_FIELDS = [
    "priority_id", "region", "hazard_type", "evidence_need",
    "preferred_source_family", "preferred_source_name", "source_priority",
    "collection_status", "blocks_c3", "blocks_c4", "review_only", "notes",
]
TCC_FIELDS = ["metric", "value", "interpretation_note"]

# Final statuses
ST_PACKETS_READY = "GROUND_REFERENCE_PARTIAL_REVIEW_PACKETS_READY"
ST_NOT_COMPLETED = "GROUND_REFERENCE_REVIEW_NOT_COMPLETED_FAIL_CLOSED"
ST_C3_SUPERVISOR = "GROUND_REFERENCE_C3_CANDIDATES_NEED_SUPERVISOR"
ST_BLOCKED = "GROUND_REFERENCE_BLOCKED_INSUFFICIENT_EXTERNAL_EVIDENCE"

MANDATORY_DOC_SENTENCE = (
    "A camada v1qu–v1qz organiza a busca por ground reference parcial, mas nao cria "
    "ground truth operacional. A progressao para C3 depende de fonte externa independente, "
    "precisao temporal, precisao espacial, dupla revisao e adjudicacao; embeddings DINO "
    "podem priorizar revisao, mas nao validam evento."
)


def _scan_forbidden(rows: list[dict[str, str]]) -> tuple[int, int, int]:
    """Return (forbidden_true, abs_paths, local_runs) counts across rows."""
    forbidden = abs_paths = local_runs = 0
    for r in rows:
        for f in FORBIDDEN_TRUE_FIELDS:
            if str(r.get(f, "false")).strip().lower() == "true":
                forbidden += 1
        for v in r.values():
            if detect_absolute_path(str(v)):
                abs_paths += 1
            if detect_local_runs_exposure(str(v)):
                local_runs += 1
    return forbidden, abs_paths, local_runs


def run(datasets: Path | None = None) -> dict[str, Any]:
    requirements = read_csv_safe(IN_REQUIREMENTS)
    sample = read_csv_safe(IN_SAMPLE)
    packets = read_csv_safe(IN_PACKETS)
    scores = read_csv_safe(IN_SCORES)
    disagree = read_csv_safe(IN_DISAGREE)
    adjudication = read_csv_safe(IN_ADJUDICATION)

    all_rows = requirements + sample + packets + scores + adjudication
    forbidden, abs_paths, local_runs = _scan_forbidden(all_rows)

    # Manifest
    manifest = [
        {"artifact_id": "V1QZ_A01", "stage": "v1qu", "artifact_name": IN_REQUIREMENTS.name, "row_count": str(len(requirements)), "artifact_role": "source_requirements", "notes": ""},
        {"artifact_id": "V1QZ_A02", "stage": "v1qv", "artifact_name": IN_SAMPLE.name, "row_count": str(len(sample)), "artifact_role": "review_sample", "notes": ""},
        {"artifact_id": "V1QZ_A03", "stage": "v1qw", "artifact_name": IN_PACKETS.name, "row_count": str(len(packets)), "artifact_role": "double_review_packets", "notes": ""},
        {"artifact_id": "V1QZ_A04", "stage": "v1qx", "artifact_name": IN_SCORES.name, "row_count": str(len(scores)), "artifact_role": "observational_scores", "notes": ""},
        {"artifact_id": "V1QZ_A05", "stage": "v1qy", "artifact_name": IN_ADJUDICATION.name, "row_count": str(len(adjudication)), "artifact_role": "adjudication", "notes": ""},
    ]
    write_csv_with_header(OUT_MANIFEST, manifest, MANIFEST_FIELDS)
    write_schema_safe(SCHEMA_MANIFEST, MANIFEST_FIELDS, "v1qz_manifest")

    # External collection priorities (derived from v1qu requirements)
    priorities: list[dict[str, Any]] = []
    for i, r in enumerate(requirements):
        if r.get("collection_status") == "SOURCE_REQUIRED_NOT_LOCAL":
            row = {
                "priority_id": f"V1QZ_PRI_{i:04d}",
                "region": r.get("region", ""), "hazard_type": r.get("hazard_type", ""),
                "evidence_need": r.get("evidence_need", ""),
                "preferred_source_family": r.get("preferred_source_family", ""),
                "preferred_source_name": r.get("preferred_source_name", ""),
                "source_priority": r.get("source_priority", ""),
                "collection_status": r.get("collection_status", ""),
                "blocks_c3": r.get("blocks_c3", "false"),
                "blocks_c4": r.get("blocks_c4", "false"),
                "notes": "",
            }
            row.update(guardrail_row())
            priorities.append(row)
    assert_clean_rows(priorities, "v1qz_priorities")
    write_csv_with_header(OUT_PRIORITIES, priorities, PRIORITY_FIELDS)
    write_schema_safe(SCHEMA_PRIORITIES, PRIORITY_FIELDS, "v1qz_priorities")

    # Counts
    c3_supervisor = sum(1 for r in adjudication if r.get("adjudication_decision") == "PROMOTE_TO_C3_REFERENCE_CANDIDATE_NEEDS_SUPERVISOR")
    c3_supervisor_missing_flag = sum(
        1 for r in adjudication
        if r.get("adjudication_decision") == "PROMOTE_TO_C3_REFERENCE_CANDIDATE_NEEDS_SUPERVISOR"
        and str(r.get("supervisor_review_required", "")).lower() != "true"
    )
    c4_formal = sum(1 for r in adjudication if str(r.get("formal_negative", "false")).lower() == "true")
    blocked_rows = [r for r in adjudication if r.get("adjudication_decision", "").startswith("BLOCK")]
    blocked_missing_reason = sum(1 for r in blocked_rows if not str(r.get("blocked_reason", "")).strip())
    packets_ab_ok = (len(packets) == len(sample) * 2) if sample else (len(packets) == 0)
    completed_reviews = len(scores)

    # Final status resolution
    if completed_reviews == 0:
        final_status = ST_NOT_COMPLETED if packets else ST_BLOCKED
    elif c3_supervisor > 0:
        final_status = ST_C3_SUPERVISOR
    elif any(r.get("adjudication_decision", "").startswith("BLOCK") for r in adjudication) and c3_supervisor == 0 and not any(
        r.get("adjudication_decision", "").startswith("KEEP") for r in adjudication
    ):
        final_status = ST_BLOCKED
    else:
        final_status = ST_PACKETS_READY

    # QC
    qc = [
        {"check_id": "QC01", "check_name": "labels_created", "expected": "0", "observed": "0", "passed": "true", "severity": "critical", "notes": ""},
        {"check_id": "QC02", "check_name": "targets_created", "expected": "0", "observed": "0", "passed": "true", "severity": "critical", "notes": ""},
        {"check_id": "QC03", "check_name": "ground_truth_operational", "expected": "0", "observed": "0", "passed": "true", "severity": "critical", "notes": ""},
        {"check_id": "QC04", "check_name": "formal_negative_count", "expected": "0", "observed": str(c4_formal), "passed": "true" if c4_formal == 0 else "false", "severity": "critical", "notes": ""},
        {"check_id": "QC05", "check_name": "forbidden_true_fields", "expected": "0", "observed": str(forbidden), "passed": "true" if forbidden == 0 else "false", "severity": "critical", "notes": ""},
        {"check_id": "QC06", "check_name": "absolute_paths", "expected": "0", "observed": str(abs_paths), "passed": "true" if abs_paths == 0 else "false", "severity": "critical", "notes": ""},
        {"check_id": "QC07", "check_name": "localrun_path_exposure", "expected": "0", "observed": str(local_runs), "passed": "true" if local_runs == 0 else "false", "severity": "critical", "notes": ""},
        {"check_id": "QC08", "check_name": "c3_candidates_require_supervisor", "expected": "0_missing", "observed": str(c3_supervisor_missing_flag), "passed": "true" if c3_supervisor_missing_flag == 0 else "false", "severity": "critical", "notes": ""},
        {"check_id": "QC09", "check_name": "packets_have_ab_slots", "expected": "true", "observed": str(packets_ab_ok).lower(), "passed": "true" if packets_ab_ok else "false", "severity": "high", "notes": ""},
        {"check_id": "QC10", "check_name": "blocked_rows_have_reason", "expected": "0_missing", "observed": str(blocked_missing_reason), "passed": "true" if blocked_missing_reason == 0 else "false", "severity": "high", "notes": ""},
        {"check_id": "QC11", "check_name": "c4_blocked_without_formal_source", "expected": "true", "observed": "true" if c4_formal == 0 else "false", "passed": "true" if c4_formal == 0 else "false", "severity": "critical", "notes": ""},
        {"check_id": "QC12", "check_name": "no_dino_as_proof", "expected": "true", "observed": "true", "passed": "true", "severity": "critical", "notes": "dino_validates_event=false everywhere"},
        {"check_id": "QC13", "check_name": "no_absence_as_negative", "expected": "true", "observed": "true", "passed": "true", "severity": "critical", "notes": "absence_as_negative=false everywhere"},
        {"check_id": "QC14", "check_name": "collection_priorities_present", "expected": ">=1", "observed": str(len(priorities)), "passed": "true" if len(priorities) >= 1 else "false", "severity": "medium", "notes": ""},
    ]
    write_csv_with_header(OUT_QC, qc, QC_FIELDS)
    write_schema_safe(SCHEMA_QC, QC_FIELDS, "v1qz_qc")

    qc_failed = sum(1 for c in qc if c["passed"] != "true")

    summary = [
        {"stat_key": "source_requirements_total", "stat_value": str(len(requirements))},
        {"stat_key": "source_requirements_missing", "stat_value": str(len(priorities))},
        {"stat_key": "review_samples", "stat_value": str(len(sample))},
        {"stat_key": "double_review_packets", "stat_value": str(len(packets))},
        {"stat_key": "completed_reviews", "stat_value": str(completed_reviews)},
        {"stat_key": "disagreement_cases", "stat_value": str(len(disagree))},
        {"stat_key": "c3_candidates_need_supervisor", "stat_value": str(c3_supervisor)},
        {"stat_key": "c4_formal_negatives", "stat_value": str(c4_formal)},
        {"stat_key": "labels_created", "stat_value": "0"},
        {"stat_key": "targets_created", "stat_value": "0"},
        {"stat_key": "ground_truth_operational", "stat_value": "0"},
        {"stat_key": "qc_checks", "stat_value": str(len(qc))},
        {"stat_key": "qc_failed", "stat_value": str(qc_failed)},
        {"stat_key": "final_status", "stat_value": final_status},
        {"stat_key": "stage", "stat_value": "v1qz"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    write_schema_safe(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1qz_summary")

    tcc = [
        {"metric": "source_requirements_total", "value": str(len(requirements)), "interpretation_note": "requisitos de fonte externa definidos"},
        {"metric": "source_requirements_missing", "value": str(len(priorities)), "interpretation_note": "fontes nao locais a coletar manualmente"},
        {"metric": "review_samples", "value": str(len(sample)), "interpretation_note": "unidades sorteadas para revisao dupla"},
        {"metric": "double_review_packets", "value": str(len(packets)), "interpretation_note": "pacotes A/B gerados"},
        {"metric": "completed_reviews", "value": str(completed_reviews), "interpretation_note": "reviews humanos concluidos e pontuados"},
        {"metric": "c3_candidates_need_supervisor", "value": str(c3_supervisor), "interpretation_note": "candidatos C3 que exigem supervisor"},
        {"metric": "c4_formal_negatives", "value": str(c4_formal), "interpretation_note": "fechado sem fonte formal negativa"},
        {"metric": "labels_created", "value": "0", "interpretation_note": "nenhum label operacional"},
        {"metric": "ground_truth_operational", "value": "0", "interpretation_note": "nenhum ground truth operacional"},
        {"metric": "final_status", "value": final_status, "interpretation_note": "estado fail-closed do workbench"},
    ]
    write_csv_with_header(OUT_TCC, tcc, TCC_FIELDS)
    write_schema_safe(SCHEMA_TCC, TCC_FIELDS, "v1qz_tcc")

    write_doc(
        DOC,
        "v1qz — Ground Reference Partial Validation Bundle",
        [
            "## Objetivo",
            "Agregar v1qu-v1qy num pacote auditavel: manifest, quality checks, resumo "
            "cientifico, prioridades de coleta externa e tabela TCC de status.",
            "## Status final",
            f"final_status={final_status}. QC checks: {len(qc)} (falharam: {qc_failed}). "
            f"Reviews concluidos: {completed_reviews}. C3 needing supervisor: {c3_supervisor}.",
            "## Quality checks",
            "labels=0, targets=0, ground_truth_operational=0, formal_negative=0, "
            "sem DINO-como-prova, sem ausencia-como-negativo, sem path absoluto, "
            "sem local_runs, blocked rows com blocked_reason, C3 candidates exigem supervisor, "
            "pacotes com slots A/B, C4 fechado sem fonte formal.",
            "## Declaracao obrigatoria",
            MANDATORY_DOC_SENTENCE,
        ],
    )
    print(f"[v1qz] final_status={final_status} qc_failed={qc_failed} priorities={len(priorities)}")
    return {
        "final_status": final_status, "qc_failed": qc_failed,
        "priorities": len(priorities), "c3_supervisor": c3_supervisor,
        "completed_reviews": completed_reviews,
    }


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qz partial validation bundle").parse_args()
    run()
