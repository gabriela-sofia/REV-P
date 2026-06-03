"""REV-P v1rp — TCC results tables for Protocol C.

Builds TCC-ready tables from existing P0/P1/P2 summaries: C-level status,
missing external sources, review workflow, and the DINO review-only role.
Read-only consolidation; no labels, targets or operational ground truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1qu_v1qz_ground_reference_common import (
    DATASETS,
    DOCS,
    SCHEMAS,
    _p,
    assert_clean_rows,
    read_csv_safe,
    write_csv_with_header,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

IN_ADJ_SUMMARY = _p("REVP_V1RP_IN_ADJ_SUMMARY", DATASETS / "protocol_c_ground_reference_adjudication_summary_v1qy.csv")
IN_PARTIAL_SUMMARY = _p("REVP_V1RP_IN_PARTIAL_SUMMARY", DATASETS / "protocol_c_ground_reference_partial_scientific_summary_v1qz.csv")
IN_REQUIREMENTS = _p("REVP_V1RP_IN_REQUIREMENTS", DATASETS / "protocol_c_official_evidence_source_requirements_v1qu.csv")
IN_GAP = _p("REVP_V1RP_IN_GAP", DATASETS / "protocol_c_official_evidence_source_gap_summary_v1qu.csv")
IN_SAMPLING_SUMMARY = _p("REVP_V1RP_IN_SAMPLING_SUMMARY", DATASETS / "protocol_c_event_patch_review_sampling_summary_v1qv.csv")
IN_GATE_SUMMARY = _p("REVP_V1RP_IN_GATE_SUMMARY", DATASETS / "protocol_c_review_supervisor_gate_scientific_summary_v1rm.csv")
IN_DINO_QUEUE = _p("REVP_V1RP_IN_DINO_QUEUE", DATASETS / "recife_dino_review_only_representation_queue_v1oz.csv")

OUT_C_LEVEL = _p("REVP_V1RP_OUT_C_LEVEL", DATASETS / "protocol_c_tcc_table_c_level_status_v1rp.csv")
OUT_SOURCES = _p("REVP_V1RP_OUT_SOURCES", DATASETS / "protocol_c_tcc_table_external_sources_v1rp.csv")
OUT_WORKFLOW = _p("REVP_V1RP_OUT_WORKFLOW", DATASETS / "protocol_c_tcc_table_review_workflow_v1rp.csv")
OUT_DINO = _p("REVP_V1RP_OUT_DINO", DATASETS / "protocol_c_tcc_table_dino_role_v1rp.csv")
SCHEMA_C_LEVEL = _p("REVP_V1RP_SCHEMA_C_LEVEL", SCHEMAS / "protocol_c_tcc_table_c_level_status_v1rp_schema.csv")
SCHEMA_SOURCES = _p("REVP_V1RP_SCHEMA_SOURCES", SCHEMAS / "protocol_c_tcc_table_external_sources_v1rp_schema.csv")
SCHEMA_WORKFLOW = _p("REVP_V1RP_SCHEMA_WORKFLOW", SCHEMAS / "protocol_c_tcc_table_review_workflow_v1rp_schema.csv")
SCHEMA_DINO = _p("REVP_V1RP_SCHEMA_DINO", SCHEMAS / "protocol_c_tcc_table_dino_role_v1rp_schema.csv")
DOC = _p("REVP_V1RP_DOC", DOCS / "revp_v1rp_tcc_protocol_c_results_tables.md")

C_LEVEL_FIELDS = ["c_level", "count", "is_operational_label", "interpretation_note"]
SOURCES_FIELDS = ["source_family", "requirements_total", "required_not_local", "interpretation_note"]
WORKFLOW_FIELDS = ["workflow_stage", "metric", "value", "interpretation_note"]
DINO_FIELDS = ["dino_aspect", "value", "interpretation_note"]


def _stat(rows: list[dict[str, str]], key: str, default: str = "0") -> str:
    for r in rows:
        if r.get("stat_key") == key:
            return r.get("stat_value", default)
    return default


def run(datasets: Path | None = None) -> dict[str, Any]:
    adj = read_csv_safe(IN_ADJ_SUMMARY)
    partial = read_csv_safe(IN_PARTIAL_SUMMARY)
    requirements = read_csv_safe(IN_REQUIREMENTS)
    sampling = read_csv_safe(IN_SAMPLING_SUMMARY)
    gate = read_csv_safe(IN_GATE_SUMMARY)
    dino_queue = read_csv_safe(IN_DINO_QUEUE)

    # --- C-level status table ---
    c1 = _stat(adj, "kept_c1_contextual")
    c2 = _stat(adj, "kept_c2_review_only")
    c3 = _stat(adj, "promote_c3_needs_supervisor")
    c4 = _stat(adj, "c4_formal_negatives_opened")
    c_level_rows = [
        {"c_level": "C1_CONTEXTUAL_ONLY", "count": c1, "is_operational_label": "false", "interpretation_note": "evidencia contextual"},
        {"c_level": "C2_REVIEW_ONLY_CANDIDATE", "count": c2, "is_operational_label": "false", "interpretation_note": "candidato review-only"},
        {"c_level": "C3_REFERENCE_CANDIDATE_NEEDS_SUPERVISOR", "count": c3, "is_operational_label": "false", "interpretation_note": "candidato C3 review-only (exige supervisor)"},
        {"c_level": "C4_NEGATIVE_BLOCKED", "count": c4, "is_operational_label": "false", "interpretation_note": "negativo formal fechado"},
    ]
    write_csv_with_header(OUT_C_LEVEL, c_level_rows, C_LEVEL_FIELDS)
    write_schema_safe(SCHEMA_C_LEVEL, C_LEVEL_FIELDS, "v1rp_c_level")

    # --- External sources table ---
    fam_total: dict[str, int] = {}
    fam_missing: dict[str, int] = {}
    for r in requirements:
        fam = r.get("preferred_source_family", "UNKNOWN")
        fam_total[fam] = fam_total.get(fam, 0) + 1
        if r.get("collection_status") == "SOURCE_REQUIRED_NOT_LOCAL":
            fam_missing[fam] = fam_missing.get(fam, 0) + 1
    sources_rows = [
        {"source_family": fam, "requirements_total": str(fam_total[fam]),
         "required_not_local": str(fam_missing.get(fam, 0)),
         "interpretation_note": "fonte externa priorizada"}
        for fam in sorted(fam_total)
    ]
    write_csv_with_header(OUT_SOURCES, sources_rows, SOURCES_FIELDS)
    write_schema_safe(SCHEMA_SOURCES, SOURCES_FIELDS, "v1rp_sources")

    # --- Review workflow table ---
    workflow_rows = [
        {"workflow_stage": "v1qv_sampling", "metric": "review_samples", "value": _stat(sampling, "sample_size"), "interpretation_note": "amostra de revisao"},
        {"workflow_stage": "v1qz_partial", "metric": "completed_reviews", "value": _stat(partial, "completed_reviews"), "interpretation_note": "reviews concluidos (P0)"},
        {"workflow_stage": "v1rm_gate", "metric": "completed_double_reviews", "value": _stat(gate, "completed_double_reviews"), "interpretation_note": "revisoes duplas completas (P2)"},
        {"workflow_stage": "v1rm_gate", "metric": "supervisor_packets", "value": _stat(gate, "supervisor_packets"), "interpretation_note": "pacotes para supervisor"},
        {"workflow_stage": "v1rm_gate", "metric": "c3_reference_candidates_review_only", "value": _stat(gate, "c3_reference_candidates_review_only"), "interpretation_note": "candidatos C3 review-only"},
        {"workflow_stage": "v1rm_gate", "metric": "final_status", "value": _stat(gate, "final_status", "WAITING"), "interpretation_note": "estado do gate"},
    ]
    write_csv_with_header(OUT_WORKFLOW, workflow_rows, WORKFLOW_FIELDS)
    write_schema_safe(SCHEMA_WORKFLOW, WORKFLOW_FIELDS, "v1rp_workflow")

    # --- DINO role table ---
    dino_rows = [
        {"dino_aspect": "dino_review_queue_rows", "value": str(len(dino_queue)), "interpretation_note": "fila de representacao review-only"},
        {"dino_aspect": "dino_validates_event", "value": "false", "interpretation_note": "DINO nunca valida evento"},
        {"dino_aspect": "dino_can_create_label", "value": "false", "interpretation_note": "DINO nunca cria label"},
        {"dino_aspect": "dino_role", "value": "REVIEW_ONLY_PRIORITIZATION", "interpretation_note": "DINO apenas prioriza revisao"},
    ]
    write_csv_with_header(OUT_DINO, dino_rows, DINO_FIELDS)
    write_schema_safe(SCHEMA_DINO, DINO_FIELDS, "v1rp_dino")

    write_doc(
        DOC,
        "v1rp — TCC Results Tables for Protocol C",
        [
            "## Objetivo",
            "Gerar tabelas TCC-ready a partir dos resumos P0/P1/P2: estado C1-C4, fontes "
            "externas faltantes, fluxo de revisao, e papel review-only do DINO.",
            "## Tabelas",
            "c_level_status, external_sources, review_workflow, dino_role.",
            "## Invariante DINO",
            "DINO e review-only: dino_validates_event=false, dino_can_create_label=false. "
            "Serve apenas para priorizar revisao supervisora, nunca como prova de evento.",
            "## Guardrails",
            "Nenhum c_level e label operacional. Nenhuma tabela cria target ou ground truth.",
        ],
    )
    print(f"[v1rp] c_level=4 sources={len(sources_rows)} workflow={len(workflow_rows)} dino={len(dino_rows)}")
    return {"sources": len(sources_rows), "workflow": len(workflow_rows)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1rp tcc results tables").parse_args()
    run()
