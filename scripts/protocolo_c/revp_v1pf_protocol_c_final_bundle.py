"""REV-P v1pf — Protocol C final bundle.

Consolidates v1pb-v1pe outputs, generates manifest, summary, and
commit candidate file list. Does not execute git operations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1pb_v1pf_common import (
    ALL_EXPECTED_OUTPUTS,
    DATASETS,
    DOCS,
    ROOT,
    SCHEMAS,
    V1OG_V1OT_OUTPUTS,
    V1OU_V1PA_OUTPUTS,
    _p,
    count_csv_rows,
    emit_doc,
    load_metric_from_summary,
    read_csv_safe,
    sha256_16,
    write_csv_with_header,
    write_schema,
)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

OUT_MANIFEST = _p("REVP_V1PF_OUT_MANIFEST", DATASETS / "recife_protocol_c_final_bundle_manifest_v1pf.csv")
OUT_SUMMARY = _p("REVP_V1PF_OUT_SUMMARY", DATASETS / "recife_protocol_c_final_bundle_summary_v1pf.csv")
OUT_COMMIT = _p("REVP_V1PF_OUT_COMMIT", DATASETS / "recife_protocol_c_final_bundle_commit_candidate_files_v1pf.csv")
SCHEMA_MANIFEST = _p("REVP_V1PF_SCHEMA_MANIFEST", SCHEMAS / "recife_protocol_c_final_bundle_manifest_v1pf_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1PF_SCHEMA_SUMMARY", SCHEMAS / "recife_protocol_c_final_bundle_summary_v1pf_schema.csv")
SCHEMA_COMMIT = _p("REVP_V1PF_SCHEMA_COMMIT", SCHEMAS / "recife_protocol_c_final_bundle_commit_candidate_files_v1pf_schema.csv")
DOC = _p("REVP_V1PF_DOC", DOCS / "revp_v1pf_protocol_c_final_bundle.md")

MANIFEST_FIELDS = [
    "artifact_path", "artifact_type", "stage", "rows", "columns",
    "sha256_16", "role", "include_in_commit", "notes",
]
SUMMARY_FIELDS = ["metric", "value", "interpretation", "status"]
COMMIT_FIELDS = ["file_path", "file_group", "include_recommendation", "reason"]

# ---------------------------------------------------------------------------
# v1pb-v1pf outputs
# ---------------------------------------------------------------------------

V1PB_V1PF_OUTPUTS = [
    ("recife_protocol_c_end_to_end_orchestration_v1pb.csv", "v1pb", "orchestration_registry"),
    ("recife_protocol_c_end_to_end_orchestration_summary_v1pb.csv", "v1pb", "orchestration_summary"),
    ("recife_protocol_c_global_invariant_audit_v1pc.csv", "v1pc", "invariant_audit"),
    ("recife_protocol_c_global_invariant_summary_v1pc.csv", "v1pc", "invariant_summary"),
    ("recife_protocol_c_tcc_table_temporal_recovery_v1pd.csv", "v1pd", "tcc_table_temporal"),
    ("recife_protocol_c_tcc_table_observed_evidence_v1pd.csv", "v1pd", "tcc_table_observed"),
    ("recife_protocol_c_tcc_table_guardrails_v1pd.csv", "v1pd", "tcc_table_guardrails"),
    ("recife_protocol_c_tcc_table_decision_levels_v1pd.csv", "v1pd", "tcc_table_decisions"),
    ("recife_protocol_c_final_bundle_manifest_v1pf.csv", "v1pf", "final_manifest"),
    ("recife_protocol_c_final_bundle_summary_v1pf.csv", "v1pf", "final_summary"),
    ("recife_protocol_c_final_bundle_commit_candidate_files_v1pf.csv", "v1pf", "commit_candidates"),
]

V1PE_DOC = "revp_v1pe_protocol_c_methodological_report.md"

# Docs from v1ou-v1pf
ALL_DOCS = [
    "revp_v1ou_external_evidence_source_inventory.md",
    "revp_v1ov_ground_reference_observed_event_registry.md",
    "revp_v1ow_evidence_strength_precision_scoring.md",
    "revp_v1ox_event_patch_linkage_registry.md",
    "revp_v1oy_ground_truth_candidate_decision_audit.md",
    "revp_v1oz_dino_review_only_representation_queue.md",
    "revp_v1pa_protocol_c_observed_evidence_bundle.md",
    "revp_v1pb_protocol_c_end_to_end_orchestrator.md",
    "revp_v1pc_protocol_c_global_invariant_auditor.md",
    "revp_v1pd_protocol_c_tcc_table_exporter.md",
    "revp_v1pe_protocol_c_methodological_report.md",
    "revp_v1pf_protocol_c_final_bundle.md",
]


def build_manifest(datasets_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # v1og-v1ot
    for filename in V1OG_V1OT_OUTPUTS:
        path = datasets_dir / filename
        n_rows = count_csv_rows(path)
        rows.append({
            "artifact_path": filename,
            "artifact_type": "CSV",
            "stage": "v1og-v1ot",
            "rows": str(n_rows) if n_rows >= 0 else "MISSING",
            "columns": "",
            "sha256_16": sha256_16(path),
            "role": "temporal_recovery",
            "include_in_commit": "true" if path.exists() else "false",
            "notes": "",
        })

    # v1ou-v1pa
    for filename in V1OU_V1PA_OUTPUTS:
        path = datasets_dir / filename
        n_rows = count_csv_rows(path)
        rows.append({
            "artifact_path": filename,
            "artifact_type": "CSV",
            "stage": "v1ou-v1pa",
            "rows": str(n_rows) if n_rows >= 0 else "MISSING",
            "columns": "",
            "sha256_16": sha256_16(path),
            "role": "observed_evidence",
            "include_in_commit": "true" if path.exists() else "false",
            "notes": "",
        })

    # v1pb-v1pf
    for filename, stage, role in V1PB_V1PF_OUTPUTS:
        path = datasets_dir / filename
        n_rows = count_csv_rows(path)
        rows.append({
            "artifact_path": filename,
            "artifact_type": "CSV",
            "stage": stage,
            "rows": str(n_rows) if n_rows >= 0 else "MISSING",
            "columns": "",
            "sha256_16": sha256_16(path),
            "role": role,
            "include_in_commit": "true" if path.exists() else "false",
            "notes": "",
        })

    return rows


def build_commit_candidates() -> list[dict[str, Any]]:
    """Generate recommended commit groups."""
    rows: list[dict[str, Any]] = []

    # Group A: v1og-v1ot temporal recovery
    for f in V1OG_V1OT_OUTPUTS:
        rows.append({
            "file_path": f"datasets/{f}",
            "file_group": "A_temporal_recovery_v1og_v1ot",
            "include_recommendation": "true",
            "reason": "Temporal recovery pipeline outputs",
        })

    # Group B: v1ou-v1pa observed evidence
    for f in V1OU_V1PA_OUTPUTS:
        rows.append({
            "file_path": f"datasets/{f}",
            "file_group": "B_observed_evidence_v1ou_v1pa",
            "include_recommendation": "true",
            "reason": "Observed evidence layer outputs",
        })

    # Group C: v1pb-v1pf finalization
    for f, stage, role in V1PB_V1PF_OUTPUTS:
        rows.append({
            "file_path": f"datasets/{f}",
            "file_group": "C_finalization_v1pb_v1pf",
            "include_recommendation": "true",
            "reason": f"Finalization: {role}",
        })

    # Group C scripts
    script_names = [
        "revp_v1ou_v1pa_common.py",
        "revp_v1ou_external_evidence_source_inventory.py",
        "revp_v1ov_ground_reference_observed_event_registry.py",
        "revp_v1ow_evidence_strength_precision_scoring.py",
        "revp_v1ox_event_patch_linkage_registry.py",
        "revp_v1oy_ground_truth_candidate_decision_audit.py",
        "revp_v1oz_dino_review_only_representation_queue.py",
        "revp_v1pa_protocol_c_observed_evidence_bundle.py",
        "revp_v1pb_v1pf_common.py",
        "revp_v1pb_protocol_c_end_to_end_orchestrator.py",
        "revp_v1pc_protocol_c_global_invariant_auditor.py",
        "revp_v1pd_protocol_c_tcc_table_exporter.py",
        "revp_v1pe_protocol_c_methodological_report.py",
        "revp_v1pf_protocol_c_final_bundle.py",
    ]
    for sn in script_names:
        group = "B_observed_evidence_v1ou_v1pa" if "v1ou" in sn or "v1ov" in sn or "v1ow" in sn or "v1ox" in sn or "v1oy" in sn or "v1oz" in sn or "v1pa" in sn else "C_finalization_v1pb_v1pf"
        rows.append({
            "file_path": f"scripts/protocolo_c/{sn}",
            "file_group": group,
            "include_recommendation": "true",
            "reason": "Pipeline script",
        })

    # Tests
    test_names = [
        "test_revp_v1ou_v1pa_observed_evidence_protocol_c.py",
        "test_revp_v1pb_v1pf_protocol_c_finalization.py",
    ]
    for tn in test_names:
        group = "B_observed_evidence_v1ou_v1pa" if "v1ou" in tn else "C_finalization_v1pb_v1pf"
        rows.append({
            "file_path": f"tests/{tn}",
            "file_group": group,
            "include_recommendation": "true",
            "reason": "Test file",
        })

    # Docs
    for d in ALL_DOCS:
        group = "B_observed_evidence_v1ou_v1pa" if any(x in d for x in ["v1ou", "v1ov", "v1ow", "v1ox", "v1oy", "v1oz", "v1pa"]) else "C_finalization_v1pb_v1pf"
        rows.append({
            "file_path": f"docs/metodologia_cientifica/{d}",
            "file_group": group,
            "include_recommendation": "true",
            "reason": "Methodology documentation",
        })

    # Schemas
    for f in V1OU_V1PA_OUTPUTS:
        schema_name = f.replace(".csv", "_schema.csv")
        schema_path = SCHEMAS / schema_name
        if schema_path.exists():
            rows.append({
                "file_path": f"datasets/schemas/{schema_name}",
                "file_group": "B_observed_evidence_v1ou_v1pa",
                "include_recommendation": "true",
                "reason": "Schema",
            })

    return rows


def run() -> None:
    datasets_dir = DATASETS

    manifest = build_manifest(datasets_dir)
    commit_candidates = build_commit_candidates()

    write_csv_with_header(OUT_MANIFEST, manifest, MANIFEST_FIELDS)
    write_schema(SCHEMA_MANIFEST, MANIFEST_FIELDS, "v1pf_final_bundle_manifest")

    write_csv_with_header(OUT_COMMIT, commit_candidates, COMMIT_FIELDS)
    write_schema(SCHEMA_COMMIT, COMMIT_FIELDS, "v1pf_commit_candidates")

    # Summary
    total_artifacts = len(manifest)
    present = sum(1 for r in manifest if r["include_in_commit"] == "true")
    missing = total_artifacts - present

    # Load invariant status
    inv_summary = datasets_dir / "recife_protocol_c_global_invariant_summary_v1pc.csv"
    inv_status = load_metric_from_summary(inv_summary, "final_status")

    # Load orchestration status
    orch_summary = datasets_dir / "recife_protocol_c_end_to_end_orchestration_summary_v1pb.csv"
    orch_status = load_metric_from_summary(orch_summary, "final_status")

    if inv_status == "GLOBAL_INVARIANTS_PASS" and missing == 0:
        final_status = "PROTOCOL_C_COMPLETE_READY_FOR_COMMIT"
    elif inv_status == "GLOBAL_INVARIANTS_WARN_ONLY" and missing == 0:
        final_status = "PROTOCOL_C_COMPLETE_WITH_WARNINGS"
    elif missing > 0:
        final_status = "PROTOCOL_C_INCOMPLETE"
    else:
        final_status = "PROTOCOL_C_COMPLETE_WITH_ISSUES"

    summary_rows = [
        {"metric": "total_artifacts", "value": str(total_artifacts),
         "interpretation": "Total de artefatos no bundle", "status": "INFO"},
        {"metric": "artifacts_present", "value": str(present),
         "interpretation": "Artefatos existentes prontos para commit", "status": "INFO"},
        {"metric": "artifacts_missing", "value": str(missing),
         "interpretation": "Artefatos esperados mas ausentes", "status": "WARN" if missing > 0 else "INFO"},
        {"metric": "invariant_audit_status", "value": inv_status,
         "interpretation": "Status da auditoria global de invariantes (v1pc)", "status": inv_status},
        {"metric": "orchestration_status", "value": orch_status,
         "interpretation": "Status da orquestracao end-to-end (v1pb)", "status": orch_status},
        {"metric": "commit_groups_recommended", "value": "3",
         "interpretation": "A: v1og-v1ot, B: v1ou-v1pa, C: v1pb-v1pf", "status": "INFO"},
        {"metric": "final_status", "value": final_status,
         "interpretation": "Status final do bundle Protocol C", "status": final_status},
    ]
    write_csv_with_header(OUT_SUMMARY, summary_rows, SUMMARY_FIELDS)
    write_schema(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1pf_final_bundle_summary")

    emit_doc(DOC, f"""# v1pf - Protocol C Final Bundle

## Objetivo

Consolidar v1pb-v1pe em manifest, summary e lista de arquivos candidatos a commit.
Nao executa git add/commit/push.

## Resultado

- Total artefatos: {total_artifacts}
- Presentes: {present}
- Ausentes: {missing}
- Invariantes: {inv_status}
- Orquestracao: {orch_status}
- Status final: {final_status}

## Commits Recomendados

### Commit A: Recuperacao Temporal (v1og-v1ot)
Outputs da cadeia temporal Sentinel: proveniencia, resolucao de data,
adjudicacao temporal, fixture audit, bundle final.

### Commit B: Camada Observacional (v1ou-v1pa)
Scripts, datasets, schemas, docs e testes da camada de evidencias externas
e decisoes C1/C2/C3/C4.

### Commit C: Finalizacao (v1pb-v1pf)
Orquestrador, auditor de invariantes, tabelas TCC, relatorio metodologico,
bundle final.

## Acoes Manuais Necessarias

1. Revisar `git status --short`
2. Revisar `git diff --stat`
3. Selecionar arquivos por grupo (A, B, C)
4. `git add <files>`
5. `git commit -m "..."`
6. Revisar antes de push
""")

    print(f"[v1pf] Bundle: {present}/{total_artifacts} artifacts present. "
          f"Invariants: {inv_status}. Status: {final_status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v1pf Protocol C final bundle")
    parser.parse_args()
    run()
