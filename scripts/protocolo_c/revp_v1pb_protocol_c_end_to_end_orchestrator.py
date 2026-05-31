"""REV-P v1pb — Protocol C end-to-end orchestrator.

Runs or checks the full Protocol C pipeline (v1og-v1pa) in order.
Modes: --dry-run, --run, --check-only (default).
Does not use internet, download data, read pixels, or alter ground truth.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
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
    sha256_16,
    write_csv_with_header,
    write_schema,
)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

OUT_REGISTRY = _p("REVP_V1PB_OUT_REGISTRY", DATASETS / "recife_protocol_c_end_to_end_orchestration_v1pb.csv")
OUT_SUMMARY = _p("REVP_V1PB_OUT_SUMMARY", DATASETS / "recife_protocol_c_end_to_end_orchestration_summary_v1pb.csv")
SCHEMA_REGISTRY = _p("REVP_V1PB_SCHEMA_REGISTRY", SCHEMAS / "recife_protocol_c_end_to_end_orchestration_v1pb_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1PB_SCHEMA_SUMMARY", SCHEMAS / "recife_protocol_c_end_to_end_orchestration_summary_v1pb_schema.csv")
DOC = _p("REVP_V1PB_DOC", DOCS / "revp_v1pb_protocol_c_end_to_end_orchestrator.md")

REGISTRY_FIELDS = [
    "step_id", "version", "script_path", "stage_group", "mode",
    "executed", "exit_code", "status", "expected_outputs_present",
    "stdout_tail", "stderr_tail", "notes",
]
SUMMARY_FIELDS = ["metric", "value", "interpretation"]

# ---------------------------------------------------------------------------
# Pipeline definition — only scripts that are safe to run without network/geopandas
# ---------------------------------------------------------------------------

SCRIPTS_DIR = ROOT / "scripts" / "protocolo_c"

PIPELINE_STEPS: list[tuple[str, str, str, list[str]]] = [
    # (version, script_name, stage_group, expected_output_files)
    ("v1og", "revp_v1og_rec_patch_provenance_graph_builder.py", "temporal_recovery",
     ["recife_patch_provenance_graph_registry.csv"]),
    ("v1om", "revp_v1om_recife_sentinel_sidecar_discovery.py", "temporal_recovery",
     ["recife_sentinel_sidecar_discovery_v1om.csv"]),
    ("v1on", "revp_v1on_sentinel_product_date_parser.py", "temporal_recovery",
     ["recife_sentinel_product_date_candidates_v1on.csv"]),
    ("v1oo", "revp_v1oo_recife_patch_scene_date_resolver_v3.py", "temporal_recovery",
     ["recife_patch_scene_date_resolved_v3_v1oo.csv"]),
    ("v1op", "revp_v1op_recife_event_patch_temporal_adjudication_v3.py", "temporal_recovery",
     ["recife_event_patch_temporal_adjudication_v3_v1op.csv"]),
    ("v1oq", "revp_v1oq_recife_c3_c4_dino_recheck_after_scene_date_v3.py", "temporal_recovery",
     ["recife_c3_plus_recheck_after_scene_date_v3_v1oq.csv"]),
    ("v1or", "revp_v1or_scene_date_recovery_v3_bundle.py", "temporal_recovery",
     ["recife_scene_date_recovery_v3_master_summary_v1or.csv"]),
    ("v1os", "revp_v1os_fixture_contamination_audit.py", "temporal_recovery",
     ["recife_fixture_contamination_audit_v1os.csv"]),
    ("v1ot", "revp_v1ot_scene_date_recovery_final_audit_bundle.py", "temporal_recovery",
     ["recife_scene_date_recovery_final_manifest_v1ot.csv",
      "recife_scene_date_recovery_final_scientific_summary_v1ot.csv"]),
    ("v1ou", "revp_v1ou_external_evidence_source_inventory.py", "observed_evidence",
     ["recife_external_evidence_source_inventory_v1ou.csv"]),
    ("v1ov", "revp_v1ov_ground_reference_observed_event_registry.py", "observed_evidence",
     ["recife_ground_reference_observed_event_registry_v1ov.csv"]),
    ("v1ow", "revp_v1ow_evidence_strength_precision_scoring.py", "observed_evidence",
     ["recife_ground_reference_evidence_scoring_v1ow.csv"]),
    ("v1ox", "revp_v1ox_event_patch_linkage_registry.py", "observed_evidence",
     ["recife_event_patch_linkage_registry_v1ox.csv"]),
    ("v1oy", "revp_v1oy_ground_truth_candidate_decision_audit.py", "observed_evidence",
     ["recife_ground_truth_candidate_decision_audit_v1oy.csv"]),
    ("v1oz", "revp_v1oz_dino_review_only_representation_queue.py", "observed_evidence",
     ["recife_dino_review_only_representation_queue_v1oz.csv"]),
    ("v1pa", "revp_v1pa_protocol_c_observed_evidence_bundle.py", "observed_evidence",
     ["recife_protocol_c_observed_evidence_manifest_v1pa.csv",
      "recife_protocol_c_observed_evidence_scientific_summary_v1pa.csv"]),
]


def _check_outputs(expected: list[str], datasets_dir: Path) -> tuple[bool, int]:
    present = sum(1 for f in expected if (datasets_dir / f).exists())
    return present == len(expected), present


def _run_script(script_path: Path, timeout: int = 300) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=ROOT, capture_output=True, text=True, timeout=timeout,
        )
        stdout_tail = result.stdout[-300:] if result.stdout else ""
        stderr_tail = result.stderr[-300:] if result.stderr else ""
        return result.returncode, stdout_tail, stderr_tail
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except Exception as e:
        return -2, "", str(e)[:200]


def run(mode: str = "check-only") -> None:
    datasets_dir = DATASETS
    rows: list[dict[str, Any]] = []

    for i, (version, script_name, stage_group, expected_outputs) in enumerate(PIPELINE_STEPS):
        script_path = SCRIPTS_DIR / script_name
        all_present, present_count = _check_outputs(expected_outputs, datasets_dir)

        if mode == "dry-run":
            rows.append({
                "step_id": f"V1PB_STEP_{i+1:03d}",
                "version": version,
                "script_path": script_name,
                "stage_group": stage_group,
                "mode": "dry-run",
                "executed": "false",
                "exit_code": "",
                "status": "DRY_RUN",
                "expected_outputs_present": f"{present_count}/{len(expected_outputs)}",
                "stdout_tail": "",
                "stderr_tail": "",
                "notes": f"Would execute {script_name}",
            })
        elif mode == "run":
            if not script_path.exists():
                status = "SCRIPT_NOT_FOUND"
                exit_code_val = -3
                stdout_t = ""
                stderr_t = f"Script not found: {script_name}"
            else:
                exit_code_val, stdout_t, stderr_t = _run_script(script_path)
                all_present, present_count = _check_outputs(expected_outputs, datasets_dir)
                if exit_code_val == 0 and all_present:
                    status = "PASS"
                elif exit_code_val == 0:
                    status = "PASS_OUTPUTS_PARTIAL"
                else:
                    status = "FAIL"
            rows.append({
                "step_id": f"V1PB_STEP_{i+1:03d}",
                "version": version,
                "script_path": script_name,
                "stage_group": stage_group,
                "mode": "run",
                "executed": "true",
                "exit_code": str(exit_code_val),
                "status": status,
                "expected_outputs_present": f"{present_count}/{len(expected_outputs)}",
                "stdout_tail": stdout_t.replace("\n", " ")[:150],
                "stderr_tail": stderr_t.replace("\n", " ")[:150],
                "notes": "",
            })
        else:  # check-only
            if all_present:
                status = "OUTPUTS_PRESENT"
            elif present_count > 0:
                status = "OUTPUTS_PARTIAL"
            else:
                status = "OUTPUTS_MISSING"
            rows.append({
                "step_id": f"V1PB_STEP_{i+1:03d}",
                "version": version,
                "script_path": script_name,
                "stage_group": stage_group,
                "mode": "check-only",
                "executed": "false",
                "exit_code": "",
                "status": status,
                "expected_outputs_present": f"{present_count}/{len(expected_outputs)}",
                "stdout_tail": "",
                "stderr_tail": "",
                "notes": "",
            })

    write_csv_with_header(OUT_REGISTRY, rows, REGISTRY_FIELDS)
    write_schema(SCHEMA_REGISTRY, REGISTRY_FIELDS, "v1pb_orchestration")

    # Summary
    total_steps = len(rows)
    present_steps = sum(1 for r in rows if r["status"] in ("PASS", "OUTPUTS_PRESENT"))
    partial_steps = sum(1 for r in rows if "PARTIAL" in str(r["status"]))
    missing_steps = sum(1 for r in rows if "MISSING" in str(r["status"]) or "FAIL" in str(r["status"]))

    if missing_steps == 0 and partial_steps == 0:
        final_status = "ORCHESTRATION_CHECK_READY"
    elif missing_steps > 0:
        final_status = "ORCHESTRATION_INCOMPLETE"
    else:
        final_status = "ORCHESTRATION_PARTIAL"

    summary_rows = [
        {"metric": "total_pipeline_steps", "value": str(total_steps),
         "interpretation": "Scripts no pipeline Protocol C v1og-v1pa"},
        {"metric": "steps_with_outputs_present", "value": str(present_steps),
         "interpretation": "Scripts cujos outputs existem e estao completos"},
        {"metric": "steps_partial", "value": str(partial_steps),
         "interpretation": "Scripts com outputs parciais"},
        {"metric": "steps_missing", "value": str(missing_steps),
         "interpretation": "Scripts com outputs ausentes"},
        {"metric": "orchestration_mode", "value": mode,
         "interpretation": f"Pipeline executado em modo {mode}"},
        {"metric": "final_status", "value": final_status,
         "interpretation": "Status da orquestracao end-to-end"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary_rows, SUMMARY_FIELDS)
    write_schema(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1pb_orchestration_summary")

    emit_doc(DOC, f"""# v1pb - Protocol C End-to-End Orchestrator

## Objetivo

Verificar ou executar o pipeline Protocol C (v1og-v1pa) de forma ordenada
e reproduzivel. Default: check-only (verifica outputs existentes sem executar).

## Resultado

- Total de steps: {total_steps}
- Outputs presentes: {present_steps}
- Parciais: {partial_steps}
- Ausentes: {missing_steps}
- Modo: {mode}
- Status final: {final_status}

## Modos

- `--dry-run`: lista ordem e arquivos esperados sem executar
- `--run`: executa scripts via subprocess
- `--check-only`: verifica outputs existentes (default)

## Nota

Nao usa internet. Nao baixa dados. Nao le pixels.
Nao altera ground truth ou decisoes cientificas.
""")

    print(f"[v1pb] {mode}: {present_steps}/{total_steps} steps ready. Status: {final_status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v1pb Protocol C end-to-end orchestrator")
    parser.add_argument("--mode", choices=["dry-run", "run", "check-only"], default="check-only")
    args = parser.parse_args()
    run(mode=args.mode)
