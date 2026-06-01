"""REV-P v1rr — Scientific roadmap bundle.

Consolidates the existing P0/P1/P2/P3 outputs into a programmatic roadmap:
what already exists, what is still missing, technical/scientific blockers, the
next manual actions, and the next programming steps. Read-only: it only reads
small expected CSVs, counts metrics, and writes four outputs. No recursion, no
subprocess, no pytest, no internet. Never creates labels, targets, operational
ground truth, or formal negatives.
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
    guardrail_row,
    read_csv_safe,
    safe_relpath,
    write_csv_with_header,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

# --- Inputs (small expected CSVs) ---
IN_PARTIAL_SUMMARY = _p("REVP_V1RR_IN_PARTIAL_SUMMARY", DATASETS / "protocol_c_ground_reference_partial_scientific_summary_v1qz.csv")
IN_PRIORITIES = _p("REVP_V1RR_IN_PRIORITIES", DATASETS / "protocol_c_ground_reference_external_collection_priorities_v1qz.csv")
IN_INTAKE_SUMMARY = _p("REVP_V1RR_IN_INTAKE_SUMMARY", DATASETS / "protocol_c_external_intake_scientific_summary_v1rf.csv")
IN_TASK_BOARD = _p("REVP_V1RR_IN_TASK_BOARD", DATASETS / "protocol_c_external_collection_task_board_v1ra.csv")
IN_GATE_SUMMARY = _p("REVP_V1RR_IN_GATE_SUMMARY", DATASETS / "protocol_c_review_supervisor_gate_scientific_summary_v1rm.csv")
IN_SUP_PACKETS = _p("REVP_V1RR_IN_SUP_PACKETS", DATASETS / "protocol_c_supervisor_review_packet_manifest_v1rj.csv")
IN_STATE_MACHINE = _p("REVP_V1RR_IN_STATE_MACHINE", DATASETS / "protocol_c_state_machine_registry_v1rn.csv")
IN_BACKLOG = _p("REVP_V1RR_IN_BACKLOG", DATASETS / "protocol_c_ground_reference_evidence_backlog_v1ro.csv")
IN_CLEVEL = _p("REVP_V1RR_IN_CLEVEL", DATASETS / "protocol_c_tcc_table_c_level_status_v1rp.csv")
IN_CLAIMS = _p("REVP_V1RR_IN_CLAIMS", DATASETS / "protocol_c_methodological_claims_audit_v1rq.csv")
IN_CLAIMS_SUMMARY = _p("REVP_V1RR_IN_CLAIMS_SUMMARY", DATASETS / "protocol_c_methodological_claims_audit_summary_v1rq.csv")

# --- Outputs ---
OUT_MANIFEST = _p("REVP_V1RR_OUT_MANIFEST", DATASETS / "protocol_c_scientific_roadmap_manifest_v1rr.csv")
OUT_SUMMARY = _p("REVP_V1RR_OUT_SUMMARY", DATASETS / "protocol_c_scientific_roadmap_summary_v1rr.csv")
OUT_QUEUE = _p("REVP_V1RR_OUT_QUEUE", DATASETS / "protocol_c_next_action_queue_v1rr.csv")
OUT_STEPS = _p("REVP_V1RR_OUT_STEPS", DATASETS / "protocol_c_programming_next_steps_v1rr.csv")
SCHEMA_MANIFEST = _p("REVP_V1RR_SCHEMA_MANIFEST", SCHEMAS / "protocol_c_scientific_roadmap_manifest_v1rr_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1RR_SCHEMA_SUMMARY", SCHEMAS / "protocol_c_scientific_roadmap_summary_v1rr_schema.csv")
SCHEMA_QUEUE = _p("REVP_V1RR_SCHEMA_QUEUE", SCHEMAS / "protocol_c_next_action_queue_v1rr_schema.csv")
SCHEMA_STEPS = _p("REVP_V1RR_SCHEMA_STEPS", SCHEMAS / "protocol_c_programming_next_steps_v1rr_schema.csv")
DOC = _p("REVP_V1RR_DOC", DOCS / "revp_v1rr_scientific_roadmap_bundle.md")

MANIFEST_FIELDS = ["artifact_id", "stage", "artifact_path", "artifact_exists", "row_count", "artifact_role", "status", "notes"]
SUMMARY_FIELDS = ["summary_id", "metric", "value", "interpretation", "methodological_status", "writing_use"]
QUEUE_FIELDS = [
    "action_id", "priority", "action_type", "region", "related_stage", "blocker",
    "recommended_action", "required_external_source", "blocks_c3", "blocks_c4",
    "review_only", "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "formal_negative", "dino_validates_event",
    "absence_as_negative", "notes",
]
STEPS_FIELDS = [
    "step_id", "priority", "module_scope", "next_script_range", "objective",
    "required_inputs", "expected_outputs", "risk", "model_recommendation",
    "done_when", "notes",
]

# Final statuses
ST_READY = "REV_P_GROUND_REFERENCE_WORKBENCH_READY_FOR_MANUAL_REVIEW"
ST_WAIT_INTAKE = "REV_P_WAITING_EXTERNAL_DOCUMENT_INTAKE"
ST_WAIT_REVIEW = "REV_P_WAITING_DOUBLE_REVIEW_RESPONSES"
ST_CLAIMS_FAIL = "REV_P_METHOD_CLAIMS_GUARDRAIL_FAIL_CLOSED"

MANDATORY_SENTENCE = (
    "O roadmap v1rr consolida o estado cientifico do Protocolo C: o projeto possui "
    "infraestrutura de revisao, intake externo, gate supervisor e DINO review-only, mas "
    "ainda aguarda documentos externos e respostas humanas para avancar de C1/C2 para "
    "candidatos C3. Nenhum rotulo operacional, target supervisionado, ground truth "
    "operacional ou negativo formal e criado nesta etapa."
)


def _stat(rows: list[dict[str, str]], key: str, default: str = "") -> str:
    for r in rows:
        if r.get("stat_key") == key:
            return r.get("stat_value", default)
    return default


def _exists(path: Path) -> str:
    return "true" if path.exists() else "false"


def _manifest_entry(aid: str, stage: str, path: Path, role: str, status: str) -> dict[str, Any]:
    rows = read_csv_safe(path)
    return {
        "artifact_id": aid, "stage": stage, "artifact_path": safe_relpath(path),
        "artifact_exists": _exists(path), "row_count": str(len(rows)),
        "artifact_role": role, "status": status, "notes": "",
    }


def run(datasets: Path | None = None) -> dict[str, Any]:
    partial = read_csv_safe(IN_PARTIAL_SUMMARY)
    priorities = read_csv_safe(IN_PRIORITIES)
    intake = read_csv_safe(IN_INTAKE_SUMMARY)
    board = read_csv_safe(IN_TASK_BOARD)
    gate = read_csv_safe(IN_GATE_SUMMARY)
    sup_packets = read_csv_safe(IN_SUP_PACKETS)
    backlog = read_csv_safe(IN_BACKLOG)
    claims_summary = read_csv_safe(IN_CLAIMS_SUMMARY)

    # --- Consolidated statuses ---
    p0_status = _stat(partial, "final_status", "UNKNOWN")
    p1_status = _stat(intake, "final_status", "UNKNOWN")
    p2_status = _stat(gate, "final_status", "UNKNOWN")
    claims_status = _stat(claims_summary, "audit_status", "UNKNOWN")
    claims_violations = int(_stat(claims_summary, "violations", "0") or "0")

    source_req_missing = _stat(partial, "source_requirements_missing", str(len(priorities)))
    external_tasks = str(len(board))
    review_packets = _stat(gate, "review_packets_available", "0")
    completed_reviews = _stat(gate, "completed_double_reviews", "0")
    supervisor_packets = str(len(sup_packets))
    c3_candidates = _stat(gate, "c3_reference_candidates_review_only", "0")

    # --- Final scientific status (priority order) ---
    intake_waiting = p1_status in ("EXTERNAL_INTAKE_WAITING_MANUAL_DOCUMENTS",
                                   "EXTERNAL_INTAKE_TASK_BOARD_READY", "UNKNOWN")
    if claims_violations > 0:
        final_status = ST_CLAIMS_FAIL
    elif intake_waiting:
        final_status = ST_WAIT_INTAKE
    elif completed_reviews == "0":
        final_status = ST_WAIT_REVIEW
    else:
        final_status = ST_READY

    # --- Manifest ---
    manifest = [
        _manifest_entry("V1RR_A01", "P0_v1qz", IN_PARTIAL_SUMMARY, "ground_reference_partial_summary", p0_status),
        _manifest_entry("V1RR_A02", "P0_v1qz", IN_PRIORITIES, "external_collection_priorities", "READY" if priorities else "EMPTY"),
        _manifest_entry("V1RR_A03", "P1_v1rf", IN_INTAKE_SUMMARY, "external_intake_summary", p1_status),
        _manifest_entry("V1RR_A04", "P1_v1ra", IN_TASK_BOARD, "external_collection_task_board", "READY" if board else "EMPTY"),
        _manifest_entry("V1RR_A05", "P2_v1rm", IN_GATE_SUMMARY, "review_supervisor_gate_summary", p2_status),
        _manifest_entry("V1RR_A06", "P2_v1rj", IN_SUP_PACKETS, "supervisor_packets", "READY" if sup_packets else "WAITING"),
        _manifest_entry("V1RR_A07", "P3_v1rn", IN_STATE_MACHINE, "state_machine_registry", "READY"),
        _manifest_entry("V1RR_A08", "P3_v1ro", IN_BACKLOG, "evidence_backlog", "READY" if backlog else "EMPTY"),
        _manifest_entry("V1RR_A09", "P3_v1rp", IN_CLEVEL, "tcc_c_level_status", "READY"),
        _manifest_entry("V1RR_A10", "P3_v1rq", IN_CLAIMS, "methodological_claims_audit", claims_status),
    ]
    write_csv_with_header(OUT_MANIFEST, manifest, MANIFEST_FIELDS)
    write_schema_safe(SCHEMA_MANIFEST, MANIFEST_FIELDS, "v1rr_manifest")

    # --- Summary ---
    def s(i, metric, value, interp, mstatus="REVIEW_ONLY", use="tcc_results"):
        return {"summary_id": f"V1RR_S{i:02d}", "metric": metric, "value": str(value),
                "interpretation": interp, "methodological_status": mstatus, "writing_use": use}
    summary = [
        s(1, "p0_status", p0_status, "estado do partial validation workbench"),
        s(2, "p1_status", p1_status, "estado do intake externo"),
        s(3, "p2_status", p2_status, "estado do gate de revisao/supervisor"),
        s(4, "p3_claims_audit_status", claims_status, "auditoria metodologica", "GUARDRAIL"),
        s(5, "source_requirements_missing", source_req_missing, "fontes externas a coletar", "REVIEW_ONLY", "methods"),
        s(6, "external_collection_tasks", external_tasks, "tarefas de coleta externa", "REVIEW_ONLY", "methods"),
        s(7, "review_packets", review_packets, "pacotes A/B disponiveis", "REVIEW_ONLY", "methods"),
        s(8, "completed_reviews", completed_reviews, "revisoes duplas completas", "REVIEW_ONLY", "results"),
        s(9, "supervisor_packets", supervisor_packets, "pacotes para supervisor", "REVIEW_ONLY", "results"),
        s(10, "c3_candidates_review_only", c3_candidates, "candidatos C3 review-only", "REVIEW_ONLY", "results"),
        s(11, "c4_formal_negatives", "0", "nenhum negativo formal por ausencia", "GUARDRAIL", "results"),
        s(12, "labels_created", "0", "nenhum label operacional", "GUARDRAIL", "results"),
        s(13, "targets_created", "0", "nenhum target supervisionado", "GUARDRAIL", "results"),
        s(14, "ground_truth_operational_created", "0", "nenhum ground truth operacional", "GUARDRAIL", "results"),
        s(15, "final_scientific_status", final_status, "estado cientifico consolidado", "REVIEW_ONLY", "discussion"),
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    write_schema_safe(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1rr_summary")

    # --- Next action queue (from backlog) ---
    queue: list[dict[str, Any]] = []
    for i, b in enumerate(backlog):
        blocker = b.get("blocker", "")
        action_type = {
            "EXTERNAL_SOURCE_NOT_LOCAL": "COLLECT_EXTERNAL_SOURCE",
            "EXTERNAL_DOCUMENTS_NOT_INTAKEN": "INTAKE_EXTERNAL_DOCUMENTS",
            "DOUBLE_REVIEW_NOT_COMPLETED": "RUN_DOUBLE_REVIEW",
            "SUPERVISOR_DECISION_PENDING": "SUPERVISOR_DECISION",
        }.get(blocker, "REVIEW")
        row = {
            "action_id": f"V1RR_ACT_{i:04d}", "priority": b.get("priority", "P2"),
            "action_type": action_type, "region": b.get("region", ""),
            "related_stage": b.get("current_state", ""), "blocker": blocker,
            "recommended_action": b.get("next_action", ""),
            "required_external_source": b.get("missing_source_name", ""),
            "blocks_c3": b.get("blocks_c3", "false"), "blocks_c4": b.get("blocks_c4", "false"),
            "notes": "manual_required",
        }
        row.update(guardrail_row())
        queue.append(row)
    assert_clean_rows(queue, "v1rr_queue")
    write_csv_with_header(OUT_QUEUE, queue, QUEUE_FIELDS)
    write_schema_safe(SCHEMA_QUEUE, QUEUE_FIELDS, "v1rr_queue")

    # --- Programming next steps (static, controlled) ---
    steps = [
        {"step_id": "V1RR_ST01", "priority": "P0", "module_scope": "external_intake",
         "next_script_range": "v1ra-v1rf (rerun apos coleta)",
         "objective": "Coletar documentos externos manualmente e validar via v1rc",
         "required_inputs": "documentos oficiais (CEMADEN/ANA/INMET/SGB/Defesa Civil/Diario Oficial)",
         "expected_outputs": "intake validado e event candidates review-only",
         "risk": "fontes podem nao estar publicamente disponiveis",
         "model_recommendation": "revisao humana", "done_when": "intake validado pass", "notes": ""},
        {"step_id": "V1RR_ST02", "priority": "P0", "module_scope": "double_review",
         "next_script_range": "v1rg-v1rm (rerun apos respostas)",
         "objective": "Preencher respostas A/B e validar/pontuar",
         "required_inputs": "respostas humanas A/B preenchidas",
         "expected_outputs": "completed review scores e supervisor packets",
         "risk": "desacordo entre revisores", "model_recommendation": "dupla revisao humana",
         "done_when": "completed reviews validadas", "notes": ""},
        {"step_id": "V1RR_ST03", "priority": "P1", "module_scope": "supervisor_gate",
         "next_script_range": "v1rj-v1rl",
         "objective": "Decisao supervisora sobre candidatos C3 review-only",
         "required_inputs": "decisao supervisor preenchida",
         "expected_outputs": "C3 candidates review-only aprovados (sem label)",
         "risk": "promocao indevida a label", "model_recommendation": "supervisor humano",
         "done_when": "decisao supervisor validada", "notes": ""},
        {"step_id": "V1RR_ST04", "priority": "P2", "module_scope": "dashboard_roadmap",
         "next_script_range": "v1rs-v1rz (futuro)",
         "objective": "Bundles adicionais de QA/relatorio quando houver evidencia real",
         "required_inputs": "outputs P0/P1/P2/P3", "expected_outputs": "relatorios TCC adicionais",
         "risk": "scripts incompletos", "model_recommendation": "implementar so com evidencia",
         "done_when": "evidencia real disponivel", "notes": ""},
    ]
    write_csv_with_header(OUT_STEPS, steps, STEPS_FIELDS)
    write_schema_safe(SCHEMA_STEPS, STEPS_FIELDS, "v1rr_steps")

    write_doc(
        DOC,
        "v1rr — Scientific Roadmap Bundle",
        [
            "## Objetivo",
            "Consolidar o estado cientifico do Protocolo C (P0/P1/P2/P3): o que ja existe, o "
            "que falta buscar, blockers tecnicos e cientificos, proximas acoes manuais e "
            "proximos passos de programacao.",
            "## Estado consolidado",
            f"p0={p0_status}; p1={p1_status}; p2={p2_status}; claims={claims_status}. "
            f"final_scientific_status={final_status}.",
            "## Acoes proximas",
            f"Fila de acoes: {len(queue)}. Passos de programacao: {len(steps)}. "
            "Inclui coleta de documentos externos e respostas de dupla revisao.",
            "## Declaracao obrigatoria",
            MANDATORY_SENTENCE,
        ],
    )
    print(f"[v1rr] final_status={final_status} actions={len(queue)} steps={len(steps)} claims={claims_status}")
    return {"final_status": final_status, "actions": len(queue), "steps": len(steps),
            "p0": p0_status, "p1": p1_status, "p2": p2_status}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1rr scientific roadmap bundle").parse_args()
    run()
