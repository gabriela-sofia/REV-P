"""REV-P v1rn — Protocol C state machine registry.

Static, auditable registry of Protocol C states (C1/C2/C3-candidate/C4-blocked
/blocked), their allowed and forbidden transitions, required gates, and the
output artifacts that prove each gate. Review-only documentation; no labels,
no targets, no operational ground truth.
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
    write_csv_with_header,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_REGISTRY = _p("REVP_V1RN_OUT_REGISTRY", DATASETS / "protocol_c_state_machine_registry_v1rn.csv")
OUT_SUMMARY = _p("REVP_V1RN_OUT_SUMMARY", DATASETS / "protocol_c_state_machine_summary_v1rn.csv")
SCHEMA_REGISTRY = _p("REVP_V1RN_SCHEMA_REGISTRY", SCHEMAS / "protocol_c_state_machine_registry_v1rn_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1RN_SCHEMA_SUMMARY", SCHEMAS / "protocol_c_state_machine_summary_v1rn_schema.csv")
DOC = _p("REVP_V1RN_DOC", DOCS / "revp_v1rn_protocol_c_state_machine_registry.md")

REGISTRY_FIELDS = [
    "state_id", "state_name", "description", "allowed_transitions",
    "forbidden_transitions", "required_gates", "proof_outputs",
    "is_operational_label", "is_open", "review_only",
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "notes",
]

SUMMARY_FIELDS = ["stat_key", "stat_value"]

# state_id, name, description, allowed, forbidden, gates, proof_outputs, is_open
_STATES: list[tuple[str, ...]] = [
    (
        "C1", "C1_CONTEXTUAL_ONLY",
        "Evidencia contextual; nao confirma evento.",
        "C1->C2",
        "C1->C3_CANDIDATE;C1->C4_BLOCKED;C1->LABEL",
        "fonte_documentada",
        "protocol_c_official_evidence_source_requirements_v1qu.csv;recife_ground_reference_observed_event_registry_v1ov.csv",
        "false",
    ),
    (
        "C2", "C2_REVIEW_ONLY_CANDIDATE",
        "Candidato review-only; precisa dupla revisao.",
        "C2->C3_CANDIDATE;C2->C1",
        "C2->LABEL;C2->C4_BLOCKED",
        "dupla_revisao;fonte_independente",
        "protocol_c_event_patch_review_sample_v1qv.csv;protocol_c_double_review_packet_manifest_v1qw.csv",
        "false",
    ),
    (
        "C3_CANDIDATE", "C3_REFERENCE_CANDIDATE_NEEDS_SUPERVISOR",
        "Candidato a referencia C3; exige supervisor; permanece review-only.",
        "C3_CANDIDATE->C2",
        "C3_CANDIDATE->LABEL;C3_CANDIDATE->GROUND_TRUTH_OPERACIONAL;C3_CANDIDATE_AUTO_PROMOTE",
        "precisao_temporal;precisao_espacial;fonte_oficial;dupla_revisao;adjudicacao;supervisor",
        "protocol_c_ground_reference_adjudication_registry_v1qy.csv;protocol_c_supervisor_review_packet_manifest_v1rj.csv;protocol_c_review_supervisor_gate_scientific_summary_v1rm.csv",
        "false",
    ),
    (
        "C4_BLOCKED", "C4_NEGATIVE_BLOCKED",
        "Negativo formal fechado; nunca aberto sem fonte formal negativa.",
        "",
        "C4_AUTO_OPEN;ABSENCE_AS_NEGATIVE",
        "fonte_formal_negativa_explicita",
        "protocol_c_ground_reference_adjudication_summary_v1qy.csv",
        "false",
    ),
    (
        "BLOCKED", "BLOCKED_INSUFFICIENT_EVIDENCE",
        "Bloqueado por evidencia insuficiente.",
        "BLOCKED->C1;BLOCKED->C2",
        "BLOCKED->LABEL",
        "coleta_externa_adicional",
        "protocol_c_ground_reference_external_collection_priorities_v1qz.csv",
        "false",
    ),
]


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sid, name, desc, allowed, forbidden, gates, proof, is_open in _STATES:
        row = {
            "state_id": sid, "state_name": name, "description": desc,
            "allowed_transitions": allowed, "forbidden_transitions": forbidden,
            "required_gates": gates, "proof_outputs": proof,
            "is_operational_label": "false", "is_open": is_open, "notes": "",
        }
        row.update(guardrail_row())
        rows.append(row)
    return rows


def run(datasets: Path | None = None) -> dict[str, Any]:
    rows = build_rows()
    assert_clean_rows(rows, "v1rn_registry")
    write_csv_with_header(OUT_REGISTRY, rows, REGISTRY_FIELDS)
    write_schema_safe(SCHEMA_REGISTRY, REGISTRY_FIELDS, "v1rn_state_machine")

    open_states = sum(1 for r in rows if r["is_open"] == "true")
    summary = [
        {"stat_key": "states_total", "stat_value": str(len(rows))},
        {"stat_key": "open_states", "stat_value": str(open_states)},
        {"stat_key": "operational_label_states", "stat_value": "0"},
        {"stat_key": "c4_open", "stat_value": "0"},
        {"stat_key": "stage", "stat_value": "v1rn"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    write_schema_safe(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1rn_summary")

    write_doc(
        DOC,
        "v1rn — Protocol C State Machine Registry",
        [
            "## Objetivo",
            "Registrar de forma auditavel os estados do Protocolo C (C1, C2, C3-candidate, "
            "C4-blocked, blocked), transicoes permitidas/proibidas, gates necessarios e os "
            "outputs que comprovam cada gate.",
            "## Invariantes",
            "Nenhum estado e label operacional. C3-candidate exige supervisor e permanece "
            "review-only. C4 nunca abre automaticamente nem por ausencia de evidencia. "
            "Promocao automatica para label e proibida.",
            "## Resultado",
            f"Estados: {len(rows)}. Estados abertos (operacionais): {open_states}. "
            "Estados de label operacional: 0.",
            "## Guardrails",
            "review_only=true. can_create_operational_label=false. ground_truth_operational=false.",
        ],
    )
    print(f"[v1rn] states={len(rows)} open={open_states}")
    return {"states": len(rows), "open": open_states}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1rn protocol C state machine").parse_args()
    run()
