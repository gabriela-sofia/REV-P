"""REV-P v1rk — Supervisor decision intake template.

Generates a fillable template for the supervisor's final decision from the
v1rj packets, plus a schema/config descriptor. Never fills a decision; never
creates a label or ground truth. Review-only.
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
    assert_clean_rows,
    guardrail_row,
    read_csv_safe,
    write_csv_with_header,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"

IN_PACKETS = _p("REVP_V1RK_IN_PACKETS", DATASETS / "protocol_c_supervisor_review_packet_manifest_v1rj.csv")
OUT_TEMPLATE = _p("REVP_V1RK_OUT_TEMPLATE", DATASETS / "protocol_c_supervisor_decision_intake_template_v1rk.csv")
OUT_SCHEMA = _p("REVP_V1RK_OUT_SCHEMA", CONFIGS / "protocol_c_supervisor_decision_schema_v1rk.csv")
SCHEMA_TEMPLATE = _p("REVP_V1RK_SCHEMA_TEMPLATE", SCHEMAS / "protocol_c_supervisor_decision_intake_template_v1rk_schema.csv")
DOC = _p("REVP_V1RK_DOC", DOCS / "revp_v1rk_supervisor_decision_intake_template.md")

TEMPLATE_FIELDS = [
    "supervisor_decision_id", "supervisor_packet_id", "review_sample_id",
    "event_id", "patch_id", "region", "supervisor_decision",
    "decision_confidence_0_4", "required_followup", "decision_note",
    "decision_status", "review_only", "can_create_operational_label",
    "can_train_model", "target_created", "ground_truth_operational", "formal_negative",
]

CONFIG_FIELDS = ["field", "required", "description", "example_placeholder"]


def build_rows(packets: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in packets:
        pid = p.get("supervisor_packet_id", "")
        rsid = p.get("review_sample_id", "")
        row = {
            "supervisor_decision_id": f"V1RK_SDEC_{rsid}",
            "supervisor_packet_id": pid, "review_sample_id": rsid,
            "event_id": p.get("event_id", ""), "patch_id": p.get("patch_id", ""),
            "region": p.get("region", ""),
            "supervisor_decision": "", "decision_confidence_0_4": "",
            "required_followup": "", "decision_note": "",
            "decision_status": "AWAITING_SUPERVISOR_DECISION",
        }
        row.update(guardrail_row())
        rows.append(row)
    return rows


def run(datasets: Path | None = None) -> dict[str, Any]:
    packets = read_csv_safe(IN_PACKETS)
    rows = build_rows(packets)
    assert_clean_rows(rows, "v1rk_template")

    write_csv_with_header(OUT_TEMPLATE, rows, TEMPLATE_FIELDS)
    write_schema_safe(SCHEMA_TEMPLATE, TEMPLATE_FIELDS, "v1rk_template")

    config_rows = [
        {"field": "supervisor_packet_id", "required": "yes", "description": "ID do pacote v1rj", "example_placeholder": "V1RJ_SPKT_..."},
        {"field": "supervisor_decision", "required": "yes", "description": "Acao do supervisor permitida", "example_placeholder": "APPROVE_C3_CANDIDATE_REVIEW_ONLY"},
        {"field": "decision_confidence_0_4", "required": "yes", "description": "Confianca 0 a 4", "example_placeholder": "3"},
        {"field": "required_followup", "required": "optional", "description": "Followup necessario", "example_placeholder": "coletar boletim oficial"},
        {"field": "decision_note", "required": "optional", "description": "Justificativa", "example_placeholder": ""},
        {"field": "decision_status", "required": "auto", "description": "Status do preenchimento", "example_placeholder": "FILLED"},
    ]
    write_csv_with_header(OUT_SCHEMA, config_rows, CONFIG_FIELDS)

    write_doc(
        DOC,
        "v1rk — Supervisor Decision Intake Template",
        [
            "## Objetivo",
            "Gerar template preenchivel para a decisao final do supervisor a partir dos "
            "pacotes v1rj. Nenhuma decisao e preenchida; nenhuma evidencia e criada.",
            "## Preenchimento seguro",
            "Preencher supervisor_decision com uma acao permitida, decision_confidence_0_4 "
            "(0-4), e notas. Apontar REVP_PROTOCOL_C_SUPERVISOR_DECISIONS_PATH para o CSV "
            "preenchido e rodar v1rl.",
            "## Resultado",
            f"Pacotes para decisao: {len(rows)}.",
            "## Guardrails",
            "Aprovar C3 candidate permanece review-only: can_create_operational_label=false, "
            "ground_truth_operational=false. C4 nunca aberto sem fonte formal negativa.",
        ],
    )
    print(f"[v1rk] decision_template_rows={len(rows)}")
    return {"rows": len(rows)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1rk supervisor decision template").parse_args()
    run()
