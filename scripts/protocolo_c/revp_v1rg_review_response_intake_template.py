"""REV-P v1rg — Review response intake template.

Generates a fillable A/B review response template from the v1qw double-review
packets. Never fills answers; never creates evidence. Includes safe-filling
instructions in the doc. Review-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1rg_v1rm_review_response_common import (
    DATASETS,
    DOCS,
    SCHEMAS,
    REVIEW_QUESTIONS,
    _p,
    assert_clean_rows,
    guardrail_row,
    normalize_region,
    normalize_reviewer_slot,
    read_csv_safe,
    write_csv_with_header,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"

IN_PACKETS = _p("REVP_V1RG_IN_PACKETS", DATASETS / "protocol_c_double_review_packet_manifest_v1qw.csv")
OUT_TEMPLATE = _p("REVP_V1RG_OUT_TEMPLATE", DATASETS / "protocol_c_review_response_intake_template_v1rg.csv")
OUT_SCHEMA = _p("REVP_V1RG_OUT_SCHEMA", CONFIGS / "protocol_c_review_response_schema_v1rg.csv")
SCHEMA_TEMPLATE = _p("REVP_V1RG_SCHEMA_TEMPLATE", SCHEMAS / "protocol_c_review_response_intake_template_v1rg_schema.csv")
DOC = _p("REVP_V1RG_DOC", DOCS / "revp_v1rg_review_response_intake_template.md")

TEMPLATE_FIELDS = [
    "response_id", "packet_id", "review_sample_id", "reviewer_slot",
    "reviewer_id_pseudonym", "event_id", "patch_id", "region",
    "question_id", "question_text", "answer_value", "confidence_0_4",
    "evidence_note", "source_reference", "uncertainty_note", "response_status",
    "review_only", "can_create_operational_label", "can_train_model",
    "target_created", "ground_truth_operational",
]

CONFIG_FIELDS = ["field", "required", "description", "example_placeholder"]

_QUESTION_TEXT = {
    "evidence_visible": "A evidencia e visivel/documentada na fonte? (sim/nao/incerto)",
    "event_supported": "A fonte sustenta a ocorrencia do evento? (sim/nao/incerto)",
    "location_supported": "A localizacao do evento e sustentada? (sim/nao/incerto)",
    "timing_supported": "A data/janela temporal e sustentada? (sim/nao/incerto)",
    "source_quality": "Qualidade da fonte (oficial/tecnica/secundaria/desconhecida)",
    "independent_source_present": "Existe fonte independente adicional? (sim/nao)",
    "uncertainty_level": "Nivel de incerteza (baixo/medio/alto)",
    "recommended_decision": "Decisao recomendada (C1/C2/C3-candidate/block)",
    "uncertainty_notes": "Notas de incerteza (texto livre)",
}


def build_rows(packets: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in packets:
        rsid = p.get("review_sample_id", "")
        slot = normalize_reviewer_slot(p.get("reviewer_slot", ""))
        packet_id = p.get("packet_id", "")
        region = normalize_region(p.get("region", ""))
        for q in REVIEW_QUESTIONS:
            row = {
                "response_id": f"V1RG_RESP_{rsid}_{slot[-1]}_{q}",
                "packet_id": packet_id, "review_sample_id": rsid,
                "reviewer_slot": slot, "reviewer_id_pseudonym": "",
                "event_id": p.get("event_id", ""), "patch_id": p.get("patch_id", ""),
                "region": region, "question_id": q, "question_text": _QUESTION_TEXT[q],
                "answer_value": "", "confidence_0_4": "",
                "evidence_note": "", "source_reference": "", "uncertainty_note": "",
                "response_status": "AWAITING_HUMAN_RESPONSE",
            }
            row.update(guardrail_row())
            rows.append(row)
    return rows


def run(datasets: Path | None = None) -> dict[str, Any]:
    packets = read_csv_safe(IN_PACKETS)
    rows = build_rows(packets)
    assert_clean_rows(rows, "v1rg_template")

    write_csv_with_header(OUT_TEMPLATE, rows, TEMPLATE_FIELDS)
    write_schema_safe(SCHEMA_TEMPLATE, TEMPLATE_FIELDS, "v1rg_template")

    config_rows = [
        {"field": "response_id", "required": "auto", "description": "ID unico da resposta", "example_placeholder": "V1RG_RESP_..."},
        {"field": "packet_id", "required": "yes", "description": "ID do pacote v1qw", "example_placeholder": "V1QW_PKT_..."},
        {"field": "reviewer_slot", "required": "yes", "description": "REVIEWER_A ou REVIEWER_B", "example_placeholder": "REVIEWER_A"},
        {"field": "reviewer_id_pseudonym", "required": "yes", "description": "Pseudonimo do revisor (sem PII)", "example_placeholder": "REV_01"},
        {"field": "question_id", "required": "yes", "description": "ID da pergunta", "example_placeholder": "event_supported"},
        {"field": "answer_value", "required": "yes", "description": "Resposta (sim/nao/incerto/texto)", "example_placeholder": "sim"},
        {"field": "confidence_0_4", "required": "yes", "description": "Confianca 0 a 4", "example_placeholder": "3"},
        {"field": "source_reference", "required": "conditional", "description": "Referencia da fonte (obrigatoria se event_supported=sim)", "example_placeholder": "Boletim Defesa Civil"},
        {"field": "response_status", "required": "auto", "description": "Status do preenchimento", "example_placeholder": "FILLED"},
    ]
    write_csv_with_header(OUT_SCHEMA, config_rows, CONFIG_FIELDS)

    n_packets = len({p.get("packet_id", "") for p in packets})
    write_doc(
        DOC,
        "v1rg — Review Response Intake Template",
        [
            "## Objetivo",
            "Gerar template preenchivel para respostas de Review A/B a partir dos pacotes "
            "v1qw. Nenhuma resposta e preenchida; nenhuma evidencia e criada.",
            "## Preenchimento seguro",
            "1) Cada revisor preenche apenas answer_value/confidence/notes do seu slot. "
            "2) Usar pseudonimo, nunca PII. 3) Nao colar paths absolutos nem referencias a "
            "diretorios locais. 4) source_reference obrigatoria quando event_supported=sim. "
            "5) Apontar REVP_PROTOCOL_C_REVIEW_RESPONSES_PATH para o CSV preenchido e rodar v1rh.",
            "## Resultado",
            f"Pacotes A/B: {n_packets}. Linhas de template (pacote x pergunta): {len(rows)}.",
            "## Guardrails",
            "review_only=true. Nenhuma resposta preenchida automaticamente. "
            "Nenhum label/target/ground truth operacional.",
        ],
    )
    print(f"[v1rg] packets={n_packets} template_rows={len(rows)}")
    return {"packets": n_packets, "rows": len(rows)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1rg review response template").parse_args()
    run()
