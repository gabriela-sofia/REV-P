"""REV-P v1qw — Double-review packet generator (A/B).

For each review_sample_id from v1qv, generates two reviewer slots
(REVIEWER_A, REVIEWER_B) and blank review forms with mandatory questions.
Never fills a final answer; only placeholders. Review-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1lj_v1lq_common import DATASETS, DOCS, SCHEMAS
from revp_v1qu_v1qz_ground_reference_common import (
    _p,
    assert_clean_rows,
    guardrail_row,
    read_csv_safe,
    write_csv_with_header,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

IN_SAMPLE = _p("REVP_V1QW_IN_SAMPLE", DATASETS / "protocol_c_event_patch_review_sample_v1qv.csv")
OUT_MANIFEST = _p("REVP_V1QW_OUT_MANIFEST", DATASETS / "protocol_c_double_review_packet_manifest_v1qw.csv")
OUT_FORMS = _p("REVP_V1QW_OUT_FORMS", DATASETS / "protocol_c_double_review_forms_v1qw.csv")
OUT_SUMMARY = _p("REVP_V1QW_OUT_SUMMARY", DATASETS / "protocol_c_double_review_packet_summary_v1qw.csv")
SCHEMA_MANIFEST = _p("REVP_V1QW_SCHEMA_MANIFEST", SCHEMAS / "protocol_c_double_review_packet_manifest_v1qw_schema.csv")
SCHEMA_FORMS = _p("REVP_V1QW_SCHEMA_FORMS", SCHEMAS / "protocol_c_double_review_forms_v1qw_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1QW_SCHEMA_SUMMARY", SCHEMAS / "protocol_c_double_review_packet_summary_v1qw_schema.csv")
DOC = _p("REVP_V1QW_DOC", DOCS / "revp_v1qw_double_review_packet_generator.md")

REVIEWERS = ["REVIEWER_A", "REVIEWER_B"]

MANIFEST_FIELDS = [
    "packet_id", "review_sample_id", "reviewer_slot", "event_id", "patch_id",
    "alias", "region", "hazard_type", "evidence_status", "dino_queue_status",
    "packet_status", "review_only", "dino_validates_event",
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "notes",
]

# Mandatory review questions (placeholders only — never filled)
REVIEW_QUESTIONS = [
    "evidence_visible",
    "event_supported",
    "location_supported",
    "timing_supported",
    "source_quality",
    "independent_source_present",
    "uncertainty_level",
    "recommended_decision",
    "uncertainty_notes",
]

FORM_FIELDS = [
    "form_id", "packet_id", "review_sample_id", "reviewer_slot",
    "question_key", "question_text", "answer_placeholder", "response_value",
    "review_only", "dino_validates_event", "notes",
]

SUMMARY_FIELDS = ["stat_key", "stat_value"]

_QUESTION_TEXT = {
    "evidence_visible": "A evidencia e visivel/documentada na fonte? (sim/nao/incerto)",
    "event_supported": "A fonte sustenta a ocorrencia do evento? (sim/nao/incerto)",
    "location_supported": "A localizacao do evento e sustentada? (sim/nao/incerto)",
    "timing_supported": "A data/janela temporal e sustentada? (sim/nao/incerto)",
    "source_quality": "Qualidade da fonte (oficial/tecnica/secundaria/desconhecida)",
    "independent_source_present": "Existe fonte independente adicional? (sim/nao)",
    "uncertainty_level": "Nivel de incerteza (baixo/medio/alto)",
    "recommended_decision": "Decisao recomendada (C1/C2/C3-candidate/bloquear)",
    "uncertainty_notes": "Notas de incerteza (texto livre)",
}


def build(sample: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest: list[dict[str, Any]] = []
    forms: list[dict[str, Any]] = []
    for s in sample:
        rsid = s.get("review_sample_id", "")
        for reviewer in REVIEWERS:
            packet_id = f"V1QW_PKT_{rsid}_{reviewer[-1]}"
            mrow = {
                "packet_id": packet_id, "review_sample_id": rsid,
                "reviewer_slot": reviewer,
                "event_id": s.get("event_id", ""), "patch_id": s.get("patch_id", ""),
                "alias": s.get("alias", ""), "region": s.get("region", ""),
                "hazard_type": s.get("hazard_type", ""),
                "evidence_status": s.get("evidence_status", ""),
                "dino_queue_status": s.get("dino_queue_status", ""),
                "packet_status": "AWAITING_HUMAN_REVIEW",
                "notes": "review_only_no_answer_prefilled",
            }
            mrow.update(guardrail_row())
            manifest.append(mrow)
            for q in REVIEW_QUESTIONS:
                frow = {
                    "form_id": f"V1QW_FORM_{rsid}_{reviewer[-1]}_{q}",
                    "packet_id": packet_id, "review_sample_id": rsid,
                    "reviewer_slot": reviewer, "question_key": q,
                    "question_text": _QUESTION_TEXT[q],
                    "answer_placeholder": "<TO_BE_FILLED_BY_HUMAN_REVIEWER>",
                    "response_value": "",
                    "review_only": "true", "dino_validates_event": "false",
                    "notes": "",
                }
                forms.append(frow)
    return manifest, forms


def run(datasets: Path | None = None) -> dict[str, Any]:
    sample = read_csv_safe(IN_SAMPLE)
    manifest, forms = build(sample)
    assert_clean_rows(manifest, "v1qw_manifest")

    write_csv_with_header(OUT_MANIFEST, manifest, MANIFEST_FIELDS)
    write_csv_with_header(OUT_FORMS, forms, FORM_FIELDS)
    write_schema_safe(SCHEMA_MANIFEST, MANIFEST_FIELDS, "v1qw_manifest")
    write_schema_safe(SCHEMA_FORMS, FORM_FIELDS, "v1qw_forms")

    n_samples = len(sample)
    n_packets = len(manifest)
    summary = [
        {"stat_key": "review_samples", "stat_value": str(n_samples)},
        {"stat_key": "packets_generated", "stat_value": str(n_packets)},
        {"stat_key": "reviewers_per_sample", "stat_value": str(len(REVIEWERS))},
        {"stat_key": "forms_generated", "stat_value": str(len(forms))},
        {"stat_key": "questions_per_form", "stat_value": str(len(REVIEW_QUESTIONS))},
        {"stat_key": "ab_balanced", "stat_value": "true" if n_packets == n_samples * 2 else "false"},
        {"stat_key": "stage", "stat_value": "v1qw"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    write_schema_safe(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1qw_summary")

    write_doc(
        DOC,
        "v1qw — Double-Review Packet Generator",
        [
            "## Objetivo",
            "Gerar pacotes de revisao dupla A/B. Cada review_sample_id gera dois slots "
            "(REVIEWER_A, REVIEWER_B) e formularios em branco com perguntas obrigatorias.",
            "## Perguntas obrigatorias",
            "evidence_visible, event_supported, location_supported, timing_supported, "
            "source_quality, independent_source_present, uncertainty_level, "
            "recommended_decision, uncertainty_notes.",
            "## Resultado",
            f"Amostras: {n_samples}. Pacotes A/B: {n_packets}. Formularios: {len(forms)}.",
            "## Guardrails",
            "Nenhuma resposta final e preenchida; apenas placeholders "
            "<TO_BE_FILLED_BY_HUMAN_REVIEWER>. dino_validates_event=false. Nenhum label.",
        ],
    )
    print(f"[v1qw] samples={n_samples} packets={n_packets} forms={len(forms)}")
    return {"samples": n_samples, "packets": n_packets, "forms": len(forms)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qw double review packets").parse_args()
    run()
