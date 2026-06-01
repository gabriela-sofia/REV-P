"""REV-P v1rx — Manual evidence collection runbook generator.

Generates the runbook for manually collecting external evidence to fill P1
(intake) and P2 (double-review responses). No science, no labels, no internet.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1rs_v1rz_integration_common import (
    DOCS, SCHEMAS, _p,
    write_csv_with_header, write_doc, write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"

OUT_RUNBOOK = _p("REVP_V1RX_OUT_RUNBOOK", DOCS / "revp_v1rx_manual_evidence_collection_runbook.md")
OUT_EV_CHECKLIST = _p("REVP_V1RX_OUT_EV_CHECKLIST", CONFIGS / "revp_external_evidence_collection_checklist_v1rx.csv")
OUT_RESP_CHECKLIST = _p("REVP_V1RX_OUT_RESP_CHECKLIST", CONFIGS / "revp_review_response_completion_checklist_v1rx.csv")
SCHEMA_EV = _p("REVP_V1RX_SCHEMA_EV", SCHEMAS / "revp_external_evidence_collection_checklist_v1rx_schema.csv")
SCHEMA_RESP = _p("REVP_V1RX_SCHEMA_RESP", SCHEMAS / "revp_review_response_completion_checklist_v1rx_schema.csv")

EV_FIELDS = ["source_id", "source_name", "source_family", "region", "priority",
             "evidence_type", "access_method", "collection_action",
             "intake_template", "review_only", "notes"]
RESP_FIELDS = ["step_id", "phase", "description", "script_to_run",
               "template_to_fill", "expected_output", "notes"]

_SOURCES = [
    ("SRC01", "CEMADEN", "OFFICIAL_HYDROMETEOROLOGICAL", "ALL", "P0",
     "hydromet_alert_pluviometry", "web/API", "Search alerts for event date/location", "v1rb intake template", "true", "cemaden.gov.br"),
    ("SRC02", "ANA / HidroWeb", "OFFICIAL_HYDROMETEOROLOGICAL", "RECIFE;CURITIBA", "P0",
     "river_level_discharge_series", "web/API", "Download station series for event dates", "v1rb intake template", "true", "hidroweb.ana.gov.br"),
    ("SRC03", "INMET / BDMEP", "OFFICIAL_HYDROMETEOROLOGICAL", "ALL", "P0",
     "rainfall_hourly_daily", "web/API", "Download precipitation series", "v1rb intake template", "true", "bdmep.inmet.gov.br"),
    ("SRC04", "SGB / CPRM", "OFFICIAL_GEOLOGICAL", "PET", "P0",
     "mass_movement_post_disaster_report", "web/download", "Search for post-disaster reports", "v1rb intake template", "true", "sgb.gov.br/rigeo"),
    ("SRC05", "Defesa Civil municipal/estadual", "OFFICIAL_CIVIL_DEFENSE", "ALL", "P0",
     "occurrence_bulletin_affected_areas", "web/email/LAI", "Request occurrence records for events", "v1rb intake template", "true", ""),
    ("SRC06", "Diário Oficial", "OFFICIAL_GOVERNMENT_PUBLICATION", "ALL", "P1",
     "decree_emergency_recognition", "web", "Search for emergency decrees for event dates", "v1rb intake template", "true", ""),
    ("SRC07", "Relatórios técnicos/artigos", "TECHNICAL_REPORT", "ALL", "P1",
     "technical_secondary_evidence", "web/academic", "Collect only when documented source", "v1rb intake template", "true", "Only if documented"),
    ("SRC08", "Mídia jornalística", "NEWS_MEDIA_SECONDARY", "ALL", "P2",
     "secondary_evidence_only", "web", "Secondary evidence; never sufficient alone for C3", "v1rb intake template", "true", "Never alone for C3"),
    ("SRC09", "Redes sociais", "SOCIAL_MEDIA_SECONDARY", "ALL", "P3",
     "triage_only", "web", "Triagem apenas; evidência fraca", "v1rb intake template", "true", "Triage only"),
]

_STEPS = [
    ("RS01", "COLLECT", "Collect official document for event/date/location",
     "manual web search", "v1rb intake template", "Document reference saved", ""),
    ("RS02", "REGISTER", "Record source metadata in intake template (v1rb)",
     "manual edit", "datasets/protocol_c_external_document_intake_template_v1rb.csv",
     "Row added with all required fields", ""),
    ("RS03", "VALIDATE", "Validate intake with v1rc",
     "python scripts/protocolo_c/revp_v1rc_external_document_intake_validator.py",
     "—", "EXTERNAL_INTAKE_VALIDATION_PASS_REVIEW_ONLY", ""),
    ("RS04", "CANDIDATES", "Generate event candidates from intake (v1rd)",
     "python scripts/protocolo_c/revp_v1rd_event_candidate_builder_from_external_intake.py",
     "—", "Event candidates review-only created", ""),
    ("RS05", "LINK", "Link event candidates to patches (v1re)",
     "python scripts/protocolo_c/revp_v1re_external_event_patch_candidate_linker.py",
     "—", "Link candidates review-only", ""),
    ("RS06", "FILL_AB", "Fill A/B review responses using v1rg template",
     "manual edit", "datasets/protocol_c_review_response_intake_template_v1rg.csv",
     "Responses filled for all samples", "Use pseudonym; no PII"),
    ("RS07", "VALIDATE_RESP", "Validate A/B responses with v1rh",
     "python scripts/protocolo_c/revp_v1rh_review_response_validator.py",
     "—", "REVIEW_RESPONSES_VALIDATION_PASS_REVIEW_ONLY", ""),
    ("RS08", "SCORE", "Score completed double-review with v1ri",
     "python scripts/protocolo_c/revp_v1ri_completed_review_scoring_replay.py",
     "—", "Review scores computed", ""),
    ("RS09", "SUPERVISOR", "Generate supervisor packet with v1rj",
     "python scripts/protocolo_c/revp_v1rj_supervisor_review_packet_generator.py",
     "—", "Supervisor packets ready if C3 candidate", ""),
    ("RS10", "SUP_FILL", "Fill supervisor decision using v1rk template",
     "manual edit", "datasets/protocol_c_supervisor_decision_intake_template_v1rk.csv",
     "Decision filled", ""),
    ("RS11", "SUP_VALIDATE", "Validate supervisor decision with v1rl",
     "python scripts/protocolo_c/revp_v1rl_supervisor_decision_validator.py",
     "—", "SUPERVISOR_DECISIONS_VALIDATION_PASS_REVIEW_ONLY", ""),
    ("RS12", "BUNDLE", "Consolidate with v1rm and v1rr",
     "python scripts/protocolo_c/revp_v1rm_review_supervisor_gate_bundle.py && "
     "python scripts/protocolo_c/revp_v1rr_scientific_roadmap_bundle.py",
     "—", "Final status updated", ""),
]


def run(datasets: Path | None = None) -> dict[str, Any]:
    ev_rows = [
        {"source_id": s[0], "source_name": s[1], "source_family": s[2],
         "region": s[3], "priority": s[4], "evidence_type": s[5],
         "access_method": s[6], "collection_action": s[7],
         "intake_template": s[8], "review_only": s[9], "notes": s[10]}
        for s in _SOURCES
    ]
    resp_rows = [
        {"step_id": s[0], "phase": s[1], "description": s[2],
         "script_to_run": s[3], "template_to_fill": s[4],
         "expected_output": s[5], "notes": s[6]}
        for s in _STEPS
    ]

    write_csv_with_header(OUT_EV_CHECKLIST, ev_rows, EV_FIELDS)
    write_csv_with_header(OUT_RESP_CHECKLIST, resp_rows, RESP_FIELDS)
    write_schema_safe(SCHEMA_EV, EV_FIELDS, "v1rx_evidence_checklist")
    write_schema_safe(SCHEMA_RESP, RESP_FIELDS, "v1rx_response_checklist")

    write_doc(OUT_RUNBOOK, "v1rx — Manual Evidence Collection Runbook", [
        "## Objetivo",
        "Guia passo a passo para coletar documentos externos (P1) e preencher respostas "
        "de revisão A/B + decisão supervisora (P2).",
        "## Fontes prioritárias",
        "| ID | Fonte | Família | Regiões | Prioridade |",
        "| -- | ----- | ------- | ------- | ---------- |",
        *[f"| {s[0]} | {s[1]} | {s[2]} | {s[3]} | {s[4]} |" for s in _SOURCES],
        "## Fluxo completo",
        *[f"**{s[0]}** [{s[1]}] {s[2]}\n- Script: `{s[3]}`\n- Template: {s[4]}\n- Esperado: {s[5]}" for s in _STEPS],
        "## Regras",
        "1. Mídia jornalística é secundária: nunca suficiente sozinha para C3.",
        "2. Redes sociais apenas para triagem: evidência fraca.",
        "3. Nunca usar ausência de evidência como negativo formal.",
        "4. DINO pode priorizar revisão, mas nunca valida evento.",
        "5. Toda decisão C3 exige supervisor humano.",
    ])

    print(f"[v1rx] sources={len(ev_rows)} steps={len(resp_rows)}")
    return {"sources": len(ev_rows), "steps": len(resp_rows)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1rx evidence collection runbook").parse_args()
    run()
