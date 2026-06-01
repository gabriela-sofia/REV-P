"""REV-P v1rb — External document intake template.

Emits a blank intake template (header-only) for documents collected MANUALLY,
plus a schema/config descriptor. No data, no internet, no labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1ra_v1rf_external_intake_common import (
    DATASETS,
    DOCS,
    INTAKE_FIELDS,
    _p,
    write_csv_with_header,
    write_doc,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"

OUT_TEMPLATE = _p("REVP_V1RB_OUT_TEMPLATE", DATASETS / "protocol_c_external_document_intake_template_v1rb.csv")
OUT_SCHEMA = _p("REVP_V1RB_OUT_SCHEMA", CONFIGS / "protocol_c_external_document_intake_schema_v1rb.csv")
DOC = _p("REVP_V1RB_DOC", DOCS / "revp_v1rb_external_document_intake_template.md")

SCHEMA_FIELDS = ["field", "required", "description", "example_placeholder"]

_FIELD_DESCRIPTIONS: dict[str, tuple[str, str, str]] = {
    "document_id": ("yes", "Identificador unico do documento coletado", "DOC_0001"),
    "source_name": ("yes", "Nome da fonte (instituicao/publicacao)", "Defesa Civil Petropolis"),
    "source_family": ("auto", "Familia de fonte (classificada automaticamente)", "OFFICIAL_CIVIL_DEFENSE"),
    "region": ("yes", "Regiao (RECIFE/PET/CURITIBA)", "PET"),
    "hazard_type": ("yes", "Tipo de ameaca", "LANDSLIDE"),
    "event_date_text": ("yes", "Data do evento (texto livre)", "15/02/2022"),
    "event_location_text": ("yes", "Local do evento (texto livre)", "Alto da Serra"),
    "url_or_reference": ("yes", "URL ou referencia (NAO sera baixada)", "https://exemplo.gov.br/doc"),
    "local_document_hash": ("optional", "Hash do documento local (sem path)", "abc123"),
    "access_date": ("yes", "Data de acesso/coleta", "2026-05-31"),
    "license_note": ("yes", "Nota de licenca/acesso", "dominio publico"),
    "evidence_type": ("yes", "Tipo de evidencia", "occurrence_bulletin"),
    "temporal_precision_claim": ("yes", "Precisao temporal alegada (DAY/MONTH/YEAR)", "DAY"),
    "spatial_precision_claim": ("yes", "Precisao espacial alegada (POINT/ADDRESS/ADMINISTRATIVE)", "ADDRESS"),
    "reviewer_notes": ("optional", "Notas do revisor", ""),
    "intake_status": ("auto", "Status do intake", "PENDING_VALIDATION"),
}


def run(datasets: Path | None = None) -> dict[str, Any]:
    # Blank template — header only, zero data rows
    write_csv_with_header(OUT_TEMPLATE, [], INTAKE_FIELDS)

    schema_rows = []
    for f in INTAKE_FIELDS:
        req, desc, example = _FIELD_DESCRIPTIONS.get(f, ("optional", f, ""))
        schema_rows.append({
            "field": f, "required": req, "description": desc,
            "example_placeholder": example,
        })
    write_csv_with_header(OUT_SCHEMA, schema_rows, SCHEMA_FIELDS)

    write_doc(
        DOC,
        "v1rb — External Document Intake Template",
        [
            "## Objetivo",
            "Fornecer um template em branco para inserir documentos externos coletados "
            "MANUALMENTE, mais um descritor de schema. Nenhum dado e preenchido; nenhuma "
            "URL e baixada.",
            "## Como usar",
            "1) Preencher o template com documentos coletados. 2) Apontar "
            "REVP_PROTOCOL_C_EXTERNAL_INTAKE_PATH para o CSV preenchido. 3) Rodar v1rc "
            "para validar.",
            "## Campos",
            "document_id, source_name, source_family, region, hazard_type, event_date_text, "
            "event_location_text, url_or_reference, local_document_hash, access_date, "
            "license_note, evidence_type, temporal_precision_claim, spatial_precision_claim, "
            "reviewer_notes, intake_status.",
            "## Guardrails",
            "Template review-only. Nao baixa nada. Nao cria label, target ou ground truth. "
            "url_or_reference nunca e acessada automaticamente.",
        ],
    )
    print(f"[v1rb] template_fields={len(INTAKE_FIELDS)} schema_rows={len(schema_rows)}")
    return {"fields": len(INTAKE_FIELDS)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1rb intake template").parse_args()
    run()
