"""REV-P v1sk — Institutional document discovery queue.

For sources without a robust direct endpoint (CEMADEN, SGB, Defesa Civil,
Diário Oficial), creates a safe discovery/download queue using existing
source requirements and backlog. No broad scraping. Review-only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1sg_v1sz_official_download_common import (
    DATASETS, DOCS, SCHEMAS, _p, guardrail_row, write_csv_with_header,
    write_doc, write_schema_for, read_csv_safe, is_allowed_domain,
    domain_from_url, classify_source_family, forbidden_guardrail_scan,
)

ROOT = Path(__file__).resolve().parents[2]

IN_REQUIREMENTS = _p("REVP_V1SK_IN_REQUIREMENTS", DATASETS / "protocol_c_official_evidence_source_requirements_v1qu.csv")
IN_BACKLOG = _p("REVP_V1SK_IN_BACKLOG", DATASETS / "protocol_c_ground_reference_evidence_backlog_v1ro.csv")
IN_ENDPOINTS = _p("REVP_V1SK_IN_ENDPOINTS", DATASETS / "protocol_c_official_source_endpoint_registry_v1sg.csv")

OUT_QUEUE = _p("REVP_V1SK_OUT_QUEUE", DATASETS / "protocol_c_institutional_document_discovery_queue_v1sk.csv")
OUT_SUMMARY = _p("REVP_V1SK_OUT_SUMMARY", DATASETS / "protocol_c_institutional_document_discovery_summary_v1sk.csv")
SCHEMA_Q = _p("REVP_V1SK_SCHEMA_Q", SCHEMAS / "protocol_c_institutional_document_discovery_queue_v1sk_schema.csv")
SCHEMA_S = _p("REVP_V1SK_SCHEMA_S", SCHEMAS / "protocol_c_institutional_document_discovery_summary_v1sk_schema.csv")
DOC = _p("REVP_V1SK_DOC", DOCS / "revp_v1sk_institutional_document_discovery_queue.md")

QUEUE_FIELDS = [
    "discovery_id", "region", "source_name", "source_family", "hazard_type",
    "evidence_need", "suggested_query", "candidate_url", "domain",
    "allowed_domain", "auto_download_allowed", "requires_manual_review",
    "priority", "blocks_c3", "blocks_c4", "review_only", "blocked_reason", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]

_QUERIES = [
    ("RECIFE", "CEMADEN", "CEMADEN Recife alerta chuva inundacao pluviometria", "https://www.gov.br/cemaden/pt-br"),
    ("RECIFE", "Defesa Civil PE", "Defesa Civil Recife alagamento ocorrencia boletim", ""),
    ("RECIFE", "Diario Oficial PE", "Diario Oficial Pernambuco situacao de emergencia Recife", ""),
    ("PET", "SGB CPRM", "SGB CPRM Petropolis movimento de massa 2022 relatorio pos-desastre", "https://rigeo.sgb.gov.br/"),
    ("PET", "Defesa Civil RJ", "Defesa Civil Petropolis deslizamento boletim ocorrencia", ""),
    ("PET", "Diario Oficial RJ", "Diario Oficial Rio de Janeiro situacao de emergencia Petropolis", ""),
    ("PET", "CEMADEN", "CEMADEN Petropolis alerta chuva deslizamento", "https://www.gov.br/cemaden/pt-br"),
    ("CURITIBA", "Defesa Civil PR", "Defesa Civil Curitiba inundacao alagamento boletim", ""),
    ("CURITIBA", "CEMADEN", "CEMADEN Curitiba alerta chuva risco hidrologico", "https://www.gov.br/cemaden/pt-br"),
    ("CURITIBA", "Diario Oficial PR", "Diario Oficial Parana situacao de emergencia Curitiba", ""),
]


def _requirements_extras(requirements: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Generate extra discovery rows from unfulfilled requirements."""
    extras: list[dict[str, Any]] = []
    for r in requirements:
        if r.get("collection_status") != "SOURCE_REQUIRED_NOT_LOCAL":
            continue
        name = r.get("preferred_source_name", "")
        family = classify_source_family(name)
        if family in ("OFFICIAL_HYDROMETEOROLOGICAL",):
            continue  # handled by v1sh/v1sj
        region = r.get("region", "")
        need = r.get("evidence_need", "")
        row = {
            "discovery_id": "", "region": region, "source_name": name,
            "source_family": family, "hazard_type": r.get("hazard_type", ""),
            "evidence_need": need,
            "suggested_query": f"{name} {region} {need.replace('_', ' ')}",
            "candidate_url": "", "domain": "", "allowed_domain": "NOT_CONFIGURED",
            "auto_download_allowed": "false", "requires_manual_review": "true",
            "priority": r.get("source_priority", "P2"),
            "blocks_c3": r.get("blocks_c3", "false"),
            "blocks_c4": r.get("blocks_c4", "false"),
            "review_only": "true", "blocked_reason": "ENDPOINT_CONFIG_REQUIRED",
            "notes": "from_v1qu_requirements",
        }
        extras.append(row)
    return extras


def run(datasets: Path | None = None) -> dict[str, Any]:
    requirements = read_csv_safe(IN_REQUIREMENTS)
    rows: list[dict[str, Any]] = []

    for i, (region, source, query, url) in enumerate(_QUERIES):
        domain = domain_from_url(url) if url else ""
        allowed = "true" if (url and is_allowed_domain(url)) else ("false" if url else "NOT_CONFIGURED")
        row = {
            "discovery_id": f"V1SK_D{i:03d}", "region": region,
            "source_name": source, "source_family": classify_source_family(source),
            "hazard_type": "FLOOD" if region in ("RECIFE", "CURITIBA") else "LANDSLIDE",
            "evidence_need": "institutional_document",
            "suggested_query": query, "candidate_url": url, "domain": domain,
            "allowed_domain": allowed, "auto_download_allowed": "false",
            "requires_manual_review": "true",
            "priority": "P0" if region in ("RECIFE", "PET") else "P1",
            "blocks_c3": "true", "blocks_c4": "false",
            "review_only": "true", "blocked_reason": "" if url else "ENDPOINT_CONFIG_REQUIRED",
            "notes": "",
        }
        rows.append(row)

    extras = _requirements_extras(requirements)
    for j, ex in enumerate(extras):
        ex["discovery_id"] = f"V1SK_D{len(rows) + j:03d}"
        rows.append(ex)

    write_csv_with_header(OUT_QUEUE, rows, QUEUE_FIELDS)
    write_schema_for(SCHEMA_Q, QUEUE_FIELDS, "v1sk_queue")

    manual = sum(1 for r in rows if r["requires_manual_review"] == "true")
    summary = [
        {"stat_key": "discovery_items", "stat_value": str(len(rows))},
        {"stat_key": "manual_review_required", "stat_value": str(manual)},
        {"stat_key": "auto_download_possible", "stat_value": str(len(rows) - manual)},
        {"stat_key": "stage", "stat_value": "v1sk"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUM_FIELDS)
    write_schema_for(SCHEMA_S, SUM_FIELDS, "v1sk_summary")

    write_doc(DOC, "v1sk — Institutional Document Discovery Queue", [
        "## Objetivo",
        "Criar fila de descoberta/download para fontes institucionais sem endpoint robusto "
        "(CEMADEN, SGB, Defesa Civil, Diario Oficial). Nao faz scraping amplo.",
        "## Resultado",
        f"Itens de discovery: {len(rows)}. Manual review: {manual}.",
    ])
    print(f"[v1sk] discovery={len(rows)} manual={manual}")
    return {"discovery": len(rows), "manual": manual}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1sk institutional discovery queue").parse_args()
    run()
