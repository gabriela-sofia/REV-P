"""REV-P v1ra — External collection task board.

Builds a programmatic task board from v1qz external collection priorities.
Suggests search queries per region/source WITHOUT executing any internet
search. Review-only; never creates labels or targets.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1ra_v1rf_external_intake_common import (
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

IN_PRIORITIES = _p("REVP_V1RA_IN_PRIORITIES", DATASETS / "protocol_c_ground_reference_external_collection_priorities_v1qz.csv")
OUT_BOARD = _p("REVP_V1RA_OUT_BOARD", DATASETS / "protocol_c_external_collection_task_board_v1ra.csv")
OUT_SUMMARY = _p("REVP_V1RA_OUT_SUMMARY", DATASETS / "protocol_c_external_collection_task_summary_v1ra.csv")
SCHEMA_BOARD = _p("REVP_V1RA_SCHEMA_BOARD", SCHEMAS / "protocol_c_external_collection_task_board_v1ra_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1RA_SCHEMA_SUMMARY", SCHEMAS / "protocol_c_external_collection_task_summary_v1ra_schema.csv")
DOC = _p("REVP_V1RA_DOC", DOCS / "revp_v1ra_external_collection_task_board.md")

BOARD_FIELDS = [
    "task_id", "region", "source_name", "source_family", "evidence_need",
    "search_query_suggested", "expected_document_type", "priority",
    "blocks_c3", "blocks_c4", "assigned_status", "collection_status",
    "review_only", "notes",
]

SUMMARY_FIELDS = ["stat_key", "stat_value"]

_REGION_TERM = {"RECIFE": "Recife", "PET": "Petropolis", "CURITIBA": "Curitiba"}

# evidence_need -> (query_template_suffix, expected_document_type)
_NEED_QUERY = {
    "rainfall_intensity": ("precipitacao extrema data evento", "SPREADSHEET"),
    "rainfall_alert": ("alerta chuva risco data", "PDF_REPORT"),
    "river_level_discharge": ("estacao nivel vazao serie", "SPREADSHEET"),
    "field_occurrence_record": ("ocorrencia boletim alagamento deslizamento", "PDF_REPORT"),
    "mass_movement_mapping": ("movimento de massa relatorio pos-desastre", "PDF_REPORT"),
    "institutional_recognition": ("situacao de emergencia decreto reconhecimento", "OFFICIAL_GAZETTE"),
    "territorial_base": ("limites territoriais setor municipio", "GEOSPATIAL"),
}


def build_query(region: str, source_name: str, evidence_need: str) -> str:
    region_term = _REGION_TERM.get(region, region.title())
    suffix = _NEED_QUERY.get(evidence_need, (evidence_need.replace("_", " "), ""))[0]
    # source short token: keep first word group before slash
    src_token = source_name.split("(")[0].split("/")[0].strip()
    return f"{src_token} {region_term} {suffix}".strip()


def build_rows(priorities: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, p in enumerate(priorities):
        region = p.get("region", "")
        need = p.get("evidence_need", "")
        source_name = p.get("preferred_source_name", "")
        query = build_query(region, source_name, need)
        expected = _NEED_QUERY.get(need, ("", "UNKNOWN_DOCUMENT_TYPE"))[1] or "UNKNOWN_DOCUMENT_TYPE"
        row = {
            "task_id": f"V1RA_TASK_{i:04d}",
            "region": region,
            "source_name": source_name,
            "source_family": p.get("preferred_source_family", ""),
            "evidence_need": need,
            "search_query_suggested": query,
            "expected_document_type": expected,
            "priority": p.get("source_priority", ""),
            "blocks_c3": p.get("blocks_c3", "false"),
            "blocks_c4": p.get("blocks_c4", "false"),
            "assigned_status": "UNASSIGNED",
            "collection_status": "PENDING_MANUAL_COLLECTION",
            "notes": "no_internet_query_suggestion_only",
        }
        row.update(guardrail_row())
        rows.append(row)
    return rows


def run(datasets: Path | None = None) -> dict[str, Any]:
    priorities = read_csv_safe(IN_PRIORITIES)
    rows = build_rows(priorities)
    assert_clean_rows(rows, "v1ra_board")

    write_csv_with_header(OUT_BOARD, rows, BOARD_FIELDS)
    write_schema_safe(SCHEMA_BOARD, BOARD_FIELDS, "v1ra_task_board")

    by_region: dict[str, int] = {}
    by_priority: dict[str, int] = {}
    for r in rows:
        by_region[r["region"]] = by_region.get(r["region"], 0) + 1
        by_priority[r["priority"]] = by_priority.get(r["priority"], 0) + 1

    summary = [
        {"stat_key": "total_tasks", "stat_value": str(len(rows))},
        {"stat_key": "tasks_blocking_c3", "stat_value": str(sum(1 for r in rows if r["blocks_c3"] == "true"))},
    ]
    for region, n in sorted(by_region.items()):
        summary.append({"stat_key": f"region_{region.lower()}", "stat_value": str(n)})
    for prio, n in sorted(by_priority.items()):
        summary.append({"stat_key": f"priority_{prio.lower()}", "stat_value": str(n)})
    summary.append({"stat_key": "collection_mode", "stat_value": "MANUAL_NO_INTERNET"})
    summary.append({"stat_key": "stage", "stat_value": "v1ra"})
    write_csv_with_header(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    write_schema_safe(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1ra_summary")

    write_doc(
        DOC,
        "v1ra — External Collection Task Board",
        [
            "## Objetivo",
            "Construir um quadro de tarefas de coleta externa a partir das prioridades "
            "v1qz. Sugere queries de busca por regiao/fonte sem executar internet.",
            "## Queries sugeridas (exemplos)",
            "CEMADEN Recife alerta chuva risco; Defesa Civil Recife ocorrencia boletim; "
            "INMET Recife precipitacao extrema; ANA HidroWeb estacao Recife nivel; "
            "SGB CPRM Petropolis movimento de massa relatorio; "
            "Diario Oficial situacao de emergencia Petropolis.",
            "## Resultado",
            f"Total de tarefas: {len(rows)}.",
            "## Guardrails",
            "Nenhuma busca executada (no_internet). collection_status=PENDING_MANUAL_COLLECTION. "
            "review_only=true. Nenhum label/target/ground truth.",
        ],
    )
    print(f"[v1ra] tasks={len(rows)}")
    return {"tasks": len(rows)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1ra external collection task board").parse_args()
    run()
