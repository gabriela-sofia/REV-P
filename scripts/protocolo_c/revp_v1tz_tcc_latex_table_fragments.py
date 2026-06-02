"""REV-P v1tz — TCC LaTeX table fragments.

Emits LaTeX tabular fragments (as escaped text rows) from the final evidence
matrix, for direct inclusion in the TCC. Review-only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1tx_v1ub_tcc_dossier_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
    scan_guardrails, latex_table_row,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_FRG = _p("REVP_V1TZ_OUT_FRG", DATASETS / "protocol_c_tcc_latex_table_fragments_v1tz.csv")
OUT_SUM = _p("REVP_V1TZ_OUT_SUM", DATASETS / "protocol_c_tcc_latex_table_fragments_summary_v1tz.csv")
SCHEMA_F = _p("REVP_V1TZ_SCHEMA_F", SCHEMAS / "protocol_c_tcc_latex_table_fragments_v1tz_schema.csv")
SCHEMA_S = _p("REVP_V1TZ_SCHEMA_S", SCHEMAS / "protocol_c_tcc_latex_table_fragments_summary_v1tz_schema.csv")
DOC = _p("REVP_V1TZ_DOC", DOCS / "revp_v1tz_tcc_latex_table_fragments.md")

FRG_FIELDS = ["fragment_id", "table_key", "row_order", "latex_row",
              "review_only", "automated_review"]
SUM_FIELDS = ["stat_key", "stat_value"]


def run() -> dict[str, Any]:
    matrix = read_csv_safe(DATASETS / "protocol_c_final_evidence_matrix_v1ty.csv")
    matrix = [m for m in matrix if not m.get("case_id", "").startswith("FAIL_CLOSED")]

    rows: list[dict[str, Any]] = []
    order = 0

    header = latex_table_row(["Caso", "Regiao", "Externa", "Hidromet",
                              "DINO", "Patch", "Decisao review-only"])
    rows.append({"fragment_id": "V1TZ_HDR", "table_key": "evidence_matrix",
                 "row_order": str(order), "latex_row": header,
                 "review_only": "true", "automated_review": "true"})
    order += 1
    for m in matrix:
        rows.append({
            "fragment_id": f"V1TZ_{order:04d}", "table_key": "evidence_matrix",
            "row_order": str(order),
            "latex_row": latex_table_row([
                m.get("case_id", ""), m.get("region", ""),
                m.get("external_present", ""), m.get("hydromet_context", ""),
                m.get("dino_context", ""), m.get("patch_link", ""),
                m.get("supervisor_decision", ""),
            ]),
            "review_only": "true", "automated_review": "true",
        })
        order += 1

    caption = (r"\caption{Matriz de evidencia review-only (revisao automatizada). "
               r"DINO e hidromet sao contexto; ausencia nao e negativo; sem "
               r"promocao automatica de C3 e C4 fechado.}")
    rows.append({"fragment_id": "V1TZ_CAP", "table_key": "evidence_matrix",
                 "row_order": str(order), "latex_row": caption,
                 "review_only": "true", "automated_review": "true"})

    if len(rows) == 1:
        rows.append({"fragment_id": "V1TZ_EMPTY", "table_key": "evidence_matrix",
                     "row_order": "1", "latex_row": latex_table_row(["(sem casos)"]),
                     "review_only": "true", "automated_review": "true"})

    viol = scan_guardrails(rows, "v1tz")
    if viol:
        raise ValueError(f"Guardrail violations v1tz: {viol[:3]}")

    write_csv_with_header(OUT_FRG, rows, FRG_FIELDS)
    write_schema(SCHEMA_F, FRG_FIELDS, "v1tz_fragments")

    summary = [
        {"stat_key": "fragment_rows", "stat_value": str(len(rows))},
        {"stat_key": "matrix_cases", "stat_value": str(len(matrix))},
        {"stat_key": "stage", "stat_value": "v1tz"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1tz_summary")

    write_doc(DOC, "v1tz — TCC LaTeX Table Fragments", [
        "## Objetivo",
        "Fragmentos LaTeX (tabular) a partir da matriz final de evidencia, prontos "
        "para inclusao no TCC.",
        f"## Resultado\nLinhas de fragmento: {len(rows)}.",
        "## Limitacao",
        "Conteudo review-only; nao expressa evento validado nem ground truth.",
    ])
    print(f"[v1tz] fragment_rows={len(rows)} cases={len(matrix)}")
    return {"fragment_rows": len(rows), "cases": len(matrix)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1tz latex fragments").parse_args()
    run()
