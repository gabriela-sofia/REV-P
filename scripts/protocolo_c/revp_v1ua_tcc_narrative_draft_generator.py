"""REV-P v1ua — TCC narrative draft generator.

Generates a Portuguese technical narrative draft per case plus a global
paragraph, from the automated review-only results. Review-only; never states a
validated event or operational ground truth.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1tx_v1ub_tcc_dossier_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
    guardrail_row_review, scan_guardrails, narrative_for_case,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_NAR = _p("REVP_V1UA_OUT_NAR", DATASETS / "protocol_c_tcc_narrative_draft_v1ua.csv")
OUT_SUM = _p("REVP_V1UA_OUT_SUM", DATASETS / "protocol_c_tcc_narrative_draft_summary_v1ua.csv")
SCHEMA_N = _p("REVP_V1UA_SCHEMA_N", SCHEMAS / "protocol_c_tcc_narrative_draft_v1ua_schema.csv")
SCHEMA_S = _p("REVP_V1UA_SCHEMA_S", SCHEMAS / "protocol_c_tcc_narrative_draft_summary_v1ua_schema.csv")
DOC = _p("REVP_V1UA_DOC", DOCS / "revp_v1ua_tcc_narrative_draft_generator.md")

NAR_FIELDS = ["narrative_id", "scope", "case_id", "narrative_text",
              "review_only", "automated_review",
              "internal_review_automated_for_review_only",
              "requires_external_observational_evidence_for_operational_claim",
              "automatic_c3_promotion", "c4_opened",
              "can_create_operational_label", "can_train_model", "target_created",
              "ground_truth_operational", "formal_negative",
              "dino_validates_event", "hydromet_validates_event",
              "hydromet_is_negative_evidence", "absence_as_negative", "notes"]
SUM_FIELDS = ["stat_key", "stat_value"]

GLOBAL_TEXT = (
    "Este trabalho consolidou os casos por revisao automatizada review-only, "
    "integrando contexto hidrometeorologico (INMET) e representacao estrutural "
    "DINO apenas como contexto. A camada organiza e adjudica internamente para "
    "fins metodologicos, sem promover C3 automaticamente (contagem zero), sem "
    "abrir C4, sem criar rotulos, alvos, referencia operacional confirmada ou "
    "negativos formais, e sem tratar ausencia como negativo. Afirmacoes "
    "operacionais permanecem condicionadas a fonte observacional independente.")


def run() -> dict[str, Any]:
    cases = read_csv_safe(DATASETS / "protocol_c_unified_evidence_case_index_v1tn.csv")
    sup = {r.get("case_id", ""): r for r in
           read_csv_safe(DATASETS / "protocol_c_automated_supervisor_adjudication_v1tr.csv")}

    rows: list[dict[str, Any]] = []
    rows.append({
        "narrative_id": "V1UA_GLOBAL", "scope": "global", "case_id": "",
        "narrative_text": GLOBAL_TEXT, "notes": "", **guardrail_row_review(),
    })
    for c in cases:
        cid = c.get("case_id", "")
        if cid.startswith("FAIL_CLOSED"):
            continue
        rows.append({
            "narrative_id": f"V1UA_{cid}", "scope": "case", "case_id": cid,
            "narrative_text": narrative_for_case(c, sup.get(cid, {})),
            "notes": "", **guardrail_row_review(),
        })

    viol = scan_guardrails(rows, "v1ua")
    if viol:
        raise ValueError(f"Guardrail violations v1ua: {viol[:3]}")

    write_csv_with_header(OUT_NAR, rows, NAR_FIELDS)
    write_schema(SCHEMA_N, NAR_FIELDS, "v1ua_narrative")

    case_rows = sum(1 for r in rows if r["scope"] == "case")
    summary = [
        {"stat_key": "narrative_rows", "stat_value": str(len(rows))},
        {"stat_key": "case_narratives", "stat_value": str(case_rows)},
        {"stat_key": "stage", "stat_value": "v1ua"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1ua_summary")

    write_doc(DOC, "v1ua — TCC Narrative Draft Generator", [
        "## Objetivo",
        "Gerar rascunho de narrativa tecnica em portugues por caso e um paragrafo "
        "global, a partir da revisao automatizada review-only.",
        f"## Resultado\nNarrativas: {len(rows)} (casos: {case_rows}).",
        "## Limitacao",
        "Nao afirma evento validado nem ground truth operacional. DINO/hidromet "
        "sao contexto; ausencia nao e negativo; C3 automatico = 0; C4 fechado.",
    ])
    print(f"[v1ua] narratives={len(rows)} case_narratives={case_rows}")
    return {"narratives": len(rows), "case_narratives": case_rows}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1ua narrative draft").parse_args()
    run()
