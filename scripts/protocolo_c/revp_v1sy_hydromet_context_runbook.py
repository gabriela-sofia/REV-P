"""REV-P v1sy — Hydrometeorological context runbook.

Generates a methodology doc explaining how INMET/ANA data serves as context
(not validation) for Protocol C events. No computation; doc-only output.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1sr_v1sz_hydromet_context_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    write_csv_with_header, write_schema, write_doc,
    read_csv_safe,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_SUM = _p("REVP_V1SY_OUT_SUM", DATASETS / "protocol_c_hydromet_context_runbook_summary_v1sy.csv")
SCHEMA_S = _p("REVP_V1SY_SCHEMA_S", SCHEMAS / "protocol_c_hydromet_context_runbook_summary_v1sy_schema.csv")
DOC = _p("REVP_V1SY_DOC", DOCS / "revp_v1sy_hydromet_context_runbook.md")

SUM_FIELDS = ["stat_key", "stat_value"]

_MANDATORY = (
    "A camada v1sr–v1sz usa dados hidrometeorológicos oficiais apenas como "
    "contexto temporal e regional review-only. Precipitação observada, "
    "proximidade de estação ou janela temporal compatível não validam "
    "automaticamente evento, não criam ground truth operacional, não criam "
    "negativo formal e não substituem revisão humana."
)


def run() -> dict[str, Any]:
    # Read summary stats from upstream stages for doc content
    sr_sum = {r["stat_key"]: r["stat_value"] for r in
              read_csv_safe(DATASETS / "protocol_c_inmet_station_region_proximity_summary_v1sr.csv")}
    ss_sum = {r["stat_key"]: r["stat_value"] for r in
              read_csv_safe(DATASETS / "protocol_c_event_date_windows_summary_v1ss.csv")}
    st_sum = {r["stat_key"]: r["stat_value"] for r in
              read_csv_safe(DATASETS / "protocol_c_inmet_precipitation_event_window_summary_v1st.csv")}
    sx_sum = {r["stat_key"]: r["stat_value"] for r in
              read_csv_safe(DATASETS / "protocol_c_hydromet_context_guardrail_summary_v1sx.csv")}

    n_stations   = sr_sum.get("stations_within_100km", "0")
    n_windows    = ss_sum.get("windows_total", "0")
    n_ctx        = st_sum.get("context_rows", "0")
    guardrail_st = sx_sum.get("audit_status", "UNKNOWN")

    doc_sections = [
        "## Declaracao obrigatoria",
        _MANDATORY,
        "## Como usar dados INMET/ANA como contexto",
        "1. **Identificar estações próximas** (v1sr): usar somente estações dentro "
        "de 100 km da região de interesse como contexto espacial plausível.",
        "2. **Construir janelas temporais** (v1ss): T-7 a T+1 relativas à data "
        "documentada do evento. Janela não implica causalidade.",
        "3. **Consultar precipitação no período** (v1st): valores de precipitação "
        "diária como descrição do contexto meteorológico. Ausência de precipitação "
        "não é evidência negativa de evento.",
        "4. **Calcular features contextuais** (v1su): acumulados 1d/3d/7d são "
        "descritivos; não podem ser usados como target ou label supervisionado.",
        "5. **Crosswalk de intake** (v1sv): para cada janela, criar entrada de "
        "intake manual no v1rb com todos os campos review-only.",
        "6. **Revisão humana obrigatória**: qualquer interpretação de causalidade "
        "entre precipitação e evento requer revisão por especialista.",
        "## O que não fazer",
        "- Não tratar compatibilidade de precipitação como validação de evento.",
        "- Não tratar ausência de precipitação como prova de que o evento não ocorreu.",
        "- Não usar features de precipitação como target supervisionado.",
        "- Não abrir C4 (análise exploratória supervisionada) com estes dados.",
        "- Não citar DINO como validação de contexto.",
        "## Status do pipeline",
        f"- Estações INMET dentro de 100km: {n_stations}",
        f"- Janelas de evento construídas: {n_windows}",
        f"- Linhas de contexto de precipitação: {n_ctx}",
        f"- Guardrail audit status: {guardrail_st}",
        "## Fontes",
        "INMET — Instituto Nacional de Meteorologia. Dados históricos automáticos. "
        "Fonte pública oficial. Verificar licença antes de publicação.",
    ]

    write_doc(DOC, "v1sy — Hydrometeorological Context Runbook", doc_sections)

    summary = [
        {"stat_key": "runbook_emitted",      "stat_value": "true"},
        {"stat_key": "mandatory_clause",     "stat_value": "INCLUDED"},
        {"stat_key": "stations_in_scope",    "stat_value": n_stations},
        {"stat_key": "event_windows",        "stat_value": n_windows},
        {"stat_key": "context_rows",         "stat_value": n_ctx},
        {"stat_key": "guardrail_status",     "stat_value": guardrail_st},
        {"stat_key": "stage",                "stat_value": "v1sy"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1sy_summary")

    print(f"[v1sy] runbook emitted stations={n_stations} windows={n_windows} guardrail={guardrail_st}")
    return {"runbook": str(DOC.name), "guardrail": guardrail_st}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1sy hydromet runbook").parse_args()
    run()
