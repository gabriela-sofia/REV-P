"""REV-P v1te — TCC hydromet correction and evidence tables.

Generates three TCC-ready tables documenting: (1) the coordinate correction
from v1si→v1ta, (2) hydromet evidence bridge summary, (3) methodological
limitations including the parse-bug provenance and contextual-only use.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1ta_v1tf_inmet_canonical_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_COR = _p("REVP_V1TE_OUT_COR", DATASETS / "protocol_c_tcc_table_inmet_coordinate_correction_v1te.csv")
OUT_EVB = _p("REVP_V1TE_OUT_EVB", DATASETS / "protocol_c_tcc_table_hydromet_evidence_bridge_v1te.csv")
OUT_LIM = _p("REVP_V1TE_OUT_LIM", DATASETS / "protocol_c_tcc_table_hydromet_limitations_v1te.csv")
SCHEMA_C = _p("REVP_V1TE_SCHEMA_C", SCHEMAS / "protocol_c_tcc_table_inmet_coordinate_correction_v1te_schema.csv")
SCHEMA_E = _p("REVP_V1TE_SCHEMA_E", SCHEMAS / "protocol_c_tcc_table_hydromet_evidence_bridge_v1te_schema.csv")
SCHEMA_L = _p("REVP_V1TE_SCHEMA_L", SCHEMAS / "protocol_c_tcc_table_hydromet_limitations_v1te_schema.csv")
DOC      = _p("REVP_V1TE_DOC",      DOCS    / "revp_v1te_tcc_hydromet_correction_evidence_tables.md")

COR_FIELDS = ["metric", "value", "interpretation_note"]
EVB_FIELDS = ["metric", "value", "interpretation_note"]
LIM_FIELDS = ["limitation_id", "aspect", "description", "implication", "notes"]


def _stat(rows: list[dict[str, str]], key: str, default: str = "0") -> str:
    for r in rows:
        if r.get("stat_key") == key:
            return r.get("stat_value", default)
    return default


def run() -> dict[str, Any]:
    tb_sum = read_csv_safe(DATASETS / "protocol_c_inmet_coordinate_parse_discrepancy_summary_v1tb.csv")
    ta_sum = read_csv_safe(DATASETS / "protocol_c_inmet_canonical_station_registry_summary_v1ta.csv")
    td_sum = read_csv_safe(DATASETS / "protocol_c_hydromet_event_evidence_bridge_summary_v1td.csv")
    tc_sum = read_csv_safe(DATASETS / "protocol_c_inmet_canonical_precipitation_index_summary_v1tc.csv")

    # Table 1: Coordinate correction
    cor_rows: list[dict[str, Any]] = [
        {"metric": "stations_in_v1si",        "value": _stat(tb_sum, "stations_compared"),
         "interpretation_note": "Estações extraídas pelo parser v1si (coordenadas quebradas)"},
        {"metric": "stations_corrected_v1ta", "value": _stat(tb_sum, "corrected_in_v1ta"),
         "interpretation_note": "Estações com coordenadas corrigidas em v1ta (decimal-vírgula→ponto)"},
        {"metric": "canonical_total",         "value": _stat(ta_sum, "canonical_stations_total"),
         "interpretation_note": "Total no registry canônico v1ta"},
        {"metric": "within_100km_canonical",  "value": _stat(ta_sum, "within_100km"),
         "interpretation_note": "Estações canônicas dentro de 100km das regiões-alvo"},
        {"metric": "region_matching_affected", "value": _stat(tb_sum, "affects_region_matching"),
         "interpretation_note": "Estações com matching de região afetado pelo bug de parse"},
        {"metric": "v1si_not_modified",        "value": "true",
         "interpretation_note": "v1si preservado intacto; correção apenas em v1ta"},
    ]
    write_csv_with_header(OUT_COR, cor_rows, COR_FIELDS)
    write_schema(SCHEMA_C, COR_FIELDS, "v1te_correction")

    # Table 2: Evidence bridge
    evb_rows: list[dict[str, Any]] = [
        {"metric": "event_windows",           "value": _stat(td_sum, "bridge_rows"),
         "interpretation_note": "Janelas de evento com pacote de evidência hidromet"},
        {"metric": "rows_with_rain_data",     "value": _stat(td_sum, "rows_with_rain_data"),
         "interpretation_note": "Janelas com dados de precipitação canônica disponíveis"},
        {"metric": "canonical_precip_records","value": _stat(tc_sum, "total_daily_records"),
         "interpretation_note": "Registros diários de precipitação no índice canônico v1tc"},
        {"metric": "validates_event",         "value": "false",
         "interpretation_note": "Nenhum evento validado automaticamente"},
        {"metric": "absence_as_negative",     "value": "false",
         "interpretation_note": "Ausência de dados não é evidência negativa"},
    ]
    write_csv_with_header(OUT_EVB, evb_rows, EVB_FIELDS)
    write_schema(SCHEMA_E, EVB_FIELDS, "v1te_evidence_bridge")

    # Table 3: Limitations
    lim_rows: list[dict[str, Any]] = [
        {"limitation_id": "LIM_TE01", "aspect": "parse_bug_v1si",
         "description": "v1si usou parse sem conversão de vírgula decimal; coordenadas resultantes inválidas.",
         "implication": "Matching de região em v1si é não-confiável. Usar v1ta para análises downstream.",
         "notes": ""},
        {"limitation_id": "LIM_TE02", "aspect": "coordinate_correction",
         "description": "v1ta corrige via releitura dos ZIPs originais com replace(',','.').",
         "implication": "Correção é auditável; provenance preservado. v1si não alterado.",
         "notes": ""},
        {"limitation_id": "LIM_TE03", "aspect": "contextual_use_only",
         "description": "Precipitação INMET é usada como contexto temporal/regional, não como prova de evento.",
         "implication": "Compatibilidade de precipitação não valida deslizamento ou inundação.",
         "notes": ""},
        {"limitation_id": "LIM_TE04", "aspect": "absence_not_negative",
         "description": "Ausência de precipitação em estação próxima não constitui evidência negativa.",
         "implication": "Evento pode ter ocorrido por acumulados anteriores, chuva localizada ou outras causas.",
         "notes": ""},
        {"limitation_id": "LIM_TE05", "aspect": "hourly_to_daily_aggregation",
         "description": "Dados INMET são horários; v1tc agrega para diário.",
         "implication": "Eventos de curta duração (<1h) podem ter total diário subestimado.",
         "notes": ""},
        {"limitation_id": "LIM_TE06", "aspect": "station_density",
         "description": "Densidade de estações INMET é baixa em regiões serranas como Petrópolis.",
         "implication": "Estação mais próxima pode não representar condições no local exato do evento.",
         "notes": ""},
    ]
    write_csv_with_header(OUT_LIM, lim_rows, LIM_FIELDS)
    write_schema(SCHEMA_L, LIM_FIELDS, "v1te_limitations")

    write_doc(DOC, "v1te — TCC Hydromet Correction and Evidence Tables", [
        "## Objetivo",
        "Tabelas TCC-prontas: correção de coordenadas v1si→v1ta, resumo de "
        "evidência hidromet, e tabela de limitações metodológicas.",
        f"## Resultado\nCorreção: {len(cor_rows)} métricas. "
        f"Bridge: {len(evb_rows)} métricas. Limitações: {len(lim_rows)}.",
    ])
    print(f"[v1te] cor={len(cor_rows)} bridge={len(evb_rows)} limitations={len(lim_rows)}")
    return {"correction_rows": len(cor_rows), "bridge_rows": len(evb_rows),
            "limitation_rows": len(lim_rows)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1te TCC hydromet tables").parse_args()
    run()
