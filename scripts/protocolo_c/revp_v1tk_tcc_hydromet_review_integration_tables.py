"""REV-P v1tk — TCC hydromet review integration tables.

Three TCC-ready tables: review packets, supervisor addendum summary, and
overclaim controls. Makes explicit that INMET data is contextual only and
that C3 still requires independent observational evidence.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1tg_v1tm_hydromet_review_integration_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_PKT = _p("REVP_V1TK_OUT_PKT", DATASETS / "protocol_c_tcc_table_hydromet_review_packets_v1tk.csv")
OUT_SUP = _p("REVP_V1TK_OUT_SUP", DATASETS / "protocol_c_tcc_table_hydromet_supervisor_addendum_v1tk.csv")
OUT_OC  = _p("REVP_V1TK_OUT_OC",  DATASETS / "protocol_c_tcc_table_hydromet_overclaim_controls_v1tk.csv")
SCHEMA_P= _p("REVP_V1TK_SCHEMA_P",SCHEMAS  / "protocol_c_tcc_table_hydromet_review_packets_v1tk_schema.csv")
SCHEMA_S= _p("REVP_V1TK_SCHEMA_S",SCHEMAS  / "protocol_c_tcc_table_hydromet_supervisor_addendum_v1tk_schema.csv")
SCHEMA_O= _p("REVP_V1TK_SCHEMA_O",SCHEMAS  / "protocol_c_tcc_table_hydromet_overclaim_controls_v1tk_schema.csv")
DOC     = _p("REVP_V1TK_DOC",     DOCS     / "revp_v1tk_tcc_hydromet_review_integration_tables.md")

MET_FIELDS = ["metric", "value", "interpretation_note"]
OC_FIELDS  = ["control_id", "aspect", "description", "implication"]


def _stat(rows: list[dict[str, str]], key: str, default: str = "0") -> str:
    for r in rows:
        if r.get("stat_key") == key:
            return r.get("stat_value", default)
    return default


def run() -> dict[str, Any]:
    tg_sum = read_csv_safe(DATASETS / "protocol_c_hydromet_evidence_packet_summary_v1tg.csv")
    th_sum = read_csv_safe(DATASETS / "protocol_c_hydromet_double_review_addendum_summary_v1th.csv")
    ti_sum = read_csv_safe(DATASETS / "protocol_c_hydromet_review_scoring_summary_v1ti.csv")
    tj_sum = read_csv_safe(DATASETS / "protocol_c_supervisor_hydromet_addendum_summary_v1tj.csv")

    # Table 1: Review packets
    pkt_rows: list[dict[str, Any]] = [
        {"metric": "hydromet_packets",      "value": _stat(tg_sum,"total_packets"),
         "interpretation_note": "Pacotes de evidencia hidromet por evento candidato"},
        {"metric": "context_available",     "value": _stat(tg_sum,"context_available"),
         "interpretation_note": "Pacotes com contexto disponivel (estacao <100km + precip)"},
        {"metric": "double_review_addenda", "value": _stat(th_sum,"addenda_total"),
         "interpretation_note": "Addenda para revisao A/B com perguntas INMET"},
        {"metric": "review_forms",          "value": _stat(th_sum,"form_rows"),
         "interpretation_note": "Formularios de revisao hidromet gerados"},
        {"metric": "responses_empty",       "value": _stat(th_sum,"responses_empty"),
         "interpretation_note": "Respostas pendentes de preenchimento humano"},
        {"metric": "validates_event",       "value": "false",
         "interpretation_note": "Nenhum evento validado automaticamente"},
    ]
    write_csv_with_header(OUT_PKT, pkt_rows, MET_FIELDS)
    write_schema(SCHEMA_P, MET_FIELDS, "v1tk_packets")

    # Table 2: Supervisor addendum
    sup_rows: list[dict[str, Any]] = [
        {"metric": "supervisor_addenda",     "value": _stat(tj_sum,"addenda_total"),
         "interpretation_note": "Addenda gerados para supervisor"},
        {"metric": "attached_to_supervisor", "value": _stat(tj_sum,"attached_to_supervisor"),
         "interpretation_note": "Addenda vinculados a pacote supervisor existente"},
        {"metric": "standalone_waiting",     "value": _stat(tj_sum,"standalone_waiting"),
         "interpretation_note": "Addenda standalone aguardando pacote supervisor"},
        {"metric": "independent_source_required", "value": _stat(tj_sum,"independent_source_required"),
         "interpretation_note": "Addenda marcando necessidade de fonte observacional independente"},
        {"metric": "scoring_status",         "value": _stat(ti_sum,"overall_status"),
         "interpretation_note": "Estado atual do scoring contextual"},
    ]
    write_csv_with_header(OUT_SUP, sup_rows, MET_FIELDS)
    write_schema(SCHEMA_S, MET_FIELDS, "v1tk_supervisor")

    # Table 3: Overclaim controls
    oc_rows: list[dict[str, Any]] = [
        {"control_id": "OC01", "aspect": "precipitation_is_context",
         "description": "Chuva INMET registrada e contexto hidrometeorologico oficial, nao validacao de inundacao/deslizamento.",
         "implication": "Revisor nao pode usar rain_7d como criterio de validacao de evento."},
        {"control_id": "OC02", "aspect": "absence_is_not_negative",
         "description": "Ausencia ou baixa precipitacao em estacao INMET nao constitui negativo formal.",
         "implication": "C3 nao pode ser negado somente por falta de chuva INMET."},
        {"control_id": "OC03", "aspect": "station_proximity_is_not_validation",
         "description": "Estacao proxima nao prova que evento ocorreu naquele ponto especifico.",
         "implication": "Distancia de 5km nao substitui foto, boletim ou documento observacional."},
        {"control_id": "OC04", "aspect": "c3_requires_independent_source",
         "description": "Promocao para C3 exige fonte observacional independente alem do INMET.",
         "implication": "Contexto hidromet e necessario mas nao suficiente para C3."},
        {"control_id": "OC05", "aspect": "c4_not_opened_by_hydromet",
         "description": "C4 nao pode ser aberto com base em dados INMET.",
         "implication": "Abertura de C4 exige revisao metodologica separada e explicitamente aprovada."},
        {"control_id": "OC06", "aspect": "hydromet_does_not_replace_review",
         "description": "Addenda hidromet sao auxiliares — nao substituem revisao A/B nem supervisao.",
         "implication": "Respostas ao addendum devem ser preenchidas por revisor humano qualificado."},
    ]
    write_csv_with_header(OUT_OC, oc_rows, OC_FIELDS)
    write_schema(SCHEMA_O, OC_FIELDS, "v1tk_overclaim")

    write_doc(DOC, "v1tk — TCC Hydromet Review Integration Tables", [
        "## Objetivo",
        "Tabelas TCC: pacotes de revisao, addendum supervisor e controles de overclaim.",
        f"## Resultado\nPacotes: {_stat(tg_sum,'total_packets')}. "
        f"Addenda: {_stat(th_sum,'addenda_total')}. Overclaim controls: {len(oc_rows)}.",
    ])
    print(f"[v1tk] pkt={len(pkt_rows)} sup={len(sup_rows)} overclaim_controls={len(oc_rows)}")
    return {"pkt_rows": len(pkt_rows), "sup_rows": len(sup_rows), "oc_rows": len(oc_rows)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1tk TCC review tables").parse_args()
    run()
