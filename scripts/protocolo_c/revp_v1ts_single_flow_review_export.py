"""REV-P v1ts — Single-flow review export.

Produces one ordered, readable flow per case (header, evidence, hydromet, DINO
limitation, Reviewer A/B, consensus, supervisor, blockers, next action, claim
safety, TCC-ready summary). This is the artefact that replaces reading dozens
of technical CSVs. Review-only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1tn_v1tw_automated_review_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
    guardrail_row_review, scan_guardrails,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_FLOW = _p("REVP_V1TS_OUT_FLOW", DATASETS / "protocol_c_single_flow_review_export_v1ts.csv")
OUT_SEC  = _p("REVP_V1TS_OUT_SEC",  DATASETS / "protocol_c_single_flow_review_export_sections_v1ts.csv")
OUT_SUM  = _p("REVP_V1TS_OUT_SUM",  DATASETS / "protocol_c_single_flow_review_export_summary_v1ts.csv")
SCHEMA_F = _p("REVP_V1TS_SCHEMA_F", SCHEMAS  / "protocol_c_single_flow_review_export_v1ts_schema.csv")
SCHEMA_E = _p("REVP_V1TS_SCHEMA_E", SCHEMAS  / "protocol_c_single_flow_review_export_sections_v1ts_schema.csv")
SCHEMA_S = _p("REVP_V1TS_SCHEMA_S", SCHEMAS  / "protocol_c_single_flow_review_export_summary_v1ts_schema.csv")
DOC      = _p("REVP_V1TS_DOC",      DOCS     / "revp_v1ts_single_flow_review_export.md")

SECTION_KEYS = [
    "case_header", "evidence_summary", "hydromet_summary", "dino_limitation",
    "reviewer_a_decision", "reviewer_b_decision", "consensus_divergence",
    "supervisor_adjudication", "blockers", "next_action", "claim_safety_status",
    "tcc_ready_summary",
]

FLOW_FIELDS = [
    "case_id", "region", "hazard_type", "event_window",
    "case_header", "evidence_summary", "hydromet_summary", "dino_limitation",
    "reviewer_a_decision", "reviewer_b_decision", "consensus_divergence",
    "supervisor_adjudication", "blockers", "next_action",
    "claim_safety_status", "tcc_ready_summary",
    "review_only", "automated_review",
    "internal_review_automated_for_review_only",
    "requires_external_observational_evidence_for_operational_claim",
    "automatic_c3_promotion", "c4_opened",
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "formal_negative",
    "dino_validates_event", "hydromet_validates_event",
    "hydromet_is_negative_evidence", "absence_as_negative", "notes",
]
SEC_FIELDS = ["case_id", "section_order", "section_key", "section_text",
              "review_only", "automated_review"]
SUM_FIELDS = ["stat_key", "stat_value"]


def run() -> dict[str, Any]:
    cases = read_csv_safe(DATASETS / "protocol_c_unified_evidence_case_index_v1tn.csv")
    decisions = read_csv_safe(DATASETS / "protocol_c_automated_reviewer_ab_decisions_v1tp.csv")
    consensus = read_csv_safe(DATASETS / "protocol_c_review_consensus_divergence_adjudication_v1tq.csv")
    supervisor = read_csv_safe(DATASETS / "protocol_c_automated_supervisor_adjudication_v1tr.csv")

    dec_by_case: dict[str, dict[str, dict[str, str]]] = {}
    for d in decisions:
        dec_by_case.setdefault(d.get("case_id", ""), {})[d.get("reviewer_slot", "")] = d
    con_by_case = {c.get("case_id", ""): c for c in consensus}
    sup_by_case = {s.get("case_id", ""): s for s in supervisor}

    flow_rows: list[dict[str, Any]] = []
    sec_rows: list[dict[str, Any]] = []

    for c in cases:
        cid = c.get("case_id", "")
        if cid.startswith("FAIL_CLOSED"):
            continue
        region = c.get("region", "")
        hazard = c.get("hazard_type", "")
        window = c.get("event_window", "")
        a = dec_by_case.get(cid, {}).get("A", {})
        b = dec_by_case.get(cid, {}).get("B", {})
        con = con_by_case.get(cid, {})
        sup = sup_by_case.get(cid, {})

        sup_decision = sup.get("supervisor_decision", "N/A")
        ready_tcc = sup.get("ready_for_tcc_discussion", "false")

        sections = {
            "case_header": f"Caso {cid} | {region} | {hazard} | janela {window}.",
            "evidence_summary": (
                f"Externa: {c.get('external_evidence_status','')}; "
                f"Patch: {c.get('patch_link_status','')}; "
                f"Estado Protocolo C: {c.get('protocol_c_state','')}."),
            "hydromet_summary": (
                f"Hidromet: {c.get('hydromet_status','')} — contexto, não validação."),
            "dino_limitation": (
                f"DINO: {c.get('dino_status','')} — representação review-only, nunca prova."),
            "reviewer_a_decision": (
                f"Reviewer A (conservador): {a.get('recommended_review_only_status','N/A')} "
                f"(conf {a.get('review_only_confidence_score','')}, "
                f"overclaim {a.get('overclaim_risk','')})."),
            "reviewer_b_decision": (
                f"Reviewer B (integrador): {b.get('recommended_review_only_status','N/A')} "
                f"(conf {b.get('review_only_confidence_score','')}, "
                f"overclaim {b.get('overclaim_risk','')})."),
            "consensus_divergence": (
                f"Consenso: {con.get('consensus_status','N/A')} "
                f"(divergência {con.get('divergence_type','NONE')})."),
            "supervisor_adjudication": (
                f"Supervisor automatizado: {sup_decision} "
                f"(final review-only {sup.get('final_for_review_only_use','false')}; "
                f"validação operacional false)."),
            "blockers": f"Blockers: {c.get('blocking_factors','')}.",
            "next_action": f"Próxima ação: {c.get('next_required_action','')}.",
            "claim_safety_status": (
                "REVIEW_ONLY_SAFE: sem C3 automático, sem C4, sem ground truth, "
                "sem label/target/negativo formal; fonte externa exigida p/ operacional."),
            "tcc_ready_summary": (
                f"Pronto para discussão no TCC: {ready_tcc} "
                f"(uso review-only, não operacional)."),
        }

        row: dict[str, Any] = {
            "case_id": cid, "region": region, "hazard_type": hazard,
            "event_window": window, "notes": "",
        }
        row.update(sections)
        row.update(guardrail_row_review())
        flow_rows.append(row)

        for i, key in enumerate(SECTION_KEYS):
            sec_rows.append({
                "case_id": cid, "section_order": str(i), "section_key": key,
                "section_text": sections[key],
                "review_only": "true", "automated_review": "true",
            })

    if not flow_rows:
        flow_rows = [{
            "case_id": "FAIL_CLOSED_NO_CASES", "region": "", "hazard_type": "",
            "event_window": "", "notes": "no inputs",
            **{k: "" for k in SECTION_KEYS}, **guardrail_row_review(),
        }]
        sec_rows = [{
            "case_id": "FAIL_CLOSED_NO_CASES", "section_order": "0",
            "section_key": "case_header", "section_text": "no inputs",
            "review_only": "true", "automated_review": "true",
        }]

    for label, rws in (("v1ts_flow", flow_rows), ("v1ts_sec", sec_rows)):
        viol = scan_guardrails(rws, label)
        if viol:
            raise ValueError(f"Guardrail violations {label}: {viol[:3]}")

    write_csv_with_header(OUT_FLOW, flow_rows, FLOW_FIELDS)
    write_csv_with_header(OUT_SEC, sec_rows, SEC_FIELDS)
    write_schema(SCHEMA_F, FLOW_FIELDS, "v1ts_flow")
    write_schema(SCHEMA_E, SEC_FIELDS, "v1ts_sections")

    summary = [
        {"stat_key": "flow_rows",        "stat_value": str(len(flow_rows))},
        {"stat_key": "section_rows",     "stat_value": str(len(sec_rows))},
        {"stat_key": "sections_per_case","stat_value": str(len(SECTION_KEYS))},
        {"stat_key": "stage",            "stat_value": "v1ts"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1ts_summary")

    write_doc(DOC, "v1ts — Single-Flow Review Export", [
        "## Objetivo",
        "Fluxo único, ordenado e legível por caso: header, evidência, hidromet, "
        "limitação do DINO, Reviewer A/B, consenso/divergência, supervisor, "
        "blockers, próxima ação, claim safety e resumo pronto-para-TCC.",
        f"## Resultado\nFluxos: {len(flow_rows)}. Seções: {len(sec_rows)} "
        f"({len(SECTION_KEYS)} por caso).",
        "## Limitação",
        "Substitui a leitura de dezenas de CSVs técnicos. Não cria label/target/"
        "ground truth, não promove C3, não abre C4, não usa hidromet/DINO como prova.",
    ])
    print(f"[v1ts] flow={len(flow_rows)} sections={len(sec_rows)}")
    return {"flow": len(flow_rows), "sections": len(sec_rows)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1ts single-flow review export").parse_args()
    run()
