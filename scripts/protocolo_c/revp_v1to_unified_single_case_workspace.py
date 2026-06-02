"""REV-P v1to — Unified single-case workspace.

Renders one readable workspace per case so a reviewer can read a case without
opening dozens of technical CSVs. Sections are emitted as rows. Review-only.
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

OUT_WS   = _p("REVP_V1TO_OUT_WS",   DATASETS / "protocol_c_unified_single_case_workspace_v1to.csv")
OUT_SEC  = _p("REVP_V1TO_OUT_SEC",  DATASETS / "protocol_c_unified_single_case_workspace_sections_v1to.csv")
OUT_SUM  = _p("REVP_V1TO_OUT_SUM",  DATASETS / "protocol_c_unified_single_case_workspace_summary_v1to.csv")
SCHEMA_W = _p("REVP_V1TO_SCHEMA_W", SCHEMAS  / "protocol_c_unified_single_case_workspace_v1to_schema.csv")
SCHEMA_E = _p("REVP_V1TO_SCHEMA_E", SCHEMAS  / "protocol_c_unified_single_case_workspace_sections_v1to_schema.csv")
SCHEMA_S = _p("REVP_V1TO_SCHEMA_S", SCHEMAS  / "protocol_c_unified_single_case_workspace_summary_v1to_schema.csv")
DOC      = _p("REVP_V1TO_DOC",      DOCS     / "revp_v1to_unified_single_case_workspace.md")

WS_FIELDS = [
    "case_id", "region", "hazard_type", "event_window",
    "case_summary", "external_evidence_summary", "hydromet_summary",
    "dino_role", "patch_event_link", "blockers", "overclaim_risks",
    "next_steps", "review_only_status",
    "review_only", "automated_review",
    "internal_review_automated_for_review_only",
    "requires_external_observational_evidence_for_operational_claim",
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "formal_negative",
    "automatic_c3_promotion", "c4_opened",
    "dino_validates_event", "hydromet_validates_event",
    "hydromet_is_negative_evidence", "absence_as_negative", "notes",
]
SEC_FIELDS = ["case_id", "section_order", "section_key", "section_text",
              "review_only", "automated_review"]
SUM_FIELDS = ["stat_key", "stat_value"]

SECTION_KEYS = [
    "case_summary", "external_evidence_summary", "hydromet_summary",
    "dino_role", "patch_event_link", "blockers", "overclaim_risks",
    "next_steps", "review_only_status",
]


def run() -> dict[str, Any]:
    cases = read_csv_safe(DATASETS / "protocol_c_unified_evidence_case_index_v1tn.csv")

    ws_rows: list[dict[str, Any]] = []
    sec_rows: list[dict[str, Any]] = []

    for c in cases:
        cid = c.get("case_id", "")
        if cid.startswith("FAIL_CLOSED"):
            continue
        region = c.get("region", "")
        hazard = c.get("hazard_type", "")
        window = c.get("event_window", "")
        ext = c.get("external_evidence_status", "")
        hyd = c.get("hydromet_status", "")
        dino = c.get("dino_status", "")
        patch = c.get("patch_link_status", "")
        readiness = c.get("case_readiness_status", "")
        blockers = c.get("blocking_factors", "")
        nxt = c.get("next_required_action", "")

        case_summary = (f"Caso {cid} ({region}, {hazard}) janela {window}. "
                        f"Prontidão: {readiness}.")
        ext_summary = f"Fonte externa: {ext}. Exigida p/ afirmação operacional."
        hyd_summary = f"Hidromet (INMET): {hyd} — contexto, não validação."
        dino_role = f"DINO: {dino} — representação visual review-only, nunca prova."
        patch_link = f"Patch/evento: {patch}."
        overclaim = ("Risco de overclaim se hidromet/DINO forem tratados como prova; "
                     "ausência de chuva não é negativo.")
        next_steps = f"Próxima ação: {nxt}."
        ro_status = f"Status review-only: {readiness} (sem C3 automático, sem C4)."

        sections = {
            "case_summary": case_summary,
            "external_evidence_summary": ext_summary,
            "hydromet_summary": hyd_summary,
            "dino_role": dino_role,
            "patch_event_link": patch_link,
            "blockers": f"Blockers: {blockers}.",
            "overclaim_risks": overclaim,
            "next_steps": next_steps,
            "review_only_status": ro_status,
        }

        row: dict[str, Any] = {
            "case_id": cid, "region": region, "hazard_type": hazard,
            "event_window": window,
            "case_summary": case_summary,
            "external_evidence_summary": ext_summary,
            "hydromet_summary": hyd_summary,
            "dino_role": dino_role,
            "patch_event_link": patch_link,
            "blockers": f"Blockers: {blockers}.",
            "overclaim_risks": overclaim,
            "next_steps": next_steps,
            "review_only_status": ro_status,
            "notes": "",
        }
        row.update(guardrail_row_review())
        ws_rows.append(row)

        for i, key in enumerate(SECTION_KEYS):
            sec_rows.append({
                "case_id": cid, "section_order": str(i), "section_key": key,
                "section_text": sections[key],
                "review_only": "true", "automated_review": "true",
            })

    if not ws_rows:
        ws_rows = [{
            "case_id": "FAIL_CLOSED_NO_CASES", "region": "", "hazard_type": "",
            "event_window": "", "case_summary": "no cases",
            "external_evidence_summary": "", "hydromet_summary": "",
            "dino_role": "", "patch_event_link": "", "blockers": "",
            "overclaim_risks": "", "next_steps": "", "review_only_status": "",
            "notes": "no v1tn input", **guardrail_row_review(),
        }]
        sec_rows = [{
            "case_id": "FAIL_CLOSED_NO_CASES", "section_order": "0",
            "section_key": "case_summary", "section_text": "no cases",
            "review_only": "true", "automated_review": "true",
        }]

    for label, rws in (("v1to_ws", ws_rows), ("v1to_sec", sec_rows)):
        viol = scan_guardrails(rws, label)
        if viol:
            raise ValueError(f"Guardrail violations {label}: {viol[:3]}")

    write_csv_with_header(OUT_WS, ws_rows, WS_FIELDS)
    write_csv_with_header(OUT_SEC, sec_rows, SEC_FIELDS)
    write_schema(SCHEMA_W, WS_FIELDS, "v1to_workspace")
    write_schema(SCHEMA_E, SEC_FIELDS, "v1to_sections")

    summary = [
        {"stat_key": "workspaces_total", "stat_value": str(len(ws_rows))},
        {"stat_key": "sections_total",   "stat_value": str(len(sec_rows))},
        {"stat_key": "sections_per_case","stat_value": str(len(SECTION_KEYS))},
        {"stat_key": "stage",            "stat_value": "v1to"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1to_summary")

    write_doc(DOC, "v1to — Unified Single-Case Workspace", [
        "## Objetivo",
        "Workspace única e legível por caso (resumo, evidência externa, hidromet, "
        "papel do DINO, patch/evento, blockers, riscos de overclaim, próximos "
        "passos e status review-only).",
        f"## Resultado\nWorkspaces: {len(ws_rows)}. Seções: {len(sec_rows)}.",
        "## Limitação",
        "Permite ler um caso sem abrir dezenas de CSVs. Não cria label/target/"
        "ground truth, não promove C3, não abre C4, não usa hidromet/DINO como prova.",
    ])
    print(f"[v1to] workspaces={len(ws_rows)} sections={len(sec_rows)}")
    return {"workspaces": len(ws_rows), "sections": len(sec_rows)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1to single-case workspace").parse_args()
    run()
