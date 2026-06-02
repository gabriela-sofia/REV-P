"""REV-P v1tn — Unified evidence case index.

One case per event candidate/region. Consolidates external, hydromet, DINO
(review-only), patch-link and Protocol C state. Computes readiness, blockers
and next action. Never decides C3/C4.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1tn_v1tw_automated_review_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
    guardrail_row_review, scan_guardrails, hash_short,
    normalize_region, normalize_case_id,
    summarize_external_evidence, summarize_hydromet_evidence,
    summarize_dino_role, summarize_patch_context, summarize_protocol_c_state,
    classify_case_readiness, next_required_action, blocking_factors,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_IDX  = _p("REVP_V1TN_OUT_IDX",  DATASETS / "protocol_c_unified_evidence_case_index_v1tn.csv")
OUT_SUM  = _p("REVP_V1TN_OUT_SUM",  DATASETS / "protocol_c_unified_evidence_case_index_summary_v1tn.csv")
SCHEMA_I = _p("REVP_V1TN_SCHEMA_I", SCHEMAS  / "protocol_c_unified_evidence_case_index_v1tn_schema.csv")
SCHEMA_S = _p("REVP_V1TN_SCHEMA_S", SCHEMAS  / "protocol_c_unified_evidence_case_index_summary_v1tn_schema.csv")
DOC      = _p("REVP_V1TN_DOC",      DOCS     / "revp_v1tn_unified_evidence_case_index.md")

IDX_FIELDS = [
    "case_id", "event_candidate_id", "event_id", "patch_id", "region",
    "hazard_type", "event_window", "external_evidence_status", "hydromet_status",
    "dino_status", "patch_link_status", "protocol_c_state",
    "case_readiness_status", "blocking_factors", "next_required_action",
    "review_only", "automated_review",
    "automatic_c3_promotion", "c4_opened",
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "formal_negative",
    "internal_review_automated_for_review_only",
    "requires_external_observational_evidence_for_operational_claim",
    "dino_validates_event", "hydromet_validates_event",
    "hydromet_is_negative_evidence", "absence_as_negative", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def _index(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        out.setdefault(r.get(key, ""), []).append(r)
    return out


def run() -> dict[str, Any]:
    packets   = read_csv_safe(DATASETS / "protocol_c_hydromet_evidence_packet_registry_v1tg.csv")
    windows   = read_csv_safe(DATASETS / "protocol_c_event_date_windows_v1ss.csv")
    candidates= read_csv_safe(DATASETS / "protocol_c_external_event_candidates_v1rd.csv")
    links     = read_csv_safe(DATASETS / "protocol_c_external_event_patch_link_candidates_v1re.csv")
    dino      = read_csv_safe(DATASETS / "protocol_c_dino_review_only_representation_queue_v1oz.csv")
    backlog   = read_csv_safe(DATASETS / "protocol_c_ground_reference_evidence_backlog_v1ro.csv")

    packet_by_cand = {p.get("event_candidate_id", ""): p for p in packets}
    window_by_cand = {w.get("event_candidate_id", ""): w
                      for w in windows if not w.get("blocked_reason")}
    cand_by_id     = _index(candidates, "event_candidate_id")
    links_by_cand  = _index(links, "event_candidate_id")
    dino_by_cand   = _index(dino, "event_candidate_id")

    candidate_ids: list[str] = []
    seen: set[str] = set()
    for src in (packets, windows, candidates):
        for r in src:
            cid = r.get("event_candidate_id", "")
            if cid and cid not in seen:
                seen.add(cid)
                candidate_ids.append(cid)

    rows: list[dict[str, Any]] = []
    for cid in candidate_ids:
        pkt = packet_by_cand.get(cid)
        win = window_by_cand.get(cid)
        region = normalize_region(
            (pkt or win or {}).get("region", "")
            or (cand_by_id.get(cid, [{}])[0]).get("region", ""))
        hazard = (win or pkt or {}).get("hazard_type", "") \
            or (cand_by_id.get(cid, [{}])[0]).get("hazard_type", "")
        event_window = (pkt or {}).get("event_window", "") or (
            f"{win.get('window_start','')} to {win.get('window_end','')}" if win else "")
        temporal_status = (win or {}).get("temporal_precision_status", "")

        ext_status   = summarize_external_evidence(cand_by_id.get(cid, []))
        hyd_status   = summarize_hydromet_evidence(pkt)
        dino_status  = summarize_dino_role(dino_by_cand.get(cid, []))
        patch_status = summarize_patch_context(links_by_cand.get(cid, []))
        state        = summarize_protocol_c_state(region, backlog)

        readiness = classify_case_readiness(
            ext_status, hyd_status, event_window, temporal_status, patch_status)

        patch_id = ""
        if links_by_cand.get(cid):
            patch_id = links_by_cand[cid][0].get("patch_id", "")

        row: dict[str, Any] = {
            "case_id":                  normalize_case_id(f"CASE_{region}_{hash_short(cid, 10)}"),
            "event_candidate_id":       cid,
            "event_id":                 cid,
            "patch_id":                 patch_id,
            "region":                   region,
            "hazard_type":              hazard,
            "event_window":             event_window,
            "external_evidence_status": ext_status,
            "hydromet_status":          hyd_status,
            "dino_status":              dino_status,
            "patch_link_status":        patch_status,
            "protocol_c_state":         state,
            "case_readiness_status":    readiness,
            "blocking_factors":         blocking_factors(
                ext_status, hyd_status, event_window, temporal_status, patch_status),
            "next_required_action":     next_required_action(readiness),
            "notes":                    "",
        }
        row.update(guardrail_row_review())
        rows.append(row)

    if not rows:
        rows = [{
            "case_id": "FAIL_CLOSED_NO_CASES", "event_candidate_id": "",
            "event_id": "", "patch_id": "", "region": "", "hazard_type": "",
            "event_window": "", "external_evidence_status": "EXTERNAL_SOURCE_ABSENT_LOCAL",
            "hydromet_status": "HYDROMET_CONTEXT_ABSENT",
            "dino_status": "DINO_NOT_PRESENT_CONTEXT_ONLY",
            "patch_link_status": "PATCH_LINK_ABSENT",
            "protocol_c_state": "PROTOCOL_C_NO_BACKLOG_RECORD",
            "case_readiness_status": "CASE_BLOCKED_INSUFFICIENT_EVIDENCE",
            "blocking_factors": "NO_INPUTS",
            "next_required_action": "GATHER_MINIMUM_EVIDENCE_BEFORE_REVIEW",
            "notes": "no inputs", **guardrail_row_review(),
        }]

    viol = scan_guardrails(rows, "v1tn")
    if viol:
        raise ValueError(f"Guardrail violations v1tn: {viol[:3]}")

    write_csv_with_header(OUT_IDX, rows, IDX_FIELDS)
    write_schema(SCHEMA_I, IDX_FIELDS, "v1tn_case_index")

    ready = sum(1 for r in rows if r["case_readiness_status"]
                == "CASE_READY_FOR_REVIEW_ONLY_ADJUDICATION")
    ctx = sum(1 for r in rows if r["case_readiness_status"]
              == "CASE_CONTEXT_AVAILABLE_NEEDS_EXTERNAL_SOURCE")
    blocked = sum(1 for r in rows if r["case_readiness_status"]
                  == "CASE_BLOCKED_INSUFFICIENT_EVIDENCE")
    summary = [
        {"stat_key": "cases_total",                   "stat_value": str(len(rows))},
        {"stat_key": "cases_ready_for_adjudication",  "stat_value": str(ready)},
        {"stat_key": "cases_context_needs_external",  "stat_value": str(ctx)},
        {"stat_key": "cases_blocked",                 "stat_value": str(blocked)},
        {"stat_key": "automatic_c3_promotion",        "stat_value": "false"},
        {"stat_key": "c4_opened",                     "stat_value": "false"},
        {"stat_key": "stage",                         "stat_value": "v1tn"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1tn_summary")

    write_doc(DOC, "v1tn — Unified Evidence Case Index", [
        "## Objetivo",
        "Consolidar, por caso (evento/candidato/região), as evidências externa, "
        "hidromet, DINO review-only, patch-link e estado Protocolo C, com status "
        "de prontidão, blockers e próxima ação.",
        f"## Resultado\nCasos: {len(rows)}. Contexto disponível precisando de "
        f"fonte externa: {ctx}. Bloqueados: {blocked}.",
        "## Limitação",
        "Índice não decide C3/C4, não cria label/target/ground truth e não trata "
        "hidromet/DINO como prova. Fonte observacional externa segue exigida para "
        "afirmação operacional.",
    ])
    print(f"[v1tn] cases={len(rows)} ctx_needs_external={ctx} blocked={blocked}")
    return {"cases": len(rows), "ready": ready, "ctx": ctx, "blocked": blocked}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1tn unified evidence case index").parse_args()
    run()
