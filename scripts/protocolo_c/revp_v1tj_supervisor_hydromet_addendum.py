"""REV-P v1tj — Supervisor hydromet addendum.

Attaches hydromet context to supervisor review packets (v1rj), or emits
standalone waiting entries when no supervisor packet exists.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1tg_v1tm_hydromet_review_integration_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
    guardrail_row_extended, scan_guardrails_extended,
    hash_short, parse_float_safe,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_ADD  = _p("REVP_V1TJ_OUT_ADD",  DATASETS / "protocol_c_supervisor_hydromet_addendum_v1tj.csv")
OUT_SUM  = _p("REVP_V1TJ_OUT_SUM",  DATASETS / "protocol_c_supervisor_hydromet_addendum_summary_v1tj.csv")
SCHEMA_A = _p("REVP_V1TJ_SCHEMA_A", SCHEMAS  / "protocol_c_supervisor_hydromet_addendum_v1tj_schema.csv")
SCHEMA_S = _p("REVP_V1TJ_SCHEMA_S", SCHEMAS  / "protocol_c_supervisor_hydromet_addendum_summary_v1tj_schema.csv")
DOC      = _p("REVP_V1TJ_DOC",      DOCS     / "revp_v1tj_supervisor_hydromet_addendum.md")

ADD_FIELDS = [
    "supervisor_addendum_id", "supervisor_packet_id", "event_candidate_id",
    "region", "hydromet_packet_id", "nearest_station_code",
    "nearest_station_distance_km", "rain_7d", "hydromet_support_level",
    "supervisor_notes_prompt",
    "supports_manual_review", "does_not_validate_event",
    "independent_observational_source_still_required",
    "review_only", "hydromet_validates_event", "hydromet_is_negative_evidence",
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "formal_negative", "absence_as_negative", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def run() -> dict[str, Any]:
    sup_manifest = read_csv_safe(DATASETS / "protocol_c_supervisor_review_packet_manifest_v1rj.csv")
    packets      = read_csv_safe(DATASETS / "protocol_c_hydromet_evidence_packet_registry_v1tg.csv")
    scores       = read_csv_safe(DATASETS / "protocol_c_hydromet_review_scores_v1ti.csv")

    # Index supervisor packets by event_id
    sup_by_event: dict[str, list[dict]] = {}
    for r in sup_manifest:
        eid = r.get("event_id", "") or r.get("event_candidate_id", "")
        sup_by_event.setdefault(eid, []).append(r)

    # Index scores by event_candidate_id
    scores_by_cid: dict[str, dict] = {r.get("event_candidate_id",""): r for r in scores}

    rows: list[dict[str, Any]] = []
    attached = 0
    standalone = 0

    for pkt in packets:
        cid    = pkt.get("event_candidate_id", "")
        region = pkt.get("region", "")
        pid    = pkt.get("hydromet_packet_id", "")
        code   = pkt.get("nearest_station_code", "")
        dist   = pkt.get("nearest_station_distance_km", "")
        r7d    = pkt.get("rain_7d", "")
        lvl    = pkt.get("hydromet_support_level", "")
        score  = scores_by_cid.get(cid, {})

        # Build supervisor prompt
        overclaim = parse_float_safe(score.get("overclaim_risk_score", "0.5"), 0.5)
        prompt = (
            f"Contexto INMET: rain_7d={r7d}mm, estacao={code} (~{dist}km). "
            f"overclaim_risk={overclaim:.1f}. "
            "Confirmar que evidencia hidromet e contextual apenas e que "
            "fonte observacional independente e necessaria."
        )

        sup_pkts = sup_by_event.get(cid, [])
        if not sup_pkts:
            # Try partial match
            for eid, pkts in sup_by_event.items():
                if cid in eid or eid in cid:
                    sup_pkts = pkts
                    break

        if sup_pkts:
            for sp in sup_pkts:
                row: dict[str, Any] = {
                    "supervisor_addendum_id":    f"V1TJ_ADD_{hash_short(cid+sp.get('packet_id',''),10)}",
                    "supervisor_packet_id":      sp.get("packet_id", ""),
                    "event_candidate_id":        cid,
                    "region":                    region,
                    "hydromet_packet_id":        pid,
                    "nearest_station_code":      code,
                    "nearest_station_distance_km": dist,
                    "rain_7d":                   r7d,
                    "hydromet_support_level":    lvl,
                    "supervisor_notes_prompt":   prompt,
                    "supports_manual_review":    "true",
                    "does_not_validate_event":   "true",
                    "independent_observational_source_still_required": "true",
                    "notes": "",
                }
                row.update(guardrail_row_extended())
                rows.append(row)
            attached += 1
        else:
            row = {
                "supervisor_addendum_id":    f"V1TJ_ADD_{hash_short(cid+'standalone',10)}",
                "supervisor_packet_id":      "",
                "event_candidate_id":        cid,
                "region":                    region,
                "hydromet_packet_id":        pid,
                "nearest_station_code":      code,
                "nearest_station_distance_km": dist,
                "rain_7d":                   r7d,
                "hydromet_support_level":    lvl,
                "supervisor_notes_prompt":   prompt,
                "supports_manual_review":    "true",
                "does_not_validate_event":   "true",
                "independent_observational_source_still_required": "true",
                "notes": "no supervisor packet — standalone waiting",
            }
            row.update(guardrail_row_extended())
            rows.append(row)
            standalone += 1

    if not rows:
        rows = [{
            "supervisor_addendum_id": "FAIL_CLOSED_NO_PACKETS",
            "supervisor_packet_id": "", "event_candidate_id": "",
            "region": "", "hydromet_packet_id": "",
            "nearest_station_code": "", "nearest_station_distance_km": "",
            "rain_7d": "", "hydromet_support_level": "HYDROMET_CONTEXT_WAITING_EVENT_WINDOW",
            "supervisor_notes_prompt": "",
            "supports_manual_review": "true",
            "does_not_validate_event": "true",
            "independent_observational_source_still_required": "true",
            "notes": "", **guardrail_row_extended(),
        }]

    violations = scan_guardrails_extended(rows, "v1tj_addendum")
    if violations:
        raise ValueError(f"Guardrail violations in v1tj: {violations[:3]}")

    write_csv_with_header(OUT_ADD, rows, ADD_FIELDS)
    write_schema(SCHEMA_A, ADD_FIELDS, "v1tj_addendum")

    req_ind = sum(1 for r in rows if r.get("independent_observational_source_still_required") == "true")
    summary = [
        {"stat_key": "addenda_total",                   "stat_value": str(len(rows))},
        {"stat_key": "attached_to_supervisor",          "stat_value": str(attached)},
        {"stat_key": "standalone_waiting",              "stat_value": str(standalone)},
        {"stat_key": "independent_source_required",     "stat_value": str(req_ind)},
        {"stat_key": "validates_event",                 "stat_value": "false"},
        {"stat_key": "stage",                           "stat_value": "v1tj"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1tj_summary")

    write_doc(DOC, "v1tj — Supervisor Hydromet Addendum", [
        "## Objetivo",
        "Anexo com contexto INMET para pacotes do supervisor (v1rj). "
        "Quando sem match, emite standalone waiting.",
        f"## Resultado\nAddenda: {len(rows)}. Standalone: {standalone}. "
        f"independent_source_required: {req_ind}.",
        "## Regra",
        "does_not_validate_event=true. independent_observational_source_still_required=true.",
    ])
    print(f"[v1tj] addenda={len(rows)} attached={attached} standalone={standalone}")
    return {"addenda": len(rows), "attached": attached, "standalone": standalone}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1tj supervisor hydromet addendum").parse_args()
    run()
