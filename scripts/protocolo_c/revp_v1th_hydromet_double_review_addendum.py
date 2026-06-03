"""REV-P v1th — Hydromet addendum for double review packets.

Generates hydromet-specific review questions as an addendum to existing
v1qw packets, or standalone when no v1qw match exists. Answer slots are
always empty — filled by reviewer only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1tg_v1tm_hydromet_review_integration_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
    guardrail_row_extended, scan_guardrails_extended,
    hash_short, build_hydromet_question_set,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_MAN  = _p("REVP_V1TH_OUT_MAN",  DATASETS / "protocol_c_hydromet_double_review_addendum_manifest_v1th.csv")
OUT_FORM = _p("REVP_V1TH_OUT_FORM", DATASETS / "protocol_c_hydromet_double_review_addendum_forms_v1th.csv")
OUT_SUM  = _p("REVP_V1TH_OUT_SUM",  DATASETS / "protocol_c_hydromet_double_review_addendum_summary_v1th.csv")
SCHEMA_M = _p("REVP_V1TH_SCHEMA_M", SCHEMAS  / "protocol_c_hydromet_double_review_addendum_manifest_v1th_schema.csv")
SCHEMA_F = _p("REVP_V1TH_SCHEMA_F", SCHEMAS  / "protocol_c_hydromet_double_review_addendum_forms_v1th_schema.csv")
SCHEMA_S = _p("REVP_V1TH_SCHEMA_S", SCHEMAS  / "protocol_c_hydromet_double_review_addendum_summary_v1th_schema.csv")
DOC      = _p("REVP_V1TH_DOC",      DOCS     / "revp_v1th_hydromet_double_review_addendum.md")

MAN_FIELDS = [
    "addendum_id", "packet_id", "event_candidate_id", "region",
    "source_packet_type", "addendum_status",
    "review_only", "hydromet_validates_event", "hydromet_is_negative_evidence",
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "formal_negative", "absence_as_negative", "notes",
]
FORM_FIELDS = [
    "form_id", "addendum_id", "packet_id", "event_candidate_id", "region",
    "question_key", "question_text", "answer_placeholder", "response_value",
    "review_only", "hydromet_validates_event", "hydromet_is_negative_evidence",
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "formal_negative", "absence_as_negative", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def run() -> dict[str, Any]:
    v1qw_manifest = read_csv_safe(DATASETS / "protocol_c_double_review_packet_manifest_v1qw.csv")
    packets_registry = read_csv_safe(DATASETS / "protocol_c_hydromet_evidence_packet_registry_v1tg.csv")

    # Index packets by event_candidate_id
    packets_by_cid: dict[str, dict] = {r["event_candidate_id"]: r for r in packets_registry}

    # Index v1qw packets by event_id (close-enough match to event_candidate_id)
    v1qw_by_event: dict[str, list[dict]] = {}
    for r in v1qw_manifest:
        eid = r.get("event_id", "")
        v1qw_by_event.setdefault(eid, []).append(r)

    man_rows:  list[dict[str, Any]] = []
    form_rows: list[dict[str, Any]] = []

    # For each hydromet packet, create addendum
    for pkt in packets_registry:
        cid    = pkt.get("event_candidate_id", "")
        region = pkt.get("region", "")
        r7d    = pkt.get("rain_7d", "")
        code   = pkt.get("nearest_station_code", "")
        dist   = pkt.get("nearest_station_distance_km", "")

        # Match to v1qw packets by event_id
        matched_pkts = v1qw_by_event.get(cid, [])
        # Also try partial match: cid might be embedded in event_id
        if not matched_pkts:
            for eid, pkts in v1qw_by_event.items():
                if cid in eid or eid in cid:
                    matched_pkts = pkts
                    break

        if matched_pkts:
            # Attach to each matched v1qw packet
            for v1qw_pkt in matched_pkts:
                addendum_id = f"V1TH_ADD_{hash_short(cid + v1qw_pkt.get('packet_id',''), 10)}"
                man_row: dict[str, Any] = {
                    "addendum_id":        addendum_id,
                    "packet_id":          v1qw_pkt.get("packet_id", ""),
                    "event_candidate_id": cid,
                    "region":             region,
                    "source_packet_type": "V1QW_DOUBLE_REVIEW",
                    "addendum_status":    "ADDENDUM_ATTACHED",
                    "notes":              "",
                }
                man_row.update(guardrail_row_extended())
                man_rows.append(man_row)

                for q in build_hydromet_question_set(cid, region, r7d, code, dist):
                    form_row: dict[str, Any] = {
                        "form_id":            f"V1TH_FORM_{hash_short(addendum_id+q['question_key'],12)}",
                        "addendum_id":        addendum_id,
                        "packet_id":          v1qw_pkt.get("packet_id", ""),
                        "event_candidate_id": cid,
                        "region":             region,
                        "question_key":       q["question_key"],
                        "question_text":      q["question_text"],
                        "answer_placeholder": "<TO_BE_FILLED_BY_REVIEWER>",
                        "response_value":     "",
                        "notes":              "",
                    }
                    form_row.update(guardrail_row_extended())
                    form_rows.append(form_row)
        else:
            # Standalone addendum
            addendum_id = f"V1TH_ADD_{hash_short(cid+'standalone', 10)}"
            man_row = {
                "addendum_id":        addendum_id,
                "packet_id":          "",
                "event_candidate_id": cid,
                "region":             region,
                "source_packet_type": "STANDALONE_NO_V1QW_MATCH",
                "addendum_status":    "ADDENDUM_STANDALONE_REVIEW_ONLY",
                "notes":              "no matching v1qw packet found",
            }
            man_row.update(guardrail_row_extended())
            man_rows.append(man_row)

            for q in build_hydromet_question_set(cid, region, r7d, code, dist):
                form_row = {
                    "form_id":            f"V1TH_FORM_{hash_short(addendum_id+q['question_key'],12)}",
                    "addendum_id":        addendum_id,
                    "packet_id":          "",
                    "event_candidate_id": cid,
                    "region":             region,
                    "question_key":       q["question_key"],
                    "question_text":      q["question_text"],
                    "answer_placeholder": "<TO_BE_FILLED_BY_REVIEWER>",
                    "response_value":     "",
                    "notes":              "",
                }
                form_row.update(guardrail_row_extended())
                form_rows.append(form_row)

    if not man_rows:
        man_rows = [{
            "addendum_id": "FAIL_CLOSED_NO_PACKETS", "packet_id": "",
            "event_candidate_id": "", "region": "",
            "source_packet_type": "NONE", "addendum_status": "FAIL_CLOSED_NO_PACKETS",
            "notes": "", **guardrail_row_extended(),
        }]

    for lst, label in [(man_rows, "v1th_manifest"), (form_rows, "v1th_forms")]:
        violations = scan_guardrails_extended(lst, label)
        if violations:
            raise ValueError(f"Guardrail violations in {label}: {violations[:3]}")

    write_csv_with_header(OUT_MAN, man_rows, MAN_FIELDS)
    write_csv_with_header(OUT_FORM, form_rows, FORM_FIELDS)
    write_schema(SCHEMA_M, MAN_FIELDS, "v1th_manifest")
    write_schema(SCHEMA_F, FORM_FIELDS, "v1th_forms")

    attached   = sum(1 for r in man_rows if r["addendum_status"] == "ADDENDUM_ATTACHED")
    standalone = sum(1 for r in man_rows if "STANDALONE" in r["addendum_status"])
    empty_resp = sum(1 for r in form_rows if r.get("response_value") == "")
    summary = [
        {"stat_key": "addenda_total",       "stat_value": str(len(man_rows))},
        {"stat_key": "attached_to_v1qw",    "stat_value": str(attached)},
        {"stat_key": "standalone",          "stat_value": str(standalone)},
        {"stat_key": "form_rows",           "stat_value": str(len(form_rows))},
        {"stat_key": "responses_empty",     "stat_value": str(empty_resp)},
        {"stat_key": "stage",               "stat_value": "v1th"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1th_summary")

    write_doc(DOC, "v1th — Hydromet Double Review Addendum", [
        "## Objetivo",
        "Addendum de revisão A/B com perguntas específicas para contexto INMET. "
        "Respostas sempre vazias — preenchidas pelo revisor humano.",
        f"## Resultado\nAddenda: {len(man_rows)}. "
        f"Attached: {attached}. Standalone: {standalone}. Forms: {len(form_rows)}.",
        "## Regra",
        "Respostas às perguntas não implicam validação de evento.",
    ])
    print(f"[v1th] addenda={len(man_rows)} attached={attached} standalone={standalone} forms={len(form_rows)}")
    return {"addenda": len(man_rows), "attached": attached, "forms": len(form_rows)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1th hydromet double review addendum").parse_args()
    run()
