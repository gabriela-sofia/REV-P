"""REV-P v1nm - C4 recheck after official-response intake."""

from __future__ import annotations

import argparse
import json

from revp_v1ni_v1nn_common import DATASETS, DOCS, SCHEMAS, c3_event_count, formal_negative_count_from_adjudication, gazette_formal_negative_count, read_csv, write_doc, write_outputs


OUT_C4 = DATASETS / "c4_recheck_after_official_negative_intake.csv"
OUT_READY = DATASETS / "c4_label_readiness_after_official_negative_intake.csv"
OUT_SUMMARY = DATASETS / "protocol_c_official_negative_intake_summary.csv"
SCHEMA_C4 = SCHEMAS / "c4_recheck_after_official_negative_intake_schema.csv"
SCHEMA_READY = SCHEMAS / "c4_label_readiness_after_official_negative_intake_schema.csv"
DOC = DOCS / "protocolo_c_recheck_c4_pos_intake_oficial_v1nm.md"

C4_FIELDS = ["recheck_id", "c3_event_count", "gazette_formal_negative_count", "official_intake_response_count", "formal_negative_count", "official_request_queue_status", "intake_status", "adjudication_status", "summary_decision", "can_create_operational_label", "can_train_model", "dino_role", "next_scientific_action"]
READY_FIELDS = ["readiness_id", "positive_c3_ready", "formal_negative_count", "negative_gate", "split_leakage_gate", "training_boundary", "can_create_operational_label", "can_train_model", "dino_role", "notes"]
SUMMARY_FIELDS = ["summary_id", "c3_event_count", "official_request_target_count", "official_question_count", "official_intake_response_count", "formal_negative_count", "summary_decision", "remaining_blocker", "can_create_operational_label", "can_train_model", "next_scientific_action"]


def build_recheck() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    c3_count = c3_event_count()
    intake_rows = read_csv(DATASETS / "official_negative_response_intake_registry.csv")
    adj_rows = read_csv(DATASETS / "strict_formal_negative_adjudication_registry.csv")
    targets = read_csv(DATASETS / "official_negative_evidence_request_target_registry.csv")
    questions = read_csv(DATASETS / "official_negative_request_question_bank.csv")
    response_count = sum(1 for row in intake_rows if row.get("intake_status") != "NO_OFFICIAL_RESPONSE_INTAKE")
    formal_negatives = formal_negative_count_from_adjudication()
    intake_status = "NO_OFFICIAL_RESPONSE_INTAKE" if response_count == 0 else "RESPONSE_PRESENT_NEEDS_REVIEW"
    adjudication_status = "BLOCKED_NO_OFFICIAL_RESPONSE_INTAKE" if response_count == 0 else "BLOCKED_PENDING_STRICT_REVIEW"
    decision = "C4_OPEN" if formal_negatives > 0 else "C4_BLOCKED_NO_FORMAL_NEGATIVES"
    c4 = [
        {
            "recheck_id": "C4_OFFICIAL_INTAKE_RECHECK_V1NM",
            "c3_event_count": str(c3_count),
            "gazette_formal_negative_count": str(gazette_formal_negative_count()),
            "official_intake_response_count": str(response_count),
            "formal_negative_count": str(formal_negatives),
            "official_request_queue_status": "REQUEST_QUEUE_PREPARED_NOT_EVIDENCE",
            "intake_status": intake_status,
            "adjudication_status": adjudication_status if not adj_rows else adj_rows[0].get("adjudication_status", adjudication_status),
            "summary_decision": decision,
            "can_create_operational_label": "false",
            "can_train_model": "false",
            "dino_role": "REVIEW_ONLY_REPRESENTATION",
            "next_scientific_action": "obter resposta oficial com declaracao negativa explicita",
        }
    ]
    ready = [
        {
            "readiness_id": "LABEL_READINESS_OFFICIAL_INTAKE_V1NM",
            "positive_c3_ready": "true" if c3_count >= 9 else "partial",
            "formal_negative_count": str(formal_negatives),
            "negative_gate": "FAIL_FORMAL_NEGATIVES_ZERO" if formal_negatives == 0 else "PENDING_C4_REVIEW",
            "split_leakage_gate": "BLOCKED_SPLIT_LEAKAGE_NOT_READY",
            "training_boundary": "BLOCKED_SUPERVISED_TRAINING_NOT_ALLOWED",
            "can_create_operational_label": "false",
            "can_train_model": "false",
            "dino_role": "REVIEW_ONLY_REPRESENTATION",
            "notes": "C4 does not open without adjudicated formal negatives; request queue and DINO triage are not labels.",
        }
    ]
    summary = [
        {
            "summary_id": "PROTOCOL_C_OFFICIAL_NEGATIVE_INTAKE_SUMMARY_V1NM",
            "c3_event_count": str(c3_count),
            "official_request_target_count": str(len(targets)),
            "official_question_count": str(len(questions)),
            "official_intake_response_count": str(response_count),
            "formal_negative_count": str(formal_negatives),
            "summary_decision": decision,
            "remaining_blocker": "NO_FORMAL_NEGATIVES" if formal_negatives == 0 else "SPLIT_LEAKAGE_NOT_READY",
            "can_create_operational_label": "false",
            "can_train_model": "false",
            "next_scientific_action": "obter resposta oficial com declaracao negativa explicita",
        }
    ]
    return c4, ready, summary


def write_method_doc() -> None:
    write_doc(
        DOC,
        "Protocolo C - recheck C4 pos intake oficial v1nm",
        [
            "Esta rechecagem integra C3, Diario Oficial v1na-v1nh, fila de pedidos v1ni-v1nj, intake v1nk e adjudicacao v1nl.",
            "No estado sem resposta oficial real, a decisao permanece C4_BLOCKED_NO_FORMAL_NEGATIVES.",
            "can_create_operational_label=false, can_train_model=false e DINO permanece REVIEW_ONLY_REPRESENTATION.",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--emit-evidence", action="store_true")
    args = parser.parse_args()
    if OUT_C4.exists() and OUT_READY.exists() and OUT_SUMMARY.exists() and not args.force:
        print(json.dumps({"stage": "v1nm", "status": "EXISTING_OUTPUTS_PRESERVED"}))
        return
    c4, ready, summary = build_recheck()
    if args.force or args.emit_evidence:
        write_method_doc()
        write_outputs(
            [(OUT_C4, c4, C4_FIELDS), (OUT_READY, ready, READY_FIELDS), (OUT_SUMMARY, summary, SUMMARY_FIELDS)],
            [(SCHEMA_C4, C4_FIELDS, "v1nm C4 recheck after official intake"), (SCHEMA_READY, READY_FIELDS, "v1nm label readiness after official intake")],
            [DOC],
        )
    print(json.dumps({"stage": "v1nm", "summary_decision": summary[0]["summary_decision"], "can_train_model": "false"}, indent=2))


if __name__ == "__main__":
    main()
