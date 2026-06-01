"""REV-P v1rh — Review response validator.

Validates manually-filled A/B review responses from
REVP_PROTOCOL_C_REVIEW_RESPONSES_PATH. When absent, fail-closed / waiting.
Emits one validation row per check per (sample, reviewer slot). Review-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1rg_v1rm_review_response_common import (
    DATASETS,
    DOCS,
    SCHEMAS,
    REQUIRED_QUESTIONS,
    _p,
    allow_synthetic,
    assert_clean_rows,
    detect_absolute_path,
    detect_local_runs_exposure,
    guardrail_row,
    normalize_decision,
    normalize_reviewer_slot,
    read_csv_safe,
    responses_path,
    write_csv_with_header,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

IN_PACKETS = _p("REVP_V1RH_IN_PACKETS", DATASETS / "protocol_c_double_review_packet_manifest_v1qw.csv")
OUT_VALIDATION = _p("REVP_V1RH_OUT_VALIDATION", DATASETS / "protocol_c_review_response_validation_v1rh.csv")
OUT_SUMMARY = _p("REVP_V1RH_OUT_SUMMARY", DATASETS / "protocol_c_review_response_validation_summary_v1rh.csv")
SCHEMA_VALIDATION = _p("REVP_V1RH_SCHEMA_VALIDATION", SCHEMAS / "protocol_c_review_response_validation_v1rh_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1RH_SCHEMA_SUMMARY", SCHEMAS / "protocol_c_review_response_validation_summary_v1rh_schema.csv")
DOC = _p("REVP_V1RH_DOC", DOCS / "revp_v1rh_review_response_validator.md")

VALIDATION_FIELDS = [
    "validation_id", "packet_id", "review_sample_id", "reviewer_slot",
    "check_name", "status", "severity", "observed_value", "expected_value",
    "blocked_reason", "review_only", "can_create_operational_label",
    "can_train_model", "target_created", "ground_truth_operational",
    "formal_negative", "dino_validates_event", "absence_as_negative", "notes",
]

SUMMARY_FIELDS = ["stat_key", "stat_value"]

WAITING = "REVIEW_RESPONSES_WAITING_MANUAL_INPUT"
PASS = "REVIEW_RESPONSES_VALIDATION_PASS_REVIEW_ONLY"
FAIL_CLOSED = "REVIEW_RESPONSES_VALIDATION_FAIL_CLOSED"

_FORBIDDEN_TRUE = ["can_create_operational_label", "can_train_model",
                   "target_created", "ground_truth_operational", "formal_negative",
                   "dino_validates_event", "absence_as_negative"]


def _group(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        rsid = str(r.get("review_sample_id", "")).strip()
        slot = normalize_reviewer_slot(r.get("reviewer_slot", ""))
        if not rsid:
            continue
        key = (rsid, slot)
        g = groups.setdefault(key, {"packet_id": r.get("packet_id", ""), "answers": {}, "rows": []})
        qid = str(r.get("question_id", "")).strip()
        val = str(r.get("answer_value", "")).strip()
        if qid:
            g["answers"][qid] = val
        g["rows"].append(r)
    return groups


def _checks_for_group(idx_start: int, key: tuple[str, str], g: dict[str, Any],
                      packet_ids: set[str]) -> list[dict[str, Any]]:
    rsid, slot = key
    answers = g["answers"]
    rows = g["rows"]
    packet_id = g["packet_id"]
    base = {"packet_id": packet_id, "review_sample_id": rsid, "reviewer_slot": slot}
    out: list[dict[str, Any]] = []
    idx = idx_start

    def emit(name, ok, sev, obs, exp, reason=""):
        nonlocal idx
        row = dict(base)
        row.update({
            "validation_id": f"V1RH_VAL_{idx:05d}", "check_name": name,
            "status": "PASS" if ok else "FAIL", "severity": sev,
            "observed_value": str(obs)[:80], "expected_value": exp,
            "blocked_reason": "" if ok else (reason or name), "notes": "",
        })
        row.update(guardrail_row())
        out.append(row)
        idx += 1

    emit("packet_id_exists", packet_id in packet_ids,
         "critical", packet_id, "known_packet_id", "PACKET_NOT_FOUND")
    emit("reviewer_slot_valid", slot in ("REVIEWER_A", "REVIEWER_B"),
         "critical", slot, "REVIEWER_A|REVIEWER_B", "INVALID_SLOT")

    missing_q = [q for q in REQUIRED_QUESTIONS if not str(answers.get(q, "")).strip()]
    emit("all_required_questions_answered", not missing_q, "high",
         ",".join(missing_q) or "complete", "all_required_answered",
         "MISSING_REQUIRED_ANSWERS")

    confidences = [str(r.get("confidence_0_4", "")).strip() for r in rows if str(r.get("confidence_0_4", "")).strip()]
    bad_conf = [c for c in confidences if not (c.isdigit() and 0 <= int(c) <= 4)]
    emit("confidence_in_range_0_4", not bad_conf, "high",
         ",".join(bad_conf) or "ok", "0..4", "CONFIDENCE_OUT_OF_RANGE")

    decision = normalize_decision(answers.get("recommended_decision", ""))
    emit("recommended_decision_allowed", bool(decision), "high",
         answers.get("recommended_decision", ""), "allowed_decision_token",
         "DECISION_NOT_ALLOWED")

    event_supported = str(answers.get("event_supported", "")).strip().lower() in ("sim", "yes", "true", "1", "y")
    has_source = any(str(r.get("source_reference", "")).strip() for r in rows)
    emit("source_reference_when_event_supported", (not event_supported) or has_source,
         "high", "present" if has_source else "absent",
         "source_reference_required_if_event_supported", "MISSING_SOURCE_REFERENCE")

    blob = " ".join(str(v) for r in rows for v in r.values())
    unsafe = detect_absolute_path(blob) or detect_local_runs_exposure(blob)
    emit("no_absolute_path", not unsafe, "critical",
         "unsafe" if unsafe else "safe", "no_absolute_path_no_localrun", "UNSAFE_PATH")

    low = blob.lower()
    has_synth = ("synthetic" in low or "fixture" in low or "test_only" in low)
    emit("no_synthetic_unless_sandbox", (not has_synth) or allow_synthetic(),
         "critical", "synthetic" if has_synth else "real",
         "no_synthetic_marker_unless_sandbox", "SYNTHETIC_NOT_ALLOWED")

    forbidden_hit = any(
        str(r.get(f, "false")).strip().lower() == "true"
        for r in rows for f in _FORBIDDEN_TRUE
    )
    emit("no_label_target_ground_truth_true", not forbidden_hit, "critical",
         "violation" if forbidden_hit else "clean", "all_guardrails_false",
         "FORBIDDEN_TRUE_FIELD")

    return out


def run(datasets: Path | None = None) -> dict[str, Any]:
    path = responses_path()
    packet_ids = {p.get("packet_id", "") for p in read_csv_safe(IN_PACKETS) if p.get("packet_id", "")}

    rows: list[dict[str, Any]] = []
    status = WAITING
    if path is not None:
        responses = read_csv_safe(path)
        filled = [r for r in responses if str(r.get("answer_value", "")).strip()]
        if filled:
            groups = _group(filled)
            idx = 0
            for key in sorted(groups):
                checks = _checks_for_group(idx, key, groups[key], packet_ids)
                idx += len(checks)
                rows.extend(checks)
            any_fail = any(r["status"] == "FAIL" and r["severity"] in ("critical", "high") for r in rows)
            status = FAIL_CLOSED if any_fail else PASS

    assert_clean_rows(rows, "v1rh_validation")
    write_csv_with_header(OUT_VALIDATION, rows, VALIDATION_FIELDS)
    write_schema_safe(SCHEMA_VALIDATION, VALIDATION_FIELDS, "v1rh_validation")

    passed = sum(1 for r in rows if r["status"] == "PASS")
    failed = sum(1 for r in rows if r["status"] == "FAIL")
    groups_n = len({(r["review_sample_id"], r["reviewer_slot"]) for r in rows})
    summary = [
        {"stat_key": "validation_status", "stat_value": status},
        {"stat_key": "response_groups", "stat_value": str(groups_n)},
        {"stat_key": "checks_total", "stat_value": str(len(rows))},
        {"stat_key": "checks_passed", "stat_value": str(passed)},
        {"stat_key": "checks_failed", "stat_value": str(failed)},
        {"stat_key": "responses_path_present", "stat_value": "true" if path else "false"},
        {"stat_key": "synthetic_allowed", "stat_value": "true" if allow_synthetic() else "false"},
        {"stat_key": "stage", "stat_value": "v1rh"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    write_schema_safe(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1rh_summary")

    write_doc(
        DOC,
        "v1rh — Review Response Validator",
        [
            "## Objetivo",
            "Validar respostas A/B preenchidas manualmente "
            "(REVP_PROTOCOL_C_REVIEW_RESPONSES_PATH). Sem o arquivo, "
            "REVIEW_RESPONSES_WAITING_MANUAL_INPUT. Uma linha por checagem por (sample, slot).",
            "## Checagens",
            "packet_id_exists, reviewer_slot_valid, all_required_questions_answered, "
            "confidence_in_range_0_4, recommended_decision_allowed, "
            "source_reference_when_event_supported, no_absolute_path, "
            "no_synthetic_unless_sandbox, no_label_target_ground_truth_true.",
            "## Resultado",
            f"Status: {status}. Grupos: {groups_n}. Checagens: {len(rows)} "
            f"(passou {passed}, falhou {failed}).",
            "## Guardrails",
            "review_only=true. Fixture/synthetic so com sandbox explicito. "
            "Nenhum label/target/ground truth/negativo formal.",
        ],
    )
    print(f"[v1rh] status={status} groups={groups_n} checks={len(rows)} failed={failed}")
    return {"status": status, "groups": groups_n, "checks": len(rows), "failed": failed}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1rh review response validator").parse_args()
    run()
