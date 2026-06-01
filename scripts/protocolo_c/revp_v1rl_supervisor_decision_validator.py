"""REV-P v1rl — Supervisor decision validator.

Validates manually-filled supervisor decisions from
REVP_PROTOCOL_C_SUPERVISOR_DECISIONS_PATH. When absent, waiting / fail-closed.
Even when the supervisor approves a C3 candidate, it stays review-only:
can_create_operational_label=false, ground_truth_operational=false. C4 is
never opened without an explicit formal negative source.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1rg_v1rm_review_response_common import (
    DATASETS,
    DOCS,
    SCHEMAS,
    SUP_APPROVE_C3,
    _p,
    assert_clean_rows,
    detect_absolute_path,
    detect_local_runs_exposure,
    guardrail_row,
    normalize_supervisor_action,
    read_csv_safe,
    supervisor_decisions_path,
    write_csv_with_header,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

IN_PACKETS = _p("REVP_V1RL_IN_PACKETS", DATASETS / "protocol_c_supervisor_review_packet_manifest_v1rj.csv")
OUT_VALIDATION = _p("REVP_V1RL_OUT_VALIDATION", DATASETS / "protocol_c_supervisor_decision_validation_v1rl.csv")
OUT_SUMMARY = _p("REVP_V1RL_OUT_SUMMARY", DATASETS / "protocol_c_supervisor_decision_validation_summary_v1rl.csv")
SCHEMA_VALIDATION = _p("REVP_V1RL_SCHEMA_VALIDATION", SCHEMAS / "protocol_c_supervisor_decision_validation_v1rl_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1RL_SCHEMA_SUMMARY", SCHEMAS / "protocol_c_supervisor_decision_validation_summary_v1rl_schema.csv")
DOC = _p("REVP_V1RL_DOC", DOCS / "revp_v1rl_supervisor_decision_validator.md")

VALIDATION_FIELDS = [
    "validation_id", "supervisor_packet_id", "review_sample_id",
    "supervisor_decision", "check_name", "status", "severity",
    "observed_value", "expected_value", "maintains_review_only", "c4_opened",
    "blocked_reason", "review_only", "can_create_operational_label",
    "can_train_model", "target_created", "ground_truth_operational",
    "formal_negative", "dino_validates_event", "absence_as_negative", "notes",
]

SUMMARY_FIELDS = ["stat_key", "stat_value"]

WAITING = "SUPERVISOR_DECISIONS_WAITING_MANUAL_INPUT"
PASS = "SUPERVISOR_DECISIONS_VALIDATION_PASS_REVIEW_ONLY"
FAIL_CLOSED = "SUPERVISOR_DECISIONS_VALIDATION_FAIL_CLOSED"

_FORBIDDEN_TRUE = ["can_create_operational_label", "can_train_model",
                   "target_created", "ground_truth_operational"]


def _validate_decision(idx_start: int, d: dict[str, str], packet_ids: set[str]) -> list[dict[str, Any]]:
    pid = d.get("supervisor_packet_id", "")
    rsid = d.get("review_sample_id", "")
    action = normalize_supervisor_action(d.get("supervisor_decision", ""))
    base = {"supervisor_packet_id": pid, "review_sample_id": rsid,
            "supervisor_decision": action or d.get("supervisor_decision", "")}
    out: list[dict[str, Any]] = []
    idx = idx_start

    def emit(name, ok, sev, obs, exp, reason="", c4="false", review_only="true"):
        nonlocal idx
        row = dict(base)
        row.update({
            "validation_id": f"V1RL_VAL_{idx:05d}", "check_name": name,
            "status": "PASS" if ok else "FAIL", "severity": sev,
            "observed_value": str(obs)[:80], "expected_value": exp,
            "maintains_review_only": review_only, "c4_opened": c4,
            "blocked_reason": "" if ok else (reason or name), "notes": "",
        })
        row.update(guardrail_row())
        out.append(row)
        idx += 1

    emit("supervisor_packet_exists", pid in packet_ids,
         "critical", pid, "known_packet_id", "PACKET_NOT_FOUND")
    emit("supervisor_decision_allowed", bool(action), "critical",
         d.get("supervisor_decision", ""), "allowed_supervisor_action", "DECISION_NOT_ALLOWED")

    conf = str(d.get("decision_confidence_0_4", "")).strip()
    emit("decision_confidence_in_range", conf.isdigit() and 0 <= int(conf) <= 4,
         "high", conf, "0..4", "CONFIDENCE_OUT_OF_RANGE")

    # Approving a C3 candidate must remain review-only
    label_true = str(d.get("can_create_operational_label", "false")).strip().lower() == "true"
    gt_true = str(d.get("ground_truth_operational", "false")).strip().lower() == "true"
    emit("approval_stays_review_only", not (action == SUP_APPROVE_C3 and (label_true or gt_true)),
         "critical", "violation" if (label_true or gt_true) else "review_only",
         "approve_c3_review_only", "APPROVAL_TRIED_TO_CREATE_LABEL")

    # C4 never opened without explicit formal negative source
    formal = str(d.get("formal_negative", "false")).strip().lower() == "true"
    emit("c4_not_opened_without_formal_source", not formal, "critical",
         "c4_opened" if formal else "c4_closed", "formal_negative_false",
         "C4_OPENED_WITHOUT_FORMAL_SOURCE", c4="true" if formal else "false")

    blob = " ".join(str(v) for v in d.values())
    unsafe = detect_absolute_path(blob) or detect_local_runs_exposure(blob)
    emit("no_absolute_path", not unsafe, "critical",
         "unsafe" if unsafe else "safe", "no_absolute_path_no_localrun", "UNSAFE_PATH")

    forbidden_hit = any(str(d.get(f, "false")).strip().lower() == "true" for f in _FORBIDDEN_TRUE)
    emit("no_label_target_ground_truth_true", not forbidden_hit, "critical",
         "violation" if forbidden_hit else "clean", "all_guardrails_false", "FORBIDDEN_TRUE_FIELD")

    return out


def run(datasets: Path | None = None) -> dict[str, Any]:
    path = supervisor_decisions_path()
    packet_ids = {p.get("supervisor_packet_id", "") for p in read_csv_safe(IN_PACKETS) if p.get("supervisor_packet_id", "")}

    rows: list[dict[str, Any]] = []
    status = WAITING
    approved_c3 = 0
    if path is not None:
        decisions = [d for d in read_csv_safe(path) if str(d.get("supervisor_decision", "")).strip()]
        if decisions:
            idx = 0
            for d in decisions:
                checks = _validate_decision(idx, d, packet_ids)
                idx += len(checks)
                rows.extend(checks)
                if normalize_supervisor_action(d.get("supervisor_decision", "")) == SUP_APPROVE_C3:
                    approved_c3 += 1
            any_fail = any(r["status"] == "FAIL" and r["severity"] in ("critical", "high") for r in rows)
            status = FAIL_CLOSED if any_fail else PASS

    assert_clean_rows(rows, "v1rl_validation")
    write_csv_with_header(OUT_VALIDATION, rows, VALIDATION_FIELDS)
    write_schema_safe(SCHEMA_VALIDATION, VALIDATION_FIELDS, "v1rl_validation")

    passed = sum(1 for r in rows if r["status"] == "PASS")
    failed = sum(1 for r in rows if r["status"] == "FAIL")
    decisions_n = len({r["supervisor_packet_id"] for r in rows})
    summary = [
        {"stat_key": "validation_status", "stat_value": status},
        {"stat_key": "decisions_examined", "stat_value": str(decisions_n)},
        {"stat_key": "checks_total", "stat_value": str(len(rows))},
        {"stat_key": "checks_passed", "stat_value": str(passed)},
        {"stat_key": "checks_failed", "stat_value": str(failed)},
        {"stat_key": "approved_c3_candidates_review_only", "stat_value": str(approved_c3 if status == PASS else 0)},
        {"stat_key": "c4_formal_negatives_opened", "stat_value": "0"},
        {"stat_key": "decisions_path_present", "stat_value": "true" if path else "false"},
        {"stat_key": "stage", "stat_value": "v1rl"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    write_schema_safe(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1rl_summary")

    write_doc(
        DOC,
        "v1rl — Supervisor Decision Validator",
        [
            "## Objetivo",
            "Validar decisoes do supervisor preenchidas manualmente "
            "(REVP_PROTOCOL_C_SUPERVISOR_DECISIONS_PATH). Sem o arquivo, "
            "SUPERVISOR_DECISIONS_WAITING_MANUAL_INPUT.",
            "## Garantias",
            "Aprovar C3 candidate permanece review-only (can_create_operational_label=false, "
            "ground_truth_operational=false). C4 nunca aberto sem fonte formal negativa.",
            "## Resultado",
            f"Status: {status}. Decisoes: {decisions_n}. Checagens: {len(rows)} "
            f"(passou {passed}, falhou {failed}).",
            "## Guardrails",
            "review_only=true. formal_negative=false esperado. Nenhum label/target/ground truth.",
        ],
    )
    print(f"[v1rl] status={status} decisions={decisions_n} checks={len(rows)} failed={failed}")
    return {"status": status, "decisions": decisions_n, "checks": len(rows), "failed": failed}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1rl supervisor decision validator").parse_args()
    run()
