"""Tests for REV-P Protocol C v1rg-v1rm review/supervisor gate workflow.

All script outputs are redirected to tmp_path; the real datasets/ tree is
never written. Manual responses and supervisor decisions are simulated via
fixture CSVs and env vars. Fixtures live only in tmp_path.
"""
from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "protocolo_c"
sys.path.insert(0, str(SCRIPTS))

import revp_v1rg_v1rm_review_response_common as RC  # noqa: E402
import revp_v1qu_v1qz_ground_reference_common as G  # noqa: E402

v1rg = importlib.import_module("revp_v1rg_review_response_intake_template")
v1rh = importlib.import_module("revp_v1rh_review_response_validator")
v1ri = importlib.import_module("revp_v1ri_completed_review_scoring_replay")
v1rj = importlib.import_module("revp_v1rj_supervisor_review_packet_generator")
v1rk = importlib.import_module("revp_v1rk_supervisor_decision_intake_template")
v1rl = importlib.import_module("revp_v1rl_supervisor_decision_validator")
v1rm = importlib.import_module("revp_v1rm_review_supervisor_gate_bundle")

RESP_ENV = "REVP_PROTOCOL_C_REVIEW_RESPONSES_PATH"
SUP_ENV = "REVP_PROTOCOL_C_SUPERVISOR_DECISIONS_PATH"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _header(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh), [])


def _redirect(monkeypatch, mod, tmp: Path) -> None:
    for name in dir(mod):
        if name.startswith(("OUT_", "SCHEMA_", "DOC")):
            val = getattr(mod, name)
            if isinstance(val, Path):
                monkeypatch.setattr(mod, name, tmp / val.name)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


RESP_FIELDS = [
    "response_id", "packet_id", "review_sample_id", "reviewer_slot",
    "question_id", "question_text", "answer_value", "confidence_0_4",
    "source_reference", "uncertainty_note", "response_status",
    "review_only", "can_create_operational_label", "can_train_model",
    "target_created", "ground_truth_operational",
]

_DEFAULT_ANSWERS = {
    "evidence_visible": "sim", "event_supported": "sim", "location_supported": "sim",
    "timing_supported": "sim", "source_quality": "oficial SGB CPRM",
    "independent_source_present": "sim", "uncertainty_level": "baixo",
    "recommended_decision": "C3-candidate", "uncertainty_notes": "",
}


def _slot_rows(rsid, packet_id, slot, answers, *, confidence="3", source_ref="Boletim SGB CPRM"):
    rows = []
    for q, v in answers.items():
        rows.append({
            "response_id": f"R_{rsid}_{slot}_{q}", "packet_id": packet_id,
            "review_sample_id": rsid, "reviewer_slot": slot,
            "question_id": q, "answer_value": v, "confidence_0_4": confidence,
            "source_reference": source_ref, "response_status": "FILLED",
            "review_only": "true", "can_create_operational_label": "false",
            "can_train_model": "false", "target_created": "false",
            "ground_truth_operational": "false",
        })
    return rows


def _responses(rsid="RS1", *, both=True, b_decision=None, timing="sim",
               location="sim", source_ref="Boletim SGB CPRM", confidence="3",
               source_quality="oficial SGB CPRM"):
    a = dict(_DEFAULT_ANSWERS, timing_supported=timing, location_supported=location,
             source_quality=source_quality)
    rows = _slot_rows(rsid, f"PKT_{rsid}_A", "REVIEWER_A", a, confidence=confidence, source_ref=source_ref)
    if both:
        b = dict(a)
        if b_decision:
            b["recommended_decision"] = b_decision
        rows += _slot_rows(rsid, f"PKT_{rsid}_B", "REVIEWER_B", b, confidence=confidence, source_ref=source_ref)
    return rows


def _write_responses(tmp, rows):
    p = tmp / "responses.csv"
    _write_csv(p, rows, RESP_FIELDS)
    return p


def _write_packets(tmp, rsids=("RS1",)):
    fields = ["packet_id", "review_sample_id", "reviewer_slot", "event_id", "patch_id", "region"]
    rows = []
    for rsid in rsids:
        for slot in ("REVIEWER_A", "REVIEWER_B"):
            rows.append({"packet_id": f"PKT_{rsid}_{slot[-1]}", "review_sample_id": rsid,
                         "reviewer_slot": slot, "event_id": f"EV_{rsid}",
                         "patch_id": f"PET_1000{rsid[-1]}", "region": "PET"})
    p = tmp / "packets.csv"
    _write_csv(p, rows, fields)
    return p


def _write_sample(tmp, rsids=("RS1",)):
    fields = ["review_sample_id", "event_id", "patch_id", "region", "evidence_status"]
    rows = [{"review_sample_id": r, "event_id": f"EV_{r}", "patch_id": f"PET_1000{r[-1]}",
             "region": "PET", "evidence_status": "official"} for r in rsids]
    p = tmp / "sample.csv"
    _write_csv(p, rows, fields)
    return p


def _run_validate(monkeypatch, tmp, rows):
    _redirect(monkeypatch, v1rh, tmp)
    monkeypatch.setattr(v1rh, "IN_PACKETS", _write_packets(tmp))
    monkeypatch.setenv(RESP_ENV, str(_write_responses(tmp, rows)))
    return v1rh.run()


def _run_score(monkeypatch, tmp, rows):
    _run_validate(monkeypatch, tmp, rows)
    _redirect(monkeypatch, v1ri, tmp)
    monkeypatch.setattr(v1ri, "IN_SAMPLE", _write_sample(tmp))
    monkeypatch.setattr(v1ri, "IN_VALIDATION_SUMMARY", tmp / v1rh.OUT_SUMMARY.name)
    return v1ri.run()


# ===========================================================================
# Common helpers
# ===========================================================================

def test_common_empty_csv_header(tmp_path):
    RC.write_csv_with_header(tmp_path / "x.csv", [], ["a", "b"])
    assert _header(tmp_path / "x.csv") == ["a", "b"]

def test_common_mask_absolute_path():
    assert "C:" not in RC.mask_path(r"C:\Users\gabriela\f.tif")

def test_common_forbidden_literal_not_exposed():
    # the masker neutralizes the literal path-dir reference
    out = RC.mask_path("local_runs/x/y")
    assert "REDACTED" in out

def test_common_normalize_reviewer_slot_a():
    assert RC.normalize_reviewer_slot("a") == "REVIEWER_A"

def test_common_normalize_reviewer_slot_b():
    assert RC.normalize_reviewer_slot("REVIEWER_B") == "REVIEWER_B"

def test_common_normalize_reviewer_slot_unknown():
    assert RC.normalize_reviewer_slot("X") == "UNKNOWN_SLOT"

def test_common_normalize_decision_c3():
    assert RC.normalize_decision("C3-candidate") == RC.C3_NEEDS_SUPERVISOR

def test_common_normalize_decision_c2():
    assert RC.normalize_decision("C2") == RC.KEEP_C2_REVIEW_ONLY

def test_common_normalize_decision_invalid():
    assert RC.normalize_decision("maybe") == ""

def test_common_normalize_supervisor_action():
    assert RC.normalize_supervisor_action("approve c3") == RC.SUP_APPROVE_C3


# ===========================================================================
# v1rg — template
# ===========================================================================

def test_v1rg_template_created(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rg, tmp_path)
    monkeypatch.setattr(v1rg, "OUT_SCHEMA", tmp_path / v1rg.OUT_SCHEMA.name)
    monkeypatch.setattr(v1rg, "IN_PACKETS", _write_packets(tmp_path))
    v1rg.run()
    assert (tmp_path / v1rg.OUT_TEMPLATE.name).exists()

def test_v1rg_preserves_ab(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rg, tmp_path)
    monkeypatch.setattr(v1rg, "OUT_SCHEMA", tmp_path / v1rg.OUT_SCHEMA.name)
    monkeypatch.setattr(v1rg, "IN_PACKETS", _write_packets(tmp_path))
    v1rg.run()
    rows = _read(tmp_path / v1rg.OUT_TEMPLATE.name)
    slots = {r["reviewer_slot"] for r in rows}
    assert slots == {"REVIEWER_A", "REVIEWER_B"}

def test_v1rg_no_answers_filled(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rg, tmp_path)
    monkeypatch.setattr(v1rg, "OUT_SCHEMA", tmp_path / v1rg.OUT_SCHEMA.name)
    monkeypatch.setattr(v1rg, "IN_PACKETS", _write_packets(tmp_path))
    v1rg.run()
    rows = _read(tmp_path / v1rg.OUT_TEMPLATE.name)
    assert all(r["answer_value"] == "" for r in rows)

def test_v1rg_schema_and_doc(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rg, tmp_path)
    monkeypatch.setattr(v1rg, "OUT_SCHEMA", tmp_path / v1rg.OUT_SCHEMA.name)
    monkeypatch.setattr(v1rg, "IN_PACKETS", _write_packets(tmp_path))
    v1rg.run()
    assert (tmp_path / v1rg.SCHEMA_TEMPLATE.name).exists()
    assert (tmp_path / v1rg.DOC.name).exists()


# ===========================================================================
# v1rh — validator
# ===========================================================================

def test_v1rh_waiting_without_env(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rh, tmp_path)
    monkeypatch.delenv(RESP_ENV, raising=False)
    out = v1rh.run()
    assert out["status"] == "REVIEW_RESPONSES_WAITING_MANUAL_INPUT"
    assert _header(tmp_path / v1rh.OUT_VALIDATION.name)

def test_v1rh_accepts_valid_ab(monkeypatch, tmp_path):
    out = _run_validate(monkeypatch, tmp_path, _responses())
    assert out["status"] == "REVIEW_RESPONSES_VALIDATION_PASS_REVIEW_ONLY"

def test_v1rh_blocks_unknown_packet(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rh, tmp_path)
    # packets file empty -> packet not found
    _write_csv(tmp_path / "packets.csv", [], ["packet_id"])
    monkeypatch.setattr(v1rh, "IN_PACKETS", tmp_path / "packets.csv")
    monkeypatch.setenv(RESP_ENV, str(_write_responses(tmp_path, _responses())))
    v1rh.run()
    rows = _read(tmp_path / v1rh.OUT_VALIDATION.name)
    pe = [r for r in rows if r["check_name"] == "packet_id_exists"]
    assert pe and pe[0]["status"] == "FAIL"

def test_v1rh_blocks_invalid_slot(monkeypatch, tmp_path):
    rows = _responses()
    for r in rows:
        if r["reviewer_slot"] == "REVIEWER_B":
            r["reviewer_slot"] = "C"
    _run_validate(monkeypatch, tmp_path, rows)
    v = _read(tmp_path / v1rh.OUT_VALIDATION.name)
    bad = [r for r in v if r["check_name"] == "reviewer_slot_valid" and r["status"] == "FAIL"]
    assert bad

def test_v1rh_blocks_bad_confidence(monkeypatch, tmp_path):
    _run_validate(monkeypatch, tmp_path, _responses(confidence="7"))
    v = _read(tmp_path / v1rh.OUT_VALIDATION.name)
    bad = [r for r in v if r["check_name"] == "confidence_in_range_0_4" and r["status"] == "FAIL"]
    assert bad

def test_v1rh_blocks_invalid_decision(monkeypatch, tmp_path):
    _run_validate(monkeypatch, tmp_path, _responses(b_decision="maybe"))
    # reviewer B decision invalid -> at least one FAIL on recommended_decision_allowed
    v = _read(tmp_path / v1rh.OUT_VALIDATION.name)
    bad = [r for r in v if r["check_name"] == "recommended_decision_allowed" and r["status"] == "FAIL"]
    assert bad

def test_v1rh_requires_source_when_event_supported(monkeypatch, tmp_path):
    _run_validate(monkeypatch, tmp_path, _responses(source_ref=""))
    v = _read(tmp_path / v1rh.OUT_VALIDATION.name)
    bad = [r for r in v if r["check_name"] == "source_reference_when_event_supported" and r["status"] == "FAIL"]
    assert bad


# ===========================================================================
# v1ri — scoring
# ===========================================================================

def test_v1ri_fail_closed_without_responses(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ri, tmp_path)
    monkeypatch.delenv(RESP_ENV, raising=False)
    monkeypatch.setattr(v1ri, "IN_VALIDATION_SUMMARY", tmp_path / "nope.csv")
    out = v1ri.run()
    assert out["status"] == "COMPLETED_REVIEW_NOT_AVAILABLE_FAIL_CLOSED"
    assert _header(tmp_path / v1ri.OUT_SCORES.name)

def test_v1ri_scores_agreement(monkeypatch, tmp_path):
    out = _run_score(monkeypatch, tmp_path, _responses())
    assert out["scored"] == 1
    rows = _read(tmp_path / v1ri.OUT_SCORES.name)
    assert rows[0]["disagreement_flag"] == "false"
    assert float(rows[0]["reviewer_agreement_score"]) == 1.0

def test_v1ri_scores_disagreement(monkeypatch, tmp_path):
    _run_score(monkeypatch, tmp_path, _responses(b_decision="C2"))
    rows = _read(tmp_path / v1ri.OUT_SCORES.name)
    assert rows[0]["disagreement_flag"] == "true"
    dis = _read(tmp_path / v1ri.OUT_DISAGREE.name)
    assert len(dis) == 1

def test_v1ri_requires_ab(monkeypatch, tmp_path):
    _run_score(monkeypatch, tmp_path, _responses(both=False))
    rows = _read(tmp_path / v1ri.OUT_SCORES.name)
    assert rows[0]["reviewer_b_present"] == "false"

def test_v1ri_unilateral_incomplete(monkeypatch, tmp_path):
    _run_score(monkeypatch, tmp_path, _responses(both=False))
    rows = _read(tmp_path / v1ri.OUT_SCORES.name)
    assert rows[0]["recommended_decision"] == RC.KEEP_C2_REVIEW_ONLY
    assert rows[0]["disagreement_type"] == "INCOMPLETE_REVIEW"

def test_v1ri_composite_computed(monkeypatch, tmp_path):
    _run_score(monkeypatch, tmp_path, _responses())
    rows = _read(tmp_path / v1ri.OUT_SCORES.name)
    assert float(rows[0]["composite_review_score"]) > 0.0

def test_v1ri_strong_review_is_c3_candidate(monkeypatch, tmp_path):
    _run_score(monkeypatch, tmp_path, _responses())
    rows = _read(tmp_path / v1ri.OUT_SCORES.name)
    assert rows[0]["recommended_decision"] == RC.C3_NEEDS_SUPERVISOR
    assert rows[0]["supervisor_review_required"] == "true"

def test_v1ri_low_temporal_blocks_c3(monkeypatch, tmp_path):
    _run_score(monkeypatch, tmp_path, _responses(timing="nao"))
    rows = _read(tmp_path / v1ri.OUT_SCORES.name)
    assert "C3_REFERENCE" not in rows[0]["recommended_decision"]

def test_v1ri_never_creates_label(monkeypatch, tmp_path):
    _run_score(monkeypatch, tmp_path, _responses())
    rows = _read(tmp_path / v1ri.OUT_SCORES.name)
    assert all(r["can_create_operational_label"] == "false" for r in rows)
    assert all(r["ground_truth_operational"] == "false" for r in rows)


# ===========================================================================
# v1rj — supervisor packets
# ===========================================================================

def _run_packets(monkeypatch, tmp, rows):
    _run_score(monkeypatch, tmp, rows)
    _redirect(monkeypatch, v1rj, tmp)
    monkeypatch.setattr(v1rj, "IN_SCORES", tmp / v1ri.OUT_SCORES.name)
    return v1rj.run()

def test_v1rj_waiting_without_completed(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rj, tmp_path)
    RC.write_csv_with_header(tmp_path / v1ri.OUT_SCORES.name, [], v1ri.SCORE_FIELDS)
    monkeypatch.setattr(v1rj, "IN_SCORES", tmp_path / v1ri.OUT_SCORES.name)
    out = v1rj.run()
    assert out["status"] == "SUPERVISOR_PACKETS_WAITING_COMPLETED_REVIEWS"

def test_v1rj_created_for_strong_review(monkeypatch, tmp_path):
    out = _run_packets(monkeypatch, tmp_path, _responses())
    assert out["packets"] == 1
    rows = _read(tmp_path / v1rj.OUT_MANIFEST.name)
    assert rows[0]["supervisor_action_required"] == RC.SUP_APPROVE_C3

def test_v1rj_blocks_disagreement(monkeypatch, tmp_path):
    _run_packets(monkeypatch, tmp_path, _responses(b_decision="C2"))
    rows = _read(tmp_path / v1rj.OUT_MANIFEST.name)
    assert rows and rows[0]["supervisor_action_required"] == RC.SUP_REQUEST_MORE

def test_v1rj_requires_source(monkeypatch, tmp_path):
    _run_packets(monkeypatch, tmp_path, _responses(source_quality="blog noticia"))
    rows = _read(tmp_path / v1rj.OUT_MANIFEST.name)
    # weak source -> no supervisor_review_required upstream -> no packet, or block source
    assert all(r["supervisor_action_required"] != RC.SUP_APPROVE_C3 for r in rows)

def test_v1rj_promote_review_only(monkeypatch, tmp_path):
    _run_packets(monkeypatch, tmp_path, _responses())
    rows = _read(tmp_path / v1rj.OUT_MANIFEST.name)
    assert rows[0]["can_promote_to_c3_candidate"] == "true"
    assert rows[0]["can_create_operational_label"] == "false"


# ===========================================================================
# v1rk — supervisor decision template
# ===========================================================================

def test_v1rk_template_created(monkeypatch, tmp_path):
    _run_packets(monkeypatch, tmp_path, _responses())
    _redirect(monkeypatch, v1rk, tmp_path)
    monkeypatch.setattr(v1rk, "OUT_SCHEMA", tmp_path / v1rk.OUT_SCHEMA.name)
    monkeypatch.setattr(v1rk, "IN_PACKETS", tmp_path / v1rj.OUT_MANIFEST.name)
    v1rk.run()
    rows = _read(tmp_path / v1rk.OUT_TEMPLATE.name)
    assert len(rows) == 1
    assert rows[0]["supervisor_decision"] == ""


# ===========================================================================
# v1rl — supervisor decision validator
# ===========================================================================

def _write_decisions(tmp, packet_id="V1RJ_SPKT_RS1", decision="APPROVE_C3_CANDIDATE_REVIEW_ONLY",
                     confidence="3", **over):
    fields = ["supervisor_decision_id", "supervisor_packet_id", "review_sample_id",
              "supervisor_decision", "decision_confidence_0_4", "review_only",
              "can_create_operational_label", "ground_truth_operational", "formal_negative"]
    row = {"supervisor_decision_id": "SD1", "supervisor_packet_id": packet_id,
           "review_sample_id": "RS1", "supervisor_decision": decision,
           "decision_confidence_0_4": confidence, "review_only": "true",
           "can_create_operational_label": "false", "ground_truth_operational": "false",
           "formal_negative": "false"}
    row.update(over)
    p = tmp / "decisions.csv"
    _write_csv(p, [row], fields)
    return p

def _write_sup_packets(tmp, packet_id="V1RJ_SPKT_RS1"):
    fields = ["supervisor_packet_id", "review_sample_id"]
    p = tmp / "sup_packets.csv"
    _write_csv(p, [{"supervisor_packet_id": packet_id, "review_sample_id": "RS1"}], fields)
    return p

def test_v1rl_waiting_without_env(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rl, tmp_path)
    monkeypatch.delenv(SUP_ENV, raising=False)
    out = v1rl.run()
    assert out["status"] == "SUPERVISOR_DECISIONS_WAITING_MANUAL_INPUT"
    assert _header(tmp_path / v1rl.OUT_VALIDATION.name)

def test_v1rl_accepts_valid_decision(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rl, tmp_path)
    monkeypatch.setattr(v1rl, "IN_PACKETS", _write_sup_packets(tmp_path))
    monkeypatch.setenv(SUP_ENV, str(_write_decisions(tmp_path)))
    out = v1rl.run()
    assert out["status"] == "SUPERVISOR_DECISIONS_VALIDATION_PASS_REVIEW_ONLY"

def test_v1rl_blocks_invalid_decision(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rl, tmp_path)
    monkeypatch.setattr(v1rl, "IN_PACKETS", _write_sup_packets(tmp_path))
    monkeypatch.setenv(SUP_ENV, str(_write_decisions(tmp_path, decision="DO_WHATEVER")))
    v1rl.run()
    rows = _read(tmp_path / v1rl.OUT_VALIDATION.name)
    bad = [r for r in rows if r["check_name"] == "supervisor_decision_allowed" and r["status"] == "FAIL"]
    assert bad

def test_v1rl_keeps_review_only_on_approval(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rl, tmp_path)
    monkeypatch.setattr(v1rl, "IN_PACKETS", _write_sup_packets(tmp_path))
    # approval that illegally tries to create a label must FAIL
    monkeypatch.setenv(SUP_ENV, str(_write_decisions(tmp_path, can_create_operational_label="true")))
    v1rl.run()
    rows = _read(tmp_path / v1rl.OUT_VALIDATION.name)
    bad = [r for r in rows if r["check_name"] == "approval_stays_review_only" and r["status"] == "FAIL"]
    assert bad

def test_v1rl_no_c4_without_formal(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1rl, tmp_path)
    monkeypatch.setattr(v1rl, "IN_PACKETS", _write_sup_packets(tmp_path))
    monkeypatch.setenv(SUP_ENV, str(_write_decisions(tmp_path, formal_negative="true")))
    v1rl.run()
    rows = _read(tmp_path / v1rl.OUT_VALIDATION.name)
    bad = [r for r in rows if r["check_name"] == "c4_not_opened_without_formal_source" and r["status"] == "FAIL"]
    assert bad


# ===========================================================================
# v1rm — bundle
# ===========================================================================

def _full(monkeypatch, tmp, *, with_responses=False, with_supervisor=False):
    # v1rg
    _redirect(monkeypatch, v1rg, tmp)
    monkeypatch.setattr(v1rg, "OUT_SCHEMA", tmp / v1rg.OUT_SCHEMA.name)
    monkeypatch.setattr(v1rg, "IN_PACKETS", _write_packets(tmp))
    v1rg.run()
    # v1rh / v1ri
    if with_responses:
        _run_score(monkeypatch, tmp, _responses())
    else:
        _redirect(monkeypatch, v1rh, tmp)
        monkeypatch.delenv(RESP_ENV, raising=False)
        monkeypatch.setattr(v1rh, "IN_PACKETS", _write_packets(tmp))
        v1rh.run()
        _redirect(monkeypatch, v1ri, tmp)
        monkeypatch.setattr(v1ri, "IN_SAMPLE", _write_sample(tmp))
        monkeypatch.setattr(v1ri, "IN_VALIDATION_SUMMARY", tmp / v1rh.OUT_SUMMARY.name)
        v1ri.run()
    # v1rj
    _redirect(monkeypatch, v1rj, tmp)
    monkeypatch.setattr(v1rj, "IN_SCORES", tmp / v1ri.OUT_SCORES.name)
    v1rj.run()
    # v1rk
    _redirect(monkeypatch, v1rk, tmp)
    monkeypatch.setattr(v1rk, "OUT_SCHEMA", tmp / v1rk.OUT_SCHEMA.name)
    monkeypatch.setattr(v1rk, "IN_PACKETS", tmp / v1rj.OUT_MANIFEST.name)
    v1rk.run()
    # v1rl
    _redirect(monkeypatch, v1rl, tmp)
    monkeypatch.setattr(v1rl, "IN_PACKETS", tmp / v1rj.OUT_MANIFEST.name)
    if with_supervisor:
        monkeypatch.setenv(SUP_ENV, str(_write_decisions(tmp)))
    else:
        monkeypatch.delenv(SUP_ENV, raising=False)
    v1rl.run()
    # v1rm
    _redirect(monkeypatch, v1rm, tmp)
    monkeypatch.setattr(v1rm, "IN_TEMPLATE", tmp / v1rg.OUT_TEMPLATE.name)
    monkeypatch.setattr(v1rm, "IN_RESP_VALIDATION", tmp / v1rh.OUT_VALIDATION.name)
    monkeypatch.setattr(v1rm, "IN_RESP_VAL_SUMMARY", tmp / v1rh.OUT_SUMMARY.name)
    monkeypatch.setattr(v1rm, "IN_SCORES", tmp / v1ri.OUT_SCORES.name)
    monkeypatch.setattr(v1rm, "IN_DISAGREE", tmp / v1ri.OUT_DISAGREE.name)
    monkeypatch.setattr(v1rm, "IN_SCORE_SUMMARY", tmp / v1ri.OUT_SUMMARY.name)
    monkeypatch.setattr(v1rm, "IN_SUP_PACKETS", tmp / v1rj.OUT_MANIFEST.name)
    monkeypatch.setattr(v1rm, "IN_SUP_TEMPLATE", tmp / v1rk.OUT_TEMPLATE.name)
    monkeypatch.setattr(v1rm, "IN_SUP_VALIDATION", tmp / v1rl.OUT_VALIDATION.name)
    monkeypatch.setattr(v1rm, "IN_SUP_VAL_SUMMARY", tmp / v1rl.OUT_SUMMARY.name)
    return v1rm.run()

def test_v1rm_waiting_without_responses(monkeypatch, tmp_path):
    out = _full(monkeypatch, tmp_path, with_responses=False)
    assert out["final_status"] == v1rm.ST_WAITING

def test_v1rm_supervisor_packets_ready(monkeypatch, tmp_path):
    out = _full(monkeypatch, tmp_path, with_responses=True, with_supervisor=False)
    assert out["final_status"] == v1rm.ST_PACKETS_READY

def test_v1rm_c3_candidates_with_supervisor(monkeypatch, tmp_path):
    out = _full(monkeypatch, tmp_path, with_responses=True, with_supervisor=True)
    assert out["final_status"] == v1rm.ST_C3
    assert out["c3_review_only"] == 1

def test_v1rm_summary_labels_zero(monkeypatch, tmp_path):
    _full(monkeypatch, tmp_path, with_responses=True)
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1rm.OUT_SUMMARY.name)}
    assert summ["labels_created"] == "0"

def test_v1rm_summary_targets_zero(monkeypatch, tmp_path):
    _full(monkeypatch, tmp_path, with_responses=True)
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1rm.OUT_SUMMARY.name)}
    assert summ["targets_created"] == "0"

def test_v1rm_summary_ground_truth_zero(monkeypatch, tmp_path):
    _full(monkeypatch, tmp_path, with_responses=True)
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1rm.OUT_SUMMARY.name)}
    assert summ["ground_truth_operational_created"] == "0"

def test_v1rm_summary_formal_negative_zero(monkeypatch, tmp_path):
    _full(monkeypatch, tmp_path, with_responses=True, with_supervisor=True)
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1rm.OUT_SUMMARY.name)}
    assert summ["c4_formal_negatives"] == "0"

def test_v1rm_qc_all_pass(monkeypatch, tmp_path):
    _full(monkeypatch, tmp_path, with_responses=True, with_supervisor=True)
    qc = _read(tmp_path / v1rm.OUT_QC.name)
    assert all(c["passed"] == "true" for c in qc)

def test_v1rm_c3_needs_supervisor(monkeypatch, tmp_path):
    _full(monkeypatch, tmp_path, with_responses=True)
    scores = _read(tmp_path / v1ri.OUT_SCORES.name)
    for r in scores:
        if r["recommended_decision"] == RC.C3_NEEDS_SUPERVISOR:
            assert r["supervisor_review_required"] == "true"

def test_v1rm_c3_not_a_label(monkeypatch, tmp_path):
    _full(monkeypatch, tmp_path, with_responses=True, with_supervisor=True)
    # even with C3 candidates, no label/ground-truth created
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1rm.OUT_SUMMARY.name)}
    assert summ["c3_reference_candidates_review_only"] == "1"
    assert summ["labels_created"] == "0"

def test_v1rm_blocked_rows_have_reason(monkeypatch, tmp_path):
    _full(monkeypatch, tmp_path, with_responses=True, with_supervisor=True)
    for name in (v1rh.OUT_VALIDATION.name, v1rl.OUT_VALIDATION.name):
        for r in _read(tmp_path / name):
            if r.get("status") == "FAIL":
                assert r.get("blocked_reason")

def test_v1rm_mandatory_sentence(monkeypatch, tmp_path):
    _full(monkeypatch, tmp_path, with_responses=True)
    doc = (tmp_path / v1rm.DOC.name).read_text(encoding="utf-8")
    assert "permanece review-only" in doc
    assert "nao substitui ground truth validado em campo" in doc

def test_v1rm_tcc_table_exists(monkeypatch, tmp_path):
    _full(monkeypatch, tmp_path, with_responses=True)
    metrics = {r["metric"] for r in _read(tmp_path / v1rm.OUT_TCC.name)}
    assert "final_status" in metrics and "labels_created" in metrics


# ===========================================================================
# Guardrails (negative)
# ===========================================================================

def test_guardrail_label_true():
    with pytest.raises(ValueError):
        G.assert_no_forbidden_true([{"can_create_operational_label": "true"}], "t")

def test_guardrail_train_true():
    with pytest.raises(ValueError):
        G.assert_no_forbidden_true([{"can_train_model": "true"}], "t")

def test_guardrail_target_true():
    with pytest.raises(ValueError):
        G.assert_no_forbidden_true([{"target_created": "true"}], "t")

def test_guardrail_ground_truth_true():
    with pytest.raises(ValueError):
        G.assert_no_forbidden_true([{"ground_truth_operational": "true"}], "t")

def test_guardrail_dino_validates_true():
    with pytest.raises(ValueError):
        G.assert_no_forbidden_true([{"dino_validates_event": "true"}], "t")

def test_guardrail_absence_as_negative_true():
    with pytest.raises(ValueError):
        G.assert_no_forbidden_true([{"absence_as_negative": "true"}], "t")


# ===========================================================================
# Hygiene
# ===========================================================================

def test_no_real_dataset_writes(monkeypatch, tmp_path):
    before = {p.name: p.stat().st_mtime for p in (ROOT / "datasets").glob("*v1rm*")}
    _full(monkeypatch, tmp_path, with_responses=True)
    after = {p.name: p.stat().st_mtime for p in (ROOT / "datasets").glob("*v1rm*")}
    assert before == after

def test_schemas_and_docs_in_tmp(monkeypatch, tmp_path):
    _full(monkeypatch, tmp_path, with_responses=True)
    assert (tmp_path / v1rm.SCHEMA_QC.name).exists()
    assert (tmp_path / v1rm.DOC.name).exists()
