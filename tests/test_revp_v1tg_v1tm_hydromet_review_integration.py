"""Tests — REV-P Protocol C v1tg-v1tm hydromet review integration.

All outputs redirected to tmp_path. No network. No real dataset writes.
"""
from __future__ import annotations
import csv, importlib, os, subprocess, sys
from pathlib import Path
import pytest

ROOT    = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "protocolo_c"
sys.path.insert(0, str(SCRIPTS))

import revp_v1tg_v1tm_hydromet_review_integration_common as C  # noqa: E402

v1tg = importlib.import_module("revp_v1tg_hydromet_evidence_packet_registry")
v1th = importlib.import_module("revp_v1th_hydromet_double_review_addendum")
v1ti = importlib.import_module("revp_v1ti_hydromet_review_scoring_adapter")
v1tj = importlib.import_module("revp_v1tj_supervisor_hydromet_addendum")
v1tk = importlib.import_module("revp_v1tk_tcc_hydromet_review_integration_tables")
v1tl = importlib.import_module("revp_v1tl_hydromet_review_integration_guardrail_audit")
v1tm = importlib.import_module("revp_v1tm_hydromet_review_integration_bundle")


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


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _redirect(monkeypatch, mod, tmp: Path) -> None:
    for name in dir(mod):
        if name.startswith(("OUT_", "SCHEMA_", "DOC", "IN_")):
            val = getattr(mod, name)
            if isinstance(val, Path):
                monkeypatch.setattr(mod, name, tmp / val.name)


def _gr() -> dict[str, str]:
    return C.guardrail_row_extended()


def _write_bridge(tmp: Path, cid: str = "EVT_PET_001", r7d: str = "110.0"):
    gr = _gr()
    rows = [{"hydromet_evidence_id": "HID01", "event_candidate_id": cid,
             "region": "PET", "event_window": "2022-02-12 to 2022-02-20",
             "nearest_station_code": "A636", "nearest_station_distance_km": "4.5",
             "rain_1d": "20.6", "rain_3d": "75.2", "rain_7d": r7d,
             "max_1d_in_window": "25.0", "does_not_validate_event": "true",
             "evidence_role": "HYDROMETEOROLOGICAL_CONTEXT_REVIEW_ONLY", **gr}]
    _write_csv(tmp / "protocol_c_hydromet_event_evidence_bridge_v1td.csv",
               rows, list(rows[0].keys()))


def _write_station_reg(tmp: Path):
    gr = _gr()
    rows = [{"station_code": "A636", "station_name": "PETROPOLIS-NORTE",
             "nearest_region": "PET", "within_100km": "true", **gr}]
    _write_csv(tmp / "protocol_c_inmet_canonical_station_registry_v1ta.csv",
               rows, list(rows[0].keys()))


def _write_windows(tmp: Path, cid: str = "EVT_PET_001"):
    gr = _gr()
    rows = [{"event_window_id": "W001", "event_candidate_id": cid,
             "region": "PET", "parsed_date": "2022-02-19",
             "blocked_reason": "", **gr}]
    _write_csv(tmp / "protocol_c_event_date_windows_v1ss.csv",
               rows, list(rows[0].keys()))


def _write_features(tmp: Path, wid: str = "W001"):
    gr = _gr()
    rows = [{"feature_id": "FID01", "event_window_id": wid,
             "region": "PET", "station_code": "A636",
             "rain_1d": "20.6", "rain_3d": "75.2", "rain_7d": "110.0",
             "max_1d_in_window": "25.0",
             "feature_status": "ROLLING_CONTEXT_REVIEW_ONLY", **gr}]
    _write_csv(tmp / "protocol_c_rolling_rainfall_context_features_v1su.csv",
               rows, list(rows[0].keys()))


def _write_packets(tmp: Path, cid: str = "EVT_PET_001"):
    gr = _gr()
    rows = [{"hydromet_packet_id": "PKT001", "event_candidate_id": cid,
             "region": "PET", "event_window": "...",
             "nearest_station_code": "A636", "nearest_station_name": "PET-N",
             "nearest_station_distance_km": "4.5",
             "rain_1d": "20.6", "rain_3d": "75.2", "rain_7d": "110.0",
             "max_1d_in_window": "25.0",
             "station_coverage_status": "CLOSE_STATION_WITHIN_25KM",
             "precipitation_context_status": "HIGH_RAINFALL_CONTEXT_REVIEW_ONLY",
             "hydromet_support_level": "HYDROMET_CONTEXT_AVAILABLE",
             "evidence_role": "HYDROMETEOROLOGICAL_CONTEXT_REVIEW_ONLY", **gr}]
    _write_csv(tmp / "protocol_c_hydromet_evidence_packet_registry_v1tg.csv",
               rows, list(rows[0].keys()))


# ---------------------------------------------------------------------------
# common — classification helpers
# ---------------------------------------------------------------------------

def test_classify_station_close():
    assert C.classify_station_coverage(10.0) == "CLOSE_STATION_WITHIN_25KM"

def test_classify_station_near():
    assert C.classify_station_coverage(40.0) == "NEAR_STATION_WITHIN_50KM"

def test_classify_station_distant():
    assert C.classify_station_coverage(80.0) == "DISTANT_STATION_WITHIN_100KM"

def test_classify_station_very_distant():
    assert C.classify_station_coverage(200.0) == "VERY_DISTANT_STATION_BEYOND_100KM"

def test_classify_station_no_station():
    assert C.classify_station_coverage(0.0) == "NO_STATION"

def test_classify_precip_high():
    assert C.classify_precipitation_context(200.0, 60.0) == "HIGH_RAINFALL_CONTEXT_REVIEW_ONLY"

def test_classify_precip_moderate():
    assert C.classify_precipitation_context(80.0, 25.0) == "MODERATE_RAINFALL_CONTEXT_REVIEW_ONLY"

def test_classify_precip_low():
    assert C.classify_precipitation_context(10.0, 5.0) == "LOW_RAINFALL_CONTEXT_REVIEW_ONLY"

def test_classify_precip_zero():
    assert C.classify_precipitation_context(0.0, 0.0) == "ZERO_RAINFALL_CONTEXT_REVIEW_ONLY"

def test_classify_precip_missing():
    assert C.classify_precipitation_context(-1.0, -1.0) == "PRECIPITATION_DATA_MISSING"

def test_classify_hydromet_available():
    assert C.classify_hydromet_support_level(10.0, 100.0, True) == "HYDROMET_CONTEXT_AVAILABLE"

def test_classify_hydromet_limited_distance():
    assert C.classify_hydromet_support_level(200.0, 100.0, True) == "HYDROMET_CONTEXT_LIMITED_STATION_DISTANCE"

def test_classify_hydromet_waiting_window():
    assert C.classify_hydromet_support_level(10.0, 100.0, False) == "HYDROMET_CONTEXT_WAITING_EVENT_WINDOW"

def test_classify_hydromet_missing_precip():
    assert C.classify_hydromet_support_level(10.0, -1.0, True) == "HYDROMET_CONTEXT_MISSING_PRECIPITATION"

def test_build_question_set_count():
    qs = C.build_hydromet_question_set("EVT001", "PET", "110", "A636", "4.5")
    assert len(qs) == 7

def test_build_question_set_keys():
    qs = C.build_hydromet_question_set("EVT001", "PET", "110", "A636", "4.5")
    keys = {q["question_key"] for q in qs}
    assert "hydromet_overclaim_risk" in keys
    assert "hydromet_requires_independent_source" in keys

def test_guardrail_row_extended():
    r = C.guardrail_row_extended()
    assert r["hydromet_validates_event"] == "false"
    assert r["hydromet_is_negative_evidence"] == "false"
    assert r["review_only"] == "true"

def test_scan_guardrails_extended_detects_validates():
    issues = C.scan_guardrails_extended([{"hydromet_validates_event": "true"}], "t")
    assert any("hydromet_validates_event" in i for i in issues)

def test_scan_guardrails_extended_detects_negative():
    issues = C.scan_guardrails_extended([{"hydromet_is_negative_evidence": "true"}], "t")
    assert any("hydromet_is_negative_evidence" in i for i in issues)

def test_scan_guardrails_extended_clean():
    assert C.scan_guardrails_extended([C.guardrail_row_extended()], "t") == []


# ---------------------------------------------------------------------------
# v1tg — packet registry
# ---------------------------------------------------------------------------

def test_v1tg_packet_registry_fixture(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tg, tmp_path)
    _write_bridge(tmp_path)
    _write_station_reg(tmp_path)
    _write_windows(tmp_path)
    _write_features(tmp_path)
    monkeypatch.setattr(v1tg, "DATASETS", tmp_path)
    v1tg.run()
    rows = _read(tmp_path / v1tg.OUT_PKT.name)
    assert len(rows) >= 1
    assert rows[0]["hydromet_support_level"] == "HYDROMET_CONTEXT_AVAILABLE"

def test_v1tg_fail_closed_no_bridge(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tg, tmp_path)
    monkeypatch.setattr(v1tg, "DATASETS", tmp_path)
    v1tg.run()
    assert _header(tmp_path / v1tg.OUT_PKT.name) != []

def test_v1tg_never_validated(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tg, tmp_path)
    _write_bridge(tmp_path)
    _write_station_reg(tmp_path)
    _write_windows(tmp_path)
    _write_features(tmp_path)
    monkeypatch.setattr(v1tg, "DATASETS", tmp_path)
    v1tg.run()
    for r in _read(tmp_path / v1tg.OUT_PKT.name):
        assert "validated" not in r.get("hydromet_support_level", "").lower()
        assert r.get("hydromet_validates_event") == "false"

def test_v1tg_no_label(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tg, tmp_path)
    _write_bridge(tmp_path)
    _write_station_reg(tmp_path)
    _write_windows(tmp_path)
    _write_features(tmp_path)
    monkeypatch.setattr(v1tg, "DATASETS", tmp_path)
    v1tg.run()
    for r in _read(tmp_path / v1tg.OUT_PKT.name):
        assert r.get("can_create_operational_label") == "false"


# ---------------------------------------------------------------------------
# v1th — addendum
# ---------------------------------------------------------------------------

def _write_v1qw_manifest(tmp: Path, event_id: str = "EVT_PET_001"):
    _write_csv(tmp / "protocol_c_double_review_packet_manifest_v1qw.csv",
               [{"packet_id": "PKT_A", "event_id": event_id,
                 "review_sample_id": "SMP01", "reviewer_slot": "A",
                 "patch_id": "PET01", "alias": "PET01"}],
               ["packet_id","event_id","review_sample_id","reviewer_slot","patch_id","alias"])


def test_v1th_generates_questions(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1th, tmp_path)
    _write_packets(tmp_path)
    monkeypatch.setattr(v1th, "DATASETS", tmp_path)
    v1th.run()
    forms = _read(tmp_path / v1th.OUT_FORM.name)
    assert len(forms) >= 7

def test_v1th_responses_empty(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1th, tmp_path)
    _write_packets(tmp_path)
    monkeypatch.setattr(v1th, "DATASETS", tmp_path)
    v1th.run()
    for r in _read(tmp_path / v1th.OUT_FORM.name):
        assert r.get("response_value") == ""

def test_v1th_standalone_when_no_v1qw(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1th, tmp_path)
    _write_packets(tmp_path)
    # No v1qw manifest
    _write_csv(tmp_path / "protocol_c_double_review_packet_manifest_v1qw.csv", [],
               ["packet_id","event_id","review_sample_id","reviewer_slot","patch_id","alias"])
    monkeypatch.setattr(v1th, "DATASETS", tmp_path)
    v1th.run()
    manifest = _read(tmp_path / v1th.OUT_MAN.name)
    assert any("STANDALONE" in r.get("addendum_status", "") for r in manifest)

def test_v1th_attached_when_v1qw_matches(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1th, tmp_path)
    _write_packets(tmp_path, cid="EVT_PET_001")
    _write_v1qw_manifest(tmp_path, event_id="EVT_PET_001")
    monkeypatch.setattr(v1th, "DATASETS", tmp_path)
    v1th.run()
    manifest = _read(tmp_path / v1th.OUT_MAN.name)
    assert any(r.get("addendum_status") == "ADDENDUM_ATTACHED" for r in manifest)

def test_v1th_no_label(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1th, tmp_path)
    _write_packets(tmp_path)
    monkeypatch.setattr(v1th, "DATASETS", tmp_path)
    v1th.run()
    for r in _read(tmp_path / v1th.OUT_FORM.name):
        assert r.get("can_create_operational_label") == "false"


# ---------------------------------------------------------------------------
# v1ti — scoring adapter
# ---------------------------------------------------------------------------

def _write_addenda(tmp: Path, cid: str = "EVT_PET_001"):
    gr = _gr()
    rows = [{"addendum_id": "ADD001", "event_candidate_id": cid,
             "region": "PET", "addendum_status": "ADDENDUM_STANDALONE_REVIEW_ONLY", **gr}]
    _write_csv(tmp / "protocol_c_hydromet_double_review_addendum_manifest_v1th.csv",
               rows, list(rows[0].keys()))
    _write_csv(tmp / "protocol_c_hydromet_double_review_addendum_forms_v1th.csv",
               [], ["form_id","addendum_id","event_candidate_id","question_key","response_value"])


def test_v1ti_waiting_without_env(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ti, tmp_path)
    _write_addenda(tmp_path)
    monkeypatch.setattr(v1ti, "DATASETS", tmp_path)
    monkeypatch.delenv("REVP_PROTOCOL_C_HYDROMET_REVIEW_RESPONSES_PATH", raising=False)
    result = v1ti.run()
    assert result["status"] == v1ti.ST_WAITING

def test_v1ti_scores_not_empty_with_responses(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ti, tmp_path)
    _write_addenda(tmp_path)
    # Write a mock responses file
    resp = [{"addendum_id": "ADD001",
             "question_key": "hydromet_precipitation_present",
             "response_value": "sim"}]
    resp_path = tmp_path / "responses.csv"
    _write_csv(resp_path, resp, ["addendum_id","question_key","response_value"])
    monkeypatch.setenv("REVP_PROTOCOL_C_HYDROMET_REVIEW_RESPONSES_PATH", str(resp_path))
    monkeypatch.setattr(v1ti, "DATASETS", tmp_path)
    v1ti.run()
    rows = _read(tmp_path / v1ti.OUT_SCR.name)
    assert any(r.get("hydromet_context_score") != "" for r in rows)

def test_v1ti_no_label(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ti, tmp_path)
    _write_addenda(tmp_path)
    monkeypatch.setattr(v1ti, "DATASETS", tmp_path)
    monkeypatch.delenv("REVP_PROTOCOL_C_HYDROMET_REVIEW_RESPONSES_PATH", raising=False)
    v1ti.run()
    for r in _read(tmp_path / v1ti.OUT_SCR.name):
        assert r.get("can_create_operational_label") == "false"
        assert r.get("target_created") == "false"
        assert r.get("ground_truth_operational") == "false"
        assert r.get("formal_negative") == "false"

def test_v1ti_absence_not_negative(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ti, tmp_path)
    _write_addenda(tmp_path)
    monkeypatch.setattr(v1ti, "DATASETS", tmp_path)
    monkeypatch.delenv("REVP_PROTOCOL_C_HYDROMET_REVIEW_RESPONSES_PATH", raising=False)
    v1ti.run()
    for r in _read(tmp_path / v1ti.OUT_SCR.name):
        assert r.get("absence_as_negative") == "false"

def test_v1ti_does_not_influence_c4(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1ti, tmp_path)
    _write_addenda(tmp_path)
    monkeypatch.setattr(v1ti, "DATASETS", tmp_path)
    monkeypatch.delenv("REVP_PROTOCOL_C_HYDROMET_REVIEW_RESPONSES_PATH", raising=False)
    v1ti.run()
    summ = {r["stat_key"]: r["stat_value"] for r in _read(tmp_path / v1ti.OUT_SUM.name)}
    assert summ["influences_c4"] == "false"


# ---------------------------------------------------------------------------
# v1tj — supervisor addendum
# ---------------------------------------------------------------------------

def _write_scores(tmp: Path, cid: str = "EVT_PET_001"):
    gr = _gr()
    rows = [{"score_id": "SC001", "event_candidate_id": cid,
             "region": "PET", "addendum_id": "ADD001",
             "overclaim_risk_score": "0.8",
             "scoring_status": "HYDROMET_REVIEW_RESPONSES_WAITING", **gr}]
    _write_csv(tmp / "protocol_c_hydromet_review_scores_v1ti.csv",
               rows, list(rows[0].keys()))


def test_v1tj_standalone_when_no_supervisor(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tj, tmp_path)
    _write_packets(tmp_path)
    _write_scores(tmp_path)
    # Empty supervisor manifest
    _write_csv(tmp_path / "protocol_c_supervisor_review_packet_manifest_v1rj.csv",
               [], ["packet_id","event_id"])
    monkeypatch.setattr(v1tj, "DATASETS", tmp_path)
    v1tj.run()
    rows = _read(tmp_path / v1tj.OUT_ADD.name)
    assert any("standalone" in r.get("notes","").lower() for r in rows)

def test_v1tj_independent_source_required(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tj, tmp_path)
    _write_packets(tmp_path)
    _write_scores(tmp_path)
    _write_csv(tmp_path / "protocol_c_supervisor_review_packet_manifest_v1rj.csv",
               [], ["packet_id","event_id"])
    monkeypatch.setattr(v1tj, "DATASETS", tmp_path)
    v1tj.run()
    for r in _read(tmp_path / v1tj.OUT_ADD.name):
        assert r.get("independent_observational_source_still_required") == "true"

def test_v1tj_does_not_validate(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tj, tmp_path)
    _write_packets(tmp_path)
    _write_scores(tmp_path)
    _write_csv(tmp_path / "protocol_c_supervisor_review_packet_manifest_v1rj.csv",
               [], ["packet_id","event_id"])
    monkeypatch.setattr(v1tj, "DATASETS", tmp_path)
    v1tj.run()
    for r in _read(tmp_path / v1tj.OUT_ADD.name):
        assert r.get("does_not_validate_event") == "true"
        assert r.get("hydromet_validates_event") == "false"


# ---------------------------------------------------------------------------
# v1tk — TCC tables
# ---------------------------------------------------------------------------

def _write_v1tk_summaries(tmp: Path):
    for fn, kv in [
        ("protocol_c_hydromet_evidence_packet_summary_v1tg.csv",
         [("total_packets","9"),("context_available","9")]),
        ("protocol_c_hydromet_double_review_addendum_summary_v1th.csv",
         [("addenda_total","9"),("form_rows","63"),("responses_empty","63")]),
        ("protocol_c_hydromet_review_scoring_summary_v1ti.csv",
         [("score_rows","9"),("overall_status","HYDROMET_REVIEW_RESPONSES_WAITING")]),
        ("protocol_c_supervisor_hydromet_addendum_summary_v1tj.csv",
         [("addenda_total","9"),("independent_source_required","9")]),
    ]:
        _write_csv(tmp / fn, [{"stat_key": k,"stat_value": v} for k,v in kv],
                   ["stat_key","stat_value"])


def test_v1tk_tables_exist(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tk, tmp_path)
    _write_v1tk_summaries(tmp_path)
    monkeypatch.setattr(v1tk, "DATASETS", tmp_path)
    v1tk.run()
    assert (tmp_path / v1tk.OUT_PKT.name).exists()
    assert (tmp_path / v1tk.OUT_SUP.name).exists()
    assert (tmp_path / v1tk.OUT_OC.name).exists()

def test_v1tk_overclaim_controls(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tk, tmp_path)
    _write_v1tk_summaries(tmp_path)
    monkeypatch.setattr(v1tk, "DATASETS", tmp_path)
    v1tk.run()
    oc = _read(tmp_path / v1tk.OUT_OC.name)
    assert len(oc) >= 4
    aspects = [r.get("aspect","") for r in oc]
    assert "absence_is_not_negative" in aspects
    assert "c3_requires_independent_source" in aspects


# ---------------------------------------------------------------------------
# v1tl — guardrail audit
# ---------------------------------------------------------------------------

def _write_clean_v1tg_v1tk(tmp: Path):
    gr = _gr()
    targets = [
        "protocol_c_hydromet_evidence_packet_registry_v1tg.csv",
        "protocol_c_hydromet_evidence_packet_summary_v1tg.csv",
        "protocol_c_hydromet_double_review_addendum_manifest_v1th.csv",
        "protocol_c_hydromet_double_review_addendum_forms_v1th.csv",
        "protocol_c_hydromet_review_scores_v1ti.csv",
        "protocol_c_supervisor_hydromet_addendum_v1tj.csv",
        "protocol_c_tcc_table_hydromet_review_packets_v1tk.csv",
        "protocol_c_tcc_table_hydromet_supervisor_addendum_v1tk.csv",
        "protocol_c_tcc_table_hydromet_overclaim_controls_v1tk.csv",
    ]
    for fn in targets:
        _write_csv(tmp / fn, [{"region":"PET","review_only":"true",**gr}],
                   ["region","review_only"]+list(gr.keys()))


def test_v1tl_all_pass_clean(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tl, tmp_path)
    _write_clean_v1tg_v1tk(tmp_path)
    monkeypatch.setattr(v1tl, "DATASETS", tmp_path)
    v1tl.run()
    summ = {r["stat_key"]:r["stat_value"] for r in _read(tmp_path / v1tl.OUT_SUM.name)}
    assert summ["audit_status"] == "GUARDRAIL_PASS_ALL"

def test_v1tl_detects_hydromet_validates(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tl, tmp_path)
    bad = {"region":"PET","review_only":"true","hydromet_validates_event":"true",
           "hydromet_is_negative_evidence":"false"}
    _write_csv(tmp_path / "protocol_c_hydromet_evidence_packet_registry_v1tg.csv",
               [bad], list(bad.keys()))
    for fn in [t for t in [
        "protocol_c_hydromet_evidence_packet_summary_v1tg.csv",
        "protocol_c_hydromet_double_review_addendum_manifest_v1th.csv",
        "protocol_c_hydromet_double_review_addendum_forms_v1th.csv",
        "protocol_c_hydromet_review_scores_v1ti.csv",
        "protocol_c_supervisor_hydromet_addendum_v1tj.csv",
        "protocol_c_tcc_table_hydromet_review_packets_v1tk.csv",
        "protocol_c_tcc_table_hydromet_supervisor_addendum_v1tk.csv",
        "protocol_c_tcc_table_hydromet_overclaim_controls_v1tk.csv",
    ]]:
        _write_csv(tmp_path / fn, [], ["region"])
    monkeypatch.setattr(v1tl, "DATASETS", tmp_path)
    v1tl.run()
    audit = _read(tmp_path / v1tl.OUT_AUD.name)
    assert any(r.get("audit_status") == "FAIL" for r in audit)

def test_v1tl_detects_hydromet_is_negative(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tl, tmp_path)
    bad = {"region":"PET","review_only":"true","hydromet_validates_event":"false",
           "hydromet_is_negative_evidence":"true"}
    _write_csv(tmp_path / "protocol_c_hydromet_evidence_packet_registry_v1tg.csv",
               [bad], list(bad.keys()))
    for fn in [
        "protocol_c_hydromet_evidence_packet_summary_v1tg.csv",
        "protocol_c_hydromet_double_review_addendum_manifest_v1th.csv",
        "protocol_c_hydromet_double_review_addendum_forms_v1th.csv",
        "protocol_c_hydromet_review_scores_v1ti.csv",
        "protocol_c_supervisor_hydromet_addendum_v1tj.csv",
        "protocol_c_tcc_table_hydromet_review_packets_v1tk.csv",
        "protocol_c_tcc_table_hydromet_supervisor_addendum_v1tk.csv",
        "protocol_c_tcc_table_hydromet_overclaim_controls_v1tk.csv",
    ]:
        _write_csv(tmp_path / fn, [], ["region"])
    monkeypatch.setattr(v1tl, "DATASETS", tmp_path)
    v1tl.run()
    audit = _read(tmp_path / v1tl.OUT_AUD.name)
    assert any(r.get("audit_status") == "FAIL" for r in audit)


# ---------------------------------------------------------------------------
# v1tm — bundle
# ---------------------------------------------------------------------------

def _write_all_tm_summaries(tmp: Path, guardrail_ok: bool = True, waiting: bool = True):
    g = "GUARDRAIL_PASS_ALL" if guardrail_ok else "GUARDRAIL_FAIL_CLOSED"
    sc_st = "HYDROMET_REVIEW_RESPONSES_WAITING" if waiting else "HYDROMET_REVIEW_CONTEXT_SCORED_REVIEW_ONLY"
    for fn, kv in [
        ("protocol_c_hydromet_evidence_packet_summary_v1tg.csv",
         [("total_packets","9"),("context_available","9")]),
        ("protocol_c_hydromet_double_review_addendum_summary_v1th.csv",
         [("addenda_total","9"),("form_rows","63"),("responses_empty","63")]),
        ("protocol_c_hydromet_review_scoring_summary_v1ti.csv",
         [("score_rows","9"),("overall_status",sc_st),("influences_label","false"),("influences_c4","false")]),
        ("protocol_c_supervisor_hydromet_addendum_summary_v1tj.csv",
         [("addenda_total","9"),("independent_source_required","9"),("validates_event","false")]),
        ("protocol_c_hydromet_review_integration_guardrail_summary_v1tl.csv",
         [("audit_status",g),("total_violations","0" if guardrail_ok else "1"),("files_audited","9")]),
    ]:
        _write_csv(tmp / fn, [{"stat_key":k,"stat_value":v} for k,v in kv],
                   ["stat_key","stat_value"])


def test_v1tm_status_ready(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tm, tmp_path)
    _write_all_tm_summaries(tmp_path, guardrail_ok=True, waiting=False)
    for attr in ("IN_TG","IN_TH","IN_TI","IN_TJ","IN_TL"):
        monkeypatch.setattr(v1tm, attr, tmp_path / getattr(v1tm, attr).name)
    result = v1tm.run()
    assert result["final_status"] == v1tm.ST_READY

def test_v1tm_status_waiting(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tm, tmp_path)
    _write_all_tm_summaries(tmp_path, guardrail_ok=True, waiting=True)
    for attr in ("IN_TG","IN_TH","IN_TI","IN_TJ","IN_TL"):
        monkeypatch.setattr(v1tm, attr, tmp_path / getattr(v1tm, attr).name)
    result = v1tm.run()
    assert result["final_status"] == v1tm.ST_WAIT_RESPONSES

def test_v1tm_guardrail_fail_closed(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tm, tmp_path)
    _write_all_tm_summaries(tmp_path, guardrail_ok=False, waiting=True)
    for attr in ("IN_TG","IN_TH","IN_TI","IN_TJ","IN_TL"):
        monkeypatch.setattr(v1tm, attr, tmp_path / getattr(v1tm, attr).name)
    result = v1tm.run()
    assert result["final_status"] == v1tm.ST_GUARDRAIL

def test_v1tm_zero_labels(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tm, tmp_path)
    _write_all_tm_summaries(tmp_path, waiting=True)
    for attr in ("IN_TG","IN_TH","IN_TI","IN_TJ","IN_TL"):
        monkeypatch.setattr(v1tm, attr, tmp_path / getattr(v1tm, attr).name)
    v1tm.run()
    summ = {r["stat_key"]:r["stat_value"] for r in _read(tmp_path / v1tm.OUT_SUM.name)}
    assert summ["labels_created"] == "0"
    assert summ["targets_created"] == "0"
    assert summ["ground_truth_operational_created"] == "0"
    assert summ["c4_formal_negatives"] == "0"

def test_v1tm_mandatory_clause(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tm, tmp_path)
    _write_all_tm_summaries(tmp_path)
    for attr in ("IN_TG","IN_TH","IN_TI","IN_TJ","IN_TL"):
        monkeypatch.setattr(v1tm, attr, tmp_path / getattr(v1tm, attr).name)
    v1tm.run()
    doc = (tmp_path / v1tm.DOC.name).read_text(encoding="utf-8")
    assert "v1tg" in doc and "v1tm" in doc
    assert "observacional independente" in doc


# ---------------------------------------------------------------------------
# Schema / doc / hygiene
# ---------------------------------------------------------------------------

def test_v1tg_schema_exists(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tg, tmp_path)
    monkeypatch.setattr(v1tg, "DATASETS", tmp_path)
    v1tg.run()
    assert (tmp_path / v1tg.SCHEMA_P.name).exists()

def test_v1tg_doc_exists(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tg, tmp_path)
    monkeypatch.setattr(v1tg, "DATASETS", tmp_path)
    v1tg.run()
    assert (tmp_path / v1tg.DOC.name).exists()

def test_no_real_dataset_writes(monkeypatch, tmp_path):
    before = {p.name: p.stat().st_mtime for p in (ROOT/"datasets").glob("*v1tg*")}
    _redirect(monkeypatch, v1tg, tmp_path)
    monkeypatch.setattr(v1tg, "DATASETS", tmp_path)
    v1tg.run()
    after = {p.name: p.stat().st_mtime for p in (ROOT/"datasets").glob("*v1tg*")}
    assert before == after

def test_staged_empty():
    result = subprocess.run(
        ["git","diff","--cached","--name-only"],
        cwd=str(ROOT), capture_output=True, text=True
    )
    assert result.stdout.strip() == ""
