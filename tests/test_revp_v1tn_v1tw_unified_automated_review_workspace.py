"""Tests — REV-P Protocol C v1tn-v1tw Automated Review Adjudication.

All outputs redirected to tmp_path. No network. No real dataset writes.
"""
from __future__ import annotations
import csv, importlib, sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "protocolo_c"
sys.path.insert(0, str(SCRIPTS))

import revp_v1tn_v1tw_automated_review_common as C  # noqa: E402

v1tn = importlib.import_module("revp_v1tn_unified_evidence_case_index")
v1to = importlib.import_module("revp_v1to_unified_single_case_workspace")
v1tp = importlib.import_module("revp_v1tp_automated_reviewer_ab_adjudication")
v1tq = importlib.import_module("revp_v1tq_review_consensus_divergence_adjudication")
v1tr = importlib.import_module("revp_v1tr_automated_supervisor_adjudication")
v1ts = importlib.import_module("revp_v1ts_single_flow_review_export")
v1tt = importlib.import_module("revp_v1tt_automated_review_tcc_tables")
v1tu = importlib.import_module("revp_v1tu_proof_of_review_only_validation_audit")
v1tv = importlib.import_module("revp_v1tv_unified_review_guardrail_audit")
v1tw = importlib.import_module("revp_v1tw_unified_automated_review_bundle")

ALL_MODULES = [v1tn, v1to, v1tp, v1tq, v1tr, v1ts, v1tt, v1tu, v1tv, v1tw]


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
    monkeypatch.setattr(mod, "DATASETS", tmp)


def _gr() -> dict[str, str]:
    return C.guardrail_row_review()


# ----- fixture writers -------------------------------------------------------

def _write_packet(tmp: Path, cid: str = "EVT_PET_001", region: str = "PET",
                  support: str = "HYDROMET_CONTEXT_AVAILABLE"):
    rows = [{
        "hydromet_packet_id": "PKT01", "event_candidate_id": cid,
        "region": region, "event_window": "2022-02-12 to 2022-02-20",
        "nearest_station_code": "A636", "nearest_station_name": "PET-N",
        "nearest_station_distance_km": "4.5", "rain_1d": "20", "rain_3d": "75",
        "rain_7d": "110", "max_1d_in_window": "25",
        "station_coverage_status": "CLOSE_STATION_WITHIN_25KM",
        "precipitation_context_status": "HIGH_RAINFALL_CONTEXT_REVIEW_ONLY",
        "hydromet_support_level": support, "hazard_type": "FLOOD_LANDSLIDE",
        "evidence_role": "HYDROMETEOROLOGICAL_CONTEXT_REVIEW_ONLY",
    }]
    _write_csv(tmp / "protocol_c_hydromet_evidence_packet_registry_v1tg.csv",
               rows, list(rows[0].keys()))


def _write_window(tmp: Path, cid: str = "EVT_PET_001", region: str = "PET",
                  blocked: str = ""):
    rows = [{
        "event_window_id": "W001", "event_candidate_id": cid, "region": region,
        "hazard_type": "FLOOD_LANDSLIDE", "event_date_text": "19/02/2022",
        "parsed_date": "2022-02-19", "window_start": "2022-02-12",
        "window_end": "2022-02-20", "window_days": "9",
        "temporal_precision_status": "DATE_PARSED_OK", "source_block": "v1ir",
        "blocked_reason": blocked,
    }]
    _write_csv(tmp / "protocol_c_event_date_windows_v1ss.csv",
               rows, list(rows[0].keys()))


def _write_backlog(tmp: Path, region: str = "PET"):
    rows = [{
        "backlog_id": "BL01", "region": region, "event_id": "", "patch_id": "",
        "hazard_type": "LANDSLIDE", "missing_source_family": "OFFICIAL_HYDROMETEOROLOGICAL",
        "missing_source_name": "CEMADEN", "blocker": "EXTERNAL_SOURCE_NOT_LOCAL",
        "next_action": "MANUAL", "blocks_c3": "true", "blocks_c4": "false",
        "priority": "P0", "current_state": "BLOCKED_INSUFFICIENT_EVIDENCE",
        "status": "OPEN",
    }]
    _write_csv(tmp / "protocol_c_ground_reference_evidence_backlog_v1ro.csv",
               rows, list(rows[0].keys()))


def _write_candidates(tmp: Path, cid: str = "", status: str = "ACCEPTED",
                      region: str = "PET"):
    rows = []
    if cid:
        rows = [{
            "event_candidate_id": cid, "document_id": "DOC1", "region": region,
            "hazard_type": "FLOOD", "source_name": "DefesaCivil",
            "source_family": "OFFICIAL_CIVIL_DEFENSE", "event_date_text": "19/02/2022",
            "event_location_text": "Centro", "temporal_precision_claim": "DAY",
            "spatial_precision_claim": "NEIGHBORHOOD", "license_status": "OPEN",
            "candidate_status": status,
        }]
    _write_csv(tmp / "protocol_c_external_event_candidates_v1rd.csv", rows,
               ["event_candidate_id", "document_id", "region", "hazard_type",
                "source_name", "source_family", "event_date_text",
                "event_location_text", "temporal_precision_claim",
                "spatial_precision_claim", "license_status", "candidate_status"])


def _write_links(tmp: Path, cid: str = "", patch_id: str = "P001",
                 conf: str = "HIGH", region: str = "PET"):
    rows = []
    if cid:
        rows = [{
            "link_candidate_id": "LC1", "event_candidate_id": cid,
            "patch_id": patch_id, "region": region, "hazard_type": "FLOOD",
            "link_basis": "COORD", "link_confidence": conf,
            "link_status": "CANDIDATE",
        }]
    _write_csv(tmp / "protocol_c_external_event_patch_link_candidates_v1re.csv",
               rows, ["link_candidate_id", "event_candidate_id", "patch_id",
                      "region", "hazard_type", "link_basis", "link_confidence",
                      "link_status"])


def _setup_inputs(tmp: Path, **kw):
    cid = kw.get("cid", "EVT_PET_001")
    _write_packet(tmp, cid, support=kw.get("support", "HYDROMET_CONTEXT_AVAILABLE"))
    _write_window(tmp, cid, blocked=kw.get("blocked", ""))
    _write_backlog(tmp)
    _write_candidates(tmp, kw.get("ext_cid", ""), kw.get("ext_status", "ACCEPTED"))
    _write_links(tmp, kw.get("link_cid", ""), conf=kw.get("link_conf", "HIGH"))


def _run_chain(monkeypatch, tmp: Path, **kw):
    _setup_inputs(tmp, **kw)
    for mod in ALL_MODULES:
        _redirect(monkeypatch, mod, tmp)
    for mod in ALL_MODULES:
        mod.run()


# ===========================================================================
# 0. staged area
# ===========================================================================

def test_staged_area_empty_at_start():
    import subprocess
    out = subprocess.run(["git", "diff", "--cached", "--name-only"],
                         cwd=ROOT, capture_output=True, text=True)
    assert out.stdout.strip() == "", f"staged not empty: {out.stdout}"


# ===========================================================================
# 1. common — guardrails / normalisers
# ===========================================================================

def test_guardrail_row_review_required_true():
    r = _gr()
    assert r["review_only"] == "true"
    assert r["automated_review"] == "true"
    assert r["internal_review_automated_for_review_only"] == "true"
    assert r["requires_external_observational_evidence_for_operational_claim"] == "true"

def test_guardrail_row_review_forbidden_false():
    r = _gr()
    for f in C.FORBIDDEN_TRUE_FLAGS:
        assert r[f] == "false"

def test_scan_guardrails_clean():
    assert C.scan_guardrails([_gr()], "t") == []

def test_scan_guardrails_detects_label():
    bad = _gr(); bad["can_create_operational_label"] = "true"
    assert any("can_create_operational_label" in i for i in C.scan_guardrails([bad], "t"))

def test_scan_guardrails_detects_c4():
    bad = _gr(); bad["c4_opened"] = "true"
    assert any("c4_opened" in i for i in C.scan_guardrails([bad], "t"))

def test_scan_guardrails_detects_c3():
    bad = _gr(); bad["automatic_c3_promotion"] = "true"
    assert any("automatic_c3_promotion" in i for i in C.scan_guardrails([bad], "t"))

def test_scan_guardrails_detects_hydromet_validates():
    bad = _gr(); bad["hydromet_validates_event"] = "true"
    assert any("hydromet_validates_event" in i for i in C.scan_guardrails([bad], "t"))

def test_scan_guardrails_detects_dino_validates():
    bad = _gr(); bad["dino_validates_event"] = "true"
    assert any("dino_validates_event" in i for i in C.scan_guardrails([bad], "t"))

def test_scan_guardrails_detects_absence_negative():
    bad = _gr(); bad["absence_as_negative"] = "true"
    assert any("absence_as_negative" in i for i in C.scan_guardrails([bad], "t"))

def test_scan_guardrails_detects_required_false():
    bad = _gr(); bad["review_only"] = "false"
    assert any("review_only" in i for i in C.scan_guardrails([bad], "t"))

def test_scan_guardrails_detects_abs_path():
    bad = _gr(); bad["notes"] = r"C:\Users\x"
    assert any("abs_path" in i for i in C.scan_guardrails([bad], "t"))

def test_scan_guardrails_detects_local_runs():
    bad = _gr(); bad["notes"] = "local" + "_runs/foo"
    assert any("local_runs_exposure" in i for i in C.scan_guardrails([bad], "t"))

def test_normalize_case_id():
    assert C.normalize_case_id("case x 1") == "CASE_X_1"

def test_normalize_bool_true():
    assert C.normalize_bool("sim") == "true"

def test_normalize_bool_false():
    assert C.normalize_bool("no") == "false"


# ===========================================================================
# 2. common — summarisers
# ===========================================================================

def test_summarize_external_absent():
    assert C.summarize_external_evidence([]) == "EXTERNAL_SOURCE_ABSENT_LOCAL"

def test_summarize_external_present():
    assert C.summarize_external_evidence(
        [{"candidate_status": "ACCEPTED"}]) == "EXTERNAL_CANDIDATE_PRESENT_REVIEW_ONLY"

def test_summarize_external_weak():
    assert C.summarize_external_evidence(
        [{"candidate_status": "DRAFT"}]) == "EXTERNAL_CANDIDATE_WEAK_REVIEW_ONLY"

def test_summarize_hydromet_available():
    assert C.summarize_hydromet_evidence(
        {"hydromet_support_level": "HYDROMET_CONTEXT_AVAILABLE"}) \
        == "HYDROMET_CONTEXT_AVAILABLE_REVIEW_ONLY"

def test_summarize_hydromet_absent():
    assert C.summarize_hydromet_evidence(None) == "HYDROMET_CONTEXT_ABSENT"

def test_summarize_dino_review_only():
    assert C.summarize_dino_role([{"x": "1"}]) == "DINO_REPRESENTATION_REVIEW_ONLY_CONTEXT"

def test_summarize_dino_absent():
    assert C.summarize_dino_role([]) == "DINO_NOT_PRESENT_CONTEXT_ONLY"

def test_summarize_patch_absent():
    assert C.summarize_patch_context([]) == "PATCH_LINK_ABSENT"

def test_summarize_patch_present():
    assert C.summarize_patch_context(
        [{"link_confidence": "HIGH"}]) == "PATCH_LINK_CANDIDATE_REVIEW_ONLY"

def test_summarize_protocol_c_blocked():
    bl = [{"region": "PET", "current_state": "BLOCKED_INSUFFICIENT_EVIDENCE",
           "status": "OPEN"}]
    assert C.summarize_protocol_c_state("PET", bl) == "PROTOCOL_C_BLOCKED_INSUFFICIENT_EVIDENCE"

def test_summarize_protocol_c_no_record():
    assert C.summarize_protocol_c_state("RECIFE", []) == "PROTOCOL_C_NO_BACKLOG_RECORD"


# ===========================================================================
# 3. common — readiness / reviewer / consensus / supervisor classifiers
# ===========================================================================

def test_classify_case_ready():
    s = C.classify_case_readiness(
        "EXTERNAL_CANDIDATE_PRESENT_REVIEW_ONLY",
        "HYDROMET_CONTEXT_AVAILABLE_REVIEW_ONLY", "win", "DATE_PARSED_OK",
        "PATCH_LINK_CANDIDATE_REVIEW_ONLY")
    assert s == "CASE_READY_FOR_REVIEW_ONLY_ADJUDICATION"

def test_classify_case_needs_external():
    s = C.classify_case_readiness(
        "EXTERNAL_SOURCE_ABSENT_LOCAL",
        "HYDROMET_CONTEXT_AVAILABLE_REVIEW_ONLY", "win", "DATE_PARSED_OK",
        "PATCH_LINK_ABSENT")
    assert s == "CASE_CONTEXT_AVAILABLE_NEEDS_EXTERNAL_SOURCE"

def test_classify_case_blocked_no_window():
    s = C.classify_case_readiness("EXTERNAL_SOURCE_ABSENT_LOCAL",
                                  "HYDROMET_CONTEXT_ABSENT", "", "", "PATCH_LINK_ABSENT")
    assert s == "CASE_BLOCKED_INSUFFICIENT_EVIDENCE"

def test_next_required_action_external():
    assert C.next_required_action("CASE_CONTEXT_AVAILABLE_NEEDS_EXTERNAL_SOURCE") \
        == "COLLECT_EXTERNAL_OBSERVATIONAL_SOURCE_FOR_OPERATIONAL_CLAIM"

def test_blocking_factors_lists_external():
    bf = C.blocking_factors("EXTERNAL_SOURCE_ABSENT_LOCAL",
                            "HYDROMET_CONTEXT_AVAILABLE_REVIEW_ONLY", "win",
                            "DATE_PARSED_OK", "PATCH_LINK_ABSENT")
    assert "NO_EXTERNAL_OBSERVATIONAL_SOURCE" in bf
    assert "NO_PATCH_LINK" in bf

def test_reviewer_a_more_conservative_than_b():
    args = ("EXTERNAL_SOURCE_ABSENT_LOCAL", "HYDROMET_CONTEXT_AVAILABLE_REVIEW_ONLY",
            "win", "DATE_PARSED_OK", "PATCH_LINK_ABSENT", "DINO_NOT_PRESENT_CONTEXT_ONLY")
    a = C.reviewer_dimensions("conservative", *args)
    b = C.reviewer_dimensions("integrator", *args)
    assert float(a["review_only_confidence_score"]) < float(b["review_only_confidence_score"])

def test_reviewer_dino_always_limited():
    d = C.reviewer_dimensions("integrator", "EXTERNAL_SOURCE_ABSENT_LOCAL",
                              "HYDROMET_CONTEXT_AVAILABLE_REVIEW_ONLY", "win",
                              "DATE_PARSED_OK", "PATCH_LINK_ABSENT",
                              "DINO_REPRESENTATION_REVIEW_ONLY_CONTEXT")
    assert d["dino_role_correctly_limited"] == "true"

def test_classify_review_decision_needs_spatial_conservative():
    dims = C.reviewer_dimensions("conservative", "EXTERNAL_SOURCE_ABSENT_LOCAL",
                                 "HYDROMET_CONTEXT_AVAILABLE_REVIEW_ONLY", "win",
                                 "DATE_PARSED_OK", "PATCH_LINK_ABSENT",
                                 "DINO_NOT_PRESENT_CONTEXT_ONLY")
    assert C.classify_automated_review_decision("conservative", dims) \
        == "AUTOMATED_REVIEW_NEEDS_SPATIAL_PRECISION"

def test_classify_review_decision_validated_integrator():
    dims = C.reviewer_dimensions("integrator", "EXTERNAL_SOURCE_ABSENT_LOCAL",
                                 "HYDROMET_CONTEXT_AVAILABLE_REVIEW_ONLY", "win",
                                 "DATE_PARSED_OK", "PATCH_LINK_ABSENT",
                                 "DINO_NOT_PRESENT_CONTEXT_ONLY")
    assert C.classify_automated_review_decision("integrator", dims) \
        == "AUTOMATED_REVIEW_VALIDATED_FOR_REVIEW_ONLY_USE"

def test_classify_review_decisions_in_allowed_set():
    for prof in ("conservative", "integrator"):
        dims = C.reviewer_dimensions(prof, "EXTERNAL_SOURCE_ABSENT_LOCAL",
                                     "HYDROMET_CONTEXT_AVAILABLE_REVIEW_ONLY", "win",
                                     "DATE_PARSED_OK", "PATCH_LINK_ABSENT",
                                     "DINO_NOT_PRESENT_CONTEXT_ONLY")
        assert C.classify_automated_review_decision(prof, dims) in C.REVIEWER_DECISIONS

def test_consensus_validated():
    s, d = C.classify_consensus("AUTOMATED_REVIEW_VALIDATED_FOR_REVIEW_ONLY_USE",
                                "AUTOMATED_REVIEW_VALIDATED_FOR_REVIEW_ONLY_USE")
    assert s == "AUTOMATED_CONSENSUS_VALIDATED_FOR_REVIEW_ONLY_USE"

def test_consensus_divergence():
    s, d = C.classify_consensus("AUTOMATED_REVIEW_NEEDS_SPATIAL_PRECISION",
                                "AUTOMATED_REVIEW_VALIDATED_FOR_REVIEW_ONLY_USE")
    assert s == "AUTOMATED_DIVERGENCE_REQUIRES_SUPERVISOR_ADJUDICATION"
    assert d != "NONE"

def test_consensus_needs_external():
    s, _ = C.classify_consensus("AUTOMATED_REVIEW_CONTEXT_OK_BUT_NEEDS_EXTERNAL_SOURCE",
                                "AUTOMATED_REVIEW_CONTEXT_OK_BUT_NEEDS_EXTERNAL_SOURCE")
    assert s == "AUTOMATED_CONSENSUS_BLOCKED_NEEDS_EXTERNAL_SOURCE"

def test_consensus_temporal_spatial():
    s, _ = C.classify_consensus("AUTOMATED_REVIEW_NEEDS_TEMPORAL_PRECISION",
                                "AUTOMATED_REVIEW_NEEDS_SPATIAL_PRECISION")
    assert s == "AUTOMATED_CONSENSUS_BLOCKED_TEMPORAL_SPATIAL"

def test_consensus_in_allowed_set():
    s, _ = C.classify_consensus("AUTOMATED_REVIEW_OVERCLAIM_RISK", "AUTOMATED_REVIEW_OVERCLAIM_RISK")
    assert s in C.CONSENSUS_STATUSES

def test_supervisor_adjudication_required_true():
    assert C.supervisor_adjudication_required(
        "AUTOMATED_DIVERGENCE_REQUIRES_SUPERVISOR_ADJUDICATION") == "true"

def test_supervisor_precheck_pass():
    assert C.classify_supervisor_precheck(True) == "SUPERVISOR_PRECHECK_PASS"

def test_supervisor_precheck_fail():
    assert C.classify_supervisor_precheck(False) == "SUPERVISOR_PRECHECK_FAIL_GUARDRAIL"

def test_supervisor_decision_ready_tcc_on_consensus():
    d = C.classify_supervisor_decision(
        "AUTOMATED_CONSENSUS_VALIDATED_FOR_REVIEW_ONLY_USE", 0.8, True, False)
    assert d == "AUTOMATED_SUPERVISOR_READY_FOR_TCC_DISCUSSION"

def test_supervisor_decision_divergence_validated():
    d = C.classify_supervisor_decision(
        "AUTOMATED_DIVERGENCE_REQUIRES_SUPERVISOR_ADJUDICATION", 0.8, True, False)
    assert d == "AUTOMATED_SUPERVISOR_VALIDATED_FOR_REVIEW_ONLY_USE"

def test_supervisor_decision_waiting_external():
    d = C.classify_supervisor_decision(
        "AUTOMATED_CONSENSUS_BLOCKED_NEEDS_EXTERNAL_SOURCE", 0.4, True, False)
    assert d == "AUTOMATED_SUPERVISOR_WAITING_EXTERNAL_OBSERVATIONAL_SOURCE"

def test_supervisor_decision_overclaim_blocked():
    d = C.classify_supervisor_decision(
        "AUTOMATED_CONSENSUS_OVERCLAIM_RISK", 0.8, True, False)
    assert d == "AUTOMATED_SUPERVISOR_BLOCKED_OVERCLAIM_RISK"

def test_supervisor_decisions_in_allowed_set():
    d = C.classify_supervisor_decision(
        "AUTOMATED_CONSENSUS_BLOCKED_TEMPORAL_SPATIAL", 0.1, True, False)
    assert d in C.SUPERVISOR_DECISIONS

def test_supervisor_final_for_review_only():
    assert C.supervisor_final_for_review_only(
        "AUTOMATED_SUPERVISOR_VALIDATED_FOR_REVIEW_ONLY_USE") == "true"
    assert C.supervisor_final_for_review_only(
        "AUTOMATED_SUPERVISOR_WAITING_EXTERNAL_OBSERVATIONAL_SOURCE") == "false"

def test_review_only_validation_status_validated():
    assert C.classify_review_only_validation_status(
        "AUTOMATED_SUPERVISOR_VALIDATED_FOR_REVIEW_ONLY_USE") == "VALIDATED_FOR_REVIEW_ONLY_USE"

def test_review_only_validation_status_external_required():
    assert C.classify_review_only_validation_status(
        "AUTOMATED_SUPERVISOR_WAITING_EXTERNAL_OBSERVATIONAL_SOURCE") \
        == "EXTERNAL_OBSERVATIONAL_EVIDENCE_REQUIRED_FOR_OPERATIONAL_CLAIM"

def test_build_reviewer_rubric_two_profiles():
    profs = {r["reviewer_profile"] for r in C.build_reviewer_rubric()}
    assert profs == {"conservative", "integrator"}

def test_build_supervisor_rubric_no_operational():
    for r in C.build_supervisor_rubric():
        assert r["operational_validation"] == "false"
        assert r["supervisor_final_operational_decision_allowed"] == "false"

def test_build_tcc_case_summary_safe():
    s = C.build_tcc_case_summary({"case_id": "X", "region": "PET"})
    assert s["claim_safety"] == "REVIEW_ONLY_SAFE_NO_OPERATIONAL_CLAIM"


# ===========================================================================
# 4. v1tn — case index
# ===========================================================================

def test_v1tn_case_index_created(monkeypatch, tmp_path):
    _setup_inputs(tmp_path)
    _redirect(monkeypatch, v1tn, tmp_path)
    v1tn.run()
    rows = _read(tmp_path / v1tn.OUT_IDX.name)
    assert len(rows) == 1
    assert rows[0]["case_id"].startswith("CASE_PET_")

def test_v1tn_fail_closed_no_inputs(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tn, tmp_path)
    v1tn.run()
    rows = _read(tmp_path / v1tn.OUT_IDX.name)
    assert rows and rows[0]["case_id"] == "FAIL_CLOSED_NO_CASES"

def test_v1tn_hydromet_status_consolidated(monkeypatch, tmp_path):
    _setup_inputs(tmp_path)
    _redirect(monkeypatch, v1tn, tmp_path)
    v1tn.run()
    rows = _read(tmp_path / v1tn.OUT_IDX.name)
    assert rows[0]["hydromet_status"] == "HYDROMET_CONTEXT_AVAILABLE_REVIEW_ONLY"

def test_v1tn_external_status_consolidated(monkeypatch, tmp_path):
    _setup_inputs(tmp_path)
    _redirect(monkeypatch, v1tn, tmp_path)
    v1tn.run()
    rows = _read(tmp_path / v1tn.OUT_IDX.name)
    assert rows[0]["external_evidence_status"] == "EXTERNAL_SOURCE_ABSENT_LOCAL"

def test_v1tn_dino_status_review_only(monkeypatch, tmp_path):
    _setup_inputs(tmp_path)
    _redirect(monkeypatch, v1tn, tmp_path)
    v1tn.run()
    rows = _read(tmp_path / v1tn.OUT_IDX.name)
    assert rows[0]["dino_status"] == "DINO_NOT_PRESENT_CONTEXT_ONLY"

def test_v1tn_patch_status(monkeypatch, tmp_path):
    _setup_inputs(tmp_path, link_cid="EVT_PET_001")
    _redirect(monkeypatch, v1tn, tmp_path)
    v1tn.run()
    rows = _read(tmp_path / v1tn.OUT_IDX.name)
    assert rows[0]["patch_link_status"] == "PATCH_LINK_CANDIDATE_REVIEW_ONLY"

def test_v1tn_next_action_correct(monkeypatch, tmp_path):
    _setup_inputs(tmp_path)
    _redirect(monkeypatch, v1tn, tmp_path)
    v1tn.run()
    rows = _read(tmp_path / v1tn.OUT_IDX.name)
    assert rows[0]["next_required_action"] \
        == "COLLECT_EXTERNAL_OBSERVATIONAL_SOURCE_FOR_OPERATIONAL_CLAIM"

def test_v1tn_no_c3_c4(monkeypatch, tmp_path):
    _setup_inputs(tmp_path)
    _redirect(monkeypatch, v1tn, tmp_path)
    v1tn.run()
    for r in _read(tmp_path / v1tn.OUT_IDX.name):
        assert r["automatic_c3_promotion"] == "false"
        assert r["c4_opened"] == "false"


# ===========================================================================
# 5. v1to — workspace
# ===========================================================================

def test_v1to_workspace_readable(monkeypatch, tmp_path):
    _setup_inputs(tmp_path)
    for mod in (v1tn, v1to):
        _redirect(monkeypatch, mod, tmp_path)
    v1tn.run(); v1to.run()
    rows = _read(tmp_path / v1to.OUT_WS.name)
    assert rows and "Caso" in rows[0]["case_summary"]

def test_v1to_sections_created(monkeypatch, tmp_path):
    _setup_inputs(tmp_path)
    for mod in (v1tn, v1to):
        _redirect(monkeypatch, mod, tmp_path)
    v1tn.run(); v1to.run()
    secs = _read(tmp_path / v1to.OUT_SEC.name)
    assert len(secs) == len(v1to.SECTION_KEYS)

def test_v1to_workspace_has_hydromet_summary(monkeypatch, tmp_path):
    _setup_inputs(tmp_path)
    for mod in (v1tn, v1to):
        _redirect(monkeypatch, mod, tmp_path)
    v1tn.run(); v1to.run()
    rows = _read(tmp_path / v1to.OUT_WS.name)
    assert "contexto" in rows[0]["hydromet_summary"].lower()


# ===========================================================================
# 6. v1tp — reviewer A/B
# ===========================================================================

def _run_to(monkeypatch, tmp, upto, **kw):
    _setup_inputs(tmp, **kw)
    order = [v1tn, v1to, v1tp, v1tq, v1tr, v1ts, v1tt, v1tu, v1tv, v1tw]
    for mod in order:
        _redirect(monkeypatch, mod, tmp)
    for mod in order:
        mod.run()
        if mod is upto:
            break

def test_v1tp_reviewer_a_decisions(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tp)
    rows = [r for r in _read(tmp_path / v1tp.OUT_DEC.name) if r["reviewer_slot"] == "A"]
    assert len(rows) == 1

def test_v1tp_reviewer_b_decisions(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tp)
    rows = [r for r in _read(tmp_path / v1tp.OUT_DEC.name) if r["reviewer_slot"] == "B"]
    assert len(rows) == 1

def test_v1tp_a_more_conservative(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tp)
    rows = _read(tmp_path / v1tp.OUT_DEC.name)
    a = next(r for r in rows if r["reviewer_slot"] == "A")
    b = next(r for r in rows if r["reviewer_slot"] == "B")
    assert float(a["review_only_confidence_score"]) < float(b["review_only_confidence_score"])

def test_v1tp_automated_review_true(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tp)
    for r in _read(tmp_path / v1tp.OUT_DEC.name):
        assert r["automated_review"] == "true"

def test_v1tp_human_replaced_true(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tp)
    for r in _read(tmp_path / v1tp.OUT_DEC.name):
        assert r["internal_review_automated_for_review_only"] == "true"

def test_v1tp_requires_external_true(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tp)
    for r in _read(tmp_path / v1tp.OUT_DEC.name):
        assert r["requires_external_observational_evidence_for_operational_claim"] == "true"

def test_v1tp_no_label_target_groundtruth(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tp)
    for r in _read(tmp_path / v1tp.OUT_DEC.name):
        assert r["can_create_operational_label"] == "false"
        assert r["target_created"] == "false"
        assert r["ground_truth_operational"] == "false"

def test_v1tp_no_formal_negative(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tp)
    for r in _read(tmp_path / v1tp.OUT_DEC.name):
        assert r["formal_negative"] == "false"

def test_v1tp_no_absence_negative(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tp)
    for r in _read(tmp_path / v1tp.OUT_DEC.name):
        assert r["absence_as_negative"] == "false"

def test_v1tp_no_dino_proof(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tp)
    for r in _read(tmp_path / v1tp.OUT_DEC.name):
        assert r["dino_validates_event"] == "false"

def test_v1tp_no_hydromet_proof(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tp)
    for r in _read(tmp_path / v1tp.OUT_DEC.name):
        assert r["hydromet_validates_event"] == "false"

def test_v1tp_no_c3_c4(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tp)
    for r in _read(tmp_path / v1tp.OUT_DEC.name):
        assert r["automatic_c3_promotion"] == "false"
        assert r["c4_opened"] == "false"

def test_v1tp_rubric_present(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tp)
    assert len(_read(tmp_path / v1tp.OUT_RUB.name)) > 0


# ===========================================================================
# 7. v1tq — consensus / divergence
# ===========================================================================

def test_v1tq_consensus_created(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tq)
    rows = _read(tmp_path / v1tq.OUT_CON.name)
    assert len(rows) == 1

def test_v1tq_divergence_detected(monkeypatch, tmp_path):
    # default fixture: no patch -> A NEEDS_SPATIAL vs B VALIDATED -> divergence
    _run_to(monkeypatch, tmp_path, v1tq)
    rows = _read(tmp_path / v1tq.OUT_CON.name)
    assert rows[0]["consensus_status"] == "AUTOMATED_DIVERGENCE_REQUIRES_SUPERVISOR_ADJUDICATION"
    assert rows[0]["supervisor_adjudication_required"] == "true"

def test_v1tq_external_required_flag(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tq)
    for r in _read(tmp_path / v1tq.OUT_CON.name):
        assert r["external_observational_evidence_required_for_operational_claim"] == "true"

def test_v1tq_no_c3(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tq)
    for r in _read(tmp_path / v1tq.OUT_CON.name):
        assert r["automatic_c3_promotion"] == "false"


# ===========================================================================
# 8. v1tr — supervisor
# ===========================================================================

def test_v1tr_supervisor_created(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tr)
    rows = _read(tmp_path / v1tr.OUT_SUP.name)
    assert len(rows) == 1
    assert rows[0]["supervisor_decision"] in C.SUPERVISOR_DECISIONS

def test_v1tr_operational_validation_false(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tr)
    for r in _read(tmp_path / v1tr.OUT_SUP.name):
        assert r["operational_validation"] == "false"

def test_v1tr_final_operational_decision_not_allowed(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tr)
    for r in _read(tmp_path / v1tr.OUT_SUP.name):
        assert r["supervisor_final_operational_decision_allowed"] == "false"

def test_v1tr_automated_supervisor_flag(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tr)
    for r in _read(tmp_path / v1tr.OUT_SUP.name):
        assert r["automated_supervisor_adjudication"] == "true"
        assert r["does_not_validate_event"] == "true"

def test_v1tr_external_required_when_no_source(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tr)
    for r in _read(tmp_path / v1tr.OUT_SUP.name):
        assert r["external_observational_source_required_for_operational_claim"] == "true"


# ===========================================================================
# 9. v1ts — single flow
# ===========================================================================

def test_v1ts_flow_all_sections(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1ts)
    secs = _read(tmp_path / v1ts.OUT_SEC.name)
    keys = {s["section_key"] for s in secs}
    for k in v1ts.SECTION_KEYS:
        assert k in keys

def test_v1ts_flow_replaces_many_csvs(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1ts)
    rows = _read(tmp_path / v1ts.OUT_FLOW.name)
    r = rows[0]
    # one row carries header, evidence, hydromet, dino, A, B, consensus, supervisor
    for f in ("case_header", "evidence_summary", "hydromet_summary",
              "dino_limitation", "reviewer_a_decision", "reviewer_b_decision",
              "consensus_divergence", "supervisor_adjudication", "blockers",
              "next_action", "claim_safety_status", "tcc_ready_summary"):
        assert r[f]

def test_v1ts_claim_safety(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1ts)
    rows = _read(tmp_path / v1ts.OUT_FLOW.name)
    assert "REVIEW_ONLY_SAFE" in rows[0]["claim_safety_status"]


# ===========================================================================
# 10. v1tt — TCC tables
# ===========================================================================

def test_v1tt_case_status_table(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tt)
    rows = _read(tmp_path / v1tt.OUT_STATUS.name)
    assert rows and "review_only_validation_status" in rows[0]

def test_v1tt_outcomes_table(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tt)
    rows = _read(tmp_path / v1tt.OUT_OUTCOME.name)
    assert rows and rows[0]["reviewer_a_status"]

def test_v1tt_blockers_table(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tt)
    rows = _read(tmp_path / v1tt.OUT_BLOCK.name)
    assert rows and "C3" in rows[0]["why_c3_not_automatic"]

def test_v1tt_claim_safety_table(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tt)
    rows = _read(tmp_path / v1tt.OUT_SAFETY.name)
    assert rows[0]["dino_validates_event"] == "false"
    assert rows[0]["hydromet_is_negative_evidence"] == "false"
    assert rows[0]["absence_as_negative"] == "false"


# ===========================================================================
# 11. v1tu — proof audit
# ===========================================================================

def test_v1tu_checks_present(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tu)
    rows = _read(tmp_path / v1tu.OUT_PROOF.name)
    r = rows[0]
    for f in ("has_workspace", "has_evidence_summary", "has_hydromet_summary",
              "dino_role_limited", "has_reviewer_a", "has_reviewer_b",
              "has_consensus_or_divergence", "has_supervisor_adjudication"):
        assert r[f] == "true"

def test_v1tu_marks_validated_review_only(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tu)
    rows = _read(tmp_path / v1tu.OUT_PROOF.name)
    assert rows[0]["review_only_validation_status"] == "VALIDATED_FOR_REVIEW_ONLY_USE"

def test_v1tu_marks_not_operational_ground_truth(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tu)
    for r in _read(tmp_path / v1tu.OUT_PROOF.name):
        assert r["not_operational_ground_truth"] == "true"

def test_v1tu_marks_not_automatic_c3(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tu)
    for r in _read(tmp_path / v1tu.OUT_PROOF.name):
        assert r["not_automatic_c3"] == "true"

def test_v1tu_marks_not_c4(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tu)
    for r in _read(tmp_path / v1tu.OUT_PROOF.name):
        assert r["not_c4"] == "true"

def test_v1tu_proof_complete(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tu)
    assert _read(tmp_path / v1tu.OUT_PROOF.name)[0]["proof_status"] \
        == "REVIEW_ONLY_PROOF_COMPLETE"


# ===========================================================================
# 12. v1tv — guardrail audit
# ===========================================================================

def test_v1tv_all_clean(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tv)
    rows = _read(tmp_path / v1tv.OUT_AUDIT.name)
    assert all(r["audit_status"] == "GUARDRAIL_CLEAN" for r in rows)

def test_v1tv_detects_automatic_c3(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tv)
    inj = tmp_path / "protocol_c_automated_reviewer_ab_decisions_v1tp.csv"
    rows = _read(inj)
    rows[0]["automatic_c3_promotion"] = "true"
    _write_csv(inj, rows, list(rows[0].keys()))
    v1tv.run()
    aud = _read(tmp_path / v1tv.OUT_AUDIT.name)
    hit = next(r for r in aud if r["source_file"].endswith("v1tp.csv"))
    assert int(hit["forbidden_true_hits"]) >= 1

def test_v1tv_detects_c4_opened(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tv)
    inj = tmp_path / "protocol_c_unified_evidence_case_index_v1tn.csv"
    rows = _read(inj)
    rows[0]["c4_opened"] = "true"
    _write_csv(inj, rows, list(rows[0].keys()))
    v1tv.run()
    aud = _read(tmp_path / v1tv.OUT_AUDIT.name)
    hit = next(r for r in aud if r["source_file"].endswith("v1tn.csv"))
    assert int(hit["forbidden_true_hits"]) >= 1

def test_v1tv_detects_human_review_completed(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tv)
    inj = tmp_path / "protocol_c_single_flow_review_export_v1ts.csv"
    rows = _read(inj)
    rows[0]["notes"] = "human review completed"
    _write_csv(inj, rows, list(rows[0].keys()))
    v1tv.run()
    aud = _read(tmp_path / v1tv.OUT_AUDIT.name)
    hit = next(r for r in aud if r["source_file"].endswith(
        "single_flow_review_export_v1ts.csv"))
    assert int(hit["forbidden_phrase_hits"]) >= 1

def test_v1tv_detects_operationally_validated(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tv)
    inj = tmp_path / "protocol_c_single_flow_review_export_v1ts.csv"
    rows = _read(inj)
    rows[0]["notes"] = "operationally validated event"
    _write_csv(inj, rows, list(rows[0].keys()))
    v1tv.run()
    aud = _read(tmp_path / v1tv.OUT_AUDIT.name)
    hit = next(r for r in aud if r["source_file"].endswith(
        "single_flow_review_export_v1ts.csv"))
    assert int(hit["forbidden_phrase_hits"]) >= 1

def test_v1tv_detects_ground_truth_confirmed(monkeypatch, tmp_path):
    _run_to(monkeypatch, tmp_path, v1tv)
    inj = tmp_path / "protocol_c_proof_of_review_only_validation_audit_v1tu.csv"
    rows = _read(inj)
    rows[0]["notes"] = "ground truth confirmed"
    _write_csv(inj, rows, list(rows[0].keys()))
    v1tv.run()
    aud = _read(tmp_path / v1tv.OUT_AUDIT.name)
    hit = next(r for r in aud if r["source_file"].endswith("v1tu.csv"))
    assert int(hit["forbidden_phrase_hits"]) >= 1


# ===========================================================================
# 13. v1tw — bundle
# ===========================================================================

def test_v1tw_manifest_present(monkeypatch, tmp_path):
    _run_chain(monkeypatch, tmp_path)
    rows = _read(tmp_path / v1tw.OUT_MAN.name)
    assert len(rows) == len(v1tw.ARTIFACTS)

def test_v1tw_status_ready_or_validated(monkeypatch, tmp_path):
    _run_chain(monkeypatch, tmp_path)
    sci = {r["metric_key"]: r["metric_value"]
           for r in _read(tmp_path / v1tw.OUT_SCI.name)}
    assert sci["final_status"] in (
        "UNIFIED_AUTOMATED_REVIEW_READY_FOR_TCC_DISCUSSION",
        "UNIFIED_AUTOMATED_REVIEW_VALIDATED_FOR_REVIEW_ONLY_USE",
    )

def test_v1tw_metrics_zero_forbidden(monkeypatch, tmp_path):
    _run_chain(monkeypatch, tmp_path)
    sci = {r["metric_key"]: r["metric_value"]
           for r in _read(tmp_path / v1tw.OUT_SCI.name)}
    for k in ("automatic_c3_promotions", "c4_opened_count", "labels_created",
              "targets_created", "ground_truth_operational_created",
              "formal_negatives_created", "guardrail_violations"):
        assert sci[k] == "0"

def test_v1tw_quality_checks_pass(monkeypatch, tmp_path):
    _run_chain(monkeypatch, tmp_path)
    qc = _read(tmp_path / v1tw.OUT_QC.name)
    assert all(r["check_result"] == "PASS" for r in qc)

def test_v1tw_mandatory_doc_phrase(monkeypatch, tmp_path):
    _run_chain(monkeypatch, tmp_path)
    doc = (tmp_path / v1tw.DOC.name).read_text(encoding="utf-8")
    norm = " ".join(doc.split())
    phrase = " ".join(v1tw.MANDATORY_PHRASE.split())
    assert phrase in norm

def test_v1tw_divergence_counted(monkeypatch, tmp_path):
    _run_chain(monkeypatch, tmp_path)
    sci = {r["metric_key"]: r["metric_value"]
           for r in _read(tmp_path / v1tw.OUT_SCI.name)}
    assert int(sci["divergence_rows"]) >= 1


# ===========================================================================
# 14. cross-cutting: schemas/docs exist; chain guardrail-clean
# ===========================================================================

def test_all_schemas_and_docs_written(monkeypatch, tmp_path):
    _run_chain(monkeypatch, tmp_path)
    for mod in ALL_MODULES:
        for name in dir(mod):
            if name.startswith("SCHEMA_") or name == "DOC":
                val = getattr(mod, name)
                if isinstance(val, Path):
                    assert val.exists(), f"missing {mod.__name__}.{name}"

def test_chain_outputs_guardrail_clean(monkeypatch, tmp_path):
    _run_chain(monkeypatch, tmp_path)
    out_files = list(tmp_path.glob("protocol_c_*_v1t[n-w]*.csv"))
    assert out_files
    for f in out_files:
        if "schema" in f.name:
            continue
        viol = C.scan_guardrails(_read(f), f.name)
        assert viol == [], f"{f.name}: {viol[:2]}"

def test_chain_no_absolute_paths(monkeypatch, tmp_path):
    _run_chain(monkeypatch, tmp_path)
    for f in tmp_path.glob("protocol_c_*_v1t*.csv"):
        text = f.read_text(encoding="utf-8")
        assert not C.ABS_PATH_RE.search(text), f"abs path in {f.name}"

def test_chain_no_local_runs(monkeypatch, tmp_path):
    _run_chain(monkeypatch, tmp_path)
    lit = "local" + "_runs"
    for f in tmp_path.glob("protocol_c_*_v1t*.csv"):
        assert lit not in f.read_text(encoding="utf-8").lower(), f"local_runs in {f.name}"

def test_outputs_written_only_in_tmp(monkeypatch, tmp_path):
    _run_chain(monkeypatch, tmp_path)
    # every redirected OUT path lives under tmp_path
    for mod in ALL_MODULES:
        for name in dir(mod):
            if name.startswith("OUT_"):
                val = getattr(mod, name)
                if isinstance(val, Path):
                    assert tmp_path in val.parents


# ===========================================================================
# 15. divergence-vs-consensus fixtures
# ===========================================================================

def test_consensus_validated_with_external_no_hydromet(monkeypatch, tmp_path):
    # external present + patch + window, hydromet absent -> both VALIDATED
    _write_window(tmp_path, "EVT_PET_001")
    _write_backlog(tmp_path)
    _write_candidates(tmp_path, "EVT_PET_001", "ACCEPTED")
    _write_links(tmp_path, "EVT_PET_001", conf="HIGH")
    # no hydromet packet on purpose
    _write_csv(tmp_path / "protocol_c_hydromet_evidence_packet_registry_v1tg.csv",
               [], ["hydromet_packet_id", "event_candidate_id", "region",
                    "hydromet_support_level"])
    for mod in (v1tn, v1to, v1tp, v1tq):
        _redirect(monkeypatch, mod, tmp_path)
    for mod in (v1tn, v1to, v1tp, v1tq):
        mod.run()
    rows = _read(tmp_path / v1tq.OUT_CON.name)
    assert rows[0]["consensus_status"] == "AUTOMATED_CONSENSUS_VALIDATED_FOR_REVIEW_ONLY_USE"

# ===========================================================================
# 16. terminology guard for versioned v1tn-v1tw files
# Forbidden review-layer labels are built by concatenation so this test source
# stays self-clean and only real violations are flagged. DINO is preserved.
# ===========================================================================

import re as _re

_A = "A" + "I"          # uppercase two-letter label (en)
_B = "I" + "A"          # uppercase two-letter label (pt)
_C = "a" + "i"          # lowercase identifier fragment
_FORBIDDEN_PATTERNS = [
    _re.compile(r"\b" + _A + r"\b"),
    _re.compile(r"\b" + _B + r"\b"),
    _re.compile(_C + "_"),
    _re.compile("_" + _C + "_"),
    _re.compile("_" + _C + r"\b"),
    _re.compile("(?i)artificial " + "intelligence"),
    _re.compile("(?i)assistida por " + _B),
    _re.compile("(?i)autonomous " + _A),
    _re.compile("(?i)" + _A + "-assisted"),
    _re.compile("(?i)human_review_" + "replaced"),
    _re.compile(r"\b" + "LL" + "M" + r"\b"),
    _re.compile(r"(?i)\b" + "cla" + "ude" + r"\b"),
    _re.compile("(?i)" + "chat" + "gpt"),
]


def _versioned_v1tn_v1tw_files() -> list[Path]:
    out: list[Path] = []
    out += list(SCRIPTS.glob("revp_v1t[n-w]*.py"))
    out += list((ROOT / "datasets").glob("protocol_c_*_v1t[n-w]*.csv"))
    out += list((ROOT / "datasets" / "schemas").glob("protocol_c_*_v1t[n-w]*_schema.csv"))
    out += list((ROOT / "docs" / "metodologia_cientifica").glob("revp_v1t[n-w]*.md"))
    out += [Path(__file__)]
    return out


def test_no_forbidden_review_layer_labels_in_versioned_files():
    offenders: list[str] = []
    for f in _versioned_v1tn_v1tw_files():
        text = f.read_text(encoding="utf-8")
        for line in text.splitlines():
            for pat in _FORBIDDEN_PATTERNS:
                if pat.search(line):
                    offenders.append(f"{f.name}: {line.strip()[:80]}")
    assert offenders == [], f"forbidden review-layer labels: {offenders[:5]}"


def test_dino_terminology_preserved_allowed():
    # DINO must remain usable; ensure the guard does not flag it.
    for pat in _FORBIDDEN_PATTERNS:
        assert not pat.search("DINO representation review-only context")


def test_staged_area_empty_at_end():
    import subprocess
    out = subprocess.run(["git", "diff", "--cached", "--name-only"],
                         cwd=ROOT, capture_output=True, text=True)
    assert out.stdout.strip() == "", f"staged not empty: {out.stdout}"
