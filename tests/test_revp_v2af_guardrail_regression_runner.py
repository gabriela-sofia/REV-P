from pathlib import Path

from tests.test_revp_v2af_qa_input_manifest_builder import install_base_inputs, set_env, write_csv
import scripts.protocolo_c.revp_v2af_common as common

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "v2af"


def test_guardrail_regression_clean(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    rows = common.run_guardrail_regression_runner(common.parse_args([]))
    assert rows
    assert all(r["status"] == "PASS" for r in rows)


def test_guardrail_regression_detects_violations(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    bad = str(data / "tampered.csv")
    write_csv(bad, ["event_patch_candidate_id", "crosswalk_inferred", "overlay_status", "src_path"],
              [{"event_patch_candidate_id": "EPCX", "crosswalk_inferred": "true", "overlay_status": "ALLOWED", "src_path": "C:\\Users\\x\\raw.tif"}])
    rows = common.run_guardrail_regression_runner(common.parse_args([]), files=[bad])
    counts = {r["check_type"]: int(r["violation_count"]) for r in rows}
    assert counts["forbidden_true_value"] >= 1
    assert counts["overlay_or_gr_or_training_released"] >= 1
    assert counts["absolute_path"] >= 1
    assert any(r["status"] == "FAIL" for r in rows)


def test_guardrail_regression_detects_violation_fixture(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    fixture = str(FIXTURE_DIR / "guardrail_violation.csv")
    rows = common.run_guardrail_regression_runner(common.parse_args([]), files=[fixture])
    counts = {r["check_type"]: int(r["violation_count"]) for r in rows}
    assert counts["forbidden_true_value"] >= 2  # crosswalk_inferred + sentinel_date_inferred
    assert counts["absolute_path"] >= 1
    assert counts["overlay_or_gr_or_training_released"] >= 1
