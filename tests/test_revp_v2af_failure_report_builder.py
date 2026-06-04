import csv

from tests.test_revp_v2af_qa_input_manifest_builder import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2af_common as common


def _run_chain(data):
    install_base_inputs(data)
    common.run_expected_count_validator(common.parse_args([]))
    common.run_guardrail_regression_runner(common.parse_args([]))
    common.run_canonical_registry_regression_runner(common.parse_args([]))
    common.run_event_patch_v2_regression_runner(common.parse_args([]))


def test_failure_report_no_failures_when_clean(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    _run_chain(data)
    rows = common.run_failure_report_builder(common.parse_args([]))
    assert len(rows) == 1
    assert rows[0]["check"] == "NO_FAILURES_DETECTED"
    assert rows[0]["blocking_status"] == "NON_BLOCKING"


def test_failure_report_captures_failures(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    # Drop a package so expected-count fails, then build chain + report.
    path = str(data / "v2ac_event_patch_v2_package_registry.csv")
    with open(path, newline="", encoding="utf-8") as f:
        reg = list(csv.DictReader(f))
        cols = list(reg[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(reg[:-2])
    common.run_expected_count_validator(common.parse_args([]))
    common.run_guardrail_regression_runner(common.parse_args([]))
    common.run_canonical_registry_regression_runner(common.parse_args([]))
    common.run_event_patch_v2_regression_runner(common.parse_args([]))
    rows = common.run_failure_report_builder(common.parse_args([]))
    assert any(r["check"] != "NO_FAILURES_DETECTED" for r in rows)
    assert all(r["blocking_status"] == "BLOCKING" for r in rows)
