import scripts.protocolo_c.revp_v2ap_common as common
from tests.test_revp_v2ap_common import install_all, read_csv


def test_completion_report(tmp_path, monkeypatch):
    datasets, protocol, docs = install_all(tmp_path, monkeypatch)
    common.run_master_orchestrator(common.parse_args([]))
    by_metric = {r["metric"]: r for r in read_csv(protocol / "v2ap_completion_report.csv")}
    assert by_metric["candidates_loaded"]["value"] == "9"
    assert by_metric["patch_truth_allowed"]["value"] == "0"
    assert by_metric["patch_truth_allowed"]["status"] == "GUARDRAIL_OK"
    assert by_metric["guardrail_regression_failures"]["status"] == "PASS"
    md = (docs / "v2ap_completion_report.md").read_text(encoding="utf-8")
    common.assert_safe_text(md)
