import scripts.protocolo_c.revp_v2an_common as common
from tests.test_revp_v2an_common import install_all, read_csv


def test_completion_report_records_sprint(tmp_path, monkeypatch):
    data, protocol, docs, _ = install_all(tmp_path, monkeypatch)
    common.run_master_orchestrator(common.parse_args([]))
    by_metric = {r["metric"]: r for r in read_csv(protocol / "v2an_completion_report.csv")}
    assert by_metric["candidates_normalized"]["value"] == "9"
    assert by_metric["candidates_normalized"]["status"] == "PASS"
    assert by_metric["ground_truth_blocked"]["status"] == "PASS"
    assert by_metric["guardrail_regression_failures"]["status"] == "PASS"
    assert by_metric["patch_link_overlay_ready"]["value"] == "0"
    assert by_metric["sentinel_crosswalks_closed"]["value"] == "0"
    md = (docs / "v2an_completion_report.md").read_text(encoding="utf-8")
    common.assert_safe_text(md)
