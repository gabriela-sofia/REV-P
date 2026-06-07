import scripts.protocolo_c.revp_v2aq_common as common
from tests.test_revp_v2aq_common import install_all, read_csv


def test_completion_report(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson = install_all(tmp_path, monkeypatch)
    common.run_master_orchestrator(common.parse_args([]))
    by_metric = {r["metric"]: r for r in read_csv(protocol / "v2aq_completion_report.csv")}
    assert by_metric["candidates_loaded"]["value"] == "9"
    assert by_metric["ground_truth_blocked_all"]["status"] == "PASS"
    assert by_metric["guardrail_regression_failures"]["status"] == "PASS"
    assert int(by_metric["geojson_geometry_null"]["value"]) == 9
    md = (docs / "v2aq_completion_report.md").read_text(encoding="utf-8")
    common.assert_safe_text(md)
