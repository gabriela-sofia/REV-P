import scripts.protocolo_c.revp_v2am_common as common
from tests.test_revp_v2am_common import install_all, read_csv


def test_completion_report_records_pipeline(tmp_path, monkeypatch):
    data, docs, atlas, _ = install_all(tmp_path, monkeypatch)
    common.run_master_orchestrator(common.parse_args([]))
    common.run_guardrail_regression(common.parse_args([]))
    common.run_completion_report(common.parse_args([]))
    by_metric = {r["metric"]: r for r in read_csv(data / "v2am_completion_report.csv")}
    assert int(by_metric["dag_nodes"]["value"]) == 9
    assert int(by_metric["dag_edges"]["value"]) == 8
    assert by_metric["final_audit_violations"]["status"] == "PASS"
    assert by_metric["guardrail_regression_failures"]["status"] == "PASS"
    assert by_metric["main_manuscript_overwritten"]["value"] == "false"
    assert by_metric["next_action_rank_1"]["value"] == "MANUAL_APPENDIX_REVIEW_AND_ORIENTATION_MEETING"
    md = (atlas / "v2am_completion_report.md").read_text(encoding="utf-8")
    common.assert_safe_text(md)
