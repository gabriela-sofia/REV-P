import scripts.protocolo_c.revp_v2al_common as common
from tests.test_revp_v2al_common import install_all, read_csv


def test_completion_report_records_pipeline(tmp_path, monkeypatch):
    data, docs, integration = install_all(tmp_path, monkeypatch)
    common.run_all(common.parse_args([]))
    rows = read_csv(data / "v2al_completion_report.csv")
    by_metric = {r["metric"]: r for r in rows}
    assert by_metric["markdown_bundles_created"]["value"] == "4"
    assert by_metric["latex_bundles_created"]["value"] == "4"
    assert by_metric["claim_violations"]["value"] == "0"
    assert by_metric["claim_violations"]["status"] == "PASS"
    assert by_metric["safe_language_regression_failures"]["value"] == "0"
    assert by_metric["main_manuscript_overwritten"]["value"] == "false"
    assert by_metric["next_action_rank_1"]["value"] == "MANUAL_TCC_INTEGRATION_REVIEW"
    md = (integration / "v2al_completion_report.md").read_text(encoding="utf-8")
    assert "Main manuscript overwritten: false" in md
    common.assert_safe_manuscript_language(md)
