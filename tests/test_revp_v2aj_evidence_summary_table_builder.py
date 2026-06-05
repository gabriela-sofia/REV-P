import scripts.protocolo_c.revp_v2aj_common as common
from tests.test_revp_v2aj_common import install_inputs, set_env


def test_evidence_summary_exports_required_metrics(tmp_path, monkeypatch):
    data, _ = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    rows = common.run_evidence_summary_table_builder(common.parse_args([]))
    metrics = {r["metric_name"]: r for r in rows}
    assert metrics["total_candidates"]["metric_value"] == "2"
    assert metrics["total_assignments"]["metric_value"] == "4"
    assert metrics["next_action_rank_1"]["metric_value"] == "HUMAN_REVIEW_EXECUTION_OR_SAFE_TCC_EXPORT"
