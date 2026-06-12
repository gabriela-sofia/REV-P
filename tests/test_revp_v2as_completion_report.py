import scripts.protocolo_c.revp_v2as_common as common
from tests.test_revp_v2as_common import install_all


def test_completion_report_no_operational_ground_truth(tmp_path, monkeypatch):
    install_all(tmp_path, monkeypatch)
    common.run_master_orchestrator(common.parse_args([]))
    rows = common.run_completion_report(common.parse_args([]))
    by = {r["metric"]: r for r in rows}
    assert by["final_decision"]["status"] == "NO_OPERATIONAL_GROUND_TRUTH"
    assert by["geojson_geometry_real"]["value"] == "0"
    assert by["geojson_geometry_null"]["value"] == "4"
    assert by["guardrail_regression_failures"]["status"] == "PASS"
    assert by["patch_truth_boundary_blocked_all"]["status"] == "PASS"
    assert by["next_action_rank_1"]["value"] == "MANUAL_DIGITIZE_EVENT_GEOMETRY_FROM_OFFICIAL_SOURCES"
