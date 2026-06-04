from tests.test_revp_v1uw_curitiba_event_source_snapshotter import install_inputs, set_env
import scripts.protocolo_c.revp_v1uw_curitiba_common as common


def test_blocker_builder_includes_alert_not_ground_truth(tmp_path, monkeypatch):
    data, _raw = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    common.run_event_candidate_status_builder(common.parse_args([]))
    rows = common.run_ground_reference_blocker_builder(common.parse_args([]))
    blockers = {r["blocker"] for r in rows}
    assert "alert_is_not_ground_truth" in blockers
    assert all(r["ground_truth_operational"] == "false" for r in rows)
