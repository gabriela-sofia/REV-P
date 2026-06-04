from tests.test_revp_v1ut_recife_coordinate_asset_locator import install_asset, set_env
import scripts.protocolo_c.revp_v1ut_recife_common as common


def test_candidate_audit_caps_status_and_blocks_truth_labels(tmp_path, monkeypatch):
    data, raw = set_env(tmp_path, monkeypatch)
    install_asset(data, raw, "recife_coordinate_hazard_window.csv",
                  classification="OCCURRENCE_COORDINATES_CANDIDATE")
    common.run_coordinate_asset_locator()
    common.run_coordinate_schema_reparser()
    common.run_event_window_coordinate_filter()
    common.run_hazard_coordinate_crossfilter()
    rows = common.run_coordinate_candidate_audit()
    assert rows[0]["candidate_status"] == common.MAX_STATUS
    assert rows[0]["ground_truth_operational"] == "false"
    assert rows[0]["can_create_training_label"] == "false"
    assert rows[0]["no_coordinates_invented"] == "true"
