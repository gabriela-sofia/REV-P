from tests.test_revp_v1ut_recife_coordinate_asset_locator import install_asset, set_env
import scripts.protocolo_c.revp_v1ut_recife_common as common


def test_overlay_preflight_blocker_never_executes_overlay(tmp_path, monkeypatch):
    data, raw = set_env(tmp_path, monkeypatch)
    install_asset(data, raw, "recife_coordinate_hazard_window.csv",
                  classification="OCCURRENCE_COORDINATES_CANDIDATE")
    common.run_coordinate_asset_locator()
    common.run_coordinate_schema_reparser()
    common.run_event_window_coordinate_filter()
    common.run_hazard_coordinate_crossfilter()
    common.run_coordinate_candidate_audit()
    rows = common.run_overlay_preflight_blocker()
    assert rows[0]["can_execute_overlay_now"] == "false"
    assert rows[0]["no_overlay_executed"] == "true"
