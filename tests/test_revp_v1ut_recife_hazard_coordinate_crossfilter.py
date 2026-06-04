from tests.test_revp_v1ut_recife_coordinate_asset_locator import install_asset, set_env
import scripts.protocolo_c.revp_v1ut_recife_common as common


def test_hazard_crossfilter_promotes_only_window_hazard_occurrence_coordinate(tmp_path, monkeypatch):
    data, raw = set_env(tmp_path, monkeypatch)
    install_asset(data, raw, "recife_coordinate_hazard_window.csv",
                  classification="OCCURRENCE_COORDINATES_CANDIDATE")
    common.run_coordinate_asset_locator()
    common.run_coordinate_schema_reparser()
    common.run_event_window_coordinate_filter()
    rows = common.run_hazard_coordinate_crossfilter()
    assert rows[0]["can_promote_to_coordinate_candidate"] == "true"
    assert rows[0]["hazard_coordinate_status"] == "COORDINATE_WINDOW_HAZARD_CANDIDATE_FOR_REVIEW"
