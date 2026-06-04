from tests.test_revp_v1ut_recife_coordinate_asset_locator import install_asset, set_env
import scripts.protocolo_c.revp_v1ut_recife_common as common


def test_event_window_filter_marks_core_and_outside_dates(tmp_path, monkeypatch):
    data, raw = set_env(tmp_path, monkeypatch)
    install_asset(data, raw, "recife_coordinate_hazard_window.csv", "core_asset")
    install_asset(data, raw, "recife_coordinate_outside_window.csv", "outside_asset")
    common.run_coordinate_asset_locator()
    common.run_coordinate_schema_reparser()
    rows = common.run_event_window_coordinate_filter()
    assert {r["event_window_match"] for r in rows} == {"REC_2022_CORE_WINDOW", "OUTSIDE_REC_2022_CORE_WINDOW"}
    assert all(r["raw_values_versioned"] == "false" for r in rows)
