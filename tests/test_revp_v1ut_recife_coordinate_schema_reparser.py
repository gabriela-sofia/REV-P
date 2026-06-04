from tests.test_revp_v1ut_recife_coordinate_asset_locator import install_asset, set_env
import scripts.protocolo_c.revp_v1ut_recife_common as common


def test_schema_reparser_parses_decimal_comma_without_geocoding(tmp_path, monkeypatch):
    data, raw = set_env(tmp_path, monkeypatch)
    install_asset(data, raw, "recife_coordinate_hazard_window.csv")
    common.run_coordinate_asset_locator()
    rows = common.run_coordinate_schema_reparser()
    assert rows[0]["decimal_comma_detected"] == "true"
    assert rows[0]["rows_in_recife_plausible_range"] == "1"
    assert rows[0]["geocoding_executed"] == "false"
    assert rows[0]["centroid_used"] == "false"
