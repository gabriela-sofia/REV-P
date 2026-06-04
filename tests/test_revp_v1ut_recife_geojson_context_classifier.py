from tests.test_revp_v1ut_recife_coordinate_asset_locator import install_asset, set_env
import scripts.protocolo_c.revp_v1ut_recife_common as common


def test_geojson_context_classifier_blocks_equipment_and_region_layers(tmp_path, monkeypatch):
    data, raw = set_env(tmp_path, monkeypatch)
    install_asset(data, raw, "recife_equipment_context.geojson", "equipment", "geospatial_vector",
                  "CONTEXTUAL_OFFICIAL_LAYER", "geometry")
    install_asset(data, raw, "recife_bairro_regional.geojson", "bairro", "geospatial_vector",
                  "CONTEXTUAL_OFFICIAL_LAYER", "geometry")
    common.run_coordinate_asset_locator()
    rows = common.run_geojson_context_classifier()
    assert len(rows) == 2
    assert all(r["can_promote_to_occurrence_candidate"] == "false" for r in rows)
    assert {r["coordinate_role"] for r in rows} >= {"ADMIN_REGION_GEOMETRY", "CONTEXTUAL_EQUIPMENT_OR_INFRASTRUCTURE_COORDINATE"}
