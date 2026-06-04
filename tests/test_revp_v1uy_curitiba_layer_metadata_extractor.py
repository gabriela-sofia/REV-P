from tests.test_revp_v1uy_curitiba_content_mismatch_resolver import install_base_inputs, set_env
import scripts.protocolo_c.revp_v1uy_curitiba_common as common


def test_layer_metadata_extractor_hashes_fields_without_features(tmp_path, monkeypatch):
    data, v1ux_raw = set_env(tmp_path, monkeypatch)
    install_base_inputs(data, v1ux_raw)
    common.run_geodata_endpoint_probe(common.parse_args([]))
    rows = common.run_layer_metadata_extractor(common.parse_args([]))
    assert any(r["has_date_field"] == "true" and r["has_hazard_field"] == "true" for r in rows)
    assert all(r["fields_hash"] for r in rows)
    assert all("data_evento" not in str(r) for r in rows)
