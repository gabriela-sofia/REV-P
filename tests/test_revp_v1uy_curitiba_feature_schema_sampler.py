from tests.test_revp_v1uy_curitiba_content_mismatch_resolver import install_base_inputs, set_env
import scripts.protocolo_c.revp_v1uy_curitiba_common as common


def test_feature_schema_sampler_does_not_version_raw_geometry(tmp_path, monkeypatch):
    data, v1ux_raw = set_env(tmp_path, monkeypatch)
    install_base_inputs(data, v1ux_raw)
    common.run_geodata_endpoint_probe(common.parse_args([]))
    common.run_layer_metadata_extractor(common.parse_args([]))
    rows = common.run_feature_schema_sampler(common.parse_args([]))
    assert rows
    assert all(r["geometry_sample_versioned"] == "false" for r in rows)
    assert all(r["raw_feature_versioned"] == "false" for r in rows)
    assert all(r["sampled_feature_count"] == "0" for r in rows)
