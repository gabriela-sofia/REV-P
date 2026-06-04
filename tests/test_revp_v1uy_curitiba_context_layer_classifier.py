from tests.test_revp_v1uy_curitiba_content_mismatch_resolver import install_base_inputs, set_env
import scripts.protocolo_c.revp_v1uy_curitiba_common as common


def test_context_layer_classifier_keeps_context_out_of_ground_reference(tmp_path, monkeypatch):
    data, v1ux_raw = set_env(tmp_path, monkeypatch)
    install_base_inputs(data, v1ux_raw)
    common.run_geodata_endpoint_probe(common.parse_args([]))
    common.run_layer_metadata_extractor(common.parse_args([]))
    rows = common.run_context_layer_classifier(common.parse_args([]))
    assert any(r["layer_class"] == "DRAINAGE_CONTEXT_LAYER" for r in rows)
    assert any(r["layer_class"] == "EVENT_SPECIFIC_OCCURRENCE_LAYER_CANDIDATE" for r in rows)
    assert all(r["can_create_ground_reference"] == "false" for r in rows)
