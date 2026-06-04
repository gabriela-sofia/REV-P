from tests.test_revp_v1uy_curitiba_content_mismatch_resolver import install_base_inputs, set_env
import scripts.protocolo_c.revp_v1uy_curitiba_common as common


def test_possible_occurrence_audit_requires_gates_and_blocks_overlay(tmp_path, monkeypatch):
    data, v1ux_raw = set_env(tmp_path, monkeypatch)
    install_base_inputs(data, v1ux_raw)
    common.run_geodata_endpoint_probe(common.parse_args([]))
    common.run_layer_metadata_extractor(common.parse_args([]))
    common.run_context_layer_classifier(common.parse_args([]))
    rows = common.run_possible_occurrence_layer_audit(common.parse_args([]))
    assert rows
    assert any(r["can_advance_to_controlled_download"] == "true" for r in rows)
    assert all(r["can_advance_to_overlay_preflight"] == "false" for r in rows)
    assert all(r["can_create_training_label"] == "false" for r in rows)
