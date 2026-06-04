from tests.test_revp_v2ae_canonical_region_registry_builder import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2ae_common as common


def test_safe_use_policy_prohibits_truth_label_overlay(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    rows = common.run_safe_use_policy_registry_builder(common.parse_args([]))
    assert any(r["scope"] == "GLOBAL" for r in rows)
    assert {r["region"] for r in rows if r["scope"] == "REGION"} == {"REC", "PET", "CUR"}
    for r in rows:
        for prohibited in ("ground_truth", "training_label", "overlay_truth", "event_validated_by_sentinel", "context_layer_as_occurrence"):
            assert prohibited in r["prohibited_use"]
        assert "review_only" in r["safe_use"]
        assert r["can_create_ground_reference"] == "false"
        assert r["can_create_training_label"] == "false"
