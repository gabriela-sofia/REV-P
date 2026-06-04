from tests.test_revp_v2ac_event_patch_v2_package_builder import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2ac_common as common


def test_blocker_normalizer_preserves_and_adds_blockers(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    common.run_event_patch_v2_package_builder(common.parse_args([]))
    rows = common.run_blocker_field_normalizer(common.parse_args([]))
    epc0 = [r for r in rows if r["event_patch_candidate_id"] == "EPC0"]
    present = {r["blocker"]: r["present"] for r in epc0}
    # Canonical blockers always recorded, present flagged true/false.
    for blocker in common.CANONICAL_BLOCKERS:
        assert blocker in present
    # Persisting structural blockers must be present.
    assert present["no_overlay"] == "true"
    assert present["no_ground_reference"] == "true"
    assert present["no_training_label"] == "true"
    assert present["patch_truth_forbidden"] == "true"
    assert present["no_explicit_anchor_crosswalk"] == "true"
    # Unlinkable date blocker present for EPC0 (parallel-namespace date).
    assert present["unlinkable_sentinel_date"] == "true"
