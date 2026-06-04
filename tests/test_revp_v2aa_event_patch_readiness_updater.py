from tests.test_revp_v2aa_patch_source_registry_scanner import (
    install_base_inputs, set_env,
)
import scripts.protocolo_c.revp_v2aa_common as common


def _run_to_readiness(data, scan):
    install_base_inputs(data, scan)
    common.run_patch_source_registry_scanner(common.parse_args([]))
    common.run_sentinel_filename_date_extractor(common.parse_args([]))
    common.run_sentinel_sidecar_metadata_resolver(common.parse_args([]))
    common.run_patch_date_candidate_consolidator(common.parse_args([]))
    common.run_sentinel_date_confidence_audit(common.parse_args([]))
    common.run_event_patch_temporal_distance_builder(common.parse_args([]))
    return common.run_event_patch_readiness_updater(common.parse_args([]))


def test_readiness_updater_improves_only_temporal_dimensions(tmp_path, monkeypatch):
    data, scan = set_env(tmp_path, monkeypatch)
    rows = _run_to_readiness(data, scan)
    # Overlay / ground reference / training stay BLOCKED for every candidate.
    blocked_dims = {"overlay_readiness", "ground_reference_readiness", "training_readiness"}
    for r in rows:
        if r["dimension"] in blocked_dims:
            assert r["classification"] == "BLOCKED"
    # The candidate with a recovered usable date improves its sentinel/temporal support.
    s2a = [r for r in rows if r["patch_id"] == "P_S2A"]
    dims = {r["dimension"]: r["classification"] for r in s2a}
    assert dims["sentinel_date_support"] == "RECOVERED_USABLE"
    assert dims["temporal_linkage"] == "IMPROVED_REVIEW_ONLY"
    assert dims["contextual_review_readiness"] == "CONTEXTUAL_REVIEW_READY"
    # The no-date candidate stays blocked on sentinel/temporal support.
    nodate = {r["dimension"]: r["classification"] for r in rows if r["patch_id"] == "P_NODATE"}
    assert nodate["sentinel_date_support"] == "STILL_MISSING_OR_BLOCKED"
    assert all(r["can_create_training_label"] == "false" for r in rows)
    assert all(r["sentinel_date_inferred"] == "false" for r in rows)
