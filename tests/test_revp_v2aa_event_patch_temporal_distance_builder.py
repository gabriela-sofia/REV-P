from tests.test_revp_v2aa_patch_source_registry_scanner import (
    install_base_inputs, set_env,
)
import scripts.protocolo_c.revp_v2aa_common as common


def _run_to_temporal(data, scan):
    install_base_inputs(data, scan)
    common.run_patch_source_registry_scanner(common.parse_args([]))
    common.run_sentinel_filename_date_extractor(common.parse_args([]))
    common.run_sentinel_sidecar_metadata_resolver(common.parse_args([]))
    common.run_patch_date_candidate_consolidator(common.parse_args([]))
    common.run_sentinel_date_confidence_audit(common.parse_args([]))
    return common.run_event_patch_temporal_distance_builder(common.parse_args([]))


def test_temporal_distance_does_not_become_truth(tmp_path, monkeypatch):
    data, scan = set_env(tmp_path, monkeypatch)
    rows = _run_to_temporal(data, scan)
    by_patch = {r["patch_id"]: r for r in rows}
    # P_S2A (2022-05-25) falls within the event window 2022-05-24..2022-05-25.
    assert by_patch["P_S2A"]["temporal_class"] == "WITHIN_EVENT_WINDOW"
    assert by_patch["P_S2A"]["usable_for_contextual_review"] == "true"
    # P_S2B (2022-05-30) is shortly after the window.
    assert by_patch["P_S2B"]["temporal_class"] == "POST_EVENT_NEAR"
    # P_NODATE has no recoverable date -> blocked.
    assert by_patch["P_NODATE"]["temporal_class"] == "TEMPORAL_DISTANCE_BLOCKED_NO_DATE"
    # No temporal row ever creates overlay readiness or ground reference.
    assert all(r["usable_for_overlay_preflight"] == "false" for r in rows)
    assert all(r["can_create_ground_reference"] == "false" for r in rows)
