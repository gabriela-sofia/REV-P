from tests.test_revp_v2aa_patch_source_registry_scanner import (
    install_base_inputs, set_env,
)
import scripts.protocolo_c.revp_v2aa_common as common


def _run_to_reduction(data, scan):
    install_base_inputs(data, scan)
    common.run_patch_source_registry_scanner(common.parse_args([]))
    common.run_sentinel_filename_date_extractor(common.parse_args([]))
    common.run_sentinel_sidecar_metadata_resolver(common.parse_args([]))
    common.run_patch_date_candidate_consolidator(common.parse_args([]))
    common.run_sentinel_date_confidence_audit(common.parse_args([]))
    common.run_event_patch_temporal_distance_builder(common.parse_args([]))
    return common.run_multiregion_temporal_blocker_reducer(common.parse_args([]))


def test_blocker_reducer_quantifies_no_sentinel_date_reduction(tmp_path, monkeypatch):
    data, scan = set_env(tmp_path, monkeypatch)
    rows = _run_to_reduction(data, scan)
    rec = {r["region"]: r for r in rows}.get("REC")
    assert rec is not None
    # REC has 3 patches in the registry resolution; P_S2A and P_S2B recovered.
    assert int(rec["total_patches"]) == 3
    assert int(rec["patches_with_recovered_date"]) >= 2
    assert int(rec["missing_or_blocked_dates"]) >= 1
    assert int(rec["event_patch_candidates_improved"]) >= 1
    assert rec["blocker_reduction_status"] in {
        "BLOCKER_PARTIALLY_REDUCED", "BLOCKER_FULLY_REDUCED",
    }
    # Counts are consistent: recovered + missing == total.
    assert int(rec["patches_with_recovered_date"]) + int(rec["missing_or_blocked_dates"]) == int(rec["total_patches"])
