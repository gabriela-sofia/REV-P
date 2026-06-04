from tests.test_revp_v2aa_patch_source_registry_scanner import (
    install_base_inputs, set_env,
)
import scripts.protocolo_c.revp_v2aa_common as common


def _run_to_confidence(data, scan):
    install_base_inputs(data, scan)
    common.run_patch_source_registry_scanner(common.parse_args([]))
    common.run_sentinel_filename_date_extractor(common.parse_args([]))
    common.run_sentinel_sidecar_metadata_resolver(common.parse_args([]))
    common.run_patch_date_candidate_consolidator(common.parse_args([]))
    return common.run_sentinel_date_confidence_audit(common.parse_args([]))


def test_confidence_audit_only_releases_high_and_medium(tmp_path, monkeypatch):
    data, scan = set_env(tmp_path, monkeypatch)
    rows = _run_to_confidence(data, scan)
    by_patch = {r["patch_id"]: r for r in rows}
    # Canonical Sentinel filename -> HIGH and usable.
    assert by_patch["P_S2A"]["confidence_class"] == "HIGH_CONFIDENCE"
    assert by_patch["P_S2A"]["usable_for_temporal_linkage"] == "true"
    # Conflict -> blocked, not usable.
    assert by_patch["P_CONFLICT"]["confidence_class"] == "BLOCKED_CONFLICT"
    assert by_patch["P_CONFLICT"]["usable_for_temporal_linkage"] == "false"
    # Ambiguous / no-date -> MISSING, not usable.
    assert by_patch["P_AMB"]["confidence_class"] == "MISSING"
    assert by_patch["P_AMB"]["usable_for_temporal_linkage"] == "false"
    # Every usable date is HIGH or MEDIUM confidence (never LOW/BLOCKED/MISSING).
    for r in rows:
        if r["usable_for_temporal_linkage"] == "true":
            assert r["confidence_class"] in {"HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE"}
