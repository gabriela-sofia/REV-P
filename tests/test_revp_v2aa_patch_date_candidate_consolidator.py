from tests.test_revp_v2aa_patch_source_registry_scanner import (
    install_base_inputs, set_env,
)
import scripts.protocolo_c.revp_v2aa_common as common


def _run_to_consolidation(data, scan):
    install_base_inputs(data, scan)
    common.run_patch_source_registry_scanner(common.parse_args([]))
    common.run_sentinel_filename_date_extractor(common.parse_args([]))
    common.run_sentinel_sidecar_metadata_resolver(common.parse_args([]))
    return common.run_patch_date_candidate_consolidator(common.parse_args([]))


def test_consolidator_blocks_date_conflict(tmp_path, monkeypatch):
    data, scan = set_env(tmp_path, monkeypatch)
    rows = _run_to_consolidation(data, scan)
    by_patch = {r["patch_id"]: r for r in rows}
    # P_CONFLICT has two different scene_date values -> conflict blocked, no selection.
    conflict = by_patch["P_CONFLICT"]
    assert conflict["consolidation_status"] == "DATE_CONFLICT_BLOCKED"
    assert conflict["conflict_status"] == "CONFLICT"
    assert conflict["selected_sentinel_date"] == ""
    assert conflict["sentinel_date_recovered"] == "false"
    # P_S2A confirmed from a single source.
    assert by_patch["P_S2A"]["consolidation_status"] in {
        "DATE_CONFIRMED_SINGLE_SOURCE", "DATE_CONFIRMED_MULTI_SOURCE_AGREE",
    }
    assert by_patch["P_S2A"]["selected_sentinel_date"] == "2022-05-25"
    # Nothing is ever marked inferred.
    assert all(r["sentinel_date_inferred"] == "false" for r in rows)
