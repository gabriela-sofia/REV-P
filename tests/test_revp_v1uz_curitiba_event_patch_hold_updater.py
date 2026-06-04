from tests.test_revp_v1uz_curitiba_context_only_hold_builder import (
    install_base_inputs, set_env,
)
import scripts.protocolo_c.revp_v1uz_common as common


def test_event_patch_hold_update_keeps_candidate_only(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    rows = common.run_curitiba_event_patch_hold_updater(common.parse_args([]))
    assert len(rows) == 3
    assert all(r["event_patch_candidate_only"] == "true" for r in rows)
    assert all(r["context_only_hold_status"] == "CONTEXT_ONLY_HOLD" for r in rows)
    assert all(r["sentinel_date_status"] == "SENTINEL_DATE_MISSING" for r in rows)
    assert all(r["occurrence_geometry_status"] == "NO_OCCURRENCE_GEOMETRY" for r in rows)
    assert all(r["overlay_status"] == "BLOCKED" for r in rows)
    assert all(r["linkage_basis"] == "REGION_ONLY_EVENT_CANDIDATE" for r in rows)
    assert all(r["can_create_ground_reference"] == "false" for r in rows)
    assert all(r["can_create_training_label"] == "false" for r in rows)
