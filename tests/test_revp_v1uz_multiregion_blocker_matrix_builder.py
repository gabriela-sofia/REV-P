from tests.test_revp_v1uz_curitiba_context_only_hold_builder import (
    install_base_inputs, set_env,
)
import scripts.protocolo_c.revp_v1uz_common as common


def _prepare(data):
    install_base_inputs(data)
    common.run_curitiba_context_only_hold_builder(common.parse_args([]))
    common.run_multiregion_closure_status_builder(common.parse_args([]))


def test_blocker_matrix_includes_observed_geometry_and_sentinel_date(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    _prepare(data)
    rows = common.run_multiregion_blocker_matrix_builder(common.parse_args([]))
    blockers = {r["blocker"] for r in rows}
    assert "no_observed_geometry" in blockers
    assert "no_sentinel_date" in blockers
    assert "no_occurrence_coordinates" in blockers
    assert "patch_truth_forbidden" in blockers
    # Every region keeps observed geometry and sentinel date blocked.
    for region in ("REC", "PET", "CUR"):
        region_rows = {r["blocker"]: r for r in rows if r["region"] == region}
        assert region_rows["no_observed_geometry"]["status"] == "BLOCKED"
        assert region_rows["no_sentinel_date"]["status"] == "BLOCKED"
    assert all(r["can_create_ground_reference"] == "false" for r in rows)
