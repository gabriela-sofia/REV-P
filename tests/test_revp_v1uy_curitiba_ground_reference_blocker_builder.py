from tests.test_revp_v1uy_curitiba_content_mismatch_resolver import set_env
import scripts.protocolo_c.revp_v1uy_curitiba_common as common


def test_ground_reference_blocker_builder_blocks_required_reasons(tmp_path, monkeypatch):
    data, _v1ux_raw = set_env(tmp_path, monkeypatch)
    rows = common.run_ground_reference_blocker_builder(common.parse_args([]))
    blockers = {r["blocker"] for r in rows}
    assert {"no_observed_geometry", "no_occurrence_table", "no_ground_reference", "patch_truth_forbidden"}.issubset(blockers)
    assert all(r["status"] == "BLOCKED" for r in rows)
    assert all(r["ground_truth_operational"] == "false" for r in rows)
