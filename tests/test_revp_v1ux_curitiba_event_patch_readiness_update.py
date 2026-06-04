from tests.test_revp_v1ux_curitiba_download_target_builder import run_fixture_pipeline, set_env
import scripts.protocolo_c.revp_v1ux_curitiba_common as common


def test_readiness_update_preserves_blocked_operational_gates(tmp_path, monkeypatch):
    data, raw = set_env(tmp_path, monkeypatch)
    run_fixture_pipeline(data, raw)
    rows = common.load_csv(common.dataset_path("v1ux_curitiba_event_patch_readiness_update.csv"))
    dims = {r["dimension"] for r in rows}
    assert {"overlay_readiness", "ground_reference_readiness", "training_readiness"}.issubset(dims)
    assert all(r["ground_truth_operational"] == "false" for r in rows)
    assert all(r["can_create_ground_reference"] == "false" for r in rows)
    assert all(r["no_coordinates_invented"] == "true" for r in rows)
