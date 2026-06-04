from tests.test_revp_v1uy_curitiba_content_mismatch_resolver import run_pipeline, set_env
import scripts.protocolo_c.revp_v1uy_curitiba_common as common


def test_controlled_feature_download_planner_plans_without_execution(tmp_path, monkeypatch):
    data, v1ux_raw = set_env(tmp_path, monkeypatch)
    run_pipeline(data, v1ux_raw)
    rows = common.load_csv(common.dataset_path("v1uy_curitiba_controlled_feature_download_plan.csv"))
    assert any(r["plan_status"] == "CONTROLLED_DOWNLOAD_CANDIDATE_FOR_V1UZ" for r in rows)
    assert all(r["can_execute_now"] == "false" for r in rows)
    assert all(r["raw_data_versioned"] == "false" for r in rows)
