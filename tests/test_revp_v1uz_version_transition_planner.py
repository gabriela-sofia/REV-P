from tests.test_revp_v1uz_curitiba_context_only_hold_builder import (
    install_base_inputs, set_env,
)
import scripts.protocolo_c.revp_v1uz_common as common


def test_transition_planner_does_not_start_next_version(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    common.run_next_programming_target_ranker(common.parse_args([]))
    rows = common.run_version_transition_planner(common.parse_args([]))
    assert len(rows) == 1
    plan = rows[0]
    assert plan["implementation_not_started"] == "true"
    assert plan["selected_next_target"] == "SENTINEL_DATE_RECOVERY_FOR_EVENT_PATCH_PACKAGES"
    assert plan["selected_version"].startswith("v2aa")
    assert "Sentinel Date Recovery" in plan["selected_version"]
    # Guardrails are carried into the plan, not relaxed.
    assert "can_create_ground_reference=false" in plan["guardrails"]
    assert "no_overlay_executed=true" in plan["guardrails"]
