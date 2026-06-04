from tests.test_revp_v1uv_curitiba_source_target_builder import install_candidate_discovery, set_env
import scripts.protocolo_c.revp_v1uv_curitiba_common as common


def test_event_registry_updater_keeps_missing_without_strong_candidate(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_candidate_discovery(data, official=True, date="", hazard="alagamento")
    common.run_candidate_event_builder(common.parse_args([]))
    common.run_event_evidence_audit(common.parse_args([]))
    rows = common.run_event_registry_updater(common.parse_args([]))
    assert rows[0]["new_status"] == "CUR_EVENT_REGISTRY_MISSING"
    assert rows[0]["can_create_ground_reference"] == "false"
