from tests.test_revp_v1uv_curitiba_source_target_builder import install_candidate_discovery, set_env
import scripts.protocolo_c.revp_v1uv_curitiba_common as common


def test_candidate_builder_does_not_invent_date_or_event_id(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_candidate_discovery(data, official=True, date="", hazard="alagamento")
    rows = common.run_candidate_event_builder(common.parse_args([]))
    assert rows[0]["event_id_candidate"] == ""
    assert rows[0]["can_enter_multiregion_registry"] == "false"
