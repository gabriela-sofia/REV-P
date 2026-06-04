from tests.test_revp_v1uu_recife_contextual_coordinate_asset_classifier import install_minimal_inputs, set_env
import scripts.protocolo_c.revp_v1uu_recife_common as common


def test_next_action_ranker_does_not_insist_on_recife_coordinate_recovery(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_minimal_inputs(data)
    common.run_contextual_coordinate_asset_classifier()
    rows = common.run_next_action_ranker()
    assert rows[0]["next_action"] != "RECIFE_COORDINATE_RECOVERY"
    assert rows[0]["next_action"] == "CURITIBA_EVENT_REGISTRY_AND_PUBLIC_SOURCE_DISCOVERY"
