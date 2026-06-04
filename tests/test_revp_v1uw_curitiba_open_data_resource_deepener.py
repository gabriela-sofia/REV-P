from tests.test_revp_v1uw_curitiba_event_source_snapshotter import install_inputs, set_env
import scripts.protocolo_c.revp_v1uw_curitiba_common as common


def test_open_data_deepener_does_not_download_or_create_event(tmp_path, monkeypatch):
    data, _raw = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    rows = common.run_open_data_resource_deepener(common.parse_args([]))
    assert rows[0]["download_priority"] == "HIGH"
    assert rows[0]["can_create_ground_reference"] == "false"
