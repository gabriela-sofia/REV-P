from tests.test_revp_v1uw_curitiba_event_source_snapshotter import install_inputs, set_env
import scripts.protocolo_c.revp_v1uw_curitiba_common as common


def test_geocuritiba_context_layer_never_becomes_event(tmp_path, monkeypatch):
    data, _raw = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    rows = common.run_geocuritiba_layer_deepener(common.parse_args([]))
    assert rows[0]["event_specificity"] == "CONTEXT_LAYER_NOT_EVENT"
    assert rows[0]["can_support_observed_occurrence"] == "false"
