from tests.test_revp_v1uw_curitiba_event_source_snapshotter import install_inputs, set_env
import scripts.protocolo_c.revp_v1uw_curitiba_common as common


def test_hydromet_anchor_does_not_support_occurrence_claim(tmp_path, monkeypatch):
    data, _raw = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    rows = common.run_hydromet_anchor_resolver(common.parse_args([]))
    assert rows[0]["hydromet_support_class"] == "HYDROMET_ANCHOR_AVAILABLE"
    assert rows[0]["can_support_occurrence_claim"] == "false"
