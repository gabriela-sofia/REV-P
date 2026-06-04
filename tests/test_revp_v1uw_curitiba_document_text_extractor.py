from tests.test_revp_v1uw_curitiba_event_source_snapshotter import install_inputs, set_env
import scripts.protocolo_c.revp_v1uw_curitiba_common as common


def test_text_extractor_does_not_version_full_text(tmp_path, monkeypatch):
    data, _raw = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    common.run_event_source_snapshotter(common.parse_args(["--dry-run"]))
    rows = common.run_document_text_extractor(common.parse_args([]))
    assert rows[0]["raw_text_versioned"] == "false"
    assert int(rows[0]["term_signal_count"]) > 0
