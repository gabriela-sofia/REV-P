from tests.test_revp_v1ux_curitiba_download_target_builder import install_inputs, set_env
import scripts.protocolo_c.revp_v1ux_curitiba_common as common


def test_downloader_writes_raw_only_manifest_metadata(tmp_path, monkeypatch):
    data, raw = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    common.run_download_target_builder(common.parse_args([]))
    rows = common.run_public_artifact_downloader(common.parse_args(["--dry-run"]))
    assert rows
    assert list(raw.iterdir())
    assert all(r["raw_data_versioned"] == "false" for r in rows)
    assert all("\\" not in r["safe_filename"] and "/" not in r["safe_filename"] for r in rows)
    assert all(str(tmp_path) not in str(r) for r in rows)
