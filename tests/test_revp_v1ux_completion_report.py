from tests.test_revp_v1ux_curitiba_download_target_builder import install_inputs, read_csv, set_env
import scripts.protocolo_c.revp_v1ux_curitiba_common as common


def test_completion_report_writes_public_artifacts_and_blockers(tmp_path, monkeypatch):
    data, _raw = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    result = common.run_all(common.parse_args(["--dry-run"]))
    blockers = read_csv(data / "v1ux_curitiba_ground_reference_blocker_matrix.csv")
    manifest = read_csv(data / "v1ux_versionable_artifacts_manifest.csv")
    docs = "\n".join(path.read_text(encoding="utf-8") for path in (tmp_path / "docs" / "metodologia_cientifica").glob("protocolo_c_*v1ux*.md"))
    assert result["downloads"] > 0
    assert len(manifest) >= 20
    assert all(r["ground_truth_operational"] == "false" for r in blockers)
    assert all(r["can_create_ground_reference"] == "false" for r in blockers)
    assert all(r["raw_data_versioned"] == "false" for r in blockers)
    assert "Claude" not in docs and "Codex" not in docs and "LLM" not in docs and "assistente" not in docs
