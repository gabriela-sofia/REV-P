from tests.test_revp_v1uy_curitiba_content_mismatch_resolver import install_base_inputs, read_csv, set_env
import scripts.protocolo_c.revp_v1uy_curitiba_common as common


def test_completion_report_writes_manifest_and_technical_docs(tmp_path, monkeypatch):
    data, v1ux_raw = set_env(tmp_path, monkeypatch)
    install_base_inputs(data, v1ux_raw)
    result = common.run_all(common.parse_args([]))
    manifest = read_csv(data / "v1uy_versionable_artifacts_manifest.csv")
    docs = "\n".join(p.read_text(encoding="utf-8") for p in (tmp_path / "docs" / "metodologia_cientifica").glob("*v1uy*.md"))
    assert result["classes"] > 0
    assert len(manifest) >= 18
    assert "Ground reference remains blocked" in docs
    assert "Claude" not in docs and "Codex" not in docs and "LLM" not in docs and "assistente" not in docs
