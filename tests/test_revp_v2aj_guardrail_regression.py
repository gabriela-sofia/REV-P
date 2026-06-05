import scripts.protocolo_c.revp_v2aj_common as common
from tests.test_revp_v2aj_common import install_inputs, set_env, write_csv


def test_guardrail_flags_unmarked_markdown_overclaim(tmp_path, monkeypatch):
    data, docs = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    (docs / "bad_v2aj.md").write_text("ground truth validado\n", encoding="utf-8")
    rows = common.run_guardrail_regression(common.parse_args([]))
    assert any(r["artifact_path"].endswith("bad_v2aj.md") and r["status"] == "FAIL" for r in rows)


def test_guardrail_passes_generated_outputs(tmp_path, monkeypatch):
    data, _ = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    common.run_all(common.parse_args([]))
    rows = common.run_guardrail_regression(common.parse_args([]))
    assert all(r["status"] == "PASS" for r in rows)


def test_guardrail_flags_absolute_and_local_paths(tmp_path, monkeypatch):
    data, _ = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    write_csv(data / "v2aj_bad_paths.csv", ["source_artifact"], [{"source_artifact": r"C:\Users\gabriela\x.csv"}, {"source_artifact": "local_only/x.csv"}])
    rows = common.run_guardrail_regression(common.parse_args([]))
    bad = [r for r in rows if r["artifact_path"].endswith("v2aj_bad_paths.csv") and r["status"] == "FAIL"]
    assert bad
