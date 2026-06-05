import scripts.protocolo_c.revp_v2ai_common as common
from tests.test_revp_v2ai_common import install_v2ah, set_env, write_csv


def test_guardrail_regression_flags_fake_review_and_paths(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_v2ah(data)
    write_csv(data / "v2ai_bad.csv", ["human_review_completed", "source_path"], [{"human_review_completed": "true", "source_path": "local_only/raw.xlsx"}])
    rows = common.run_guardrail_regression(common.parse_args([]))
    failures = [r for r in rows if r["artifact_path"].endswith("v2ai_bad.csv") and r["status"] == "FAIL"]
    assert failures


def test_guardrail_regression_passes_generated_v2ai(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_v2ah(data)
    common.run_all(common.parse_args([]))
    rows = common.run_guardrail_regression(common.parse_args([]))
    assert all(r["status"] == "PASS" for r in rows)
