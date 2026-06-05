import scripts.protocolo_c.revp_v2ah_common as common
from tests.test_revp_v2ah_common import install_inputs, set_env, write_csv


def test_guardrail_regression_flags_forbidden_true(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    write_csv(data / "v2ah_bad.csv", ["ground_truth"], [{"ground_truth": "true"}])
    rows = common.run_guardrail_regression(common.parse_args([]))
    bad = [r for r in rows if r["artifact_path"].endswith("v2ah_bad.csv") and r["status"] == "FAIL"]
    assert bad


def test_guardrail_regression_passes_generated_outputs(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    common.run_all(common.parse_args([]))
    rows = common.run_guardrail_regression(common.parse_args([]))
    assert all(r["status"] == "PASS" for r in rows)
