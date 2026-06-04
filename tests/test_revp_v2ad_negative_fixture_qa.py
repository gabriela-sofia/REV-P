from tests.test_revp_v2ad_package_contract_qa import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2ad_common as common


def test_negative_fixture_qa_detects_all_injected_violations(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    rows = common.run_negative_fixture_qa(common.parse_args([]))
    assert len(rows) == 10
    # Every fixture row is handled correctly (violation detected or no false positive).
    assert all(r["status"].startswith("PASS") for r in rows)
    # Each injected (non-clean) fixture is detected.
    injected = [r for r in rows if r["injected_violation"] != "none"]
    assert injected
    assert all(r["detected"] == "true" for r in injected)
    # Clean fixtures produce no detection (no false positive).
    clean = [r for r in rows if r["injected_violation"] == "none"]
    assert clean
    assert all(r["detected"] == "false" for r in clean)
