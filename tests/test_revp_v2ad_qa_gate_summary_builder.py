from tests.test_revp_v2ad_package_contract_qa import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2ad_common as common


def test_gate_summary_pass_with_expected_blockers(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    common.run_all(common.parse_args([]))
    rows = common.run_qa_gate_summary_builder(common.parse_args([]))
    overall = next(r for r in rows if r["qa_group"] == "OVERALL")
    assert overall["gate_status"] == "QA_PASS_WITH_EXPECTED_BLOCKERS"
    assert int(overall["failed_checks"]) == 0
    assert int(overall["expected_blockers"]) >= 1
    # No per-group gate is QA_FAIL on real data.
    assert all(r["gate_status"] != "QA_FAIL" for r in rows)
