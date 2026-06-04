from tests.test_revp_v2af_qa_input_manifest_builder import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2af_common as common


def test_gate_pass_with_expected_blockers(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    common.run_all(common.parse_args([]))
    rows = common.run_qa_gate_orchestrator(common.parse_args([]))
    overall = next(r for r in rows if r["qa_component"] == "OVERALL")
    assert overall["gate_status"] == "QA_AUTOMATION_PASS_WITH_EXPECTED_BLOCKERS"
    assert int(overall["failed_checks"]) == 0
    assert overall["required_action"] == "next_stage_may_start"
    assert all(r["gate_status"] != "QA_AUTOMATION_FAIL" for r in rows)
