from tests.test_revp_v2ad_package_contract_qa import install_base_inputs, set_env, write_csv
import scripts.protocolo_c.revp_v2ad_common as common


def test_guardrail_qa_clean_on_real_outputs(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    common.run_package_contract_qa(common.parse_args([]))
    rows = common.run_guardrail_qa(common.parse_args([]))
    assert rows
    assert all(r["status"] == "PASS" for r in rows)


def test_guardrail_qa_detects_forbidden_true_and_overlay_release(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    bad = str(data / "tampered.csv")
    write_csv(bad, ["event_patch_candidate_id", "can_create_ground_reference", "overlay_status"],
              [{"event_patch_candidate_id": "EPCX", "can_create_ground_reference": "true", "overlay_status": "ALLOWED"}])
    rows = common.run_guardrail_qa(common.parse_args([]), artifacts=[bad])
    counts = {r["check_type"]: int(r["violation_count"]) for r in rows}
    assert counts["forbidden_true_value"] >= 1
    assert counts["overlay_or_gr_or_training_released"] >= 1
    assert any(r["status"] == "FAIL" for r in rows)
