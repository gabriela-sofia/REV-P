import os

from tests.test_revp_v2af_qa_input_manifest_builder import install_base_inputs, read_csv, set_env
import scripts.protocolo_c.revp_v2af_common as common


def test_completion_report_and_manifest(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    result = common.run_all(common.parse_args([]))
    assert result["gate"] == "QA_AUTOMATION_PASS_WITH_EXPECTED_BLOCKERS"
    assert result["no_failures"] is True
    assert result["missing_artifacts"] == 0
    assert result["next_target"]

    next_actions = read_csv(str(data / "v2af_next_actions_registry.csv"))
    assert next_actions[0]["status"] == "RECOMMENDED_NEXT_STEP"

    manifest = read_csv(str(data / "v2af_versionable_artifacts_manifest.csv"))
    assert manifest
    assert all(r["is_versionable"] == "true" for r in manifest)
    assert all(r["protocol_version"] == "v2af" for r in manifest)

    docs = str(tmp_path / "docs" / "metodologia_cientifica")
    status = open(os.path.join(docs, "protocolo_c_status_atual_v2af.md"), encoding="utf-8").read()
    assert common.MAX_STATUS in status
    report = open(os.path.join(docs, "protocolo_c_relatorio_v2af_event_patch_v2_qa_automation.md"), encoding="utf-8").read()
    assert "no overlay" in report.lower()
    assert "ground reference" in report.lower()
    lowered = report.lower()
    for token in ("claude", "codex", "llm", "assistant", "openai", "anthropic"):
        assert token not in lowered


def test_no_forbidden_true_in_any_v2af_csv(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    common.run_all(common.parse_args([]))
    for name in os.listdir(str(data)):
        if not name.startswith("v2af_") or not name.endswith(".csv"):
            continue
        for row in read_csv(str(data / name)):
            for key in ("can_create_ground_reference", "can_create_training_label",
                        "ground_truth_operational", "crosswalk_inferred", "sentinel_date_inferred"):
                if key in row:
                    assert row[key] != "true", f"{name} has {key}=true"
