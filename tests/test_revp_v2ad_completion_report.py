import os

from tests.test_revp_v2ad_package_contract_qa import install_base_inputs, read_csv, set_env
import scripts.protocolo_c.revp_v2ad_common as common


def test_completion_report_and_manifest(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    result = common.run_all(common.parse_args([]))
    assert result["gate"] == "QA_PASS_WITH_EXPECTED_BLOCKERS"
    assert result["negatives_ok"] is True
    assert result["next_target"]

    next_actions = read_csv(str(data / "v2ad_next_actions_registry.csv"))
    assert next_actions[0]["status"] == "RECOMMENDED_NEXT_STEP"

    manifest = read_csv(str(data / "v2ad_versionable_artifacts_manifest.csv"))
    assert manifest
    assert all(r["is_versionable"] == "true" for r in manifest)
    assert all(r["protocol_version"] == "v2ad" for r in manifest)

    docs = str(tmp_path / "docs" / "metodologia_cientifica")
    status = open(os.path.join(docs, "protocolo_c_status_atual_v2ad.md"), encoding="utf-8").read()
    assert common.MAX_STATUS in status
    report = open(os.path.join(docs, "protocolo_c_relatorio_v2ad_event_patch_v2_qa_harness.md"), encoding="utf-8").read()
    assert "no overlay" in report.lower()
    assert "ground reference" in report.lower()
    lowered = report.lower()
    for token in ("claude", "codex", "llm", "assistant", "openai", "anthropic"):
        assert token not in lowered


def test_no_forbidden_true_in_any_v2ad_csv(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    common.run_all(common.parse_args([]))
    for name in os.listdir(str(data)):
        if not name.startswith("v2ad_") or not name.endswith(".csv"):
            continue
        for row in read_csv(str(data / name)):
            for key in ("can_create_ground_reference", "can_create_training_label",
                        "ground_truth_operational", "crosswalk_inferred", "sentinel_date_inferred"):
                if key in row:
                    assert row[key] != "true", f"{name} has {key}=true"
