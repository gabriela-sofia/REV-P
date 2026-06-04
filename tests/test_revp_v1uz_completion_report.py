import os

from tests.test_revp_v1uz_curitiba_context_only_hold_builder import (
    install_base_inputs, read_csv, set_env,
)
import scripts.protocolo_c.revp_v1uz_common as common


def test_completion_report_summarizes_status_and_manifest(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    result = common.run_all(common.parse_args([]))
    assert result["next_target"] == "SENTINEL_DATE_RECOVERY_FOR_EVENT_PATCH_PACKAGES"

    next_actions = read_csv(str(data / "v1uz_next_actions_registry.csv"))
    assert next_actions[0]["action_type"] == "SENTINEL_DATE_RECOVERY_FOR_EVENT_PATCH_PACKAGES"
    assert next_actions[0]["status"] == "RECOMMENDED_NEXT_STEP"

    manifest = read_csv(str(data / "v1uz_versionable_artifacts_manifest.csv"))
    assert manifest, "manifest is empty"
    assert all(r["is_versionable"] == "true" for r in manifest)
    assert all(r["protocol_version"] == "v1uz" for r in manifest)

    report = os.path.join(
        str(tmp_path / "docs" / "metodologia_cientifica"),
        "protocolo_c_relatorio_v1uz_curitiba_context_only_hold_multiregion_rerank.md",
    )
    text = open(report, encoding="utf-8").read()
    assert "CURITIBA_CONTEXT_ONLY_HOLD_NON_OPERATIONAL" in text
    assert "RECIFE_CONTEXTUAL_COORDINATE_LAYER_CONSOLIDATED_NON_OPERATIONAL" in text
    assert "PETROPOLIS_DOCUMENT_ONLY_NO_GEODATA" in text
    # No assistant/tool attribution in docs.
    lowered = text.lower()
    for token in ("claude", "codex", "llm", "assistant", "openai", "anthropic"):
        assert token not in lowered


def test_no_forbidden_true_values_in_any_v1uz_csv(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    common.run_all(common.parse_args([]))
    for name in os.listdir(str(data)):
        if not name.startswith("v1uz_") or not name.endswith(".csv"):
            continue
        for row in read_csv(str(data / name)):
            for key in ("can_create_ground_reference", "can_create_training_label", "ground_truth_operational"):
                if key in row:
                    assert row[key] != "true", f"{name} has {key}=true"
