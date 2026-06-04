import os

from tests.test_revp_v2aa_patch_source_registry_scanner import (
    install_base_inputs, read_csv, set_env,
)
import scripts.protocolo_c.revp_v2aa_common as common


def test_completion_report_and_manifest(tmp_path, monkeypatch):
    data, scan = set_env(tmp_path, monkeypatch)
    install_base_inputs(data, scan)
    result = common.run_all(common.parse_args([]))
    assert result["recovered"] >= 2
    assert result["next_target"]

    next_actions = read_csv(str(data / "v2aa_next_actions_registry.csv"))
    assert next_actions[0]["status"] == "RECOMMENDED_NEXT_STEP"

    manifest = read_csv(str(data / "v2aa_versionable_artifacts_manifest.csv"))
    assert manifest
    assert all(r["is_versionable"] == "true" for r in manifest)
    assert all(r["protocol_version"] == "v2aa" for r in manifest)

    report = os.path.join(
        str(tmp_path / "docs" / "metodologia_cientifica"),
        "protocolo_c_relatorio_v2aa_sentinel_date_recovery.md",
    )
    text = open(report, encoding="utf-8").read()
    assert common.MAX_STATUS in open(
        os.path.join(str(tmp_path / "docs" / "metodologia_cientifica"), "protocolo_c_status_atual_v2aa.md"),
        encoding="utf-8",
    ).read()
    assert "no overlay" in text.lower()
    assert "ground reference" in text.lower()
    lowered = text.lower()
    for token in ("claude", "codex", "llm", "assistant", "openai", "anthropic"):
        assert token not in lowered


def test_no_forbidden_true_or_inferred_in_any_v2aa_csv(tmp_path, monkeypatch):
    data, scan = set_env(tmp_path, monkeypatch)
    install_base_inputs(data, scan)
    common.run_all(common.parse_args([]))
    for name in os.listdir(str(data)):
        if not name.startswith("v2aa_") or not name.endswith(".csv"):
            continue
        for row in read_csv(str(data / name)):
            for key in ("can_create_ground_reference", "can_create_training_label",
                        "ground_truth_operational", "sentinel_date_inferred"):
                if key in row:
                    assert row[key] != "true", f"{name} has {key}=true"
