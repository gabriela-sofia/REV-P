import os

from tests.test_revp_v2af_qa_input_manifest_builder import install_base_inputs, set_env, write_csv
import scripts.protocolo_c.revp_v2af_common as common


def test_freshness_clean(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    rows = common.run_artifact_freshness_auditor(common.parse_args([]))
    required = [r for r in rows if r["required"] == "true"]
    assert all(r["freshness_status"] == "FRESH_ENOUGH_FOR_QA" for r in required)


def test_freshness_detects_empty_and_missing(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    # Empty a required artifact (header only, no rows).
    write_csv(str(data / "v2ae_canonical_event_registry.csv"), ["canonical_event_id", "event_id", "canonical_event_status"], [])
    # Remove another required artifact entirely.
    os.remove(str(data / "v2ac_event_patch_v2_package_registry.csv"))
    rows = common.run_artifact_freshness_auditor(common.parse_args([]))
    by_path = {os.path.basename(r["artifact_path"]): r for r in rows}
    assert by_path["v2ae_canonical_event_registry.csv"]["freshness_status"] == "EMPTY_ARTIFACT"
    assert by_path["v2ac_event_patch_v2_package_registry.csv"]["freshness_status"] == "MISSING_ARTIFACT"
