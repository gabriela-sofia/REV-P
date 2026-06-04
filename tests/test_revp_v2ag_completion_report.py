import os

import scripts.protocolo_c.revp_v2ag_common as common
from tests.test_revp_v2ag_crosswalk_source_inventory import install_packages, read_csv, set_env, write_csv


def test_completion_report_writes_manifest_and_preserves_guardrails(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_packages(data)
    write_csv(data / "explicit.csv", ["event_patch_candidate_id", "patch_id", "refpatch_id"], [{"event_patch_candidate_id": "EPC_SYN_001", "patch_id": "REC_00001", "refpatch_id": "REFPATCH_SYN_DATE_001"}])
    common.run_all(common.parse_args([]))
    manifest = read_csv(data / "v2ag_versionable_artifacts_manifest.csv")
    blocker = read_csv(data / "v2ag_ground_reference_blocker_matrix.csv")
    assert manifest
    assert all("local_only/" not in r["artifact_path"] for r in manifest)
    assert all(r["can_create_ground_reference"] == "false" for r in blocker)
    assert all(r["can_create_training_label"] == "false" for r in blocker)
    assert all(r["ground_truth_operational"] == "false" for r in blocker)
    report = os.path.join(common.DOCS_DIR, "protocolo_c_relatorio_v2ag_sentinel_date_crosswalk_discovery.md")
    with open(report, encoding="utf-8") as f:
        text = f.read()
    assert "Codex" not in text
    assert "Claude" not in text
