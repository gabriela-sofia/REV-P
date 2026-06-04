from pathlib import Path

import scripts.protocolo_c.revp_v2ag_common as common
from tests.test_revp_v2ag_crosswalk_source_inventory import install_packages, set_env, write_csv


def test_key_extractor_hashes_raw_values_and_classifies_namespaces(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_packages(data)
    write_csv(data / "candidate_registry.csv", ["event_patch_candidate_id", "patch_id", "refpatch_id"], [{"event_patch_candidate_id": "EPC_SYN_001", "patch_id": "REC_00001", "refpatch_id": "REFPATCH_SYN_DATE_001"}])
    common.run_crosswalk_source_inventory(common.parse_args([]))
    rows = common.run_patch_identity_key_extractor(common.parse_args([]))
    assert rows
    assert all(r["raw_value_versioned"] == "false" for r in rows)
    assert all("REC_00001" not in r.values() for r in rows)
    assert any(r["key_value_class"] == "EVENT_PATCH_NUMERIC_ID" for r in rows)
    assert any(r["namespace_hint"] == "ANCHOR_REFPATCH_NAMESPACE" for r in rows)
