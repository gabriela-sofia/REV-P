import scripts.protocolo_c.revp_v1us_common as common
from tests.test_revp_v1us_patch_registry_resolver import set_env, build_chain


def test_recife_locality_only_preserved(tmp_path, monkeypatch):
    data, root, _, _ = set_env(tmp_path, monkeypatch)
    build_chain(data, root)
    rows = common.run_external_evidence_attacher()
    rec = [r for r in rows if r["event_id"] == "REC_2022_05_24_30"]
    assert rec and all(r["evidence_type"] == "LOCALITY_ONLY_HUMAN_REVIEW" for r in rec)
    assert all(r["evidence_strength"] == "STRONG_CONTEXTUAL_LOCALITY_ONLY" for r in rec)
    assert all(r["can_support_overlay"] == "false" for r in rec)
    assert all(r["can_create_ground_reference"] == "false" for r in rec)


def test_petropolis_document_only_no_geodata_preserved(tmp_path, monkeypatch):
    data, root, _, _ = set_env(tmp_path, monkeypatch)
    build_chain(data, root)
    rows = common.run_external_evidence_attacher()
    pet = [r for r in rows if r["event_id"].startswith("PET_")]
    assert pet
    assert all("no_geo" in r["evidence_limitations"] or "no_geometry" in r["evidence_limitations"] for r in pet)
    assert all(r["can_support_overlay"] == "false" for r in pet)
