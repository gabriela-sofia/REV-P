import scripts.protocolo_c.revp_v2ah_common as common
from tests.test_revp_v2ah_common import install_inputs, set_env


def test_dossier_index_uses_hashes_and_no_raw_paths(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    rows = common.run_candidate_evidence_dossier_index(common.parse_args([]))
    assert rows
    assert all(r["use_status"] == "REVIEW_ONLY_NOT_OPERATIONAL" for r in rows)
    assert all("datasets/protocolo_c/" in r["source_artifacts"] for r in rows)
    assert all("C:" not in r["source_artifacts"] for r in rows)
    assert all(r["source_artifact_hashes"] for r in rows)
