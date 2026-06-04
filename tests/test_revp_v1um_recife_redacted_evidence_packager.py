import scripts.protocolo_c.revp_v1um_recife_common as common
from tests.test_revp_v1um_recife_locality_candidate_sampler import make_base


def test_redacted_evidence_package_removes_literal_address(tmp_path, monkeypatch):
    data = make_base(tmp_path, monkeypatch)
    common.run_locality_candidate_sampler(str(data / "v1um_recife_locality_candidate_sample_registry.csv"))
    common.run_locality_text_normalizer(str(data / "v1um_recife_locality_normalization_registry.csv"))
    common.run_hazard_semantics_ranker(str(data / "v1um_recife_hazard_semantics_rank_registry.csv"))
    out = data / "evidence.csv"
    rows = common.run_redacted_evidence_packager(str(out))
    content = out.read_text(encoding="utf-8")
    assert rows[0]["public_redaction_status"] == "HASHES_FLAGS_ONLY_NO_LITERAL_SENSITIVE_VALUES"
    assert "Rua " not in content
    assert rows[0]["ground_truth_operational"] == "false"
