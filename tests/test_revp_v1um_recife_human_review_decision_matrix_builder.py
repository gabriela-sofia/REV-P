import scripts.protocolo_c.revp_v1um_recife_common as common
from tests.test_revp_v1um_recife_locality_candidate_sampler import make_base


def test_human_review_decision_matrix_does_not_create_truth(tmp_path, monkeypatch):
    data = make_base(tmp_path, monkeypatch)
    common.run_locality_candidate_sampler(str(data / "v1um_recife_locality_candidate_sample_registry.csv"))
    common.run_locality_text_normalizer(str(data / "v1um_recife_locality_normalization_registry.csv"))
    common.run_hazard_semantics_ranker(str(data / "v1um_recife_hazard_semantics_rank_registry.csv"))
    common.run_redacted_evidence_packager(str(data / "v1um_recife_redacted_evidence_package_registry.csv"))
    rows = common.run_human_review_decision_matrix_builder(str(data / "decision.csv"))
    assert rows[0]["operational_decision"] == "NOT_OPERATIONAL"
    assert rows[0]["ground_reference_status"] == "NOT_CREATED"
    assert rows[0]["label_status"] == "NOT_CREATED"
