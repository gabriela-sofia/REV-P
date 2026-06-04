import os

import scripts.protocolo_c.revp_v1um_recife_common as common
from tests.test_revp_v1um_recife_locality_candidate_sampler import make_base


def test_completion_report_writes_manifest_and_status(tmp_path, monkeypatch):
    data = make_base(tmp_path, monkeypatch)
    docs = tmp_path / "docs" / "metodologia_cientifica"
    configs = tmp_path / "configs" / "protocolo_c"
    review = tmp_path / "docs" / "review_packages" / "protocolo_c"
    docs.mkdir(parents=True)
    configs.mkdir(parents=True)
    review.mkdir(parents=True)
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CONFIG_DIR", str(configs))
    monkeypatch.setattr(common, "REVIEW_DIR", str(review))
    monkeypatch.setattr(common, "V1UM_ARTIFACTS", [])
    common.run_locality_candidate_sampler(str(data / "v1um_recife_locality_candidate_sample_registry.csv"))
    common.run_locality_text_normalizer(str(data / "v1um_recife_locality_normalization_registry.csv"))
    common.run_hazard_semantics_ranker(str(data / "v1um_recife_hazard_semantics_rank_registry.csv"))
    common.run_human_review_batch_builder(str(data / "v1um_recife_human_review_batch_registry.csv"), str(review))
    common.run_redacted_evidence_packager(str(data / "v1um_recife_redacted_evidence_package_registry.csv"))
    common.run_neighborhood_signal_aggregator(str(data / "v1um_recife_neighborhood_signal_aggregation.csv"))
    common.run_human_review_decision_matrix_builder(str(data / "v1um_recife_human_review_decision_matrix.csv"))
    common.run_non_overlay_readiness_matrix(str(data / "v1um_recife_non_overlay_readiness_matrix.csv"))
    result = common.run_completion_report()
    assert result["locality_only_candidates_processed"] == 3
    assert result["next_action"] == "v1un - Human Review Evidence Consolidation Registry"
    assert os.path.exists(docs / "protocolo_c_status_atual_v1um.md")
    assert os.path.exists(data / "v1um_versionable_artifacts_manifest.csv")
