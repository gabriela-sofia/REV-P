import scripts.protocolo_c.revp_v1um_recife_common as common
from tests.test_revp_v1um_recife_locality_candidate_sampler import make_base


def test_human_review_batch_creates_safe_markdown(tmp_path, monkeypatch):
    data = make_base(tmp_path, monkeypatch)
    docs = tmp_path / "review_packages"
    common.run_locality_candidate_sampler(str(data / "v1um_recife_locality_candidate_sample_registry.csv"))
    common.run_locality_text_normalizer(str(data / "v1um_recife_locality_normalization_registry.csv"))
    common.run_hazard_semantics_ranker(str(data / "v1um_recife_hazard_semantics_rank_registry.csv"))
    rows = common.run_human_review_batch_builder(str(data / "batch.csv"), str(docs))
    content = (docs / "v1um_recife_human_review_batch_01.md").read_text(encoding="utf-8")
    assert rows[0]["human_review_package_created"] == "true"
    assert rows[0]["human_review_status"] == "PREPARED_NOT_OPERATIONAL"
    assert "Rua " not in content
    assert "no_overlay_executed: true" in content
