import scripts.protocolo_c.revp_v1um_recife_common as common
from tests.test_revp_v1um_recife_locality_candidate_sampler import make_base


def test_aggregation_does_not_create_geometry(tmp_path, monkeypatch):
    data = make_base(tmp_path, monkeypatch)
    common.run_locality_candidate_sampler(str(data / "v1um_recife_locality_candidate_sample_registry.csv"))
    common.run_locality_text_normalizer(str(data / "v1um_recife_locality_normalization_registry.csv"))
    common.run_hazard_semantics_ranker(str(data / "v1um_recife_hazard_semantics_rank_registry.csv"))
    common.run_redacted_evidence_packager(str(data / "v1um_recife_redacted_evidence_package_registry.csv"))
    rows = common.run_neighborhood_signal_aggregator(str(data / "aggregation.csv"))
    assert rows[0]["can_support_overlay"] == "false"
    assert rows[0]["can_create_ground_reference"] == "false"
    assert "geometry" not in rows[0]
