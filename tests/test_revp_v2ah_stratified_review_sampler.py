import scripts.protocolo_c.revp_v2ah_common as common
from tests.test_revp_v2ah_common import install_inputs, set_env


def test_sampler_is_review_only_and_deterministic(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    common.run_candidate_reference_review_queue(common.parse_args([]))
    first = common.run_stratified_review_sampler(common.parse_args([]))
    second = common.run_stratified_review_sampler(common.parse_args([]))
    assert first == second
    assert all(r["sample_seed"] == "20260604" for r in first)
    assert all("training" in r["forbidden_use"] for r in first)
